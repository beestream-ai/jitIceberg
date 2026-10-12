/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.spark.sql.iceberg.gpu

import org.apache.iceberg.expressions.{Expression => IcebergExpression}
import org.apache.iceberg.expressions.Expressions
import org.apache.iceberg.spark.SparkV2Filters
import org.apache.iceberg.spark.source.SparkTable
import org.apache.iceberg.tornadovm.DeviceValue
import org.apache.iceberg.tornadovm.spark.GpuAggregates
import org.apache.iceberg.types.Type.TypeID
import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.Strategy
import org.apache.spark.sql.catalyst.expressions.{Left => _, Right => _, _}
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2ScanRelation
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2Strategy
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import scala.collection.JavaConverters._

/** `spark.sql.extensions=org.apache.spark.sql.iceberg.gpu.GpuAggregateExtensions` */
class GpuAggregateExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectPlannerStrategy(_ => GpuAggregateStrategy)
  }
}

/** What one aggregate function of the query becomes: device sums and the row count. */
sealed trait DeviceAggregate
case object RowCount extends DeviceAggregate
case class SumOf(value: Int) extends DeviceAggregate
case class AverageOf(value: Int) extends DeviceAggregate

/**
 * Plans a grouped aggregate over an Iceberg table -- SUM, AVG and COUNT of products of double
 * columns and `1 - c` / `1 + c` factors, grouped by up to four integer, date, timestamp or short
 * string columns, under a filter the device can apply -- as one device scan
 * ([[GpuAggregateExec]]): merge-on-read deletes, the filter, the arithmetic and the grouping all
 * on the GPU, only the groups back on the host. Anything else plans as Spark would.
 *
 * Local mode only: the device scan runs in the driver, which there is the executor too.
 */
object GpuAggregateStrategy extends Strategy with PredicateHelper with Logging {
  val Enabled = "spark.iceberg.gpu.enabled"
  val AggregateEnabled = "spark.iceberg.gpu.aggregate.enabled"

  /** Read options that pick another state of the table than its current one. */
  private val TimeTravel = Set(
    "snapshot-id",
    "as-of-timestamp",
    "branch",
    "tag",
    "start-snapshot-id",
    "end-snapshot-id",
    "start-timestamp",
    "end-timestamp")

  /** Set while a declined aggregate plans on the CPU, so that it does not come back here. */
  private[gpu] val planningFallback = new ThreadLocal[Boolean] {
    override def initialValue(): Boolean = false
  }

  private def enabled: Boolean = {
    val env = Option(SparkEnv.get).map(_.conf)
    def flag(key: String, default: String): Boolean =
      SQLConf.get.getConfString(key, env.map(_.get(key, default)).getOrElse(default)).toBoolean
    flag(Enabled, "false") && flag(AggregateEnabled, "true") &&
      env.exists(_.get("spark.master", "").startsWith("local"))
  }

  override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
    case agg: Aggregate if !planningFallback.get() && enabled =>
      deviceSpec(agg) match {
        case Right(spec) => Seq(GpuAggregateExec(agg.output, spec, agg))
        case Left(reason) =>
          logInfo(s"GPU aggregate declined: $reason")
          Nil
      }
    case _ => Nil
  }

  private def deviceSpec(agg: Aggregate): Either[String, GpuAggregateSpec] = {
    for {
      source <- scanOf(agg.child)
      (conditions, aliases, relation) = source
      groupExpressions = agg.groupingExpressions.map(substitute(_, aliases))
      resultExpressions = agg.aggregateExpressions.map(
        substitute(_, aliases).asInstanceOf[NamedExpression])
      table <- relation.relation.table match {
        case t: SparkTable
            if t.snapshotId() == null &&
              (t.branch() == null || t.branch() == "main") &&
              !relation.relation.options.keySet().asScala.exists(TimeTravel.contains) =>
          Right(t.table())
        case other => Left(s"not the current state of an Iceberg table: $other")
      }
      filter <- icebergFilter(conditions)
      bucket <- groupExpressions match {
        case Seq(BucketOf(buckets, a)) =>
          field(table, a).flatMap { f =>
            if (f.isRequired && f.`type`().typeId() == TypeID.LONG) Right(Some((f.name(), buckets)))
            else Left(s"bucket of ${a.name}: not a required bigint column")
          }
        case _ => Right(None)
      }
      groups <-
        if (bucket.isDefined) Right(Nil)
        else
          traverse(groupExpressions) {
            case a: AttributeReference => groupColumn(table, a)
            case other => Left(s"group expression is not a column: $other")
          }
      _ <- if (groups.size <= 4) Right(()) else Left("more than 4 group columns")
      aggregates = resultExpressions
        .flatMap(_.collect { case ae: AggregateExpression => ae })
        .distinct
      translated <- {
        val values = scala.collection.mutable.ArrayBuffer[DeviceValue]()
        traverse(aggregates)(ae => deviceAggregate(table, ae, values).map(ae -> _))
          .map(pairs => (pairs, values.toList))
      }
      (pairs, values) = translated
      _ <- if (values.nonEmpty) Right(()) else Left("no SUM or AVG")
      sumColumns <-
        if (bucket.isEmpty) Right(Nil)
        else
          traverse(pairs.collect { case (ae, kind) if kind != RowCount => ae }) { ae =>
            ae.aggregateFunction.children match {
              case Seq(a: AttributeReference) if a.dataType == DoubleType => Right(a.name)
              case _ => Left(s"bucket aggregates sum plain double columns only: $ae")
            }
          }
    } yield GpuAggregateSpec(
      table,
      filter,
      groupExpressions,
      groups,
      bucket,
      sumColumns,
      values,
      pairs.toMap,
      resultExpressions)
  }

  /** `bucket(n, column)`, Iceberg's bucket transform of a bigint column, as Spark calls it. */
  private object BucketOf {
    def unapply(e: Expression): Option[(Int, AttributeReference)] = e match {
      case s: org.apache.spark.sql.catalyst.expressions.objects.StaticInvoke
          if s.staticObject == classOf[org.apache.iceberg.spark.functions.BucketFunction.BucketLong] =>
        arguments(s.arguments)
      case a: ApplyFunctionExpression
          if a.function.isInstanceOf[org.apache.iceberg.spark.functions.BucketFunction.BucketLong] =>
        arguments(a.children)
      case _ => None
    }

    private def arguments(args: Seq[Expression]): Option[(Int, AttributeReference)] = args match {
      case Seq(Literal(n: Int, IntegerType), a: AttributeReference) => Some((n, a))
      case _ => None
    }
  }

  /**
   * Projections and filters down to an Iceberg scan: the filter's conditions, and what each
   * projected alias stands for, in terms of the scan's columns.
   */
  private def scanOf(plan: LogicalPlan)
      : Either[String, (Seq[Expression], Map[ExprId, Expression], DataSourceV2ScanRelation)] =
    plan match {
      case Project(list, child) if list.forall(e => e.deterministic) =>
        scanOf(child).flatMap { case (conditions, aliases, relation) =>
          traverse(list) {
            case _: Attribute => Right(None)
            case alias @ Alias(e, _) if !e.exists(_.isInstanceOf[AggregateExpression]) =>
              Right(Some(alias.exprId -> substitute(e, aliases)))
            case other => Left(s"projection: $other")
          }.map(added => (conditions, aliases ++ added.flatten, relation))
        }
      case Filter(condition, child) =>
        scanOf(child).map { case (conditions, aliases, relation) =>
          (splitConjunctivePredicates(substitute(condition, aliases)) ++ conditions, aliases, relation)
        }
      case relation: DataSourceV2ScanRelation => Right((Nil, Map.empty, relation))
      case other => Left(s"not a filtered scan: ${other.nodeName}")
    }

  private def substitute(e: Expression, aliases: Map[ExprId, Expression]): Expression =
    if (aliases.isEmpty) e
    else e.transformUp { case a: AttributeReference if aliases.contains(a.exprId) => aliases(a.exprId) }

  /** Every condition as an Iceberg expression, or a refusal: the device must apply all. */
  private def icebergFilter(conditions: Seq[Expression]): Either[String, IcebergExpression] =
    traverse(conditions) { condition =>
      DataSourceV2Strategy
        .translateFilterV2(condition)
        .flatMap(predicate => Option(SparkV2Filters.convert(predicate)))
        .toRight(s"filter not translatable: $condition")
    }.map(_.foldLeft(Expressions.alwaysTrue(): IcebergExpression)(Expressions.and))

  private def field(table: org.apache.iceberg.Table, a: Attribute) =
    Option(table.schema().findField(a.name)).toRight(s"not a table column: ${a.name}")

  private def groupColumn(table: org.apache.iceberg.Table, a: Attribute): Either[String, String] =
    field(table, a).flatMap { f =>
      f.`type`().typeId() match {
        case TypeID.INTEGER | TypeID.LONG | TypeID.DATE | TypeID.TIMESTAMP | TypeID.STRING =>
          Right(f.name())
        case TypeID.DECIMAL if a.dataType.asInstanceOf[DecimalType].precision <= 18 =>
          Right(f.name())
        case other => Left(s"cannot group by $other")
      }
    }

  private def deviceAggregate(
      table: org.apache.iceberg.Table,
      ae: AggregateExpression,
      values: scala.collection.mutable.ArrayBuffer[DeviceValue]): Either[String, DeviceAggregate] = {
    if (ae.isDistinct || ae.filter.isDefined) {
      return Left(s"DISTINCT or FILTER: $ae")
    }
    def add(value: DeviceValue): Int = {
      values += value
      values.size - 1
    }
    def valueIndex(child: Expression): Either[String, Int] = deviceValue(table, child).map(add)
    ae.aggregateFunction match {
      case Count(children) =>
        traverse(children) {
          case l: Literal if l.value != null => Right(())
          case a: AttributeReference => nullFree(table, a)
          case other => Left(s"COUNT of $other")
        }.map(_ => RowCount)
      case s: Sum if s.child.dataType == DoubleType => valueIndex(s.child).map(SumOf)
      case a: Average if a.child.dataType == DoubleType => valueIndex(a.child).map(AverageOf)
      case s: Sum => exactColumn(table, s.child).map(add).map(SumOf)
      case a: Average => exactColumn(table, a.child).map(add).map(AverageOf)
      case other => Left(s"aggregate not supported: $other")
    }
  }

  /**
   * An int, bigint or decimal (up to 18 digits, as its unscaled value) column whose sum the
   * device's doubles hold exactly, by the files' bounds: Spark sums it as a bigint or a decimal,
   * exactly too.
   */
  private def exactColumn(
      table: org.apache.iceberg.Table,
      child: Expression): Either[String, DeviceValue] = child match {
    // Spark's DecimalAggregates has already turned SUM and AVG of a short decimal into ones of its
    // unscaled value: what the device sums.
    case UnscaledValue(a: AttributeReference) if exactType(a.dataType) => exactColumn(table, a)
    case a: AttributeReference if exactType(a.dataType) =>
      for {
        f <- field(table, a)
        _ <- nullFree(table, a)
        _ <-
          if (GpuAggregates.exactSum(table, a.name)) Right(())
          else Left(s"sum of ${a.name} may not be exact in doubles")
      } yield DeviceValue.of(f.name())
    case other => Left(s"aggregate of $other")
  }

  private def exactType(dataType: DataType): Boolean = dataType match {
    case IntegerType | LongType => true
    case d: DecimalType => d.precision <= 18
    case _ => false
  }

  private sealed trait Factor
  private case class Plain(column: String) extends Factor
  private case class OneMinus(column: String) extends Factor
  private case class OnePlus(column: String) extends Factor

  /**
   * A product of non-null numeric columns and `1 - c`, `1 + c` factors, with at least one plain
   * column: what [[DeviceValue]] evaluates. Non-null, so that a group's sum is never SQL's NULL
   * and its row count is AVG's count.
   */
  private def deviceValue(
      table: org.apache.iceberg.Table,
      expr: Expression): Either[String, DeviceValue] = {
    def column(e: Expression): Either[String, String] = e match {
      case a: AttributeReference if a.dataType == DoubleType => numeric(a)
      case Cast(a: AttributeReference, DoubleType, _, _)
          if a.dataType == IntegerType || a.dataType == LongType =>
        numeric(a)
      case other => Left(s"not a numeric column: $other")
    }
    def numeric(a: AttributeReference): Either[String, String] =
      field(table, a).flatMap { f =>
        f.`type`().typeId() match {
          case TypeID.DOUBLE | TypeID.INTEGER | TypeID.LONG => nullFree(table, a).map(_ => f.name())
          case other => Left(s"not a device value column: $other")
        }
      }
    def one(l: Literal): Boolean = l.value != null && l.value.toString.toDouble == 1.0
    def factors(e: Expression): Either[String, Seq[Factor]] = e match {
      case Multiply(left, right, _) =>
        for (l <- factors(left); r <- factors(right)) yield l ++ r
      case Subtract(l: Literal, c, _) if one(l) => column(c).map(name => Seq(OneMinus(name)))
      case Add(l: Literal, c, _) if one(l) => column(c).map(name => Seq(OnePlus(name)))
      case Add(c, l: Literal, _) if one(l) => column(c).map(name => Seq(OnePlus(name)))
      case c => column(c).map(name => Seq(Plain(name)))
    }
    factors(expr).flatMap { all =>
      val (plain, rest) = all.partition(_.isInstanceOf[Plain])
      if (plain.isEmpty) {
        Left(s"no plain column factor: $expr")
      } else if (all.size > DeviceValue.MAX_FACTORS) {
        Left(s"more than ${DeviceValue.MAX_FACTORS} factors: $expr")
      } else {
        val first = DeviceValue.of(plain.head.asInstanceOf[Plain].column)
        Right((plain.tail ++ rest).foldLeft(first) {
          case (value, Plain(c)) => value.times(c)
          case (value, OneMinus(c)) => value.timesOneMinus(c)
          case (value, OnePlus(c)) => value.timesOnePlus(c)
        })
      }
    }
  }

  private def nullFree(table: org.apache.iceberg.Table, a: Attribute): Either[String, Unit] =
    if (!a.nullable || GpuAggregates.nullFree(table, a.name)) Right(())
    else Left(s"${a.name} may hold nulls")

  private def traverse[A, B](items: Seq[A])(f: A => Either[String, B]): Either[String, Seq[B]] =
    items.foldLeft(Right(Vector.empty): Either[String, Vector[B]]) { (acc, item) =>
      for (done <- acc; next <- f(item)) yield done :+ next
    }
}
