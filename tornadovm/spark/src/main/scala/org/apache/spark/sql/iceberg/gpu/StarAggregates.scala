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
import org.apache.iceberg.tornadovm.DeviceDimension
import org.apache.iceberg.tornadovm.DeviceProbe
import org.apache.iceberg.tornadovm.DeviceValue
import org.apache.iceberg.tornadovm.spark.GpuAggregates
import org.apache.iceberg.types.Type.TypeID
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Left => _, Right => _, _}
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2ScanRelation
import org.apache.spark.sql.types._
import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * A dimension of a star aggregate: its plan (with the query's filters on it), the fact column
 * joined to its key, and the dimension attributes the query groups by.
 */
case class StarDimension(
    plan: LogicalPlan,
    factColumn: String,
    key: Attribute,
    attributes: Seq[Attribute],
    probeAttributes: Seq[Attribute] = Nil)

/** A probe key part on the fact side: a fact column, or a dimension's probe attribute. */
sealed trait ProbePart
case class FactPart(column: String) extends ProbePart
case class AttributePart(dimension: Int, attribute: Int) extends ProbePart

/**
 * The other side of a probe join ([[ProbeAggregates]]): a plan Spark runs and collects, whose
 * columns are each key part's value, then the compared value when there is a comparison; the
 * fact side's parts; how the fact's column compares with the build's value; and how many build
 * columns follow, summed over each fact row's joined rows.
 */
case class StarProbe(
    build: LogicalPlan,
    parts: Seq[ProbePart],
    comparison: Option[(String, DeviceProbe.Comparison)],
    sums: Int = 0)

/**
 * An aggregate of `CASE WHEN predicate THEN value [ELSE 0 | NULL] END` whose predicate reads
 * dimension attributes only: the device aggregates the value ({@code base}, or the row count when
 * -1), the predicate's attributes join the finest group keys, and the driver keeps or zeroes each
 * finest group's part.
 */
case class Conditional(base: Int, predicate: Expression, elseZero: Boolean)

/**
 * An aggregate over inner joins of an Iceberg fact table with small dimensions, as the device and
 * the driver compute it ([[StarAggregates]]):
 *
 *   - Spark runs each dimension's plan (its filters included) and the driver turns the rows into a
 *     mask over the key range and one code a key for the tuple of attributes the query uses;
 *   - the device scans the fact table (deletes and the fact's own filters applied), joins every
 *     dimension by key lookup and groups by the codes and the fact's group columns: the finest
 *     grouping the query needs, with sums, row counts and non-null counts;
 *   - the driver decodes the groups, applies the query's `Expand` (ROLLUP, CUBE, GROUPING SETS) and
 *     group expressions, merges the sums and counts and builds the result rows.
 *
 * A dimension whose keys are not dense, or not unique, fails the run; the aggregate then runs on
 * the CPU.
 */
case class GpuStarAggregateSpec(
    table: org.apache.iceberg.Table,
    filter: IcebergExpression,
    dimensions: Seq[StarDimension],
    factKeys: Seq[(Attribute, String)],
    values: Seq[DeviceValue],
    conditionals: Seq[Conditional],
    aggregates: Map[AggregateExpression, DeviceAggregate],
    expand: Option[(Seq[Seq[Expression]], Seq[Attribute])],
    groupExpressions: Seq[Expression],
    resultExpressions: Seq[NamedExpression],
    probe: Option[StarProbe] = None)
    extends DeviceAggregateSpec
    with org.apache.spark.internal.Logging {

  // The probe's collected build rows, for this run.
  @transient private var buildSide: Option[ProbeAggregates.BuildRows] = None

  /** The attributes of a finest group, in order: each dimension's, then the fact's keys. */
  private def finest: Seq[Attribute] = dimensions.flatMap(_.attributes) ++ factKeys.map(_._1)

  override def run(): Seq[InternalRow] = {
    val session = SparkSession.active
    val start = System.nanoTime()
    // The dimensions' Spark jobs run side by side: each is small, and they would add up in line.
    val built = {
      import scala.concurrent.{Await, Future}
      import scala.concurrent.ExecutionContext.Implicits.global
      val jobs = dimensions.map(d => Future(StarAggregates.build(session, d)))
      val buildRows = probe.map(p => Future(ProbeAggregates.collect(session, p)))
      val dims = jobs.map(Await.result(_, scala.concurrent.duration.Duration.Inf))
      buildSide = buildRows.map(Await.result(_, scala.concurrent.duration.Duration.Inf))
      dims
    }
    val (deviceDimensions, deviceProbe) = probe match {
      case Some(p) => ProbeAggregates.prepare(p, buildSide.get, dimensions, built)
      case None => (built.map(_._1), null)
    }
    val dimensionsBuilt = System.nanoTime()
    // Without fact keys the finest groups are at most the product of the dimensions' tuples. Past
    // a few million the device's per-batch groups and the driver's merge cost more than Spark's own
    // aggregate (q67: 4.8M groups, 52 s against 5.4 s), so Spark runs it.
    val most = built.filter(_._2.nonEmpty).map(_._2.length.toDouble).product
    // A probe's groups are bounded by its build side, not by the dimensions' tuples (q72: 41K
    // groups of an item x warehouse x week product of billions).
    if (factKeys.isEmpty && probe.isEmpty && most > StarAggregates.maxGroups) {
      throw new UnsupportedOperationException(
        f"up to $most%.0f finest groups, over ${StarAggregates.maxGroups}%.0f")
    }
    val grouped = dimensions.zipWithIndex.filter(_._1.attributes.nonEmpty)
    val keys =
      grouped.map { case (_, d) => DeviceDimension.Key.attribute(d, 0) } ++
        factKeys.map { case (_, column) => DeviceDimension.Key.column(column) }
    val groups = GpuAggregates.starGroups(
      table,
      filter,
      deviceDimensions.asJava,
      keys.asJava,
      values.asJava,
      deviceProbe)
    val device = System.nanoTime()

    val input = finest
    val predicates = conditionals.map(c => BindReferences.bindReference(c.predicate, input))
    val (bound, expanded) = expand match {
      case Some((projections, output)) =>
        (
          projections.map(_.map(e => BindReferences.bindReference(e, input))),
          groupExpressions.map(e => BindReferences.bindReference(e, output)))
      case None => (Seq.empty, groupExpressions.map(e => BindReferences.bindReference(e, input)))
    }
    val merged = mutable.LinkedHashMap[Seq[Any], Array[Double]]()
    def add(key: Seq[Any], totals: Array[Double]): Unit =
      merged.get(key) match {
        case Some(sum) => for (i <- sum.indices) sum(i) += totals(i)
        case None => merged(key) = totals.clone()
      }
    groups.asScala.foreach { case (key, totals) =>
      val codes = key.asScala
      val row = new Array[Any](input.size)
      var at = 0
      grouped.zipWithIndex.foreach { case ((d, i), k) =>
        val tuple = built(i)._2(codes(k).asInstanceOf[Number].intValue())
        System.arraycopy(tuple, 0, row, at, d.attributes.size)
        at += d.attributes.size
      }
      factKeys.zipWithIndex.foreach { case ((a, _), k) =>
        row(at) = DeviceResults.sparkValue(a.dataType, codes(grouped.size + k))
        at += 1
      }
      val finestRow = new GenericInternalRow(row)
      val parts = withConditionals(totals, finestRow, predicates)
      if (bound.isEmpty) {
        add(expanded.map(_.eval(finestRow)), parts)
      } else {
        bound.foreach { projection =>
          val out = new GenericInternalRow(projection.map(_.eval(finestRow)).toArray)
          add(expanded.map(_.eval(out)), parts)
        }
      }
    }
    val result =
      if (merged.isEmpty && groupExpressions.isEmpty) {
        Seq(new DeviceResults(Nil, slots, aggregates, resultExpressions).row(Nil, null))
      } else {
        val results = new DeviceResults(groupExpressions, slots, aggregates, resultExpressions)
        merged.toSeq.map { case (key, totals) => results.row(key, totals) }
      }
    val end = System.nanoTime()
    logInfo(
      f"star aggregate ${table.name()}: dimensions ${(dimensionsBuilt - start) / 1e6}%.1f ms" +
        f" (${built.map(_._2.length).mkString("/")} tuples), device ${(device - dimensionsBuilt) / 1e6}%.1f ms" +
        f" (${groups.size()} groups), driver ${(end - device) / 1e6}%.1f ms (${merged.size} groups out)" +
        s"; scan: ${GpuAggregates.lastStarMetrics()}")
    result
  }

  /** Device values, then one slot a conditional aggregate. */
  private def slots: Int = values.size + conditionals.size

  /**
   * A finest group's totals with the conditionals' slots: sums, the row count, non-null counts,
   * each laid out over {@link #slots} values as [[DeviceResults]] reads them.
   */
  private def withConditionals(
      totals: Array[Double],
      row: InternalRow,
      predicates: Seq[Expression]): Array[Double] = {
    if (conditionals.isEmpty) {
      return totals
    }
    val n = values.size
    val m = conditionals.size
    val rows = totals(n)
    val out = new Array[Double](2 * (n + m) + 1)
    System.arraycopy(totals, 0, out, 0, n)
    out(n + m) = rows
    System.arraycopy(totals, n + 1, out, n + m + 1, n)
    conditionals.zipWithIndex.foreach { case (c, i) =>
      if (predicates(i).eval(row) == true) {
        out(n + i) = if (c.base < 0) rows else totals(c.base)
        out(2 * n + m + 1 + i) = if (c.base < 0) rows else totals(n + 1 + c.base)
      } else {
        // ELSE 0: every row a non-null 0; ELSE NULL: nothing.
        out(2 * n + m + 1 + i) = if (c.elseZero) rows else 0
      }
    }
    out
  }

  override def toString: String =
    s"star ${table.name()} filter=$filter " +
      dimensions.map(d => s"${d.factColumn}->${d.key.name}${d.attributes.map(_.name).mkString("(", ",", ")")}")
        .mkString("joins=[", ", ", "]") +
      s" keys=${factKeys.map(_._2).mkString("[", ", ", "]")} values=${values.mkString("[", ", ", "]")}" +
      (if (conditionals.isEmpty) "" else s" when=${conditionals.map(_.predicate).mkString("[", ", ", "]")}")
}

/**
 * Plans star aggregates: an `Aggregate`, possibly over an `Expand`, over projections and filters,
 * over a tree of inner joins whose leaves are Iceberg scans -- the largest the fact table, every
 * other one a dimension of at most `spark.iceberg.gpu.aggregate.star.maxDimensionRows` rows -- in
 * which every dimension joins the fact table by one equality of an integer fact column with its
 * key. Conditions on one table alone are that table's filters (the fact's must be ones the device
 * applies); the aggregates are those the device computes, over fact columns. Anything else
 * declines, and Spark plans the query as usual.
 */
object StarAggregates extends PredicateHelper {
  val MaxDimensionRows = "spark.iceberg.gpu.aggregate.star.maxDimensionRows"

  val MaxGroups = "spark.iceberg.gpu.aggregate.star.maxGroups"

  /** Most finest groups the dimensions' tuples may make before the star runs on Spark. */
  private[gpu] def maxGroups: Double = {
    val env = Option(org.apache.spark.SparkEnv.get).map(_.conf)
    org.apache.spark.sql.internal.SQLConf.get
      .getConfString(MaxGroups, env.map(_.get(MaxGroups, "4194304")).getOrElse("4194304"))
      .toDouble
  }

  /** Key ranges up to this many keys are dense enough, or 4 a dimension row. */
  private val DenseKeys = 1 << 20
  private val MaxKeys = 1 << 26

  import GpuAggregateStrategy.traverse

  /**
   * Stars that failed on the device (too many groups, sparse or repeated keys), by their plan:
   * planned again -- the next run of the same query -- they decline at once instead of building
   * their dimensions to fail again.
   */
  private val failed = java.util.Collections.newSetFromMap(
    java.util.Collections.synchronizedMap(new java.util.LinkedHashMap[String, java.lang.Boolean](64, 0.75f, true) {
      override def removeEldestEntry(eldest: java.util.Map.Entry[String, java.lang.Boolean]): Boolean =
        size() > 1024
    }))

  /**
   * An aggregate's identity across plannings: its canonical plan, with the column names and table
   * snapshots a canonical plan leaves out (it names every attribute "none").
   */
  private[gpu] def identity(agg: LogicalPlan): String = {
    val relations = agg.collect { case r: DataSourceV2ScanRelation => r }
    relations.map { r =>
      val snapshot = GpuAggregateStrategy.currentTable(r).toOption
        .map(t => s"${t.location()}@${Option(t.currentSnapshot()).map(_.snapshotId()).orNull}")
      s"$snapshot${r.output.map(_.name)}"
    }.mkString(";") + agg.canonicalized.toString
  }

  private[gpu] def remember(agg: LogicalPlan): Unit = failed.add(identity(agg))

  private[gpu] def failedBefore(agg: LogicalPlan): Boolean = failed.contains(identity(agg))

  def spec(agg: Aggregate): Either[String, GpuStarAggregateSpec] =
    if (failedBefore(agg)) Left("failed on the device before") else plan(agg)

  private def plan(agg: Aggregate): Either[String, GpuStarAggregateSpec] = {
    val (expand, below) = agg.child match {
      case e: Expand => (Some(e), e.child)
      case other => (None, other)
    }
    for {
      upper <- aboveJoins(below)
      (upperConditions, upperAliases, joins) = upper
      tree <- joinTree(joins)
      (leaves, joinConditions) = tree
      _ <- if (leaves.size >= 2) Right(()) else Left("not a join")
      sized <- traverse(leaves) { leaf =>
        leaf.collect { case r: DataSourceV2ScanRelation => r } match {
          case Seq(relation) =>
            GpuAggregateStrategy.currentTable(relation).map(t => (leaf, relation, rowCount(t)))
          case _ => Left(s"a join input that is not one Iceberg scan: ${leaf.nodeName}")
        }
      }
      (factLeaf, factRelation, _) = sized.maxBy(_._3)
      dimensionLeaves = sized.filter(_._1 ne factLeaf).map(_._1)
      maxRows = sized.filter(_._1 ne factLeaf).map(_._3).max
      _ <-
        if (maxRows <= dimensionRowLimit) Right(())
        else Left(s"a dimension of $maxRows rows: not a star")
      fact <- GpuAggregateStrategy.scanOf(factLeaf)
      (factConditions, factAliases, _) = fact
      table <- GpuAggregateStrategy.currentTable(factRelation)
      factOutput = factLeaf.outputSet
      // A fact attribute as the scan's column.
      factColumn = (a: Attribute) =>
        GpuAggregateStrategy.substitute(a, factAliases) match {
          case c: AttributeReference => GpuAggregateStrategy.field(table, c).map(_.name())
          case other => Left(s"not a fact column: $other")
        }
      classified <- traverse[Expression, Either[Expression, DimensionPart]](
        (upperConditions ++ joinConditions).map(GpuAggregateStrategy.substitute(_, upperAliases))) {
        condition =>
          val refs = condition.references
          if (refs.subsetOf(factOutput)) {
            Right(Left(GpuAggregateStrategy.substitute(condition, factAliases)))
          } else {
            dimensionLeaves.indexWhere(l => refs.subsetOf(l.outputSet)) match {
              case d if d >= 0 => Right(Right((d, None, condition)))
              case _ =>
                condition match {
                  case EqualTo(a: Attribute, b: Attribute) => edge(a, b, factOutput, dimensionLeaves)
                  case other => Left(s"join condition: $other")
                }
            }
          }
      }
      factFilters = classified.collect { case Left(c) => c }
      filter <- GpuAggregateStrategy.icebergFilter(factFilters ++ factConditions)
      dimensionParts = classified.collect { case Right(part) => part }
      // The Expand's value columns (the same child attribute in every projection) and what the
      // aggregates and group keys read.
      passThrough = expand.toSeq.flatMap { e =>
        e.output.indices.flatMap { i =>
          e.projections.map(_(i)).distinct match {
            case Seq(a: Attribute) => Some(e.output(i).exprId -> (a: Expression))
            case _ => None
          }
        }
      }.toMap
      // Without an Expand, the outputs over the joins' attributes, as for a plain device aggregate.
      resultExpressions = agg.aggregateExpressions.map {
        case a: Attribute if upperAliases.contains(a.exprId) && expand.isEmpty =>
          Alias(upperAliases(a.exprId), a.name)(exprId = a.exprId)
        case named if expand.isEmpty =>
          GpuAggregateStrategy.substitute(named, upperAliases).asInstanceOf[NamedExpression]
        case named => named
      }
      aggregateExpressions = resultExpressions
        .flatMap(_.collect { case ae: AggregateExpression => ae })
        .distinct
      translated <- {
        val values = mutable.ArrayBuffer[DeviceValue]()
        val conditionals = mutable.ArrayBuffer[(AggregateExpression, Conditional)]()
        def onDevice(overJoins: AggregateExpression) = {
          val overScan = GpuAggregateStrategy
            .substitute(overJoins, factAliases)
            .asInstanceOf[AggregateExpression]
          GpuAggregateStrategy.deviceAggregate(table, overScan, values, nullable = true)
        }
        traverse(aggregateExpressions) { ae =>
          val overJoins = GpuAggregateStrategy
            .substitute(GpuAggregateStrategy.substitute(ae, passThrough), upperAliases)
            .asInstanceOf[AggregateExpression]
          if (overJoins.references.subsetOf(factOutput)) {
            onDevice(overJoins).map(kind => Some(ae -> kind))
          } else {
            conditional(overJoins, factOutput, dimensionLeaves) match {
              case Some((plain, predicate, elseZero)) =>
                onDevice(plain).flatMap {
                  case SumOf(v) => Right(SumOf(v))
                  case AverageOf(v) => Right(AverageOf(v))
                  case CountOf(v) => Right(CountOf(v))
                  case RowCount => Right(CountOf(-1))
                  case other => Left(s"conditional $other")
                }.map { kind =>
                  val base = kind match {
                    case SumOf(v) => v
                    case AverageOf(v) => v
                    case CountOf(v) => v
                    case _ => -1
                  }
                  conditionals += ae -> Conditional(base, predicate, elseZero)
                  None
                }
              case None => Left(s"aggregate of dimension columns: $ae")
            }
          }
        }.map { pairs =>
          // A conditional's slot follows the device values.
          val slotted = conditionals.zipWithIndex.map { case ((ae, c), i) =>
            val slot = values.size + i
            ae -> (ae.aggregateFunction match {
              case _: org.apache.spark.sql.catalyst.expressions.aggregate.Sum => SumOf(slot)
              case _: org.apache.spark.sql.catalyst.expressions.aggregate.Average => AverageOf(slot)
              case _ => CountOf(slot)
            })
          }
          ((pairs.flatten ++ slotted).toMap, values.toList, conditionals.map(_._2).toList)
        }
      }
      (aggregates, values, conditionalAggregates) = translated
      _ <- if (values.nonEmpty) Right(()) else Left("no SUM or AVG")
      // What the group keys are computed from, over the joins' output.
      keyed = expand match {
        case Some(e) =>
          val valuePositions = passThrough.keySet
          (
            Some(e.projections.map(_.zip(e.output).map {
              case (_, out) if valuePositions.contains(out.exprId) => Literal(null, out.dataType)
              case (p, _) => GpuAggregateStrategy.substitute(p, upperAliases)
            })),
            agg.groupingExpressions)
        case None =>
          (None, agg.groupingExpressions.map(GpuAggregateStrategy.substitute(_, upperAliases)))
      }
      (projections, groupExpressions) = keyed
      keyInputs = AttributeSet(
        projections.getOrElse(Seq(groupExpressions)).flatten.flatMap(_.references) ++
          conditionalAggregates.flatMap(_.predicate.references))
      _ <-
        if (keyInputs.forall(a => factOutput.contains(a) || dimensionLeaves.exists(_.outputSet.contains(a))))
          Right(())
        else Left(s"a group key not over the joins: $keyInputs")
      // A fact column as a group key leaves the groups unbounded (TPC-DS q51, q65: millions), and
      // the device's per-batch groups and the driver's merge then cost more than Spark's aggregate.
      _ <- keyInputs.toSeq.filter(factOutput.contains) match {
        case Seq() => Right(())
        case keys => Left(s"fact columns as group keys: ${keys.map(_.name).mkString(", ")}")
      }
      factKeys <- traverse(keyInputs.toSeq.filter(factOutput.contains)) { a =>
        GpuAggregateStrategy.substitute(a, factAliases) match {
          case c: AttributeReference => GpuAggregateStrategy.groupColumn(table, c).map(a -> _)
          case other => Left(s"a fact group key that is not a column: $other")
        }
      }
      dimensions <- traverse(dimensionLeaves.indices) { d =>
        val parts = dimensionParts.filter(_._1 == d)
        val filters = parts.collect { case (_, None, c) => c }
        parts.collect { case (_, Some(join), _) => join } match {
          case Seq((factAttribute, key)) =>
            for {
              column <- factColumn(factAttribute)
              _ <- table.schema().findField(column).`type`().typeId() match {
                case TypeID.INTEGER | TypeID.LONG => Right(())
                case other => Left(s"join on a $other fact column")
              }
              _ <- key.dataType match {
                case IntegerType | LongType => Right(())
                case other => Left(s"dimension key of $other")
              }
              // The key's range, by the dimension's file statistics, must be dense enough to
              // index by; the run checks the rows it reads again.
              _ <- denseByStatistics(dimensionLeaves(d), key)
            } yield {
              val leaf = dimensionLeaves(d)
              StarDimension(
                if (filters.isEmpty) leaf else Filter(filters.reduce(And), leaf),
                column,
                key,
                keyInputs.toSeq.filter(leaf.outputSet.contains))
            }
          case other => Left(s"dimension ${dimensionLeaves(d).nodeName} joined ${other.size} times")
        }
      }
      // A dimension grouped by its attributes has at most as many tuples as rows; one with more
      // rows than the finest groups allowed (customer, 500K) most likely has as many tuples too,
      // and building it to find out costs more than the aggregate (TPC-DS q74, q23).
      _ <- dimensions.zip(dimensionLeaves).find { case (d, leaf) =>
        d.attributes.nonEmpty && sized.find(_._1 eq leaf).exists(_._3 > maxGroups)
      } match {
        case Some((d, _)) => Left(s"grouped by attributes of ${d.key.name}'s table, over $maxGroups rows")
        case None => Right(())
      }
      deviceKeys = dimensions.count(_.attributes.nonEmpty) + factKeys.size
      _ <- if (deviceKeys <= 4) Right(()) else Left(s"$deviceKeys device group keys, at most 4")
    } yield GpuStarAggregateSpec(
      table,
      filter,
      dimensions,
      factKeys,
      values,
      conditionalAggregates,
      aggregates,
      projections.map(p => (p, expand.get.output)),
      groupExpressions,
      resultExpressions)
  }

  /** A condition on one dimension: a filter, or (with the fact attribute and key) its join. */
  private type DimensionPart = (Int, Option[(Attribute, Attribute)], Expression)

  private[gpu] def dimensionRowLimit: Long = {
    val key = MaxDimensionRows
    val env = Option(org.apache.spark.SparkEnv.get).map(_.conf)
    org.apache.spark.sql.internal.SQLConf.get
      .getConfString(key, env.map(_.get(key, "10000000")).getOrElse("10000000"))
      .toLong
  }

  private[gpu] def denseByStatistics(leaf: LogicalPlan, key: Attribute): Either[String, Unit] = {
    val column = leaf.collectFirst {
      case r: DataSourceV2ScanRelation if r.outputSet.contains(key) => r
    }
    column match {
      case Some(relation) =>
        GpuAggregateStrategy.currentTable(relation).flatMap { table =>
          if (GpuAggregates.denseKeys(table, key.name, 4.0, DenseKeys.toDouble)) Right(())
          else Left(s"dimension key ${key.name}: not dense by its statistics")
        }
      case None => Right(())
    }
  }

  private[gpu] def rowCount(table: org.apache.iceberg.Table): Long =
    Option(table.currentSnapshot())
      .flatMap(s => Option(s.summary().get("total-records")))
      .map(_.toLong)
      .getOrElse(0L)

  /**
   * `f(CASE WHEN p THEN v [ELSE 0 | NULL] END)` -- also under `UnscaledValue` or a cast -- with `p`
   * over dimension attributes and `v` over fact columns: `f(v)`, `p`, and whether the else is 0.
   */
  private def conditional(
      ae: AggregateExpression,
      factOutput: AttributeSet,
      dimensions: Seq[LogicalPlan]): Option[(AggregateExpression, Expression, Boolean)] = {
    def zero(e: Option[Expression]): Option[Boolean] = e match {
      case None => Some(false)
      case Some(Literal(null, _)) => Some(false)
      case Some(l: Literal) if l.value.toString.toDouble == 0.0 => Some(true)
      case _ => None
    }
    def split(e: Expression): Option[(Expression, Expression, Boolean)] = e match {
      case CaseWhen(Seq((p, v)), otherwise) =>
        zero(otherwise).map(z => (p, v, z))
      case UnscaledValue(c) => split(c).map { case (p, v, z) => (p, UnscaledValue(v), z) }
      case c: Cast => split(c.child).map { case (p, v, z) => (p, c.withNewChildren(Seq(v)), z) }
      case _ => None
    }
    ae.aggregateFunction.children match {
      case Seq(argument) =>
        split(argument).filter { case (p, v, _) =>
          v.references.subsetOf(factOutput) &&
          p.references.nonEmpty &&
          p.references.forall(a => dimensions.exists(_.outputSet.contains(a)))
        }.map { case (p, v, z) =>
          (ae.copy(aggregateFunction = ae.aggregateFunction.withNewChildren(Seq(v))
            .asInstanceOf[org.apache.spark.sql.catalyst.expressions.aggregate.AggregateFunction]),
            p,
            z)
        }
      case _ => None
    }
  }

  /** A join edge: a fact attribute equal to a dimension's key. */
  private def edge(
      a: Attribute,
      b: Attribute,
      factOutput: AttributeSet,
      dimensions: Seq[LogicalPlan])
      : Either[String, Either[Expression, DimensionPart]] = {
    val (factSide, other) =
      if (factOutput.contains(a)) (a, b) else if (factOutput.contains(b)) (b, a) else (null, null)
    val d = if (other == null) -1 else dimensions.indexWhere(_.outputSet.contains(other))
    if (factSide == null || d < 0) Left(s"join condition: $a = $b")
    else Right(Right((d, Some((factSide, other)), EqualTo(a, b))))
  }

  /** Projections and filters above the joins: their conditions and aliases. */
  private[gpu] def aboveJoins(plan: LogicalPlan)
      : Either[String, (Seq[Expression], Map[ExprId, Expression], LogicalPlan)] = plan match {
    case Project(list, child) if list.forall(_.deterministic) =>
      aboveJoins(child).flatMap { case (conditions, aliases, joins) =>
        traverse(list) {
          case _: Attribute => Right(None)
          case alias @ Alias(e, _) => Right(Some(alias.exprId -> GpuAggregateStrategy.substitute(e, aliases)))
          case other => Left(s"projection: $other")
        }.map(added => (conditions, aliases ++ added.flatten, joins))
      }
    case Filter(condition, child) =>
      aboveJoins(child).map { case (conditions, aliases, joins) =>
        (splitConjunctivePredicates(GpuAggregateStrategy.substitute(condition, aliases)) ++ conditions, aliases, joins)
      }
    case j: Join => Right((Nil, Map.empty, j))
    case other => Left(s"not a filtered scan or a join: ${other.nodeName}")
  }

  /** The leaves of a tree of inner joins (what holds no join) and every condition in it. */
  private[gpu] def joinTree(plan: LogicalPlan): Either[String, (Seq[LogicalPlan], Seq[Expression])] =
    plan match {
      case Join(left, right, Inner, condition, _) =>
        for (l <- joinTree(left); r <- joinTree(right))
          yield (l._1 ++ r._1, l._2 ++ r._2 ++ condition.toSeq.flatMap(splitConjunctivePredicates))
      case j: Join => Left(s"${j.joinType} join")
      case p if !p.exists(_.isInstanceOf[Join]) => Right((Seq(p), Nil))
      case Project(list, child) if list.forall(_.isInstanceOf[Attribute]) => joinTree(child)
      case Filter(condition, child) =>
        joinTree(child).map { case (leaves, conditions) =>
          (leaves, conditions ++ splitConjunctivePredicates(condition))
        }
      case other => Left(s"between joins: ${other.nodeName}")
    }

  /**
   * A dimension as the device takes it, and its attribute tuples by code: Spark runs the
   * dimension's plan; its keys must be unique and dense.
   */
  /**
   * Built dimensions by their plan and their tables' snapshots: the same filtered dimension
   * (TPC-DS's `date_dim` of a year, say) recurs across queries and runs, and building it is a
   * Spark job and a pass over its rows.
   */
  /** A built dimension: the device's, its attribute tuples by code, and each probe attribute's raw values by key. */
  type Built = (DeviceDimension, Array[Array[Any]], Seq[Array[Long]])

  private val built = java.util.Collections.synchronizedMap(
    new java.util.LinkedHashMap[String, Built](64, 0.75f, true) {
      override def removeEldestEntry(eldest: java.util.Map.Entry[String, Built]): Boolean =
        size() > 32
    })

  private def dimensionKey(d: StarDimension): Option[String] = {
    val snapshots = d.plan.collect { case r: DataSourceV2ScanRelation => r }.map { r =>
      GpuAggregateStrategy.currentTable(r).toOption.map { t =>
        s"${t.location()}@${Option(t.currentSnapshot()).map(_.snapshotId()).orNull}"
      }
    }
    // A canonical plan names every attribute "none" and refers to columns by position: two
    // dimensions on the same table reading other columns (q42's item categories, q52's brands)
    // look alike unless the columns' names are part of the key too.
    val columns = d.plan.collect { case r: DataSourceV2ScanRelation => r.output.map(_.name) }
    if (snapshots.exists(_.isEmpty)) None
    else
      Some(
        snapshots.flatten.mkString(",") + "|" + d.factColumn + "|" + columns.mkString(";") + "|" +
          (d.key +: d.attributes).map(_.name).mkString(",") + "|" +
          d.probeAttributes.map(_.name).mkString(",") + "|" +
          Project((d.key +: d.attributes) ++ d.probeAttributes, d.plan).canonicalized.toString)
  }

  private[gpu] def build(session: SparkSession, d: StarDimension): Built =
    dimensionKey(d) match {
      case Some(key) =>
        Option(built.get(key)).getOrElse {
          val made = buildNow(session, d)
          built.put(key, made)
          made
        }
      case None => buildNow(session, d)
    }

  private def buildNow(session: SparkSession, d: StarDimension): Built = {
    val rows = Dataset
      .ofRows(session, Project((d.key +: d.attributes) ++ d.probeAttributes, d.plan))
      .queryExecution
      .executedPlan
      .executeCollect()
    def key(row: InternalRow): Long =
      if (d.key.dataType == IntegerType) row.getInt(0).toLong else row.getLong(0)
    val live = rows.filterNot(_.isNullAt(0))
    val first = if (live.isEmpty) 0L else live.map(key).min
    val last = if (live.isEmpty) 0L else live.map(key).max
    val range = last - first + 1
    if (range > math.min(MaxKeys.toLong, math.max(DenseKeys.toLong, 4L * live.length)) ||
        first < Int.MinValue || last > Int.MaxValue) {
      throw new UnsupportedOperationException(
        s"Dimension ${d.key.name}: keys $first..$last for ${live.length} rows are not dense")
    }
    val keep = new Array[Boolean](range.toInt)
    val codes = new Array[Int](range.toInt)
    // A probe attribute's integer value a key (Long.MinValue: absent or null, joins nothing).
    val raw = d.probeAttributes.map(_ => Array.fill(range.toInt)(Long.MinValue))
    val rawAt = 1 + d.attributes.size
    val dictionary = mutable.HashMap[Seq[Any], Int]()
    val tuples = mutable.ArrayBuffer[Array[Any]]()
    live.foreach { row =>
      val at = (key(row) - first).toInt
      if (keep(at)) {
        throw new UnsupportedOperationException(s"Dimension ${d.key.name}: key ${key(row)} repeats")
      }
      keep(at) = true
      d.probeAttributes.indices.foreach { j =>
        if (!row.isNullAt(rawAt + j)) {
          raw(j)(at) =
            if (d.probeAttributes(j).dataType == IntegerType) row.getInt(rawAt + j).toLong
            else row.getLong(rawAt + j)
        }
      }
      if (d.attributes.nonEmpty) {
        val tuple = d.attributes.indices.map(i => row.get(i + 1, d.attributes(i).dataType))
        codes(at) = dictionary.getOrElseUpdate(tuple, { tuples += tuple.toArray; tuples.size - 1 })
      }
    }
    val dimension = DeviceDimension.of(d.factColumn, first.toInt, keep)
    (if (d.attributes.isEmpty) dimension else dimension.withAttribute(codes), tuples.toArray, raw)
  }
}
