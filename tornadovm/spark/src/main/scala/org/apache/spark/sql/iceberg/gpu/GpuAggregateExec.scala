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
import org.apache.iceberg.tornadovm.DeviceValue
import org.apache.iceberg.tornadovm.spark.GpuAggregates
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Left => _, Right => _, _}
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateFunction
import org.apache.spark.sql.catalyst.expressions.aggregate.DeclarativeAggregate
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.LeafExecNode
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.execution.metric.SQLMetrics
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String
import scala.collection.JavaConverters._

/** An aggregate as the device computes it: see [[GpuAggregateStrategy]]. */
case class GpuAggregateSpec(
    table: org.apache.iceberg.Table,
    filter: IcebergExpression,
    groupExpressions: Seq[Expression],
    groupColumns: Seq[String],
    bucket: Option[(String, Int)],
    sumColumns: Seq[String],
    values: Seq[DeviceValue],
    aggregates: Map[AggregateExpression, DeviceAggregate],
    resultExpressions: Seq[NamedExpression]) {

  /** The result rows: each output expression with its aggregates and group columns filled in. */
  def run(): Seq[InternalRow] = {
    // In bucket mode the values are the sum columns, in order, so the totals line up the same.
    val result = GpuAggregates.groups(
      org.apache.spark.sql.SparkSession.active,
      table,
      filter,
      groupColumns.asJava,
      bucket.map(_._1).orNull,
      bucket.map(_._2).getOrElse(0),
      sumColumns.asJava,
      values.asJava)
    val groups = result.asScala.toVector.map { case (key, totals) =>
      (key.asScala.toVector, totals)
    }
    if (groups.isEmpty && groupExpressions.isEmpty) {
      // A global aggregate of no rows is one row: counts 0, sums and averages NULL.
      Seq(row(Seq.empty, null))
    } else {
      groups.map { case (key, totals) => row(key, totals) }
    }
  }

  private def row(key: Seq[AnyRef], totals: Array[Double]): InternalRow = {
    val rows = if (totals == null) 0L else totals(values.size).toLong
    // The rows where value v is not null: COUNT(v), AVG's divisor, 0 when SUM is NULL.
    def nonNull(v: Int): Long = if (totals == null) 0L else totals(values.size + 1 + v).toLong
    val keyLiterals = groupExpressions.zip(key).map { case (e, value) =>
      e.canonicalized -> Literal(sparkValue(e.dataType, value), e.dataType)
    }.toMap
    InternalRow.fromSeq(resultExpressions.map { output =>
      output.transformDown {
        case ae: AggregateExpression =>
          val value: Any = aggregates(ae) match {
            case RowCount => rows
            case CountOf(v) => nonNull(v)
            case SumOf(v) if nonNull(v) == 0 => null
            case AverageOf(v) if nonNull(v) == 0 => null
            case SumOf(v) => finish(ae.aggregateFunction, totals(v), nonNull(v))
            case AverageOf(v) => finish(ae.aggregateFunction, totals(v), nonNull(v))
          }
          Literal(value, ae.dataType)
        case e if keyLiterals.contains(e.canonicalized) => keyLiterals(e.canonicalized)
      }.eval(null)
    })
  }

  /**
   * The aggregate's result as Spark computes it from its buffer -- the sum, the row count, and for
   * a decimal SUM whether it saw no row -- here the device's: Spark's own evaluate expression
   * applies its result type, rounding and overflow rules. A sum the device holds exactly (a
   * decimal's unscaled value, an integer) goes into the buffer exactly.
   */
  private def finish(function: AggregateFunction, sum: Double, rows: Long): Any = {
    val buffer = function.asInstanceOf[DeclarativeAggregate]
    val attributes = buffer.aggBufferAttributes
    val sumType = attributes.head.dataType
    val sumValue: Any = sumType match {
      case DoubleType => sum
      case LongType => Math.round(sum)
      case d: DecimalType =>
        val scale = function.children.head.dataType.asInstanceOf[DecimalType].scale
        Decimal(java.math.BigDecimal.valueOf(Math.round(sum), scale), d.precision, d.scale)
      case other => throw new IllegalStateException(s"sum buffer of $other")
    }
    val literals: Map[ExprId, Literal] = attributes.map { a =>
      a.exprId -> (a.name match {
        case "sum" => Literal(sumValue, sumType)
        case "count" => Literal(rows, LongType)
        case "isEmpty" => Literal(false, BooleanType)
        case other => throw new IllegalStateException(s"buffer attribute $other of $function")
      })
    }.toMap
    buffer.evaluateExpression.transform {
      case a: AttributeReference if literals.contains(a.exprId) => literals(a.exprId)
    }.eval(null)
  }

  /** A device group key -- Long, String or null -- as Spark's internal value. */
  private def sparkValue(dataType: DataType, value: AnyRef): Any = (dataType, value) match {
    case (_, null) => null
    case (IntegerType | DateType, n: java.lang.Number) => n.intValue()
    case (LongType | TimestampType, n: java.lang.Number) => n.longValue()
    case (StringType, s: String) => UTF8String.fromString(s)
    case (d: DecimalType, n: java.lang.Number) =>
      Decimal.createUnsafe(n.longValue(), d.precision, d.scale)
    case (t, v) => throw new IllegalStateException(s"group key $v for $t")
  }

  override def toString: String =
    s"${table.name()} filter=$filter group=${groupColumns.mkString("[", ", ", "]")} " +
      bucket.map { case (c, n) => s"bucket($n, $c) " }.getOrElse("") +
      s"values=${values.mkString("[", ", ", "]")}"
}

/**
 * A grouped aggregate over an Iceberg table, computed by one device scan -- in the driver in local
 * mode, else in a one-task job on an executor -- with the result rows built in the driver; when the
 * device cannot run it after all (a group string over 8 bytes, say), the aggregate is planned
 * and run on the CPU instead. The spec and plan are the driver's, where it runs.
 */
case class GpuAggregateExec(
    output: Seq[Attribute],
    @transient spec: GpuAggregateSpec,
    @transient logical: LogicalPlan)
    extends LeafExecNode {

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "deviceTime" -> SQLMetrics.createTimingMetric(sparkContext, "device aggregate time"))

  override protected def doExecute(): RDD[InternalRow] = {
    val start = System.nanoTime()
    val rows =
      try {
        Some(spec.run())
      } catch {
        case e: RuntimeException =>
          logWarning(s"GPU aggregate failed, aggregating on the CPU: ${e.getMessage}")
          None
      }
    rows match {
      case Some(result) =>
        longMetric("deviceTime") += (System.nanoTime() - start) / 1000000
        longMetric("numOutputRows") += result.size
        val toUnsafe = UnsafeProjection.create(schema)
        sparkContext.parallelize(result.map(r => toUnsafe(r).copy(): InternalRow).toVector, 1)
      case None => cpu()
    }
  }

  private def cpu(): RDD[InternalRow] = {
    GpuAggregateStrategy.planningFallback.set(true)
    try {
      Dataset.ofRows(session, logical).queryExecution.toRdd
    } finally {
      GpuAggregateStrategy.planningFallback.set(false)
    }
  }

  override def simpleString(maxFields: Int): String = s"GpuIcebergAggregate $spec"
}
