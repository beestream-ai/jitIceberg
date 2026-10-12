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
    val result = bucket match {
      case Some((column, buckets)) =>
        GpuAggregates.aggregateByBucket(table, filter, column, buckets, sumColumns.asJava)
      case None =>
        GpuAggregates.aggregate(table, filter, groupColumns.asJava, values.asJava).groups()
    }
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
    val keyLiterals = groupExpressions.zip(key).map { case (e, value) =>
      e.canonicalized -> Literal(sparkValue(e.dataType, value), e.dataType)
    }.toMap
    InternalRow.fromSeq(resultExpressions.map { output =>
      output.transformDown {
        case ae: AggregateExpression =>
          val value: Any = aggregates(ae) match {
            case RowCount => rows
            case SumOf(_) if rows == 0 => null
            case SumOf(v) if ae.dataType == LongType => Math.round(totals(v))
            case SumOf(v) => totals(v)
            case AverageOf(_) if rows == 0 => null
            case AverageOf(v) => totals(v) / rows
          }
          Literal(value, ae.dataType)
        case e if keyLiterals.contains(e.canonicalized) => keyLiterals(e.canonicalized)
      }.eval(null)
    })
  }

  /** A device group key -- Long, String or null -- as Spark's internal value. */
  private def sparkValue(dataType: DataType, value: AnyRef): Any = (dataType, value) match {
    case (_, null) => null
    case (IntegerType | DateType, n: java.lang.Number) => n.intValue()
    case (LongType | TimestampType, n: java.lang.Number) => n.longValue()
    case (StringType, s: String) => UTF8String.fromString(s)
    case (t, v) => throw new IllegalStateException(s"group key $v for $t")
  }

  override def toString: String =
    s"${table.name()} filter=$filter group=${groupColumns.mkString("[", ", ", "]")} " +
      bucket.map { case (c, n) => s"bucket($n, $c) " }.getOrElse("") +
      s"values=${values.mkString("[", ", ", "]")}"
}

/**
 * A grouped aggregate over an Iceberg table, computed by one device scan in the driver; when the
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
