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

/** What [[GpuAggregateExec]] runs: an aggregate's result rows, computed with the device. */
trait DeviceAggregateSpec {
  def run(): Seq[InternalRow]
}

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
    resultExpressions: Seq[NamedExpression])
    extends DeviceAggregateSpec {

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

  private lazy val results =
    new DeviceResults(groupExpressions, values.size, aggregates, resultExpressions)

  private def row(key: Seq[AnyRef], totals: Array[Double]): InternalRow =
    results.row(
      groupExpressions.zip(key).map { case (e, value) => DeviceResults.sparkValue(e.dataType, value) },
      totals)

  override def toString: String =
    s"${table.name()} filter=$filter group=${groupColumns.mkString("[", ", ", "]")} " +
      bucket.map { case (c, n) => s"bucket($n, $c) " }.getOrElse("") +
      s"values=${values.mkString("[", ", ", "]")}"
}

/**
 * The rows of a device aggregate, from its groups' sums and counts: the output expressions are
 * bound once -- each group expression and aggregate a slot of an input row, each aggregate's
 * evaluation (Spark's own, over its buffer) bound to the buffer the device's totals fill -- and
 * evaluated per group.
 */
class DeviceResults(
    groupExpressions: Seq[Expression],
    values: Int,
    aggregates: Map[AggregateExpression, DeviceAggregate],
    resultExpressions: Seq[NamedExpression]) {
  private val ordered = aggregates.keys.toVector
  private val slots = ordered.zipWithIndex.toMap
  private val groups = groupExpressions.size
  private val keySlots = groupExpressions.zipWithIndex.map { case (e, i) => e.canonicalized -> i }.toMap

  private val outputs: Seq[Expression] = resultExpressions.map { output =>
    output.transformDown {
      case ae: AggregateExpression => BoundReference(groups + slots(ae), ae.dataType, nullable = true)
      case e if keySlots.contains(e.canonicalized) =>
        BoundReference(keySlots(e.canonicalized), e.dataType, nullable = true)
    }
  }

  /** Per aggregate with a sum: its buffer's fields and Spark's evaluation bound to them. */
  private case class Finish(sumType: DataType, scale: Int, fields: Seq[String], evaluate: Expression)

  private val finishes: Vector[Finish] = ordered.map { ae =>
    aggregates(ae) match {
      case SumOf(_) | AverageOf(_) | SumAll(_) =>
        val buffer = ae.aggregateFunction.asInstanceOf[DeclarativeAggregate]
        val attributes = buffer.aggBufferAttributes
        val scale = ae.aggregateFunction.children.head.dataType match {
          case d: DecimalType => d.scale
          case _ => 0
        }
        Finish(
          attributes.head.dataType,
          scale,
          attributes.map(_.name),
          BindReferences.bindReference(buffer.evaluateExpression, attributes))
      case _ => null
    }
  }

  /**
   * One result row. {@code key} holds the group expressions' values, as Spark's internal values;
   * {@code totals} the group's sums, its row count and each value's non-null count, or null for a
   * global aggregate of no rows (counts 0, sums and averages NULL).
   */
  def row(key: Seq[Any], totals: Array[Double]): InternalRow = {
    val rows = if (totals == null) 0L else totals(values).toLong
    // The rows where value v is not null: COUNT(v), AVG's divisor, 0 when SUM is NULL.
    def nonNull(v: Int): Long = if (totals == null) 0L else totals(values + 1 + v).toLong
    val input = new Array[Any](groups + ordered.size)
    key.copyToArray(input)
    var a = 0
    while (a < ordered.size) {
      input(groups + a) = aggregates(ordered(a)) match {
        case RowCount => rows
        case CountOf(v) => nonNull(v)
        case SumOf(v) if nonNull(v) == 0 => null
        case AverageOf(v) if nonNull(v) == 0 => null
        case SumOf(v) => finish(finishes(a), totals(v), nonNull(v))
        case SumAll(_) if rows == 0 => null
        case SumAll(v) => finish(finishes(a), totals(v), rows)
        case AverageOf(v) => finish(finishes(a), totals(v), nonNull(v))
      }
      a += 1
    }
    val row = new GenericInternalRow(input)
    InternalRow.fromSeq(outputs.map(_.eval(row)))
  }

  /**
   * The aggregate's result as Spark computes it from its buffer -- the sum, the row count, and for
   * a decimal SUM whether it saw no row -- here the device's: Spark's own evaluate expression
   * applies its result type, rounding and overflow rules. A sum the device holds exactly (a
   * decimal's unscaled value, an integer) goes into the buffer exactly.
   */
  private def finish(f: Finish, sum: Double, rows: Long): Any = {
    val sumValue: Any = f.sumType match {
      case DoubleType => sum
      case LongType => Math.round(sum)
      case d: DecimalType =>
        Decimal(java.math.BigDecimal.valueOf(Math.round(sum), f.scale), d.precision, d.scale)
      case other => throw new IllegalStateException(s"sum buffer of $other")
    }
    val buffer = f.fields.map {
      case "sum" => sumValue
      case "count" => rows
      case "isEmpty" => false
      case other => throw new IllegalStateException(s"buffer attribute $other")
    }
    f.evaluate.eval(new GenericInternalRow(buffer.toArray[Any]))
  }
}

object DeviceResults {
  /** A device group key -- Long, String or null -- as Spark's internal value. */
  def sparkValue(dataType: DataType, value: AnyRef): Any = (dataType, value) match {
    case (_, null) => null
    case (IntegerType | DateType, n: java.lang.Number) => n.intValue()
    case (LongType | TimestampType, n: java.lang.Number) => n.longValue()
    case (StringType, s: String) => UTF8String.fromString(s)
    case (d: DecimalType, n: java.lang.Number) =>
      Decimal.createUnsafe(n.longValue(), d.precision, d.scale)
    case (t, v) => throw new IllegalStateException(s"group key $v for $t")
  }
}

/**
 * A grouped aggregate over an Iceberg table, computed by one device scan -- in the driver in local
 * mode, else in a one-task job on an executor -- with the result rows built in the driver; when the
 * device cannot run it after all (a group string over 8 bytes, say), the aggregate is planned
 * and run on the CPU instead. The spec and plan are the driver's, where it runs.
 */
case class GpuAggregateExec(
    output: Seq[Attribute],
    @transient spec: DeviceAggregateSpec,
    @transient logical: LogicalPlan)
    extends LeafExecNode {

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "deviceTime" -> SQLMetrics.createTimingMetric(sparkContext, "device aggregate time"))

  /**
   * Starts the aggregate on a driver thread and returns at once: Spark materializes a query's
   * stages one after another, so a subquery computed here in line would hold back the others --
   * TPC-DS q77's six star aggregates ran in series while Spark's own plans overlap. The one-row
   * RDD's task waits for the result. Local mode only (the strategy's precondition): the task
   * finds the result in this JVM. When the device cannot run it after all, the same thread
   * aggregates on the CPU and collects the rows.
   */
  override protected def doExecute(): RDD[InternalRow] = {
    val id = GpuAggregateExec.start(this)
    sparkContext.parallelize(Seq(id), 1).mapPartitions(ids => GpuAggregateExec.await(ids.next()))
  }

  private[gpu] def compute(): Seq[InternalRow] = {
    val start = System.nanoTime()
    val rows =
      try {
        Some(spec.run())
      } catch {
        case e: RuntimeException =>
          logWarning(s"GPU aggregate failed, aggregating on the CPU: ${e.getMessage}")
          spec match {
            case star: GpuStarAggregateSpec => StarAggregates.remember(star)
            case _ =>
          }
          None
      }
    rows match {
      case Some(result) =>
        longMetric("deviceTime") += (System.nanoTime() - start) / 1000000
        longMetric("numOutputRows") += result.size
        val toUnsafe = UnsafeProjection.create(schema)
        result.map(r => toUnsafe(r).copy(): InternalRow).toVector
      case None => cpu()
    }
  }

  private def cpu(): Seq[InternalRow] = {
    GpuAggregateStrategy.planningFallback.set(true)
    try {
      Dataset.ofRows(session, logical).queryExecution.executedPlan.executeCollect().toVector
    } finally {
      GpuAggregateStrategy.planningFallback.set(false)
    }
  }

  override def simpleString(maxFields: Int): String = s"GpuIcebergAggregate $spec"
}

object GpuAggregateExec {
  private val threads = java.util.concurrent.Executors.newCachedThreadPool { runnable =>
    val thread = new Thread(runnable, "gpu-aggregate")
    thread.setDaemon(true)
    thread
  }
  private val ids = new java.util.concurrent.atomic.AtomicLong()

  // Results by id; kept for the last few so that an RDD computed again still finds its rows.
  private val results = java.util.Collections.synchronizedMap(
    new java.util.LinkedHashMap[Long, java.util.concurrent.Future[Seq[InternalRow]]](64, 0.75f, false) {
      override def removeEldestEntry(
          eldest: java.util.Map.Entry[Long, java.util.concurrent.Future[Seq[InternalRow]]]): Boolean =
        size() > 64
    })

  private[gpu] def start(exec: GpuAggregateExec): Long = {
    val id = ids.incrementAndGet()
    val session = exec.session
    results.put(
      id,
      threads.submit { () =>
        org.apache.spark.sql.SparkSession.setActiveSession(session)
        exec.compute()
      })
    id
  }

  private[gpu] def await(id: Long): Iterator[InternalRow] =
    Option(results.get(id))
      .getOrElse(throw new IllegalStateException(s"GPU aggregate $id is no longer kept"))
      .get()
      .iterator
}
