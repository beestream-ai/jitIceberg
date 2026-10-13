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

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Left => _, Right => _, _}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Count, Sum}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.types._

/**
 * An aggregate without group keys over a UNION ALL (TPC-DS q23a's `sum(sales)` over catalog and
 * web sales): each branch is an aggregate of its own, planned for the device like any other (a
 * star, say), and the branches' partial results are combined on the driver. SUM and COUNT only,
 * whose partials add up: a SUM is the sum of the branches' non-null sums (null when all are), a
 * COUNT the sum of the counts.
 */
object UnionAggregates {
  import GpuAggregateStrategy.traverse

  def spec(
      agg: Aggregate,
      branchSpec: Aggregate => Either[String, DeviceAggregateSpec]): Either[String, GpuUnionAggregateSpec] =
    agg.child match {
      case union: Union if agg.groupingExpressions.isEmpty && !union.byName && !union.allowMissingCol =>
        val aggregates = agg.aggregateExpressions.flatMap(_.collect { case ae: AggregateExpression => ae }).distinct
        for {
          _ <- traverse(aggregates) { ae =>
            ae.aggregateFunction match {
              case _ if ae.isDistinct || ae.filter.isDefined => Left(s"DISTINCT or FILTER: $ae")
              case _: Sum | _: Count => Right(())
              case other => Left(s"over a union: $other")
            }
          }
          branches <- traverse(union.children) { child =>
            // The union's output attributes are its first child's; each branch reads its own.
            val byPosition = AttributeMap(union.output.zip(child.output))
            val parts = aggregates.zipWithIndex.map { case (ae, k) =>
              Alias(ae.transform { case a: Attribute if byPosition.contains(a) => byPosition(a) }, s"part_$k")()
            }
            val branch = Aggregate(Nil, parts, child)
            branchSpec(branch).left.map(reason => s"branch: $reason").map(branch -> _)
          }
        } yield GpuUnionAggregateSpec(branches, aggregates, agg.aggregateExpressions)
      case _: Union => Left("a union: group keys or by name")
      case other => Left(s"not a union: ${other.nodeName}")
    }
}

/** An aggregate over a union as [[UnionAggregates]] plans it: each branch's, then their sums. */
case class GpuUnionAggregateSpec(
    branches: Seq[(Aggregate, DeviceAggregateSpec)],
    aggregates: Seq[AggregateExpression],
    resultExpressions: Seq[NamedExpression])
    extends DeviceAggregateSpec {

  override def run(): Seq[InternalRow] = {
    val partials = branches.map { case (branch, spec) =>
      try spec.run().head
      catch {
        case e: RuntimeException =>
          // The branch plans again elsewhere (a subquery's own aggregate): decline it then.
          spec match {
            case _: GpuStarAggregateSpec => StarAggregates.remember(branch)
            case _ =>
          }
          throw e
      }
    }
    val totals = aggregates.zipWithIndex.map { case (ae, k) =>
      val dataType = ae.dataType
      val values = partials.flatMap(row => if (row.isNullAt(k)) None else Some(row.get(k, dataType)))
      val total: Any = ae.aggregateFunction match {
        case _: Count => values.map(_.asInstanceOf[Long]).sum
        case _ if values.isEmpty => null
        case _ =>
          dataType match {
            case LongType => values.map(_.asInstanceOf[Long]).sum
            case DoubleType => values.map(_.asInstanceOf[Double]).sum
            case d: DecimalType =>
              val sum = values.map(_.asInstanceOf[Decimal]).reduce(_ + _)
              // Spark's SUM is null when its result overflows the type.
              if (sum.changePrecision(d.precision, d.scale)) sum else null
            case other => throw new UnsupportedOperationException(s"SUM of $other over a union")
          }
      }
      ae.canonicalized -> Literal(total, dataType)
    }.toMap
    val values = resultExpressions.map { e =>
      e.transform { case ae: AggregateExpression => totals(ae.canonicalized) }.eval(InternalRow.empty)
    }
    Seq(InternalRow.fromSeq(values))
  }
}
