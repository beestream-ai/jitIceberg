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
package org.apache.iceberg.tornadovm.spark;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.Snapshot;
import org.apache.iceberg.Table;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.spark.source.SerializableTableWithSize;
import org.apache.iceberg.tornadovm.DeviceDimension;
import org.apache.iceberg.tornadovm.DeviceMergeOnReadScan;
import org.apache.iceberg.tornadovm.DeviceProbe;
import org.apache.iceberg.tornadovm.DeviceValue;
import org.apache.iceberg.tornadovm.GroupedAggregate;
import org.apache.iceberg.types.Types;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

/**
 * Grouped sums on the device for Spark's aggregate planning (GpuAggregateStrategy): one device scan
 * a table snapshot and filter, kept across queries -- its plans, buffers and kernels -- for the few
 * most recently used, as each holds device memory.
 */
public final class GpuAggregates {
  // A TPC-DS query reads up to six fact tables (q77: three channels' sales and returns), each its
  // own scan; their host buffers are only resident where touched (LazyZeroes).
  private static final int KEPT = Integer.getInteger("iceberg.tornadovm.keptScans", 8);

  private static final Map<String, DeviceMergeOnReadScan> SCANS =
      new LinkedHashMap<>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, DeviceMergeOnReadScan> eldest) {
          if (size() > KEPT) {
            // Closed once the map's lock is released, under the scan's own: a query may still be
            // running on it.
            EVICTED.add(eldest.getValue());
            return true;
          }
          return false;
        }
      };

  // Scans dropped from SCANS, to close outside its lock (guarded by SCANS).
  private static final List<DeviceMergeOnReadScan> EVICTED = new ArrayList<>();

  /**
   * Runs {@code work} on the kept scan of a table snapshot and filter, holding that scan only:
   * aggregates on different scans run side by side (a query's subqueries over several fact tables),
   * ones on the same scan in turn.
   */
  private static <T> T withScan(
      Table table, Expression filter, java.util.function.Function<DeviceMergeOnReadScan, T> work) {
    while (true) {
      DeviceMergeOnReadScan scan = scan(table, filter);
      synchronized (scan) {
        if (!scan.isClosed()) {
          return work.apply(scan);
        }
      }
    }
  }

  private static void closeEvicted() {
    List<DeviceMergeOnReadScan> closing;
    synchronized (SCANS) {
      closing = new ArrayList<>(EVICTED);
      EVICTED.clear();
    }
    for (DeviceMergeOnReadScan scan : closing) {
      synchronized (scan) {
        scan.close();
      }
    }
  }

  static {
    ExitCleanup.register(
        () -> {
          synchronized (SCANS) {
            SCANS.values().forEach(DeviceMergeOnReadScan::close);
          }
        });
  }

  // What a snapshot's file metrics say of a column; a snapshot never changes.
  private static final Map<String, Boolean> COLUMN_FACTS =
      new java.util.concurrent.ConcurrentHashMap<>();

  private GpuAggregates() {}

  /** Frees every kept scan's device memory -- for a compaction, which needs the room. */
  static void releaseAll() {
    synchronized (SCANS) {
      EVICTED.addAll(SCANS.values());
      SCANS.clear();
    }
    closeEvicted();
  }

  private static String factKey(Table table, String column, String fact) {
    Snapshot snapshot = table.currentSnapshot();
    return table.location()
        + "@"
        + (snapshot == null ? "empty" : snapshot.snapshotId())
        + "|"
        + column
        + "|"
        + fact;
  }

  /**
   * The groups of an aggregate -- by the given columns, or by {@code bucket(buckets, bucketColumn)}
   * when that is not null -- computed where the GPU is: in this JVM in local mode, else in a
   * one-task Spark job on an executor, the table travelling as a {@link SerializableTableWithSize}.
   */
  public static Map<List<Object>, double[]> groups(
      SparkSession spark,
      Table table,
      Expression filter,
      List<String> groupColumns,
      String bucketColumn,
      int buckets,
      List<String> sumColumns,
      List<DeviceValue> values) {
    if (spark.sparkContext().isLocal()) {
      return groupsHere(table, filter, groupColumns, bucketColumn, buckets, sumColumns, values);
    }
    Table shipped = SerializableTableWithSize.copyOf(table);
    List<String> groupBy = new ArrayList<>(groupColumns);
    List<String> sums = new ArrayList<>(sumColumns);
    List<DeviceValue> sumValues = new ArrayList<>(values);
    return JavaSparkContext.fromSparkContext(spark.sparkContext())
        .parallelize(List.of(0), 1)
        .map(
            ignored -> groupsHere(shipped, filter, groupBy, bucketColumn, buckets, sums, sumValues))
        .collect()
        .get(0);
  }

  private static Map<List<Object>, double[]> groupsHere(
      Table table,
      Expression filter,
      List<String> groupColumns,
      String bucketColumn,
      int buckets,
      List<String> sumColumns,
      List<DeviceValue> values) {
    Map<List<Object>, double[]> groups =
        bucketColumn != null
            ? aggregateByBucket(table, filter, bucketColumn, buckets, sumColumns)
            : aggregate(table, filter, groupColumns, values).groups();
    // Serializable keys and map, for the trip back from an executor.
    LinkedHashMap<List<Object>, double[]> copy = new LinkedHashMap<>();
    groups.forEach((key, totals) -> copy.put(new ArrayList<>(key), totals));
    return copy;
  }

  /**
   * The groups of a star aggregate ({@link DeviceMergeOnReadScan#starAggregate}), in this JVM: the
   * scan is kept per fact table snapshot and filter, the dimensions are the run's.
   */
  public static Map<List<Object>, double[]> starGroups(
      Table table,
      Expression filter,
      List<DeviceDimension> dimensions,
      List<DeviceDimension.Key> keys,
      List<DeviceValue> values) {
    return starGroups(table, filter, dimensions, keys, values, null);
  }

  /** {@link #starGroups(Table, Expression, List, List, List)} joined with a probe (or null). */
  public static Map<List<Object>, double[]> starGroups(
      Table table,
      Expression filter,
      List<DeviceDimension> dimensions,
      List<DeviceDimension.Key> keys,
      List<DeviceValue> values,
      DeviceProbe probe) {
    return withScan(
        table,
        filter,
        scan -> {
          Map<List<Object>, double[]> groups =
              scan.starAggregate(dimensions, keys, values, probe).groups();
          lastStarMetrics = String.valueOf(scan.metrics());
          return groups;
        });
  }

  private static volatile String lastStarMetrics = "";

  /** The device scan's metrics of the last star aggregate, for the planner's log. */
  public static String lastStarMetrics() {
    return lastStarMetrics;
  }

  /**
   * Sums of the values and the row count per group, over the live rows of the table's current
   * snapshot that pass the filter. Throws when the device cannot run it (a group string longer than
   * 8 bytes, a filter it lacks); the caller then aggregates on the CPU.
   */
  public static GroupedAggregate aggregate(
      Table table, Expression filter, List<String> groupColumns, List<DeviceValue> values) {
    return withScan(table, filter, scan -> scan.aggregate(groupColumns, values));
  }

  /**
   * Sums of the columns and the row count per {@code bucket(buckets, column)} of a required LONG
   * column -- Iceberg's bucket transform, evaluated on the device -- as {@link
   * GroupedAggregate#groups()} gives them: keyed by the bucket, sums, the row count, then each
   * column's non-null count.
   */
  public static Map<List<Object>, double[]> aggregateByBucket(
      Table table, Expression filter, String column, int buckets, List<String> sumColumns) {
    org.apache.iceberg.tornadovm.AggregateResult result;
    result =
        withScan(
            table,
            filter,
            scan -> scan.aggregateByBucket(column, buckets, sumColumns.toArray(new String[0])));
    Map<List<Object>, double[]> groups = new LinkedHashMap<>();
    result
        .groups()
        .forEach(
            (bucket, group) -> {
              int n = sumColumns.size();
              double[] totals = java.util.Arrays.copyOf(group.sums(), 2 * n + 1);
              totals[n] = group.count();
              // The summed columns are null-free (the strategy checks): each counts every row.
              java.util.Arrays.fill(totals, n + 1, 2 * n + 1, group.count());
              groups.put(List.of((long) bucket), totals);
            });
    return groups;
  }

  private static DeviceMergeOnReadScan scan(Table table, Expression filter) {
    Snapshot snapshot = table.currentSnapshot();
    String key =
        table.location()
            + "@"
            + (snapshot == null ? "empty" : snapshot.snapshotId())
            + "|"
            + filter;
    DeviceMergeOnReadScan scan;
    synchronized (SCANS) {
      scan = SCANS.computeIfAbsent(key, ignored -> DeviceMergeOnReadScan.of(table, filter));
    }
    closeEvicted();
    return scan;
  }

  /**
   * Whether a sum of the integer or decimal column over the table, accumulated in doubles, is
   * exact: the files' row count times the largest magnitude their bounds allow -- a decimal's
   * unscaled -- stays under 2^53.
   */
  public static boolean exactSum(Table table, String column) {
    return exactProduct(table, List.of(column));
  }

  /**
   * Whether the device's doubles hold the sum of the product of these int, bigint or decimal
   * columns (decimals as unscaled values) exactly, by the files' bounds: rows times the product of
   * the columns' largest magnitudes below 2^53.
   */
  public static boolean exactProduct(Table table, List<String> columns) {
    double bound = rows(table);
    for (String column : columns) {
      bound *= largest(table, column);
    }
    return bound < 0x1p53;
  }

  /**
   * Whether an int or bigint column's values, by the files' bounds, span at most {@code perRow}
   * keys a row of the table (or {@code floor} keys): a dimension key a star aggregate can index by.
   * False when the bounds are unknown.
   */
  public static boolean denseKeys(Table table, String column, double perRow, double floor) {
    return COLUMN_FACTS.computeIfAbsent(
        factKey(table, column, "dense " + perRow + " " + floor),
        ignored -> computeDenseKeys(table, column, perRow, floor));
  }

  private static boolean computeDenseKeys(Table table, String column, double perRow, double floor) {
    Types.NestedField field = table.schema().findField(column);
    if (field == null) {
      return false;
    }
    double low = Double.POSITIVE_INFINITY;
    double high = Double.NEGATIVE_INFINITY;
    try (CloseableIterable<FileScanTask> tasks = table.newScan().includeColumnStats().planFiles()) {
      for (FileScanTask task : tasks) {
        Map<Integer, java.nio.ByteBuffer> lower = task.file().lowerBounds();
        Map<Integer, java.nio.ByteBuffer> upper = task.file().upperBounds();
        if (lower == null || upper == null) {
          return false;
        }
        java.nio.ByteBuffer min = lower.get(field.fieldId());
        java.nio.ByteBuffer max = upper.get(field.fieldId());
        if (min == null || max == null) {
          return false;
        }
        low =
            Math.min(
                low,
                ((Number) org.apache.iceberg.types.Conversions.fromByteBuffer(field.type(), min))
                    .doubleValue());
        high =
            Math.max(
                high,
                ((Number) org.apache.iceberg.types.Conversions.fromByteBuffer(field.type(), max))
                    .doubleValue());
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return high < low || high - low + 1 <= Math.max(floor, perRow * rows(table));
  }

  private static double rows(Table table) {
    Snapshot snapshot = table.currentSnapshot();
    return snapshot == null
        ? 0
        : Double.parseDouble(snapshot.summary().getOrDefault("total-records", "Infinity"));
  }

  // A column's largest magnitude in a snapshot, by its files' bounds; infinite when unknown.
  private static final Map<String, Double> LARGEST = new java.util.concurrent.ConcurrentHashMap<>();

  private static double largest(Table table, String column) {
    return LARGEST.computeIfAbsent(
        factKey(table, column, "largest"), ignored -> computeLargest(table, column));
  }

  private static double computeLargest(Table table, String column) {
    Types.NestedField field = table.schema().findField(column);
    if (field == null) {
      return Double.POSITIVE_INFINITY;
    }
    double largest = 0;
    try (CloseableIterable<FileScanTask> tasks = table.newScan().includeColumnStats().planFiles()) {
      for (FileScanTask task : tasks) {
        Map<Integer, java.nio.ByteBuffer> lower = task.file().lowerBounds();
        Map<Integer, java.nio.ByteBuffer> upper = task.file().upperBounds();
        if (lower == null || upper == null) {
          return Double.POSITIVE_INFINITY;
        }
        java.nio.ByteBuffer low = lower.get(field.fieldId());
        java.nio.ByteBuffer high = upper.get(field.fieldId());
        if (low == null || high == null) {
          return Double.POSITIVE_INFINITY;
        }
        Number min = org.apache.iceberg.types.Conversions.fromByteBuffer(field.type(), low);
        Number max = org.apache.iceberg.types.Conversions.fromByteBuffer(field.type(), high);
        largest = Math.max(largest, Math.max(magnitude(min), magnitude(max)));
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return largest;
  }

  private static double magnitude(Number value) {
    return Math.abs(
        value instanceof java.math.BigDecimal decimal
            ? decimal.unscaledValue().doubleValue()
            : value.doubleValue());
  }

  /**
   * Whether no data file of the table's current snapshot holds a null in the column, by the files'
   * metrics: a required column, or every file counting zero nulls.
   */
  public static boolean nullFree(Table table, String column) {
    return COLUMN_FACTS.computeIfAbsent(
        factKey(table, column, "nullFree"), ignored -> computeNullFree(table, column));
  }

  private static boolean computeNullFree(Table table, String column) {
    Types.NestedField field = table.schema().findField(column);
    if (field == null) {
      return false;
    }
    if (field.isRequired()) {
      return true;
    }
    try (CloseableIterable<FileScanTask> tasks = table.newScan().includeColumnStats().planFiles()) {
      for (FileScanTask task : tasks) {
        Map<Integer, Long> counts = task.file().nullValueCounts();
        Long count = counts == null ? null : counts.get(field.fieldId());
        if (count == null || count != 0) {
          return false;
        }
      }
      return true;
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }
}
