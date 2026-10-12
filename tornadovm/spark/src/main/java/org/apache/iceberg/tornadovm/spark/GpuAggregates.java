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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.Snapshot;
import org.apache.iceberg.Table;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.tornadovm.DeviceMergeOnReadScan;
import org.apache.iceberg.tornadovm.DeviceValue;
import org.apache.iceberg.tornadovm.GroupedAggregate;
import org.apache.iceberg.types.Types;

/**
 * Grouped sums on the device for Spark's aggregate planning (GpuAggregateStrategy): one device scan
 * a table snapshot and filter, kept across queries -- its plans, buffers and kernels -- for the few
 * most recently used, as each holds device memory.
 */
public final class GpuAggregates {
  private static final int KEPT = 4;

  private static final Map<String, DeviceMergeOnReadScan> SCANS =
      new LinkedHashMap<>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, DeviceMergeOnReadScan> eldest) {
          if (size() > KEPT) {
            eldest.getValue().close();
            return true;
          }
          return false;
        }
      };

  // Spark gives the driver no lifecycle hook for a cache like this; the device is freed at exit.
  @SuppressWarnings("ShutdownHook")
  private static void closeAtExit() {
    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  synchronized (SCANS) {
                    SCANS.values().forEach(DeviceMergeOnReadScan::close);
                  }
                }));
  }

  static {
    closeAtExit();
  }

  // What a snapshot's file metrics say of a column; a snapshot never changes.
  private static final Map<String, Boolean> COLUMN_FACTS =
      new java.util.concurrent.ConcurrentHashMap<>();

  private GpuAggregates() {}

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
   * Sums of the values and the row count per group, over the live rows of the table's current
   * snapshot that pass the filter. Throws when the device cannot run it (a group string longer than
   * 8 bytes, a filter it lacks); the caller then aggregates on the CPU.
   */
  public static GroupedAggregate aggregate(
      Table table, Expression filter, List<String> groupColumns, List<DeviceValue> values) {
    Snapshot snapshot = table.currentSnapshot();
    String key =
        table.location()
            + "@"
            + (snapshot == null ? "empty" : snapshot.snapshotId())
            + "|"
            + filter;
    synchronized (SCANS) {
      DeviceMergeOnReadScan scan =
          SCANS.computeIfAbsent(key, ignored -> DeviceMergeOnReadScan.of(table, filter));
      return scan.aggregate(groupColumns, values);
    }
  }

  /**
   * Sums of the columns and the row count per {@code bucket(buckets, column)} of a required LONG
   * column -- Iceberg's bucket transform, evaluated on the device -- as {@link
   * GroupedAggregate#groups()} gives them: keyed by the bucket, sums then the row count.
   */
  public static Map<List<Object>, double[]> aggregateByBucket(
      Table table, Expression filter, String column, int buckets, List<String> sumColumns) {
    org.apache.iceberg.tornadovm.AggregateResult result;
    synchronized (SCANS) {
      result =
          scan(table, filter).aggregateByBucket(column, buckets, sumColumns.toArray(new String[0]));
    }
    Map<List<Object>, double[]> groups = new LinkedHashMap<>();
    result
        .groups()
        .forEach(
            (bucket, group) -> {
              double[] totals = java.util.Arrays.copyOf(group.sums(), sumColumns.size() + 1);
              totals[sumColumns.size()] = group.count();
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
    return SCANS.computeIfAbsent(key, ignored -> DeviceMergeOnReadScan.of(table, filter));
  }

  /**
   * Whether a sum of the integer or decimal column over the table, accumulated in doubles, is
   * exact: the files' row count times the largest magnitude their bounds allow -- a decimal's
   * unscaled -- stays under 2^53.
   */
  public static boolean exactSum(Table table, String column) {
    return COLUMN_FACTS.computeIfAbsent(
        factKey(table, column, "exact"), ignored -> computeExactSum(table, column));
  }

  private static boolean computeExactSum(Table table, String column) {
    Types.NestedField field = table.schema().findField(column);
    if (field == null) {
      return false;
    }
    double rows = 0;
    double largest = 0;
    try (CloseableIterable<FileScanTask> tasks = table.newScan().includeColumnStats().planFiles()) {
      for (FileScanTask task : tasks) {
        Map<Integer, java.nio.ByteBuffer> lower = task.file().lowerBounds();
        Map<Integer, java.nio.ByteBuffer> upper = task.file().upperBounds();
        if (lower == null || upper == null) {
          return false;
        }
        java.nio.ByteBuffer low = lower.get(field.fieldId());
        java.nio.ByteBuffer high = upper.get(field.fieldId());
        if (low == null || high == null) {
          return false;
        }
        Number min = org.apache.iceberg.types.Conversions.fromByteBuffer(field.type(), low);
        Number max = org.apache.iceberg.types.Conversions.fromByteBuffer(field.type(), high);
        largest = Math.max(largest, Math.max(magnitude(min), magnitude(max)));
        rows += task.file().recordCount();
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return rows * largest < 0x1p53;
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
