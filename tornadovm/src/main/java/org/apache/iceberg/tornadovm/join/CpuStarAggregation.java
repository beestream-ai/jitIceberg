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
package org.apache.iceberg.tornadovm.join;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.iceberg.arrow.vectorized.ColumnVector;
import org.apache.iceberg.arrow.vectorized.ColumnarBatch;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.relocated.com.google.common.collect.Maps;
import org.apache.iceberg.tornadovm.CpuMergeOnReadScan;
import org.apache.iceberg.tornadovm.DeviceValue;
import org.apache.iceberg.tornadovm.GroupedAggregate;

/**
 * {@link StarAggregation} on the CPU: the same dimensions (masks and codes by key), the same probe
 * (packed keys, a sorted build side, weights and build sums) and the same aggregation (dense by
 * codes where the device aggregates densely, hashed otherwise), over the CPU scan's live rows on
 * every core. It exists to measure what the GPU adds to the join techniques, apart from what the
 * techniques add to Spark's plans; the results are the device's, row for row.
 *
 * <p>Integer, long and double columns only (the benchmarks it serves); a decimal fails.
 */
public final class CpuStarAggregation {
  private CpuStarAggregation() {}

  public static GroupedAggregate run(
      CpuMergeOnReadScan scan,
      List<DeviceDimension> dimensions,
      List<DeviceDimension.Key> keys,
      List<DeviceValue> values) {
    return run(scan, dimensions, keys, values, null, List.of());
  }

  public static GroupedAggregate run(
      CpuMergeOnReadScan scan,
      List<DeviceDimension> dimensions,
      List<DeviceDimension.Key> keys,
      List<DeviceValue> values,
      DeviceProbe probe,
      List<StarAggregation.ProbeSum> probeSums) {
    Plan plan = new Plan(dimensions, keys, values, probe, probeSums);
    GroupedAggregate result = new GroupedAggregate(values.size() + probeSums.size());
    scan.visit(
        plan.columns, index -> new Visitor(plan, index), visitors -> plan.finish(visitors, result));
    return result;
  }

  /** What every visitor shares: the inputs and how they map onto the scan's columns. */
  private static final class Plan {
    final List<DeviceDimension> dimensions;
    final List<DeviceDimension.Key> keys;
    final List<DeviceValue> values;
    final DeviceProbe probe;
    final List<StarAggregation.ProbeSum> probeSums;
    final List<String> columns = Lists.newArrayList();
    // Dense grouping: every key a code, at most 2^20 groups (as the device).
    final int[] cardinalities;
    final int[] multipliers;
    final int denseGroups;
    final int width;

    Plan(
        List<DeviceDimension> dimensions,
        List<DeviceDimension.Key> keys,
        List<DeviceValue> values,
        DeviceProbe probe,
        List<StarAggregation.ProbeSum> probeSums) {
      this.dimensions = dimensions;
      this.keys = keys;
      this.values = values;
      this.probe = probe;
      this.probeSums = probeSums;
      dimensions.forEach(d -> add(d.factColumn()));
      keys.forEach(
          k -> add(k.isAttribute() ? dimensions.get(k.dimension()).factColumn() : k.column()));
      values.forEach(v -> v.factors().forEach(f -> add(f.column())));
      if (probe != null) {
        probe
            .parts()
            .forEach(
                p ->
                    add(p.isAttribute() ? dimensions.get(p.dimension()).factColumn() : p.column()));
        if (probe.column() != null) {
          add(probe.column());
        }
      }
      this.cardinalities = new int[keys.size()];
      this.multipliers = new int[keys.size()];
      long groups = keys.isEmpty() ? 0 : 1;
      for (int k = keys.size() - 1; k >= 0; k--) {
        DeviceDimension.Key key = keys.get(k);
        if (!key.isAttribute()) {
          groups = 0;
          break;
        }
        int[] codes = dimensions.get(key.dimension()).attributes().get(key.attribute());
        cardinalities[k] = 1 + Arrays.stream(codes).max().orElse(0);
        multipliers[k] = (int) groups;
        groups *= cardinalities[k];
        if (groups > (1 << 20)) {
          groups = 0;
          break;
        }
      }
      this.denseGroups = (int) groups;
      // Per group: the values' sums, the row count, the values' non-null counts.
      int n = values.size() + probeSums.size();
      this.width = 2 * n + 1;
    }

    private void add(String column) {
      if (!columns.contains(column)) {
        columns.add(column);
      }
    }

    void finish(List<Visitor> visitors, GroupedAggregate result) {
      int n = values.size() + probeSums.size();
      double[] sums = new double[n];
      double[] nonNull = new double[n];
      if (denseGroups > 0) {
        double[] total = new double[denseGroups * width];
        for (Visitor v : visitors) {
          for (int i = 0; i < total.length; i++) {
            total[i] += v.dense[i];
          }
        }
        for (int g = 0; g < denseGroups; g++) {
          double rows = total[g * width + n];
          if (rows == 0) {
            continue;
          }
          List<Object> key = Lists.newArrayListWithCapacity(keys.size());
          for (int k = 0; k < keys.size(); k++) {
            key.add((long) ((g / multipliers[k]) % cardinalities[k]));
          }
          System.arraycopy(total, g * width, sums, 0, n);
          System.arraycopy(total, g * width + n + 1, nonNull, 0, n);
          result.add(key, sums, rows, nonNull);
        }
      } else {
        for (Visitor v : visitors) {
          v.hashed.forEach(
              (key, totals) -> {
                System.arraycopy(totals, 0, sums, 0, n);
                System.arraycopy(totals, n + 1, nonNull, 0, n);
                result.add(key, sums, totals[n], nonNull);
              });
        }
      }
    }
  }

  /** One data file's live rows, on the scan thread that reads it. */
  private static final class Visitor implements CpuMergeOnReadScan.RowVisitor {
    private final Plan plan;
    private final int[] dimensionColumn;
    private final int[] keyColumn;
    private final int[][] valueColumn;
    private final int[] partColumn;
    private final int compareColumn;
    final double[] dense;
    final Map<List<Object>, double[]> hashed = Maps.newHashMap();
    private final double[] row;
    private final double[] rowSums;
    private final double[] rowCounts;
    private final long[] dimensionKey;

    Visitor(Plan plan, Map<String, Integer> index) {
      this.plan = plan;
      this.dimensionKey = new long[plan.dimensions.size()];
      this.dimensionColumn =
          plan.dimensions.stream().mapToInt(d -> index.get(d.factColumn())).toArray();
      this.keyColumn =
          plan.keys.stream()
              .mapToInt(
                  k ->
                      index.get(
                          k.isAttribute()
                              ? plan.dimensions.get(k.dimension()).factColumn()
                              : k.column()))
              .toArray();
      this.valueColumn = new int[plan.values.size()][];
      for (int v = 0; v < valueColumn.length; v++) {
        valueColumn[v] =
            plan.values.get(v).factors().stream().mapToInt(f -> index.get(f.column())).toArray();
      }
      this.partColumn =
          plan.probe == null
              ? new int[0]
              : plan.probe.parts().stream()
                  .mapToInt(
                      p ->
                          index.get(
                              p.isAttribute()
                                  ? plan.dimensions.get(p.dimension()).factColumn()
                                  : p.column()))
                  .toArray();
      this.compareColumn =
          plan.probe == null || plan.probe.column() == null ? -1 : index.get(plan.probe.column());
      this.dense = plan.denseGroups > 0 ? new double[plan.denseGroups * plan.width] : null;
      this.row = new double[plan.width];
      int sums = plan.probe == null ? 0 : plan.probe.sums().size();
      this.rowSums = new double[sums];
      this.rowCounts = new double[sums];
    }

    // How each column reads, resolved on its first value: 4 (getInt), 8 (getLong) or 'd'
    // (getDouble). Iceberg's accessor decides what works: a dictionary-encoded BIGINT page arrives
    // as
    // an IntVector of dictionary offsets whose accessor reads only getLong, so the vector's own
    // values
    // must never be read directly.
    private final int[] kind = new int[64];

    private int kind(ColumnVector vector, int column, int r) {
      if (kind[column] == 0) {
        int[] order =
            vector.getFieldVector() instanceof org.apache.arrow.vector.Float8Vector
                ? new int[] {'d'}
                : new int[] {4, 8, 'd'};
        for (int k : order) {
          try {
            if (k == 'd') {
              vector.getDouble(r);
            } else if (k == 8) {
              vector.getLong(r);
            } else {
              vector.getInt(r);
            }
            kind[column] = k;
            break;
          } catch (UnsupportedOperationException e) {
            // the next way of reading it
          }
        }
        if (kind[column] == 0) {
          throw new UnsupportedOperationException(
              "CPU star aggregation cannot read " + vector.getFieldVector().getField());
        }
      }
      return kind[column];
    }

    private long integer(ColumnVector vector, int column, int r) {
      int k = kind(vector, column, r);
      if (k == 'd') {
        throw new UnsupportedOperationException(
            "not an integer column: " + vector.getFieldVector().getField());
      }
      return k == 4 ? vector.getInt(r) : vector.getLong(r);
    }

    private double number(ColumnVector vector, int column, int r) {
      int k = kind(vector, column, r);
      return k == 'd' ? vector.getDouble(r) : k == 4 ? vector.getInt(r) : vector.getLong(r);
    }

    @Override
    public void accept(ColumnarBatch batch, int r) {
      // The dimensions' masks: a null, out-of-range or masked-out key drops the row.
      for (int d = 0; d < dimensionKey.length; d++) {
        ColumnVector vector = batch.column(dimensionColumn[d]);
        if (vector.isNullAt(r)) {
          return;
        }
        DeviceDimension dimension = plan.dimensions.get(d);
        long at = integer(vector, dimensionColumn[d], r) - dimension.firstKey();
        if (at < 0 || at >= dimension.keys() || dimension.keep()[(int) at] == 0) {
          return;
        }
        dimensionKey[d] = at;
      }
      // The probe: the packed key, the matching build rows that pass the comparison.
      double weight = 1;
      int n = plan.values.size();
      if (plan.probe != null) {
        weight = probe(batch, r, dimensionKey);
        if (weight == 0) {
          return;
        }
      }
      Arrays.fill(row, 0);
      for (int v = 0; v < n; v++) {
        double value = 0;
        double term = 0;
        boolean isNull = false;
        List<DeviceValue.Factor> factors = plan.values.get(v).factors();
        for (int f = 0; f < factors.size(); f++) {
          DeviceValue.Factor factor = factors.get(f);
          ColumnVector vector = batch.column(valueColumn[v][f]);
          if (vector.isNullAt(r)) {
            isNull = true;
            break;
          }
          double x = factor.offset() + factor.scale() * number(vector, valueColumn[v][f], r);
          if (factor.coefficient() != 0) {
            value += term;
            term = factor.coefficient() * x;
          } else {
            term *= x;
          }
        }
        if (!isNull) {
          row[v] = weight * (value + term);
          row[plan.values.size() + plan.probeSums.size() + 1 + v] = weight;
        }
      }
      for (int s = 0; s < plan.probeSums.size(); s++) {
        StarAggregation.ProbeSum sum = plan.probeSums.get(s);
        boolean kept =
            sum.maskedBy() < 0
                || row[plan.values.size() + plan.probeSums.size() + 1 + sum.maskedBy()] != 0;
        row[n + s] = kept ? rowSums[sum.column()] : 0;
        row[2 * (n + plan.probeSums.size()) - (plan.probeSums.size() - s) + 1] =
            kept ? rowCounts[sum.column()] : 0;
      }
      row[n + plan.probeSums.size()] = weight;
      if (dense != null) {
        int group = 0;
        for (int k = 0; k < plan.keys.size(); k++) {
          DeviceDimension.Key key = plan.keys.get(k);
          int code =
              plan.dimensions
                  .get(key.dimension())
                  .attributes()
                  .get(key.attribute())[(int) dimensionKey[key.dimension()]];
          group += code * plan.multipliers[k];
        }
        int base = group * plan.width;
        for (int c = 0; c < plan.width; c++) {
          dense[base + c] += row[c];
        }
      } else {
        List<Object> key = Lists.newArrayListWithCapacity(plan.keys.size());
        for (int k = 0; k < plan.keys.size(); k++) {
          DeviceDimension.Key spec = plan.keys.get(k);
          if (spec.isAttribute()) {
            key.add(
                (long)
                    plan.dimensions
                        .get(spec.dimension())
                        .attributes()
                        .get(spec.attribute())[(int) dimensionKey[spec.dimension()]]);
          } else {
            ColumnVector vector = batch.column(keyColumn[k]);
            key.add(vector.isNullAt(r) ? null : (Object) integer(vector, keyColumn[k], r));
          }
        }
        double[] totals = hashed.computeIfAbsent(key, ignored -> new double[plan.width]);
        for (int c = 0; c < plan.width; c++) {
          totals[c] += row[c];
        }
      }
    }

    /** The row's join weight, and its build sums in rowSums and rowCounts; 0 drops the row. */
    private double probe(ColumnarBatch batch, int r, long[] dimensionKey) {
      DeviceProbe probe = plan.probe;
      long key = 0;
      boolean inRange = true;
      List<DeviceProbe.Part> parts = probe.parts();
      for (int p = 0; p < parts.size(); p++) {
        DeviceProbe.Part part = parts.get(p);
        long value;
        if (part.isAttribute()) {
          value =
              plan.dimensions
                  .get(part.dimension())
                  .attributes()
                  .get(part.attribute())[(int) dimensionKey[part.dimension()]];
        } else {
          ColumnVector vector = batch.column(partColumn[p]);
          if (vector.isNullAt(r)) {
            inRange = false;
            continue;
          }
          value = integer(vector, partColumn[p], r);
        }
        long shifted = value - part.offset();
        if (shifted < 0 || shifted >= part.range()) {
          inRange = false;
        }
        key += shifted * part.multiplier();
      }
      boolean rowNull = false;
      long rowValue = 0;
      if (compareColumn >= 0) {
        ColumnVector vector = batch.column(compareColumn);
        rowNull = vector.isNullAt(r);
        rowValue = rowNull ? 0 : integer(vector, compareColumn, r);
      }
      Arrays.fill(rowSums, 0);
      Arrays.fill(rowCounts, 0);
      double weight = 0;
      if (inRange) {
        long[] keys = probe.keys();
        long[] values = probe.values();
        int j = Arrays.binarySearch(keys, key);
        if (j >= 0) {
          while (j > 0 && keys[j - 1] == key) {
            j--;
          }
          for (; j < keys.length && keys[j] == key; j++) {
            if (passes(probe.comparison(), rowNull, rowValue, values[j])) {
              weight += 1;
              for (int s = 0; s < rowSums.length; s++) {
                double v = probe.sums().get(s)[j];
                if (!Double.isNaN(v)) {
                  rowSums[s] += v;
                  rowCounts[s] += 1;
                }
              }
            }
          }
        }
      }
      return weight == 0 && probe.outer() ? 1 : weight;
    }

    private static boolean passes(
        DeviceProbe.Comparison how, boolean rowNull, long row, long build) {
      if (how == DeviceProbe.Comparison.ANY) {
        return true;
      }
      if (rowNull) {
        return false;
      }
      return switch (how) {
        case LESS -> row < build;
        case LESS_EQUAL -> row <= build;
        case EQUAL -> row == build;
        case GREATER -> row > build;
        case GREATER_EQUAL -> row >= build;
        default -> true;
      };
    }
  }
}
