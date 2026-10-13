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
package org.apache.iceberg.tornadovm.bench;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.apache.iceberg.data.IcebergGenerics;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.tornadovm.DeviceDimension;
import org.apache.iceberg.tornadovm.DeviceMergeOnReadScan;
import org.apache.iceberg.tornadovm.DeviceValue;
import org.apache.iceberg.tornadovm.GroupedAggregate;

/**
 * Device-join prototype (DEVICE-JOINS.md, phase 1 step 1): TPC-DS q22 as a star aggregate. The
 * dimensions are read on the host into key sets (the date filter and the warehouse inner join
 * become filters on the fact's foreign keys), one device aggregate groups inventory by inv_item_sk
 * (every grouping column is an item attribute), and the host decodes the item keys and computes the
 * ROLLUP.
 *
 * <p>Usage: {@code StarQ22 <tpcds warehouse dir> <runs> <out file>}; prints per-phase times.
 */
public final class StarQ22 {
  private StarQ22() {}

  public static void main(String[] args) throws IOException {
    String warehouse = args[0];
    int runs = Integer.parseInt(args[1]);
    // -Dstarq22.reuse=true keeps one device scan across runs (its regions stay allocated).
    boolean reuse = Boolean.getBoolean("starq22.reuse");
    // -Dstarq22.star=true joins on the device (DeviceMergeOnReadScan.starAggregate): the dimensions
    // become masks indexed by key, and one unfiltered scan of inventory serves every run.
    boolean star = Boolean.getBoolean("starq22.star");
    DeviceMergeOnReadScan kept = null;
    for (int run = 0; run < runs; run++) {
      long t0 = System.nanoTime();
      // Dimensions on the host: small, read once a query.
      List<Object> dates = new ArrayList<>();
      try (CloseableIterable<Record> rows =
          IcebergGenerics.read(MorTables.load(warehouse + "/date_dim"))
              .select("d_date_sk", "d_month_seq")
              .where(
                  Expressions.and(
                      Expressions.greaterThanOrEqual("d_month_seq", 1200),
                      Expressions.lessThanOrEqual("d_month_seq", 1211)))
              .build()) {
        for (Record r : rows) {
          Number month = (Number) r.getField("d_month_seq");
          if (month != null && month.intValue() >= 1200 && month.intValue() <= 1211) {
            dates.add(r.getField("d_date_sk"));
          }
        }
      }
      List<Object> warehouses = new ArrayList<>();
      try (CloseableIterable<Record> rows =
          IcebergGenerics.read(MorTables.load(warehouse + "/warehouse"))
              .select("w_warehouse_sk")
              .build()) {
        for (Record r : rows) {
          warehouses.add(r.getField("w_warehouse_sk"));
        }
      }
      Map<Long, String[]> items = new HashMap<>();
      try (CloseableIterable<Record> rows =
          IcebergGenerics.read(MorTables.load(warehouse + "/item"))
              .select("i_item_sk", "i_product_name", "i_brand", "i_class", "i_category")
              .build()) {
        for (Record r : rows) {
          items.put(
              ((Number) r.getField("i_item_sk")).longValue(),
              new String[] {
                (String) r.getField("i_product_name"),
                (String) r.getField("i_brand"),
                (String) r.getField("i_class"),
                (String) r.getField("i_category")
              });
        }
      }
      long t1 = System.nanoTime();
      List<DeviceDimension> dimensions =
          ImmutableList.of(
              dimension("inv_date_sk", dates),
              dimension("inv_warehouse_sk", warehouses),
              dimension("inv_item_sk", new ArrayList<>(items.keySet())));

      // The fact on the device: deletes applied, joins as key filters, grouped by the item key.
      Expression filter =
          Expressions.and(
              Expressions.in("inv_date_sk", dates), Expressions.in("inv_warehouse_sk", warehouses));
      GroupedAggregate groups;
      DeviceMergeOnReadScan scan =
          kept != null
              ? kept
              : DeviceMergeOnReadScan.of(
                  MorTables.load(warehouse + "/inventory"),
                  star ? Expressions.alwaysTrue() : filter);
      try {
        groups =
            star
                ? scan.starAggregate(
                    dimensions,
                    ImmutableList.of(DeviceDimension.Key.column("inv_item_sk")),
                    ImmutableList.of(DeviceValue.of("inv_quantity_on_hand")))
                : scan.aggregate(
                    ImmutableList.of("inv_item_sk"),
                    ImmutableList.of(DeviceValue.of("inv_quantity_on_hand")));
        System.out.println("scan: " + scan.metrics());
      } catch (Exception e) {
        throw new IllegalStateException(e);
      } finally {
        if (reuse || star) {
          kept = scan;
        } else {
          scan.close();
        }
      }
      long t2 = System.nanoTime();

      // ROLLUP(product_name, brand, class, category) of AVG(qoh) = SUM / COUNT(non-null), on the
      // host.
      Map<List<String>, double[]> rollup = new HashMap<>();
      long factRows = 0;
      for (Map.Entry<List<Object>, double[]> e : groups.groups().entrySet()) {
        Object key = e.getKey().get(0);
        String[] item = key == null ? null : items.get(((Number) key).longValue());
        if (item == null) {
          continue; // inner join with item
        }
        double[] totals = e.getValue(); // sum, rows, non-null count
        factRows += (long) totals[1];
        for (int level = 4; level >= 0; level--) {
          List<String> k = new ArrayList<>(4);
          for (int c = 0; c < 4; c++) {
            k.add(c < level ? item[c] : null);
          }
          k.add(Integer.toString(level)); // grouping level keeps NULL attributes apart from levels
          double[] acc = rollup.computeIfAbsent(k, ignored -> new double[2]);
          acc[0] += totals[0];
          acc[1] += totals[2];
        }
      }
      List<Object[]> result = new ArrayList<>();
      for (Map.Entry<List<String>, double[]> e : rollup.entrySet()) {
        double[] acc = e.getValue();
        Double avg = acc[1] == 0 ? null : acc[0] / acc[1];
        List<String> k = e.getKey();
        result.add(new Object[] {k.get(0), k.get(1), k.get(2), k.get(3), avg});
      }
      Comparator<Object> nullsFirst =
          Comparator.nullsFirst((a, b) -> ((Comparable) a).compareTo(b));
      result.sort(
          (a, b) -> {
            int c = nullsFirst.compare(a[4], b[4]);
            for (int i = 0; c == 0 && i < 4; i++) {
              c = nullsFirst.compare(a[i], b[i]);
            }
            return c;
          });
      List<Object[]> top = result.subList(0, Math.min(100, result.size()));
      long t3 = System.nanoTime();

      System.out.printf(
          Locale.ROOT,
          "run=%d dims_ms=%.1f device_ms=%.1f host_rollup_ms=%.1f total_ms=%.1f groups=%d fact_rows=%d dates=%d%n",
          run,
          (t1 - t0) / 1e6,
          (t2 - t1) / 1e6,
          (t3 - t2) / 1e6,
          (t3 - t0) / 1e6,
          groups.groups().size(),
          factRows,
          dates.size());
      if (run == runs - 1) {
        try (PrintWriter out = new PrintWriter(args[2], StandardCharsets.UTF_8)) {
          for (Object[] row : top) {
            out.printf(
                Locale.ROOT,
                "%s|%s|%s|%s|%s%n",
                row[0],
                row[1],
                row[2],
                row[3],
                row[4] == null ? "null" : String.format(Locale.ROOT, "%.6g", (Double) row[4]));
          }
        }
      }
    }
    if (kept != null) {
      kept.close();
    }
  }

  /** A dimension that keeps the given keys, as a mask over their range. */
  private static DeviceDimension dimension(String factColumn, List<?> keys) {
    long first = Long.MAX_VALUE;
    long last = Long.MIN_VALUE;
    for (Object key : keys) {
      long value = ((Number) key).longValue();
      first = Math.min(first, value);
      last = Math.max(last, value);
    }
    boolean[] keep = new boolean[(int) (last - first + 1)];
    for (Object key : keys) {
      keep[(int) (((Number) key).longValue() - first)] = true;
    }
    return DeviceDimension.of(factColumn, (int) first, keep);
  }
}
