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
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;
import org.apache.iceberg.Table;
import org.apache.iceberg.data.IcebergGenerics;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.tornadovm.AggregateResult;
import org.apache.iceberg.tornadovm.CpuMergeOnReadScan;
import org.apache.iceberg.tornadovm.DeviceMergeOnReadScan;
import org.apache.iceberg.tornadovm.MergeOnReadScan;

/**
 * Runs one engine and one query over benchmark tables and prints one CSV line a run.
 *
 * <pre>
 *   MorBench &lt;warehouse&gt; &lt;engine: gpu|cpu|generics&gt; &lt;query: q1|q2|q3&gt; &lt;runs&gt; &lt;table&gt;...
 * </pre>
 *
 * <ul>
 *   <li>q1 -- {@code SELECT count(*), sum(v1)};
 *   <li>q2 -- {@code SELECT k, count(*), sum(v1), sum(v2), sum(v3), sum(v4) GROUP BY k};
 *   <li>q3 -- {@code SELECT id, k, v1}, every live row copied into host arrays;
 *   <li>q4 -- {@code SELECT bucket(16, id), count(*), sum(v1) GROUP BY 1}, Iceberg's bucket
 *       partition transform.
 * </ul>
 *
 * <p>The first run of each table is a warm-up (TornadoVM's JIT compiles the kernels, the page cache
 * fills) and is reported but marked. The checksum lets runs of different engines be compared: the
 * live row count, then a sum over the result.
 */
public final class MorBench {

  private static final String[] SUMS = {"v1", "v2", "v3", "v4"};

  private MorBench() {}

  public static void main(String[] args) throws IOException {
    String warehouse = args[0];
    String engine = args[1];
    String query = args[2];
    int runs = Integer.parseInt(args[3]);
    int threads = Integer.getInteger("threads", Runtime.getRuntime().availableProcessors());
    System.out.println("table,engine,query,run,warmup,millis,live_rows,checksum,breakdown");
    for (int t = 4; t < args.length; t++) {
      Table table = MorTables.load(warehouse + "/" + args[t]);
      // One scan object a table, as an engine keeps its operators: a device scan keeps its buffers
      // and compiled plan between runs, so only the first run of a table pays for them.
      MergeOnReadScan scan =
          "gpu".equals(engine)
              ? DeviceMergeOnReadScan.of(table)
              : "cpu".equals(engine) ? CpuMergeOnReadScan.of(table, threads) : null;
      for (int run = 0; run < runs; run++) {
        long start = System.nanoTime();
        double[] answer;
        String breakdown = "";
        if (scan == null) {
          answer = generics(table, query);
        } else {
          answer = run(scan, query);
          breakdown = scan.metrics().toString();
        }
        double millis = (System.nanoTime() - start) / 1e6;
        System.out.println(
            String.format(
                Locale.ROOT,
                "%s,%s,%s,%d,%s,%.1f,%d,%.6e,\"%s\"",
                args[t],
                engine,
                query,
                run,
                run == 0 ? "yes" : "no",
                millis,
                (long) answer[0],
                answer[1],
                breakdown));
      }
      if (scan instanceof AutoCloseable closeable) {
        try {
          closeable.close();
        } catch (Exception e) {
          throw new IllegalStateException(e);
        }
      }
    }
  }

  /** The live row count and a checksum of the answer. */
  private static double[] run(MergeOnReadScan scan, String query) {
    return switch (query) {
      case "q1" -> {
        AggregateResult result = scan.aggregate(null, "v1");
        yield new double[] {result.rows(), sum(result)};
      }
      case "q2" -> {
        AggregateResult result = scan.aggregate("k", SUMS);
        yield new double[] {result.rows(), sum(result) + result.groups().size()};
      }
      case "q4" -> {
        AggregateResult result = scan.aggregateByBucket("id", 16, "v1");
        yield new double[] {result.rows(), sum(result) + result.groups().size()};
      }
      case "q3" -> {
        LongAdder ids = new LongAdder();
        DoubleAdder values = new DoubleAdder();
        long rows =
            scan.materialize(
                ImmutableList.of("id", "k", "v1"),
                batch -> {
                  int id = batch.column("id");
                  int k = batch.column("k");
                  int v1 = batch.column("v1");
                  long idSum = 0;
                  double v1Sum = 0;
                  for (int r = 0; r < batch.rows(); r++) {
                    idSum += batch.getLong(id, r) + batch.getInt(k, r);
                    v1Sum += batch.getDouble(v1, r);
                  }
                  ids.add(idSum);
                  values.add(v1Sum);
                });
        yield new double[] {rows, ids.sum() + values.sum()};
      }
      default -> throw new IllegalArgumentException("Unknown query " + query);
    };
  }

  private static double sum(AggregateResult result) {
    double total = 0;
    for (AggregateResult.Group group : result.groups().values()) {
      for (double value : group.sums()) {
        total += value;
      }
    }
    return total;
  }

  /** Iceberg's own row-at-a-time merge-on-read reader, one thread: the reference answer. */
  private static double[] generics(Table table, String query) throws IOException {
    List<String> columns =
        "q3".equals(query)
            ? ImmutableList.of("id", "k", "v1")
            : "q1".equals(query)
                ? ImmutableList.of("v1")
                : ImmutableList.of("k", "v1", "v2", "v3", "v4");
    long rows = 0;
    double checksum = 0;
    java.util.Set<Integer> groups = new java.util.HashSet<>();
    try (CloseableIterable<Record> records = IcebergGenerics.read(table).select(columns).build()) {
      for (Record record : records) {
        rows++;
        switch (query) {
          case "q1" -> checksum += (Double) record.getField("v1");
          case "q4" -> {
            long id = (Long) record.getField("id");
            groups.add((org.apache.iceberg.util.BucketUtil.hash(id) & Integer.MAX_VALUE) % 16);
            checksum += (Double) record.getField("v1");
          }
          case "q2" -> {
            groups.add((Integer) record.getField("k"));
            for (String sum : SUMS) {
              checksum += (Double) record.getField(sum);
            }
          }
          default ->
              checksum +=
                  (Long) record.getField("id")
                      + (Integer) record.getField("k")
                      + (Double) record.getField("v1");
        }
      }
    }
    return new double[] {rows, checksum + groups.size()};
  }
}
