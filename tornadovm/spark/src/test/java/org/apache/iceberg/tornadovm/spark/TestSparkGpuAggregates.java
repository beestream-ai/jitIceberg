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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import java.nio.file.Path;
import java.time.LocalDate;
import java.util.List;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.tornadovm.DeviceMergeOnReadScan;
import org.apache.iceberg.tornadovm.bench.MorTables;
import org.apache.iceberg.types.Types;
import org.apache.spark.SparkEnv;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Spark SQL aggregates over a merge-on-read table with the GPU aggregate planning on and off: the
 * same groups and, to rounding, the same sums -- whether the device ran the aggregate, declined it
 * at planning, or failed it at run time and left it to the CPU.
 */
public class TestSparkGpuAggregates {
  private static SparkSession spark;

  @TempDir private Path temp;

  @BeforeAll
  public static void startSpark() {
    assumeThat(DeviceMergeOnReadScan.isAvailable()).as("TornadoVM with the cuDF shim").isTrue();
    spark =
        SparkSession.builder()
            .master("local[4]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .config(
                "spark.sql.extensions", "org.apache.spark.sql.iceberg.gpu.GpuAggregateExtensions")
            .config("spark.sql.catalog.default_iceberg", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.default_iceberg.type", "hadoop")
            .config(
                "spark.sql.catalog.default_iceberg.warehouse",
                System.getProperty("java.io.tmpdir") + "/iceberg-gpu-spark-test")
            .getOrCreate();
  }

  @AfterAll
  public static void stopSpark() {
    if (spark != null) {
      spark.stop();
    }
  }

  private static void gpu(boolean on) {
    SparkEnv.get().conf().set(GpuColumnarReaderProvider.ENABLED, Boolean.toString(on));
    // These tables are small: read them on the device anyway.
    SparkEnv.get().conf().set(GpuColumnarReaderProvider.MIN_ROWS, "0");
  }

  /** The query's rows, sorted, and whether the device aggregate produced them. */
  private record Answer(List<Row> rows, boolean onDevice) {}

  private static Answer run(String sql) {
    Dataset<Row> frame = spark.sql(sql);
    List<Row> rows = Lists.newArrayList(frame.collectAsList());
    rows.sort((a, b) -> a.toString().compareTo(b.toString()));
    String plan = frame.queryExecution().executedPlan().toString();
    return new Answer(rows, plan.contains("GpuIcebergAggregate"));
  }

  private static void assertSameAnswer(List<Row> gpu, List<Row> cpu) {
    assertThat(gpu).hasSameSizeAs(cpu);
    for (int r = 0; r < cpu.size(); r++) {
      Row expected = cpu.get(r);
      Row actual = gpu.get(r);
      assertThat(actual.size()).isEqualTo(expected.size());
      for (int c = 0; c < expected.size(); c++) {
        Object want = expected.get(c);
        Object got = actual.get(c);
        if (want instanceof Double wanted && got instanceof Double value) {
          assertThat(value)
              .as("row %s column %d", expected, c)
              .isCloseTo(wanted, org.assertj.core.api.Assertions.within(1e-9 * Math.abs(wanted)));
        } else {
          assertThat(got).as("row %s column %d", expected, c).isEqualTo(want);
        }
      }
    }
  }

  @Test
  public void testSameAggregatesOnTheDevice() {
    Schema schema =
        new Schema(
            Types.NestedField.required(1, "id", Types.LongType.get()),
            Types.NestedField.optional(2, "flag", Types.StringType.get()),
            Types.NestedField.optional(3, "status", Types.StringType.get()),
            Types.NestedField.optional(4, "shipdate", Types.DateType.get()),
            Types.NestedField.optional(5, "price", Types.DoubleType.get()),
            Types.NestedField.optional(6, "disc", Types.DoubleType.get()),
            Types.NestedField.optional(7, "tax", Types.DoubleType.get()),
            Types.NestedField.optional(8, "qty", Types.IntegerType.get()),
            Types.NestedField.optional(9, "sparse", Types.DoubleType.get()),
            Types.NestedField.optional(10, "comment", Types.StringType.get()));
    String location = temp.resolve("lineitem").toString();
    Table table = MorTables.create(location, schema, 2);
    List<DataFile> files = Lists.newArrayList();
    String[] flags = {"A", "N", "R"};
    for (long f = 0; f < 3; f++) {
      long index = f;
      files.add(
          MorTables.writeDataFile(
              table,
              20_000,
              i -> {
                long id = index * 1_000_000L + i;
                GenericRecord record = GenericRecord.create(schema);
                record.setField("id", id);
                record.setField("flag", flags[(int) (id % 3)]);
                record.setField("status", id % 7 < 3 ? "F" : "O");
                record.setField("shipdate", LocalDate.of(1992, 1, 1).plusDays(id % 2500));
                record.setField("price", 900.0 + (id % 1000) * 1.37);
                record.setField("disc", (id % 11) / 100.0);
                record.setField("tax", (id % 9) / 100.0);
                record.setField("qty", (int) (1 + id % 50));
                record.setField("sparse", id % 5 == 0 ? null : id * 0.25);
                record.setField("comment", "a comment longer than eight bytes " + (id % 4));
                return record;
              }));
    }
    MorTables.append(table, files);
    MorTables.deletePositions(
        table, files, ImmutableList.of(new long[] {0, 5, 17, 19_999}, new long[] {3}, new long[0]));
    MorTables.deleteEqual(table, "id", ImmutableList.of(7L, 1_000_008L, 2_000_009L));
    spark.read().format("iceberg").load(location).createOrReplaceTempView("lineitem");

    // TPC-H Q1 and Q6 shapes, an empty global aggregate, int and bigint sums exact in
    // doubles, Iceberg's bucket transform as the group.
    List<String> onDevice =
        ImmutableList.of(
            "SELECT flag, status, sum(qty), sum(price), sum(price * (1 - disc)),"
                + " sum(price * (1 - disc) * (1 + tax)), avg(qty), avg(price), avg(disc), count(*)"
                + " FROM lineitem WHERE shipdate <= DATE '1998-09-02' GROUP BY flag, status",
            "SELECT count(*), sum(price * disc) FROM lineitem WHERE shipdate >= DATE '1994-01-01'"
                + " AND shipdate < DATE '1995-01-01' AND disc >= 0.05 AND disc <= 0.07"
                + " AND qty < 24",
            "SELECT count(*), sum(price), avg(tax) FROM lineitem WHERE price < 0",
            "SELECT shipdate, avg(price * qty) FROM lineitem WHERE id < 3000 GROUP BY shipdate",
            "SELECT flag, sum(id) FROM lineitem GROUP BY flag",
            "SELECT default_iceberg.system.bucket(16, id) b, count(*), sum(price), avg(disc)"
                + " FROM lineitem WHERE disc > 0.02 GROUP BY b");
    // Declined when planned (a SUM over nulls), or failed on the device (a group
    // string over 8 bytes): the CPU answers.
    List<String> onCpu =
        ImmutableList.of(
            "SELECT flag, sum(sparse) FROM lineitem GROUP BY flag",
            // TPC-DS q54's shape: the group key a projected expression, declined without failing.
            "SELECT segment, count(*), sum(price) FROM (SELECT CAST(price / 50 AS INT) segment,"
                + " price FROM lineitem) GROUP BY segment",
            "SELECT comment, sum(price) FROM lineitem GROUP BY comment");

    List<Answer> cpu = Lists.newArrayList();
    for (String sql : onDevice) {
      cpu.add(run(sql));
    }
    for (String sql : onCpu) {
      cpu.add(run(sql));
    }
    gpu(true);
    List<Answer> device = Lists.newArrayList();
    try {
      for (String sql : onDevice) {
        device.add(run(sql));
      }
      for (String sql : onCpu) {
        device.add(run(sql));
      }
    } finally {
      gpu(false);
    }

    for (int q = 0; q < cpu.size(); q++) {
      assertThat(cpu.get(q).onDevice()).isFalse();
      assertThat(cpu.get(q).rows()).isNotEmpty();
      assertSameAnswer(device.get(q).rows(), cpu.get(q).rows());
    }
    for (int q = 0; q < onDevice.size(); q++) {
      assertThat(device.get(q).onDevice()).as(onDevice.get(q)).isTrue();
    }
    for (int q = 0; q < onCpu.size() - 1; q++) {
      assertThat(device.get(onDevice.size() + q).onDevice()).as(onCpu.get(q)).isFalse();
    }
  }

  /**
   * Decimal columns, as TPC-DS has them: rows read through the GPU reader, and SUM, AVG and group
   * keys of decimals from the device aggregate, equal to Spark's exact decimal answers.
   */
  @Test
  public void testDecimalReadsAndAggregates() {
    Schema schema =
        new Schema(
            Types.NestedField.required(1, "id", Types.LongType.get()),
            Types.NestedField.optional(2, "price", Types.DecimalType.of(7, 2)),
            Types.NestedField.optional(3, "net", Types.DecimalType.of(15, 2)),
            Types.NestedField.optional(4, "band", Types.DecimalType.of(3, 1)),
            Types.NestedField.optional(5, "store", Types.IntegerType.get()));
    String location = temp.resolve("sales").toString();
    Table table = MorTables.create(location, schema, 2);
    List<DataFile> files = Lists.newArrayList();
    for (long f = 0; f < 3; f++) {
      long index = f;
      files.add(
          MorTables.writeDataFile(
              table,
              10_000,
              i -> {
                long id = index * 1_000_000L + i;
                GenericRecord record = GenericRecord.create(schema);
                record.setField("id", id);
                record.setField(
                    "price", java.math.BigDecimal.valueOf((id * 7919) % 9_999_999 - 4_999_999, 2));
                record.setField(
                    "net", java.math.BigDecimal.valueOf(9_999_999_999L - id * 1_234L, 2));
                record.setField("band", java.math.BigDecimal.valueOf(id % 5, 1));
                record.setField("store", (int) (id % 7));
                return record;
              }));
    }
    MorTables.append(table, files);
    MorTables.deletePositions(
        table, files, ImmutableList.of(new long[] {1, 2, 3}, new long[0], new long[] {9_999}));
    MorTables.deleteEqual(table, "id", ImmutableList.of(11L, 1_000_012L));
    spark.read().format("iceberg").load(location).createOrReplaceTempView("sales");

    List<String> onDevice =
        ImmutableList.of(
            "SELECT band, store, sum(price), avg(price), sum(net), avg(net), count(*) FROM sales"
                + " WHERE price > -1234.56 GROUP BY band, store",
            "SELECT sum(price), avg(net), count(*) FROM sales WHERE band = 0.3");
    String rows = "SELECT * FROM sales WHERE price < 100.00";

    List<Answer> cpu = Lists.newArrayList();
    for (String sql : onDevice) {
      cpu.add(run(sql));
    }
    Answer cpuRows = run(rows);
    gpu(true);
    List<Answer> device = Lists.newArrayList();
    Answer gpuRows;
    long tasks = GpuPartitionReader.TASKS.get();
    try {
      for (String sql : onDevice) {
        device.add(run(sql));
      }
      gpuRows = run(rows);
    } finally {
      gpu(false);
    }
    for (int q = 0; q < onDevice.size(); q++) {
      assertThat(cpu.get(q).rows()).isNotEmpty();
      assertThat(device.get(q).onDevice()).as(onDevice.get(q)).isTrue();
      assertThat(device.get(q).rows()).containsExactlyElementsOf(cpu.get(q).rows());
    }
    assertThat(GpuPartitionReader.TASKS.get()).as("GPU reader tasks").isGreaterThan(tasks);
    assertThat(cpuRows.rows()).isNotEmpty();
    assertThat(gpuRows.rows()).containsExactlyElementsOf(cpuRows.rows());
  }
}
