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
package org.apache.iceberg.tornadovm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.stream.LongStream;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableProperties;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.IcebergGenerics;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableMap;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.tornadovm.bench.MorTables;
import org.apache.iceberg.types.Types;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Every device scan is checked against Iceberg's own merge-on-read reader, {@link IcebergGenerics},
 * on tables that exercise one delete mechanism each and then all of them at once.
 */
public class TestDeviceMergeOnRead {

  private static final String[] SUMS = {"v1", "v2", "v3", "v4"};

  @TempDir private Path temp;

  @BeforeAll
  public static void deviceMustBeAvailable() {
    assumeThat(DeviceMergeOnReadScan.isAvailable())
        .as("TornadoVM with a cuDF shim that has readParquetColumns")
        .isTrue();
  }

  private static Record row(Schema schema, long fileIndex, long i) {
    GenericRecord record = GenericRecord.create(schema);
    record.setField("id", fileIndex * 1_000_000L + i);
    record.setField("k", (int) (i % 7));
    record.setField("v1", i * 0.25);
    record.setField("v2", (double) -i);
    record.setField("v3", (double) (i % 3));
    record.setField("v4", 1.0);
    return record;
  }

  private static List<DataFile> writeFiles(Table table, int firstIndex, long... rows) {
    List<DataFile> files = Lists.newArrayList();
    for (int f = 0; f < rows.length; f++) {
      long index = firstIndex + f;
      files.add(MorTables.writeDataFile(table, rows[f], i -> row(table.schema(), index, i)));
    }
    MorTables.append(table, files);
    return files;
  }

  private static long[] everyNth(long rows, int n) {
    return LongStream.range(0, rows).filter(i -> i % n == 0).toArray();
  }

  private static long[] randomPositions(long rows, double fraction, long seed) {
    Random random = new Random(seed);
    return LongStream.range(0, rows).filter(i -> random.nextDouble() < fraction).toArray();
  }

  /** Position deletes, then equality deletes, then a file appended after them: v2. */
  @Test
  public void testPositionAndEqualityDeletesV2() throws IOException {
    Table table = MorTables.create(temp.resolve("v2").toString(), MorTables.SCHEMA, 2);
    List<DataFile> files = writeFiles(table, 0, 1000, 2500, 700);
    MorTables.deletePositions(
        table,
        files,
        ImmutableList.of(everyNth(1000, 10), randomPositions(2500, 0.3, 1), new long[0]));
    MorTables.deleteEqual(
        table, "id", ImmutableList.of(5L, 7L, 1_000_123L, 1_000_124L, 2_000_699L, 42_000_000L));
    // Appended after the equality delete, so id 5 in this file must survive it.
    List<DataFile> late = writeFiles(table, 0, 300);
    MorTables.deleteEqual(table, "id", ImmutableList.of(6L, 2_000_000L));

    assertMatchesIceberg(table);
    assertThat(late).hasSize(1);
  }

  /** Deletion vectors and equality deletes: v3. */
  @Test
  public void testDeletionVectorsV3() throws IOException {
    Table table = MorTables.create(temp.resolve("v3").toString(), MorTables.SCHEMA, 3);
    List<DataFile> files = writeFiles(table, 0, 4000, 3000);
    MorTables.deletePositions(
        table, files, ImmutableList.of(randomPositions(4000, 0.05, 2), everyNth(3000, 3)));
    MorTables.deleteEqual(table, "id", ImmutableList.of(1L, 2L, 1_000_002L));
    assertMatchesIceberg(table);
  }

  /** Only appends: the scan must equal a plain read. */
  @Test
  public void testNoDeletes() throws IOException {
    Table table = MorTables.create(temp.resolve("plain").toString(), MorTables.SCHEMA, 2);
    writeFiles(table, 0, 1234, 1);
    assertMatchesIceberg(table);
  }

  /** Small row groups, so positions have to stay right across row-group boundaries. */
  @Test
  public void testManyRowGroups() throws IOException {
    Table table =
        MorTables.create(
            temp.resolve("groups").toString(),
            MorTables.SCHEMA,
            2,
            ImmutableMap.of(
                TableProperties.PARQUET_ROW_GROUP_SIZE_BYTES, "16384",
                TableProperties.PARQUET_PAGE_ROW_LIMIT, "500"));
    List<DataFile> files = writeFiles(table, 0, 60_000);
    MorTables.deletePositions(table, files, ImmutableList.of(randomPositions(60_000, 0.2, 3)));
    assertMatchesIceberg(table);
  }

  /**
   * Optional columns: a null key is deleted only by a null in the delete set, and a null value is
   * skipped by SUM but its row still counts.
   */
  @Test
  public void testNullsV2() throws IOException {
    Schema schema =
        new Schema(
            Types.NestedField.optional(1, "id", Types.LongType.get()),
            Types.NestedField.required(2, "k", Types.IntegerType.get()),
            Types.NestedField.optional(3, "v1", Types.DoubleType.get()),
            Types.NestedField.required(4, "v2", Types.DoubleType.get()),
            Types.NestedField.required(5, "v3", Types.DoubleType.get()),
            Types.NestedField.required(6, "v4", Types.DoubleType.get()));
    Table table = MorTables.create(temp.resolve("nulls").toString(), schema, 2);
    DataFile file =
        MorTables.writeDataFile(
            table,
            2000,
            i -> {
              Record record = row(schema, 0, i);
              if (i % 5 == 0) {
                record.setField("id", null);
              }
              if (i % 4 == 0) {
                record.setField("v1", null);
              }
              return record;
            });
    MorTables.append(table, ImmutableList.of(file));
    MorTables.deletePositions(table, ImmutableList.of(file), ImmutableList.of(everyNth(2000, 7)));
    assertMatchesIceberg(table);

    List<Object> keys = Lists.newArrayList();
    keys.add(3L);
    keys.add(11L);
    keys.add(null);
    MorTables.deleteEqual(table, "id", keys);
    assertMatchesIceberg(table);
  }

  /** An equality key of type INT, which the device probes as 32-bit keys. */
  @Test
  public void testIntEqualityKey() throws IOException {
    Table table = MorTables.create(temp.resolve("intkey").toString(), MorTables.SCHEMA, 2);
    writeFiles(table, 0, 3000, 1500);
    MorTables.deleteEqual(table, "k", ImmutableList.of(2, 5));
    assertMatchesIceberg(table);
  }

  // ---------------------------------------------------------------------------------------------

  private static void assertMatchesIceberg(Table table) throws IOException {
    table.refresh();
    Map<Integer, double[]> expectedByK = new TreeMap<>();
    double[] expectedAll = new double[2];
    List<String> expectedRows = Lists.newArrayList();
    try (CloseableIterable<Record> records = IcebergGenerics.read(table).build()) {
      for (Record record : records) {
        int k = (Integer) record.getField("k");
        double[] totals = expectedByK.computeIfAbsent(k, ignored -> new double[1 + SUMS.length]);
        totals[0] += 1;
        for (int c = 0; c < SUMS.length; c++) {
          Object value = record.getField(SUMS[c]);
          totals[1 + c] += value == null ? 0.0 : (Double) value;
        }
        expectedAll[0] += 1;
        Object v1 = record.getField("v1");
        expectedAll[1] += v1 == null ? 0.0 : (Double) v1;
        expectedRows.add(record.getField("id") + "/" + k + "/" + v1);
      }
    }

    assertScanMatches(DeviceMergeOnReadScan.of(table), expectedByK, expectedAll, expectedRows);
    assertScanMatches(CpuMergeOnReadScan.of(table, 4), expectedByK, expectedAll, expectedRows);
  }

  private static void assertScanMatches(
      MergeOnReadScan scan,
      Map<Integer, double[]> expectedByK,
      double[] expectedAll,
      List<String> expectedRows) {
    AggregateResult grouped = scan.aggregate("k", SUMS);
    assertThat(grouped.groups().keySet()).isEqualTo(expectedByK.keySet());
    for (Map.Entry<Integer, double[]> entry : expectedByK.entrySet()) {
      AggregateResult.Group group = grouped.groups().get(entry.getKey());
      assertThat(group.count())
          .as("count of k=%s", entry.getKey())
          .isEqualTo((long) entry.getValue()[0]);
      for (int c = 0; c < SUMS.length; c++) {
        assertThat(group.sums()[c])
            .as("%s of k=%s", SUMS[c], entry.getKey())
            .isCloseTo(entry.getValue()[1 + c], org.assertj.core.data.Offset.offset(1e-6));
      }
    }

    AggregateResult total = scan.aggregate(null, "v1");
    if (expectedAll[0] == 0) {
      assertThat(total.groups()).isEmpty();
    } else {
      assertThat(total.groups().keySet()).containsExactly(0);
      assertThat(total.groups().get(0).count()).isEqualTo((long) expectedAll[0]);
      assertThat(total.groups().get(0).sums()[0])
          .isCloseTo(expectedAll[1], org.assertj.core.data.Offset.offset(1e-6));
    }

    List<String> rows = java.util.Collections.synchronizedList(Lists.newArrayList());
    long live =
        scan.materialize(
            ImmutableList.of("id", "k", "v1"),
            batch -> {
              for (int r = 0; r < batch.rows(); r++) {
                Object id = batch.isNull("id", r) ? null : batch.getLong("id", r);
                Object v1 = batch.isNull("v1", r) ? null : batch.getDouble("v1", r);
                rows.add(id + "/" + batch.getInt("k", r) + "/" + v1);
              }
            });
    assertThat(live).isEqualTo(expectedRows.size());
    String[] actual = rows.toArray(new String[0]);
    String[] expected = expectedRows.toArray(new String[0]);
    Arrays.sort(actual);
    Arrays.sort(expected);
    assertThat(actual).containsExactly(expected);
  }
}
