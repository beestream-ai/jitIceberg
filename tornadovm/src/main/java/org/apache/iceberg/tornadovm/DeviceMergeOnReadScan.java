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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.FileContent;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.Table;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.relocated.com.google.common.collect.Maps;
import org.apache.iceberg.types.Types;
import uk.ac.manchester.tornado.api.DataRange;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.cudf.Cudf;
import uk.ac.manchester.tornado.cudf.enums.CudfAggregation;
import uk.ac.manchester.tornado.cudf.enums.CudfType;

/**
 * A merge-on-read scan of an Iceberg table on an NVIDIA GPU.
 *
 * <p>Planning stays on the CPU and is Iceberg's own: {@code planFiles()} decides which delete files
 * apply to which data file. Each data file then goes through one TornadoVM execution plan, built
 * once for the whole scan and reused for every file:
 *
 * <ol>
 *   <li>cuDF decodes the projected Parquet columns straight into TornadoVM's device buffers;
 *   <li>cuDF probes the equality-delete key set with the key column, when equality deletes apply;
 *   <li>a TornadoVM kernel combines the position-delete bitmap (v2 position deletes or a v3
 *       deletion vector, packed on the host), the probe and Iceberg's null semantics into one live
 *       byte a row;
 *   <li>the consumer: either a cuDF group-by over the live rows, of which only the groups come
 *       back, or cuDF compaction and TornadoVM gathers, after which the live rows come back.
 * </ol>
 *
 * <p>While the device works on one file, a small host pool reads the next files' footers and delete
 * files. Supported: local Parquet data files, INT/DATE, LONG/TIMESTAMP/TIME and DOUBLE columns
 * (required or optional), position deletes, deletion vectors, and equality deletes on one INT or
 * LONG column. Anything else is refused with {@link UnsupportedOperationException} rather than
 * answered wrongly.
 */
public final class DeviceMergeOnReadScan implements MergeOnReadScan {

  private static final int HOST_THREADS = 4;

  private final Table table;
  private ScanMetrics metrics = new ScanMetrics();

  private DeviceMergeOnReadScan(Table table) {
    this.table = table;
  }

  public static DeviceMergeOnReadScan of(Table table) {
    return new DeviceMergeOnReadScan(table);
  }

  /**
   * Whether this JVM can run device scans: TornadoVM with a cuDF shim that has the typed reader.
   */
  public static boolean isAvailable() {
    try {
      return Cudf.isParquetAvailable() && Cudf.isColumnsAvailable();
    } catch (LinkageError e) {
      return false;
    }
  }

  @Override
  public ScanMetrics metrics() {
    return metrics;
  }

  /**
   * {@code SELECT groupColumn, COUNT(*), SUM(c1), ... FROM table GROUP BY groupColumn} over the
   * live rows. With a null group column, one group keyed 0. Sum columns must be DOUBLE; nulls are
   * skipped as SQL skips them. The group column must be a required INT.
   */
  @Override
  public AggregateResult aggregate(String groupColumn, String... sumColumns) {
    if (sumColumns.length == 0) {
      throw new IllegalArgumentException("aggregate needs at least one sum column");
    }
    Scan scan = new Scan(groupColumn, ImmutableList.copyOf(sumColumns));
    if (groupColumn != null) {
      Types.NestedField key = table.schema().findField(groupColumn);
      if (key.isOptional() || ScanLayout.cudfType(key) != CudfType.INT32) {
        throw new UnsupportedOperationException("The group column must be a required INT: " + key);
      }
    }
    int[] validColumns = new int[sumColumns.length];
    for (int c = 0; c < sumColumns.length; c++) {
      Types.NestedField field = table.schema().findField(sumColumns[c]);
      if (ScanLayout.cudfType(field) != CudfType.FLOAT64) {
        throw new UnsupportedOperationException("Sum columns must be DOUBLE: " + field);
      }
      validColumns[c] = field.isOptional() ? scan.layout.readIndex(field.fieldId()) : -1;
    }
    if (scan.layout.slots(CudfType.FLOAT64) != sumColumns.length) {
      throw new UnsupportedOperationException("The sum columns must be distinct DOUBLE columns");
    }

    int capacity = scan.capacity;
    int groupOffset =
        groupColumn == null
            ? -1
            : scan.layout.slot(table.schema().findField(groupColumn).fieldId()) * capacity;
    IntArray groupKeys = new IntArray(capacity);
    IntArray sumKeys = new IntArray(capacity);
    DoubleArray sums = new DoubleArray((long) sumColumns.length * capacity);
    IntArray sumGroups = new IntArray(1);
    IntArray countKeys = new IntArray(capacity);
    DoubleArray counts = new DoubleArray(capacity);
    IntArray countGroups = new IntArray(1);
    IntArray validColumn = IntArray.fromArray(validColumns);

    TaskGraph graph = scan.region("aggregate");
    graph.task("groups", DeviceKernels::groupKeys, scan.live, scan.ints, groupOffset, groupKeys);
    if (scan.layout.anyOptional()) {
      graph
          .transferToDevice(DataTransferMode.FIRST_EXECUTION, validColumn)
          .task(
              "nulls",
              DeviceKernels::zeroNulls,
              scan.doubles,
              scan.valid,
              validColumn,
              sumColumns.length,
              capacity);
    }
    graph
        .libraryTask(
            "sum",
            Cudf::groupAggregate,
            capacity,
            sumColumns.length,
            CudfAggregation.SUM.code(),
            groupKeys,
            scan.doubles,
            sumKeys,
            sums,
            sumGroups)
        .libraryTask(
            "count",
            Cudf::groupAggregate,
            capacity,
            1,
            CudfAggregation.COUNT.code(),
            groupKeys,
            scan.doubles,
            countKeys,
            counts,
            countGroups)
        .transferToHost(
            DataTransferMode.UNDER_DEMAND,
            sumKeys,
            sums,
            sumGroups,
            countKeys,
            counts,
            countGroups);

    AggregateResult result = new AggregateResult(sumColumns.length);
    scan.run(
        graph,
        executed -> {
          executed.transferToHost(sumGroups, countGroups);
          int groups = sumGroups.get(0);
          int countGroupsN = countGroups.get(0);
          if (groups == 0) {
            return 0L;
          }
          executed.transferToHost(new DataRange(sumKeys).withSize(groups));
          executed.transferToHost(new DataRange(countKeys).withSize(countGroupsN));
          executed.transferToHost(new DataRange(counts).withSize(countGroupsN));
          for (int c = 0; c < sumColumns.length; c++) {
            executed.transferToHost(
                new DataRange(sums).withOffset((long) c * capacity).withSize(groups));
          }
          Map<Integer, Long> rowsByKey = Maps.newHashMap();
          for (int g = 0; g < countGroupsN; g++) {
            rowsByKey.put(countKeys.get(g), (long) counts.get(g));
          }
          long live = 0;
          double[] values = new double[sumColumns.length];
          for (int g = 0; g < groups; g++) {
            int key = sumKeys.get(g);
            if (key == DeviceKernels.DEAD_GROUP) {
              continue;
            }
            for (int c = 0; c < values.length; c++) {
              values[c] = sums.get(c * capacity + g);
            }
            long rows = rowsByKey.get(key);
            result.add(key, rows, values);
            live += rows;
          }
          return live;
        });
    return result;
  }

  /**
   * {@code SELECT columns FROM table} over the live rows, one batch a data file. Live rows are
   * compacted on the device and only they are copied back.
   *
   * @return the number of live rows
   */
  @Override
  public long materialize(List<String> columns, Consumer<LiveRows> consumer) {
    Scan scan = new Scan(null, columns);
    ScanLayout layout = scan.layout;
    int capacity = scan.capacity;
    int intSlots = layout.slots(CudfType.INT32);
    int longSlots = layout.slots(CudfType.INT64);
    int doubleSlots = layout.slots(CudfType.FLOAT64);
    int validSlots = layout.anyOptional() ? layout.fields().size() : 0;

    IntArray positions = new IntArray(capacity);
    IntArray liveCount = new IntArray(1);
    IntArray outInts = new IntArray(Math.max(1L, (long) intSlots * capacity));
    LongArray outLongs = new LongArray(Math.max(1L, (long) longSlots * capacity));
    DoubleArray outDoubles = new DoubleArray(Math.max(1L, (long) doubleSlots * capacity));
    ByteArray outValid = new ByteArray(Math.max(1L, (long) validSlots * capacity));

    TaskGraph graph = scan.region("materialize");
    graph.libraryTask(
        "compact", Cudf::selectedIndices, capacity, capacity, scan.live, positions, liveCount);
    if (intSlots > 0) {
      graph.task(
          "gatherInts",
          DeviceKernels::gatherInts,
          positions,
          liveCount,
          scan.ints,
          intSlots,
          capacity,
          outInts);
    }
    if (longSlots > 0) {
      graph.task(
          "gatherLongs",
          DeviceKernels::gatherLongs,
          positions,
          liveCount,
          scan.longs,
          longSlots,
          capacity,
          outLongs);
    }
    if (doubleSlots > 0) {
      graph.task(
          "gatherDoubles",
          DeviceKernels::gatherDoubles,
          positions,
          liveCount,
          scan.doubles,
          doubleSlots,
          capacity,
          outDoubles);
    }
    if (validSlots > 0) {
      graph.task(
          "gatherValid",
          DeviceKernels::gatherBytes,
          positions,
          liveCount,
          scan.valid,
          validSlots,
          capacity,
          outValid);
    }
    graph.transferToHost(
        DataTransferMode.UNDER_DEMAND, liveCount, outInts, outLongs, outDoubles, outValid);

    Map<String, Integer> idByName = Maps.newHashMap();
    for (Types.NestedField field : layout.fields()) {
      idByName.put(field.name(), field.fieldId());
    }
    return scan.run(
        graph,
        executed -> {
          executed.transferToHost(liveCount);
          int rows = liveCount.get(0);
          if (rows > 0) {
            copySlots(executed, outInts, intSlots, capacity, rows);
            copySlots(executed, outLongs, longSlots, capacity, rows);
            copySlots(executed, outDoubles, doubleSlots, capacity, rows);
            copySlots(executed, outValid, validSlots, capacity, rows);
          }
          consumer.accept(
              new HostBatch(
                  layout, idByName, rows, capacity, outInts, outLongs, outDoubles, outValid));
          return (long) rows;
        });
  }

  private static void copySlots(
      TornadoExecutionResult executed,
      uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray array,
      int slots,
      int stride,
      int rows) {
    for (int s = 0; s < slots; s++) {
      executed.transferToHost(new DataRange(array).withOffset((long) s * stride).withSize(rows));
    }
  }

  /** What each file's execution hands back to the scan loop: the live rows it found. */
  @FunctionalInterface
  private interface FileResult {
    long consume(TornadoExecutionResult executed);
  }

  /** The device buffers and host holders shared by every query shape, and the per-file loop. */
  private final class Scan {
    private final List<FileScanTask> tasks;
    private final ScanLayout layout;
    private final HostDeletes host;
    private final Integer equalityKeyId;
    private final CudfType equalityKeyType;
    private final int capacity;
    private final int equalityCapacity;

    private final StringBuilder path = new StringBuilder();
    private final long[] rowsHolder = new long[1];
    private final int[] columns;
    private final int[] probeSizes = new int[2];
    private final IntArray meta = new IntArray(DeviceKernels.META_SIZE);
    private final ByteArray positionBits;
    private final ByteArray inSet;
    private final IntArray setInts;
    private final LongArray setLongs;
    private final IntArray ints;
    private final LongArray longs;
    private final DoubleArray doubles;
    private final ByteArray valid;
    private final ByteArray live;

    Scan(String groupColumn, List<String> projection) {
      metrics = new ScanMetrics();
      this.start = System.nanoTime();
      try (CloseableIterable<FileScanTask> planned = table.newScan().planFiles()) {
        this.tasks = Lists.newArrayList(planned);
      } catch (java.io.IOException e) {
        throw new java.io.UncheckedIOException(e);
      }
      this.equalityKeyId = HostDeletes.equalityKeyId(tasks);
      this.layout = new ScanLayout(table.schema(), equalityKeyId, groupColumn, projection);
      this.host = new HostDeletes(table, layout, equalityKeyId);
      this.equalityKeyType = equalityKeyId == null ? null : layout.type(equalityKeyId);

      long maxRows = 1;
      long maxSet = 1;
      for (FileScanTask task : tasks) {
        maxRows = Math.max(maxRows, task.file().recordCount());
        long setRows = 0;
        for (DeleteFile delete : task.deletes()) {
          if (delete.content() == FileContent.EQUALITY_DELETES) {
            setRows += delete.recordCount();
          }
        }
        maxSet = Math.max(maxSet, setRows);
      }
      if (maxRows > Integer.MAX_VALUE / 8 || maxSet > Integer.MAX_VALUE) {
        throw new UnsupportedOperationException(
            "Data file too large for one device read: " + maxRows);
      }
      this.capacity = (int) maxRows;
      this.equalityCapacity = (int) maxSet;
      this.columns = new int[layout.fields().size()];

      boolean equality = equalityKeyId != null;
      this.positionBits = new ByteArray((capacity + 7) / 8);
      this.inSet = new ByteArray(equality ? capacity : 1);
      this.setInts =
          new IntArray(equality && equalityKeyType == CudfType.INT32 ? equalityCapacity : 1);
      this.setLongs =
          new LongArray(equality && equalityKeyType == CudfType.INT64 ? equalityCapacity : 1);
      this.ints = new IntArray(Math.max(1L, (long) layout.slots(CudfType.INT32) * capacity));
      this.longs = new LongArray(Math.max(1L, (long) layout.slots(CudfType.INT64) * capacity));
      this.doubles =
          new DoubleArray(Math.max(1L, (long) layout.slots(CudfType.FLOAT64) * capacity));
      this.valid =
          new ByteArray(layout.anyOptional() ? (long) layout.fields().size() * capacity : 1L);
      this.live = new ByteArray(capacity);
      metrics.plan((System.nanoTime() - start) / 1e6);
    }

    private final long start;

    /** Read, probe and the live-row kernel: the part of the graph every query shape shares. */
    TaskGraph region(String name) {
      TaskGraph graph =
          new TaskGraph(name)
              .transferToDevice(DataTransferMode.EVERY_EXECUTION, meta, positionBits);
      boolean equality = equalityKeyId != null;
      if (equality) {
        graph.transferToDevice(
            DataTransferMode.FIRST_EXECUTION,
            equalityKeyType == CudfType.INT32 ? setInts : setLongs);
      }
      graph.libraryTask(
          "read",
          Cudf::readParquetColumns,
          path,
          0,
          0,
          columns,
          layout.types(),
          rowsHolder,
          capacity,
          ints,
          longs,
          doubles,
          valid,
          layout.anyOptional());
      int keyValidOffset = 0;
      if (equality) {
        // The key is slot 0 of its type, so the probe reads it from offset 0.
        if (equalityKeyType == CudfType.INT32) {
          graph.libraryTask("probe", Cudf::containedIn, probeSizes, ints, setInts, inSet);
        } else {
          graph.libraryTask("probe", Cudf::containedIn, probeSizes, longs, setLongs, inSet);
        }
        keyValidOffset = layout.readIndex(equalityKeyId) * capacity;
      }
      graph.task(
          "live", DeviceKernels::liveRows, meta, positionBits, inSet, valid, keyValidOffset, live);
      return graph;
    }

    long run(TaskGraph graph, FileResult perFile) {
      ExecutorService pool = Executors.newFixedThreadPool(HOST_THREADS);
      long liveRows = 0;
      try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
        List<CompletableFuture<HostDeletes.FilePrep>> preps = Lists.newArrayList();
        for (FileScanTask task : tasks) {
          preps.add(CompletableFuture.supplyAsync(() -> host.prepare(task), pool));
        }
        String uploadedSet = null;
        boolean keyNullable =
            equalityKeyId != null && table.schema().findField(equalityKeyId).isOptional();
        for (CompletableFuture<HostDeletes.FilePrep> future : preps) {
          HostDeletes.FilePrep prep = future.join();
          metrics.deleteLoad(prep.hostNanos() / 1e6);
          path.setLength(0);
          path.append(prep.path());
          rowsHolder[0] = prep.rows();
          System.arraycopy(prep.columns(), 0, columns, 0, columns.length);

          meta.set(DeviceKernels.META_ROWS, (int) prep.rows());
          byte[] bits = prep.positionBits();
          meta.set(DeviceKernels.META_HAS_POSITION_DELETES, bits == null ? 0 : 1);
          if (bits != null) {
            MemorySegment.copy(
                MemorySegment.ofArray(bits), 0, positionBits.getSegment(), 0, bits.length);
          }
          HostDeletes.EqualitySet set = prep.equality();
          meta.set(DeviceKernels.META_HAS_EQUALITY_DELETES, set == null ? 0 : 1);
          meta.set(DeviceKernels.META_KEY_NULLABLE, keyNullable ? 1 : 0);
          if (set != null) {
            meta.set(DeviceKernels.META_EQUALITY_SET_HAS_NULL, set.hasNull() ? 1 : 0);
            if (!set.id().equals(uploadedSet)) {
              upload(plan, set);
              uploadedSet = set.id();
            }
          }

          long deviceStart = System.nanoTime();
          TornadoExecutionResult executed = plan.execute();
          long copyStart = System.nanoTime();
          long rows = perFile.consume(executed);
          long end = System.nanoTime();
          metrics.device((copyStart - deviceStart) / 1e6);
          metrics.copyBack((end - copyStart) / 1e6);
          metrics.file(prep.rows());
          metrics.live(rows);
          liveRows += rows;
        }
      } catch (TornadoExecutionPlanException e) {
        throw new IllegalStateException("Device scan failed", e);
      } finally {
        pool.shutdownNow();
      }
      metrics.total((System.nanoTime() - start) / 1e6);
      return liveRows;
    }

    /** Puts a new equality-delete set on the device; files that share a set share one upload. */
    private void upload(TornadoExecutionPlan plan, HostDeletes.EqualitySet set) {
      long[] keys = set.keys();
      probeSizes[0] = capacity;
      probeSizes[1] = keys.length;
      if (equalityKeyType == CudfType.INT32) {
        for (int i = 0; i < keys.length; i++) {
          setInts.set(i, (int) keys[i]);
        }
        plan.transferToDevice(setInts);
      } else {
        MemorySegment.copy(
            MemorySegment.ofArray(keys),
            0,
            setLongs.getSegment(),
            0,
            (long) keys.length * ValueLayout.JAVA_LONG.byteSize());
        plan.transferToDevice(setLongs);
      }
    }
  }
}
