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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.FileContent;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.RewriteFiles;
import org.apache.iceberg.Table;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;
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
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;
import uk.ac.manchester.tornado.cudf.Cudf;
import uk.ac.manchester.tornado.cudf.enums.CudfAggregation;
import uk.ac.manchester.tornado.cudf.enums.CudfType;
import uk.ac.manchester.tornado.cudf.enums.ParquetColumnType;

/**
 * A merge-on-read scan of an Iceberg table on an NVIDIA GPU.
 *
 * <p>Planning stays on the CPU and is Iceberg's own: {@code planFiles()} decides which delete files
 * apply to which data file. Consecutive data files that share their equality deletes are grouped
 * into batches of up to {@code iceberg.tornadovm.batch-rows} rows, and each batch goes through one
 * TornadoVM execution plan:
 *
 * <ol>
 *   <li>cuDF decodes the projected columns of all the batch's Parquet files, in one call, straight
 *       into TornadoVM's device buffers;
 *   <li>cuDF reads the {@code pos} column of their file-scoped position-delete files, and the key
 *       column of their equality-delete files, into device buffers too;
 *   <li>cuDF probes the equality-delete set with the key column;
 *   <li>a TornadoVM kernel folds the probe, deletion vectors (packed into a bitmap on the host) and
 *       Iceberg's null semantics into one live byte a row, and a second scatters the position
 *       deletes into it;
 *   <li>the consumer: a cuDF group-by of which only the groups come back, or cuDF compaction and
 *       TornadoVM gathers after which only the live rows come back.
 * </ol>
 *
 * <p>A host pool reads footers, deletion vectors and anything the device path does not cover while
 * the device works on earlier batches. The device buffers and the compiled plan of each query shape
 * are kept by this object and reused by its later queries; {@link #close()} releases them.
 *
 * <p>Supported: local Parquet data files, INT/DATE, LONG/TIMESTAMP/TIME and DOUBLE columns
 * (required or optional), position deletes, deletion vectors, and equality deletes on one INT or
 * LONG column. Anything else is refused with {@link UnsupportedOperationException} rather than
 * answered wrongly.
 */
public final class DeviceMergeOnReadScan implements MergeOnReadScan, AutoCloseable {

  private static final int HOST_THREADS = 8;

  /** Characters of paths one batch may name in one holder: data files, delete files. */
  private static final int PATH_CAPACITY = 1 << 16;

  /**
   * Rows a device batch may hold. Files are grouped up to this many rows into one read, because a
   * read of one small file is mostly latency; a file larger than this is a batch on its own.
   */
  private static final long BATCH_ROWS = Long.getLong("iceberg.tornadovm.batch-rows", 4_000_000L);

  /**
   * Batches in flight at once, each in its own region -- its own buffers, plan and CUDA stream --
   * so that one batch's file reads and host-to-device copies overlap another's decoding.
   */
  private static final int STREAMS = Integer.getInteger("iceberg.tornadovm.streams", 4);

  private final Table table;
  private final Expression filter;
  private final Map<String, List<Region>> regions = Maps.newHashMap();
  private ScanMetrics metrics = new ScanMetrics();

  private DeviceMergeOnReadScan(Table table, Expression filter) {
    this.table = table;
    this.filter = filter;
  }

  public static DeviceMergeOnReadScan of(Table table) {
    return new DeviceMergeOnReadScan(table, Expressions.alwaysTrue());
  }

  /**
   * A scan of the rows that match {@code filter}. Iceberg prunes files with it while planning, and
   * the device applies each file's residual -- what statistics could not decide -- to every row.
   * See {@link DevicePredicate} for what a residual may contain.
   */
  public static DeviceMergeOnReadScan of(Table table, Expression filter) {
    return new DeviceMergeOnReadScan(table, filter);
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

  @Override
  public void close() {
    regions.values().forEach(list -> list.forEach(Region::close));
    regions.clear();
  }

  @Override
  public AggregateResult aggregate(String groupColumn, String... sumColumns) {
    return aggregate(groupColumn, 0, sumColumns);
  }

  @Override
  public AggregateResult aggregateByBucket(String column, int buckets, String... sumColumns) {
    Types.NestedField field = table.schema().findField(column);
    if (field.isOptional() || field.type().typeId() != org.apache.iceberg.types.Type.TypeID.LONG) {
      throw new UnsupportedOperationException(
          "The bucket column must be a required LONG: " + field);
    }
    if (buckets <= 0) {
      throw new IllegalArgumentException("Invalid bucket count: " + buckets);
    }
    return aggregate(column, buckets, sumColumns);
  }

  /** With {@code buckets > 0}, groups by {@code bucket(buckets, groupColumn)} of a LONG column. */
  private AggregateResult aggregate(String groupColumn, int buckets, String... sumColumns) {
    if (sumColumns.length == 0) {
      throw new IllegalArgumentException("aggregate needs at least one sum column");
    }
    metrics = new ScanMetrics();
    long start = System.nanoTime();
    if (groupColumn != null && buckets == 0) {
      Types.NestedField key = table.schema().findField(groupColumn);
      if (key.isOptional() || ScanLayout.cudfType(key) != CudfType.INT32) {
        throw new UnsupportedOperationException("The group column must be a required INT: " + key);
      }
    }
    for (String column : sumColumns) {
      if (ScanLayout.cudfType(table.schema().findField(column)) != CudfType.FLOAT64) {
        throw new UnsupportedOperationException("Sum columns must be DOUBLE: " + column);
      }
    }
    Planned planned = new Planned(groupColumn, ImmutableList.copyOf(sumColumns));
    // The group-by reads the first sumColumns slots of the FP64 buffer, so the sum columns have to
    // be those slots, in order; columns only the filter reads come after them.
    for (int c = 0; c < sumColumns.length; c++) {
      if (planned.layout.slot(table.schema().findField(sumColumns[c]).fieldId()) != c) {
        throw new UnsupportedOperationException("The sum columns must be distinct DOUBLE columns");
      }
    }
    String key = "aggregate:" + groupColumn + ":" + buckets + ":" + String.join(",", sumColumns);
    List<Region> streams =
        reuse(key, planned, () -> new AggregateRegion(planned, groupColumn, buckets, sumColumns));
    metrics.plan((System.nanoTime() - start) / 1e6);

    AggregateResult result = new AggregateResult(sumColumns.length);
    run(
        streams,
        planned,
        (region, executed) -> ((AggregateRegion) region).collect(executed, result));
    metrics.total((System.nanoTime() - start) / 1e6);
    return result;
  }

  @Override
  public long materialize(List<String> columns, Consumer<LiveRows> consumer) {
    metrics = new ScanMetrics();
    long start = System.nanoTime();
    Planned planned = new Planned(null, columns);
    String key = "materialize:" + String.join(",", columns);
    List<Region> streams = reuse(key, planned, () -> new MaterializeRegion(planned));
    metrics.plan((System.nanoTime() - start) / 1e6);
    long live =
        run(
            streams,
            planned,
            (region, executed) -> ((MaterializeRegion) region).collect(executed, consumer));
    metrics.total((System.nanoTime() - start) / 1e6);
    return live;
  }

  /** What {@link #rewriteDataFiles()} did. */
  public static final class RewriteResult {
    private final int dataFilesRemoved;
    private final int deleteFilesRemoved;
    private final int dataFilesAdded;
    private final long rows;

    RewriteResult(int dataFilesRemoved, int deleteFilesRemoved, int dataFilesAdded, long rows) {
      this.dataFilesRemoved = dataFilesRemoved;
      this.deleteFilesRemoved = deleteFilesRemoved;
      this.dataFilesAdded = dataFilesAdded;
      this.rows = rows;
    }

    public int dataFilesRemoved() {
      return dataFilesRemoved;
    }

    public int deleteFilesRemoved() {
      return deleteFilesRemoved;
    }

    public int dataFilesAdded() {
      return dataFilesAdded;
    }

    public long rows() {
      return rows;
    }
  }

  /**
   * Compacts the table on the device: merge-on-read into copy-on-write.
   *
   * <p>Every data file is read with its deletes applied, as a scan reads it; the live rows of each
   * batch are compacted and written, still on the device, to a new ZSTD Parquet file by libcudf --
   * with Iceberg's field ids, so Iceberg reads it back by id. The files' metrics come from the
   * footers libcudf wrote. One {@code RewriteFiles} commit then swaps every old data file and every
   * delete file for the new files, validated against the snapshot that was planned, so a concurrent
   * commit fails it rather than being lost.
   *
   * <p>Format version 2 only: a v3 table's rewrite also has to carry each row's lineage ({@code
   * _row_id}), which this does not write yet.
   */
  public RewriteResult rewriteDataFiles() {
    if (filter.op() != Expression.Operation.TRUE) {
      throw new UnsupportedOperationException("A rewrite of part of a table is not supported");
    }
    int formatVersion =
        ((org.apache.iceberg.HasTableOperations) table).operations().current().formatVersion();
    if (formatVersion >= 3) {
      throw new UnsupportedOperationException(
          "Rewriting a v3 table needs row lineage, which the device writer does not carry yet");
    }
    metrics = new ScanMetrics();
    long start = System.nanoTime();
    List<String> columns = Lists.newArrayList();
    for (Types.NestedField field : table.schema().columns()) {
      if (field.type().typeId() == org.apache.iceberg.types.Type.TypeID.TIME) {
        throw new UnsupportedOperationException("TIME columns cannot be written on the device");
      }
      ScanLayout.cudfType(field); // refuses what the device cannot read
      columns.add(field.name());
    }
    Planned planned = new Planned(null, columns);
    List<Region> streams = reuse("rewrite", planned, () -> new RewriteRegion(planned));
    metrics.plan((System.nanoTime() - start) / 1e6);

    List<DataFile> written = java.util.Collections.synchronizedList(Lists.newArrayList());
    long rows =
        run(
            streams,
            planned,
            (region, executed) -> ((RewriteRegion) region).collect(executed, written));

    java.util.Set<String> deletesSeen = new java.util.HashSet<>();
    RewriteFiles rewrite = table.newRewrite();
    if (planned.snapshotId != null) {
      rewrite.validateFromSnapshot(planned.snapshotId);
    }
    for (FileScanTask task : planned.tasks) {
      rewrite.deleteFile(task.file());
      for (DeleteFile delete : task.deletes()) {
        if (deletesSeen.add(delete.location())) {
          rewrite.deleteFile(delete);
        }
      }
    }
    written.forEach(rewrite::addFile);
    rewrite.commit();
    metrics.total((System.nanoTime() - start) / 1e6);
    return new RewriteResult(planned.tasks.size(), deletesSeen.size(), written.size(), rows);
  }

  /** The regions kept for a query shape, one a stream, rebuilt when this scan outgrows them. */
  private List<Region> reuse(
      String key, Planned planned, java.util.function.Supplier<Region> create) {
    List<Region> kept = regions.get(key);
    if (kept != null && kept.get(0).fits(planned)) {
      return kept;
    }
    if (kept != null) {
      kept.forEach(Region::close);
    }
    int streams = Math.max(1, Math.min(STREAMS, planned.batches.size()));
    List<Region> created = Lists.newArrayList();
    for (int i = 0; i < streams; i++) {
      created.add(create.get());
    }
    regions.put(key, created);
    return created;
  }

  /**
   * Runs every batch of a query on the given regions, one thread a region taking the next batch as
   * it finishes one; {@code perBatch} returns the live rows a batch had. Host preparation of every
   * batch is started up front on its own pool.
   */
  private long run(
      List<Region> streams,
      Planned planned,
      java.util.function.ToLongBiFunction<Region, TornadoExecutionResult> perBatch) {
    ExecutorService hostPool = Executors.newFixedThreadPool(HOST_THREADS);
    java.util.concurrent.atomic.AtomicInteger next =
        new java.util.concurrent.atomic.AtomicInteger();
    java.util.concurrent.atomic.AtomicLong liveRows = new java.util.concurrent.atomic.AtomicLong();
    try {
      List<CompletableFuture<HostDeletes.FilePrep>> preps = Lists.newArrayList();
      for (List<FileScanTask> batch : planned.batches) {
        preps.add(CompletableFuture.supplyAsync(() -> planned.host.prepare(batch), hostPool));
      }
      List<java.util.concurrent.Future<?>> workers = Lists.newArrayList();
      for (Region region : streams) {
        workers.add(
            region.thread.submit(
                () -> {
                  TornadoExecutionPlan executionPlan = region.plan();
                  for (int i = next.getAndIncrement();
                      i < preps.size();
                      i = next.getAndIncrement()) {
                    HostDeletes.FilePrep prep = preps.get(i).join();
                    metrics.deleteLoad(prep.hostNanos() / 1e6);
                    region.stage(executionPlan, prep, planned.keyNullable);
                    if (region.filtered) {
                      planned
                          .batchFilters
                          .get(i)
                          .encode(
                              region.layout,
                              table.schema(),
                              region.capacity,
                              region.filterTerms,
                              region.filterLongs,
                              region.filterDoubles);
                    }
                    long deviceStart = System.nanoTime();
                    TornadoExecutionResult executed = executionPlan.execute();
                    long copyStart = System.nanoTime();
                    long rows = perBatch.applyAsLong(region, executed);
                    long end = System.nanoTime();
                    metrics.device((copyStart - deviceStart) / 1e6);
                    metrics.copyBack((end - copyStart) / 1e6);
                    metrics.batch(prep.rows(), prep.files());
                    metrics.live(rows);
                    liveRows.addAndGet(rows);
                  }
                }));
      }
      for (java.util.concurrent.Future<?> worker : workers) {
        worker.get();
      }
    } catch (java.util.concurrent.ExecutionException e) {
      throw new IllegalStateException("Device scan failed", e.getCause());
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(e);
    } finally {
      hostPool.shutdownNow();
    }
    return liveRows.get();
  }

  /** One query's planning: Iceberg's file tasks, grouped into device batches, and their sizes. */
  private final class Planned {
    private final List<List<FileScanTask>> batches = Lists.newArrayList();
    private final List<FileScanTask> tasks;
    private final Long snapshotId;
    private final ScanLayout layout;
    private final HostDeletes host;
    private final Integer equalityKeyId;
    private final CudfType equalityKeyType;
    private final boolean keyNullable;
    private final int capacity;
    private final int equalityCapacity;
    private final int positionCapacity;
    private final int maxFiles;
    private final boolean filtered;
    private final List<DevicePredicate> batchFilters = Lists.newArrayList();

    Planned(String groupColumn, List<String> projection) {
      this.snapshotId =
          table.currentSnapshot() == null ? null : table.currentSnapshot().snapshotId();
      try (CloseableIterable<FileScanTask> planned = table.newScan().filter(filter).planFiles()) {
        this.tasks = Lists.newArrayList(planned);
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
      this.equalityKeyId = HostDeletes.equalityKeyId(tasks);
      this.filtered = filter.op() != Expression.Operation.TRUE;
      List<String> read = Lists.newArrayList(projection);
      for (int fieldId : DevicePredicate.fieldIds(filter, table.schema())) {
        String name = table.schema().findField(fieldId).name();
        if (!read.contains(name)) {
          read.add(name);
        }
      }
      this.layout = new ScanLayout(table.schema(), equalityKeyId, groupColumn, read);
      this.host = new HostDeletes(table, layout, equalityKeyId, true);
      this.equalityKeyType = equalityKeyId == null ? null : layout.type(equalityKeyId);
      this.keyNullable =
          equalityKeyId != null && table.schema().findField(equalityKeyId).isOptional();

      List<FileScanTask> current = Lists.newArrayList();
      long currentRows = 0;
      String currentSet = null;
      for (FileScanTask task : tasks) {
        long rows = task.file().recordCount();
        // Files batch together only if they share their equality deletes and their residual.
        String set = HostDeletes.equalityId(task) + "|" + task.residual();
        if (!current.isEmpty() && (currentRows + rows > BATCH_ROWS || !set.equals(currentSet))) {
          batches.add(current);
          current = Lists.newArrayList();
          currentRows = 0;
        }
        current.add(task);
        currentRows += rows;
        currentSet = set;
      }
      if (!current.isEmpty()) {
        batches.add(current);
      }
      for (List<FileScanTask> batch : batches) {
        batchFilters.add(
            filtered
                ? DevicePredicate.compile(batch.get(0).residual(), table.schema())
                : DevicePredicate.alwaysTrue());
      }

      long maxRows = 1;
      long maxSet = 1;
      long maxPositions = 1;
      int files = 1;
      for (List<FileScanTask> batch : batches) {
        maxRows = Math.max(maxRows, batch.stream().mapToLong(t -> t.file().recordCount()).sum());
        files = Math.max(files, batch.size());
        long positionRows = 0;
        for (FileScanTask task : batch) {
          for (DeleteFile delete : task.deletes()) {
            if (HostDeletes.readOnDevice(delete, task.file().location())) {
              positionRows += delete.recordCount();
            }
          }
        }
        maxPositions = Math.max(maxPositions, positionRows);
        long setRows = 0;
        for (DeleteFile delete : batch.get(0).deletes()) {
          if (delete.content() == FileContent.EQUALITY_DELETES) {
            setRows += delete.recordCount();
          }
        }
        maxSet = Math.max(maxSet, setRows);
      }
      if (maxRows > Integer.MAX_VALUE / 8
          || maxSet > Integer.MAX_VALUE
          || maxPositions > Integer.MAX_VALUE) {
        throw new UnsupportedOperationException("Batch too large for one device read: " + maxRows);
      }
      this.capacity = (int) maxRows;
      this.equalityCapacity = (int) maxSet;
      this.positionCapacity = (int) maxPositions;
      this.maxFiles = files;
    }

    /** What decides the shape of the graph, apart from sizes. */
    String signature() {
      StringBuilder signature = new StringBuilder();
      for (int c = 0; c < layout.fields().size(); c++) {
        Types.NestedField field = layout.fields().get(c);
        signature.append(field.fieldId()).append(field.isOptional() ? "?" : "").append(',');
      }
      return signature.append(equalityKeyId).append(keyNullable).append(filtered).toString();
    }
  }

  /**
   * The device buffers, host holders and execution plan of one query shape: read, delete
   * application and live-row kernel, which every shape shares, plus what the shape adds.
   */
  private abstract static class Region implements AutoCloseable {
    final String signature;
    final ScanLayout layout;
    final Integer equalityKeyId;
    final CudfType equalityKeyType;
    final int capacity;
    final int equalityCapacity;
    final int positionCapacity;
    final int maxFiles;

    // The path holders are handed to TornadoVM, which serializes a holder object the first time it
    // transfers it and sizes that buffer then; a holder whose array later grows overflows it. So
    // each holder gets a fixed capacity up front, and a longer path list is refused.
    final StringBuilder path = new StringBuilder(PATH_CAPACITY);
    final long[] rowsHolder = new long[1];
    final int[] columns;
    final int[] probeSizes = new int[2];
    final IntArray meta = new IntArray(DeviceKernels.META_SIZE);
    final ByteArray positionBits;
    final StringBuilder positionPath = new StringBuilder(PATH_CAPACITY);
    final long[] positionRowsHolder = new long[1];
    final int[] positionColumn = new int[1];
    final LongArray positions;
    final IntArray segmentStarts;
    final IntArray segmentOffsets;
    final StringBuilder setPath = new StringBuilder(PATH_CAPACITY);
    final long[] setRowsHolder = new long[1];
    final int[] setColumn = new int[1];
    final boolean filtered;
    final IntArray filterTerms =
        new IntArray(1 + DevicePredicate.TERM_INTS * DevicePredicate.MAX_TERMS);
    final LongArray filterLongs = new LongArray(DevicePredicate.MAX_CONSTANTS);
    final DoubleArray filterDoubles = new DoubleArray(DevicePredicate.MAX_CONSTANTS);
    final ByteArray inSet;
    final IntArray setInts;
    final LongArray setLongs;
    final IntArray ints;
    final LongArray longs;
    final DoubleArray doubles;
    final ByteArray valid;
    final ByteArray live;

    private TornadoExecutionPlan plan;
    private String setOnDevice;

    /**
     * The one thread that builds, runs and closes this region's plan. TornadoVM keys its CUDA
     * streams by plan and thread, but caches a library's context -- and with it the stream cuDF
     * runs on -- by plan alone; a plan run from a second thread would put its copies and kernels on
     * one stream and its cuDF calls on another, with nothing ordering the two.
     */
    final ExecutorService thread =
        Executors.newSingleThreadExecutor(
            runnable -> {
              Thread worker = new Thread(runnable, "tornadovm-region");
              worker.setDaemon(true);
              return worker;
            });

    Region(Planned planned) {
      this.signature = planned.signature();
      this.filtered = planned.filtered;
      this.layout = planned.layout;
      this.equalityKeyId = planned.equalityKeyId;
      this.equalityKeyType = planned.equalityKeyType;
      this.capacity = planned.capacity;
      this.equalityCapacity = planned.equalityCapacity;
      this.positionCapacity = planned.positionCapacity;
      this.maxFiles = planned.maxFiles;
      this.columns = new int[layout.fields().size()];
      this.probeSizes[0] = capacity;

      boolean equality = equalityKeyId != null;
      this.positionBits = new ByteArray((capacity + 7) / 8);
      this.positions = new LongArray(positionCapacity);
      this.segmentStarts = new IntArray(maxFiles);
      this.segmentOffsets = new IntArray(maxFiles);
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
    }

    boolean fits(Planned planned) {
      return signature.equals(planned.signature())
          && capacity >= planned.capacity
          && equalityCapacity >= planned.equalityCapacity
          && positionCapacity >= planned.positionCapacity
          && maxFiles >= planned.maxFiles;
    }

    /** What this shape adds after the live-row kernel. */
    abstract void consumer(TaskGraph graph);

    TornadoExecutionPlan plan() {
      if (plan == null) {
        TaskGraph graph = shared();
        consumer(graph);
        plan = new TornadoExecutionPlan(graph.snapshot());
      }
      return plan;
    }

    private TaskGraph shared() {
      boolean equality = equalityKeyId != null;
      TaskGraph graph =
          new TaskGraph("mor")
              .transferToDevice(
                  DataTransferMode.EVERY_EXECUTION,
                  meta,
                  positionBits,
                  segmentStarts,
                  segmentOffsets);
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
      // Position-delete files the device reads itself: their pos columns, concatenated.
      graph.libraryTask(
          "readPositions",
          Cudf::readParquetColumns,
          positionPath,
          0,
          0,
          positionColumn,
          new CudfType[] {CudfType.INT64},
          positionRowsHolder,
          positionCapacity,
          new IntArray(1),
          positions,
          new DoubleArray(1),
          new ByteArray(1),
          false);
      int keyValidOffset = 0;
      if (equality) {
        // Equality-delete files the device reads itself, straight into the set; an execution whose
        // set is unchanged reads nothing and probes the copy already there.
        boolean intKey = equalityKeyType == CudfType.INT32;
        graph.libraryTask(
            "readSet",
            Cudf::readParquetColumns,
            setPath,
            0,
            0,
            setColumn,
            new CudfType[] {equalityKeyType},
            setRowsHolder,
            equalityCapacity,
            intKey ? setInts : new IntArray(1),
            intKey ? new LongArray(1) : setLongs,
            new DoubleArray(1),
            new ByteArray(1),
            false);
        // The key is slot 0 of its type, so the probe reads it from offset 0.
        if (intKey) {
          graph.libraryTask("probe", Cudf::containedIn, probeSizes, ints, setInts, inSet);
        } else {
          graph.libraryTask("probe", Cudf::containedIn, probeSizes, longs, setLongs, inSet);
        }
        keyValidOffset = layout.readIndex(equalityKeyId) * capacity;
      }
      graph
          .task(
              "live",
              DeviceKernels::liveRows,
              meta,
              positionBits,
              inSet,
              valid,
              keyValidOffset,
              live)
          .task(
              "deletePositions",
              DeviceKernels::deletePositions,
              positions,
              meta,
              segmentStarts,
              segmentOffsets,
              live);
      if (filtered) {
        graph
            .transferToDevice(
                DataTransferMode.EVERY_EXECUTION, filterTerms, filterLongs, filterDoubles)
            .task(
                "filter",
                DeviceKernels::filterRows,
                filterTerms,
                filterLongs,
                filterDoubles,
                ints,
                longs,
                doubles,
                valid,
                capacity,
                live);
      }
      return graph;
    }

    /** Points the plan's holders and small inputs at one batch. */
    /** What a shape sets up per batch, after the shared staging; nothing by default. */
    void stageOutput() {}

    void stage(TornadoExecutionPlan executionPlan, HostDeletes.FilePrep prep, boolean keyNullable) {
      stageOutput();
      path.setLength(0);
      hold(path, prep.path());
      rowsHolder[0] = prep.rows();
      System.arraycopy(prep.columns(), 0, columns, 0, columns.length);
      meta.set(DeviceKernels.META_ROWS, (int) prep.rows());

      byte[] bits = prep.positionBits();
      meta.set(DeviceKernels.META_HAS_POSITION_DELETES, bits == null ? 0 : 1);
      if (bits != null) {
        MemorySegment.copy(
            MemorySegment.ofArray(bits), 0, positionBits.getSegment(), 0, bits.length);
      }

      positionPath.setLength(0);
      hold(positionPath, prep.positionPaths());
      positionRowsHolder[0] = prep.positionRows();
      positionColumn[0] = Math.max(0, prep.positionColumn());
      meta.set(DeviceKernels.META_POSITION_ROWS, (int) prep.positionRows());
      meta.set(DeviceKernels.META_POSITION_SEGMENTS, prep.segmentStarts().size());
      for (int g = 0; g < prep.segmentStarts().size(); g++) {
        segmentStarts.set(g, prep.segmentStarts().get(g));
        segmentOffsets.set(g, prep.segmentOffsets().get(g));
      }

      String equalityId = prep.equalityId();
      meta.set(DeviceKernels.META_HAS_EQUALITY_DELETES, equalityId.isEmpty() ? 0 : 1);
      meta.set(DeviceKernels.META_KEY_NULLABLE, keyNullable ? 1 : 0);
      setPath.setLength(0);
      setRowsHolder[0] = 0;
      if (!equalityId.isEmpty() && !equalityId.equals(setOnDevice)) {
        if (prep.setPaths() != null) {
          hold(setPath, prep.setPaths());
          setRowsHolder[0] = prep.setRows();
          setColumn[0] = prep.setColumn();
          probeSizes[1] = (int) prep.setRows();
          meta.set(DeviceKernels.META_EQUALITY_SET_HAS_NULL, 0);
        } else {
          upload(executionPlan, prep.equality());
          meta.set(DeviceKernels.META_EQUALITY_SET_HAS_NULL, prep.equality().hasNull() ? 1 : 0);
        }
        setOnDevice = equalityId;
      }
    }

    private static void hold(StringBuilder holder, String paths) {
      if (paths.length() >= PATH_CAPACITY) {
        throw new UnsupportedOperationException(
            "A batch names "
                + paths.length()
                + " characters of paths; the limit is "
                + PATH_CAPACITY);
      }
      holder.append(paths);
    }

    /** Puts a host-loaded equality-delete set on the device. */
    private void upload(TornadoExecutionPlan executionPlan, HostDeletes.EqualitySet set) {
      long[] keys = set.keys();
      probeSizes[1] = keys.length;
      if (equalityKeyType == CudfType.INT32) {
        for (int i = 0; i < keys.length; i++) {
          setInts.set(i, (int) keys[i]);
        }
        executionPlan.transferToDevice(setInts);
      } else {
        MemorySegment.copy(
            MemorySegment.ofArray(keys),
            0,
            setLongs.getSegment(),
            0,
            (long) keys.length * ValueLayout.JAVA_LONG.byteSize());
        executionPlan.transferToDevice(setLongs);
      }
    }

    @Override
    public void close() {
      try {
        thread
            .submit(
                () -> {
                  if (plan != null) {
                    try {
                      plan.close();
                    } catch (TornadoExecutionPlanException e) {
                      throw new IllegalStateException(e);
                    }
                    plan = null;
                  }
                })
            .get();
      } catch (java.util.concurrent.ExecutionException e) {
        throw new IllegalStateException("Cannot release the device plan", e.getCause());
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException(e);
      } finally {
        thread.shutdown();
      }
    }
  }

  /** GROUP BY on the device: only each batch's groups come back. */
  private final class AggregateRegion extends Region {
    private final int sumColumns;
    private final int groupOffset;
    private final int buckets;
    private final IntArray groupKeys;
    private final IntArray sumKeys;
    private final DoubleArray sums;
    private final IntArray sumGroups = new IntArray(1);
    private final IntArray countKeys;
    private final DoubleArray counts;
    private final IntArray countGroups = new IntArray(1);
    private final IntArray validColumn;

    AggregateRegion(Planned planned, String groupColumn, int buckets, String[] sumNames) {
      super(planned);
      this.sumColumns = sumNames.length;
      this.buckets = buckets;
      this.groupOffset =
          groupColumn == null
              ? -1
              : layout.slot(table.schema().findField(groupColumn).fieldId()) * capacity;
      int[] validColumns = new int[sumNames.length];
      for (int c = 0; c < sumNames.length; c++) {
        Types.NestedField field = table.schema().findField(sumNames[c]);
        validColumns[c] = field.isOptional() ? layout.readIndex(field.fieldId()) : -1;
      }
      this.validColumn = IntArray.fromArray(validColumns);
      this.groupKeys = new IntArray(capacity);
      this.sumKeys = new IntArray(capacity);
      this.sums = new DoubleArray((long) sumColumns * capacity);
      this.countKeys = new IntArray(capacity);
      this.counts = new DoubleArray(capacity);
    }

    @Override
    void consumer(TaskGraph graph) {
      if (buckets > 0) {
        graph.task(
            "buckets",
            DeviceKernels::bucketGroupKeys,
            live,
            longs,
            groupOffset,
            buckets,
            groupKeys);
      } else {
        graph.task("groups", DeviceKernels::groupKeys, live, ints, groupOffset, groupKeys);
      }
      if (layout.anyOptional()) {
        graph
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, validColumn)
            .task(
                "nulls",
                DeviceKernels::zeroNulls,
                doubles,
                valid,
                validColumn,
                sumColumns,
                capacity);
      }
      graph
          .libraryTask(
              "sum",
              Cudf::groupAggregate,
              capacity,
              sumColumns,
              CudfAggregation.SUM.code(),
              groupKeys,
              doubles,
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
              doubles,
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
    }

    /** Copies back one batch's groups and merges them as partial aggregates. */
    long collect(TornadoExecutionResult executed, AggregateResult result) {
      executed.transferToHost(sumGroups, countGroups);
      int groups = sumGroups.get(0);
      int counted = countGroups.get(0);
      if (groups == 0) {
        return 0L;
      }
      executed.transferToHost(new DataRange(sumKeys).withSize(groups));
      executed.transferToHost(new DataRange(countKeys).withSize(counted));
      executed.transferToHost(new DataRange(counts).withSize(counted));
      for (int c = 0; c < sumColumns; c++) {
        executed.transferToHost(
            new DataRange(sums).withOffset((long) c * capacity).withSize(groups));
      }
      Map<Integer, Long> rowsByKey = Maps.newHashMap();
      for (int g = 0; g < counted; g++) {
        rowsByKey.put(countKeys.get(g), (long) counts.get(g));
      }
      long liveRows = 0;
      double[] values = new double[sumColumns];
      for (int g = 0; g < groups; g++) {
        int key = sumKeys.get(g);
        if (key == DeviceKernels.DEAD_GROUP) {
          continue;
        }
        for (int c = 0; c < sumColumns; c++) {
          values[c] = sums.get(c * capacity + g);
        }
        long rows = rowsByKey.get(key);
        result.add(key, rows, values);
        liveRows += rows;
      }
      return liveRows;
    }
  }

  /** Projection on the device: live rows are compacted, and only they come back. */
  private static class MaterializeRegion extends Region {
    private final int intSlots;
    private final int longSlots;
    private final int doubleSlots;
    private final int validSlots;
    private final IntArray compacted;
    private final IntArray liveCount = new IntArray(1);
    private final IntArray outInts;
    private final LongArray outLongs;
    private final DoubleArray outDoubles;
    private final ByteArray outValid;
    private final Map<String, Integer> idByName = Maps.newHashMap();

    IntArray liveCount() {
      return liveCount;
    }

    IntArray outInts() {
      return outInts;
    }

    LongArray outLongs() {
      return outLongs;
    }

    DoubleArray outDoubles() {
      return outDoubles;
    }

    ByteArray outValid() {
      return outValid;
    }

    MaterializeRegion(Planned planned) {
      super(planned);
      this.intSlots = layout.slots(CudfType.INT32);
      this.longSlots = layout.slots(CudfType.INT64);
      this.doubleSlots = layout.slots(CudfType.FLOAT64);
      this.validSlots = layout.anyOptional() ? layout.fields().size() : 0;
      this.compacted = new IntArray(capacity);
      this.outInts = new IntArray(Math.max(1L, (long) intSlots * capacity));
      this.outLongs = new LongArray(Math.max(1L, (long) longSlots * capacity));
      this.outDoubles = new DoubleArray(Math.max(1L, (long) doubleSlots * capacity));
      this.outValid = new ByteArray(Math.max(1L, (long) validSlots * capacity));
      for (Types.NestedField field : layout.fields()) {
        idByName.put(field.name(), field.fieldId());
      }
    }

    @Override
    void consumer(TaskGraph graph) {
      graph.libraryTask(
          "compact", Cudf::selectedIndices, capacity, capacity, live, compacted, liveCount);
      if (intSlots > 0) {
        graph.task(
            "gatherInts",
            DeviceKernels::gatherInts,
            compacted,
            liveCount,
            ints,
            intSlots,
            capacity,
            outInts);
      }
      if (longSlots > 0) {
        graph.task(
            "gatherLongs",
            DeviceKernels::gatherLongs,
            compacted,
            liveCount,
            longs,
            longSlots,
            capacity,
            outLongs);
      }
      if (doubleSlots > 0) {
        graph.task(
            "gatherDoubles",
            DeviceKernels::gatherDoubles,
            compacted,
            liveCount,
            doubles,
            doubleSlots,
            capacity,
            outDoubles);
      }
      if (validSlots > 0) {
        graph.task(
            "gatherValid",
            DeviceKernels::gatherBytes,
            compacted,
            liveCount,
            valid,
            validSlots,
            capacity,
            outValid);
      }
      graph.transferToHost(
          DataTransferMode.UNDER_DEMAND, liveCount, outInts, outLongs, outDoubles, outValid);
    }

    /** Copies back one batch's live rows and hands them to the consumer. */
    long collect(TornadoExecutionResult executed, Consumer<LiveRows> consumer) {
      executed.transferToHost(liveCount);
      int rows = liveCount.get(0);
      if (rows > 0) {
        copySlots(executed, outInts, intSlots, rows);
        copySlots(executed, outLongs, longSlots, rows);
        copySlots(executed, outDoubles, doubleSlots, rows);
        copySlots(executed, outValid, validSlots, rows);
      }
      consumer.accept(
          new HostBatch(layout, idByName, rows, capacity, outInts, outLongs, outDoubles, outValid));
      return rows;
    }

    private void copySlots(
        TornadoExecutionResult executed, TornadoNativeArray array, int slots, int rows) {
      for (int s = 0; s < slots; s++) {
        executed.transferToHost(
            new DataRange(array).withOffset((long) s * capacity).withSize(rows));
      }
    }
  }

  /** Compaction: the live rows of a batch, compacted and written to a new Parquet file. */
  private final class RewriteRegion extends MaterializeRegion {
    private final StringBuilder outputPath = new StringBuilder(PATH_CAPACITY);
    private final String names;
    private final int[] fieldIds;
    private final ParquetColumnType[] types;
    private final boolean[] optional;
    private String location;

    RewriteRegion(Planned planned) {
      super(planned);
      List<Types.NestedField> fields = layout.fields();
      this.fieldIds = new int[fields.size()];
      this.types = new ParquetColumnType[fields.size()];
      this.optional = new boolean[fields.size()];
      StringBuilder joined = new StringBuilder();
      for (int c = 0; c < fields.size(); c++) {
        Types.NestedField field = fields.get(c);
        fieldIds[c] = field.fieldId();
        optional[c] = field.isOptional();
        types[c] =
            switch (field.type().typeId()) {
              case INTEGER -> ParquetColumnType.INT32;
              case DATE -> ParquetColumnType.DATE;
              case LONG -> ParquetColumnType.INT64;
              case TIMESTAMP -> ParquetColumnType.TIMESTAMP_MICROS;
              case DOUBLE -> ParquetColumnType.FLOAT64;
              default ->
                  throw new UnsupportedOperationException(
                      "Cannot write " + field + " on the device");
            };
        if (c > 0) {
          joined.append('\n');
        }
        joined.append(field.name());
      }
      this.names = joined.toString();
    }

    @Override
    void consumer(TaskGraph graph) {
      super.consumer(graph);
      graph.libraryTask(
          "write",
          Cudf::writeParquetColumns,
          outputPath,
          names,
          fieldIds,
          types,
          optional,
          new long[] {-1},
          liveCount(),
          capacity,
          outInts(),
          outLongs(),
          outDoubles(),
          outValid(),
          2,
          0);
    }

    /** A new file a batch, where Iceberg's location provider puts data files. */
    @Override
    void stageOutput() {
      location =
          table
              .locationProvider()
              .newDataLocation("gpu-rewrite-" + java.util.UUID.randomUUID() + ".parquet");
      outputPath.setLength(0);
      outputPath.append(HostDeletes.localPath(location));
    }

    /** The batch's file as an Iceberg data file, with metrics from the footer libcudf wrote. */
    long collect(TornadoExecutionResult executed, List<DataFile> written) {
      executed.transferToHost(liveCount());
      int rows = liveCount().get(0);
      org.apache.iceberg.io.InputFile file = table.io().newInputFile(location);
      if (rows == 0) {
        table.io().deleteFile(location);
        return 0;
      }
      org.apache.iceberg.Metrics fileMetrics =
          org.apache.iceberg.parquet.ParquetUtil.fileMetrics(
              file, org.apache.iceberg.MetricsConfig.forTable(table));
      written.add(
          org.apache.iceberg.DataFiles.builder(table.spec())
              .withPath(location)
              .withFormat(org.apache.iceberg.FileFormat.PARQUET)
              .withFileSizeInBytes(file.getLength())
              .withMetrics(fileMetrics)
              .build());
      return rows;
    }
  }
}
