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
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.FileContent;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.MetadataColumns;
import org.apache.iceberg.RewriteFiles;
import org.apache.iceberg.Table;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.relocated.com.google.common.collect.Maps;
import org.apache.iceberg.relocated.com.google.common.collect.Sets;
import org.apache.iceberg.types.Types;
import uk.ac.manchester.tornado.api.DataRange;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
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

  /** The longest string the device packs into an INT64. */
  static final int STRING_BYTES = 8;

  private static final boolean DEBUG = Boolean.getBoolean("iceberg.tornadovm.debug");

  /**
   * How equality deletes are probed: {@code sorted} (the default) sorts the delete keys on the
   * device once a set and binary-searches them from a kernel on every batch; {@code hash} uses
   * cuDF's contains, which hashes the set again on every batch.
   */
  private static final boolean SORTED_PROBE =
      !"hash".equals(System.getProperty("iceberg.tornadovm.probe", "sorted"));

  /** Characters of paths one batch may name in one holder: data files, delete files. */
  private static final int PATH_CAPACITY = 1 << 16;

  /**
   * Rows a device batch may hold. Files are grouped up to this many rows into one read, because a
   * read of one small file is mostly latency; a file larger than this is a batch on its own. The
   * bigger the read, the cheaper cuDF's ZSTD decompression of it: for 100M rows in 1.56M-row files,
   * 1, 2 and 5 files a read cost 435, 233 and 137 ms of decompression a scan, so the default fits
   * five such files.
   */
  private static final long BATCH_ROWS = Long.getLong("iceberg.tornadovm.batch-rows", 8_000_000L);

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
    sharedPools.values().forEach(SharedPool::close);
    sharedPools.clear();
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

  /**
   * Sums of {@link DeviceValue}s and the row count per group of up to four columns -- integer,
   * long, date, timestamp or string (of up to 8 bytes) -- over the live rows that pass the scan's
   * filter: the shape of TPC-H's Q1 and, with no group columns, Q6.
   *
   * <p>One kernel builds the group keys as INT64 columns (a string as its packed value), another
   * evaluates every value of every row, and cuDF groups by all keys at once
   * (Cudf.groupAggregateLongs); only the groups come back to the host.
   */
  public GroupedAggregate aggregate(List<String> groupColumns, List<DeviceValue> values) {
    if (groupColumns.size() > 4 || values.isEmpty()) {
      throw new IllegalArgumentException("Up to 4 group columns and at least one value");
    }
    metrics = new ScanMetrics();
    long start = System.nanoTime();
    List<String> projection = Lists.newArrayList(groupColumns);
    for (String column : groupColumns) {
      org.apache.iceberg.types.Type.TypeID type = column(column).type().typeId();
      switch (type) {
        case INTEGER, LONG, DATE, TIMESTAMP, STRING, DECIMAL -> {}
        default -> throw new UnsupportedOperationException("Cannot group by " + column(column));
      }
    }
    for (DeviceValue value : values) {
      for (DeviceValue.Factor factor : value.factors()) {
        org.apache.iceberg.types.Type.TypeID type = column(factor.column()).type().typeId();
        // A decimal sums as its unscaled value.
        if (type != org.apache.iceberg.types.Type.TypeID.INTEGER
            && type != org.apache.iceberg.types.Type.TypeID.LONG
            && type != org.apache.iceberg.types.Type.TypeID.DECIMAL
            && type != org.apache.iceberg.types.Type.TypeID.DOUBLE) {
          throw new UnsupportedOperationException("Not a numeric column: " + factor.column());
        }
        if (!projection.contains(factor.column())) {
          projection.add(factor.column());
        }
      }
    }
    Planned planned = new Planned(null, projection);
    String key = "grouped:" + groupColumns + ":" + values;
    List<Region> streams =
        reuse(key, planned, () -> new GroupedRegion(planned, groupColumns, values));
    metrics.plan((System.nanoTime() - start) / 1e6);
    GroupedAggregate result = new GroupedAggregate(values.size());
    run(streams, planned, (region, executed) -> ((GroupedRegion) region).collect(executed, result));
    metrics.total((System.nanoTime() - start) / 1e6);
    return result;
  }

  private Types.NestedField column(String name) {
    Types.NestedField field = table.schema().findField(name);
    if (field == null) {
      throw new IllegalArgumentException("No column " + name);
    }
    return field;
  }

  /** Whether the device scan can read this task: a whole data file, as it reads files. */
  public static boolean readsWholeFile(FileScanTask task) {
    return HostDeletes.wholeFile(task);
  }

  /**
   * {@link #materialize(List, Consumer)} over file scan tasks someone else planned -- a Spark
   * task's -- rather than the table's: their data files read with their delete files applied.
   * Concurrent calls share the device: their batches go into one queue that every stream takes
   * from, so the streams stay busy while each caller waits only for its own batches.
   */
  public long materialize(
      List<FileScanTask> tasks, List<String> columns, Consumer<LiveRows> consumer) {
    if (tasks.isEmpty()) {
      return 0;
    }
    Planned planned = new Planned(null, columns, tasks);
    SharedPool pool = sharedPools.computeIfAbsent(planned.signature(), ignored -> new SharedPool());
    return pool.run(
        planned, (region, executed) -> ((MaterializeRegion) region).collect(executed, consumer));
  }

  private final Map<String, SharedPool> sharedPools =
      new java.util.concurrent.ConcurrentHashMap<>();

  /** Regions that take batches from any caller's request, for one projection. */
  private final class SharedPool {
    private final java.util.concurrent.LinkedBlockingQueue<Work> queue =
        new java.util.concurrent.LinkedBlockingQueue<>();
    private final java.util.concurrent.locks.ReentrantReadWriteLock lock =
        new java.util.concurrent.locks.ReentrantReadWriteLock();
    private final ExecutorService hostPool =
        Executors.newFixedThreadPool(
            HOST_THREADS,
            runnable -> {
              Thread thread = new Thread(runnable, "tornadovm-host");
              thread.setDaemon(true);
              return thread;
            });
    private List<Region> regions;

    /** A batch of a request; a null planned stops the region that takes it. */
    private record Work(
        Planned planned,
        int index,
        CompletableFuture<HostDeletes.FilePrep> prep,
        java.util.function.ToLongBiFunction<Region, TornadoExecutionResult> perBatch,
        ScanMetrics metrics,
        CompletableFuture<Long> done) {}

    long run(
        Planned planned,
        java.util.function.ToLongBiFunction<Region, TornadoExecutionResult> perBatch) {
      ensureFits(planned);
      lock.readLock().lock();
      try {
        ScanMetrics batchMetrics = new ScanMetrics();
        List<CompletableFuture<Long>> done = Lists.newArrayList();
        for (int i = 0; i < planned.batches.size(); i++) {
          List<FileScanTask> batch = planned.batches.get(i);
          CompletableFuture<HostDeletes.FilePrep> prep =
              CompletableFuture.supplyAsync(() -> planned.host.prepare(batch), hostPool);
          CompletableFuture<Long> finished = new CompletableFuture<>();
          done.add(finished);
          queue.add(new Work(planned, i, prep, perBatch, batchMetrics, finished));
        }
        long rows = 0;
        for (CompletableFuture<Long> finished : done) {
          rows += finished.join();
        }
        metrics = batchMetrics;
        return rows;
      } catch (java.util.concurrent.CompletionException e) {
        throw e.getCause() instanceof RuntimeException runtime
            ? runtime
            : new IllegalStateException("Device scan failed", e.getCause());
      } finally {
        lock.readLock().unlock();
      }
    }

    /** Builds the regions, or rebuilds them bigger once no request is in flight. */
    private void ensureFits(Planned planned) {
      lock.readLock().lock();
      try {
        if (regions != null && regions.get(0).fits(planned)) {
          return;
        }
      } finally {
        lock.readLock().unlock();
      }
      lock.writeLock().lock();
      try {
        if (regions != null && regions.get(0).fits(planned)) {
          return;
        }
        if (regions != null) {
          for (int r = 0; r < regions.size(); r++) {
            queue.add(new Work(null, -1, null, null, null, null));
          }
          regions.forEach(Region::close);
        }
        List<Region> created = Lists.newArrayList();
        for (int r = 0; r < STREAMS; r++) {
          Region region = new MaterializeRegion(planned);
          created.add(region);
          region.thread.execute(() -> loop(region));
        }
        regions = created;
      } finally {
        lock.writeLock().unlock();
      }
    }

    void close() {
      lock.writeLock().lock();
      try {
        if (regions != null) {
          for (int r = 0; r < regions.size(); r++) {
            queue.add(new Work(null, -1, null, null, null, null));
          }
          regions.forEach(Region::close);
          regions = null;
        }
        hostPool.shutdown();
      } finally {
        lock.writeLock().unlock();
      }
    }

    private void loop(Region region) {
      TornadoExecutionPlan executionPlan = region.plan();
      while (true) {
        Work work;
        try {
          work = queue.take();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          return;
        }
        if (work.planned() == null) {
          return;
        }
        try {
          work.done()
              .complete(
                  executeBatch(
                      region,
                      executionPlan,
                      work.planned(),
                      work.index(),
                      work.prep().join(),
                      work.perBatch(),
                      work.metrics()));
        } catch (Throwable failure) {
          work.done().completeExceptionally(failure);
        }
      }
    }
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
   * <p>A v3 table's rewrite also writes each row's lineage -- {@code _row_id} and {@code
   * _last_updated_sequence_number}, which the old files left to inheritance -- as columns of the
   * new files, so rows keep them. Data files that already store lineage columns are refused.
   */
  public RewriteResult rewriteDataFiles() {
    return rewrite(List.of());
  }

  /**
   * {@link #rewriteDataFiles()}, writing each new file's rows in Iceberg's Z-order of the given
   * columns -- the order {@code rewrite_data_files(strategy => 'sort', sort_order =>
   * 'zorder(...)')} writes -- so that files cluster on all of them at once.
   *
   * <p>A kernel computes every row's Z-order value as Iceberg's {@code ZOrderByteUtils} does, cuDF
   * sorts the rows of a batch by it, and the live rows are gathered in that order and written. Each
   * output file is sorted; files are not sorted against each other (a batch is the unit, as a file
   * group is Spark's). Up to {@value DeviceKernels#MAX_ZORDER_COLUMNS} integer, long, date,
   * timestamp or double columns.
   */
  public RewriteResult rewriteDataFilesZOrder(String... zorderColumns) {
    if (zorderColumns.length == 0 || zorderColumns.length > DeviceKernels.MAX_ZORDER_COLUMNS) {
      throw new IllegalArgumentException(
          "Z-order takes 1 to " + DeviceKernels.MAX_ZORDER_COLUMNS + " columns");
    }
    return rewrite(List.of(zorderColumns));
  }

  private RewriteResult rewrite(List<String> zorderColumns) {
    if (filter.op() != Expression.Operation.TRUE) {
      throw new UnsupportedOperationException("A rewrite of part of a table is not supported");
    }
    int formatVersion =
        ((org.apache.iceberg.HasTableOperations) table).operations().current().formatVersion();
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
    List<Region> streams =
        reuse(
            "rewrite" + (formatVersion >= 3 ? "-lineage" : "") + "-z" + zorderColumns,
            planned,
            () -> new RewriteRegion(planned, formatVersion >= 3, zorderColumns));
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
    // Planned tasks someone else will send more of get every stream from the start.
    int streams = Math.max(1, planned.given ? STREAMS : Math.min(STREAMS, planned.batches.size()));
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
  /** Per field of the layout, whether every file of the batch counts no null in it. */
  private static boolean[] nullFree(ScanLayout layout, List<FileScanTask> batch) {
    List<Types.NestedField> fields = layout.fields();
    boolean[] free = new boolean[fields.size()];
    for (int c = 0; c < free.length; c++) {
      Types.NestedField field = fields.get(c);
      boolean none = true;
      for (FileScanTask task : batch) {
        Map<Integer, Long> counts = task.file().nullValueCounts();
        Long count = counts == null ? null : counts.get(field.fieldId());
        if (count == null || count != 0) {
          none = false;
          break;
        }
      }
      free[c] = field.isRequired() || none;
    }
    return free;
  }

  /** One batch on one region: stage it, run the plan, hand the result over. */
  private long executeBatch(
      Region region,
      TornadoExecutionPlan executionPlan,
      Planned planned,
      int i,
      HostDeletes.FilePrep prep,
      java.util.function.ToLongBiFunction<Region, TornadoExecutionResult> perBatch,
      ScanMetrics into) {
    into.deleteLoad(prep.hostNanos() / 1e6);
    region.nullFree = nullFree(region.layout, planned.batches.get(i));
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
    region.checkStrings(executed);
    long copyStart = System.nanoTime();
    long rows = perBatch.applyAsLong(region, executed);
    if (DEBUG) {
      System.err.printf(
          "[mor] batch %d region %d files %d rows %d live %d%n",
          i, System.identityHashCode(region), prep.files(), prep.rows(), rows);
    }
    long end = System.nanoTime();
    into.device((copyStart - deviceStart) / 1e6);
    into.copyBack((end - copyStart) / 1e6);
    into.batch(prep.rows(), prep.files());
    into.live(rows);
    return rows;
  }

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
                    long rows =
                        executeBatch(
                            region,
                            executionPlan,
                            planned,
                            i,
                            preps.get(i).join(),
                            perBatch,
                            metrics);
                    liveRows.addAndGet(rows);
                  }
                }));
      }
      for (java.util.concurrent.Future<?> worker : workers) {
        worker.get();
      }
    } catch (java.util.concurrent.ExecutionException e) {
      for (Throwable cause = e.getCause(); cause != null; cause = cause.getCause()) {
        if (cause instanceof UnsupportedOperationException unsupported) {
          throw unsupported;
        }
        if (String.valueOf(cause.getMessage()).contains("a column was sized for")) {
          // The string read found more bytes than 8 a row: some value is longer than that.
          throw new UnsupportedOperationException(
              "A string column holds values longer than "
                  + STRING_BYTES
                  + " bytes, which the device does not support",
              e.getCause());
        }
      }
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
    private final List<Integer> keyIds;
    private final CudfType equalityKeyType;
    private final boolean keyNullable;
    private final int capacity;
    private final int equalityCapacity;
    private final int positionCapacity;
    private final int maxFiles;
    private final boolean filtered;
    private final List<DevicePredicate> batchFilters = Lists.newArrayList();
    // Bytes a string column of a batch may decode to, from the footers' size statistics.
    private final long stringStride;
    // Columns whose values the device compares -- filters and keys: a string among them must pack.
    private final java.util.Set<Integer> keyedIds = new java.util.HashSet<>();
    private final boolean given;

    Planned(String groupColumn, List<String> projection) {
      this(groupColumn, projection, null);
    }

    /**
     * A given task's residual as a device filter, or null when the device cannot run it: an
     * operation it lacks, or a string column, whose values may not pack into a key.
     */
    private DevicePredicate deviceResidual(Expression residual) {
      if (residual.op() == Expression.Operation.TRUE) {
        return null;
      }
      for (int fieldId : DevicePredicate.fieldIds(residual, table.schema())) {
        if (table.schema().findField(fieldId).type().typeId()
            == org.apache.iceberg.types.Type.TypeID.STRING) {
          return null;
        }
      }
      try {
        return DevicePredicate.compile(residual, table.schema());
      } catch (UnsupportedOperationException e) {
        return null;
      }
    }

    /**
     * {@code given}: file scan tasks someone else planned (a Spark task's, say) instead of the
     * table's. Their sizes are rounded up, so that the next group of tasks most likely fits the
     * same device buffers and plans.
     */
    Planned(String groupColumn, List<String> projection, List<FileScanTask> given) {
      this.snapshotId =
          table.currentSnapshot() == null ? null : table.currentSnapshot().snapshotId();
      this.given = given != null;
      if (given != null) {
        this.tasks = Lists.newArrayList(given);
      } else {
        try (CloseableIterable<FileScanTask> planned = table.newScan().filter(filter).planFiles()) {
          this.tasks = Lists.newArrayList(planned);
        } catch (IOException e) {
          throw new UncheckedIOException(e);
        }
      }
      this.equalityKeyId = HostDeletes.equalityKeyId(tasks);
      if (equalityKeyId != null
          && table.schema().findField(equalityKeyId).type().typeId()
              == org.apache.iceberg.types.Type.TypeID.STRING) {
        throw new UnsupportedOperationException(
            "Equality deletes on a string column are not supported on the device");
      }
      List<Integer> allKeyIds = HostDeletes.equalityKeyIds(tasks);
      this.keyIds = allKeyIds.size() > 1 ? allKeyIds : List.of();
      // Given tasks carry the caller's filter as their residuals. The caller applies it again
      // (Spark does), so the device only narrows what comes back, and a residual it cannot run
      // simply filters nothing.
      Map<Expression, DevicePredicate> residuals = Maps.newHashMap();
      Set<Integer> filterIds = Sets.newHashSet(DevicePredicate.fieldIds(filter, table.schema()));
      if (given != null) {
        for (FileScanTask task : tasks) {
          residuals.computeIfAbsent(
              task.residual(),
              residual -> {
                DevicePredicate compiled = deviceResidual(residual);
                if (compiled != null) {
                  filterIds.addAll(DevicePredicate.fieldIds(residual, table.schema()));
                }
                return compiled == null ? DevicePredicate.alwaysTrue() : compiled;
              });
        }
      }
      this.filtered =
          filter.op() != Expression.Operation.TRUE
              || residuals.keySet().stream()
                  .anyMatch(r -> r.op() != Expression.Operation.TRUE && deviceResidual(r) != null);
      List<String> read = Lists.newArrayList(projection);
      for (int keyId : keyIds) {
        String name = table.schema().findField(keyId).name();
        if (!read.contains(name)) {
          read.add(name);
        }
      }
      for (int fieldId : filterIds) {
        String name = table.schema().findField(fieldId).name();
        if (!read.contains(name)) {
          read.add(name);
        }
      }
      this.layout = new ScanLayout(table.schema(), equalityKeyId, groupColumn, read);
      this.host = new HostDeletes(table, layout, equalityKeyId, true).withKeyIds(keyIds);
      this.equalityKeyType = equalityKeyId == null ? null : layout.type(equalityKeyId);
      this.keyNullable =
          equalityKeyId != null && table.schema().findField(equalityKeyId).isOptional();

      // Big batches decompress cheaply, but a small table cut into a few of them leaves streams
      // idle:
      // a batch holds at most an even share of the table a stream, up to BATCH_ROWS.
      long totalRows = tasks.stream().mapToLong(t -> t.file().recordCount()).sum();
      // Given tasks share the streams with other callers' requests, so they are not cut smaller.
      long batchRows =
          given != null
              ? BATCH_ROWS
              : Math.max(1, Math.min(BATCH_ROWS, (totalRows + STREAMS - 1) / STREAMS));
      List<FileScanTask> current = Lists.newArrayList();
      long currentRows = 0;
      String currentSet = null;
      for (FileScanTask task : tasks) {
        long rows = task.file().recordCount();
        // Files batch together only if they share their equality deletes and their residual.
        String set = HostDeletes.equalityId(task) + "|" + task.residual();
        if (!current.isEmpty() && (currentRows + rows > batchRows || !set.equals(currentSet))) {
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
        Expression residual = batch.get(0).residual();
        batchFilters.add(
            !filtered
                ? DevicePredicate.alwaysTrue()
                : given != null
                    ? residuals.get(residual)
                    : DevicePredicate.compile(residual, table.schema()));
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
      // Rounded up to whole thread blocks: TornadoVM's CUDA scheduler picks a block size that
      // divides
      // the loop bound exactly, and a bound with no such divisor (a prime row count, say) gets
      // one-thread blocks -- 28x slower kernels for a batch of 7,987,813 rows. The padding rows are
      // never live.
      this.capacity =
          (int)
              Math.min(
                  Integer.MAX_VALUE / 8,
                  (Math.max(maxRows, given != null ? Math.min(BATCH_ROWS, roundUp(maxRows)) : 0)
                          + 1023)
                      / 1024
                      * 1024);
      keyedIds.addAll(DevicePredicate.fieldIds(filter, table.schema()));
      if (equalityKeyId != null) {
        keyedIds.add(equalityKeyId);
      }
      keyedIds.addAll(keyIds);
      long stride = stringStride(batches, layout, capacity);
      this.stringStride =
          given != null && stride > 1
              ? Math.min(roundUp(stride), Integer.MAX_VALUE / Math.max(1, layout.stringCount()))
              : stride;
      boolean padded = given != null;
      this.equalityCapacity = (int) (padded ? roundUp(maxSet) : maxSet);
      this.positionCapacity = (int) (padded ? roundUp(maxPositions) : maxPositions);
      this.maxFiles = padded ? Math.max(64, files) : files;
    }

    /**
     * The bytes any string column of any batch decodes to, from the data files' footers
     * (parquet-java records them); a writer that left them out gets 16 bytes a row, and a batch
     * that needs more is refused rather than truncated.
     */
    private static long stringStride(
        List<List<FileScanTask>> batches, ScanLayout layout, int rows) {
      if (layout.stringCount() == 0) {
        return 1;
      }
      long most = 0;
      for (List<FileScanTask> batch : batches) {
        for (int s = 0; s < layout.stringCount(); s++) {
          int fieldId = layout.fields().get(layout.numericCount() + s).fieldId();
          long bytes = 0;
          for (FileScanTask task : batch) {
            long file =
                HostDeletes.stringBytes(HostDeletes.localPath(task.file().location()), fieldId);
            if (file < 0) {
              bytes = 16L * rows;
              break;
            }
            bytes += file;
          }
          most = Math.max(most, bytes);
        }
      }
      long stride = (most + 1023) / 1024 * 1024 + 1024;
      if (stride * layout.stringCount() > Integer.MAX_VALUE) {
        throw new UnsupportedOperationException(
            "String columns of " + stride + " bytes a batch are beyond one device buffer");
      }
      return stride;
    }

    /** The next power of two, at least 1024. */
    private static long roundUp(long value) {
      return Math.max(1024, Long.highestOneBit(Math.max(1, value - 1)) << 1);
    }

    /** What decides the shape of the graph, apart from sizes. */
    String signature() {
      StringBuilder signature = new StringBuilder();
      for (int c = 0; c < layout.fields().size(); c++) {
        Types.NestedField field = layout.fields().get(c);
        signature.append(field.fieldId()).append(field.isOptional() ? "?" : "").append(',');
      }
      return signature
          .append(equalityKeyId)
          .append(keyIds)
          .append(keyNullable)
          .append(filtered)
          .toString();
    }
  }

  /**
   * The device buffers, host holders and execution plan of one query shape: read, delete
   * application and live-row kernel, which every shape shares, plus what the shape adds.
   */
  private abstract static class Region implements AutoCloseable {
    /**
     * Per field of the layout, whether no file of the current batch holds a null there (by its
     * metrics), so that its validity bytes need not come back.
     */
    boolean[] nullFree = new boolean[0];

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
    final IntArray sortedInts;
    final LongArray sortedLongs;
    final int[] sortSize = new int[1];
    final IntArray setSize = new IntArray(1);
    // Equality deletes over several columns: the set as loaded, then in hash order.
    final int keyColumns;
    final IntArray keySpec;
    final LongArray rawHashes;
    final LongArray rawKeys;
    final ByteArray rawNulls;
    final IntArray keyOrder;
    final int[] keySortSize = {0, 1};
    final IntArray keySetChanged = new IntArray(1);
    final LongArray keyHashes;
    final LongArray keyValues;
    final ByteArray keyNulls;
    final IntArray ints;
    final LongArray longs;
    final DoubleArray doubles;
    final ByteArray valid;
    final ByteArray live;
    final int[] stringColumns;
    final IntArray stringOffsets;
    final ByteArray stringChars;
    final IntArray stringErrors;
    final long stringStride;
    final java.util.Set<Integer> keyedIds;

    /**
     * Refuses a batch whose filter or key column held a string longer than {@link #STRING_BYTES}:
     * those are compared packed, and a longer value would compare wrongly. Strings only returned to
     * the host can be of any length.
     */
    void checkStrings(TornadoExecutionResult executed) {
      if (stringColumns.length > 0) {
        executed.transferToHost(stringErrors);
        for (int s = 0; s < stringColumns.length; s++) {
          Types.NestedField field = layout.fields().get(layout.numericCount() + s);
          if (stringErrors.get(s) != 0 && keyedIds.contains(field.fieldId())) {
            throw new UnsupportedOperationException(
                "String column "
                    + field.name()
                    + " holds a value longer than "
                    + STRING_BYTES
                    + " bytes; filters and keys on such columns are not supported on the device");
          }
        }
      }
    }

    private TornadoExecutionPlan plan;
    final KernelContext context = new KernelContext();
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
      if (layout.numericCount() == 0) {
        throw new UnsupportedOperationException("A device scan reads at least one numeric column");
      }
      this.columns = new int[layout.numericCount()];
      this.stringColumns = new int[layout.stringCount()];
      long strings = layout.stringCount();
      this.stringOffsets = new IntArray(strings == 0 ? 1 : (int) (strings * (capacity + 1L)));
      this.stringStride = planned.stringStride;
      this.stringChars = new ByteArray(strings == 0 ? 1L : strings * stringStride);
      this.stringErrors = new IntArray((int) Math.max(1, strings));
      this.keyedIds = new java.util.HashSet<>(planned.keyedIds);
      this.probeSizes[0] = capacity;

      boolean equality = equalityKeyId != null;
      this.keyColumns = planned.keyIds.size();
      boolean composite = keyColumns > 1;
      long keyCells = composite ? (long) keyColumns * equalityCapacity : 1L;
      this.keySpec = new IntArray(Math.max(1, 3 * keyColumns));
      for (int c = 0; c < keyColumns; c++) {
        int keyId = planned.keyIds.get(c);
        keySpec.set(3 * c, layout.type(keyId) == CudfType.INT32 ? 0 : 1);
        keySpec.set(3 * c + 1, layout.slot(keyId));
        keySpec.set(
            3 * c + 2,
            layout.fields().get(layout.readIndex(keyId)).isOptional()
                ? layout.readIndex(keyId)
                : -1);
      }
      this.rawHashes = new LongArray(composite ? equalityCapacity : 1);
      this.rawKeys = new LongArray(keyCells);
      this.rawNulls = new ByteArray(keyCells);
      this.keyOrder = new IntArray(composite ? equalityCapacity : 1);
      this.keyHashes = new LongArray(composite ? equalityCapacity : 1);
      this.keyValues = new LongArray(keyCells);
      this.keyNulls = new ByteArray(keyCells);
      this.positionBits = new ByteArray((capacity + 7) / 8);
      this.positions = new LongArray(positionCapacity);
      this.segmentStarts = new IntArray(maxFiles);
      this.segmentOffsets = new IntArray(maxFiles);
      this.inSet = new ByteArray(equality || composite ? capacity : 1);
      this.setInts =
          new IntArray(equality && equalityKeyType == CudfType.INT32 ? equalityCapacity : 1);
      this.setLongs =
          new LongArray(equality && equalityKeyType == CudfType.INT64 ? equalityCapacity : 1);
      boolean sortedInt = equality && SORTED_PROBE && equalityKeyType == CudfType.INT32;
      boolean sortedLong = equality && SORTED_PROBE && equalityKeyType == CudfType.INT64;
      this.sortedInts = new IntArray(sortedInt ? equalityCapacity : 1);
      this.sortedLongs = new LongArray(sortedLong ? equalityCapacity : 1);
      this.ints = new IntArray(Math.max(1L, (long) layout.slots(CudfType.INT32) * capacity));
      this.longs = new LongArray(Math.max(1L, (long) layout.slots(CudfType.INT64) * capacity));
      this.doubles =
          new DoubleArray(Math.max(1L, (long) layout.slots(CudfType.FLOAT64) * capacity));
      // The string read writes validity for its columns whether or not they are optional.
      this.valid =
          new ByteArray(
              layout.anyOptional() || layout.stringCount() > 0
                  ? (long) layout.fields().size() * capacity
                  : 1L);
      this.live = new ByteArray(capacity);
    }

    boolean fits(Planned planned) {
      return signature.equals(planned.signature())
          && capacity >= planned.capacity
          && equalityCapacity >= planned.equalityCapacity
          && positionCapacity >= planned.positionCapacity
          && maxFiles >= planned.maxFiles
          && stringStride >= planned.stringStride;
    }

    /** Threads a block for every kernel of the scan. */
    static final int BLOCK = 256;

    /**
     * The launch shape of every kernel, set here rather than left to TornadoVM's scheduler: each
     * kernel takes a KernelContext and guards its index, and runs on whole blocks of {@link #BLOCK}
     * threads over its own bound -- the batch, the position-delete entries, or the equality-delete
     * set. Names a shape does not use are never looked up.
     */
    private GridScheduler grids() {
      GridScheduler scheduler = new GridScheduler();
      String[] perRow = {
        "packStrings",
        "probe",
        "keyProbe",
        "live",
        "filter",
        "buckets",
        "groups",
        "nulls",
        "gatherInts",
        "gatherLongs",
        "gatherDoubles",
        "gatherValid",
        "groupKeys",
        "values",
        "zorder",
        "lineage"
      };
      for (String task : perRow) {
        scheduler.addWorkerGrid("mor." + task, grid(capacity));
      }
      scheduler.addWorkerGrid("mor.deletePositions", grid(positionCapacity));
      scheduler.addWorkerGrid("mor.keyGather", grid(equalityCapacity));
      return scheduler;
    }

    private static WorkerGrid grid(long bound) {
      WorkerGrid grid = new WorkerGrid1D((int) ((Math.max(1, bound) + BLOCK - 1) / BLOCK * BLOCK));
      grid.setLocalWork(BLOCK, 1, 1);
      return grid;
    }

    /** What this shape adds after the live-row kernel. */
    abstract void consumer(TaskGraph graph);

    TornadoExecutionPlan plan() {
      if (plan == null) {
        TaskGraph graph = shared();
        consumer(graph);
        plan = new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(grids());
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
      if (layout.stringCount() > 0) {
        // STRING columns of the same files: cuDF decodes them into offsets and bytes, and a kernel
        // packs each value -- up to 8 bytes -- into its INT64 slot, where filters, group keys and
        // the rest treat it as a number that orders as the string does.
        graph
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, stringErrors)
            .libraryTask(
                "readStrings",
                Cudf::readParquetStringColumns,
                path,
                stringColumns,
                rowsHolder,
                capacity,
                stringOffsets,
                stringChars,
                stringStride,
                valid,
                layout.numericCount())
            .task(
                "packStrings",
                DeviceKernels::packStrings,
                context,
                meta,
                stringOffsets,
                stringChars,
                layout.stringCount(),
                layout.firstStringSlot(),
                capacity,
                (int) stringStride,
                longs,
                stringErrors)
            .transferToHost(DataTransferMode.UNDER_DEMAND, stringErrors);
      }
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
            // As the file holds it: a decimal key is read unscaled (CudfType.DECIMAL64).
            new CudfType[] {layout.readType(equalityKeyId)},
            setRowsHolder,
            equalityCapacity,
            intKey ? setInts : new IntArray(1),
            intKey ? new LongArray(1) : setLongs,
            new DoubleArray(1),
            new ByteArray(1),
            false);
        // The key is slot 0 of its type, so the probe reads it from offset 0.
        if (SORTED_PROBE) {
          // Sorted once a set (sortSize is 0 when the set has not changed), then a binary search
          // from a kernel for every batch: no hash table rebuilt per batch.
          graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, setSize);
          if (intKey) {
            graph
                .libraryTask("sortSet", Cudf::sortKeys, sortSize, setInts, sortedInts)
                .task(
                    "probe",
                    DeviceKernels::probeSortedInts,
                    context,
                    sortedInts,
                    setSize,
                    ints,
                    inSet);
          } else {
            graph
                .libraryTask("sortSet", Cudf::sortKeys, sortSize, setLongs, sortedLongs)
                .task(
                    "probe",
                    DeviceKernels::probeSortedLongs,
                    context,
                    sortedLongs,
                    setSize,
                    longs,
                    inSet);
          }
        } else if (intKey) {
          graph.libraryTask("probe", Cudf::containedIn, probeSizes, ints, setInts, inSet);
        } else {
          graph.libraryTask("probe", Cudf::containedIn, probeSizes, longs, setLongs, inSet);
        }
        keyValidOffset = layout.readIndex(equalityKeyId) * capacity;
      }
      if (keyColumns > 1) {
        // Several key columns: the set, loaded on the host, is put in hash order on the device when
        // it changes; every batch then binary-searches it by hash and compares the columns.
        graph
            .transferToDevice(
                DataTransferMode.FIRST_EXECUTION, rawHashes, rawKeys, rawNulls, keySpec)
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, keySetChanged, setSize)
            .libraryTask(
                "keySort",
                Cudf::sortedOrderLongs,
                keySortSize,
                rawHashes,
                equalityCapacity,
                keyOrder)
            .task(
                "keyGather",
                DeviceKernels::gatherKeySet,
                context,
                keySetChanged,
                setSize,
                keyOrder,
                rawHashes,
                rawKeys,
                rawNulls,
                keyColumns,
                equalityCapacity,
                keyHashes,
                keyValues,
                keyNulls)
            .task(
                "keyProbe",
                DeviceKernels::probeKeySet,
                context,
                keySpec,
                keyColumns,
                keyHashes,
                keyValues,
                keyNulls,
                setSize,
                equalityCapacity,
                ints,
                longs,
                valid,
                capacity,
                inSet);
      }
      graph
          .task(
              "live",
              DeviceKernels::liveRows,
              context,
              meta,
              positionBits,
              inSet,
              valid,
              keyValidOffset,
              live)
          .task(
              "deletePositions",
              DeviceKernels::deletePositions,
              context,
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
                context,
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

    /** What a shape sets up per batch, before the shared staging; nothing by default. */
    void stageOutput(HostDeletes.FilePrep prep) {}

    /** Points the plan's holders and small inputs at one batch. */
    void stage(TornadoExecutionPlan executionPlan, HostDeletes.FilePrep prep, boolean keyNullable) {
      stageOutput(prep);
      path.setLength(0);
      hold(path, prep.path());
      rowsHolder[0] = prep.rows();
      System.arraycopy(prep.columns(), 0, columns, 0, columns.length);
      System.arraycopy(prep.columns(), columns.length, stringColumns, 0, stringColumns.length);
      for (int s = 0; s < stringColumns.length; s++) {
        stringErrors.set(s, 0);
      }
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
      sortSize[0] = 0;
      keySortSize[0] = 0;
      keySetChanged.set(0, 0);
      if (!equalityId.isEmpty() && !equalityId.equals(setOnDevice)) {
        if (prep.keySet() != null) {
          uploadKeySet(executionPlan, prep.keySet());
          meta.set(DeviceKernels.META_EQUALITY_SET_HAS_NULL, 0);
          meta.set(DeviceKernels.META_KEY_NULLABLE, 0);
        } else if (prep.setPaths() != null) {
          hold(setPath, prep.setPaths());
          setRowsHolder[0] = prep.setRows();
          setColumn[0] = prep.setColumn();
          probeSizes[1] = (int) prep.setRows();
          meta.set(DeviceKernels.META_EQUALITY_SET_HAS_NULL, 0);
          sortSize[0] = probeSizes[1];
          setSize.set(0, probeSizes[1]);
        } else {
          upload(executionPlan, prep.equality());
          meta.set(DeviceKernels.META_EQUALITY_SET_HAS_NULL, prep.equality().hasNull() ? 1 : 0);
          sortSize[0] = probeSizes[1];
          setSize.set(0, probeSizes[1]);
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

    /** Puts a host-loaded multi-column set on the device, for this execution to sort. */
    private void uploadKeySet(TornadoExecutionPlan executionPlan, HostDeletes.KeySet set) {
      int m = set.size();
      MemorySegment.copy(
          MemorySegment.ofArray(set.hashes()), 0, rawHashes.getSegment(), 0, (long) m * Long.BYTES);
      for (int c = 0; c < keyColumns; c++) {
        MemorySegment.copy(
            MemorySegment.ofArray(set.keys()),
            (long) c * m * Long.BYTES,
            rawKeys.getSegment(),
            (long) c * equalityCapacity * Long.BYTES,
            (long) m * Long.BYTES);
        MemorySegment.copy(
            MemorySegment.ofArray(set.nulls()),
            (long) c * m,
            rawNulls.getSegment(),
            (long) c * equalityCapacity,
            m);
      }
      executionPlan.transferToDevice(rawHashes, rawKeys, rawNulls);
      keySortSize[0] = m;
      keySetChanged.set(0, 1);
      setSize.set(0, m);
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
            context,
            live,
            longs,
            groupOffset,
            buckets,
            groupKeys);
      } else {
        graph.task("groups", DeviceKernels::groupKeys, context, live, ints, groupOffset, groupKeys);
      }
      if (layout.anyOptional()) {
        graph
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, validColumn)
            .task(
                "nulls",
                DeviceKernels::zeroNulls,
                context,
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

    /** The rows the gathers take, in the order they write them: the live rows, by default. */
    IntArray positions() {
      return compacted;
    }

    /** What a shape adds between the compaction and the gathers; nothing by default. */
    void beforeGather(TaskGraph graph) {}

    int longSlots() {
      return longSlots;
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
      this(planned, 0);
    }

    /** {@code extraLongSlots} more INT64 output slots, after the gathered ones, for a subclass. */
    MaterializeRegion(Planned planned, int extraLongSlots) {
      super(planned);
      this.intSlots = layout.slots(CudfType.INT32);
      this.longSlots = layout.slots(CudfType.INT64);
      this.doubleSlots = layout.slots(CudfType.FLOAT64);
      this.validSlots = layout.anyOptional() ? layout.fields().size() : 0;
      this.compacted = new IntArray(capacity);
      this.outInts = new IntArray(Math.max(1L, (long) intSlots * capacity));
      this.outLongs = new LongArray(Math.max(1L, (long) (longSlots + extraLongSlots) * capacity));
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
      beforeGather(graph);
      if (intSlots > 0) {
        graph.task(
            "gatherInts",
            DeviceKernels::gatherInts,
            context,
            positions(),
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
            context,
            positions(),
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
            context,
            positions(),
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
            context,
            positions(),
            liveCount,
            valid,
            validSlots,
            capacity,
            outValid);
      }
      graph.transferToHost(
          DataTransferMode.UNDER_DEMAND, liveCount, outInts, outLongs, outDoubles, outValid);
      if (stringColumns.length > 0) {
        // Strings are not gathered on the device: the live rows' indices, offsets and bytes come
        // back, and the host reads each live value where the read left it.
        graph.transferToHost(DataTransferMode.UNDER_DEMAND, compacted, stringOffsets, stringChars);
      }
    }

    /** Copies back one batch's live rows and hands them to the consumer. */
    long collect(TornadoExecutionResult executed, Consumer<LiveRows> consumer) {
      executed.transferToHost(liveCount);
      int rows = liveCount.get(0);
      if (rows > 0) {
        copySlots(executed, outInts, 0, intSlots, rows);
        // A string's INT64 slot holds its packed key, which the host never reads: it takes the
        // string from its offsets and chars.
        copySlots(executed, outLongs, 0, layout.firstStringSlot(), rows);
        copySlots(
            executed, outLongs, layout.firstStringSlot() + layout.stringCount(), longSlots, rows);
        copySlots(executed, outDoubles, 0, doubleSlots, rows);
        for (int c = 0; c < validSlots; c++) {
          if (!nullFree[c]) {
            executed.transferToHost(
                new DataRange(outValid).withOffset((long) c * capacity).withSize(rows));
          }
        }
      }
      consumer.accept(
          new HostBatch(
              layout,
              idByName,
              rows,
              capacity,
              outInts,
              outLongs,
              outDoubles,
              outValid,
              nullFree.clone(),
              rows > 0 ? copyStrings(executed, rows) : null));
      return rows;
    }

    /** A batch's string columns, for the rows the read produced, and the live rows' indices. */
    private HostBatch.StringColumns copyStrings(TornadoExecutionResult executed, int rows) {
      if (stringColumns.length == 0) {
        return null;
      }
      int read = meta.get(DeviceKernels.META_ROWS);
      executed.transferToHost(new DataRange(compacted).withSize(rows));
      byte[][] chars = new byte[stringColumns.length][];
      for (int s = 0; s < stringColumns.length; s++) {
        long offsetsBase = (long) s * (capacity + 1);
        executed.transferToHost(
            new DataRange(stringOffsets).withOffset(offsetsBase).withSize(read + 1L));
        int bytes = stringOffsets.get((int) (offsetsBase + read));
        if (bytes > 0) {
          executed.transferToHost(
              new DataRange(stringChars).withOffset(s * stringStride).withSize(bytes));
        }
        chars[s] =
            stringChars
                .getSegment()
                .asSlice(s * stringStride, bytes)
                .toArray(ValueLayout.JAVA_BYTE);
      }
      return new HostBatch.StringColumns(compacted, stringOffsets, chars, layout.numericCount());
    }

    private void copySlots(
        TornadoExecutionResult executed, TornadoNativeArray array, int from, int to, int rows) {
      for (int s = from; s < to; s++) {
        executed.transferToHost(
            new DataRange(array).withOffset((long) s * capacity).withSize(rows));
      }
    }
  }

  /** Sums of expressions per group of several columns: the TPC-H Q1 and Q6 shape. */
  private final class GroupedRegion extends Region {
    private final int keys;
    private final int groupKeyColumns;
    private final int values;
    private final boolean[] stringKey;
    private final boolean[] nullableKey;
    private final IntArray groupSpec;
    private final IntArray valueSpec;
    private final DoubleArray valueConstants;
    private final LongArray keyColumnsData;
    private final DoubleArray valueData;
    private final LongArray outKeys;
    private final DoubleArray outSums;
    private final IntArray outGroups = new IntArray(1);
    private final int[] size;

    GroupedRegion(Planned planned, List<String> groupColumns, List<DeviceValue> sums) {
      super(planned);
      this.keys = groupColumns.size();
      this.groupKeyColumns = Math.max(1, keys);
      this.values = sums.size();
      this.stringKey = new boolean[keys];
      this.nullableKey = new boolean[keys];
      this.groupSpec = new IntArray(Math.max(1, 3 * keys));
      for (int k = 0; k < keys; k++) {
        Types.NestedField field = column(groupColumns.get(k));
        stringKey[k] = layout.isString(field.fieldId());
        keyedIds.add(field.fieldId()); // grouped on the device: a string key has to pack
        nullableKey[k] = field.isOptional();
        groupSpec.set(3 * k, layout.type(field.fieldId()) == CudfType.INT32 ? 0 : 1);
        groupSpec.set(3 * k + 1, layout.slot(field.fieldId()));
        groupSpec.set(3 * k + 2, field.isOptional() ? layout.readIndex(field.fieldId()) : -1);
      }
      this.valueSpec = new IntArray(values * DeviceKernels.VALUE_SPEC_INTS);
      this.valueConstants = new DoubleArray(2L * values * DeviceValue.MAX_FACTORS);
      for (int v = 0; v < values; v++) {
        List<DeviceValue.Factor> factors = sums.get(v).factors();
        int base = v * DeviceKernels.VALUE_SPEC_INTS;
        valueSpec.set(base, factors.size());
        for (int f = 0; f < factors.size(); f++) {
          DeviceValue.Factor factor = factors.get(f);
          Types.NestedField field = column(factor.column());
          CudfType type = layout.type(field.fieldId());
          valueSpec.set(
              base + 1 + 3 * f, type == CudfType.INT32 ? 0 : type == CudfType.INT64 ? 1 : 2);
          valueSpec.set(base + 2 + 3 * f, layout.slot(field.fieldId()));
          valueSpec.set(
              base + 3 + 3 * f, field.isOptional() ? layout.readIndex(field.fieldId()) : -1);
          int c = 2 * (v * DeviceValue.MAX_FACTORS + f);
          valueConstants.set(c, factor.offset());
          valueConstants.set(c + 1, factor.scale());
        }
      }
      this.keyColumnsData = new LongArray((long) groupKeyColumns * capacity);
      this.valueData = new DoubleArray((long) (values + 1) * capacity);
      this.outKeys = new LongArray((long) groupKeyColumns * capacity);
      this.outSums = new DoubleArray((long) (values + 1) * capacity);
      this.size = new int[] {capacity, groupKeyColumns, values + 1};
    }

    @Override
    void consumer(TaskGraph graph) {
      graph
          .transferToDevice(DataTransferMode.FIRST_EXECUTION, groupSpec, valueSpec, valueConstants)
          .task(
              "groupKeys",
              DeviceKernels::groupKeyColumns,
              context,
              groupSpec,
              keys,
              live,
              ints,
              longs,
              valid,
              capacity,
              keyColumnsData)
          .task(
              "values",
              DeviceKernels::valueColumns,
              context,
              valueSpec,
              valueConstants,
              values,
              live,
              ints,
              longs,
              doubles,
              valid,
              capacity,
              valueData)
          .libraryTask(
              "groupBy",
              Cudf::groupAggregateLongs,
              size,
              CudfAggregation.SUM.code(),
              keyColumnsData,
              valueData,
              capacity,
              outKeys,
              outSums,
              outGroups)
          .transferToHost(DataTransferMode.UNDER_DEMAND, outKeys, outSums, outGroups);
    }

    /** Copies back one batch's groups and merges them; returns the batch's live rows. */
    long collect(TornadoExecutionResult executed, GroupedAggregate result) {
      executed.transferToHost(outGroups);
      int groups = outGroups.get(0);
      if (groups == 0) {
        return 0L;
      }
      for (int k = 0; k < groupKeyColumns; k++) {
        executed.transferToHost(
            new DataRange(outKeys).withOffset((long) k * capacity).withSize(groups));
      }
      for (int v = 0; v <= values; v++) {
        executed.transferToHost(
            new DataRange(outSums).withOffset((long) v * capacity).withSize(groups));
      }
      long liveRows = 0;
      double[] sums = new double[values];
      for (int g = 0; g < groups; g++) {
        if (outKeys.get(g) == DeviceKernels.DEAD_KEY) {
          continue;
        }
        List<Object> key = Lists.newArrayListWithCapacity(keys);
        for (int k = 0; k < keys; k++) {
          long value = outKeys.get((long) k * capacity + g);
          if (nullableKey[k] && value == DeviceKernels.NULL_KEY) {
            key.add(null);
          } else {
            key.add(stringKey[k] ? ScanLayout.unpackString(value) : (Object) value);
          }
        }
        for (int v = 0; v < values; v++) {
          sums[v] = outSums.get((long) v * capacity + g);
        }
        double rows = outSums.get((long) values * capacity + g);
        result.add(key, sums, rows);
        liveRows += (long) rows;
      }
      return liveRows;
    }
  }

  /** Compaction: the live rows of a batch, compacted and written to a new Parquet file. */
  private final class RewriteRegion extends MaterializeRegion {
    private final StringBuilder outputPath = new StringBuilder(PATH_CAPACITY);
    private final String names;
    private final int[] fieldIds;
    private final ParquetColumnType[] types;
    private final boolean[] optional;
    private final boolean lineage;
    private final IntArray fileStarts;
    private final IntArray fileCount = new IntArray(1);
    private final LongArray firstRowIds;
    private final LongArray sequences;
    private final int zColumns;
    private final IntArray zSpec;
    private final int[] zSize;
    private final LongArray zKeys;
    private final IntArray zOrder;
    private String location;

    /**
     * {@code lineage}: a v3 table's rewrite, which also writes each row's {@code _row_id} and
     * {@code _last_updated_sequence_number}, as two INT64 columns after the table's.
     */
    RewriteRegion(Planned planned, boolean lineage, List<String> zorderColumns) {
      super(planned, lineage ? 2 : 0);
      this.lineage = lineage;
      this.zColumns = zorderColumns.size();
      this.zSpec = new IntArray(Math.max(1, 3 * zColumns));
      this.zSize = new int[] {0, zColumns + 1};
      this.zKeys = new LongArray(zColumns == 0 ? 1L : (long) (zColumns + 1) * capacity);
      this.zOrder = new IntArray(zColumns == 0 ? 1 : capacity);
      for (int c = 0; c < zColumns; c++) {
        Types.NestedField field = table.schema().findField(zorderColumns.get(c));
        if (field == null) {
          throw new IllegalArgumentException("No column " + zorderColumns.get(c));
        }
        CudfType type = layout.type(field.fieldId());
        zSpec.set(3 * c, type == CudfType.INT32 ? 0 : type == CudfType.INT64 ? 1 : 2);
        zSpec.set(3 * c + 1, layout.slot(field.fieldId()));
        zSpec.set(
            3 * c + 2,
            layout.anyOptional() && field.isOptional() ? layout.readIndex(field.fieldId()) : -1);
      }
      this.fileStarts = new IntArray(maxFiles);
      this.firstRowIds = new LongArray(maxFiles);
      this.sequences = new LongArray(maxFiles);
      List<Types.NestedField> fields = Lists.newArrayList(layout.fields());
      if (lineage) {
        fields.add(MetadataColumns.ROW_ID);
        fields.add(MetadataColumns.LAST_UPDATED_SEQUENCE_NUMBER);
      }
      this.fieldIds = new int[fields.size()];
      this.types = new ParquetColumnType[fields.size()];
      this.optional = new boolean[fields.size()];
      StringBuilder joined = new StringBuilder();
      for (int c = 0; c < fields.size(); c++) {
        Types.NestedField field = fields.get(c);
        fieldIds[c] = field.fieldId();
        // Lineage is written for every row, so its columns carry no nulls.
        optional[c] = field.isOptional() && c < layout.fields().size();
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
    IntArray positions() {
      return zColumns == 0 ? super.positions() : zOrder;
    }

    @Override
    void beforeGather(TaskGraph graph) {
      if (zColumns > 0) {
        graph
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, zSpec)
            .task(
                "zorder",
                DeviceKernels::zorderKeys,
                context,
                zSpec,
                zColumns,
                live,
                ints,
                longs,
                doubles,
                valid,
                capacity,
                zKeys)
            .libraryTask("zsort", Cudf::sortedOrderLongs, zSize, zKeys, capacity, zOrder);
      }
    }

    @Override
    void consumer(TaskGraph graph) {
      super.consumer(graph);
      if (lineage) {
        graph
            .transferToDevice(
                DataTransferMode.EVERY_EXECUTION, fileStarts, fileCount, firstRowIds, sequences)
            .task(
                "lineage",
                DeviceKernels::lineage,
                context,
                positions(),
                liveCount(),
                fileStarts,
                fileCount,
                firstRowIds,
                sequences,
                longSlots(),
                capacity,
                outLongs());
      }
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
          codec(),
          0);
    }

    /**
     * The table's Parquet codec ({@code write.parquet.compression-codec}, ZSTD by default), or the
     * {@code iceberg.tornadovm.write-codec} override; the device writer has ZSTD, Snappy and none.
     */
    private int codec() {
      String codec =
          System.getProperty(
              "iceberg.tornadovm.write-codec",
              table
                  .properties()
                  .getOrDefault(
                      org.apache.iceberg.TableProperties.PARQUET_COMPRESSION,
                      org.apache.iceberg.TableProperties.PARQUET_COMPRESSION_DEFAULT));
      return switch (codec.toLowerCase(java.util.Locale.ROOT)) {
        case "zstd" -> 2;
        case "snappy" -> 1;
        case "uncompressed", "none" -> 0;
        default ->
            throw new UnsupportedOperationException("The device writer has no " + codec + " codec");
      };
    }

    /** A new file a batch, where Iceberg's location provider puts data files. */
    @Override
    void stageOutput(HostDeletes.FilePrep prep) {
      zSize[0] = (int) prep.rows();
      if (lineage) {
        if (prep.storedLineage()) {
          throw new UnsupportedOperationException(
              "A data file already stores row lineage columns; rewriting it again is not supported");
        }
        List<Integer> starts = prep.fileStarts();
        fileCount.set(0, starts.size());
        for (int f = 0; f < starts.size(); f++) {
          Long firstRowId = prep.firstRowIds().get(f);
          Long sequence = prep.sequences().get(f);
          if (firstRowId == null || sequence == null) {
            throw new UnsupportedOperationException(
                "A data file of a v3 table has no first row id or data sequence number");
          }
          fileStarts.set(f, starts.get(f));
          firstRowIds.set(f, firstRowId);
          sequences.set(f, sequence);
        }
      }
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
