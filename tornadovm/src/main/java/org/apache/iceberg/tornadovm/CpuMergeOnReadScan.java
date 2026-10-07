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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import org.apache.iceberg.BaseCombinedScanTask;
import org.apache.iceberg.CombinedScanTask;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Table;
import org.apache.iceberg.arrow.vectorized.ArrowReader;
import org.apache.iceberg.arrow.vectorized.ColumnVector;
import org.apache.iceberg.arrow.vectorized.ColumnarBatch;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.io.CloseableIterator;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.relocated.com.google.common.collect.Maps;
import org.apache.iceberg.util.BucketUtil;
import uk.ac.manchester.tornado.cudf.enums.CudfType;

/**
 * The CPU baseline: the same merge-on-read algorithm as {@link DeviceMergeOnReadScan}, on all
 * cores, with Iceberg's fastest Java reader.
 *
 * <p>That reader is the vectorized Arrow reader, and it refuses any file that has deletes ({@code
 * ArrowReader}: "Cannot read files that require applying delete files", apache/iceberg#2487). So
 * each file is handed to it without its deletes, and the deletes are applied here, from the same
 * host preparation the device scan uses: the position bitmap from {@code BaseDeleteLoader} and the
 * equality keys, probed through a primitive hash set. One data file a task, one task a thread.
 */
public final class CpuMergeOnReadScan implements MergeOnReadScan {

  private static final int BATCH_ROWS = 65536;

  private final Table table;
  private final int threads;
  private ScanMetrics metrics = new ScanMetrics();

  private CpuMergeOnReadScan(Table table, int threads) {
    this.table = table;
    this.threads = threads;
  }

  public static CpuMergeOnReadScan of(Table table, int threads) {
    return new CpuMergeOnReadScan(table, threads);
  }

  @Override
  public ScanMetrics metrics() {
    return metrics;
  }

  @Override
  public AggregateResult aggregate(String groupColumn, String... sumColumns) {
    return aggregate(groupColumn, 0, sumColumns);
  }

  @Override
  public AggregateResult aggregateByBucket(String column, int buckets, String... sumColumns) {
    return aggregate(column, buckets, sumColumns);
  }

  private AggregateResult aggregate(String groupColumn, int buckets, String... sumColumns) {
    AggregateResult result = new AggregateResult(sumColumns.length);
    List<String> projection = Lists.newArrayList();
    if (groupColumn != null) {
      projection.add(groupColumn);
    }
    projection.addAll(ImmutableList.copyOf(sumColumns));
    run(
        projection,
        (file, columnIndex) -> new GroupSink(groupColumn, buckets, sumColumns, columnIndex),
        sinks -> {
          synchronized (result) {
            for (GroupSink sink : sinks) {
              sink.mergeInto(result);
            }
          }
        });
    return result;
  }

  /** The consumer is called once a data file, after every file has been read. */
  @Override
  public long materialize(List<String> columns, Consumer<LiveRows> consumer) {
    return run(
        columns,
        (file, columnIndex) -> new RowSink(columnIndex, file.recordCount()),
        sinks -> {
          for (RowSink sink : sinks) {
            consumer.accept(sink);
          }
        });
  }

  /** What one thread accumulates over the live rows of one data file. */
  private interface Sink {
    void accept(ColumnarBatch batch, int row);

    int rows();
  }

  private interface SinkFactory<S extends Sink> {
    S create(DataFile file, Map<String, Integer> columnIndex);
  }

  private <S extends Sink> long run(
      List<String> projection, SinkFactory<S> factory, Consumer<List<S>> finish) {
    metrics = new ScanMetrics();
    long start = System.nanoTime();
    List<FileScanTask> tasks;
    try (CloseableIterable<FileScanTask> planned = table.newScan().planFiles()) {
      tasks = Lists.newArrayList(planned);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    Integer equalityKeyId = HostDeletes.equalityKeyId(tasks);
    List<String> read = Lists.newArrayList(projection);
    String keyName = null;
    if (equalityKeyId != null) {
      keyName = table.schema().findField(equalityKeyId).name();
      if (!read.contains(keyName)) {
        read.add(0, keyName);
      }
    }
    // The reader returns columns in schema order, so index them by name from its schema.
    org.apache.iceberg.Schema readSchema = table.schema().select(read);
    Map<String, Integer> columnIndex = Maps.newHashMap();
    for (int i = 0; i < readSchema.columns().size(); i++) {
      columnIndex.put(readSchema.columns().get(i).name(), i);
    }
    ScanLayout layout = new ScanLayout(table.schema(), equalityKeyId, null, read);
    HostDeletes host = new HostDeletes(table, layout, equalityKeyId, false);
    metrics.plan((System.nanoTime() - start) / 1e6);

    boolean keyIsInt = keyName != null && layout.type(equalityKeyId) == CudfType.INT32;
    Integer keyColumn = keyName == null ? null : columnIndex.get(keyName);
    Map<String, PrimitiveHash.LongSet> sets = new ConcurrentHashMap<>();

    ExecutorService pool = Executors.newFixedThreadPool(threads);
    long live = 0;
    try {
      List<Future<S>> futures = Lists.newArrayList();
      for (FileScanTask task : tasks) {
        futures.add(
            pool.submit(
                () -> {
                  HostDeletes.FilePrep prep = host.prepare(task);
                  byte[] bits = prep.positionBits();
                  HostDeletes.EqualitySet equality = prep.equality();
                  PrimitiveHash.LongSet set =
                      equality == null
                          ? null
                          : sets.computeIfAbsent(
                              equality.id(), ignored -> new PrimitiveHash.LongSet(equality.keys()));
                  boolean setHasNull = equality != null && equality.hasNull();
                  S sink = factory.create(task.file(), columnIndex);
                  try (ArrowReader reader =
                          new ArrowReader(table.newScan().select(read), BATCH_ROWS, true);
                      CloseableIterator<ColumnarBatch> batches =
                          reader.open(
                              CloseableIterable.withNoopClose(
                                  ImmutableList.<CombinedScanTask>of(
                                      new BaseCombinedScanTask(new WithoutDeletes(task)))))) {
                    long position = 0;
                    while (batches.hasNext()) {
                      ColumnarBatch batch = batches.next();
                      ColumnVector key = keyColumn == null ? null : batch.column(keyColumn);
                      int rows = batch.numRows();
                      for (int r = 0; r < rows; r++) {
                        long pos = position + r;
                        if (bits != null && ((bits[(int) (pos >> 3)] >> (pos & 7)) & 1) == 1) {
                          continue;
                        }
                        if (set != null) {
                          if (key.isNullAt(r)) {
                            if (setHasNull) {
                              continue;
                            }
                          } else if (set.contains(keyIsInt ? key.getInt(r) : key.getLong(r))) {
                            continue;
                          }
                        }
                        sink.accept(batch, r);
                      }
                      position += rows;
                    }
                  }
                  return sink;
                }));
      }
      List<S> sinks = Lists.newArrayList();
      for (Future<S> future : futures) {
        S sink = future.get();
        sinks.add(sink);
      }
      finish.accept(sinks);
      for (S sink : sinks) {
        live += sink.rows();
      }
      for (FileScanTask task : tasks) {
        metrics.file(task.file().recordCount());
      }
      metrics.live(live);
    } catch (Exception e) {
      throw new IllegalStateException("CPU scan failed", e);
    } finally {
      pool.shutdownNow();
    }
    metrics.total((System.nanoTime() - start) / 1e6);
    return live;
  }

  /** Grouped COUNT and SUMs over one file's live rows, with SQL's null-skipping SUM. */
  private static final class GroupSink implements Sink {
    private final Integer groupColumn;
    private final int buckets;
    private final int[] sumColumns;
    private final PrimitiveHash.IntIndex groups = new PrimitiveHash.IntIndex();
    private long[] counts = new long[64];
    private double[] sums;
    private long rows = 0;

    GroupSink(String group, int buckets, String[] sumNames, Map<String, Integer> columnIndex) {
      this.groupColumn = group == null ? null : columnIndex.get(group);
      this.buckets = buckets;
      this.sumColumns = new int[sumNames.length];
      for (int c = 0; c < sumNames.length; c++) {
        sumColumns[c] = columnIndex.get(sumNames[c]);
      }
      this.sums = new double[64 * sumNames.length];
    }

    @Override
    public void accept(ColumnarBatch batch, int row) {
      int key;
      if (groupColumn == null) {
        key = 0;
      } else if (buckets > 0) {
        long value = batch.column(groupColumn).getLong(row);
        key = (BucketUtil.hash(value) & Integer.MAX_VALUE) % buckets;
      } else {
        key = batch.column(groupColumn).getInt(row);
      }
      int g = groups.indexOf(key);
      if (g == counts.length) {
        counts = java.util.Arrays.copyOf(counts, counts.length * 2);
        sums = java.util.Arrays.copyOf(sums, sums.length * 2);
      }
      counts[g]++;
      for (int c = 0; c < sumColumns.length; c++) {
        ColumnVector vector = batch.column(sumColumns[c]);
        if (!vector.isNullAt(row)) {
          sums[g * sumColumns.length + c] += vector.getDouble(row);
        }
      }
      rows++;
    }

    void mergeInto(AggregateResult result) {
      int[] keys = groups.keysByIndex();
      double[] values = new double[sumColumns.length];
      for (int g = 0; g < keys.length; g++) {
        System.arraycopy(sums, g * sumColumns.length, values, 0, sumColumns.length);
        result.add(keys[g], counts[g], values);
      }
    }

    @Override
    public int rows() {
      return (int) rows;
    }
  }

  /** One file's live rows, copied out of Arrow into arrays, as the device scan copies them back. */
  private static final class RowSink implements Sink, LiveRows {
    private final Map<String, Integer> columnIndex;
    private final Object[] columns;
    private final boolean[][] nulls;
    private int rows = 0;

    RowSink(Map<String, Integer> columnIndex, long capacity) {
      this.columnIndex = columnIndex;
      this.columns = new Object[columnIndex.size()];
      this.nulls = new boolean[columnIndex.size()][];
      this.capacityHint = (int) capacity;
    }

    private final int capacityHint;

    @Override
    public void accept(ColumnarBatch batch, int row) {
      for (int c = 0; c < columns.length; c++) {
        ColumnVector vector = batch.column(c);
        if (columns[c] == null) {
          columns[c] = allocate(vector, capacityHint);
          nulls[c] = new boolean[capacityHint];
        }
        boolean isNull = vector.isNullAt(row);
        nulls[c][rows] = isNull;
        if (isNull) {
          continue;
        }
        Object column = columns[c];
        if (column instanceof long[] longs) {
          longs[rows] = vector.getLong(row);
        } else if (column instanceof int[] ints) {
          ints[rows] = vector.getInt(row);
        } else {
          ((double[]) column)[rows] = vector.getDouble(row);
        }
      }
      rows++;
    }

    private static Object allocate(ColumnVector vector, int capacity) {
      org.apache.arrow.vector.FieldVector arrow = vector.getFieldVector();
      if (arrow instanceof org.apache.arrow.vector.BigIntVector
          || arrow instanceof org.apache.arrow.vector.TimeStampVector) {
        return new long[capacity];
      } else if (arrow instanceof org.apache.arrow.vector.IntVector
          || arrow instanceof org.apache.arrow.vector.DateDayVector) {
        return new int[capacity];
      }
      return new double[capacity];
    }

    @Override
    public int rows() {
      return rows;
    }

    @Override
    public int column(String name) {
      return columnIndex.get(name);
    }

    @Override
    public boolean isNull(int column, int row) {
      return nulls[column][row];
    }

    @Override
    public int getInt(int column, int row) {
      return ((int[]) columns[column])[row];
    }

    @Override
    public long getLong(int column, int row) {
      return ((long[]) columns[column])[row];
    }

    @Override
    public double getDouble(int column, int row) {
      return ((double[]) columns[column])[row];
    }
  }

  /** A data-file task with its delete files removed, so the vectorized reader will read it. */
  private static final class WithoutDeletes implements FileScanTask {
    private final FileScanTask task;

    WithoutDeletes(FileScanTask task) {
      this.task = task;
    }

    @Override
    public DataFile file() {
      return task.file();
    }

    @Override
    public List<DeleteFile> deletes() {
      return ImmutableList.of();
    }

    @Override
    public PartitionSpec spec() {
      return task.spec();
    }

    @Override
    public long start() {
      return task.start();
    }

    @Override
    public long length() {
      return task.length();
    }

    @Override
    public Expression residual() {
      return task.residual();
    }

    @Override
    public Iterable<FileScanTask> split(long splitSize) {
      return ImmutableList.of(this);
    }
  }
}
