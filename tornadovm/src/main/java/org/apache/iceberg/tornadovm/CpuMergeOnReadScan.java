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
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.io.CloseableIterator;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.relocated.com.google.common.collect.Maps;
import org.apache.iceberg.types.Types;
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
  private final Expression filter;
  private ScanMetrics metrics = new ScanMetrics();

  private CpuMergeOnReadScan(Table table, int threads, Expression filter) {
    this.table = table;
    this.threads = threads;
    this.filter = filter;
  }

  public static CpuMergeOnReadScan of(Table table, int threads) {
    return new CpuMergeOnReadScan(table, threads, Expressions.alwaysTrue());
  }

  /** Same contract as {@link DeviceMergeOnReadScan#of(Table, Expression)}. */
  public static CpuMergeOnReadScan of(Table table, int threads, Expression filter) {
    return new CpuMergeOnReadScan(table, threads, filter);
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
        (file, columnIndex) -> new RowSink(columnIndex, file.recordCount(), table.schema()),
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
    try (CloseableIterable<FileScanTask> planned = table.newScan().filter(filter).planFiles()) {
      tasks = Lists.newArrayList(planned);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    Integer equalityKeyId = HostDeletes.equalityKeyId(tasks);
    List<Integer> allKeyIds = HostDeletes.equalityKeyIds(tasks);
    List<Integer> keyIds = allKeyIds.size() > 1 ? allKeyIds : List.of();
    List<String> read = Lists.newArrayList(projection);
    for (int keyId : keyIds) {
      String name = table.schema().findField(keyId).name();
      if (!read.contains(name)) {
        read.add(name);
      }
    }
    for (int fieldId : DevicePredicate.fieldIds(filter, table.schema())) {
      String name = table.schema().findField(fieldId).name();
      if (!read.contains(name)) {
        read.add(name);
      }
    }
    boolean filtered = filter.op() != Expression.Operation.TRUE;
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
    HostDeletes host = new HostDeletes(table, layout, equalityKeyId, false).withKeyIds(keyIds);
    int[] keyColumns = new int[keyIds.size()];
    boolean[] keyInts = new boolean[keyIds.size()];
    for (int c = 0; c < keyColumns.length; c++) {
      Types.NestedField field = table.schema().findField(keyIds.get(c));
      keyColumns[c] = columnIndex.get(field.name());
      keyInts[c] = layout.type(field.fieldId()) == CudfType.INT32;
    }
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
                  HostDeletes.KeySet keySet = prep.keySet();
                  long[] keyValues = new long[keyColumns.length];
                  byte[] keyNulls = new byte[keyColumns.length];
                  S sink = factory.create(task.file(), columnIndex);
                  RowFilter rowFilter =
                      filtered
                          ? new RowFilter(
                              DevicePredicate.compile(task.residual(), table.schema()),
                              table.schema(),
                              columnIndex)
                          : null;
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
                        if (keySet != null) {
                          for (int c = 0; c < keyColumns.length; c++) {
                            ColumnVector column = batch.column(keyColumns[c]);
                            keyNulls[c] = (byte) (column.isNullAt(r) ? 1 : 0);
                            keyValues[c] =
                                keyNulls[c] == 1
                                    ? 0
                                    : keyInts[c] ? column.getInt(r) : column.getLong(r);
                          }
                          if (keySet.contains(keyValues, keyNulls)) {
                            continue;
                          }
                        }
                        if (rowFilter != null && !rowFilter.test(batch, r)) {
                          continue;
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

  /**
   * The residual filter as the device evaluates it -- the same compiled terms -- read straight from
   * Arrow vectors without boxing a value per row.
   */
  private static final class RowFilter {
    private final int[] columns;
    private final int[] kinds;
    private final int[] ops;
    private final long[][] longs;
    private final double[][] doubles;

    RowFilter(
        DevicePredicate predicate, org.apache.iceberg.Schema schema, Map<String, Integer> index) {
      List<DevicePredicate.Term> terms = predicate.terms();
      this.columns = new int[terms.size()];
      this.kinds = new int[terms.size()];
      this.ops = new int[terms.size()];
      this.longs = new long[terms.size()][];
      this.doubles = new double[terms.size()][];
      for (int t = 0; t < terms.size(); t++) {
        DevicePredicate.Term term = terms.get(t);
        ops[t] = term.op();
        if (term.op() == DevicePredicate.OP_FALSE) {
          continue;
        }
        columns[t] = index.get(schema.findField(term.fieldId()).name());
        kinds[t] = term.kind().code();
        longs[t] = term.longs();
        doubles[t] = term.doubles();
      }
    }

    boolean test(ColumnarBatch batch, int row) {
      for (int t = 0; t < ops.length; t++) {
        int op = ops[t];
        if (op == DevicePredicate.OP_FALSE) {
          return false;
        }
        ColumnVector vector = batch.column(columns[t]);
        boolean isNull = vector.isNullAt(row);
        if (op == DevicePredicate.OP_IS_NULL) {
          if (!isNull) {
            return false;
          }
        } else if (op == DevicePredicate.OP_NOT_NULL) {
          if (isNull) {
            return false;
          }
        } else if (isNull) {
          // Iceberg's evaluator orders nulls first, as the device kernel does.
          if (op != DevicePredicate.OP_LT
              && op != DevicePredicate.OP_LT_EQ
              && op != DevicePredicate.OP_NOT_EQ) {
            return false;
          }
        } else if (kinds[t] == CudfType.FLOAT64.code()) {
          if (!compare(vector.getDouble(row), op, doubles[t])) {
            return false;
          }
        } else {
          long value = kinds[t] == CudfType.INT64.code() ? vector.getLong(row) : vector.getInt(row);
          if (!compare(value, op, longs[t])) {
            return false;
          }
        }
      }
      return true;
    }

    private static boolean compare(double value, int op, double[] constants) {
      if (op == DevicePredicate.OP_IN) {
        for (double constant : constants) {
          if (value == constant) {
            return true;
          }
        }
        return false;
      }
      double constant = constants[0];
      return switch (op) {
        case DevicePredicate.OP_LT -> value < constant;
        case DevicePredicate.OP_LT_EQ -> value <= constant;
        case DevicePredicate.OP_GT -> value > constant;
        case DevicePredicate.OP_GT_EQ -> value >= constant;
        case DevicePredicate.OP_EQ -> value == constant;
        default -> value != constant;
      };
    }

    private static boolean compare(long value, int op, long[] constants) {
      if (op == DevicePredicate.OP_IN) {
        for (long constant : constants) {
          if (value == constant) {
            return true;
          }
        }
        return false;
      }
      long constant = constants[0];
      return switch (op) {
        case DevicePredicate.OP_LT -> value < constant;
        case DevicePredicate.OP_LT_EQ -> value <= constant;
        case DevicePredicate.OP_GT -> value > constant;
        case DevicePredicate.OP_GT_EQ -> value >= constant;
        case DevicePredicate.OP_EQ -> value == constant;
        default -> value != constant;
      };
    }
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

    RowSink(Map<String, Integer> columnIndex, long capacity, org.apache.iceberg.Schema schema) {
      this.columnIndex = columnIndex;
      this.columns = new Object[columnIndex.size()];
      this.nulls = new boolean[columnIndex.size()][];
      this.capacityHint = (int) capacity;
      this.types = new CudfType[columnIndex.size()];
      columnIndex.forEach(
          (name, index) -> types[index] = ScanLayout.cudfType(schema.findField(name)));
    }

    private final CudfType[] types;

    private final int capacityHint;

    @Override
    public void accept(ColumnarBatch batch, int row) {
      for (int c = 0; c < columns.length; c++) {
        ColumnVector vector = batch.column(c);
        if (columns[c] == null) {
          columns[c] = allocate(types[c], capacityHint);
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

    /** The column's array, by its Iceberg type: the Arrow vector class depends on the encoding. */
    private static Object allocate(CudfType type, int capacity) {
      return switch (type) {
        case INT64 -> new long[capacity];
        case INT32 -> new int[capacity];
        case FLOAT64 -> new double[capacity];
      };
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
