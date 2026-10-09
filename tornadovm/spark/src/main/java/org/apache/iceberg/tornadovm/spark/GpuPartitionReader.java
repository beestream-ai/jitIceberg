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

import java.nio.ByteBuffer;
import java.util.Iterator;
import java.util.List;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.ScanTaskGroup;
import org.apache.iceberg.Schema;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.spark.SparkSchemaUtil;
import org.apache.iceberg.tornadovm.DeviceMergeOnReadScan;
import org.apache.iceberg.tornadovm.LiveRows;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.Types;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A Spark task's data files, read on the GPU: the device scan decodes them and applies their
 * deletes, and each batch of live rows it returns is copied into Spark column vectors, in the order
 * Spark expects. The task's batches are read on the first call to {@link #next()}.
 */
final class GpuPartitionReader implements PartitionReader<ColumnarBatch> {
  /** Spark tasks read on the device in this JVM, for tests and logs. */
  static final java.util.concurrent.atomic.AtomicLong TASKS =
      new java.util.concurrent.atomic.AtomicLong();

  private final DeviceMergeOnReadScan scan;
  private final List<FileScanTask> tasks;
  private final Schema schema;
  private final StructType sparkSchema;
  private Iterator<ColumnarBatch> batches;
  private ColumnarBatch current;

  GpuPartitionReader(
      DeviceMergeOnReadScan scan, ScanTaskGroup<FileScanTask> taskGroup, Schema schema) {
    this.scan = scan;
    this.tasks = Lists.newArrayList(taskGroup.tasks());
    this.schema = schema;
    this.sparkSchema = SparkSchemaUtil.convert(schema);
  }

  @Override
  public boolean next() {
    if (batches == null) {
      List<ColumnarBatch> read = Lists.newArrayList();
      List<String> columns = Lists.newArrayList();
      for (Types.NestedField field : schema.columns()) {
        columns.add(field.name());
      }
      scan.materialize(tasks, columns, batch -> read.add(toSpark(batch)));
      TASKS.incrementAndGet();
      batches = read.iterator();
    }
    if (current != null) {
      current.close();
      current = null;
    }
    if (batches.hasNext()) {
      current = batches.next();
      return true;
    }
    return false;
  }

  @Override
  public ColumnarBatch get() {
    return current;
  }

  @Override
  public void close() {
    if (current != null) {
      current.close();
    }
    while (batches != null && batches.hasNext()) {
      batches.next().close();
    }
  }

  /** One device batch as Spark column vectors: the device's buffers are reused for the next. */
  private ColumnarBatch toSpark(LiveRows batch) {
    int rows = batch.rows();
    List<Types.NestedField> fields = schema.columns();
    ColumnVector[] vectors = new ColumnVector[fields.size()];
    for (int c = 0; c < fields.size(); c++) {
      Types.NestedField field = fields.get(c);
      OnHeapColumnVector vector =
          new OnHeapColumnVector(Math.max(1, rows), sparkSchema.fields()[c].dataType());
      int column = batch.column(field.name());
      Type.TypeID type = field.type().typeId();
      for (int r = 0; r < rows; r++) {
        if (batch.isNull(column, r)) {
          vector.putNull(r);
          continue;
        }
        switch (type) {
          case INTEGER, DATE -> vector.putInt(r, batch.getInt(column, r));
          case LONG, TIMESTAMP -> vector.putLong(r, batch.getLong(column, r));
          case FLOAT -> vector.putFloat(r, (float) batch.getDouble(column, r));
          case DOUBLE -> vector.putDouble(r, batch.getDouble(column, r));
          case STRING -> {
            ByteBuffer utf8 = batch.getUtf8(column, r);
            vector.putByteArray(
                r, utf8.array(), utf8.arrayOffset() + utf8.position(), utf8.remaining());
          }
          default -> throw new UnsupportedOperationException("Column " + field);
        }
      }
      vectors[c] = vector;
    }
    return new ColumnarBatch(vectors, rows);
  }
}
