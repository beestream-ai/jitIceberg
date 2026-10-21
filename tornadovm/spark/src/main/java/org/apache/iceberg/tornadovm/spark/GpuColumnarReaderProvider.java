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

import org.apache.iceberg.FileFormat;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.MetadataColumns;
import org.apache.iceberg.ScanTaskGroup;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.spark.source.ColumnarReaderProvider;
import org.apache.iceberg.tornadovm.DeviceMergeOnReadScan;
import org.apache.iceberg.types.Types;
import org.apache.spark.SparkEnv;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Spark reads of Iceberg tables on the GPU: when {@code spark.iceberg.gpu.enabled} is true, a Spark
 * task's data files are read by the TornadoVM + cuDF merge-on-read scan -- decoded, their position
 * deletes, deletion vectors and equality deletes applied on the device -- and handed to Spark as
 * columnar batches of the live rows. Anything the device cannot read the way Iceberg's reader would
 * is declined and left to that reader: other file formats, split files, metadata columns, and
 * column types the device does not have.
 */
public class GpuColumnarReaderProvider implements ColumnarReaderProvider {
  private static final org.slf4j.Logger LOG =
      org.slf4j.LoggerFactory.getLogger(GpuColumnarReaderProvider.class);

  /** The Spark configuration key that turns GPU reads on. */
  public static final String ENABLED = "spark.iceberg.gpu.enabled";

  /**
   * The fewest rows a Spark task must read for the device to read it: below that, a GPU round trip
   * costs more than Spark's own reader takes for a small file -- a dimension table, say.
   */
  public static final String MIN_ROWS = "spark.iceberg.gpu.minRows";

  /**
   * The Spark configuration key that keeps reads on Spark while aggregates and compaction use the
   * GPU: for join-heavy queries, whose rows Spark's CPU joins would have to get back anyway.
   */
  public static final String READ_ENABLED = "spark.iceberg.gpu.read.enabled";

  private static final long DEFAULT_MIN_ROWS = 1_000_000L;

  @Override
  public PartitionReader<ColumnarBatch> create(
      Table table,
      ScanTaskGroup<FileScanTask> taskGroup,
      Schema expectedSchema,
      boolean caseSensitive) {
    SparkEnv env = SparkEnv.get();
    if (env == null
        || !env.conf().getBoolean(ENABLED, false)
        || !env.conf().getBoolean(READ_ENABLED, true)) {
      return null;
    }
    if (!DeviceMergeOnReadScan.isAvailable()) {
      LOG.warn("GPU reads are on but TornadoVM with the cuDF shim is not available");
      return null;
    }
    if (!readable(expectedSchema)) {
      LOG.info("Reading on the CPU: the device does not read the columns of {}", expectedSchema);
      return null;
    }
    long rows = taskGroup.tasks().stream().mapToLong(task -> task.file().recordCount()).sum();
    if (rows < env.conf().getLong(MIN_ROWS, DEFAULT_MIN_ROWS)) {
      LOG.debug("Reading on the CPU: {} rows", rows);
      return null;
    }
    for (FileScanTask task : taskGroup.tasks()) {
      if (task.file().format() != FileFormat.PARQUET
          || !DeviceMergeOnReadScan.readsWholeFile(task)) {
        LOG.info(
            "Reading on the CPU: {} is a {} split [{}, +{}) of {} bytes",
            task.file().location(),
            task.file().format(),
            task.start(),
            task.length(),
            task.file().fileSizeInBytes());
        return null;
      }
    }
    return new GpuPartitionReader(GpuScans.scan(table), taskGroup, expectedSchema);
  }

  /** Whether every expected column is a type the device reads and returns. */
  static boolean readable(Schema schema) {
    for (Types.NestedField field : schema.columns()) {
      if (MetadataColumns.isMetadataColumn(field.fieldId())) {
        return false;
      }
      switch (field.type().typeId()) {
        case INTEGER, LONG, FLOAT, DOUBLE, DATE, TIMESTAMP, STRING -> {}
        case DECIMAL -> {
          // Unscaled in an INT64, so up to 18 digits.
          if (((Types.DecimalType) field.type()).precision() > 18) {
            return false;
          }
        }
        default -> {
          return false;
        }
      }
    }
    // The device scan reads at least one numeric column, strings riding along.
    return schema.columns().stream()
        .anyMatch(field -> field.type().typeId() != org.apache.iceberg.types.Type.TypeID.STRING);
  }
}
