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

  @Override
  public PartitionReader<ColumnarBatch> create(
      Table table,
      ScanTaskGroup<FileScanTask> taskGroup,
      Schema expectedSchema,
      boolean caseSensitive) {
    SparkEnv env = SparkEnv.get();
    if (env == null || !env.conf().getBoolean(ENABLED, false)) {
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
    return !schema.columns().isEmpty();
  }
}
