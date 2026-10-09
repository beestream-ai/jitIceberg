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

import java.util.Map;
import java.util.Set;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.Table;
import org.apache.iceberg.actions.FileRewriteRunner;
import org.apache.iceberg.actions.RewriteDataFiles.FileGroupInfo;
import org.apache.iceberg.actions.RewriteFileGroup;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableSet;
import org.apache.iceberg.relocated.com.google.common.collect.Sets;
import org.apache.iceberg.spark.actions.FileRewriteRunnerProvider;
import org.apache.iceberg.tornadovm.DeviceMergeOnReadScan;
import org.apache.iceberg.types.Types;
import org.apache.spark.SparkEnv;
import org.apache.spark.sql.SparkSession;

/**
 * Spark's bin-pack {@code rewrite_data_files} on the GPU: each file group's live rows -- deletes
 * applied -- are read, gathered and written to new Parquet files on the device, and the action
 * commits them as it would Spark's. Turned on by {@code spark.iceberg.gpu.enabled} (and off again
 * by {@code spark.iceberg.gpu.rewrite.enabled=false}), in local mode, for an unsorted table --
 * partitioned or not, on local disk or an object store -- whose columns the device writes: int,
 * bigint, float, double, date, timestamp, string and decimals of up to 18 digits. Anything else
 * gets Spark's own runner.
 */
public class GpuFileRewriteRunnerProvider implements FileRewriteRunnerProvider {
  private static final org.slf4j.Logger LOG =
      org.slf4j.LoggerFactory.getLogger(GpuFileRewriteRunnerProvider.class);

  /** File groups rewritten on the device in this JVM, for tests and logs. */
  static final java.util.concurrent.atomic.AtomicLong GROUPS =
      new java.util.concurrent.atomic.AtomicLong();

  /** The Spark configuration key that keeps rewrites on the CPU while reads use the GPU. */
  public static final String REWRITE_ENABLED = "spark.iceberg.gpu.rewrite.enabled";

  /** The name that selects this provider. */
  public static final String NAME = "tornadovm";

  @Override
  public String name() {
    return NAME;
  }

  @Override
  public FileRewriteRunner<FileGroupInfo, FileScanTask, DataFile, RewriteFileGroup> create(
      SparkSession spark, Table table) {
    SparkEnv env = SparkEnv.get();
    if (env == null
        || !env.conf().getBoolean(GpuPartitionReaderFactoryProvider.ENABLED, false)
        || !env.conf().getBoolean(REWRITE_ENABLED, true)
        || !spark.sparkContext().isLocal()
        || !DeviceMergeOnReadScan.isAvailable()) {
      return null;
    }
    String reason = unsupported(table);
    if (reason != null) {
      LOG.info("Rewriting {} with Spark: {}", table.name(), reason);
      return null;
    }
    LOG.info("Rewriting {} on the GPU", table.name());
    return new GpuRunner(table);
  }

  /** Why the device cannot rewrite the table as Spark's bin-pack would, or null. */
  static String unsupported(Table table) {
    if (table.specs().size() > 1) {
      // Files of an older spec would be rewritten into the current one, which needs Spark's
      // repartitioning; a table that never changed its spec has one file group a partition.
      return "its partition spec evolved";
    }
    if (table.sortOrder().isSorted()) {
      return "sorted (Spark's writer would order the rows)";
    }
    for (Types.NestedField field : table.schema().columns()) {
      switch (field.type().typeId()) {
        case INTEGER, LONG, FLOAT, DOUBLE, DATE, TIMESTAMP, STRING -> {}
        case DECIMAL -> {
          if (((Types.DecimalType) field.type()).precision() > 18) {
            return field.name() + " has over 18 digits";
          }
        }
        default -> {
          return field.name() + " is a " + field.type() + " column";
        }
      }
    }
    return null;
  }

  private static final class GpuRunner
      implements FileRewriteRunner<FileGroupInfo, FileScanTask, DataFile, RewriteFileGroup> {
    private final Table table;

    GpuRunner(Table table) {
      this.table = table;
    }

    @Override
    public String description() {
      return "GPU-BIN-PACK";
    }

    @Override
    public Set<String> validOptions() {
      return ImmutableSet.of();
    }

    @Override
    public void init(Map<String, String> options) {}

    @Override
    public Set<DataFile> rewrite(RewriteFileGroup group) {
      // The aggregates' kept scans hold device memory a rewrite needs; they rebuild when next used.
      GpuAggregates.releaseAll();
      Set<DataFile> written =
          Sets.newHashSet(
              GpuScans.rewriteScan(table)
                  .rewriteGroup(group.fileScanTasks(), group.maxOutputFileSize()));
      GROUPS.incrementAndGet();
      return written;
    }
  }
}
