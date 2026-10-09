<!--
  - Licensed to the Apache Software Foundation (ASF) under one
  - or more contributor license agreements.  See the NOTICE file
  - distributed with this work for additional information
  - regarding copyright ownership.  The ASF licenses this file
  - to you under the Apache License, Version 2.0 (the
  - "License"); you may not use this file except in compliance
  - with the License.  You may obtain a copy of the License at
  -
  -   http://www.apache.org/licenses/LICENSE-2.0
  -
  - Unless required by applicable law or agreed to in writing,
  - software distributed under the License is distributed on an
  - "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  - KIND, either express or implied.  See the License for the
  - specific language governing permissions and limitations
  - under the License.
  -->

# Iceberg on the GPU for Spark 3.5

This module plugs the GPU merge-on-read scan of `iceberg-tornadovm` (TornadoVM + cuDF, NVIDIA GPUs
only) into unmodified Spark SQL. It does three things, and in each case anything it cannot do
exactly is left to Spark:

| What | How it hooks in | When it is used |
|---|---|---|
| **Reads**: a Spark task's data files are read on the GPU with their position, deletion-vector and equality deletes applied and Spark's pushed filters evaluated, and come back as `ColumnarBatch`es | `ColumnarReaderProvider`, a ServiceLoader SPI in Iceberg's Spark module | Parquet files read whole; columns are int, bigint, float, double, date, timestamp, string or decimal up to 18 digits, and at least one is not a string; a task of at least `spark.iceberg.gpu.minRows` rows |
| **Grouped aggregates**: an `Aggregate` over filters and projections of one Iceberg table runs as a single GPU scan, and only the groups come back | planner strategy (`GpuAggregateExtensions`) | SUM / AVG / COUNT of products of double columns and `1 - c` / `1 + c`; integer and decimal sums the files' bounds prove exact; up to 4 group columns or Iceberg's `bucket(n, bigint)`; value columns null-free by metrics |
| **Compaction**: `rewrite_data_files` (bin-pack) rewrites each file group on the GPU, and Iceberg's action plans and commits as usual | `FileRewriteRunnerProvider`, a ServiceLoader SPI in Iceberg's Spark module | unsorted tables, partitioned or not (one partition spec), whose columns are int, bigint, float, double, date, timestamp, string or decimal up to 18 digits |

All three work on tables on local disk or on an object store (S3, GCS, ADLS: anything Iceberg's
`FileIO` reads), with merge-on-read deletes of every kind, and on evolved schemas: columns added
after a file was written read as nulls, and int→long and float→double promotions are widened on
the device.

Answers match Spark's. The tests compare every path with the GPU on and off. On TPC-DS SF10 after
merge-on-read maintenance, all 102 runnable queries return identical results on Spark, Spark + GPU
and Spark RAPIDS.

## Where it helps, and where it does not

The GPU wins when the work stays on the device:

- **Aggregations**: TPC-H Q1 and Q6 on SF10 run 3–5× faster than Spark and 2–8× faster than Spark
  RAPIDS.
- **Selective filters**: only matching rows cross PCIe.
- **Expensive deletes**: equality deletes, many delete files.
- **Compaction**: see the project's results.

It does not win on join-heavy queries such as TPC-DS. There Spark's joins run on the CPU, and a
reader that copies rows back is about as fast as 32 CPU cores reading them, or slower (0.87× geomean
on TPC-DS SF10). Turn reads off and keep aggregates and compaction with:

```
spark.iceberg.gpu.read.enabled=false
```

## Requirements

- An NVIDIA GPU (compute capability 7.0 or newer), CUDA 12 driver, Linux x86_64.
- JDK 21. The device scan uses Java's foreign-memory API, which needs `--enable-preview` on 21.
- Spark 3.5 (Scala 2.12), **local mode**: the GPU work runs in the driver JVM.
- libcudf 26.08 and a TornadoVM SDK built with the cuDF shim, including TornadoVM #1215
  (decimals), #1219 (cuDF's memory pool gives memory back), #1220 (evolved schemas) and #1221
  (strings in compaction). The steps are in [../README.md](../README.md#set-up).

Check all of it, and whether your tables qualify, with:

```bash
tornadovm/preflight.sh /data/wh/db/sales        # table directories are optional
```

## Build and run

```
export TORNADOVM_HOME=...        # the SDK above
./gradlew -DsparkVersions=3.5 :iceberg-tornadovm-spark:sparkGpuBundle
build=tornadovm/spark/build/spark-gpu
SPARK_HOME=... $build/bin/spark-submit-gpu --driver-memory 24g my_job.py
```

`sparkGpuBundle` writes `build/spark-gpu/`:

- `jars/`: this module, Iceberg's modules (with the SQL extensions, for `CALL`), and the Parquet,
  Avro, Arrow and ORC versions they need. Everything a Spark distribution already ships is left out.
- `tornado-modules/` and `tornado-argfile`: TornadoVM's module path, without the jars that would
  clash with Spark's. TornadoVM's own argfile puts its whole jar directory on the module path, and
  Java then resolves every jar there as an automatic module, including JMH, log4j and junit.
- `bin/spark-submit-gpu`: `spark-submit` in local mode, with the class path, the driver JVM options
  and the GPU settings filled in. Its arguments are `spark-submit`'s.

Compaction then runs as it always does, on the GPU when the table qualifies:

```sql
CALL local.system.rewrite_data_files(table => 'db.sales')
```

## Configuration

| Spark setting | Default | |
|---|---|---|
| `spark.iceberg.gpu.enabled` | `false` | turns all three on (`spark-submit-gpu` sets it) |
| `spark.iceberg.gpu.read.enabled` | `true` | reads on the GPU (`false` keeps aggregates and compaction on it) |
| `spark.iceberg.gpu.minRows` | `1000000` | fewest rows a Spark task must read for the GPU to read it; smaller tasks are faster on Spark's reader |
| `spark.iceberg.gpu.aggregate.enabled` | `true` | the aggregate strategy |
| `spark.iceberg.gpu.rewrite.enabled` | `true` | compaction on the GPU |

| JVM property (driver) | Default | |
|---|---|---|
| `tornado.device.memory` | `20GB` (`GPU_DEVICE_MEMORY`) | TornadoVM's share of device memory. Leave a few GB for cuDF's own allocations |
| `iceberg.tornadovm.streams` | `4` | concurrent device streams |
| `iceberg.tornadovm.rewriteStreams` | the streams | concurrent streams of a compaction's file group. Each holds a batch's read, gathered and encoded columns at once |
| `iceberg.tornadovm.rewriteBatchBytes` | 768 MB | most decoded bytes in one compaction batch; wide tables (e.g. with strings) get fewer rows per batch. A batch that runs out of device memory is retried at half the rows on one stream |
| `iceberg.tornadovm.cacheDir` | `java.io.tmpdir`/`iceberg-tornadovm-cache` | local copies of object-store files |
| `iceberg.tornadovm.cacheBytes` | 64 GB | the cache's size; least recently used files go first |
| `iceberg.tornadovm.fetchThreads` | `16` | parallel fetches from the object store |
| `iceberg.tornadovm.sharedPools` | `6` | table and projection shapes that keep device buffers. Past this, the least recently used idle ones are freed |
| `iceberg.tornadovm.spark.debug` | `false` | per-task timings on stderr |
| `iceberg.tornadovm.debug` | `false` | per-batch and per-compaction-group sizes on stderr |

## Limits

- Local mode only.
- No rewrites of sorted tables, of tables whose partition spec evolved, or of decimals over 18
  digits.
- Object-store files are fetched whole into a local cache before cuDF reads them (and a
  compaction's files are uploaded after it writes them); the cache needs local disk.
- Columns added with a default value (v3), required added columns and added string columns are
  declined.
- Nested types are not read.
- Reads take whole Parquet files only, as Iceberg plans unsplit files. A file split inside a task is
  read by Spark.
- Strings longer than 8 bytes cannot be group keys or filter terms on the device. Such an aggregate
  runs on Spark, after a failed first attempt on the device.
