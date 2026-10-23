<!--
  - Licensed to the Apache Software Foundation (ASF) under one or more
  - contributor license agreements.  See the NOTICE file distributed with
  - this work for additional information regarding copyright ownership.
  - The ASF licenses this file to You under the Apache License, Version 2.0
  - (the "License"); you may not use this file except in compliance with
  - the License.  You may obtain a copy of the License at
  -
  -   http://www.apache.org/licenses/LICENSE-2.0
  -
  - Unless required by applicable law or agreed to in writing, software
  - distributed under the License is distributed on an "AS IS" BASIS,
  - WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  - See the License for the specific language governing permissions and
  - limitations under the License.
  -->


# iceberg-tornadovm: Iceberg merge-on-read and maintenance on the GPU

Iceberg's per-row work, run on an NVIDIA GPU from plain Java: merge-on-read scans that apply
position deletes, deletion vectors and equality deletes, compaction, and the conversion of equality
deletes into deletion vectors. Iceberg still does the planning and the commits. The GPU only
replaces the work in between.

It is a library on Iceberg's public APIs. It needs no engine and no change to Iceberg. The GPU code
is ordinary Java methods that [TornadoVM](https://github.com/beehive-lab/TornadoVM) compiles to
CUDA, plus [cuDF](https://github.com/rapidsai/cudf) for Parquet. Every operation has a CPU twin
(`CpuMergeOnReadScan`), the same algorithm on all cores, for comparison and as the reference.

## What it does

```java
try (DeviceMergeOnReadScan scan = DeviceMergeOnReadScan.of(table)) {
  // Scans: every delete applied on the device; only the answer comes back.
  AggregateResult byK = scan.aggregate("k", "v1", "v2");            // GROUP BY k: COUNT(*), SUM(v1), SUM(v2)
  AggregateResult buckets = scan.aggregateByBucket("id", 16, "v1");   // GROUP BY bucket(16, id)
  long live = scan.materialize(List.of("id", "k", "v1"), rows -> { ... }); // live rows, to the host

  // Maintenance: each is one Iceberg commit, validated against the snapshot it planned.
  scan.convertEqualityDeletes();       // equality deletes -> one deletion vector per data file (v3)
  scan.rewriteDataFiles();             // compaction: deletes applied, files rewritten
  scan.rewriteDataFilesZOrder("k", "v1");
}
// Filters: Iceberg prunes files, the device applies the residual.
DeviceMergeOnReadScan.of(table, Expressions.and(lessThan("id", 25_000_000L), greaterThan("v2", 0.5)));
```

**`convertEqualityDeletes()`** reads only the equality key columns of the data files that
equality deletes apply to. It applies every delete as a scan would, marks the rows merge-on-read
drops, and writes one deletion vector per data file, with that file's earlier deletion vector or
file-scoped position deletes folded in. One `RowDelta` adds the vectors and removes the
equality-delete files and the deletes they replace, validated against the planned snapshot.
These are the rules Flink's `ConvertEqualityDeletes` maintenance task follows (apache/iceberg
#16858, #16874, #17189), here without a Flink job. Iceberg v4 forbids writing new equality
deletes (#17783), so this is the conversion v4 tables need.

## How a batch runs

1. **Plan.** `table.newScan().planFiles()` decides which delete files apply to which data file,
   by sequence number. Consecutive files that share their equality deletes form a batch of up to
   `iceberg.tornadovm.batch-rows` rows (8M by default).
2. **Prepare on the host,** ahead of the GPU: Parquet footers, and deletion vectors through
   `BaseDeleteLoader`.
3. **Run on the GPU,** in one TornadoVM execution plan a batch:
   - cuDF reads the batch's data files, its position-delete files and its equality-delete keys;
   - TornadoVM kernels fold position deletes, deletion vectors, equality hits and Iceberg's null
     rules into one live mask;
   - then one consumer runs: a cuDF group-by, a gather of live rows, a Parquet write, or the
     positions of dropped rows.
   Nothing returns to the host between these steps. `iceberg.tornadovm.streams` batches (3 by
   default) run at once, each on its own CUDA stream.
4. **Commit** through Iceberg: `RewriteFiles` for compaction, `RowDelta` for deletion vectors.

**Object stores** (S3, GCS, ADLS) go through the table's `FileIO`. cuDF opens files by path, so a
scan fetches only the column chunks it reads, each with one ranged read, into a sparse local copy
that cuDF opens as the original file. Delete files and rewrites fetch whole files. The cache is
`iceberg.tornadovm.cacheDir` (64 GB by default, least recently used first).

## What the device takes

| | Supported | Refused with an error |
|---|---|---|
| Files | Parquet, whole files; local or through `FileIO` | other formats, split files |
| Types | int, long, date, timestamp, float, double, decimal (precision <= 18), string | nested types, decimal > 18, `time` in writes |
| Deletes | v2 position deletes, v3 deletion vectors, equality deletes on int/long keys (one or several columns) | equality deletes on string keys |
| Tables | v2 and v3, partitioned or not, evolved schemas (added columns read as null) | |
| Conversion | v3 tables | v2 tables, partition-scoped position delete files |

Anything refused fails with `UnsupportedOperationException`. Nothing is answered wrongly.

## Iceberg APIs it uses

All public, the same ones Iceberg's own engines use. Planning: `Table.newScan()`, `FileScanTask`,
`DeleteFile`. Commits: `RowDelta`, `RewriteFiles`. Deletes: `BaseDeleteLoader`,
`PositionDeleteIndex`, `BaseDVFileWriter`, `ContentFileUtil`. Writing: `OutputFileFactory`,
`ParquetUtil.fileMetrics`, `MetricsConfig`. It is not a File Format API model, because that API
reads one file at a time and applies deletes above it (#12774, #13433). Nothing here needs to live
in Iceberg's tree (compare #15674).

## Correctness

The tests check every device and CPU result against `IcebergGenerics`, Iceberg's own merge-on-read
reader:
- **`TestDeviceMergeOnRead`:** v2 position deletes, v3 deletion vectors, equality deletes before
  and after an append (the sequence-number rules), several key columns, nulls in keys and values,
  filters, strings, decimals, compaction with v3 row lineage, z-order, and conversion to deletion
  vectors (GPU and CPU twin give the same vectors).
- **`TestDeviceEvolutionAndStorage`:** evolved schemas, object-store tables, partitioned compaction
  and partitioned conversion.

The benchmarks print each answer's row count and checksum, and every engine must agree.

## Set up

1. **libcudf** from NVIDIA's wheels (about 1.3 GB, once; no conda): `source tornadovm/cudf-env.sh`.
   `CUDF_PACKAGE=libcudf-cu13` selects the CUDA 13 build.
2. **TornadoVM with the cuDF shim**, built with the CUDA backend (released SDKs don't ship the
   shim): branch `iceberg-bench` of `mikepapadim/TornadoVM`, which is master plus three changes
   in review.
   ```bash
   make BACKEND=cuda && source setvars.sh      # with CUDF_HOME from step 1; JDK 21
   ```
3. **Check** the JDK, SDK, shim, libcudf and GPU: `tornadovm/preflight.sh` (OK / WARN / FAIL a
   line).

The module is part of the build only when `TORNADOVM_HOME` is set. Run `./gradlew
:iceberg-tornadovm:test`, then the benchmarks:

```bash
tornadovm/run.sh gen /data/wh 64 1562500                    # 100M rows, a table per delete layout
tornadovm/run.sh gpu /data/wh q2 6 none pos05 dv05 eq1m     # and `cpu` for the CPU twin
MAIN=org.apache.iceberg.tornadovm.bench.ConvertBench tornadovm/run.sh gpu /data/wh gpu 4 eq1m eq10m
```

To run everything on a new machine with one command (checks, build, tables, every suite,
summary), use `mikepapadim/iceberg-gpu-bench`.

## Beyond the core

- `org.apache.iceberg.tornadovm.join`: star aggregates and probe joins over the scan's grouped
  pipeline ([DEVICE-JOINS.md](DEVICE-JOINS.md)). Optional; nothing in the core depends on it.
- `tornadovm/spark`: a Spark integration (reader, aggregates, rewrite runner). Separate and
  optional ([spark/README.md](spark/README.md)).
