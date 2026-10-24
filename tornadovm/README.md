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
  scan.convertEqualityDeletes();       // equality deletes -> a deletion vector per data file (v3),
                                       //   or a file-scoped position-delete file (v2)
  scan.deleteKeys("id", keys);         // DELETE ... WHERE id IN (keys), as deletion vectors
  scan.computeTableStats("id", "k");   // NDV theta sketches in a Puffin file, as compute_table_stats
  scan.rewriteDataFiles();             // compaction: deletes applied, files rewritten
  scan.rewriteDataFilesSorted();       // compaction in the table's sort order, sort_order_id set
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
deletes (#17783), so this is the conversion v4 tables need. On a v2 table each data file gets one
file-scoped position-delete file instead.

**`deleteKeys(column, keys)`** deletes by key without writing an equality-delete file: what a CDC
writer does for keys it has seen before (Spark and Flink write deletion vectors only for positions
they already know, #11561, #14197). Iceberg prunes files by the column's bounds (`IN` up to
`iceberg.tornadovm.deleteKeys.inLimit` keys, 1000 by default, otherwise the key range); the keys
stay in memory; the device finds their rows. The `RowDelta` also validates that no data file added
since could hold one of the keys.

**`computeTableStats(columns...)`** writes what Spark's `compute_table_stats` writes: an
`apache-datasketches-theta-v1` blob a column (MurmurHash3 seed 9001 over Iceberg's serialization,
4096 entries) with the `ndv` property, in a Puffin file set on the snapshot. The device hashes and
sorts; the host merges the smallest hashes.

**`rewriteDataFilesSorted()`** is `rewrite_data_files(strategy => 'sort')` with the table's sort
order: identity fields, either direction and null order, and the order's id on every new file
(`sort_order_id`, as Spark's writes set it since #15150 and #15832). Each output file is sorted;
files are not sorted against each other, as with Spark's file groups. For selective compaction,
`rewriteGroup(tasks, targetFileBytes)` and `rewriteGroupSorted(...)` rewrite a file group that any
planner chose (a `where`, a delete-file threshold) and leave the commit to the caller.

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
| Conversion, delete by key | v3 (deletion vectors) and v2 (file-scoped position deletes); int/long keys | partition-scoped position delete files |
| Sorted rewrite | up to 4 identity sort fields on int, long, date, timestamp, double | transforms (bucket, truncate, ...) in the sort order |

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
  filters, strings, decimals, compaction with v3 row lineage, z-order, sorted rewrites (checked
  with Iceberg's `SortOrderComparators`), conversion to deletion vectors and to v2 position deletes
  (GPU and CPU twin give the same vectors), delete by key, and NDV sketches (checked against
  DataSketches' own update sketch).
- **`TestDeviceEvolutionAndStorage`:** evolved schemas, object-store tables, partitioned compaction
  and partitioned conversion.

The benchmarks print each answer's row count and checksum, and every engine must agree.

## Set up

1. **libcudf** from NVIDIA's wheels (about 1.3 GB, once; no conda): `source tornadovm/cudf-env.sh`.
   `CUDF_PACKAGE=libcudf-cu13` selects the CUDA 13 build.
2. **TornadoVM with the cuDF shim**, built with the CUDA backend (released SDKs don't ship the
   shim): branch `iceberg-bench` of `mikepapadim/TornadoVM`, which is master plus three changes
   in review upstream (beehive-lab/TornadoVM #1276, #1277, #1278).
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
tornadovm/run.sh gpu /data/wh sorted 4 pos05 eq1m             # sorted rewrite; `spark` for Spark's
MAIN=org.apache.iceberg.tornadovm.bench.ConvertBench tornadovm/run.sh gpu /data/wh gpu 4 eq1m eq10m
MAIN=org.apache.iceberg.tornadovm.bench.NewOpsBench tornadovm/run.sh gpu /data/wh deletekeys 3 none 1000 1000000
MAIN=org.apache.iceberg.tornadovm.bench.NewOpsBench tornadovm/run.sh gpu /data/wh convertv2 3 eq1m eq10m
MAIN=org.apache.iceberg.tornadovm.bench.StatsBench tornadovm/run.sh gpu /data/wh id,k,v1,v2 3 none eq1m
```

The Spark baselines run the same operations through Spark 3.5 and Iceberg's procedures:
`BENCH_SCRIPT=spark_delete_keys.py` (`MERGE ... WHEN MATCHED THEN DELETE`) and
`BENCH_SCRIPT=stats_bench.py` (`compute_table_stats`) with `tornadovm/run.sh spark`.

| Property | Default | |
|---|---|---|
| `iceberg.tornadovm.batch-rows` | 8M | rows a batch |
| `iceberg.tornadovm.streams` | 3 | batches in flight, one CUDA stream each |
| `iceberg.tornadovm.deleteKeys.inLimit` | 1000 | up to this many keys prune by `IN`, more by range |
| `iceberg.tornadovm.positionWriters` | 0 (auto: one per 32k positions, up to a core each) | v2 position-delete writers |
| `tornado.cuda.sched` | `yield` | `blocking`: waiting threads sleep; less host CPU, a little slower (TornadoVM #1277) |

To run everything on a new machine with one command (checks, build, tables, every suite,
summary), use `mikepapadim/iceberg-gpu-bench`.

## Beyond the core

- `org.apache.iceberg.tornadovm.join`: star aggregates and probe joins over the scan's grouped
  pipeline ([DEVICE-JOINS.md](DEVICE-JOINS.md)). Optional; nothing in the core depends on it.
- `tornadovm/spark`: a Spark integration (reader, aggregates, rewrite runner). Separate and
  optional ([spark/README.md](spark/README.md)).
