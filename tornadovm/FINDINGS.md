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

# Findings: Iceberg merge-on-read, filters and compaction on the GPU (TornadoVM + cuDF)

## Summary

Data: 100M rows in 64 Parquet files, with every kind of Iceberg delete. The GPU scan runs
merge-on-read, filters and compaction.

**Scans (Q1–Q5)**
- **Without equality deletes:** 1.5–3.4x faster than the same algorithm on all 32 CPU threads, and
  1.6–3.3x faster than Spark 3.5.
- **With equality deletes:** 2.8–7.8x faster than the CPU, and 4–200x faster than Spark. Spark runs
  out of memory on 10M equality deletes at a 40 GB heap, except under Q5's file pruning, where it
  takes 21 s.

**Compaction** (rewriting every data file with its deletes applied)
- **Without equality deletes:** 1.3–1.4x faster than Spark's `rewrite_data_files`.
- **With equality deletes:** 3.2–3.5x faster; at 10M equality deletes the GPU rewrites the table in
  2.4 s and Spark runs out of memory.

**Correctness**
- Every engine returns the same live-row count and checksum on every table and query: 51 of 51,
  warm-up runs included.
- Compaction is checked by an independent CPU scan of the rewritten table.

**Iceberg's single-threaded reader:** `IcebergGenerics` is 50–325x slower than the GPU.

**Bound:** the device time goes to cuDF reading and, for compaction, writing Parquet. The
TornadoVM kernels that apply Iceberg's deletes, filters and transforms take microseconds a batch.

## What runs where

**CPU (Iceberg's own planning)**
- `planFiles(filter)` prunes files by their statistics and decides which delete files apply to
  which data file, by sequence number.

**Host pool, ahead of the device**
- reads Parquet footers;
- loads deletion vectors with `BaseDeleteLoader`;
- names the delete files that the device will read itself.

**Device, one TornadoVM execution plan a batch of files, 4 in flight (8 for compaction)**
- `Cudf.readParquetColumns`: the data files of the batch in one cuDF call, and the `pos` and key
  columns of their delete files.
- `Cudf.containedIn`: the equality-delete hash probe.
- TornadoVM kernels, written in Java:
  - the live-row mask, combining the deletion vectors, the probe and Iceberg's null rules;
  - the position-delete scatter;
  - the residual filter, compiled from the Iceberg expression;
  - Iceberg's `bucket[N]` Murmur3 transform;
  - the gathers.
- The consumer, one of:
  - cuDF group-by (only the groups come back);
  - cuDF compaction and gathers (only the live rows come back);
  - compaction: `Cudf.writeParquetColumns` writes the live rows to a new ZSTD Parquet file, with
    Iceberg field ids, straight from device memory. One `RewriteFiles` commit, with metrics from
    the footers libcudf wrote, then swaps the old data and delete files for the new ones.

## Setup

**Hardware**
- **GPU:** NVIDIA RTX 4090 (24 GB), driver 610.57.
- **CPU:** Intel i9-13900K, 32 threads, 62 GB RAM.
- **Storage:** local NVMe; data in the page cache.

**Software**
- **TornadoVM:** `develop` plus the PRs listed below (`results/tornadovm-commits.txt`), JDK 21;
  libcudf 26.08 (cu12 wheels).
- **Iceberg:** 1.12.0 plus this module.
- **Spark:** 3.5.9 with `iceberg-spark-runtime-3.5_2.12-1.12.0`.

**Data** (`GenerateTables`)
- 64 data files of 1,562,500 rows: `id long, k int (0..999), v1..v4 double`, all required.
- 3.1 GB of zstd Parquet, Iceberg's default writer.
- One table per delete scenario, all sharing the same data files.
- Position deletes are file-scoped, as Iceberg writes them by default.

**Engines**

| engine | what it is |
|---|---|
| gpu | `DeviceMergeOnReadScan`: cuDF Parquet read/write, set probe, group-by, compaction; TornadoVM kernels |
| cpu | `CpuMergeOnReadScan`: Iceberg's vectorized `ArrowReader` a file, the same delete and filter evaluation, 32 threads |
| spark | Spark 3.5.9 `local[32]`, 40 GB heap, Iceberg's vectorized merge-on-read, `split-size` 32 MB. Compaction: `rewrite_data_files` with `rewrite-all`, 256 MB file groups, 32 concurrent groups, 128 MB target files |
| generics | `IcebergGenerics`, Iceberg's row-at-a-time reader, 1 thread (Q2 only) |

**Method**
- **Timing:** one warm-up and five timed runs per table and query; the tables report the median.
  A GPU or CPU scan object is kept across a table's runs.
- **Q3:** Spark writes to its `noop` sink, which materializes rows without copying them; the
  GPU and CPU copy every live row into host arrays and sum them.
- **Compaction:** each run rewrites the table, checks the rewritten table with an independent CPU
  scan (Q2's answer must not change), then restores the original snapshot and deletes the new files.
- **Raw data:** per-run CSVs, with the GPU's per-stage breakdown, are in `results/`.

## Results


### Q1 `SELECT count(*), sum(v1)`

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | GPU vs cpu | GPU vs spark | same answer |
|---|---|---|---|---|---|---|---|
| no deletes | 100,000,000 | 128 | 285 | 263 | 2.2x | 2.0x | yes |
| 1% position deletes (v2) | 99,001,414 | 170 | 303 | 275 | 1.8x | 1.6x | yes |
| 5% position deletes (v2) | 95,000,729 | 179 | 264 | 291 | 1.5x | 1.6x | yes |
| 20% position deletes (v2) | 79,999,955 | 181 | 414 | 309 | 2.3x | 1.7x | yes |
| 5% deletion vectors (v3) | 95,000,729 | 129 | 207 | 287 | 1.6x | 2.2x | yes |
| 10k equality deletes | 99,990,001 | 146 | 411 | 643 | 2.8x | 4.4x | yes |
| 1M equality deletes | 99,005,073 | 146 | 568 | 3,063 | 3.9x | 20.9x | yes |
| 10M equality deletes | 90,483,415 | 217 | 1,249 | failed (OOM) | 5.8x |  | yes |
| 5% position + 1M equality | 94,055,589 | 227 | 628 | 3,342 | 2.8x | 14.7x | yes |

### Q2 `SELECT k, count(*), sum(v1..v4) GROUP BY k` (1000 groups)

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | generics (ms) | GPU vs cpu | GPU vs spark | GPU vs generics | same answer |
|---|---|---|---|---|---|---|---|---|---|
| no deletes | 100,000,000 | 285 | 962 | 925 | 15,246 | 3.4x | 3.3x | 53.6x | yes |
| 1% position deletes (v2) | 99,001,414 | 355 | 957 | 1,001 | 20,198 | 2.7x | 2.8x | 56.9x | yes |
| 5% position deletes (v2) | 95,000,729 | 351 | 989 | 959 | 20,422 | 2.8x | 2.7x | 58.1x | yes |
| 20% position deletes (v2) | 79,999,955 | 411 | 1,061 | 907 | 20,157 | 2.6x | 2.2x | 49.1x | yes |
| 5% deletion vectors (v3) | 95,000,729 | 295 | 924 | 904 | 19,390 | 3.1x | 3.1x | 65.7x | yes |
| 10k equality deletes | 99,990,001 | 305 | 1,130 | 1,346 | 20,201 | 3.7x | 4.4x | 66.2x | yes |
| 1M equality deletes | 99,005,073 | 310 | 1,268 | 4,142 | 30,898 | 4.1x | 13.4x | 99.8x | yes |
| 10M equality deletes | 90,483,415 | 390 | 1,900 | failed (OOM) | 126,541 | 4.9x |  | 324.6x | yes |
| 5% position + 1M equality | 94,055,589 | 401 | 1,334 | 3,855 | 34,407 | 3.3x | 9.6x | 85.7x | yes |

### Q3 `SELECT id, k, v1` -- every live row materialized on the host

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | GPU vs cpu | GPU vs spark | same answer |
|---|---|---|---|---|---|---|---|
| no deletes | 100,000,000 | 278 | 701 | 448 | 2.5x | 1.6x | yes |
| 1% position deletes (v2) | 99,001,414 | 331 | 790 | 515 | 2.4x | 1.6x | yes |
| 5% position deletes (v2) | 95,000,729 | 324 | 743 | 547 | 2.3x | 1.7x | yes |
| 20% position deletes (v2) | 79,999,955 | 311 | 829 | 540 | 2.7x | 1.7x | yes |
| 5% deletion vectors (v3) | 95,000,729 | 284 | 722 | 491 | 2.5x | 1.7x | yes |
| 10k equality deletes | 99,990,001 | 288 | 876 | 797 | 3.0x | 2.8x | yes |
| 1M equality deletes | 99,005,073 | 282 | 853 | 3,660 | 3.0x | 13.0x | yes |
| 10M equality deletes | 90,483,415 | 315 | 1,628 | failed (OOM) | 5.2x |  | yes |
| 5% position + 1M equality | 94,055,589 | 352 | 918 | 3,907 | 2.6x | 11.1x | yes |

### Q4 `SELECT bucket(16, id), count(*), sum(v1) GROUP BY 1` -- Iceberg's bucket transform

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | GPU vs cpu | GPU vs spark | same answer |
|---|---|---|---|---|---|---|---|
| no deletes | 100,000,000 | 148 | 385 | 409 | 2.6x | 2.8x | yes |
| 1% position deletes (v2) | 99,001,414 | 195 | 488 | 430 | 2.5x | 2.2x | yes |
| 5% position deletes (v2) | 95,000,729 | 202 | 480 | 447 | 2.4x | 2.2x | yes |
| 20% position deletes (v2) | 79,999,955 | 211 | 534 | 394 | 2.5x | 1.9x | yes |
| 5% deletion vectors (v3) | 95,000,729 | 149 | 334 | 440 | 2.3x | 3.0x | yes |
| 10k equality deletes | 99,990,001 | 148 | 408 | 698 | 2.8x | 4.7x | yes |
| 1M equality deletes | 99,005,073 | 149 | 623 | 3,272 | 4.2x | 22.0x | yes |
| 10M equality deletes | 90,483,415 | 218 | 1,232 | failed (OOM) | 5.7x |  | yes |
| 5% position + 1M equality | 94,055,589 | 230 | 673 | 3,216 | 2.9x | 14.0x | yes |

### Q5 `SELECT k, count(*), sum(v1) WHERE id < 25000000 AND v2 > 0.5 GROUP BY k` -- file pruning + residual filter

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | GPU vs cpu | GPU vs spark | same answer |
|---|---|---|---|---|---|---|---|
| no deletes | 12,498,401 | 85 | 203 | 251 | 2.4x | 2.9x | yes |
| 1% position deletes (v2) | 12,373,518 | 104 | 218 | 267 | 2.1x | 2.6x | yes |
| 5% position deletes (v2) | 11,874,204 | 103 | 212 | 280 | 2.1x | 2.7x | yes |
| 20% position deletes (v2) | 10,000,063 | 104 | 262 | 298 | 2.5x | 2.9x | yes |
| 5% deletion vectors (v3) | 11,874,204 | 81 | 207 | 270 | 2.5x | 3.3x | yes |
| 10k equality deletes | 12,497,207 | 77 | 238 | 390 | 3.1x | 5.1x | yes |
| 1M equality deletes | 12,374,374 | 79 | 330 | 933 | 4.2x | 11.8x | yes |
| 10M equality deletes | 11,310,079 | 106 | 831 | 21,205 | 7.8x | 199.5x | yes |
| 5% position + 1M equality | 11,756,308 | 132 | 358 | 987 | 2.7x | 7.5x | yes |

### Compaction -- rewrite every data file with its deletes applied (GPU: rewriteDataFiles; Spark: rewrite_data_files); answer = Q2 on the rewritten table

| table | live rows | gpu (ms) | spark (ms) | GPU vs spark | same answer |
|---|---|---|---|---|---|
| no deletes | 100,000,000 | 2,528 | 3,276 | 1.3x | yes |
| 5% position deletes (v2) | 95,000,729 | 2,659 | 3,368 | 1.3x | yes |
| 20% position deletes (v2) | 79,999,955 | 2,212 | 3,141 | 1.4x | yes |
| 1M equality deletes | 99,005,073 | 2,558 | 8,928 | 3.5x | yes |
| 10M equality deletes | 90,483,415 | 2,435 | failed (OOM) |  | yes |
| 5% position + 1M equality | 94,055,589 | 2,700 | 8,620 | 3.2x | yes |


## What this shows, and what it does not

- **Merge-on-read is cheap on the device; equality deletes are where CPUs hurt.** A CPU engine
  probes a hash set per row, and Spark builds one per task. On the GPU the probe is one cuDF call
  per batch, and the delete logic is a few microseconds of kernel time.
- **The planning stays Iceberg's.** Filters prune files by statistics (Q5 reads 16 of 64 files),
  delete applicability follows sequence numbers, and a compaction commits through `RewriteFiles`
  validated against the planned snapshot.
- **Compaction is bound by writing Parquet.** The GPU's lead over Spark is smallest (1.3–1.4x)
  where there is little to merge; with equality deletes Spark's merge dominates and the GPU's
  lead is 3.2–3.5x.
- **The data is cached.** It is local and in the page cache, so this measures decode and compute.
  On object storage, a scan would mostly wait on I/O.
- **The baselines.** The CPU baseline is our code on Iceberg's fastest Java reader. Spark is a
  real engine with overheads the prototype does not have. Spark RAPIDS was not compared.
- **Supported scope:**
  - local Parquet; INT/LONG/DATE/TIMESTAMP/DOUBLE columns;
  - filters that are conjunctions of comparisons, `IN` and `IS [NOT] NULL`;
  - single-column equality deletes; unsplit files;
  - compaction of v2 tables only, since v3 needs row lineage.

  Anything else is refused.

## TornadoVM changes made along the way

| PR | what |
|---|---|
| beehive-lab/TornadoVM#1200 (merged) | `readParquetColumns` (typed, nullable, several files a read), `containedIn`; library-task text arguments passed by reference rather than serialized to the device |
| #1205 (merged) | `writeParquetColumns`: device buffers to Parquet, with field ids, null masks, statistics; row count from a device counter |
| #1203 (merged) | Device-only intermediates (written and read only on the device) were never freed when a plan closed: a process running large plans one after another ran out of device memory |
| #1204 (merged) | The zero-fill of a recycled write-only buffer ran on the legacy stream, unordered with TornadoVM's non-blocking streams, and could wipe what a plan had just written. Under concurrency this gave wrong rows on first executions, until fixed |
| #1202 (open) | Compiler plugins registered for methods that do not exist stopped the CUDA backend from starting under a blanket `-ea` (Gradle, Maven, IDEs) |

Two related issues are already covered by other PRs:
- **#1185 and #1187:** library calls followed the stream of the thread that created the context.

## Traps

- **GPU JVM heap:** keep it under 32 GB. TornadoVM's object serialization assumes compressed oops.
- **TornadoVM device budget:** raise `tornado.device.memory`. `run.sh` uses 20 GB.
- **Position deletes on the device:** the delete files must be file-scoped, that is written with
  `MetricsConfig.forPositionDelete()`.
- **Null semantics:** Iceberg's row evaluator orders nulls first, so a null satisfies `<`, `<=` and
  `!=`. The device follows it.
- **Concurrency:** a TornadoVM plan must run on one thread, because its CUDA streams are per thread.
  Each region owns one.
