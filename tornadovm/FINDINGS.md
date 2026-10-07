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

# Findings: Iceberg merge-on-read on the GPU (TornadoVM + cuDF)

## Summary

On 100M rows (64 Parquet files) with every kind of Iceberg delete, the GPU scan is **2-3x faster than
the same algorithm on all 32 CPU threads, and 1.4-3x faster than Spark 3.5**:

- **Equality deletes:** the gap widens to **4-6x vs the CPU and 8-23x vs Spark**. At 10M equality
  deletes, Spark runs out of memory with a 40 GB heap.
- **Correctness:** every engine returns the same live-row count and checksum on every table and
  query (36 of 36).
- **Iceberg's single-threaded reader:** `IcebergGenerics` is 40-360x slower than the GPU.

The device time goes almost entirely to cuDF reading Parquet: about 88% of a 1M-row file. The
TornadoVM kernels that apply Iceberg's deletes take about 20 us a batch. The GPU path is bound by
how fast data reaches the device, not by the merge-on-read logic.

## Setup

**Hardware**

- **GPU:** NVIDIA RTX 4090 (24 GB), driver 610.57.
- **CPU:** Intel i9-13900K, 32 threads, 62 GB RAM.
- **Storage:** local NVMe; the data is in the page cache (warm runs).

**Software**

- **TornadoVM:** `develop` plus beehive-lab/TornadoVM#1200, JDK 21, CUDA backend; libcudf 26.08
  (cu12 wheels).
- **Iceberg:** 1.12.0 plus this module.
- **Spark:** 3.5.9 with `iceberg-spark-runtime-3.5_2.12-1.12.0`.

**Data** (`GenerateTables`)

- 64 data files of 1,562,500 rows each: `id long, k int (0..999), v1..v4 double`, all required.
- 3.1 GB of zstd Parquet, Iceberg's default writer settings.
- Every table shares the same data files; each one adds the deletes of one scenario.
- Position deletes are file-scoped, as Iceberg writes them by default
  (`write.delete.granularity=file`).
- Equality-delete keys are random existing ids.

**Engines**

| engine | what it is |
|---|---|
| gpu | `DeviceMergeOnReadScan`: cuDF Parquet reads and set probe, TornadoVM kernels, 3 streams, 8M-row batches |
| cpu | `CpuMergeOnReadScan`: Iceberg's vectorized `ArrowReader` per file, the same delete application, 32 threads |
| spark | Spark 3.5.9 `local[32]`, 40 GB heap, Iceberg's vectorized merge-on-read, `split-size` 32 MB (faster than the 128 MB default here) |
| generics | `IcebergGenerics`, Iceberg's row-at-a-time reader, 1 thread (Q2 only) |

**Method**

- Each engine runs one warm-up and five timed runs per table. The tables report the median of the
  five.
- A GPU or CPU scan object is kept for all of a table's runs, the way an engine keeps its
  operators, so only the warm-up pays for TornadoVM's JIT and buffer allocation.
- Spark's time is that of the query's action. Q3 there writes to Spark's `noop` sink, which
  materializes rows without copying them anywhere; the GPU and CPU instead copy every live row into
  host arrays and sum them.
- Raw per-run CSVs, with the GPU's per-stage breakdown, are in `results/`.

## Results


### Q1 `SELECT count(*), sum(v1)`

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | GPU vs cpu | GPU vs spark | same answer |
|---|---|---|---|---|---|---|---|
| no deletes | 100,000,000 | 115 | 312 | 292 | 2.7x | 2.5x | yes |
| 1% position deletes (v2) | 99,001,414 | 161 | 316 | 282 | 2.0x | 1.8x | yes |
| 5% position deletes (v2) | 95,000,729 | 161 | 351 | 300 | 2.2x | 1.9x | yes |
| 20% position deletes (v2) | 79,999,955 | 162 | 348 | 296 | 2.1x | 1.8x | yes |
| 5% deletion vectors (v3) | 95,000,729 | 120 | 277 | 315 | 2.3x | 2.6x | yes |
| 10k equality deletes | 99,990,001 | 152 | 412 | 658 | 2.7x | 4.3x | yes |
| 1M equality deletes | 99,005,073 | 143 | 542 | 2,887 | 3.8x | 20.2x | yes |
| 10M equality deletes | 90,483,415 | 190 | 1,213 | failed (OOM) | 6.4x |  | yes |
| 5% position + 1M equality | 94,055,589 | 221 | 615 | 3,358 | 2.8x | 15.2x | yes |

### Q2 `SELECT k, count(*), sum(v1..v4) GROUP BY k` (1000 groups)

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | generics (ms) | GPU vs cpu | GPU vs spark | GPU vs generics | same answer |
|---|---|---|---|---|---|---|---|---|---|
| no deletes | 100,000,000 | 292 | 890 | 905 | 12,808 | 3.0x | 3.1x | 43.9x | yes |
| 1% position deletes (v2) | 99,001,414 | 381 | 964 | 923 | 16,392 | 2.5x | 2.4x | 43.0x | yes |
| 5% position deletes (v2) | 95,000,729 | 388 | 967 | 1,009 | 16,677 | 2.5x | 2.6x | 43.0x | yes |
| 20% position deletes (v2) | 79,999,955 | 409 | 1,068 | 943 | 16,631 | 2.6x | 2.3x | 40.6x | yes |
| 5% deletion vectors (v3) | 95,000,729 | 296 | 936 | 913 | 15,956 | 3.2x | 3.1x | 53.8x | yes |
| 10k equality deletes | 99,990,001 | 312 | 1,070 | 1,369 | 18,817 | 3.4x | 4.4x | 60.4x | yes |
| 1M equality deletes | 99,005,073 | 308 | 1,268 | 3,895 | 28,900 | 4.1x | 12.6x | 93.7x | yes |
| 10M equality deletes | 90,483,415 | 340 | 1,873 | failed (OOM) | 123,486 | 5.5x |  | 362.8x | yes |
| 5% position + 1M equality | 94,055,589 | 456 | 1,284 | 3,794 | 31,771 | 2.8x | 8.3x | 69.7x | yes |

### Q3 `SELECT id, k, v1` -- every live row materialized on the host

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | GPU vs cpu | GPU vs spark | same answer |
|---|---|---|---|---|---|---|---|
| no deletes | 100,000,000 | 285 | 778 | 504 | 2.7x | 1.8x | yes |
| 1% position deletes (v2) | 99,001,414 | 338 | 690 | 457 | 2.0x | 1.4x | yes |
| 5% position deletes (v2) | 95,000,729 | 323 | 716 | 570 | 2.2x | 1.8x | yes |
| 20% position deletes (v2) | 79,999,955 | 312 | 844 | 502 | 2.7x | 1.6x | yes |
| 5% deletion vectors (v3) | 95,000,729 | 284 | 696 | 525 | 2.4x | 1.8x | yes |
| 10k equality deletes | 99,990,001 | 290 | 752 | 812 | 2.6x | 2.8x | yes |
| 1M equality deletes | 99,005,073 | 289 | 875 | 3,739 | 3.0x | 12.9x | yes |
| 10M equality deletes | 90,483,415 | 296 | 1,607 | failed (OOM) | 5.4x |  | yes |
| 5% position + 1M equality | 94,055,589 | 355 | 897 | 3,470 | 2.5x | 9.8x | yes |

### Q4 `SELECT bucket(16, id), count(*), sum(v1) GROUP BY 1` -- Iceberg's bucket transform

| table | live rows | gpu (ms) | cpu (ms) | spark (ms) | GPU vs cpu | GPU vs spark | same answer |
|---|---|---|---|---|---|---|---|
| no deletes | 100,000,000 | 146 | 432 | 362 | 3.0x | 2.5x | yes |
| 1% position deletes (v2) | 99,001,414 | 184 | 477 | 423 | 2.6x | 2.3x | yes |
| 5% position deletes (v2) | 95,000,729 | 186 | 390 | 424 | 2.1x | 2.3x | yes |
| 20% position deletes (v2) | 79,999,955 | 188 | 532 | 436 | 2.8x | 2.3x | yes |
| 5% deletion vectors (v3) | 95,000,729 | 146 | 312 | 425 | 2.1x | 2.9x | yes |
| 10k equality deletes | 99,990,001 | 160 | 357 | 719 | 2.2x | 4.5x | yes |
| 1M equality deletes | 99,005,073 | 147 | 517 | 3,320 | 3.5x | 22.6x | yes |
| 10M equality deletes | 90,483,415 | 190 | 1,225 | failed (OOM) | 6.5x |  | yes |
| 5% position + 1M equality | 94,055,589 | 218 | 642 | 3,485 | 2.9x | 16.0x | yes |

## Where the time goes (GPU)

- **Per 1M-row file, profiled with one stream:**
  - `readParquetColumns` (cuDF): 9.7 ms of 11.1 ms;
  - the two cuDF group-bys: 0.8 ms;
  - each TornadoVM kernel: about 20 us.
- **Batching:** 1M-row reads are latency-bound. Reading several files in one cuDF call took the
  device time of Q2 on 4M rows from about 65 ms to 40 ms.
- **Streams:** three batches in flight, each in its own region with its own CUDA stream, overlap
  file reads and copies with decoding. That took Q2 from 470 ms to 290 ms, about 10.7 GB/s of
  Parquet.
- **Deletes on the device:** reading delete files on the device took the host delete-load time off
  the critical path. `BaseDeleteLoader` reading position-delete files row by row had cost 100-170 ms
  per 4 files, more than the device work itself.
- **Plan reuse:** keeping buffers and plans between queries took Q2 on 4M rows from 135 ms to 40 ms.

## What this shows, and what it does not

- **Merge-on-read is cheap on the device.** Folding position deletes, deletion vectors, equality
  deletes and null semantics into a live mask is a few microseconds of kernel time on top of the
  read. Equality deletes in particular are where CPU engines hurt (hash-set probes per row, and in
  Spark a set per task), and where the GPU is furthest ahead.
- **The hybrid split works.** cuDF does what needs cross-row cooperation: Parquet decoding,
  hashing, group-by, compaction. Plain Java kernels JIT-compiled by TornadoVM do the
  format-specific per-row work, including Iceberg's Murmur3 `bucket[N]` transform (Q4), on the same
  device buffers.
- **The data is cached.** It is local and in the page cache, so this measures decode and compute.
  On object storage the scan would mostly wait on I/O.
- **The baselines are a prototype harness, not production engines.** The CPU baseline is our own
  code on Iceberg's fastest Java reader. Spark is a real engine and adds overhead the prototype does
  not have (task scheduling, row conversion). Neither is Spark RAPIDS, which this was not compared
  with.
- **Supported scope:** local Parquet; INT/LONG/DATE/TIMESTAMP/DOUBLE columns; equality deletes on
  one column; unsplit files. Anything else is refused.

## Bugs and traps found

**In TornadoVM**

- **Fixed in beehive-lab/TornadoVM#1200.** A library task's `StringBuilder` path holder was
  serialized to the device:
  - `BufferOverflowException` once the holder grew;
  - `unimplemented: field type [B`, reliably with compressed oops off.

  It is now passed by reference.
- **Reported in the PR, not fixed.**
  - CUDA streams are keyed by (plan, thread), but a library's context, and with it cuDF's stream,
    is cached by plan only. A plan executed from a second thread runs its cuDF calls on another
    stream from its copies and kernels. The workaround here is one owning thread per plan.
  - `tornado.device.memory` (4 GB by default) keeps counting buffers released by closed plans.
    `run.sh` raises it to 20 GB.
- **Pre-existing and worked around.** With `-ea`, Graal checks a CUDA plugin for
  `TileContext.view(ByteArray,...)` that `tornado-api` does not have. Tests run with
  `-da:tornado... -da:uk.ac.manchester.tornado...`.

**In Iceberg**

- `ArrowReader` refuses any file with deletes (apache/iceberg#2487).
- Position-delete files count as file-scoped only when `file_path` has full bounds, so they must be
  written with `MetricsConfig.forPositionDelete()`, as Iceberg's own writer factories do.

**In Spark + Iceberg**

- 10M equality deletes run out of memory at `local[32]` with a 40 GB heap: each task holds its own
  set of the keys.

**Correctness note**

- One warm-up run of Q3 on `eq10m` returned 91,227,442 live rows instead of 90,483,415. It was
  before the holder fix above, and it has not recurred in more than 60 full scans since. Its cause is
  not proven.
