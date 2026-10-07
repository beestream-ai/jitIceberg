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

# iceberg-tornadovm: merge-on-read on the GPU, from Java

A prototype of Iceberg **merge-on-read** scans on an NVIDIA GPU, built from two pieces that already
exist on the JVM:

- **RAPIDS cuDF** through [TornadoVM](https://github.com/beehive-lab/TornadoVM)'s `tornado-cudf`
  binding: Parquet decoding, the hash probe of equality deletes, group-by, stream compaction;
- **TornadoVM kernels written in Java**, JIT-compiled to CUDA: the Iceberg-specific parts that cuDF
  has no operator for -- folding position deletes, deletion vectors, equality-delete hits and
  Iceberg's null semantics into one live mask, scattering position deletes, gathering live rows,
  and Iceberg's `bucket[N]` transform.

Both work on the same device buffers. Nothing round-trips through the host between them.

Planning stays on the CPU and is Iceberg's own: `planFiles()` decides which delete files apply to which data
file, by sequence number, and the device scan applies exactly those.

## What it does

```java
try (DeviceMergeOnReadScan scan = DeviceMergeOnReadScan.of(table)) {
  AggregateResult byK = scan.aggregate("k", "v1", "v2");          // GROUP BY k: COUNT(*), SUM(v1), SUM(v2)
  AggregateResult buckets = scan.aggregateByBucket("id", 16, "v1"); // GROUP BY bucket(16, id)
  long live = scan.materialize(List.of("id", "k", "v1"), rows -> { ... }); // live rows, on the host
}
```

Each query goes through these steps:

1. **Plan and batch.** Iceberg plans the file tasks. Consecutive data files that share their
   equality deletes are grouped into batches of up to `iceberg.tornadovm.batch-rows` rows (8M by
   default).
2. **Prepare on the host.** A host pool, running ahead of the device, does the work per batch:
   - reads the Parquet footers, to map field ids to columns;
   - loads deletion vectors with `BaseDeleteLoader` into a bitmap;
   - names the delete files the device will read itself.
3. **Run on the device.** Each batch runs through one TornadoVM execution plan:
   - `Cudf.readParquetColumns` reads every data file of the batch in one call;
   - the `pos` column of its file-scoped position-delete files is read the same way;
   - the key column of its equality-delete files is read the same way;
   - `Cudf.containedIn` probes the equality-delete set;
   - TornadoVM kernels build the live mask and apply the position deletes;
   - then the consumer runs: a cuDF group-by (only the groups come back), or compaction plus
     gathers (only the live rows come back).
4. **Overlap batches.** `iceberg.tornadovm.streams` regions (3 by default) run batches
   concurrently, each with its own buffers, plan and CUDA stream, so one batch's file reads overlap
   another's decoding. A scan keeps its regions between queries.

Supported:

- local Parquet data files;
- INT/DATE, LONG/TIMESTAMP/TIME and DOUBLE columns, required or optional;
- v2 position deletes, v3 deletion vectors;
- equality deletes on one INT or LONG column.

Anything else (object stores, strings, multi-column equality deletes, split files, schema evolution
that adds columns) is refused with `UnsupportedOperationException` rather than answered wrongly.

`CpuMergeOnReadScan` is the same algorithm on all cores. It reads with Iceberg's vectorized
`ArrowReader`, which on its own refuses any file with deletes (#2487), so it is handed each file
without its deletes and applies them itself, with the same delete preparation as the GPU path.

## Correctness

`TestDeviceMergeOnRead` checks every device and CPU scan against `IcebergGenerics`, Iceberg's own
merge-on-read reader. It covers:

- v2 position deletes;
- v3 deletion vectors;
- equality deletes committed before and after an append, which tests the sequence-number rules;
- nulls in keys and values, including a null in the delete set;
- an INT equality key;
- many row groups;
- `bucket(16, id)` against `Transforms.bucket`;
- each device scan run twice, reusing its plans.

On the benchmark tables every engine returns the same row count and checksum.

## Running

Requirements:

- a TornadoVM SDK whose cuDF shim has `readParquetColumns`
  ([beehive-lab/TornadoVM#1200](https://github.com/beehive-lab/TornadoVM/pull/1200)), built with
  `make BACKEND=cuda` and `CUDF_HOME` pointing at libcudf (the `libcudf-cu12` wheels are enough);
- JDK 21;
- every RAPIDS library directory on `LD_LIBRARY_PATH` (see TornadoVM's `tornado-cudf/README.md`).

The module is part of the build only when `TORNADOVM_HOME` is set.

```bash
source $TORNADOVM/setvars.sh                      # TORNADOVM_HOME, JAVA_HOME (21)
./gradlew :iceberg-tornadovm:test

tornadovm/run.sh gen   /data/wh 64 1562500        # 100M rows, one table a delete scenario
tornadovm/run.sh gpu   /data/wh q2 6 none pos05 dv05 eq1m mixed
tornadovm/run.sh cpu   /data/wh q2 6 none pos05 dv05 eq1m mixed
SPLIT_SIZE=33554432 tornadovm/run.sh spark /data/wh q2 6 none pos05 dv05 eq1m mixed   # SPARK_HOME, ICEBERG_SPARK_JAR
python3 tornadovm/summarize.py results/
```

The results are in [FINDINGS.md](FINDINGS.md).
