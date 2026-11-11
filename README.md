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

# jitIceberg: Apache Iceberg on the GPU

jitIceberg is [Apache Iceberg](https://iceberg.apache.org) `main` with Iceberg's per-row work moved
to an NVIDIA GPU. The GPU code is plain Java that [TornadoVM](https://github.com/beehive-lab/TornadoVM)
compiles to CUDA at run time (hence *jit*), plus [cuDF](https://github.com/rapidsai/cudf) for
Parquet. Iceberg keeps doing what it does today: planning, delete-file resolution, validation and
commits. Only the work in between runs on the device, and anything the GPU cannot do exactly falls
back to Iceberg's own code.

## What runs on the GPU

| Operation | API | In Spark 4.2 |
|---|---|---|
| Merge-on-read scans with position deletes, deletion vectors and equality deletes, filters and grouped aggregates | `DeviceMergeOnReadScan.aggregate / materialize` | reads and `GROUP BY` queries |
| Compaction (bin-pack, sorted, z-order), deletes applied | `rewriteDataFiles`, `rewriteDataFilesSorted`, `rewriteDataFilesZOrder` | `CALL system.rewrite_data_files` |
| Equality deletes converted to deletion vectors (v3) or file-scoped position deletes (v2) | `convertEqualityDeletes` | |
| `DELETE ... WHERE key IN (...)` written as deletion vectors | `deleteKeys` | |
| NDV statistics (theta sketches in Puffin, as `compute_table_stats`) | `computeTableStats` | |

Every operation has a CPU twin (`CpuMergeOnReadScan`, the same algorithm on all cores) and is tested
against Iceberg's own reader.

## How fast

A grouped aggregate (`SELECT k, count(*), sum(v1..v4) GROUP BY k`) over 100M rows after
merge-on-read deletes, one RTX 4090 against the CPU twin on all 32 threads of an i9-13900K (warm
medians of 4 runs; every answer identical):

| Table | GPU | CPU (32 threads) | Speed-up | Host CPU time, GPU / CPU |
|---|---|---|---|---|
| no deletes | 256 ms | 868 ms | 3.4x | 1.3 s / 23.3 s |
| 5% position deletes | 292 ms | 900 ms | 3.1x | 1.4 s / 23.7 s |
| 5% deletion vectors | 261 ms | 862 ms | 3.3x | 1.3 s / 22.4 s |
| 1M equality deletes | 270 ms | 1152 ms | 4.3x | 1.3 s / 30.6 s |
| 10M equality deletes | 271 ms | 1582 ms | 5.8x | 1.3 s / 38.4 s |

The GPU's time hardly moves with the deletes, and it leaves the host's cores free. More results,
including Spark, Spark RAPIDS, compaction and TPC-DS, are in
[tornadovm/FINDINGS.md](tornadovm/FINDINGS.md).

## Quick start

You need an NVIDIA GPU (compute capability 7.0+) with a CUDA 13 driver, Linux x86_64 with glibc
2.34+, and [SDKMAN](https://sdkman.io).

```bash
git clone git@github.com:beestream-ai/jitIceberg.git && cd jitIceberg

sdk install java 21.0.2-open
sdk install tornadovm 8.1.0-jdk21-cuda
export JAVA_HOME=~/.sdkman/candidates/java/21.0.2-open
export TORNADOVM_HOME=~/.sdkman/candidates/tornadovm/8.1.0-jdk21-cuda
source tornadovm/cudf-env.sh        # libcudf 26.08 + CUDA 13 runtime from NVIDIA's wheels, once
tornadovm/preflight.sh              # checks JDK, TornadoVM, cuDF, GPU, driver, memory

./gradlew :iceberg-tornadovm:test   # 38 tests, every result checked against Iceberg's reader
tornadovm/run.sh gen /tmp/wh 4 1000000
tornadovm/run.sh gpu /tmp/wh q2 3 none pos05 dv05 eq1m
tornadovm/run.sh cpu /tmp/wh q2 3 none pos05 dv05 eq1m   # the CPU twin: same answers
```

From Java:

```java
try (DeviceMergeOnReadScan scan = DeviceMergeOnReadScan.of(table)) {
  AggregateResult byK = scan.aggregate("k", "v1", "v2");   // GROUP BY k on the GPU, deletes applied
  scan.convertEqualityDeletes();                           // one Iceberg commit
  scan.rewriteDataFiles();                                 // compaction, one Iceberg commit
}
```

Start the JVM with TornadoVM's argfile and `--enable-native-access=ALL-UNNAMED`; see
[tornadovm/README.md](tornadovm/README.md).

## Spark 4.2

```bash
export SPARK_HOME=/opt/spark-4.2.1-bin-hadoop3
./gradlew -DsparkVersions=4.2 :iceberg-tornadovm-spark:sparkGpuBundle
tornadovm/spark/build/spark-gpu/bin/spark-submit-gpu --driver-memory 16g \
    tornadovm/spark/examples/quickstart.py /tmp/quickstart-wh
```

```
aggregate: GPU 0.09 s (on the GPU: True), Spark 0.21 s, same answer: True
... INFO GpuFileRewriteRunnerProvider: Rewriting local.db.sales on the GPU
compaction: 0.54 s
rows after compaction: 19000000 (expected 19000000)
```

Your own jobs run unchanged under `spark-submit-gpu`. Reads, aggregates and compaction go to the GPU
when they qualify, and stay on Spark otherwise. See [tornadovm/spark/README.md](tornadovm/spark/README.md).

## What is in this repository

| Path | |
|---|---|
| [`tornadovm/`](tornadovm/README.md) | `iceberg-tornadovm`: the GPU library on Iceberg's public APIs, its CPU twin, benchmarks and scripts |
| [`tornadovm/spark/`](tornadovm/spark/README.md) | `iceberg-tornadovm-spark`: Spark 4.2 reads, aggregates and compaction on the GPU, `spark-submit-gpu`, examples |
| `spark/v4.2` | Iceberg's Spark 4.2 module with two engine-neutral extension points, `PartitionReaderFactoryProvider` and `FileRewriteRunnerProvider`, proposed upstream in [apache/iceberg#18897](https://github.com/apache/iceberg/pull/18897) |
| everything else | Apache Iceberg `main`, unchanged apart from [apache/iceberg#18631](https://github.com/apache/iceberg/pull/18631) (equality field IDs kept when delete files are copied) |

The GPU modules are part of the build only when `TORNADOVM_HOME` is set, so without it this builds
and tests as Apache Iceberg does.

## License

Apache License 2.0, as Apache Iceberg. TornadoVM is Apache 2.0; cuDF is Apache 2.0.

---

# Apache Iceberg


[![](https://github.com/apache/iceberg/actions/workflows/java-ci.yml/badge.svg)](https://github.com/apache/iceberg/actions/workflows/java-ci.yml)
[![Slack](https://img.shields.io/badge/chat-on%20Slack-brightgreen.svg)](https://apache-iceberg.slack.com/)

Iceberg is a high-performance format for huge analytic tables. Iceberg brings the reliability and simplicity of SQL tables to big data, while making it possible for engines like Spark, Trino, Flink, Presto, Hive and Impala to safely work with the same tables, at the same time.

Background and documentation is available at <https://iceberg.apache.org>


## Status

Iceberg is under active development at the Apache Software Foundation.

The [Iceberg format specification][iceberg-spec] is stable and new features are added with each version.

The core Java library is located in this repository and is the reference implementation for other libraries.

[Documentation][iceberg-docs] is available for all libraries and integrations.

[iceberg-docs]: https://iceberg.apache.org/docs/latest/
[iceberg-spec]: https://iceberg.apache.org/spec/

## Collaboration

Iceberg tracks issues in GitHub and prefers to receive contributions as pull requests.

Community discussions happen primarily on the [dev mailing list][dev-list] or on specific issues.

[dev-list]: mailto:dev@iceberg.apache.org


### Building

Iceberg is built using Gradle with Java 17 or 21.

* To invoke a build and run tests: `./gradlew build`
* To skip tests: `./gradlew build -x test -x integrationTest`
* To fix code style for default versions: `./gradlew spotlessApply`
* To fix code style for all versions of Spark/Hive/Flink:`./gradlew spotlessApply -DallModules`

Iceberg table support is organized in library modules:

* `iceberg-common` contains utility classes used in other modules
* `iceberg-api` contains the public Iceberg API
* `iceberg-core` contains implementations of the Iceberg API and support for Avro data files, **this is what processing engines should depend on**
* `iceberg-parquet` is an optional module for working with tables backed by Parquet files
* `iceberg-arrow` is an optional module for reading Parquet into Arrow memory
* `iceberg-orc` is an optional module for working with tables backed by ORC files
* `iceberg-hive-metastore` is an implementation of Iceberg tables backed by the Hive metastore Thrift client
* `iceberg-data` is an optional module for working with tables directly from JVM applications

Iceberg also has modules for adding Iceberg support to processing engines:

* `iceberg-spark` is an implementation of Spark's Datasource V2 API for Iceberg with submodules for each Spark version (use [runtime jars](https://iceberg.apache.org/multi-engine-support/#runtime-jar) for a shaded version to avoid dependency conflicts)
* `iceberg-flink` contains classes for integrating with Apache Flink (use [iceberg-flink-runtime](https://iceberg.apache.org/multi-engine-support/#runtime-jar) for a shaded version)
* `iceberg-mr` contains an InputFormat and other classes for integrating with Apache Hive

---
**NOTE**

The tests require Docker to execute. On macOS (with Docker Desktop), you might need to create a symbolic name to the docker socket in order to be detected by the tests:

```
sudo ln -s $HOME/.docker/run/docker.sock /var/run/docker.sock
```

In some cases the testcontainer may exit with an initialization error because of an illegal state exception in the GenericContainer.  One work around for this problem is to set `selinux` into permissive mode before running the tests. 

```
sudo setenforce Permissive
./gradlew ...
sudo setenforce Enforcing
```

---

#### Documentation

For information about building the documentation, see [here](site/README.md).

### Engine Compatibility

See the [Multi-Engine Support](https://iceberg.apache.org/multi-engine-support/) page to know about Iceberg compatibility with different Spark, Flink and Hive versions.
For other engines such as Presto or Trino, please visit their websites for Iceberg integration details.

### Implementations

This repository contains the Java implementation of Iceberg. Other implementations can be found at:

* **Go**: [iceberg-go](https://github.com/apache/iceberg-go)
* **PyIceberg** (Python): [iceberg-python](https://github.com/apache/iceberg-python)
* **Rust**: [iceberg-rust](https://github.com/apache/iceberg-rust)
* **C++**: [iceberg-cpp](https://github.com/apache/iceberg-cpp)
