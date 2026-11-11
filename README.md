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

![Iceberg](https://iceberg.apache.org/assets/images/Iceberg-logo.svg)

[![](https://github.com/apache/iceberg/actions/workflows/java-ci.yml/badge.svg)](https://github.com/apache/iceberg/actions/workflows/java-ci.yml)
[![Slack](https://img.shields.io/badge/chat-on%20Slack-brightgreen.svg)](https://apache-iceberg.slack.com/)

Iceberg is a high-performance format for huge analytic tables. Iceberg brings the reliability and simplicity of SQL tables to big data, while making it possible for engines like Spark, Trino, Flink, Presto, Hive and Impala to safely work with the same tables, at the same time.

Background and documentation is available at <https://iceberg.apache.org>

## Iceberg on the GPU (this branch)

This branch is Apache Iceberg's `main` plus Iceberg's per-row work on an NVIDIA GPU through
[TornadoVM](https://github.com/beehive-lab/TornadoVM) and [cuDF](https://github.com/rapidsai/cudf):
merge-on-read scans with every kind of delete, compaction, equality deletes converted to deletion
vectors, and, in Spark 4.2, reads, grouped aggregates and `rewrite_data_files` on the GPU. It adds:

| Where | What |
|---|---|
| [`tornadovm/`](tornadovm/README.md) | `iceberg-tornadovm`: the GPU library on Iceberg's public APIs, with set-up and a quick start |
| [`tornadovm/spark/`](tornadovm/spark/README.md) | `iceberg-tornadovm-spark`: Spark 4.2 on the GPU, `spark-submit-gpu` and a runnable example |
| `spark/v4.2` | two engine-neutral extension points the Spark integration plugs into (proposed to Apache Iceberg): `PartitionReaderFactoryProvider` and `FileRewriteRunnerProvider` |

In short, with an NVIDIA GPU, JDK 21 and SDKMAN:

```bash
sdk install tornadovm 8.1.0-jdk21-cuda
export TORNADOVM_HOME=~/.sdkman/candidates/tornadovm/8.1.0-jdk21-cuda
source tornadovm/cudf-env.sh          # libcudf + CUDA 13 runtime from NVIDIA's wheels
tornadovm/preflight.sh                # checks everything
./gradlew :iceberg-tornadovm:test
```


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
