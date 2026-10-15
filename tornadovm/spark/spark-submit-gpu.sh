#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# spark-submit with Iceberg on the GPU: reads, grouped aggregates and bin-pack compaction
# (rewrite_data_files) run on an NVIDIA GPU through TornadoVM and cuDF; everything else is Spark's.
# Local mode only. Arguments are spark-submit's; GPU_DEVICE_MEMORY (default 20GB) caps TornadoVM's
# device memory, GPU_JAVA_OPTS adds driver JVM options.
#
#   spark-submit-gpu --driver-memory 24g [--conf ...] app.py|--class ... app.jar
#
# Needs SPARK_HOME (Spark 3.5, Scala 2.12), JAVA_HOME (JDK 21), and libcudf and TornadoVM's native
# libraries on LD_LIBRARY_PATH (see the README).
set -euo pipefail
: "${SPARK_HOME:?set SPARK_HOME to a Spark 3.5 distribution}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CP=$(ls "$HERE"/jars/*.jar | paste -sd: -)
exec "$SPARK_HOME/bin/spark-submit" \
  --master "local[*]" \
  --driver-class-path "$CP" \
  --conf spark.driver.extraJavaOptions="--enable-preview @$HERE/tornado-argfile -Dtornado.device.memory=${GPU_DEVICE_MEMORY:-20GB} -da:tornado... -da:uk.ac.manchester.tornado... ${GPU_JAVA_OPTS:-}" \
  --conf spark.iceberg.gpu.enabled=true \
  --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,org.apache.spark.sql.iceberg.gpu.GpuAggregateExtensions \
  "$@"
