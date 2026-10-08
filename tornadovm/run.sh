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
# Merge-on-read benchmarks: the GPU scan, the same algorithm on the CPU, Iceberg's own reader, Spark.
#
#   run.sh gen      <warehouse> <files> <rowsPerFile>
#   run.sh gpu      <warehouse> <q1|q2|q3|q4> <runs> <table>...     TornadoVM JVM, CUDA device
#   run.sh cpu      <warehouse> <q1|q2|q3|q4> <runs> <table>...     plain JVM, all cores (THREADS=n)
#   run.sh generics <warehouse> <q1|q2|q3|q4> <runs> <table>...     plain JVM, IcebergGenerics, 1 thread
#   run.sh spark    <warehouse> <q1|q2|q3|q4> <runs> <table>...     Spark local[*] + iceberg-spark-runtime
#
# Needs: TORNADOVM_HOME (a TornadoVM SDK with the cuDF shim), JAVA_HOME (JDK 21), libcudf on
# LD_LIBRARY_PATH; for spark, SPARK_HOME and ICEBERG_SPARK_JAR.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
: "${TORNADOVM_HOME:?set TORNADOVM_HOME to a TornadoVM SDK}"
: "${JAVA_HOME:?set JAVA_HOME to JDK 21}"
HEAP="${HEAP:-40g}"

classpath() {
    local file="$HERE/build/runtime-classpath.txt"
    (cd "$ROOT" && ./gradlew -q :iceberg-tornadovm:printRuntimeClasspath) > "$file.tmp"
    mv "$file.tmp" "$file"
    cat "$file"
}

mode="${1:?mode}"; shift
case "$mode" in
    gen)
        CP="$(classpath):$(ls "$TORNADOVM_HOME"/share/java/tornado/tornado-{api,cudf}-*.jar | tr '\n' ':')"
        exec "$JAVA_HOME/bin/java" -Xmx"$HEAP" --enable-preview -cp "$CP" \
            org.apache.iceberg.tornadovm.bench.GenerateTables "$@"
        ;;
    gpu)
        CP="$(classpath)"
        # The argfile puts TornadoVM on the module path; -da for TornadoVM packages, see build.gradle.
        # tornado.device.memory is TornadoVM's own device budget (4 GB by default); buffers a closed
        # plan released stay counted against it until reused by the same access type.
        # The heap stays under 32 GB: TornadoVM serializes object arguments (the path holders)
        # assuming compressed oops, which a larger heap turns off.
        exec "$JAVA_HOME/bin/java" @"$TORNADOVM_HOME/tornado-argfile" -Xmx"${GPU_HEAP:-24g}" \
            -Dtornado.device.memory="${TORNADO_DEVICE_MEMORY:-20GB}" --add-opens=java.base/java.nio=ALL-UNNAMED \
            -da:tornado... -da:uk.ac.manchester.tornado... -cp "$CP" \
            org.apache.iceberg.tornadovm.bench.MorBench "$1" gpu "${@:2}"
        ;;
    cpu|generics)
        CP="$(classpath):$(ls "$TORNADOVM_HOME"/share/java/tornado/tornado-{api,cudf}-*.jar | tr '\n' ':')"
        exec "$JAVA_HOME/bin/java" -Xmx"$HEAP" --enable-preview --add-opens=java.base/java.nio=ALL-UNNAMED -Dthreads="${THREADS:-$(nproc)}" -cp "$CP" \
            org.apache.iceberg.tornadovm.bench.MorBench "$1" "$mode" "${@:2}"
        ;;
    spark)
        : "${SPARK_HOME:?set SPARK_HOME}"; : "${ICEBERG_SPARK_JAR:?set ICEBERG_SPARK_JAR}"
        exec "$SPARK_HOME/bin/spark-submit" --master "local[${THREADS:-$(nproc)}]" \
            --driver-memory "$HEAP" --jars "$ICEBERG_SPARK_JAR" \
            --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
            --conf spark.sql.catalog.local=org.apache.iceberg.spark.SparkCatalog \
            --conf spark.sql.catalog.local.type=hadoop \
            --conf spark.sql.catalog.local.warehouse="$(dirname "$1")" \
            --conf spark.ui.enabled=false \
            "$HERE/spark_bench.py" "$@"
        ;;
    rapids)
        # Spark with the RAPIDS Accelerator on the same GPU; NOT_ON_GPU explains every operator it
        # leaves on the CPU.
        : "${SPARK_HOME:?set SPARK_HOME}"; : "${ICEBERG_SPARK_JAR:?set ICEBERG_SPARK_JAR}"; : "${RAPIDS_JAR:?set RAPIDS_JAR}"
        ENGINE=rapids exec "$SPARK_HOME/bin/spark-submit" --master "local[${THREADS:-$(nproc)}]" \
            --driver-memory "$HEAP" --jars "$ICEBERG_SPARK_JAR,$RAPIDS_JAR" \
            --conf spark.plugins=com.nvidia.spark.SQLPlugin \
            --conf spark.rapids.sql.enabled=true \
            --conf spark.rapids.sql.explain="${RAPIDS_EXPLAIN:-NOT_ON_GPU}" \
            --conf spark.rapids.sql.concurrentGpuTasks="${RAPIDS_TASKS:-2}" \
            --conf spark.rapids.memory.pinnedPool.size=4g \
            --conf spark.rapids.sql.format.iceberg.enabled=true \
            --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
            --conf spark.sql.catalog.local=org.apache.iceberg.spark.SparkCatalog \
            --conf spark.sql.catalog.local.type=hadoop \
            --conf spark.sql.catalog.local.warehouse="$(dirname "$1")" \
            --conf spark.ui.enabled=false \
            "$HERE/spark_bench.py" "$@"
        ;;
    *)
        echo "unknown mode $mode" >&2; exit 2 ;;
esac
