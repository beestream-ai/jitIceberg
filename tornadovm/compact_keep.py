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
# Compacts each table once on the GPU's runner and KEEPS the result (compact_bench.py rolls back):
# the layout experiment rewrites tables with a chosen codec and row-group size
# (-Diceberg.tornadovm.write-codec, -Diceberg.tornadovm.write-row-group-rows in the driver JVM) and
# then reads the rewritten tables. Prints the rewrite's time and the new files' count and bytes.
#   BENCH_SCRIPT=compact_keep.py run.sh spark-gpu <warehouse>/<namespace> <table>...
import os
import sys
import time

from pyspark.sql import SparkSession

warehouse, tables = sys.argv[1], sys.argv[2:]
namespace = os.path.basename(warehouse.rstrip("/"))
spark = SparkSession.builder.appName("compact-keep").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
jvm = spark.sparkContext._jvm
print("table,millis,rewritten,added,rows,data_files,data_bytes,delete_files", flush=True)
for t in tables:
    name = f"local.{namespace}.{t}"
    table = jvm.org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark._jsparkSession, name)
    begin = time.perf_counter()
    result = (jvm.org.apache.iceberg.spark.actions.SparkActions.get(spark._jsparkSession)
              .rewriteDataFiles(table).option("rewrite-all", "true").execute())
    millis = (time.perf_counter() - begin) * 1000
    stats = spark.sql(f"SELECT count(*), sum(file_size_in_bytes), sum(record_count) FROM {name}.data_files").collect()[0]
    deletes = spark.sql(f"SELECT count(*) FROM {name}.delete_files").collect()[0][0]
    print(f"{t},{millis:.0f},{result.rewrittenDataFilesCount()},{result.addedDataFilesCount()},"
          f"{stats[2]},{stats[0]},{stats[1]},{deletes}", flush=True)
