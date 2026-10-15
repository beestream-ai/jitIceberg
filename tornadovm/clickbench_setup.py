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
# ClickBench's hits table (the Parquet file ClickBench's Spark entry reads, ~100M rows) as an Iceberg
# table `hits`, in one of three namespaces:
#   plain  as loaded
#   dv     format v3, the rows with WatchID % 100 = 7 deleted by Spark's merge-on-read DELETE
#          (deletion vectors)
#   eq     format v2, the same rows' WatchIDs written to <warehouse>/_keys/hits.txt for
#          AddEqualityDeletes (equality deletes)
#
#   BENCH_SCRIPT=clickbench_setup.py run.sh spark <warehouse>/<namespace> <hits.parquet> <mode>
import os
import sys

from pyspark.sql import SparkSession

warehouse, source, mode = sys.argv[1:4]
namespace = os.path.basename(warehouse.rstrip("/"))
spark = SparkSession.builder.appName(f"clickbench-setup-{mode}").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
version = "3" if mode == "dv" else "2"
props = (f"'format-version'='{version}','write.delete.mode'='merge-on-read',"
         "'write.update.mode'='merge-on-read','write.target-file-size-bytes'='134217728',"
         "'write.parquet.compression-codec'='zstd'")
spark.sql(f"CREATE NAMESPACE IF NOT EXISTS local.{namespace}")
spark.read.parquet(source).createOrReplaceTempView("src")
table = f"local.{namespace}.hits"
spark.sql(f"DROP TABLE IF EXISTS {table}")
# Ordered by CounterID and EventDate, as ClickBench's own schema clusters the table.
spark.sql(f"CREATE TABLE {table} USING iceberg TBLPROPERTIES ({props}) AS SELECT * FROM src ORDER BY CounterID, EventDate")
if mode == "dv":
    spark.sql(f"DELETE FROM {table} WHERE WatchID % 100 = 7")
elif mode == "eq":
    os.makedirs(os.path.join(warehouse, "_keys"), exist_ok=True)
    keys = spark.sql(f"SELECT WatchID FROM {table} WHERE WatchID % 100 = 7")
    with open(os.path.join(warehouse, "_keys", "hits.txt"), "w") as f:
        for row in keys.toLocalIterator():
            f.write(f"{row[0]}\n")
n = spark.sql(f"SELECT count(*) FROM {table}").collect()[0][0]
files = spark.sql(f"SELECT count(*) FROM {table}.files").collect()[0][0]
print(f"CLICKBENCH {namespace}.hits: {n} rows, {files} files", flush=True)
