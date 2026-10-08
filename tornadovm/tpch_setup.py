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
# TPC-H lineitem as Iceberg v2 merge-on-read tables, from DuckDB's dbgen Parquet (decimals cast to
# double): lineitem (no deletes), lineitem_pos (Spark DELETE in merge-on-read mode: position
# deletes) and lineitem_eq (the same rows, for equality deletes on l_orderkey, which Spark cannot
# write; the keys go to a text file for the Java writer). The deleted orders, 1% of them, stand in
# for ten runs of TPC-H's RF2 refresh.
#
#   spark-submit ... tpch_setup.py <warehouse> <namespace> <lineitem parquet dir> <keys file>
import sys

from pyspark.sql import SparkSession

warehouse, namespace, source, keys_file = sys.argv[1:5]
spark = SparkSession.builder.appName("tpch-setup").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
spark.read.parquet(source).createOrReplaceTempView("src")
props = ("'format-version'='2','write.delete.mode'='merge-on-read','write.update.mode'='merge-on-read',"
         "'write.target-file-size-bytes'='134217728','write.parquet.compression-codec'='zstd'")
for table in ["lineitem", "lineitem_pos", "lineitem_eq"]:
    spark.sql(f"DROP TABLE IF EXISTS local.{namespace}.{table}")
    spark.sql(f"CREATE TABLE local.{namespace}.{table} USING iceberg TBLPROPERTIES ({props}) "
              f"AS SELECT * FROM src ORDER BY l_orderkey")
spark.sql(f"DELETE FROM local.{namespace}.lineitem_pos WHERE l_orderkey % 100 = 7")
keys = [r[0] for r in spark.sql(
    f"SELECT DISTINCT l_orderkey FROM local.{namespace}.lineitem WHERE l_orderkey % 100 = 7 ORDER BY 1").collect()]
with open(keys_file, "w") as f:
    f.write("\n".join(str(k) for k in keys))
for table in ["lineitem", "lineitem_pos", "lineitem_eq"]:
    n = spark.sql(f"SELECT count(*) FROM local.{namespace}.{table}").collect()[0][0]
    files = spark.sql(f"SELECT count(*) FROM local.{namespace}.{table}.files").collect()[0][0]
    print(f"TPCH {table}: {n} rows, {files} files", flush=True)
print(f"TPCH keys: {len(keys)} orders", flush=True)
