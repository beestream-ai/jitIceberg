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
# The upsert stream of CdcBench through Spark's own MERGE, with the table in copy-on-write or in
# merge-on-read mode: each round upserts <upserts> existing ids with new values, then runs Q2. One CSV
# line a round: mode, round, merge ms, Q2 ms, live rows, data files, delete files.
#
#   BENCH_SCRIPT=spark_cdc.py run.sh spark <warehouse>/<namespace> <base table> <rounds> <upserts> <cow|mor>...
import os
import sys
import time

from pyspark.sql import SparkSession

warehouse, base, rounds, upserts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
modes = sys.argv[5:] or ["cow", "mor"]
namespace = os.path.basename(warehouse.rstrip("/"))
spark = SparkSession.builder.appName("spark-cdc").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
source = f"local.{namespace}.{base}"
rows = spark.sql(f"SELECT count(*) FROM {source}").collect()[0][0]
print("mode,round,merge_ms,q2_ms,live_rows,data_files,delete_files", flush=True)

for mode in modes:
    table = f"local.{namespace}.cdc_{mode}"
    spark.sql(f"DROP TABLE IF EXISTS {table} PURGE")
    write_mode = "copy-on-write" if mode == "cow" else "merge-on-read"
    spark.sql(
        f"CREATE TABLE {table} USING iceberg TBLPROPERTIES ('format-version'='3', "
        f"'write.merge.mode'='{write_mode}', 'write.update.mode'='{write_mode}', "
        f"'write.delete.mode'='{write_mode}') AS SELECT * FROM {source}")
    for r in range(1, rounds + 1):
        # The same ids in both modes: a seeded sample of existing ids, new values. MERGE needs a
        # deterministic source, so the new values are hashes of (id, round), not rand().
        values = ", ".join(f"pmod(hash(id, {r}, {j}), 1000000) / 1000000.0 AS v{j}" for j in range(1, 5))
        spark.sql(
            f"SELECT id, k, {values} FROM {source} WHERE pmod(hash(id, {r}), {rows}) < {upserts}"
        ).createOrReplaceTempView("updates")
        start = time.perf_counter()
        spark.sql(f"MERGE INTO {table} t USING updates s ON t.id = s.id "
                  "WHEN MATCHED THEN UPDATE SET *")
        merge_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        result = spark.sql(f"SELECT k, count(*) c, sum(v1), sum(v2), sum(v3), sum(v4) FROM {table} GROUP BY k").collect()
        q2_ms = (time.perf_counter() - start) * 1000
        live = sum(row["c"] for row in result)
        files = spark.sql(f"SELECT content, count(*) FROM {table}.files GROUP BY content").collect()
        data_files = sum(row[1] for row in files if row[0] == 0)
        delete_files = sum(row[1] for row in files if row[0] != 0)
        print(f"{mode},{r},{merge_ms:.1f},{q2_ms:.1f},{live},{data_files},{delete_files}", flush=True)
    spark.sql(f"DROP TABLE IF EXISTS {table} PURGE")
