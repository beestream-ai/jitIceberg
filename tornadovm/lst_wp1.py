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
# A longevity run in the style of LST-Bench's WP1, on the TPC-DS tables tpcds_setup.py made: rounds
# of data maintenance -- Spark merge-on-read DELETE and UPDATE of every sales and returns table,
# keyed so that each round touches every data file -- each followed by a single-user run of a query
# set, on Spark and on Spark + GPU (reads and aggregates) in the same JVM. Then a compaction
# (rewrite_data_files, bin-pack) on Spark's runner and on the GPU's, from the same state, and the
# query set once more. Every query's result is checked against Spark's in the same round.
#
#   BENCH_SCRIPT=lst_wp1.py run.sh spark-gpu <warehouse>/<namespace> <query dir> <rounds> <q>...
import hashlib
import os
import statistics
import sys
import time
from decimal import Decimal

from pyspark.sql import SparkSession

warehouse, query_dir, rounds, queries = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
namespace = os.path.basename(warehouse.rstrip("/"))
spark = SparkSession.builder.appName("lst-wp1").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.sql(f"USE local.{namespace}")
jvm = spark.sparkContext._jvm
env_conf = jvm.org.apache.spark.SparkEnv.get().conf()
FACTS = [("store_sales", "ss_ticket_number", "ss_quantity", "ss_net_paid"),
         ("catalog_sales", "cs_order_number", "cs_quantity", "cs_net_paid"),
         ("web_sales", "ws_order_number", "ws_quantity", "ws_net_paid"),
         ("store_returns", "sr_ticket_number", "sr_return_quantity", "sr_net_loss"),
         ("catalog_returns", "cr_order_number", "cr_return_quantity", "cr_net_loss"),
         ("web_returns", "wr_order_number", "wr_return_quantity", "wr_net_loss")]
sql = {q: open(f"{query_dir}/{q}.sql").read().strip().rstrip(";") for q in queries}


def gpu(on):
    value = "true" if on else "false"
    env_conf.set("spark.iceberg.gpu.enabled", value)
    spark.conf.set("spark.iceberg.gpu.enabled", value)


def checksum(rows):
    def canonical(v):
        return f"{float(v):.6g}" if isinstance(v, (float, Decimal)) else str(v)
    lines = sorted("|".join(canonical(v) for v in row) for row in rows)
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]


def single_user(label):
    """The query set on each engine: per query the median of 3 after a warm-up."""
    totals = {}
    answers = {}
    for engine in ("spark", "gpu"):
        gpu(engine == "gpu")
        total = 0.0
        for q in queries:
            spark.sql(sql[q]).collect()
            times = []
            for _ in range(3):
                start = time.perf_counter()
                rows = spark.sql(sql[q]).collect()
                times.append((time.perf_counter() - start) * 1000)
            total += statistics.median(times)
            answers.setdefault(q, []).append(checksum(rows))
        totals[engine] = total
    wrong = [q for q, a in answers.items() if len(set(a)) != 1]
    deletes = sum(spark.sql(f"SELECT count(*) FROM local.{namespace}.{t}.files WHERE content > 0")
                  .collect()[0][0]
                  for t, *_ in FACTS)
    print(f"{label},{totals['spark']:.0f},{totals['gpu']:.0f},{deletes},{'ok' if not wrong else wrong}",
          flush=True)


def maintenance(r):
    gpu(False)
    start = time.perf_counter()
    for table, key, quantity, amount in FACTS:
        spark.sql(f"DELETE FROM {table} WHERE {key} % 100 = {(13 * r) % 100}")
        spark.sql(f"UPDATE {table} SET {quantity} = {quantity} + 1, {amount} = {amount} + 1 "
                  f"WHERE {key} % 100 = {(13 * r + 50) % 100}")
    return (time.perf_counter() - start) * 1000


print("phase,spark_ms,spark_gpu_ms,fact_delete_files,answers", flush=True)
single_user("round0")
for r in range(1, rounds + 1):
    dm = maintenance(r)
    print(f"dm{r},{dm:.0f},,,", flush=True)
    single_user(f"round{r}")

# Compaction from the same state on each runner; the GPU's result is the one kept.
snapshots = {}
for table, *_ in FACTS:
    t = jvm.org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark._jsparkSession, table)
    snapshots[table] = (t, t.currentSnapshot().snapshotId())
times = {}
for engine in ("spark", "gpu"):
    total = 0.0
    for table, (t, snapshot) in snapshots.items():
        t.refresh()
        t.manageSnapshots().rollbackTo(snapshot).commit()
        t.refresh()
        gpu(engine == "gpu")
        start = time.perf_counter()
        (jvm.org.apache.iceberg.spark.actions.SparkActions.get(spark._jsparkSession)
         .rewriteDataFiles(t).option("rewrite-all", "true")
         .option("remove-dangling-deletes", "true").execute())
        total += (time.perf_counter() - start) * 1000
        if engine == "spark":
            # Spark's compaction is undone for the GPU's run; its files go with its snapshot.
            t.refresh()
            rewritten = t.currentSnapshot().snapshotId()
            t.manageSnapshots().rollbackTo(snapshot).commit()
            t.refresh()
            t.expireSnapshots().expireSnapshotId(rewritten).commit()
    times[engine] = total
print(f"compaction,{times['spark']:.0f},{times['gpu']:.0f},,", flush=True)
single_user("compacted")
