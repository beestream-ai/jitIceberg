#
# Licensed to the Apache Software Foundation (ASF) under the Apache License, Version 2.0.
#
# TPC-DS as Iceberg v2 merge-on-read tables maintained with EQUALITY deletes, the way a
# change-data-capture writer leaves them, holding exactly the rows tpcds_setup.py's position-delete
# maintenance leaves: rows with key % 50 = 7 deleted, rows with key % 100 = 11 updated (+1 on a
# quantity and on an amount). Two phases around the Java tool that writes the equality deletes:
#
#   BENCH_SCRIPT=tpcds_setup_eq.py run.sh spark <warehouse>/<namespace> <parquet dir> load
#   (AddKeyEqualityDeletes for every table: % 50 = 7 and % 100 = 11, five commits each)
#   BENCH_SCRIPT=tpcds_setup_eq.py run.sh spark <warehouse>/<namespace> <parquet dir> append
#
# `load` creates the tables and saves the updated rows (as they will be) under <warehouse>/_updates;
# `append` appends them after the equality deletes, in a later commit: their sequence number is
# higher, so the deletes do not apply to them.
import os
import sys

from pyspark.sql import SparkSession

warehouse, source, phase = sys.argv[1:4]
namespace = os.path.basename(warehouse.rstrip("/"))
updates = os.path.join(warehouse, "_updates")
spark = SparkSession.builder.appName(f"tpcds-setup-eq-{phase}").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
props = ("'format-version'='2','write.delete.mode'='merge-on-read','write.update.mode'='merge-on-read',"
         "'write.merge.mode'='merge-on-read','write.target-file-size-bytes'='134217728',"
         "'write.parquet.compression-codec'='zstd'")
mutations = [
    ("store_sales", "ss_ticket_number", "ss_quantity", "ss_net_paid"),
    ("catalog_sales", "cs_order_number", "cs_quantity", "cs_net_paid"),
    ("web_sales", "ws_order_number", "ws_quantity", "ws_net_paid"),
    ("store_returns", "sr_ticket_number", "sr_return_quantity", "sr_net_loss"),
    ("catalog_returns", "cr_order_number", "cr_return_quantity", "cr_net_loss"),
    ("web_returns", "wr_order_number", "wr_return_quantity", "wr_net_loss"),
    ("inventory", "inv_item_sk", "inv_quantity_on_hand", None),
]
if phase == "load":
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS local.{namespace}")
    for table in sorted(os.listdir(source)):
        spark.read.parquet(f"{source}/{table}").createOrReplaceTempView("src")
        spark.sql(f"DROP TABLE IF EXISTS local.{namespace}.{table}")
        spark.sql(f"CREATE TABLE local.{namespace}.{table} USING iceberg TBLPROPERTIES ({props}) AS SELECT * FROM src")
    for table, key, quantity, amount in mutations:
        t = spark.table(f"local.{namespace}.{table}")
        changed = [f"{quantity} + 1 AS {quantity}" if c == quantity else
                   f"{amount} + 1 AS {amount}" if c == amount else c for c in t.columns]
        t.where(f"{key} % 100 = 11").selectExpr(*changed).write.mode("overwrite").parquet(f"{updates}/{table}")
        print(f"saved updates of {table}", flush=True)
elif phase == "append":
    for table, key, quantity, amount in mutations:
        spark.read.parquet(f"{updates}/{table}").writeTo(f"local.{namespace}.{table}").append()
    for table in sorted(os.listdir(source)):
        t = f"local.{namespace}.{table}"
        n = spark.sql(f"SELECT count(*) FROM {t}").collect()[0][0]
        files = spark.sql(f"SELECT content, count(*) FROM {t}.files GROUP BY content ORDER BY content").collect()
        print(f"TPCDS {table}: {n} rows, files by content {[(r[0], r[1]) for r in files]}", flush=True)
else:
    sys.exit(f"unknown phase {phase}")
