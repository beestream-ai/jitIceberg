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
# The Star Schema Benchmark as Iceberg tables, from the .tbl files of SSB's dbgen (the copy in
# Crystal's repository, so both systems read the same rows). Every numeric column is an integer, as
# in Crystal's columnar loader. <mode> picks lineorder's maintenance:
#   plain  no deletes (the comparison with Crystal, which runs on device-resident columns)
#   dv     format v3, rows with lo_orderkey % 100 = 7 deleted by Spark's merge-on-read DELETE
#          (deletion vectors); the dimension tables are the plain ones
#   eq     format v2, the same rows' keys left for AddKeyEqualityDeletes (run afterwards)
#
#   run.sh spark <warehouse>/<namespace> <tbl dir> <mode>     (BENCH_SCRIPT=ssb_setup.py)
import os
import sys

from pyspark.sql import SparkSession

warehouse, source, mode = sys.argv[1:4]
namespace = os.path.basename(warehouse.rstrip("/"))
spark = SparkSession.builder.appName(f"ssb-setup-{mode}").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
version = "3" if mode == "dv" else "2"
props = (f"'format-version'='{version}','write.delete.mode'='merge-on-read',"
         "'write.update.mode'='merge-on-read','write.target-file-size-bytes'='134217728',"
         "'write.parquet.compression-codec'='zstd'")
tables = {
    "lineorder": ("lo_orderkey int, lo_linenumber int, lo_custkey int, lo_partkey int, lo_suppkey int, "
                  "lo_orderdate int, lo_orderpriority string, lo_shippriority string, lo_quantity int, "
                  "lo_extendedprice int, lo_ordtotalprice int, lo_discount int, lo_revenue int, "
                  "lo_supplycost int, lo_tax int, lo_commitdate int, lo_shipmode string", "lo_orderdate"),
    "part": ("p_partkey int, p_name string, p_mfgr string, p_category string, p_brand1 string, "
             "p_color string, p_type string, p_size int, p_container string", "p_partkey"),
    "supplier": ("s_suppkey int, s_name string, s_address string, s_city string, s_nation string, "
                 "s_region string, s_phone string", "s_suppkey"),
    "customer": ("c_custkey int, c_name string, c_address string, c_city string, c_nation string, "
                 "c_region string, c_phone string, c_mktsegment string", "c_custkey"),
    "ddate": ("d_datekey int, d_date string, d_dayofweek string, d_month string, d_year int, "
              "d_yearmonthnum int, d_yearmonth string, d_daynuminweek int, d_daynuminmonth int, "
              "d_daynuminyear int, d_monthnuminyear int, d_weeknuminyear int, d_sellingseason string, "
              "d_lastdayinweekfl int, d_lastdayinmonthfl int, d_holidayfl int, d_weekdayfl int", "d_datekey"),
}
files = {"ddate": "date"}
spark.sql(f"CREATE NAMESPACE IF NOT EXISTS local.{namespace}")
for table, (schema, order) in tables.items():
    # dbgen ends every line with the separator: one empty trailing column, dropped.
    frame = (spark.read.option("sep", "|").schema(schema + ", _end string")
             .csv(os.path.join(source, files.get(table, table) + ".tbl")).drop("_end"))
    frame.createOrReplaceTempView("src")
    spark.sql(f"DROP TABLE IF EXISTS local.{namespace}.{table}")
    spark.sql(f"CREATE TABLE local.{namespace}.{table} USING iceberg TBLPROPERTIES ({props}) "
              f"AS SELECT * FROM src ORDER BY {order}")
if mode == "dv":
    spark.sql(f"DELETE FROM local.{namespace}.lineorder WHERE lo_orderkey % 100 = 7")
for table in tables:
    n = spark.sql(f"SELECT count(*) FROM local.{namespace}.{table}").collect()[0][0]
    count = spark.sql(f"SELECT count(*) FROM local.{namespace}.{table}.files").collect()[0][0]
    deletes = spark.sql(f"SELECT count(*) FROM local.{namespace}.{table}.delete_files").collect()[0][0]
    print(f"SSB {namespace}.{table}: {n} rows, {count} data files, {deletes} delete files", flush=True)
