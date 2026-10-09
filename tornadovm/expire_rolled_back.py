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
# Removes what benchmark runs left behind: snapshots that are not ancestors of a table's current
# snapshot -- a compaction that was rolled back -- and the files only they reference, through
# Iceberg's own snapshot expiry. The current state and its whole history are untouched.
#
#   BENCH_SCRIPT=expire_rolled_back.py run.sh spark <warehouse>/<namespace>
import os
import sys

from pyspark.sql import SparkSession

warehouse = sys.argv[1]
namespace = os.path.basename(warehouse.rstrip("/"))
spark = SparkSession.builder.appName("expire-rolled-back").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
jvm = spark.sparkContext._jvm
for row in spark.sql(f"SHOW TABLES IN local.{namespace}").collect():
    name = f"local.{namespace}.{row.tableName}"
    table = jvm.org.apache.iceberg.spark.Spark3Util.loadIcebergTable(spark._jsparkSession, name)
    current = table.currentSnapshot()
    if current is None:
        continue
    lineage = set()
    snapshot = current
    while snapshot is not None:
        lineage.add(snapshot.snapshotId())
        parent = snapshot.parentId()
        snapshot = table.snapshot(parent) if parent is not None else None
    stray = [s.snapshotId() for s in table.snapshots() if s.snapshotId() not in lineage]
    if stray:
        expire = table.expireSnapshots()
        for sid in stray:
            expire = expire.expireSnapshotId(sid)
        expire.commit()
    print(f"{name}: expired {len(stray)} rolled-back snapshots", flush=True)
