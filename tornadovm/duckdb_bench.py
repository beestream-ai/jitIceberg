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
# MorBench's Q1 (count + sum) and Q2 (grouped sums) through DuckDB's Iceberg extension, on the
# benchmark tables with their deletes applied by DuckDB: a CPU engine baseline. One CSV line a run,
# with live rows and a checksum to compare against MorBench; a run past the time limit is reported
# as a timeout, a run whose process dies (DuckDB out of memory) as failed with its exit code, and in
# either case the table's remaining runs are skipped. DUCKDB_MEMORY caps DuckDB's own memory limit, so
# that it spills or fails cleanly before the operating system kills it.
#
#   python3 duckdb_bench.py <warehouse> <runs> <table>...
#       [DUCKDB_THREADS=n] [DUCKDB_TIMEOUT_S=600] [DUCKDB_MEMORY=40GB]
import glob
import multiprocessing
import os
import queue
import sys
import time

QUERIES = {
    "q1": "SELECT count(*) AS n, sum(v1) AS s FROM t",
    "q2": "SELECT k, count(*) AS n, sum(v1) AS a, sum(v2) AS b, sum(v3) AS c, sum(v4) AS d FROM t GROUP BY k",
}


def run(meta, sql, threads, out):
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{os.environ.get('DUCKDB_MEMORY', '40GB')}'")
    con.execute("LOAD iceberg")
    con.execute(f"CREATE VIEW t AS SELECT * FROM iceberg_scan('{meta}')")
    start = time.perf_counter()
    try:
        rows = con.execute(sql).fetchall()
    except duckdb.Error as e:
        out.put(("error", type(e).__name__ + ": " + str(e).splitlines()[0]))
        return
    ms = (time.perf_counter() - start) * 1000
    live = sum(r[1] if len(r) > 2 else r[0] for r in rows)
    checksum = sum(sum(float(x or 0) for x in (r[2:] if len(r) > 2 else r[1:])) for r in rows)
    out.put((ms, live, checksum))


def main():
    warehouse, runs, tables = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    threads = int(os.environ.get("DUCKDB_THREADS", os.cpu_count()))
    limit = float(os.environ.get("DUCKDB_TIMEOUT_S", 600))
    print("table,engine,query,run,warmup,ms,live_rows,checksum", flush=True)
    for table in tables:
        meta = sorted(glob.glob(os.path.join(warehouse, table, "metadata", "*.metadata.json")),
                      key=os.path.getmtime)[-1]
        for query, sql in QUERIES.items():
            for r in range(runs):
                out = multiprocessing.Queue()
                proc = multiprocessing.Process(target=run, args=(meta, sql, threads, out))
                proc.start()
                proc.join(limit)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                    print(f"{table},duckdb,{query},{r},{'yes' if r == 0 else 'no'},timeout>{limit:.0f}s,,", flush=True)
                    break
                try:
                    result = out.get(timeout=10)
                except queue.Empty:
                    result = ("died", f"exit code {proc.exitcode}")
                if isinstance(result[0], str):
                    detail = result[1].replace(",", ";").replace('"', "'")
                    print(f"{table},duckdb,{query},{r},{'yes' if r == 0 else 'no'},failed,,,\"{result[0]}: {detail}\"", flush=True)
                    break
                ms, live, checksum = result
                print(f"{table},duckdb,{query},{r},{'yes' if r == 0 else 'no'},{ms:.1f},{live},{checksum:.6e}", flush=True)


if __name__ == "__main__":
    main()
