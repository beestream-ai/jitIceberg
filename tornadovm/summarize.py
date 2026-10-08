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
# Summarizes MorBench / spark_bench.py CSVs: the median of the timed runs (warm-up excluded) per
# table, engine and query, the GPU speed-up over each other engine, and whether every engine
# returned the same answer (live rows exactly, checksum to a relative 1e-9).
#
#   python3 summarize.py <results-dir>     prints Markdown tables
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

TABLES = ["none", "pos01", "pos05", "pos20", "dv05", "eq10k", "eq1m", "eq10m", "mixed"]
ENGINES = ["gpu", "cpu", "spark", "generics"]
LABEL = {
    "none": "no deletes",
    "pos01": "1% position deletes (v2)",
    "pos05": "5% position deletes (v2)",
    "pos20": "20% position deletes (v2)",
    "dv05": "5% deletion vectors (v3)",
    "eq10k": "10k equality deletes",
    "eq1m": "1M equality deletes",
    "eq10m": "10M equality deletes",
    "mixed": "5% position + 1M equality",
}

runs = defaultdict(list)
answers = defaultdict(set)
failed = set()
for path in glob.glob(os.path.join(sys.argv[1], "*-q*.csv")):
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["table"], row["engine"], row["query"])
            if row["run"] == "failed":
                failed.add(key)
                continue
            if not row.get("millis"):
                continue
            if row["warmup"] == "no":
                runs[key].append(float(row["millis"]))
            answers[(row["table"], row["query"])].add((row["engine"], row["live_rows"], row["checksum"]))


def agree(table, query):
    seen = answers[(table, query)]
    rows = {r for _, r, _ in seen}
    sums = [float(c) for _, _, c in seen if c not in ("", "None")]
    if len(rows) != 1:
        return "rows differ: " + ", ".join(sorted(f"{e}={r}" for e, r, _ in seen))
    if sums and (max(sums) - min(sums)) > 1e-9 * max(abs(max(sums)), 1.0):
        return "checksums differ"
    return "yes"


for query, title in [
    ("q1", "Q1 `SELECT count(*), sum(v1)`"),
    ("q2", "Q2 `SELECT k, count(*), sum(v1..v4) GROUP BY k` (1000 groups)"),
    ("q3", "Q3 `SELECT id, k, v1` -- every live row materialized on the host"),
    ("q4", "Q4 `SELECT bucket(16, id), count(*), sum(v1) GROUP BY 1` -- Iceberg's bucket transform"),
    ("q5", "Q5 `SELECT k, count(*), sum(v1) WHERE id < 25000000 AND v2 > 0.5 GROUP BY k` -- file pruning + residual filter"),
]:
    present = [e for e in ENGINES if any((t, e, query) in runs for t in TABLES)]
    if not present:
        continue
    print(f"\n### {title}\n")
    header = "| table | live rows | " + " | ".join(f"{e} (ms)" for e in present)
    header += " | " + " | ".join(f"GPU vs {e}" for e in present if e != "gpu") + " | same answer |"
    print(header)
    print("|" + "---|" * (2 + len(present) + len([e for e in present if e != "gpu"]) + 1))
    for table in TABLES:
        medians = {e: statistics.median(runs[(table, e, query)]) for e in present if runs.get((table, e, query))}
        if "gpu" not in medians:
            continue
        rows = next(iter(r for _, r, _ in answers[(table, query)]))
        cells = [LABEL[table], f"{int(rows):,}"]
        def cell(e):
            if e in medians:
                return f"{medians[e]:,.0f}"
            return "failed (OOM)" if (table, e, query) in failed else ""

        cells += [cell(e) for e in present]
        cells += [f"{medians[e] / medians['gpu']:.1f}x" if e in medians else "" for e in present if e != "gpu"]
        cells.append(agree(table, query))
        print("| " + " | ".join(cells) + " |")
