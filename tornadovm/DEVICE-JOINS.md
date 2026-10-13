# Device joins: design

Status: steps 1-4 done (prototype, device joins, Spark strategy, TPC-DS run); step 5 (cost decision) and wider coverage open. Private (fork `feat/tornadovm-merge-on-read`). Measurements below are
from this machine (RTX 4090, i9-13900K, `local[32]`) on TPC-DS SF10 after merge-on-read maintenance.

## Why

After v0.2 the GPU path is at parity with Spark on TPC-DS (1.01x geomean). Reads only pay with
equality deletes, and aggregates only where the aggregate sits directly on one Iceberg table. On
TPC-DS almost none do: when the aggregate strategy logged its decisions, nearly every one of the
102 queries was declined as "not a filtered scan", because the aggregate sits on joins.

Joins are where Spark RAPIDS earns its 1.3x total on the same tables (98.5 s against Spark's
128.7 s). Fifteen queries carry 33 of the 41 s it saves:

| Query | Spark | RAPIDS | Saved | Plan shape (static, AQE off) |
|---|---|---|---|---|
| q72 | 11.6 s | 5.5 s | 6.0 s | catalog_sales ⋈ inventory (SMJ) + 7 broadcast dims, 2 left outer |
| q67 | 5.6 s | 1.1 s | 4.5 s | store_sales ⋈ date, store, item (3 broadcast) → ROLLUP → windows |
| q4 | 6.4 s | 3.3 s | 3.0 s | three channels × customer, 11 SMJ |
| q22 | 3.8 s | 0.8 s | 3.0 s | inventory ⋈ date, item, warehouse (3 broadcast) → ROLLUP |
| q64 | 6.3 s | 3.5 s | 2.8 s | 13 tables, 15 SMJ + 20 broadcast |
| q23a/b | 4.4/4.6 s | 2.1/2.4 s | 4.4 s | semi-joins (IN subqueries) across channels |
| q80 | 3.6 s | 1.9 s | 1.6 s | three channels, each a star with a left outer to its returns, ROLLUP |
| q95, q25, q29, q93, q17, q78, q11 | 1.0-3.0 s each | | 0.9-1.4 s each | sales ⋈ returns ⋈ sales (SMJ), semi-joins |

Two families:

- **A. Star aggregates**: one fact table, inner joins to small dimension tables (Spark broadcasts
  them), then an aggregate, often under ROLLUP (`Expand`) and windows. q67, q22 purely; q80 and
  q72 for most of their work.
- **B. Fact-to-fact joins**: two large tables joined (sort-merge in Spark), semi-joins from `IN`
  subqueries. q4, q11, q64, q23, q78, q17/q25/q29, q93, q95.

Family A is the first target: the largest single wins (q67 5.2x, q22 4.8x on RAPIDS), the least
device memory, and a shape the device does well without a general join engine.

## Phase 1: star aggregates

### The idea: dimensions as arrays indexed by key

In a star schema a dimension's surrogate key is a dense integer (TPC-DS: 1..N, `date_dim` from
2415022). A join of the fact table to a dimension is then not a hash join but a gather:

- a **filter** on the dimension (`d_year = 2001`, `i_category IN (...)`) becomes a byte mask
  indexed by key, computed once from the (small) dimension;
- an **attribute** the query groups by or computes with (`i_brand`, `d_moy`) becomes an array of
  dictionary codes (strings) or values (numbers) indexed by key.

Per fact row the device reads the foreign key, looks up `mask[fk]` (an inner join drops a row whose
key is null, absent or filtered out) and gathers `code[fk]`. No hash table, no build phase on the
device, and no strings on the device: group keys are int codes, decoded on the host after the
aggregation (the existing 8-byte string limit does not apply).

When a dimension's keys are not dense (the range is much larger than the row count) or not
integers, the dimension falls back to a hash probe through cuDF (`Cudf.containedIn` for the mask,
`Cudf.innerJoin` for the gather map). That path is phase 2; phase 1 declines such plans.

### Execution, per batch of fact files

1. Read the fact columns the query needs (foreign keys, measures, the fact's own filter columns)
   with the existing device scan: deletes applied, the fact's pushed filters on the device.
2. For each dimension: `live &= mask_d[fk_d]`, and gather the attribute codes the aggregate groups
   by (TornadoVM kernels; the masks and code arrays are uploaded once per query, they are small:
   `item` is 102K keys at SF10, `date_dim` 73K, `customer` 500K).
3. Group by the codes (and fact columns) with `Cudf.groupAggregateLongs`: SUM and COUNT, AVG as
   SUM and COUNT, MIN and MAX as separate calls (decomposable aggregates only).
4. Return the groups. The host decodes the codes into the dimension values.

The result is a **partial aggregate at the finest grouping the query needs**. Spark finishes it:
ROLLUP's `Expand` and the final aggregate, windows, ORDER BY / LIMIT. All of these then run over the
groups (thousands to a few million rows) instead of the joined fact rows.

### Spark integration

As for the aggregate strategy: a planner strategy injected through `spark.sql.extensions`. Spark
itself is not modified.

- **Match**: an `Aggregate` (with or without an `Expand` below it) over projections and filters,
  over a tree of inner joins in which:
  - exactly one leaf is an Iceberg scan, the fact table, and every other leaf is an Iceberg scan
    under the broadcast threshold (a dimension);
  - each join condition is an equality between a fact column and a dimension's key column;
  - the aggregates are decomposable (SUM, COUNT, AVG, MIN, MAX; no DISTINCT).
- **Plan**: a leaf `GpuStarAggregateExec` that outputs the partial aggregate's rows (grouping
  columns and aggregate buffers), with Spark's final aggregation (and `Expand`, windows) above it.
- **Decline** (Spark plans as usual) for outer joins, non-equi conditions, a dimension too large or
  with sparse keys, more group keys than the device groups by, or an aggregate the device does not
  compute. A device failure at run time re-plans on the CPU, as the aggregate strategy does today.

The dimension masks and code arrays are built on the host by Spark itself (a small query on the
dimension, with its filters), so dimension filters of any complexity (LIKE, IN lists, expressions)
are evaluated by Spark, not the device.

### What phase 1 covers, and the expected effect

The plan-shape scan (`tornadovm/plan_shapes.py`, static plans of all 103 queries) finds
**20 pure single-fact star aggregates**: one fact table, only inner broadcast joins, an aggregate.
q3 q12 q19 q20 q21 q22 q36 q42 q43 q52 q53 q55 q62 q63 q67 q86 q89 q96 q98 q99. Together they take
14.4 s on Spark and 8.6 s on RAPIDS (11% of the suite's time). The other 81 queries that read a
fact table (114.3 s) need phase 2 for all or part of their work.

Most of the 20 are short (150-400 ms on Spark). The read policy showed that the device's fixed cost
per scan (~100 ms and up) loses on such queries, so phase 1's wins concentrate where the fact scan is
large and the post-join work is heavy: q67 (5.6 s on Spark, 1.1 s on RAPIDS) and q22 (3.8 s, 0.8 s).
q22 reads 4 columns of `inventory` (133M rows at SF10), and its work after the scan is gathers and
one group-by. Its device time should be comparable to the compaction scans (a few hundred ms). The
prototype checks that before anything is built into Spark, and step 5's cost decision keeps the
short star queries on Spark unless they too win.

### Plan of work

1. Prototype on q22 through the Java API (`DeviceMergeOnReadScan`), no Spark: masks and code arrays
   from `date_dim`, `item`, `warehouse`, the gather kernel, the group-by; compare groups with
   Spark's partial aggregate; measure device time. Go/no-go on that number.
   **Result (2026-10-13, `bench/StarQ22.java`, SF10, 5 runs):** the 100 rows equal Spark's q22
   answer (names exact, averages within 1e-3). 25.8M fact rows after the date and warehouse
   filters, 49,640 groups at (item); warm runs: dimensions 51-68 ms, device scan + group-by
   826-911 ms, host ROLLUP and top 100 88-132 ms, **967-1111 ms in total** (cold 3.1 s) against
   Spark's 3.8 s and RAPIDS' 0.8 s. The device part is above the few hundred ms guessed above, so
   the next steps should look at where it goes (the prototype filters with an `IN` list and groups
   by `inv_item_sk` through the existing aggregate, without the mask/gather kernels). Go.

   **Where the time goes** (`scan.metrics()` plus temporary probes): of the ~0.8 s scan, ~420 ms
   is building the device regions -- the host buffers of four streams sized to 8M-row batches,
   zero-filled by `Arena.allocate` -- because each new `DeviceMergeOnReadScan` (one per filter)
   builds its own. File planning is 22 ms. With one scan kept across runs
   (`-Dstarq22.reuse=true`) the device part is **255-262 ms** and the whole query **0.40-0.41 s**:
   9x Spark, 2x RAPIDS. Design consequence for step 2: the dimension masks must be run-time
   inputs of a scan keyed by the fact table's snapshot (and its own filter), not part of the scan's
   filter, so one set of regions serves every star query on that fact table. (One run of five
   rebuilt its regions once without a visible cause; a 6-run repeat with a probe did not.)
2. The kernels and the plan in `iceberg-tornadovm` (engine-neutral, like the scan).

   **Done (2026-10-13):** `DeviceDimension` (a mask and attribute codes over a dense key range),
   `DeviceMergeOnReadScan.starAggregate(dimensions, keys, values)` with `DeviceDimension.Key` a
   fact column or a dimension attribute, and two kernels in the grouped pipeline: `starFilter`
   (after the delete and filter kernels: `live &= mask[fk - first]`, null and out-of-range keys
   dropped) and `starKeys` (after `groupKeys`: a gathered key's foreign key becomes the
   attribute's code). The masks and codes are uploaded once a run, not once a batch. The regions
   are keyed by the star's shape (join columns, keys, room rounded to a power of two), so a scan of
   the fact table serves every query on it. `TestDeviceMergeOnRead.testStarAggregate` checks
   against IcebergGenerics under position and equality deletes, with two dimensions on one column,
   a range that drops keys, and dimensions changed on the same scan. q22 with
   `-Dstarq22.star=true` (all three joins on the device, one unfiltered inventory scan): 100/100
   rows equal Spark's, **device 246-260 ms, query 0.40-0.47 s warm** (Spark 3.8 s, RAPIDS 0.8 s).
3. The Spark strategy and `GpuStarAggregateExec`; tests comparing answers with the strategy on and
   off, as `TestSparkGpuAggregates` does.

   **Done (2026-10-13):** `StarAggregates` (Scala) matches an `Aggregate` (optionally over an
   `Expand`) over projections/filters over inner joins whose leaves are Iceberg scans: the largest
   is the fact table, every other leaf at most `spark.iceberg.gpu.aggregate.star.maxDimensionRows`
   rows (10M) and joined by one equality of an integer fact column with its key. At run time Spark
   runs the dimension plans (concurrently), the driver builds masks and one code per key for the
   tuple of attributes the query uses (one device key per dimension, at most four), the device
   joins and groups, and the driver applies the `Expand` and group expressions and merges the
   partial sums (result expressions compiled once per run). Dimensions with repeated or sparse keys
   fail the run, which falls back to the CPU. Opt-in: `spark.iceberg.gpu.aggregate.star=true`.
   `TestSparkGpuAggregates.testStarAggregates`: q3 and q22 shapes, three joins, CUBE, a global
   aggregate, three declines/fallbacks.

   Two device findings on the way: (a) cuDF's hash group-by is slow when most rows are dropped by
   the joins (one hot "dead" key next to many live groups: 176 ms for a q63 shape against 116 ms
   with every row live), so a star whose keys are all dimension codes with at most 2^20 combined
   groups aggregates densely (`denseClear`/`denseAggregate`, atomic adds, dropped rows cost one
   load): 65 ms. (b) Every query shape had its own batch-sized regions, never freed: seven star
   queries on `store_sales` ran out of direct memory (25 GB). Aggregate shapes are now an LRU of
   `iceberg.tornadovm.aggregateShapes` (3) region sets JVM-wide.

4. TPC-DS with the strategy on, per query against v0.2 and RAPIDS; the plan-shape scan says which
   queries it reached.

   **Done (2026-10-13), 20 star queries, SF10, median of 3, both arms the same day**
   (`results-star/`, `report/star.py` in the results repo): star off 15.10 s, on **10.91 s**,
   geomean **1.37x** (1.56x over the 15 taken), all 20 answers equal. Taken: q3 1.69x, q12 1.01x,
   q20 1.20x, q21 1.01x, q22 **5.76x** (4.0 to 0.69 s), q36 1.91x, q42 1.70x, q43 1.81x, q52 1.68x,
   q53 1.69x, q55 1.66x, q63 1.63x, q86 0.93x, q89 1.37x, q98 1.13x. Left to Spark: q67 (see
   below), q62/q99 (CASE on fact columns), q96 (COUNT only), q19 (a dimension joined to another).

   **Coverage added (2026-10-13):** aggregates of `CASE WHEN p THEN v [ELSE 0 | NULL] END` with
   `p` over dimension attributes (q21 1.10x, q43 1.56x): `p`'s attributes join the finest keys,
   the device sums `v`, the driver keeps or zeroes each finest group's part. Exact sums of decimal
   products (`price * cast(qty as decimal)`, bounded by the files' statistics) and
   `SUM(COALESCE(v, 0))`.

   **q67 does not pay as built:** 4.76M finest groups (item 102K tuples x 51 stores x 12 months).
   The device part took 13 s (merging millions of groups a batch into a Java map) and the
   driver's Expand of nine grouping sets 25 s: 52 s against Spark's 5.4 s, the same answer. A
   star without fact keys whose dimension tuples multiply past
   `spark.iceberg.gpu.aggregate.star.maxGroups` (2^22) now fails over to Spark before the scan
   (q67 5.5 s). Merging on the device and finishing in Spark would not rescue q67 either: its
   finest grouping (item x store x month) keeps 4.76M groups of the ~5-6M fact rows a year
   holds, so the partial aggregate barely shrinks what Spark's nine-way ROLLUP and window sort
   then process -- where its 5.4 s go. Out of scope for star aggregates.

   **The whole suite (102 queries, same day, `results-star/full/`):** the first run with the
   strategy on was a net loss, 126.5 to 164.7 s (0.95x geomean), every answer equal: the strategy
   also takes star sub-aggregates of other queries, and three kinds lost. (1) Fact columns as
   group keys (q51 1.6 to 25.8 s, q65, q1, q74: 260K to 4.5M groups) -- now declined at planning.
   (2) Dimensions built only to fail at run time (q26/q27: `cd_demo_sk` sparse; q74/q23:
   customer's 500K tuples) -- now declined at planning by the key column's file bounds and by a
   grouped dimension's row count over `maxGroups` (now 2^17), and a star that fails at run time
   is remembered and declines when planned again. (3) Region creation per query shape (q77: six
   fact tables) -- host buffers are now anonymous `mmap` pages (`LazyZeroes`, needs
   `--enable-native-access=ALL-UNNAMED`), so creating a region costs ~1 ms instead of ~430 ms of
   zero-filling, and eight scans / six shapes are kept. After these: **127.0 to 121.6 s, 1.07x
   geomean, 102/102 equal**; q22 5.66x, q43 2.17x, q63 2.05x. Still losing: q57 0.74x, q47 0.84x
   (stars that fail at run time on their first plan) and q77 0.79x: its six small stars ran one
   after another under the aggregates' lock while Spark overlaps the subqueries.

   **Concurrent stars:** `GpuAggregates` now holds each scan's own lock, not one for all (evicted
   scans close outside the map's lock, a closed scan is re-opened); region sets an aggregate runs
   on are never evicted; and `GpuAggregateExec` starts its work on a driver thread and returns a
   one-row RDD whose task waits for it (local mode), so a query's stages overlap instead of
   computing each device aggregate in line. A failure there aggregates on the CPU and collects the
   rows. Whole suite: **127.0 to 123.6 s, geomean 1.08x, 102/102 equal**; q77 now 1.36x, q22
   5.85x, q43 2.14x, q66 1.92x, q63 1.96x. The few left below 0.9x (q47, q92, q1, q4, q71) do not
   run a star or are within a 5-run noise check.
5. Cost decision, as for reads: offload a star aggregate only where it pays (measure, do not guess).

**Built dimensions kept (2026-10-13):** each star built its dimensions again every run, a Spark
job and a pass over the rows (q22: ~150 ms of ~650 ms). They are now kept (32, least recently
used first) by the dimension's plan and its tables' snapshots. The first key was the canonical
plan alone, and it was wrong: canonicalization names every attribute "none" and refers to
columns by position, so q42's item dimension (categories) and q52's (brands) looked alike and
q52 and q63 got wrong answers in the benchmark's equality check. The key now holds the
relations' column names and the key's and attributes' names too (the failure memory's
signature as well), and `testStarAggregates` runs two such queries back to back -- it fails
without the fix. The 20 star queries: **15.10 to 10.06 s, 1.66x geomean, 20/20 equal; q22
7.62x** (4.0 to 0.53 s). The whole suite: **127.0 to 123.3 s, 1.115x geomean, 102/102 equal**.

## Phase 2: fact-to-fact joins

Scoped 2026-10-13 from the operators' own metrics (`joins/metrics.py` in the results repo: the
final adaptive plan of a Spark run, each operator's timing metrics and row counts), on the
queries where RAPIDS saves most:

- **q72 (11.6 s on Spark):** `catalog_sales` (14.0M rows after its filters) sort-merge joined
  with `inventory` (123M) on the item alone yields **1.06 billion rows**, which broadcast joins
  with `date_dim` d1/d2 (`d1.d_week_seq = d2.d_week_seq`), `household_demographics` and
  `customer_demographics` then cut to 7.5M: ~190 s of task time, ~6 s of the wall. The week
  equality is a join key Spark cannot see, because it goes through two dimensions. With dense
  dimension lookups on the device it can: each side computes its week from its date key
  (`d_week_seq[date - first]`), filters by its own dimensions, and the two sides join on
  (item, week) -- about 10 inventory rows (warehouses) per sale instead of ~75. The join is
  cuDF's (`Cudf.innerJoin` on a packed int key: item < 2^17, week < 2^14), then
  `inv_quantity_on_hand < cs_quantity`, the promotion and returns left outer joins, and the
  aggregate. This is phase 2's first target: the largest single saving in the suite, and one
  RAPIDS does not find either (it joins the billion rows faster).
- **q4, q11, q74 (and q23's subqueries):** each channel's sales joined with `customer` (500K
  keys, dense) and the year, aggregated by customer and year: a star, but with ~1M finest groups,
  which phase 1 declines. Its partial aggregate shrinks 28M rows to ~1M (unlike q67), so it
  pays once the device keeps a dense star's groups across batches (one copy back, no per-batch
  map) and the finest groups go back to Spark as rows of its partial aggregate for Spark to
  finish in parallel, instead of the driver building them.
- **q64, q17/q25/q29, q93, q95:** sales joined with returns on (item, ticket/order): a
  key-preserving join of a fact with a smaller fact (returns are ~10% of sales) -- a semi-join
  mask built from the returns' keys (`Cudf.containedIn`) or a hash join, then a star. Their
  scans and sorts dominate (q64: 28M `store_sales` rows sorted for the merge join).

Order: q72 as a prototype through the Java API (as q22 was), then the large-group stars (q4
family), then sales-returns.

**Sales with their returns (measured 2026-10-13, `metrics.py`):** q50, q17, q25, q29 and q93 (1.3
to 2.5 s each, ~10.6 s together) all sort-merge join `store_sales` (28.2M rows) with
`store_returns` (2.8M) on (customer, item, ticket) for ~1.9M matches -- the 28M-row sort
dominates (10-17 s of task time each) -- and q17/q25/q29 then join `catalog_sales` (14.1M) on
(customer, item). Returns are unique by (ticket, item), so this is a lookup, not a
many-to-many join: the probe's build side, but carrying values the aggregate needs -- SUM of
`sr_net_loss` (q25), the day differences of `CASE WHEN sr_returned_date_sk - ss_sold_date_sk <=
30` (q50), the returned quantity (q93). Next primitive: `DeviceProbe` build rows with value
columns, summed per fact row over its matches (SUM over the join is then sums of build values
plus weighted fact values), and conditional indicators over a fact column and a build column.

**q72 prototype (2026-10-13, `bench/StarQ72.java`): 100/100 rows equal to Spark's, ~0.94 s warm
against Spark's ~10.5 s and RAPIDS' 5.5 s.** Measured sizes (`joins/q72sizes.py`): the sales
side keeps 87,805 rows after its dimension filters (85,979 distinct (item, week), at most 4 a
key, quantities 1-101); the inventory side has 26.7M rows in those 53 weeks. The new primitive
is `DeviceProbe`, an inner join of a star's fact rows with a small host-built table that
contributes only multiplicity: the probe key is a packed sum of fact columns and dimension
attribute codes (`item * 64 + week code`), the build side sorted keys with one value each, and a
comparison of a fact column with that value (`inv_quantity_on_hand < cs_quantity`). Two kernels:
`probeWeights` (after `starFilter`: binary search, then the key's few build rows; weight 0
drops the row) and `applyWeights` (after the value columns: each row's sums and counts times
its weight). The prototype runs two stars: catalog sales by (item, sold date, ship date,
quantity) under the d1/household/customer filters (~200 ms), the host applies
`ship > sold + 5` and maps sold dates to weeks (~20 ms), then inventory grouped by item
description, warehouse and week with the probe (~300 ms): 40,749 groups, as Spark's. The rest
is reading the dimensions on the host (~400 ms, mostly `customer_demographics`' 1.9M rows), which
Spark would do in the planned version. `TestDeviceMergeOnRead.testStarAggregateWithProbe`
checks the probe against IcebergGenerics (an attribute and a column in the key, repeated and
absent build keys, the comparison). Promotion and returns, q72's left outer joins, match at
most one row each (their keys are primary keys) and count rows that are never null, so they
do not change the answer; a planned version must check that, or join them.

## Risks

- **Group count**: if a query groups by near-unique columns, the partial aggregate is as large as
  the fact rows and Spark's remaining work does not shrink. The prototype measures group counts.
- **Planner fragility**: matching join trees through Spark internals (`private[sql]`) ties the
  strategy to Spark minor versions, as the aggregate strategy already is.
- **Dense keys are a TPC-DS property**: real star schemas usually have them (surrogate keys), but
  the plan must check density per dimension and decline otherwise.
