/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.tornadovm.bench;

import java.util.Locale;
import java.util.SplittableRandom;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;

/**
 * The dense aggregate's accumulation, two ways: Floe's double atomics (exact because file
 * statistics bound every sum below 2^53) against native 64-bit integer atomics. Same rows, same
 * group keys, four value columns of integers (unscaled decimals), from one group (every thread
 * contends) to millions. Inputs stay on the device; each timed run clears the output and
 * accumulates every row. Both results are checked against a CPU reference.
 *
 * <pre>
 *   AtomicsBench &lt;rows&gt; &lt;runs&gt; &lt;groups&gt;...
 * </pre>
 */
public final class AtomicsBench {
  private static final int COLUMNS = 4;
  private static final int BLOCK = 256;

  private AtomicsBench() {}

  static void clearDouble(KernelContext context, DoubleArray out) {
    int index = context.globalIdx;
    if (index < out.getSize()) {
      out.set(index, 0.0);
    }
  }

  static void clearLong(KernelContext context, LongArray out) {
    int index = context.globalIdx;
    if (index < out.getSize()) {
      out.set(index, 0L);
    }
  }

  static void sumDouble(
      KernelContext context,
      IntArray group,
      LongArray values,
      int rows,
      int groups,
      DoubleArray out) {
    int row = context.globalIdx;
    if (row < rows) {
      int slot = group.get(row);
      for (int c = 0; c < COLUMNS; c++) {
        context.atomicAdd(out, c * groups + slot, (double) values.get((long) c * rows + row));
      }
    }
  }

  static void sumLong(
      KernelContext context,
      IntArray group,
      LongArray values,
      int rows,
      int groups,
      LongArray out) {
    int row = context.globalIdx;
    if (row < rows) {
      int slot = group.get(row);
      for (int c = 0; c < COLUMNS; c++) {
        context.atomicAdd(out, c * groups + slot, values.get((long) c * rows + row));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    int rows = Integer.parseInt(args[0]);
    int runs = Integer.parseInt(args[1]);
    SplittableRandom random = new SplittableRandom(42);
    LongArray values = new LongArray((long) COLUMNS * rows);
    for (long i = 0; i < values.getSize(); i++) {
      values.set((int) i, random.nextLong(1_000_000));
    }
    System.out.println("groups,kind,run,warmup,ms,exact");
    for (int a = 2; a < args.length; a++) {
      int groups = Integer.parseInt(args[a]);
      IntArray group = new IntArray(rows);
      for (int i = 0; i < rows; i++) {
        group.set(i, random.nextInt(groups));
      }
      long[] reference = new long[COLUMNS * groups];
      for (int c = 0; c < COLUMNS; c++) {
        for (int i = 0; i < rows; i++) {
          reference[c * groups + group.get(i)] += values.get((long) c * rows + i);
        }
      }
      DoubleArray outDouble = new DoubleArray(COLUMNS * groups);
      LongArray outLong = new LongArray(COLUMNS * groups);
      run(groups, "double", rows, runs, group, values, outDouble, null, reference);
      run(groups, "int64", rows, runs, group, values, null, outLong, reference);
    }
  }

  private static void run(
      int groups,
      String kind,
      int rows,
      int runs,
      IntArray group,
      LongArray values,
      DoubleArray outDouble,
      LongArray outLong,
      long[] reference)
      throws TornadoExecutionPlanException {
    KernelContext context = new KernelContext();
    boolean dbl = outDouble != null;
    TaskGraph graph =
        new TaskGraph("atomics").transferToDevice(DataTransferMode.FIRST_EXECUTION, group, values);
    if (dbl) {
      graph
          .task("clear", AtomicsBench::clearDouble, context, outDouble)
          .task("sum", AtomicsBench::sumDouble, context, group, values, rows, groups, outDouble)
          .transferToHost(DataTransferMode.UNDER_DEMAND, outDouble);
    } else {
      graph
          .task("clear", AtomicsBench::clearLong, context, outLong)
          .task("sum", AtomicsBench::sumLong, context, group, values, rows, groups, outLong)
          .transferToHost(DataTransferMode.UNDER_DEMAND, outLong);
    }
    GridScheduler grids = new GridScheduler();
    WorkerGrid1D clear = new WorkerGrid1D(roundUp(COLUMNS * groups));
    clear.setLocalWork(BLOCK, 1, 1);
    WorkerGrid1D sum = new WorkerGrid1D(roundUp(rows));
    sum.setLocalWork(BLOCK, 1, 1);
    grids.addWorkerGrid("atomics.clear", clear);
    grids.addWorkerGrid("atomics.sum", sum);
    try (TornadoExecutionPlan plan =
        new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(grids)) {
      for (int r = 0; r <= runs; r++) {
        long start = System.nanoTime();
        TornadoExecutionResult result = plan.execute();
        double ms = (System.nanoTime() - start) / 1e6;
        boolean exact = true;
        if (r == runs) {
          result.transferToHost(dbl ? outDouble : outLong);
          for (int i = 0; i < reference.length; i++) {
            long got = dbl ? (long) outDouble.get(i) : outLong.get(i);
            exact &= got == reference[i];
          }
        }
        System.out.printf(
            Locale.ROOT,
            "%d,%s,%d,%s,%.2f,%s%n",
            groups,
            kind,
            r,
            r == 0 ? "yes" : "no",
            ms,
            r == runs ? exact : "");
      }
    }
  }

  private static int roundUp(int count) {
    return (count + BLOCK - 1) / BLOCK * BLOCK;
  }
}
