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
package org.apache.iceberg.tornadovm;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;

/**
 * The Iceberg-specific parts of a merge-on-read scan, as TornadoVM kernels.
 *
 * <p>These are plain Java methods: TornadoVM JIT-compiles them to CUDA at the first execution of
 * the plan that holds them. cuDF does the work that needs cross-row cooperation (Parquet decoding,
 * the equality-delete hash probe, grouping, compaction); what is left is per-row and specific to
 * the table format, and is written here once, in Java.
 *
 * <p>Every kernel runs over the plan's capacity rather than the current file's row count, because a
 * plan is built once and reused for every file; {@code meta[META_ROWS]} says how many rows are
 * real.
 */
final class DeviceKernels {

  /** Rows in the current file. */
  static final int META_ROWS = 0;

  /** 1 when the current file has position deletes or a deletion vector. */
  static final int META_HAS_POSITION_DELETES = 1;

  /** 1 when the current file has equality deletes to apply. */
  static final int META_HAS_EQUALITY_DELETES = 2;

  /**
   * 1 when the equality-delete set contains a null key, which deletes every row whose key is null.
   */
  static final int META_EQUALITY_SET_HAS_NULL = 3;

  /** 1 when the equality key column is optional, so its validity bytes have to be consulted. */
  static final int META_KEY_NULLABLE = 4;

  /** Position-delete entries read on the device for the current batch. */
  static final int META_POSITION_ROWS = 5;

  /** Runs of those entries, one a data file that has them. */
  static final int META_POSITION_SEGMENTS = 6;

  static final int META_SIZE = 7;

  /**
   * The group key of a deleted or padding row. It is grouped like any other key and dropped on the
   * host, which lets cuDF's group-by run over the plan's whole capacity without a compaction first.
   */
  static final int DEAD_GROUP = Integer.MIN_VALUE;

  private DeviceKernels() {}

  /**
   * Merge-on-read in one pass: a row is live when it is a real row, its position is not set in the
   * file's deletion bitmap, and its key is not in the equality-delete set.
   *
   * <p>Equality deletes follow Iceberg's semantics for nulls: a null in the delete set deletes rows
   * whose key is null, and a null key is never deleted by a non-null entry.
   *
   * @param positionBits one bit a row, bit {@code i & 7} of byte {@code i >> 3}; set means deleted
   * @param inEqualitySet one byte a row from cuDF's set probe; read only when the file has equality
   *     deletes
   * @param valid validity bytes from the Parquet read; read only when the key is nullable
   * @param keyValidOffset where the key column's validity bytes start in {@code valid}
   */
  static void liveRows(
      IntArray meta,
      ByteArray positionBits,
      ByteArray inEqualitySet,
      ByteArray valid,
      int keyValidOffset,
      ByteArray live) {
    for (@Parallel int i = 0; i < live.getSize(); i++) {
      byte keep = 0;
      if (i < meta.get(META_ROWS)) {
        keep = 1;
        if (meta.get(META_HAS_POSITION_DELETES) == 1
            && ((positionBits.get(i >> 3) >> (i & 7)) & 1) == 1) {
          keep = 0;
        }
        if (meta.get(META_HAS_EQUALITY_DELETES) == 1) {
          boolean keyIsNull =
              meta.get(META_KEY_NULLABLE) == 1 && valid.get(keyValidOffset + i) == 0;
          if (keyIsNull) {
            if (meta.get(META_EQUALITY_SET_HAS_NULL) == 1) {
              keep = 0;
            }
          } else if (inEqualitySet.get(i) == 1) {
            keep = 0;
          }
        }
      }
      live.set(i, keep);
    }
  }

  /**
   * Applies position-delete files the device read itself: entry {@code j} is a row position within
   * its data file, and the run of entries it belongs to says where that file starts in the batch.
   * Runs are few (one a data file of the batch), so each thread finds its own by a short scan. Only
   * zeroes are written, so threads that delete the same row do not race.
   *
   * @param segmentStarts the first entry of each run
   * @param segmentOffsets the batch row at which each run's data file starts
   */
  static void deletePositions(
      LongArray positions,
      IntArray meta,
      IntArray segmentStarts,
      IntArray segmentOffsets,
      ByteArray live) {
    for (@Parallel int j = 0; j < positions.getSize(); j++) {
      if (j < meta.get(META_POSITION_ROWS)) {
        int segments = meta.get(META_POSITION_SEGMENTS);
        int s = 0;
        while (s + 1 < segments && segmentStarts.get(s + 1) <= j) {
          s++;
        }
        long row = segmentOffsets.get(s) + positions.get(j);
        live.set((int) row, (byte) 0);
      }
    }
  }

  /**
   * Applies a scan's residual filter to the live rows: a conjunction of compiled terms (see {@link
   * DevicePredicate}), each a column compared with constants. A row stays live only if every term
   * holds. Nulls follow Iceberg's row evaluator, which orders them first.
   */
  static void filterRows(
      IntArray terms,
      LongArray longConstants,
      DoubleArray doubleConstants,
      IntArray ints,
      LongArray longs,
      DoubleArray doubles,
      ByteArray valid,
      int stride,
      ByteArray live) {
    for (@Parallel int i = 0; i < live.getSize(); i++) {
      if (live.get(i) == 1) {
        int count = terms.get(0);
        boolean keep = true;
        for (int t = 0; t < count; t++) {
          int base = 1 + t * DevicePredicate.TERM_INTS;
          int kind = terms.get(base);
          int op = terms.get(base + 1);
          int offset = terms.get(base + 2);
          int validColumn = terms.get(base + 3);
          int first = terms.get(base + 4);
          int constants = terms.get(base + 5);
          boolean isNull = validColumn >= 0 && valid.get(validColumn * stride + i) == 0;
          boolean pass;
          if (op == DevicePredicate.OP_FALSE) {
            pass = false;
          } else if (op == DevicePredicate.OP_IS_NULL) {
            pass = isNull;
          } else if (op == DevicePredicate.OP_NOT_NULL) {
            pass = !isNull;
          } else if (isNull) {
            // Iceberg's own evaluator compares with nulls first: a null is below every value.
            pass =
                op == DevicePredicate.OP_LT
                    || op == DevicePredicate.OP_LT_EQ
                    || op == DevicePredicate.OP_NOT_EQ;
          } else if (kind == 2) {
            pass = compareDouble(doubles.get(offset + i), op, doubleConstants, first, constants);
          } else if (kind == 1) {
            pass = compareLong(longs.get(offset + i), op, longConstants, first, constants);
          } else {
            pass = compareLong(ints.get(offset + i), op, longConstants, first, constants);
          }
          keep = keep && pass;
        }
        if (!keep) {
          live.set(i, (byte) 0);
        }
      }
    }
  }

  private static boolean compareLong(
      long value, int op, LongArray constants, int first, int count) {
    boolean pass;
    if (op == DevicePredicate.OP_IN) {
      pass = false;
      for (int c = 0; c < count; c++) {
        if (value == constants.get(first + c)) {
          pass = true;
        }
      }
    } else {
      long constant = constants.get(first);
      if (op == DevicePredicate.OP_LT) {
        pass = value < constant;
      } else if (op == DevicePredicate.OP_LT_EQ) {
        pass = value <= constant;
      } else if (op == DevicePredicate.OP_GT) {
        pass = value > constant;
      } else if (op == DevicePredicate.OP_GT_EQ) {
        pass = value >= constant;
      } else if (op == DevicePredicate.OP_EQ) {
        pass = value == constant;
      } else {
        pass = value != constant;
      }
    }
    return pass;
  }

  private static boolean compareDouble(
      double value, int op, DoubleArray constants, int first, int count) {
    boolean pass;
    if (op == DevicePredicate.OP_IN) {
      pass = false;
      for (int c = 0; c < count; c++) {
        if (value == constants.get(first + c)) {
          pass = true;
        }
      }
    } else {
      double constant = constants.get(first);
      if (op == DevicePredicate.OP_LT) {
        pass = value < constant;
      } else if (op == DevicePredicate.OP_LT_EQ) {
        pass = value <= constant;
      } else if (op == DevicePredicate.OP_GT) {
        pass = value > constant;
      } else if (op == DevicePredicate.OP_GT_EQ) {
        pass = value >= constant;
      } else if (op == DevicePredicate.OP_EQ) {
        pass = value == constant;
      } else {
        pass = value != constant;
      }
    }
    return pass;
  }

  /**
   * The key each row is grouped under: its value of the group column, or {@link #DEAD_GROUP} when
   * the row is deleted. With no group column ({@code keyOffset < 0}) every live row is group 0.
   */
  static void groupKeys(ByteArray live, IntArray ints, int keyOffset, IntArray groups) {
    for (@Parallel int i = 0; i < groups.getSize(); i++) {
      int key = DEAD_GROUP;
      if (live.get(i) == 1) {
        key = keyOffset < 0 ? 0 : ints.get(keyOffset + i);
      }
      groups.set(i, key);
    }
  }

  /**
   * Zeroes the nulls of optional value columns, so that cuDF's SUM ignores them as SQL's does.
   *
   * @param validColumn for each FP64 slot, the index of its column in the validity bytes, or -1 for
   *     a required column
   */
  static void zeroNulls(
      DoubleArray doubles, ByteArray valid, IntArray validColumn, int slots, int stride) {
    for (@Parallel int i = 0; i < stride; i++) {
      for (int s = 0; s < slots; s++) {
        int column = validColumn.get(s);
        if (column >= 0 && valid.get(column * stride + i) == 0) {
          doubles.set(s * stride + i, 0.0);
        }
      }
    }
  }

  /** Moves the surviving rows of every INT32 slot to the front, in their original order. */
  static void gatherInts(
      IntArray positions, IntArray count, IntArray source, int slots, int stride, IntArray target) {
    for (@Parallel int j = 0; j < stride; j++) {
      if (j < count.get(0)) {
        int row = positions.get(j);
        for (int s = 0; s < slots; s++) {
          target.set(s * stride + j, source.get(s * stride + row));
        }
      }
    }
  }

  /** {@link #gatherInts} for INT64 slots. */
  static void gatherLongs(
      IntArray positions,
      IntArray count,
      LongArray source,
      int slots,
      int stride,
      LongArray target) {
    for (@Parallel int j = 0; j < stride; j++) {
      if (j < count.get(0)) {
        int row = positions.get(j);
        for (int s = 0; s < slots; s++) {
          target.set(s * stride + j, source.get(s * stride + row));
        }
      }
    }
  }

  /** {@link #gatherInts} for FP64 slots. */
  static void gatherDoubles(
      IntArray positions,
      IntArray count,
      DoubleArray source,
      int slots,
      int stride,
      DoubleArray target) {
    for (@Parallel int j = 0; j < stride; j++) {
      if (j < count.get(0)) {
        int row = positions.get(j);
        for (int s = 0; s < slots; s++) {
          target.set(s * stride + j, source.get(s * stride + row));
        }
      }
    }
  }

  /** {@link #gatherInts} for validity bytes. */
  static void gatherBytes(
      IntArray positions,
      IntArray count,
      ByteArray source,
      int slots,
      int stride,
      ByteArray target) {
    for (@Parallel int j = 0; j < stride; j++) {
      if (j < count.get(0)) {
        int row = positions.get(j);
        for (int s = 0; s < slots; s++) {
          target.set(s * stride + j, source.get(s * stride + row));
        }
      }
    }
  }

  /**
   * Groups live rows by Iceberg's {@code bucket[N]} partition transform of a long column: Murmur3
   * x86 32-bit of the value's 8 little-endian bytes, seed 0, made non-negative and taken modulo N
   * -- the function of {@code org.apache.iceberg.util.BucketUtil}, so a row lands in the bucket
   * Iceberg would write it to. Deleted and padding rows get {@link #DEAD_GROUP}.
   */
  static void bucketGroupKeys(
      ByteArray live, LongArray longs, int offset, int buckets, IntArray groups) {
    for (@Parallel int i = 0; i < groups.getSize(); i++) {
      int key = DEAD_GROUP;
      if (live.get(i) == 1) {
        long value = longs.get(offset + i);
        int h1 = mixH1(0, mixK1((int) value));
        h1 = mixH1(h1, mixK1((int) (value >>> 32)));
        key = (fmix(h1, 8) & Integer.MAX_VALUE) % buckets;
      }
      groups.set(i, key);
    }
  }

  private static int mixK1(int k) {
    int k1 = k * 0xcc9e2d51;
    k1 = (k1 << 15) | (k1 >>> 17);
    return k1 * 0x1b873593;
  }

  private static int mixH1(int h, int k1) {
    int h1 = h ^ k1;
    h1 = (h1 << 13) | (h1 >>> 19);
    return h1 * 5 + 0xe6546b64;
  }

  private static int fmix(int h, int length) {
    int h1 = h ^ length;
    h1 ^= h1 >>> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >>> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >>> 16;
    return h1;
  }
}
