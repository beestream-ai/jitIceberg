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

import java.util.Arrays;
import java.util.List;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;

/**
 * An inner join of a star aggregate's fact rows with a small table the caller has already reduced
 * on the host (another fact table's rows after their own filters, say): the table is sorted keys
 * and one value each, and a fact row joins every build row with its key -- built from fact columns
 * and dimension attributes, {@code sum(part * multiplier)} -- whose value passes a comparison with
 * one of the row's columns. The build side contributes only multiplicity: the row's sums and counts
 * are taken {@code weight} times, and a row that joins nothing is dropped.
 *
 * <p>TPC-DS q72: inventory rows join catalog sales on (item, week), the week a date dimension's
 * attribute, where {@code inv_quantity_on_hand < cs_quantity}.
 */
public final class DeviceProbe {
  /** How the fact row's column compares with a build row's value. */
  public enum Comparison {
    ANY,
    LESS,
    LESS_EQUAL,
    EQUAL,
    GREATER,
    GREATER_EQUAL
  }

  /**
   * A key part: a fact column, or attribute {@code attribute} of dimension {@code dimension}; its
   * value less {@code offset}, which must lie in [0, {@code range}) for the row to join anything,
   * times {@code multiplier}.
   */
  record Part(String column, int dimension, int attribute, int multiplier, int offset, int range) {
    boolean isAttribute() {
      return dimension >= 0;
    }
  }

  private final long[] keys;
  private final long[] values;
  private final List<Part> parts;
  private final String column;
  private final Comparison comparison;

  private DeviceProbe(
      long[] keys, long[] values, List<Part> parts, String column, Comparison comparison) {
    this.keys = keys;
    this.values = values;
    this.parts = parts;
    this.column = column;
    this.comparison = comparison;
  }

  /** The build rows: a key and a value each, in any order. */
  public static DeviceProbe of(long[] keys, long[] values) {
    if (keys.length != values.length) {
      throw new IllegalArgumentException(keys.length + " keys for " + values.length + " values");
    }
    Integer[] order = new Integer[keys.length];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    Arrays.sort(order, (a, b) -> Long.compare(keys[a], keys[b]));
    long[] sortedKeys = new long[keys.length];
    long[] sortedValues = new long[keys.length];
    for (int i = 0; i < order.length; i++) {
      sortedKeys[i] = keys[order[i]];
      sortedValues[i] = values[order[i]];
    }
    return new DeviceProbe(sortedKeys, sortedValues, ImmutableList.of(), null, Comparison.ANY);
  }

  /** Adds {@code column * multiplier} of the fact row to its key. */
  public DeviceProbe keyColumn(String factColumn, int multiplier) {
    return keyColumn(factColumn, multiplier, 0, Integer.MAX_VALUE);
  }

  /**
   * Adds {@code (column - offset) * multiplier} to the key; a row whose column is outside [offset,
   * offset + range) joins nothing, so parts cannot alias one another.
   */
  public DeviceProbe keyColumn(String factColumn, int multiplier, int offset, int range) {
    return with(new Part(factColumn, -1, -1, multiplier, offset, range));
  }

  /** Adds the code of the row's dimension attribute, times {@code multiplier}, to its key. */
  public DeviceProbe keyAttribute(int dimension, int attribute, int multiplier) {
    return keyAttribute(dimension, attribute, multiplier, 0, Integer.MAX_VALUE);
  }

  /** {@link #keyAttribute(int, int, int)} of {@code (code - offset)}, in [0, range) to join. */
  public DeviceProbe keyAttribute(
      int dimension, int attribute, int multiplier, int offset, int range) {
    return with(new Part(null, dimension, attribute, multiplier, offset, range));
  }

  /** Joins only build rows whose value passes {@code factColumn comparison value}. */
  public DeviceProbe where(String factColumn, Comparison how) {
    return new DeviceProbe(keys, values, parts, factColumn, how);
  }

  private DeviceProbe with(Part part) {
    return new DeviceProbe(
        keys,
        values,
        ImmutableList.<Part>builder().addAll(parts).add(part).build(),
        column,
        comparison);
  }

  long[] keys() {
    return keys;
  }

  long[] values() {
    return values;
  }

  List<Part> parts() {
    return parts;
  }

  String column() {
    return column;
  }

  Comparison comparison() {
    return comparison;
  }

  int rows() {
    return keys.length;
  }

  @Override
  public String toString() {
    return "probe "
        + parts
        + " of "
        + keys.length
        + " rows"
        + (column == null ? "" : " where " + column + " " + comparison);
  }
}
