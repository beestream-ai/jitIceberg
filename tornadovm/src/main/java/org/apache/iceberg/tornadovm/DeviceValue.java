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

import java.util.List;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;

/**
 * A value to sum per group, evaluated on the device row by row: a product of factors {@code (a + b
 * * column)} -- a column, {@code 1 - column}, {@code 1 + column} -- which covers the arithmetic of
 * TPC-H's pricing queries, {@code l_extendedprice * (1 - l_discount) * (1 + l_tax)} for one. A row
 * where any factor's column is null contributes 0, as SQL's SUM skips a null product.
 */
public final class DeviceValue implements java.io.Serializable {
  /** Most factors in one value. */
  public static final int MAX_FACTORS = 4;

  /**
   * {@code offset + scale * column}; a non-zero {@code coefficient} starts a new term of the sum,
   * which it multiplies.
   */
  record Factor(String column, double offset, double scale, double coefficient)
      implements java.io.Serializable {}

  private final List<Factor> factors;
  // A probe's build column summed over the build rows a fact row joins, or -1.
  private final int probeColumn;

  private DeviceValue(List<Factor> factors) {
    this(factors, -1);
  }

  private DeviceValue(List<Factor> factors, int probeColumn) {
    if (factors.size() > MAX_FACTORS) {
      throw new IllegalArgumentException("At most " + MAX_FACTORS + " factors: " + factors);
    }
    this.factors = factors;
    this.probeColumn = probeColumn;
  }

  /**
   * Build column {@code column} of a star aggregate's {@link DeviceProbe}: per fact row, the sum of
   * its values over the build rows the row joins (nulls skipped, and counted apart).
   */
  public static DeviceValue probe(int column) {
    return new DeviceValue(ImmutableList.of(), column);
  }

  int probeColumn() {
    return probeColumn;
  }

  /** The column itself. */
  public static DeviceValue of(String column) {
    return new DeviceValue(ImmutableList.of(new Factor(column, 0, 1, 1)));
  }

  public DeviceValue times(String column) {
    return with(new Factor(column, 0, 1, 0));
  }

  /** Times {@code 1 - column}. */
  public DeviceValue timesOneMinus(String column) {
    return with(new Factor(column, 1, -1, 0));
  }

  /** Times {@code 1 + column}. */
  public DeviceValue timesOnePlus(String column) {
    return with(new Factor(column, 1, 1, 0));
  }

  /**
   * This value plus {@code other}: a sum of terms, each a product. A row where any term's column is
   * null contributes nothing, as SQL's {@code a + b} is null when either is.
   */
  public DeviceValue plus(DeviceValue other) {
    return sum(other, 1);
  }

  /** Minus this value. */
  public DeviceValue negate() {
    ImmutableList.Builder<Factor> negated = ImmutableList.builder();
    for (Factor f : factors) {
      negated.add(new Factor(f.column(), f.offset(), f.scale(), -f.coefficient()));
    }
    return new DeviceValue(negated.build(), probeColumn);
  }

  /** This value minus {@code other}. */
  public DeviceValue minus(DeviceValue other) {
    return sum(other, -1);
  }

  private DeviceValue sum(DeviceValue other, double sign) {
    if (probeColumn >= 0 || other.probeColumn >= 0) {
      throw new IllegalArgumentException("A probe value is not a term");
    }
    ImmutableList.Builder<Factor> terms = ImmutableList.<Factor>builder().addAll(factors);
    for (Factor f : other.factors) {
      terms.add(
          f.coefficient() == 0
              ? f
              : new Factor(f.column(), f.offset(), f.scale(), sign * f.coefficient()));
    }
    return new DeviceValue(terms.build());
  }

  List<Factor> factors() {
    return factors;
  }

  private DeviceValue with(Factor factor) {
    return new DeviceValue(ImmutableList.<Factor>builder().addAll(factors).add(factor).build());
  }

  @Override
  public String toString() {
    if (probeColumn >= 0) {
      return "probe[" + probeColumn + "]";
    }
    StringBuilder text = new StringBuilder();
    for (Factor factor : factors) {
      if (text.length() > 0) {
        text.append(factor.coefficient() == 0 ? " * " : factor.coefficient() < 0 ? " - " : " + ");
      }
      text.append(
          factor.offset() == 0
              ? factor.column()
              : "("
                  + factor.offset()
                  + (factor.scale() < 0 ? " - " : " + ")
                  + factor.column()
                  + ")");
    }
    return text.toString();
  }
}
