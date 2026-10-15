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
  /**
   * One factor of a value: {@code offset + scale * column}, starting a new term (multiplied by
   * {@code coefficient}) when the coefficient is not zero, multiplying the current term otherwise.
   */
  public record Factor(String column, double offset, double scale, double coefficient)
      implements java.io.Serializable {}

  private final List<Factor> factors;

  private DeviceValue(List<Factor> factors) {
    if (factors.size() > MAX_FACTORS) {
      throw new IllegalArgumentException("At most " + MAX_FACTORS + " factors: " + factors);
    }
    this.factors = factors;
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
    return new DeviceValue(negated.build());
  }

  /** This value minus {@code other}. */
  public DeviceValue minus(DeviceValue other) {
    return sum(other, -1);
  }

  private DeviceValue sum(DeviceValue other, double sign) {
    ImmutableList.Builder<Factor> terms = ImmutableList.<Factor>builder().addAll(factors);
    for (Factor f : other.factors) {
      terms.add(
          f.coefficient() == 0
              ? f
              : new Factor(f.column(), f.offset(), f.scale(), sign * f.coefficient()));
    }
    return new DeviceValue(terms.build());
  }

  public List<Factor> factors() {
    return factors;
  }

  /** The columns the value reads. */
  public List<String> columns() {
    List<String> columns = new java.util.ArrayList<>();
    for (Factor f : factors) {
      if (!columns.contains(f.column())) {
        columns.add(f.column());
      }
    }
    return columns;
  }

  /**
   * The largest magnitude one row's value can have, given each column's largest magnitude: the sum
   * over terms of the product of each factor's {@code |offset| + |scale| * largest}.
   */
  public double bound(java.util.function.ToDoubleFunction<String> largest) {
    double bound = 0;
    double term = 0;
    for (Factor f : factors) {
      double factor =
          Math.abs(f.offset()) + Math.abs(f.scale()) * largest.applyAsDouble(f.column());
      if (f.coefficient() != 0) {
        bound += term;
        term = Math.abs(f.coefficient()) * factor;
      } else {
        term *= factor;
      }
    }
    return bound + term;
  }

  private DeviceValue with(Factor factor) {
    return new DeviceValue(ImmutableList.<Factor>builder().addAll(factors).add(factor).build());
  }

  @Override
  public String toString() {
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
