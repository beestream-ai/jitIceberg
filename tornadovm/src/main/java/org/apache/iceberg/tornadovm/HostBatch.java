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
import java.util.Map;
import org.apache.iceberg.types.Types;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.cudf.enums.CudfType;

/**
 * The live rows of one data file, copied back from the device: column-major, one typed buffer a
 * physical type, each column at element {@code slot * stride}.
 */
public final class HostBatch implements LiveRows {
  private final ScanLayout layout;
  private final Map<String, Integer> idByName;
  private final int rows;
  private final int stride;
  private final IntArray ints;
  private final LongArray longs;
  private final DoubleArray doubles;
  private final ByteArray valid;

  HostBatch(
      ScanLayout layout,
      Map<String, Integer> idByName,
      int rows,
      int stride,
      IntArray ints,
      LongArray longs,
      DoubleArray doubles,
      ByteArray valid) {
    this.layout = layout;
    this.idByName = idByName;
    this.rows = rows;
    this.stride = stride;
    this.ints = ints;
    this.longs = longs;
    this.doubles = doubles;
    this.valid = valid;
  }

  @Override
  public int rows() {
    return rows;
  }

  public List<Types.NestedField> columns() {
    return layout.fields();
  }

  private int base(String column, CudfType expected) {
    int id = idByName.get(column);
    if (layout.type(id) != expected) {
      throw new IllegalArgumentException(column + " is " + layout.type(id) + ", not " + expected);
    }
    return layout.slot(id) * stride;
  }

  @Override
  public boolean isNull(String column, int row) {
    return layout.anyOptional()
        && valid.get(layout.readIndex(idByName.get(column)) * stride + row) == 0;
  }

  @Override
  public int getInt(String column, int row) {
    return ints.get(base(column, CudfType.INT32) + row);
  }

  @Override
  public long getLong(String column, int row) {
    return longs.get(base(column, CudfType.INT64) + row);
  }

  @Override
  public double getDouble(String column, int row) {
    return doubles.get(base(column, CudfType.FLOAT64) + row);
  }
}
