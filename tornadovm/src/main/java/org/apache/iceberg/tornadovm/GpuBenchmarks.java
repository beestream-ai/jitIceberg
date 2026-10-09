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
import java.util.concurrent.atomic.AtomicLong;
import org.apache.iceberg.Schema;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.types.Types;

/** Helpers for benchmarks that measure the device scan against Iceberg's own readers. */
public final class GpuBenchmarks {

  private GpuBenchmarks() {}

  /**
   * Materializes every live row of the table with all of its top-level columns on the host and
   * reads every value, folding them into {@code checksum} so that none can be skipped: the work
   * Spark does to build every row of a scan. Returns the live row count.
   */
  public static long materializeAll(
      DeviceMergeOnReadScan scan, Schema schema, AtomicLong checksum) {
    List<String> columns = Lists.newArrayList();
    for (Types.NestedField field : schema.columns()) {
      columns.add(field.name());
    }
    return scan.materialize(
        columns,
        batch -> {
          int count = columns.size();
          int[] index = new int[count];
          char[] kind = new char[count];
          for (int c = 0; c < count; c++) {
            Types.NestedField field = schema.findField(columns.get(c));
            index[c] = batch.column(field.name());
            kind[c] =
                switch (field.type().typeId()) {
                  case INTEGER, DATE -> 'i';
                  case LONG, TIMESTAMP -> 'l';
                  case FLOAT, DOUBLE -> 'd';
                  case STRING -> 's';
                  default -> throw new UnsupportedOperationException("Column " + field);
                };
          }
          long sum = 0;
          for (int r = 0; r < batch.rows(); r++) {
            for (int c = 0; c < count; c++) {
              if (batch.isNull(index[c], r)) {
                sum += 31;
                continue;
              }
              sum =
                  sum * 17
                      + switch (kind[c]) {
                        case 'i' -> batch.getInt(index[c], r);
                        case 'l' -> batch.getLong(index[c], r);
                        case 'd' -> Double.doubleToRawLongBits(batch.getDouble(index[c], r));
                        default -> batch.getString(index[c], r).hashCode();
                      };
            }
          }
          checksum.addAndGet(sum);
        });
  }
}
