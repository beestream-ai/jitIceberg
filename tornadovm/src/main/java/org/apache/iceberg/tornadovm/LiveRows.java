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

/** The live rows of one data file after merge-on-read, column by name. */
public interface LiveRows {
  int rows();

  /** The index of a column, for the by-index accessors: resolve once, read many rows. */
  int column(String name);

  boolean isNull(int column, int row);

  int getInt(int column, int row);

  long getLong(int column, int row);

  double getDouble(int column, int row);

  default boolean isNull(String column, int row) {
    return isNull(column(column), row);
  }

  default int getInt(String column, int row) {
    return getInt(column(column), row);
  }

  default long getLong(String column, int row) {
    return getLong(column(column), row);
  }

  default double getDouble(String column, int row) {
    return getDouble(column(column), row);
  }
}
