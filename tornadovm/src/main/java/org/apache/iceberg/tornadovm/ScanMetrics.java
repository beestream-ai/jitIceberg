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

import java.util.Locale;

/** Where a scan spent its time, in milliseconds of wall clock, plus the work it did. */
public final class ScanMetrics {
  private double planMillis;
  private double deleteLoadMillis;
  private double deviceMillis;
  private double copyBackMillis;
  private double totalMillis;
  private long files;
  private long rowsRead;
  private long rowsLive;

  void plan(double millis) {
    planMillis += millis;
  }

  /**
   * Host time spent loading delete files and footers; overlapped with device work when it can be.
   */
  void deleteLoad(double millis) {
    deleteLoadMillis += millis;
  }

  void device(double millis) {
    deviceMillis += millis;
  }

  void copyBack(double millis) {
    copyBackMillis += millis;
  }

  void total(double millis) {
    totalMillis = millis;
  }

  void file(long rows) {
    files++;
    rowsRead += rows;
  }

  void live(long rows) {
    rowsLive += rows;
  }

  public double planMillis() {
    return planMillis;
  }

  public double deleteLoadMillis() {
    return deleteLoadMillis;
  }

  public double deviceMillis() {
    return deviceMillis;
  }

  public double copyBackMillis() {
    return copyBackMillis;
  }

  public double totalMillis() {
    return totalMillis;
  }

  public long files() {
    return files;
  }

  public long rowsRead() {
    return rowsRead;
  }

  public long rowsLive() {
    return rowsLive;
  }

  @Override
  public String toString() {
    return String.format(
        Locale.ROOT,
        "total %.1f ms | plan %.1f, delete+footer load %.1f (host, overlapped), device %.1f, copy-back %.1f"
            + " | %d files, %d rows read, %d live",
        totalMillis,
        planMillis,
        deleteLoadMillis,
        deviceMillis,
        copyBackMillis,
        files,
        rowsRead,
        rowsLive);
  }
}
