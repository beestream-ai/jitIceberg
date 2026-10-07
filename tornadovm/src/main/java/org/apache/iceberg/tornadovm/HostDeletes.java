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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.FileContent;
import org.apache.iceberg.FileFormat;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.data.BaseDeleteLoader;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetReaders;
import org.apache.iceberg.deletes.PositionDeleteIndex;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.relocated.com.google.common.collect.Maps;
import org.apache.iceberg.types.Types;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.schema.MessageType;

/**
 * The host half of a merge-on-read scan: everything about one data file that is metadata rather
 * than data, prepared off the critical path while the device works on the previous file.
 *
 * <ul>
 *   <li>the file's local path and row count, from the manifest;
 *   <li>which Parquet column each projected field is, from the footer, by field id;
 *   <li>its position deletes or deletion vector, loaded by Iceberg's own {@link BaseDeleteLoader}
 *       and packed into one bit a row;
 *   <li>the equality-delete key set that applies to it, shared between files and loaded once.
 * </ul>
 *
 * <p>Which delete files apply to which data file -- the sequence-number rules -- is decided by
 * Iceberg's scan planning before any of this runs; {@link FileScanTask#deletes()} is already that
 * answer.
 */
final class HostDeletes {

  /** One data file, ready for the device. */
  static final class FilePrep {
    private final String path;
    private final long rows;
    private final int[] columns;
    private final byte[] positionBits;
    private final EqualitySet equality;
    private final long hostNanos;

    FilePrep(
        String path,
        long rows,
        int[] columns,
        byte[] positionBits,
        EqualitySet equality,
        long hostNanos) {
      this.path = path;
      this.rows = rows;
      this.columns = columns;
      this.positionBits = positionBits;
      this.equality = equality;
      this.hostNanos = hostNanos;
    }

    String path() {
      return path;
    }

    long rows() {
      return rows;
    }

    int[] columns() {
      return columns;
    }

    /** One bit a row, set when the row is deleted; null when the file has no position deletes. */
    byte[] positionBits() {
      return positionBits;
    }

    /** The equality-delete keys for this file, or null when none apply. */
    EqualitySet equality() {
      return equality;
    }

    long hostNanos() {
      return hostNanos;
    }
  }

  /** The keys of a set of equality-delete files over one column. */
  static final class EqualitySet {
    private final String id;
    private final long[] keys;
    private final boolean hasNull;

    EqualitySet(String id, long[] keys, boolean hasNull) {
      this.id = id;
      this.keys = keys;
      this.hasNull = hasNull;
    }

    /** The delete files' locations, sorted: two files with the same id share one device upload. */
    String id() {
      return id;
    }

    long[] keys() {
      return keys;
    }

    boolean hasNull() {
      return hasNull;
    }
  }

  private final Table table;
  private final ScanLayout layout;
  private final Integer equalityKeyId;
  private final BaseDeleteLoader deleteLoader;
  private final Map<String, EqualitySet> equalitySets = Maps.newConcurrentMap();

  HostDeletes(Table table, ScanLayout layout, Integer equalityKeyId) {
    this.table = table;
    this.layout = layout;
    this.equalityKeyId = equalityKeyId;
    this.deleteLoader = new BaseDeleteLoader(file -> table.io().newInputFile(file.location()));
  }

  /**
   * The single column every equality delete in the scan is keyed on, or null when there are none.
   * Multi-column equality deletes are declined rather than half-applied.
   */
  static Integer equalityKeyId(List<FileScanTask> tasks) {
    Integer key = null;
    for (FileScanTask task : tasks) {
      for (DeleteFile delete : task.deletes()) {
        if (delete.content() != FileContent.EQUALITY_DELETES) {
          continue;
        }
        List<Integer> ids = delete.equalityFieldIds();
        if (ids.size() != 1 || (key != null && !key.equals(ids.get(0)))) {
          throw new UnsupportedOperationException(
              "Equality deletes on several columns are not supported on the device: " + ids);
        }
        key = ids.get(0);
      }
    }
    return key;
  }

  /** A local path cuDF can open. Object stores are not wired up in this prototype. */
  static String localPath(String location) {
    if (location.startsWith("file:")) {
      return Paths.get(URI.create(location)).toString();
    }
    if (location.startsWith("/")) {
      return location;
    }
    throw new UnsupportedOperationException("Not a local file: " + location);
  }

  FilePrep prepare(FileScanTask task) {
    long start = System.nanoTime();
    String location = task.file().location();
    String path = localPath(location);
    long rows = task.file().recordCount();
    if (task.file().format() != FileFormat.PARQUET) {
      throw new UnsupportedOperationException("Not a Parquet data file: " + location);
    }
    if (task.start() != 0 || task.length() < task.file().fileSizeInBytes()) {
      throw new UnsupportedOperationException("Split data files are not supported: " + location);
    }

    int[] columns = layout.parquetColumns(footerSchema(path), location);

    List<DeleteFile> positionDeletes = Lists.newArrayList();
    List<DeleteFile> equalityDeletes = Lists.newArrayList();
    for (DeleteFile delete : task.deletes()) {
      if (delete.content() == FileContent.EQUALITY_DELETES) {
        equalityDeletes.add(delete);
      } else {
        positionDeletes.add(delete);
      }
    }

    byte[] bits = null;
    if (!positionDeletes.isEmpty()) {
      PositionDeleteIndex index = deleteLoader.loadPositionDeletes(positionDeletes, location);
      byte[] packed = new byte[(int) ((rows + 7) / 8)];
      index.forEach(
          position -> {
            if (position < rows) {
              packed[(int) (position >> 3)] |= (byte) (1 << (position & 7));
            }
          });
      bits = packed;
    }

    EqualitySet equality = equalityDeletes.isEmpty() ? null : equalitySet(equalityDeletes);
    return new FilePrep(path, rows, columns, bits, equality, System.nanoTime() - start);
  }

  private static MessageType footerSchema(String path) {
    try (ParquetFileReader reader = ParquetFileReader.open(new LocalInputFile(Paths.get(path)))) {
      return reader.getFileMetaData().getSchema();
    } catch (IOException e) {
      throw new UncheckedIOException("Cannot read the footer of " + path, e);
    }
  }

  private EqualitySet equalitySet(List<DeleteFile> deletes) {
    String id =
        deletes.stream()
            .map(DeleteFile::location)
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.joining(","));
    return equalitySets.computeIfAbsent(id, ignored -> loadEqualitySet(id, deletes));
  }

  /**
   * Reads the key column of each equality-delete file. Keys are widened to long on the host; the
   * device set has the key column's own width.
   */
  private EqualitySet loadEqualitySet(String id, List<DeleteFile> deletes) {
    Types.NestedField key = table.schema().findField(equalityKeyId);
    Schema keySchema = new Schema(key);
    long[] keys = new long[1024];
    int size = 0;
    boolean hasNull = false;
    for (DeleteFile delete : deletes) {
      if (delete.format() != FileFormat.PARQUET) {
        throw new UnsupportedOperationException("Not a Parquet delete file: " + delete.location());
      }
      try (CloseableIterable<Record> records =
          Parquet.read(table.io().newInputFile(delete.location()))
              .project(keySchema)
              .createReaderFunc(
                  fileSchema -> GenericParquetReaders.buildReader(keySchema, fileSchema))
              .build()) {
        for (Record record : records) {
          Object value = record.get(0);
          if (value == null) {
            hasNull = true;
            continue;
          }
          if (size == keys.length) {
            keys = Arrays.copyOf(keys, keys.length * 2);
          }
          keys[size++] = ((Number) value).longValue();
        }
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
    return new EqualitySet(id, Arrays.copyOf(keys, size), hasNull);
  }

  Integer equalityKeyId() {
    return equalityKeyId;
  }
}
