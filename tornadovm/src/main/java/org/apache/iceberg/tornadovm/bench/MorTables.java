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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.LongFunction;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.AppendFiles;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.FileFormat;
import org.apache.iceberg.MetricsConfig;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.RowDelta;
import org.apache.iceberg.Schema;
import org.apache.iceberg.SortOrder;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableProperties;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.deletes.BaseDVFileWriter;
import org.apache.iceberg.deletes.DVFileWriter;
import org.apache.iceberg.deletes.EqualityDeleteWriter;
import org.apache.iceberg.deletes.PositionDelete;
import org.apache.iceberg.deletes.PositionDeleteWriter;
import org.apache.iceberg.hadoop.HadoopTables;
import org.apache.iceberg.io.DataWriter;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.io.OutputFileFactory;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableMap;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.types.Types;

/**
 * Builds merge-on-read Iceberg tables on the local file system with Iceberg's own writers: data
 * files, v2 position-delete files (one per data file, as {@code write.delete.granularity=file}
 * produces), v3 deletion vectors, and equality-delete files. Each delete commit is a row delta, so
 * sequence numbers -- which decide what a delete applies to -- are Iceberg's.
 */
public final class MorTables {

  /** {@code id long, k int, v1..v4 double}, all required: the benchmark table. */
  public static final Schema SCHEMA =
      new Schema(
          Types.NestedField.required(1, "id", Types.LongType.get()),
          Types.NestedField.required(2, "k", Types.IntegerType.get()),
          Types.NestedField.required(3, "v1", Types.DoubleType.get()),
          Types.NestedField.required(4, "v2", Types.DoubleType.get()),
          Types.NestedField.required(5, "v3", Types.DoubleType.get()),
          Types.NestedField.required(6, "v4", Types.DoubleType.get()));

  private MorTables() {}

  public static Table create(String location, Schema schema, int formatVersion) {
    return create(location, schema, formatVersion, ImmutableMap.of());
  }

  public static Table create(
      String location, Schema schema, int formatVersion, Map<String, String> properties) {
    Map<String, String> props =
        ImmutableMap.<String, String>builder()
            .putAll(properties)
            .put(TableProperties.FORMAT_VERSION, String.valueOf(formatVersion))
            .buildKeepingLast();
    return new HadoopTables(new Configuration())
        .create(schema, PartitionSpec.unpartitioned(), SortOrder.unsorted(), props, location);
  }

  public static Table load(String location) {
    return new HadoopTables(new Configuration()).load(location);
  }

  /**
   * Writes one data file of {@code rows} rows, row {@code i} given by {@code row}; not committed.
   */
  public static DataFile writeDataFile(Table table, long rows, LongFunction<Record> row) {
    OutputFile out = newFile(table, "data");
    try {
      DataWriter<Record> writer =
          Parquet.writeData(out)
              .forTable(table)
              .createWriterFunc(GenericParquetWriter::create)
              .withSpec(table.spec())
              .overwrite()
              .build();
      try (DataWriter<Record> closing = writer) {
        for (long i = 0; i < rows; i++) {
          closing.write(row.apply(i));
        }
      }
      return writer.toDataFile();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  public static void append(Table table, List<DataFile> files) {
    AppendFiles append = table.newAppend();
    files.forEach(append::appendFile);
    append.commit();
  }

  /**
   * Deletes rows of data files by position, in one commit: a position-delete file per data file in
   * a v2 table, a deletion vector per data file in a v3 table.
   *
   * @param positions for each data file, its deleted positions
   */
  public static void deletePositions(Table table, List<DataFile> files, List<long[]> positions) {
    List<DeleteFile> deletes = Lists.newArrayList();
    int formatVersion = formatVersion(table);
    try {
      if (formatVersion >= 3) {
        OutputFileFactory factory =
            OutputFileFactory.builderFor(table, 1, System.nanoTime())
                .format(FileFormat.PUFFIN)
                .build();
        DVFileWriter writer = new BaseDVFileWriter(factory, path -> null);
        try (DVFileWriter closing = writer) {
          for (int f = 0; f < files.size(); f++) {
            String path = files.get(f).location();
            for (long position : positions.get(f)) {
              closing.delete(path, position, table.spec(), null);
            }
          }
        }
        deletes.addAll(writer.result().deleteFiles());
      } else {
        for (int f = 0; f < files.size(); f++) {
          if (positions.get(f).length == 0) {
            continue;
          }
          PositionDeleteWriter<Record> writer =
              Parquet.writeDeletes(newFile(table, "pos-deletes"))
                  .setAll(table.properties())
                  // As Iceberg's own writer factories do: full bounds on file_path, which is what
                  // marks a delete file as scoped to one data file.
                  .metricsConfig(MetricsConfig.forPositionDelete())
                  .withSpec(table.spec())
                  .overwrite()
                  .buildPositionWriter();
          PositionDelete<Record> delete = PositionDelete.create();
          String path = files.get(f).location();
          try (PositionDeleteWriter<Record> closing = writer) {
            for (long position : positions.get(f)) {
              closing.write(delete.set(path, position));
            }
          }
          deletes.add(writer.toDeleteFile());
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    RowDelta delta = table.newRowDelta();
    deletes.forEach(delta::addDeletes);
    delta.commit();
  }

  /**
   * Deletes every row whose {@code column} equals one of {@code keys} -- a null key deletes the
   * rows where the column is null -- in one equality-delete file and one commit. It applies to the
   * data files committed before it, and not to any committed after.
   */
  public static void deleteEqual(Table table, String column, List<Object> keys) {
    Types.NestedField field = table.schema().findField(column);
    Schema keySchema = new Schema(field);
    try {
      EqualityDeleteWriter<Record> writer =
          Parquet.writeDeletes(newFile(table, "eq-deletes"))
              .forTable(table)
              .withSpec(table.spec())
              .rowSchema(keySchema)
              .createWriterFunc(GenericParquetWriter::create)
              .equalityFieldIds(field.fieldId())
              .overwrite()
              .buildEqualityWriter();
      org.apache.iceberg.data.GenericRecord record =
          org.apache.iceberg.data.GenericRecord.create(keySchema);
      try (EqualityDeleteWriter<Record> closing = writer) {
        for (Object key : keys) {
          closing.write(record.copy(column, key));
        }
      }
      table.newRowDelta().addDeletes(writer.toDeleteFile()).commit();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  public static int formatVersion(Table table) {
    return ((org.apache.iceberg.HasTableOperations) table).operations().current().formatVersion();
  }

  private static OutputFile newFile(Table table, String kind) {
    String name = kind + "-" + UUID.randomUUID() + ".parquet";
    return table.io().newOutputFile(table.locationProvider().newDataLocation(name));
  }
}
