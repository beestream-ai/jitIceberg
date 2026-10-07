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
import org.apache.iceberg.Schema;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.relocated.com.google.common.collect.Maps;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.Types;
import org.apache.parquet.schema.MessageType;
import uk.ac.manchester.tornado.cudf.enums.CudfType;

/**
 * Where each projected column lives on the device: which typed buffer, and which slot of it.
 *
 * <p>cuDF writes the k-th column of a physical type at element {@code k * capacity} of that type's
 * buffer. The order of the read is chosen so that the columns a library call reads from offset 0
 * are slot 0: the equality-delete key first (the set probe reads it), then the group key.
 */
final class ScanLayout {

  private final List<Types.NestedField> fields = Lists.newArrayList();
  private final List<CudfType> types = Lists.newArrayList();
  private final Map<Integer, Integer> slotById = Maps.newHashMap();
  private final Map<Integer, Integer> readIndexById = Maps.newHashMap();
  private final int[] slotsPerType = new int[3];
  private boolean anyOptional = false;

  ScanLayout(Schema schema, Integer equalityKeyId, String groupColumn, List<String> columns) {
    if (equalityKeyId != null) {
      add(schema.findField(equalityKeyId));
    }
    if (groupColumn != null) {
      add(field(schema, groupColumn));
    }
    for (String column : columns) {
      add(field(schema, column));
    }
  }

  private static Types.NestedField field(Schema schema, String name) {
    Types.NestedField field = schema.findField(name);
    if (field == null) {
      throw new IllegalArgumentException("No column " + name + " in " + schema);
    }
    return field;
  }

  private void add(Types.NestedField field) {
    if (slotById.containsKey(field.fieldId())) {
      return;
    }
    CudfType type = cudfType(field);
    readIndexById.put(field.fieldId(), fields.size());
    slotById.put(field.fieldId(), slotsPerType[type.code()]++);
    fields.add(field);
    types.add(type);
    anyOptional |= field.isOptional();
  }

  /** The device type of an Iceberg primitive, or a refusal: nothing here is ever cast. */
  static CudfType cudfType(Types.NestedField field) {
    Type.TypeID id = field.type().typeId();
    return switch (id) {
      case INTEGER, DATE -> CudfType.INT32;
      case LONG, TIMESTAMP, TIME -> CudfType.INT64;
      case DOUBLE -> CudfType.FLOAT64;
      default ->
          throw new UnsupportedOperationException(
              "Column "
                  + field.name()
                  + " of type "
                  + field.type()
                  + " cannot be read on the device");
    };
  }

  List<Types.NestedField> fields() {
    return fields;
  }

  CudfType[] types() {
    return types.toArray(new CudfType[0]);
  }

  int slots(CudfType type) {
    return slotsPerType[type.code()];
  }

  int slot(int fieldId) {
    return slotById.get(fieldId);
  }

  /** The column's position in the read, which is also its slot in the validity bytes. */
  int readIndex(int fieldId) {
    return readIndexById.get(fieldId);
  }

  CudfType type(int fieldId) {
    return types.get(readIndex(fieldId));
  }

  boolean anyOptional() {
    return anyOptional;
  }

  /**
   * The schema index of each projected column in one data file, resolved by Iceberg field id the
   * way Iceberg's own readers resolve them, so renamed and reordered columns read correctly.
   */
  int[] parquetColumns(MessageType fileSchema, String location) {
    int[] columns = new int[fields.size()];
    for (int c = 0; c < fields.size(); c++) {
      Types.NestedField field = fields.get(c);
      columns[c] = -1;
      for (int i = 0; i < fileSchema.getFieldCount(); i++) {
        org.apache.parquet.schema.Type column = fileSchema.getType(i);
        if (column.getId() != null && column.getId().intValue() == field.fieldId()) {
          columns[c] = i;
          break;
        }
      }
      if (columns[c] < 0) {
        // A column added after the file was written reads as null; that needs a default the
        // device read cannot produce, so the scan declines rather than guess.
        throw new UnsupportedOperationException(
            "Data file " + location + " has no column with field id " + field.fieldId());
      }
    }
    return columns;
  }
}
