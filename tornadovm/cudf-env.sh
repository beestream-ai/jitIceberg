#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Sets up libcudf from NVIDIA's Python wheels (no conda, no Python environment): downloads the
# libcudf-cu12 wheels and their dependencies once into DIR, unpacks them, and exports CUDF_HOME and
# RMM_HOME (to build TornadoVM's cuDF shim) and LD_LIBRARY_PATH (to run it).
#
#   source tornadovm/cudf-env.sh [DIR]       DIR defaults to ~/.iceberg-tornadovm/cudf
#
# CUDF_VERSION picks the release (default 26.8.*, the one this module is tested with).

CUDF_VERSION="${CUDF_VERSION:-26.8.*}"
CUDF_DIR="${1:-$HOME/.iceberg-tornadovm/cudf}"

if [ ! -s "$CUDF_DIR/ld_library_path" ]; then
    echo "Installing libcudf $CUDF_VERSION into $CUDF_DIR" >&2
    mkdir -p "$CUDF_DIR/wheels"
    python3 -m pip download --quiet --timeout 120 --retries 10 --only-binary=:all: --python-version 3.12 \
        --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
        --extra-index-url https://pypi.nvidia.com -d "$CUDF_DIR/wheels" "libcudf-cu12==$CUDF_VERSION" \
        || { echo "cudf-env.sh: downloading the libcudf wheels failed" >&2; return 1 2>/dev/null || exit 1; }
    for wheel in "$CUDF_DIR"/wheels/*.whl; do
        python3 -m zipfile -e "$wheel" "$CUDF_DIR"
    done
    rm -rf "$CUDF_DIR/wheels"
    find "$CUDF_DIR" -name '*.so*' -type f -printf '%h\n' | sort -u | paste -sd: > "$CUDF_DIR/ld_library_path"
fi

export CUDF_HOME="$CUDF_DIR/libcudf"
export RMM_HOME="$CUDF_DIR/librmm"
export LD_LIBRARY_PATH="$(cat "$CUDF_DIR/ld_library_path")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
