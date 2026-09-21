#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
wheel=$1
destination=$2
project=$3

# NVIDIA libraries remain external dependencies supplied by the cuda12 extra.
# cuBLAS loads cuBLASLt transitively, so neither library belongs in qsimcirq.
auditwheel repair --plat manylinux_2_28_x86_64 \
  --exclude libcudart.so.12 --exclude libcublas.so.12 \
  --exclude libcublasLt.so.12 --exclude libcustatevec.so.1 \
  --wheel-dir "$destination" "$wheel"

# cibuildwheel gives each repair invocation its own destination directory.
for repaired in "$destination"/*.whl; do
  python "$project/dev_tools/check_gpu_wheel.py" "$repaired"
done
