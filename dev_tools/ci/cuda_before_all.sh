#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Runs as root in the manylinux_2_28 x86_64 build container. Only developer
# packages are installed; no NVIDIA driver or GPU is needed to build a wheel.
set -euo pipefail

test "$(uname -m)" = x86_64
dnf install -y dnf-plugins-core gcc-c++
dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 cuda-cccl-12-9 \
  libcublas-devel-12-9

# Compile against the oldest supported cuStateVec ABI/API. This wheel contains
# headers and the versioned library, but no unversioned libcustatevec.so link.
/opt/python/cp312-cp312/bin/python -m pip install --no-deps --only-binary=:all: \
  --target /opt/qsim-cuda12 custatevec-cu12==1.11.0
test -f /opt/qsim-cuda12/cuquantum/include/custatevecEx.h
test -f /opt/qsim-cuda12/cuquantum/lib/libcustatevec.so.1
/usr/local/cuda-12.9/bin/nvcc --version
