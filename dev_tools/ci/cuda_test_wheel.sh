#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

# Run in a clean Linux x86_64 environment with Python matching the wheel ABI.
# Pass --gpu to require real execution by all three backends, without skips.
set -euo pipefail
wheel=$(realpath "$1")
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mode=with-extras
if [[ ${2:-} == --gpu ]]; then
  mode=gpu
elif [[ $# -ne 1 ]]; then
  echo "Usage: $0 WHEEL [--gpu]" >&2
  exit 2
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python -m venv "$work/venv"
python="$work/venv/bin/python"
unset LD_LIBRARY_PATH PYTHONPATH
cd "$work"
"$python" -m pip install --upgrade pip
"$python" -m pip install "$wheel" pytest
"$python" -m pip check
QSIM_GPU_WHEEL_TEST=without-extras "$python" -m pytest --import-mode=importlib -v \
  "$project/qsimcirq_tests/gpu_wheel_test.py" \
  "$project/qsimcirq_tests/qsimcirq_test.py"

# Upgrade the very same artifact, rather than resolving qsimcirq from PyPI.
"$python" -m pip install "$wheel[cuda12]"
"$python" -m pip check
QSIM_GPU_WHEEL_TEST="$mode" "$python" -m pytest --import-mode=importlib -v \
  "$project/qsimcirq_tests/gpu_wheel_test.py"

if [[ $mode == gpu ]]; then
  QSIM_GPU_WHEEL_TEST=gpu "$python" -m pytest --import-mode=importlib -v \
    "$project/qsimcirq_tests/qsimcirq_test.py"
fi
