# Copyright 2026 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility tests for the native GPU backend detection functions."""

import subprocess
import sys
import types

import pytest

import qsimcirq
from qsimcirq import qsim_decide


def test_detect_instructions():
    assert qsim_decide.detect_instructions() in (0, 1, 2, 3)


def test_detect_matches_imported_backends():
    gpu = qsimcirq.qsim_gpu
    expected_gpu = 10
    if gpu is not None:
        expected_gpu = 3 if gpu.__name__.endswith("qsim_hip") else 0
    assert qsim_decide.detect_gpu() == expected_gpu
    assert qsim_decide.detect_custatevec() == (
        1 if qsimcirq.qsim_custatevec is not None else 11
    )
    assert qsim_decide.detect_custatevecex() == (
        2 if qsimcirq.qsim_custatevecex is not None else 12
    )


@pytest.mark.parametrize("backend, expected", [(None, 10), ("cuda", 0), ("hip", 3)])
def test_detect_gpu_tracks_backend(monkeypatch, backend, expected):
    module = types.ModuleType(f"qsimcirq.qsim_{backend}") if backend else None
    monkeypatch.setattr(qsimcirq, "qsim_gpu", module)
    assert qsim_decide.detect_gpu() == expected


@pytest.mark.parametrize(
    "attribute, function, available, missing",
    [
        ("qsim_custatevec", qsim_decide.detect_custatevec, 1, 11),
        ("qsim_custatevecex", qsim_decide.detect_custatevecex, 2, 12),
    ],
)
def test_custatevec_detected_independently(
    monkeypatch, attribute, function, available, missing
):
    monkeypatch.setattr(qsimcirq, "qsim_gpu", None)
    monkeypatch.setattr(qsimcirq, attribute, None)
    assert function() == missing
    monkeypatch.setattr(qsimcirq, attribute, types.ModuleType(attribute))
    assert function() == available


def test_native_extension_does_not_import_package(tmp_path):
    # Load the real extension by filename, then simulate a partially imported
    # package. Neither case may recursively import qsimcirq or query a driver.
    script = """
import importlib.util
import sys
import types
spec = importlib.util.spec_from_file_location("qsim_decide", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "qsimcirq" not in sys.modules
assert module.detect_gpu() == 10
assert module.detect_custatevec() == 11
assert module.detect_custatevecex() == 12
assert "qsimcirq" not in sys.modules
sys.modules["qsimcirq"] = types.ModuleType("qsimcirq")
assert module.detect_gpu() == 10
assert module.detect_custatevec() == 11
assert module.detect_custatevecex() == 12
"""
    subprocess.run(
        [sys.executable, "-c", script, qsim_decide.__file__],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
