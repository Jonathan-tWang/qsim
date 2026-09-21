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

"""Diagnostics for optional GPU imports and simulator backend errors."""

import importlib
import types

import pytest

import qsimcirq
import qsimcirq.qsim_simulator
from qsimcirq import _gpu_status

_MODULES = ("qsim_gpu", "qsim_custatevec", "qsim_custatevecex")


@pytest.fixture
def no_gpu_modules(monkeypatch):
    for name in _MODULES:
        monkeypatch.setattr(qsimcirq, name, None)
        monkeypatch.setattr(qsimcirq.qsim_simulator, name, None)
    monkeypatch.setattr(qsimcirq, "_import_errors", {})
    monkeypatch.setattr(qsimcirq, "_missing_modules", set())
    monkeypatch.setattr(qsimcirq, "_gpu_backend", "cuda")


def test_absent_module_is_distinct_from_missing_dependency(monkeypatch, no_gpu_modules):
    def absent(name):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    monkeypatch.setattr(importlib, "import_module", absent)
    assert qsimcirq._try_import("qsim_cuda") is None
    assert "qsim_cuda" in qsimcirq._missing_modules
    assert "was not built or installed" in _gpu_status.gpu_backend_reason(0)
    assert "pip install" not in _gpu_status.gpu_backend_reason(0)

    def missing_dependency(name):
        raise ModuleNotFoundError("No module named 'dependency'", name="dependency")

    monkeypatch.setattr(importlib, "import_module", missing_dependency)
    assert qsimcirq._try_import("qsim_cuda") is None
    assert "qsim_cuda" not in qsimcirq._missing_modules
    reason = _gpu_status.gpu_backend_reason(0)
    assert "failed to import: No module named 'dependency'" in reason


def test_success_clears_previous_error(monkeypatch, no_gpu_modules):
    qsimcirq._missing_modules.add("qsim_cuda")
    qsimcirq._import_errors["qsim_cuda"] = "old error"
    module = types.ModuleType("qsimcirq.qsim_cuda")
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    assert qsimcirq._try_import("qsim_cuda") is module
    assert "qsim_cuda" not in qsimcirq._import_errors
    assert "qsim_cuda" not in qsimcirq._missing_modules


def test_unexpected_import_failures_propagate(monkeypatch, no_gpu_modules):
    def broken(name):
        raise RuntimeError("extension initialization bug")

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(RuntimeError, match="extension initialization bug"):
        qsimcirq._try_import("qsim_cuda")


@pytest.mark.parametrize(
    "mode, module",
    [
        (0, "qsim_cuda"),
        (1, "qsim_custatevec"),
        (2, "qsim_custatevecex"),
        (3, "qsim_custatevecex"),
    ],
)
def test_simulator_reports_requested_backend(no_gpu_modules, mode, module):
    qsimcirq._import_errors.update(
        qsim_cuda="libcudart.so.12 missing",
        qsim_custatevec="libcustatevec.so.1 missing",
        qsim_custatevecex=(
            "undefined symbol: custatevecExStateVectorCreateSingleProcess"
        ),
    )
    with pytest.raises(ValueError) as excinfo:
        qsimcirq.QSimSimulator(
            qsim_options=qsimcirq.QSimOptions(use_gpu=True, gpu_mode=mode)
        )
    assert f"qsimcirq.{module} failed to import" in str(excinfo.value)
    assert qsimcirq._import_errors[module] in str(excinfo.value)


def test_hip_failure_has_hip_diagnostic(monkeypatch, no_gpu_modules):
    monkeypatch.setattr(qsimcirq, "_gpu_backend", "hip")
    qsimcirq._import_errors["qsim_hip"] = (
        "libamdhip64.so.6: cannot open shared object file"
    )
    reason = _gpu_status.gpu_backend_reason(0)
    assert "qsim_hip failed to import" in reason
    assert "libamdhip64.so.6" in reason
    assert "NVIDIA" not in reason
    assert "pip install" not in reason
    assert "HIP build" in _gpu_status.gpu_backend_reason(1)


def test_available_backend_has_no_error(monkeypatch, no_gpu_modules):
    monkeypatch.setattr(qsimcirq, "qsim_gpu", types.ModuleType("qsimcirq.qsim_cuda"))
    assert _gpu_status.gpu_backend_reason(0) is None
