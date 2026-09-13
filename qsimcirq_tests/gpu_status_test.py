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

"""Tests for qsimcirq.gpu_status() and the GPU import guards (qsim #601, C2).

Every test runs on a CPU-only machine: `qsim_decide.gpu_probe()` and the GPU
module attributes are replaced with `monkeypatch`.
"""

import dataclasses
import importlib
import types

import pytest

import qsimcirq
import qsimcirq.qsim_simulator
from qsimcirq import _gpu_status, qsim_decide

_MODULES = ("qsim_gpu", "qsim_custatevec", "qsim_custatevecex")
_LIBCUDART_ERROR = "libcudart.so.12: cannot open shared object file"


def _healthy_probe(**overrides):
    """A C1 probe dict describing a machine where everything is present."""
    probe = {
        "platform": "linux",
        "driver_library_found": True,
        "driver_version": 12080,
        "device_count": 1,
        "cuda_runtime_found": True,
        "cuda_runtime_version": 12090,
        "cublas_found": True,
        "custatevec_found": True,
        "custatevec_ex_found": True,
        "hip_compiled": False,
        "errors": [],
    }
    probe.update(overrides)
    return probe


def _set_probe(monkeypatch, probe):
    monkeypatch.setattr(qsim_decide, "gpu_probe", lambda: dict(probe), raising=False)


def _set_modules(monkeypatch, gpu=None, custatevec=None, custatevecex=None):
    for name, value in zip(_MODULES, (gpu, custatevec, custatevecex)):
        monkeypatch.setattr(qsimcirq, name, value)
        monkeypatch.setattr(qsimcirq.qsim_simulator, name, value)


@pytest.fixture
def linux_x86_64(monkeypatch):
    """Pretend the host is Linux x86_64 so the platform check passes."""
    monkeypatch.setattr(_gpu_status, "_host_platform", lambda: "linux")
    monkeypatch.setattr(_gpu_status, "_is_x86_64", lambda: True)


@pytest.fixture
def no_gpu_modules(monkeypatch, linux_x86_64):
    """No GPU module loaded and no import errors recorded."""
    _set_modules(monkeypatch)
    monkeypatch.setattr(qsimcirq, "_import_errors", {})


def test_gpu_status_is_exported():
    assert "gpu_status" in qsimcirq.__all__
    assert "GpuStatus" in qsimcirq.__all__
    assert qsimcirq.GpuStatus is _gpu_status.GpuStatus
    assert isinstance(qsimcirq.gpu_status(), qsimcirq.GpuStatus)


def test_gpu_status_on_this_machine():
    status = qsimcirq.gpu_status()
    assert status.available == (qsimcirq.qsim_gpu is not None)
    assert status.custatevec == (qsimcirq.qsim_custatevec is not None)
    assert status.custatevecex == (qsimcirq.qsim_custatevecex is not None)
    assert isinstance(status.probe, dict)
    if status.available:
        assert status.backend in ("cuda", "hip")
        assert status.reason is None
    else:
        assert status.backend is None
        assert isinstance(status.reason, str) and status.reason
    assert status == qsimcirq.gpu_status()  # deterministic


def test_gpu_status_is_frozen():
    status = qsimcirq.gpu_status()
    with pytest.raises(dataclasses.FrozenInstanceError):
        status.available = not status.available  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"platform": "darwin"}, _gpu_status.REASON_PLATFORM),
        ({"platform": "windows"}, _gpu_status.REASON_PLATFORM),
        ({"driver_library_found": False}, _gpu_status.REASON_NO_DRIVER),
        ({"driver_version": None}, _gpu_status.REASON_NO_DRIVER),
        ({"driver_version": 11080}, _gpu_status.REASON_DRIVER_TOO_OLD),
        ({"device_count": 0}, _gpu_status.REASON_NO_DEVICES),
        ({"cuda_runtime_found": False}, _gpu_status.REASON_NO_RUNTIME),
        ({"custatevec_found": False}, _gpu_status.REASON_NO_CUSTATEVEC),
    ],
)
def test_reason_branches(monkeypatch, no_gpu_modules, overrides, expected):
    probe = _healthy_probe(**overrides)
    _set_probe(monkeypatch, probe)
    status = qsimcirq.gpu_status()
    assert not status.available
    assert status.backend is None
    assert status.reason == expected
    assert status.probe == probe


def test_reason_strings_match_contract_c2():
    assert _gpu_status.REASON_NO_DRIVER == "No NVIDIA driver found (libcuda.so.1)."
    assert _gpu_status.REASON_DRIVER_TOO_OLD == (
        "NVIDIA driver too old for CUDA 12 (need >= 525)."
    )
    assert _gpu_status.REASON_NO_DEVICES == "No CUDA devices visible."
    assert _gpu_status.REASON_NO_RUNTIME == (
        'CUDA runtime not installed; run: pip install "qsimcirq[cuda12]".'
    )
    assert _gpu_status.REASON_NO_CUSTATEVEC == (
        'cuStateVec not installed; run: pip install "qsimcirq[cuda12]".'
    )
    assert (
        _gpu_status.REASON_PLATFORM == "GPU support is only available on Linux x86_64."
    )


def test_reason_priority_order(monkeypatch, no_gpu_modules):
    """When several things are wrong, the earliest in the C2 chain wins."""
    monkeypatch.setitem(qsimcirq._import_errors, "qsim_cuda", _LIBCUDART_ERROR)
    everything_wrong = _healthy_probe(
        driver_library_found=False,
        driver_version=None,
        device_count=0,
        cuda_runtime_found=False,
        custatevec_found=False,
    )
    chain = [
        ("driver_library_found", True, _gpu_status.REASON_NO_DRIVER),
        ("driver_version", 11000, _gpu_status.REASON_NO_DRIVER),
        ("driver_version", 12000, _gpu_status.REASON_DRIVER_TOO_OLD),
        ("device_count", 2, _gpu_status.REASON_NO_DEVICES),
        ("cuda_runtime_found", True, _gpu_status.REASON_NO_RUNTIME),
    ]
    probe = everything_wrong
    for key, fixed_value, expected_before_fix in chain:
        _set_probe(monkeypatch, probe)
        assert qsimcirq.gpu_status().reason == expected_before_fix
        probe = dict(probe, **{key: fixed_value})
    # Driver, devices and runtime are fine: the import error comes next ...
    _set_probe(monkeypatch, probe)
    assert qsimcirq.gpu_status().reason == (
        f"qsimcirq.qsim_cuda failed to import: {_LIBCUDART_ERROR}."
    )
    # ... and cuStateVec last.
    monkeypatch.delitem(qsimcirq._import_errors, "qsim_cuda")
    assert qsimcirq.gpu_status().reason == _gpu_status.REASON_NO_CUSTATEVEC


def test_platform_check_uses_probe_platform_and_host_arch(monkeypatch, no_gpu_modules):
    _set_probe(monkeypatch, _healthy_probe(driver_library_found=False))
    monkeypatch.setattr(_gpu_status, "_is_x86_64", lambda: False)
    assert qsimcirq.gpu_status().reason == _gpu_status.REASON_PLATFORM
    monkeypatch.setattr(_gpu_status, "_is_x86_64", lambda: True)
    assert qsimcirq.gpu_status().reason == _gpu_status.REASON_NO_DRIVER


def test_load_qsim_gpu_records_import_error(monkeypatch, no_gpu_modules):
    """A GPU module whose shared libraries are missing yields None, not a crash."""
    attempted = []

    def fake_import(name):
        attempted.append(name)
        raise ImportError(_LIBCUDART_ERROR)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(qsim_decide, "detect_gpu", lambda: 0)
    monkeypatch.setattr(qsim_decide, "detect_custatevec", lambda: 1)
    monkeypatch.setattr(qsim_decide, "detect_custatevecex", lambda: 2)

    assert qsimcirq._load_qsim_gpu() is None
    assert qsimcirq._load_qsim_custatevec() is None
    assert qsimcirq._load_qsim_custatevecex() is None
    assert attempted == [
        "qsimcirq.qsim_cuda",
        "qsimcirq.qsim_custatevec",
        "qsimcirq.qsim_custatevecex",
    ]
    assert qsimcirq._import_errors == {
        "qsim_cuda": _LIBCUDART_ERROR,
        "qsim_custatevec": _LIBCUDART_ERROR,
        "qsim_custatevecex": _LIBCUDART_ERROR,
    }

    _set_probe(monkeypatch, _healthy_probe())
    status = qsimcirq.gpu_status()
    assert status.available is False
    assert status.reason == f"qsimcirq.qsim_cuda failed to import: {_LIBCUDART_ERROR}."


def test_load_qsim_gpu_records_hip_import_error(monkeypatch, no_gpu_modules):
    def fake_import(name):
        raise ImportError(f"{name}: libamdhip64.so.6: cannot open shared object file")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(qsim_decide, "detect_gpu", lambda: 3)
    assert qsimcirq._load_qsim_gpu() is None
    assert "qsim_hip" in qsimcirq._import_errors
    _set_probe(monkeypatch, _healthy_probe(hip_compiled=True))
    assert qsimcirq.gpu_status().reason.startswith(
        "qsimcirq.qsim_hip failed to import: qsimcirq.qsim_hip: libamdhip64.so.6"
    )


def test_import_error_reason_has_single_trailing_period(monkeypatch, no_gpu_modules):
    monkeypatch.setitem(qsimcirq._import_errors, "qsim_cuda", "No module named 'x'.")
    _set_probe(monkeypatch, _healthy_probe())
    assert qsimcirq.gpu_status().reason == (
        "qsimcirq.qsim_cuda failed to import: No module named 'x'."
    )


def test_loaders_do_not_import_when_detect_says_no(monkeypatch, no_gpu_modules):
    def fake_import(name):
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(qsim_decide, "detect_gpu", lambda: 10)
    monkeypatch.setattr(qsim_decide, "detect_custatevec", lambda: 11)
    monkeypatch.setattr(qsim_decide, "detect_custatevecex", lambda: 12)
    assert qsimcirq._load_qsim_gpu() is None
    assert qsimcirq._load_qsim_custatevec() is None
    assert qsimcirq._load_qsim_custatevecex() is None
    assert qsimcirq._import_errors == {}


def test_fallback_without_gpu_probe(monkeypatch, no_gpu_modules):
    """qsim_decide without gpu_probe() (pre-C1 build): reasons come from codes."""
    monkeypatch.delattr(qsim_decide, "gpu_probe", raising=False)
    monkeypatch.setattr(qsim_decide, "detect_gpu", lambda: 10)
    monkeypatch.setattr(qsim_decide, "detect_custatevec", lambda: 11)

    status = qsimcirq.gpu_status()
    assert status.probe == {}
    assert status.available is False
    assert status.reason == _gpu_status.REASON_NO_DEVICES

    monkeypatch.setattr(qsim_decide, "detect_gpu", lambda: 0)
    assert qsimcirq.gpu_status().reason == _gpu_status.REASON_NO_CUSTATEVEC

    monkeypatch.setitem(qsimcirq._import_errors, "qsim_cuda", _LIBCUDART_ERROR)
    assert qsimcirq.gpu_status().reason == (
        f"qsimcirq.qsim_cuda failed to import: {_LIBCUDART_ERROR}."
    )

    monkeypatch.delitem(qsimcirq._import_errors, "qsim_cuda")
    monkeypatch.setattr(qsim_decide, "detect_custatevec", lambda: 1)
    assert isinstance(qsimcirq.gpu_status().reason, str)  # never None when unavailable


def test_fallback_without_gpu_probe_off_platform(monkeypatch, no_gpu_modules):
    monkeypatch.delattr(qsim_decide, "gpu_probe", raising=False)
    monkeypatch.setattr(_gpu_status, "_host_platform", lambda: "darwin")
    assert qsimcirq.gpu_status().reason == _gpu_status.REASON_PLATFORM


def test_probe_that_raises_or_misbehaves_is_tolerated(monkeypatch, no_gpu_modules):
    def bad_probe():
        raise RuntimeError("boom")

    monkeypatch.setattr(qsim_decide, "gpu_probe", bad_probe, raising=False)
    monkeypatch.setattr(qsim_decide, "detect_gpu", lambda: 10)
    status = qsimcirq.gpu_status()
    assert status.probe == {}
    assert status.reason == _gpu_status.REASON_NO_DEVICES

    monkeypatch.setattr(qsim_decide, "gpu_probe", lambda: "not a dict", raising=False)
    assert qsimcirq.gpu_status().probe == {}


def test_available_cuda(monkeypatch, linux_x86_64):
    fake_cuda = types.ModuleType("qsimcirq.qsim_cuda")
    fake_custatevec = types.ModuleType("qsimcirq.qsim_custatevec")
    _set_modules(monkeypatch, gpu=fake_cuda, custatevec=fake_custatevec)
    monkeypatch.setattr(qsimcirq, "_import_errors", {"qsim_custatevecex": "x"})
    probe = _healthy_probe(custatevec_ex_found=False)
    _set_probe(monkeypatch, probe)

    status = qsimcirq.gpu_status()
    assert status == qsimcirq.GpuStatus(
        available=True,
        backend="cuda",
        custatevec=True,
        custatevecex=False,
        reason=None,
        probe=probe,
    )


def test_available_hip(monkeypatch, linux_x86_64):
    _set_modules(monkeypatch, gpu=types.ModuleType("qsimcirq.qsim_hip"))
    monkeypatch.setattr(qsimcirq, "_import_errors", {})
    _set_probe(monkeypatch, _healthy_probe(hip_compiled=True))
    status = qsimcirq.gpu_status()
    assert status.available is True
    assert status.backend == "hip"
    assert status.reason is None
    assert status.probe == {}


@pytest.mark.parametrize(
    "gpu_mode, original_wording",
    [
        (0, "GPU execution requested, but not supported."),
        (1, "cuStateVec GPU execution requested, but not supported."),
        (2, "cuStateVecEx GPU execution requested, but not supported."),
    ],
)
def test_simulator_error_includes_reason(
    monkeypatch, no_gpu_modules, gpu_mode, original_wording
):
    _set_probe(monkeypatch, _healthy_probe(driver_library_found=False))
    options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=gpu_mode)
    with pytest.raises(ValueError) as excinfo:
        qsimcirq.QSimSimulator(qsim_options=options)
    message = str(excinfo.value)
    assert original_wording in message
    assert message.endswith(" " + _gpu_status.REASON_NO_DRIVER)


def test_simulator_error_without_reason_keeps_original_message(
    monkeypatch, linux_x86_64
):
    """gmode=1 with a working CUDA backend: reason is None, message unchanged."""
    _set_modules(monkeypatch, gpu=types.ModuleType("qsimcirq.qsim_cuda"))
    monkeypatch.setattr(qsimcirq, "_import_errors", {})
    _set_probe(monkeypatch, _healthy_probe(custatevec_found=False))
    options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
    with pytest.raises(ValueError) as excinfo:
        qsimcirq.QSimSimulator(qsim_options=options)
    assert str(excinfo.value).endswith("you may need to compile qsim locally.")
