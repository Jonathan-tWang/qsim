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

"""Tests for the runtime GPU probe in the qsim_decide extension module."""

import subprocess
import sys
import time

import pytest

from qsimcirq import qsim_decide

# The GPUCapabilities enum values returned by the detect_* functions.
CUDA = 0
CUSTATEVEC = 1
CUSTATEVECEX = 2
HIP = 3
NO_GPU = 10
NO_CUSTATEVEC = 11
NO_CUSTATEVECEX = 12

# The keys of gpu_probe() and the Python types allowed for each value.
PROBE_KEY_TYPES: dict[str, tuple[type, ...]] = {
    "platform": (str,),
    "driver_library_found": (bool,),
    "driver_version": (int, type(None)),
    "device_count": (int,),
    "cuda_runtime_found": (bool,),
    "cuda_runtime_version": (int, type(None)),
    "cublas_found": (bool,),
    "custatevec_found": (bool,),
    "custatevec_ex_found": (bool,),
    "hip_compiled": (bool,),
    "errors": (list,),
}

FOUND_KEYS = [
    "driver_library_found",
    "cuda_runtime_found",
    "cublas_found",
    "custatevec_found",
    "custatevec_ex_found",
]


def expected_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in ("win32", "cygwin"):
        return "windows"
    return "other"


def test_gpu_probe_keys_and_types():
    probe = qsim_decide.gpu_probe()
    assert isinstance(probe, dict)
    assert set(probe) == set(PROBE_KEY_TYPES)
    for key, allowed in PROBE_KEY_TYPES.items():
        # bool is a subclass of int, so compare exact types.
        assert type(probe[key]) in allowed, f"{key}={probe[key]!r}"
    assert probe["platform"] in ("linux", "darwin", "windows", "other")
    assert probe["device_count"] >= 0
    assert all(isinstance(error, str) and error for error in probe["errors"])


def test_gpu_probe_platform_matches_host():
    assert qsim_decide.gpu_probe()["platform"] == expected_platform()


def test_gpu_probe_returns_a_fresh_dict():
    probe = qsim_decide.gpu_probe()
    probe["device_count"] = 99
    probe["errors"].append("mutated by test")
    fresh = qsim_decide.gpu_probe()
    assert fresh["device_count"] != 99
    assert "mutated by test" not in fresh["errors"]


def test_gpu_probe_repeated_calls_are_consistent():
    first = qsim_decide.gpu_probe()
    assert all(qsim_decide.gpu_probe() == first for _ in range(5))


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="Linux-only probe")
def test_gpu_probe_on_non_linux_platform():
    probe = qsim_decide.gpu_probe()
    assert probe["platform"] != "linux"
    assert probe["device_count"] == 0
    assert probe["driver_version"] is None
    assert probe["cuda_runtime_version"] is None
    assert not any(probe[key] for key in FOUND_KEYS)
    assert probe["errors"], "the errors list must explain why nothing was found"
    assert qsim_decide.detect_gpu() in (HIP, NO_GPU)
    assert qsim_decide.detect_custatevec() == NO_CUSTATEVEC
    assert qsim_decide.detect_custatevecex() == NO_CUSTATEVECEX


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only")
def test_gpu_probe_on_linux_without_driver():
    probe = qsim_decide.gpu_probe()
    if probe["driver_library_found"]:
        pytest.skip("an NVIDIA driver is installed on this machine")
    assert probe["platform"] == "linux"
    assert probe["driver_version"] is None
    assert probe["device_count"] == 0
    assert probe["errors"], "a missing driver must be reported in errors"
    assert any("libcuda.so.1" in error for error in probe["errors"])
    assert qsim_decide.detect_gpu() in (HIP, NO_GPU)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only")
def test_gpu_probe_on_linux_with_driver():
    probe = qsim_decide.gpu_probe()
    if not probe["driver_library_found"]:
        pytest.skip("no NVIDIA driver on this machine")
    assert probe["driver_version"] is not None
    assert probe["driver_version"] > 0


def test_gpu_probe_version_fields_follow_found_flags():
    probe = qsim_decide.gpu_probe()
    if not probe["driver_library_found"]:
        assert probe["driver_version"] is None
        assert probe["device_count"] == 0
    if not probe["cuda_runtime_found"]:
        assert probe["cuda_runtime_version"] is None
    if not probe["custatevec_found"]:
        assert not probe["custatevec_ex_found"]


def test_detect_gpu_is_consistent_with_probe():
    probe = qsim_decide.gpu_probe()
    cuda_usable = (
        probe["driver_library_found"]
        and probe["device_count"] > 0
        and probe["cuda_runtime_found"]
    )
    if cuda_usable:
        expected_gpu = CUDA
    elif probe["hip_compiled"]:
        expected_gpu = HIP
    else:
        expected_gpu = NO_GPU
    assert qsim_decide.detect_gpu() == expected_gpu

    custatevec_usable = (
        expected_gpu == CUDA and probe["cublas_found"] and probe["custatevec_found"]
    )
    expected_custatevec = CUSTATEVEC if custatevec_usable else NO_CUSTATEVEC
    assert qsim_decide.detect_custatevec() == expected_custatevec

    custatevecex_usable = custatevec_usable and probe["custatevec_ex_found"]
    expected_custatevecex = CUSTATEVECEX if custatevecex_usable else NO_CUSTATEVECEX
    assert qsim_decide.detect_custatevecex() == expected_custatevecex


def test_detect_instructions_is_unchanged():
    assert qsim_decide.detect_instructions() in (0, 1, 2, 3)


def test_gpu_probe_cached_calls_are_fast():
    qsim_decide.gpu_probe()  # Make sure the cached result exists.
    start = time.perf_counter()
    for _ in range(100):
        qsim_decide.gpu_probe()
        qsim_decide.detect_gpu()
    assert time.perf_counter() - start < 0.1


# Loads the extension module directly from its file (bypassing the qsimcirq
# package, whose import already fills the probe cache) and times the first,
# uncached probe in a fresh interpreter.
_COLD_PROBE_SCRIPT = """
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("qsim_decide", sys.argv[1])
module = importlib.util.module_from_spec(spec)
start = time.perf_counter()
spec.loader.exec_module(module)
probe = module.gpu_probe()
print(time.perf_counter() - start)
"""


def test_gpu_probe_first_call_is_fast():
    if qsim_decide.gpu_probe()["device_count"] > 0:
        pytest.skip("timing bound only applies without a GPU")
    output = subprocess.run(
        [sys.executable, "-c", _COLD_PROBE_SCRIPT, qsim_decide.__file__],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    assert float(output) < 0.1
