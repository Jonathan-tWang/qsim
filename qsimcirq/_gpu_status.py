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

"""Diagnostics explaining whether, and why not, qsimcirq's GPU backends loaded.

`qsimcirq.gpu_status()` combines the runtime probe exposed by the `qsim_decide`
extension (`gpu_probe()`, qsim issue #601 contract C1) with the ImportErrors
recorded by `qsimcirq/__init__.py` while loading the optional `qsim_cuda`,
`qsim_custatevec` and `qsim_custatevecex` modules. The `reason` strings below
are part of the public contract (C2): tests and documentation match on their
exact prefixes, so change them only together with those.
"""

import dataclasses
import platform
import sys
from typing import Any, Callable, Dict, Optional

REASON_PLATFORM = "GPU support is only available on Linux x86_64."
REASON_NO_DRIVER = "No NVIDIA driver found (libcuda.so.1)."
REASON_DRIVER_TOO_OLD = "NVIDIA driver too old for CUDA 12 (need >= 525)."
REASON_NO_DEVICES = "No CUDA devices visible."
REASON_NO_RUNTIME = 'CUDA runtime not installed; run: pip install "qsimcirq[cuda12]".'
REASON_NO_CUSTATEVEC = 'cuStateVec not installed; run: pip install "qsimcirq[cuda12]".'
# Used by QSimSimulator for gpu_mode 1/2 when the CUDA backend itself loaded.
REASON_CUSTATEVEC_TOO_OLD = (
    "cuStateVec too old for the Ex API (need cuStateVec >= 1.10)."
)
# Terminal fallback when nothing above applies (e.g. the module was unloaded
# after import); not one of the contract strings, but never None.
REASON_NOT_LOADED = "qsimcirq.qsim_cuda is not loaded."

# cuDriverGetVersion() encodes CUDA 12.0 as 12000; driver 525 is the first to
# report it.
_MIN_DRIVER_VERSION = 12000

# Codes returned by qsim_decide.detect_gpu() / detect_custatevec(); see C1.
_CUDA = 0
_NO_GPU = 10
_CUSTATEVEC = 1


@dataclasses.dataclass(frozen=True)
class GpuStatus:
    """Snapshot of qsimcirq's GPU support on this machine.

    Attributes:
        available: whether `qsimcirq.qsim_gpu` loaded (i.e. `use_gpu=True`
            can work).
        backend: `"cuda"`, `"hip"`, or `None` when unavailable.
        custatevec: whether `qsimcirq.qsim_custatevec` loaded (`gpu_mode=1`).
        custatevecex: whether `qsimcirq.qsim_custatevecex` loaded
            (`gpu_mode=2`).
        reason: `None` when available; otherwise one human-readable sentence
            saying why not, starting with one of the `REASON_*` prefixes in
            this module.
        probe: the raw `qsim_decide.gpu_probe()` dictionary, or `{}` when the
            probe is unavailable (older `qsim_decide`) or the backend is HIP.
    """

    available: bool
    backend: Optional[str]
    custatevec: bool
    custatevecex: bool
    reason: Optional[str]
    probe: Dict[str, Any]


def gpu_status() -> GpuStatus:
    """Return a `GpuStatus` describing the GPU backends of this qsimcirq install.

    Never raises: every failure of the underlying probe degrades to a status
    with an empty `probe` and a best-effort `reason`.
    """
    # Imported lazily: this module is imported by qsimcirq/__init__.py after
    # the GPU loaders have run, and reading the attributes at call time keeps
    # the function honest if they are later replaced (e.g. by tests).
    import qsimcirq
    from qsimcirq import qsim_decide

    qsim_gpu = qsimcirq.qsim_gpu
    errors = dict(getattr(qsimcirq, "_import_errors", {}))
    probe = _run_probe(qsim_decide)

    available = qsim_gpu is not None
    backend = _backend(qsim_gpu)
    if backend == "hip":
        probe = {}
    reason = None if available else _reason(probe, errors, qsim_decide)
    return GpuStatus(
        available=available,
        backend=backend,
        custatevec=qsimcirq.qsim_custatevec is not None,
        custatevecex=qsimcirq.qsim_custatevecex is not None,
        reason=reason,
        probe=probe,
    )


def _run_probe(qsim_decide: Any) -> Dict[str, Any]:
    """Call `qsim_decide.gpu_probe()` if it exists; `{}` on any problem."""
    probe_fn: Optional[Callable[[], Any]] = getattr(qsim_decide, "gpu_probe", None)
    if probe_fn is None:
        return {}
    try:
        result = probe_fn()
    except Exception:  # C1 says the probe never raises; be robust anyway.
        return {}
    return dict(result) if isinstance(result, dict) else {}


def _backend(qsim_gpu: Any) -> Optional[str]:
    if qsim_gpu is None:
        return None
    name = getattr(qsim_gpu, "__name__", "")
    return "hip" if name.endswith("qsim_hip") else "cuda"


def _host_platform() -> str:
    """The C1 `platform` value for this interpreter (used when no probe)."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in ("win32", "cygwin"):
        return "windows"
    return "other"


def _is_x86_64() -> bool:
    return platform.machine().lower() in ("x86_64", "amd64")


def _detect(qsim_decide: Any, name: str, default: int) -> int:
    fn = getattr(qsim_decide, name, None)
    if fn is None:
        return default
    try:
        return int(fn())
    except Exception:
        return default


def _reason(probe: Dict[str, Any], errors: Dict[str, str], qsim_decide: Any) -> str:
    """Pick the single most actionable reason, in the C2 priority order.

    platform -> driver missing -> driver too old -> no devices -> runtime
    missing -> import error -> cuStateVec missing.
    """
    platform_name = probe.get("platform") if probe else _host_platform()
    if platform_name != "linux":
        return REASON_PLATFORM
    # The wheels only ship GPU modules for x86_64, but a source build on another
    # Linux architecture (e.g. aarch64 with a driver) is diagnosed normally.
    if not _is_x86_64() and not (probe and probe.get("driver_library_found")):
        return REASON_PLATFORM

    if probe:
        driver_version = probe.get("driver_version")
        if not probe.get("driver_library_found") or driver_version is None:
            return REASON_NO_DRIVER
        if driver_version < _MIN_DRIVER_VERSION:
            return REASON_DRIVER_TOO_OLD
        if not probe.get("device_count"):
            return REASON_NO_DEVICES
        if not probe.get("cuda_runtime_found"):
            return REASON_NO_RUNTIME

    for module in ("qsim_cuda", "qsim_hip"):
        if module in errors:
            return f"qsimcirq.{module} failed to import: {errors[module].rstrip('.')}."

    if probe:
        if not probe.get("custatevec_found"):
            return REASON_NO_CUSTATEVEC
    else:
        # No gpu_probe() (qsim_decide predates contract C1): the integer
        # codes are all the diagnostics available.
        if _detect(qsim_decide, "detect_gpu", _NO_GPU) != _CUDA:
            return REASON_NO_DEVICES
        if _detect(qsim_decide, "detect_custatevec", 0) != _CUSTATEVEC:
            return REASON_NO_CUSTATEVEC

    return REASON_NOT_LOADED


def custatevec_reason(gpu_mode: int) -> Optional[str]:
    """Why `gpu_mode` 1 or 2 is unavailable although the CUDA backend loaded.

    `GpuStatus.reason` is `None` whenever `qsimcirq.qsim_gpu` loaded (contract
    C2), so `QSimSimulator` uses this to explain a missing cuStateVec or
    cuStateVecEx module. Returns `None` when nothing specific can be said
    (no probe, HIP backend, or the module is present).
    """
    import qsimcirq
    from qsimcirq import qsim_decide

    module = "qsim_custatevecex" if gpu_mode >= 2 else "qsim_custatevec"
    if getattr(qsimcirq, module) is not None:
        return None
    probe = _run_probe(qsim_decide)
    if probe and probe.get("hip_compiled") and not probe.get("driver_library_found"):
        return None
    if probe:
        if not probe.get("cublas_found") or not probe.get("custatevec_found"):
            return REASON_NO_CUSTATEVEC
        if gpu_mode >= 2 and not probe.get("custatevec_ex_found"):
            return REASON_CUSTATEVEC_TOO_OLD
    errors = dict(getattr(qsimcirq, "_import_errors", {}))
    if module in errors:
        return f"qsimcirq.{module} failed to import: {errors[module].rstrip('.')}."
    return None
