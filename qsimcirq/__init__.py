# pylint: disable=wrong-import-position
# Copyright 2019 Google LLC. All Rights Reserved.
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

import importlib

from qsimcirq import qsim_decide

# ImportError messages from loading the optional GPU modules, keyed by module
# name ("qsim_cuda", "qsim_hip", "qsim_custatevec", "qsim_custatevecex").
# Read by the private diagnostics appended to QSimSimulator errors.
_import_errors: dict[str, str] = {}
_missing_modules: set[str] = set()
_gpu_backend = "cuda"


def _load_simd_qsim():
    instr = qsim_decide.detect_instructions()
    if instr == 0:
        qsim = importlib.import_module("qsimcirq.qsim_avx512")
    elif instr == 1:
        qsim = importlib.import_module("qsimcirq.qsim_avx2")
    elif instr == 2:
        qsim = importlib.import_module("qsimcirq.qsim_sse")
    else:
        qsim = importlib.import_module("qsimcirq.qsim_basic")
    return qsim


def _try_import(name: str):
    """Load an optional backend and preserve the loader's diagnostic on failure."""
    qualified_name = f"qsimcirq.{name}"
    _import_errors.pop(name, None)
    _missing_modules.discard(name)
    try:
        return importlib.import_module(qualified_name)
    except ImportError as error:
        _import_errors[name] = str(error)
        # Distinguish an absent extension from one with missing dependencies.
        if isinstance(error, ModuleNotFoundError) and error.name == qualified_name:
            _missing_modules.add(name)
        return None


def _load_qsim_gpu():
    global _gpu_backend
    # A HIP installation keeps its backend preference even if a dependency is
    # missing, including on machines that also have an NVIDIA driver installed.
    hip = _try_import("qsim_hip")
    if "qsim_hip" not in _missing_modules:
        _gpu_backend = "hip"
        return hip
    _gpu_backend = "cuda"
    return _try_import("qsim_cuda")


def _load_qsim_custatevec():
    return None if _gpu_backend == "hip" else _try_import("qsim_custatevec")


def _load_qsim_custatevecex():
    return None if _gpu_backend == "hip" else _try_import("qsim_custatevecex")


qsim = _load_simd_qsim()
qsim_gpu = _load_qsim_gpu()
qsim_custatevec = _load_qsim_custatevec()
qsim_custatevecex = _load_qsim_custatevecex()

# Note: the following imports must remain at the bottom of this file.

from qsimcirq._version import __version__

from .qsim_circuit import QSimCircuit, add_op_to_circuit, add_op_to_opstring
from .qsim_simulator import QSimOptions, QSimSimulator
from .qsimh_simulator import QSimhSimulator

__all__ = [
    "QSimCircuit",
    "add_op_to_circuit",
    "add_op_to_opstring",
    "QSimOptions",
    "QSimSimulator",
    "QSimhSimulator",
    "qsim",
    "qsim_gpu",
    "qsim_custatevec",
    "qsim_custatevecex",
    "__version__",
]
