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
# Read by qsimcirq.gpu_status(); private otherwise.
_import_errors: dict[str, str] = {}


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
    """Import `qsimcirq.<name>`; on ImportError record the message and return None.

    The GPU modules are optional: a missing shared library (e.g. libcudart.so.12
    or libcustatevec.so.1) or a CPU-only build must not break `import qsimcirq`.
    `qsimcirq.gpu_status()` reports the recorded message as the reason.
    """
    try:
        return importlib.import_module(f"qsimcirq.{name}")
    except ImportError as e:
        _import_errors[name] = str(e)
        return None


def _load_qsim_gpu():
    instr = qsim_decide.detect_gpu()
    if instr == 0:
        qsim_gpu = _try_import("qsim_cuda")
    elif instr == 3:
        qsim_gpu = _try_import("qsim_hip")
    else:
        qsim_gpu = None
    return qsim_gpu


def _load_qsim_custatevec():
    instr = qsim_decide.detect_custatevec()
    if instr == 1:
        qsim_custatevec = _try_import("qsim_custatevec")
    else:
        qsim_custatevec = None
    return qsim_custatevec


def _load_qsim_custatevecex():
    instr = qsim_decide.detect_custatevecex()
    if instr == 2:
        qsim_custatevecex = _try_import("qsim_custatevecex")
    else:
        qsim_custatevecex = None
    return qsim_custatevecex


qsim = _load_simd_qsim()
qsim_gpu = _load_qsim_gpu()
qsim_custatevec = _load_qsim_custatevec()
qsim_custatevecex = _load_qsim_custatevecex()

# Note: the following imports must remain at the bottom of this file.

from qsimcirq._version import __version__

from ._gpu_status import GpuStatus, gpu_status
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
    "gpu_status",
    "GpuStatus",
    "__version__",
]
