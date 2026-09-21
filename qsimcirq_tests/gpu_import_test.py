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

"""Subprocess regressions using real native extensions and dynamic libraries.

These tests need a C compiler, but no GPU toolkit or hardware. They exercise the
platform loader and the installed qsimcirq initializer, including dependencies
that are accessible only through an extension's private runtime search path.
"""

import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

import qsimcirq

_BACKENDS = ("qsim_cuda", "qsim_hip", "qsim_custatevec", "qsim_custatevecex")


def _compiler():
    # A Python installation can retain an SDK path that no longer exists after
    # Xcode is upgraded. The active compiler supplies its current SDK on macOS.
    if sys.platform == "darwin":
        return [shutil.which("cc") or "cc"]
    return shlex.split(sysconfig.get_config_var("CC") or "cc")


def _compile(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def native_backends(tmp_path_factory):
    if sys.platform not in ("linux", "darwin"):
        pytest.skip("native loader fixtures require Linux or macOS")
    compiler = _compiler()
    if not shutil.which(compiler[0]):
        pytest.skip("a C compiler is required for native loader fixtures")
    destination = tmp_path_factory.mktemp("gpu-native-extensions")
    private = destination / "private"
    private.mkdir()
    library_suffix = ".dylib" if sys.platform == "darwin" else ".so"
    for name in _BACKENDS:
        dependency = f"lib{name}_dependency{library_suffix}"
        source = private / f"{name}.c"
        source.write_text("int qsim_test_dependency(void) { return 601; }\n")
        shared_flags = (
            ["-dynamiclib", f"-Wl,-install_name,@rpath/{dependency}"]
            if sys.platform == "darwin"
            else ["-shared", f"-Wl,-soname,{dependency}"]
        )
        _compile(
            compiler
            + ["-fPIC", str(source), "-o", str(private / dependency)]
            + shared_flags
        )
        source = destination / f"{name}.c"
        source.write_text(
            "#include <Python.h>\n"
            "extern int qsim_test_dependency(void);\n"
            "static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "
            f'"{name}", NULL, -1, NULL}};\n'
            f"PyMODINIT_FUNC PyInit_{name}(void) {{\n"
            "  PyObject *result = PyModule_Create(&module);\n"
            "  if (result != NULL) PyModule_AddIntConstant("
            'result, "backend_value", qsim_test_dependency());\n'
            "  return result;\n"
            "}\n"
        )
        rpath = (
            "@loader_path/private" if sys.platform == "darwin" else "$ORIGIN/private"
        )
        linker = (
            compiler + ["-bundle", "-undefined", "dynamic_lookup"]
            if sys.platform == "darwin"
            else shlex.split(sysconfig.get_config_var("LDSHARED"))
        )
        _compile(
            linker
            + [
                "-fPIC",
                f"-I{sysconfig.get_path('include')}",
                str(source),
                f"-L{private}",
                f"-l{name}_dependency",
                f"-Wl,-rpath,{rpath}",
                "-o",
                str(destination / f"{name}{sysconfig.get_config_var('EXT_SUFFIX')}"),
            ]
        )
    return destination


@pytest.fixture
def isolated_package(tmp_path):
    destination = tmp_path / "qsimcirq"
    shutil.copytree(
        Path(qsimcirq.__file__).parent,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__", *[name + "*" for name in _BACKENDS]
        ),
    )
    return destination


def _install_backend(package, artifacts, name):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    shutil.copy2(artifacts / f"{name}{suffix}", package)
    (package / "private").mkdir(exist_ok=True)
    dependency = next((artifacts / "private").glob(f"lib{name}_dependency.*"))
    shutil.copy2(dependency, package / "private")
    return package / "private" / dependency.name


def _run(package, script, extra_environment=None):
    environment = dict(os.environ, PYTHONPATH=str(package.parent), PYTHONNOUSERSITE="1")
    environment.update(extra_environment or {})
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_CPU_CHECK = """
import cirq
import numpy as np
q = cirq.LineQubit(0)
result = qsimcirq.QSimSimulator().simulate(cirq.Circuit(cirq.X(q)))
np.testing.assert_allclose(result.final_state_vector, [0, 1], atol=1e-6)
"""


def test_cpu_install_survives_absent_gpu_extensions(isolated_package):
    _run(
        isolated_package,
        """
import qsimcirq
from qsimcirq import _gpu_status
assert qsimcirq.qsim_gpu is None
assert qsimcirq.qsim_custatevec is None
assert qsimcirq.qsim_custatevecex is None
assert "was not built or installed" in _gpu_status.gpu_backend_reason(0)
assert "pip install" not in _gpu_status.gpu_backend_reason(0)
assert qsimcirq.qsim_decide.detect_gpu() == 10
""" + _CPU_CHECK,
    )


def test_extensions_find_private_dependencies(isolated_package, native_backends):
    for name in _BACKENDS:
        if name != "qsim_hip":
            _install_backend(isolated_package, native_backends, name)
    _run(
        isolated_package,
        """
import qsimcirq
assert qsimcirq.qsim_gpu is not None
assert qsimcirq.qsim_custatevec is not None
assert qsimcirq.qsim_custatevecex is not None
assert qsimcirq.qsim_gpu.backend_value == 601
assert qsimcirq.qsim_decide.detect_gpu() == 0
assert qsimcirq.qsim_decide.detect_custatevec() == 1
assert qsimcirq.qsim_decide.detect_custatevecex() == 2
""" + _CPU_CHECK,
        {"LD_LIBRARY_PATH": "", "DYLD_LIBRARY_PATH": ""},
    )


def test_missing_shared_dependency_preserves_cpu(isolated_package, native_backends):
    dependency = _install_backend(isolated_package, native_backends, "qsim_cuda")
    dependency.unlink()
    _run(
        isolated_package,
        """
import qsimcirq
from qsimcirq import _gpu_status
assert qsimcirq.qsim_gpu is None
assert "qsim_cuda failed to import" in _gpu_status.gpu_backend_reason(0)
assert "libqsim_cuda_dependency" in _gpu_status.gpu_backend_reason(0)
assert "qsim_cuda" not in qsimcirq._missing_modules
assert qsimcirq.qsim_decide.detect_gpu() == 10
""" + _CPU_CHECK,
    )


@pytest.mark.parametrize("broken_hip", [False, True])
def test_hip_installation_keeps_preference(
    isolated_package, native_backends, broken_hip
):
    for name in _BACKENDS:
        dependency = _install_backend(isolated_package, native_backends, name)
        if broken_hip and name == "qsim_hip":
            dependency.unlink()
    _run(
        isolated_package,
        f"""
import sys
import qsimcirq
from qsimcirq import _gpu_status
assert qsimcirq._gpu_backend == "hip"
assert "qsimcirq.qsim_cuda" not in sys.modules
assert qsimcirq.qsim_custatevec is None
assert qsimcirq.qsim_custatevecex is None
if {broken_hip!r}:
    assert qsimcirq.qsim_gpu is None
    assert "libqsim_hip_dependency" in _gpu_status.gpu_backend_reason(0)
    assert "NVIDIA" not in _gpu_status.gpu_backend_reason(0)
    assert qsimcirq.qsim_decide.detect_gpu() == 10
else:
    assert qsimcirq.qsim_gpu.__name__ == "qsimcirq.qsim_hip"
    assert qsimcirq.qsim_decide.detect_gpu() == 3
""" + _CPU_CHECK,
    )


def test_reload_discards_stale_import_errors(isolated_package, native_backends):
    # Missing modules can appear after an optional dependency is installed. A
    # package reload should report the new imports, without old diagnostics.
    _run(
        isolated_package,
        f"""
import importlib
from pathlib import Path
import shutil
import qsimcirq
assert qsimcirq.qsim_gpu is None
source = Path({str(native_backends)!r})
package = Path(qsimcirq.__file__).parent
shutil.copytree(source / "private", package / "private")
for name in ("qsim_cuda", "qsim_custatevec", "qsim_custatevecex"):
    shutil.copy2(source / (name + {sysconfig.get_config_var('EXT_SUFFIX')!r}), package)
importlib.invalidate_caches()
importlib.reload(qsimcirq)
assert qsimcirq.qsim_gpu is not None
assert qsimcirq.qsim_custatevec is not None
assert qsimcirq.qsim_custatevecex is not None
assert "qsim_cuda" not in qsimcirq._import_errors
assert "qsim_cuda" not in qsimcirq._missing_modules
assert qsimcirq.qsim_decide.detect_gpu() == 0
""" + _CPU_CHECK,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux NVIDIA driver soname")
def test_import_and_diagnostics_do_not_open_nvidia_driver(
    isolated_package, native_backends
):
    for name in _BACKENDS:
        if name != "qsim_hip":
            _install_backend(isolated_package, native_backends, name)
    traps = isolated_package.parent / "driver-traps"
    traps.mkdir()
    source = traps / "trap.c"
    # Exit if the Python package's backend selection code probes these driver
    # libraries. The fake extensions themselves use only private dependencies;
    # installed-wheel tests cover imports of the real CUDA extensions.
    source.write_text(
        "#include <stdlib.h>\n"
        "__attribute__((constructor)) static void trap(void) { _Exit(85); }\n"
        "int cuInit(unsigned int flags) { _Exit(86); }\n"
        "int cuDeviceGetCount(int *count) { _Exit(87); }\n"
    )
    compiler = _compiler()
    for library in (
        "libcuda.so.1",
        "libcudart.so.12",
        "libcublas.so.12",
        "libcustatevec.so.1",
    ):
        _compile(
            compiler + ["-shared", "-fPIC", str(source), "-o", str(traps / library)]
        )
    _run(
        isolated_package,
        """
import os
import qsimcirq
assert qsimcirq.qsim_gpu is not None
assert qsimcirq.qsim_decide.detect_gpu() == 0
assert qsimcirq.qsim_decide.detect_custatevec() == 1
assert qsimcirq.qsim_decide.detect_custatevecex() == 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pid = os.fork()
if pid == 0:
    assert qsimcirq.qsim_gpu is not None
    os._exit(0)
assert os.waitpid(pid, 0)[1] == 0
""",
        {"LD_LIBRARY_PATH": str(traps)},
    )
