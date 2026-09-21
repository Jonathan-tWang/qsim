# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Installed-wheel checks; GPU execution is opt-in and never silently skipped.

QSIM_GPU_WHEEL_TEST=without-extras checks CPU fallback without NVIDIA packages.
QSIM_GPU_WHEEL_TEST=with-extras checks pip library resolution without a driver.
QSIM_GPU_WHEEL_TEST=gpu additionally executes every GPU backend against Cirq.
The default runs only the CPU checks, including in cibuildwheel's build image.
"""

import importlib
import importlib.metadata
import os
from pathlib import Path

import cirq
import numpy as np
import pytest

import qsimcirq

MODE = os.environ.get("QSIM_GPU_WHEEL_TEST", "cpu")
GPU_MODULES = ("qsim_cuda", "qsim_custatevec", "qsim_custatevecex")
NVIDIA_PACKAGES = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "custatevec-cu12",
)


def _circuit():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.Circuit(cirq.H.on_each(*qubits))
    for layer in range(3):
        circuit.append(cirq.CZ(qubits[i], qubits[i + 1]) for i in range(5))
        circuit.append(
            cirq.rx(0.2 + layer / 7 + i / 11)(q) for i, q in enumerate(qubits)
        )
        circuit.append(cirq.rz(0.1 + i / 13)(q) for i, q in enumerate(qubits))
    return circuit


def test_cpu_simulation():
    circuit = _circuit()
    actual = qsimcirq.QSimSimulator().simulate(circuit).final_state_vector
    expected = cirq.Simulator().simulate(circuit).final_state_vector
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_cpu_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m"))
    result = qsimcirq.QSimSimulator().run(circuit, repetitions=20)
    np.testing.assert_array_equal(result.measurements["m"], np.ones((20, 2)))


@pytest.mark.skipif(
    MODE == "cpu", reason="Requires an explicit installed-wheel test mode"
)
def test_wheel_installation():
    assert MODE in ("without-extras", "with-extras", "gpu"), MODE
    package = Path(qsimcirq.__file__).resolve().parent
    for module in GPU_MODULES:
        assert list(
            package.glob(f"{module}.*.so")
        ), f"Missing wheel extension: {module}"

    if MODE == "without-extras":
        for distribution in NVIDIA_PACKAGES:
            with pytest.raises(importlib.metadata.PackageNotFoundError):
                importlib.metadata.distribution(distribution)
        assert qsimcirq.qsim_gpu is None
        assert qsimcirq.qsim_custatevec is None
        assert qsimcirq.qsim_custatevecex is None
        for mode in range(3):
            with pytest.raises(ValueError, match="GPU|cuStateVec"):
                qsimcirq.QSimSimulator(
                    qsimcirq.QSimOptions(use_gpu=True, gpu_mode=mode)
                )
    else:
        for distribution in NVIDIA_PACKAGES:
            assert importlib.metadata.version(distribution)
        assert importlib.import_module("qsimcirq.qsim_cuda") is qsimcirq.qsim_gpu

        if MODE == "gpu":
            assert (
                importlib.import_module("qsimcirq.qsim_custatevec")
                is qsimcirq.qsim_custatevec
            )
            assert (
                importlib.import_module("qsimcirq.qsim_custatevecex")
                is qsimcirq.qsim_custatevecex
            )
        else:
            # cuStateVec 1.11 links to libnvidia-ml.so.1 from the NVIDIA
            # driver. Some build images provide its stub library and some do
            # not; both must retain CPU functionality, while qsim_cuda itself
            # remains importable without either a driver or stub.
            for mode, module in ((1, "qsim_custatevec"), (2, "qsim_custatevecex")):
                backend = getattr(qsimcirq, module)
                if backend is None:
                    assert "libnvidia-ml.so.1" in qsimcirq._import_errors[module]
                    with pytest.raises(ValueError, match="libnvidia-ml.so.1"):
                        qsimcirq.QSimSimulator(
                            qsimcirq.QSimOptions(use_gpu=True, gpu_mode=mode)
                        )
                else:
                    assert importlib.import_module(f"qsimcirq.{module}") is backend

        # Assert that the actual ELF loader resolved libraries from this venv,
        # not a toolkit accidentally inherited from the build environment.
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
        relative_paths = ["nvidia/cuda_runtime/lib/libcudart.so.12"]
        if any(
            backend is not None
            for backend in (qsimcirq.qsim_custatevec, qsimcirq.qsim_custatevecex)
        ):
            relative_paths += [
                "nvidia/cublas/lib/libcublas.so.12",
                "nvidia/cublas/lib/libcublasLt.so.12",
                "cuquantum/lib/libcustatevec.so.1",
            ]
        for relative_path in relative_paths:
            assert str(package.parent / relative_path) in maps, relative_path


@pytest.mark.skipif(
    MODE != "gpu", reason="Set QSIM_GPU_WHEEL_TEST=gpu on an NVIDIA GPU"
)
@pytest.mark.parametrize("gpu_mode", range(3))
@pytest.mark.parametrize("initial_state", [0, 21, "superposition"])
def test_gpu_simulation(gpu_mode, initial_state):
    circuit = _circuit()
    if initial_state == "superposition":
        rng = np.random.default_rng(601)
        initial_state = (rng.normal(size=64) + 1j * rng.normal(size=64)).astype(
            np.complex64
        )
        initial_state /= np.linalg.norm(initial_state)
    simulator = qsimcirq.QSimSimulator(
        qsimcirq.QSimOptions(use_gpu=True, gpu_mode=gpu_mode)
    )
    actual = simulator.simulate(circuit, initial_state=initial_state).final_state_vector
    expected = cirq.Simulator().simulate(circuit, initial_state=initial_state)
    np.testing.assert_allclose(
        actual, expected.final_state_vector, atol=3e-6, rtol=3e-5
    )


@pytest.mark.skipif(
    MODE != "gpu", reason="Set QSIM_GPU_WHEEL_TEST=gpu on an NVIDIA GPU"
)
@pytest.mark.parametrize("gpu_mode", range(3))
def test_gpu_amplitudes_and_expectations(gpu_mode):
    circuit = _circuit()
    simulator = qsimcirq.QSimSimulator(
        qsimcirq.QSimOptions(use_gpu=True, gpu_mode=gpu_mode)
    )
    reference = cirq.Simulator()
    bitstrings = [0, 1, 21, 42, 63]
    np.testing.assert_allclose(
        simulator.compute_amplitudes(circuit, bitstrings=bitstrings),
        reference.compute_amplitudes(circuit, bitstrings=bitstrings),
        atol=3e-6,
        rtol=3e-5,
    )
    q0, q1 = cirq.LineQubit.range(2)
    observables = [cirq.Z(q0), cirq.X(q0) * cirq.Y(q1)]
    np.testing.assert_allclose(
        simulator.simulate_expectation_values(circuit, observables),
        reference.simulate_expectation_values(circuit, observables),
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.skipif(
    MODE != "gpu", reason="Set QSIM_GPU_WHEEL_TEST=gpu on an NVIDIA GPU"
)
@pytest.mark.parametrize("gpu_mode", range(3))
def test_gpu_measurement(gpu_mode):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m"))
    simulator = qsimcirq.QSimSimulator(
        qsimcirq.QSimOptions(use_gpu=True, gpu_mode=gpu_mode)
    )
    result = simulator.run(circuit, repetitions=20)
    np.testing.assert_array_equal(result.measurements["m"], np.ones((20, 2)))
