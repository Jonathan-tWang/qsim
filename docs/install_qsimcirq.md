# Installing qsimcirq

The qsim-Cirq Python interface is available as a PyPI package for Linux, MacOS and Windows users.
For all others, Dockerfiles are provided to install qsim in a containerized
environment.

**Note:** The core qsim library (under
[lib/](https://github.com/quantumlib/qsim/blob/main/lib)) can be included
directly in C++ code without building and installing the qsimcirq interface.

## Before installation

Prior to installation, consider creating a
[virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Prerequisites for installing and running qsim are included in the
[`requirements.txt`](https://github.com/quantumlib/qsim/blob/main/requirements.txt)
file, and will be automatically installed along with qsimcirq when you install
it with pip.

If you'd like to develop qsimcirq, a separate set of dependencies are defined
in the
[`pyproject.toml`](https://github.com/quantumlib/qsim/blob/main/pyproject.toml)
file. Using pip version 25.1 or higher, you can install them with the following
commands:

```shell
pip install -r requirements.txt
pip install --group dev
```

## Linux installation

We provide `qsimcirq` Python wheels on 64-bit `x86` architectures with
`Python 3.{10,11,12,13}`. Simply run `pip3 install qsimcirq` for the standard
installation.

CUDA-enabled Linux x86_64 wheels can be produced with the dedicated
`dev_tools/ci/cuda_wheels.toml` cibuildwheel configuration. Those wheels
contain the CUDA and cuStateVec extension modules, while NVIDIA's much larger
runtime libraries remain separate dependencies. Install a CUDA-enabled wheel
artifact with its `cuda12` extra to supply CUDA 12, cuBLAS, and cuStateVec:

```shell
pip3 install "/path/to/qsimcirq-VERSION-cpPYTHON-cpPYTHON-manylinux_2_28_x86_64.whl[cuda12]"
```

The extra does not add GPU extensions to a CPU-only wheel. It is restricted to
Linux x86_64 by environment markers and requires cuStateVec 1.11 or newer.
The CUDA wheel contains native code for compute capabilities 7.5 through 12.0,
so it supports Turing and newer NVIDIA GPUs. A compatible NVIDIA driver is
still required when a simulation is run.

Importing `qsimcirq` does not initialize the GPU or create a CUDA context.
GPU availability is checked by CUDA when the first GPU simulation runs. A CPU
simulation continues to work if an optional GPU extension or one of its
shared-library dependencies cannot be loaded. If a requested extension could
not load, `QSimSimulator` includes the dynamic loader's error in its existing
“not supported” exception.

## MacOS installation

We provide `qsimcirq` Python wheels on `x86` and Apple Silicon architectures
with `Python 3.{10,11,12,13}`.

Simply run `pip3 install qsimcirq`.

Note that, due to architectural differences, CUDA support is not available on
MacOS. The version of `qsimcirq` on MacOS will only use the CPU, without GPU
acceleration.

## Windows installation

We provide `qsimcirq` Python wheels on 64-bit `x86` and `amd64` architectures
with `Python 3.{10,11,12,13}`.

Simply run `pip3 install qsimcirq`.

## Help! There's no compatible wheel for my machine!

If existing wheels do no meet your needs, please open an issue with your
machine configuration (i.e., CPU architecture, Python version) and consider
using the [Docker config](./docker.md) provided in the qsim GitHub repository.

## Testing

After installing `qsimcirq` on your machine, you can test the installation by
copying [qsimcirq_tests/qsimcirq_test.py](qsimcirq_tests/qsimcirq_test.py)
to your machine and running `python3 -m pytest qsimcirq_test.py`.

The file `qsimcirq_test.py` also has examples of how to use qsimcirq.

**Note:** Because of how Python searches for modules, the test file cannot
be run from inside a clone of the qsim repository, or from any parent
directory of such a repository. Failure to meet this criteria may result
in misbehaving tests (e.g., false positives after a failed installation).
