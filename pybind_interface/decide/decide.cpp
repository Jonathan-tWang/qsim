// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>

namespace py = pybind11;

#if defined(_WIN32) && (defined(_M_IX86) || defined(_M_X64))
//  Windows with cpuid
#include <intrin.h>
#define cpuid(info, x)    __cpuidex(info, x, 0)

#elif defined(__x86_64__) || defined(__i386__)
// GCC Intrinsics for x86/x86_64
#include <cpuid.h>
void cpuid(int info[4], int infoType){
    __cpuid_count(infoType, 0, info[0], info[1], info[2], info[3]);
}

#endif

enum Instructions { AVX512F = 0, AVX2 = 1, SSE4_1 = 2, BASIC = 3};

int detect_instructions() {
  Instructions instr = BASIC;

  #if (defined(_WIN32) && (defined(_M_IX86) || defined(_M_X64))) || defined(__x86_64__) || defined(__i386__)
  // Existing x86/x86_64 specific instruction set detection logic
  int info[4];
  cpuid(info, 0);
  int nIds = info[0];
  if (nIds >= 1) {
    cpuid(info, 1);
    if ((info[2] & (1 << 19)) != 0) {
      instr = SSE4_1;
    }
  }
  if (nIds >= 7) {
    cpuid(info, 7);
    if ((info[1] & (1 << 5)) != 0) {
      instr = AVX2;
    }
    if ((info[1] & (1 << 16)) != 0) {
      instr = AVX512F;
    }
  }
  #endif

  return static_cast<int>(instr);
}

enum GPUCapabilities {
    CUDA = 0, CUSTATEVEC = 1, CUSTATEVECEX = 2, HIP = 3, NO_GPU = 10,
    NO_CUSTATEVEC = 11, NO_CUSTATEVECEX = 12 };

// GPU extensions are optional and loaded by qsimcirq/__init__.py. Inspect the
// result of those imports without opening CUDA libraries or initializing a
// driver. Looking in sys.modules also makes direct calls during package
// initialization safe: an unavailable attribute simply means no backend yet.
py::object loaded_backend(const char* name) {
  py::dict modules = py::reinterpret_borrow<py::dict>(PyImport_GetModuleDict());
  if (!modules.contains("qsimcirq")) {
    return py::none();
  }
  return py::getattr(modules["qsimcirq"], name, py::none());
}

int detect_gpu() {
  py::object gpu = loaded_backend("qsim_gpu");
  if (gpu.is_none()) {
    return NO_GPU;
  }
  return py::str(gpu.attr("__name__")).equal(py::str("qsimcirq.qsim_hip"))
      ? HIP : CUDA;
}

int detect_custatevec() {
  return loaded_backend("qsim_custatevec").is_none()
      ? NO_CUSTATEVEC : CUSTATEVEC;
}

int detect_custatevecex() {
  return loaded_backend("qsim_custatevecex").is_none()
      ? NO_CUSTATEVECEX : CUSTATEVECEX;
}

PYBIND11_MODULE(qsim_decide, m) {
  m.doc() = "CPU instruction and imported GPU backend detection";
  m.def("detect_instructions", &detect_instructions, "Detect SIMD");
  m.def("detect_gpu", &detect_gpu,
        "Return the imported CUDA/HIP backend code, without querying devices");
  m.def("detect_custatevec", &detect_custatevec,
        "Return whether the cuStateVec extension imported");
  m.def("detect_custatevecex", &detect_custatevecex,
        "Return whether the cuStateVecEx extension imported");
}
