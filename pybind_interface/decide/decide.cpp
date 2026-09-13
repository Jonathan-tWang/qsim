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

#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

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

// GPU detection is performed at runtime, not at compile time: this module is
// always built as a plain C++ extension (never by nvcc or hipcc) and must not
// link against any CUDA library. On Linux it probes for the NVIDIA driver and
// the CUDA/cuQuantum shared libraries with dlopen(), which means a single
// wheel can decide at import time whether the GPU modules are usable on the
// machine it runs on. The libraries can come from a system CUDA toolkit or
// from the pip wheels installed by `pip install "qsimcirq[cuda12]"`. HIP
// support is still decided at compile time (ROCm wheels are out of scope).

// The outcome of probing the running system for NVIDIA GPU support. This is
// the C++ counterpart of the dict returned by gpu_probe().
struct GPUProbe {
  std::string platform;
  bool driver_library_found = false;
  bool has_driver_version = false;
  int driver_version = 0;
  int device_count = 0;
  bool cuda_runtime_found = false;
  bool has_cuda_runtime_version = false;
  int cuda_runtime_version = 0;
  bool cublas_found = false;
  bool custatevec_found = false;
  bool custatevec_ex_found = false;
  bool hip_compiled = false;
  std::vector<std::string> errors;
};

#if defined(__linux__)

// Appends "<what>: <dlerror()>" to errors, using a fallback text if the
// loader did not leave an error message behind.
void record_dl_error(const std::string& what,
                     std::vector<std::string>& errors) {
  const char* msg = dlerror();
  errors.push_back(what + ": " +
                   (msg != nullptr ? msg : "unknown dlopen/dlsym error"));
}

// dlsym() wrapper that records a failure in errors and returns nullptr.
void* load_symbol(void* handle, const char* name,
                  std::vector<std::string>& errors) {
  dlerror();  // Clear any stale error message.
  void* sym = dlsym(handle, name);
  if (sym == nullptr) {
    record_dl_error(std::string("dlsym(") + name + ")", errors);
  }
  return sym;
}

// Opens the shared library `soname`, recording a failure in errors. The
// libraries are opened RTLD_LOCAL so that their symbols never leak into the
// global namespace of the process (the real GPU modules load them again).
void* load_library(const char* soname, std::vector<std::string>& errors) {
  dlerror();  // Clear any stale error message.
  void* handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    record_dl_error(std::string("dlopen(") + soname + ")", errors);
  }
  return handle;
}

// Probes the NVIDIA driver (libcuda.so.1) for its version and the number of
// CUDA devices. The driver API prototypes are declared locally so that this
// file needs neither cuda.h nor -lcuda to build.
void probe_driver(GPUProbe& probe) {
  using CUresult = int;  // CUDA_SUCCESS == 0.
  using cuInit_t = CUresult (*)(unsigned int);
  using cuDriverGetVersion_t = CUresult (*)(int*);
  using cuDeviceGetCount_t = CUresult (*)(int*);

  void* driver = load_library("libcuda.so.1", probe.errors);
  if (driver == nullptr) {
    return;
  }
  probe.driver_library_found = true;

  auto cu_driver_get_version = reinterpret_cast<cuDriverGetVersion_t>(
      load_symbol(driver, "cuDriverGetVersion", probe.errors));
  if (cu_driver_get_version != nullptr) {
    int version = 0;
    CUresult result = cu_driver_get_version(&version);
    if (result == 0) {
      probe.has_driver_version = true;
      probe.driver_version = version;
    } else {
      probe.errors.push_back("cuDriverGetVersion failed with CUresult " +
                             std::to_string(result));
    }
  }

  auto cu_init = reinterpret_cast<cuInit_t>(
      load_symbol(driver, "cuInit", probe.errors));
  auto cu_device_get_count = reinterpret_cast<cuDeviceGetCount_t>(
      load_symbol(driver, "cuDeviceGetCount", probe.errors));
  if (cu_init != nullptr && cu_device_get_count != nullptr) {
    CUresult result = cu_init(0);
    if (result != 0) {
      probe.errors.push_back("cuInit(0) failed with CUresult " +
                             std::to_string(result));
    } else {
      int count = 0;
      result = cu_device_get_count(&count);
      if (result == 0 && count > 0) {
        probe.device_count = count;
      } else if (result != 0) {
        probe.errors.push_back("cuDeviceGetCount failed with CUresult " +
                               std::to_string(result));
      }
    }
  }

  dlclose(driver);
}

// Probes for the CUDA runtime, cuBLAS and cuStateVec shared libraries. Each
// library is opened by its versioned soname (the name the GPU modules are
// linked against) and closed again immediately.
void probe_libraries(GPUProbe& probe) {
  using cudaRuntimeGetVersion_t = int (*)(int*);  // cudaSuccess == 0.

  void* cudart = load_library("libcudart.so.12", probe.errors);
  if (cudart != nullptr) {
    probe.cuda_runtime_found = true;
    auto cuda_runtime_get_version = reinterpret_cast<cudaRuntimeGetVersion_t>(
        load_symbol(cudart, "cudaRuntimeGetVersion", probe.errors));
    if (cuda_runtime_get_version != nullptr) {
      int version = 0;
      int result = cuda_runtime_get_version(&version);
      if (result == 0) {
        probe.has_cuda_runtime_version = true;
        probe.cuda_runtime_version = version;
      } else {
        probe.errors.push_back("cudaRuntimeGetVersion failed with cudaError " +
                               std::to_string(result));
      }
    }
    dlclose(cudart);
  }

  void* cublas = load_library("libcublas.so.12", probe.errors);
  if (cublas != nullptr) {
    probe.cublas_found = true;
    dlclose(cublas);
  }

  void* custatevec = load_library("libcustatevec.so.1", probe.errors);
  if (custatevec != nullptr) {
    probe.custatevec_found = true;
    // The cuStateVecEx API exists from cuStateVec 1.10 onwards.
    void* ex_symbol = load_symbol(
        custatevec, "custatevecExStateVectorCreateSingleProcess", probe.errors);
    probe.custatevec_ex_found = ex_symbol != nullptr;
    dlclose(custatevec);
  }
}

#endif  // defined(__linux__)

// Runs the probe once. Never throws, never prints and never exits: every
// failure is reported through GPUProbe::errors instead.
GPUProbe run_gpu_probe() {
  GPUProbe probe;

  #if defined(__linux__)
  probe.platform = "linux";
  #elif defined(__APPLE__)
  probe.platform = "darwin";
  #elif defined(_WIN32)
  probe.platform = "windows";
  #else
  probe.platform = "other";
  #endif

  #ifdef __HIP__
  probe.hip_compiled = true;
  #endif

  #if defined(__linux__)
  probe_driver(probe);
  probe_libraries(probe);
  #else
  probe.errors.push_back(
      "CUDA GPU support is only available on Linux; no CUDA libraries were "
      "searched for on platform '" + probe.platform + "'.");
  #endif

  return probe;
}

// The probe result is cached for the lifetime of the process, so repeated
// calls to gpu_probe() and the detect_* functions are free.
const GPUProbe& gpu_probe_result() {
  static const GPUProbe probe = run_gpu_probe();
  return probe;
}

// Converts the cached probe result to a Python dict with the keys and types
// fixed by the qsim_decide API.
py::dict gpu_probe() {
  const GPUProbe& probe = gpu_probe_result();

  py::list errors;
  for (const std::string& error : probe.errors) {
    errors.append(error);
  }

  py::dict result;
  result["platform"] = probe.platform;
  result["driver_library_found"] = probe.driver_library_found;
  result["driver_version"] = probe.has_driver_version
      ? py::object(py::int_(probe.driver_version)) : py::object(py::none());
  result["device_count"] = probe.device_count;
  result["cuda_runtime_found"] = probe.cuda_runtime_found;
  result["cuda_runtime_version"] = probe.has_cuda_runtime_version
      ? py::object(py::int_(probe.cuda_runtime_version))
      : py::object(py::none());
  result["cublas_found"] = probe.cublas_found;
  result["custatevec_found"] = probe.custatevec_found;
  result["custatevec_ex_found"] = probe.custatevec_ex_found;
  result["hip_compiled"] = probe.hip_compiled;
  result["errors"] = errors;
  return result;
}

// CUDA is usable when a driver with at least one device and the CUDA runtime
// library are both present; HIP is decided at compile time.
int detect_gpu() {
  const GPUProbe& probe = gpu_probe_result();
  GPUCapabilities gpu = NO_GPU;
  if (probe.driver_library_found && probe.device_count > 0 &&
      probe.cuda_runtime_found) {
    gpu = CUDA;
  } else if (probe.hip_compiled) {
    gpu = HIP;
  }
  return gpu;
}

// cuStateVec additionally needs cuBLAS and the cuStateVec library itself.
int detect_custatevec() {
  const GPUProbe& probe = gpu_probe_result();
  GPUCapabilities gpu = NO_CUSTATEVEC;
  if (detect_gpu() == CUDA && probe.cublas_found && probe.custatevec_found) {
    gpu = CUSTATEVEC;
  }
  return gpu;
}

// cuStateVecEx additionally needs a cuStateVec library that exports the Ex
// API (cuStateVec >= 1.10).
int detect_custatevecex() {
  const GPUProbe& probe = gpu_probe_result();
  GPUCapabilities gpu = NO_CUSTATEVECEX;
  if (detect_custatevec() == CUSTATEVEC && probe.custatevec_ex_found) {
    gpu = CUSTATEVECEX;
  }
  return gpu;
}

PYBIND11_MODULE(qsim_decide, m) {
  m.doc() = "pybind11 plugin";  // optional module docstring

  // Methods for returning amplitudes
  m.def("detect_instructions", &detect_instructions, "Detect SIMD");

  // Detect available GPUs.
  m.def("detect_gpu", &detect_gpu, "Detect GPU");

  // Detect cuStateVec.
  m.def("detect_custatevec", &detect_custatevec, "Detect cuStateVec");

  // Detect cuStateVecEx.
  m.def("detect_custatevecex", &detect_custatevecex, "Detect cuStateVecEx");

  // Diagnostics behind the detect_* answers.
  m.def("gpu_probe", &gpu_probe,
        "Return a dict describing the NVIDIA driver and CUDA libraries found "
        "at runtime");
}
