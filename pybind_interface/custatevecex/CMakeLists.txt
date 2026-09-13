# Copyright 2025 Google LLC. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.28)
project(qsim LANGUAGES CXX CUDA)

if(MSVC)
    add_compile_options(/openmp)
else()
    add_compile_options(-fno-lto)
endif()

if(APPLE)
    include_directories(
      "/usr/local/include"
      "/usr/local/opt/llvm/include"
      "/opt/homebrew/include"
      "/opt/homebrew/opt/llvm@19/include"
    )
    link_directories(
      "/usr/local/lib"
      "/usr/local/opt/llvm/lib"
      "/opt/homebrew/lib"
      "/opt/homebrew/opt/llvm@19/lib"
    )
endif()

set(CMAKE_CXX_STANDARD 17)

include(../GetPybind11.cmake)
find_package(Python3 3.10 REQUIRED)

include_directories(${pybind11_INCLUDE_DIRS})

# cuBLAS comes from the CUDA toolkit that provides nvcc.
find_package(CUDAToolkit REQUIRED)

# cuStateVec comes from CUQUANTUM_ROOT, which may be either the cuQuantum SDK
# layout (lib/libcustatevec.so -> libcustatevec.so.1) or the custatevec-cu12
# pip wheel layout (<site-packages>/cuquantum, which only ships the versioned
# libcustatevec.so.1). Look there first, then in the default library paths.
include_directories($ENV{CUQUANTUM_ROOT}/include)
find_library(QSIM_CUSTATEVEC_LIBRARY
    NAMES custatevec libcustatevec.so.1
    PATHS $ENV{CUQUANTUM_ROOT}/lib $ENV{CUQUANTUM_ROOT}/lib64
    NO_DEFAULT_PATH
)
find_library(QSIM_CUSTATEVEC_LIBRARY NAMES custatevec libcustatevec.so.1)
if(NOT QSIM_CUSTATEVEC_LIBRARY)
    message(FATAL_ERROR "${MSG_PREFIX} could not find the cuStateVec library "
        "(libcustatevec.so or libcustatevec.so.1). Set CUQUANTUM_ROOT to the "
        "cuQuantum SDK directory or to the custatevec-cu12 wheel directory "
        "(<site-packages>/cuquantum). Searched $ENV{CUQUANTUM_ROOT}/lib, "
        "$ENV{CUQUANTUM_ROOT}/lib64 and the default library paths.")
endif()
message(STATUS "${MSG_PREFIX} found cuStateVec library "
               "${QSIM_CUSTATEVEC_LIBRARY}")

add_library(qsim_custatevecex MODULE pybind_main_custatevecex.cpp)
set_target_properties(qsim_custatevecex PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_source_files_properties(pybind_main_custatevecex.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(qsim_custatevecex PRIVATE
    qsim_openmp_config ${QSIM_CUSTATEVEC_LIBRARY} CUDA::cublas
)
qsim_apply_wheel_rpath(qsim_custatevecex)
