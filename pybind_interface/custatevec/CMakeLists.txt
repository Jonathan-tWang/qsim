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

include_directories("$ENV{CUQUANTUM_ROOT}/include")
include(../cmake/qsim_custatevec.cmake)
qsim_find_custatevec(QSIM_CUSTATEVEC_LIBRARY)
message(STATUS "${MSG_PREFIX} found cuStateVec library "
               "${QSIM_CUSTATEVEC_LIBRARY}")

add_library(qsim_custatevec MODULE pybind_main_custatevec.cpp)
set_target_properties(qsim_custatevec PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_source_files_properties(pybind_main_custatevec.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(qsim_custatevec PRIVATE
    qsim_openmp_config ${QSIM_CUSTATEVEC_LIBRARY} CUDA::cublas
)
qsim_apply_wheel_rpath(qsim_custatevec)
