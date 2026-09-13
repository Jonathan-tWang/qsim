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
project(qsim LANGUAGES CXX)

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

# qsim_decide is a plain C++ module on every platform. It detects GPU support
# at runtime by dlopen()-ing the NVIDIA driver and CUDA libraries (see
# decide.cpp), so it must never be compiled by nvcc/hipcc and must not link
# against any CUDA library, even when a CUDA compiler is available.
pybind11_add_module(qsim_decide decide.cpp)

target_link_libraries(qsim_decide PRIVATE qsim_openmp_config)

# dlopen()/dlsym() live in libdl on glibc versions older than 2.34.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(qsim_decide PRIVATE ${CMAKE_DL_LIBS})
endif()

# QSIM_WHEEL_RPATH is applied by pybind_interface/cmake/qsim_wheel_rpath.cmake (WP2)
