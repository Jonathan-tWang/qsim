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

# NVIDIA's pip package contains only libcustatevec.so.1, whereas SDK installs
# also provide the unversioned linker symlink. Resolve both layouts, and repeat
# the lookup on reconfiguration when CUQUANTUM_ROOT may have changed.
function(qsim_find_custatevec output_variable)
    find_library(custatevec_library
        NAMES custatevec libcustatevec.so.1
        PATHS "$ENV{CUQUANTUM_ROOT}/lib" "$ENV{CUQUANTUM_ROOT}/lib64"
        NO_DEFAULT_PATH
        NO_CACHE
    )
    find_library(custatevec_library
        NAMES custatevec libcustatevec.so.1 NO_CACHE
    )
    if(NOT custatevec_library)
        message(FATAL_ERROR "Could not find libcustatevec.so or libcustatevec.so.1. "
            "Set CUQUANTUM_ROOT to the cuQuantum SDK directory or the "
            "custatevec-cu12 package's cuquantum directory.")
    endif()
    set(${output_variable} "${custatevec_library}" PARENT_SCOPE)
endfunction()
