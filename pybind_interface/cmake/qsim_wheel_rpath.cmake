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

# Summary: opt-in relocatable RPATH for the qsimcirq wheel build (issue #601).
#
# qsim_apply_wheel_rpath(<target>) is a no-op unless the CMake option
# QSIM_WHEEL_RPATH is ON (it is OFF by default; see the root CMakeLists.txt).
# When it is ON, <target> is linked with a DT_RPATH entry (not DT_RUNPATH) that
# resolves, relative to the module's own location in site-packages/qsimcirq/,
# to the directories where the nvidia-cuda-runtime-cu12, nvidia-cublas-cu12 and
# custatevec-cu12 wheels install their shared libraries. DT_RPATH is searched
# before LD_LIBRARY_PATH, so a system CUDA toolkit cannot shadow the wheels; if
# the wheels are not installed, the loader falls through to its normal search
# path and a system toolkit still works. The build-tree RPATH that CMake would
# otherwise add (absolute paths into the build machine) is suppressed so the
# resulting modules are relocatable. This only applies to Linux; on other
# platforms the function does nothing.

function(qsim_apply_wheel_rpath target)
    if(NOT QSIM_WHEEL_RPATH)
        return()
    endif()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(STATUS "${MSG_PREFIX} QSIM_WHEEL_RPATH has no effect on "
                       "${CMAKE_SYSTEM_NAME}; ignoring it for ${target}")
        return()
    endif()
    # nvidia-cuda-runtime-cu12, nvidia-cublas-cu12 and custatevec-cu12 install
    # to site-packages/nvidia/{cuda_runtime,cublas}/lib and
    # site-packages/cuquantum/lib; $ORIGIN is site-packages/qsimcirq.
    string(JOIN ":" wheel_rpath
        "$ORIGIN/../nvidia/cuda_runtime/lib"
        "$ORIGIN/../nvidia/cublas/lib"
        "$ORIGIN/../cuquantum/lib"
    )
    set_target_properties(${target} PROPERTIES
        SKIP_BUILD_RPATH ON
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "${wheel_rpath}"
    )
    # Ask the linker for a DT_RPATH entry rather than the DT_RUNPATH it would
    # emit by default; only DT_RPATH takes precedence over LD_LIBRARY_PATH.
    target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
    message(STATUS "${MSG_PREFIX} ${target}: wheel RPATH ${wheel_rpath}")
endfunction()
