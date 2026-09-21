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

"""Private diagnostics for optional GPU extension imports."""

from typing import Optional


def gpu_backend_reason(gpu_mode: int) -> Optional[str]:
    """Explain why the requested backend did not import."""
    import qsimcirq

    if gpu_mode == 0:
        if qsimcirq.qsim_gpu is not None:
            return None
        module = "qsim_hip" if qsimcirq._gpu_backend == "hip" else "qsim_cuda"
    else:
        module = "qsim_custatevec" if gpu_mode == 1 else "qsim_custatevecex"
        if getattr(qsimcirq, module) is not None:
            return None
        if qsimcirq._gpu_backend == "hip":
            return "cuStateVec backends are not available in this HIP build."
    if module in qsimcirq._missing_modules:
        return (
            f"qsimcirq.{module} was not built or installed. "
            "Install a qsimcirq build with this backend."
        )
    error = qsimcirq._import_errors.get(module)
    if error:
        return f"qsimcirq.{module} failed to import: {error.rstrip('.')}."
    return f"qsimcirq.{module} is not loaded."
