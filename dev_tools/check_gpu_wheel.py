# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Check the repaired CUDA wheel without importing or initializing CUDA.

Run on Linux with binutils installed: python dev_tools/check_gpu_wheel.py WHEEL.
The external NVIDIA libraries are intentional exceptions to auditwheel's
dependency bundling; the matching cuda12 extra must supply them at runtime.
"""

import argparse
import email
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

GPU_MODULES = ("qsim_cuda", "qsim_custatevec", "qsim_custatevecex")
CPU_MODULES = ("qsim_basic", "qsim_decide", "qsim_sse", "qsim_avx2", "qsim_avx512")
NVIDIA_LIBRARY = re.compile(r"lib(?:cuda|cudart|cublas|custatevec|nvrtc|nvJitLink)")
WHEEL_RPATHS = {
    "$ORIGIN/../nvidia/cuda_runtime/lib",
    "$ORIGIN/../nvidia/cublas/lib",
    "$ORIGIN/../cuquantum/lib",
}
EXTRA_PACKAGES = {
    "nvidia-cuda-runtime-cu12": {">=12.0", "<13"},
    "nvidia-cublas-cu12": {">=12.0", "<13"},
    "custatevec-cu12": {">=1.11", "<2"},
}


def check_wheel(wheel: Path, max_bytes: int = 100_000_000) -> dict:
    """Raise ValueError for an incomplete, nonrelocatable, or oversized wheel."""
    size = wheel.stat().st_size
    if size >= max_bytes:
        raise ValueError(f"Wheel is {size} bytes; expected less than {max_bytes}")
    if "manylinux_2_28_x86_64" not in wheel.name:
        raise ValueError("Expected a repaired manylinux_2_28_x86_64 wheel")

    report = {"wheel": wheel.name, "bytes": size, "modules": {}}
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as temp:
        names = archive.namelist()
        bundled = [name for name in names if NVIDIA_LIBRARY.match(Path(name).name)]
        if bundled:
            raise ValueError(f"NVIDIA libraries must not be bundled: {bundled}")
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("Expected exactly one wheel METADATA file")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        if "cuda12" not in metadata.get_all("Provides-Extra", []):
            raise ValueError("Wheel metadata is missing the cuda12 extra")
        requirements = [
            Requirement(value) for value in metadata.get_all("Requires-Dist", [])
        ]
        for package, expected_specifiers in EXTRA_PACKAGES.items():
            matches = [req for req in requirements if req.name == package]
            if len(matches) != 1:
                raise ValueError(
                    f"Missing platform-guarded cuda12 dependency: {package}"
                )
            requirement = matches[0]
            if {str(specifier) for specifier in requirement.specifier} != (
                expected_specifiers
            ):
                raise ValueError(f"{package} has incorrect version bounds")
            marker = requirement.marker
            marker_cases = (
                ("cuda12", "linux", "x86_64", True),
                ("", "linux", "x86_64", False),
                ("cuda12", "win32", "x86_64", False),
                ("cuda12", "linux", "aarch64", False),
            )
            if marker is None or any(
                marker.evaluate(
                    {
                        "extra": extra,
                        "sys_platform": platform,
                        "platform_machine": machine,
                    }
                )
                != expected
                for extra, platform, machine, expected in marker_cases
            ):
                raise ValueError(f"{package} has incorrect cuda12 platform markers")

        modules = {}
        for name in names:
            if name.startswith("qsimcirq/") and name.endswith(".so"):
                module = Path(name).name.split(".")[0]
                if module in modules:
                    raise ValueError(f"Duplicate extension: {module}")
                modules[module] = name
        required = (*GPU_MODULES, *CPU_MODULES)
        if missing := set(required) - modules.keys():
            raise ValueError(f"Wheel is missing extensions: {sorted(missing)}")

        for module, name in sorted(modules.items()):
            path = Path(temp) / Path(name).name
            path.write_bytes(archive.read(name))
            header = subprocess.check_output(["readelf", "-h", str(path)], text=True)
            if "Advanced Micro Devices X86-64" not in header:
                raise ValueError(f"{module} is not an x86_64 ELF extension")
            dynamic = subprocess.check_output(["readelf", "-d", str(path)], text=True)
            needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)
            paths = re.findall(r"\((?:RPATH|RUNPATH)\).*\[(.*?)\]", dynamic)
            rpaths = {entry for value in paths for entry in value.split(":")}
            if any(not entry.startswith("$ORIGIN/") for entry in rpaths):
                raise ValueError(
                    f"{module} has a nonrelocatable library path: {rpaths}"
                )
            if module in GPU_MODULES:
                required_libraries = {"libcudart.so.12"}
                if module != "qsim_cuda":
                    required_libraries.add("libcustatevec.so.1")
                if module == "qsim_custatevec":
                    required_libraries.add("libcublas.so.12")
                if not required_libraries <= set(needed):
                    raise ValueError(f"{module} lacks dynamic dependencies: {needed}")
                if not WHEEL_RPATHS <= rpaths or "(RPATH)" not in dynamic:
                    raise ValueError(f"{module} is missing transitive wheel RPATHs")
            elif any(NVIDIA_LIBRARY.match(lib) for lib in needed):
                raise ValueError(f"CPU module {module} depends on NVIDIA: {needed}")
            if "libcuda.so.1" in needed:
                raise ValueError(f"{module} must not link directly to the CUDA driver")
            report["modules"][module] = {"needed": needed, "rpath": sorted(rpaths)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--max-bytes", type=int, default=100_000_000)
    args = parser.parse_args()
    print(json.dumps(check_wheel(args.wheel, args.max_bytes), indent=2))


if __name__ == "__main__":
    main()
