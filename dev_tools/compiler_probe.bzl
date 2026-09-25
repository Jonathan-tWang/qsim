# Copyright 2026 Google LLC
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

"""Repository rule to probe the C++ compiler's SFrame support and -march=native."""

def _compiler_probe_impl(repository_ctx):
    # Try to determine the compiler. Prefer CC from environment.
    cc = repository_ctx.os.environ.get("CC", "c++")

    # Run a test compilation to test if the flag is recognized.
    # This is hacky and I wish there was a better way.
    res = repository_ctx.execute([
        cc,
        "-Wa,--gsframe=no",
        "-c",
        "-x",
        "c++",
        "/dev/null",
        "-o",
        "/dev/null",
    ])

    supports_gsframe = (res.return_code == 0)

    # Identify what -march=native selects on this machine. Code compiled with
    # -march=native depends on the build machine's CPU, but the compile command
    # line does not, so machines with the same toolchain compute the same action
    # key for it. A shared disk or remote cache can then hand an object built on
    # a CPU with, e.g., AVX-512 to a machine without it, where the test dies with
    # an illegal instruction. Adding this ID to such command lines prevents that.
    # The ID is a hash of the compiler's predefined macros, which name the
    # instruction set extensions it may use (__AVX2__, __AVX512F__, ...), sorted
    # so that their order doesn't matter. Use the same compiler as Bazel's
    # auto-configured C++ toolchain: $CC if set, otherwise gcc.
    res = repository_ctx.execute([
        repository_ctx.os.environ.get("CC", "gcc"),
        "-march=native",
        "-dM",
        "-E",
        "-x",
        "c++",
        "/dev/null",
    ])
    if res.return_code == 0:
        native_arch_id = str(hash("\n".join(sorted(res.stdout.splitlines()))))
    else:
        native_arch_id = "unknown"

    repository_ctx.file("BUILD.bazel", "package(default_visibility = ['//visibility:public'])\n")
    repository_ctx.file(
        "compiler_config.bzl",
        "SUPPORTS_GSFRAME = %s\nNATIVE_ARCH_ID = \"%s\"\n" % (supports_gsframe, native_arch_id),
    )

compiler_probe = repository_rule(
    implementation = _compiler_probe_impl,
    local = True,
    environ = ["CC", "PATH"],
)
