#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Build and run the real host-compatible headers with minimal include stubs.

This is a CPU-only Windows fallback. It does not emulate Ascend events, cache
coherence or device compilation. All build outputs stay outside the checkout.
"""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def run(command, workdir):
    result = subprocess.run(command, cwd=workdir, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {command}\n{result.stdout}\n{result.stderr}")


def extract_function(text, name):
    begin = text.index(name)
    opening = text.index("{", begin)
    depth = 1
    index = opening + 1
    while depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[begin:index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=Path(
        "C:/Program Files/Microsoft Visual Studio/2022/Community/SDK/ScopeCppSDK/vc15/VC/bin/cl.exe"))
    parser.add_argument("--output-dir", type=Path, default=Path("D:/workspace/mega_moe_ring_cpu_validation"))
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    linker = compiler.with_name("link.exe")
    if not compiler.is_file() or not linker.is_file():
        parser.error("MSVC compiler and link.exe are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="layout_", dir=args.output_dir.resolve()))
    source_dir = Path(__file__).resolve().parent
    source = source_dir / "check_production_layout.cpp"

    def build_and_check(name, input_source, expected=0, defines=()):
        obj = output / f"{name}.obj"
        exe = output / f"{name}.exe"
        compile_command = [str(compiler), "/nologo", "/utf-8", "/std:c++17", "/O1", "/GS-", "/c",
                           f"/I{source_dir / 'stubs'}", f"/Fo{obj}", str(input_source)]
        compile_command.extend(f"/D{define}" for define in defines)
        run(compile_command, output)
        run([str(linker), "/nologo", "/NODEFAULTLIB", "/ENTRY:main", "/SUBSYSTEM:CONSOLE",
             f"/OUT:{exe}", str(obj)], output)
        result = subprocess.run([str(exe)], cwd=output)
        if result.returncode != expected:
            raise RuntimeError(f"{name}: returned {result.returncode}, expected {expected}")
        print(f"{name}: PASS (exit {result.returncode})")

    build_and_check("layout_check", source)
    build_and_check("negative_control", source, 91, ("CPU_RING_FORCE_FAILURE=1",))
    # Compile and execute the exact new UT bodies as well. Existing tiling
    # cases depend on unavailable TilingContext/runtime libraries and are not
    # part of this fallback. Fail if the source boundaries or case names change.
    host_ut = source_dir.parent / "op_host/arch35/test_mega_moe_tiling.cpp"
    host_text = host_ut.read_text(encoding="utf-8")
    begin = host_text.index("static MegaMoeTilingData MakeMteRingLayoutCase()")
    end = host_text.rindex("} // namespace MegaMoeUT")
    fragment = host_text[begin:end]
    names = re.findall(r"TEST_F\(MegaMoeArch35TilingTest, (MteRing\w+)\)", fragment)
    if len(names) != 4:
        raise RuntimeError(f"expected four ring layout UTs, found {names}")
    extracted = output / "extracted_host_layout_cases.cpp"
    extracted.write_text(
        f'#include "{(source_dir / "cpu_test_support.h").as_posix()}"\n'
        '#define CPU_RING_EMBEDDED_MAIN 1\n'
        f'#include "{source.as_posix()}"\n'
        'namespace MegaMoeUT {\n' + fragment + '\n}\n'
        'extern "C" int main() {\n'
        + '\n'.join(f'    MegaMoeUT::{name}();' for name in names)
        + '\n    return CpuCheckFailureLine;\n}\n', encoding="utf-8")
    build_and_check("extracted_host_layout_cases", extracted)
    # The Torch helper carries its own upper-bound formula because its JIT
    # source is shipped without kernel headers. Execute that exact integer
    # function against the actual workspace allocator over supported modes.
    torch_source = source_dir.parents[2] / "torch_extension/csrc/mega_moe.cpp"
    torch_text = torch_source.read_text(encoding="utf-8")
    constants_begin = torch_text.index("constexpr int64_t ALIGN_32 =")
    constants_end = torch_text.index("int64_t CeilAlign(", constants_begin)
    torch_fragment = (torch_text[constants_begin:constants_end]
                      + extract_function(torch_text, "int64_t CeilAlign(") + "\n"
                      + extract_function(torch_text, "int64_t CeilDiv(") + "\n"
                      + extract_function(torch_text, "int64_t CalcCclWorkspaceSizeA5("))
    extracted_bound = output / "extracted_torch_bound.cpp"
    extracted_bound.write_text(
        f'#include "{(source_dir / "cpu_test_support.h").as_posix()}"\n'
        '#define CPU_RING_EMBEDDED_MAIN 1\n'
        f'#include "{source.as_posix()}"\n'
        '#define TORCH_CHECK(condition, ...) do { if (!(condition)) return -1; } while (0)\n'
        'namespace TorchBound {\n' + torch_fragment + '\n}\n'
        f'#include "{(source_dir / "check_torch_bound.inc").as_posix()}"\n'
        'extern "C" int main() { return RunTorchBoundChecks(); }\n', encoding="utf-8")
    build_and_check("extracted_torch_bound", extracted_bound)
    print("production WorkspaceLayout: 34,560 parameter combinations")
    print("production padding/row mapping: 2,872,320 row cases")
    print("new host layout UTs: 4 exact-source cases (freestanding assertion adapter)")
    print("Torch CCL bound: 2,624,400 real workspace configurations (up to 4 shared experts)")
    print(f"build artifacts: {output}")


if __name__ == "__main__":
    main()
