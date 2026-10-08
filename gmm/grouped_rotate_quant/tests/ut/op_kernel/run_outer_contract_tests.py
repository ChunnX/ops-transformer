# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compile the real outer entry using host-only SDK substitutes, not device math."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--compiler-arg", action="append", default=[])
    parser.add_argument("--ops-tensor", type=Path, help="ops-tensor tree containing the grouped backend")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    tensor = args.ops_tensor or root.parents[4].parent / "ops-tensor_gmm_rotate_quant"
    if not (tensor / "include/blaze/attention/block/grouped_rotate_quant_problem.h").is_file():
        parser.error("pass --ops-tensor pointing to the adapted ops-tensor source tree")
    compiler = [args.compiler, *args.compiler_arg, "-std=c++17", "-Wall", "-Wextra", "-Werror"]
    with tempfile.TemporaryDirectory(prefix="grq_outer_") as directory:
        output = Path(directory) / "contract.exe"

        def compile_and_run(source, flags, expected_error=None):
            result = subprocess.run(
                [*compiler, "-I", str(root / "stubs"), "-I", str(tensor / "include"),
                 *flags, str(root / source), "-o", str(output)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            if expected_error is not None:
                if result.returncode == 0 or expected_error not in result.stderr:
                    raise RuntimeError(f"Expected compile rejection: {expected_error}\n{result.stderr}")
                return
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            subprocess.run([str(output)], check=True)

        for sdk in (8, 9):
            for x in ("half", "bfloat16_t"):
                for y in ("fp4x2_e2m1_t", "fp8_e4m3fn_t", "fp8_e5m2_t"):
                    flags = [f"-DASC_DEVKIT_MAJOR={sdk}", f"-DDTYPE_X={x}", f"-DDTYPE_ROTATION={x}",
                             f"-DDTYPE_Y=AscendC::{y}", "-DDTYPE_SCALE=AscendC::fp8_e8m0_t"]
                    compile_and_run("test_outer_contract.cpp", flags)
        compile_and_run("test_outer_contract.cpp", [*flags, "-DGRQ_TEST_MISSING_BACKEND"],
                        "grouped Tensor API backend is not integrated")
        compile_and_run("test_tiling_abi.cpp", [])
        compile_and_run("test_tiling_abi.cpp", ["-D__CCE_AICORE__=310"])
    print("PASS: 12 real adapter configurations, unavailable-binding rejection, 2 ABI checks")


if __name__ == "__main__":
    main()
