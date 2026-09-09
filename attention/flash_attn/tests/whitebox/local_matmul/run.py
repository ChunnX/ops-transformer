# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Compile the real matmul header against a host recorder; optionally test fault detection."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutations", action="store_true", help="Require five deliberate faults to be detected")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    kernel = here.parents[2] / "op_kernel" / "arch35"
    with tempfile.TemporaryDirectory(prefix="fa-matmul-whitebox-") as directory:
        work = Path(directory)
        def build(include, executable):
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-Wno-missing-field-initializers", "-I", str(here), "-I", str(include),
                            str(here / "test_local_matmul.cpp"), "-o", str(executable)], check=True)
        binary = work / "test"
        build(kernel, binary)
        subprocess.run([str(binary)], check=True, timeout=30)
        if args.mutations:
            original = (kernel / "flash_attn_matmul.h").read_text()
            mutations = {
                "A slot fails to rotate": ("l0aIdx ^= 1U;", "l0aIdx ^= 0U;"),
                "single B uses stale index": ("singleL0B ? 0U : l0bIdx", "l0bIdx"),
                "K continuation clears accumulator": ("(k == 0)", "true"),
                "N result offset overlaps": ("cL0Tensor[n * L0Coffset]", "cL0Tensor[0 * L0Coffset]"),
                "split forward barrier missing": ("Mutex::Lock<PIPE_M>(bMutex);", ";"),
            }
            for name, (old, new) in mutations.items():
                assert old in original
                (work / "flash_attn_matmul.h").write_text(original.replace(old, new))
                build(work, binary)
                result = subprocess.run([str(binary)], capture_output=True, timeout=30)
                assert result.returncode != 0 and b"instruction contract mismatch" in result.stderr, name
                print(f"Detected: {name}", flush=True)


if __name__ == "__main__":
    main()
