#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# PyPTO kernel codegen driver (host-only, no NPU / no correctness or performance tests).
#
# Invoked from CMake (configure stage) for operators marked with enable_pypto_kernel(<op_file>):
#   pypto_codegen.py --py-file <op_kernel/<op_file>.py> --out-dir <pypto gen dir> \
#                    --op-file <op_file> --soc <ascend_compute_unit>
#
# It imports the kernel python file, runs `pl.generate_binary_headers()` (pure host-side codegen),
# and copies the infer artifacts into --out-dir:
#   *_tiling.h, *_tilingkey.h, *_pypto_infer.cpp
# The migrated opc still re-generates concrete kernels per tilingkey at build stage; the configure-stage
# infer cpp is only used to precompile .i and infer metadata consistently with compile_op.
# -----------------------------------------------------------------------------------------------------------
import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

# soc (ASCEND_COMPUTE_UNIT, lowercased) -> pypto arch token used by PYPTO_JIT_ARCH.
_ARCH_MAP = {
    "ascend910b": "a3",
    "ascend910c": "a3",
    "ascend950": "a5",
}


def _load_kernels(py_file: Path):
    """Import the kernel module and return all @pl.jit kernel objects defined in it."""
    sys.path.insert(0, str(py_file.parent))

    import pypto_pro.runtime.opc.pypto_compile as pto_compile
    from pypto_pro.runtime.jit import _TileJitKernel

    spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernels = [v for v in vars(module).values() if isinstance(v, _TileJitKernel)]
    return pto_compile, kernels


def main():
    parser = argparse.ArgumentParser(description="PyPTO kernel codegen driver")
    parser.add_argument("--py-file", required=True, help="kernel python source file")
    parser.add_argument("--out-dir", required=True, help="directory to place generated artifacts")
    parser.add_argument("--op-file", required=True, help="op file stem (e.g. flash_attention_score_apt)")
    parser.add_argument("--soc", required=True, help="ascend compute unit (e.g. ascend950)")
    args = parser.parse_args()

    soc = args.soc.strip().lower()
    arch = _ARCH_MAP.get(soc)
    if arch:
        os.environ["PYPTO_JIT_ARCH"] = arch  # pypto reads this during compile
    else:
        print(f"[pypto_codegen] warning: unknown soc '{soc}', letting pypto auto-detect arch", file=sys.stderr)

    py_file = Path(args.py_file).resolve()
    if not py_file.is_file():
        raise SystemExit(f"[pypto_codegen] py-file not found: {py_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pto_compile, kernels = _load_kernels(py_file)
    if not kernels:
        raise SystemExit(f"[pypto_codegen] no @pl.jit kernel found in {py_file}")

    copied = []
    for kernel in kernels:
        # generate_binary_headers() emits host tiling headers plus a self-contained cpp used only for .i infer.
        # The build-stage opc still produces the real per-tilingkey kernel sources.
        binary_dir = Path(pto_compile.generate_binary_headers(kernel)).resolve()
        for pattern in ("*_tiling.h", "*_tilingkey.h", "*_pypto_infer.cpp"):
            for src in sorted(binary_dir.glob(pattern)):
                shutil.copy(src, out_dir / src.name)
                copied.append(src.name)

    print(f"[pypto_codegen] {args.op_file}: generated {sorted(set(copied))} into {out_dir}")


if __name__ == "__main__":
    main()
