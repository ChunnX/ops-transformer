#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# PyPTO kernel codegen driver (host-only, no NPU / no correctness or performance tests).
#
# Invoked from CMake (configure stage) for operators marked with enable_pypto_kernel(<op_file>):
#   pypto_codegen.py --py-file <op_kernel/<op_file>.py> --out-dir <pypto gen dir> \
#                    --op-file <op_file> --soc <ascend_compute_unit>
#
# It imports the kernel python file, runs `fe.compile()` (pure host-side C++ codegen),
# and copies the generated artifacts into --out-dir:
#   <op_file>.cpp, kernel.cpp, *_tiling.h, *_tilingkey.h
# -----------------------------------------------------------------------------------------------------------
import argparse
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

# soc (ASCEND_COMPUTE_UNIT, lowercased) -> pypto arch token used by PYPTO_JIT_ARCH.
_ARCH_MAP = {
    "ascend910b": "a3",
    "ascend910c": "a3",
    "ascend950": "a5",
}


def _load_kernels(py_file: Path):
    """Import the kernel module and return all @fe.jit kernel objects defined in it."""
    sys.path.insert(0, str(py_file.parent))

    import pypto_pro.runtime as fe
    from pypto_pro.runtime.jit import _TileJitKernel

    spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernels = [v for v in vars(module).values() if isinstance(v, _TileJitKernel)]
    return fe, kernels


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

    stamp = out_dir / ".codegen.done"
    if stamp.is_file():
        py_mtime = py_file.stat().st_mtime
        stamp_mtime = stamp.stat().st_mtime
        if py_mtime <= stamp_mtime:
            print(f"[pypto_codegen] {args.op_file}: skipped (up-to-date)")
            return
    
    fe, kernels = _load_kernels(py_file)
    if not kernels:
        raise SystemExit(f"[pypto_codegen] no @fe.jit kernel found in {py_file}")

    copied = []
    for kernel in kernels:
        _t0 = time.perf_counter()
        binary_dir = Path(fe.compile(kernel)).resolve()
        _t1 = time.perf_counter()
        _parse_t = getattr(kernel, "_last_parse_time", 0.0)
        _codegen_t = getattr(kernel, "_last_codegen_time", 0.0)
        for pattern in (f"{args.op_file}.cpp", "kernel.cpp", "*_tiling.h", "*_tilingkey.h"):
            for src in sorted(binary_dir.glob(pattern)):
                shutil.copy(src, out_dir / src.name)
                copied.append(src.name)
        _t2 = time.perf_counter()
        shutil.rmtree(binary_dir.parent, ignore_errors=True)
        for _l in [
            f"{args.op_file}\tparse\t{_parse_t:.3f}s",
            f"{args.op_file}\tcodegen\t{_codegen_t:.3f}s",
            f"{args.op_file}\tcopy\t{_t2 - _t1:.3f}s",
            f"{args.op_file}\ttotal\t{_t2 - _t0:.3f}s",
        ]:
            print(f"[pypto-timing] {_l}", file=sys.stderr)

    print(f"[pypto_codegen] {args.op_file}: generated {sorted(set(copied))} into {out_dir}")
    stamp.touch()

if __name__ == "__main__":
    main()
