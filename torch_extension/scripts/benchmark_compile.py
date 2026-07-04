# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Benchmark JIT compilation time for each operator in torch_extension.

Each operator is compiled in an ISOLATED subprocess via sys.modules stubbing:
only the target operator's .cpp is compiled per run, not the entire package.

Usage:
    python scripts/benchmark_compile.py [--warm-cache] [--no-cache-clear] [--verbose]

Options:
    --warm-cache      Also measure warm-cache (cached) compilation time.
    --no-cache-clear  Skip clearing the torch extensions cache (faster re-runs).
    --verbose, -v     Show subprocess stderr in real-time (for debugging hangs).
    --quiet, -q       Suppress compilation output, only show progress.
    --op OP_NAME      Test only a specific operator.
    --list            List available operators without running benchmarks.
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def get_ops_dir():
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / "cann_ops_transformer" / "ops"


def get_package_dir():
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / "cann_ops_transformer"


def get_torch_extensions_cache_dirs():
    home = Path.home()
    cache_dirs = [
        home / ".cache" / "torch_extensions",
        home / ".cache" / "torch_extensions_npu",
        home / "Ascend" / "torch_extensions",
    ]
    env_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if env_dir:
        cache_dirs.append(Path(env_dir))
    return cache_dirs


def clear_torch_extensions_cache():
    """Remove all JIT and compiler caches to force full recompilation."""
    for d in get_torch_extensions_cache_dirs():
        if d.exists():
            try:
                shutil.rmtree(d)
            except OSError:
                pass

    try:
        subprocess.run(["ccache", "-C"], capture_output=True, timeout=10)
    except Exception:
        pass


def discover_operators():
    ops_dir = get_ops_dir()
    csrc_dir = ops_dir / "csrc"

    if not csrc_dir.exists():
        print(f"[ERROR] csrc directory not found: {csrc_dir}")
        sys.exit(1)

    cpp_stems = {f.stem for f in csrc_dir.glob("*.cpp")}

    ops = []
    for py_file in sorted(ops_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        stem = py_file.stem
        if stem not in cpp_stems:
            continue

        has_op_builder = False
        try:
            content = py_file.read_text(encoding="utf-8")
            has_op_builder = "OpBuilder" in content and "class " in content
        except Exception:
            pass

        if not has_op_builder:
            continue

        ops.append(stem)

    return ops


def _build_isolated_import_py(op_name):
    """
    Build a Python script that imports ONLY the target operator module.

    The core problem: `from cann_ops_transformer.ops.flash_attn import *`
    triggers `cann_ops_transformer/__init__.py` -> `ops/__init__.py` ->
    `from .xxx import *` for ALL operators, causing all .cpp files to be
    compiled in every subprocess. Not just the target one.

    Fix: Pre-register stub modules in sys.modules for `cann_ops_transformer`
    and `cann_ops_transformer.ops` so their __init__.py files are never
    executed. Then use importlib to load only the target module directly.
    """
    package_path = str(get_package_dir().resolve())

    return """\
import sys, types, os, importlib

_pkg = r'{package_path}'

def _make_stub(name, path):
    m = types.ModuleType(name)
    m.__file__ = os.path.join(path, '__init__.py')
    m.__path__ = [path]
    return m

sys.modules['cann_ops_transformer'] = _make_stub(
    'cann_ops_transformer', _pkg)
sys.modules['cann_ops_transformer.ops'] = _make_stub(
    'cann_ops_transformer.ops', os.path.join(_pkg, 'ops'))
sys.modules['cann_ops_transformer.op_builder'] = _make_stub(
    'cann_ops_transformer.op_builder', os.path.join(_pkg, 'op_builder'))

importlib.import_module('cann_ops_transformer.ops.{op_name}')
""".format(package_path=package_path, op_name=op_name)


def measure_compile(op_name, *, timeout=600, verbose=False, quiet=False):
    """
    Launch a subprocess that compiles only ONE operator.
    Shows real-time elapsed while waiting.
    """
    code = _build_isolated_import_py(op_name)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if quiet:
        stdout_dest = subprocess.DEVNULL
    else:
        stdout_dest = sys.stdout

    start = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=stdout_dest,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stderr_lines = []

    def read_stderr():
        for line in iter(proc.stderr.readline, ""):
            stderr_lines.append(line)
            if not quiet:
                sys.stderr.write(f"  [{op_name}] {line}")
                sys.stderr.flush()

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    poll_interval = 2.0
    try:
        while True:
            try:
                proc.wait(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start
                sys.stdout.write(f"\r  [{op_name}] compiling... ({elapsed:.0f}s)")
                sys.stdout.flush()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    elapsed = time.perf_counter() - start
    t.join(timeout=2)

    stderr_text = "".join(stderr_lines)

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    stderr_tail = ""
    if proc.returncode != 0 and stderr_text:
        lines = [l for l in stderr_text.strip().split("\n") if l.strip()]
        stderr_tail = "\n".join(lines[-8:])

    return elapsed, proc.returncode, stderr_tail


def format_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{s:.1f}s"


def main():
    parser = argparse.ArgumentParser(description="Benchmark torch_extension operator compilation time")
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument("--no-cache-clear", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show subprocess stderr in real-time")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress compilation output, only show progress")
    parser.add_argument("--op", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    operators = discover_operators()

    if args.list:
        print(f"Found {len(operators)} operators with JIT compilation:")
        for op in operators:
            print(f"  {op}")
        return

    if args.op:
        if args.op not in operators:
            print(f"[ERROR] Unknown operator: {args.op}")
            print(f"Available: {', '.join(operators)}")
            sys.exit(1)
        operators = [args.op]

    if not operators:
        print("No operators found.")
        return

    print(f"Benchmarking {len(operators)} operator(s) in isolated subprocesses")
    if not args.no_cache_clear:
        print("Cache: cleared before each operator")
    print()
    sys.stdout.flush()

    has_warm = args.warm_cache
    header = f"{'Operator':<40} {'Cold (s)':<12} {'Warm (s)':<12} {'Status'}"
    if not has_warm:
        header = f"{'Operator':<40} {'Time (s)':<12} {'Status'}"
    print(header)
    print("-" * (70 if has_warm else 62))
    sys.stdout.flush()

    results = []
    for idx, op in enumerate(operators):
        if not args.no_cache_clear:
            clear_torch_extensions_cache()
            time.sleep(0.3)

        print(f"[{idx + 1}/{len(operators)}] ", end="")
        sys.stdout.flush()

        cold_elapsed, cold_rc, cold_err = measure_compile(
            op, timeout=600, verbose=args.verbose, quiet=args.quiet)
        cold_status = "OK" if cold_rc == 0 else f"FAIL [{cold_rc}]"

        if has_warm:
            warm_elapsed, warm_rc, warm_err = measure_compile(
                op, timeout=120, verbose=False, quiet=True)
            warm_status = "OK" if warm_rc == 0 else f"FAIL [{warm_rc}]"
            status = warm_status if warm_rc != 0 else cold_status
            print(f"{op:<40} {cold_elapsed:<12.2f} {warm_elapsed:<12.2f} {status}")
            results.append((op, cold_elapsed, warm_elapsed, cold_rc))
        else:
            print(f"{op:<40} {cold_elapsed:<12.2f} {cold_status}")
            results.append((op, cold_elapsed, cold_rc))

        sys.stdout.flush()

        if cold_rc != 0 and cold_err:
            for line in cold_err.split("\n"):
                print(f"  [ERROR] {line}")
            sys.stdout.flush()

    print()
    print("=" * 70)
    ok_count = sum(1 for r in results if r[-1] == 0)
    total_cold = sum(r[1] for r in results if r[-1] == 0)
    print(f"Total: {len(results)} operators, {ok_count} passed, {len(results) - ok_count} failed")
    sys.stdout.flush()

    if ok_count > 0:
        ok_list = [r for r in results if r[-1] == 0]
        sorted_list = sorted(ok_list, key=lambda x: x[1], reverse=True)
        print(f"Sum of cold compilation time: {format_time(total_cold)}")
        print(f"Slowest:  {sorted_list[0][0]} ({format_time(sorted_list[0][1])})")
        print(f"Fastest:  {sorted_list[-1][0]} ({format_time(sorted_list[-1][1])})")


if __name__ == "__main__":
    main()
