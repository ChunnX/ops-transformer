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
    python scripts/benchmark_compile.py [--per-phase] [--ftime-trace]
                                        [--op OP_NAME] [--warm-cache] [--no-cache-clear]

Options:
    --per-phase, -p     Parse .ninja_log for per-file (.cpp/.so) compile/link breakdown.
    --ftime-trace, -c   Switch to clang++, add -ftime-trace, parse per-header timing.
    --warm-cache        Also measure warm-cache (cached) compilation time.
    --no-cache-clear    Skip clearing the torch extensions cache (faster re-runs).
    --verbose, -v       Show subprocess stderr in real-time (for debugging hangs).
    --quiet, -q         Suppress compilation output, only show progress.
    --op OP_NAME        Test only a specific operator.
    --list              List available operators without running benchmarks.
"""

import argparse
import json
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


def find_build_dir(op_name):
    for cache_dir in get_torch_extensions_cache_dirs():
        if not cache_dir.exists():
            continue
        for ninja_log in cache_dir.rglob(".ninja_log"):
            try:
                ninja_log.relative_to(cache_dir)
            except ValueError:
                continue
            if op_name in str(ninja_log.parent):
                return ninja_log.parent
    return None


def parse_ninja_log(log_path):
    steps = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                start_ms = int(parts[0])
                end_ms = int(parts[1])
            except ValueError:
                continue
            d = end_ms - start_ms
            if d < 0:
                continue
            output = parts[3]
            basename = os.path.basename(output)
            ext = os.path.splitext(basename)[1]
            if ext in (".o", ".obj"):
                st = "compile"
            elif ext in (".so", ".dylib", ".pyd", ".dll"):
                st = "link"
            else:
                st = "other"
            steps.append({"output": basename, "duration_ms": d, "step_type": st})
    return steps


def print_ninja_breakdown(steps, indent="  "):
    if not steps:
        return

    def _fmt(ms):
        if ms < 1000:
            return f"{ms}ms"
        elif ms < 60000:
            return f"{ms / 1000:.2f}s"
        else:
            m, s = divmod(ms / 1000, 60)
            return f"{int(m)}m{s:.1f}s"

    print(f"{indent}Ninja steps ({len(steps)} items):")
    for cat, label in [("compile", "Compile"), ("link", "Link"), ("other", "Other")]:
        items = [s for s in steps if s["step_type"] == cat]
        if not items:
            continue
        total = sum(s["duration_ms"] for s in items)
        print(f"{indent}  [{label}] {len(items)} file(s), total {_fmt(total)}")
        m = max(len(s["output"]) for s in items)
        for s in sorted(items, key=lambda x: x["duration_ms"], reverse=True):
            bar = _bar(s["duration_ms"], max(s["duration_ms"] for s in items))
            print(f"{indent}    {s['output']:<{m}}  {_fmt(s['duration_ms']):>8}  {bar}")


def _bar(v, mv, w=20):
    if mv <= 0:
        return ""
    r = min(v / mv, 1.0)
    return "[" + "#" * int(r * w) + "-" * (w - int(r * w)) + "]"


def parse_clang_ftime_trace(build_dir):
    json_files = list(build_dir.glob("*.json"))
    if not json_files:
        return None
    results = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        events = data.get("traceEvents", [])
        if not events:
            continue
        total_us = 0
        headers = {}
        for e in events:
            name = e.get("name", "")
            dur = e.get("dur", 0)
            if name == "Total ExecuteCompiler":
                total_us = dur
            if dur <= 0:
                continue
            if name in ("Total Source", "Total Frontend", "Total Backend",
                        "Total ReadSource", "Total WriteSource"):
                continue
            detail = (e.get("args") or {}).get("detail", "")
            if not detail:
                continue
            p = Path(detail)
            if p.suffix in (".h", ".hpp", ".hxx", ".h++"):
                headers[detail] = headers.get(detail, 0) + dur
        sorted_h = sorted(headers.items(), key=lambda x: x[1], reverse=True)
        results.append({"json_file": jf.name, "total_us": total_us, "headers": sorted_h})
    return results


def print_clang_breakdown(trace_results, indent="  "):
    if not trace_results:
        return
    for r in trace_results:
        print(f"{indent}Clang -ftime-trace ({r['json_file']}):")
        print(f"{indent}  Total compilation: {r['total_us'] / 1e6:.2f}s")
        if not r["headers"]:
            continue
        print(f"{indent}  Top headers by time:")
        mv = r["headers"][0][1]
        top_n = min(20, len(r["headers"]))
        for _, (path, dur) in enumerate(r["headers"][:top_n]):
            sec = dur / 1e6
            bar = _bar(int(dur / 1000), int(mv / 1000))
            print(f"{indent}    {os.path.basename(path):<50} {sec:>8.2f}s  {bar}")
        if len(r["headers"]) > top_n:
            n = len(r["headers"]) - top_n
            print(f"{indent}    ({n} more headers not shown)")


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


def _build_isolated_import_py(op_name, compiler_flags=None):
    package_path = str(get_package_dir().resolve())

    flags_patch = ""
    if compiler_flags:
        flags_patch = """\
import cann_ops_transformer.op_builder.builder as _builder
_orig_cxx_args = _builder.OpBuilder.cxx_args
def _patched_cxx_args(self):
    return _orig_cxx_args(self) + {compiler_flags}
_builder.OpBuilder.cxx_args = _patched_cxx_args
""".format(compiler_flags=repr(compiler_flags))

    return ("""\
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

""" + flags_patch + """
importlib.import_module('cann_ops_transformer.ops.{op_name}')
_module = sys.modules['cann_ops_transformer.ops.{op_name}']
for _attr in dir(_module):
    if _attr.endswith('_op_builder'):
        _builder_inst = getattr(_module, _attr)
        if hasattr(_builder_inst, 'load'):
            _builder_inst.load()
""").format(package_path=package_path, op_name=op_name)


def measure_compile(op_name, *, timeout=600, verbose=False, quiet=False,
                    ftime_trace=False):
    compiler_flags = []
    clang_env = {}
    if ftime_trace:
        compiler_flags.append("-ftime-trace")
        clang_env = {"CXX": "clang++", "CC": "clang"}

    code = _build_isolated_import_py(op_name, compiler_flags=compiler_flags or None)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.update(clang_env)

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
    parser.add_argument("--per-phase", "-p", action="store_true",
                        help="Parse .ninja_log for per-file compile/link breakdown")
    parser.add_argument("--ftime-trace", "-c", action="store_true",
                        help="Switch to clang++, add -ftime-trace, parse per-header timing")
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
    if args.ftime_trace:
        print("Compiler: clang++ (--ftime-trace)")
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
            op, timeout=600, verbose=args.verbose, quiet=args.quiet,
            ftime_trace=args.ftime_trace)
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

        if cold_rc == 0:
            if args.per_phase:
                bd = find_build_dir(op)
                if bd:
                    steps = parse_ninja_log(bd / ".ninja_log")
                    print_ninja_breakdown(steps)
                else:
                    print(f"  (no .ninja_log found)")

            if args.ftime_trace:
                bd = find_build_dir(op)
                if bd:
                    tr = parse_clang_ftime_trace(bd)
                    if tr:
                        print_clang_breakdown(tr)
                    else:
                        print(f"  (no -ftime-trace .json found)")
                else:
                    print(f"  (no build dir found)")

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
