# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Performance runner: per-case msprof invocation, surviving crashes.

接口与 flash_attn 的 tests/pytests/utils/perf_runner.py 一致（build_case_script /
run_all_perf / build_batch_script / run_batch_msprof），内核调用段按 MLA 适配：
metadata 走 flash_mla_metadata，kernel 走 flash_mla_with_kvcache
（参数组由 core.data.build_flash_mla_with_kvcache_params 构造，与 NPU 后端同源）。
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_OP_MSG_TAG = "FlashMlaWithKvcache"


def _pytests_dir() -> str:
    """本 runner 所在 pytests 目录（生成的 msprof 脚本据此插 sys.path）。"""
    return str(Path(__file__).resolve().parent.parent)


def _serialize_params(c: Dict) -> str:
    s = {}
    for k, v in c.items():
        if hasattr(v, "shape"):
            s[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            s[k] = list(v)
        else:
            s[k] = v
    return repr(s)


def build_case_script(case: Dict, runs: int, case_name: str, device_id: int = 0) -> str:
    """Build a single-case Python script for msprof."""
    is_hot = case.get("S1", 9999) > 16
    mode = "hot" if is_hot else "cold"
    lines = [
        "import sys, os",
        f"_pytests_dir = {repr(_pytests_dir())}",
        "if _pytests_dir not in sys.path:",
        "    sys.path.insert(0, _pytests_dir)",
        "import torch, torch_npu",
        f"torch.npu.set_device({device_id})",
        "from core.data import flash_mla_with_kvcache_inputs, build_flash_mla_with_kvcache_params",
        "from cann_ops_transformer.ops import flash_mla_with_kvcache, flash_mla_metadata",
        "device = torch.device('npu:0')",
        f"c = {_serialize_params(case)}",
        f"print('CASE {case_name}  {mode}  runs={runs}')",
        "def _run_once():",
        "    inputs = flash_mla_with_kvcache_inputs.generate(",
        "        c, device=device, layout=c.get('layout_q', 'TND'))",
        "    meta_kwargs, kernel_kwargs, _ = build_flash_mla_with_kvcache_params(c, device, inputs)",
        "    metadata = flash_mla_metadata(**meta_kwargs)",
        "    torch.npu.synchronize()",
        "    flash_mla_with_kvcache(inputs['q'], inputs['k_cache'],",
        "                          metadata=metadata, **kernel_kwargs)",
        "    torch.npu.synchronize()",
        "    del inputs",
        "",
    ]
    if is_hot:
        # hot：inputs/metadata 只在循环外生成一次
        lines += [
            "_inputs = flash_mla_with_kvcache_inputs.generate(",
            "    c, device=device, layout=c.get('layout_q', 'TND'))",
            "_meta_kwargs, _kernel_kwargs, _ = build_flash_mla_with_kvcache_params(c, device, _inputs)",
            "_metadata = flash_mla_metadata(**_meta_kwargs)",
            "torch.npu.synchronize()",
            f"for _ in range({runs}):",
            "    flash_mla_with_kvcache(_inputs['q'], _inputs['k_cache'],",
            "                          metadata=_metadata, **_kernel_kwargs)",
            "    torch.npu.synchronize()",
            "torch.npu.empty_cache()",
        ]
    else:
        lines += [
            f"for _ in range({runs}):",
            "    _run_once()",
            "    torch.npu.empty_cache()",
        ]
    lines.append("print('CASE_DONE')")
    return "\n".join(lines)


def run_all_perf(
    cases: List[Dict],
    runs: int,
    cold_thr: int,
    output_dir: Path,
    case_names: List[str] = None,
    device_id: int = 0,
    one_by_one: bool = False,
) -> Optional[Path]:
    """Run each case as a separate msprof invocation. Returns merged op_summary.csv."""
    import csv as csv_mod

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = case_names or [f"case_{i}" for i in range(len(cases))]
    log_lines = []
    all_csvs = []

    perf_csv = output_dir / "perf.csv"
    csv_f = open(perf_csv, "w", newline="")
    csv_w = csv_mod.DictWriter(csv_f, ["#", "case", "dir", "avg_us", "mode", "error"])
    csv_w.writeheader()

    for i, (c, name) in enumerate(zip(cases, names)):
        tag = f"{'hot' if c.get('S1', 9999) > cold_thr else 'cold'}{i}"
        case_dir_name = name.replace("/", "_")
        case_dir = output_dir / case_dir_name
        case_dir.mkdir(parents=True, exist_ok=True)

        script = build_case_script(c, runs, name, device_id)
        script_path = case_dir / "_perf_sweep.py"
        script_path.write_text(script)

        msprof_dir = case_dir / "msprof_output"
        if msprof_dir.exists():
            shutil.rmtree(msprof_dir)

        cmd = [
            "msprof",
            f"--output={msprof_dir}",
            "--aic-mode=task-based",
            sys.executable,
            str(script_path),
        ]
        print(f"[msprof {i + 1}/{len(cases)}] {name}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        crashed = False
        error_msg = ""
        avg_us = 0.0
        mode = "hot" if c.get("S1", 9999) > cold_thr else "cold"

        if result.returncode != 0:
            crashed = True
            error_msg = f"exit={result.returncode}"
        if "CASE_DONE" not in stdout:
            crashed = True
            error_msg = error_msg or "CASE_DONE not found"
        if stderr:
            error_msg = (error_msg + "; " + stderr)[:500]

        log_lines.append(
            json.dumps(
                {
                    "tag": tag,
                    "ok": not crashed,
                    "error": error_msg,
                    "name": name,
                }
            )
        )

        prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
        if prof_dirs:
            csv_files = sorted(
                prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv")
            )
            if csv_files:
                dst = case_dir / "op_summary.csv"
                shutil.copy(csv_files[0], dst)
                all_csvs.append(dst)
                avg_us = _quick_avg(dst, runs)
                print(f"  [{name}] OK  avg={avg_us:.1f}us")
            else:
                print(f"  [{name}] NO CSV")
                crashed = True
                error_msg = error_msg or "no op_summary.csv"
        else:
            error_msg = error_msg or "no PROF directory"
            print(f"  [{name}] NO PROF_DIR {error_msg}")
            crashed = True

        csv_w.writerow(
            {
                "#": i + 1,
                "case": name,
                "dir": case_dir_name,
                "avg_us": "" if crashed else f"{avg_us:.1f}",
                "mode": "CRASH" if crashed else mode,
                "error": error_msg,
            }
        )
        csv_f.flush()

    csv_f.close()

    # Write log
    log_path = output_dir / "_perf_log.jsonl"
    with open(log_path, "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Merge all op_summary.csv into one
    if not all_csvs:
        return None

    merged = output_dir / "op_summary.csv"
    _merge_csvs(all_csvs, merged)
    print(f"[perf] merged {len(all_csvs)} CSV → {merged}")
    return merged


def build_batch_script(
    cases: List[Dict],
    runs: int,
    cold_thr: int,
    case_names: List[str] = None,
    device_id: int = 0,
    output_dir: str = ".",
) -> str:
    """Build a single script with all cases for batch msprof."""
    hot = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) > cold_thr]
    cold = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) <= cold_thr]
    names = case_names or [f"case_{i}" for i in range(len(cases))]
    hot_names = repr([names[i] for i, _ in hot])
    cold_names = repr([names[i] for i, _ in cold])
    hot_dicts = repr([_serialize_params(c) for _, c in hot])
    cold_dicts = repr([_serialize_params(c) for _, c in cold])

    lines = [
        "import sys, os, traceback",
        f"_pytests_dir = {repr(_pytests_dir())}",
        "if _pytests_dir not in sys.path:",
        "    sys.path.insert(0, _pytests_dir)",
        "import torch, torch_npu",
        f"torch.npu.set_device({device_id})",
        "from core.data import flash_mla_with_kvcache_inputs, build_flash_mla_with_kvcache_params",
        "from cann_ops_transformer.ops import flash_mla_with_kvcache, flash_mla_metadata",
        "device = torch.device('npu:0')",
        f"_hot = {hot_dicts}",
        f"_cold = {cold_dicts}",
        f"_hot_names = {hot_names}",
        f"_cold_names = {cold_names}",
        f"_perf_log = open({repr(str(Path(output_dir) / '_perf_log.jsonl'))}, 'w')",
        "def _run_one(c, tag, name):",
        "    try:",
        "        inputs = flash_mla_with_kvcache_inputs.generate(",
        "            c, device=device, layout=c.get('layout_q', 'TND'))",
        "        meta_kwargs, kernel_kwargs, _ = build_flash_mla_with_kvcache_params(c, device, inputs)",
        "        metadata = flash_mla_metadata(**meta_kwargs)",
        "        torch.npu.synchronize()",
        f"        for _ in range({runs}):",
        "            flash_mla_with_kvcache(inputs['q'], inputs['k_cache'],",
        "                                  metadata=metadata, **kernel_kwargs)",
        "            torch.npu.synchronize()",
        '        _perf_log.write(\'{"tag":"\' + tag + \'","ok":true}\\n\'); _perf_log.flush()',
        "        print(f'[CASE_END] {name}  OK')",
        "    except Exception as e:",
        "        traceback.print_exc()",
        '        _perf_log.write(\'{"tag":"\' + tag + \'","ok":false,"error":"\' + str(e).replace(chr(34), chr(39)) + \'"}\\n\')',
        "        _perf_log.flush()",
        "        print(f'[CASE_END] {name}  CRASH  {e}')",
        "    finally:",
        "        torch.npu.synchronize()",
        "        torch.npu.empty_cache()",
        "        import gc; gc.collect()",
        "",
    ]

    for idx, _ in enumerate(hot):
        lines.append(f"print(f'  [batch] {idx + 1}/{len(hot)}  {{_hot_names[{idx}]}}')")
        lines.append(f"_run_one(_hot[{idx}], 'hot{idx}', _hot_names[{idx}])")
    for idx, _ in enumerate(cold):
        lines.append(
            f"print(f'  [batch] {idx + 1}/{len(cold)}  {{_cold_names[{idx}]}}')"
        )
        lines.append(f"_run_one(_cold[{idx}], 'cold{idx}', _cold_names[{idx}])")

    lines.append("_perf_log.close()")
    lines.append("print('PERF_DONE')")
    return "\n".join(lines)


def run_batch_msprof(script: str, output_dir: Path) -> Optional[Path]:
    """Run batch msprof, return path to op_summary.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_perf_sweep.py").write_text(script)

    msprof_dir = output_dir / "msprof_output"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    cmd = [
        "msprof",
        f"--output={msprof_dir}",
        "--aic-mode=task-based",
        sys.executable,
        str(output_dir / "_perf_sweep.py"),
    ]
    print(f"[msprof] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(f"[msprof stderr]\n{result.stderr.strip()}")
    if result.returncode != 0:
        print(f"[msprof] exit={result.returncode}")

    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    if not prof_dirs:
        print("[msprof] no PROF_*")
        return None
    csv_files = sorted(prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv"))
    if not csv_files:
        print("[msprof] no op_summary CSV")
        return None
    dst = output_dir / "op_summary.csv"
    shutil.copy(csv_files[0], dst)
    print(f"[msprof] op_summary -> {dst}")
    return dst


def _merge_csvs(csv_paths: List[Path], out_path: Path):
    import csv as csv_mod

    rows = []
    header = None
    for cp in csv_paths:
        with open(cp) as f:
            reader = csv_mod.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                rows.append(row)
    if header is None:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _quick_avg(csv_path: Path, runs: int, op_name: str = _OP_MSG_TAG) -> float:
    """Quickly compute avg duration from a single op_summary.csv."""
    from utils.perf_parser import parse_op_summary

    entries = parse_op_summary(str(csv_path), op_name)
    durs = [d for _, d in entries]
    if not durs:
        return 0.0
    is_hot = len(durs) >= runs
    if is_hot:
        durs = durs[1:]  # discard 1st
    else:
        durs = sorted(durs)
        if len(durs) >= 3:
            durs = durs[1:-1]
    return sum(durs) / len(durs) if durs else 0.0
