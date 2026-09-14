#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""msprof 性能采集(最小版, 抽取自 ops-transformer-testkit core/profiler.py)。

以 output_dir 为 cwd 运行生成的脚本, task-based 模式采集 AI Core kernel,
然后从最新的 PROF_*/mindstudio_profiler_output 拷出 op_summary CSV。
"""

import shutil
import subprocess
import sys
from pathlib import Path


def run_msprof(script: str, output_dir: Path, label: str = "case",
               total_timeout: int = 7200) -> Path:
    """在 output_dir 下写脚本并用 msprof 执行, 返回拷出的 op_summary.csv 路径。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{label}.py"
    script_path.write_text(script)

    msprof_dir = output_dir / "msprof"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    cmd = [
        "msprof",
        f"--output={msprof_dir}",
        "--aic-mode=task-based",
        sys.executable,
        str(script_path),
    ]
    log_path = output_dir / f"{label}.msprof.log"
    print(f"[msprof] {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        subprocess.run(
            cmd, cwd=output_dir, stdout=log_file, stderr=subprocess.STDOUT,
            timeout=total_timeout, check=False,
        )

    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    if not prof_dirs:
        raise FileNotFoundError(f"No PROF_* found in {msprof_dir}, see {log_path}")
    csv_files = sorted(
        prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No op_summary CSV in {prof_dirs[0]}, see {log_path}"
        )

    dst = output_dir / "op_summary.csv"
    shutil.copy(csv_files[0], dst)
    print(f"[msprof] {label} -> {dst}")
    return dst
