#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""非法输入拒绝套件：用 examples 驱动器（`--neg <mode>`）逐一构造非法输入，
要求算子 GetWorkspaceSize 返回非 0（驱动器退出码 != 0 且打印 "GetWorkspaceSize failed"）。

覆盖 op_host/kv_cache_turbo_quant_tiling.cpp 的全部校验点：
  kv_vectors            : rank=3 / dim2=128 / 非空 / dtype=BF16 / head 数 ∈ [4, 32]
  rotation_matrix / qjl : rank=2、[128, 128]、dtype=FP32
  mse_bits              : ∈ {2, 3, 4}

用法:
  export KCTQ_BIN=<构建出的 test_aclnn_kv_cache_turbo_quant 路径>
  python invalid_cases.py [--work <目录>] [--json 结果.json]
退出码: 0 = 所有非法用例被拒绝，且合法基准被接受；1 = 有用例被错误接受/基准被拒。
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (用例名, 驱动器的 --neg 模式, 说明)
MODES = [
    ("valid_base", "", "合法基准（应被接受，套件自检）"),
    ("mse_bits_5", "mse-bits-5", "mse_bits 越界（5）"),
    ("mse_bits_1", "mse-bits-1", "mse_bits 越界（1）"),
    ("kv_rank2", "kv-rank2", "kv_vectors rank=2"),
    ("head_dim_64", "head-dim-64", "head_dim=64"),
    ("heads_below_min", "heads-2", "head 数 = 2（<4）"),
    ("heads_above_max", "heads-64", "head 数 = 64（>32）"),
    ("kv_dtype_fp32", "kv-fp32", "kv_vectors dtype=FP32"),
    ("rot_dim_64", "rot-64", "rotation_matrix=[64,64]"),
    ("rot_dtype_bf16", "rot-bf16", "rotation_matrix dtype=BF16"),
    ("qjl_dtype_bf16", "qjl-bf16", "qjl_matrix dtype=BF16"),
    ("empty_tokens", "empty", "空 tensor（tokens=0）"),
]
TOKENS, HEADS, BITS = 8, 8, 3


def ensure_inputs(work: Path, py: str) -> Path:
    data = work / "data"
    if not (data / "meta.json").exists():
        subprocess.run([py, str(HERE / "gen_test_data.py"), "--out", str(data)], check=True)
    kv_dir = work / "inputs" / "neg_base"
    if not (kv_dir / "kv_vectors.bin").exists():
        kv_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([py, str(HERE / "gen_case_input.py"), "--out", str(kv_dir),
                        "--tokens", str(TOKENS), "--heads", str(HEADS)], check=True)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(HERE / "_out"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    bin_path = os.environ.get("KCTQ_BIN", "/tmp/kctq_example/test_aclnn_kv_cache_turbo_quant")
    if not os.access(bin_path, os.X_OK):
        raise SystemExit(f"找不到可执行文件 {bin_path}（先 bash examples/build_and_run.sh <pkg>/packages）")
    work = Path(args.work)
    data = ensure_inputs(work, sys.executable)
    kv_dir = work / "inputs" / "neg_base"
    out_dir = work / "out" / "invalid"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, wrong = [], []
    for name, mode, why in MODES:
        cmd = [bin_path, "--kv", str(kv_dir / "kv_vectors.bin"), "--rot", str(data / "rotation_matrix.bin"),
               "--qjl", str(data / "qjl_matrix.bin"), "--out", str(out_dir), "--tokens", str(TOKENS),
               "--heads", str(HEADS), "--bits", str(BITS), "--warmup", "1", "--repeat", "1", "--label", name]
        if mode:
            cmd += ["--neg", mode]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        rejected = proc.returncode != 0
        entry = {"case": name, "mode": mode or "(none)", "why": why, "exit_code": proc.returncode,
                 "rejected": rejected}
        if rejected:
            entry["error"] = next((ln.strip() for ln in log.splitlines()
                                   if "failed" in ln or "NOT REJECTED" in ln), "")
        expect_reject = bool(mode)
        entry["ok"] = rejected == expect_reject
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if not entry["ok"]:
            wrong.append(name)

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    n_neg = sum(1 for r in results if r["mode"] != "(none)")
    n_rej = sum(1 for r in results if r["mode"] != "(none)" and r["rejected"])
    base_ok = all(not r["rejected"] for r in results if r["case"] == "valid_base")
    print(f"\n== 汇总：非法用例 {n_neg} 个被拒绝 {n_rej} 个；合法基准{'被接受' if base_ok else '被拒绝（异常）'} ==")
    if wrong:
        print("异常用例：", wrong)
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
