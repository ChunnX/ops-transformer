#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""OOM 编译包精度扫：T=1..200，N=1..12，H=7168，need_backward 开/关。

用法:
  cd attention/attn_res_fwd/tests/pytest
  source env_kdy.sh && source cann_env.sh
  DEVICE_ID=7 python grid_oom_t1_200_precision.py
  # 出错即停：默认开启；全量继续：STOP_ON_FAIL=0
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from precision_compare import data_compare  # noqa: E402

from attn_res_fwd_golden import compute_mere_mare
from cann_ops_transformer.ops.attn_res_fwd import attn_res_fwd

BF16_THD = 1.0 / 128.0
FP32_THD = 1.0 / 1024.0
H = 7168
SEED = 0
T_MIN = int(os.environ.get("T_MIN", "1"))
T_MAX = int(os.environ.get("T_MAX", "200"))
N_VALUES = list(range(1, 13))
STOP_ON_FAIL = os.environ.get("STOP_ON_FAIL", "1") != "0"


def golden_forward(ps, br, pw, nw, eps, need_bwd: bool):
    v = torch.cat((br, ps.unsqueeze(1)), dim=1).float()
    mean_sq = v.pow(2).mean(-1)
    inv = 1.0 / torch.sqrt(mean_sq + eps)
    k = v * inv.unsqueeze(-1)
    scores = (k * (nw.float() * pw.squeeze(0).float())).sum(-1)
    probs = scores.softmax(-1)
    hidden = (probs.unsqueeze(-1) * v).sum(1).to(torch.bfloat16)
    if need_bwd:
        return hidden, inv, probs
    return hidden


def check_tensor(actual: torch.Tensor, expect: torch.Tensor, thd: float) -> Tuple[bool, float, str, float]:
    import contextlib
    import io

    a = actual.detach().float().cpu()
    e = expect.detach().float().cpu()
    mere, _ = compute_mere_mare(a, e)
    finite = bool(torch.isfinite(a).all().item())
    with contextlib.redirect_stdout(io.StringIO()):
        dc, pct, _ = data_compare(a.numpy().reshape(-1), e.numpy().reshape(-1), thd, thd, thd)
    ok = finite and dc == "success"
    return ok, mere, dc, float(pct)


def run_one(fn, T: int, N: int, need_bwd: bool, device: str) -> dict:
    torch.manual_seed(SEED)
    dtype = torch.bfloat16
    ps = torch.randn(T, H, device=device, dtype=dtype)
    br = torch.randn(T, N, H, device=device, dtype=dtype)
    pw = torch.randn(1, H, device=device, dtype=dtype)
    nw = torch.randn(H, device=device, dtype=dtype)
    eps = 1e-5

    gold = golden_forward(ps.cpu(), br.cpu(), pw.cpu(), nw.cpu(), eps, need_bwd)
    out = fn(ps, br, pw, nw, eps, need_bwd)
    torch.npu.synchronize()

    row = {"T": T, "N": N, "H": H, "need_backward": need_bwd, "pass": True}
    if need_bwd:
        h, inv, probs = out
        gh, gi, gp = gold
        ok_h, mere_h, dc_h, pct_h = check_tensor(h, gh.to(device), BF16_THD)
        ok_i, mere_i, dc_i, pct_i = check_tensor(inv, gi.to(device), FP32_THD)
        ok_p, mere_p, dc_p, pct_p = check_tensor(probs, gp.to(device), FP32_THD)
        row.update(
            {
                "hidden_ok": ok_h,
                "hidden_mere": mere_h,
                "hidden_dc": dc_h,
                "hidden_pct": pct_h,
                "inv_ok": ok_i,
                "inv_mere": mere_i,
                "inv_dc": dc_i,
                "inv_pct": pct_i,
                "probs_ok": ok_p,
                "probs_mere": mere_p,
                "probs_dc": dc_p,
                "probs_pct": pct_p,
                "pass": ok_h and ok_i and ok_p,
            }
        )
    else:
        h = out
        gh = gold
        ok_h, mere_h, dc_h, pct_h = check_tensor(h, gh.to(device), BF16_THD)
        row.update(
            {
                "hidden_ok": ok_h,
                "hidden_mere": mere_h,
                "hidden_dc": dc_h,
                "hidden_pct": pct_h,
                "pass": ok_h,
            }
        )
    return row


def main() -> None:
    device_id = int(os.environ.get("DEVICE_ID", "7"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    cases = [(t, n) for t in range(T_MIN, T_MAX + 1) for n in N_VALUES]
    total = len(cases) * 2
    print(f"Using NPU device: {device_id}")
    print(f"OOM precision grid: T=[{T_MIN},{T_MAX}] N={N_VALUES} H={H}")
    print(f"shapes={len(cases)} × need_backward={{False,True}} → total={total}")
    print(f"STOP_ON_FAIL={STOP_ON_FAIL}")

    out_dir = os.path.join(
        "profiler_output", f"grid_oom_t{T_MIN}_{T_MAX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    rows: List[dict] = []
    n_fail = 0
    done = 0
    for i, (T, N) in enumerate(cases, 1):
        for need_bwd in (False, True):
            done += 1
            tag = f"[{done}/{total}] T={T} N={N} need_bwd={need_bwd}"
            try:
                row = run_one(attn_res_fwd, T, N, need_bwd, device)
                rows.append(row)
                status = "PASS" if row["pass"] else "FAIL"
                if not row["pass"]:
                    n_fail += 1
                    print(
                        f"{tag} {status} h={row.get('hidden_dc')} "
                        f"inv={row.get('inv_dc')} probs={row.get('probs_dc')} "
                        f"mere_h={row.get('hidden_mere')} mere_i={row.get('inv_mere')}",
                        flush=True,
                    )
                    if STOP_ON_FAIL:
                        break
                elif done % 50 == 0 or T in (1, 50, 100, 150, 200) and N == 1 and not need_bwd:
                    print(f"{tag} {status}", flush=True)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                rows.append(
                    {
                        "T": T,
                        "N": N,
                        "H": H,
                        "need_backward": need_bwd,
                        "pass": False,
                        "error": str(e),
                    }
                )
                print(f"{tag} ERROR {e}", flush=True)
                if STOP_ON_FAIL:
                    break
            torch.npu.empty_cache()
        else:
            continue
        break

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "grid_oom_precision.csv")
    df.to_csv(csv_path, index=False)
    n_pass = int(df["pass"].sum()) if len(df) else 0
    print("\n===== summary =====")
    print(f"ran={len(df)} pass={n_pass} fail={len(df) - n_pass}")
    if n_fail:
        fails = df[~df["pass"]]
        print(fails.head(20).to_string(index=False))
        fails.to_csv(os.path.join(out_dir, "failures_detail.csv"), index=False)
    print(f"csv: {csv_path}")
    raise SystemExit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
