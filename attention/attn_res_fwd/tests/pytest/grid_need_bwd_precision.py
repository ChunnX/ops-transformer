#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""泛化精度：H=7168，T 采样 ∈[1,16384]，N=1..12，need_backward 开/关。

用法:
  cd attention/attn_res_fwd/tests/pytest
  source env_kdy.sh && source cann_env.sh
  python grid_need_bwd_precision.py
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
from test_attn_res_fwd_paramset import GRID_N_VALUES, GRID_T_VALUES, build_grid_tn_7168_cases

BF16_THD = 1.0 / 128.0
FP32_THD = 1.0 / 1024.0
H = 7168
SEED = 0


def golden_forward(ps, br, pw, nw, eps, need_bwd: bool):
    """CPU golden；inv 用 1/sqrt（对齐 kernel InvRmsInPlace）。"""
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


def worst_point(actual: torch.Tensor, expect: torch.Tensor, n_cols: int | None = None) -> dict:
    """最大绝对误差点位与数值。"""
    a = actual.detach().float().cpu().contiguous()
    e = expect.detach().float().cpu().contiguous()
    diff = (a - e).abs()
    flat = diff.view(-1)
    idx = int(flat.argmax().item())
    act_v = float(a.view(-1)[idx].item())
    exp_v = float(e.view(-1)[idx].item())
    abs_v = float(flat[idx].item())
    rel_v = abs_v / max(abs(exp_v), 1e-10)
    info = {
        "worst_idx": idx,
        "worst_act": act_v,
        "worst_exp": exp_v,
        "worst_abs": abs_v,
        "worst_rel": rel_v,
    }
    if n_cols is not None and n_cols > 0:
        info["worst_token"] = idx // n_cols
        info["worst_branch"] = idx % n_cols
    return info


def check_tensor(name: str, actual: torch.Tensor, expect: torch.Tensor, thd: float) -> Tuple[bool, float, str, float]:
    """静默比对：finite + data_compare(pct/max)，不打印明细。返回 (ok, mere, dc, pct)。"""
    import contextlib
    import io

    a = actual.detach().float().cpu()
    e = expect.detach().float().cpu()
    mere, _ = compute_mere_mare(a, e)
    finite = bool(torch.isfinite(a).all().item())
    with contextlib.redirect_stdout(io.StringIO()):
        dc, pct, max_err = data_compare(a.numpy().reshape(-1), e.numpy().reshape(-1), thd, thd, thd)
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
        b = N + 1
        ok_h, mere_h, dc_h, pct_h = check_tensor("hidden", h, gh.to(device), BF16_THD)
        ok_i, mere_i, dc_i, pct_i = check_tensor("inv_rms", inv, gi.to(device), FP32_THD)
        ok_p, mere_p, dc_p, pct_p = check_tensor("probs", probs, gp.to(device), FP32_THD)
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
        if not ok_h:
            for k, v in worst_point(h, gh, H).items():
                row[f"hidden_{k}"] = v
        if not ok_i:
            for k, v in worst_point(inv, gi, b).items():
                row[f"inv_{k}"] = v
        if not ok_p:
            for k, v in worst_point(probs, gp, b).items():
                row[f"probs_{k}"] = v
    else:
        h = out
        gh = gold
        ok_h, mere_h, dc_h, pct_h = check_tensor("hidden", h, gh.to(device), BF16_THD)
        row.update(
            {
                "hidden_ok": ok_h,
                "hidden_mere": mere_h,
                "hidden_dc": dc_h,
                "hidden_pct": pct_h,
                "inv_ok": None,
                "inv_mere": None,
                "inv_dc": None,
                "probs_ok": None,
                "probs_mere": None,
                "probs_dc": None,
                "pass": ok_h,
            }
        )
        if not ok_h:
            for k, v in worst_point(h, gh, H).items():
                row[f"hidden_{k}"] = v
    return row


def main() -> None:
    device_id = int(os.environ.get("DEVICE_ID", "7"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    print(f"Using NPU device: {device_id}")
    print(f"GRID_T={GRID_T_VALUES}")
    print(f"GRID_N={GRID_N_VALUES}")

    cases = build_grid_tn_7168_cases()
    print(f"shapes={len(cases)} × need_backward={{False,True}} → total={len(cases) * 2}")

    out_dir = os.path.join(
        "profiler_output", f"grid_need_bwd_prec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    rows: List[dict] = []
    n_fail = 0
    for i, c in enumerate(cases, 1):
        T, N = c["T"], c["N"]
        for need_bwd in (False, True):
            tag = f"[{i}/{len(cases)}] T={T} N={N} need_bwd={need_bwd}"
            try:
                row = run_one(attn_res_fwd, T, N, need_bwd, device)
                rows.append(row)
                status = "PASS" if row["pass"] else "FAIL"
                if not row["pass"]:
                    n_fail += 1
                    msg = (
                        f"{tag} {status} "
                        f"h={row.get('hidden_dc')} inv={row.get('inv_dc')} probs={row.get('probs_dc')} "
                        f"inv_mere={row.get('inv_mere')} inv_pct={row.get('inv_pct')}"
                    )
                    if row.get("inv_worst_idx") is not None:
                        msg += (
                            f" | inv_worst idx={row['inv_worst_idx']} "
                            f"token={row.get('inv_worst_token')} branch={row.get('inv_worst_branch')} "
                            f"act={row.get('inv_worst_act')} exp={row.get('inv_worst_exp')} "
                            f"abs={row.get('inv_worst_abs')} rel={row.get('inv_worst_rel')}"
                        )
                    if row.get("hidden_worst_idx") is not None:
                        msg += (
                            f" | h_worst idx={row['hidden_worst_idx']} "
                            f"act={row.get('hidden_worst_act')} exp={row.get('hidden_worst_exp')}"
                        )
                    if row.get("probs_worst_idx") is not None:
                        msg += (
                            f" | p_worst idx={row['probs_worst_idx']} "
                            f"act={row.get('probs_worst_act')} exp={row.get('probs_worst_exp')}"
                        )
                    print(msg, flush=True)
                else:
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
            torch.npu.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "grid_need_bwd_precision.csv")
    df.to_csv(csv_path, index=False)

    total = len(df)
    n_pass = int(df["pass"].sum()) if total else 0
    by_flag = df.groupby("need_backward")["pass"].agg(["sum", "count"])
    print("\n===== summary =====")
    print(f"total={total} pass={n_pass} fail={total - n_pass}")
    print(by_flag.to_string())
    if n_fail:
        fails = df[~df["pass"]]
        print("\n----- failures (head 30) -----")
        cols = [
            c
            for c in (
                "T",
                "N",
                "need_backward",
                "hidden_dc",
                "inv_dc",
                "probs_dc",
                "inv_mere",
                "inv_pct",
                "inv_worst_idx",
                "inv_worst_token",
                "inv_worst_branch",
                "inv_worst_act",
                "inv_worst_exp",
                "inv_worst_abs",
                "inv_worst_rel",
                "error",
            )
            if c in fails.columns
        ]
        print(fails[cols].head(30).to_string(index=False))
        detail_path = os.path.join(out_dir, "failures_detail.csv")
        fails.to_csv(detail_path, index=False)
        print(f"failures_detail: {detail_path}")
    print(f"\ncsv: {csv_path}")
    raise SystemExit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
