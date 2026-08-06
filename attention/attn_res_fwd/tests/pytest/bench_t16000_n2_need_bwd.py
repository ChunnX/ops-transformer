#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""T=16000 N=2 H=7168：融合算子 need_backward 开/关 × 精度(golden/小算子) + 性能 + Ascend Profiler。

用法:
  cd attention/attn_res_fwd/tests/pytest
  source env_kdy.sh && source cann_env.sh
  DEVICE_ID=7 python bench_t16000_n2_need_bwd.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
import torch_npu

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[4] / "torch_extension"))

from precision_compare import data_compare  # noqa: E402
from attn_res_fwd_golden import compute_mere_mare  # noqa: E402
from attn_res_fwd_torch_runner import get_attn_res_fwd_fn  # noqa: E402

T, N, H = 16000, 2, 7168
B = N + 1
SEED = 0
BF16_THD = 1.0 / 128.0
FP32_THD = 1.0 / 1024.0
WARMUP = 3
ITERS = 10


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


def small_ops_forward(ps, br, pw, nw, eps, need_bwd: bool):
    """NPU 小算子分解（与 bench_attn_res_fwd.apply_attn_res_small_ops 一致）。"""
    v = torch.cat((br, ps.unsqueeze(1)), dim=1)
    v_f = v.float()
    inv = torch.rsqrt(v_f.pow(2).mean(dim=-1) + eps)
    k = v_f * inv.unsqueeze(-1)
    scores = (k * (nw.float() * pw.squeeze(0).float())).sum(dim=-1)
    probs = torch.softmax(scores, dim=-1)
    hidden = (probs.unsqueeze(-1) * v_f).sum(dim=1).to(v.dtype)
    if need_bwd:
        return hidden, inv, probs
    return hidden


def check_tensor(name: str, actual: torch.Tensor, expect: torch.Tensor, thd: float) -> dict:
    a = actual.detach().float().cpu()
    e = expect.detach().float().cpu()
    mere, mare = compute_mere_mare(a, e)
    finite = bool(torch.isfinite(a).all().item())
    with contextlib.redirect_stdout(io.StringIO()):
        dc, pct, max_err = data_compare(a.numpy().reshape(-1), e.numpy().reshape(-1), thd, thd, thd)
    ok = finite and dc == "success"
    return {
        "name": name,
        "ok": ok,
        "dc": dc,
        "pct": float(pct),
        "mere": float(mere),
        "mare": float(mare),
        "max_err": float(max_err),
        "finite": finite,
    }


def bench_ms(fn, *args) -> float:
    for _ in range(WARMUP):
        fn(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / ITERS


def _find_profiler_csv(base_dir: str, filename: str) -> Optional[str]:
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _parse_kernel_duration_us(profiler_dir: str) -> Tuple[Optional[float], Optional[str]]:
    csv_path = _find_profiler_csv(profiler_dir, "kernel_details.csv")
    if csv_path is None:
        return None, None
    df = pd.read_csv(csv_path)
    duration_col = next((c for c in df.columns if c.strip().startswith("Duration")), None)
    if duration_col is None:
        return None, csv_path
    return float(df[duration_col].astype(float).sum()), csv_path


def _parse_op_statistic_time_us(profiler_dir: str) -> Tuple[Optional[float], Optional[str]]:
    csv_path = _find_profiler_csv(profiler_dir, "op_statistic.csv")
    if csv_path is None:
        return None, None
    df = pd.read_csv(csv_path)
    total_time_col = next((c for c in df.columns if "Total Time" in c), None)
    if total_time_col is None:
        return None, csv_path
    return float(df[total_time_col].astype(float).sum()), csv_path


def _create_profiler(result_path: str):
    os.makedirs(result_path, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path),
    )


def profile_once(name: str, fn: Callable, args, output_root: str) -> dict:
    profiler_dir = os.path.join(output_root, name)
    for _ in range(WARMUP):
        fn(*args)
        torch.npu.synchronize()
    prof = _create_profiler(profiler_dir)
    prof.start()
    fn(*args)
    torch.npu.synchronize()
    prof.stop()

    kernel_us, kernel_csv = _parse_kernel_duration_us(profiler_dir)
    op_us, op_csv = _parse_op_statistic_time_us(profiler_dir)
    print(f"\n===== profiler ({name}) =====")
    print(f"profiler_dir: {profiler_dir}")
    if kernel_us is not None:
        print(f"kernel_details Duration(us) sum: {kernel_us:.3f}")
        print(f"kernel_details csv: {kernel_csv}")
    else:
        print("kernel_details: missing")
    if op_us is not None:
        print(f"op_statistic Total Time(us) sum: {op_us:.3f}")
        print(f"op_statistic csv: {op_csv}")
    else:
        print("op_statistic: missing")
    return {
        "name": name,
        "profiler_dir": profiler_dir,
        "kernel_duration_us": kernel_us,
        "op_total_time_us": op_us,
    }


def main() -> int:
    device_id = int(os.environ.get("DEVICE_ID", "7"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    print(f"device={device} shape=T={T} N={N} H={H} B={B} warmup={WARMUP} iters={ITERS}")

    torch.manual_seed(SEED)
    dtype = torch.bfloat16
    ps = torch.randn(T, H, device=device, dtype=dtype)
    br = torch.randn(T, N, H, device=device, dtype=dtype)
    pw = torch.randn(1, H, device=device, dtype=dtype)
    nw = torch.randn(H, device=device, dtype=dtype)
    eps = 1e-5

    fused = get_attn_res_fwd_fn()
    n_fail = 0

    print("\n========== PRECISION ==========")
    for need_bwd in (False, True):
        tag = f"need_bwd={need_bwd}"
        print(f"\n----- {tag} -----")
        gold = golden_forward(ps.cpu(), br.cpu(), pw.cpu(), nw.cpu(), eps, need_bwd)
        small = small_ops_forward(ps, br, pw, nw, eps, need_bwd)
        torch.npu.synchronize()
        out = fused(ps, br, pw, nw, eps, need_bwd)
        torch.npu.synchronize()

        if need_bwd:
            h, inv, probs = out
            gh, gi, gp = gold
            sh, si, sp = small
            checks = [
                check_tensor("hidden_vs_golden", h, gh, BF16_THD),
                check_tensor("inv_vs_golden", inv, gi, FP32_THD),
                check_tensor("probs_vs_golden", probs, gp, FP32_THD),
                check_tensor("hidden_vs_small", h, sh, BF16_THD),
                check_tensor("inv_vs_small", inv, si, FP32_THD),
                check_tensor("probs_vs_small", probs, sp, FP32_THD),
            ]
        else:
            h = out
            gh = gold
            sh = small
            checks = [
                check_tensor("hidden_vs_golden", h, gh, BF16_THD),
                check_tensor("hidden_vs_small", h, sh, BF16_THD),
            ]

        for c in checks:
            status = "PASS" if c["ok"] else "FAIL"
            if not c["ok"]:
                n_fail += 1
            print(
                f"  [{status}] {c['name']}: dc={c['dc']} pct={c['pct']:.2f} "
                f"mere={c['mere']:.6g} mare={c['mare']:.6g} finite={c['finite']}"
            )

        # small vs golden（确认小算子本身可信）
        if need_bwd:
            sh, si, sp = small
            gh, gi, gp = gold
            for name, a, e, thd in (
                ("small_hidden_vs_golden", sh, gh, BF16_THD),
                ("small_inv_vs_golden", si, gi, FP32_THD),
                ("small_probs_vs_golden", sp, gp, FP32_THD),
            ):
                c = check_tensor(name, a, e, thd)
                status = "PASS" if c["ok"] else "FAIL"
                if not c["ok"]:
                    n_fail += 1
                print(
                    f"  [{status}] {c['name']}: dc={c['dc']} pct={c['pct']:.2f} "
                    f"mere={c['mere']:.6g}"
                )
        else:
            c = check_tensor("small_hidden_vs_golden", small, gold, BF16_THD)
            status = "PASS" if c["ok"] else "FAIL"
            if not c["ok"]:
                n_fail += 1
            print(
                f"  [{status}] {c['name']}: dc={c['dc']} pct={c['pct']:.2f} mere={c['mere']:.6g}"
            )

    print("\n========== PERFORMANCE (ms/op) ==========")
    rows = []
    for need_bwd in (False, True):
        ms_fused = bench_ms(fused, ps, br, pw, nw, eps, need_bwd)
        ms_small = bench_ms(small_ops_forward, ps, br, pw, nw, eps, need_bwd)
        spd = ms_small / ms_fused if ms_fused > 0 else float("nan")
        rows.append((need_bwd, ms_fused, ms_small, spd))
        print(
            f"  need_bwd={need_bwd}: fused={ms_fused:.3f} ms  "
            f"small_ops={ms_small:.3f} ms  speedup={spd:.3f}x"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = str(SCRIPT_DIR / "profiler_output" / f"bench_t16000_n2_{ts}")
    os.makedirs(output_root, exist_ok=True)
    print(f"\n========== ASCEND PROFILER ==========")
    print(f"output_root: {output_root}")
    prof_rows = []
    for need_bwd in (False, True):
        tag = "bwd1" if need_bwd else "bwd0"
        prof_rows.append(
            profile_once(f"fused_{tag}", fused, (ps, br, pw, nw, eps, need_bwd), output_root)
        )
        prof_rows.append(
            profile_once(
                f"small_{tag}", small_ops_forward, (ps, br, pw, nw, eps, need_bwd), output_root
            )
        )

    print("\n========== SUMMARY ==========")
    print(f"{'need_bwd':<10} {'fused_ms':>10} {'small_ms':>10} {'speedup':>10}")
    for need_bwd, ms_f, ms_s, spd in rows:
        print(f"{str(need_bwd):<10} {ms_f:10.3f} {ms_s:10.3f} {spd:9.3f}x")
    print(f"\n{'name':<14} {'kernel_us':>12} {'op_us':>12}")
    for r in prof_rows:
        ku = r["kernel_duration_us"]
        ou = r["op_total_time_us"]
        ku_s = f"{ku:.1f}" if ku is not None else "NA"
        ou_s = f"{ou:.1f}" if ou is not None else "NA"
        print(f"{r['name']:<14} {ku_s:>12} {ou_s:>12}")
    print(f"\nprofiler_root: {output_root}")
    print(f"precision_fails={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
