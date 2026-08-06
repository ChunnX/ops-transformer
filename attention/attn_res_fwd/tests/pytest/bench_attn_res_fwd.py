#!/usr/bin/env python3
"""AttnResFwd 精度(data_compare) + 性能基准（H=7168, N<8, T 采样）。

用法:
  cd attention/attn_res_fwd/tests/pytest
  source env_kdy.sh && source cann_env.sh
  python bench_attn_res_fwd.py --mode fused --step baseline
  python bench_attn_res_fwd.py --mode compare --step vs_small_ops --append ../../docs/GETVALUE_REMOVAL_PERF.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import torch
import torch_npu

from attn_res_fwd_golden import apply_attn_res_cust, compare_with_data_compare
from attn_res_fwd_torch_runner import get_attn_res_fwd_fn
from npu_device import resolve_device_id

# H=7168, N<8, T 从 1 到 16384 挑 10 个典型 shape
BENCH_CASES: List[Dict] = [
    {"id": "t1_n1", "T": 1, "N": 1, "H": 7168, "seed": 0},
    {"id": "t1_n4", "T": 1, "N": 4, "H": 7168, "seed": 0},
    {"id": "t1_n7", "T": 1, "N": 7, "H": 7168, "seed": 0},
    {"id": "t16_n1", "T": 16, "N": 1, "H": 7168, "seed": 0},
    {"id": "t16_n4", "T": 16, "N": 4, "H": 7168, "seed": 0},
    {"id": "t128_n2", "T": 128, "N": 2, "H": 7168, "seed": 0},
    {"id": "t512_n6", "T": 512, "N": 6, "H": 7168, "seed": 0},
    {"id": "t1024_n1", "T": 1024, "N": 1, "H": 7168, "seed": 0},
    {"id": "t4096_n4", "T": 4096, "N": 4, "H": 7168, "seed": 0},
    {"id": "t16384_n2", "T": 16384, "N": 2, "H": 7168, "seed": 0},
    {"id": "prod", "T": 544, "N": 12, "H": 7168, "seed": 0},
]


def apply_attn_res_small_ops(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """NPU 小算子分解：cat → rmsnorm(rsqrt) → dot → softmax → weighted sum（逐步 torch 下发）。"""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_f = v.float()
    inv_rms = torch.rsqrt(v_f.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
    k = v_f * inv_rms
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(dim=-1)
    probs = torch.softmax(scores, dim=-1).unsqueeze(1)
    hidden = (probs.unsqueeze(-1) * v_f.unsqueeze(1)).sum(dim=2).squeeze(1)
    return hidden.to(v.dtype)


def _prepare(T: int, N: int, H: int, seed: int, device: str, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    ps = torch.randn(T, H, device=device, dtype=dtype)
    br = torch.randn(T, N, H, device=device, dtype=dtype)
    pw = torch.randn(1, H, device=device, dtype=dtype)
    nw = torch.randn(H, device=device, dtype=dtype)
    return ps, br, pw, nw, 1e-5


def _bench_ms(fn: Callable, ps, br, pw, nw, eps, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(ps, br, pw, nw, eps)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(ps, br, pw, nw, eps)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _run_one_impl(
    impl: str,
    fn: Callable,
    step: str,
    warmup: int,
    iters: int,
    dev: int,
    d: str,
) -> List[Tuple]:
    rows = []
    hdr = f"{'case':<12} {'T':>6} {'N':>3} {'impl':<10} {'dc':>8} {'pct':>7} {'mare':>10} {'ms':>10}"
    print(hdr)
    for c in BENCH_CASES:
        T, N, H, seed = c["T"], c["N"], c["H"], c["seed"]
        ps, br, pw, nw, eps = _prepare(T, N, H, seed, d)
        ps_cpu, br_cpu, pw_cpu, nw_cpu = ps.cpu(), br.cpu(), pw.cpu(), nw.cpu()
        out = fn(ps, br, pw, nw, eps)
        torch.npu.synchronize()
        gold = apply_attn_res_cust(ps_cpu.float(), br_cpu.float(), pw_cpu.float(), nw_cpu.float(), eps)
        st = compare_with_data_compare(out.cpu(), gold)
        ms = _bench_ms(fn, ps, br, pw, nw, eps, warmup, iters)
        print(
            f"{c['id']:<12} {T:6d} {N:3d} {impl:<10} {st['data_compare']:>8s} "
            f"{st['pct']:7.2f} {st['mare']:10.6g} {ms:10.3f}"
        )
        rows.append((step, c["id"], T, N, H, impl, st["data_compare"], st["pct"], st["mare"], ms))
    return rows


def run_step(step: str, mode: str, warmup: int, iters: int) -> List[Tuple]:
    dev = resolve_device_id(force_auto=True)
    torch.npu.set_device(dev)
    d = f"npu:{dev}"
    fused_fn = get_attn_res_fwd_fn()
    small_fn = apply_attn_res_small_ops
    print(f"step={step} mode={mode} device={dev} warmup={warmup} iters={iters}")

    if mode == "fused":
        return _run_one_impl("fused", fused_fn, step, warmup, iters, dev, d)
    if mode == "small":
        return _run_one_impl("small_ops", small_fn, step, warmup, iters, dev, d)

    fused_rows = _run_one_impl("fused", fused_fn, step, warmup, iters, dev, d)
    print()
    small_rows = _run_one_impl("small_ops", small_fn, step, warmup, iters, dev, d)

    fused_ms = {r[1]: r[9] for r in fused_rows}
    cmp_rows = []
    print(f"\n{'case':<12} {'T':>6} {'N':>3} {'fused_ms':>10} {'small_ms':>10} {'speedup':>8} {'dc_fused':>10}")
    for r in small_rows:
        cid, T, N, H = r[1], r[2], r[3], r[4]
        fms = fused_ms[cid]
        sms = r[9]
        spd = sms / fms if fms > 0 else float("nan")
        dc_f = next(x[6] for x in fused_rows if x[1] == cid)
        print(f"{cid:<12} {T:6d} {N:3d} {fms:10.3f} {sms:10.3f} {spd:8.3f}x {dc_f:>10s}")
        cmp_rows.append((step, cid, T, N, H, fms, sms, spd, dc_f, r[6]))
    return cmp_rows


def append_md(path: str, step: str, rows: List[Tuple], mode: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {step} ({ts}) mode={mode}\n\n")
        if mode == "compare" and rows and len(rows[0]) >= 10:
            f.write("| case | T | N | H | fused_ms | small_ms | speedup | dc_fused | dc_small |\n")
            f.write("|------|---|---|---|----------|----------|---------|----------|----------|\n")
            for r in rows:
                _, cid, T, N, H, fms, sms, spd, dc_f, dc_s = r
                f.write(
                    f"| {cid} | {T} | {N} | {H} | {fms:.3f} | {sms:.3f} | {spd:.3f}x | {dc_f} | {dc_s} |\n"
                )
        else:
            f.write("| case | T | N | H | impl | data_compare | pct | mare | ms/op |\n")
            f.write("|------|---|---|---|------|--------------|-----|------|-------|\n")
            for r in rows:
                if len(r) == 9:
                    _, cid, T, N, H, dc, pct, mare, ms = r
                    f.write(f"| {cid} | {T} | {N} | {H} | fused | {dc} | {pct:.2f} | {mare:.6g} | {ms:.3f} |\n")
                else:
                    _, cid, T, N, H, impl, dc, pct, mare, ms = r
                    f.write(
                        f"| {cid} | {T} | {N} | {H} | {impl} | {dc} | {pct:.2f} | {mare:.6g} | {ms:.3f} |\n"
                    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="baseline", help="步骤标签")
    ap.add_argument("--mode", choices=("fused", "small", "compare"), default="compare")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--append", default="", help="追加到 markdown")
    args = ap.parse_args()
    rows = run_step(args.step, args.mode, args.warmup, args.iters)
    if args.append:
        append_md(args.append, args.step, rows, args.mode)
        print(f"appended -> {args.append}")
    if args.mode == "compare":
        dc_fail = sum(1 for r in rows if r[8] != "success" or r[9] != "success")
    else:
        dc_fail = sum(1 for r in rows if (r[6] if len(r) > 9 else r[5]) != "success")
    if dc_fail:
        print(f"WARNING: {dc_fail} cases data_compare != success")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
