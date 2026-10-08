#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""causal_conv1d 反向性能对比：PyTorch 小算子拼接(A) vs 自定义 kernel(B)。

A 路线（小算子拼接标杆，用户指定基准）：
  channel-first 布局（kimik3 原始形态）：
    x_c = x.transpose(1, 2)                     # (B, D, T)
    w_c = weight.view(D, 1, kW)                 # depthwise groups=D
    z   = F.conv1d(F.pad(x_c, (kW-1, 0)), w_c, bias)
    y   = F.silu(z)
    loss = (y * grad_y_c).sum()                 # 上游梯度注入
    loss.backward() 由 autograd 自动拼小算子完成反向
  退路：F.conv1d 在 torch_npu bf16/groups=12288 不支持时，改手动拼接
  （F.pad + slice 移位 + mul + add 组合出 z，同样建图 backward）。
  注：A 前向输出 y 为 channel-first (B, D, T)（PyTorch conv1d 原生形态），
      未含输出转回 (B, T, D) 的转置开销（该口径对 A 略有利）。
      三路梯度经 autograd 自动回到 (B,T,D)/(kW,D)/(D,) 布局，与 B 对齐。

B 路线（自定义 kernel）：
  causal_conv1d_fn_autograd = CausalConv1dFnAutoGrad（前向 aclnnCausalConv1dFn，
  反向 aclnnCausalConv1dGrad 自定义算子）。

对比协议（公平性）：
  - 同输入数据（同 seed 同分布，A/B 各自 clone 同一组张量）
  - 同 shape 矩阵：B in {1,2,4} x T in {512,1024,2048,4096,8192}（15 组）
  - 计时三组（torch_npu.npu.Event，device 时间）：
      fwd_only : torch.no_grad() 下前向
      bwd_only : 前向建图一次，retain_graph 循环 backward
                 （先跑 >=3 次预热，再计 >=20 次）
      total    : 每 iter 完整前向 + backward（完整 autograd 路径）
  - 输出每 shape：A bwd / B bwd / 加速比 speedup=A_bwd/B_bwd（<1 表示自定义更慢）

正确性（对比有意义的前提）：A/B 三路梯度 vs CPU golden.py
  bf16 混合容差 rtol=atol=4e-3，matched_ratio >= 0.99 判通过。

用法：
  python3 profile_bwd_compare_composite_vs_kernel.py [--verify-only] [--quick]
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# ---- 环境与路径（必须在 import torch / torch_npu 之前） ----
_THIS_DIR = Path(__file__).resolve().parent
_CONV1D_DIR = _THIS_DIR.parent.parent                                # mamba/causal_conv1d
_SRC_ROOT = _CONV1D_DIR.parents[1]                                   # ops-transformer 根
_GRAD_REPO = Path(
    "/workspace/huangqiang/hq_claude/code/cannbot-skills/plugins-official/"
    "ops-registry-invoke/operators/causal_conv1d_grad"
)
_CUSTOM_OPP_VENDOR = _GRAD_REPO / "custom_opp" / "vendors" / "causal_conv1d_grad_custom"

os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", str(_CUSTOM_OPP_VENDOR))
sys.path.insert(0, str(_SRC_ROOT / "torch_extension"))  # 源码仓 cann_ops_transformer 优先
sys.path.insert(0, str(_CONV1D_DIR))                    # torch_extension 包可导入
sys.path.insert(0, str(_GRAD_REPO / "tests" / "st"))    # golden.py

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: E402

from torch_extension.causal_conv1d_grad_ext import (  # noqa: E402
    causal_conv1d_fn_autograd,
)
from golden import causal_conv1d_grad_golden  # noqa: E402

# ---- kimik3 场景参数 ----
BATCH_SIZES = [1, 2, 4]
SEQ_LENS = [512, 1024, 2048, 4096, 8192]
DIM = 12288
KW = 4
DTYPE = torch.bfloat16
ACTIVATION = "silu"
DEVICE = "npu:0"

RTOL = 4e-3
ATOL = 4e-3
THRESHOLD = 0.99

WARMUP_ITERS = 10   # >= 10；bwd_only 预热 >= 3 次由此覆盖
PROFILE_ITERS = 30  # >= 20


# =========================================================================
# A 路线：小算子拼接（conv1d 原生 / manual 移位拼接退路）
# =========================================================================
def a_forward(x, weight, bias, mode):
    """A 路线前向：输入 (B,T,D)/(kW,D)/(D,)，输出 y 为 channel-first (B,D,T)。"""
    x_c = x.transpose(1, 2)  # (B, D, T)
    t_len = x.shape[1]
    if mode == "conv1d":
        # depthwise + 因果卷积方向对齐（与 golden z[t]=Σ_j w[j]·x[t-j] 一致）：
        #   1. flip(0)：PyTorch conv1d 是互相关（z[t]=Σ_k w_c[k]·x_pad[t+k]），
        #      golden 语义需 w_c[k]=weight[kW-1-k]，即核翻转（真卷积）
        #   2. t().reshape(D,1,kW)：weight (kW,D) -> (D,1,kW)；直接
        #      weight.view(D,1,kW) 是行优先跨维乱序，语义错误
        #   3. groups=DIM：depthwise 必需（任务伪代码注释为 depthwise，原生
        #      接口要求显式 groups）
        # 三者的梯度（flip/transpose/reshape 的 backward）均由 autograd 自动反传，
        # dweight 自动回到 (kW, D) 布局。
        w_c = weight.flip(0).t().reshape(DIM, 1, KW)
        # causal：左 pad kW-1、conv padding=0
        z = F.conv1d(F.pad(x_c, (KW - 1, 0)), w_c, bias, groups=DIM)
    elif mode == "manual":
        xp = F.pad(x_c, (KW - 1, 0))  # (B, D, T+kW-1)
        z = (weight[0].view(1, -1, 1) * xp[:, :, KW - 0:t_len + KW - 0]
             + weight[1].view(1, -1, 1) * xp[:, :, KW - 1:t_len + KW - 1]
             + weight[2].view(1, -1, 1) * xp[:, :, KW - 2:t_len + KW - 2]
             + weight[3].view(1, -1, 1) * xp[:, :, KW - 3:t_len + KW - 3]
             + bias.view(1, -1, 1))
    else:
        raise ValueError(f"unknown a-mode: {mode}")
    y = F.silu(z)
    return y


def a_build_loss(x, weight, bias, grad_y, mode):
    """A 路线建图（任务协议伪代码）：返回 (y, loss)，上游梯度经 loss 注入。"""
    y = a_forward(x, weight, bias, mode)
    loss = (y * grad_y.transpose(1, 2)).sum()
    return y, loss


def probe_a_mode():
    """探测 F.conv1d 在 torch_npu bf16/groups=12288 下是否可用（小 shape 前向+反向）。"""
    try:
        x = torch.randn(1, 16, DIM, dtype=DTYPE, device=DEVICE)
        w = torch.randn(KW, DIM, dtype=DTYPE, device=DEVICE)
        b = torch.randn(DIM, dtype=DTYPE, device=DEVICE)
        gy = torch.randn(1, 16, DIM, dtype=DTYPE, device=DEVICE)
        xl, wl, bl = (t.detach().clone().requires_grad_(True) for t in (x, w, b))
        _, loss = a_build_loss(xl, wl, bl, gy, "conv1d")
        torch.autograd.grad(loss, (xl, wl, bl))
        torch.npu.synchronize()
        return "conv1d"
    except Exception as e:  # noqa: BLE001
        print(f"[probe] F.conv1d 路线不可用（{type(e).__name__}: {str(e)[:120]}），"
              f"退手动移位拼接")
        return "manual"


# =========================================================================
# 输入与计时
# =========================================================================
def make_inputs(B, T, seed=1234):
    """kimik3 场景输入：3D 固定 batch、等长序列、conv_states 冷启动全零。

    A/B 同 seed 同分布；B 路线额外需要 conv_states / query_start_loc。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(B, T, DIM, generator=gen, dtype=torch.float32)).to(DTYPE).to(DEVICE)
    weight = (torch.randn(KW, DIM, generator=gen, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
    bias = (torch.randn(DIM, generator=gen, dtype=torch.float32) * 0.01).to(DTYPE).to(DEVICE)
    grad_y = (torch.randn(B, T, DIM, generator=gen, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
    conv_states = torch.zeros(B, KW - 1, DIM, dtype=DTYPE, device=DEVICE)
    query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=DEVICE) * T
    return x, weight, bias, grad_y, conv_states, query_start_loc


def timed_loop(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


# =========================================================================
# 正确性验证（A/B vs golden）
# =========================================================================
def compare_tensors(npu_out: np.ndarray, golden_out: np.ndarray, name: str):
    """与 ST（test_minimal.py）/ test_autograd_bf16_correctness.py 口径一致。"""
    assert npu_out.shape == golden_out.shape, \
        f"{name} shape mismatch: {npu_out.shape} vs {golden_out.shape}"
    abs_err = np.abs(npu_out.astype(np.float32) - golden_out.astype(np.float32))
    rel_err = abs_err / (np.abs(golden_out.astype(np.float32)) + 1e-8)
    match = (abs_err <= ATOL) | (rel_err <= RTOL)
    matched_ratio = float(np.mean(match))
    print(f"    [{name}] matched_ratio={matched_ratio:.4f} (threshold>={THRESHOLD}) "
          f"abs_err max={abs_err.max():.3e} mean={abs_err.mean():.3e}")
    return matched_ratio


def verify_one(B, T, a_mode):
    """单 shape 正确性：A/B 三路梯度 vs golden。返回两路线各自的 {dx,dw,db} 比率。"""
    x, weight, bias, grad_y, conv_states, qsl = make_inputs(B, T, seed=42)

    dx_ref, dw_ref, db_ref = causal_conv1d_grad_golden(
        grad_y.float().cpu().numpy(), x.float().cpu().numpy(),
        weight.float().cpu().numpy(), bias.float().cpu().numpy(),
    )

    # ---- A 路线：conv1d/manual 拼接 autograd ----
    print(f"  [A 路线 mode={a_mode}]")
    xl = x.detach().clone().requires_grad_(True)
    wl = weight.detach().clone().requires_grad_(True)
    bl = bias.detach().clone().requires_grad_(True)
    _, loss = a_build_loss(xl, wl, bl, grad_y, a_mode)
    dx_a, dw_a, db_a = torch.autograd.grad(loss, (xl, wl, bl))
    torch.npu.synchronize()
    assert dx_a.shape == (B, T, DIM) and dw_a.shape == (KW, DIM) and db_a.shape == (DIM,)
    a_ratios = {
        "dx": compare_tensors(dx_a.float().cpu().numpy(), dx_ref, "A dx"),
        "dweight": compare_tensors(dw_a.float().cpu().numpy(), dw_ref, "A dweight"),
        "dbias": compare_tensors(db_a.float().cpu().numpy(), db_ref, "A dbias"),
    }
    del loss, dx_a, dw_a, db_a, xl, wl, bl

    # ---- B 路线：自定义 kernel autograd ----
    print("  [B 路线 causal_conv1d_grad kernel]")
    xl = x.detach().clone().requires_grad_(True)
    wl = weight.detach().clone().requires_grad_(True)
    bl = bias.detach().clone().requires_grad_(True)
    y = causal_conv1d_fn_autograd(xl, wl, bl, conv_states, qsl, activation=ACTIVATION)
    dx_b, dw_b, db_b = torch.autograd.grad(y, (xl, wl, bl), grad_outputs=grad_y)
    torch.npu.synchronize()
    b_ratios = {
        "dx": compare_tensors(dx_b.float().cpu().numpy(), dx_ref, "B dx"),
        "dweight": compare_tensors(dw_b.float().cpu().numpy(), dw_ref, "B dweight"),
        "dbias": compare_tensors(db_b.float().cpu().numpy(), db_ref, "B dbias"),
    }
    del y, dx_b, dw_b, db_b, xl, wl, bl
    torch.npu.empty_cache()
    return a_ratios, b_ratios


# =========================================================================
# 性能测试
# =========================================================================
def bench_a(x, weight, bias, grad_y, a_mode, warmup, iters):
    def fwd_only():
        with torch.no_grad():
            a_forward(x, weight, bias, a_mode)

    fwd_us = timed_loop(fwd_only, warmup, iters)

    xl = x.detach().clone().requires_grad_(True)
    wl = weight.detach().clone().requires_grad_(True)
    bl = bias.detach().clone().requires_grad_(True)
    _, loss = a_build_loss(xl, wl, bl, grad_y, a_mode)

    def bwd_only():
        torch.autograd.grad(loss, (xl, wl, bl), retain_graph=True)

    bwd_us = timed_loop(bwd_only, warmup, iters)
    del loss, xl, wl, bl

    def fwd_bwd():
        xl2 = x.detach().requires_grad_(True)
        wl2 = weight.detach().requires_grad_(True)
        bl2 = bias.detach().requires_grad_(True)
        _, loss2 = a_build_loss(xl2, wl2, bl2, grad_y, a_mode)
        torch.autograd.grad(loss2, (xl2, wl2, bl2))

    total_us = timed_loop(fwd_bwd, warmup, iters)
    return {"fwd_us": fwd_us, "bwd_us": bwd_us, "total_us": total_us}


def bench_b(x, weight, bias, grad_y, conv_states, qsl, warmup, iters):
    def fwd_only():
        with torch.no_grad():
            causal_conv1d_fn_autograd(x, weight, bias, conv_states, qsl,
                                      activation=ACTIVATION)

    fwd_us = timed_loop(fwd_only, warmup, iters)

    xl = x.detach().clone().requires_grad_(True)
    wl = weight.detach().clone().requires_grad_(True)
    bl = bias.detach().clone().requires_grad_(True)
    y = causal_conv1d_fn_autograd(xl, wl, bl, conv_states, qsl, activation=ACTIVATION)

    def bwd_only():
        torch.autograd.grad(y, (xl, wl, bl), grad_outputs=grad_y, retain_graph=True)

    bwd_us = timed_loop(bwd_only, warmup, iters)
    del y, xl, wl, bl

    def fwd_bwd():
        xl2 = x.detach().requires_grad_(True)
        wl2 = weight.detach().clone().requires_grad_(True)
        bl2 = bias.detach().clone().requires_grad_(True)
        yy = causal_conv1d_fn_autograd(xl2, wl2, bl2, conv_states, qsl,
                                       activation=ACTIVATION)
        torch.autograd.grad(yy, (xl2, wl2, bl2), grad_outputs=grad_y)

    total_us = timed_loop(fwd_bwd, warmup, iters)
    return {"fwd_us": fwd_us, "bwd_us": bwd_us, "total_us": total_us}


def bench_one_shape(B, T, a_mode, warmup, iters):
    """单 shape：同输入采集 A/B 三组计时。异常时如实记录 error，不中断整体。"""
    out = {"batch": B, "T": T}
    x, weight, bias, grad_y, conv_states, qsl = make_inputs(B, T)

    for route, bench in (("A", None), ("B", None)):
        try:
            if route == "A":
                res = bench_a(x, weight, bias, grad_y, a_mode, warmup, iters)
            else:
                res = bench_b(x, weight, bias, grad_y, conv_states, qsl, warmup, iters)
            res["ratio_bwd_fwd"] = res["bwd_us"] / res["fwd_us"]
            res["status"] = "ok"
            out[route] = res
        except Exception as e:  # noqa: BLE001
            print(f"  [route {route}] ERROR: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc(limit=3)
            out[route] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            torch.npu.empty_cache()

    if out.get("A", {}).get("status") == "ok" and out.get("B", {}).get("status") == "ok":
        out["speedup_bwd_B_vs_A"] = out["A"]["bwd_us"] / out["B"]["bwd_us"]  # <1 自定义更慢
        out["speedup_total_B_vs_A"] = out["A"]["total_us"] / out["B"]["total_us"]

    del x, weight, bias, grad_y, conv_states, qsl
    torch.npu.empty_cache()
    return out


def summarize(results):
    """加速比汇总：全局 geomean/mean/min/max + 按 T 分桶。"""
    def _stats(vals):
        if not vals:
            return None
        n = len(vals)
        geomean = math.exp(sum(math.log(v) for v in vals) / n)
        return {"count": n, "geomean": geomean, "mean": sum(vals) / n,
                "min": min(vals), "max": max(vals)}

    ok = [r for r in results
          if r.get("A", {}).get("status") == "ok" and r.get("B", {}).get("status") == "ok"]
    summary = {
        "shapes_total": len(results),
        "shapes_ok": len(ok),
        "shapes_error": len(results) - len(ok),
        "speedup_bwd": _stats([r["speedup_bwd_B_vs_A"] for r in ok]),
        "speedup_total": _stats([r["speedup_total_B_vs_A"] for r in ok]),
        "speedup_bwd_by_T": {},
        "a_ratio_bwd_fwd": _stats([r["A"]["ratio_bwd_fwd"] for r in ok]),
        "b_ratio_bwd_fwd": _stats([r["B"]["ratio_bwd_fwd"] for r in ok]),
    }
    for t in sorted({r["T"] for r in ok}):
        summary["speedup_bwd_by_T"][str(t)] = _stats(
            [r["speedup_bwd_B_vs_A"] for r in ok if r["T"] == t])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="仅跑正确性验证")
    parser.add_argument("--quick", action="store_true", help="缩减矩阵（B=1，T in 512/2048/8192）")
    args = parser.parse_args()

    torch.npu.set_device(0)
    print("=" * 100)
    print("causal_conv1d 反向性能对比：小算子拼接(A) vs 自定义 kernel(B)  [bf16, NPU Ascend950PR]")
    print("=" * 100)
    print(f"DIM={DIM}, kW={KW}, dtype={DTYPE}, activation={ACTIVATION}, device={DEVICE}")
    print(f"warmup={WARMUP_ITERS}, iters={PROFILE_ITERS}, "
          f"rtol={RTOL}, atol={ATOL}, threshold={THRESHOLD}")
    print(f"ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH')}")

    a_mode = probe_a_mode()
    print(f"[A 路线实现模式] {a_mode}")

    # ---- 正确性验证（对比有意义的前提） ----
    verify_shapes = [(2, 16), (1, 512), (1, 8192)]
    correctness = {"a_mode": a_mode, "shapes": []}
    print("\n---- 正确性验证：A/B vs CPU golden (bf16 混合容差) ----")
    for (B, T) in verify_shapes:
        print(f"  == shape B={B}, T={T} ==")
        a_ratios, b_ratios = verify_one(B, T, a_mode)
        a_pass = all(v >= THRESHOLD for v in a_ratios.values())
        b_pass = all(v >= THRESHOLD for v in b_ratios.values())
        correctness["shapes"].append({
            "B": B, "T": T,
            "A": {**a_ratios, "pass": a_pass},
            "B": {**b_ratios, "pass": b_pass},
        })
        print(f"  => A pass={a_pass}, B pass={b_pass}")
    correctness["a_all_pass"] = all(s["A"]["pass"] for s in correctness["shapes"])
    correctness["b_all_pass"] = all(s["B"]["pass"] for s in correctness["shapes"])
    print(f"正确性汇总：A={'PASS' if correctness['a_all_pass'] else 'FAIL'}, "
          f"B={'PASS' if correctness['b_all_pass'] else 'FAIL'}")

    if args.verify_only:
        out_json = _THIS_DIR / "bwd_compare_results.json"
        out_json.write_text(json.dumps({"meta": {}, "correctness": correctness,
                                        "results": [], "summary": None}, indent=2))
        print(f"(verify-only) correctness saved to {out_json}")
        return 0

    # ---- 性能矩阵 ----
    batches = [1] if args.quick else BATCH_SIZES
    seqlens = [512, 2048, 8192] if args.quick else SEQ_LENS

    print("\n---- 性能矩阵 ----")
    results = []
    for B in batches:
        for T in seqlens:
            t0 = time.time()
            r = bench_one_shape(B, T, a_mode, WARMUP_ITERS, PROFILE_ITERS)
            results.append(r)
            if r.get("A", {}).get("status") == "ok" and r.get("B", {}).get("status") == "ok":
                print(f"[B={B}, T={T:5d}] "
                      f"A: fwd={r['A']['fwd_us']:9.1f} bwd={r['A']['bwd_us']:9.1f} "
                      f"tot={r['A']['total_us']:9.1f} (bwd/fwd={r['A']['ratio_bwd_fwd']:5.2f}) | "
                      f"B: fwd={r['B']['fwd_us']:9.1f} bwd={r['B']['bwd_us']:9.1f} "
                      f"tot={r['B']['total_us']:9.1f} (bwd/fwd={r['B']['ratio_bwd_fwd']:5.2f}) | "
                      f"speedup(bwd B:A)={r['speedup_bwd_B_vs_A']:6.3f} "
                      f"({time.time()-t0:.1f}s)")
            else:
                print(f"[B={B}, T={T:5d}] FAILED: "
                      f"A={r.get('A', {}).get('status')} B={r.get('B', {}).get('status')} "
                      f"({time.time()-t0:.1f}s)")

    summary = summarize(results)

    # ---- 汇总表 ----
    print("\n" + "=" * 118)
    print("性能对比汇总 (bf16, D=12288, kW=4, silu)  时间单位 us；speedup=A_bwd/B_bwd, <1 表示自定义 kernel 更慢")
    print("=" * 118)
    hdr = (f"{'B':<3} {'T':<6} | {'A fwd':>9} {'A bwd':>9} {'A tot':>9} {'A b/f':>6} | "
           f"{'B fwd':>9} {'B bwd':>9} {'B tot':>9} {'B b/f':>6} | {'spd bwd':>8} {'spd tot':>8}")
    print(hdr)
    print("-" * 118)
    for r in results:
        if r.get("A", {}).get("status") == "ok" and r.get("B", {}).get("status") == "ok":
            a, b = r["A"], r["B"]
            print(f"{r['batch']:<3} {r['T']:<6} | {a['fwd_us']:>9.1f} {a['bwd_us']:>9.1f} "
                  f"{a['total_us']:>9.1f} {a['ratio_bwd_fwd']:>6.2f} | "
                  f"{b['fwd_us']:>9.1f} {b['bwd_us']:>9.1f} {b['total_us']:>9.1f} "
                  f"{b['ratio_bwd_fwd']:>6.2f} | {r['speedup_bwd_B_vs_A']:>8.3f} "
                  f"{r['speedup_total_B_vs_A']:>8.3f}")
        else:
            print(f"{r['batch']:<3} {r['T']:<6} | ERROR: A={r.get('A', {}).get('status')} "
                  f"B={r.get('B', {}).get('status')}")
    print("=" * 118)

    sb = summary.get("speedup_bwd") or {}
    print(f"speedup(bwd B:A) geomean={sb.get('geomean', float('nan')):.3f} "
          f"mean={sb.get('mean', float('nan')):.3f} min={sb.get('min', float('nan')):.3f} "
          f"max={sb.get('max', float('nan')):.3f} over {sb.get('count', 0)} shapes")
    for t, st in (summary.get("speedup_bwd_by_T") or {}).items():
        print(f"  T={t:>5}: geomean={st['geomean']:.3f} mean={st['mean']:.3f} "
              f"min={st['min']:.3f} max={st['max']:.3f} ({st['count']} shapes)")

    # ---- JSON 落盘 ----
    payload = {
        "meta": {
            "title": "causal_conv1d bwd compare: composite(A) vs custom kernel(B)",
            "soc": "Ascend950PR", "dtype": "bfloat16", "dim": DIM, "kW": KW,
            "activation": ACTIVATION, "a_mode": a_mode,
            "warmup_iters": WARMUP_ITERS, "profile_iters": PROFILE_ITERS,
            "timing": "torch_npu.npu.Event (device time)",
            "note_a_layout": "A forward output is channel-first (B,D,T), "
                             "transpose-back to (B,T,D) NOT included (favors A)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "correctness": correctness,
        "results": results,
        "summary": summary,
    }
    out_json = _THIS_DIR / "bwd_compare_results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nresults saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
