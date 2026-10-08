#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""kimik3 场景正反向串联性能测试（NPU, bf16, autograd 绑定版）。

场景：causal depthwise conv1d (B, T, 12288), kW=4, groups=12288, activation=silu
前向：causal_conv1d_fn（OpBuilder/aclnnCausalConv1dFn，PrivateUse1）
反向：causal_conv1d_grad（自定义算子包 aclnnCausalConv1dGrad，经
      CausalConv1dFnAutoGrad 绑定为 autograd 反向）

计时口径（torch_npu.npu.Event，device 时间）：
  - fwd_only : torch.no_grad() 下前向
  - bwd_only : 预建图后 torch.autograd.grad(y, (x,w,b), grad_outputs=gy, retain_graph=True) 循环
  - total    : 每 iter 前向 + torch.autograd.grad（不 retain，完整正反向串联）

用法：
  ASCEND_CUSTOM_OPP_PATH=<...>/custom_opp/vendors/causal_conv1d_grad_custom \
  python3 profile_kimik3_bf16_autograd.py [--quick]
"""
import argparse
import json
import os
import sys
import time
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402

from torch_extension.causal_conv1d_grad_ext import (  # noqa: E402
    causal_conv1d_fn_autograd,
)

# ---- kimik3 场景参数 ----
BATCH_SIZES = [1, 2, 4]
SEQ_LENS = [512, 1024, 2048, 4096, 8192]
DIM = 12288
KW = 4
DTYPE = torch.bfloat16
ACTIVATION = "silu"
DEVICE = "npu:0"

WARMUP_ITERS = 10
PROFILE_ITERS = 50


def make_inputs(B, T):
    """kimik3 场景输入：3D 固定 batch、等长序列、conv_states 冷启动全零。"""
    x = torch.randn(B, T, DIM, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(KW, DIM, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(DIM, dtype=DTYPE, device=DEVICE)
    conv_states = torch.zeros(B, KW - 1, DIM, dtype=DTYPE, device=DEVICE)
    query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=DEVICE) * T
    return x, weight, bias, conv_states, query_start_loc


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


def bench_one(B, T, warmup, iters):
    x, weight, bias, conv_states, qsl = make_inputs(B, T)
    gy = torch.randn(B, T, DIM, dtype=DTYPE, device=DEVICE)

    # 1) 前向 only（no_grad）
    def fwd_only():
        with torch.no_grad():
            y = causal_conv1d_fn_autograd(x, weight, bias, conv_states, qsl,
                                          activation=ACTIVATION)
        return y

    fwd_us = timed_loop(fwd_only, warmup, iters)

    # 2) 反向 only（预建图 + retain_graph 复用）
    x_leaf = x.detach().clone().requires_grad_(True)
    w_leaf = weight.detach().clone().requires_grad_(True)
    b_leaf = bias.detach().clone().requires_grad_(True)
    y = causal_conv1d_fn_autograd(x_leaf, w_leaf, b_leaf, conv_states, qsl,
                                  activation=ACTIVATION)

    def bwd_only():
        torch.autograd.grad(y, (x_leaf, w_leaf, b_leaf), grad_outputs=gy,
                            retain_graph=True)

    bwd_us = timed_loop(bwd_only, warmup, iters)
    del y, x_leaf, w_leaf, b_leaf

    # 3) 前向 + 反向 total（每 iter 完整串联，图随 iter 释放）
    def fwd_bwd():
        xl = x.detach().requires_grad_(True)
        wl = weight.detach().requires_grad_(True)
        bl = bias.detach().requires_grad_(True)
        yy = causal_conv1d_fn_autograd(xl, wl, bl, conv_states, qsl,
                                       activation=ACTIVATION)
        torch.autograd.grad(yy, (xl, wl, bl), grad_outputs=gy)

    total_us = timed_loop(fwd_bwd, warmup, iters)

    # 指标
    flop_fwd = 2 * B * T * DIM * KW
    tflops_fwd = flop_fwd / (fwd_us * 1e-6) / 1e12
    return {
        "B": B, "T": T, "fwd_us": fwd_us, "bwd_us": bwd_us, "total_us": total_us,
        "ratio_bwd_fwd": bwd_us / fwd_us,
        "tflops_fwd": tflops_fwd,
        "toks_per_s_fwd_bwd": B * T / (total_us * 1e-6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="仅跑 B=1 的缩减矩阵")
    args = parser.parse_args()

    torch.npu.set_device(0)
    print("=" * 96)
    print("kimik3 场景正反向串联性能测试 (autograd 绑定, bf16, NPU)")
    print("=" * 96)
    print(f"DIM={DIM}, kW={KW}, dtype={DTYPE}, activation={ACTIVATION}, device={DEVICE}")
    print(f"warmup={WARMUP_ITERS}, iters={PROFILE_ITERS}")
    print(f"ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH')}")

    batches = [1] if args.quick else BATCH_SIZES
    seqlens = [512, 2048, 8192] if args.quick else SEQ_LENS

    results = []
    for B in batches:
        for T in seqlens:
            t0 = time.time()
            r = bench_one(B, T, WARMUP_ITERS, PROFILE_ITERS)
            results.append(r)
            print(f"[B={B}, T={T:5d}] fwd={r['fwd_us']:10.2f}us  bwd={r['bwd_us']:10.2f}us  "
                  f"total={r['total_us']:10.2f}us  bwd/fwd={r['ratio_bwd_fwd']:6.2f}  "
                  f"({time.time()-t0:.1f}s)")

    # 汇总表
    print("\n" + "=" * 100)
    print("性能汇总 (bf16, D=12288, kW=4, silu)")
    print("=" * 100)
    hdr = (f"{'B':<3} {'T':<6} {'Fwd(us)':>10} {'Bwd(us)':>10} {'Total(us)':>11} "
           f"{'Bwd/Fwd':>8} {'Fwd TFLOPS':>11} {'Ktok/s(f+b)':>12}")
    print(hdr)
    print("-" * 100)
    for r in results:
        print(f"{r['B']:<3} {r['T']:<6} {r['fwd_us']:>10.2f} {r['bwd_us']:>10.2f} "
              f"{r['total_us']:>11.2f} {r['ratio_bwd_fwd']:>8.2f} {r['tflops_fwd']:>11.3f} "
              f"{r['toks_per_s_fwd_bwd']/1000:>12.2f}")
    print("=" * 100)

    out_json = _THIS_DIR / "kimik3_perf_bf16_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"results saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
