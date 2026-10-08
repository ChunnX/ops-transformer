#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
kimik3 场景性能采集脚本
场景：causal depthwise conv1d (B, T, 12288), kW=4, groups=12288, activation=silu
用法：
  # 单次运行（供 msprof 包裹）
  python3 tests/perf/profile_kimik3.py

  # msprof 采集（7 组 aic-metrics）
  msprof --output=./msprof_kimik3 --aic-metrics=ArithmeticUtilization \
         --application="python3 tests/perf/profile_kimik3.py"
"""
import os
import sys
import torch
import torch_npu

# 载入算子
from cann_ops_transformer.ops import causal_conv1d_fn

# kimik3 场景参数
BATCH_SIZES = [1, 2, 4]
SEQ_LENS = [512, 1024, 2048, 4096, 8192]
DIM = 12288
KW = 4
DTYPE = torch.float16
ACTIVATION = "silu"
DEVICE = "npu:0"

# 性能采集参数
WARMUP_ITERS = 10
PROFILE_ITERS = 100


def make_inputs(B, T, D, kW, dtype, device):
    """构造 kimik3 场景输入"""
    x = torch.randn(B, T, D, dtype=dtype, device=device)
    weight = torch.randn(kW, D, dtype=dtype, device=device)
    bias = torch.randn(D, dtype=dtype, device=device)
    conv_states = torch.zeros(B, kW - 1, D, dtype=dtype, device=device)
    query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=device) * T
    cache_indices = torch.arange(B, dtype=torch.int32, device=device)
    has_initial_state = torch.zeros(B, dtype=torch.int32, device=device)
    return x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state


def benchmark_case(B, T):
    """对单个 case 进行性能测试"""
    print(f"\n{'='*60}")
    print(f"Case: B={B}, T={T}, D={DIM}, kW={KW}, dtype={DTYPE}, act={ACTIVATION}")
    print(f"{'='*60}")

    # 准备输入
    x, weight, bias, conv_states, qsl, cache_idx, has_init = make_inputs(
        B, T, DIM, KW, DTYPE, DEVICE
    )

    # Warmup
    print(f"Warmup: {WARMUP_ITERS} iters...")
    for _ in range(WARMUP_ITERS):
        _ = causal_conv1d_fn(
            x, weight, bias, conv_states, qsl,
            cache_indices=cache_idx,
            has_initial_state=has_init,
            activation=ACTIVATION
        )
        torch_npu.npu.synchronize()

    # Profile
    print(f"Profiling: {PROFILE_ITERS} iters...")
    torch_npu.npu.synchronize()
    start_event = torch_npu.npu.Event(enable_timing=True)
    end_event = torch_npu.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(PROFILE_ITERS):
        _ = causal_conv1d_fn(
            x, weight, bias, conv_states, qsl,
            cache_indices=cache_idx,
            has_initial_state=has_init,
            activation=ACTIVATION
        )
    end_event.record()
    torch_npu.npu.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_us = (elapsed_ms * 1000) / PROFILE_ITERS

    print(f"Average time: {avg_time_us:.2f} us ({elapsed_ms / PROFILE_ITERS:.4f} ms)")
    print(f"Throughput: {PROFILE_ITERS / (elapsed_ms / 1000):.2f} ops/s")

    # 计算 FLOP（粗估）
    # 前向卷积：B * T * D * kW * 2（MAC） + B * T * D * 8（SiLU：exp + div + mul * 4）
    flop = B * T * DIM * (KW * 2 + 8)
    tflops = (flop / (avg_time_us * 1e-6)) / 1e12
    print(f"Estimated TFLOPS: {tflops:.2f}")

    return avg_time_us


def main():
    torch_npu.npu.set_device(0)
    print("kimik3 场景性能采集")
    print(f"DIM={DIM}, kW={KW}, dtype={DTYPE}, activation={ACTIVATION}")
    print(f"Warmup={WARMUP_ITERS}, Profile={PROFILE_ITERS}")

    results = []
    for B in BATCH_SIZES:
        for T in SEQ_LENS:
            try:
                avg_time = benchmark_case(B, T)
                results.append((B, T, avg_time))
            except Exception as e:
                print(f"[ERROR] B={B}, T={T} failed: {e}")

    # 汇总报告
    print(f"\n{'='*80}")
    print("性能汇总")
    print(f"{'='*80}")
    print(f"{'Batch':<8} {'SeqLen':<8} {'Time(us)':<12} {'Throughput(Ktok/s)':<20}")
    print(f"{'-'*80}")
    for B, T, avg_time in results:
        throughput = (B * T) / (avg_time * 1e-6) / 1000  # Ktokens/s
        print(f"{B:<8} {T:<8} {avg_time:<12.2f} {throughput:<20.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
