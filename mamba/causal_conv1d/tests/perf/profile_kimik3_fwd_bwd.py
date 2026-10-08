#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
kimik3 场景正向+反向联合性能测试
场景：causal depthwise conv1d (B, T, 12288), kW=4, groups=12288, activation=silu
正向：causal_conv1d_fn
反向：causal_conv1d_grad（待实现算子）

用法：
  # 直接运行（获取端到端耗时）
  python3 tests/perf/profile_kimik3_fwd_bwd.py

  # msprof 采集（深度分析）
  msprof --output=./msprof_kimik3_fwd_bwd --aic-metrics=ArithmeticUtilization \
         --application="python3 tests/perf/profile_kimik3_fwd_bwd.py"
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
    """构造 kimik3 场景输入（requires_grad=True）"""
    x = torch.randn(B, T, D, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(kW, D, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    conv_states = torch.zeros(B, kW - 1, D, dtype=dtype, device=device)
    query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=device) * T
    cache_indices = torch.arange(B, dtype=torch.int32, device=device)
    has_initial_state = torch.zeros(B, dtype=torch.int32, device=device)
    return x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state


def benchmark_forward_only(B, T):
    """仅前向性能测试"""
    print(f"\n{'='*70}")
    print(f"[前向] B={B}, T={T}, D={DIM}, kW={KW}, dtype={DTYPE}, act={ACTIVATION}")
    print(f"{'='*70}")

    x, weight, bias, conv_states, qsl, cache_idx, has_init = make_inputs(
        B, T, DIM, KW, DTYPE, DEVICE
    )

    # Warmup
    print(f"Warmup: {WARMUP_ITERS} iters...")
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
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
        with torch.no_grad():
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

    print(f"前向平均耗时: {avg_time_us:.2f} us ({elapsed_ms / PROFILE_ITERS:.4f} ms)")

    # 计算 FLOP（前向卷积 + SiLU）
    flop_fwd = B * T * DIM * (KW * 2 + 8)
    tflops_fwd = (flop_fwd / (avg_time_us * 1e-6)) / 1e12
    print(f"前向 TFLOPS: {tflops_fwd:.2f}")

    return avg_time_us


def benchmark_forward_backward(B, T):
    """前向+反向联合性能测试"""
    print(f"\n{'='*70}")
    print(f"[前向+反向] B={B}, T={T}, D={DIM}, kW={KW}, dtype={DTYPE}, act={ACTIVATION}")
    print(f"{'='*70}")

    # Warmup
    print(f"Warmup: {WARMUP_ITERS} iters...")
    for _ in range(WARMUP_ITERS):
        x, weight, bias, conv_states, qsl, cache_idx, has_init = make_inputs(
            B, T, DIM, KW, DTYPE, DEVICE
        )
        y = causal_conv1d_fn(
            x, weight, bias, conv_states, qsl,
            cache_indices=cache_idx,
            has_initial_state=has_init,
            activation=ACTIVATION
        )
        loss = y.sum()
        loss.backward()
        torch_npu.npu.synchronize()

    # Profile
    print(f"Profiling: {PROFILE_ITERS} iters...")
    torch_npu.npu.synchronize()
    start_event = torch_npu.npu.Event(enable_timing=True)
    end_event = torch_npu.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(PROFILE_ITERS):
        x, weight, bias, conv_states, qsl, cache_idx, has_init = make_inputs(
            B, T, DIM, KW, DTYPE, DEVICE
        )
        y = causal_conv1d_fn(
            x, weight, bias, conv_states, qsl,
            cache_indices=cache_idx,
            has_initial_state=has_init,
            activation=ACTIVATION
        )
        loss = y.sum()
        loss.backward()
    end_event.record()
    torch_npu.npu.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_us = (elapsed_ms * 1000) / PROFILE_ITERS

    print(f"前向+反向平均耗时: {avg_time_us:.2f} us ({elapsed_ms / PROFILE_ITERS:.4f} ms)")

    # 粗估反向 FLOP（约为前向 2-3 倍）
    flop_fwd = B * T * DIM * (KW * 2 + 8)
    flop_bwd = flop_fwd * 2.5  # 反向：重算 z + silu' + grad_z + dx/dweight/dbias
    flop_total = flop_fwd + flop_bwd
    tflops_total = (flop_total / (avg_time_us * 1e-6)) / 1e12
    print(f"前向+反向 TFLOPS: {tflops_total:.2f}")

    return avg_time_us


def benchmark_backward_only(B, T, fwd_time_us, fwd_bwd_time_us):
    """推导反向耗时（从联合测试中减去前向）"""
    bwd_time_us = fwd_bwd_time_us - fwd_time_us
    print(f"\n{'='*70}")
    print(f"[反向推导] B={B}, T={T}")
    print(f"{'='*70}")
    print(f"前向耗时: {fwd_time_us:.2f} us")
    print(f"前向+反向耗时: {fwd_bwd_time_us:.2f} us")
    print(f"反向推导耗时: {bwd_time_us:.2f} us")

    # 反向 FLOP
    flop_bwd = B * T * DIM * (KW * 2 + 8) * 2.5
    tflops_bwd = (flop_bwd / (bwd_time_us * 1e-6)) / 1e12
    print(f"反向 TFLOPS: {tflops_bwd:.2f}")

    return bwd_time_us


def main():
    torch_npu.npu.set_device(0)
    print("=" * 80)
    print("kimik3 场景正向+反向联合性能测试")
    print("=" * 80)
    print(f"DIM={DIM}, kW={KW}, dtype={DTYPE}, activation={ACTIVATION}")
    print(f"Warmup={WARMUP_ITERS}, Profile={PROFILE_ITERS}")

    results = []
    for B in BATCH_SIZES:
        for T in SEQ_LENS:
            try:
                # 1. 仅前向
                fwd_time = benchmark_forward_only(B, T)

                # 2. 前向+反向
                fwd_bwd_time = benchmark_forward_backward(B, T)

                # 3. 推导反向
                bwd_time = benchmark_backward_only(B, T, fwd_time, fwd_bwd_time)

                results.append((B, T, fwd_time, bwd_time, fwd_bwd_time))
            except Exception as e:
                print(f"\n[ERROR] B={B}, T={T} failed: {e}")
                import traceback
                traceback.print_exc()

    # 汇总报告
    print(f"\n{'='*100}")
    print("性能汇总")
    print(f"{'='*100}")
    print(f"{'Batch':<8} {'SeqLen':<8} {'Fwd(us)':<12} {'Bwd(us)':<12} {'Fwd+Bwd(us)':<15} {'Throughput(Ktok/s)':<20}")
    print(f"{'-'*100}")
    for B, T, fwd_time, bwd_time, fwd_bwd_time in results:
        # Throughput 基于前向+反向
        throughput = (B * T) / (fwd_bwd_time * 1e-6) / 1000  # Ktokens/s
        print(f"{B:<8} {T:<8} {fwd_time:<12.2f} {bwd_time:<12.2f} {fwd_bwd_time:<15.2f} {throughput:<20.2f}")
    print(f"{'='*100}")

    # 额外：反向/前向比率分析
    print(f"\n{'='*100}")
    print("反向/前向比率分析")
    print(f"{'='*100}")
    print(f"{'Batch':<8} {'SeqLen':<8} {'Bwd/Fwd 比率':<15} {'说明':<50}")
    print(f"{'-'*100}")
    for B, T, fwd_time, bwd_time, fwd_bwd_time in results:
        ratio = bwd_time / fwd_time if fwd_time > 0 else 0
        comment = ""
        if ratio < 1.5:
            comment = "反向效率高（接近前向）"
        elif ratio < 2.5:
            comment = "反向效率正常（约 2x 前向）"
        else:
            comment = "反向效率偏低（需优化）"
        print(f"{B:<8} {T:<8} {ratio:<15.2f} {comment:<50}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
