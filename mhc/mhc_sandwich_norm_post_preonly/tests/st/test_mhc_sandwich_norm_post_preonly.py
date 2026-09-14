#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
test_mhc_sandwich_norm_post_preonly.py

Self-contained ST for MhcSandwichNormPostPreonly operator.

Test levels (select via pytest -k):
    L0 eager:  pytest -k 'l0_eager'         24 cases   ~5s    (快速验证)
    L0 graph:  pytest -k 'l0_graph'         8 cases   ~10s   (图模式)
    L0 all:    pytest -k 'l0'               48 cases   ~15s   (默认推荐)
"""
import warnings
import builtins

_original_warn = warnings.warn

def _filtered_warn(message, category=None, stacklevel=1, source=None):
    if isinstance(message, str) and "The following torchair config or properties may not take effect" in message:
        return  
    if category is UserWarning and "reduce-overhead mode" in str(message):
        return
    _original_warn(message, category, stacklevel, source)

warnings.warn = _filtered_warn
builtins.warn = _filtered_warn

import itertools

import torch
import torch_npu
import omni_custom_ops
import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

N = 4
D_ALL = [2560, 5120]
DTYPE_ALL = [torch.bfloat16, torch.float16]
GAMMA2_ALL = [True, False]

EPS = 1e-7
GOLDEN_CPU_DTYPE = torch.float64

# ============================================================================
# Precision config (aligned with mhc ST convention)
# ============================================================================

DEFAULT_PRECISION_LEVEL = "L1"

PRECISION_LEVELS = {
    "L0": {"MARE": 10.0, "MERE": 2.0, "RMSE": 2.0},
    "L1": {"MARE": 5.0, "MERE": 1.5, "RMSE": 1.5},
    "L2": {"MARE": 2.0, "MERE": 1.2, "RMSE": 1.2},
}

SMALL_VALUE_CONFIG = {
    torch.float16: {"threshold": 2**-11, "error": 2**-16},
    torch.bfloat16: {"threshold": 2**-8, "error": 2**-16},
    torch.float32: {"threshold": 2**-14, "error": 2**-30},
}


# ============================================================================
# Precision utilities
# ============================================================================

def _relative_error(actual, golden):
    return torch.abs(actual - golden) / (torch.abs(golden) + EPS)


def _calc_error_metrics(actual, golden):
    """Compute MARE, MERE, RMSE over all elements (caller filters domain)."""
    re = _relative_error(actual, golden)
    mare = re.max().item()
    mere = re.mean().item()
    rmse = torch.sqrt(torch.mean((actual - golden) ** 2)).item()
    return {"MARE": mare, "MERE": mere, "RMSE": rmse}


def _split_normal_and_small_domain(actual, golden, dtype):
    """Split into normal domain (|golden| >= threshold) and small domain."""
    threshold = SMALL_VALUE_CONFIG[dtype]["threshold"]
    abs_golden = torch.abs(golden)
    normal_mask = abs_golden >= threshold
    small_mask = abs_golden < threshold
    return actual[normal_mask], golden[normal_mask], actual[small_mask], golden[small_mask]


def _normal_domain_pass(npu_normal, bench_normal, golden_normal, precision_level=DEFAULT_PRECISION_LEVEL):
    """Normal domain: ratio = npu_metric / (bench_metric + eps) must be within limits."""
    if golden_normal.numel() == 0:
        return True
    npu_metrics = _calc_error_metrics(npu_normal, golden_normal)
    bench_metrics = _calc_error_metrics(bench_normal, golden_normal)
    ratio = {k: npu_metrics[k] / (bench_metrics[k] + EPS) for k in npu_metrics}
    limits = PRECISION_LEVELS[precision_level]
    for k, limit in limits.items():
        if ratio[k] > limit:
            print(f"[NormalDomain] FAIL on {k}: ratio={ratio[k]:.4f} > {limit}")
            print(f"  NPU:   {npu_metrics}")
            print(f"  Bench: {bench_metrics}")
            return False
    return True


def _small_domain_pass(npu_small, bench_small, golden_small, dtype):
    """Small domain: ErrorCount ratio (npu/bench) must be <= 2.0."""
    if golden_small.numel() == 0:
        return True
    err = SMALL_VALUE_CONFIG[dtype]["error"]

    def error_count(actual):
        return int((torch.abs(actual - golden_small) > err).sum().item())

    npu_cnt = error_count(npu_small)
    bench_cnt = error_count(bench_small)
    ratio = npu_cnt / max(bench_cnt, 1)
    if ratio > 2.0:
        print(f"[SmallDomain] FAIL: NPU ErrCnt={npu_cnt}, Bench ErrCnt={bench_cnt}, Ratio={ratio:.3f}")
        return False
    return True


def _precision_check(name, npu_val, bench_val, golden_val, dtype, precision_level=DEFAULT_PRECISION_LEVEL):
    """Full precision check: normal domain + small domain, both must pass."""
    hp = GOLDEN_CPU_DTYPE
    npu_d, bench_d, gold_d = npu_val.to(hp), bench_val.to(hp), golden_val.to(hp)

    npu_normal, gold_normal, npu_small, gold_small = _split_normal_and_small_domain(npu_d, gold_d, dtype)
    bench_normal, _, bench_small, _ = _split_normal_and_small_domain(bench_d, gold_d, dtype)

    normal_ok = _normal_domain_pass(npu_normal, bench_normal, gold_normal, precision_level)
    small_ok = _small_domain_pass(npu_small, bench_small, gold_small, dtype)

    passed = normal_ok and small_ok
    if not passed:
        print(f"[{name}] normal_pass={normal_ok}, small_pass={small_ok}")
    return passed


# ============================================================================
# Reference implementations
# ============================================================================


def _rms_norm(x, gamma, eps):
    return (
        x
        * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        * gamma.unsqueeze(0)
    )


def _golden_cpu_fp64(inputs, norm_eps=1e-6, hc_eps=1e-6):
    hp = torch.float64
    h_out = inputs["h_out"].to(hp)
    residual = inputs["residual"].to(hp)
    h_post, h_res = inputs["h_post"].to(hp), inputs["h_res"].to(hp)
    phi, alpha_0, bias = (
        inputs["phi"].to(hp),
        inputs["alpha_0"].to(hp),
        inputs["bias"].to(hp),
    )
    gamma_0, gamma_1 = inputs["gamma_0"].to(hp), inputs["gamma_1"].to(hp)
    gamma_2 = inputs["gamma_2"].to(hp) if inputs["gamma_2"] is not None else None

    T, D = h_out.shape
    h_normed = _rms_norm(h_out, gamma_0, norm_eps)
    x2 = h_post.unsqueeze(-1) * h_normed.unsqueeze(-2) + torch.matmul(
        h_res.transpose(-1, -2), residual
    )

    if gamma_2 is not None:
        x2f = x2.reshape(T, N * D)
        x2 = (
            x2f
            * torch.rsqrt(torch.mean(x2f * x2f, dim=-1, keepdim=True) + norm_eps)
            * gamma_2.unsqueeze(0)
        ).reshape(T, N, D)

    x2_out = x2.clone()
    x2f = x2.reshape(T, N * D)
    inv_rms = torch.rsqrt(
        torch.sum(x2f * x2f, dim=-1, keepdim=True) / (N * D) + norm_eps
    )
    xhat = torch.matmul(x2f, phi.t())
    gate = (
        1.0 / (1.0 + torch.exp(-(alpha_0 * (xhat * inv_rms) + bias.unsqueeze(0))))
        + hc_eps
    )
    h_in = torch.sum(gate.unsqueeze(-1) * x2, dim=-2)
    h_in_prime = _rms_norm(h_in, gamma_1, norm_eps)
    return h_in_prime, x2_out


def _bench_npu_fp32(inputs, norm_eps=1e-6, hc_eps=1e-6):
    io_dtype = inputs["h_out"].dtype
    h_out = inputs["h_out"].npu().float()
    residual = inputs["residual"].npu().float()
    h_post, h_res = inputs["h_post"].npu(), inputs["h_res"].npu()
    phi, alpha_0, bias = (
        inputs["phi"].npu(),
        inputs["alpha_0"].npu(),
        inputs["bias"].npu(),
    )
    gamma_0, gamma_1 = inputs["gamma_0"].npu(), inputs["gamma_1"].npu()
    gamma_2 = inputs["gamma_2"].npu() if inputs["gamma_2"] is not None else None

    T, D = h_out.shape
    h_normed = (
        h_out
        * torch.rsqrt(torch.mean(h_out * h_out, dim=-1, keepdim=True) + norm_eps)
        * gamma_0.unsqueeze(0)
    )
    h_normed = h_normed.to(io_dtype).float()

    x2 = h_post.unsqueeze(-1) * h_normed.unsqueeze(-2) + torch.matmul(
        h_res.transpose(-1, -2), residual
    )
    x2 = x2.to(io_dtype).float()

    if gamma_2 is not None:
        x2f = x2.reshape(T, N * D)
        x2 = (
            x2f
            * torch.rsqrt(torch.mean(x2f * x2f, dim=-1, keepdim=True) + norm_eps)
            * gamma_2.unsqueeze(0)
        ).reshape(T, N, D)
        x2 = x2.to(io_dtype).float()

    x2_out = x2.clone()
    x2f = x2.reshape(T, N * D)
    inv_rms = torch.rsqrt(
        torch.sum(x2f * x2f, dim=-1, keepdim=True) / (N * D) + norm_eps
    )
    xhat = torch.matmul(x2f, phi.t())
    gate = torch.sigmoid(alpha_0 * (xhat * inv_rms) + bias.unsqueeze(0)) + hc_eps
    h_in = torch.sum(gate.unsqueeze(-1) * x2, dim=-2)
    h_in = h_in.to(io_dtype).float()

    h_in_prime = (
        h_in
        * torch.rsqrt(torch.mean(h_in * h_in, dim=-1, keepdim=True) + norm_eps)
        * gamma_1.unsqueeze(0)
    )
    return h_in_prime.to(io_dtype), x2_out.to(io_dtype)


def _bench_npu_fp32_v2(inputs, norm_eps=1e-6, hc_eps=1e-6):
    io_dtype = inputs["h_out"].dtype
    h_out = inputs["h_out"].npu().float()
    residual = inputs["residual"].npu().float()
    h_post, h_res = inputs["h_post"].npu(), inputs["h_res"].npu()
    phi, alpha_0, bias = (
        inputs["phi"].npu(),
        inputs["alpha_0"].npu(),
        inputs["bias"].npu(),
    )
    gamma_0, gamma_1 = inputs["gamma_0"].npu(), inputs["gamma_1"].npu()
    gamma_2 = inputs["gamma_2"].npu() if inputs["gamma_2"] is not None else None

    T, D = h_out.shape
    h_normed = (
        h_out
        * torch.rsqrt(torch.mean(h_out * h_out, dim=-1, keepdim=True) + norm_eps)
        * gamma_0.unsqueeze(0)
    )
    h_normed = h_normed.to(io_dtype).float()

    x2 = h_post.unsqueeze(-1) * h_normed.unsqueeze(-2) + torch.matmul(
        h_res.transpose(-1, -2), residual
    )
    x2 = x2.to(io_dtype).float()

    if gamma_2 is not None:
        x2f = x2.reshape(T, N * D)
        x2 = (
            x2f
            * torch.rsqrt(torch.mean(x2f * x2f, dim=-1, keepdim=True) + norm_eps)
            * gamma_2.unsqueeze(0)
        ).reshape(T, N, D)
        x2 = x2.to(io_dtype).float()

    x2_out = x2.clone()
    x2f = x2.reshape(T, N * D)
    inv_rms = torch.rsqrt(
        torch.sum(x2f * x2f, dim=-1, keepdim=True) / (N * D) + norm_eps
    )
    xhat = torch.matmul(x2f, phi.t())
    gate = torch.sigmoid(alpha_0 * (xhat * inv_rms) + bias.unsqueeze(0)) + hc_eps
    h_in = torch.sum(gate.unsqueeze(-1) * x2, dim=-2)
    h_in = h_in.to(io_dtype).float()

    h_in_prime = (
        h_in
        * torch.rsqrt(torch.mean(h_in * h_in, dim=-1, keepdim=True) + norm_eps)
        * gamma_1.unsqueeze(0)
    )
    return h_in_prime.to(io_dtype), x2_out.to(io_dtype), h_in_prime

def _npu_custom_op(inputs, norm_eps=1e-6, hc_eps=1e-6):
    return torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(
        inputs["h_out"].npu(),
        inputs["residual"].npu(),
        inputs["h_post"].npu(),
        inputs["h_res"].npu(),
        inputs["phi"].npu(),
        inputs["alpha_0"].npu(),
        inputs["bias"].npu(),
        inputs["gamma_0"].npu(),
        inputs["gamma_1"].npu(),
        inputs["gamma_2"].npu() if inputs["gamma_2"] is not None else None,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )

def _npu_custom_op_v2(inputs, norm_eps=1e-6, hc_eps=1e-6):
    return torch.ops.custom.npu_mhc_sandwich_norm_post_preonly_v2(
        inputs["h_out"].npu(),
        inputs["residual"].npu(),
        inputs["h_post"].npu(),
        inputs["h_res"].npu(),
        inputs["phi"].npu(),
        inputs["alpha_0"].npu(),
        inputs["bias"].npu(),
        inputs["gamma_0"].npu(),
        inputs["gamma_1"].npu(),
        gamma_2=inputs["gamma_2"].npu() if inputs["gamma_2"] is not None else None,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        return_h_in_f32=True,
    )

class _OpModule(torch.nn.Module):
    """Wrap custom op as nn.Module for torch.compile / NPUGraph."""

    def forward(self, h_out, residual, h_post, h_res, phi, alpha_0, bias,
                gamma_0, gamma_1, gamma_2, norm_eps, hc_eps):
        return torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(
            h_out, residual, h_post, h_res, phi, alpha_0, bias,
            gamma_0, gamma_1, gamma_2, norm_eps=norm_eps, hc_eps=hc_eps,
        )


def _to_npu_args(inputs, norm_eps, hc_eps):
    return (
        inputs["h_out"].npu(), inputs["residual"].npu(),
        inputs["h_post"].npu(), inputs["h_res"].npu(),
        inputs["phi"].npu(), inputs["alpha_0"].npu(), inputs["bias"].npu(),
        inputs["gamma_0"].npu(), inputs["gamma_1"].npu(),
        inputs["gamma_2"].npu() if inputs["gamma_2"] is not None else None,
        norm_eps, hc_eps,
    )


compiled = torch.compile(
    _OpModule(), fullgraph=False, backend="npugraph_ex",
    dynamic=False, options={"force_eager": True},
)


def _npu_graph_op(inputs, norm_eps=1e-6, hc_eps=1e-6):
    """Call fused operator via torch.compile with npugraph_ex backend."""
    torch._dynamo.reset()
    return compiled(*_to_npu_args(inputs, norm_eps, hc_eps))


def _npu_aclgraph_eager_op(inputs, norm_eps=1e-6, hc_eps=1e-6):
    """Call fused operator via aclgraph eager (NPUGraph capture + replay)."""
    npu_args = _to_npu_args(inputs, norm_eps, hc_eps)
    torch.npu.synchronize()
    g = torch_npu.npu.NPUGraph()
    with torch.npu.graph(g, auto_dispatch_capture=True):
        res = torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(*npu_args)
    g.replay()
    return res


# ============================================================================
# Input generation
# ============================================================================


def _generate_inputs(B, S, D, dtype, gamma2, seed=42):
    torch.manual_seed(seed)
    T = B * S
    return {
        "h_out": torch.randn(T, D).to(dtype),
        "residual": torch.randn(T, N, D).to(dtype),
        "h_post": torch.randn(T, N),
        "h_res": torch.randn(T, N, N),
        "phi": torch.randn(N, N * D),
        "alpha_0": torch.randn(1),
        "bias": torch.randn(N),
        "gamma_0": torch.randn(D),
        "gamma_1": torch.randn(D),
        "gamma_2": torch.randn(N * D) if gamma2 else None,
    }


def _generate_inputs_uniform(B, S, D, dtype, gamma2, seed=42):
    """ATK-style uniform distribution with 3sigma ranges."""
    gen = torch.Generator().manual_seed(seed)
    T = B * S

    def u(shape, lo, hi, dt):
        return (torch.rand(shape, generator=gen) * (hi - lo) + lo).to(dt)

    return {
        "h_out": u((T, D), -1, 1, dtype),
        "residual": u((T, N, D), -1, 1, dtype),
        "h_post": u((T, N), -1, 1, torch.float32),
        "h_res": u((T, N, N), -1, 1, torch.float32),
        "phi": u((N, N * D), -0.24, 0.24, torch.float32),
        "alpha_0": u((1,), -1.28, 1.84, torch.float32),
        "bias": u((N,), -1, 1, torch.float32),
        "gamma_0": u((D,), -0.11, 0.85, torch.float32),
        "gamma_1": u((D,), 0.45, 1.35, torch.float32),
        "gamma_2": u((N * D,), -0.10, 0.26, torch.float32) if gamma2 else None,
    }


# ============================================================================
# Verification helpers
# ============================================================================


def _run_and_verify(B, S, D, dtype, gamma2, norm_eps=1e-6, hc_eps=1e-6):
    inputs = _generate_inputs(B, S, D, dtype, gamma2)
    golden_h, golden_x = _golden_cpu_fp64(inputs, norm_eps, hc_eps)
    bench_h, bench_x = _bench_npu_fp32(inputs, norm_eps, hc_eps)
    npu_h, npu_x = _npu_custom_op(inputs, norm_eps, hc_eps)

    tag = f"B{B}S{S}_D{D}_{'bf16' if dtype == torch.bfloat16 else 'fp16'}_{'g2' if gamma2 else 'nog2'}"
    ok_h = _precision_check(f"h_in_prime {tag}", npu_h.cpu().float(), bench_h.cpu().float(), golden_h, dtype)
    ok_x = _precision_check(f"x_2_out {tag}", npu_x.cpu().float(), bench_x.cpu().float(), golden_x, dtype)
    assert ok_h, f"h_in_prime precision FAILED for {tag}"
    assert ok_x, f"x_2_out precision FAILED for {tag}"


def _run_and_verify_v2(B, S, D, dtype, gamma2, norm_eps=1e-6, hc_eps=1e-6):
    inputs = _generate_inputs(B, S, D, dtype, gamma2)
    golden_h, golden_x = _golden_cpu_fp64(inputs, norm_eps, hc_eps)
    bench_h, bench_x, bench_h_f32 = _bench_npu_fp32_v2(inputs, norm_eps, hc_eps)
    npu_h, npu_x, npu_f32 = _npu_custom_op_v2(inputs, norm_eps, hc_eps)

    tag = f"B{B}S{S}_D{D}_{'bf16' if dtype == torch.bfloat16 else 'fp16'}_{'g2' if gamma2 else 'nog2'}"
    ok_h = _precision_check(f"h_in_prime {tag}", npu_h.cpu().float(), bench_h.cpu().float(), golden_h, dtype)
    ok_x = _precision_check(f"x_2_out {tag}", npu_x.cpu().float(), bench_x.cpu().float(), golden_x, dtype)
    ok_h_f32 = _precision_check(f"h_in_f32 {tag}", npu_f32.cpu().float(), bench_h_f32.cpu().float(), golden_h, dtype)
    assert ok_h, f"h_in_prime precision FAILED for {tag}"
    assert ok_x, f"x_2_out precision FAILED for {tag}"
    assert ok_h_f32, f"h_in_f32 precision FAILED for {tag}"


def _run_and_verify_graph(B, S, D, dtype, gamma2, norm_eps=1e-6, hc_eps=1e-6):
    inputs = _generate_inputs(B, S, D, dtype, gamma2)
    golden_h, golden_x = _golden_cpu_fp64(inputs, norm_eps, hc_eps)
    bench_h, bench_x = _bench_npu_fp32(inputs, norm_eps, hc_eps)
    graph_h, graph_x = _npu_graph_op(inputs, norm_eps, hc_eps)

    tag = f"graph_B{B}S{S}_D{D}_{'bf16' if dtype == torch.bfloat16 else 'fp16'}_{'g2' if gamma2 else 'nog2'}"
    ok_h = _precision_check(f"h_in_prime {tag}", graph_h.cpu().float(), bench_h.cpu().float(), golden_h, dtype)
    ok_x = _precision_check(f"x_2_out {tag}", graph_x.cpu().float(), bench_x.cpu().float(), golden_x, dtype)
    assert ok_h, f"h_in_prime precision FAILED for {tag}"
    assert ok_x, f"x_2_out precision FAILED for {tag}"


# ============================================================================
# Case generation
# ============================================================================


def _dtype_tag(dt):
    return "bf16" if dt == torch.bfloat16 else "fp16"


# L0: T=4, 3 factor pairs x 2 D x 2 dtype x 2 gamma2 = 24
L0_PARAMS = [
    (B, S, D, dt, g2)
    for B, S in [(1, 4), (2, 2), (4, 1)]
    for D in D_ALL
    for dt in DTYPE_ALL
    for g2 in GAMMA2_ALL
]
L0_IDS = [
    f"{_dtype_tag(dt)}_B{B}S{S}_D{D}_{'g2' if g2 else 'nog2'}"
    for B, S, D, dt, g2 in L0_PARAMS
]

# L0_GRAPH: T=4, 1 factor pair x 2 D x 2 dtype x 2 gamma2 = 8 (仅通路测试)
L0_GRAPH_PARAMS = [
    (B, S, D, dt, g2)
    for B, S in [(1, 4)]
    for D in D_ALL
    for dt in DTYPE_ALL
    for g2 in GAMMA2_ALL
]
L0_GRAPH_IDS = [
    f"{_dtype_tag(dt)}_B{B}S{S}_D{D}_{'g2' if g2 else 'nog2'}"
    for B, S, D, dt, g2 in L0_GRAPH_PARAMS
]


# ============================================================================
# Tests
# ============================================================================
# --- L0 eager (24 cases) ---
@pytest.mark.parametrize("B, S, D, dtype, gamma2", L0_PARAMS, ids=L0_IDS)
@pytest.mark.resources(device="npu:*", npus_per_node=1)
def test_l0_eager(B, S, D, dtype, gamma2):
    _run_and_verify(B, S, D, dtype, gamma2)


# --- L0 graph (8 cases) ---
@pytest.mark.parametrize("B, S, D, dtype, gamma2", L0_GRAPH_PARAMS, ids=L0_GRAPH_IDS)
@pytest.mark.resources(device="npu:*", npus_per_node=1, cann_min_ver="9.1.0")
def test_l0_graph(B, S, D, dtype, gamma2):
    _run_and_verify_graph(B, S, D, dtype, gamma2)

@pytest.mark.resources(device="npu:*", npus_per_node=1)
def test_v2_B24_S4_D2560_bf16_noGamma2():
    _run_and_verify_v2(24, 4, 2560, torch.bfloat16, False)

@pytest.mark.resources(device="npu:*", npus_per_node=1)
def test_v2_B20_S4_D5120_fp16_Gamma2():
    _run_and_verify_v2(20, 4, 5120, torch.float16, True)


# ============================================================================
# aclgraph静态kernel编译模式测试
# ============================================================================

class _StaticKernelModule(torch.nn.Module):
    """静态kernel编译模式需要在forward上加@torch.inference_mode()"""

    @torch.inference_mode()
    def forward(self, h_out, residual, h_post, h_res, phi, alpha_0, bias,
                gamma_0, gamma_1, gamma_2, norm_eps, hc_eps):
        return torch.ops.custom.npu_mhc_sandwich_norm_post_preonly(
            h_out, residual, h_post, h_res, phi, alpha_0, bias,
            gamma_0, gamma_1, gamma_2, norm_eps=norm_eps, hc_eps=hc_eps,
        )


@pytest.mark.resources(device="npu:*", npus_per_node=1, cann_min_ver="9.1.0")
def test_aclgraph_static_kernel_bf16_B1S4_D2560_g2():
    """BF16 aclgraph静态kernel编译模式: B=1, S=4, D=2560, gamma2=True"""
    B, S, D = 1, 4, 2560
    dtype = torch.bfloat16
    gamma2 = True
    norm_eps = 1e-6
    hc_eps = 1e-6

    # 1. 生成输入
    inputs = _generate_inputs(B, S, D, dtype, gamma2)

    # 2. CPU金标准
    golden_h, golden_x = _golden_cpu_fp64(inputs, norm_eps, hc_eps)

    # 3. NPU bench标杆
    bench_h, bench_x = _bench_npu_fp32(inputs, norm_eps, hc_eps)

    # 4. 准备NPU输入
    npu_args = _to_npu_args(inputs, norm_eps, hc_eps)

    # 5. 初始化模型并编译（使用npugraph_ex + static_kernel_compile）
    model = _StaticKernelModule().to("npu")
    model_compiled = torch.compile(
        model,
        fullgraph=True,
        backend="npugraph_ex",
        dynamic=False,
        options={"force_eager": True, "static_kernel_compile": True},
    )

    # 6. 模型预执行一次，完成静态kernel编译
    model_compiled(*npu_args)

    # 7. 录制ACLGraph
    g = torch_npu.npu.NPUGraph()
    with torch.npu.graph(g):
        npu_h, npu_x = model_compiled(*npu_args)

    # 8. 重放ACLGraph
    g.replay()
    torch.npu.synchronize()

    # 9. 精度校验
    tag = f"static_kernel_B{B}S{S}_D{D}_bf16_g2"
    ok_h = _precision_check(f"h_in_prime {tag}", npu_h.cpu().float(), bench_h.cpu().float(), golden_h, dtype)
    ok_x = _precision_check(f"x_2_out {tag}", npu_x.cpu().float(), bench_x.cpu().float(), golden_x, dtype)
    assert ok_h, f"h_in_prime precision FAILED for {tag}"
    assert ok_x, f"x_2_out precision FAILED for {tag}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "l0"])
