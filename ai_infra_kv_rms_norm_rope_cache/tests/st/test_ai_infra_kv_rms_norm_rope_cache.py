import torch
import torch_npu
import torch.nn.functional as F
from functorch import make_fx
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair import logger
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices
from torch._inductor.compile_fx import graph_returns_tuple
import omni_custom_ops

import pytest
import numpy as np
from typing import Optional, Tuple
from functools import partial
import time
import unittest
import copy
import logging

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

EPS: float = 1e-7
GOLDEN_CPU_DTYPE = torch.float64  # cpu golden升精度的位数

# ===== 精度等级配置 =====
DEFAULT_PRECISION_LEVEL = "L1"

precision_levels_config = {
    "L0": {"MARE": 10.0, "MERE": 2.0, "RMSE": 2.0},
    "L1": {"MARE": 5.0, "MERE": 1.5, "RMSE": 1.5},
    "L2": {"MARE": 2.0, "MERE": 1.2, "RMSE": 1.2},
}


# ===== 小值域配置 =====
small_value_config = {
    torch.float16: {"threshold": 2**-11, "error": 2**-16},
    torch.bfloat16: {"threshold": 2**-8, "error": 2**-16},
    torch.float32: {"threshold": 2**-14, "error": 2**-30},
}


# ═══════════════════════════════════════════════════════════
# 精度检测工具函数
# ═══════════════════════════════════════════════════════════

def absolute_error(actual: torch.Tensor, golden: torch.Tensor) -> torch.Tensor:
    """
    AE = |actual - golden|
    """
    return torch.abs(actual - golden)


def relative_error(actual: torch.Tensor, golden: torch.Tensor) -> torch.Tensor:
    """
    RE = |actual - golden| / (|golden| + eps)
    """
    return torch.abs(actual - golden) / (torch.abs(golden) + EPS)


# 最大相对误差（MARE）
def max_relative_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    """
    MARE = max(RE)
    """
    re: torch.Tensor = relative_error(actual, golden)
    return re.max().item()


# 平均相对误差（MERE）
def mean_relative_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    """
    MERE = mean(RE)
    """
    re: torch.Tensor = relative_error(actual, golden)
    return re.mean().item()


# 均方根误差（RMSE）
def root_mean_squared_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    """
    RMSE = sqrt(mean((actual - golden)^2))
    """
    diff: torch.Tensor = actual - golden
    mse: torch.Tensor = torch.mean(diff * diff)
    rmse: torch.Tensor = torch.sqrt(mse)
    return rmse.item()


# 统一误差统计（dict 返回）
def calc_error_metrics(actual: torch.Tensor, golden: torch.Tensor) -> dict[str, float]:
    """
    返回所有误差指标
    """
    assert actual.shape == golden.shape
    # assert golden.dtype == torch.float32
    assert golden.dtype == GOLDEN_CPU_DTYPE

    return {
        "MARE": max_relative_error(actual, golden),
        "MERE": mean_relative_error(actual, golden),
        "RMSE": root_mean_squared_error(actual, golden),
    }


# 按 tensor 划分【正常域 / 小值域】
def split_normal_and_small_domain(
    actual: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
):
    """
    返回：
      normal_actual, normal_golden,
      small_actual, small_golden
    """
    threshold = small_value_config[dtype]["threshold"]

    abs_golden = torch.abs(golden)
    normal_mask = abs_golden >= threshold
    small_mask = abs_golden < threshold

    return (
        actual[normal_mask],
        golden[normal_mask],
        actual[small_mask],
        golden[small_mask],
    )


# 正常域精度检查
def normal_domain_pass(
    npu_normal: torch.Tensor,
    gpu_normal: torch.Tensor,
    golden: torch.Tensor,
    precision_level: str = DEFAULT_PRECISION_LEVEL,
) -> bool:
    """
    正常域必须达到指定精度等级(如 L2)
    """
    npu_metrics = calc_error_metrics(npu_normal, golden)
    gpu_metrics = calc_error_metrics(gpu_normal, golden)

    # k遍历 MARE, MERE, RMSE
    ratio = {k: npu_metrics[k] / (gpu_metrics[k] + 1e-7) for k in npu_metrics}

    limits = precision_levels_config[precision_level]

    print("[NormalDomain] ratio:", ratio)
    print("[NormalDomain] required level:", precision_level)

    for k, limit in limits.items():
        if ratio[k] > limit:
            print(f"[NormalDomain] FAIL on {k}: {ratio[k]} > {limit}")
            return False

    print("[NormalDomain] PASS")
    return True


# 小值域精度检查（ErrorCount ratio）
def small_domain_pass(
    npu_actual: torch.Tensor,
    gpu_actual: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
) -> bool:
    cfg = small_value_config[dtype]
    threshold = cfg["threshold"]
    err = cfg["error"]

    def error_count(actual):
        mask = (torch.abs(golden) < threshold) & (torch.abs(actual - golden) > err)
        return int(mask.sum().item())

    npu_cnt = error_count(npu_actual)
    gpu_cnt = error_count(gpu_actual)

    ratio = npu_cnt / max(gpu_cnt, 1)

    print(
        f"[SmallDomain] NPU ErrorCount={npu_cnt}, "
        f"GPU ErrorCount={gpu_cnt}, Ratio={ratio:.3f}"
    )

    return ratio <= 2.0


# 最终统一接口
def precision_check(
    npu_out: torch.Tensor,
    gpu_out: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
    precision_level: str = DEFAULT_PRECISION_LEVEL,
) -> bool:
    """
    正常域 + 小值域 都通过，才算 PASS。

    当 gpu_out 与 golden 的误差极小（ratio 分母接近零）时，
    退化为直接对比 npu_out vs golden 的绝对指标。
    """
    (npu_normal, golden_normal, npu_small, golden_small) = (
        split_normal_and_small_domain(npu_out, golden, dtype)
    )

    (gpu_normal, _, gpu_small, _) = split_normal_and_small_domain(
        gpu_out, golden, dtype
    )

    # 1) 正常域
    normal_ok = True
    if golden_normal.numel() > 0:
        gpu_metrics = calc_error_metrics(gpu_normal, golden_normal)
        # 当 gpu_ref 误差极小时，ratio 分母接近零会导致爆炸
        # 此时退化为直接检查 npu vs golden 的绝对误差
        if gpu_metrics["MERE"] < 1e-6:
            npu_metrics = calc_error_metrics(npu_normal, golden_normal)
            print(f"[NormalDomain] gpu_ref nearly exact (MERE={gpu_metrics['MERE']:.2e}), fallback to absolute check")
            print(f"[NormalDomain] npu vs golden: MARE={npu_metrics['MARE']:.6f}, MERE={npu_metrics['MERE']:.6f}, RMSE={npu_metrics['RMSE']:.6f}")
            # bf16 的 relative error 上界约为 2^-7 ≈ 0.0078，给一定余量
            normal_ok = npu_metrics["MERE"] < 0.01
            if normal_ok:
                print("[NormalDomain] PASS (absolute)")
            else:
                print(f"[NormalDomain] FAIL (absolute): MERE {npu_metrics['MERE']:.6f} >= 0.01")
        else:
            normal_ok = normal_domain_pass(
                npu_normal, gpu_normal, golden_normal, precision_level
            )

    # 2) 小值域
    small_ok = True
    if golden_small.numel() > 0:
        small_ok = small_domain_pass(npu_small, gpu_small, golden_small, dtype)

    print(f"[Final] normal_pass={normal_ok}, small_pass={small_ok}")

    return normal_ok and small_ok


# ═══════════════════════════════════════════════════════════
# CPU调用 — CPU Golden Reference
# ═══════════════════════════════════════════════════════════

def cpu_rms_norm(x, gamma, eps=1e-5):
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma  (FP64 精度计算)"""
    xf = x.double()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return ((xf * torch.rsqrt(var + eps)) * gamma.double()).to(x.dtype)


def cpu_rope(x, cos_val, sin_val):
    """RoPE: 直接对 half 格式（前半/后半）做 RotateHalf + cos/sin 旋转。(FP64 精度计算)"""
    xf = x.double()
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (xf * cos_val.double() + rot * sin_val.double()).to(x.dtype)


def npu_rms_norm(x, gamma, eps=1e-5):
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma  (FP64 精度计算)"""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return ((xf * torch.rsqrt(var + eps)) * gamma.float()).to(x.dtype)


def npu_rope(x, cos_val, sin_val):
    """RoPE: 直接对 half 格式（前半/后半）做 RotateHalf + cos/sin 旋转。(FP64 精度计算)"""
    xf = x.float()
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (xf * cos_val.float() + rot * sin_val.float()).to(x.dtype)


def half_to_interleave(x):
    """将 half 格式转换为 interleave 格式。

    half 格式:        [a0, a1, a2, ..., b0, b1, b2, ...]
    interleave 格式:  [a0, b0, a1, b1, a2, b2, ...]

    NPU 算子内部会做 deinterleave(interleave -> half)
    所以输入 NPU 前需要先将 half 格式转成 interleave 格式。
    """
    shape = x.shape
    half = shape[-1] // 2
    x_first = x[..., :half]
    x_second = x[..., half:]
    return torch.stack([x_first, x_second], dim=-1).reshape(shape)


def cpu_dynamic_block_quant(x_2d, group_size=128):
    """CPU 版 dynamic_block_quant: 对称 INT8 分组量化 (FP64 精度计算)

    对 x_2d 的每行按 group_size 分组，每组取 max(|x|) 计算 scale = max_abs / 127，
    量化为 INT8: round(x / scale), clamp 到 [-128, 127]。

    Args:
        x_2d: (N, D) float tensor
        group_size: 每组元素数

    Returns:
        quant_int8: (N, D) int8 tensor
        scales: (N, D // group_size) float32 tensor
    """
    N, D = x_2d.shape
    num_groups = D // group_size
    xf = x_2d.double().reshape(N, num_groups, group_size)
    max_abs = xf.abs().amax(dim=-1)  # (N, num_groups)
    scales = (max_abs / 127.0).float()  # (N, num_groups)
    # 避免 scale 为 0
    scales_safe = scales.double().clamp(min=1e-10)
    quant = torch.round(xf / scales_safe.unsqueeze(-1))
    quant_int8 = quant.to(torch.int8).reshape(N, D)
    return quant_int8, scales


def npu_dynamic_block_quant(x_2d, group_size=128):
    """NPU 版 dynamic_block_quant: 对称 INT8 分组量化 (FP32 精度计算)

    与 cpu_dynamic_block_quant 逻辑相同，但用 float32 而非 float64，
    作为三路对比中的 NPU golden baseline。

    Args:
        x_2d: (N, D) float tensor (on any device)
        group_size: 每组元素数

    Returns:
        quant_int8: (N, D) int8 tensor
        scales: (N, D // group_size) float32 tensor
    """
    N, D = x_2d.shape
    num_groups = D // group_size
    xf = x_2d.float().reshape(N, num_groups, group_size)
    max_abs = xf.abs().amax(dim=-1)  # (N, num_groups)
    scales = max_abs / 127.0         # (N, num_groups) float32
    scales_safe = scales.clamp(min=1e-10)
    quant = torch.round(xf / scales_safe.unsqueeze(-1))
    quant_int8 = quant.to(torch.int8).reshape(N, D)
    return quant_int8, scales


def cpu_reference(kv, gamma, cos_val, sin_val, eps=1e-5, rms_size=512):
    """完整 CPU golden: 拆分 kv -> RMSNorm + RoPE

    Args:
        kv: (..., HIDDEN_SIZE) 输入
        gamma: (RMS_SIZE,) RMSNorm 权重
        cos_val: (..., ROPE_SIZE) cos 值
        sin_val: (..., ROPE_SIZE) sin 值
        eps: RMSNorm epsilon

    Returns:
        y_out: (..., RMS_SIZE) RMSNorm 输出
        k_embed_out: (..., ROPE_SIZE) RoPE 输出
    """
    rms_in  = kv[..., :rms_size]   # [..., 512]
    rope_in = kv[..., rms_size:]   # [..., 64]
    y_out       = cpu_rms_norm(rms_in, gamma, eps)
    k_embed_out = cpu_rope(rope_in, cos_val, sin_val)
    return y_out, k_embed_out


# ═══════════════════════════════════════════════════════════
# NPU调用 — 在 NPU 上运行 CPU 参考实现 (device="npu")
# ═══════════════════════════════════════════════════════════

def npu_reference(kv, gamma, cos_val, sin_val, eps=1e-5, rms_size=512):
    """Same as cpu_reference but runs on NPU tensors (as gpu_out baseline for precision_check)."""
    rms_in  = kv[..., :rms_size]
    rope_in = kv[..., rms_size:]
    y_out       = npu_rms_norm(rms_in, gamma, eps)
    k_embed_out = npu_rope(rope_in, cos_val, sin_val)
    return y_out, k_embed_out


# ═══════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════

def make_noncontig_cache(
    block_num, block_size, num_heads, head_dim, pad, dtype, seed,
    device="npu", storage_offset=None
):
    """构造 dim0 非连续的 cache tensor

    Args:
        block_num: block 数量
        block_size: 每个 block 的 slot 数
        dim: 最后一维大小
        pad: 填充元素数 (使 dim0 stride > 连续 stride)
        dtype: 数据类型
        seed: seed
        device: 设备

    Returns:
        非连续 cache tensor: (block_num, block_size, seq_len, head_dim)
    """
    torch.manual_seed(seed)

    stride_0 = block_size * num_heads * head_dim + pad
    if dtype.is_floating_point:
        base = torch.randn(block_num, stride_0, dtype=dtype, device=device)
    else:
        base = torch.zeros(block_num, stride_0, dtype=dtype, device=device)
    cache = torch.as_strided(
        base,
        size=(block_num, block_size, num_heads, head_dim),
        stride=(stride_0, head_dim, head_dim, num_heads),
        storage_offset=storage_offset
    )
    return cache


def make_noncontig(data: torch.Tensor, pads: tuple, device=None) -> torch.Tensor:
    """构造非连续 tensor (仿照 test_ai_infra_causal_conv1d.py)

    在每个维度末尾添加 pad 个元素的填充，返回原始 shape 的切片（非连续视图）。

    Args:
        data: 原始 tensor
        pads: 每个维度的填充大小，长度须等于 data.ndim
        device: 目标设备，None 则沿用 data.device

    Returns:
        非连续 tensor，shape 与 data 相同，但 stride 更大
    """
    dev = device if device is not None else data.device
    padded_shape = tuple(s + p for s, p in zip(data.shape, pads))
    buf = torch.zeros(padded_shape, dtype=data.dtype, device=dev)
    idx = tuple(slice(s) for s in data.shape)
    buf[idx] = data.to(dev)
    return buf[idx]


def make_inputs(
    batch_size, seq_len, block_num, block_size, seed,
    dtype=torch.bfloat16, device="cpu", rms_size=512, rope_size=64
):
    """构造公共输入: kv, gamma, cos, sin, index

    Args:
        batch_size: batch 大小
        seq_len: 序列长度
        dtype: 数据类型
        device: 设备

    Returns:
        kv, gamma, cos, sin, index
    """
    torch.manual_seed(seed)
    kv    = torch.randn(batch_size, 1, seq_len, rms_size + rope_size, dtype=dtype, device=device)
    gamma = torch.randn(rms_size, dtype=dtype, device=device)
    cos   = torch.randn(batch_size, 1, seq_len, rope_size, dtype=dtype, device=device)
    sin   = torch.randn(batch_size, 1, seq_len, rope_size, dtype=dtype, device=device)
    index = torch.randperm(block_num * block_size, dtype=torch.int64)[:batch_size * seq_len]

    if device != "cpu":
        index = index.to(device)

    return kv, gamma, cos, sin, index


def compare(name, npu_tensor, cpu_tensor, atol=1e-2, rtol=1e-2):
    """对比 NPU 结果与 CPU golden"""
    npu_cpu = npu_tensor.cpu().float()
    cpu_f   = cpu_tensor.float()
    match   = torch.allclose(npu_cpu, cpu_f, atol=atol, rtol=rtol)
    diff    = (npu_cpu - cpu_f).abs()
    '''if match:
        print(f"  [PASS] {name}: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
    else:
        print(f"  [FAIL] {name}: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")'''

    if not match:
        print(f"  [FAIL] {name}: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
    return match


# ═══════════════════════════════════════════════════════════
# 网络定义类
# ═══════════════════════════════════════════════════════════
class KvRmsNormRopeCacheModel(torch.nn.Module):
    def forward(
        self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, v=None,
        epsilon=1e-6, cache_mode="Norm", rotary_mode="interleave-half",
        quant_mode="static", is_output_kv=False
    ):
        return torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
            kv, gamma, cos, sin, index, k_cache, ckv_cache,
            k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
            k_rope_offset=k_rope_offset, c_kv_offset=c_kv_offset, v=v,
            epsilon=epsilon, cache_mode=cache_mode,
            rotary_mode=rotary_mode, quant_mode=quant_mode,
            is_output_kv=is_output_kv
        )


# ═══════════════════════════════════════════════════════════
# 测试类 1: TestCustomAiInfraKvRmsNormRopeCacheV2
# ═══════════════════════════════════════════════════════════
class TestCustomAiInfraKvRmsNormRopeCacheV2(TestCase):
    def setUp(self):
        self._t0 = time.time()
        self.model = KvRmsNormRopeCacheModel().npu()
        self.npu_backend = None
        self.model_compiled = None

    def tearDown(self):
        print(f"[Timing] {self._testMethodName} elapsed: {time.time() - self._t0:.3f}s")

    def call_eager(self, *args, **kwargs):
        return torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(*args, **kwargs)

    def call_aclgraph_eager(
        self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale=None, c_kv_scale=None,
        k_rope_offset=None, c_kv_offset=None, v=None,
        epsilon=1e-5, cache_mode="Norm",
        rotary_mode="interleave-half", quant_mode="static",
        is_output_kv=False, model=None
    ):
        model = KvRmsNormRopeCacheModel().npu()
        model = torch.compile(model, backend="eager", dynamic=False)
        g = torch_npu.npu.NPUGraph()
        with torch.npu.graph(g, auto_dispatch_capture=True):
            res = model(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                k_rope_offset=k_rope_offset, c_kv_offset=c_kv_offset, v=v,
                epsilon=epsilon, cache_mode=cache_mode,
                rotary_mode=rotary_mode, quant_mode=quant_mode,
                is_output_kv=is_output_kv)
        g.replay()
        return res

    def _run_test_SWA(
        self, batch_size, block_num, block_size, seq_len, pad,
        cache_mode, rotary_mode, quant_mode, is_output_kv, seed, dtype, op_caller,
        epsilon=1e-5, rope_size=64, rms_size=512, storage_offset=None
    ):
        kv, gamma, cos, sin, index = make_inputs(
            batch_size=batch_size, seq_len=seq_len, block_num=block_num, block_size=block_size,
            seed=seed, dtype=torch.bfloat16, device="npu")

        # 1) CPU golden (FP64)
        y_golden, k_golden = cpu_reference(
            kv.cpu().to(GOLDEN_CPU_DTYPE), gamma.cpu().to(GOLDEN_CPU_DTYPE),
            cos.cpu().to(GOLDEN_CPU_DTYPE), sin.cpu().to(GOLDEN_CPU_DTYPE), epsilon)

        # 2) NPU golden (same math on NPU tensors)
        npu_y_golden, npu_k_golden = npu_reference(kv, gamma, cos, sin, epsilon)

        # 3) NPU 算子调用
        k_cache = make_noncontig_cache(
            block_num=block_num, block_size=block_size, num_heads=1, head_dim=rope_size,
            pad=pad, dtype=dtype, seed=seed, device="npu", storage_offset=storage_offset)
        ckv_cache = make_noncontig_cache(
            block_num=block_num, block_size=block_size, num_heads=1, head_dim=rms_size,
            pad=pad, dtype=dtype, seed=seed, device="npu", storage_offset=storage_offset)

        # 根据注入的执行器不同通路调用算子
        _, _ = op_caller(
                kv, gamma, cos, sin, index, k_cache, ckv_cache,
                epsilon=epsilon, cache_mode=cache_mode, rotary_mode=rotary_mode,
                quant_mode=quant_mode, is_output_kv=is_output_kv)

        # 4) 从 cache 中按 index 提取 NPU 算子输出
        num_tokens = batch_size * seq_len
        index_cpu = index.cpu()
        npu_ckv_list, npu_k_list = [], []
        for t in range(num_tokens):
            idx = index_cpu[t].item()
            blk, pos = idx // block_size, idx % block_size
            npu_ckv_list.append(ckv_cache[blk, pos, 0, :].cpu().float())
            npu_k_list.append(k_cache[blk, pos, 0, :].cpu().float())
        npu_ckv_flat = torch.stack(npu_ckv_list)
        npu_k_flat = torch.stack(npu_k_list)

        # 5) Flatten golden tensors
        golden_ckv = y_golden.reshape(num_tokens, rms_size).to(GOLDEN_CPU_DTYPE)
        golden_k = k_golden.reshape(num_tokens, rope_size).to(GOLDEN_CPU_DTYPE)
        gpu_ckv = npu_y_golden.cpu().reshape(num_tokens, rms_size).float()
        gpu_k = npu_k_golden.cpu().reshape(num_tokens, rope_size).float()

        # 6) precision_check: npu_out vs npu_golden vs cpu_golden
        print(f"\n[SWA] Precision check: ckv (RMSNorm)")
        ok_ckv = precision_check(npu_ckv_flat, gpu_ckv, golden_ckv, dtype)
        print(f"[SWA] Precision check: k (RoPE)")
        ok_k = precision_check(npu_k_flat, gpu_k, golden_k, dtype)
        self.assertTrue(ok_ckv and ok_k, "SWA precision check failed")

    def _run_test_SFA_NO_QUANT(
        self, batch_size, block_num, block_size, seq_len, pad,
        cache_mode, rotary_mode, quant_mode, is_output_kv, seed, dtype, op_caller,
        epsilon=1e-5, rope_size=64, rms_size=512, storage_offset=None
    ):
        kv, gamma, cos, sin, index = make_inputs(
            batch_size=batch_size, seq_len=seq_len, block_num=block_num, block_size=block_size,
            seed=seed, dtype=torch.bfloat16, device="npu")

        # 1) CPU golden (FP64)
        y_golden, k_golden = cpu_reference(
            kv.cpu().to(GOLDEN_CPU_DTYPE), gamma.cpu().to(GOLDEN_CPU_DTYPE),
            cos.cpu().to(GOLDEN_CPU_DTYPE), sin.cpu().to(GOLDEN_CPU_DTYPE), epsilon)

        # 2) NPU golden (same math on NPU tensors)
        npu_y_golden, npu_k_golden = npu_reference(kv, gamma, cos, sin, epsilon)

        # 3) NPU 算子调用
        ckv_cache = make_noncontig_cache(
            block_num=block_num, block_size=block_size, num_heads=1, head_dim=rms_size+rope_size,
            pad=pad, dtype=dtype, seed=seed, device="npu", storage_offset=storage_offset)

        # 根据注入的执行器不同通路调用算子
        _, _ = op_caller(
                kv, gamma, cos, sin, index, None, ckv_cache,
                epsilon=epsilon, cache_mode=cache_mode, rotary_mode=rotary_mode,
                quant_mode=quant_mode, is_output_kv=is_output_kv)

        # 4) 从 cache 中按 index 提取 NPU 算子输出
        num_tokens = batch_size * seq_len
        index_cpu = index.cpu()
        npu_ckv_list, npu_k_list = [], []
        for t in range(num_tokens):
            idx = index_cpu[t].item()
            blk, pos = idx // block_size, idx % block_size
            npu_ckv_list.append(ckv_cache[blk, pos, 0, :rms_size].cpu().float())
            npu_k_list.append(ckv_cache[blk, pos, 0, rms_size:].cpu().float())
        npu_ckv_flat = torch.stack(npu_ckv_list)
        npu_k_flat = torch.stack(npu_k_list)

        # 5) Flatten golden tensors
        golden_ckv = y_golden.reshape(num_tokens, rms_size).to(GOLDEN_CPU_DTYPE)
        golden_k = k_golden.reshape(num_tokens, rope_size).to(GOLDEN_CPU_DTYPE)
        gpu_ckv = npu_y_golden.cpu().reshape(num_tokens, rms_size).float()
        gpu_k = npu_k_golden.cpu().reshape(num_tokens, rope_size).float()

        # 6) precision_check: npu_out vs npu_golden vs cpu_golden
        print(f"\n[SFA_NO_QUANT] Precision check: ckv (RMSNorm)")
        ok_ckv = precision_check(npu_ckv_flat, gpu_ckv, golden_ckv, dtype)
        print(f"[SFA_NO_QUANT] Precision check: k (RoPE)")
        ok_k = precision_check(npu_k_flat, gpu_k, golden_k, dtype)
        self.assertTrue(ok_ckv and ok_k, "SFA_NO_QUANT precision check failed")

    def _run_test_SFA_C8(
        self, batch_size, block_num, block_size, seq_len, pad,
        cache_mode, rotary_mode, quant_mode, is_output_kv, seed, dtype, op_caller,
        epsilon=1e-5, rope_size=64, rms_size=512, quant_group_size=128, storage_offset=None
    ):
        """Case 2.3: uint8 非连续 cache，混合类型量化模式，三路对比。

        每行布局 (字节):
          [0 : rms_size]                      → INT8  ckv (rms_size bytes)
          [rms_size : rms_size+rope_size*2]   → BF16  k_pe (rope_size elements)
          [rms_size+rope_size*2 : row_bytes]  → FP32  scales (rms_size//quant_group groups)

        三路:
          NPU 算子: npu_rms_norm + npu_dynamic_block_quant + npu_rotary_mul，结果写入 cache
          NPU golden: npu_rms_norm + npu_dynamic_block_quant(float32) + npu_rope，在 NPU 上算
          CPU golden: cpu_rms_norm + cpu_dynamic_block_quant(float64) + cpu_rope，在 CPU 上算
        """
        scale_dims = rms_size // quant_group_size         # e.g. 4
        ckv_bytes  = rms_size * 1                    # INT8: 1 byte/elem
        kpe_bytes  = rope_size * 2                   # BF16: 2 bytes/elem
        scl_bytes  = scale_dims * 4                  # FP32: 4 bytes/group
        row_bytes  = ckv_bytes + kpe_bytes + scl_bytes
        num_tokens = batch_size * seq_len

        # 1) 公共输入 (NPU)
        kv, gamma, cos, sin, index = make_inputs(
            batch_size=batch_size, seq_len=seq_len,
            block_num=block_num, block_size=block_size,
            seed=seed, dtype=dtype, device="npu",
            rms_size=rms_size, rope_size=rope_size)
        rms_in  = kv[..., :rms_size]
        rope_in = kv[..., rms_size:]

        # 2) CPU golden (FP64)
        y_normed_cpu = cpu_rms_norm(
            rms_in.cpu().to(GOLDEN_CPU_DTYPE),
            gamma.cpu().to(GOLDEN_CPU_DTYPE), epsilon
        )
        y_normed_2d_cpu = y_normed_cpu.reshape(num_tokens, rms_size)
        ckv_int8_cpu, scales_cpu = cpu_dynamic_block_quant(y_normed_2d_cpu, group_size=quant_group_size)
        k_rope_cpu = cpu_rope(
            rope_in.cpu().to(GOLDEN_CPU_DTYPE),
            cos.cpu().to(GOLDEN_CPU_DTYPE),
            sin.cpu().to(GOLDEN_CPU_DTYPE)
        )
        k_rope_flat_cpu = k_rope_cpu.reshape(num_tokens, rope_size)

        # 3) NPU golden (FP32) — npu_rms_norm + npu_dynamic_block_quant + npu_rope
        y_normed_npu = npu_rms_norm(rms_in, gamma, epsilon)
        y_normed_2d_npu = y_normed_npu.reshape(num_tokens, rms_size)
        ckv_int8_npu, scales_npu = npu_dynamic_block_quant(y_normed_2d_npu, group_size=quant_group_size)
        k_rope_npu = npu_rope(rope_in, cos, sin)
        k_rope_flat_npu = k_rope_npu.reshape(num_tokens, rope_size)

        # 4) 构造 uint8 非连续 cache (NPU, shape: block_num x block_size x 1 x row_bytes)
        cache = make_noncontig_cache(
            block_num=block_num, block_size=block_size, num_heads=1, head_dim=row_bytes,
            pad=pad, dtype=torch.int8, seed=seed, device="npu",
            storage_offset=None)

        print(f"  cache: shape={tuple(cache.shape)}, "
          f"stride={cache.stride()}, is_contiguous={cache.is_contiguous()}")
        
        _, _ = op_caller(
                kv, gamma, cos, sin, index, None, cache,
                epsilon=epsilon, cache_mode=cache_mode, rotary_mode=rotary_mode,
                quant_mode=quant_mode, is_output_kv=is_output_kv)

        # 5) 从 cache 按 index 读回，展开三组 tensor
        npu_ckv_list, npu_kpe_list, npu_scale_list = [], [], []
        for t in range(num_tokens):
            idx = index.cpu()[t].item()
            blk, pos = idx // block_size, idx % block_size
            row = cache[blk, pos, 0, :].cpu().contiguous()
            npu_ckv_list.append(row[:ckv_bytes].view(torch.int8).float())
            npu_kpe_list.append(row[ckv_bytes:ckv_bytes + kpe_bytes].view(dtype).float())
            npu_scale_list.append(row[ckv_bytes + kpe_bytes:].view(torch.float32))
        npu_ckv_flat   = torch.stack(npu_ckv_list)    # (N, rms_size)  float32
        npu_kpe_flat   = torch.stack(npu_kpe_list)    # (N, rope_size) float32
        npu_scale_flat = torch.stack(npu_scale_list)  # (N, scale_dims) float32

        # NPU golden → float32 for comparison
        gpu_ckv_flat   = ckv_int8_npu.cpu().float()
        gpu_kpe_flat   = k_rope_flat_npu.cpu().float()
        gpu_scale_flat = scales_npu.cpu()

        # CPU golden → GOLDEN_CPU_DTYPE for precision_check, float32 for scale
        golden_ckv_flat   = ckv_int8_cpu.to(GOLDEN_CPU_DTYPE)
        golden_kpe_flat   = k_rope_flat_cpu.to(GOLDEN_CPU_DTYPE)
        golden_scale_flat = scales_cpu  # float32

        # 6) 三路 precision_check
        print(f"\n[SFA_C8] Precision check: ckv (INT8 quant)")
        ok_ckv = precision_check(npu_ckv_flat, gpu_ckv_flat, golden_ckv_flat, dtype)
        print(f"[SFA_C8] Precision check: k_pe (RoPE)")
        ok_kpe = precision_check(npu_kpe_flat, gpu_kpe_flat, golden_kpe_flat, dtype)

        # scale 用 allclose 直接对比 (float32，量级相近，不需要 normal/small 分域)
        ok_scale_npu = torch.allclose(npu_scale_flat, gpu_scale_flat, atol=1e-3, rtol=1e-2)
        ok_scale_cpu = torch.allclose(npu_scale_flat, golden_scale_flat, atol=1e-3, rtol=1e-2)
        ok_scale = ok_scale_npu and ok_scale_cpu
        if ok_scale:
            print("[SFA_C8] scale (FP32): PASS")
        else:
            diff_npu = (npu_scale_flat - gpu_scale_flat).abs().max().item()
            diff_cpu = (npu_scale_flat - golden_scale_flat).abs().max().item()
            print(f"[SFA_C8] scale (FP32): FAIL  npu_diff={diff_npu:.6f}  cpu_diff={diff_cpu:.6f}")

        self.assertTrue(ok_ckv and ok_kpe and ok_scale, "SFA_C8 precision check failed")

    
    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_ai_infra_kv_rmsnorm_rope_cache_v2_SFA_NQ_bf16_01(self):
        self._run_test_SFA_NO_QUANT(
            batch_size=4, block_num=16384, block_size=64, seq_len=32000, pad=64,
            cache_mode="PA_BNSD", rotary_mode="half", quant_mode="static",
            is_output_kv=False, seed=44, dtype=torch.bfloat16, op_caller=self.call_eager
        )
    
    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_ai_infra_kv_rmsnorm_rope_cache_v2_SWA_bf16_01(self):
        self._run_test_SWA(
            batch_size=1, block_num=4096, block_size=64, seq_len=2048, pad=64,
            cache_mode="PA_BNSD", rotary_mode="half", quant_mode="static",
            is_output_kv=False, seed=44, dtype=torch.bfloat16, op_caller=self.call_aclgraph_eager
        )
        
    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_ai_infra_kv_rmsnorm_rope_cache_v2_SFA_C8_bf16_01(self):
        self._run_test_SFA_C8(
            batch_size=1, block_num=5000, block_size=64, seq_len=5, pad=256,
            cache_mode="PA_BNSD", rotary_mode="half", quant_mode="pertile128",
            is_output_kv=False, seed=42, dtype=torch.bfloat16, op_caller=self.call_eager
        )


# ═══════════════════════════════════════════════════════════
# 测试类 2: TestNPUKvRmsNormRopeCache
# ═══════════════════════════════════════════════════════════
class TestNPUKvRmsNormRopeCache(TestCase):

    dim_kv = [512, 192]
    dim_v = [None, 128]
    k_cache_len = [64, 192]
    ckv_cache_len = [512, 128]

    def setUp(self):
        self._t0 = time.time()

    def tearDown(self):
        print(f"[Timing] {self._testMethodName} elapsed: {time.time() - self._t0:.3f}s")

    def generate_inputs(
        self, method_mode, batch_size, seq_len, page_num, page_size,
        quant_mode, cache_mode, output_mode, input_dtype
    ):
        kv = (torch.randn(batch_size, 1, seq_len, 576, dtype=input_dtype) if method_mode == 0
              else torch.randn(batch_size, 1, seq_len, 192, dtype=input_dtype))
        gamma = torch.randn(self.dim_kv[method_mode], dtype=input_dtype)
        cos = torch.randn(batch_size, 1, seq_len, 64, dtype=input_dtype)
        sin = torch.randn(batch_size, 1, seq_len, 64, dtype=input_dtype)
        v = (None if method_mode == 0
             else torch.randn(batch_size, 1, seq_len, self.dim_v[method_mode], dtype=input_dtype))
        if cache_mode != "Norm":
            k_cache = torch.ones(page_num, page_size, 1, self.k_cache_len[method_mode], dtype=input_dtype) * 9
            ckv_cache = torch.ones(page_num, page_size, 1, self.ckv_cache_len[method_mode], dtype=input_dtype) * 9
            if "BLK" in cache_mode:
                index_shape = (batch_size * ((seq_len + page_size - 1) // page_size),)
                index = torch.arange(start=0, end=index_shape[0] * page_size, step=page_size, dtype=torch.int64)
            else:
                index_shape = (batch_size * seq_len,)
                index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64)
        if quant_mode == 1:
            k_cache = k_cache.to(torch.int8)
            ckv_cache = ckv_cache.to(torch.int8)
            k_rope_scale = torch.randn(self.k_cache_len[method_mode], dtype=torch.float32)
            c_kv_scale = torch.randn(self.ckv_cache_len[method_mode], dtype=torch.float32)
        else:
            k_rope_scale = None
            c_kv_scale = None

        # call golden
        kv = (-2 + 10) * kv - 10
        gamma = (-10 + 1000) * gamma - 1000
        sin = (0.01 + 0.01) * sin - 0.01
        v = None if method_mode == 0 else (-2 + 10) * v - 10

        return kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, v, cache_mode, output_mode, input_dtype

    def supported_op_exec(
        self, method_mode, kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
        v=None, epsilon=1e-05, cache_mode="Norm", is_output_kv=False
    ):

        # golden function
        def round_float_to_int8(src_tensor):
            return torch.round(src_tensor).clip(-128, 127).to(torch.int8)

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        batch_size, _, seq_len, _ = kv.shape
        if "PA" in cache_mode:
            block_num, block_size, _, _ = k_cache.shape
        if "BLK" in cache_mode:
            index_page_id_length = index.shape[0] // batch_size
        if k_rope_scale is None:
            quantMode = 0
            d0 = 16
        else:
            quantMode = 1
            d0 = 32

        if method_mode == 0:
            # split input
            rms_in, rope_in = kv.split([512, 64], dim=-1)
            rms_in = rms_in.to(torch.float32)
            rope_in = rope_in.to(torch.float32)

            # calc rmsnorm
            y = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + epsilon)
            y = y * gamma.to(torch.float32)

            # calc rope
            k = rope_in.view(batch_size, 1, seq_len, 32, 2).transpose(4, 3).reshape(batch_size, 1, seq_len, 64)
            k_embed = (k * cos.to(torch.float32)) + (rotate_half(k) * sin.to(torch.float32))

            # copy outputs
            k_embed_out = k_embed.clone().to(kv.dtype)
            y_out = y.clone().to(kv.dtype)
            if quantMode == 1:
                k_embed = round_float_to_int8(k_embed * k_rope_scale)
                y = round_float_to_int8(y * c_kv_scale)
            else:
                k_embed = k_embed.to(k_cache.dtype)
                y = y.to(k_cache.dtype)
        else:
            # calc rmsnorm and rope
            rms_in = kv.to(torch.float32)
            v_in = v.to(torch.float32)
            y = rms_in / torch.sqrt(torch.mean(rms_in ** 2, dim=-1, keepdim=True) + epsilon)
            y = y * gamma.to(torch.float32)
            k = y[..., :64].view(batch_size, 1, seq_len, 32, 2).transpose(4, 3).reshape(batch_size, 1, seq_len, 64)
            k_embed = (k * cos.to(torch.float32)) + (rotate_half(k) * sin.to(torch.float32))
            y = torch.cat([k_embed, y[..., 64:]], dim=-1)

            # copy outputs
            k_embed_out = k_embed.clone().to(kv.dtype)
            y_out = y.clone().to(kv.dtype)
            if quantMode == 1:
                k_embed = round_float_to_int8(y * k_rope_scale)
                y = round_float_to_int8(v_in * c_kv_scale)
            else:
                k_embed = y.to(k_cache.dtype)
                y = v_in.to(k_cache.dtype)

        # scatter update
        if cache_mode == "Norm":
            """ Norm mode """
            pass

        elif cache_mode in ("PA", "PA_BNSD"):
            k_cache = k_cache.reshape(-1, self.k_cache_len[method_mode])
            ckv_cache = ckv_cache.reshape(-1, self.ckv_cache_len[method_mode])
            for batch_id in range(batch_size):
                for token_id in range(seq_len):
                    offset = index[batch_id * seq_len + token_id]
                    if offset >= 0:
                        k_cache[offset, :] = k_embed[batch_id, 0, token_id, :]
                        ckv_cache[offset, :] = y[batch_id, 0, token_id, :]
            k_cache = k_cache.reshape(block_num, block_size, 1, self.k_cache_len[method_mode])
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, self.ckv_cache_len[method_mode])

        elif cache_mode == "PA_BLK_NZ":
            """ PA_BLK_NZ mode """
            k_cache = k_cache.reshape(block_num, 1, -1, block_size, d0)
            ckv_cache = ckv_cache.reshape(block_num, 1, -1, block_size, d0)
            for batch_id in range(batch_size):
                for tokenInCurrentBatch in range(seq_len):
                    tokenId = batch_id * seq_len + tokenInCurrentBatch
                    indexPageId = tokenInCurrentBatch // block_size
                    pageOffset = index[batch_id *
                                    index_page_id_length + indexPageId]
                    if pageOffset < 0:
                        continue
                    pageId = pageOffset // block_size
                    tokenOffsetInCurrentPage = tokenInCurrentBatch % block_size
                    k_cache[pageId, 0, :, tokenOffsetInCurrentPage,
                            :] = k_embed[batch_id, 0, tokenInCurrentBatch, :].reshape(-1, d0)
                    ckv_cache[pageId, 0, :, tokenOffsetInCurrentPage,
                            :] = y[batch_id, 0, tokenInCurrentBatch, :].reshape(-1, d0)
            k_cache = k_cache.reshape(block_num, block_size, 1, self.k_cache_len[method_mode])
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, self.ckv_cache_len[method_mode])

        elif cache_mode == "PA_BLK_BNSD":
            """ PA_BLK_BNSD mode """
            k_cache = k_cache.reshape(block_num, block_size, 1, -1)
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, -1)
            for batch_id in range(batch_size):
                for tokenInCurrentBatch in range(seq_len):
                    tokenId = batch_id * seq_len + tokenInCurrentBatch
                    indexPageId = tokenInCurrentBatch // block_size
                    pageOffset = index[batch_id *
                                    index_page_id_length + indexPageId]
                    if pageOffset < 0:
                        continue
                    pageId = pageOffset // block_size
                    tokenOffsetInCurrentPage = tokenInCurrentBatch % block_size
                    k_cache[pageId, tokenOffsetInCurrentPage, 0,
                            :] = k_embed[batch_id, 0, tokenInCurrentBatch, :]
                    ckv_cache[pageId, tokenOffsetInCurrentPage, 0,
                            :] = y[batch_id, 0, tokenInCurrentBatch, :]
            k_cache = k_cache.reshape(block_num, block_size, 1, self.k_cache_len[method_mode])
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, self.ckv_cache_len[method_mode])

        elif cache_mode == "PA_NZ":
            """ PA_NZ """
            if k_rope_scale is not None:
                d0 = 32
            else:
                d0 = 16
            k_cache = k_cache.reshape(block_num, 1, -1, block_size, d0)
            ckv_cache = ckv_cache.reshape(block_num, 1, -1, block_size, d0)
            for batch_id in range(batch_size):
                for tokenInCurrentBatch in range(seq_len):
                    pageOffset = index[batch_id *
                                    seq_len + tokenInCurrentBatch]
                    if pageOffset >= 0:
                        pageId = pageOffset // block_size
                        tokenOffsetInCurrentPage = pageOffset % block_size
                        k_cache[pageId, 0, :, tokenOffsetInCurrentPage,
                                :] = k_embed[batch_id, 0, tokenInCurrentBatch, :].reshape(-1, d0)
                        ckv_cache[pageId, 0, :, tokenOffsetInCurrentPage,
                                :] = y[batch_id, 0, tokenInCurrentBatch, :].reshape(-1, d0)
            k_cache = k_cache.reshape(block_num, block_size, 1, self.k_cache_len[method_mode])
            ckv_cache = ckv_cache.reshape(block_num, block_size, 1, self.ckv_cache_len[method_mode])

        return k_cache, ckv_cache, k_embed_out, y_out

    def custom_op_exec(
        self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
        v=None, epsilon=1e-05, cache_mode="Norm", rotary_mode="interleave-half",
        quant_mode="static", is_output_kv=False
    ):
        kv = kv.npu()
        gamma = gamma.npu()
        cos = cos.npu()
        sin = sin.npu()
        index = index.npu()
        k_cache = k_cache.npu()
        ckv_cache = ckv_cache.npu()
        if k_rope_scale is not None:
            k_rope_scale = k_rope_scale.npu()
            c_kv_scale = c_kv_scale.npu()
        if v is not None:
            v = v.npu()

        k_rope_npu, c_kv_npu = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
                kv, gamma, cos, sin, index,
                k_cache, ckv_cache,
                k_rope_scale=k_rope_scale,
                c_kv_scale=c_kv_scale,
                k_rope_offset=k_rope_offset,
                c_kv_offset=c_kv_offset,
                v=v,
                epsilon=epsilon,
                cache_mode=cache_mode,
                rotary_mode=rotary_mode,
                quant_mode=quant_mode,
                is_output_kv=is_output_kv
            )

        k_cache_cpu = k_cache.cpu()
        ckv_cache_cpu = ckv_cache.cpu()
        k_rope_cpu = k_rope_npu.cpu()
        c_kv_cpu = c_kv_npu.cpu()
        torch._dynamo.reset()

        return k_cache_cpu, ckv_cache_cpu, k_rope_cpu, c_kv_cpu

    def _run_test(
        self, method_mode, batch_size, seq_len, page_num, page_size,
        quant_mode, cache_mode, output_mode, input_dtype, test_name, device="npu"
    ):
        kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, v, cache_mode, output_mode, input_dtype = self.generate_inputs(
            method_mode, batch_size, seq_len, page_num, page_size,
            quant_mode, cache_mode, output_mode, input_dtype)

        k_rope_scale_cpu = k_rope_scale.to(GOLDEN_CPU_DTYPE) if k_rope_scale is not None else k_rope_scale
        c_kv_scale_cpu = c_kv_scale.to(GOLDEN_CPU_DTYPE) if c_kv_scale is not None else c_kv_scale
        v_cpu = v.to(GOLDEN_CPU_DTYPE) if v is not None else v
        v_npu = v.to(device) if v is not None else v

        golden_out = self.supported_op_exec(
            method_mode, kv.to(GOLDEN_CPU_DTYPE), gamma.to(GOLDEN_CPU_DTYPE),
            cos.to(GOLDEN_CPU_DTYPE), sin.to(GOLDEN_CPU_DTYPE), index,
            k_cache.to(GOLDEN_CPU_DTYPE), ckv_cache.to(GOLDEN_CPU_DTYPE),
            k_rope_scale=k_rope_scale_cpu, c_kv_scale=c_kv_scale_cpu,
            k_rope_offset=None, c_kv_offset=None, v=v_cpu,
            epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        k_rope_scale_npu = k_rope_scale.to(device) if k_rope_scale is not None else k_rope_scale
        c_kv_scale_npu = c_kv_scale.to(device) if c_kv_scale is not None else c_kv_scale

        golden_out_npu = self.supported_op_exec(
            method_mode, kv.to(device), gamma.to(device),
            cos.to(device), sin.to(device), index.to(device),
            k_cache.to(device), ckv_cache.to(device),
            k_rope_scale=k_rope_scale_npu, c_kv_scale=c_kv_scale_npu,
            k_rope_offset=None, c_kv_offset=None, v=v_npu,
            epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        # call npu api
        npu_out = self.custom_op_exec(
            kv, gamma, cos, sin, index, k_cache, ckv_cache,
            k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
            k_rope_offset=None, c_kv_offset=None, v=v,
            epsilon=1e-05, cache_mode=cache_mode, is_output_kv=output_mode)

        k_cache_npu, ckv_cache_npu, k_rope_npu, c_kv_npu = npu_out
        k_cache_cpu, ckv_cache_cpu, k_rope_cpu, c_kv_cpu = golden_out
        k_cache_npu_golden, ckv_cache_npu_golden, k_rope_npu_golden, c_kv_npu_golden = golden_out_npu

        k_cache_cpu = k_cache_cpu.to(GOLDEN_CPU_DTYPE)
        ckv_cache_cpu = ckv_cache_cpu.to(GOLDEN_CPU_DTYPE)
        k_rope_cpu = k_rope_cpu.to(GOLDEN_CPU_DTYPE)
        c_kv_cpu = c_kv_cpu.to(GOLDEN_CPU_DTYPE)

        ok_k = precision_check(k_cache_npu.cpu(), k_cache_npu_golden.cpu(), k_cache_cpu, torch.bfloat16)
        ok_ckv = precision_check(ckv_cache_npu.cpu(), ckv_cache_npu_golden.cpu(), ckv_cache_cpu, torch.bfloat16)
        self.assertTrue(ok_ckv and ok_k, f"{test_name} precision check failed")

        if output_mode:
            ok_k = precision_check(k_rope_npu.cpu(), k_rope_npu_golden.cpu(), k_rope_cpu, torch.bfloat16)
            ok_ckv = precision_check(c_kv_npu.cpu(), c_kv_npu_golden.cpu(), c_kv_cpu, torch.bfloat16)
            self.assertTrue(ok_ckv and ok_k, f"{test_name} output_kv precision check failed")


    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_npu_kv_rmsnorm_rope_cache_PA_BNSD(self):
        self._run_test(
            method_mode=0, batch_size=64, seq_len=1, page_num=576, page_size=128,
            quant_mode=0, cache_mode="PA_BNSD", output_mode=False,
            input_dtype=torch.bfloat16, test_name="test_npu_kv_rmsnorm_rope_cache_PA_BNSD")

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_npu_kv_rmsnorm_rope_cache_PA_BLK_NZ(self):
        self._run_test(
            method_mode=0, batch_size=64, seq_len=1, page_num=576, page_size=128,
            quant_mode=0, cache_mode="PA_BLK_NZ", output_mode=False,
            input_dtype=torch.bfloat16, test_name="test_npu_kv_rmsnorm_rope_cache_PA_BLK_NZ")

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_npu_kv_rmsnorm_rope_cache_PA_BLK_BNSD(self):
        self._run_test(
            method_mode=0, batch_size=64, seq_len=1, page_num=576, page_size=128,
            quant_mode=0, cache_mode="PA_BLK_BNSD", output_mode=False,
            input_dtype=torch.bfloat16, test_name="test_npu_kv_rmsnorm_rope_cache_PA_BLK_BNSD")

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_npu_kv_rmsnorm_rope_cache_PA_NZ(self):
        self._run_test(
            method_mode=0, batch_size=64, seq_len=1, page_num=576, page_size=128,
            quant_mode=0, cache_mode="PA_NZ", output_mode=False,
            input_dtype=torch.bfloat16, test_name="test_npu_kv_rmsnorm_rope_cache_PA_NZ")

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_npu_kv_rmsnorm_rope_cache_PA_BNSD_v2(self):
        self._run_test(
            method_mode=1, batch_size=64, seq_len=1, page_num=576, page_size=128,
            quant_mode=0, cache_mode="PA_BNSD", output_mode=False,
            input_dtype=torch.bfloat16, test_name="test_npu_kv_rmsnorm_rope_cache_PA_BNSD_v2")

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_npu_kv_rmsnorm_rope_cache_PA_BLK_NZ_v2(self, device="npu"):
        self._run_test(
            method_mode=1, batch_size=64, seq_len=1, page_num=576, page_size=128,
            quant_mode=0, cache_mode="PA_BLK_NZ", output_mode=False,
            input_dtype=torch.bfloat16, test_name="test_npu_kv_rmsnorm_rope_cache_PA_BLK_NZ_v2")

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_npu_kv_rmsnorm_rope_cache_PA_BLK_BNSD_v2(self, device="npu"):
        self._run_test(
            method_mode=1, batch_size=64, seq_len=1, page_num=576, page_size=128,
            quant_mode=0, cache_mode="PA_BLK_BNSD", output_mode=False,
            input_dtype=torch.bfloat16, test_name="test_npu_kv_rmsnorm_rope_cache_PA_BLK_BNSD_v2")

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_npu_kv_rmsnorm_rope_cache_PA_NZ_v2(self, device="npu"):
        self._run_test(
            method_mode=1, batch_size=64, seq_len=1, page_num=576, page_size=128,
            quant_mode=0, cache_mode="PA_NZ", output_mode=False,
            input_dtype=torch.bfloat16, test_name="test_npu_kv_rmsnorm_rope_cache_PA_NZ_v2")


if __name__ == "__main__":
    unittest.main()
