#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""
昇腾精度标准 2.0 精度比对脚本 (Ascend Precision Standard 2.0)

核心特性:
  1. ULP (Unit in the Last Place) 误差分析 —— 最科学的浮点精度度量
  2. MERE (Mean Relative Error) + MARE (Max Relative Error) 双指标体系
  3. 场景化分层阈值: 正常值域 / 小值域(≈0) / Inf·NaN 特殊处理
  4. 按数据类型自动适配阈值 (FP32/FP16/BF16/FP8/INT8)
  5. 规约次数感知的容差 —— 累加/归约操作会放大误差
  6. 余弦相似度 —— 整体一致性评估
  7. ULP 分位数统计 (P50/P95/P99)
  8. 与既有 result_compare_method.py 兼容的 API

评判逻辑 (与 CANN-Bench 对齐):
  - MERE < 对应类型的 1 ULP  →  Pass
  - MARE < 10 × MERE        →  Pass
  - 小值域: 绝对误差 < 小值容差 →  Pass
  - 失败元素占比 < 0.5%       →  Pass
"""

import logging
import math
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Union

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s]-%(filename)s:%(lineno)04d - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

_EPS = 1e-10

# ============================================================================
# 1. 数据类型精度配置表 (基于 IEEE 754 及昇腾硬件特性)
# ============================================================================
# 1 ULP ≈ 2^(-mantissa_bits) × value
# FP32: mantissa=23 bits → 1 ULP ≈ 1.19e-7
# FP16: mantissa=10 bits → 1 ULP ≈ 9.77e-4
# BF16: mantissa=7  bits → 1 ULP ≈ 7.81e-3

@dataclass
class PrecisionThreshold:
    """单个数据类型的精度阈值配置"""
    mere: float          # 平均相对误差阈值 (1 ULP)
    mare: float          # 最大相对误差阈值 (10 × MERE)
    small_value: float   # 小值域阈值 (绝对值低于此值视为小值)
    small_atol: float    # 小值域绝对误差容忍
    fail_ratio: float    # 允许的失败元素占比上限
    cos_sim_min: float   # 最低余弦相似度

# 各数据类型精度阈值
PRECISION_CONFIG: Dict[torch.dtype, PrecisionThreshold] = {
    torch.float32: PrecisionThreshold(
        mere=1.2e-7, mare=1.2e-6, small_value=1e-5, small_atol=1e-6,
        fail_ratio=0.001, cos_sim_min=0.999999
    ),
    torch.float16: PrecisionThreshold(
        mere=1.0e-3, mare=1.0e-2, small_value=1e-3, small_atol=1e-5,
        fail_ratio=0.005, cos_sim_min=0.999
    ),
    torch.bfloat16: PrecisionThreshold(
        mere=8.0e-3, mare=8.0e-2, small_value=1e-2, small_atol=1e-4,
        fail_ratio=0.005, cos_sim_min=0.999
    ),
}

# FP8 类型 (PyTorch 中通常由自定义 dtype 或 uint8 承载)
FP8_CONFIG = PrecisionThreshold(
    mere=2.0e-1, mare=1.0, small_value=1e-1, small_atol=1e-2,
    fail_ratio=0.01, cos_sim_min=0.99
)

INT8_CONFIG = PrecisionThreshold(
    mere=1e-3, mare=1e-2, small_value=0.5, small_atol=1.0,
    fail_ratio=0.001, cos_sim_min=0.999999
)


# ============================================================================
# 2. ULP 误差计算
# ============================================================================

def compute_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算两个浮点张量之间的 ULP (Unit in the Last Place) 差值。

    原理: 将 IEEE 754 浮点数的位表示重新解释为有符号整数,
          转换为能正确排序的整数后再计算差值。
          ±0.0 被视为同值 (ULP=0)。

    Args:
        a, b: 相同 shape 的 float32 张量

    Returns:
        ulp_diff: int64 张量, 逐元素 ULP 差值
    """
    sign_mask = 0x7FFFFFFF
    sign_bit = 0x80000000

    a_int = a.view(torch.int32).to(torch.int64)
    b_int = b.view(torch.int32).to(torch.int64)

    # IEEE 754 → 有序整数映射:
    #   正数: 置符号位为1, 正数之间整数顺序与浮点顺序一致
    #   负数: 全部位取反, 这样负数之间整数顺序也与浮点顺序一致
    #   结果: 0x007FFFFF(-Inf) < ... < 0x7FFFFFFF(-0.0) < 0x80000000(+0.0) < 0xFF7FFFFF(+Inf)
    a_sign = a_int & sign_bit
    b_sign = b_int & sign_bit

    a_ordered = torch.where(a_sign != 0,
                            sign_mask - (a_int & sign_mask),   # 负数: 取反
                            a_int | sign_bit)                   # 正数: 只翻符号位
    b_ordered = torch.where(b_sign != 0,
                            sign_mask - (b_int & sign_mask),
                            b_int | sign_bit)

    ulp_diff = torch.abs(a_ordered - b_ordered)

    # ±0.0 视为同值: 两者绝对值都为零时 ULP=0
    a_abs = a_int & sign_mask
    b_abs = b_int & sign_mask
    both_zero = (a_abs == 0) & (b_abs == 0)
    ulp_diff = torch.where(both_zero, torch.zeros_like(ulp_diff), ulp_diff)

    return ulp_diff


# ============================================================================
# 3. 精度指标计算
# ============================================================================

@dataclass
class PrecisionReport:
    """精度比对综合报告"""
    test_name: str = ""
    output_dtype: str = ""
    total_elements: int = 0

    # 基础误差
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    rmse: float = 0.0

    # 相对误差 (正常值域)
    mere: float = 0.0           # Mean Relative Error
    mare: float = 0.0           # Max Relative Error

    # ULP 误差
    ulp_max: int = 0
    ulp_mean: float = 0.0
    ulp_p50: float = 0.0        # 中位数
    ulp_p95: float = 0.0
    ulp_p99: float = 0.0

    # 场景化统计
    normal_count: int = 0
    small_count: int = 0
    overflow_count: int = 0     # Golden 中的 Inf/NaN
    fail_count: int = 0
    fail_ratio: float = 0.0

    # 大值域相对误差
    mere_big: float = 0.0
    mare_big: float = 0.0
    big_count: int = 0

    # 小值域统计
    small_err_count: int = 0
    small_err_ratio: float = 0.0

    # 整体一致性
    cosine_similarity: float = 0.0
    max_ulp_bin: int = 0        # 最大误差所在的位置 (index)

    # 判定
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    # 原始数据引用 (用于 licensee 自行提取)
    _abs_error: Optional[torch.Tensor] = field(default=None, repr=False)
    _rel_error: Optional[torch.Tensor] = field(default=None, repr=False)
    _ulp_diff: Optional[torch.Tensor] = field(default=None, repr=False)
    _fail_mask: Optional[torch.Tensor] = field(default=None, repr=False)


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个一维张量的余弦相似度"""
    a_norm = torch.norm(a).item()
    b_norm = torch.norm(b).item()
    if a_norm < 1e-12 and b_norm < 1e-12:
        return 1.0
    if a_norm < 1e-12 or b_norm < 1e-12:
        return 0.0
    dot = torch.dot(a, b).item()
    cos_sim = dot / (a_norm * b_norm)
    return max(-1.0, min(1.0, cos_sim))


def compute_precision_report(
    golden: torch.Tensor,
    result: torch.Tensor,
    test_name: str = "",
    config: Optional[PrecisionThreshold] = None,
    compute_count: int = 1,
) -> PrecisionReport:
    """
    核心精度评测函数 — 对标昇腾精度标准 2.0。

    Args:
        golden:   标杆结果 (CPU/FP64 高精度参考)
        result:   NPU 计算结果
        test_name: 测试用例名称
        config:   自定义精度阈值, 为 None 则根据 result.dtype 自动选取
        compute_count: 累加/归约次数, 用于动态放宽阈值

    Returns:
        PrecisionReport 包含所有精度指标及通过/失败判定
    """
    report = PrecisionReport(test_name=test_name)

    # --- 基本校验 ---
    golden_f = golden.detach().cpu().flatten()
    result_f = result.detach().cpu().flatten()

    report.total_elements = int(golden_f.numel())
    report.output_dtype = str(result.dtype)

    if report.total_elements == 0:
        report.passed = True
        return report

    if golden_f.numel() != result_f.numel():
        report.failure_reasons.append(f"Shape mismatch: golden={golden_f.numel()} result={result_f.numel()}")
        return report

    # --- 确定阈值配置 ---
    if config is None:
        dtype = result.dtype
        if str(dtype) in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
            config = FP8_CONFIG
        elif dtype in (torch.int8, torch.uint8, torch.int32):
            config = INT8_CONFIG
        else:
            config = PRECISION_CONFIG.get(dtype, PRECISION_CONFIG[torch.float32])

    # 规约次数修正: 误差随 sqrt(compute_count) 增长
    reduction_factor = max(1.0, math.sqrt(float(compute_count)) * 0.5)
    mere_threshold = config.mere * reduction_factor
    mare_threshold = config.mare * reduction_factor

    # --- 转为 float32 统一计算 ---
    golden_f32 = golden_f.to(torch.float32)
    result_f32 = result_f.to(torch.float32)

    # --- Inf/NaN 检测 ---
    golden_inf = torch.isinf(golden_f32)
    golden_nan = torch.isnan(golden_f32)
    result_inf = torch.isinf(result_f32)
    result_nan = torch.isnan(result_f32)

    report.overflow_count = int((golden_inf | golden_nan).sum().item())

    # Inf 位置: 确认结果也是一致的 Inf
    inf_match = golden_inf & result_inf
    inf_mismatch = golden_inf ^ result_inf
    nan_match = golden_nan & result_nan

    if inf_mismatch.any():
        report.failure_reasons.append(
            f"Inf mismatch: {int(inf_mismatch.sum().item())} elements"
        )

    # NaN 位置: 确认结果也是 NaN
    nan_mismatch = golden_nan ^ result_nan
    if nan_mismatch.any():
        report.failure_reasons.append(
            f"NaN mismatch: {int(nan_mismatch.sum().item())} elements"
        )

    # --- 排除 Inf/NaN 后的有效元素 ---
    valid_mask = ~(golden_inf | golden_nan | result_inf | result_nan)
    valid_golden = golden_f32[valid_mask]
    valid_result = result_f32[valid_mask]
    valid_count = int(valid_mask.sum().item())

    if valid_count == 0:
        report.passed = (inf_mismatch.sum().item() == 0 and nan_mismatch.sum().item() == 0)
        if report.passed:
            report.cosine_similarity = 1.0
        return report

    # --- 基础误差 ---
    abs_error = torch.abs(valid_result - valid_golden)
    report.max_abs_error = float(abs_error.max().item())
    report.mean_abs_error = float(abs_error.mean().item())
    report.rmse = float(torch.sqrt((abs_error ** 2).mean()).item())

    # --- ULP 误差 ---
    ulp_diff = compute_ulp_diff(valid_result, valid_golden)
    report.ulp_max = int(ulp_diff.max().item())
    report.ulp_mean = float(ulp_diff.float().mean().item())

    ulp_sorted = ulp_diff.sort().values
    n = ulp_sorted.numel()
    report.ulp_p50 = float(ulp_sorted[n // 2].item())
    report.ulp_p95 = float(ulp_sorted[int(n * 0.95)].item())
    report.ulp_p99 = float(ulp_sorted[int(n * 0.99)].item())

    report.max_ulp_bin = int(ulp_diff.argmax().item())

    report._abs_error = abs_error
    report._ulp_diff = ulp_diff

    # --- 余弦相似度 ---
    report.cosine_similarity = compute_cosine_similarity(valid_golden, valid_result)

    # --- 场景分离: 大值 vs 小值 ---
    golden_abs = torch.abs(valid_golden)
    is_big = golden_abs >= config.small_value
    is_small = ~is_big

    report.normal_count = int(is_big.sum().item())
    report.small_count = int(is_small.sum().item())

    # --- 大值域相对误差: MERE / MARE ---
    if is_big.any():
        big_golden = valid_golden[is_big]
        big_result = valid_result[is_big]
        rel_error_big = torch.abs(big_result - big_golden) / (torch.abs(big_golden) + _EPS)
        report.mere = float(rel_error_big.mean().item())
        report.mare = float(rel_error_big.max().item())
        report.mere_big = report.mere
        report.mare_big = report.mare
        report.big_count = int(is_big.sum().item())
        report._rel_error = rel_error_big

    # --- 小值域: 使用绝对误差 ---
    if is_small.any():
        small_abs = abs_error[is_small]
        small_err_mask = small_abs > config.small_atol
        report.small_err_count = int(small_err_mask.sum().item())
        report.small_err_ratio = report.small_err_count / max(report.small_count, 1)

    # --- 构造失败掩码 ---
    fail_mask_big = torch.zeros(valid_count, dtype=torch.bool)
    fail_mask_small = torch.zeros(valid_count, dtype=torch.bool)

    if is_big.any():
        fail_mask_big[is_big] = rel_error_big > mare_threshold

    if is_small.any():
        fail_mask_small[is_small] = abs_error[is_small] > config.small_atol

    fail_mask = fail_mask_big | fail_mask_small
    report.fail_count = int(fail_mask.sum().item())
    report.fail_ratio = report.fail_count / max(report.total_elements, 1)
    report._fail_mask = fail_mask

    # --- 判定 ---
    reasons = []

    if report.mere > mere_threshold:
        reasons.append(f"MERE({report.mere:.2e}) > threshold({mere_threshold:.2e})")

    if report.mare > mare_threshold:
        reasons.append(f"MARE({report.mare:.2e}) > threshold({mare_threshold:.2e})")

    if report.small_err_ratio > 0.01:
        reasons.append(f"Small-value error ratio({report.small_err_ratio:.4f}) > 1%")

    if report.fail_ratio > config.fail_ratio:
        reasons.append(f"Fail ratio({report.fail_ratio:.4f}) > threshold({config.fail_ratio})")

    if report.cosine_similarity < config.cos_sim_min:
        reasons.append(f"Cosine sim({report.cosine_similarity:.6f}) < min({config.cos_sim_min})")

    for reason in report.failure_reasons:
        reasons.append(reason)

    report.failure_reasons = reasons
    report.passed = len(reasons) == 0

    return report


# ============================================================================
# 4. 兼容旧版 API 的 check_result 函数
# ============================================================================

def check_result_v2(
    expect,              # Golden / CPU 参考
    npu_result,          # NPU 输出
    test_name: str = "",
    config: Optional[PrecisionThreshold] = None,
    compute_count: int = 1,
    verbose: bool = True,
) -> Tuple[str, float, float]:
    """
    兼容现有 result_compare_method.check_result 调用惯例。

    Returns:
        (result_str, fulfill_percent, max_relative_error)
          result_str: "Pass" or "Failed"
          fulfill_percent: 通过元素百分比 (0-100)
          max_relative_error: 最大相对误差
    """
    report = compute_precision_report(
        golden=expect,
        result=npu_result,
        test_name=test_name,
        config=config,
        compute_count=compute_count,
    )

    if report.total_elements == 0:
        return "Pass", 100.0, 0.0

    fulfill_percent = (1.0 - report.fail_ratio) * 100.0
    max_rel = report.mare if report.mare > 0 else report.max_abs_error

    if verbose:
        display_precision_report(report)

    result_str = "Pass" if report.passed else "Failed"
    return result_str, fulfill_percent, max_rel


# ============================================================================
# 5. 报告展示
# ============================================================================

def display_precision_report(report: PrecisionReport):
    """格式化打印精度报告"""
    SEP = "=" * 72
    SUB = "-" * 72

    logger.info(SEP)
    logger.info("  昇腾精度标准 2.0 — 精度比对报告")
    if report.test_name:
        logger.info("  Test: %s", report.test_name)
    logger.info(SUB)

    # 基本信息
    logger.info("  %-24s %s", "Dtype:", report.output_dtype)
    logger.info("  %-24s %d", "Total elements:", report.total_elements)
    logger.info("  %-24s %d (large) + %d (small)", "Valid elements:",
                report.normal_count, report.small_count)
    if report.overflow_count > 0:
        logger.info("  %-24s %d  ⚠", "Overflow (Inf/NaN):", report.overflow_count)
    logger.info(SUB)

    # 绝对误差
    logger.info("  [绝对误差]")
    logger.info("    %-22s %.6e", "Max Absolute Error:", report.max_abs_error)
    logger.info("    %-22s %.6e", "Mean Absolute Error:", report.mean_abs_error)
    logger.info("    %-22s %.6e", "RMSE:", report.rmse)

    # 相对误差 (大值域)
    logger.info("  [相对误差 — 大值域 (|x|≥threshold)]")
    logger.info("    %-22s %d elems", "Big-value count:", report.big_count)
    logger.info("    %-22s %.6e  (MERE)", "Mean Rel Error:", report.mere)
    logger.info("    %-22s %.6e  (MARE)", "Max Rel Error:", report.mare)

    # 小值域
    logger.info("  [小值域 (|x|<threshold)]")
    logger.info("    %-22s %d elems", "Small-value count:", report.small_count)
    logger.info("    %-22s %d (%.2f%%)", "Small errors:",
                report.small_err_count, report.small_err_ratio * 100)

    # ULP 统计
    logger.info("  [ULP 误差分布]")
    logger.info("    %-22s %d", "Max ULP:", report.ulp_max)
    logger.info("    %-22s %.2f", "Mean ULP:", report.ulp_mean)
    logger.info("    %-22s %.1f / %.1f / %.1f", "P50/P95/P99:",
                report.ulp_p50, report.ulp_p95, report.ulp_p99)

    # 整体指标
    logger.info("  [整体一致性]")
    logger.info("    %-22s %.8f", "Cosine Similarity:", report.cosine_similarity)
    logger.info("    %-22s %d / %d (%.4f%%)", "Fail elements:",
                report.fail_count, report.total_elements, report.fail_ratio * 100)

    logger.info(SUB)

    # 判定结果
    if report.passed:
        logger.info("  >>> 判定: PASS ✓")
    else:
        logger.info("  >>> 判定: FAIL ✗")
        for i, reason in enumerate(report.failure_reasons, 1):
            logger.info("      [%d] %s", i, reason)

    logger.info(SEP)


# ============================================================================
# 6. 批量比较 + 差分分析
# ============================================================================

def compare_multi_output(
    outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    config: Optional[PrecisionThreshold] = None,
    compute_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, PrecisionReport]:
    """
    对多个输出进行批量精度比较。

    Args:
        outputs: {output_name: (golden, result)} 的字典
        config:  全局精度阈值 (None=自动)
        compute_counts: {output_name: compute_count}, 用于归约容忍

    Returns:
        {output_name: PrecisionReport}
    """
    reports = {}
    for name, (golden, result) in outputs.items():
        cc = compute_counts.get(name, 1) if compute_counts else 1
        reports[name] = compute_precision_report(
            golden=golden, result=result,
            test_name=name, config=config, compute_count=cc
        )
    return reports


def analyze_failure_distribution(
    golden: torch.Tensor,
    result: torch.Tensor,
    report: PrecisionReport,
    top_n: int = 10,
):
    """
    分析失败元素的分布特征 (沿最后一维统计)。

    对 FlashAttention 等多维张量 (B, H, S, D),
    统计各 batch/head/seq 维度的失败率 Top-N,
    辅助排查与 tiling/block 边界相关的精度问题。
    """
    if report.total_elements == 0 or report.fail_count == 0:
        logger.info("  No failures to analyze.")
        return

    golden_f = golden.detach().cpu().flatten().to(torch.float32)
    result_f = result.detach().cpu().flatten().to(torch.float32)

    valid_mask = ~(torch.isinf(golden_f) | torch.isnan(golden_f) |
                   torch.isinf(result_f) | torch.isnan(result_f))

    abs_error = torch.abs(result_f - golden_f) * valid_mask.float()
    rel_error = abs_error / (torch.abs(golden_f) + _EPS) * valid_mask.float()

    # Top-N 最大相对误差位置
    logger.info("  [Top-%d 最大相对误差位置]", top_n)
    top_values, top_indices = torch.topk(rel_error, min(top_n, rel_error.numel()))
    for rank, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist()), 1):
        logger.info("    #%d  idx=%d  golden=%.6e  result=%.6e  abs_err=%.6e  rel_err=%.6e",
                    rank, idx,
                    float(golden_f[idx]), float(result_f[idx]),
                    float(abs_error[idx]), float(val))

    # 按多维拆解 (假设 2-4 维张量)
    shape = golden.shape
    if len(shape) >= 2:
        logger.info("  [按最后两维的失败统计]")
        last_dim = shape[-1]
        fail_2d = (rel_error.reshape(-1, last_dim) > report.mare).float()
        fail_per_row = fail_2d.sum(dim=1)
        top_fail_rows = torch.topk(fail_per_row, min(top_n, fail_per_row.numel()))
        for rank, (row_idx, fail_cnt) in enumerate(
            zip(top_fail_rows.indices.tolist(), top_fail_rows.values.tolist()), 1
        ):
            logger.info("    #%d  row=%d  fail_elements=%d/%d (%.1f%%)",
                        rank, row_idx, int(fail_cnt), last_dim,
                        float(fail_cnt) / last_dim * 100)


# ============================================================================
# 7. CV (Co-Verification) 模式 — 对标 chunk_gated_delta_rule_golden.py
# ============================================================================

@dataclass
class CVThreshold:
    """CV 模式归一化率阈值"""
    max_re_rate: float = 5.0      # 最大相对误差率 (CV_MAX_RE)
    avg_re_rate: float = 1.5      # 平均相对误差率 (CV_AVER_RE)
    rmse_rate: float = 1.5        # 均方根误差率 (CV_RMSE)
    small_val_rate: float = 2.0   # 小值域错误率 (CV_SMALL_VAL)
    err_floor: float = 2.0 ** -8  # 归一化分母最小值 (err_threshold)
    fallback_max_re: float = 2.0 ** -7  # NPU绝对误差低于此值直接Pass


@dataclass
class CVPrecisionReport:
    """CV模式精度比对综合报告"""
    test_name: str = ""
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    # NPU vs Golden (低精度基准) 原始误差
    npu_mere: float = 0.0
    npu_mare: float = 0.0
    npu_rmse: float = 0.0
    npu_small_err_count: int = 0
    npu_small_err_ratio: float = 0.0
    npu_max_abs: float = 0.0

    # Benchmark vs Golden (高精度基准自身精度损失)
    bench_mere: float = 0.0
    bench_mare: float = 0.0
    bench_rmse: float = 0.0
    bench_small_err_count: int = 0
    bench_small_err_ratio: float = 0.0

    # CV 归一化率 (= NPU_err / bench_err)
    mere_rate: float = 0.0
    mare_rate: float = 0.0
    rmse_rate: float = 0.0
    small_err_rate: float = 0.0

    # 误差均衡性
    error_balance: float = 0.0

    # ULP / 余弦 (从 NPU vs Golden 继承)
    cosine_similarity: float = 0.0
    ulp_max: int = 0
    ulp_p50: float = 0.0
    ulp_p95: float = 0.0
    ulp_p99: float = 0.0
    total_elements: int = 0

    # 子报告引用
    npu_detail: Optional[PrecisionReport] = field(default=None, repr=False)
    bench_detail: Optional[PrecisionReport] = field(default=None, repr=False)


def compute_cv_report(
    golden: torch.Tensor,
    benchmark: torch.Tensor,
    actual: torch.Tensor,
    test_name: str = "",
    cv_config: Optional[CVThreshold] = None,
    dtype_config: Optional[PrecisionThreshold] = None,
    compute_count: int = 1,
) -> CVPrecisionReport:
    """
    CV (Co-Verification) 精度比对 — 昇腾精度标准 2.0。

    三层精度参考:
      golden   — 高精度基准 (FP32或FP64输入的计算结果, 如 cgdr_golden 输出)
      benchmark — 同dtype基准 (与NPU相同输入dtype的计算结果, 如 cgdr_benchmark 输出)
      actual   — NPU算子输出

    原理:
      rate = NPU→golden误差 / max(benchmark→golden误差, err_floor)
      将NPU误差归一化到benchmark自身的精度损失上。

    Args:
        golden:    高精度参考结果 (FP32/FP64输入)
        benchmark: 同dtype输入参考结果 (BF16/FP16输入, FP32计算)
        actual:    NPU算子输出
        test_name: 测试名称
        cv_config: CV阈值配置 (None=使用默认)

    Returns:
        CVPrecisionReport
    """
    if cv_config is None:
        cv_config = CVThreshold()

    report = CVPrecisionReport(test_name=test_name)

    # NPU vs Golden 的详细报告
    npu_detail = compute_precision_report(
        golden=golden, result=actual,
        test_name=f"{test_name}_NPU",
        config=dtype_config, compute_count=compute_count,
    )
    report.npu_detail = npu_detail

    # Benchmark vs Golden 的详细报告 (高精度基准自身的精度损失)
    bench_detail = compute_precision_report(
        golden=golden, result=benchmark,
        test_name=f"{test_name}_BENCH",
        config=dtype_config, compute_count=compute_count,
    )
    report.bench_detail = bench_detail

    report.total_elements = npu_detail.total_elements

    # --- 原始误差 ---
    report.npu_mere = npu_detail.mere
    report.npu_mare = npu_detail.mare
    report.npu_rmse = npu_detail.rmse
    report.npu_small_err_count = npu_detail.small_err_count
    report.npu_small_err_ratio = npu_detail.small_err_ratio
    report.npu_max_abs = npu_detail.max_abs_error

    report.bench_mere = bench_detail.mere
    report.bench_mare = bench_detail.mare
    report.bench_rmse = bench_detail.rmse
    report.bench_small_err_count = bench_detail.small_err_count
    report.bench_small_err_ratio = bench_detail.small_err_ratio

    # --- CV 归一化率 ---
    report.mere_rate = report.npu_mere / max(report.bench_mere, cv_config.err_floor)
    report.mare_rate = report.npu_mare / max(report.bench_mare, cv_config.err_floor)
    report.rmse_rate = report.npu_rmse / max(report.bench_rmse, cv_config.err_floor)
    report.small_err_rate = report.npu_small_err_ratio / max(report.bench_small_err_ratio, cv_config.err_floor) \
        if report.bench_small_err_ratio > 0 else report.npu_small_err_ratio / max(cv_config.err_floor, 1e-8)

    # --- ULP / 余弦 ---
    report.cosine_similarity = npu_detail.cosine_similarity
    report.ulp_max = npu_detail.ulp_max
    report.ulp_p50 = npu_detail.ulp_p50
    report.ulp_p95 = npu_detail.ulp_p95
    report.ulp_p99 = npu_detail.ulp_p99

    # --- 误差均衡性 ---
    golden_f32 = golden.detach().cpu().flatten().to(torch.float32)
    actual_f32 = actual.detach().cpu().flatten().to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden_f32), min=1.0)
    report.error_balance = float(torch.mean((actual_f32 - golden_f32) / golden_nmax).item())

    # --- 判定 ---
    reasons = []

    if report.mare_rate >= cv_config.max_re_rate:
        reasons.append(f"MARE_rate({report.mare_rate:.2f}) >= {cv_config.max_re_rate}")
    if report.mere_rate >= cv_config.avg_re_rate:
        reasons.append(f"MERE_rate({report.mere_rate:.2f}) >= {cv_config.avg_re_rate}")
    if report.rmse_rate >= cv_config.rmse_rate:
        reasons.append(f"RMSE_rate({report.rmse_rate:.2f}) >= {cv_config.rmse_rate}")
    if report.small_err_rate >= cv_config.small_val_rate:
        reasons.append(f"Small_err_rate({report.small_err_rate:.2f}) >= {cv_config.small_val_rate}")

    # Fallback: NPU绝对误差本身就很小
    if reasons and report.npu_max_abs < cv_config.fallback_max_re:
        reasons.clear()

    report.failure_reasons = reasons
    report.passed = len(reasons) == 0

    return report


def display_cv_report(report: CVPrecisionReport):
    """格式化打印CV精度报告"""
    SEP = "=" * 72
    SUB = "-" * 72

    logger.info(SEP)
    logger.info("  昇腾精度标准 2.0 — CV (Co-Verification) 精度报告")
    if report.test_name:
        logger.info("  Test: %s", report.test_name)
    logger.info(SEP)
    logger.info("  %-24s %d", "Total elements:", report.total_elements)
    logger.info(SUB)

    # NPU原始误差
    logger.info("  [NPU vs Golden] (算子原始误差)")
    logger.info("    %-22s %.6e  |  benchmark: %.6e", "MARE:",
                report.npu_mare, report.bench_mare)
    logger.info("    %-22s %.6e  |  benchmark: %.6e", "MERE:",
                report.npu_mere, report.bench_mere)
    logger.info("    %-22s %.6e  |  benchmark: %.6e", "RMSE:",
                report.npu_rmse, report.bench_rmse)
    logger.info("    %-22s %.6e  |  benchmark: %.6e", "MaxAbs:",
                report.npu_max_abs, report.npu_max_abs)
    logger.info("    %-22s %.4f  |  benchmark: %.4f", "SmallErr%:",
                report.npu_small_err_ratio, report.bench_small_err_ratio)
    logger.info(SUB)

    # CV归一化率
    logger.info("  [CV 归一化率]  (NPU_err / max(bench_err, 2^-8))")
    logger.info("    %-22s %.2f  (threshold < %.1f)", "MARE_rate:",
                report.mare_rate, CVThreshold.max_re_rate)
    logger.info("    %-22s %.2f  (threshold < %.1f)", "MERE_rate:",
                report.mere_rate, CVThreshold.avg_re_rate)
    logger.info("    %-22s %.2f  (threshold < %.1f)", "RMSE_rate:",
                report.rmse_rate, CVThreshold.rmse_rate)
    logger.info("    %-22s %.2f  (threshold < %.1f)", "SmallErr_rate:",
                report.small_err_rate, CVThreshold.small_val_rate)
    logger.info(SUB)

    # 整体一致性
    logger.info("  [整体一致性]")
    logger.info("    %-22s %.8f", "Cosine Similarity:", report.cosine_similarity)
    logger.info("    %-22s %.6e", "Error Balance:", report.error_balance)
    logger.info("    %-22s %d  (P50:%.0f P95:%.0f P99:%.0f)", "Max ULP:",
                report.ulp_max, report.ulp_p50, report.ulp_p95, report.ulp_p99)
    logger.info(SUB)

    if report.passed:
        logger.info("  >>> 判定: PASS ✓")
    else:
        logger.info("  >>> 判定: FAIL ✗")
        for i, reason in enumerate(report.failure_reasons, 1):
            logger.info("      [%d] %s", i, reason)

    logger.info(SEP)


def check_result_cv(
    golden: torch.Tensor,
    benchmark: torch.Tensor,
    npu_result: torch.Tensor,
    test_name: str = "",
    cv_config: Optional[CVThreshold] = None,
    dtype_config: Optional[PrecisionThreshold] = None,
    mode: str = "auto",
    verbose: bool = True,
) -> Tuple[str, Dict]:
    """
    CV 模式精度比对 — 兼容旧版调用。

    Args:
        golden:     高精度参考 (FP32/FP64 golden)
        benchmark:  同dtype参考 (BF16/FP16 benchmark)
        npu_result: NPU输出
        mode:       "cv" | "simple" | "both" | "auto"
                    auto: 有benchmark时自动用CV, 否则用simple

    Returns:
        (result_str, stats_dict)
    """
    if mode == "auto":
        use_cv = benchmark is not None
    elif mode == "cv":
        use_cv = True
    elif mode == "simple":
        use_cv = False
    elif mode == "both":
        use_cv = True
    else:
        use_cv = benchmark is not None

    if not use_cv:
        result_str, pct, max_err = check_result_v2(
            expect=golden, npu_result=npu_result,
            test_name=test_name, config=dtype_config, verbose=verbose,
        )
        return result_str, {"pass_pct": pct, "max_rel_err": max_err}

    cv_report = compute_cv_report(
        golden=golden, benchmark=benchmark, actual=npu_result,
        test_name=test_name, cv_config=cv_config, dtype_config=dtype_config,
    )

    if verbose:
        display_cv_report(cv_report)

    if mode != "both":
        result_str = "Pass" if cv_report.passed else "Failed"
        return result_str, {
            "mare_rate": cv_report.mare_rate,
            "mere_rate": cv_report.mere_rate,
            "rmse_rate": cv_report.rmse_rate,
            "cosine_sim": cv_report.cosine_similarity,
            "ulp_max": cv_report.ulp_max,
        }

    # both: 同时输出simple和cv两个判定
    simple_result, pct, max_err = check_result_v2(
        expect=golden, npu_result=npu_result,
        test_name=test_name, config=dtype_config, verbose=False,
    )
    return simple_result, {
        "simple_pass_pct": pct,
        "simple_max_rel_err": max_err,
        "cv_mare_rate": cv_report.mare_rate,
        "cv_passed": cv_report.passed,
    }


# ============================================================================
# 8. 旧版兼容通道
# ============================================================================

def check_result(expect, npu_result, benchmark=None):
    """
    完全兼容现有 result_compare_method.py 的 check_result 接口。

    Args:
        expect:     golden 或 benchmark (取决于调用方含义)
        npu_result: NPU输出
        benchmark:  可选的高精度基准, 提供后自动启用CV模式

    兼容旧版两参数调用:
        check_result(golden, npu_result) → simple模式
    新版三参数调用:
        check_result(golden, benchmark, npu_result) → CV模式
    """
    if benchmark is not None:
        result_str, stats = check_result_cv(
            golden=expect, benchmark=benchmark, npu_result=npu_result, verbose=False,
        )
        fulfill_percent = stats.get("mare_rate", 0)
        max_error = stats.get("cosine_sim", 0)
        return result_str, fulfill_percent, max_error
    return check_result_v2(expect=expect, npu_result=npu_result, verbose=False)