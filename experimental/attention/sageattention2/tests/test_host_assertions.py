# SageAttention2 宿主机数值断言与仿真对齐测试（M4.1 门禁）
import math
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from sageattention2 import sageattn, sageattn_qk_int8_pv_fp8_asc


def _sdpa_ground_truth(q, k, v, is_causal=False, sm_scale=None):
    head_dim = q.shape[-1]
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=is_causal, scale=scale
    ).to(q.dtype)


def test_assertion_kappa_two_tiers():
    """断言 1: κ 双档位（2.25 vs 448.0）数值差异与 P 域约束。"""
    torch.manual_seed(101)
    b, hq, sq, d = 1, 4, 128, 64
    q = torch.randn(b, hq, sq, d, dtype=torch.float16)
    k = torch.randn(b, hq, sq, d, dtype=torch.float16)
    v = torch.randn(b, hq, sq, d, dtype=torch.float16)

    out_fp16_accum = sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32+fp16")
    out_fp32_accum = sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32")

    diff = (out_fp16_accum - out_fp32_accum).abs().max().item()
    assert diff > 1e-4, f"kappa=2.25 vs kappa=448 must produce non-zero difference, got {diff}"


def test_assertion_granularity_non_zero_difference():
    """断言 2: per_warp 与 per_thread 分段非零差异（严禁空桩违约）。"""
    torch.manual_seed(202)
    b, hq, sq, d = 1, 4, 128, 128
    # 构造通道方差不均匀的输入以放大分段 scale 的差异
    weights = torch.linspace(0.1, 10.0, d, dtype=torch.float16)
    q = torch.randn(b, hq, sq, d, dtype=torch.float16) * weights
    k = torch.randn(b, hq, sq, d, dtype=torch.float16) * weights
    v = torch.randn(b, hq, sq, d, dtype=torch.float16)

    out_warp = sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran="per_warp")
    out_thread = sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran="per_thread")

    diff = (out_warp - out_thread).abs().max().item()
    assert diff > 1e-3, f"per_warp and per_thread must yield distinct results, got diff={diff}"


def test_assertion_lse_contract():
    """断言 3: LSE 核内闭环数学契约（DESIGN_SPEC §4.6 修正口径：lse = ln(denom) + m' + s*corr，不加 lnκ）。"""
    torch.manual_seed(303)
    b, hq, sq, d = 1, 2, 64, 64
    sm_scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, hq, sq, d, dtype=torch.float16)
    k = torch.randn(b, hq, sq, d, dtype=torch.float16)
    v = torch.randn(b, hq, sq, d, dtype=torch.float16)

    out, lse = sageattn(q, k, v, sm_scale=sm_scale, return_lse=True, smooth_k=True)

    # 理论 ground truth: 全量浮点计算的 logsumexp(QK^T * sm_scale)
    score_gt = torch.einsum("bhsd,bhkd->bhsk", q.float(), k.float()) * sm_scale
    lse_gt = torch.logsumexp(score_gt, dim=-1)

    # 验证 LSE 紧密对齐理论真值（低比特 QK 量化下平均误差约 0.003，最大误差 < 0.02）
    lse_diff = (lse - lse_gt).abs().mean().item()
    lse_max = (lse - lse_gt).abs().max().item()
    assert lse_diff < 0.02, f"LSE mean error {lse_diff} exceeds tight tolerance 0.02"
    assert lse_max < 0.05, f"LSE max error {lse_max} exceeds tight tolerance 0.05"


@pytest.mark.parametrize("case_cfg", [
    {"name": "P-01", "b": 1, "hq": 4, "hk": 4, "sq": 128, "sk": 128, "d": 128, "dtype": torch.float16, "causal": False},
    {"name": "P-02", "b": 1, "hq": 4, "hk": 4, "sq": 256, "sk": 256, "d": 128, "dtype": torch.float16, "causal": False},
    {"name": "P-03", "b": 1, "hq": 4, "hk": 4, "sq": 256, "sk": 256, "d": 128, "dtype": torch.float16, "causal": True},
    {"name": "P-04", "b": 2, "hq": 4, "hk": 4, "sq": 128, "sk": 128, "d": 128, "dtype": torch.float16, "causal": False},
    {"name": "P-05_GQA", "b": 1, "hq": 8, "hk": 2, "sq": 128, "sk": 128, "d": 128, "dtype": torch.float16, "causal": False},
    {"name": "P-06_BF16", "b": 1, "hq": 4, "hk": 4, "sq": 128, "sk": 128, "d": 128, "dtype": torch.bfloat16, "causal": False},
    {"name": "P-07_Long", "b": 1, "hq": 2, "hk": 2, "sq": 512, "sk": 512, "d": 128, "dtype": torch.float16, "causal": False},
])
def test_assertion_lb_sdpa_matrix(case_cfg):
    """断言 4: P-01 ~ P-07 对应规格以 FP32 SDPA 为基准（L-B 门禁要求：max≤2, mean≤1.2, RMSE≤1.2）。"""
    torch.manual_seed(404)
    b, hq, hk, sq, sk, d = case_cfg["b"], case_cfg["hq"], case_cfg["hk"], case_cfg["sq"], case_cfg["sk"], case_cfg["d"]
    dtype = case_cfg["dtype"]
    causal = case_cfg["causal"]

    q = torch.randn(b, hq, sq, d, dtype=dtype)
    k = torch.randn(b, hk, sk, d, dtype=dtype)
    v = torch.randn(b, hk, sk, d, dtype=dtype)

    out = sageattn(q, k, v, is_causal=causal)

    # 针对 GQA 广播 K/V
    if hq != hk:
        group = hq // hk
        k_ref = k.repeat_interleave(group, dim=1)
        v_ref = v.repeat_interleave(group, dim=1)
    else:
        k_ref, v_ref = k, v

    out_sdpa = _sdpa_ground_truth(q, k_ref, v_ref, is_causal=causal)

    diff = (out.float() - out_sdpa.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    matched_ratio = (diff <= 0.1).float().mean().item()

    assert max_err <= 2.0, f"[{case_cfg['name']}] max_err {max_err:.4f} > 2.0"
    assert mean_err <= 1.2, f"[{case_cfg['name']}] mean_err {mean_err:.4f} > 1.2"
    assert rmse <= 1.2, f"[{case_cfg['name']}] rmse {rmse:.4f} > 1.2"
    assert matched_ratio >= 0.95, f"[{case_cfg['name']}] matched_ratio {matched_ratio:.4f} < 0.95"
