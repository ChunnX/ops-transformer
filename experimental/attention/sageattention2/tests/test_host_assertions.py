# SageAttention2 宿主机数值断言与仿真对齐测试（M4.1 门禁）
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from sageattention2 import create_qkv, sageattn, sageattn_qk_int8_pv_fp8_asc


def _sdpa_ground_truth(q, k, v, is_causal=False, sm_scale=None):
    head_dim = q.shape[-1]
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=is_causal, scale=scale
    ).to(q.dtype)


def test_assertion_kappa_two_tiers():
    """断言 1: κ 双档位（2.25 vs 448.0）数值差异与 P 域约束。"""
    torch.manual_seed(101)
    q, k, v = create_qkv(b=1, hq=4, sq=128, d=64)

    out_fp16_accum = sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32+fp16")
    out_fp32_accum = sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32")

    diff = (out_fp16_accum - out_fp32_accum).abs().max().item()
    assert diff > 1e-4, f"kappa=2.25 vs kappa=448 must produce non-zero difference, got {diff}"


def test_assertion_granularity_non_zero_difference():
    """断言 2: per_warp 与 per_thread 分段非零差异（严禁空桩违约）。"""
    torch.manual_seed(202)
    d = 128
    # 构造通道方差不均匀的输入以放大分段 scale 的差异
    weights = torch.linspace(0.1, 10.0, d, dtype=torch.float16)
    q, k, v = create_qkv(b=1, hq=4, sq=128, d=d)
    q = q * weights
    k = k * weights

    out_warp = sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran="per_warp")
    out_thread = sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran="per_thread")

    diff = (out_warp - out_thread).abs().max().item()
    assert diff > 1e-3, f"per_warp and per_thread must yield distinct results, got diff={diff}"


def test_assertion_lse_contract():
    """断言 3: LSE 核内闭环数学契约（DESIGN_SPEC §4.6 修正口径：lse = ln(denom) + m' + s*corr，不加 lnκ）。"""
    torch.manual_seed(303)
    d = 64
    sm_scale = 1.0 / math.sqrt(d)
    q, k, v = create_qkv(b=1, hq=2, sq=64, d=d)


    _out, lse = sageattn(q, k, v, sm_scale=sm_scale, return_lse=True, smooth_k=True)

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
    causal = case_cfg["causal"]
    q, k, v = create_qkv(**case_cfg)

    out = sageattn(q, k, v, is_causal=causal)

    # 针对 GQA 广播 K/V
    if case_cfg["hq"] != case_cfg["hk"]:
        group = case_cfg["hq"] // case_cfg["hk"]
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


def test_assertion_deterministic_repeatability():
    """断言 5: 确定性逐位一致（D-R8 确定性单写者树设计，无原子加随机竞争）。"""
    torch.manual_seed(505)
    q, k, v = create_qkv(b=2, hq=4, hk=2, sq=128, sk=128, d=64)

    # 1. sageattn 默认分发两次执行逐位一致
    out1, lse1 = sageattn(q, k, v, return_lse=True)
    out2, lse2 = sageattn(q, k, v, return_lse=True)
    assert torch.equal(out1, out2), "out must be bitwise identical across calls"
    assert torch.equal(lse1, lse2), "lse must be bitwise identical across calls"

    # 2. per_thread 显式入口两次执行逐位一致
    out_pt1, lse_pt1 = sageattn_qk_int8_pv_fp8_asc(
        q, k, v, qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp16", return_lse=True
    )
    out_pt2, lse_pt2 = sageattn_qk_int8_pv_fp8_asc(
        q, k, v, qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp16", return_lse=True
    )
    assert torch.equal(out_pt1, out_pt2), "per_thread out must be bitwise identical across calls"
    assert torch.equal(lse_pt1, lse_pt2), "per_thread lse must be bitwise identical across calls"


def test_assertion_torch_compile_eager_and_fullgraph():
    """断言 6: PyTorch torch.compile(eager, fullgraph=True) 整图编译与无图中断验证。"""
    torch.manual_seed(606)
    q, k, v = create_qkv(b=1, hq=4, sq=64, d=64)

    compiled_fn = torch.compile(sageattn, backend="eager", fullgraph=True)
    out_compiled = compiled_fn(q, k, v)
    out_eager = sageattn(q, k, v)


    assert torch.equal(out_compiled, out_eager), "torch.compile output must match eager output"


def test_assertion_meta_tensor_shape_inference():
    """断言 7: Meta / FakeTensor 设备上的张量形状推导契约验证。"""
    b, hq, hk, sq, sk, d = 1, 8, 2, 256, 256, 128
    qm = torch.empty(b, hq, sq, d, device="meta", dtype=torch.float16)
    km = torch.empty(b, hk, sk, d, device="meta", dtype=torch.float16)
    vm = torch.empty(b, hk, sk, d, device="meta", dtype=torch.float16)

    out_meta, lse_meta = sageattn(qm, km, vm, return_lse=True)
    assert out_meta.device.type == "meta"
    assert out_meta.shape == (b, hq, sq, d)
    assert lse_meta.device.type == "meta"
    assert lse_meta.shape == (b, hq, sq)
