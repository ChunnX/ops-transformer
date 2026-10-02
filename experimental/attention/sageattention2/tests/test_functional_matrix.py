# SageAttention2 功能验收测试矩阵（TC-01 ~ TC-10 全覆盖）
import math
import sys
from pathlib import Path
import warnings
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from sageattention2 import sageattn, sageattn_qk_int8_pv_fp8_asc, replace_sdpa
from sageattention2.reference import sageattn_reference


def _create_tensors(b=2, hq=4, hk=4, sq=64, sk=64, d=64, dtype=torch.float16, layout="HND"):
    if layout == "HND":
        q = torch.randn(b, hq, sq, d, dtype=dtype)
        k = torch.randn(b, hk, sk, d, dtype=dtype)
        v = torch.randn(b, hk, sk, d, dtype=dtype)
    else:  # NHD
        q = torch.randn(b, sq, hq, d, dtype=dtype)
        k = torch.randn(b, sk, hk, d, dtype=dtype)
        v = torch.randn(b, sk, hk, d, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tc01_dispatch_and_layout(dtype):
    """TC-01: 分发入口与 HND / NHD 布局验证（覆盖 FP16 与 BF16）。"""
    # HND 布局
    q_hnd, k_hnd, v_hnd = _create_tensors(dtype=dtype, layout="HND")
    out_hnd = sageattn(q_hnd, k_hnd, v_hnd, tensor_layout="HND")
    assert out_hnd.shape == q_hnd.shape
    assert out_hnd.dtype == dtype
    assert torch.all(torch.isfinite(out_hnd))

    # NHD 布局
    q_nhd, k_nhd, v_nhd = _create_tensors(dtype=dtype, layout="NHD")
    out_nhd = sageattn(q_nhd, k_nhd, v_nhd, tensor_layout="NHD")
    assert out_nhd.shape == q_nhd.shape
    assert out_nhd.dtype == dtype
    assert torch.all(torch.isfinite(out_nhd))

    # NHD 与 HND 数值等价性验证
    q_nhd_from_hnd = q_hnd.permute(0, 2, 1, 3).contiguous()
    k_nhd_from_hnd = k_hnd.permute(0, 2, 1, 3).contiguous()
    v_nhd_from_hnd = v_hnd.permute(0, 2, 1, 3).contiguous()
    out_nhd2 = sageattn(q_nhd_from_hnd, k_nhd_from_hnd, v_nhd_from_hnd, tensor_layout="NHD")
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(out_hnd, out_nhd2.permute(0, 2, 1, 3), atol=tol, rtol=tol)


@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("accum", ["fp32", "fp32+fp32", "fp32+fp16"])
def test_tc02_gran_accum_matrix(gran, accum):
    """TC-02: gran × accum 全组合（2 × 3 = 6 档）。"""
    q, k, v = _create_tensors(b=1, hq=2, hk=2, sq=64, sk=64, d=64)
    out = sageattn_qk_int8_pv_fp8_asc(
        q, k, v,
        qk_quant_gran=gran,
        pv_accum_dtype=accum,
    )
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.all(torch.isfinite(out))


def test_tc03_smooth_and_warning():
    """TC-03: smooth 开关与 warning 触发机制。"""
    q, k, v = _create_tensors(b=1, hq=2, hk=2, sq=64, sk=64, d=64)

    # smooth_k=False 应正常执行
    out_no_sk = sageattn(q, k, v, smooth_k=False)
    assert out_no_sk.shape == q.shape
    assert torch.all(torch.isfinite(out_no_sk))

    # smooth_v=True 在 fp32+fp16 档须触发 UserWarning 并忽略
    with pytest.warns(UserWarning, match="smooth_v is only supported with pv_accum_dtype='fp32'"):
        out_warn = sageattn(q, k, v, pv_accum_dtype="fp32+fp16", smooth_v=True)
    assert out_warn.shape == q.shape

    # smooth_v=True 在 fp32+fp32 档亦须触发 UserWarning 并忽略
    with pytest.warns(UserWarning, match="smooth_v is only supported with pv_accum_dtype='fp32'"):
        out_warn32 = sageattn(q, k, v, pv_accum_dtype="fp32+fp32", smooth_v=True)
    assert out_warn32.shape == q.shape

    # smooth_v=True 在 fp32 档不应触发 warning 且正常执行
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out_smooth_v = sageattn(q, k, v, pv_accum_dtype="fp32", smooth_v=True)
    assert out_smooth_v.shape == q.shape
    assert torch.all(torch.isfinite(out_smooth_v))


def test_tc04_return_lse():
    """TC-04: return_lse 契约验证。"""
    b, hq, sq, d = 2, 4, 64, 64
    q, k, v = _create_tensors(b=b, hq=hq, hk=hq, sq=sq, sk=sq, d=d)

    # return_lse=False 返回单张量
    out_only = sageattn(q, k, v, return_lse=False)
    assert isinstance(out_only, torch.Tensor)

    # return_lse=True 返回 (out, lse)
    out, lse = sageattn(q, k, v, return_lse=True)
    assert out.shape == (b, hq, sq, d)
    assert lse.shape == (b, hq, sq)
    assert lse.dtype == torch.float32
    assert torch.all(torch.isfinite(lse))

    # NHD 布局下的 LSE 形状
    q_nhd, k_nhd, v_nhd = _create_tensors(b=b, hq=hq, hk=hq, sq=sq, sk=sq, d=d, layout="NHD")
    out_nhd, lse_nhd = sageattn(q_nhd, k_nhd, v_nhd, tensor_layout="NHD", return_lse=True)
    assert out_nhd.shape == (b, sq, hq, d)
    assert lse_nhd.shape == (b, hq, sq)
    assert lse_nhd.dtype == torch.float32


@pytest.mark.parametrize("d_og, d_pad", [(48, 64), (96, 128)])
def test_tc05_head_dim_pad_and_crop(d_og, d_pad):
    """TC-05: head_dim pad（48→64、96→128）与输出裁剪回原 D；数值化补「仅形状」缺口（todo §四）。"""
    torch.manual_seed(505)
    q, k, v = _create_tensors(b=1, hq=2, hk=2, sq=64, sk=64, d=d_og)
    out = sageattn(q, k, v)
    assert out.shape == (1, 2, 64, d_og)
    assert out.dtype == q.dtype
    assert torch.all(torch.isfinite(out))

    # 数值口径 1（独立 oracle）：pad 路径输出以 FP32 SDPA 为基准
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=1.0 / math.sqrt(d_og))
    diff = (out.float() - out_sdpa).abs()
    assert diff.max().item() <= 0.2, f"max_err {diff.max().item():.4f} > 0.2 (d={d_og})"
    assert diff.mean().item() <= 0.05, f"mean_err {diff.mean().item():.4f} > 0.05 (d={d_og})"
    rmse = torch.sqrt((diff ** 2).mean()).item()
    assert rmse <= 0.05, f"rmse {rmse:.4f} > 0.05 (d={d_og})"
    matched = (diff <= 0.1).float().mean().item()
    assert matched >= 0.99, f"matched_ratio {matched:.4f} < 0.99 (d={d_og})"

    # 数值口径 2（todo 原文「D=48 输出 vs D=64 直接计算一致性」）：同一输入零 pad 到 d_pad 直接
    # 计算（sm_scale 沿用 1/sqrt(d_og)，与内核 pad 路径口径一致），裁回 d_og 后须逐位一致
    direct = sageattn_reference(
        torch.nn.functional.pad(q, (0, d_pad - d_og)),
        torch.nn.functional.pad(k, (0, d_pad - d_og)),
        torch.nn.functional.pad(v, (0, d_pad - d_og)),
        sm_scale=1.0 / math.sqrt(d_og))
    assert torch.equal(out, direct[..., :d_og]), \
        f"pad-to-{d_pad} direct computation must match native d={d_og} output bitwise after crop"

    # 数值精度对齐断言与数学零贡献性证明：
    # 构造原 D 输入经零填充至 d_pad 的基准张量，固定与原 D 相同的 sm_scale
    sm_scale = 1.0 / math.sqrt(d_og)
    q_pad = torch.nn.functional.pad(q, (0, d_pad - d_og))
    k_pad = torch.nn.functional.pad(k, (0, d_pad - d_og))
    v_pad = torch.nn.functional.pad(v, (0, d_pad - d_og))
    out_pad = sageattn(q_pad, k_pad, v_pad, sm_scale=sm_scale)

    # 1. 断言 D 裁切输出与前 d_og 维直接计算结果逐元素等价
    assert torch.equal(out, out_pad[..., :d_og])
    # 2. 断言填充尾部为严格全零，证明零填充在 INT8/FP8 下的数学零贡献性
    assert torch.all(out_pad[..., d_og:] == 0)


def test_tc06_gqa_and_unequal_seq():
    """TC-06: GQA（Hq % Hk == 0）与非对称序列长度（Sq != Sk）。"""
    # GQA: Hq=8, Hk=2 (group=4), Sq=64, Sk=128
    b, hq, hk, sq, sk, d = 2, 8, 2, 64, 128, 64
    q, k, v = _create_tensors(b=b, hq=hq, hk=hk, sq=sq, sk=sk, d=d)
    out = sageattn(q, k, v)
    assert out.shape == (b, hq, sq, d)
    assert torch.all(torch.isfinite(out))


def test_tc07_causal():
    """TC-07: 因果遮蔽（is_causal=True）及屏蔽未来信息校验。"""
    b, hq, sq, d = 1, 2, 64, 64
    q, k, v = _create_tensors(b=b, hq=hq, hk=hq, sq=sq, sk=sq, d=d)
    out_causal = sageattn(q, k, v, is_causal=True)
    assert out_causal.shape == (b, hq, sq, d)
    assert torch.all(torch.isfinite(out_causal))

    # 因果性数值不变性验证：
    # 1. 在 smooth_k=False 时，修改 k 和 v 的后半部分，q 前半部分的输出严格不变
    out_no_sk = sageattn(q, k, v, is_causal=True, smooth_k=False)
    k_modified = k.clone()
    v_modified = v.clone()
    k_modified[:, :, 32:, :] += torch.randn_like(k_modified[:, :, 32:, :])
    v_modified[:, :, 32:, :] += torch.randn_like(v_modified[:, :, 32:, :])
    out_no_sk_mod = sageattn(q, k_modified, v_modified, is_causal=True, smooth_k=False)
    assert torch.equal(out_no_sk[:, :, :32, :], out_no_sk_mod[:, :, :32, :])

    # 2. 在 smooth_k=True 时（默认），修改 v 的后半部分，q 前半部分的输出严格不变
    v_only_mod = v.clone()
    v_only_mod[:, :, 32:, :] += torch.randn_like(v_only_mod[:, :, 32:, :])
    out_causal_v_mod = sageattn(q, k, v_only_mod, is_causal=True, smooth_k=True)
    assert torch.equal(out_causal[:, :, :32, :], out_causal_v_mod[:, :, :32, :])


def test_tc08_illegal_inputs():
    """TC-08: 非法参数与越界输入校验（严格抛出 ValueError）。"""
    q, k, v = _create_tensors(b=1, hq=4, hk=4, sq=64, sk=64, d=64)

    # 1. head_dim > 128
    q_bad_d = torch.randn(1, 4, 64, 160, dtype=torch.float16)
    k_bad_d = torch.randn(1, 4, 64, 160, dtype=torch.float16)
    v_bad_d = torch.randn(1, 4, 64, 160, dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        sageattn(q_bad_d, k_bad_d, v_bad_d)

    # 2. Hq % Hk != 0
    q_bad_h = torch.randn(1, 5, 64, 64, dtype=torch.float16)
    k_bad_h = torch.randn(1, 3, 64, 64, dtype=torch.float16)
    v_bad_h = torch.randn(1, 3, 64, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="divisible"):
        sageattn(q_bad_h, k_bad_h, v_bad_h)

    # 3. is_causal 时 Sq != Sk
    q_diff_s = torch.randn(1, 4, 64, 64, dtype=torch.float16)
    k_diff_s = torch.randn(1, 4, 128, 64, dtype=torch.float16)
    v_diff_s = torch.randn(1, 4, 128, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="is_causal is valid only when qo_len equals kv_len"):
        sageattn(q_diff_s, k_diff_s, v_diff_s, is_causal=True)

    # 4. attn_mask 非 None 报错
    with pytest.raises(ValueError, match="attn_mask"):
        sageattn(q, k, v, attn_mask=torch.zeros(64, 64))

    # 5. tensor_layout 非法（如 "XYZ"、"INVALID"）
    with pytest.raises(ValueError, match="tensor_layout"):
        sageattn(q, k, v, tensor_layout="XYZ")
    with pytest.raises(ValueError, match="tensor_layout"):
        sageattn(q, k, v, tensor_layout="INVALID")

    # 6. dtype 不匹配或不支持（如 float32）
    q_f32 = q.float()
    with pytest.raises(ValueError, match="torch.float16 or torch.bfloat16"):
        sageattn(q_f32, k.float(), v.float())

    # 7. q.dtype != k.dtype 或 q.dtype != v.dtype 混用报错
    k_bf16 = k.to(torch.bfloat16)
    with pytest.raises(ValueError, match="q/k/v must share the same dtype"):
        sageattn(q, k_bf16, v)
    v_bf16 = v.to(torch.bfloat16)
    with pytest.raises(ValueError, match="q/k/v must share the same dtype"):
        sageattn(q, k, v_bf16)

    # 8. 非连续末维 (stride(-1) != 1) 报错（分别对 q, k, v 注入非连续末维）
    q_non_contig = torch.randn(1, 4, 64, 128, dtype=torch.float16)[:, :, :, ::2]
    with pytest.raises(ValueError, match="q must have stride\\(-1\\) == 1"):
        sageattn(q_non_contig, k, v)
    k_non_contig = torch.randn(1, 4, 64, 128, dtype=torch.float16)[:, :, :, ::2]
    with pytest.raises(ValueError, match="k must have stride\\(-1\\) == 1"):
        sageattn(q, k_non_contig, v)
    v_non_contig = torch.randn(1, 4, 64, 128, dtype=torch.float16)[:, :, :, ::2]
    with pytest.raises(ValueError, match="v must have stride\\(-1\\) == 1"):
        sageattn(q, k, v_non_contig)

    # 9. 非法量化粒度 qk_quant_gran 报错（如 "per_token"、"per_channel"）
    with pytest.raises(ValueError, match="qk_quant_gran"):
        sageattn(q, k, v, qk_quant_gran="per_token")
    with pytest.raises(ValueError, match="qk_quant_gran"):
        sageattn(q, k, v, qk_quant_gran="per_channel")

    # 10. 非法累加档位 pv_accum_dtype 报错（如 "int8"、"fp16"）
    with pytest.raises(ValueError, match="pv_accum_dtype"):
        sageattn(q, k, v, pv_accum_dtype="int8")
    with pytest.raises(ValueError, match="pv_accum_dtype"):
        sageattn(q, k, v, pv_accum_dtype="fp16")


def test_tc08_illegal_inputs_extended():
    """TC-08 补充: _ops 校验分支中上述用例未覆盖的非法输入（shape 族 / sm_scale 越界）。"""
    q, k, v = _create_tensors(b=1, hq=4, hk=4, sq=64, sk=64, d=64)

    # 1. 非 4D 输入（3D）
    with pytest.raises(ValueError, match="4D"):
        sageattn(q[0], k[0], v[0])

    # 2. batch 不一致
    q_b2 = torch.randn(2, 4, 64, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="batch"):
        sageattn(q_b2, k, v)

    # 3. q 与 k 的 head_dim 不一致
    k_d128 = torch.randn(1, 4, 64, 128, dtype=torch.float16)
    v_d128 = torch.randn(1, 4, 64, 128, dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        sageattn(q, k_d128, v_d128)

    # 4. k/v shape 不一致（同 b/h/d 但序列长不同）
    v_s32 = torch.randn(1, 4, 32, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="k and v must share the same shape"):
        sageattn(q, k, v_s32)

    # 5. sm_scale 非有限（inf / NaN）
    with pytest.raises(ValueError, match="sm_scale"):
        sageattn(q, k, v, sm_scale=float("inf"))
    with pytest.raises(ValueError, match="sm_scale"):
        sageattn(q, k, v, sm_scale=float("nan"))


def test_tc09_scaled_sequences():
    """TC-09: 序列长度测试（含非对称及 S=8192 长序列真实校验）。"""
    # 1. 非对称序列长
    q, k, v = _create_tensors(b=1, hq=2, hk=2, sq=256, sk=512, d=64)
    out = sageattn(q, k, v)
    assert out.shape == (1, 2, 256, 64)
    assert torch.all(torch.isfinite(out))

    # 2. 8K 长序列真实分发测试（S >= 8K 验收用例契约）
    q_8k, k_8k, v_8k = _create_tensors(b=1, hq=1, hk=1, sq=8192, sk=8192, d=64)
    out_8k = sageattn(q_8k, k_8k, v_8k)
    assert out_8k.shape == (1, 1, 8192, 64)
    assert torch.all(torch.isfinite(out_8k))


def test_tc10_replace_sdpa_smoke():
    """TC-10: replace_sdpa 上下文管理器冒烟与恢复验证。"""
    q, k, v = _create_tensors(b=1, hq=2, hk=2, sq=64, sk=64, d=64)

    orig_sdpa = torch.nn.functional.scaled_dot_product_attention

    with replace_sdpa():
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert out.shape == q.shape
        assert torch.all(torch.isfinite(out))

        # dropout 不支持报错
        with pytest.raises(ValueError, match="dropout"):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.5)

        # attn_mask 不支持报错
        with pytest.raises(ValueError, match="attn_mask"):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=torch.zeros(64, 64))

    # 退出上下文后恢复原始 SDPA
    assert torch.nn.functional.scaled_dot_product_attention is orig_sdpa


def test_tc10_replace_sdpa_params_and_overrides():
    """TC-10 增强: 验证 replace_sdpa 上下文传递 scale/sm_scale/is_causal 无冲突且生效。"""
    q, k, v = _create_tensors(b=1, hq=2, hk=2, sq=64, sk=64, d=64)

    # 1. replace_sdpa 传递 sm_scale 与 scale
    with replace_sdpa(sm_scale=0.05):
        out_sm = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert out_sm.shape == q.shape

    with replace_sdpa(scale=0.05):
        out_sc = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(out_sm, out_sc, atol=1e-3, rtol=1e-3)

    # 2. 调用方显式 scale 覆盖上下文默认值
    with replace_sdpa(scale=0.05):
        out_override = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.1)
        out_direct = sageattn(q, k, v, sm_scale=0.1)
        assert torch.allclose(out_override, out_direct, atol=1e-3, rtol=1e-3)

    # 3. replace_sdpa 传递 is_causal
    with replace_sdpa(is_causal=True):
        out_causal = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out_causal_direct = sageattn(q, k, v, is_causal=True)
        assert torch.equal(out_causal, out_causal_direct)

