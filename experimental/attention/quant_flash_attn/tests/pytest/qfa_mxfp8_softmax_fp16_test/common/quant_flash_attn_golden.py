#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
MxFP8 Softmax FP16 Golden（quant_mode=3 / BNSD / 非 PA）

功能：生成 BNSD fp8 数据（5D scale）→ CPU fp32 softmax golden → NPU 调用 → 精度对比
量化：
  - Q/K: 沿 D 每 64 元素一组 shared exponent（e8m0），descale [B,N,S,D//64,2]
  - V:   沿 S 每 64 元素一组 shared exponent（e8m0），descale [B,N,S//64,D,2]
"""

import logging
import math

import torch
import torch_npu

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn
    _HAS_NPU = True
except ImportError as _e:  # noqa: F841
    _HAS_NPU = False

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 全局参数（test_runner.apply_params 会按 PARAM_MAP 覆盖这些变量）
# ==============================================================================
B = 1
N_q = 1
N_kv = 1
D = 128

CU_SEQLENS_Q = None
CU_SEQLENS_KV = None
SEQUSED_Q = [256]
SEQUSED_KV = [256]
MAX_SEQLEN_Q = -1
MAX_SEQLEN_KV = -1

ENABLE_PA = False
KV_CACHE_LAYOUT = None
BLOCK_SIZE = None
SPARSE_MODE = 0
Q_SCALE_LAYOUT = "BNSD"
P_SCALE = 1.0
SOFTMAX_SCALE = None

DATA_RANGE_Q = 1.0
DATA_RANGE_K = 1.0
DATA_RANGE_V = 1.0
DATA_RANGE_QR = 1.0
DATA_RANGE_KR = 1.0

ENABLE_LSE = False
DEVICE_ID = 0
IS_CONTIGUOUS = True

# ==============================================================================
# 常量
# ==============================================================================
QUANT_MODE = 3  # A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16
INPUT_LAYOUT = "BNSD"
FP8_DTYPE = torch.float8_e4m3fn
QUANT_GROUP_SIZE = 64  # Q/K 沿 D，V 沿 S，每 64 元素一组
E8M0_MIN_POSITIVE = 2 ** (-127)
SEED_Q = 54
SEED_K = 3
SEED_V = 4


def _get_seqused_q():
    return SEQUSED_Q


def _get_seqused_kv():
    return SEQUSED_KV


# ==============================================================================
# e8m0 编解码
# ==============================================================================
def fp32_to_e8m0fnu(tensor_fp32):
    """FP32 -> e8m0fnu：提取 IEEE754 biased exponent（8bit）。"""
    bits = tensor_fp32.float().view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    return biased_exp.view(torch.float8_e8m0fnu)


def e8m0_to_fp32(tensor_e8m0):
    """e8m0fnu -> FP32：2^(biased_exp - 127)。"""
    biased_exp = tensor_e8m0.view(torch.uint8).to(torch.int32)
    return torch.pow(2.0, biased_exp - 127).to(torch.float32)


def sanitize_e8m0_scale(scale, name="scale"):
    """e8m0 无 0 值语义，非有限/0 值替换为最小正数。"""
    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    bad = ~torch.isfinite(result) | (result == 0)
    if int(bad.sum().item()):
        logger.info("[WARN] %s: replace %d bad scale value(s)", name, int(bad.sum().item()))
        result[bad] = E8M0_MIN_POSITIVE
    return result


# ==============================================================================
# 量化 scale（descale 原始 fp32）
# ==============================================================================
def _get_qk_block_scale(tensor):
    """Q/K：沿 D 每 QUANT_GROUP_SIZE 元素一组 shared exponent -> [B,N,S,D//G]。"""
    B_, N_, S_, D_ = tensor.shape
    num_groups = D_ // QUANT_GROUP_SIZE
    grouped = tensor.reshape(B_, N_, S_, num_groups, QUANT_GROUP_SIZE)
    max_vals = torch.max(torch.abs(grouped), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8  # e4m3fn emax=8
    return 2 ** shared_exp  # [B,N,S,G]


def _get_v_block_scale(tensor):
    """V：沿 S 每 QUANT_GROUP_SIZE 元素一组 shared exponent（每组内每 D 一值）-> [B,N,S//G,D]。"""
    B_, N_, S_, D_ = tensor.shape
    num_groups = S_ // QUANT_GROUP_SIZE
    grouped = tensor.reshape(B_, N_, num_groups, QUANT_GROUP_SIZE, D_)
    max_vals = torch.max(torch.abs(grouped), dim=-2)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8
    return 2 ** shared_exp  # [B,N,G,D]


def _pack_scale_5d(scale):
    """scale [..., G] -> [..., G, 2]，尾轴 2 重复（L0 排布最小 2Byte）。"""
    return scale.unsqueeze(-1).expand(*scale.shape, 2).contiguous()


def _quantize_qk(tensor, scale):
    scale_exp = scale.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[..., : tensor.shape[-1]]
    return (tensor / scale_exp).clamp(-448.0, 448.0).to(FP8_DTYPE)


def _quantize_v(tensor, scale):
    # scale [B,N,G,D] 沿 S 扩到 [B,N,S,D]
    scale_exp = scale.repeat_interleave(QUANT_GROUP_SIZE, dim=2)[:, :, : tensor.shape[2], :]
    return (tensor / scale_exp).clamp(-448.0, 448.0).to(FP8_DTYPE)


# ==============================================================================
# 数据生成
# ==============================================================================
def generate_data():
    max_sq = MAX_SEQLEN_Q if (MAX_SEQLEN_Q is not None and MAX_SEQLEN_Q > 0) else max(_get_seqused_q())
    max_skv = MAX_SEQLEN_KV if (MAX_SEQLEN_KV is not None and MAX_SEQLEN_KV > 0) else max(_get_seqused_kv())

    torch.manual_seed(SEED_Q)
    q_fp16 = (torch.rand(B, N_q, max_sq, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_Q
    torch.manual_seed(SEED_K)
    k_fp16 = (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_K
    torch.manual_seed(SEED_V)
    v_fp16 = (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_V

    q_scale = _get_qk_block_scale(q_fp16.float())
    k_scale = _get_qk_block_scale(k_fp16.float())
    v_scale = _get_v_block_scale(v_fp16.float())

    q_fp8 = _quantize_qk(q_fp16.float(), q_scale)
    k_fp8 = _quantize_qk(k_fp16.float(), k_scale)
    v_fp8 = _quantize_v(v_fp16.float(), v_scale)

    # descale: fp32 -> e8m0 -> 5D packing（尾轴 2 重复）
    deq_q = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(q_scale, "q_scale")))
    deq_k = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(k_scale, "k_scale")))
    deq_v = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(v_scale, "v_scale")))

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32)

    logger.info("[DATA] q=%s k=%s v=%s", tuple(q_fp8.shape), tuple(k_fp8.shape), tuple(v_fp8.shape))
    logger.info("[DATA] q_descale=%s k_descale=%s v_descale=%s", tuple(deq_q.shape), tuple(deq_k.shape), tuple(deq_v.shape))

    # 返回 (q, k, v, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table)
    return q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, None, None, None


# ==============================================================================
# CPU Golden（fp32 softmax，反量化按 5D descale）
# ==============================================================================
def cpu_mxfp8_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    softmax_scale=None,
    qr_bf16=None,
    kr_bf16=None,
):
    if actual_seq_q is None:
        actual_seq_q = _get_seqused_q()
    if actual_seq_kv is None:
        actual_seq_kv = _get_seqused_kv()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # 反量化：descale 取尾轴第一个有效值 [..., 0]，e8m0 -> fp32，再扩到逐元素
    q_scale = e8m0_to_fp32(dequant_scale_q[..., 0]).repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[..., : q_fp8.shape[-1]]
    k_scale = e8m0_to_fp32(dequant_scale_k[..., 0]).repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[..., : k_fp8.shape[-1]]
    v_scale = e8m0_to_fp32(v_descale[..., 0]).repeat_interleave(QUANT_GROUP_SIZE, dim=2)[:, :, : v_fp8.shape[2], :]

    q_t = q_fp8.float() * q_scale
    k_t = k_fp8.float() * k_scale
    v_t = v_fp8.float() * v_scale

    # GQA 广播
    if N_q != N_kv:
        g = N_q // N_kv
        k_t = k_t.repeat_interleave(g, dim=1)
        v_t = v_t.repeat_interleave(g, dim=1)

    # QK^T + scale + softmax + PV
    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * softmax_scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_t)

    out = out.to(torch.bfloat16)
    b, n, s, d = out.shape
    lse = torch.logsumexp(scores, dim=-1).to(torch.float32)  # [B,N,S]
    logger.info("[CPU] out=%s", tuple(out.shape))
    return out, lse


# ==============================================================================
# NPU 调用
# ==============================================================================
def npu_mxfp8_fa(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    qr_bf16=None,
    kr_bf16=None,
):
    if not _HAS_NPU:
        raise ImportError(
            "cann_ops_transformer.ops.quant_flash_attn is not available. "
            "Please check that cann_ops_transformer is installed and all .so are compiled."
        )

    seqused_q_t = torch.tensor(seqused_q, dtype=torch.int32).npu() if seqused_q is not None else None
    seqused_kv_t = torch.tensor(seqused_kv, dtype=torch.int32).npu() if seqused_kv is not None else None

    q_npu = q_fp8.npu()
    k_npu = k_fp8.npu()
    v_npu = v_fp8.npu()
    deq_q_npu = dequant_scale_q.npu()
    deq_k_npu = dequant_scale_k.npu()
    deq_v_npu = v_descale.npu()
    p_scale_npu = p_scale.npu()

    ss = 1.0 / math.sqrt(D) if SOFTMAX_SCALE is None else SOFTMAX_SCALE

    torch.npu.synchronize()

    metadata = quant_flash_attn_metadata(
        num_heads_q=N_q,
        num_heads_kv=N_kv,
        head_dim=D,
        quant_mode=QUANT_MODE,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        mask_mode=SPARSE_MODE,
        layout_q=INPUT_LAYOUT,
        layout_q_descale=Q_SCALE_LAYOUT,
        layout_kv=INPUT_LAYOUT,
        layout_out=INPUT_LAYOUT,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )

    atten_out, lse_out = quant_flash_attn(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        q_descale=deq_q_npu,
        k_descale=deq_k_npu,
        v_descale=deq_v_npu,
        quant_mode=QUANT_MODE,
        block_table=None,
        p_scale=p_scale_npu,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        attn_mask=None,
        metadata=metadata,
        softmax_scale=ss,
        mask_mode=SPARSE_MODE,
        layout_q=INPUT_LAYOUT,
        layout_q_descale=Q_SCALE_LAYOUT,
        layout_kv=INPUT_LAYOUT,
        layout_out=INPUT_LAYOUT,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        return_softmax_lse=ENABLE_LSE,
    )
    torch.npu.synchronize()
    return atten_out, lse_out


# ==============================================================================
# layout 转换（BNSD 单模板下基本是原样返回）
# ==============================================================================
def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    if layout == "BNSD":
        return tensor_bnsd
    raise ValueError(f"Unsupported layout: {layout}")


def fill_tnd_padding(tensor_tnd, seq_lens, cu_seqlens, fill_value=float("inf")):
    return tensor_tnd