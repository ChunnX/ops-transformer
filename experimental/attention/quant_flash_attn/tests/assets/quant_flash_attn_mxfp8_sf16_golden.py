#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
# PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MxFP8 Softmax FP16 Golden（quant_mode=3 / BNSD / 非 PA）—— ttk assets 版。

与 pytest 版（tests/pytest/qfa_mxfp8_softmax_fp16_test/common/quant_flash_attn_golden.py）
同源数学：MXFP8 组量化（Q/K 沿 D 每 64 一组、V 沿 S ceil 组）→ CPU fp16 softmax
管线逐位仿真（mm1Res fp16 + (1,8,10) deqScalar 截断 + k 域整数链 + pscale e8m0
往返 + 空批 -inf）。

ttk 适配差异：
  - 参数来自 CSV attributes dict（_apply_golden_globals），seqused 为 *_values 列表、
    data_range 为 "min,max" 字符串
  - generate_data 返回 dict（15 slot 命名与参考 assets 对齐），供 inputs/golden 共用
  - _inject_physical_s_override：ttk 按 CSV tensor_view_shapes 预分配张量，物理 S 维
    以实际张量 shape 为唯一真源（防 CSV/golden 公式漂移）
"""

import logging
import math

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 全局参数（_apply_golden_globals 按 CSV attributes 覆盖）
# ==============================================================================
B = 1
N_q = 1
N_kv = 1
D = 128

SEQUSED_Q = None
SEQUSED_KV = None
MAX_SEQLEN_Q = -1
MAX_SEQLEN_KV = -1

P_SCALE = 1.0
SOFTMAX_SCALE = None
ENABLE_LSE = False

# 数据幅值：value = offset + (rand*2-1)*range，即均匀 [offset-range, offset+range]
# （CSV data_range_* "min,max" → offset=(min+max)/2, range=(max-min)/2）
DATA_RANGE_Q = 1.0
DATA_RANGE_K = 1.0
DATA_RANGE_V = 1.0
DATA_OFFSET_Q = 0.0
DATA_OFFSET_K = 0.0
DATA_OFFSET_V = 0.0

QUANT_MODE = 3  # A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16
INPUT_LAYOUT = "BNSD"
FP8_DTYPE = torch.float8_e4m3fn
QUANT_GROUP_SIZE = 64  # Q/K 沿 D，V 沿 S，每 64 元素一组
E8M0_MIN_POSITIVE = 2 ** (-127)
SEED_Q = 54
SEED_K = 3
SEED_V = 4

# 物理 S 覆盖（ttk 张量实际 shape，None = 未注入）
_SQ_PHYSICAL = None
_SKV_PHYSICAL = None


# ==============================================================================
# ttk attributes → 全局参数
# ==============================================================================
def _parse_data_range(value):
    """CSV data_range（"min,max" 字符串或标量）→ (offset, range)。"""
    if value is None:
        return 0.0, 1.0
    if isinstance(value, str):
        parts = [float(x) for x in value.split(",") if x.strip() != ""]
        if len(parts) == 1:
            return 0.0, abs(parts[0])
        lo, hi = parts[0], parts[-1]
        return (lo + hi) / 2.0, (hi - lo) / 2.0
    return 0.0, float(value)


def _seq_list(values):
    """CSV *_values 列表（[] 或 None → None，即满 shape 无 varlen）。"""
    if values is None:
        return None
    if isinstance(values, str):
        values = values.strip()
        if not values or values in ("[]", "None"):
            return None
        import ast

        values = ast.literal_eval(values)
    values = list(values)
    return values or None


def _apply_golden_globals(attrs):
    """CSV attributes dict → golden 模块全局（数据幅值/长度/标量）。"""
    global B, N_q, N_kv, D
    global SEQUSED_Q, SEQUSED_KV, MAX_SEQLEN_Q, MAX_SEQLEN_KV
    global P_SCALE, SOFTMAX_SCALE, ENABLE_LSE
    global DATA_RANGE_Q, DATA_RANGE_K, DATA_RANGE_V
    global DATA_OFFSET_Q, DATA_OFFSET_K, DATA_OFFSET_V

    B = int(attrs.get("B", B))
    N_q = int(attrs.get("N_q", N_q))
    N_kv = int(attrs.get("N_kv", attrs.get("N", N_kv)))
    D = int(attrs.get("D", D))

    SEQUSED_Q = _seq_list(attrs.get("seqused_q_values"))
    SEQUSED_KV = _seq_list(attrs.get("seqused_kv_values"))
    MAX_SEQLEN_Q = int(attrs.get("max_seqlen_q", -1))
    MAX_SEQLEN_KV = int(attrs.get("max_seqlen_kv", -1))

    p_scale_value = attrs.get("p_scale_value")
    P_SCALE = 1.0 if p_scale_value is None else float(p_scale_value)

    scale = attrs.get("softmax_scale")
    SOFTMAX_SCALE = None if scale in (None, "", 0) else float(scale)

    ENABLE_LSE = bool(attrs.get("return_softmax_lse", False))

    DATA_OFFSET_Q, DATA_RANGE_Q = _parse_data_range(attrs.get("data_range_q"))
    DATA_OFFSET_K, DATA_RANGE_K = _parse_data_range(attrs.get("data_range_k"))
    DATA_OFFSET_V, DATA_RANGE_V = _parse_data_range(attrs.get("data_range_v"))


def _apply_kwargs_globals(attrs):
    """兼容参考 assets 的两段式接口：额外 kwargs 目前均与 golden 无关（G 已折入
    N_q/N_kv、layout 恒 BNSD、win/mask 本方案不支持），仅吸收 device_id 类键。"""
    _ = attrs


def _inject_physical_s_override(q, v, layout_q="BNSD", layout_kv="BNSD"):
    """以 ttk 实际分配的张量 shape 覆盖张量 S 维（CSV 是唯一真源）。

    BNSD：q [B,N,S1,D]、v [B,N,S2,D]；仅记录物理维，seqused 语义不变。
    """
    global _SQ_PHYSICAL, _SKV_PHYSICAL
    _SQ_PHYSICAL = int(q.shape[2]) if (q is not None and q.ndim >= 3) else None
    _SKV_PHYSICAL = int(v.shape[2]) if (v is not None and v.ndim >= 3) else None


def _clear_physical_s_override():
    global _SQ_PHYSICAL, _SKV_PHYSICAL
    _SQ_PHYSICAL = None
    _SKV_PHYSICAL = None


def _tensor_seq_dims():
    """张量 S 维：物理覆盖优先，回退 max(seqused)/max_seqlen（pytest 同款公式）。"""
    max_sq = _SQ_PHYSICAL
    if max_sq is None:
        max_sq = (
            MAX_SEQLEN_Q
            if (MAX_SEQLEN_Q is not None and MAX_SEQLEN_Q > 0)
            else max(SEQUSED_Q or [0])
        )
    max_skv = _SKV_PHYSICAL
    if max_skv is None:
        max_skv = (
            MAX_SEQLEN_KV
            if (MAX_SEQLEN_KV is not None and MAX_SEQLEN_KV > 0)
            else max(SEQUSED_KV or [0])
        )
    return max_sq, max_skv


# ==============================================================================
# e8m0 编解码（pytest 同源）
# ==============================================================================
def fp32_to_e8m0fnu(tensor_fp32):
    bits = tensor_fp32.float().view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    return biased_exp.view(torch.float8_e8m0fnu)


def e8m0_to_fp32(tensor_e8m0):
    biased_exp = tensor_e8m0.view(torch.uint8).to(torch.int32)
    return torch.pow(2.0, biased_exp - 127).to(torch.float32)


def sanitize_e8m0_scale(scale, name="scale"):
    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    bad = ~torch.isfinite(result) | (result == 0)
    if int(bad.sum().item()):
        logger.info(
            "[WARN] %s: replace %d bad scale value(s)", name, int(bad.sum().item())
        )
        result[bad] = E8M0_MIN_POSITIVE
    return result


# ==============================================================================
# 量化 scale（pytest 同源：V 沿 S ceil 组，尾组 0 补齐）
# ==============================================================================
def _get_qk_block_scale(tensor):
    B_, N_, S_, D_ = tensor.shape
    num_groups = D_ // QUANT_GROUP_SIZE
    grouped = tensor.reshape(B_, N_, S_, num_groups, QUANT_GROUP_SIZE)
    max_vals = torch.max(torch.abs(grouped), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8
    return 2**shared_exp


def _get_v_block_scale(tensor):
    B_, N_, S_, D_ = tensor.shape
    num_groups = (S_ + QUANT_GROUP_SIZE - 1) // QUANT_GROUP_SIZE
    pad_rows = num_groups * QUANT_GROUP_SIZE - S_
    if pad_rows:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    grouped = tensor.reshape(B_, N_, num_groups, QUANT_GROUP_SIZE, D_)
    max_vals = torch.max(torch.abs(grouped), dim=-2)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8
    return 2**shared_exp


def _pack_scale_5d(scale):
    return scale.unsqueeze(-1).expand(*scale.shape, 2).contiguous()


def _quantize_qk(tensor, scale):
    scale_exp = scale.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[
        ..., : tensor.shape[-1]
    ]
    return (tensor / scale_exp).clamp(-448.0, 448.0).to(FP8_DTYPE)


def _quantize_v(tensor, scale):
    scale_exp = scale.repeat_interleave(QUANT_GROUP_SIZE, dim=2)[
        :, :, : tensor.shape[2], :
    ]
    return (tensor / scale_exp).clamp(-448.0, 448.0).to(FP8_DTYPE)


# ==============================================================================
# 数据生成（15 slot dict，参考 assets 命名）
# ==============================================================================
def generate_data():
    max_sq, max_skv = _tensor_seq_dims()

    torch.manual_seed(SEED_Q)
    q_fp16 = (
        DATA_OFFSET_Q
        + (torch.rand(B, N_q, max_sq, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_Q
    )
    torch.manual_seed(SEED_K)
    k_fp16 = (
        DATA_OFFSET_K
        + (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_K
    )
    torch.manual_seed(SEED_V)
    v_fp16 = (
        DATA_OFFSET_V
        + (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_V
    )

    q_scale = _get_qk_block_scale(q_fp16.float())
    k_scale = _get_qk_block_scale(k_fp16.float())
    v_scale = _get_v_block_scale(v_fp16.float())

    q_fp8 = _quantize_qk(q_fp16.float(), q_scale)
    k_fp8 = _quantize_qk(k_fp16.float(), k_scale)
    v_fp8 = _quantize_v(v_fp16.float(), v_scale)

    deq_q = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(q_scale, "q_scale")))
    deq_k = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(k_scale, "k_scale")))
    deq_v = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(v_scale, "v_scale")))

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32)

    logger.info(
        "[DATA] q=%s k=%s v=%s",
        tuple(q_fp8.shape),
        tuple(k_fp8.shape),
        tuple(v_fp8.shape),
    )

    return {
        "q": q_fp8,
        "k": k_fp8,
        "v": v_fp8,
        "q_descale": deq_q,
        "k_descale": deq_k,
        "v_descale": deq_v,
        "block_table": None,
        "p_scale": p_scale,
        "cu_seqlens_q": None,
        "cu_seqlens_kv": None,
        "act_seq_lens_q": list(SEQUSED_Q) if SEQUSED_Q else None,
        "act_seq_lens_kv": list(SEQUSED_KV) if SEQUSED_KV else None,
        "sinks": None,
        "attn_mask": None,
    }


# ==============================================================================
# CPU Golden（NPU fp16 softmax 管线逐位仿真，pytest 同源）
# ==============================================================================
def cpu_mxfp8_sf16_golden(data_dict):
    q_fp8 = data_dict["q"]
    k_fp8 = data_dict["k"]
    v_fp8 = data_dict["v"]
    dequant_scale_q = data_dict["q_descale"]
    dequant_scale_k = data_dict["k_descale"]
    v_descale = data_dict["v_descale"]
    p_scale = data_dict["p_scale"]
    actual_seq_q = data_dict.get("act_seq_lens_q") or SEQUSED_Q
    actual_seq_kv = data_dict.get("act_seq_lens_kv") or SEQUSED_KV

    if softmax_scale_value() is None:
        softmax_scale = 1.0 / math.sqrt(D)
    else:
        softmax_scale = softmax_scale_value()
    softmax_scale_f32 = torch.tensor(softmax_scale, dtype=torch.float32)
    # FixpipeMm1 deqScalar 的 (1,8,10) 硬件格式截断（尾数低 13 位丢弃）
    softmax_scale_hw = (softmax_scale_f32.view(torch.int32) & -8192).view(torch.float32)

    q_scale = e8m0_to_fp32(dequant_scale_q[..., 0]).repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[..., : q_fp8.shape[-1]]
    k_scale = e8m0_to_fp32(dequant_scale_k[..., 0]).repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[..., : k_fp8.shape[-1]]
    v_scale = e8m0_to_fp32(v_descale[..., 0]).repeat_interleave(
        QUANT_GROUP_SIZE, dim=2
    )[:, :, : v_fp8.shape[2], :]

    q_t = q_fp8.float() * q_scale
    k_t = k_fp8.float() * k_scale
    v_t = v_fp8.float() * v_scale

    if N_q != N_kv:
        g = N_q // N_kv
        k_t = k_t.repeat_interleave(g, dim=1)
        v_t = v_t.repeat_interleave(g, dim=1)

    LN2 = math.log(2.0)
    F16 = torch.float16
    LN2_F16 = torch.tensor(LN2, dtype=F16)
    INV_LN2_F16 = torch.tensor(1.0 / LN2, dtype=F16)
    Q_BLOCK = 128
    K_BLOCK = 256

    b_, n_ = q_t.shape[0], q_t.shape[1]
    sq, skv = q_t.shape[2], k_t.shape[2]
    dv = v_t.shape[-1]
    out = torch.zeros([b_, n_, sq, dv], dtype=torch.float32)
    lse = torch.zeros([b_, n_, sq], dtype=torch.float32)
    # per-batch seqused：逐 batch 循环；空批（kv=0 且 q>0）lse = -inf、输出 0（分母保护）
    for b in range(b_):
        sq_b = int(actual_seq_q[b])
        skv_b = int(actual_seq_kv[b])
        for q0 in range(0, sq_b, Q_BLOCK):
            q1 = min(q0 + Q_BLOCK, sq_b)
            qi = q_t[b : b + 1, :, q0:q1, :]
            m_acc = None
            s_acc = torch.zeros([1, n_, q1 - q0, 1], dtype=torch.float32)
            o_acc = torch.zeros([1, n_, q1 - q0, dv], dtype=torch.float32)
            for kv0 in range(0, skv_b, K_BLOCK):
                kv1 = min(kv0 + K_BLOCK, skv_b)
                s_blk = torch.matmul(
                    qi, k_t[b : b + 1, :, kv0:kv1, :].transpose(-1, -2)
                )
                s_blk = (s_blk.double() * softmax_scale_hw.double()).to(F16)
                m_blk_start = m_acc
                sub_ps = []
                for s0 in range(kv0, kv1, 128):
                    s1_ = min(s0 + 128, kv1)
                    s_sub = s_blk[:, :, :, s0 - kv0 : s1_ - kv0]
                    k_cur = torch.ceil(s_sub.amax(dim=-1, keepdim=True) * INV_LN2_F16)
                    k_i = k_cur if m_acc is None else torch.maximum(m_acc, k_cur)
                    m_acc = k_i
                    norm = k_i * LN2_F16
                    x = s_sub - norm
                    p16 = torch.exp(x)
                    sub_ps.append((p16, k_i, s0, s1_))
                k_final = m_acc
                num_blk = torch.zeros([1, n_, q1 - q0, dv], dtype=torch.float32)
                den_blk = torch.zeros([1, n_, q1 - q0, 1], dtype=torch.float32)
                for p16, k_i, s0, s1_ in sub_ps:
                    ps_u8 = torch.clamp(
                        k_i.float() - k_final.float() + 127.0, min=0.0, max=255.0
                    ).to(torch.uint8)
                    ps = e8m0_to_fp32(ps_u8.view(torch.float8_e8m0fnu))
                    p8 = p16.float().to(FP8_DTYPE).to(torch.float32) * ps
                    num_blk = num_blk + torch.matmul(p8, v_t[b : b + 1, :, s0:s1_, :])
                    den_blk = den_blk + p8.sum(dim=-1, keepdim=True)
                factor = (
                    torch.zeros([1, n_, q1 - q0, 1], dtype=torch.float32)
                    if m_blk_start is None
                    else torch.exp2(
                        torch.clamp(m_blk_start.float() - k_final.float(), min=-127.0)
                    )
                )
                o_acc = o_acc * factor + num_blk
                s_acc = s_acc * factor + den_blk
            s_safe = torch.where(s_acc > 0, s_acc, torch.ones_like(s_acc))
            out[b : b + 1, :, q0:q1, :] = o_acc / s_safe
            # 空批保护：seqused_kv[b]=0 时 m_acc 恒 None → softmax 空集 lse = -inf
            if m_acc is None:
                lse[b : b + 1, :, q0:q1] = float("-inf")
            else:
                lse[b : b + 1, :, q0:q1] = (
                    m_acc.float() * LN2 + torch.log(s_safe)
                ).squeeze(-1)

    out = out.to(torch.bfloat16)
    logger.info("[CPU] out=%s lse=%s", tuple(out.shape), tuple(lse.shape))
    return out, lse


def softmax_scale_value():
    return SOFTMAX_SCALE
