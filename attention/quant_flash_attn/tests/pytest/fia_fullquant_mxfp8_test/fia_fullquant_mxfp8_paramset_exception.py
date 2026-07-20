#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
MxFP8 异常用例 paramset。

思路2 实现：3 个泛化基座 (A1/A2/A3) + 篡改清单 EXC_CASES。
每条篡改项 = (case_name, base_name, mutator)。
mutator(inputs) 直接修改 prepare_npu_inputs 返回的入参字典，并返回 (meta_overrides, main_overrides)。
test runner 调用 _call_npu_fa_op(**inputs, meta_overrides=, main_overrides=)，预期抛错。
"""

import torch

# ==============================================================================
# 基座定义
# 与 fia_fullquant_mxfp8_paramset_common 的 case 字段结构一致，供 test_runner.apply_params 使用。
# 选择依据：
#   A1 TND-prefill: G*Q_S>80 (G=4,Q_S=32 → 128)，覆盖非PA场景的 descale shape / layout / mask / block_table 篡改
#   A2 PA_BNBD-decode: G*Q_S<=80 (G=8,Q_S=4 → 32)，覆盖PA场景的 block_table / seqused_kv / block_size / descale 篡改
#   A3 PA_NZ-decode: 同 A2 但 NZ layout，覆盖 NZ 专属 5D/6D shape 篡改
# ==============================================================================

BASE_CASES = {
    "EXC_BASE_A1_TND_PREFILL_NOMASK": {
        "name": "EXC_BASE_A1_TND_PREFILL_NOMASK",
        "B": 2,
        "N_q": 8,
        "N_kv": 2,
        "D": 128,
        "cu_seqlens_q": [0, 32, 64],
        "cu_seqlens_kv": [0, 64, 128],
        "seqused_q": [32, 32],
        "seqused_kv": [64, 64],
        "max_seqlen_q": 32,
        "max_seqlen_kv": 64,
        "enable_pa": False,
        "kv_cache_layout": "TND",
        "block_size": 512,
        "mask_mode": 0,
        "q_scale_layout": "TND",
        "p_scale": 1.0,
        "softmax_scale": None,
        "enable_lse": False,
    },
    "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL": {
        "name": "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL",
        "B": 1,
        "N_q": 64,
        "N_kv": 8,
        "D": 128,
        "cu_seqlens_q": [0, 4],
        "cu_seqlens_kv": None,
        "seqused_q": [4],
        "seqused_kv": [1024],
        "max_seqlen_q": 4,
        "max_seqlen_kv": 1024,
        "enable_pa": True,
        "kv_cache_layout": "BnNBsD",
        "block_size": 512,
        "mask_mode": 3,
        "q_scale_layout": "N2TGD",
        "p_scale": 1.0,
        "softmax_scale": None,
        "enable_lse": False,
    },
    "EXC_BASE_A3_PA_NZ_DECODE_CAUSAL": {
        "name": "EXC_BASE_A3_PA_NZ_DECODE_CAUSAL",
        "B": 1,
        "N_q": 64,
        "N_kv": 8,
        "D": 128,
        "cu_seqlens_q": [0, 4],
        "cu_seqlens_kv": None,
        "seqused_q": [4],
        "seqused_kv": [1024],
        "max_seqlen_q": 4,
        "max_seqlen_kv": 1024,
        "enable_pa": True,
        "kv_cache_layout": "PA_NZ",
        "block_size": 1024,
        "mask_mode": 3,
        "q_scale_layout": "N2TGD",
        "p_scale": 1.0,
        "softmax_scale": None,
        "enable_lse": False,
    },
}


# ==============================================================================
# mutator 工具
# ==============================================================================

def _ret(meta=None, main=None):
    """统一返回 (meta_overrides, main_overrides)，空则用空 dict。"""
    return meta or {}, main or {}


def _fake_int32_tensor(shape, device="npu"):
    """构造一个指定 shape 的 int32 NPU tensor，用于 block_table 等篡改。"""
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _fake_e8m0_tensor(shape, device="npu"):
    """构造一个指定 shape 的 float8_e8m0fnu NPU tensor。

    注意: NPU 不允许直接用 torch.zeros() 创建 E8M0 tensor (DT_FLOAT8_E8M0)，必须先在 CPU 创建再搬运。
    """
    return torch.zeros(shape, dtype=torch.float8_e8m0fnu, device="cpu").npu()


# ==============================================================================
# A1 基座 (TND-prefill-nomask) 篡改项
# ==============================================================================

def mut_a1_layout_kv_BSND(inputs):
    inputs["layout_kv"] = "BSND"
    return _ret()


def mut_a1_layout_kv_BNSD(inputs):
    inputs["layout_kv"] = "BNSD"
    return _ret()


def mut_a1_layout_kv_PA_BBND(inputs):
    inputs["layout_kv"] = "PA_BBND"
    return _ret()


def mut_a1_layout_out_BSND(inputs):
    inputs["layout_out"] = "BSND"
    return _ret()


def mut_a1_mask_mode_4(inputs):
    inputs["sparse_mode"] = 4
    return _ret()


def mut_a1_mask_mode0_with_attn_mask(inputs):
    # mask_mode 改 0，但 mask tensor 仍存在 (A1 基座 mask=None，需先构造一个)
    inputs["sparse_mode"] = 0
    inputs["mask"] = torch.zeros(2048, 2048, dtype=torch.int8, device=inputs["q"].device)
    return _ret()


def mut_a1_mask_mode3_no_attn_mask(inputs):
    # mask_mode 改 3，但 mask=None
    inputs["sparse_mode"] = 3
    inputs["mask"] = None
    return _ret()


def mut_a1_nonPA_block_table_provided(inputs):
    inputs["block_table"] = _fake_int32_tensor((1, 2), device=inputs["q"].device)
    return _ret()


def mut_a1_cu_seqlens_kv_none(inputs):
    inputs["cu_seqlens_kv"] = None
    return _ret()


def mut_a1_q_descale_wrong_dtype(inputs):
    # MxFP8 要求 E8M0，改为 fp32。E8M0 不能直接在 NPU 上 cast，需先搬到 CPU
    s = inputs["dequant_scale_q"]
    inputs["dequant_scale_q"] = s.cpu().to(torch.float32).npu()
    return _ret()


def mut_a1_q_descale_wrong_dim(inputs):
    # 4D → 3D
    s = inputs["dequant_scale_q"]
    inputs["dequant_scale_q"] = s.reshape(s.shape[0], s.shape[1], -1)
    return _ret()


def mut_a1_q_descale_wrong_shape(inputs):
    # 末维 2 → 3，E8M0 需在 CPU 上创建
    s = inputs["dequant_scale_q"]
    inputs["dequant_scale_q"] = _fake_e8m0_tensor((s.shape[0], s.shape[1], s.shape[2], 3), device=s.device)
    return _ret()


def mut_a1_k_descale_wrong_shape(inputs):
    s = inputs["dequant_scale_k"]
    inputs["dequant_scale_k"] = _fake_e8m0_tensor((s.shape[0], s.shape[1], s.shape[2], 3), device=s.device)
    return _ret()


def mut_a1_q_wrong_dtype(inputs):
    inputs["q"] = inputs["q"].to(torch.float16)
    return _ret()


def mut_a1_q_wrong_dim(inputs):
    q = inputs["q"]
    inputs["q"] = q.unsqueeze(0)  # 3D → 4D
    return _ret()


def mut_a1_quant_mode_2(inputs):
    return _ret(main={"quant_mode": 2})


def mut_a1_meta_head_dim_96(inputs):
    return _ret(meta={"head_dim": 96})


def mut_a1_meta_num_heads_not_divisible(inputs):
    return _ret(meta={"num_heads_q": 7, "num_heads_kv": 2})


def mut_a1_meta_mask_mode_5(inputs):
    return _ret(meta={"mask_mode": 5})


def mut_a1_meta_layout_q_invalid(inputs):
    return _ret(meta={"layout_q": "XXXX"})


# ==============================================================================
# A2 基座 (PA_BNBD-decode-causal) 篡改项
# ==============================================================================

def mut_a2_pa_block_table_none(inputs):
    inputs["block_table"] = None
    return _ret()


def mut_a2_pa_block_table_wrong_dtype(inputs):
    inputs["block_table"] = inputs["block_table"].to(torch.float32)
    return _ret()


def mut_a2_pa_block_table_wrong_shape(inputs):
    bt = inputs["block_table"]
    # 改成 1D
    inputs["block_table"] = bt.reshape(-1)
    return _ret()


def mut_a2_pa_seqused_kv_none(inputs):
    inputs["seqused_kv"] = None
    return _ret()


def mut_a2_pa_block_size_256(inputs):
    # 篡改 k/v/scale 的 Bs 维为 256 (原 512)，block_table 同步扩 Bn
    # 原 k: (Bn, KV_N, 512, D)，改 (2*Bn, KV_N, 256, D)
    k = inputs["k"]
    v = inputs["v"]
    kv_n = k.shape[1]
    d = k.shape[-1]
    bn_old = k.shape[0]
    bs_new = 256
    bn_new = bn_old * 2
    inputs["k"] = k.reshape(bn_new, kv_n, bs_new, d).contiguous()
    inputs["v"] = v.reshape(bn_new, kv_n, bs_new, d).contiguous()
    # block_table 改成 (B, bn_new)
    bt = inputs["block_table"]
    b = bt.shape[0]
    new_bt = torch.zeros(b, bn_new, dtype=bt.dtype, device=bt.device)
    new_bt[:, :bt.shape[1]] = bt
    inputs["block_table"] = new_bt
    # k_descale: (Bn, KV_N, Bs, D/64, 2) → Bs 维改 256
    ks = inputs["dequant_scale_k"]
    if ks.dim() == 5 and ks.shape[2] == 512:
        inputs["dequant_scale_k"] = ks.reshape(bn_new, kv_n, bs_new, ks.shape[3], 2).contiguous()
    vs = inputs["dequant_scale_v"]
    if vs.dim() == 5 and vs.shape[1] == 512:
        inputs["dequant_scale_v"] = vs.reshape(bn_new, kv_n, bs_new // 64, vs.shape[3], 2).contiguous()
    return _ret()


def mut_a2_pa_k_descale_wrong_shape(inputs):
    s = inputs["dequant_scale_k"]
    inputs["dequant_scale_k"] = _fake_e8m0_tensor((s.shape[0], s.shape[1], s.shape[2], s.shape[3], 3), device=s.device)
    return _ret()


def mut_a2_pa_v_descale_wrong_shape(inputs):
    s = inputs["dequant_scale_v"]
    inputs["dequant_scale_v"] = _fake_e8m0_tensor((s.shape[0], s.shape[1], s.shape[2], s.shape[3], 3), device=s.device)
    return _ret()


def mut_a2_pa_q_descale_4D_with_N2TGD(inputs):
    # decode 应 5D，改 4D。E8M0 需在 CPU 创建
    s = inputs["dequant_scale_q"]
    inputs["dequant_scale_q"] = _fake_e8m0_tensor((s.shape[1], s.shape[0] * s.shape[2], s.shape[3], 2), device=s.device)
    return _ret()


def mut_a2_pa_attn_mask_wrong_shape(inputs):
    m = inputs["mask"]
    inputs["mask"] = m[:1024, :1024].contiguous()
    return _ret()


def mut_a2_pa_mask_mode0_with_attn_mask(inputs):
    inputs["sparse_mode"] = 0
    # mask 仍存在 (A2 基座 mask 为 causal mask)
    return _ret()


def mut_a2_pa_layout_out_BNSD(inputs):
    inputs["layout_out"] = "BNSD"
    return _ret()


# ==============================================================================
# A3 基座 (PA_NZ-decode-causal) 篡改项
# ==============================================================================

def mut_a3_nz_k_wrong_shape(inputs):
    k = inputs["k"]
    # NZ: (Bn, KV_N, D/32, Bs, 32)，篡改 D/32 维
    inputs["k"] = k.reshape(k.shape[0], k.shape[1], -1, k.shape[4], 1).repeat(1, 1, 1, 1, 32).contiguous()
    return _ret()


def mut_a3_nz_k_descale_wrong_6D(inputs):
    s = inputs["dequant_scale_k"]
    # 6D (Bn, KV_N, Bs/16, D/64, 16, 2)，篡改末维。E8M0 需在 CPU 创建
    inputs["dequant_scale_k"] = _fake_e8m0_tensor((s.shape[0], s.shape[1], s.shape[2], s.shape[3], s.shape[4], 3),
                                                  device=s.device)
    return _ret()


def mut_a3_nz_v_descale_wrong_6D(inputs):
    s = inputs["dequant_scale_v"]
    inputs["dequant_scale_v"] = _fake_e8m0_tensor((s.shape[0], s.shape[1], s.shape[2], s.shape[3], s.shape[4], 3),
                                                  device=s.device)
    return _ret()


# ==============================================================================
# 篡改清单
# 每条 = (case_name, base_name, mutator, expected_keyword)
# expected_keyword 用于断言报错信息 (可选，None 表示仅断言抛错)
# ==============================================================================

EXC_CASES = [
    # ---- A1: TND-prefill-nomask ----
    ("a1_layout_kv_BSND", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_layout_kv_BSND, "layout_kv"),
    ("a1_layout_kv_BNSD", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_layout_kv_BNSD, "layout_kv"),
    ("a1_layout_kv_PA_BBND", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_layout_kv_PA_BBND, "layout_kv"),
    ("a1_layout_out_BSND", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_layout_out_BSND, "layout_out"),
    ("a1_mask_mode_4", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_mask_mode_4, "mask_mode"),
    ("a1_mask_mode0_with_attn_mask", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_mask_mode0_with_attn_mask, "attn_mask"),
    ("a1_mask_mode3_no_attn_mask", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_mask_mode3_no_attn_mask, "attn_mask"),
    ("a1_nonPA_block_table_provided", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_nonPA_block_table_provided, "block_table"),
    ("a1_cu_seqlens_kv_none", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_cu_seqlens_kv_none, "cu_seqlens_kv"),
    ("a1_q_descale_wrong_dtype", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_q_descale_wrong_dtype, "dtype"),
    ("a1_q_descale_wrong_dim", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_q_descale_wrong_dim, "q_descale"),
    ("a1_q_descale_wrong_shape", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_q_descale_wrong_shape, "q_descale"),
    ("a1_k_descale_wrong_shape", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_k_descale_wrong_shape, "k_descale"),
    ("a1_q_wrong_dtype", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_q_wrong_dtype, "dtype"),
    ("a1_q_wrong_dim", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_q_wrong_dim, "q"),
    ("a1_quant_mode_2", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_quant_mode_2, "quant_mode"),
    ("a1_meta_head_dim_96", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_meta_head_dim_96, "headDim"),
    ("a1_meta_num_heads_not_divisible", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_meta_num_heads_not_divisible, "divisible"),
    ("a1_meta_mask_mode_5", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_meta_mask_mode_5, "maskMode"),
    ("a1_meta_layout_q_invalid", "EXC_BASE_A1_TND_PREFILL_NOMASK", mut_a1_meta_layout_q_invalid, "layoutQ"),
    # ---- A2: PA_BNBD-decode-causal ----
    ("a2_pa_block_table_none", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_block_table_none, "block_table"),
    ("a2_pa_block_table_wrong_dtype", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_block_table_wrong_dtype, "dtype"),
    ("a2_pa_block_table_wrong_shape", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_block_table_wrong_shape, "shape"),
    ("a2_pa_seqused_kv_none", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_seqused_kv_none, "seqused_kv"),
    ("a2_pa_block_size_256", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_block_size_256, "block_size"),
    ("a2_pa_k_descale_wrong_shape", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_k_descale_wrong_shape, "k_descale"),
    ("a2_pa_v_descale_wrong_shape", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_v_descale_wrong_shape, "v_descale"),
    ("a2_pa_q_descale_4D_with_N2TGD", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_q_descale_4D_with_N2TGD, "q_descale"),
    ("a2_pa_attn_mask_wrong_shape", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_attn_mask_wrong_shape, "shape"),
    ("a2_pa_mask_mode0_with_attn_mask", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_mask_mode0_with_attn_mask, "attn_mask"),
    ("a2_pa_layout_out_BNSD", "EXC_BASE_A2_PA_BNBD_DECODE_CAUSAL", mut_a2_pa_layout_out_BNSD, "layout_out"),
    # ---- A3: PA_NZ-decode-causal ----
    ("a3_nz_k_wrong_shape", "EXC_BASE_A3_PA_NZ_DECODE_CAUSAL", mut_a3_nz_k_wrong_shape, "shape"),
    ("a3_nz_k_descale_wrong_6D", "EXC_BASE_A3_PA_NZ_DECODE_CAUSAL", mut_a3_nz_k_descale_wrong_6D, "k_descale"),
    ("a3_nz_v_descale_wrong_6D", "EXC_BASE_A3_PA_NZ_DECODE_CAUSAL", mut_a3_nz_v_descale_wrong_6D, "v_descale"),
]
