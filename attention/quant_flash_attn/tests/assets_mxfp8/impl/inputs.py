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

import logging
import math
import os
import sys

import numpy
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import quant_flash_attn_golden as mxfp8_golden_mod

logger = logging.getLogger(__name__)

_SEED_MAP = {"q": 54, "k": 3, "v": 4}


def _write_int32_list(slot, values, slot_name):
    """把 attributes 里的 int 值列表写进 CSV tensor slot（覆盖 ttk 随机生成）。

    空 slot（shape (0,)，无数据）或 values 为 None 时跳过。
    dst 可为 numpy 或 CPU torch tensor（ttk 经 backend.inputs_from_numpy 传入）。

    按 slot 的目标 dtype 写入（不硬编码 int32）——保留 CSV tensor_dtypes。
    正常用例 slot 是 int32；异常用例若把 cu_seqlens/seqused 设为 int8，
    此处用 int8 写（值可能溢出并抛错，或被 AICPU 校验拦截），tensor_dtypes 真正生效。
    """
    if slot is None:
        return
    if values is None:
        return
    # 空 slot（shape (0,)，无数据）→ 跳过，不覆盖（PA 布局下 cu_seqlens/seqused 可能
    # 留空，属性 value 是无意义的残留值）。
    if isinstance(slot, torch.Tensor):
        if slot.numel() == 0:
            return
    elif numpy.asarray(slot).size == 0:
        return
    if isinstance(slot, torch.Tensor):
        arr = torch.as_tensor(list(values), dtype=slot.dtype)
        if tuple(slot.shape) != tuple(arr.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= attribute value {tuple(arr.shape)} ({values!r}). "
                f"Check Excel shape vs cu_seqlens/seqused value."
            )
        slot.copy_(arr)
    else:
        target_dtype = numpy.asarray(slot).dtype
        arr = numpy.array(list(values), dtype=target_dtype)
        if tuple(slot.shape) != tuple(arr.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= attribute value {tuple(arr.shape)} ({values!r}). "
                f"Check Excel shape vs cu_seqlens/seqused value."
            )
        slot[...] = arr


def _write_causal_mask(slot, mask_mode, mask_shape):
    """mask_mode != 0 时生成上三角 causal mask 写入 attn_mask slot。

    直调 op 后主算子的 attn_mask 直接来自 CSV slot (ttk 随机生成), 需在这里用与
    golden _build_causal_mask 一致的方式重建: triu(ones, diagonal=1)。
    mask_mode == 0 或 slot 为 None 时跳过。
    """
    if mask_mode == 0 or slot is None:
        return
    shape = tuple(int(v) for v in mask_shape) if mask_shape else (2048, 2048)
    causal = torch.triu(torch.ones(shape, dtype=torch.int8), diagonal=1)
    if isinstance(slot, torch.Tensor):
        if tuple(slot.shape) != tuple(causal.shape):
            raise ValueError(
                f"[INPUTS] attn_mask shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= causal {tuple(causal.shape)}"
            )
        slot.copy_(causal.to(slot.dtype))
    else:
        arr = causal.numpy().astype(numpy.asarray(slot).dtype)
        if tuple(slot.shape) != tuple(arr.shape):
            raise ValueError(
                f"[INPUTS] attn_mask shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= causal {tuple(arr.shape)}"
            )
        slot[...] = arr


def generate_qfa_mxfp8_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    quant_mode: int,
    block_table: torch.Tensor,
    p_scale: torch.Tensor,
    cu_seqlens_q_t: torch.Tensor,
    cu_seqlens_kv_t: torch.Tensor,
    seqused_q_t: torch.Tensor,
    seqused_kv_t: torch.Tensor,
    sinks_t: torch.Tensor,
    attn_mask_t: torch.Tensor,
    metadata_t: torch.Tensor,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **kwargs,
):
    # —— op-schema 直调: 额外适配参数与真实值 sidecar 经 kwargs 传入 ——
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    enable_pa = bool(kwargs.get("enable_pa", False))
    kv_cache_layout = kwargs.get("kv_cache_layout") or layout_kv
    block_size = kwargs.get("block_size", 0)
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    cu_seqlens_q = kwargs.get("cu_seqlens_q_values")
    cu_seqlens_kv = kwargs.get("cu_seqlens_kv_values")
    seqused_q = kwargs.get("seqused_q_values")
    seqused_kv = kwargs.get("seqused_kv_values")
    input_layout = kwargs.get("input_layout", "TND")
    # cu_seqlens -> actual_seq (差分还原)
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]
    actual_seq_q = (
        [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
        if len(cu_seqlens_q) > 1
        else [0]
    )
    if len(cu_seqlens_kv) > 1:
        actual_seq_kv = [
            cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]
            for i in range(len(cu_seqlens_kv) - 1)
        ]
    elif seqused_kv is not None and len(seqused_kv) > 0:
        actual_seq_kv = list(seqused_kv)
    else:
        actual_seq_kv = [0]

    # batch_size: CSV 原始值 (可为 -1), 透传给 metadata;
    # B: 从 cu_seqlens_q 推导的正整数, 供 BNSD 张量生成。
    B = max(1, len(cu_seqlens_q) - 1) if cu_seqlens_q and len(cu_seqlens_q) >= 2 else 1

    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    for gkey, gval in [
        ("B", B),
        ("N_q", N_q),
        ("N_kv", N_kv),
        ("D", D),
        ("FP8_DTYPE", torch.float8_e4m3fn),
        ("QUANT_GROUP_SIZE", 32),
        ("CU_SEQLENS_Q", cu_seqlens_q),
        ("CU_SEQLENS_KV", cu_seqlens_kv if cu_seqlens_kv else None),
        ("SEQUSED_Q", list(seqused_q) if seqused_q is not None else None),
        ("SEQUSED_KV", list(seqused_kv) if seqused_kv is not None else None),
        ("MAX_SEQLEN_Q", max_seqlen_q),
        ("MAX_SEQLEN_KV", max_seqlen_kv),
        ("ENABLE_PA", enable_pa),
        ("KV_CACHE_LAYOUT", kv_cache_layout),
        ("BLOCK_SIZE", block_size),
        ("Q_SCALE_LAYOUT", q_scale_layout),
        ("INPUT_LAYOUT", input_layout),
        ("LAYOUT_Q", layout_q),
        ("LAYOUT_Q_DESCALE", layout_q_descale),
        ("LAYOUT_KV", layout_kv),
        ("LAYOUT_OUT", layout_out),
    ]:
        setattr(mxfp8_golden_mod, gkey, gval)

    # csv input_data_ranges 列由 ttk 作为 input_ranges kwarg 传入
    input_ranges = kwargs.get("input_ranges")

    def _range_of(idx, default=(-1.0, 1.0)):
        if input_ranges is None or idx >= len(input_ranges):
            return default
        r = input_ranges[idx]
        if r is None:
            return default
        return (float(r[0]), float(r[1]))

    rq_min, rq_max = _range_of(0)
    rk_min, rk_max = _range_of(1)
    rv_min, rv_max = _range_of(2)

    # ----- Step 1: 生成 bf16 BNSD random  -----
    torch.manual_seed(_SEED_MAP["q"])
    q_bf16 = (
        torch.rand(B, N_q, max_sq, D, dtype=torch.bfloat16) * (rq_max - rq_min) + rq_min
    )
    torch.manual_seed(_SEED_MAP["k"])
    k_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rk_max - rk_min)
        + rk_min
    )
    torch.manual_seed(_SEED_MAP["v"])
    v_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rv_max - rv_min)
        + rv_min
    )

    # ----- Step 2: 量化 bf16 -> fp8 BNSD + fp32 scale BNSD  -----
    fp8_dtype = torch.float8_e4m3fn
    group_size = 32
    fp8_max = 448.0

    quant_scale_q_bnsd = mxfp8_golden_mod.get_mxfp8_per_token_group_quant_scale(
        q_bf16, fp8_dtype, group_size
    )
    quant_scale_k_bnsd = mxfp8_golden_mod.get_mxfp8_per_token_group_quant_scale(
        k_bf16, fp8_dtype, group_size
    )
    quant_scale_v_bnsd = mxfp8_golden_mod.get_mxfp8_per_channel_group_quant_scale(
        v_bf16, fp8_dtype, group_size
    )

    q_fp8_bnsd_f32 = (
        mxfp8_golden_mod.mxfp8_per_token_group_quant(
            q_bf16, quant_scale_q_bnsd, group_size
        )
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    k_fp8_bnsd_f32 = (
        mxfp8_golden_mod.mxfp8_per_token_group_quant(
            k_bf16, quant_scale_k_bnsd, group_size
        )
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    v_fp8_bnsd_f32 = (
        mxfp8_golden_mod.mxfp8_per_channel_group_quant(
            v_bf16, quant_scale_v_bnsd, group_size
        )
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )

    # ----- Step 3: layout 转换 (BNSD -> final layout) -----
    q_fp8_final = mxfp8_golden_mod.convert_q_bnsd_to_layout(
        q_fp8_bnsd_f32, actual_seq_q, "TND", cu_seqlens=cu_seqlens_q
    )
    q_scale_final_layout = mxfp8_golden_mod.convert_q_scale_bnsd_to_layout(
        quant_scale_q_bnsd, actual_seq_q, q_scale_layout, cu_seqlens=cu_seqlens_q
    )

    if enable_pa:
        total_blocks_k = int(dequant_scale_k.shape[0])
        total_blocks_v = int(dequant_scale_v.shape[0])
        total_blocks = min(total_blocks_k, total_blocks_v)

        if block_table.size == 0:
            raise ValueError(
                "[INPUTS] enable_pa=True but block_table CSV shape is (0,) — "
                "Excel AI column empty for PA row"
            )
        max_blocks = block_table.shape[1] if block_table.ndim >= 2 else 0
        if max_blocks <= 0:
            raise ValueError(
                f"[INPUTS] PA mode block_table max_blocks={max_blocks} invalid"
            )
        torch.manual_seed(42)  # 确定性 block_table
        blockid_pool = torch.randperm(total_blocks, dtype=torch.int32)
        bt_real = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(actual_seq_kv[b] / block_size)
            for j in range(n_blocks):
                bt_real[b, j] = blockid_pool[idx % total_blocks]
                idx += 1
        k_fp8_final = mxfp8_golden_mod.mxfp8_pa_preprocessing(
            k_fp8_bnsd_f32,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_k,
        )
        v_fp8_final = mxfp8_golden_mod.mxfp8_pa_preprocessing(
            v_fp8_bnsd_f32,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_v,
        )

        k_scale_pa = mxfp8_golden_mod.mxfp8_pa_preprocessing(
            quant_scale_k_bnsd,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=True,
            is_vscale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_k,
        )
        v_scale_pa = mxfp8_golden_mod.mxfp8_pa_preprocessing(
            quant_scale_v_bnsd,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=True,
            is_vscale=True,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_v,
        )
        k_scale_final_layout = k_scale_pa
        v_scale_final_layout = v_scale_pa
    else:
        # 非 PA 模式: k/v -> TND
        k_fp8_final = mxfp8_golden_mod.convert_kv_bnsd_to_layout(
            k_fp8_bnsd_f32,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        v_fp8_final = mxfp8_golden_mod.convert_kv_bnsd_to_layout(
            v_fp8_bnsd_f32,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        # k/v scale: convert 函数内部做 pack, 输入 UNPACKED 4D scale (B, N, S, D)
        k_scale_final_layout = mxfp8_golden_mod.convert_k_scale_bnsd_to_layout(
            quant_scale_k_bnsd,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        v_scale_final_layout = mxfp8_golden_mod.convert_v_scale_bnsd_to_layout(
            quant_scale_v_bnsd, actual_seq_kv, "TND"
        )

    # ----- Step 4: fp32 scale -> e8m0 -----

    q_scale_e8m0 = mxfp8_golden_mod.fp32_to_e8m0fnu_safe(
        q_scale_final_layout, "Q scale"
    )
    k_scale_e8m0 = mxfp8_golden_mod.fp32_to_e8m0fnu_safe(
        k_scale_final_layout, "K scale"
    )
    v_scale_e8m0 = mxfp8_golden_mod.fp32_to_e8m0fnu_safe(
        v_scale_final_layout, "V scale"
    )

    def _inplace_write(dst_np, src_torch, slot_name):
        if tuple(dst_np.shape) != tuple(src_torch.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: CSV storage {tuple(dst_np.shape)} "
                f"!= computed {tuple(src_torch.shape)}. "
                f"Check Excel shape vs layout conversion logic."
            )
        dst_dtype_np = dst_np.dtype
        ts = str(src_torch.dtype)
        if "float8" in ts:
            src_np = src_torch.view(dst_dtype_np)
        else:
            src_np = src_torch.numpy()
        if src_np.dtype != dst_dtype_np:
            raise ValueError(
                f"[INPUTS] {slot_name} dtype mismatch: CSV storage {dst_dtype_np} "
                f"!= computed {src_np.dtype} (torch {src_torch.dtype}). "
                f"Check CSV dtype vs inputs.py quant/layout output dtype."
            )
        # numpy in-place 拷贝 (同 dtype 同 shape, bit-exact 内存拷贝)
        dst_np[...] = src_np

    q.copy_(q_fp8_final)
    k.copy_(k_fp8_final)
    v.copy_(v_fp8_final)
    dequant_scale_q.copy_(q_scale_e8m0)
    dequant_scale_k.copy_(k_scale_e8m0)
    dequant_scale_v.copy_(v_scale_e8m0)

    # ----- p_scale: in-place 写 fp32 -----
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    p_scale[...] = p_scale_value

    if enable_pa:
        block_table[...] = bt_real
    elif block_table is not None:
        block_table[...] = 0

    # ----- 新增 slot 8-14: 用 attributes 真实值覆盖 ttk 随机生成的 cu_seqlens/seqused。
    # sinks/metadata (slot 12,14) 无 value，保持 None 语义（CSV shape (0,) → 空 tensor，
    # sink/golden 不消费）；attn_mask (slot 13) 不传值覆盖——mask 由 golden
    # _build_causal_mask() 按 attributes 里 attn_mask_shape 重建。 -----
    _write_int32_list(cu_seqlens_q_t, cu_seqlens_q, "cu_seqlens_q (slot 8)")
    _write_int32_list(cu_seqlens_kv_t, cu_seqlens_kv, "cu_seqlens_kv (slot 9)")
    _write_int32_list(seqused_q_t, seqused_q, "seqused_q (slot 10)")
    _write_int32_list(seqused_kv_t, seqused_kv, "seqused_kv (slot 11)")
    _write_causal_mask(attn_mask_t, mask_mode, kwargs.get("attn_mask_shape"))

    logger.info(
        "[INPUTS] in-place wrote fp8 q/k/v (q=%s), e8m0 descale (dq=%s, dk=%s, dv=%s), "
        "fp32 p_scale, int32 block_table (pa=%s)",
        tuple(q.shape),
        tuple(dequant_scale_q.shape),
        tuple(dequant_scale_k.shape),
        tuple(dequant_scale_v.shape),
        enable_pa,
    )
