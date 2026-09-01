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
"""Populate the QuantFlashAttn metadata slot before the main API call.

独立于 TTK: TTK 在 H2D 之后把主算子入参传给本 hook, 本 hook 调伴随算子
``quant_flash_attn_metadata`` 生成 metadata 并写回 ``metadata`` slot, 同时把
case 参数注入对应 golden 模块全局变量 (供 prepare_npu_inputs / compare 使用)。
profiling 和结果处理留在 TTK, 本 hook 始终返回 ``None``。
"""

import logging
import os
import sys

import torch
import torch_npu

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import quant_flash_attn_golden as mxfp8_golden_mod
import quant_flash_attn_fp8_golden as fp8_golden_mod
import quant_flash_attn_hif8_golden as hif8_golden_mod

logger = logging.getLogger(__name__)

_metadata_op = None


def _resolve_metadata_op():
    """惰性解析 quant_flash_attn_metadata 算子入口。

    优先走 cann_ops_transformer.ops python 包路径 (与 CSV api_name 一致的
    import 链路, 不依赖 torch.ops 命名空间注册时机); 包不可用时回退到
    torch.ops.cann_ops_transformer 命名空间。
    """
    global _metadata_op
    if _metadata_op is not None:
        return _metadata_op
    try:
        from cann_ops_transformer.ops import quant_flash_attn_metadata

        _metadata_op = quant_flash_attn_metadata
        return _metadata_op
    except (ImportError, AttributeError):
        pass
    _metadata_op = torch.ops.cann_ops_transformer.quant_flash_attn_metadata
    return _metadata_op


def _tolist_t(t, default=None):
    """把 tensor slot 读回 list (空 tensor -> default)。"""
    if t is None:
        return default
    return t.detach().cpu().tolist() if t.numel() > 0 else default


def _get(kw, name, default=None):
    v = kw.get(name)
    return default if v is None else v


def _apply_golden_globals(params, quant_mode=1):
    """把 case 参数注入 golden 模块全局变量 (按 quant_mode 选择目标模块).

    quant_mode=6 → fp8_golden_mod (GQA FP8 全量化路径)
    quant_mode=0 → hif8_golden_mod (HIF8 per-tensor 量化路径)
    其他 → mxfp8_golden_mod (MXFP8 路径)
    """
    if quant_mode == 6:
        target = fp8_golden_mod
    elif quant_mode == 0:
        target = hif8_golden_mod
    else:
        target = mxfp8_golden_mod
    for k, v in params.items():
        setattr(target, k, v)


def run(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    *,
    block_table=None,
    p_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    **kwargs,
):
    """生成并写回主算子调用的 metadata slot。"""
    if metadata is None:
        raise ValueError("QuantFlashAttn npu_preprocess requires metadata")

    device_id = _get(kwargs, "device_id", 0)
    torch_npu.npu.set_device(int(device_id))

    # ---- case 参数 (CSV attributes, 经 kwargs 传入) ----
    batch_size = _get(kwargs, "batch_size")
    N_q = _get(kwargs, "N_q")
    N_kv = _get(kwargs, "N_kv")
    D = _get(kwargs, "D")
    max_seqlen_q = _get(kwargs, "max_seqlen_q")
    max_seqlen_kv = _get(kwargs, "max_seqlen_kv")
    enable_pa = bool(_get(kwargs, "enable_pa", False))
    kv_cache_layout = _get(kwargs, "kv_cache_layout", "BnNBsD")
    block_size = _get(kwargs, "block_size", 0)
    mask_mode = _get(kwargs, "mask_mode", 0)
    q_scale_layout = _get(kwargs, "q_scale_layout", "TND")
    enable_lse = bool(_get(kwargs, "enable_lse", False))
    input_layout = _get(kwargs, "input_layout", "TND")
    is_contiguous = _get(kwargs, "is_contiguous", True)
    softmax_scale = _get(kwargs, "softmax_scale")
    layout_q = _get(kwargs, "layout_q", "TND")
    layout_q_descale = _get(kwargs, "layout_q_descale", "TND")
    layout_kv = _get(kwargs, "layout_kv", "TND")
    layout_out = _get(kwargs, "layout_out", "TND")

    # cu_seqlens/seqused 真实值已由 inputs.py in-place 写入 tensor slot,
    # 这里从 tensor 读回 list 计算 B / 注入 golden 全局变量。
    cu_seqlens_q_list = _tolist_t(cu_seqlens_q, [0])
    cu_seqlens_kv_list = _tolist_t(cu_seqlens_kv, [0])
    seqused_q_list = _tolist_t(seqused_q)
    seqused_kv_list = _tolist_t(seqused_kv)

    # batch_size: CSV 原始值 (可为 -1), 透传给 metadata;
    # B: 从 cu_seqlens_q 推导的正整数, 供 golden/prepare_npu_inputs 生成 BNSD 张量。
    B = (
        batch_size
        if batch_size is not None and batch_size > 0
        else (
            max(1, len(cu_seqlens_q_list) - 1)
            if cu_seqlens_q_list and len(cu_seqlens_q_list) >= 2
            else len(list(seqused_q_list))
            if seqused_q_list is not None and len(list(seqused_q_list)) > 0
            else 1
        )
    )

    attn_mask_shape = tuple(attn_mask.shape) if attn_mask is not None else None
    p_scale_val = (
        float(p_scale.item())
        if isinstance(p_scale, torch.Tensor) and p_scale.numel() > 0
        else (float(p_scale) if p_scale is not None else None)
    )

    _apply_golden_globals(
        {
            "B": B,
            "BATCH_SIZE": batch_size,
            "N_q": N_q,
            "N_kv": N_kv,
            "D": D,
            "CU_SEQLENS_Q": cu_seqlens_q_list,
            "CU_SEQLENS_KV": cu_seqlens_kv_list,
            "SEQUSED_Q": seqused_q_list,
            "SEQUSED_KV": seqused_kv_list,
            "MAX_SEQLEN_Q": max_seqlen_q,
            "MAX_SEQLEN_KV": max_seqlen_kv,
            "ENABLE_PA": enable_pa,
            "KV_CACHE_LAYOUT": kv_cache_layout,
            "BLOCK_SIZE": block_size,
            "SPARSE_MODE": mask_mode,
            "ATTN_MASK_SHAPE": attn_mask_shape or (2048, 2048),
            "Q_SCALE_LAYOUT": q_scale_layout,
            "P_SCALE": p_scale_val,
            "ENABLE_LSE": enable_lse,
            "FP8_DTYPE": torch.float8_e4m3fn,
            "QUANT_MODE": int(quant_mode),
            "QUANT_GROUP_SIZE": 32,
            "INPUT_LAYOUT": input_layout,
            "LAYOUT_Q": layout_q,
            "LAYOUT_Q_DESCALE": layout_q_descale,
            "LAYOUT_KV": layout_kv,
            "LAYOUT_OUT": layout_out,
            "IS_CONTIGUOUS": is_contiguous,
            "DEVICE_ID": device_id,
            "SOFTMAX_SCALE": softmax_scale,
            "SEED_Q": kwargs.get("seed_q"),
            "SEED_K": kwargs.get("seed_k"),
            "SEED_V": kwargs.get("seed_v"),
            "DATA_RANGE_Q": kwargs.get("data_range_q"),
            "DATA_RANGE_K": kwargs.get("data_range_k"),
            "DATA_RANGE_V": kwargs.get("data_range_v"),
        },
        quant_mode=int(quant_mode),
    )

    # ---- 构建 metadata 并写回 slot ----
    is_tnd_q = layout_q == "TND"
    is_tnd_kv = layout_kv == "TND"
    nq = N_q if N_q is not None else int(q.shape[1])
    nkv = N_kv if N_kv is not None else int(k.shape[1])
    head_dim = D if D is not None else int(q.shape[-1])

    logger.info(
        "[QFA_PREPROCESS] build metadata (quant_flash_attn_metadata), "
        "quant_mode=%d, N_q=%s, N_kv=%s, D=%s, batch_size=%s",
        int(quant_mode),
        nq,
        nkv,
        head_dim,
        batch_size,
    )
    generated = _resolve_metadata_op()(
        int(nq),
        int(nkv),
        int(head_dim),
        int(quant_mode),
        cu_seqlens_q=cu_seqlens_q if is_tnd_q else None,
        cu_seqlens_kv=cu_seqlens_kv if is_tnd_kv else None,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        v_descale=v_descale,
        batch_size=batch_size if not is_tnd_q else None,
        mask_mode=int(mask_mode),
        layout_q=layout_q,
        layout_q_descale=layout_q_descale,
        layout_kv=layout_kv,
        layout_out=layout_out,
        max_seqlen_q=int(max_seqlen_q) if max_seqlen_q is not None else -1,
        max_seqlen_kv=int(max_seqlen_kv) if max_seqlen_kv is not None else -1,
    )

    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "QuantFlashAttn metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    logger.info(
        "[QFA_PREPROCESS] metadata written to slot, shape=%s", tuple(metadata.shape)
    )
    return None
