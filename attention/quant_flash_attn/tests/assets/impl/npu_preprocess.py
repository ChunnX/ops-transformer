#!/usr/bin/python
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

The framework passes the main API arguments after H2D; this hook invokes the
companion torch operator ``quant_flash_attn_metadata`` and updates
``metadata`` in place.  Profiling and result handling remain in TTK, and the
hook always returns ``None``.
"""

import logging

import torch


OPERATOR = "quant_flash_attn"


def get_attribute(kwargs, name, default=None, aliases=()):
    for key in (name, f"pytest_{name}", *aliases):
        value = kwargs.get(key)
        if value is not None:
            return value
    return default


def _none_if_empty(tensor):
    """空 slot（shape (0,)，如 PA 布局下的 cu_seqlens/seqused）视为 None。"""
    if tensor is None:
        return None
    if torch.is_tensor(tensor) and tensor.numel() == 0:
        return None
    return tensor


def move_to_device(value, target):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def _num_heads_q_from_shape(q, layout_q):
    if layout_q == "NTD":
        return int(q.shape[0])
    if layout_q in ("TND", "BNSD"):
        return int(q.shape[1])
    return int(q.shape[2])


def _num_heads_kv_from_shape(k, layout_kv):
    if layout_kv == "PA_BBND":
        return int(k.shape[2])
    if layout_kv in ("TND", "BNSD", "PA_BNBD", "PA_NZ"):
        return int(k.shape[1])
    return int(k.shape[2])


def run_metadata(arguments, metadata):
    import cann_ops_transformer  # noqa: F401  (注册 torch.ops.cann_ops_transformer)

    return torch.ops.cann_ops_transformer.quant_flash_attn_metadata(
        int(arguments["num_heads_q"]),
        int(arguments["num_heads_kv"]),
        int(arguments["head_dim"]),
        int(arguments["quant_mode"]),
        cu_seqlens_q=move_to_device(arguments.get("cu_seqlens_q"), metadata),
        cu_seqlens_kv=move_to_device(arguments.get("cu_seqlens_kv"), metadata),
        seqused_q=move_to_device(arguments.get("seqused_q"), metadata),
        seqused_kv=move_to_device(arguments.get("seqused_kv"), metadata),
        v_descale=move_to_device(arguments.get("v_descale"), metadata),
        batch_size=arguments.get("batch_size"),
        max_seqlen_q=arguments.get("max_seqlen_q"),
        max_seqlen_kv=arguments.get("max_seqlen_kv"),
        mask_mode=int(arguments["mask_mode"]),
        win_left=int(arguments["win_left"]),
        win_right=int(arguments["win_right"]),
        layout_q=str(arguments["layout_q"]),
        layout_q_descale=str(arguments["layout_q_descale"]),
        layout_kv=str(arguments["layout_kv"]),
        layout_out=str(arguments["layout_out"]),
    )


def run(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    block_table,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    sinks,
    attn_mask,
    metadata,
    softmax_scale,
    mask_mode,
    win_left,
    win_right,
    max_seqlen_q,
    max_seqlen_kv,
    layout_q,
    layout_q_descale,
    layout_kv,
    layout_out,
    return_softmax_lse,
    **kwargs,
):
    """Generate and copy metadata for a main QuantFlashAttn invocation.

    位置参数对齐 cann_ops_transformer::quant_flash_attn schema 的 27 个参数
    (q/k/v/q_descale/k_descale/v_descale/quant_mode/block_table/p_scale/
    cu_seqlens_q/cu_seqlens_kv/seqused_q/seqused_kv/sinks/attn_mask/metadata/
    softmax_scale/mask_mode/win_left/win_right/max_seqlen_q/max_seqlen_kv/
    layout_q/layout_q_descale/layout_kv/layout_out/return_softmax_lse);
    N_q/N_kv/D/batch_size 等非 schema 属性由 TTK 以 kwargs 传入。
    """
    del v, q_descale, k_descale, block_table, p_scale, sinks, attn_mask
    del softmax_scale, return_softmax_lse
    if metadata is None:
        raise ValueError("QuantFlashAttn npu_preprocess requires metadata")

    layout_q = str(layout_q)
    layout_kv = str(layout_kv)
    is_tnd_q = layout_q == "TND"
    is_tnd_kv = layout_kv == "TND"

    cu_seqlens_q = _none_if_empty(cu_seqlens_q)
    cu_seqlens_kv = _none_if_empty(cu_seqlens_kv)
    seqused_q = _none_if_empty(seqused_q)
    seqused_kv = _none_if_empty(seqused_kv)

    # batch_size: CSV 原始值 (可为 -1/None), 透传; TND 时 metadata 算子要求 None。
    batch_size = get_attribute(kwargs, "batch_size")
    if batch_size is not None:
        batch_size = int(batch_size)

    arguments = {
        "num_heads_q": int(
            get_attribute(
                kwargs,
                "N_q",
                _num_heads_q_from_shape(q, layout_q),
                ("num_heads_q",),
            )
        ),
        "num_heads_kv": int(
            get_attribute(
                kwargs,
                "N_kv",
                _num_heads_kv_from_shape(k, layout_kv),
                ("num_heads_kv",),
            )
        ),
        "head_dim": int(get_attribute(kwargs, "D", int(q.shape[-1]), ("head_dim",))),
        "quant_mode": int(quant_mode),
        # cu_seqlens 仅在对应 TND layout 下透传 (与 metadata 算子约定一致)。
        "cu_seqlens_q": cu_seqlens_q if is_tnd_q else None,
        "cu_seqlens_kv": cu_seqlens_kv if is_tnd_kv else None,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "v_descale": v_descale,
        "batch_size": None if is_tnd_q else batch_size,
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_kv": int(max_seqlen_kv),
        "mask_mode": int(mask_mode),
        "win_left": int(win_left),
        "win_right": int(win_right),
        "layout_q": layout_q,
        "layout_q_descale": str(layout_q_descale),
        "layout_kv": layout_kv,
        "layout_out": str(layout_out),
    }

    testcase_name = kwargs.get("testcase_name")
    logging.info(
        "[%s] build QFA metadata (quant_mode=%d, layout_q=%s, layout_kv=%s)",
        testcase_name,
        arguments["quant_mode"],
        layout_q,
        layout_kv,
    )
    generated = run_metadata(arguments, metadata)
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "QFA metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    return None
