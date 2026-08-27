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
"""在 quant_flash_attn 主算子调用前填充 metadata slot。

本模块与 TTK 无关。框架在 H2D 之后传入主算子参数；此 hook 调用配套的
torch 算子并原地更新 ``metadata``。性能采集和结果处理仍由 TTK 负责，
hook 始终返回 ``None``。
"""

import importlib.util
import logging
import sys
from pathlib import Path

import torch


OPERATOR = "quant_flash_attn"
METADATA_INDEX = 14  # metadata_t 是 CSV slot 14


def load_metadata_protocol():
    name = "qfa_ttk_metadata_protocol"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("metadata_protocol.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def get_attribute(kwargs, name, default=None):
    """从 kwargs 取属性值，支持 pytest_ 前缀别名。"""
    value = kwargs.get(name)
    if value is None:
        value = kwargs.get(f"pytest_{name}")
    return default if value is None else value


def get_values(kwargs, name):
    """从 kwargs 取 *_values 列表（兼容 tensor 与 list）。"""
    value = kwargs.get(f"{name}_values")
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def max_sequence(prefix, used, fallback):
    """从 prefix 或 used 推导最大序列长度。"""
    if used is not None:
        return int(torch.as_tensor(used).detach().max().item())
    if prefix is not None:
        values = torch.as_tensor(prefix).detach().reshape(-1)
        if values.numel() > 1:
            return int((values[1:] - values[:-1]).max().item())
    return int(fallback)


def build_metadata_arguments(q, dequant_scale_v, cu_seqlens_q, cu_seqlens_kv,
                             seqused_q, seqused_kv, kwargs):
    """从 qfa 主算子入参推导 metadata 算子参数。

    qfa 的 metadata 算子签名:
      quant_flash_attn_metadata(num_heads_q, num_heads_kv, head_dim, quant_mode,
          cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None,
          v_descale=None, batch_size=None, max_seqlen_q=-1, max_seqlen_kv=-1,
          mask_mode=NO_MASK, win_left=-1, win_right=-1,
          layout_q="BSND", layout_q_descale="BSND", layout_kv="BSND", layout_out="BSND")

    与 smla 不同，qfa 没有 cu_seqlens_ori_kv/cu_seqlens_cmp_kv/seqused_ori_kv/
    seqused_cmp_kv/cmp_residual_kv 等稀疏相关参数。
    """
    layout_q = get_attribute(kwargs, "layout_q", "BSND")
    layout_kv = get_attribute(kwargs, "layout_kv", "BSND")
    layout_q_descale = get_attribute(kwargs, "layout_q_descale", "BSND")
    layout_out = get_attribute(kwargs, "layout_out", "BSND")

    q_shape = tuple(int(value) for value in q.shape)
    head_dim = int(get_attribute(kwargs, "D", q_shape[-1]))

    # num_heads_q 从 kwargs 或 q shape 推导（按 layout_q 判断 head 维位置）
    if layout_q == "TND":
        num_heads_q = int(get_attribute(kwargs, "N_q", q_shape[1] if len(q_shape) > 1 else 0))
    elif layout_q == "BSND":
        num_heads_q = int(get_attribute(kwargs, "N_q", q_shape[2] if len(q_shape) > 2 else 0))
    elif layout_q == "NTD":
        num_heads_q = int(get_attribute(kwargs, "N_q", q_shape[0] if len(q_shape) > 0 else 0))
    else:
        num_heads_q = int(get_attribute(kwargs, "N_q", 0))

    num_heads_kv = int(get_attribute(kwargs, "N_kv", 0))

    cu_q_values = get_values(kwargs, "cu_seqlens_q")
    cu_kv_values = get_values(kwargs, "cu_seqlens_kv")
    seq_q_values = get_values(kwargs, "seqused_q")
    seq_kv_values = get_values(kwargs, "seqused_kv")

    # batch_size: CSV 原始值（可为 -1），无值时从 cu_seqlens/seqused 推导
    batch_size = get_attribute(kwargs, "batch_size")
    if batch_size is None:
        if seq_q_values is not None:
            batch_size = len(seq_q_values)
        elif cu_q_values is not None:
            batch_size = len(cu_q_values) - 1
        elif seqused_q is not None:
            batch_size = int(torch.as_tensor(seqused_q).numel())
        elif cu_seqlens_q is not None:
            batch_size = int(torch.as_tensor(cu_seqlens_q).numel()) - 1
        else:
            batch_size = 0

    # max_seqlen_q/kv: 从 kwargs 取（-1 为显式值），无值时从 cu_seqlens/seqused 推导
    max_seqlen_q_attr = get_attribute(kwargs, "max_seqlen_q")
    if max_seqlen_q_attr is not None:
        max_seqlen_q = int(max_seqlen_q_attr)
    else:
        q_fallback = q_shape[1] if layout_q == "BSND" else q_shape[0]
        max_seqlen_q = max_sequence(
            cu_q_values if cu_q_values is not None else cu_seqlens_q,
            seq_q_values if seq_q_values is not None else seqused_q,
            q_fallback,
        )

    max_seqlen_kv_attr = get_attribute(kwargs, "max_seqlen_kv")
    if max_seqlen_kv_attr is not None:
        max_seqlen_kv = int(max_seqlen_kv_attr)
    else:
        max_seqlen_kv = max_sequence(
            cu_kv_values if cu_kv_values is not None else cu_seqlens_kv,
            seq_kv_values if seq_kv_values is not None else seqused_kv,
            0,
        )

    return {
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
        "quant_mode": int(get_attribute(kwargs, "quant_mode", 1)),
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "v_descale": dequant_scale_v,
        "batch_size": int(batch_size),
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "mask_mode": int(get_attribute(kwargs, "mask_mode", 0)),
        "layout_q": layout_q,
        "layout_q_descale": layout_q_descale,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
    }


def move_to_device(value, target):
    """将值移动到 target 的 device（None 透传）。"""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def run_metadata(arguments, metadata):
    """调用 quant_flash_attn_metadata 算子构建 metadata。"""
    import cann_ops_transformer

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
        batch_size=int(arguments["batch_size"]),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_kv=int(arguments["max_seqlen_kv"]),
        mask_mode=int(arguments["mask_mode"]),
        layout_q=str(arguments["layout_q"]),
        layout_q_descale=str(arguments["layout_q_descale"]),
        layout_kv=str(arguments["layout_kv"]),
        layout_out=str(arguments["layout_out"]),
    )


def run(q, k, v, dequant_scale_q, dequant_scale_k, dequant_scale_v,
        p_scale, block_table, cu_seqlens_q_t, cu_seqlens_kv_t,
        seqused_q_t, seqused_kv_t, sinks_t, attn_mask_t, metadata_t,
        **kwargs):
    """为 quant_flash_attn 主算子调用生成并拷贝 metadata。

    逻辑: 检查 metadata 是否已物化 → 读 sidecar → 回退到 build_metadata_arguments
    → 调 run_metadata → copy 到占位 → rewrite_metadata_input。
    """
    if metadata_t is None:
        raise ValueError("quant_flash_attn npu_preprocess requires metadata")
    protocol = load_metadata_protocol()
    testcase_name = kwargs.get("testcase_name")
    if protocol.metadata_is_materialized(metadata_t):
        logging.info("[%s] reuse nonzero QFA metadata input", testcase_name)
        return None

    arguments = protocol.load_metadata_inputs(OPERATOR, testcase_name)
    if arguments is not None:
        source = "manual-data sidecar"
    else:
        arguments = build_metadata_arguments(
            q, dequant_scale_v, cu_seqlens_q_t, cu_seqlens_kv_t,
            seqused_q_t, seqused_kv_t, kwargs,
        )
        source = "main API fallback (sidecar unavailable)"
    logging.info("[%s] build QFA metadata from %s", testcase_name, source)
    generated = run_metadata(arguments, metadata_t)
    if tuple(metadata_t.shape) != tuple(generated.shape):
        raise ValueError(
            "QFA metadata shape mismatch: "
            f"placeholder={tuple(metadata_t.shape)}, generated={tuple(generated.shape)}"
        )
    metadata_t.copy_(generated.to(dtype=metadata_t.dtype, device=metadata_t.device))
    rewritten = protocol.rewrite_metadata_input(
        OPERATOR,
        testcase_name,
        METADATA_INDEX,
        metadata_t,
    )
    if rewritten is not None:
        logging.info("[%s] rewrote QFA metadata input: %s", testcase_name, rewritten)
    return None
