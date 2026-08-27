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
"""独立 quant_flash_attn_metadata 测试的输入定制。

qfa 的 metadata 算子直接接收 cu_seqlens_q/cu_seqlens_kv/seqused_q/seqused_kv
作为参数（而非拷贝到 tensor slot），因此 metadata_inputs 主要做 sidecar 加载
和参数准备——复用主输入生成路径，但 sidecar 化 metadata 参数。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


_VECTOR_NAMES = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
)


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


def load_inputs_module():
    """动态加载 inputs 模块，避免循环导入。"""
    name = "qfa_ttk_inputs"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("inputs.py")
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


def load_sidecar_values(kwargs):
    """加载当前 testcase 的 sidecar，无则返回空 dict。"""
    metadata_input = load_metadata_protocol().load_metadata_inputs(
        "quant_flash_attn", kwargs.get("testcase_name")
    )
    return {} if metadata_input is None else metadata_input


def copy_int_list(target, values, name):
    """把 int 值列表写进 CSV tensor slot（覆盖 ttk 随机生成）。

    空 slot（shape (0,)）或 values 为 None 时跳过。
    """
    if target is None or values is None:
        return
    if isinstance(target, torch.Tensor):
        if target.numel() == 0:
            return
        arr = torch.as_tensor(list(values), dtype=target.dtype)
        if tuple(target.shape) != tuple(arr.shape):
            raise ValueError(
                f"QFA metadata {name} shape mismatch: CSV slot {tuple(target.shape)} "
                f"!= value {tuple(arr.shape)}"
            )
        target.copy_(arr)
    else:
        target_np = np.asarray(target)
        if target_np.size == 0:
            return
        arr = np.array(list(values), dtype=target_np.dtype)
        if tuple(target_np.shape) != tuple(arr.shape):
            raise ValueError(
                f"QFA metadata {name} shape mismatch: CSV slot {tuple(target_np.shape)} "
                f"!= value {tuple(arr.shape)}"
            )
        target[...] = arr


def generate_qfa_metadata_inputs(
    q,
    k,
    v,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    cu_seqlens_q_t,
    cu_seqlens_kv_t,
    seqused_q_t,
    seqused_kv_t,
    sinks_t,
    attn_mask_t,
    metadata_t,
    **kwargs,
):
    """为独立 metadata 测试准备输入: 复用主输入生成路径, sidecar 化 metadata 参数。

    接收与主输入生成相同的 15 个位置 tensor 参数 + kwargs。
    复用主输入生成路径填充所有 tensor slot（含 v_descale 量化），
    使 thin adapter run_metadata 能直接用这些 slot 调 metadata 算子。
    """
    quant_mode = kwargs.get("quant_mode", 1)
    inputs_module = load_inputs_module()
    # 复用主输入生成路径: 填充所有 tensor slot + zero metadata + 保存 sidecar
    if quant_mode == 6:
        inputs_module.generate_qfa_gqa_fp8_inputs(
            q,
            k,
            v,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            sinks_t,
            attn_mask_t,
            metadata_t,
            **kwargs,
        )
    else:
        inputs_module.generate_qfa_mxfp8_inputs(
            q,
            k,
            v,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            sinks_t,
            attn_mask_t,
            metadata_t,
            **kwargs,
        )
