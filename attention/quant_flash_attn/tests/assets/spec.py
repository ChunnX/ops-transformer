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

"""TestSpec adapter for QuantFlashAttn TTK assets."""

import importlib.util
import sys
from pathlib import Path

import torch

ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    name = f"qfa_ttk_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise RuntimeError(
            "Failed to load QuantFlashAttn assets module; "
            f"stage=impl/{stem}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


npu_preprocess_module = load_impl_module("npu_preprocess")
golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
metadata_inputs_module = load_impl_module("metadata_inputs")
compare_module = load_impl_module("compare")


# ==============================================================================
# quant_mode dispatcher
# TTK 的 Spec 类是静态绑定 (golden/customize_inputs 在类定义时即固定),
# 不能在运行时按 case 的 quant_mode 切换。因此用 dispatcher 函数包装:
#   - quant_mode=0 → HIF8 路径 (cpu_qfa_hif8 / generate_qfa_hif8_inputs)
#   - quant_mode=1 → MXFP8 路径 (cpu_qfa_mxfp8 / generate_qfa_mxfp8_inputs)
#   - quant_mode=6 → GQA FP8 路径 (cpu_qfa_gqa_fp8 / generate_qfa_gqa_fp8_inputs)
# Spec 类的 golden/customize_inputs 绑定 dispatcher, 运行时按 kwargs 派发。
# ==============================================================================

def _golden_dispatch(*args, **kwargs):
    qm = kwargs.get("quant_mode", 1)
    if qm == 6:
        return golden_module.cpu_qfa_gqa_fp8(*args, **kwargs)
    if qm == 0:
        return golden_module.cpu_qfa_hif8(*args, **kwargs)
    return golden_module.cpu_qfa_mxfp8(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    qm = kwargs.get("quant_mode", 1)
    if qm == 6:
        return inputs_module.generate_qfa_gqa_fp8_inputs(*args, **kwargs)
    if qm == 0:
        return inputs_module.generate_qfa_hif8_inputs(*args, **kwargs)
    return inputs_module.generate_qfa_mxfp8_inputs(*args, **kwargs)


# ==============================================================================
# 薄适配函数
# quant_flash_attn 的 op 签名第 7 个位置参数是 int(quant_mode), 不是 tensor,
# 不能直接用 torch.ops.cann_ops_transformer.quant_flash_attn 作为 TTK 入口。
# 这两个函数只做参数名映射, 不做 metadata 构建/golden globals/prepare_npu_inputs。
# ==============================================================================

def npu_qfa(q, k, v, dequant_scale_q, dequant_scale_k, dequant_scale_v,
             p_scale, block_table, cu_seqlens_q_t, cu_seqlens_kv_t,
             seqused_q_t, seqused_kv_t, sinks_t, attn_mask_t, metadata_t,
             *, quant_mode=1, **kwargs):
    """薄适配函数: 将 15 个 CSV 位置 tensor 映射到 quant_flash_attn op 的 kwargs。

    参数名映射: dequant_scale_* -> *_descale, *_t 后缀去掉。
    metadata 由 npu_preprocess 预填, 其余 tensor 由 inputs.py 生成。
    """
    atten_out, lse_out = torch.ops.cann_ops_transformer.quant_flash_attn(
        q,
        k,
        v,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        int(quant_mode),
        block_table=block_table,
        p_scale=p_scale,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=cu_seqlens_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        sinks=sinks_t,
        attn_mask=attn_mask_t,
        metadata=metadata_t,
        softmax_scale=kwargs.get("softmax_scale", 1.0),
        mask_mode=kwargs.get("mask_mode", 0),
        win_left=kwargs.get("win_left", -1),
        win_right=kwargs.get("win_right", -1),
        max_seqlen_q=kwargs.get("max_seqlen_q", -1),
        max_seqlen_kv=kwargs.get("max_seqlen_kv", -1),
        layout_q=kwargs.get("layout_q", "BSND"),
        layout_q_descale=kwargs.get("layout_q_descale", "BSND"),
        layout_kv=kwargs.get("layout_kv", "BSND"),
        layout_out=kwargs.get("layout_out", "BSND"),
        return_softmax_lse=bool(kwargs.get("enable_lse", False)),
    )
    if not bool(kwargs.get("enable_lse", False)):
        lse_out = None
    return atten_out, lse_out


def run_metadata(q, k, v, dequant_scale_q, dequant_scale_k, dequant_scale_v,
                 p_scale, block_table, cu_seqlens_q_t, cu_seqlens_kv_t,
                 seqused_q_t, seqused_kv_t, sinks_t, attn_mask_t, metadata_t,
                 *, quant_mode=1, **kwargs):
    """薄适配函数: 将 15 个 CSV 位置 tensor 映射到 quant_flash_attn_metadata op 的参数。

    metadata 算子直接接收 cu_seqlens_q/cu_seqlens_kv/seqused_q/seqused_kv 作为参数。
    """
    metadata = torch.ops.cann_ops_transformer.quant_flash_attn_metadata(
        int(kwargs.get("N_q", 0)),
        int(kwargs.get("N_kv", 0)),
        int(kwargs.get("D", 0)),
        int(quant_mode),
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=cu_seqlens_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        v_descale=dequant_scale_v,
        batch_size=kwargs.get("batch_size"),
        max_seqlen_q=kwargs.get("max_seqlen_q", -1),
        max_seqlen_kv=kwargs.get("max_seqlen_kv", -1),
        mask_mode=kwargs.get("mask_mode", 0),
        layout_q=kwargs.get("layout_q", "BSND"),
        layout_q_descale=kwargs.get("layout_q_descale", "BSND"),
        layout_kv=kwargs.get("layout_kv", "BSND"),
        layout_out=kwargs.get("layout_out", "BSND"),
    )
    return [metadata]


class QfaMxfp8Spec:
    """quant_flash_attn 测试规范 (MXFP8 + GQA FP8, 按 quant_mode 派发)。

    quant_mode=1: MXFP8 (Q/K per-token-group, V per-channel-group, descale=e8m0)
    quant_mode=6: GQA FP8 全量化 (Q/K per-token-head, V per-head, descale=FP32)
    """

    golden = _golden_dispatch
    customize_inputs = _inputs_dispatch
    npu_preprocess = npu_preprocess_module.run
    compare = compare_module.compare

    tolerance = {
        "float16": {
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
        },
        "bfloat16": {
            "rtol": 0.0078125,
            "ptol": 0.005,
            "atol": 0.0001,
        },
    }


class QfaMetadataSpec:
    """quant_flash_attn_metadata 独立测试规范。

    复用主输入生成路径填充 tensor slot, thin adapter run_metadata 映射到
    metadata 算子。按 quant_mode 派发到 MXFP8 或 GQA FP8 路径。
    """

    customize_inputs = metadata_inputs_module.generate_qfa_metadata_inputs


__spec__ = {
    "spec.npu_qfa": "QfaMxfp8Spec",
    "spec.run_metadata": "QfaMetadataSpec",
}
