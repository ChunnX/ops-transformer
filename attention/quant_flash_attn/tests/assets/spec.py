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

import importlib.util
import sys
from pathlib import Path

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn
except ImportError as e:
    logging.warning("Failed to import cann_ops_transformer.ops: %s", e)

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}

def load_impl_module(stem):
    # [aclgraph兼容] 模块名必须稳定(hash(path)每进程随机)且注册进 sys.modules —
    # torch.compile/npugraph_ex 的 dynamo 追踪需按名 import 回模块,
    # 未注册时报 ModuleNotFoundError: qfa_assets_impl_graph_xxx
    mod_name = f"qfa_assets_impl_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

def load_impl_module(stem):
    """懒加载 impl 模块。

    与 flash_attn 资产一致: 不在 import spec.py 时 (主进程) 级联 import
    golden/graph (其中会 import torch_npu), 避免 fork 子进程 re-init NPU 报错。
    改为在 golden/customize_inputs/compare/npu_preprocess 首次被 ttk 调用时
    (fork 之后的 worker) 才加载。
    """
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"qfa_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


# ==============================================================================
# quant_mode dispatcher
# TTK 的 Spec 类是静态绑定 (golden/customize_inputs 在类定义时即固定),
# 不能在运行时按 case 的 quant_mode 切换。因此用 dispatcher 函数包装:
#   - quant_mode=1 → MXFP8 路径 (cpu_qfa_mxfp8 / generate_qfa_mxfp8_inputs)
#   - quant_mode=6 → GQA FP8 路径 (cpu_qfa_gqa_fp8 / generate_qfa_gqa_fp8_inputs)
# Spec 类的 golden/customize_inputs 绑定 dispatcher, 运行时按 kwargs 派发。
# ==============================================================================


def _normalize_wrapper_args(args, kwargs):
    """wrapper 约定 → 直调 op 约定的参数顺序归一化。

    qfa_wrapper.npu_qfa (TTK 按 wrapper 签名构参):
        [6]=p_scale, [7]=block_table, quant_mode 仅出现在 kwargs
    直调 op (torch.ops...quant_flash_attn) / impl golden:
        [6]=quant_mode, [7]=block_table, [8]=p_scale

    检测: args[6] 非 int 即 wrapper 约定 (p_scale tensor/None)。
    归一化后 [8:] 段 (cu_seqlens...seqused_kv_tail) 两者顺序天然对齐。
    """
    if len(args) > 6 and not isinstance(args[6], int):
        qm = kwargs.pop("quant_mode", 1)
        args = list(args[:6]) + [qm, args[7], args[6]] + list(args[8:])
        # wrapper 的 enable_lse 属性 → 直调 op 的 return_softmax_lse 参数
        if "enable_lse" in kwargs and "return_softmax_lse" not in kwargs:
            kwargs["return_softmax_lse"] = kwargs.pop("enable_lse")
    return args, kwargs


def _resolve_quant_mode(args, kwargs):
    """直调 op 后 quant_mode 是位置参数 (args[6])，不再出现在 kwargs。"""
    qm = args[6] if len(args) > 6 else kwargs.get("quant_mode", 1)
    return int(qm) if qm is not None else 1


def _golden_dispatch(*args, **kwargs):
    golden_module = load_impl_module("golden")
    args, kwargs = _normalize_wrapper_args(args, kwargs)
    qm = _resolve_quant_mode(args, kwargs)
    if qm == 6:
        return golden_module.cpu_qfa_gqa_fp8(*args, **kwargs)
    if qm == 0:
        return golden_module.cpu_qfa_hif8(*args, **kwargs)
    return golden_module.cpu_qfa_mxfp8(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    inputs_module = load_impl_module("inputs")
    args, kwargs = _normalize_wrapper_args(args, kwargs)
    qm = _resolve_quant_mode(args, kwargs)
    if qm == 6:
        return inputs_module.generate_qfa_gqa_fp8_inputs(*args, **kwargs)
    if qm == 0:
        return inputs_module.generate_qfa_hif8_inputs(*args, **kwargs)
    return inputs_module.generate_qfa_mxfp8_inputs(*args, **kwargs)


class QuantFlashAttnSpec:
    """quant_flash_attn 测试规范 (MXFP8 + GQA FP8 + HIF8, 按 quant_mode 派发)。

    quant_mode=1: MXFP8 (Q/K per-token-group, V per-channel-group, descale=e8m0)
    quant_mode=6: GQA FP8 全量化 (Q/K per-token-head, V per-head, descale=FP32)
    quant_mode=0: HIF8 per-tensor 量化

    metadata 由 npu_preprocess 在主算子调用前生成并回填 metadata slot,
    主算子直接由 ttk 调用 torch.ops.cann_ops_transformer.quant_flash_attn。

    golden/customize_inputs/compare/npu_preprocess 均懒加载 impl 模块,
    避免主进程 fork 前 import torch_npu (与 flash_attn 资产一致)。
    """

    golden = staticmethod(_golden_dispatch)
    customize_inputs = staticmethod(_inputs_dispatch)

    @staticmethod
    def compare(*outputs, **kwargs):
        return load_impl_module("compare").compare(*outputs, **kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        # wrapper 约定 (qfa_wrapper.npu_qfa): metadata 由 wrapper 内部两段式
        # (quant_flash_attn_metadata + quant_flash_attn) 生成, 无需 TTK 预处理;
        # 直调 op 约定 (torch.ops...quant_flash_attn) 才需要本 hook 回填 metadata slot。
        if len(args) > 6 and not isinstance(args[6], int):
            return None
        return load_impl_module("npu_preprocess").run(*args, **kwargs)

    torch_graph = load_impl_module("graph").QuantFlashAttnAclGraph

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


class QuantFlashAttnMetadataSpec:
    """quant_flash_attn_metadata 生成器的 TestSpec。

    与 flash_attn 资产的 FlashAttnMetadataSpec 一致: 只提供 customized inputs
    (把 cu_seqlens/seqused 描述向量回填进 metadata API 输入), 无独立测试套件。
    """

    customize_inputs = load_impl_module(
        "metadata_inputs"
    ).generate_quant_flash_attn_metadata_inputs


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_flash_attn": "QuantFlashAttnSpec",
    "torch.ops.cann_ops_transformer.quant_flash_attn_metadata": (
        "QuantFlashAttnMetadataSpec"
    ),
    # wrapper 约定 (p_scale@6, quant_mode 仅kw): dispatcher 内归一化为直调 op 顺序
    "qfa_wrapper.npu_qfa": "QuantFlashAttnSpec",
    "qfa_metadata_wrapper.run_metadata": "QuantFlashAttnMetadataSpec",
}
