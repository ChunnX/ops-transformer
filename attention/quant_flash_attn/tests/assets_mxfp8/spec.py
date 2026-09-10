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
import importlib.util
from pathlib import Path

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn
except ImportError as e:
    logging.warning("Failed to import cann_ops_transformer.ops: %s", e)

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


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
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


# ==============================================================================
# MXFP8 golden / customize_inputs 绑定
# 本工程仅支持 MXFP8 (quant_mode=1) 场景:
#   - golden → impl/golden.cpu_qfa_mxfp8
#   - customize_inputs → impl/inputs.generate_qfa_mxfp8_inputs
# ==============================================================================


def _golden_dispatch(*args, **kwargs):
    golden_module = load_impl_module("golden")
    return golden_module.cpu_qfa_mxfp8(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    inputs_module = load_impl_module("inputs")
    return inputs_module.generate_qfa_mxfp8_inputs(*args, **kwargs)


class QuantFlashAttnSpec:
    """quant_flash_attn 测试规范 (仅 MXFP8, quant_mode=1)。

    quant_mode=1: MXFP8 (Q/K per-token-group, V per-channel-group, descale=e8m0)

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
}
