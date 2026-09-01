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
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def load_impl_module(stem):
    """Lazy-load impl modules to avoid import-time failures."""
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
# quant_mode dispatcher
# TTK 的 Spec 类是静态绑定 (golden/customize_inputs 在类定义时即固定),
# 不能在运行时按 case 的 quant_mode 切换。因此用 dispatcher 函数包装:
#   - quant_mode=1 → MXFP8 路径 (cpu_qfa_mxfp8 / generate_qfa_mxfp8_inputs)
#   - quant_mode=6 → GQA FP8 路径 (cpu_qfa_gqa_fp8 / generate_qfa_gqa_fp8_inputs)
#   - quant_mode=0 → HIF8 路径 (cpu_qfa_hif8 / generate_qfa_hif8_inputs)
# Spec 类的 golden/customize_inputs 绑定 dispatcher, 运行时按 kwargs 派发。
# ==============================================================================


def _golden_dispatch(*args, **kwargs):
    qm = kwargs.get("quant_mode", 1)
    golden_module = load_impl_module("golden")
    if qm == 6:
        return golden_module.cpu_qfa_gqa_fp8(*args, **kwargs)
    if qm == 0:
        return golden_module.cpu_qfa_hif8(*args, **kwargs)
    return golden_module.cpu_qfa_mxfp8(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    qm = kwargs.get("quant_mode", 1)
    inputs_module = load_impl_module("inputs")
    if qm == 6:
        return inputs_module.generate_qfa_gqa_fp8_inputs(*args, **kwargs)
    if qm == 0:
        return inputs_module.generate_qfa_hif8_inputs(*args, **kwargs)
    return inputs_module.generate_qfa_mxfp8_inputs(*args, **kwargs)


class QuantFlashAttnSpec:
    """quant_flash_attn 主算子测试规范 (MXFP8 + GQA FP8 + HIF8, 按 quant_mode 派发)。

    CSV api_name = torch.ops.cann_ops_transformer.quant_flash_attn。
    npu_preprocess 在主算子调用前构建 metadata 并注入 golden 全局变量,
    取代原 qfa_wrapper / qfa_main_wrapper。
    """

    golden = _golden_dispatch
    customize_inputs = _inputs_dispatch
    compare = staticmethod(lambda *o, **k: load_impl_module("compare").compare(*o, **k))

    torch_graph = load_impl_module("graph").QuantFlashAttnAclGraph
    npu_preprocess = load_impl_module("npu_preprocess").run

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
    """quant_flash_attn_metadata 独立测试规范 (T4 metadata-only 用例)。

    CSV api_name = torch.ops.cann_ops_transformer.quant_flash_attn_metadata。
    metadata 用例直接调 metadata 算子并返回 metadata tensor, 无 golden/compare,
    仅需 customize_inputs 把真实 cu_seqlens/seqused 等写入对应 tensor slot。
    """

    customize_inputs = _inputs_dispatch


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_flash_attn": "QuantFlashAttnSpec",
    "torch.ops.cann_ops_transformer.quant_flash_attn_metadata": "QuantFlashAttnMetadataSpec",
}
