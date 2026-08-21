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
#   - quant_mode=1 → MXFP8 路径 (cpu_qfa_mxfp8 / generate_qfa_mxfp8_inputs)
#   - quant_mode=6 → GQA FP8 路径 (cpu_qfa_gqa_fp8 / generate_qfa_gqa_fp8_inputs)
# Spec 类的 golden/customize_inputs 绑定 dispatcher, 运行时按 kwargs 派发。
# ==============================================================================

def _golden_dispatch(*args, **kwargs):
    qm = kwargs.get("quant_mode", 1)
    if qm == 6:
        return golden_module.cpu_qfa_gqa_fp8(*args, **kwargs)
    return golden_module.cpu_qfa_mxfp8(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    qm = kwargs.get("quant_mode", 1)
    if qm == 6:
        return inputs_module.generate_qfa_gqa_fp8_inputs(*args, **kwargs)
    return inputs_module.generate_qfa_mxfp8_inputs(*args, **kwargs)


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

    独立 metadata case 只生成 metadata 算子所需的 descriptor 向量
    (cu_seqlens_q/kv, seqused_q/kv), 不跑主算子。
    """

    customize_inputs = metadata_inputs_module.generate_qfa_metadata_inputs


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_flash_attn": "QfaMxfp8Spec",
    "torch.ops.cann_ops_transformer.quant_flash_attn_metadata": "QfaMetadataSpec",
}
