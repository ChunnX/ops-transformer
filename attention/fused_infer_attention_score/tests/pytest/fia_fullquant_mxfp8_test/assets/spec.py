#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""TestSpec adapter for npu_fused_infer_attention_score_v2 (MXFP8 FIA) assets.

注册 fia_mxfp8_wrapper.npu_fia_mxfp8 → FiaMxfp8Spec。
TTK e2e 模式通过 --plugin-path 指向本目录(spec.py 所在的 assets/)发现 spec,
通过 api_name = "fia_mxfp8_wrapper.npu_fia_mxfp8" 调用 wrapper(PYTHONPATH 需含 fia_fullquant_mxfp8_test/)。
"""

import importlib.util
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"fia_assets_impl_{stem}_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class FiaMxfp8Spec:
    """npu_fused_infer_attention_score_v2 (MXFP8) 测试规范。"""
    golden = golden_module.cpu_fia_mxfp8
    customize_inputs = inputs_module.generate_fia_mxfp8_inputs
    compare = compare_module.compare

    tolerance = {
        "float16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.005, "atol": 0.000025},
        "bfloat16": {"standard": "IsClose", "rtol": 0.0078125, "ptol": 0.005, "atol": 0.0001},
    }


__spec__ = {
    "fia_mxfp8_wrapper.npu_fia_mxfp8": FiaMxfp8Spec,
}
