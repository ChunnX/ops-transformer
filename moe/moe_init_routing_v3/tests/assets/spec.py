#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TestSpec adapter for MoeInitRoutingV3 assets."""

import importlib.util
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"mirv3_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class AclnnMoeInitRoutingV3Spec:
    golden = golden_module.aclnn_golden
    customize_inputs = inputs_module.aclnn_customize_inputs
    tolerance = {
        "float16": {"rtol": 0.001, "atol": 0.001},
        "bfloat16": {"rtol": 0.001, "atol": 0.001},
        "float32": {"rtol": 0.0001, "atol": 0.0001},
        "int8": {"rtol": 0, "atol": 0},
    }

    def compare(*outputs, **kwargs):
        return compare_module.compare(*outputs)


__spec__ = {
    "aclnnMoeInitRoutingV3": "AclnnMoeInitRoutingV3Spec",
}
