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

"""TestSpec adapter for QuantLightningIndexer assets."""

import importlib.util
from pathlib import Path


def _load_sibling(stem):
    path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(f"qli_assets_{stem}_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_golden = _load_sibling("_golden")
_inputs = _load_sibling("_inputs")
_compare = _load_sibling("_compare")


class QuantLightningIndexerSpec:
    golden = _golden.npu_quant_lightning_indexer
    customize_inputs = _inputs.generate_qli_inputs
    tolerance = {
        "float16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.05, "atol": 0.000025},
        "bfloat16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.05, "atol": 0.0001},
    }

    def compare(*outputs, **kwargs):
        return _compare.compare(*outputs, scores=_golden.get_topk_scores())


__spec__ = {
    "torch_npu.npu_quant_lightning_indexer": QuantLightningIndexerSpec,
}
