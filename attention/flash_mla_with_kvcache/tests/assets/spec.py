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

"""TestSpec adapter for FlashMlaWithKvcache TTK assets.

接口与 flash_attn 的 tests/assets/spec.py 一致（golden / customize_inputs /
compare / tolerance / torch_graph + __spec__ 映射）；impl 模块全部懒加载，
避免 import 时期失败。MLA 适配：golden 走 impl/golden.py 的
flash_mla_with_kvcache_golden(tensors, params)；torch_graph 指向 impl/graph.py
的占位类（结构占位，待填充）。
"""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    """Lazy-load impl modules to avoid import-time failures."""
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"mla_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class FlashMlaWithKvcacheSpec:
    """TestSpec for FlashMlaWithKvcache operator.

    Attributes are loaded lazily on first access to defer import errors.
    """

    @staticmethod
    def golden(*args, **kwargs):
        return _load_impl_module("golden").flash_mla_with_kvcache_golden(
            *args, **kwargs
        )

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load_impl_module("inputs").customize_inputs(*args, **kwargs)

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def compare(*outputs, **kwargs):
        return _load_impl_module("compare").compare(*outputs, **kwargs)

    torch_graph = _load_impl_module("graph").FlashMlaWithKvcacheAclGraph


__spec__ = {
    "flash_mla_with_kvcache_ttk_ops.flash_mla_with_kvcache_ttk": "FlashMlaWithKvcacheSpec",
}
