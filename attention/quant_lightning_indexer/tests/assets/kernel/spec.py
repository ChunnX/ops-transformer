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

"""Raw Kernel TestSpec adapter for QuantLightningIndexer."""

import importlib.util
import sys
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    name = f"qli_ttk_kernel_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    try:
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise RuntimeError(
            "Failed to load QuantLightningIndexer Kernel assets module; "
            f"stage=kernel/impl/{stem}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class QuantLightningIndexerKernelSpec:
    golden = golden_module.cpu_quant_lightning_indexer_kernel
    customize_inputs = inputs_module.generate_quant_lightning_indexer_kernel_inputs
    compare = compare_module.compare_quant_lightning_indexer_kernel


__spec__ = {
    "quant_lightning_indexer": "QuantLightningIndexerKernelSpec",
}
