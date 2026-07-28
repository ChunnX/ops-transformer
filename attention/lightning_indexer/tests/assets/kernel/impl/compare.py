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

"""Raw Kernel compare adapter for LightningIndexer."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

from ttk.utilities import numpy_to_torch_tensor


def _load_e2e_spec():
    name = "li_ttk_kernel_e2e_spec"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2] / "spec.py"
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
            "Failed to load LightningIndexer E2E compare for Kernel mode; "
            f"module={path.resolve()}; original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


E2E_SPEC = _load_e2e_spec()


def _to_torch(value):
    if value is None or hasattr(value, "detach"):
        return value
    if isinstance(value, tuple):
        return tuple(_to_torch(item) for item in value)
    if isinstance(value, list):
        return [_to_torch(item) for item in value]
    return numpy_to_torch_tensor(value)


def compare_lightning_indexer_kernel(*outputs, compare_context=None, **kwargs):
    """Reuse the E2E TopK policy and rebuild replay score context when needed."""
    if (
        compare_context is not None
        and not E2E_SPEC.golden_module.has_compare_data(compare_context.testcase_name)
    ):
        compare_context = replace(
            compare_context,
            input_tensors=_to_torch(compare_context.input_tensors),
        )
    return E2E_SPEC.LightningIndexerSpec.compare(
        *outputs, compare_context=compare_context, **kwargs
    )
