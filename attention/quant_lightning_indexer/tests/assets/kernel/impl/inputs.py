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

"""Raw Kernel input adapter for QuantLightningIndexer."""

import importlib.util
import sys
from pathlib import Path

from ttk.utilities import numpy_to_torch_tensor


def _load_e2e_inputs():
    name = "qli_ttk_kernel_e2e_inputs"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2] / "impl" / "inputs.py"
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
            "Failed to load QuantLightningIndexer E2E input adapter for Kernel mode; "
            f"module={path.resolve()}; original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


def generate_quant_lightning_indexer_kernel_inputs(
        query, key, weights, query_dequant_scale, key_dequant_scale,
        actual_seq_lengths_query, actual_seq_lengths_key, block_table, **kwargs):
    """Reuse the E2E input preparation on Kernel-mode NumPy buffers."""
    arrays = (
        query, key, weights, query_dequant_scale, key_dequant_scale,
        actual_seq_lengths_query, actual_seq_lengths_key, block_table,
    )
    tensors = tuple(numpy_to_torch_tensor(array) for array in arrays)
    params = dict(kwargs)
    query_quant_mode = params.pop("query_quant_mode", 0)
    key_quant_mode = params.pop("key_quant_mode", 0)
    _load_e2e_inputs().generate_qli_inputs(
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        tensors[4],
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_query=tensors[5],
        actual_seq_lengths_key=tensors[6],
        block_table=tensors[7],
        **params,
    )
    return arrays


__input__ = {
    "kernel": {
        "quant_lightning_indexer": "generate_quant_lightning_indexer_kernel_inputs",
    }
}
