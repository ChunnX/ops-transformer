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

"""Raw Kernel input adapter for SparseFlashAttention."""

import importlib.util
import sys
from pathlib import Path

from ttk.utilities import numpy_to_torch_tensor


def _load_e2e_inputs():
    name = "sfa_ttk_kernel_e2e_inputs"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2] / "assets" / "impl" / "inputs.py"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def generate_sparse_flash_attention_kernel_inputs(
        query, key, value, sparse_indices, block_table,
        actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope, key_rope, sinks, **kwargs):
    """Reuse the pytest-backed E2E preparation on Kernel-mode NumPy buffers."""
    arrays = (
        query, key, value, sparse_indices, block_table,
        actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope, key_rope, sinks,
    )
    tensors = tuple(numpy_to_torch_tensor(array) for array in arrays)
    params = dict(kwargs)
    scale_value = params.pop("scale_value", 1.0)
    _load_e2e_inputs().generate_sfa_inputs(
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        scale_value,
        block_table=tensors[4],
        actual_seq_lengths_query=tensors[5],
        actual_seq_lengths_kv=tensors[6],
        query_rope=tensors[7],
        key_rope=tensors[8],
        **params,
    )
    return arrays


__input__ = {
    "kernel": {
        "sparse_flash_attention": "generate_sparse_flash_attention_kernel_inputs",
    }
}
