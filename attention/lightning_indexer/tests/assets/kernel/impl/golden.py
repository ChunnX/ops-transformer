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

"""Raw Kernel CPU golden adapter for LightningIndexer."""

import importlib.util
import sys
from pathlib import Path

from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor


def _load_e2e_golden():
    name = "li_ttk_golden"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2] / "impl" / "golden.py"
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
            "Failed to load LightningIndexer E2E Golden adapter for Kernel mode; "
            f"module={path.resolve()}; original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


def cpu_lightning_indexer_kernel(
        query, key, weights, actual_seq_lengths_query,
        actual_seq_lengths_key, block_table, **kwargs):
    """Return NumPy goldens for the raw LightningIndexer signature."""
    arrays = (
        query, key, weights, actual_seq_lengths_query,
        actual_seq_lengths_key, block_table,
    )
    tensors = tuple(numpy_to_torch_tensor(array) for array in arrays)
    outputs = _load_e2e_golden().cpu_lightning_indexer(
        tensors[0],
        tensors[1],
        tensors[2],
        actual_seq_lengths_query=tensors[3],
        actual_seq_lengths_key=tensors[4],
        block_table=tensors[5],
        **kwargs,
    )
    return tuple(torch_to_numpy_tensor(output.detach().cpu()) for output in outputs)


__golden__ = {
    "kernel": {
        "lightning_indexer": "cpu_lightning_indexer_kernel",
    }
}
