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

"""Precision comparison aligned with the SparseFlashMla pytest policy."""

import importlib.util
import sys
from pathlib import Path


class PrimaryComparatorLoader:
    """Load the comparator shared with the primary assets implementation."""

    @staticmethod
    def load():
        name = "smla_ttk_primary_compare"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).resolve().parents[2] / "assets" / "impl" / "compare.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module


COMPARATOR = PrimaryComparatorLoader.load().COMPARATOR


def compare(*outputs):
    """Compare NPU outputs followed by golden outputs."""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    return [
        COMPARATOR.compare_output(npu_out, golden_out)
        for npu_out, golden_out in zip(outputs[:half], outputs[half:])
    ]
