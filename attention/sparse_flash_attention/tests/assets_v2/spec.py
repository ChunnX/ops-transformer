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

"""Standard-import control adapter for SparseFlashAttention assets."""

import sys
from pathlib import Path


REPOSITORY_ROOT = str(Path(__file__).resolve().parents[4])
path_added = REPOSITORY_ROOT not in sys.path
if path_added:
    sys.path.insert(0, REPOSITORY_ROOT)
try:
    from attention.sparse_flash_attention.tests.assets_v2.impl import golden as golden_module
    from attention.sparse_flash_attention.tests.assets_v2.impl import inputs as inputs_module
finally:
    if path_added:
        sys.path.remove(REPOSITORY_ROOT)


class SparseFlashAttentionSpec:
    golden = golden_module.cpu_sparse_flash_attention
    customize_inputs = inputs_module.generate_sfa_inputs
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


__spec__ = {
    "torch_npu.npu_sparse_flash_attention": "SparseFlashAttentionSpec",
}
