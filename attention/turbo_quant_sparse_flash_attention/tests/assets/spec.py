# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TTK TestSpec entry for the standard TurboQuant sparse attention interface."""

import importlib.util
from pathlib import Path

_path = Path(__file__).with_name("tq_sfa_reference.py")
_spec = importlib.util.spec_from_file_location("tq_sfa_reference", _path)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)


class TurboQuantSparseFlashAttentionSpec:
    golden = staticmethod(_reference.golden)
    customize_inputs = staticmethod(_reference.generate_inputs)
    compare = staticmethod(_reference.compare)
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


__spec__ = {
    "cann_ops_transformer.turbo_quant_sparse_flash_attention": "TurboQuantSparseFlashAttentionSpec",
}
