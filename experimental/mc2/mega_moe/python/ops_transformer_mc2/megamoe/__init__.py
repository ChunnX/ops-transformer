# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend MegaMoE Beta. No framework import or device initialization on import."""

from .api import SymmBuffer, fp8_fp4_mega_moe, mega_moe, transform_weights_for_mega_moe
from ops_transformer_mc2._store import StoreGroup

__all__ = [
    "SymmBuffer",
    "StoreGroup",
    "mega_moe",
    "fp8_fp4_mega_moe",
    "transform_weights_for_mega_moe",
]
