# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys

import torch
import torch_npu

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", ".."))
_MOE_EXT_ROOTS = [
    os.path.join(_REPO_ROOT, "moe", "moe_init_routing_v2", "torch_ops_extension"),
    os.path.join(_REPO_ROOT, "moe", "moe_init_routing_v2_grad", "torch_ops_extension"),
    os.path.join(_REPO_ROOT, "moe", "moe_finalize_routing_v2", "torch_ops_extension"),
    os.path.join(
        _REPO_ROOT, "moe", "moe_finalize_routing_v2_grad", "torch_ops_extension"
    ),
]
for _root in _MOE_EXT_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

from . import ops
from .ops import *
