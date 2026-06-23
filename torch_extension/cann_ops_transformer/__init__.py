# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
from importlib import import_module


_ROOT_EXPORTS = {
    "qkv_rms_norm_rope_cache_with_k_scale": ".ops.qkv_rms_norm_rope_cache_with_k_scale",
    "qkv_rms_norm_rope_cache_with_k_scale_functional": ".ops.qkv_rms_norm_rope_cache_with_k_scale",
}

__all__ = ("ops",) + tuple(_ROOT_EXPORTS)


def __getattr__(name):
    if name == "ops":
        module = import_module(".ops", __name__)
        globals()[name] = module
        return module
    module_name = _ROOT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
