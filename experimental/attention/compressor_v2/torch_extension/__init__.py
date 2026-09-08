# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys
import types
from .compressor import compressor

# 构造独立模块 cpv2，使 from cann_ops_transformer.ops.cpv2 import compressor 可用。
# 父包从 sys.modules 取 ops 模块的 __name__，不依赖当前包的路径层级。
# 本模块由 ops/__init__.py 自动发现触发 import，此时 ops 必已在 sys.modules 中。
parent_pkg = sys.modules["cann_ops_transformer.ops"].__name__
cpv2 = types.ModuleType(f"{parent_pkg}.cpv2")
cpv2.__package__ = parent_pkg
cpv2.compressor = compressor
sys.modules[f"{parent_pkg}.cpv2"] = cpv2
globals()["cpv2"] = cpv2

# 只暴露 cpv2，不暴露 compressor_v2 入口，避免与原 compressor 的 ops.compressor 冲突
__all__ = ["cpv2"]
