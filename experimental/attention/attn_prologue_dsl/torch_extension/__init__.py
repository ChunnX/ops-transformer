# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""将算子导出到共享 ds41 命名空间及 ops 顶层。"""

import sys
import types

import torch

from .attn_prologue import attn_prologue as _attn_prologue_impl


def prepare_attn_prologue(*args, **kwargs):
    """准备可重复执行的 DSL 调用；输入或权重改变后需重新准备。"""
    from ._kernel import prepare_attn_prologue as prepare

    return prepare(*args, **kwargs)


def to_nz(weight):
    """在模型加载时将一个投影权重转换为 FRACTAL_NZ。"""
    from ._kernel import to_nz as convert

    return convert(weight)


# 从当前包名推导父模块，同时支持整包、custom 和 vendor 单算子 wheel。
_parent_name = __name__.split(".ops.", 1)[0] + ".ops"
_ops_mod = sys.modules[_parent_name]
_ds41_name = _parent_name + ".ds41"
ds41 = sys.modules.get(_ds41_name)
if ds41 is None:
    ds41 = types.ModuleType(_ds41_name)
    ds41.__package__ = _parent_name
    sys.modules[_ds41_name] = ds41

attn_prologue = torch.ops.cann_ops_transformer.ds41.attn_prologue
attn_prologue_dsl = attn_prologue
ds41.attn_prologue = attn_prologue
_ops_mod.ds41 = ds41
_ops_mod.attn_prologue = attn_prologue

# prepare/to_nz 保留在算子子包，避免与其他算子的同名辅助函数冲突。
__all__ = ["ds41", "attn_prologue", "attn_prologue_dsl"]
