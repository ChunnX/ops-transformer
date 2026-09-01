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
import types
import torch
import torch_npu


def _prepare_torchair_shim():
    """torchair 是可选依赖（图模式才需要）。torch_npu 2.6.0rc1 在 torchair 缺失时会向
    sys.modules 注入 _LazyTorchair 伪模块，且一旦访问其子模块就会把内置的
    torch_npu.dynamo.torchair 实现以 `torchair.*` 别名注册进 sys.modules（即使后续
    导入失败，别名也已残留）。各算子模块 `except ImportError` 的可选依赖保护只对
    ImportError 生效，而内置实现缺少 protobuf 等依赖时抛的是 pkg_resources 的
    DistributionNotFound（非 ImportError），会穿透保护直接中断导入。

    这里显式探测 torchair 是否真实可用：探测失败时清空所有 torchair* 别名条目，
    再放入“访问即抛标准 ImportError”的占位模块，使下游 guard 正常按 ImportError 处理。
    """
    def _torchair_real():
        try:
            import torchair
            from torchair.ge import Tensor  # 仅真实安装的 torchair 提供该子模块
            return True
        except Exception:
            return False

    if _torchair_real():
        return

    # 清除内置 torchair 导入链残留的 torchair.* 别名，避免子模块直接命中真实实现
    for _key in list(sys.modules):
        if _key == 'torchair' or _key.startswith('torchair.'):
            del sys.modules[_key]

    class _NoTorchair(types.ModuleType):
        # __file__ 须为非空字符串（Python 3.10 的 inspect.getfile 用真值判断，空串会被当成
        # built-in module），避免 inspect 等遍历 sys.modules 时抛 TypeError
        __file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_torchair_stub.py')
        __path__ = []  # 空包路径：子模块导入会抛 ModuleNotFoundError（ImportError 子类）

        def __getattr__(self, name):
            raise ImportError("torchair is not available in this environment: " + name)

    _no_torchair = _NoTorchair('torchair')
    _no_torchair.__package__ = 'torchair'
    _no_torchair.__spec__ = None
    sys.modules['torchair'] = _no_torchair


_prepare_torchair_shim()

from . import ops
from .ops import *
