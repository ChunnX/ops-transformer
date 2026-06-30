# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from cann_ops_transformer.op_builder.builder import OpBuilder


class CommContextOpBuilder(OpBuilder):
    def __init__(self):
        super(CommContextOpBuilder, self).__init__("comm_context")

    def sources(self):
        return ['ops/csrc/comm_context.cpp']

    def schema(self):
        return None

    def register_meta(self):
        pass

comm_context_op_builder = CommContextOpBuilder()
_comm_context_op_module = None
_comm_context_obj = None


def _get_comm_context_op_module():
    global _comm_context_op_module
    if _comm_context_op_module is None:
        _comm_context_op_module = comm_context_op_builder.load()
    return _comm_context_op_module


def __getattr__(name):
    if name == 'CommContextManager':
        global _comm_context_obj
        if _comm_context_obj is None:
            _comm_context_obj = _get_comm_context_op_module().CommContextManager
        globals()[name] = _comm_context_obj
        return _comm_context_obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
