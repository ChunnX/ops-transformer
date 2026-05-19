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
from npu_ops_transformer.op_builder.builder import OpBuilder


class UpdateContextOpBuilder(OpBuilder):
    def __init__(self):
        super(UpdateContextOpBuilder, self).__init__("update_context")

    def sources(self):
        return ['ops/csrc/update_context.cpp']

    def schema(self) -> str:
        return ""

    def register_meta(self):
        pass

update_context_op_builder = UpdateContextOpBuilder()
op_module = update_context_op_builder.load()

CommContextManager = op_module.CommContextManager
