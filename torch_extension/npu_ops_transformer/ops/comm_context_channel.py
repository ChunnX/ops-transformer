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
import torch
import torch_npu
from npu_ops_transformer.op_builder.builder import OpBuilder


class CommContextChannelOpBuilder(OpBuilder):
    def __init__(self):
        super(CommContextChannelOpBuilder, self).__init__("comm_context_channel")

    def sources(self):
        return ['ops/csrc/test.cpp']

    def schema(self):
        return None

    def register_meta(self):
        pass

    def extra_ldflags(self):
        flags = super().extra_ldflags()
        flags.append('-L' + os.path.join(self._cann_path, 'lib64'))
        flags.append('-lhcomm')
        return flags


comm_context_channel_op_builder = CommContextChannelOpBuilder()
op_module = comm_context_channel_op_builder.load()

CommContextManagerChannel = op_module.CommContextManager