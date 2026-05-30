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
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class UpdateContextByHcclChannelOpBuilder(OpBuilder):
    def __init__(self):
        super(UpdateContextByHcclChannelOpBuilder, self).__init__("update_context_by_hccl_channel")

    def sources(self):
        return ['ops/csrc/update_context_by_hccl_channel.cpp']

    def schema(self) -> str:
        return "update_context_by_hccl_channel(str group_ep, int ep_world_size, " \
               "int ccl_buffer_size, Tensor context) -> bool"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def update_context_by_hccl_channel_meta(group_ep, ep_world_size, ccl_buffer_size, context):
            return False


update_context_by_hccl_channel_op_builder = UpdateContextByHcclChannelOpBuilder()
op_module = update_context_by_hccl_channel_op_builder.load()


@impl(AS_LIBRARY, update_context_by_hccl_channel_op_builder.name, "PrivateUse1")
def update_context_by_hccl_channel(group_ep, ep_world_size, ccl_buffer_size, context):
    return op_module.update_context_by_hccl_channel(
        group_ep, ep_world_size, ccl_buffer_size, context)
