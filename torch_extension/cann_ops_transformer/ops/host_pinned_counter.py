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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class HostPinnedCounterOpBuilder(OpBuilder):
    def __init__(self):
        super(HostPinnedCounterOpBuilder, self).__init__("host_pinned_counter")

    def sources(self):
        return ['ops/csrc/host_pinned_counter.cpp']
    
    def schema(self) -> str:
        return "host_pinned_counter() -> bool"

    def register_meta(self):
        pass

host_pinned_counter_op_builder = HostPinnedCounterOpBuilder()
op_module_host_pinned_counter = host_pinned_counter_op_builder.load()

HostPinnedCounter = op_module_host_pinned_counter.HostPinnedCounter