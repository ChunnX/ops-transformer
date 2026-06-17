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
from torch_npu.utils._error_code import ErrCode, ops_error
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class BandwidthTestOpBuilder(OpBuilder):
    def __init__(self):
        super(BandwidthTestOpBuilder, self).__init__("npu_bandwidth_test")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/bandwidth_test.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_bandwidth_test(Tensor x, Tensor dstrank_id, " \
            "str group, int world_size, int max_bs, int mode, str comm_alg, int aiv_num) " \
            "-> (Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_bandwidth_test_meta(x, dstrank_id, group, world_size, max_bs, mode, comm_alg, aiv_num):
            torch._check(
                world_size > 0,
                lambda: (
                    f"world_size should be > 0, "
                    f"but got {world_size=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )

            bs = x.size(0)
            h = x.size(1)

            max_recv_cnt = max_bs * world_size
            y = x.new_empty((max_recv_cnt, h), dtype=x.dtype)
            receive_cnt = x.new_empty((world_size,), dtype=torch.int32)
            return (y, receive_cnt)


# Instantiate the builder
bandwidth_test_op_builder = BandwidthTestOpBuilder()
op_module = bandwidth_test_op_builder.load()


@impl(AS_LIBRARY, bandwidth_test_op_builder.name, "PrivateUse1")
def _npu_bandwidth_test(x, dstrank_id, group, world_size, max_bs, mode, comm_alg, aiv_num):
    """
    Implementation for NPU.
    'PrivateUse1' is the key for custom NPU backends.
    """
    return op_module.npu_bandwidth_test(x, dstrank_id, group, world_size, max_bs, mode, comm_alg, aiv_num)


def npu_bandwidth_test(x, dstrank_id, group, world_size, max_bs, mode, comm_alg, aiv_num):

    return torch.ops.npu_ops_transformer.npu_bandwidth_test(
        x, dstrank_id, group, world_size, max_bs, mode, comm_alg, aiv_num)