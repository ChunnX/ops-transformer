# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it/or modify it under the terms and conditions of
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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class UrmaP2pBenchmarkOpBuilder(OpBuilder):
    def __init__(self):
        super(UrmaP2pBenchmarkOpBuilder, self).__init__("npu_urma_p2p_benchmark")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/urma_p2p_benchmark.cpp']

    def schema(self) -> str:
        """PyTorch operator signature. Dual-mode: SQE/Hcomm."""
        return "npu_urma_p2p_benchmark(Tensor x, " \
            "str group, int world_size, int max_bs, int mode, " \
            "int ccl_buffer_size, str comm_alg, int dst_rank_id, int aiv_num) " \
            "-> (Tensor, Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_urma_p2p_benchmark_meta(x, group, world_size, max_bs, mode,
                                         ccl_buffer_size, comm_alg, dst_rank_id, aiv_num):
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
            performance_info = x.new_empty((12,), dtype=torch.int64)  # 12 slots适配SQE(9+3)和Hcomm(5+7)
            return (y, receive_cnt, performance_info)


# Instantiate the builder
urma_p2p_benchmark_op_builder = UrmaP2pBenchmarkOpBuilder()
op_module = urma_p2p_benchmark_op_builder.load()


@impl(AS_LIBRARY, urma_p2p_benchmark_op_builder.name, "PrivateUse1")
def _npu_urma_p2p_benchmark(x, group, world_size, max_bs, mode,
                             ccl_buffer_size, comm_alg, dst_rank_id, aiv_num):
    """
    Implementation for NPU.
    'PrivateUse1' is the key for custom NPU backends.
    """
    return op_module.npu_urma_p2p_benchmark(x, group, world_size, max_bs, mode,
                                             ccl_buffer_size, comm_alg, dst_rank_id, aiv_num)

def npu_urma_p2p_benchmark(x, group, world_size, max_bs, mode,
                            ccl_buffer_size, comm_alg, dst_rank_id, aiv_num):

    return torch.ops.cann_ops_transformer.npu_urma_p2p_benchmark(
        x, group, world_size, max_bs, mode, ccl_buffer_size, comm_alg, dst_rank_id, aiv_num)
