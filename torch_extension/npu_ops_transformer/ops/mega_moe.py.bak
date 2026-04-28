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


class MegaMoeOpBuilder(OpBuilder):
    def __init__(self):
        super(MegaMoeOpBuilder, self).__init__("npu_mega_moe")

    def sources(self):
        return ['ops/csrc/mega_moe.cpp']

    def schema(self) -> str:
        return "npu_mega_moe(Tensor context, Tensor x, Tensor topk_ids, Tensor topk_weights, " \
            "Tensor[] weight1, Tensor[] weight2, int moe_expert_num, int ep_world_size, int ccl_buffer_size, *, " \
            "Tensor[]? weight_scales1=None, Tensor[]? weight_scales2=None, Tensor? x_active_mask=None, " \
            "Tensor? scales=None, int? max_recv_token_num=0, int? dispatch_quant_mode=0, " \
            "int? combine_quant_mode=0, str? comm_alg=\"\", int? global_bs=0, " \
            "int? dispatch_quant_out_type=28, int? weight1_type=28, int? weight2_type=28) -> (Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mega_moe_meta(context, x, topk_ids, topk_weights, weight1, weight2,
                              moe_expert_num, ep_world_size, ccl_buffer_size,
                              weight_scales1=None, weight_scales2=None, x_active_mask=None,
                              scales=None, max_recv_token_num=0, dispatch_quant_mode=0,
                              combine_quant_mode=0, comm_alg="", global_bs=0,
                              dispatch_quant_out_type=28, weight1_type=28, weight2_type=28):
            torch._check(
                ep_world_size != 0,
                lambda: (
                    f"ep_rank_id should not be 0, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            bs = x.size(0)
            h = x.size(1)
            local_moe_expert_num = moe_expert_num // ep_world_size
            y = x.new_empty(tuple([bs, h]), dtype=x.dtype)
            expert_token_nums = x.new_empty((local_moe_expert_num), dtype=torch.int32)
            return (y, expert_token_nums)


# Instantiate the builder
mega_moe_op_builder = MegaMoeOpBuilder()
op_module = mega_moe_op_builder.load()


@impl(AS_LIBRARY, mega_moe_op_builder.name, "PrivateUse1")
def npu_mega_moe(context, x, topk_ids, topk_weights, weight1, weight2,
                 moe_expert_num, ep_world_size, ccl_buffer_size,
                 weight_scales1=None, weight_scales2=None,
                 x_active_mask=None, scales=None, max_recv_token_num=0,
                 dispatch_quant_mode=0, combine_quant_mode=0,
                 comm_alg="", global_bs=0,
                 dispatch_quant_out_type=28, weight1_type=28, weight2_type=28):

    return op_module.npu_mega_moe(
        context, x, topk_ids, topk_weights, weight1, weight2, moe_expert_num, ep_world_size,
        ccl_buffer_size, weight_scales1, weight_scales2, x_active_mask, scales, max_recv_token_num,
        dispatch_quant_mode, combine_quant_mode, comm_alg, global_bs,
        dispatch_quant_out_type, weight1_type, weight2_type)
