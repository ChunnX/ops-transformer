# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode
try:
    import torch
    import torch_npu
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge import attr
    from typing import Optional
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    @auto_convert_to_tensor([False, False, False, False],
        [True, True, False, False])
    def MoeReRouting(tokens: Tensor, expert_token_num_per_rank: Tensor,
                      per_token_scales: Optional[Tensor], expert_topk_weight: Optional[Tensor], *,
                      expert_token_num_type: int, idx_type: int,
                      dependencies=[], node_name=None):
        """REG_OP(MoeReRouting)\n
        .INPUT(tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .INPUT(expert_token_num_per_rank, TensorType({DT_INT32, DT_INT64}))\n
        .OPTIONAL_INPUT(per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OPTIONAL_INPUT(expert_topk_weight, TensorType({DT_FLOAT}))\n
        .OUTPUT(permute_tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .OUTPUT(permute_per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OUTPUT(permute_token_idx, TensorType({DT_INT32}))\n
        .OUTPUT(expert_token_num, TensorType({DT_INT32, DT_INT64}))\n
        .OUTPUT(permute_topk_weight, TensorType({DT_FLOAT}))\n
        .ATTR(expert_token_num_type, Int, 1)\n
        .ATTR(idx_type, Int, 0)\n
        """
        inputs = {
            "tokens": tokens,
            "expert_token_num_per_rank": expert_token_num_per_rank,
            "per_token_scales": per_token_scales,
            "expert_topk_weight": expert_topk_weight,
        }

        attrs = {
            "expert_token_num_type": attr.Int(expert_token_num_type),
            "idx_type": attr.Int(idx_type),
        }

        outputs = [
            "permute_tokens",
            "permute_per_token_scales",
            "permute_token_idx",
            "expert_token_num",
            "permute_topk_weight",
        ]

        return ge_op(
            op_type="MoeReRouting",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MoeReRouting") \
            .input("tokens", "DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2") \
            .input("expert_token_num_per_rank", "DT_INT32, DT_INT64") \
            .optional_input("per_token_scales", "DT_FLOAT, DT_FLOAT8_E8M0") \
            .optional_input("expert_topk_weight", "DT_FLOAT") \
            .attr("expert_token_num_type", attr.Int(1)) \
            .attr("idx_type", attr.Int(0)) \
            .output("permute_tokens", "DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2") \
            .output("permute_per_token_scales", "DT_FLOAT, DT_FLOAT8_E8M0") \
            .output("permute_token_idx", "DT_INT32") \
            .output("expert_token_num", "DT_INT32, DT_INT64") \
            .output("permute_topk_weight", "DT_FLOAT")
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.npu_moe_re_routing.default)
    def convert_npu_moe_re_routing(
        tokens: Tensor,
        expert_token_num_per_rank: Tensor,
        *,
        per_token_scales: Tensor = None,
        expert_topk_weight: Tensor = None,
        expert_token_num_type: int = 1,
        idx_type: int = 0,
        meta_outputs: TensorSpec = None):
        return MoeReRouting(
            tokens=tokens,
            expert_token_num_per_rank=expert_token_num_per_rank,
            per_token_scales=per_token_scales,
            expert_topk_weight=expert_topk_weight,
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type)
