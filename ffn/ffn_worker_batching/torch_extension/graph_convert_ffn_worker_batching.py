# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TorchAir GE conversion for the FfnWorkerBatching ACLNN operator."""

from typing import List, Optional

try:
    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import Tensor, TensorSpec, auto_convert_to_tensor
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef, is_cann_compat
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

_FIRST_MX_TOKEN_DTYPE = 3
_FP4_TOKEN_DTYPE = 5
_OUTPUT_NAMES = (
    "y",
    "group_list",
    "session_ids",
    "micro_batch_ids",
    "token_ids",
    "expert_offsets",
    "dynamic_scale",
    "actual_token_num",
)


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False], [False])
    def FfnWorkerBatching(
        schedule_context: Tensor,
        *,
        expert_num: int,
        max_out_shape: List[int],
        token_dtype: int = 0,
        need_schedule: int = 0,
        layer_num: int = 0,
        sync_flag: bool = False,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(FfnWorkerBatching)
        .INPUT(schedule_context, TensorType({DT_INT8}))
        .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8,
                              DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
        .OUTPUT(group_list, TensorType({DT_INT64}))
        .OUTPUT(session_ids, TensorType({DT_INT32}))
        .OUTPUT(micro_batch_ids, TensorType({DT_INT32}))
        .OUTPUT(token_ids, TensorType({DT_INT32}))
        .OUTPUT(expert_offsets, TensorType({DT_INT32}))
        .OUTPUT(dynamic_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
        .OUTPUT(actual_token_num, TensorType({DT_INT64}))
        .REQUIRED_ATTR(expert_num, Int)
        .REQUIRED_ATTR(max_out_shape, ListInt)
        .ATTR(token_dtype, Int, 0)
        .ATTR(need_schedule, Int, 0)
        .ATTR(layer_num, Int, 0)
        .ATTR(sync_flag, Bool, false)
        """
        from .ffn_worker_batching import _checked_y_rows

        _checked_y_rows(max_out_shape)
        # ge_op normally checks only non-default optional attributes. Even when
        # sync_flag is False, require the UPDATED prototype: an older registration
        # has a shorter attribute array than the current tiling implementation.
        # Silently omitting the default would hide this ABI mismatch until launch.
        error = is_cann_compat(
            "FfnWorkerBatching",
            runtime_optional_inputs=(),
            runtime_optional_attrs=("sync_flag",),
        )
        if error:
            raise RuntimeError(
                "FfnWorkerBatching GE requires the updated CANN/OPP prototype "
                "containing sync_flag, including when sync_flag=False. " + error
            )

        return ge_op(
            op_type="FfnWorkerBatching",
            inputs={"schedule_context": schedule_context},
            attrs={
                "expert_num": attr.Int(expert_num),
                "max_out_shape": attr.ListInt(max_out_shape),
                "token_dtype": attr.Int(token_dtype),
                "need_schedule": attr.Int(need_schedule),
                "layer_num": attr.Int(layer_num),
                "sync_flag": attr.Bool(sync_flag),
            },
            outputs=list(_OUTPUT_NAMES),
            dependencies=[] if dependencies is None else dependencies,
            node_name=node_name,
            ir=IrDef("FfnWorkerBatching")
            .input("schedule_context", "DT_INT8")
            .required_attr("expert_num", attr.Int)
            .required_attr("max_out_shape", attr.ListInt)
            .attr("token_dtype", attr.Int(0))
            .attr("need_schedule", attr.Int(0))
            .attr("layer_num", attr.Int(0))
            .attr("sync_flag", attr.Bool(False))
            .output(
                "y",
                "DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .output("group_list", "DT_INT64")
            .output("session_ids", "DT_INT32")
            .output("micro_batch_ids", "DT_INT32")
            .output("token_ids", "DT_INT32")
            .output("expert_offsets", "DT_INT32")
            .output("dynamic_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .output("actual_token_num", "DT_INT64"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.ffn_worker_batching.default
    )
    def convert_ffn_worker_batching(
        schedule_context: Tensor,
        expert_num: int,
        max_out_shape: List[int],
        *,
        token_dtype: int = 0,
        need_schedule: int = 0,
        layer_num: int = 0,
        sync_flag: bool = False,
        mx_output_uint8: bool = False,
        meta_outputs: Optional[List[TensorSpec]] = None,
    ):
        """Convert the dispatcher call to GE, preserving attribute/output order.

        Requires the current FfnWorkerBatching OPP and TorchAir support for the
        mutable schema's auto_functionalized_v2 lowering. Referenced buffers must
        remain alive; external producers own their synchronization. Native FP4
        and MX uint8 transport currently require the npugraph_ex backend.
        """
        # This flag changes the Torch storage representation, not the GE/ACLNN
        # operator contract. Forwarding it as a GE attribute would be incorrect;
        # casting MX values to uint8 would also corrupt the raw exponent bytes.
        if token_dtype == _FP4_TOKEN_DTYPE or (
            mx_output_uint8 and token_dtype >= _FIRST_MX_TOKEN_DTYPE
        ):
            raise NotImplementedError(
                "FfnWorkerBatching GE does not support packed FP4 or MX uint8 "
                "output views; use npugraph_ex with mx_output_uint8=True. "
                f"Got token_dtype={token_dtype}, mx_output_uint8={mx_output_uint8}."
            )
        # Keep Tensor(a!) in the dispatcher schema. TorchAir functionalization
        # orders the updated context after this node; returning a ninth context
        # tensor here would break the actual eight-output GE operator contract.
        return FfnWorkerBatching(
            schedule_context,
            expert_num=expert_num,
            max_out_shape=max_out_shape,
            token_dtype=token_dtype,
            need_schedule=need_schedule,
            layer_num=layer_num,
            sync_flag=sync_flag,
        )
else:

    def convert_ffn_worker_batching(*args, **kwargs):
        """Report the optional GE dependency without breaking eager imports."""
        raise RuntimeError(
            "FfnWorkerBatching GE converter requires torchair, but torchair is not available."
        )
