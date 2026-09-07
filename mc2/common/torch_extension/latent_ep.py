# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
    15: torch.bfloat16,
    16: torch.quint4x2,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    285: torch.uint8,  # torch_npu.int4 use torch.uint8
    290: torch.uint8,  # torch_npu.hifloat8 use torch.uint8
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
    293: torch.uint8,  # torch_npu.float8_e8m0 use torch.uint8
    296: torch.uint8,  # torch_npu.float4_e2m1fn_x2 use torch.uint8
    297: torch.uint8,  # torch_npu.float4_e1m2fn_x2 use torch.uint8
}


class LatentEpOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("latent_ep_dispatch", category="mc2")

    def sources(self):
        return ["csrc/mc2/latent_ep.cpp"]

    def schema(self):
        dispatch_schema = (
            "latent_ep_dispatch(Tensor context, Tensor x, Tensor expert_ids, "
            "int ep_world_size, int ep_rank_id, int moe_expert_num, int ccl_buffer_size, "
            "*, Tensor? scales=None, Tensor? x_active_mask=None, "
            "Tensor? expert_scales=None, Tensor? elastic_info=None, Tensor? performance_info=None, "
            "Tensor? shared_expert_input=None, int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, "
            "int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=1, "
            'str comm_alg="", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0, '
            "int? moe_expert_dtype=None, int? x_dtype=None, int? x_smooth_scales_dtype=None) "
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )
        combine_schema = (
            "latent_ep_combine(Tensor context, Tensor expand_x, Tensor expert_ids, "
            "Tensor assist_info_for_combine, Tensor ep_send_counts, Tensor expert_scales, "
            "int ep_world_size, int ep_rank_id, int moe_expert_num, int ccl_buffer_size, "
            "*, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, "
            "Tensor? expand_scales=None, Tensor? shared_expert_x=None, Tensor? elastic_info=None, "
            "Tensor? ori_x=None, Tensor? const_expert_alpha_1=None, Tensor? const_expert_alpha_2=None, "
            "Tensor? const_expert_v=None, Tensor? performance_info=None, int tp_world_size=0, "
            "int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, "
            'int global_bs=0, int comm_quant_mode=0, str comm_alg="", int zero_expert_num=0, '
            "int copy_expert_num=0, int const_expert_num=0, bool enable_independent_shared_expert=False) -> (Tensor, Tensor)"
        )
        return [dispatch_schema, combine_schema]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        def has_tensor(tensor):
            return tensor is not None

        def get_dtype_from_enum(dtype):
            return TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[dtype]

        def is_packed_float4(dtype):
            return dtype in (296, 297)

        def get_dispatch_dynamic_scales_dtype(
            x, scales, x_smooth_scales_dtype, quant_mode
        ):
            dynamic_scales_dtype = torch.float32
            if quant_mode == 0:
                if (
                    x.dtype != torch.bfloat16
                    and x.dtype != torch.float16
                    and scales is not None
                ):
                    dynamic_scales_dtype = (
                        get_dtype_from_enum(x_smooth_scales_dtype)
                        if x_smooth_scales_dtype is not None
                        else scales.dtype
                    )
            elif quant_mode == 4 or quant_mode == 5:
                dynamic_scales_dtype = torch.uint8  # float8_e8m0
            return dynamic_scales_dtype

        def get_dispatch_dynamic_shape(scales, quant_mode, a, h):
            shape = tuple([a])
            if quant_mode == 0 and scales is not None:
                if scales.dim() < 2:
                    raise RuntimeError(
                        f"Expected scales to be at least 2-d, but got {scales.dim()}-d."
                    )
                shape = tuple([a, scales.shape[1]])
            elif quant_mode == 2:
                shape = tuple([a])
            elif quant_mode == 3:
                shape = tuple([a, math.ceil(h / 128)])
            elif quant_mode == 4 or quant_mode == 5:
                shape = tuple([a, (math.ceil(h / 32) + 1) // 2 * 2])
            return shape

        def check_dispatch_meta_args(
            ep_world_size,
            ep_rank_id,
            shared_expert_num,
            shared_expert_rank_num,
            expert_token_nums_type,
        ):
            torch._check(
                (ep_rank_id >= 0) and (ep_rank_id < ep_world_size),
                lambda: (
                    f"ep_rank_id should be in [0, ep_world_size), "
                    f"but got {ep_world_size=}, {ep_rank_id=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (shared_expert_rank_num >= 0)
                and (shared_expert_rank_num < ep_world_size),
                lambda: (
                    f"shared_expert_rank_num should be in [0, ep_world_size), "
                    f"but got {ep_world_size=}, {shared_expert_rank_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            is_shared_default = (shared_expert_num == 1) and (
                shared_expert_rank_num == 0
            )
            is_no_shared = (shared_expert_num == 0) and (shared_expert_rank_num == 0)
            is_valid_shared = (
                (shared_expert_num > 0)
                and ((shared_expert_rank_num // shared_expert_num) > 0)
                and ((shared_expert_rank_num % shared_expert_num) == 0)
            )
            torch._check(
                is_shared_default or is_no_shared or is_valid_shared,
                lambda: (
                    f"shared expert setting invalid, "
                    f"got {shared_expert_num=}, {shared_expert_rank_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                expert_token_nums_type in [0, 1],
                lambda: "the expert_token_nums_type should be 0 or 1"
                + ops_error(ErrCode.VALUE),
            )
            return is_shared_default, is_no_shared

        @impl(get_as_library(), "latent_ep_dispatch", "Meta")
        def npu_moe_distribute_dispatch_meta(
            context,
            x,
            expert_ids,
            ep_world_size,
            ep_rank_id,
            moe_expert_num,
            ccl_buffer_size,
            scales=None,
            x_active_mask=None,
            expert_scales=None,
            elastic_info=None,
            performance_info=None,
            shared_expert_input=None,
            tp_world_size=0,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=0,
            quant_mode=0,
            global_bs=0,
            expert_token_nums_type=1,
            comm_alg="",
            zero_expert_num=0,
            copy_expert_num=0,
            const_expert_num=0,
            moe_expert_dtype=None,
            x_dtype=None,
            x_smooth_scales_dtype=None,
        ):
            is_shared_default, is_no_shared = check_dispatch_meta_args(
                ep_world_size,
                ep_rank_id,
                shared_expert_num,
                shared_expert_rank_num,
                expert_token_nums_type,
            )

            bs = x.size(0)
            h = x.size(1)
            k = expert_ids.size(1)
            shared_front = expert_shard_type == 0
            out_dtype = torch.float8_e4m3fn if quant_mode in (4, 5) else torch.int8
            local_moe_expert_num = 1
            global_bs_real = 0
            if global_bs == 0:
                global_bs_real = bs * ep_world_size
            else:
                global_bs_real = global_bs
            a = 0
            if shared_front:
                if ep_rank_id < shared_expert_rank_num:
                    local_moe_expert_num = 1
                    max_bs = global_bs_real // ep_world_size
                    rank_num_per_shared_expert = (
                        shared_expert_rank_num // shared_expert_num
                    )
                    max_shared_group_num = (
                        ep_world_size + rank_num_per_shared_expert - 1
                    ) // rank_num_per_shared_expert
                    a = max_bs * max_shared_group_num
                else:
                    local_moe_expert_num = moe_expert_num // (
                        ep_world_size - shared_expert_rank_num
                    )
                    a = global_bs_real * min(local_moe_expert_num, k)
                if elastic_info is not None:
                    if (is_shared_default) or (is_no_shared):
                        local_moe_expert_num = max(
                            local_moe_expert_num,
                            moe_expert_num // (ep_world_size - shared_expert_rank_num),
                        )
                        a = global_bs_real * min(local_moe_expert_num, k)
                    else:
                        max_bs = global_bs_real // ep_world_size
                        rank_num_per_shared_expert = (
                            shared_expert_rank_num // shared_expert_num
                        )
                        max_shared_group_num = (
                            ep_world_size + rank_num_per_shared_expert - 1
                        ) // rank_num_per_shared_expert
                        a = max(
                            max_bs * max_shared_group_num,
                            global_bs_real
                            * min(
                                moe_expert_num
                                // (ep_world_size - shared_expert_rank_num),
                                k,
                            ),
                        )
                        local_moe_expert_num = max(
                            local_moe_expert_num,
                            moe_expert_num // (ep_world_size - shared_expert_rank_num),
                        )

            ep_recv_cnt_num = 0
            if tp_world_size == 2:
                ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size
            else:
                ep_recv_cnt_num = ep_world_size * local_moe_expert_num
            if quant_mode == 0:
                out_dtype = x.dtype
            if moe_expert_dtype is not None:
                out_dtype = get_dtype_from_enum(moe_expert_dtype)

            independent_shared_rank = (
                shared_expert_input is not None and ep_rank_id < shared_expert_rank_num
            )
            if independent_shared_rank:
                h *= 2
                out_dtype = x.dtype
            assist_info_for_combine_shape = max(bs * k, a * 128)
            expand_x_shape = tuple([max(a, a * tp_world_size), h])
            if (
                not independent_shared_rank
                and moe_expert_dtype is not None
                and is_packed_float4(moe_expert_dtype)
                and scales is None
            ):
                torch._check(
                    (h % 2) == 0,
                    lambda: "h must be divisible by 2 when moe_expert_dtype is packed float4.",
                )
                expand_x_shape = tuple([max(a, a * tp_world_size), h // 2])
            expand_x = x.new_empty(expand_x_shape, dtype=out_dtype)
            dynamic_scales_dtype = get_dispatch_dynamic_scales_dtype(
                x, scales, x_smooth_scales_dtype, quant_mode
            )
            dynamic_scales_shape = get_dispatch_dynamic_shape(
                scales, quant_mode, max(a, a * tp_world_size), h
            )
            dynamic_scales = x.new_empty(
                dynamic_scales_shape, dtype=dynamic_scales_dtype
            )
            if independent_shared_rank:
                dynamic_scales = x.new_empty((a,), dtype=torch.float32)
            expert_token_nums = x.new_empty((local_moe_expert_num), dtype=torch.int64)
            ep_recv_counts = x.new_empty((ep_recv_cnt_num), dtype=torch.int32)
            tp_recv_counts = x.new_empty((tp_world_size), dtype=torch.int32)
            expand_scales = x.new_empty((0), dtype=torch.float32)
            if has_tensor(expert_scales):
                expand_scales = x.new_empty((a), dtype=torch.float32)
            expand_idx = x.new_empty(assist_info_for_combine_shape, dtype=torch.int32)
            return (
                expand_x,
                dynamic_scales,
                expand_idx,
                expert_token_nums,
                ep_recv_counts,
                tp_recv_counts,
                expand_scales,
            )

        @impl(get_as_library(), "latent_ep_combine", "Meta")
        def npu_moe_distribute_combine_meta(
            context,
            expand_x,
            expert_ids,
            assist_info_for_combine,
            ep_send_counts,
            expert_scales,
            ep_world_size,
            ep_rank_id,
            moe_expert_num,
            ccl_buffer_size,
            tp_send_counts=None,
            x_active_mask=None,
            expand_scales=None,
            shared_expert_x=None,
            elastic_info=None,
            ori_x=None,
            const_expert_alpha_1=None,
            const_expert_alpha_2=None,
            const_expert_v=None,
            performance_info=None,
            tp_world_size=0,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=0,
            global_bs=0,
            comm_quant_mode=0,
            comm_alg="",
            zero_expert_num=0,
            copy_expert_num=0,
            const_expert_num=0,
            enable_independent_shared_expert=False,
        ):
            h = expand_x.size(1)
            if enable_independent_shared_expert and ep_rank_id < shared_expert_rank_num:
                h //= 2
            output = expand_x.new_empty((expert_ids.size(0), h))
            shared = expand_x.new_empty(
                (expert_ids.size(0), h * 2)
                if enable_independent_shared_expert
                else (0,)
            )
            return output, shared


latent_ep_op_builder = LatentEpOpBuilder()
latent_ep_op_builder._ensure_initialized()


@impl(get_as_library(), "latent_ep_dispatch", "PrivateUse1")
def _dispatch(
    context,
    x,
    expert_ids,
    ep_world_size,
    ep_rank_id,
    moe_expert_num,
    ccl_buffer_size,
    scales=None,
    x_active_mask=None,
    expert_scales=None,
    elastic_info=None,
    performance_info=None,
    shared_expert_input=None,
    tp_world_size=0,
    tp_rank_id=0,
    expert_shard_type=0,
    shared_expert_num=1,
    shared_expert_rank_num=0,
    quant_mode=0,
    global_bs=0,
    expert_token_nums_type=1,
    comm_alg="",
    zero_expert_num=0,
    copy_expert_num=0,
    const_expert_num=0,
    moe_expert_dtype=None,
    x_dtype=None,
    x_smooth_scales_dtype=None,
):
    op_module = latent_ep_op_builder.load()
    return op_module.dispatch(
        context,
        x,
        expert_ids,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        ccl_buffer_size,
        scales,
        x_active_mask,
        expert_scales,
        elastic_info,
        performance_info,
        shared_expert_input,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        quant_mode,
        global_bs,
        expert_token_nums_type,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
        moe_expert_dtype,
        x_dtype,
        x_smooth_scales_dtype,
    )


@impl(get_as_library(), "latent_ep_combine", "PrivateUse1")
def _combine(
    context,
    expand_x,
    expert_ids,
    assist_info_for_combine,
    ep_send_counts,
    expert_scales,
    ep_world_size,
    ep_rank_id,
    moe_expert_num,
    ccl_buffer_size,
    tp_send_counts=None,
    x_active_mask=None,
    expand_scales=None,
    shared_expert_x=None,
    elastic_info=None,
    ori_x=None,
    const_expert_alpha_1=None,
    const_expert_alpha_2=None,
    const_expert_v=None,
    performance_info=None,
    tp_world_size=0,
    tp_rank_id=0,
    expert_shard_type=0,
    shared_expert_num=1,
    shared_expert_rank_num=0,
    global_bs=0,
    comm_quant_mode=0,
    comm_alg="",
    zero_expert_num=0,
    copy_expert_num=0,
    const_expert_num=0,
    enable_independent_shared_expert=False,
):
    op_module = latent_ep_op_builder.load()
    return op_module.combine(
        context,
        expand_x,
        expert_ids,
        assist_info_for_combine,
        ep_send_counts,
        expert_scales,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        ccl_buffer_size,
        tp_send_counts,
        x_active_mask,
        expand_scales,
        shared_expert_x,
        elastic_info,
        ori_x,
        const_expert_alpha_1,
        const_expert_alpha_2,
        const_expert_v,
        performance_info,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        global_bs,
        comm_quant_mode,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
        enable_independent_shared_expert,
    )


dispatch = torch.ops.cann_ops_transformer.latent_ep_dispatch.default
combine = torch.ops.cann_ops_transformer.latent_ep_combine.default


def get_ccl_buffer_size(
    ep_world_size,
    max_bs,
    hidden,
    moe_expert_num,
    topk,
    shared_expert_num,
    shared_expert_rank_num,
    quant_mode=0,
    moe_expert_dtype=None,
):
    """Return bytes for both 0/1 partitions, state and per-rank address offsets."""

    def align(value, base):
        return (value + base - 1) // base * base

    if not (
        2 <= ep_world_size <= 1024
        and 1 <= max_bs <= 512
        and 1024 <= hidden <= 4096
        and 1 <= topk <= 16
    ):
        raise ValueError("Invalid EP, max_bs, latent H or topk.")
    if not (
        0 < shared_expert_num <= 4
        and 0 < shared_expert_rank_num < ep_world_size
        and shared_expert_rank_num % shared_expert_num == 0
    ):
        raise ValueError("Invalid shared expert rank groups.")
    moe_ranks = ep_world_size - shared_expert_rank_num
    if moe_expert_num <= 0 or moe_expert_num % moe_ranks:
        raise ValueError("moe_expert_num must be divisible by the number of MoE ranks.")
    local_experts = moe_expert_num // moe_ranks
    if local_experts * ep_world_size > 2048:
        raise ValueError(
            "local_experts * ep_world_size exceeds the existing state table limit."
        )
    if quant_mode not in (0, 1, 2, 3, 4, 5):
        raise ValueError("quant_mode must be in [0, 5].")
    if quant_mode in (1, 2, 3) and moe_expert_dtype is None:
        raise ValueError("moe_expert_dtype is required for quant_mode 1, 2, 3.")
    if quant_mode in (4, 5) and moe_expert_dtype not in (None, 24, 292):
        raise ValueError("MX requires E4M3FN.")
    allowed_types = {
        0: (None, 5, 15),
        1: (1, 290),
        2: (1, 23, 24, 291, 292),
        3: (23, 24, 291, 292),
        4: (None, 24, 292),
        5: (None, 24, 292),
    }
    if moe_expert_dtype not in allowed_types[quant_mode]:
        raise ValueError(
            "moe_expert_dtype is not supported for this independent dispatch quant_mode."
        )
    token_bytes = align(hidden * 2 if quant_mode == 0 else hidden, 32)
    if quant_mode == 2:
        token_bytes += 4
    elif quant_mode == 3:
        token_bytes = align(hidden, 128) + (hidden + 127) // 128 * 4
    elif quant_mode in (4, 5):
        token_bytes = align(hidden, 256) + align((hidden + 31) // 32, 2)
    moe_dispatch = (align(token_bytes, 32) + 32 + 479) // 480 * 512
    shared_dispatch = (align(hidden * 4, 32) + 32 + 479) // 480 * 512
    dispatch_bytes = (
        max_bs * ep_world_size * max(local_experts * moe_dispatch, shared_dispatch)
    )
    moe_combine = (hidden * 2 + 479) // 480 * 512
    shared_combine = (hidden * 4 + 479) // 480 * 512
    combine_bytes = max_bs * (topk * moe_combine + shared_expert_num * shared_combine)
    return align(
        (dispatch_bytes + combine_bytes) * 2 + 1024 * 1024 + ep_world_size * 1024, 512
    )


# Graph converters live beside the eager and Meta implementations.
try:
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        torch_dtype_value_to_ge_type,
        torch_dtype_value_to_ge_proto_type,
        _ge_dtype_to_ge_proto_dtype as ge_dtype_to_ge_proto_dtype,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.latent_ep_dispatch.default
    )
    def _convert_dispatch(
        context,
        x,
        expert_ids,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        ccl_buffer_size,
        *,
        scales=None,
        x_active_mask=None,
        expert_scales=None,
        elastic_info=None,
        performance_info=None,
        shared_expert_input=None,
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        quant_mode=0,
        global_bs=0,
        expert_token_nums_type=1,
        comm_alg="",
        zero_expert_num=0,
        copy_expert_num=0,
        const_expert_num=0,
        moe_expert_dtype=None,
        x_dtype=None,
        x_smooth_scales_dtype=None,
        meta_outputs=None,
    ):
        if (
            shared_expert_input is not None
            and quant_mode in (1, 2, 3)
            and moe_expert_dtype is None
        ):
            raise ValueError("moe_expert_dtype is required for quant_mode 1, 2, 3.")
        if x_dtype is not None:
            x = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
            x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
        if scales is not None and x_smooth_scales_dtype is not None:
            scales = ge.Bitcast(
                scales, type=torch_dtype_value_to_ge_type(x_smooth_scales_dtype)
            )
            scales.desc.dtype = torch_dtype_value_to_ge_proto_type(
                x_smooth_scales_dtype
            )
        moe_dtype = (
            torch_dtype_value_to_ge_type(moe_expert_dtype)
            if moe_expert_dtype is not None
            else (
                x.dtype
                if quant_mode == 0
                else (
                    DataType.DT_FLOAT8_E4M3FN
                    if quant_mode in (4, 5)
                    else DataType.DT_INT8
                )
            )
        )
        inputs = {
            "context": context,
            "x": x,
            "expert_ids": expert_ids,
            "scales": scales,
            "x_active_mask": x_active_mask,
            "expert_scales": expert_scales,
            "elastic_info": elastic_info,
            "performance_info": performance_info,
            "shared_expert_input": shared_expert_input,
        }
        attributes = {
            "ep_world_size": attr.Int(ep_world_size),
            "ep_rank_id": attr.Int(ep_rank_id),
            "moe_expert_num": attr.Int(moe_expert_num),
            "ccl_buffer_size": attr.Int(ccl_buffer_size),
            "tp_world_size": attr.Int(tp_world_size),
            "tp_rank_id": attr.Int(tp_rank_id),
            "expert_shard_type": attr.Int(expert_shard_type),
            "shared_expert_num": attr.Int(shared_expert_num),
            "shared_expert_rank_num": attr.Int(shared_expert_rank_num),
            "quant_mode": attr.Int(quant_mode),
            "global_bs": attr.Int(global_bs),
            "expert_token_nums_type": attr.Int(expert_token_nums_type),
            "comm_alg": attr.Str(comm_alg),
            "zero_expert_num": attr.Int(zero_expert_num),
            "copy_expert_num": attr.Int(copy_expert_num),
            "const_expert_num": attr.Int(const_expert_num),
            "y_dtype": attr.Int(moe_dtype),
        }
        ir = IrDef("MoeDistributeDispatchV3")
        ir.input("context", "DT_INT32")
        ir.input("x", "DT_BF16, DT_FLOAT16")
        ir.input("expert_ids", "DT_INT32")
        ir.optional_input("scales", "DT_FLOAT, DT_FLOAT8_E8M0")
        ir.optional_input("x_active_mask", "DT_BOOL")
        ir.optional_input("expert_scales", "DT_FLOAT, DT_FLOAT8_E8M0")
        ir.optional_input("elastic_info", "DT_INT32")
        ir.optional_input("performance_info", "DT_INT64")
        ir.optional_input("shared_expert_input", "DT_BF16, DT_FLOAT16")
        ir.required_attr("ep_world_size", attr.Int)
        ir.required_attr("ep_rank_id", attr.Int)
        ir.required_attr("moe_expert_num", attr.Int)
        ir.required_attr("ccl_buffer_size", attr.Int)
        ir.attr("tp_world_size", attr.Int(0))
        ir.attr("tp_rank_id", attr.Int(0))
        ir.attr("expert_shard_type", attr.Int(0))
        ir.attr("shared_expert_num", attr.Int(1))
        ir.attr("shared_expert_rank_num", attr.Int(0))
        ir.attr("quant_mode", attr.Int(0))
        ir.attr("global_bs", attr.Int(0))
        ir.attr("expert_token_nums_type", attr.Int(1))
        ir.attr("comm_alg", attr.Str(""))
        ir.attr("zero_expert_num", attr.Int(0))
        ir.attr("copy_expert_num", attr.Int(0))
        ir.attr("const_expert_num", attr.Int(0))
        ir.attr("y_dtype", attr.Int(28))
        ir.output(
            "expand_x",
            "DT_BF16, DT_FLOAT16, DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2",
        )
        ir.output("dynamic_scales", "DT_FLOAT, DT_FLOAT8_E8M0")
        ir.output("assist_info_for_combine", "DT_INT32")
        ir.output("expert_token_nums", "DT_INT64")
        ir.output("ep_recv_count", "DT_INT32")
        ir.output("tp_recv_count", "DT_INT32")
        ir.output("expand_scales", "DT_FLOAT")
        outputs = ge_op(
            op_type="MoeDistributeDispatchV3",
            inputs=inputs,
            attrs=attributes,
            outputs=[
                "expand_x",
                "dynamic_scales",
                "assist_info_for_combine",
                "expert_token_nums",
                "ep_recv_count",
                "tp_recv_count",
                "expand_scales",
            ],
            ir=ir,
        )
        shared_rank = (
            shared_expert_input is not None and ep_rank_id < shared_expert_rank_num
        )
        output_dtype = x.dtype if shared_rank else moe_dtype
        outputs[0].desc.dtype = ge_dtype_to_ge_proto_dtype(output_dtype)
        scale_dtype = (
            DataType.DT_FLOAT
            if shared_rank or quant_mode not in (4, 5)
            else DataType.DT_FLOAT8_E8M0
        )
        outputs[1].desc.dtype = ge_dtype_to_ge_proto_dtype(scale_dtype)
        if not shared_rank and moe_expert_dtype in (296, 297):
            shape = ge.Shape(outputs[0])
            half_shape = ge.Div(shape, ge.Const([1, 2], dtype=DataType.DT_INT32))
            packed_shape = ge.ConcatV2(
                [half_shape, ge.Const([2], dtype=DataType.DT_INT32)], concat_dim=0, N=2
            )
            outputs = (
                ge.Reshape(
                    ge.Bitcast(
                        ge.Reshape(outputs[0], packed_shape), type=DataType.DT_UINT8
                    ),
                    half_shape,
                ),
                *outputs[1:],
            )
        return tuple(outputs)

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.latent_ep_combine.default
    )
    def _convert_combine(
        context,
        expand_x,
        expert_ids,
        assist_info_for_combine,
        ep_send_counts,
        expert_scales,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        ccl_buffer_size,
        *,
        tp_send_counts=None,
        x_active_mask=None,
        expand_scales=None,
        shared_expert_x=None,
        elastic_info=None,
        ori_x=None,
        const_expert_alpha_1=None,
        const_expert_alpha_2=None,
        const_expert_v=None,
        performance_info=None,
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        global_bs=0,
        comm_quant_mode=0,
        comm_alg="",
        zero_expert_num=0,
        copy_expert_num=0,
        const_expert_num=0,
        enable_independent_shared_expert=False,
        meta_outputs=None,
    ):
        inputs = {
            "context": context,
            "expand_x": expand_x,
            "expert_ids": expert_ids,
            "assist_info_for_combine": assist_info_for_combine,
            "ep_send_counts": ep_send_counts,
            "expert_scales": expert_scales,
            "tp_send_counts": tp_send_counts,
            "x_active_mask": x_active_mask,
            "activation_scale": None,
            "weight_scale": None,
            "group_list": None,
            "expand_scales": expand_scales,
            "shared_expert_x": shared_expert_x,
            "elastic_info": elastic_info,
            "ori_x": ori_x,
            "const_expert_alpha_1": const_expert_alpha_1,
            "const_expert_alpha_2": const_expert_alpha_2,
            "const_expert_v": const_expert_v,
            "performance_info": performance_info,
        }
        attributes = {
            "ep_world_size": attr.Int(ep_world_size),
            "ep_rank_id": attr.Int(ep_rank_id),
            "moe_expert_num": attr.Int(moe_expert_num),
            "ccl_buffer_size": attr.Int(ccl_buffer_size),
            "tp_world_size": attr.Int(tp_world_size),
            "tp_rank_id": attr.Int(tp_rank_id),
            "expert_shard_type": attr.Int(expert_shard_type),
            "shared_expert_num": attr.Int(shared_expert_num),
            "shared_expert_rank_num": attr.Int(shared_expert_rank_num),
            "global_bs": attr.Int(global_bs),
            "out_dtype": attr.Int(0),
            "comm_quant_mode": attr.Int(comm_quant_mode),
            "group_list_type": attr.Int(0),
            "comm_alg": attr.Str(comm_alg),
            "zero_expert_num": attr.Int(zero_expert_num),
            "copy_expert_num": attr.Int(copy_expert_num),
            "const_expert_num": attr.Int(const_expert_num),
            "enable_independent_shared_expert": attr.Bool(
                enable_independent_shared_expert
            ),
        }
        ir = IrDef("MoeDistributeCombineV3")
        ir.input("context", "DT_INT32")
        ir.input("expand_x", "DT_BF16, DT_FLOAT16")
        ir.input("expert_ids", "DT_INT32")
        ir.input("assist_info_for_combine", "DT_INT32")
        ir.input("ep_send_counts", "DT_INT32")
        ir.input("expert_scales", "DT_FLOAT, DT_FLOAT8_E8M0")
        ir.optional_input("tp_send_counts", "DT_INT32")
        ir.optional_input("x_active_mask", "DT_BOOL")
        ir.optional_input("activation_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
        ir.optional_input("weight_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
        ir.optional_input("group_list", "DT_INT64")
        ir.optional_input("expand_scales", "DT_FLOAT, DT_FLOAT8_E8M0")
        ir.optional_input("shared_expert_x", "DT_BF16, DT_FLOAT16")
        ir.optional_input("elastic_info", "DT_INT32")
        ir.optional_input("ori_x", "DT_BF16, DT_FLOAT16")
        ir.optional_input("const_expert_alpha_1", "DT_BF16, DT_FLOAT16")
        ir.optional_input("const_expert_alpha_2", "DT_BF16, DT_FLOAT16")
        ir.optional_input("const_expert_v", "DT_BF16, DT_FLOAT16")
        ir.optional_input("performance_info", "DT_INT64")
        ir.required_attr("ep_world_size", attr.Int)
        ir.required_attr("ep_rank_id", attr.Int)
        ir.required_attr("moe_expert_num", attr.Int)
        ir.required_attr("ccl_buffer_size", attr.Int)
        ir.attr("tp_world_size", attr.Int(0))
        ir.attr("tp_rank_id", attr.Int(0))
        ir.attr("expert_shard_type", attr.Int(0))
        ir.attr("shared_expert_num", attr.Int(1))
        ir.attr("shared_expert_rank_num", attr.Int(0))
        ir.attr("global_bs", attr.Int(0))
        ir.attr("out_dtype", attr.Int(0))
        ir.attr("comm_quant_mode", attr.Int(0))
        ir.attr("group_list_type", attr.Int(0))
        ir.attr("comm_alg", attr.Str(""))
        ir.attr("zero_expert_num", attr.Int(0))
        ir.attr("copy_expert_num", attr.Int(0))
        ir.attr("const_expert_num", attr.Int(0))
        ir.attr("enable_independent_shared_expert", attr.Bool(False))
        ir.output(
            "x",
            "DT_BF16, DT_FLOAT16, DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2",
        )
        ir.output(
            "shared_expert_out",
            "DT_BF16, DT_FLOAT16, DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2",
        )
        outputs = ge_op(
            op_type="MoeDistributeCombineV3",
            inputs=inputs,
            attrs=attributes,
            outputs=["x", "shared_expert_out"],
            ir=ir,
        )
        return tuple(outputs)
