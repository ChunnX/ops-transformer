# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, List, Tuple

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from .comm_channel_builder_manager import CommChannelBuilderManager


_ACL_DTYPE_TO_TORCH_DTYPE = {
    5: torch.float16,
    6: torch.float32,
    15: torch.bfloat16,
}


class _AllToAllQuantMatmulOpBuilder(OpBuilder):
    def __init__(self):
        super(_AllToAllQuantMatmulOpBuilder, self).__init__("npu_all_to_all_quant_matmul")

    def sources(self):
        return ["ops/csrc/all_to_all_quant_matmul.cpp"]

    def schema(self) -> str:
        return (
            "npu_all_to_all_quant_matmul(Tensor context, Tensor x1, Tensor x2, int hccl_buffer_size, str group_name, int world_size, *, "
            "bool alltoall_out_flag=True, "
            "Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, "
            "int? x1_quant_mode=None, int? x2_quant_mode=None, "
            "int[] group_sizes=[], "
            "int? x1_dtype=None, int? x2_dtype=None, "
            "int? x1_scale_dtype=None, int? x2_scale_dtype=None, "
            "int? y_dtype=None, str comm_mode=\"ai_cpu\", int perf_mode=0"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_all_to_all_quant_matmul_meta(
            context,
            x1,
            x2,
            hccl_buffer_size,
            group_name,
            world_size,
            alltoall_out_flag=True,
            bias=None,
            x1_scale=None,
            x2_scale=None,
            x1_quant_mode=None,
            x2_quant_mode=None,
            group_sizes=[],
            x1_dtype=None,
            x2_dtype=None,
            x1_scale_dtype=None,
            x2_scale_dtype=None,
            y_dtype=None,
            comm_mode="ai_cpu",
            perf_mode=0,
        ):
            torch._check(
                world_size > 0,
                lambda: f"world_size should be greater than 0, {ops_error(ErrCode.VALUE)}.",
            )
            torch._check(
                x1.dim() == 2,
                lambda: f"x1 should be 2D, but got {x1.dim()}D, {ops_error(ErrCode.VALUE)}.",
            )
            torch._check(
                x2.dim() == 2,
                lambda: f"x2 should be 2D, but got {x2.dim()}D, {ops_error(ErrCode.VALUE)}.",
            )
            bs = x1.size(0)
            h = x1.size(1)
            n = x2.size(1)
            local_bs = bs // world_size
            out_dtype = torch.float32
            if y_dtype is not None:
                out_dtype = _ACL_DTYPE_TO_TORCH_DTYPE.get(y_dtype, out_dtype)
            y = x1.new_empty(tuple([local_bs, n]), dtype=out_dtype)
            all2all_out = x1.new_empty(
                tuple([local_bs, h * world_size]), dtype=x1.dtype
            )
            return (y, all2all_out)


_all_to_all_quant_matmul_op_builder = _AllToAllQuantMatmulOpBuilder()
_op_module = _all_to_all_quant_matmul_op_builder.load()


@impl(AS_LIBRARY, _all_to_all_quant_matmul_op_builder.name, "PrivateUse1")
def _npu_all_to_all_quant_matmul(
    context,
    x1,
    x2,
    hccl_buffer_size,
    group_name,
    world_size,
    alltoall_out_flag=True,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    x1_quant_mode=None,
    x2_quant_mode=None,
    group_sizes=[],
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode="ai_cpu",
    perf_mode=0,
):
    result = _op_module.npu_all_to_all_quant_matmul(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        world_size,
        alltoall_out_flag,
        bias,
        x1_scale,
        x2_scale,
        x1_quant_mode,
        x2_quant_mode,
        group_sizes,
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        y_dtype,
        comm_mode,
        perf_mode,
    )
    return result


def all_to_all_quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    group,
    *,
    alltoall_out_flag: bool = True,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    x1_quant_mode: int = 0,
    x2_quant_mode: int = 0,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: int = -1,
    x2_dtype: int = -1,
    x1_scale_dtype: int = -1,
    x2_scale_dtype: int = -1,
    y_dtype: int = -1,
    comm_mode: Optional[str] = None,
    perf_mode: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """AlltoAll + Matmul fused computation.

    Fuses AlltoAll communication, Permute, and Matmul into a single operator
    (communication first, then computation). Supports non-quantized, K-C
    quantized, K-C dynamic quantized, and MX-quantized scenarios.

    Args:
        x1 (Tensor): Left matrix, shape ``(BS, H)``, dtype float16/bfloat16/
            int4/float8_e4m3fn/float8_e5m2/float4_e2m1.
        x2 (Tensor): Right matrix, shape ``(H*world_size, N)``.
        group (ProcessGroup): torch.distributed process group for AlltoAll
            communication.
        alltoall_out_flag (bool): Whether to output alltoall result.
        bias (Tensor, optional): Bias added after matmul, shape ``(N,)``.
        x1_scale (Tensor, optional): Left matrix quantization scale.
        x2_scale (Tensor, optional): Right matrix quantization scale.
        x1_quant_mode (int): Left matrix quantization mode (0=none, 2=perchannel,
            3=pertoken, 6=mx, 7=pertoken-dynamic).
        x2_quant_mode (int): Right matrix quantization mode.
        group_sizes (List[int], optional): Quantization group sizes as
            ``[group_m, group_n, group_k]``.
        x1_dtype (int): x1 dtype enum for bitcast, -1 means use tensor dtype.
        x2_dtype (int): x2 dtype enum for bitcast, -1 means use tensor dtype.
        x1_scale_dtype (int): x1_scale dtype enum, -1 means use tensor dtype.
        x2_scale_dtype (int): x2_scale dtype enum, -1 means use tensor dtype.
        y_dtype (int): Output dtype enum, -1 means default float32.
        comm_mode (str): Communication engine, e.g. ``"ai_cpu"`` or ``"ccu"``.
        perf_mode (int): Perf mode, 0 means default.

    Returns:
        Tuple[Tensor, Tensor]: (y, all2all_out) where y has shape
        ``(BS/world_size, N)`` and all2all_out has shape
        ``(BS/world_size, H*world_size)``.
    """
    rank_id = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    if world_size <= 0:
        world_size = torch.distributed.get_world_size()  # fallback to default group
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
        rank_id
    )

    channel_builder = CommChannelBuilderManager(group_name)
    ctx_tag = group_name + "allto_all_matmul"
    context = channel_builder.create_context(ctx_tag)
    hccl_buffer_size = channel_builder.get_hccl_buffer_size()

    print(f"[all_to_all_quant_matmul] rank={rank_id}, world_size={world_size}, comm_mode={comm_mode}, "
          f"calling npu_all_to_all_quant_matmul ...", flush=True)
    result = torch.ops.cann_ops_transformer.npu_all_to_all_quant_matmul(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        world_size,
        alltoall_out_flag=alltoall_out_flag,
        bias=bias,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        x1_quant_mode=None if x1_quant_mode == 0 else x1_quant_mode,
        x2_quant_mode=None if x2_quant_mode == 0 else x2_quant_mode,
        group_sizes=group_sizes if group_sizes is not None else [],
        x1_dtype=None if x1_dtype == -1 else x1_dtype,
        x2_dtype=None if x2_dtype == -1 else x2_dtype,
        x1_scale_dtype=None if x1_scale_dtype == -1 else x1_scale_dtype,
        x2_scale_dtype=None if x2_scale_dtype == -1 else x2_scale_dtype,
        y_dtype=None if y_dtype == -1 else y_dtype,
        comm_mode=comm_mode,
        perf_mode=perf_mode,
    )
    print(f"[all_to_all_quant_matmul] rank={rank_id}, npu_all_to_all_quant_matmul returned", flush=True)
    return result
