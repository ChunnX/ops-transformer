# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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


class _AllGatherQuantMatmulOpBuilder(OpBuilder):
    def __init__(self):
        super(_AllGatherQuantMatmulOpBuilder, self).__init__("npu_all_gather_quant_matmul")

    def sources(self):
        return ["ops/csrc/all_gather_quant_matmul.cpp"]

    def schema(self) -> str:
        return (
            "npu_all_gather_quant_matmul(Tensor context, Tensor x1, Tensor x2, int hccl_buffer_size, str group_name, int rank_size, *, "
            "Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, "
            "int[] group_sizes=[], "
            "int? x1_dtype=None, int? x2_dtype=None, "
            "int? x1_scale_dtype=None, int? x2_scale_dtype=None, "
            "int? y_dtype=None, str comm_mode=\"urma\""
            ") -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_all_gather_quant_matmul_meta(
            context,
            x1,
            x2,
            hccl_buffer_size,
            group_name,
            rank_size,
            bias=None,
            x1_scale=None,
            x2_scale=None,
            group_sizes=[],
            x1_dtype=None,
            x2_dtype=None,
            x1_scale_dtype=None,
            x2_scale_dtype=None,
            y_dtype=None,
            comm_mode="urma",
        ):
            torch._check(
                rank_size > 0,
                lambda: f"rank_size should be greater than 0, {ops_error(ErrCode.VALUE)}.",
            )
            torch._check(
                x1.dim() == 2,
                lambda: f"x1 should be 2D, but got {x1.dim()}D, {ops_error(ErrCode.VALUE)}.",
            )
            torch._check(
                x2.dim() == 2,
                lambda: f"x2 should be 2D, but got {x2.dim()}D, {ops_error(ErrCode.VALUE)}.",
            )
            m_per_rank = x1.size(0)
            k = x1.size(1)
            n = x2.size(1)  # x2 传 .t() view [K, N]，dim1=N
            out_dtype = torch.bfloat16
            if y_dtype is not None:
                out_dtype = _ACL_DTYPE_TO_TORCH_DTYPE.get(y_dtype, out_dtype)
            y = x1.new_empty(tuple([m_per_rank * rank_size, n]), dtype=out_dtype)
            gather_out = x1.new_empty(tuple([m_per_rank * rank_size, k]), dtype=x1.dtype)
            amax_out = x1.new_empty(tuple([0]), dtype=torch.float32)
            return (y, gather_out, amax_out)


_all_gather_quant_matmul_op_builder = _AllGatherQuantMatmulOpBuilder()
_op_module = _all_gather_quant_matmul_op_builder.load()


@impl(AS_LIBRARY, _all_gather_quant_matmul_op_builder.name, "PrivateUse1")
def _npu_all_gather_quant_matmul(
    context,
    x1,
    x2,
    hccl_buffer_size,
    group_name,
    rank_size,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=[],
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode="urma",
):
    result = _op_module.npu_all_gather_quant_matmul(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        rank_size,
        bias,
        x1_scale,
        x2_scale,
        group_sizes,
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        y_dtype,
        comm_mode,
    )
    return result


def all_gather_quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    group,
    *,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: int = -1,
    x2_dtype: int = -1,
    x1_scale_dtype: int = -1,
    x2_scale_dtype: int = -1,
    y_dtype: int = -1,
    comm_mode: Optional[str] = "urma",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AllGather + MX quantized Matmul fused computation (apace UDMA path).

    Each rank gathers A from all ranks, then multiplies with its local B:
    ``y = AllGather(x1) @ x2^T``.

    Args:
        x1 (Tensor): Left matrix (per-rank A), shape ``(M_per_rank, K)``, dtype
            float8_e4m3fn/float8_e5m2/float4_e2m1fn_x2.
        x2 (Tensor): Right matrix (transposed B), shape ``(N, K)``, dtype
            float8_e4m3fn/float8_e5m2/float4_e2m1fn_x2. FP4 requires x1 and x2
            both FP4.
        group (ProcessGroup): torch.distributed process group for AllGather
            communication.
        bias (Tensor, optional): Not supported yet, must be None.
        x1_scale (Tensor): MX scale for x1, shape ``(M_per_rank, Ceil(K/64), 2)``,
            dtype float8_e8m0fnu.
        x2_scale (Tensor): MX scale for x2, shape ``(N, Ceil(K/64), 2)``, dtype
            float8_e8m0fnu.
        group_sizes (List[int], optional): Quantization group sizes, MX only
            supports ``[1, 1, 32]``. Defaults to ``[1, 1, 32]``.
        x1_dtype (int): x1 dtype enum for bitcast, -1 means use tensor dtype.
        x2_dtype (int): x2 dtype enum for bitcast, -1 means use tensor dtype.
        x1_scale_dtype (int): x1_scale dtype enum, -1 means use tensor dtype.
        x2_scale_dtype (int): x2_scale dtype enum, -1 means use tensor dtype.
        y_dtype (int): Output dtype enum, -1 means default bfloat16
            (15=bfloat16, 5=float16).
        comm_mode (str): Communication engine, only ``"urma"`` is supported.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (y, gather_out, amax_out) where y has
            shape ``(M_per_rank * rank_size, N)``, gather_out has shape
            ``(M_per_rank * rank_size, K)``, amax_out is empty tensor (not
            implemented yet). Users typically use
            ``y, _, _ = all_gather_quant_matmul(...)`` to ignore the latter two.
    """
    if group_sizes is None:
        group_sizes = [1, 1, 32]
    torch._check(
        group_sizes == [1, 1, 32],
        lambda: f"group_sizes must be [1, 1, 32] for MX quant, got {group_sizes}.",
    )
    rank_id = torch.distributed.get_rank(group)
    rank_size = torch.distributed.get_world_size(group)
    if rank_size <= 0:
        rank_size = torch.distributed.get_rank_size()  # fallback to default group
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
        rank_id
    )

    channel_builder = CommChannelBuilderManager(group_name)
    ctx_tag = group_name + "all_gather_matmul"
    context = channel_builder.create_context(ctx_tag)
    hccl_buffer_size = channel_builder.get_hccl_buffer_size()

    print(f"[all_gather_quant_matmul] rank={rank_id}, rank_size={rank_size}, comm_mode={comm_mode}, "
          f"calling npu_all_gather_quant_matmul ...", flush=True)
    result = torch.ops.cann_ops_transformer.npu_all_gather_quant_matmul(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        rank_size,
        bias=bias,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=group_sizes,
        x1_dtype=None if x1_dtype == -1 else x1_dtype,
        x2_dtype=None if x2_dtype == -1 else x2_dtype,
        x1_scale_dtype=None if x1_scale_dtype == -1 else x1_scale_dtype,
        x2_scale_dtype=None if x2_scale_dtype == -1 else x2_scale_dtype,
        y_dtype=None if y_dtype == -1 else y_dtype,
        comm_mode=comm_mode,
    )
    print(f"[all_gather_quant_matmul] rank={rank_id}, npu_all_gather_quant_matmul returned", flush=True)
    return result
