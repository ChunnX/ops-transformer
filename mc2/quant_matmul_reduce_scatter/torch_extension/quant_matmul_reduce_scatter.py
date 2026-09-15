# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import logging
from typing import Optional, List

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommChannelBuilderManager

_logger = logging.getLogger(__name__)

_ACL_DTYPE_TO_TORCH_DTYPE = {
    15: torch.bfloat16,
}


class _QuantMatmulReduceScatterOpBuilder(OpBuilder):
    def __init__(self):
        super(_QuantMatmulReduceScatterOpBuilder, self).__init__(
            "npu_quant_matmul_reduce_scatter", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/quant_matmul_reduce_scatter.cpp"]

    def schema(self) -> str:
        return (
            "npu_quant_matmul_reduce_scatter(Tensor context, Tensor x1, Tensor x2, "
            "int hccl_buffer_size, str group_name, int rank_size, *, "
            "Tensor? x1_scale=None, Tensor? x2_scale=None, "
            "int[] group_sizes=[], "
            "int? x1_dtype=None, int? x2_dtype=None, "
            "int? x1_scale_dtype=None, int? x2_scale_dtype=None, "
            "int? y_dtype=None, str comm_mode=\"ai_cpu\""
            ") -> (Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_quant_matmul_reduce_scatter_meta(
            context,
            x1,
            x2,
            hccl_buffer_size,
            group_name,
            rank_size,
            x1_scale=None,
            x2_scale=None,
            group_sizes=None,
            x1_dtype=None,
            x2_dtype=None,
            x1_scale_dtype=None,
            x2_scale_dtype=None,
            y_dtype=None,
            comm_mode="ai_cpu",
        ):
            if rank_size <= 0:
                raise ValueError("rank_size should be greater than 0.")
            if x1.dim() != 2:
                raise ValueError(f"x1 should be 2D, but got {x1.dim()}D.")
            if x2.dim() != 2:
                raise ValueError(f"x2 should be 2D, but got {x2.dim()}D.")
            m = x1.size(0)
            n = x2.size(1)
            if m % rank_size != 0:
                raise ValueError(
                    f"x1.M ({m}) must be divisible by rank_size ({rank_size})."
                )
            out_dtype = torch.bfloat16
            if y_dtype is not None:
                out_dtype = _ACL_DTYPE_TO_TORCH_DTYPE.get(y_dtype, out_dtype)
            y = x1.new_empty(tuple([m // rank_size, n]), dtype=out_dtype)
            return (y,)


_quant_matmul_reduce_scatter_op_builder = _QuantMatmulReduceScatterOpBuilder()
_op_module = _quant_matmul_reduce_scatter_op_builder.load()


@impl(get_as_library(), _quant_matmul_reduce_scatter_op_builder.name, "PrivateUse1")
def _npu_quant_matmul_reduce_scatter(
    context,
    x1,
    x2,
    hccl_buffer_size,
    group_name,
    rank_size,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode="ai_cpu",
):
    result = _op_module.npu_quant_matmul_reduce_scatter(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        rank_size,
        x1_scale,
        x2_scale,
        group_sizes if group_sizes is not None else [],
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        y_dtype,
        comm_mode,
    )
    return result


def _check_params(
    x1,
    x2,
    x1_scale,
    x2_scale,
    x1_dtype,
    x2_dtype,
    x1_scale_dtype,
    x2_scale_dtype,
    comm_mode,
):
    """Validate input parameters."""
    if x1 is None:
        raise ValueError("x1 must not be None.")
    if x2 is None:
        raise ValueError("x2 must not be None.")
    if x1.dim() != 2:
        raise ValueError(f"x1 should be 2D, but got {x1.dim()}D.")
    if x2.dim() != 2:
        raise ValueError(f"x2 should be 2D, but got {x2.dim()}D.")
    if comm_mode != "aiv_urma":
        raise ValueError(
            f"comm_mode only supports 'aiv_urma', but got '{comm_mode}'. "
            f"Please pass comm_mode='aiv_urma' explicitly."
        )
    if x1.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"x1 only supports torch.float8_e4m3fn, but got {x1.dtype}."
        )
    if x2.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"x2 only supports torch.float8_e4m3fn, but got {x2.dtype}."
        )
    fp8_e8m0_dtype_enum = 293
    if (
        x1_scale is not None
        and x1_scale.dtype == torch.uint8
        and x1_scale_dtype != fp8_e8m0_dtype_enum
    ):
        raise ValueError(
            f"x1_scale is uint8 tensor (fp8_e8m0 stored as uint8) but "
            f"x1_scale_dtype={x1_scale_dtype} is not 293 (fp8_e8m0). "
            f"Please pass x1_scale_dtype=293."
        )
    if (
        x2_scale is not None
        and x2_scale.dtype == torch.uint8
        and x2_scale_dtype != fp8_e8m0_dtype_enum
    ):
        raise ValueError(
            f"x2_scale is uint8 tensor (fp8_e8m0 stored as uint8) but "
            f"x2_scale_dtype={x2_scale_dtype} is not 293 (fp8_e8m0). "
            f"Please pass x2_scale_dtype=293."
        )


def _prepare_comm_context(group):
    """Create communication context, return context/buffer/rank info."""
    rank_id = torch.distributed.get_rank(group)
    rank_size = torch.distributed.get_world_size(group)
    if rank_size <= 0:
        rank_size = torch.distributed.get_rank_size()
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
    channel_builder = CommChannelBuilderManager(group_name)
    ctx_tag = group_name + "quant_matmul_reduce_scatter"
    context = channel_builder.create_context(ctx_tag)
    hccl_buffer_size = channel_builder.get_hccl_buffer_size()
    return context, hccl_buffer_size, rank_id, rank_size, group_name


def _check_hccl_buffer(x1, x2, rank_size, hccl_buffer_size):
    """Check comm data + 2MB reserved <= hccl_buffer_size.

    与 all_gather（通信 A 数据）不同，本算子计算在前，通信对象为 CType (BF16)：
    win 区布局 = R 个 rank 的 C 分片连续存放 = [m, n]，通信量 = m * n * 2 字节。
    """
    hccl_buffer_reserved_bytes = 2 * 1024 * 1024
    m = x1.size(0)
    n = x2.size(1)
    bytes_per_ctype = 2  # BF16
    comm_data_bytes = m * n * bytes_per_ctype
    need_bytes = comm_data_bytes + hccl_buffer_reserved_bytes
    _logger.info(
        "hccl_buffer_size=%s bytes (%.2f MB), comm_data_bytes=%s bytes (%.2f MB), "
        "need (with 2MB reserved)=%s",
        hccl_buffer_size,
        hccl_buffer_size / 1024 / 1024,
        comm_data_bytes,
        comm_data_bytes / 1024 / 1024,
        need_bytes,
    )
    if need_bytes > hccl_buffer_size:
        raise ValueError(
            f"Communication data size + 2MB reserved ({need_bytes} bytes = "
            f"comm_data {comm_data_bytes} + reserved "
            f"{hccl_buffer_reserved_bytes}) exceeds hccl_buffer_size "
            f"({hccl_buffer_size} bytes = {hccl_buffer_size / 1024 / 1024:.2f} MB). "
            f"Please increase HCCL_BUFFSIZE environment variable."
        )


def quant_matmul_reduce_scatter(
    x1: torch.Tensor,
    x2: torch.Tensor,
    group,
    *,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
    y_dtype: Optional[int] = None,
    comm_mode: Optional[str] = "ai_cpu",
) -> torch.Tensor:
    """QuantMatmul + AllToAll + ReduceSum fused computation (apace UDMA path, compute-first).

    Each rank computes ``C_r = x1 @ x2^T`` with its full-K inputs, then the M axis
    of C is split by rank_size for AllToAll communication and reduced:
    ``y = ReduceScatter_r(sum_{r} C_r)``, i.e. per-rank output
    ``(Y)[i] = sum_r (x1_r @ x2_r^T)[rows of chunk i]`` with shape
    ``(M / rank_size, N)``.

    Args:
        x1 (Tensor): Left matrix (per-rank A, full K), shape ``(M, K)``, dtype
            float8_e4m3fn. M must be divisible by rank_size.
        x2 (Tensor): Right matrix as a transpose view ``[K, N]`` of a contiguous
            ``[N, K]`` tensor (pass ``x2.t()``), dtype float8_e4m3fn.
        group (ProcessGroup): torch.distributed process group for AllToAll
            communication.
        x1_scale (Tensor, optional): MX scale for x1, shape ``(M, Ceil(K/64), 2)``,
            dtype float8_e8m0fnu. Required in MX scene.
        x2_scale (Tensor, optional): MX scale for x2, shape ``(N, Ceil(K/64), 2)``, dtype
            float8_e8m0fnu. Required in MX scene.
        group_sizes (List[int], optional): Quantization group sizes, MX only
            supports ``[1, 1, 32]``. None means auto-inferred from scale shapes.
        x1_dtype (int, optional): x1 dtype enum for bitcast, None means use tensor dtype.
        x2_dtype (int, optional): x2 dtype enum for bitcast, None means use tensor dtype.
        x1_scale_dtype (int, optional): x1_scale dtype enum for bitcast, None means use tensor dtype.
        x2_scale_dtype (int, optional): x2_scale dtype enum for bitcast, None means use tensor dtype.
        y_dtype (int, optional): Output dtype enum, None means default bfloat16.
            Only bfloat16 is supported.
        comm_mode (str, optional): Communication engine. Only ``"aiv_urma"`` is supported;
            the default ``"ai_cpu"`` is rejected, so pass ``"aiv_urma"`` explicitly.

    Returns:
        Tensor: y with shape ``(M / rank_size, N)``, dtype bfloat16.
    """
    if group_sizes is None:
        group_sizes = []
    _check_params(
        x1,
        x2,
        x1_scale,
        x2_scale,
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        comm_mode,
    )
    context, hccl_buffer_size, rank_id, rank_size, group_name = _prepare_comm_context(
        group
    )
    _check_hccl_buffer(x1, x2, rank_size, hccl_buffer_size)
    _logger.info(
        "rank=%s, rank_size=%s, comm_mode=%s, calling npu_quant_matmul_reduce_scatter ...",
        rank_id,
        rank_size,
        comm_mode,
    )
    result = torch.ops.cann_ops_transformer.npu_quant_matmul_reduce_scatter(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        rank_size,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=group_sizes,
        x1_dtype=x1_dtype,
        x2_dtype=x2_dtype,
        x1_scale_dtype=x1_scale_dtype,
        x2_scale_dtype=x2_scale_dtype,
        y_dtype=y_dtype,
        comm_mode=comm_mode,
    )
    _logger.info("rank=%s, npu_quant_matmul_reduce_scatter returned", rank_id)
    # schema 为 `-> (Tensor)`，torch.ops 返回裸 Tensor（单元素 schema 自动解包），直接返回
    return result
