# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, List, Union

import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from .comm_context import CommContextManager


class _FFNToAttentionV2OpBuilder(OpBuilder):
    def __init__(self):
        super(_FFNToAttentionV2OpBuilder, self).__init__("npu_ffn_to_attention_v2")

    def sources(self):
        return ["ops/csrc/ffn_to_attention_v2.cpp"]

    def schema(self) -> str:
        return (
            "npu_ffn_to_attention_v2(Tensor context, Tensor x, Tensor session_ids, "
            "Tensor micro_batch_ids, Tensor token_ids, Tensor expert_offsets, Tensor actual_token_num, "
            "str group, int world_size, "
            "int[] token_info_table_shape, int[] token_data_shape, *, "
            "Tensor? attn_rank_table=None) -> ()"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_ffn_to_attention_v2_meta(
            context,
            x,
            session_ids,
            micro_batch_ids,
            token_ids,
            expert_offsets,
            actual_token_num,
            group,
            world_size,
            token_info_table_shape,
            token_data_shape,
            attn_rank_table=None,
        ):
            torch._check(
                world_size > 0,
                lambda: (f"world_size should be greater than 0, {ops_error(ErrCode.VALUE)}."),
            )
            torch._check(
                x.dim() == 2,
                lambda: (f"x should be 2D, but got {x.dim()}D, {ops_error(ErrCode.VALUE)}."),
            )
            torch._check(
                len(token_info_table_shape) == 3,
                lambda: (
                    f"token_info_table_shape should have 3 elements, "
                    f"but got {len(token_info_table_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(token_data_shape) == 4,
                lambda: (
                    f"token_data_shape should have 4 elements, "
                    f"but got {len(token_data_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )


_ffn_to_attention_v2_op_builder = _FFNToAttentionV2OpBuilder()


@impl(AS_LIBRARY, _ffn_to_attention_v2_op_builder.name, "PrivateUse1")
def _npu_ffn_to_attention_v2(
    context,
    x,
    session_ids,
    micro_batch_ids,
    token_ids,
    expert_offsets,
    actual_token_num,
    group,
    world_size,
    token_info_table_shape,
    token_data_shape,
    attn_rank_table=None,
):
    _op_module = _ffn_to_attention_v2_op_builder.load()
    _op_module.npu_ffn_to_attention_v2(
        context,
        x,
        session_ids,
        micro_batch_ids,
        token_ids,
        expert_offsets,
        actual_token_num,
        attn_rank_table,
        group,
        world_size,
        token_info_table_shape,
        token_data_shape,
    )


def _get_hccl_comm_name(group, rank_id):
    """Get HCCL communicator name from a torch.distributed ProcessGroup."""
    if hasattr(group, "get_hccl_comm_name"):
        return group.get_hccl_comm_name(rank_id, init_comm=False)
    get_backend = getattr(group, "".join(["_get", "_backend"]))
    return get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=False)


def _get_npu_backend(group):
    """Get the NPU backend from a torch.distributed.ProcessGroup."""
    get_backend = getattr(group, "".join(["_get", "_backend"]))
    return get_backend(torch.device("npu"))


def _calc_window_size(token_info_table_shape, token_data_shape):
    """Calculate the required window buffer size in bytes."""
    micro_batch_num = token_info_table_shape[0]
    bs = token_info_table_shape[1]
    expert_num_per_token = token_info_table_shape[2]
    hs = token_data_shape[3]
    token_data_size = micro_batch_num * bs * expert_num_per_token * hs * 2
    token_info_size = micro_batch_num * bs * expert_num_per_token * 4
    raw_size = token_data_size + token_info_size
    mb_align = 2 * 1024 * 1024
    return ((raw_size + mb_align - 1) // mb_align) * mb_align


class FFNToAttentionV2Buffer:
    """Communication buffer for ffn_to_attention_v2, wraps CommContextManager.

    Uses the HCCL Channel backend (HcclCommMemReg + HcclChannelGetRemoteMems)
    to allocate and exchange window memory, compatible with Ascend 950.
    """

    def __init__(
        self,
        group: "torch.distributed.ProcessGroup",
        world_size: int,
        token_info_table_shape: List[int],
        token_data_shape: List[int],
    ):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self.ep_world_size = world_size

        required_buffer_size = _calc_window_size(token_info_table_shape, token_data_shape)
        self._ctx_manager = CommContextManager(
            self.group_name,
            self.ep_world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            customCclBufferSize=required_buffer_size,
        )
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size

    def destroy(self):
        self._ctx_manager.destroy()

    def get_window_addr(self) -> int:
        """Get the local window memory device address from the context."""
        import ctypes

        if self.context is None:
            return 0

        host_tensor = self.context.cpu()
        context_ptr = host_tensor.data_ptr()
        if context_ptr == 0:
            return 0

        offset_buffer_start = 2 * 4 + 8
        addr_bytes = ctypes.c_uint64.from_address(context_ptr + offset_buffer_start)

        # offset_epHcclBuffer = 2 * 4 + 8
        # offset_target = offset_epHcclBuffer + self.rank_id * 8
        # addr_bytes = ctypes.c_uint64.from_address(context_ptr + offset_target)
        return int(addr_bytes.value)


def ffn_to_attention_v2(
    context: torch.Tensor,
    x: torch.Tensor,
    session_ids: torch.Tensor,
    micro_batch_ids: torch.Tensor,
    token_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    actual_token_num: torch.Tensor,
    group: Union[str, "torch.distributed.ProcessGroup"],
    world_size: int,
    token_info_table_shape: List[int],
    token_data_shape: List[int],
    *,
    attn_rank_table: Optional[torch.Tensor] = None,
) -> None:
    """Send token data from FFN workers to Attention workers via HCCL window.

    Uses the CommContextManager-based context tensor to pass window memory
    addresses to the kernel, compatible with Ascend 950 (Channel backend).

    Args:
        context (Tensor): Communication context tensor created by
            ``FFNToAttentionV2Buffer``.
        x (Tensor): Token data to send, shape (Y, H), dtype float16/bfloat16.
        session_ids (Tensor): Per-token attention worker index, shape (Y,), dtype int32.
        micro_batch_ids (Tensor): Per-token micro batch index, shape (Y,), dtype int32.
        token_ids (Tensor): Per-token batch offset, shape (Y,), dtype int32.
        expert_offsets (Tensor): Per-token topk expert offset, shape (Y,), dtype int32.
        actual_token_num (Tensor): Actual token count to send, shape (1,), dtype int64.
        group (str | ProcessGroup): HCCL communicator name, or a
            ``torch.distributed.ProcessGroup`` from which the name is resolved.
        world_size (int): Communication domain size.
        token_info_table_shape (List[int]): 3-element list [microBatchNum, BS, expertNumPerToken].
        token_data_shape (List[int]): 4-element list [microBatchNum, BS, expertNumPerToken, HS].
        attn_rank_table (Tensor, optional): Maps attention worker id to rank id,
            shape (attnRankNum,), dtype int32. If None, worker id == rank id.

    Returns:
        None. Data is sent to peer ranks via the HCCL window; no host-visible output.
    """
    if not isinstance(group, str):
        group_name = _get_hccl_comm_name(group, torch.distributed.get_rank(group))
    else:
        group_name = group

    torch.ops.cann_ops_transformer.npu_ffn_to_attention_v2(
        context,
        x,
        session_ids,
        micro_batch_ids,
        token_ids,
        expert_offsets,
        actual_token_num,
        group_name,
        world_size,
        token_info_table_shape,
        token_data_shape,
        attn_rank_table=attn_rank_table,
    )
