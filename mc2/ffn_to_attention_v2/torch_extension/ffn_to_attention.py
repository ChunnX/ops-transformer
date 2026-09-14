# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, List

import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommContextManager


class _FFNToAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super(_FFNToAttentionOpBuilder, self).__init__(
            "npu_ffn_to_attention", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/ffn_to_attention.cpp"]

    def schema(self) -> str:
        return (
            "npu_ffn_to_attention(Tensor(a!) context, Tensor x, Tensor session_ids, "
            "Tensor micro_batch_ids, Tensor token_ids, Tensor expert_offsets, Tensor actual_token_num, "
            "str group, int world_size, "
            "int[] token_info_table_shape, int[] token_data_shape, int ccl_buffer_size, *, "
            "Tensor? attn_rank_table=None) -> ()"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_ffn_to_attention_meta(
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
            ccl_buffer_size,
            attn_rank_table=None,
        ):
            torch._check(
                world_size > 0,
                lambda: (
                    f"world_size should be greater than 0, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                x.dim() == 2,
                lambda: (
                    f"x should be 2D, but got {x.dim()}D, {ops_error(ErrCode.VALUE)}."
                ),
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


_ffn_to_attention_op_builder = _FFNToAttentionOpBuilder()
_ffn_to_attention_op_builder._ensure_initialized()


@impl(get_as_library(), _ffn_to_attention_op_builder.name, "PrivateUse1")
def _npu_ffn_to_attention(
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
    ccl_buffer_size,
    attn_rank_table=None,
):
    _op_module = _ffn_to_attention_op_builder.load()
    _op_module.npu_ffn_to_attention(
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
        ccl_buffer_size,
    )


def _get_hccl_comm_name(group, rank_id):
    """Get HCCL communicator name from a torch.distributed ProcessGroup."""
    if hasattr(group, "get_hccl_comm_name"):
        return group.get_hccl_comm_name(rank_id, init_comm=False)
    get_backend = getattr(group, "".join(["_get", "_backend"]))
    return get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=False)


def _calc_window_size(
    token_info_table_shape, token_data_shape, world_size, max_token_num=-1
):
    """Calculate the complete window capacity, including local flag and relay storage."""
    _op_module = _ffn_to_attention_op_builder.load()
    return _op_module.get_ffn_to_attention_ccl_buffer_size(
        token_info_table_shape, token_data_shape, world_size, max_token_num
    )


class FFNToAttentionBuffer:
    """Communication buffer for ffn_to_attention, wraps CommContextManager.

    Uses the HCCL Channel backend (HcclCommMemReg + HcclChannelGetRemoteMems)
    to allocate and exchange window memory, compatible with Ascend 950.
    """

    def __init__(
        self,
        group: "torch.distributed.ProcessGroup",
        world_size: int,
        token_info_table_shape: List[int],
        token_data_shape: List[int],
        *,
        max_token_num: int = -1,
    ):
        if world_size != torch.distributed.get_world_size(group):
            raise ValueError("world_size must match the communication group size")
        if (
            not isinstance(max_token_num, int)
            or isinstance(max_token_num, bool)
            or max_token_num < -1
        ):
            raise ValueError("max_token_num must be an integer >= -1")
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self.ep_world_size = world_size
        self.token_info_table_shape = list(token_info_table_shape)
        self.token_data_shape = list(token_data_shape)

        required_buffer_size = _calc_window_size(
            token_info_table_shape, token_data_shape, world_size, max_token_num
        )
        self.max_token_num = (
            world_size
            * self.token_info_table_shape[0]
            * self.token_info_table_shape[1]
            * self.token_info_table_shape[2]
            if max_token_num == -1
            else max_token_num
        )
        self.required_buffer_size = required_buffer_size
        self._capacity_shapes = (
            tuple(self.token_info_table_shape),
            tuple(self.token_data_shape),
        )
        self._ctx_manager = CommContextManager(
            self.group_name,
            self.ep_world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            commAlg="urma",
            opName="ffn_to_attention",
            customCclBufferSize=required_buffer_size,
        )
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size
        if self.ccl_buffer_size < self.required_buffer_size:
            raise RuntimeError(
                f"Context reports {self.ccl_buffer_size} bytes, but the window requires "
                f"{self.required_buffer_size} bytes"
            )

    def destroy(self):
        self._ctx_manager.destroy()
        self.context = None

    def get_window_addr(self) -> int:
        """Get the local window memory device address from the context."""
        import ctypes

        if self.context is None:
            return 0

        host_tensor = self.context.cpu()
        context_ptr = host_tensor.data_ptr()
        if context_ptr == 0:
            return 0

        offset_epHcclBuffer = 2 * 4 + 8
        offset_target = offset_epHcclBuffer + self.rank_id * 8
        addr_bytes = ctypes.c_uint64.from_address(context_ptr + offset_target)
        return int(addr_bytes.value)


def get_buffer_for_ffn_to_attention(
    group: "torch.distributed.ProcessGroup",
    world_size: int,
    token_info_table_shape: List[int],
    token_data_shape: List[int],
    *,
    max_token_num: int = -1,
) -> FFNToAttentionBuffer:
    """Create a communication buffer for ffn_to_attention.

    This is a thin factory around ``FFNToAttentionBuffer`` that computes the
    required CCL buffer size, creates the communication context, and returns a
    buffer object holding all the information ``ffn_to_attention`` needs.

    Args:
        group (ProcessGroup): HCCL process group.
        world_size (int): Communication domain size.
        token_info_table_shape (List[int]): 3-element list [microBatchNum, BS, expertNumPerToken].
        token_data_shape (List[int]): 4-element list [microBatchNum, BS, expertNumPerToken, HS].
        max_token_num (int): Capacity for local relay info (must cover x.shape[0]).
            -1 reserves world_size * microBatchNum * BS * expertNumPerToken entries.
            All ranks must use the same capacity when creating their buffers.
            Window sizing also includes local flag sources and per-rank relay list heads/tails.

    Returns:
        FFNToAttentionBuffer: Buffer holding the communication context, group
        name, world size, token table shapes and CCL buffer size.
    """
    return FFNToAttentionBuffer(
        group,
        world_size,
        token_info_table_shape,
        token_data_shape,
        max_token_num=max_token_num,
    )


def ffn_to_attention(
    buffer: FFNToAttentionBuffer,
    x: torch.Tensor,
    session_ids: torch.Tensor,
    micro_batch_ids: torch.Tensor,
    token_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    actual_token_num: torch.Tensor,
    *,
    attn_rank_table: Optional[torch.Tensor] = None,
) -> None:
    """Send token data from FFN workers to Attention workers via HCCL window.

    Uses the CommContextManager-based context tensor to pass window memory
    addresses to the kernel, compatible with Ascend 950 (Channel backend).

    Args:
        buffer (FFNToAttentionBuffer): Communication buffer created by
            ``get_buffer_for_ffn_to_attention``. Holds the context tensor,
            group name, world size, token table shapes and CCL buffer size.
        x (Tensor): Token data to send, shape (Y, H), dtype float16/bfloat16.
        session_ids (Tensor): Per-token attention worker index, shape (Y,), dtype int32.
        micro_batch_ids (Tensor): Per-token micro batch index, shape (Y,), dtype int32.
        token_ids (Tensor): Per-token batch offset, shape (Y,), dtype int32.
        expert_offsets (Tensor): Per-token topk expert offset, shape (Y,), dtype int32.
        actual_token_num (Tensor): Actual token count to send, shape (1,), dtype int64.
        attn_rank_table (Tensor, optional): Maps attention worker id to rank id,
            shape (attnRankNum,), dtype int32. If None, worker id == rank id.

    Returns:
        None. Data is sent to peer ranks via the HCCL window; no host-visible output.
    """
    if buffer.context is None:
        raise RuntimeError("FFNToAttentionBuffer has been destroyed")
    if (
        tuple(buffer.token_info_table_shape),
        tuple(buffer.token_data_shape),
    ) != buffer._capacity_shapes:
        raise ValueError(
            "Window shapes cannot change after buffer creation; create a replacement buffer"
        )
    torch._check(x.dim() == 2, lambda: "x must be a 2D tensor")
    torch._check(
        x.shape[0] <= buffer.max_token_num,
        lambda: "x.shape[0] exceeds max_token_num; all ranks must coordinate buffer replacement",
    )
    torch._check(
        x.shape[1] <= buffer.token_data_shape[3],
        lambda: "x.shape[1] exceeds the window token slot width HS",
    )
    torch.ops.cann_ops_transformer.npu_ffn_to_attention(
        buffer.context,
        x,
        session_ids,
        micro_batch_ids,
        token_ids,
        expert_offsets,
        actual_token_num,
        buffer.group_name,
        buffer.ep_world_size,
        buffer.token_info_table_shape,
        buffer.token_data_shape,
        buffer.ccl_buffer_size,
        attn_rank_table=attn_rank_table,
    )
