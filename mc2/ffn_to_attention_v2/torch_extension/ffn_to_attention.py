# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import csv
import os
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
            "Tensor? attn_rank_table=None, Tensor(a!)? perf_data=None) -> ()"
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
            perf_data=None,
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
            if perf_data is not None:
                torch._check(
                    perf_data.dim() == 2 and perf_data.size(1) == 16,
                    lambda: f"perf_data must have shape [aivNum, 16], {ops_error(ErrCode.VALUE)}.",
                )
                torch._check(
                    perf_data.dtype == torch.int64,
                    lambda: f"perf_data must have dtype int64, {ops_error(ErrCode.VALUE)}.",
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
    perf_data=None,
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
        perf_data,
        group,
        world_size,
        token_info_table_shape,
        token_data_shape,
        ccl_buffer_size,
    )


def _get_hccl_comm_name(group, rank_id):
    """Get HCCL communicator name from a torch.distributed ProcessGroup."""
    if hasattr(group, "get_hccl_comm_name"):
        group_name = group.get_hccl_comm_name(rank_id, init_comm=True)
    else:
        get_backend = getattr(group, "".join(["_get", "_backend"]))
        group_name = get_backend(torch.device("npu")).get_hccl_comm_name(
            rank_id, init_comm=True
        )
    if not group_name:
        raise RuntimeError(f"failed to initialize HCCL communicator for rank {rank_id}")
    return group_name


def _calc_window_size(token_info_table_shape, token_data_shape):
    """Calculate the required window buffer size in bytes."""
    _op_module = _ffn_to_attention_op_builder.load()
    return _op_module.get_ffn_to_attention_ccl_buffer_size(
        token_info_table_shape, token_data_shape
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
    ):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self.ep_world_size = world_size
        self.token_info_table_shape = list(token_info_table_shape)
        self.token_data_shape = list(token_data_shape)

        with torch.profiler.record_function("ffn_to_attention::calc_window_size"):
            required_buffer_size = _calc_window_size(
                token_info_table_shape, token_data_shape
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
        with torch.profiler.record_function("ffn_to_attention::create_context"):
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

        offset_epHcclBuffer = 2 * 4 + 8
        offset_target = offset_epHcclBuffer + self.rank_id * 8
        addr_bytes = ctypes.c_uint64.from_address(context_ptr + offset_target)
        return int(addr_bytes.value)


def get_buffer_for_ffn_to_attention(
    group: "torch.distributed.ProcessGroup",
    world_size: int,
    token_info_table_shape: List[int],
    token_data_shape: List[int],
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

    Returns:
        FFNToAttentionBuffer: Buffer holding the communication context, group
        name, world size, token table shapes and CCL buffer size.
    """
    return FFNToAttentionBuffer(
        group,
        world_size,
        token_info_table_shape,
        token_data_shape,
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
    perf_data: Optional[torch.Tensor] = None,
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
        perf_data (Tensor, optional): Per-AIV cycle storage, shape (aivNum, 16), dtype int64.

    Returns:
        None. Data is sent to peer ranks via the HCCL window; no host-visible output.
    """
    with torch.profiler.record_function("ffn_to_attention::dispatch"):
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
            perf_data=perf_data,
        )


def save_ffn_to_attention_perf_csv(
    perf_data: torch.Tensor,
    output_dir: str,
    rank: Optional[int] = None,
) -> str:
    """Synchronize and save per-AIV kernel cycle counters as CSV."""
    if (
        perf_data.dtype != torch.int64
        or perf_data.dim() != 2
        or perf_data.size(1) != 16
    ):
        raise ValueError("perf_data must have shape [aivNum, 16] and dtype int64")

    torch.npu.synchronize()
    values = perf_data.cpu().tolist()
    if rank is None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ffn_to_attention_perf_rank{rank}.csv")
    fields = [
        "prof1_ReadTokenMetaData_cycles",
        "prof2_LocalAddressAndOffset_cycles",
        "prof3_SendToLocal_cycles",
        "prof4_StageRemoteData_cycles",
        "prof5_CountRemoteTokens_cycles",
        "prof6_MakeBatchHandle_cycles",
        "prof7_BuildRemoteDataInfo_cycles",
        "prof8_AppendRemoteDataWrite_cycles",
        "prof9_BuildRemoteFlagInfo_cycles",
        "prof10_AppendRemoteFlagWrite_cycles",
        "prof11_BatchCommit_cycles",
        "prof12_Drain_cycles",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["rank", "aiv_id", *fields])
        for aiv_id, cycles in enumerate(values):
            writer.writerow([rank, aiv_id, *cycles[: len(fields)]])
    return output_path
