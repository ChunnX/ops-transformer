# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
High-level EP normal MoE buffer (DeepEP-style API).

Pipeline:
  npu_get_dispatch_layout  ->  npu_dispatch (moe_dispatch_prefill)  ->  expert  ->  npu_combine (moe_combine_prefill)
"""

import torch
import torch_npu
from torch_npu.utils._error_code import ErrCode, ops_error
from .update_context import update_context

# Handle tuple for npu_combine, produced by npu_dispatch:
# (reserved, reserved, reserved, assist_info_for_combine, is_token_in_rank, ep_recv_counts, topk_idx, topk_weights)
_COMBINE_HANDLE_LEN = 8


class IntWrapper:
    def __init__(self, value=0):
        self.value = value


class EventOverlap:
    """Placeholder for async event overlap; event is None until async is implemented."""

    def __init__(self, event=None):
        self.event = event


class Buffer:
    def __init__(
        self,
        group,
        *,
        local_ranksize: int = 0,
        per_round_tokens: int = 1024,
        quant_mode: int = 0,
        round_num: int = 0,
    ):
        """
        Args:
            group: torch.distributed process group for EP.
            local_ranksize: Ranks per server for layout/notify; 0 means min(8, world_size).
            per_round_tokens: Tokens per round (must match dispatch_layout / notify / dispatch).
            quant_mode: 0 = no quant, 2 = dynamic int8 (same as dispatch_normal).
            round_num: Expected tiling rounds; 0 means use layout-derived round only.
                If > 0, must equal ceil(num_tokens / per_round_tokens) after layout (adjust per_round_tokens).
        """
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id, init_comm=False
        )
        self.per_round_tokens = per_round_tokens
        self.quant_mode = quant_mode
        self._round_num_cfg = round_num
        self.round = 0

        if local_ranksize > 0:
            self.local_ranksize = local_ranksize
        else:
            self.local_ranksize = min(8, self.world_size)

        context_struct_size = 2 + 1024
        context_tensor_size = ((context_struct_size * 8) + 3) // 4
        self.context = torch.zeros(context_tensor_size, dtype=torch.int32, device="npu")
        self.ccl_buffer_size = IntWrapper()

        ok = update_context(self.group_name, self.world_size, self.ccl_buffer_size, self.context)
        torch._check(
            ok,
            lambda: f"update_context failed for group {self.group_name}.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            self.ccl_buffer_size.value > 0,
            lambda: (
                f"invalid ccl_buffer_size from update_context: {self.ccl_buffer_size.value}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

        self._reset_layout_state()
        self.num_experts = 0
        self.num_tokens = 0
        self.num_topk = 0
        self.num_worst_tokens = 0

    def _reset_layout_state(self):
        self.ext_handle_data = None
        self.notify_send_data = None
        self.send_token_idx_small = None
        self.num_tokens_per_rank = None
        self.num_tokens_per_rdma_rank = None
        self.num_tokens_per_expert = None
        self.is_token_in_rank = None

    @staticmethod
    def get_normal_ccl_buffer_size(
        world_size: int,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_moe_expert: int,
        topk: int,
        num_shared_expert: int = 0,
        num_shared_expert_ranks: int = 0,
        comm_alg: str = "",
    ) -> int:
        def inline_align(value, base):
            return (value + base - 1) // base * base

        torch._check(
            (world_size >= 2) and (world_size <= 768),
            lambda: f"world_size only support in [2, 768], but got {world_size=}.",
        )
        torch._check(
            (hidden >= 1024) and (hidden <= 8192),
            lambda: f"hidden only support in [1024, 8192], but got {hidden=}.",
        )
        torch._check(
            (num_max_dispatch_tokens_per_rank >= 1) and (num_max_dispatch_tokens_per_rank <= 512),
            lambda: (
                f"num_max_dispatch_tokens_per_rank only support in [1, 512], "
                f"but got {num_max_dispatch_tokens_per_rank=}."
            ),
        )
        torch._check(
            (num_moe_expert >= 1) and (num_moe_expert <= 1024),
            lambda: f"num_moe_expert only support in [1, 1024], but got {num_moe_expert=}.",
        )
        torch._check(
            (topk >= 1) and (topk <= 16),
            lambda: f"topk only support in [1, 16], but got {topk=}.",
        )
        torch._check(
            (num_shared_expert >= 0) and (num_shared_expert <= 4),
            lambda: f"num_shared_expert only support in [0, 4], but got {num_shared_expert=}.",
        )
        torch._check(
            world_size - num_shared_expert_ranks > 0,
            lambda: (
                f"world_size - num_shared_expert_ranks must be greater than 0, "
                f"but got {world_size=} {num_shared_expert_ranks=}."
            ),
        )
        local_moe_expert_num = num_moe_expert // (world_size - num_shared_expert_ranks)
        torch._check(
            (local_moe_expert_num * world_size > 0) and (local_moe_expert_num * world_size <= 2048),
            lambda: (
                f"local_moe_expert_num * world_size only support in (0, 2048], "
                f"but got {world_size=} {num_shared_expert_ranks=}, {local_moe_expert_num=}."
            ),
        )

        max_out_dtype_size = 2
        mb_conversion = 1024 * 1024
        ub_align = 32
        scale_expand_index_buffer = 44
        full_mesh_data_align = 480
        win_addr_align = 512

        comm_alg_support_list = ["fullmesh_v1", "fullmesh_v2", ""]
        torch._check(
            comm_alg in comm_alg_support_list,
            lambda: f"comm_alg only support {comm_alg_support_list=}, but got {comm_alg=}.",
        )

        token_actual_len = inline_align(hidden * max_out_dtype_size, ub_align) + scale_expand_index_buffer
        if comm_alg == "fullmesh_v2":
            token_need_size_dispatch = (
                inline_align(token_actual_len, full_mesh_data_align)
                // full_mesh_data_align
                * win_addr_align
            )
        else:
            token_need_size_dispatch = inline_align(token_actual_len, win_addr_align)
        token_need_size_combine = inline_align(hidden * max_out_dtype_size, win_addr_align)
        minimum_buffer_size = 2 * (
            num_max_dispatch_tokens_per_rank * token_need_size_dispatch * world_size * local_moe_expert_num
            + num_max_dispatch_tokens_per_rank * token_need_size_combine * (topk + num_shared_expert)
        ) + mb_conversion
        return inline_align(inline_align(minimum_buffer_size, mb_conversion) // mb_conversion, 2) // 2

    def update_ctx(self, new_group) -> bool:
        self.group = new_group
        self.rank_id = torch.distributed.get_rank(new_group)
        self.group_name = new_group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id, init_comm=False
        )
        new_world_size = torch.distributed.get_world_size(new_group)
        torch._check(
            new_world_size == self.world_size,
            lambda: (
                f"New world size should be the same as original world size, "
                f"but got {new_world_size=}, original={self.world_size}"
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )
        ok = update_context(self.group_name, self.world_size, self.ccl_buffer_size, self.context)
        torch._check(
            ok and self.ccl_buffer_size.value > 0,
            lambda: (
                f"update_context failed or invalid ccl_buffer_size={self.ccl_buffer_size.value}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )
        self._reset_layout_state()
        return ok

    @staticmethod
    def layout_round(num_tokens: int, per_round_tokens: int) -> int:
        return (num_tokens + per_round_tokens - 1) // per_round_tokens

    @staticmethod
    def _recv_count_to_ep_recv_counts(recv_count: torch.Tensor, round: int, moe_expert_num: int) -> torch.Tensor:
        expected = round * moe_expert_num
        torch._check(
            recv_count.numel() == expected,
            lambda: (
                f"recv_count numel {recv_count.numel()} must be round * moe_expert_num "
                f"({round} * {moe_expert_num} = {expected})."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )
        if round == 1:
            return recv_count.reshape(moe_expert_num).contiguous()
        return recv_count.reshape(round, moe_expert_num).sum(dim=0).to(torch.int32)

    def _sync_round_after_layout(self, num_tokens_per_expert: torch.Tensor, num_experts: int):
        torch._check(
            num_experts > 0 and num_tokens_per_expert.numel() % num_experts == 0,
            lambda: (
                f"num_tokens_per_expert numel {num_tokens_per_expert.numel()} "
                f"not divisible by num_experts {num_experts}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )
        layout_round = num_tokens_per_expert.numel() // num_experts
        if self._round_num_cfg > 0:
            torch._check(
                layout_round == self._round_num_cfg,
                lambda: (
                    f"round_num={self._round_num_cfg} but layout produced round={layout_round} "
                    f"(num_tokens={self.num_tokens}, per_round_tokens={self.per_round_tokens}). "
                    f"Set per_round_tokens so ceil(num_tokens/per_round_tokens)==round_num."
                    f"{ops_error(ErrCode.VALUE)}"
                ),
            )
        self.round = layout_round

    def _require_layout_done(self):
        torch._check(
            self.ext_handle_data is not None,
            lambda: (
                "npu_get_dispatch_layout must be called before npu_dispatch."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

    def npu_get_dispatch_layout(
        self,
        topk_idx,
        num_experts: int,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        if previous_event is not None or async_finish or allocate_on_comm_stream:
            pass  # reserved for async overlap; not implemented yet

        self.num_experts = num_experts
        self.num_tokens = topk_idx.size(0)
        self.num_topk = topk_idx.size(1)

        if topk_idx.dtype != torch.int64:
            topk_idx = topk_idx.to(torch.int64)
        torch._check(
            topk_idx.device.type == "npu",
            lambda: f"topk_idx must be on NPU, got device {topk_idx.device}.{ops_error(ErrCode.VALUE)}",
        )

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            ext_handle_data,
        ) = torch.ops.npu_ops_transformer.npu_dispatch_layout(
            topk_idx=topk_idx,
            num_ranks=self.world_size,
            num_experts=num_experts,
            local_ranksize=self.local_ranksize,
            per_round_tokens=self.per_round_tokens,
            rank_id=self.rank_id,
        )

        notify_send_data, send_token_idx_small = ext_handle_data
        self.ext_handle_data = ext_handle_data
        self.notify_send_data = notify_send_data
        self.send_token_idx_small = send_token_idx_small
        self.num_tokens_per_rank = num_tokens_per_rank
        self.num_tokens_per_rdma_rank = num_tokens_per_rdma_rank
        self.num_tokens_per_expert = num_tokens_per_expert
        self.is_token_in_rank = is_token_in_rank
        self._sync_round_after_layout(num_tokens_per_expert, num_experts)

        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(previous_event),
        )

    def npu_dispatch(
        self,
        x,
        handle=None,
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        is_token_in_rank=None,
        num_tokens_per_expert=None,
        topk_idx=None,
        topk_weights=None,
        expert_alignment=1,
        num_worst_tokens=0,
        config=None,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        dispatch_wait_recv_cost_stats=None,
    ):
        del handle, num_tokens_per_rdma_rank, expert_alignment, config
        del previous_event, async_finish, allocate_on_comm_stream

        self._require_layout_done()
        torch._check(self.round > 0, lambda: f"round not set; call npu_get_dispatch_layout first.{ops_error(ErrCode.VALUE)}")
        torch._check(topk_idx is not None, lambda: f"topk_idx is required.{ops_error(ErrCode.VALUE)}")
        torch._check(
            topk_idx.size(0) == self.num_tokens,
            lambda: (
                f"topk_idx batch {topk_idx.size(0)} must match layout num_tokens {self.num_tokens}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

        self.num_worst_tokens = num_worst_tokens
        topk_idx_i32 = topk_idx.to(torch.int32)

        tokens_per_rank = num_tokens_per_rank if num_tokens_per_rank is not None else self.num_tokens_per_rank
        tokens_in_rank = is_token_in_rank if is_token_in_rank is not None else self.is_token_in_rank
        tokens_per_expert = (
            num_tokens_per_expert if num_tokens_per_expert is not None else self.num_tokens_per_expert
        )
        torch._check(
            tokens_per_expert.numel() == self.num_experts * self.round,
            lambda: (
                f"num_tokens_per_expert numel {tokens_per_expert.numel()} must be "
                f"num_experts * round ({self.num_experts} * {self.round})."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

        recv_x, dynamic_scales, assist_info_for_combine, recv_count = (
            torch.ops.npu_ops_transformer.npu_moe_dispatch_prefill(
                context=self.context,
                x=x,
                topk_idx=topk_idx_i32,
                ext_handle_data=self.ext_handle_data,
                ep_world_size=self.world_size,
                ep_rank_id=self.rank_id,
                local_ranksize=self.local_ranksize,
                moe_expert_num=self.num_experts,
                ccl_buffer_size=self.ccl_buffer_size.value,
                x_scales=None,
                topk_weights=topk_weights,
                num_tokens_per_rank=tokens_per_rank,
                is_token_in_rank=tokens_in_rank,
                num_tokens_per_expert=tokens_per_expert,
                dispatch_wait_recv_cost_stats=dispatch_wait_recv_cost_stats,
                num_worst_tokens=num_worst_tokens,
                quant_mode=self.quant_mode,
                round=self.round,
                per_round_tokens=self.per_round_tokens,
            )
        )

        if self.quant_mode > 0:
            recv_x = (recv_x, dynamic_scales)

        ep_recv_counts = self._recv_count_to_ep_recv_counts(
            recv_count, self.round, self.num_experts
        )
        combine_handle = (
            None,
            None,
            None,
            assist_info_for_combine,
            tokens_in_rank,
            ep_recv_counts,
            topk_idx_i32,
            topk_weights,
        )

        return (
            recv_x,
            topk_idx_i32,
            topk_weights,
            [],
            combine_handle,
            None,
        )

    def npu_combine(
        self,
        x,
        handle,
        topk_weights=None,
        bias=None,
        config=None,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        combine_send_cost_stats=None,
    ):
        del bias, config, previous_event, async_finish, allocate_on_comm_stream

        torch._check(handle is not None, lambda: f"handle from npu_dispatch is required.{ops_error(ErrCode.VALUE)}")
        torch._check(
            len(handle) == _COMBINE_HANDLE_LEN,
            lambda: (
                f"handle must be {_COMBINE_HANDLE_LEN}-tuple from npu_dispatch, got len={len(handle)}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

        _, _, _, assist_info_for_combine, is_token_in_rank, ep_recv_counts, topk_idx, topk_weights_from_handle = (
            handle
        )
        weights = topk_weights if topk_weights is not None else topk_weights_from_handle
        torch._check(
            ep_recv_counts.numel() == self.num_experts,
            lambda: (
                f"ep_recv_counts numel {ep_recv_counts.numel()} must equal num_experts {self.num_experts}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

        real_max_bs = max(self.num_tokens, self.num_worst_tokens)

        out = torch.ops.npu_ops_transformer.npu_moe_combine_prefill(
            context=self.context,
            recv_x=x,
            token_src_info=assist_info_for_combine,
            ep_recv_counts=ep_recv_counts,
            topk_weights=weights,
            token_idx=topk_idx,
            ep_world_size=self.world_size,
            ep_rank_id=self.rank_id,
            moe_expert_num=self.num_experts,
            ccl_buffer_size=self.ccl_buffer_size.value,
            combine_send_cost_stats=combine_send_cost_stats,
            real_max_bs=real_max_bs,
            round=self.round,
            per_round_tokens=self.per_round_tokens,
        )
        return out, None, None
