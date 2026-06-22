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
from .update_context import init_mc2_context
from .dispatch_layout import op_module as dispatch_layout_op_module

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
        per_round_tokens: int = 8192,
        quant_mode: int = 0,
        round_num: int = 1,
    ):
        """
        Args:
            group: torch.distributed process group for EP.
            local_ranksize: Ranks per server for dispatch_layout only (sgl LOCAL_RANK_SIZE=8 on
                16-card); 0 auto-selects 8 when world_size>=16 else world_size. Notify/prefill always
                uses ep world_size, not this value.
            per_round_tokens: Tokens per round (sgl default 8192; must match dispatch_layout / notify / dispatch).
            quant_mode: 0 = no quant, 2 = dynamic int8 (same as dispatch_normal).
            round_num: Configured max rounds (sgl Buffer.round, default 1). Each request must satisfy
                num_tokens <= round_num * per_round_tokens and round_num * per_round_tokens <= 131072.
                Layout-derived round is ceil(num_tokens / per_round_tokens) and must not exceed round_num.
        """
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id
        )
        self.per_round_tokens = per_round_tokens
        self.quant_mode = quant_mode
        self._round_num_cfg = round_num
        self.round = 0

        if local_ranksize > 0:
            self.local_ranksize = local_ranksize
        elif self.world_size >= 16 and self.world_size % 8 == 0:
            self.local_ranksize = 8
        else:
            self.local_ranksize = self.world_size

        context_struct_size = 2 + 1024
        context_tensor_size = ((context_struct_size * 8) + 3) // 4
        self.context = torch.zeros(context_tensor_size, dtype=torch.int32, device="npu")
        self.ccl_buffer_size = IntWrapper()

        ok = init_mc2_context(self.group_name, self.world_size, self.ccl_buffer_size, self.context)
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
            self.rank_id
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
        ok = init_mc2_context(self.group_name, self.world_size, self.ccl_buffer_size, self.context)
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

    @property
    def max_token_capacity(self) -> int:
        """sgl: round * per_round_tokens upper bound for a single request."""
        return self._round_num_cfg * self.per_round_tokens

    def _check_token_capacity(self, num_tokens: int):
        torch._check(
            self._round_num_cfg > 0,
            lambda: f"round_num must be positive, got {self._round_num_cfg}.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            num_tokens <= self.max_token_capacity,
            lambda: (
                f"num_tokens={num_tokens} exceeds capacity round_num*per_round_tokens="
                f"{self._round_num_cfg}*{self.per_round_tokens}={self.max_token_capacity}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )
        torch._check(
            self.max_token_capacity <= 131072,
            lambda: (
                f"round_num*per_round_tokens={self.max_token_capacity} exceeds 131072."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

    @staticmethod
    def _recv_count_to_ep_recv_counts(recv_count: torch.Tensor, round: int, moe_expert_num: int) -> torch.Tensor:
        # prefill returns sgl-style recv_count.sum(0) -> [num_experts]
        if recv_count.numel() == moe_expert_num:
            return recv_count.contiguous()
        expected = round * moe_expert_num
        torch._check(
            recv_count.numel() == expected,
            lambda: (
                f"recv_count numel {recv_count.numel()} must be num_experts ({moe_expert_num}) "
                f"or round * moe_expert_num ({round} * {moe_expert_num} = {expected})."
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
        torch._check(
            layout_round <= self._round_num_cfg,
            lambda: (
                f"layout round={layout_round} exceeds configured round_num={self._round_num_cfg} "
                f"(num_tokens={self.num_tokens}, per_round_tokens={self.per_round_tokens}). "
                f"Increase round_num or per_round_tokens."
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
        self._check_token_capacity(self.num_tokens)

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            ext_handle_data,
        ) = dispatch_layout_op_module.npu_dispatch_layout(
            topk_idx,
            self.num_tokens,
            self.num_topk,
            num_experts,
            self.world_size,
            self.local_ranksize,
            self.per_round_tokens,
            self.rank_id,
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
        sync_finish=False,
        allocate_on_comm_stream=False,
        dispatch_wait_recv_cost_stats=None,
        *,
        async_finish=None,
    ):
        """
        MoE EP dispatch (notify_dispatch + dispatch_normal via npu_moe_dispatch_prefill).

        Reserved (ignored): handle, num_tokens_per_rdma_rank, expert_alignment, config,
        previous_event, sync_finish, allocate_on_comm_stream, async_finish.

        Returns:
            recv_x: Tensor or (Tensor, scales) when quant_mode > 0
            recv_topk_idx: None (reserved)
            recv_topk_weights: None (reserved)
            num_recv_tokens_per_expert_list: [] (reserved)
            handle: combine cache tuple
            event: EventOverlap placeholder
        """
        del handle, num_tokens_per_rdma_rank, expert_alignment, config
        del previous_event, sync_finish, allocate_on_comm_stream, async_finish

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

        x_scales = None
        if isinstance(x, tuple):
            torch._check(
                len(x) == 2,
                lambda: f"x tuple must be (tensor, scales), got len {len(x)}.{ops_error(ErrCode.VALUE)}",
            )
            x, x_scales = x
        if self.quant_mode > 0 and x_scales is not None:
            pass  # reserved for FP8 path; prefill uses quant_mode on x dtype today

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
                local_ranksize=self.world_size,
                moe_expert_num=self.num_experts,
                ccl_buffer_size=self.ccl_buffer_size.value,
                x_scales=x_scales,
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
            None,
            None,
            [],
            combine_handle,
            EventOverlap(None),
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
        """
        MoE EP combine (DeepEP-style API; kernel: combine_normal / npu_combine_normal).

        Inputs (DeepEP / UMDK):
            x: expert 输出激活，shape (num_recv_tokens, hidden)，bf16/fp16；对应算子入参 recv_x。
            handle: 必填，同一次 forward 的 npu_dispatch 返回的 8-tuple（UMDK 侧实现依赖其中元数据）。
            topk_weights: 可选，(num_tokens, num_topk) float；None 时做不加权 reduce（内部填 1.0）。
            combine_send_cost_stats: 可选 DFX buffer（UMDK Y）。

        Reserved (ignored, UMDK 占位): handle 在表中标 N 但实现仍使用；
            bias, config, previous_event, async_finish, allocate_on_comm_stream。

        Returns:
            recv_x: (num_combined_tokens, hidden)，即按原 batch 聚合后的结果（算子输出 x）。
            recv_topk_weights: None（UMDK 保留，未实现回传 reduce 后权重）。
            event: None（UMDK 保留；DeepEP 在 async_finish=True 时才有意义）。
        """
        del bias, config, previous_event, allocate_on_comm_stream
        _ = async_finish  # 保留参数，当前不驱动异步 event

        torch._check(handle is not None, lambda: f"handle from npu_dispatch is required.{ops_error(ErrCode.VALUE)}")
        torch._check(
            len(handle) == _COMBINE_HANDLE_LEN,
            lambda: (
                f"handle must be {_COMBINE_HANDLE_LEN}-tuple from npu_dispatch, got len={len(handle)}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

        _, _, _, assist_info_for_combine, _is_token_in_rank, ep_recv_counts, topk_idx, topk_weights_from_handle = (
            handle
        )
        torch._check(
            ep_recv_counts.numel() == self.num_experts,
            lambda: (
                f"ep_recv_counts numel {ep_recv_counts.numel()} must equal num_experts {self.num_experts}."
                f"{ops_error(ErrCode.VALUE)}"
            ),
        )

        recv_x_in = x
        if isinstance(x, tuple):
            torch._check(
                self.quant_mode == 0,
                lambda: (
                    "quantized x (tensor, scales) is not supported for combine; "
                    "dequant after expert first."
                    f"{ops_error(ErrCode.VALUE)}"
                ),
            )
            recv_x_in = x[0]

        weights = topk_weights if topk_weights is not None else topk_weights_from_handle
        if weights is None:
            bs, k = topk_idx.size(0), topk_idx.size(1)
            weights = torch.ones((bs, k), dtype=torch.float32, device=recv_x_in.device)

        real_max_bs = max(self.num_tokens, self.num_worst_tokens)

        recv_x = torch.ops.npu_ops_transformer.npu_combine_normal(
            context=self.context,
            recv_x=recv_x_in,
            token_src_info=assist_info_for_combine,
            ep_recv_counts=ep_recv_counts,
            recv_topk_weights=weights,
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
        return recv_x, None, None
