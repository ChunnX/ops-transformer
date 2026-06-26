# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from .moe_ep_dispatch import npu_moe_ep_dispatch, npu_moe_ep_dispatch_epilogue
from .moe_ep_combine import npu_moe_ep_combine
from .comm_context import CommContextManager
from .host_pinned_counter import HostPinnedCounter

_op = torch.ops.cann_ops_transformer


class IntWrapper:
    def __init__(self, value=0):
        self.value = value


@dataclass
class EPHandle:
    dst_buffer_slot_idx: torch.Tensor
    recv_src_metadata: torch.Tensor
    num_recv_tokens_per_rank: torch.Tensor
    num_recv_tokens_per_expert: torch.Tensor
    num_experts: int
    expert_alignment: int
    num_max_tokens_per_rank: int
    topk_idx: torch.Tensor

    @property
    def num_recv_tokens(self) -> int:
        return int(self.num_recv_tokens_per_rank.sum().item())


class MoEElasticBuffer:
    def __init__(
        self,
        group,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        num_experts: int,
        *,
        ccl_buffer_size: int = 0,
        expert_alignment: int = 1,
    ):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        self.ep_world_size = torch.distributed.get_world_size(group)
        self._ctx_manager = CommContextManager(self.group_name, self.ep_world_size, commAlg="urma", backend="channel")
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size

        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self._host_pinned_counter = None

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        *,
        handle: Optional[EPHandle] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        do_cpu_sync: Optional[bool] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
               Optional[torch.Tensor], Optional[torch.Tensor], EPHandle]:
        x, scales = x if isinstance(x, tuple) else (x, None)
        cached = handle is not None
        if cached:
            torch._check((topk_idx is None), lambda: (f"topk_idx is not surpport when cached."))
            torch._check(((do_cpu_sync is None) or (do_cpu_sync is False)),
                lambda: (f"do_cpu_sync is not surpport when cached."))
            topk_idx = handle.topk_idx
            do_cpu_sync = False
            ne, ea, nm = handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank
        else:
            torch._check(((topk_idx is not None) and (topk_weights is not None)),
                lambda: (f"topk_idx and topk_weights are required when no-cached."))
            do_cpu_sync = True if do_cpu_sync is None else do_cpu_sync
            ne = num_experts if num_experts is not None else self.num_experts
            nm = num_max_tokens_per_rank if num_max_tokens_per_rank is not None else self.num_max_tokens_per_rank
            ea = expert_alignment if expert_alignment is not None else self.expert_alignment

        cached_dst_slot = handle.dst_buffer_slot_idx if cached else None
        cached_recv_src_metadata = handle.recv_src_metadata if cached else None
        ep_world_size = self.ep_world_size

        hp_addr = 0
        if do_cpu_sync and not cached:
            if self._host_pinned_counter is None:
                self._host_pinned_counter = HostPinnedCounter()
            self._host_pinned_counter.reset()
            hp_addr = self._host_pinned_counter.device_ptr()

        num_recv_per_rank, num_recv_per_expert, dst_slot = \
            _op.npu_moe_ep_dispatch(
                self.context, x, topk_idx, topk_weights, scales, cached_dst_slot,
                ep_world_size, self.rank_id, ne, nm, self.ccl_buffer_size,
                expert_alignment=ea, do_cpu_sync=do_cpu_sync,
                host_pinned_counter_addr=hp_addr
            )

        if cached:
            actual_a = handle.recv_src_metadata.shape[0]
        elif do_cpu_sync:
            actual_a = self._host_pinned_counter.spin_wait()
        else:
            actual_a = ep_world_size * nm * min(self.num_topk, ne // ep_world_size)

        recv_x = torch.empty((actual_a, self.hidden), dtype=x.dtype, device=x.device)
        recv_topk_weights = torch.empty((actual_a,), dtype=torch.float32, device=x.device) if topk_weights is not None else None
        recv_src_meta = torch.empty((actual_a, 3), dtype=torch.int32, device=x.device)
        recv_scales = None
        if scales is not None:
            recv_scales = torch.empty((actual_a, scales.shape[1]), dtype=scales.dtype, device=x.device)

        recv_x, recv_topk_weights, recv_src_meta, recv_scales = \
            _op.npu_moe_ep_dispatch_epilogue(
                self.context, dst_slot, num_recv_per_rank, num_recv_per_expert, cached_recv_src_metadata,
                ep_world_size, self.rank_id, ne, nm, self.ccl_buffer_size, expert_alignment=ea,
                recv_x=recv_x, recv_topk_weights=recv_topk_weights, recv_src_metadata=recv_src_meta,
                recv_scales=recv_scales
            )

        recv_x = (recv_x, recv_scales) if recv_scales is not None else recv_x
        recv_topk_idx = None
        new_handle = EPHandle(
            dst_buffer_slot_idx=dst_slot,
            recv_src_metadata=recv_src_meta,
            num_recv_tokens_per_rank=num_recv_per_rank,
            num_recv_tokens_per_expert=num_recv_per_expert,
            num_experts=ne,
            expert_alignment=ea,
            num_max_tokens_per_rank=nm,
            topk_idx=(topk_idx.clone() if topk_idx is not None else handle.topk_idx),
        )
        return recv_x, recv_topk_idx, recv_topk_weights, new_handle

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        *,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bias_0, bias_1 = self._unpack_bias(bias)
        torch._check(((bias_0 is None) and (bias_1 is None)), lambda: (f"bias are not supported."))

        combined_x, combined_topk_weights = _op.npu_moe_ep_combine(
            self.context, x, handle.topk_idx, handle.recv_src_metadata,
            handle.num_recv_tokens_per_expert, topk_weights, bias_0, bias_1,
            self.ep_world_size, self.rank_id, handle.num_experts,
            handle.num_max_tokens_per_rank, self.ccl_buffer_size
        )
        return combined_x, combined_topk_weights

    @staticmethod
    def get_moe_ep_ccl_buffer_size(world_size: int, num_max_tokens_per_rank: int, hidden: int,
                                   num_experts: int, topk: int) -> int:
        def inline_align(value, base):
            return (value + base - 1) // base * base

        torch._check(((num_experts >= 1) and (num_experts <= 2048)),       # num_experts当前仅支持[1,2048]
                     lambda: (f"num_experts only support in [1, 2048], but got {num_experts=}."))
        torch._check(((topk >= 1) and (topk <= 32)),                # topk当前仅支持[1,32]
                     lambda: (f"topk only support in [1, 32], but got {topk=}."))

        win_addr_align = 512
        ub_align = 32
        max_out_dtype_size = 2  # sizeof(bf16)
        metadata_dtype_size = 4 # sizeof(int32)/sizeof(float)
        mb_conversion = 1024 * 1024
        metadata_bytes = inline_align(topk * metadata_dtype_size, ub_align)
        per_token_bytes = inline_align(hidden * max_out_dtype_size + metadata_bytes * 2 + ub_align, win_addr_align)
        minimum_buffer_size = 2 * world_size * num_max_tokens_per_rank * per_token_bytes
        ## hccl按照2*bufferSize大小开设
        ccl_buffer_size = inline_align(inline_align(minimum_buffer_size, mb_conversion) // mb_conversion, 2) // 2

        return ccl_buffer_size

    def _unpack_bias(self, bias):
        if bias is None:
            return None, None
        if isinstance(bias, torch.Tensor):
            return bias, None
        if isinstance(bias, tuple) and len(bias) == 2:
            return bias[0], bias[1]
        raise TypeError(f"unsupported bias type: {type(bias)}")

    def destroy(self):
        if self._host_pinned_counter is not None:
            del self._host_pinned_counter
            self._host_pinned_counter = None