# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist

from cann_ops_transformer.op_builder.builder import OpBuilder


class ElasticBufferOpBuilder(OpBuilder):
    """OpBuilder for ElasticBuffer operations"""

    def __init__(self):
        super(ElasticBufferOpBuilder, self).__init__("npu_elastic_buffer")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/elastic_buffer.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return None

    def register_meta(self):
        """Meta implementation (optional for JIT compiled ops)."""
        pass

    def extra_ldflags(self):
        """Extra link flags for HCCL and ACL libraries."""
        flags = super().extra_ldflags()
        flags.append("-L" + os.path.join(self._cann_path, "lib64"))
        flags.append("-lhcomm")
        flags.append("-lascendcl")
        return flags

    def include_paths(self):
        """Override include paths to ensure CANN headers are prioritized."""
        return [
            os.path.join(self._cann_path, "include"),
            os.path.join(self._torch_npu_path, "include"),
            os.path.join(self._torch_npu_path, "include/third_party/hccl/inc"),
            os.path.join(self._torch_npu_path, "include/third_party/acl/inc"),
            os.path.join(self._package_path, "common/inc"),
        ]


_elastic_buffer_op_builder = ElasticBufferOpBuilder()


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


@dataclass
class _DispatchArgs:
    x: torch.Tensor
    scales: Optional[torch.Tensor]
    topk_idx: torch.Tensor
    cached_dst_slot: Optional[torch.Tensor]
    cached_recv_src_metadata: Optional[torch.Tensor]
    num_experts: int
    num_max_tokens_per_rank: int
    expert_alignment: int
    do_cpu_sync: bool
    cached_recv_tokens: Optional[int]


class ElasticBuffer:
    """
    ElasticBuffer for distributed Engram storage management and MoE dispatch/combine operations.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        *,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: Optional[int] = None,
        hidden: Optional[int] = None,
        num_topk: Optional[int] = None,
    ):
        """
        Initialize the ElasticBuffer.

        Arguments:
            group: the distributed process group.
            num_cpu_bytes: the CPU buffer size in bytes (must be 2MB-aligned).
            num_max_tokens_per_rank: maximum MoE dispatch tokens per rank.
            hidden: hidden dimension for MoE dispatch/combine.
            num_topk: top-k value for MoE dispatch/combine.
        """
        buffer_alignment = 2 * 1024 * 1024
        torch._check((group is not None), lambda: ("group must not be None."))
        torch._check(
            (num_cpu_bytes >= 0),
            lambda: (f"num_cpu_bytes must be non-negative, got {num_cpu_bytes=}."),
        )
        torch._check(
            (num_cpu_bytes % buffer_alignment == 0),
            lambda: (
                f"num_cpu_bytes must be 2MB-aligned, got {num_cpu_bytes=}, "
                f"which is not divisible by {buffer_alignment=}."
            ),
        )
        moe_args = (num_max_tokens_per_rank, hidden, num_topk)
        torch._check(
            all(arg is None for arg in moe_args)
            or all(arg is not None for arg in moe_args),
            lambda: (
                "num_max_tokens_per_rank, hidden and num_topk "
                "must be specified together for MoE dispatch/combine."
            ),
        )

        self._group = group
        self._num_cpu_bytes = num_cpu_bytes
        self._rank_id = dist.get_rank(self._group)
        self._ep_world_size = dist.get_world_size(self._group)

        backend = self._group._get_backend(torch.device("npu"))
        self._group_name = backend.get_hccl_comm_name(self._rank_id, init_comm=False)
        if not self._group_name:
            self._group_name = backend.get_hccl_comm_name(self._rank_id, init_comm=True)
        torch._check(
            self._group_name is not None and len(self._group_name) > 0,
            lambda: "HCCL comm name is empty, please check HCCL group initialization.",
        )
        _elastic_buffer_ops = _elastic_buffer_op_builder.load()
        self._runtime = _elastic_buffer_ops.ElasticBuffer(self._group_name, self._num_cpu_bytes)

        self._num_max_tokens_per_rank = num_max_tokens_per_rank
        self._hidden = hidden
        self._num_topk = num_topk
        self._host_pinned_counter = None

    @staticmethod
    def get_engram_storage_size_hint(
        num_entries: int, hidden_size: int, dtype: torch.dtype = torch.bfloat16
    ) -> int:
        """
        Get the minimum CPU buffer size required for Engram storage.
        The returned value is aligned to 2 MB.

        Arguments:
            num_entries: the number of entries in the Engram storage (must be non-negative).
            hidden_size: the hidden dimension of each entry (must be 128-aligned and non-negative).
            dtype: the data type, defaults to `torch.bfloat16`.

        Returns:
            num_cpu_bytes: the recommended CPU buffer size in bytes (2 MB-aligned).
        """
        torch._check(
            num_entries >= 0,
            lambda: f"num_entries must be non-negative, got {num_entries}",
        )
        torch._check(
            hidden_size >= 0,
            lambda: f"hidden_size must be non-negative, got {hidden_size}",
        )
        torch._check(
            hidden_size % 128 == 0,
            lambda: f"hidden_size must be 128-aligned, got {hidden_size}",
        )
        torch._check(
            dtype in (torch.bfloat16, torch.float16, torch.float32),
            lambda: f"dtype must be bfloat16/float16/float32, got {dtype}",
        )
        _elastic_buffer_ops = _elastic_buffer_op_builder.load()
        return _elastic_buffer_ops.ElasticBuffer.get_engram_storage_size_hint(
            num_entries, hidden_size, dtype
        )

    @staticmethod
    def get_moe_ep_ccl_buffer_size(
        world_size: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        topk: int,
    ) -> int:
        def inline_align(value, base):
            return (value + base - 1) // base * base

        torch._check(
            ((num_experts >= 1) and (num_experts <= 2048)),
            lambda: (f"num_experts only support in [1, 2048], but got {num_experts=}."),
        )
        torch._check(
            ((topk >= 1) and (topk <= 32)),
            lambda: (f"topk only support in [1, 32], but got {topk=}."),
        )

        win_addr_align = 512
        ub_align = 32
        max_out_dtype_size = 2
        metadata_dtype_size = 4
        state_dtype_size = 4
        mb_conversion = 1024 * 1024
        local_experts_num = num_experts // world_size

        dispatch_count_size = world_size * inline_align(
            local_experts_num * state_dtype_size, win_addr_align
        )
        dispatch_notify_size = world_size * win_addr_align
        combine_state_size = (
            num_max_tokens_per_rank * topk * win_addr_align
            + world_size * win_addr_align
        )
        state_buffer_size = (
            dispatch_count_size + dispatch_notify_size * 2 + combine_state_size
        )

        metadata_bytes = inline_align(topk * metadata_dtype_size, ub_align)
        hidden_align = inline_align(hidden * max_out_dtype_size, ub_align)
        dispatch_per_slot_bytes = inline_align(
            hidden_align + metadata_bytes * 2 + ub_align, win_addr_align
        )
        combine_per_slot_bytes = inline_align(hidden_align + ub_align, win_addr_align)
        dispatch_buffer_size = (
            world_size * num_max_tokens_per_rank * dispatch_per_slot_bytes
        )
        combine_buffer_size = num_max_tokens_per_rank * combine_per_slot_bytes * topk

        minimum_buffer_size = (
            state_buffer_size + dispatch_buffer_size + combine_buffer_size
        )
        ccl_buffer_size = (
            inline_align(
                inline_align(minimum_buffer_size, mb_conversion) // mb_conversion, 2
            )
            // 2
        )

        return ccl_buffer_size

    def engram_write(self, storage: torch.Tensor) -> None:
        """
        Write data to the host pinned memory of ElasticBuffer.

        Arguments:
            storage: the CPU tensor to write (must be 2D, contiguous, dtype=bf16/fp16/fp32).

        Returns:
            None

        Note: barrier(with_device_sync=True) is called before and after write internally.
        """
        torch._check(
            storage.is_cpu,
            lambda: f"storage must be on CPU, got device: {storage.device}",
        )
        torch._check(
            storage.dim() == 2,
            lambda: f"storage must be 2D, got dimensions: {storage.dim()}",
        )
        torch._check(storage.is_contiguous(), lambda: "storage must be contiguous")
        torch._check(
            storage.dtype in (torch.bfloat16, torch.float16, torch.float32),
            lambda: f"storage dtype must be bfloat16/float16/float32, got: {storage.dtype}",
        )
        torch._check(
            storage.size(1) % 128 == 0,
            lambda: f"storage second dimension must be 128-aligned, got: {storage.size(1)}",
        )
        self._runtime.engram_write(storage)

    def engram_fetch(self, indices: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Fetch Engram data from remote ranks via RDMA.

        Arguments:
            indices: the indices of entries to fetch (must be 1D NPU tensor with dtype=int32).

        Returns:
            wait_callable: a callable that returns the fetched tensor when invoked.
        """
        torch._check(
            indices.device.type == torch.device("npu").type,
            lambda: f"indices must be on NPU, got device: {indices.device}",
        )
        torch._check(
            indices.dim() == 1,
            lambda: f"indices must be 1D, got dimensions: {indices.dim()}",
        )
        torch._check(
            indices.dtype == torch.int32,
            lambda: f"indices dtype must be int32, got: {indices.dtype}",
        )
        return self._runtime.engram_fetch(indices)

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        handle: Optional[EPHandle] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        do_cpu_sync: Optional[bool] = None,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        EPHandle,
    ]:
        self._ensure_moe_config()
        args = self._prepare_dispatch_args(
            x,
            topk_idx,
            handle,
            num_experts,
            num_max_tokens_per_rank,
            expert_alignment,
            do_cpu_sync,
        )
        hp_addr = self._prepare_host_counter(args.do_cpu_sync)

        num_recv_per_rank, num_recv_per_expert, dst_slot = (
            self._runtime.moe_ep_dispatch(
                args.x,
                args.topk_idx,
                topk_weights,
                args.scales,
                args.cached_dst_slot,
                self._ep_world_size,
                self._rank_id,
                args.num_experts,
                args.num_max_tokens_per_rank,
                args.expert_alignment,
                args.do_cpu_sync,
                hp_addr,
            )
        )

        actual_a = self._get_dispatch_recv_count(args)
        recv_x, recv_src_meta, recv_topk_weights, recv_scales = (
            self._allocate_dispatch_outputs(args, actual_a, topk_weights)
        )

        recv_x, recv_src_meta, recv_topk_weights, recv_scales = (
            self._runtime.moe_ep_dispatch_epilogue(
                dst_slot,
                num_recv_per_rank,
                num_recv_per_expert,
                args.cached_recv_src_metadata,
                self._ep_world_size,
                self._rank_id,
                args.num_experts,
                args.num_max_tokens_per_rank,
                args.expert_alignment,
                recv_x,
                recv_src_meta,
                recv_topk_weights,
                recv_scales,
            )
        )

        recv_x = (recv_x, recv_scales) if recv_scales is not None else recv_x
        new_handle = self._make_dispatch_handle(
            args, dst_slot, recv_src_meta, num_recv_per_rank, num_recv_per_expert
        )
        return recv_x, None, recv_topk_weights, new_handle

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        *,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._ensure_moe_config()
        bias_0, bias_1 = self._unpack_bias(bias)
        torch._check(
            ((bias_0 is None) and (bias_1 is None)), lambda: ("bias are not supported.")
        )

        combined_x, combined_topk_weights = (
            self._runtime.moe_ep_combine(
                x,
                handle.topk_idx,
                handle.recv_src_metadata,
                handle.num_recv_tokens_per_expert,
                topk_weights,
                bias_0,
                bias_1,
                self._ep_world_size,
                self._rank_id,
                handle.num_experts,
                handle.num_max_tokens_per_rank,
            )
        )
        return combined_x, combined_topk_weights

    def destroy(self) -> None:
        """
        Destroy the ElasticBuffer and free host pinned memory.

        Returns:
            None
        """
        if self._runtime is not None:
            self._runtime.destroy()
        if self._host_pinned_counter is not None:
            del self._host_pinned_counter
            self._host_pinned_counter = None

    def _ensure_moe_config(self):
        torch._check(
            (self._num_max_tokens_per_rank is not None)
            and (self._hidden is not None)
            and (self._num_topk is not None),
            lambda: (
                "num_max_tokens_per_rank, hidden and num_topk "
                "must be specified to use MoE dispatch/combine."
            ),
        )

    def _prepare_dispatch_args(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor],
        handle: Optional[EPHandle],
        num_experts: Optional[int],
        num_max_tokens_per_rank: Optional[int],
        expert_alignment: Optional[int],
        do_cpu_sync: Optional[bool],
    ) -> _DispatchArgs:
        x, scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            torch._check(
                (topk_idx is None), lambda: ("topk_idx is not supported when cached.")
            )
            torch._check(
                ((do_cpu_sync is None) or (do_cpu_sync is False)),
                lambda: ("do_cpu_sync is not supported when cached."),
            )
            return _DispatchArgs(
                x,
                scales,
                handle.topk_idx,
                handle.dst_buffer_slot_idx,
                handle.recv_src_metadata,
                handle.num_experts,
                handle.num_max_tokens_per_rank,
                handle.expert_alignment,
                False,
                handle.recv_src_metadata.shape[0],
            )

        torch._check(
            (topk_idx is not None), lambda: ("topk_idx are required when no-cached.")
        )
        torch._check(
            (num_experts is not None), lambda: ("num_experts must be specified when no-cached.")
        )
        return _DispatchArgs(
            x,
            scales,
            topk_idx,
            None,
            None,
            num_experts,
            num_max_tokens_per_rank
            if num_max_tokens_per_rank is not None
            else self._num_max_tokens_per_rank,
            1 if expert_alignment is None else expert_alignment,
            True if do_cpu_sync is None else do_cpu_sync,
            None,
        )

    def _prepare_host_counter(self, do_cpu_sync: bool) -> int:
        if not do_cpu_sync:
            return 0
        if self._host_pinned_counter is None:
            _elastic_buffer_ops = _elastic_buffer_op_builder.load()
            self._host_pinned_counter = _elastic_buffer_ops.HostPinnedCounter()
        self._host_pinned_counter.reset()
        return self._host_pinned_counter.device_ptr()

    def _get_dispatch_recv_count(self, args: _DispatchArgs) -> int:
        if args.cached_recv_tokens is not None:
            return args.cached_recv_tokens
        if args.do_cpu_sync:
            return self._host_pinned_counter.spin_wait()
        return (
            self._ep_world_size
            * args.num_max_tokens_per_rank
            * min(self._num_topk, args.num_experts // self._ep_world_size)
        )

    def _allocate_dispatch_outputs(
        self, args: _DispatchArgs, actual_a: int, topk_weights: Optional[torch.Tensor]
    ):
        recv_x = torch.empty(
            (actual_a, self._hidden), dtype=args.x.dtype, device=args.x.device
        )
        recv_src_meta = torch.empty(
            (actual_a, 4), dtype=torch.int32, device=args.x.device
        )
        recv_topk_weights = (
            None
            if topk_weights is None
            else torch.empty((actual_a,), dtype=torch.float32, device=args.x.device)
        )
        recv_scales = (
            None
            if args.scales is None
            else torch.empty(
                (actual_a, args.scales.shape[1]),
                dtype=args.scales.dtype,
                device=args.x.device,
            )
        )
        return recv_x, recv_src_meta, recv_topk_weights, recv_scales

    def _make_dispatch_handle(
        self,
        args: _DispatchArgs,
        dst_slot: torch.Tensor,
        recv_src_meta: torch.Tensor,
        num_recv_per_rank: torch.Tensor,
        num_recv_per_expert: torch.Tensor,
    ) -> EPHandle:
        topk_idx = (
            args.topk_idx
            if args.cached_recv_tokens is not None
            else args.topk_idx.clone()
        )
        return EPHandle(
            dst_buffer_slot_idx=dst_slot,
            recv_src_metadata=recv_src_meta,
            num_recv_tokens_per_rank=num_recv_per_rank,
            num_recv_tokens_per_expert=num_recv_per_expert,
            num_experts=args.num_experts,
            expert_alignment=args.expert_alignment,
            num_max_tokens_per_rank=args.num_max_tokens_per_rank,
            topk_idx=topk_idx,
        )

    def _unpack_bias(self, bias):
        if bias is None:
            return None, None
        if isinstance(bias, torch.Tensor):
            return bias, None
        if isinstance(bias, tuple) and len(bias) == 2:
            return bias[0], bias[1]
        raise TypeError(f"unsupported bias type: {type(bias)}")
