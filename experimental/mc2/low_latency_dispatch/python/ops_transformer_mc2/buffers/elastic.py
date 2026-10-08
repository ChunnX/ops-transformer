# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Adapted from DeepEP; original copyright and license notices follow.

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""Low latency expanded dispatch with explicit SHMEM resource ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from .._native import _unavailable, npu_api, require_native
from .._runtime import Runtime, rank_and_world, runtime_of, warn_unclosed
from .._runtime import integer

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist
    from ..utils.event import EventHandle, EventOverlap


class EPHandle:
    """Routing metadata owner; metadata_only distinguishes notify from payload dispatch."""

    def __init__(
        self,
        do_expand: bool,
        num_experts: int,
        expert_alignment: int,
        num_max_tokens_per_rank: int,
        num_compute_units: int,
        topk_idx: torch.Tensor,
        num_recv_tokens: int,
        num_expanded_tokens: int,
        num_recv_tokens_per_expert_list: list,
        psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
        psum_num_recv_tokens_per_expert: torch.Tensor,
        num_unaligned_recv_tokens_per_expert: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        dst_buffer_slot_idx: torch.Tensor,
        token_metadata_at_forward: Optional[torch.Tensor],
        channel_linked_list: Optional[torch.Tensor],
        is_low_latency: bool = False,
    ):
        require_native("EPHandle.__init__")
        self.do_expand = do_expand
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_compute_units = num_compute_units
        self.topk_idx = topk_idx
        self.num_recv_tokens = num_recv_tokens
        self.num_expanded_tokens = num_expanded_tokens
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.psum_num_recv_tokens_per_scaleup_rank = (
            psum_num_recv_tokens_per_scaleup_rank
        )
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.recv_src_metadata = recv_src_metadata
        self.dst_buffer_slot_idx = dst_buffer_slot_idx
        self.token_metadata_at_forward = token_metadata_at_forward
        self.channel_linked_list = channel_linked_list
        self.is_low_latency = is_low_latency
        self.token_type = None
        self.destination_index = None
        self.relay_read_index = None
        self.source_mask = None
        self.source_weights = None
        self.send_counts = None
        self.expert_recv_counts = None
        self.server_mask = None
        self.dst = None
        self.forward_list = None
        self.forward_counts = None
        self.backward_list = None
        self.backward_counts = None
        self.chunk_ranges = None
        self.chunk_masks = None
        self.rank_rows = None
        self.gather_rows = None
        self.expert_totals = None
        self.weight_return_meta = None
        self.dispatch_dst = None
        self.address_stride = None
        self.rank = None
        self.world_size = None
        self.num_tokens = None
        self.num_topk = None
        self.chunk_tokens = None
        self.trace = None
        self.hidden = None
        self.metadata_only = True
        self.metadata_version = 6
        self._buffer_token = None
        self._routing_versions = {}


class ElasticBuffer:
    """Low latency operator buffer. Destroy collectively before process-group teardown."""

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        num_bytes: Optional[int] = None,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
        prefer_overlap_with_compute: bool = True,
        sl_idx: int = 3,
        num_allocated_qps: int = 0,
        num_cpu_timeout_secs: int = 300,
        num_device_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """Initialize MTE+URMA using a group or explicit rank/world_size.

        Explicit identities bootstrap SHMEM directly and only validate locally.
        All ranks must agree on configuration and collective call order.
        num_bytes is the entire symmetric workspace, cleared at construction.
        Dispatch validates exactly 8 GiB when selecting LL; no extra allocation.
        """
        require_native("ElasticBuffer.__init__")
        if (
            num_cpu_bytes != 0
            or deterministic is not False
            or allow_hybrid_mode is not True
            or allow_multiple_reduction is not True
            or prefer_overlap_with_compute is not True
            or sl_idx != 3
            or type(num_allocated_qps) is not int
            or num_allocated_qps not in (0, 6)
            or num_device_timeout_secs != 100
        ):
            raise NotImplementedError(
                "ElasticBuffer supports default options and num_allocated_qps=0/6"
            )
        if type(use_fp8_dispatch) is not bool:
            raise TypeError("use_fp8_dispatch must be bool")
        if explicitly_destroy is not True:
            raise NotImplementedError(
                "Runtime currently requires explicitly_destroy=True"
            )
        integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 0, 255)
        integer("num_topk", num_topk, 0, 32)
        if type(hidden) is not int or hidden < 0:
            raise ValueError("hidden must be a nonnegative integer")
        if num_bytes is None:
            num_bytes = 8 << 30
        if type(num_bytes) is not int or num_bytes != 8 << 30:
            raise ValueError("Low latency dispatch requires exactly 8 GiB")
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self._handle_token = object()
        self._runtime = Runtime(
            group,
            num_bytes,
            num_cpu_timeout_secs,
            udma=True,
            rank=rank,
            world_size=world_size,
        )

    def __del__(self):
        warn_unclosed(self)

    def destroy(self) -> None:
        """Collectively release owned resources; successful repetition is safe."""
        require_native("ElasticBuffer.destroy")
        runtime_of(self).close()

    def barrier(
        self,
        use_comm_stream: bool = True,
        with_cpu_sync: bool = False,
        sequential: bool = True,
    ) -> None:
        """Run the explicitly selected host-synchronous sequential barrier."""
        require_native("ElasticBuffer.barrier")
        if (
            type(use_comm_stream) is not bool
            or with_cpu_sync is not True
            or sequential is not True
        ):
            raise NotImplementedError(
                "Only the host-synchronous sequential barrier is implemented"
            )
        runtime_of(self).barrier()

    @staticmethod
    def capture() -> EventHandle:
        """Capture a real event on the current framework stream."""
        require_native("ElasticBuffer.capture")
        from ..utils.event import EventHandle

        return EventHandle()

    def get_comm_stream(self) -> torch.Stream:
        """Return the live framework communication stream."""
        require_native("ElasticBuffer.get_comm_stream")
        runtime = runtime_of(self)
        runtime.ensure_alive()
        return runtime.stream

    def _notify_low_latency(
        self, indices: torch.Tensor, num_experts: int
    ) -> Optional[int]:
        """Return exact expanded rows using the existing EP group, or None."""
        runtime = runtime_of(self)
        if runtime.group is None:
            return None
        import torch
        import torch.distributed as dist

        if (
            dist.get_rank(runtime.group) != runtime.rank
            or dist.get_world_size(runtime.group) != runtime.world_size
        ):
            raise ValueError("LL count group must match the buffer EP rank order")
        backend = str(dist.get_backend(runtime.group))
        if backend not in ("hccl", "gloo"):
            raise NotImplementedError(
                "LL counts require the existing HCCL or Gloo EP group"
            )
        ids = indices.reshape(-1)
        valid = (ids >= 0) & (ids < num_experts)
        destinations = torch.where(valid, ids // (num_experts // runtime.world_size), 0)
        counts = torch.zeros(
            runtime.world_size, dtype=torch.int32, device=indices.device
        )
        counts.scatter_add_(0, destinations.to(torch.int64), valid.to(torch.int32))
        if backend == "gloo":
            counts = counts.cpu()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=runtime.group)
        return int(counts[runtime.rank].item())

    @staticmethod
    def _upload_low_latency(data: bytes, device: torch.device) -> torch.Tensor:
        """Upload immutable ABI bytes using allocator-owned pinned Host storage."""
        import torch

        source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        pinned = torch.empty(source.shape, dtype=torch.uint8, pin_memory=True)
        pinned.copy_(source)
        return pinned.to(device=device, non_blocking=True)

    def _low_latency_plan(self, x, topk_idx, topk_weights, num_experts, capacity):
        """Check only Host metadata before selecting LL; never read routes."""
        import torch

        runtime = runtime_of(self)
        hidden, scales = x if isinstance(x, tuple) and len(x) == 2 else (x, None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
            return None
        b, h = hidden.shape
        if (
            type(capacity) is not int
            or not b <= capacity < 256
            or runtime.world_size not in (16, 32, 64, 128)
            or topk_weights is None
        ):
            return None
        if hidden.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
            return None
        if not isinstance(topk_idx, torch.Tensor) or topk_idx.ndim != 2:
            return None
        k = topk_idx.shape[1]
        if (
            type(num_experts) is not int
            or num_experts < runtime.world_size
            or num_experts % runtime.world_size
        ):
            return None
        if (self.hidden and h != self.hidden) or (self.num_topk and k != self.num_topk):
            return None
        if hidden.dtype == torch.float8_e4m3fn:
            if (
                not isinstance(scales, torch.Tensor)
                or scales.dtype not in (torch.int32, torch.float32)
                or tuple(scales.shape) != (b, (h + 127) // 128)
            ):
                return None
        elif scales is not None:
            return None
        for value, dtype, shape in (
            (hidden, hidden.dtype, (b, h)),
            (topk_idx, torch.int64, (b, k)),
            (topk_weights, torch.float32, (b, k)),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != dtype
                or tuple(value.shape) != shape
                or not value.is_contiguous()
                or value.device.type != "npu"
                or value.device.index != runtime.device
            ):
                return None
        if scales is not None and (
            not scales.is_contiguous()
            or scales.device.type != "npu"
            or scales.device.index != runtime.device
        ):
            return None
        return require_native("ElasticBuffer.dispatch").low_latency_layout(
            runtime.world_size,
            b,
            h,
            k,
            num_experts,
            str(hidden.dtype).removeprefix("torch."),
        )

    def _dispatch_low_latency(
        self,
        x,
        topk_idx,
        topk_weights,
        num_experts,
        capacity,
        do_handle_copy,
        do_cpu_sync,
        plan,
    ):
        import torch
        from ..utils.event import EventHandle, EventOverlap

        runtime = runtime_of(self)
        native, npu = require_native("ElasticBuffer.dispatch"), npu_api()
        hidden, scales = x if isinstance(x, tuple) else (x, None)
        b, h = hidden.shape
        k = topk_idx.shape[1]
        dtype = str(hidden.dtype).removeprefix("torch.")
        with runtime.notify_lock:
            if runtime.notify_failed:
                raise RuntimeError(
                    "A previous communication launch failed; restart the distributed job"
                )
            producer = EventHandle()
            with npu.stream(runtime.stream):
                producer.current_stream_wait()
                indices = topk_idx.clone() if do_handle_copy else topk_idx
                for value in (hidden, topk_idx, indices, topk_weights, scales):
                    if value is not None:
                        value.record_stream(runtime.stream)
                # Every allocation is kept alive on error after the first collective.
                retained = (hidden, indices, topk_weights) + (
                    () if scales is None else (scales,)
                )
                runtime.notify_inflight = retained
                try:
                    rows = (
                        self._notify_low_latency(indices, num_experts)
                        if do_cpu_sync is not False
                        else None
                    )
                    c = plan["capacity_rows"]
                    output_rows = (
                        runtime.world_size
                        * capacity
                        * min(k, num_experts // runtime.world_size)
                        if rows is None
                        else c
                    )
                    if runtime.low_latency_comm_args is None:
                        runtime.low_latency_comm_args = self._upload_low_latency(
                            native.low_latency_comm_args(runtime.native), hidden.device
                        )
                    tensors = {
                        "input": hidden,
                        "indices": indices.to(torch.int32),
                        "weights": topk_weights,
                        "scales": scales,
                        "output": torch.empty(
                            (output_rows, h), dtype=hidden.dtype, device=hidden.device
                        ),
                        "output_scales": torch.empty(
                            (output_rows, (h + 127) // 128),
                            dtype=scales.dtype,
                            device=hidden.device,
                        )
                        if scales is not None
                        else None,
                        "expert_recv_counts": torch.empty(
                            (num_experts // runtime.world_size,),
                            dtype=torch.int64,
                            device=hidden.device,
                        ),
                        "send_counts": torch.empty(
                            (num_experts,), dtype=torch.int32, device=hidden.device
                        ),
                        "token_type": torch.empty(
                            (c,), dtype=torch.int32, device=hidden.device
                        ),
                        "destination_index": torch.empty(
                            (c, 3), dtype=torch.int32, device=hidden.device
                        ),
                        "relay_read_index": torch.empty(
                            (c, k, 2), dtype=torch.int32, device=hidden.device
                        ),
                        "source_mask": torch.empty(
                            (b,), dtype=torch.int32, device=hidden.device
                        ),
                        "source_weights": torch.empty(
                            (b, k), dtype=torch.float32, device=hidden.device
                        ),
                        "comm_args": runtime.low_latency_comm_args,
                        "tiling": self._upload_low_latency(
                            native.low_latency_dispatch_tiling(
                                runtime.world_size, b, h, k, num_experts, dtype
                            ),
                            hidden.device,
                        ),
                    }
                    retained += tuple(
                        value for value in tensors.values() if value is not None
                    )
                    runtime.notify_inflight = retained
                    for value in retained:
                        value.record_stream(runtime.stream)
                    native.dispatch_low_latency(
                        runtime.native,
                        tensors,
                        b,
                        h,
                        k,
                        num_experts,
                        output_rows,
                        dtype,
                        runtime.stream.npu_stream,
                    )
                    result = EPHandle(
                        True,
                        num_experts,
                        1,
                        capacity,
                        0,
                        indices,
                        runtime.world_size * capacity if rows is None else None,
                        output_rows if rows is None else rows,
                        [] if rows is None else None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        is_low_latency=True,
                    )
                    result.token_type = tensors["token_type"]
                    result.destination_index = tensors["destination_index"]
                    result.relay_read_index = tensors["relay_read_index"]
                    result.source_mask = tensors["source_mask"]
                    result.source_weights = tensors["source_weights"]
                    result.send_counts = tensors["send_counts"]
                    # Expert counts remain kernel scratch; handle count tensors stay None.
                    result.metadata_only = False
                    result._buffer_token = self._handle_token
                    result.rank, result.world_size = runtime.rank, runtime.world_size
                    result.num_tokens, result.num_topk, result.hidden = b, k, h
                    out, out_scales = tensors["output"], tensors["output_scales"]
                    if rows is not None:
                        out = out.narrow(0, 0, rows)
                        if out_scales is not None:
                            out_scales = out_scales.narrow(0, 0, rows)
                    event = EventOverlap(EventHandle(), retained)
                except Exception:
                    runtime.notify_failed = True
                    raise
            event.current_stream_wait()
            runtime.notify_inflight = None
            return (
                (out, out_scales) if scales is not None else out,
                None,
                None,
                result,
                event,
            )

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        num_compute_units: int = 0,
        num_qps: int = 0,
        previous_event: Optional[EventHandle] = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[EPHandle] = None,
        do_handle_copy: bool = True,
        do_cpu_sync: Optional[bool] = None,
        do_expand: Optional[bool] = None,
        do_zero_padding: bool = False,
        use_aligned_col_major_sf: bool = False,
    ) -> Tuple[
        Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        EPHandle,
        EventOverlap,
    ]:
        """Return (expanded_x, None, None, handle, event).

        Capacity must be below 256; BF16/FP16 and prequantized FP8 are supported.
        Combine applies the saved route weights; do not preweight expert output.
        """
        require_native("ElasticBuffer.dispatch")
        import torch
        from ..utils.event import EventHandle, EventOverlap

        runtime = runtime_of(self)
        runtime.ensure_alive()
        native, npu = require_native("ElasticBuffer.dispatch"), npu_api()
        if isinstance(handle, EPHandle) and handle.is_low_latency:
            raise NotImplementedError("Cached dispatch is not supported for LL handles")
        if handle is None:
            hidden = x[0] if isinstance(x, tuple) and len(x) == 2 else x
            if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
                capacity = (
                    (self.num_max_tokens_per_rank or hidden.shape[0])
                    if num_max_tokens_per_rank is None
                    else num_max_tokens_per_rank
                )
                within_bound = type(capacity) is int and (
                    not self.num_max_tokens_per_rank
                    or capacity <= self.num_max_tokens_per_rank
                )
                ll_options = (
                    within_bound
                    and type(do_handle_copy) is bool
                    and (do_cpu_sync is None or type(do_cpu_sync) is bool)
                    and (do_expand is None or do_expand is True)
                    and (
                        expert_alignment is None
                        or type(expert_alignment) is int
                        and expert_alignment == 1
                    )
                    and type(num_compute_units) is int
                    and num_compute_units == 0
                    and type(num_qps) is int
                    and num_qps == 0
                    and cumulative_local_expert_recv_stats is None
                    and previous_event is None
                    and previous_event_before_epilogue is None
                    and async_with_compute_stream is False
                    and allocate_on_comm_stream is False
                    and do_zero_padding is False
                    and use_aligned_col_major_sf is False
                )
                plan = (
                    self._low_latency_plan(
                        x, topk_idx, topk_weights, num_experts, capacity
                    )
                    if ll_options
                    else None
                )
                if plan is not None:
                    if runtime.capacity != 8 << 30:
                        raise ValueError(
                            "LL dispatch requires Buffer(num_bytes=8589934592), exactly 8 GiB; "
                            f"got {runtime.capacity} bytes for EP{runtime.world_size}"
                        )
                    return self._dispatch_low_latency(
                        x,
                        topk_idx,
                        topk_weights,
                        num_experts,
                        capacity,
                        do_handle_copy,
                        do_cpu_sync,
                        plan,
                    )
        raise ValueError(
            "Unsupported low latency dispatch shape/options; no high throughput fallback is built"
        )

    def _combine_low_latency(
        self,
        x,
        handle,
        topk_weights,
        bias,
        num_compute_units,
        num_qps,
        previous_event,
        previous_event_before_epilogue,
        async_with_compute_stream,
        allocate_on_comm_stream,
    ):
        import torch
        from ..utils.event import EventHandle, EventOverlap

        runtime = runtime_of(self)
        runtime.ensure_alive()
        if handle._buffer_token is not self._handle_token or handle.metadata_only:
            raise ValueError("LL handle must come from this buffer's payload dispatch")
        if topk_weights is not None or bias is not None:
            raise NotImplementedError(
                "LL combine takes its weights from the handle and does not support bias"
            )
        if (
            type(num_compute_units) is not int
            or num_compute_units != 0
            or type(num_qps) is not int
            or num_qps != 0
            or previous_event is not None
            or previous_event_before_epilogue is not None
            or async_with_compute_stream is not False
            or allocate_on_comm_stream is not False
        ):
            raise NotImplementedError(
                "LL combine supports default scheduling and stream options only"
            )
        if (
            not isinstance(x, torch.Tensor)
            or x.dtype not in (torch.bfloat16, torch.float16)
            or tuple(x.shape) != (handle.num_expanded_tokens, handle.hidden)
            or not x.is_contiguous()
            or x.device.type != "npu"
            or x.device.index != runtime.device
        ):
            raise ValueError(
                "LL combine requires contiguous BF16/FP16 expert output matching its handle"
            )
        native, npu = require_native("ElasticBuffer.combine"), npu_api()
        b, k, h, e = (
            handle.num_tokens,
            handle.num_topk,
            handle.hidden,
            handle.num_experts,
        )
        dtype = str(x.dtype).removeprefix("torch.")
        tiling = native.low_latency_combine_tiling(
            runtime.world_size, b, h, k, e, dtype
        )
        with runtime.notify_lock:
            if runtime.notify_failed:
                raise RuntimeError(
                    "A previous communication launch failed; restart the distributed job"
                )
            producer = EventHandle()
            with npu.stream(runtime.stream):
                producer.current_stream_wait()
                # The LL kernel reads selected valid rows, each with exactly 2H
                # bytes. Expert storage need not cover logical capacity C_B.
                tensors = {
                    "input": x,
                    "output": torch.empty((b, h), dtype=x.dtype, device=x.device),
                    "destination_index": handle.destination_index,
                    "send_counts": handle.send_counts,
                    "source_weights": handle.source_weights,
                    "token_type": handle.token_type,
                    "relay_read_index": handle.relay_read_index,
                    "source_mask": handle.source_mask,
                    "comm_args": runtime.low_latency_comm_args,
                    "tiling": self._upload_low_latency(tiling, x.device),
                }
                retained = tuple(tensors.values())
                for value in retained:
                    if not isinstance(value, torch.Tensor):
                        raise ValueError("LL handle is missing a required tensor")
                    if (
                        value.device.type != "npu"
                        or value.device.index != runtime.device
                    ):
                        raise ValueError(
                            "LL handle tensors must belong to the buffer NPU"
                        )
                    value.record_stream(runtime.stream)
                runtime.notify_inflight = retained
                try:
                    native.combine_low_latency(
                        runtime.native,
                        tensors,
                        b,
                        h,
                        k,
                        e,
                        dtype,
                        runtime.stream.npu_stream,
                    )
                    event = EventOverlap(EventHandle(), retained)
                except Exception:
                    runtime.notify_failed = True
                    raise
            event.current_stream_wait()
            runtime.notify_inflight = None
            return tensors["output"], None, event

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        num_compute_units: int = 0,
        num_qps: int = 0,
        previous_event: EventHandle = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Apply saved route weights and reduce BF16/FP16 expert rows to source tokens."""
        native = require_native("ElasticBuffer.combine")
        if not hasattr(native, "combine_low_latency"):
            raise NotImplementedError(
                "Build low_latency_combine/build.sh to enable Combine"
            )
        import torch
        from ..utils.event import EventHandle, EventOverlap

        if isinstance(handle, EPHandle) and handle.is_low_latency:
            return self._combine_low_latency(
                x,
                handle,
                topk_weights,
                bias,
                num_compute_units,
                num_qps,
                previous_event,
                previous_event_before_epilogue,
                async_with_compute_stream,
                allocate_on_comm_stream,
            )
        raise ValueError("combine requires a low latency handle from this buffer")
