import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, Union, Tuple

from .update_context_by_hccl_channel import update_context_by_hccl_channel

_op = torch.ops.npu_ops_transformer


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


def _get_hccl_info(group: dist.ProcessGroup) -> Tuple[str, int, int]:
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank), world, rank


class _Topology:
    __slots__ = ("num_scaleup_ranks", "num_scaleout_ranks", "intra_bw_GBps", "inter_bw_GBps")

    def __init__(self, ep_world_size: int):
        self.num_scaleup_ranks = min(ep_world_size, 128)
        self.num_scaleout_ranks = -(-ep_world_size // 128)
        self.intra_bw_GBps = 400.0
        self.inter_bw_GBps = 50.0


class MoEElasticBuffer:
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        num_experts: int,
        *,
        expert_alignment: int = 1,
    ):
        self._group_ep, ep_world_size_init, self._ep_rank_id = _get_hccl_info(group)

        assert ep_world_size_init > 1
        assert 0 <= self._ep_rank_id < ep_world_size_init
        assert num_experts % ep_world_size_init == 0, \
            f"num_experts({num_experts}) must be divisible by ep_world_size({ep_world_size_init})"

        per_slot_bytes = ((hidden * 2 + 511) // 512) * 512
        win_need = ep_world_size_init * num_max_tokens_per_rank * per_slot_bytes
        self._check_hccl_buffsize(win_need)

        self._ep_world_size = IntWrapper(ep_world_size_init)
        self._ccl_buffer_size = IntWrapper()
        context_struct_size = (2 + 1024)
        context_tensor_size = ((context_struct_size * 8) + 3) // 4
        self._context = torch.zeros(context_tensor_size, dtype=torch.int32).npu()
        update_context_by_hccl_channel(
            self._group_ep, self._ep_world_size, self._ccl_buffer_size, self._context)

        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.group = group
        self.topology = _Topology(self._ep_world_size.value)
        self._host_pinned_counter = None

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        *,
        handle: Optional[EPHandle] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        do_cpu_sync: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], EPHandle]:
        cached = handle is not None
        if cached:
            assert topk_idx is None and topk_weights is None
            assert do_cpu_sync is None or do_cpu_sync is False
            do_cpu_sync = False
            ne, ea, nm = handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank
        else:
            assert topk_idx is not None and topk_weights is not None
            do_cpu_sync = True if do_cpu_sync is None else do_cpu_sync
            ne = num_experts if num_experts is not None else self.num_experts
            nm = num_max_tokens_per_rank if num_max_tokens_per_rank is not None else self.num_max_tokens_per_rank
            ea = expert_alignment if expert_alignment is not None else self.expert_alignment

        cached_dst_slot = handle.dst_buffer_slot_idx if cached else None

        num_recv_per_rank_hp, num_recv_per_expert_hp, \
            dst_slot, recv_src_meta_internal = \
            _op.npu_moe_ep_dispatch(
                self._context, x, topk_idx, topk_weights, cached_dst_slot,
                self._ep_world_size.value, self._ep_rank_id,
                ne, nm, self._ccl_buffer_size.value,
                expert_alignment=ea,
                do_cpu_sync=do_cpu_sync and not cached,
            )

        if cached:
            actual_a = handle.recv_src_metadata.shape[0]
        elif do_cpu_sync:
            if self._host_pinned_counter is None:
                from mc2.common.op_kernel.host_pinned_counter import HostPinnedCounter
                self._host_pinned_counter = HostPinnedCounter()
            actual_a = self._host_pinned_counter.SpinWait(num_recv_per_rank_hp)
        else:
            actual_a = self._ep_world_size.value * nm

        recv_x, recv_topk_idx, recv_topk_weights, \
            recv_src_meta, num_recv_per_rank, num_recv_per_expert = \
            _op.npu_moe_ep_dispatch_epilogue(
                self._context, dst_slot, recv_src_meta_internal,
                num_recv_per_rank_hp, num_recv_per_expert_hp,
                self._ep_world_size.value, self._ep_rank_id,
                ne, nm, self._ccl_buffer_size.value,
                expert_alignment=ea,
            )

        new_handle = EPHandle(
            dst_buffer_slot_idx=dst_slot,
            recv_src_metadata=recv_src_meta,
            num_recv_tokens_per_rank=num_recv_per_rank,
            num_recv_tokens_per_expert=num_recv_per_expert,
            num_experts=ne, expert_alignment=ea, num_max_tokens_per_rank=nm,
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
        assert bias_0 is None and bias_1 is None, "bias not supported in first release"

combined_x, combined_topk_weights = _op.npu_moe_ep_combine(
            self._context, x,
            handle.dst_buffer_slot_idx,
            handle.recv_src_metadata,
            handle.num_recv_tokens_per_rank,
            topk_weights, bias_0, bias_1,
            self._ep_world_size.value, self._ep_rank_id,
            handle.num_experts, handle.num_max_tokens_per_rank, self._ccl_buffer_size.value,
        )
        return combined_x, combined_topk_weights

    @staticmethod
    def _unpack_bias(bias):
        if bias is None:
            return None, None
        if isinstance(bias, torch.Tensor):
            return bias, None
        if isinstance(bias, tuple) and len(bias) == 2:
            return bias[0], bias[1]
        raise TypeError(f"unsupported bias type: {type(bias)}")

    def _check_hccl_buffsize(self, need_bytes: int):
        avail = int(os.environ.get("HCCL_BUFFSIZE", str(200 * 1024 * 1024)))
        if need_bytes > avail:
            raise RuntimeError(
                f"HCCL_BUFFSIZE={avail} insufficient; need {need_bytes} for "
                f"ep={self._ep_world_size.value} x num_max_tokens_per_rank x per_slot_bytes")

    def destroy(self):
        if self._host_pinned_counter is not None:
            del self._host_pinned_counter
            self._host_pinned_counter = None