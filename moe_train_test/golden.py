from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch


Feature = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class GoldenHandle:
    dst_buffer_slot_idx: torch.Tensor
    recv_src_metadata: torch.Tensor
    num_recv_payloads_per_rank: torch.Tensor
    num_recv_tokens_per_expert: torch.Tensor
    topk_idx: torch.Tensor
    num_experts: int
    expert_alignment: int
    num_max_tokens_per_rank: int
    num_expanded_tokens: int

    @property
    def num_recv_payloads(self) -> int:
        return int(self.num_recv_payloads_per_rank.sum().item())

    @property
    def num_valid_expanded_tokens(self) -> int:
        return int(self.num_recv_tokens_per_expert.sum().item())


@dataclass
class DispatchGolden:
    recv_x: Feature
    recv_topk_idx: None
    recv_topk_weights: Optional[torch.Tensor]
    handle: GoldenHandle


@dataclass
class CombineGolden:
    combined_x: Feature
    combined_topk_weights: Optional[torch.Tensor]


def is_tuple_feature(x: Feature) -> bool:
    return isinstance(x, tuple)


def feature_main(x: Feature) -> torch.Tensor:
    return x[0] if is_tuple_feature(x) else x


def feature_scales(x: Feature) -> Optional[torch.Tensor]:
    return x[1] if is_tuple_feature(x) else None


def feature_num_rows(x: Feature) -> int:
    return int(feature_main(x).shape[0])


def feature_width(x: Feature) -> int:
    return int(feature_main(x).shape[1])


def assert_cpu_feature(x: Feature) -> None:
    main = feature_main(x)
    if main.device.type != "cpu":
        raise ValueError("golden 生成只允许 CPU tensor")
    scales = feature_scales(x)
    if scales is not None and scales.device.type != "cpu":
        raise ValueError("golden 生成只允许 CPU tensor")


def _is_supported_fp8_tensor_dtype(dtype: torch.dtype) -> bool:
    return str(dtype).replace("torch.", "") in ("float8_e4m3fn", "float8_e5m2")


def _is_supported_plain_tensor_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.bfloat16, torch.float16)


def _validate_train_x_dtype(x: Feature) -> None:
    main = feature_main(x)
    scales = feature_scales(x)
    if scales is None:
        if not _is_supported_plain_tensor_dtype(main.dtype):
            raise ValueError("x dtype 仅支持 bfloat16/float16 tensor 或 fp8 tuple")
        return
    if not _is_supported_fp8_tensor_dtype(main.dtype):
        raise ValueError("fp8 tuple 的 x dtype 仅支持 float8_e4m3fn 或 float8_e5m2")
    if scales.dtype not in (torch.float32, torch.uint8):
        raise ValueError("fp8 tuple 的 scales dtype 必须为 float32 或 fp8_e8m0(uint8)")


def clone_feature(x: Feature) -> Feature:
    if is_tuple_feature(x):
        return x[0].clone(), x[1].clone()
    return x.clone()


def empty_feature_like(x: Feature, rows: int) -> Feature:
    main = feature_main(x)
    out_main = torch.zeros((rows, main.shape[1]), dtype=main.dtype, device="cpu")
    scales = feature_scales(x)
    if scales is None:
        return out_main
    out_scales = torch.zeros((rows, scales.shape[1]), dtype=scales.dtype, device="cpu")
    return out_main, out_scales


def index_select_feature(x: Feature, indices: torch.Tensor) -> Feature:
    if indices.device.type != "cpu":
        indices = indices.cpu()
    main = feature_main(x).index_select(0, indices)
    scales = feature_scales(x)
    if scales is None:
        return main
    return main, scales.index_select(0, indices)


def feature_set_row(dst: Feature, dst_row: int, src: Feature, src_row: int) -> None:
    feature_main(dst)[dst_row].copy_(feature_main(src)[src_row])
    dst_scales = feature_scales(dst)
    src_scales = feature_scales(src)
    if dst_scales is not None and src_scales is not None:
        dst_scales[dst_row].copy_(src_scales[src_row])


def float32_accumulator_like(x: Feature, rows: int) -> Feature:
    main = feature_main(x)
    acc_main = torch.zeros((rows, main.shape[1]), dtype=torch.float32, device="cpu")
    scales = feature_scales(x)
    if scales is None:
        return acc_main
    acc_scales = torch.zeros((rows, scales.shape[1]), dtype=torch.float32, device="cpu")
    return acc_main, acc_scales


def accumulate_feature_row_float32(dst: Feature, dst_row: int, src: Feature, src_row: int) -> None:
    feature_main(dst)[dst_row] += feature_main(src)[src_row].float()
    dst_scales = feature_scales(dst)
    src_scales = feature_scales(src)
    if dst_scales is not None and src_scales is not None:
        dst_scales[dst_row] += src_scales[src_row].float()


def cast_accumulator_feature(acc: Feature, template: Feature) -> Feature:
    main = feature_main(acc).to(feature_main(template).dtype)
    scales = feature_scales(acc)
    template_scales = feature_scales(template)
    if scales is None or template_scales is None:
        return main
    return main, scales.to(template_scales.dtype)


def feature_equal(lhs: Feature, rhs: Feature) -> bool:
    if is_tuple_feature(lhs) != is_tuple_feature(rhs):
        return False
    if not torch.equal(feature_main(lhs), feature_main(rhs)):
        return False
    lhs_scales = feature_scales(lhs)
    rhs_scales = feature_scales(rhs)
    if lhs_scales is None or rhs_scales is None:
        return lhs_scales is rhs_scales
    return torch.equal(lhs_scales, rhs_scales)


def _validate_dispatch_inputs(
    x_per_rank: Sequence[Feature],
    topk_idx_per_rank: Sequence[torch.Tensor],
    topk_weights_per_rank: Optional[Sequence[torch.Tensor]],
    num_experts: int,
    num_max_tokens_per_rank: int,
) -> Tuple[int, int, int]:
    ep_world_size = len(x_per_rank)
    if ep_world_size == 0:
        raise ValueError("x_per_rank 不能为空")
    if num_experts % ep_world_size != 0:
        raise ValueError("num_experts 必须能整除 ep_world_size")
    if len(topk_idx_per_rank) != ep_world_size:
        raise ValueError("topk_idx_per_rank 长度必须等于 ep_world_size")
    if topk_weights_per_rank is not None and len(topk_weights_per_rank) != ep_world_size:
        raise ValueError("topk_weights_per_rank 长度必须等于 ep_world_size")

    hidden = feature_width(x_per_rank[0])
    num_topk = int(topk_idx_per_rank[0].shape[1])
    for rank, x in enumerate(x_per_rank):
        assert_cpu_feature(x)
        _validate_train_x_dtype(x)
        main = feature_main(x)
        scales = feature_scales(x)
        if len(main.shape) != 2:
            raise ValueError("x 必须是二维 tensor")
        if feature_width(x) != hidden:
            raise ValueError("所有 rank 的 hidden 必须一致")
        if feature_num_rows(x) > num_max_tokens_per_rank:
            raise ValueError("num_tokens 不能超过 num_max_tokens_per_rank")
        if scales is not None and scales.shape[0] != main.shape[0]:
            raise ValueError("scales 第 0 维必须等于 x 第 0 维")
        topk_idx = topk_idx_per_rank[rank]
        if topk_idx.device.type != "cpu":
            raise ValueError("topk_idx 必须是 CPU tensor")
        if topk_idx.shape != (feature_num_rows(x), num_topk):
            raise ValueError("topk_idx shape 必须为 [num_tokens, num_topk]")
        if topk_weights_per_rank is not None:
            topk_weights = topk_weights_per_rank[rank]
            if topk_weights.device.type != "cpu":
                raise ValueError("topk_weights 必须是 CPU tensor")
            if topk_weights.shape != topk_idx.shape:
                raise ValueError("topk_weights shape 必须等于 topk_idx")
    return ep_world_size, hidden, num_topk


def gen_dispatch_golden_cpu(
    x_per_rank: Sequence[Feature],
    topk_idx_per_rank: Sequence[torch.Tensor],
    topk_weights_per_rank: Optional[Sequence[torch.Tensor]],
    *,
    num_experts: int,
    num_max_tokens_per_rank: int,
    expert_alignment: int = 1,
    do_cpu_sync: bool = True,
) -> List[DispatchGolden]:
    """CPU-only dispatch golden。

    不调用 torch.distributed，不依赖 NPU。输出顺序固定为 local expert、source rank、
    dedup slot、source token、topk_pos。
    """
    ep_world_size, _, num_topk = _validate_dispatch_inputs(
        x_per_rank, topk_idx_per_rank, topk_weights_per_rank, num_experts, num_max_tokens_per_rank
    )
    num_local_experts = num_experts // ep_world_size

    dst_slots = [
        torch.full_like(topk_idx_per_rank[src_rank], -1, dtype=torch.int32)
        for src_rank in range(ep_world_size)
    ]
    payload_counts = torch.zeros((ep_world_size, ep_world_size), dtype=torch.int32)
    assignments: List[List[Tuple[int, int, int, int, int]]] = [[] for _ in range(ep_world_size)]

    for src_rank in range(ep_world_size):
        topk_idx = topk_idx_per_rank[src_rank].to(torch.int64)
        num_tokens = int(topk_idx.shape[0])
        slot_counter = [0 for _ in range(ep_world_size)]
        for token_idx in range(num_tokens):
            seen_dst_to_slot = {}
            for topk_pos in range(num_topk):
                expert = int(topk_idx[token_idx, topk_pos].item())
                if expert < 0:
                    dst_slots[src_rank][token_idx, topk_pos] = -1
                    continue
                if expert >= num_experts:
                    raise ValueError(f"expert id {expert} 超过 num_experts={num_experts}")
                dst_rank = expert // num_local_experts
                local_expert = expert % num_local_experts
                if dst_rank not in seen_dst_to_slot:
                    slot = slot_counter[dst_rank]
                    if slot >= num_max_tokens_per_rank:
                        raise ValueError("rank-level dedup payload 数超过 num_max_tokens_per_rank")
                    seen_dst_to_slot[dst_rank] = slot
                    slot_counter[dst_rank] += 1
                    payload_counts[dst_rank, src_rank] += 1
                else:
                    slot = seen_dst_to_slot[dst_rank]
                dst_slots[src_rank][token_idx, topk_pos] = slot
                assignments[dst_rank].append((local_expert, src_rank, slot, token_idx, topk_pos))

    result: List[DispatchGolden] = []
    for dst_rank in range(ep_world_size):
        rank_assignments = sorted(assignments[dst_rank], key=lambda item: item)
        valid_a = len(rank_assignments)
        max_topk_per_local_rank = min(num_topk, num_local_experts)
        a_alloc = valid_a if do_cpu_sync else ep_world_size * num_max_tokens_per_rank * max_topk_per_local_rank
        recv_x = empty_feature_like(x_per_rank[dst_rank], a_alloc)
        recv_src_metadata = torch.full((a_alloc, 4), -1, dtype=torch.int32)
        recv_topk_weights = None
        if topk_weights_per_rank is not None:
            recv_topk_weights = torch.zeros((a_alloc,), dtype=topk_weights_per_rank[dst_rank].dtype)
        num_recv_tokens_per_expert = torch.zeros((num_local_experts,), dtype=torch.int64)

        for row, (local_expert, src_rank, slot, token_idx, topk_pos) in enumerate(rank_assignments):
            feature_set_row(recv_x, row, x_per_rank[src_rank], token_idx)
            recv_src_metadata[row] = torch.tensor([src_rank, token_idx, topk_pos, slot], dtype=torch.int32)
            num_recv_tokens_per_expert[local_expert] += 1
            if recv_topk_weights is not None:
                recv_topk_weights[row] = topk_weights_per_rank[src_rank][token_idx, topk_pos]

        handle = GoldenHandle(
            dst_buffer_slot_idx=dst_slots[dst_rank].clone(),
            recv_src_metadata=recv_src_metadata,
            num_recv_payloads_per_rank=payload_counts[dst_rank].clone(),
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            topk_idx=topk_idx_per_rank[dst_rank].to(torch.int32).clone(),
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_expanded_tokens=a_alloc,
        )
        result.append(DispatchGolden(recv_x, None, recv_topk_weights, handle))
    return result


def gen_combine_golden_cpu(
    expanded_x_per_rank: Sequence[Feature],
    handles: Sequence[GoldenHandle],
    *,
    topk_weights_per_rank: Optional[Sequence[torch.Tensor]] = None,
) -> List[CombineGolden]:
    """CPU-only combine golden。

    Combine 依据 handle.recv_src_metadata 将 expanded rows 发回源 rank 并做 sum。
    """
    ep_world_size = len(expanded_x_per_rank)
    if len(handles) != ep_world_size:
        raise ValueError("handles 长度必须等于 ep_world_size")
    if topk_weights_per_rank is not None and len(topk_weights_per_rank) != ep_world_size:
        raise ValueError("topk_weights_per_rank 长度必须等于 ep_world_size")

    combined_x_acc = [
        float32_accumulator_like(expanded_x_per_rank[rank], int(handles[rank].topk_idx.shape[0]))
        for rank in range(ep_world_size)
    ]
    combined_topk_weights: List[Optional[torch.Tensor]] = []
    for rank, handle in enumerate(handles):
        if handle.topk_idx.device.type != "cpu":
            raise ValueError("handle 必须是 CPU golden handle")
        if topk_weights_per_rank is None:
            combined_topk_weights.append(None)
        else:
            combined_topk_weights.append(torch.zeros_like(handle.topk_idx, dtype=topk_weights_per_rank[rank].dtype))

    combine_entries: List[Tuple[int, int, int, int, int]] = []
    for sender_rank in range(ep_world_size):
        x = expanded_x_per_rank[sender_rank]
        assert_cpu_feature(x)
        metadata = handles[sender_rank].recv_src_metadata
        valid_rows = handles[sender_rank].num_valid_expanded_tokens
        if feature_num_rows(x) < valid_rows:
            raise ValueError("expanded_x 有效行数小于 handle 有效行数")
        if topk_weights_per_rank is not None and topk_weights_per_rank[sender_rank].shape[0] < valid_rows:
            raise ValueError("expanded topk_weights 有效行数不足")
        for row in range(valid_rows):
            src_rank, token_idx, topk_pos = [int(v) for v in metadata[row, :3].tolist()]
            if src_rank < 0:
                continue
            if src_rank >= ep_world_size:
                raise ValueError("recv_src_metadata src_rank 超过 ep_world_size")
            if token_idx < 0 or token_idx >= handles[src_rank].topk_idx.shape[0]:
                raise ValueError("recv_src_metadata token_idx 越界")
            if topk_pos < 0 or topk_pos >= handles[src_rank].topk_idx.shape[1]:
                raise ValueError("recv_src_metadata topk_pos 越界")
            combine_entries.append((src_rank, token_idx, topk_pos, sender_rank, row))

    # 对齐参考脚本 middle_idx 反排序后的 combine 顺序：全局原始 token 顺序、topk 顺序。
    for src_rank, token_idx, topk_pos, sender_rank, row in sorted(combine_entries):
        accumulate_feature_row_float32(combined_x_acc[src_rank], token_idx, expanded_x_per_rank[sender_rank], row)
        if topk_weights_per_rank is not None:
            combined_topk_weights[src_rank][token_idx, topk_pos] = topk_weights_per_rank[sender_rank][row]

    combined_x = [
        cast_accumulator_feature(combined_x_acc[rank], expanded_x_per_rank[rank])
        for rank in range(ep_world_size)
    ]

    return [
        CombineGolden(combined_x[rank], combined_topk_weights[rank])
        for rank in range(ep_world_size)
    ]


def expected_dispatch_rows_from_metadata(
    x_per_rank: Sequence[Feature],
    topk_weights_per_rank: Optional[Sequence[torch.Tensor]],
    metadata: torch.Tensor,
) -> Tuple[Feature, Optional[torch.Tensor]]:
    """根据 metadata 反查 dispatch 每一行应该来自哪个输入 token。"""
    if metadata.device.type != "cpu":
        metadata = metadata.cpu()
    rows = int(metadata.shape[0])
    template_rank = 0
    expected_x = empty_feature_like(x_per_rank[template_rank], rows)
    expected_weights = None
    if topk_weights_per_rank is not None:
        expected_weights = torch.zeros((rows,), dtype=topk_weights_per_rank[template_rank].dtype)
    for row in range(rows):
        src_rank, token_idx, topk_pos = [int(v) for v in metadata[row, :3].tolist()]
        if src_rank < 0:
            continue
        feature_set_row(expected_x, row, x_per_rank[src_rank], token_idx)
        if expected_weights is not None:
            expected_weights[row] = topk_weights_per_rank[src_rank][token_idx, topk_pos]
    return expected_x, expected_weights
