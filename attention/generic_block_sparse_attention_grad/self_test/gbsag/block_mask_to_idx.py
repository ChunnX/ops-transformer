"""Block sparse mask to ordered sparse block indices."""

from __future__ import annotations

import torch


def blcok_sparse_mask_to_idx(
    block_sparse_mask: torch.Tensor,
    block_shape: list,
    actual_seq_lengths: list | None,
    actual_seq_lengths_kv: list | None,
    max_q_seq: int,
    max_kv_seq: int,
    idx_flag: str,
    q_head_num: int,
    kv_head_num: int,
    is_packed_gqa: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a block mask to ordered sparse indices and row counts.

    ``block_sparse_mask`` has shape ``[B, N1, QBlocks, KVBlocks]``.  With
    ``idx_flag == "q2kvIdx"``, each Q block row contains ascending KV block
    indices.  With ``idx_flag == "kv2qIdx"``, each KV block row contains
    ascending Q block indices.  Slots after the row count are filled with
    ``-1`` and must be ignored by the consumer.

    Output shapes are ``[B,N1,QBlocks,max_kv_seq]`` for q2kvIdx and
    ``[B,N2,KVBlocks,max_q_seq]`` for kv2qIdx; counts omit the padded slots.
    ``actual_seq_lengths`` and ``actual_seq_lengths_kv`` are per-batch lists.
    Passing ``None`` means BNSD: every batch uses ``max_q_seq`` or
    ``max_kv_seq`` respectively.
    """
    if idx_flag not in ("q2kvIdx", "kv2qIdx"):
        raise ValueError(f"unsupported idx_flag: {idx_flag!r}")
    if not is_packed_gqa:
        raise ValueError("only packed GQA sparse patterns are supported")
    N1 = q_head_num
    N2 = kv_head_num
    batch_size, mask_heads, q_block_count, kv_block_count = block_sparse_mask.shape
    if mask_heads != N1:
        raise ValueError(f"mask head count {mask_heads} != q_head_num {N1}")
    if N1 % N2 != 0:
        raise ValueError("q_head_num must be divisible by kv_head_num")
    q_block_size, kv_block_size = block_shape

    if actual_seq_lengths is None:
        q_valid_blocks = torch.full(
            (batch_size,),
            q_block_count,
            dtype=torch.int32,
            device=block_sparse_mask.device,
        )
    else:
        q_valid_blocks = torch.tensor(
            [
                (length + q_block_size - 1) // q_block_size
                for length in actual_seq_lengths
            ],
            dtype=torch.int32,
            device=block_sparse_mask.device,
        )

    if actual_seq_lengths_kv is None:
        kv_valid_blocks = torch.full(
            (batch_size,),
            kv_block_count,
            dtype=torch.int32,
            device=block_sparse_mask.device,
        )
    else:
        kv_valid_blocks = torch.tensor(
            [
                (length + kv_block_size - 1) // kv_block_size
                for length in actual_seq_lengths_kv
            ],
            dtype=torch.int32,
            device=block_sparse_mask.device,
        )

    q_ids = torch.arange(q_block_count, device=block_sparse_mask.device).view(
        1, 1, q_block_count, 1
    )
    kv_ids = torch.arange(kv_block_count, device=block_sparse_mask.device).view(
        1, 1, 1, kv_block_count
    )
    valid_q = q_ids < q_valid_blocks.view(batch_size, 1, 1, 1)
    valid_kv = kv_ids < kv_valid_blocks.view(batch_size, 1, 1, 1)
    effective_mask = block_sparse_mask.to(torch.bool) & valid_q & valid_kv

    if idx_flag == "q2kvIdx":
        row_mask = effective_mask
        idx = torch.full(
            (batch_size, N1, q_block_count, max_kv_seq),
            -1,
            dtype=torch.int32,
            device=block_sparse_mask.device,
        )
    else:
        # 同一个 GQA group 内的 Q head 共享 KV->Q pattern，取每组首个 Q head。
        row_mask = effective_mask[:, :: (N1 // N2), :, :].transpose(-1, -2).contiguous()
        idx = torch.full(
            (batch_size, N2, kv_block_count, max_q_seq),
            -1,
            dtype=torch.int32,
            device=block_sparse_mask.device,
        )
    count = row_mask.sum(dim=-1, dtype=torch.int32)

    # 沿源 block 维计算有序 rank，再一次性 scatter 到输出 slot，避免 B/H/block
    # 三层 Python 循环和逐行 nonzero。额外保留一个 sentinel slot，确保无效位置
    # 不会覆盖最后一个有效 slot；最终裁剪回需求中的 max_q/max_kv_seq 宽度。
    slot_limit = max_kv_seq if idx_flag == "q2kvIdx" else max_q_seq
    rank = row_mask.to(torch.int64).cumsum(dim=-1) - 1
    sentinel = torch.full_like(rank, slot_limit)
    scatter_slot = torch.where(row_mask, rank, sentinel)
    source_ids = torch.arange(
        row_mask.shape[-1], device=row_mask.device, dtype=torch.int32
    )
    source_ids = source_ids.view(1, 1, 1, -1).expand_as(row_mask)
    packed = torch.full(
        (*row_mask.shape[:-1], slot_limit + 1),
        -1,
        dtype=torch.int32,
        device=row_mask.device,
    )
    packed.scatter_(dim=-1, index=scatter_slot, src=source_ids)
    idx.copy_(packed[..., :slot_limit])

    return idx, count


def main() -> None:
    """Construct deterministic data and execute both index directions."""

    # [B, N1, X, Y]
    block_sparse_mask = torch.tensor(
        [
            [[[1, 0, 1], [0, 1, 1]], [[1, 0, 1], [0, 1, 1]]],
            [[[1, 1, 0], [0, 1, 0]], [[1, 0, 1], [0, 1, 1]]],
        ],
        dtype=torch.bool,
    )
    # 将两个 KV-head pattern 复制到各自的两个 Q-head，构造 packed GQA 输入。
    block_sparse_mask = block_sparse_mask.repeat_interleave(2, dim=1)
    block_shape = [2, 2]
    max_q_seq = 5
    max_kv_seq = 5

    # [B, N1, X, Y]
    # [B, N1, X]
    print("block_sparse_mask")
    print(block_sparse_mask)
    print(block_sparse_mask.shape)
    q2kv_idx, q2kv_count = blcok_sparse_mask_to_idx(
        block_sparse_mask,
        block_shape,
        [3, 5],
        [5, 3],
        max_q_seq,
        max_kv_seq,
        "q2kvIdx",
        q_head_num=4,
        kv_head_num=2,
        is_packed_gqa=True,
    )

    # [B, N2, Y, X]
    # [B, N2, Y]
    kv2q_idx, kv2q_count = blcok_sparse_mask_to_idx(
        block_sparse_mask,
        block_shape,
        None,
        None,
        max_q_seq,
        max_kv_seq,
        "kv2qIdx",
        q_head_num=4,
        kv_head_num=2,
        is_packed_gqa=True,
    )
    print("q2kvIdx:")
    print(q2kv_idx)
    print(q2kv_idx.shape)
    print(q2kv_count)
    print(q2kv_count.shape)
    print("kv2qIdx:")
    print(kv2q_idx)
    print(kv2q_idx.shape)
    print(kv2q_count)
    print(kv2q_count.shape)


if __name__ == "__main__":
    main()
