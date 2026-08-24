# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side q2k → k2q CSR builder for minimax_sparse_attention_split_kv."""

from typing import Optional, Tuple, Union

import torch

SeqLike = Union[torch.Tensor, list, tuple]


def _as_int32_1d(values: SeqLike, name: str, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.int32, device=device)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank-1 [B], got {tuple(tensor.shape)}")
    return tensor


def _rows_per_batch(kv_seqlens: torch.Tensor, block_size: int) -> torch.Tensor:
    return (kv_seqlens.to(torch.int64) + int(block_size) - 1) // int(block_size)


def _packed_row_map(rows_per_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Map (batch, local_kv_block) → packed dense CSR row id (MSA / CUDA order)."""
    batch = int(rows_per_batch.numel())
    device = rows_per_batch.device
    if batch == 0:
        return torch.empty((0, 0), dtype=torch.int32, device=device), 0
    max_rows = int(rows_per_batch.max().item())
    if max_rows == 0:
        return torch.empty((batch, 0), dtype=torch.int32, device=device), 0

    k = torch.arange(max_rows, device=device, dtype=torch.int64)
    rows = rows_per_batch.to(torch.int64)
    min_k = torch.minimum(k.unsqueeze(1), rows.unsqueeze(0))
    prefix_by_k = min_k.sum(dim=1)
    greater = rows.unsqueeze(0) > k.unsqueeze(1)
    greater_i = greater.to(torch.int64)
    before = torch.cumsum(greater_i, dim=1) - greater_i
    packed = prefix_by_k.unsqueeze(1) + before
    valid = k.unsqueeze(1) < rows.unsqueeze(0)
    row_map = torch.full((batch, max_rows), -1, dtype=torch.int32, device=device)
    row_map.transpose(0, 1)[valid] = packed[valid].to(torch.int32)
    return row_map, int(rows.sum().item())


def _apply_select_num(
    select_idx: torch.Tensor, select_num_idx: Optional[torch.Tensor]
) -> torch.Tensor:
    if select_num_idx is None:
        return select_idx
    num = torch.as_tensor(select_num_idx, device=select_idx.device)
    if num.dtype != torch.int32:
        num = num.to(torch.int32)
    if num.shape != select_idx.shape[:2]:
        raise ValueError(
            f"select_num_idx shape {tuple(num.shape)} must match "
            f"select_idx[:2] {tuple(select_idx.shape[:2])}"
        )
    top_k = select_idx.size(-1)
    slot = torch.arange(top_k, device=select_idx.device)
    keep = slot.view(1, 1, top_k) < num.unsqueeze(-1)
    return torch.where(keep, select_idx, torch.full_like(select_idx, -1))


def _normalize_select_idx(
    select_idx: torch.Tensor,
    input_layout: str,
    batch: int,
    q_seq_pad: Optional[int],
) -> Tuple[torch.Tensor, int]:
    """Return [kv_heads, Q, topK] and padded Q stride (0 = packed TND)."""
    if select_idx.dtype != torch.int32:
        select_idx = select_idx.to(torch.int32)
    layout = input_layout.upper()
    if select_idx.ndim == 3:
        if layout == "TND":
            return select_idx.contiguous(), 0
        if q_seq_pad is None:
            q_tokens = int(select_idx.size(1))
            if batch <= 0 or q_tokens % batch != 0:
                raise ValueError(
                    f"BNSD/BSND 3D select_idx Q={q_tokens} is not divisible by batch={batch}; "
                    "pass 4D [kv_heads, B, S, topK] or q_seq_pad"
                )
            q_seq_pad = q_tokens // batch
        return select_idx.contiguous(), int(q_seq_pad)
    if select_idx.ndim == 4:
        kv_heads, dim1, dim2, top_k = select_idx.shape
        if layout == "TND":
            raise ValueError("TND select_idx must be [kv_heads, T, topK], got 4D")
        if dim1 == batch:
            q_seq_pad = int(dim2)
            flat = select_idx.reshape(kv_heads, batch * q_seq_pad, top_k)
            return flat.contiguous(), q_seq_pad
        raise ValueError(
            f"4D select_idx must be [kv_heads, B, S, topK] with B={batch}, "
            f"got {tuple(select_idx.shape)}"
        )
    raise ValueError(
        f"select_idx must be [kv_heads, Q, topK] or [kv_heads, B, S, topK], "
        f"got {tuple(select_idx.shape)}"
    )


def _global_to_batch_local(
    select_idx: torch.Tensor,
    q_token_lens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    device = select_idx.device
    q_offs = torch.zeros(q_token_lens.numel() + 1, dtype=torch.int64, device=device)
    q_offs[1:] = torch.cumsum(q_token_lens.to(torch.int64), 0)
    kv_rows = _rows_per_batch(kv_seqlens, block_size)
    kv_offs = torch.zeros(kv_rows.numel() + 1, dtype=torch.int64, device=device)
    kv_offs[1:] = torch.cumsum(kv_rows, 0)

    q_tokens = select_idx.size(1)
    q_ids = torch.arange(q_tokens, device=device, dtype=torch.int64)
    batch_ids = torch.searchsorted(q_offs[1:], q_ids, right=True)
    global_ids = select_idx.to(torch.int64)
    valid = global_ids >= 0
    start = kv_offs[batch_ids].view(1, q_tokens, 1)
    end = kv_offs[batch_ids + 1].view(1, q_tokens, 1)
    in_range = valid & (global_ids >= start) & (global_ids < end)
    local = global_ids - start
    return torch.where(in_range, local.to(torch.int32), torch.full_like(select_idx, -1))


def _mask_padding_tokens(
    select_idx: torch.Tensor, q_token_lens: torch.Tensor, actual_q: torch.Tensor
) -> torch.Tensor:
    """Zero out padded Q tokens when the Q axis is B*S rather than packed T."""
    if torch.equal(q_token_lens, actual_q.to(q_token_lens.dtype)):
        return select_idx
    device = select_idx.device
    q_tokens = select_idx.size(1)
    q_ids = torch.arange(q_tokens, device=device)
    q_offs = torch.zeros(q_token_lens.numel() + 1, dtype=torch.int64, device=device)
    q_offs[1:] = torch.cumsum(q_token_lens.to(torch.int64), 0)
    batch_ids = torch.searchsorted(q_offs[1:], q_ids.to(torch.int64), right=True)
    local_t = q_ids.to(torch.int64) - q_offs[batch_ids]
    keep = local_t < actual_q.to(device=device, dtype=torch.int64)[batch_ids]
    return torch.where(keep.view(1, q_tokens, 1), select_idx, torch.full_like(select_idx, -1))


def _build_csr_from_q2k(
    q2k_indices: torch.Tensor,
    q_token_lens: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    block_size: int,
    q_id_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if q2k_indices.ndim != 3:
        raise ValueError(
            f"q2k_indices must be [kv_heads, Q, topK], got {tuple(q2k_indices.shape)}"
        )

    kv_heads, total_q, top_k = q2k_indices.shape
    q_expected = int(q_token_lens.to(torch.int64).sum().item())
    if total_q != q_expected:
        raise ValueError(
            f"select_idx Q axis ({total_q}) must equal sum(q_token_lens) ({q_expected})"
        )

    device = q2k_indices.device
    rows_per_batch = _rows_per_batch(actual_seq_lengths_kv, block_size)
    row_map, total_rows = _packed_row_map(rows_per_batch)
    nnz_upper = total_q * top_k

    k2q_row_ptr = torch.zeros((kv_heads, total_rows + 1), dtype=torch.int32, device=device)
    k2q_q_indices = torch.full((kv_heads, nnz_upper), -1, dtype=torch.int32, device=device)
    k2q_slot_indices = torch.full((kv_heads, nnz_upper), -1, dtype=torch.int32, device=device)
    if total_rows == 0 or total_q == 0 or top_k == 0:
        return k2q_row_ptr, k2q_q_indices, k2q_slot_indices

    counts = torch.zeros((kv_heads, total_rows), dtype=torch.int32, device=device)
    row_all = torch.empty((kv_heads, nnz_upper), dtype=torch.int64, device=device)
    q_all = torch.empty((kv_heads, nnz_upper), dtype=torch.int32, device=device)
    slot_all = torch.empty((kv_heads, nnz_upper), dtype=torch.int32, device=device)
    valid_all = torch.empty((kv_heads, nnz_upper), dtype=torch.bool, device=device)

    q_offs = torch.zeros(q_token_lens.numel() + 1, dtype=torch.int64, device=device)
    q_offs[1:] = torch.cumsum(q_token_lens.to(torch.int64), 0)
    entry_cursor = 0

    for batch_idx, kv_rows in enumerate(rows_per_batch.tolist()):
        q_start = int(q_offs[batch_idx].item())
        q_end = int(q_offs[batch_idx + 1].item())
        q_len = q_end - q_start
        if q_len == 0:
            continue
        num_entries = q_len * top_k
        q2k_batch = q2k_indices[:, q_start:q_end, :]
        valid_batch = q2k_batch >= 0
        if valid_batch.any():
            max_valid_kv = int(q2k_batch[valid_batch].max().item())
            if max_valid_kv >= int(kv_rows):
                raise ValueError(
                    f"select_idx references kv_block {max_valid_kv} for batch {batch_idx}, "
                    f"but that batch only has {int(kv_rows)} logical kv blocks"
                )

        kv_flat = q2k_batch.reshape(kv_heads, num_entries).long()
        valid_flat = valid_batch.reshape(kv_heads, num_entries)
        safe_kv = torch.where(valid_flat, kv_flat, torch.zeros_like(kv_flat))
        row_flat = row_map[batch_idx][safe_kv]
        local_q = (
            torch.arange(q_len, device=device, dtype=torch.int32)
            .view(1, q_len, 1)
            .expand(kv_heads, q_len, top_k)
            .reshape(kv_heads, num_entries)
        )
        if q_id_stride > 0:
            q_flat = local_q + batch_idx * q_id_stride
        else:
            q_flat = local_q + q_start
        slot_flat = (
            torch.arange(top_k, device=device, dtype=torch.int32)
            .view(1, 1, top_k)
            .expand(kv_heads, q_len, top_k)
            .reshape(kv_heads, num_entries)
        )
        row_all[:, entry_cursor:entry_cursor + num_entries] = row_flat
        q_all[:, entry_cursor:entry_cursor + num_entries] = q_flat
        slot_all[:, entry_cursor:entry_cursor + num_entries] = slot_flat
        valid_all[:, entry_cursor:entry_cursor + num_entries] = valid_flat
        counts.scatter_add_(1, row_flat.to(torch.int64), valid_flat.to(torch.int32))
        entry_cursor += num_entries

    k2q_row_ptr[:, 1:] = counts.cumsum(dim=1, dtype=torch.int32)

    sort_stride = max(total_q, 1)
    invalid_key = total_rows * sort_stride
    sort_keys = torch.full_like(row_all, invalid_key, dtype=torch.int64)
    sort_keys[valid_all] = row_all[valid_all] * sort_stride + q_all[valid_all].to(torch.int64)
    _, sort_idx = sort_keys.sort(dim=1, stable=True)
    sorted_q = q_all.gather(1, sort_idx)
    sorted_slot = slot_all.gather(1, sort_idx)
    valid_counts = valid_all.sum(dim=1)
    write_mask = (
        torch.arange(nnz_upper, device=device).unsqueeze(0).expand(kv_heads, -1)
        < valid_counts.unsqueeze(1)
    )
    k2q_q_indices[write_mask] = sorted_q[write_mask]
    k2q_slot_indices[write_mask] = sorted_slot[write_mask]
    return k2q_row_ptr, k2q_q_indices, k2q_slot_indices


def build_k2q_csr(
    select_idx: torch.Tensor,
    actual_seq_lengths: SeqLike,
    actual_seq_lengths_kv: SeqLike,
    block_size: int,
    *,
    select_num_idx: Optional[torch.Tensor] = None,
    index_mode: str = "batch_local",
    input_layout: str = "TND",
    q_seq_pad: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build packed-row k2q CSR from indexer ``select_idx``.

    Training usually has per-token selected KV block ids, not CSR. This helper
    converts them to the three tensors ``minimax_sparse_attention_split_kv`` needs.

    Args:
        select_idx (Tensor): int32 selected KV block ids.
            TND: ``[kv_heads, T, topK]`` with ``T = sum(actual_seq_lengths)``.
            BNSD/BSND: ``[kv_heads, B, S, topK]`` or ``[kv_heads, B*S, topK]``.
            Invalid / padded slots use ``-1``.
        actual_seq_lengths (Tensor): int32 ``[B]`` real Q lengths (not cumsum).
        actual_seq_lengths_kv (Tensor): int32 ``[B]`` real KV lengths (not cumsum).
        block_size (int): tokens per KV block, typically 128.
        select_num_idx (Tensor, optional): int32 ``[kv_heads, Q]`` valid topK
            count per token. Extra slots are treated as ``-1``.
        index_mode (str): ``"batch_local"`` (default) if ids are per-batch
            logical blocks (lightning indexer / CUDA q2k); ``"global"`` if ids
            are packed across the batch.
        input_layout (str): ``"TND"``, ``"BNSD"`` or ``"BSND"``. Controls the
            q-token ids written into CSR (packed T vs padded ``b*S+t``).
        q_seq_pad (int, optional): padded S for BNSD/BSND when ``select_idx``
            is 3D. Default is ``Q / B``.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            ``k2q_row_ptr`` ``[kv_heads, total_kv_rows+1]``,
            ``k2q_q_indices`` / ``k2q_slot_indices`` ``[kv_heads, T*topK]``.
            Same device as ``select_idx``.
    """
    if not isinstance(select_idx, torch.Tensor):
        select_idx = torch.as_tensor(select_idx, dtype=torch.int32)
    if index_mode not in ("batch_local", "global"):
        raise ValueError(f"index_mode must be 'batch_local' or 'global', got {index_mode!r}")
    layout = input_layout.upper()
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError(f"input_layout must be TND, BNSD or BSND, got {input_layout!r}")

    actual_q = _as_int32_1d(actual_seq_lengths, "actual_seq_lengths", select_idx.device)
    actual_kv = _as_int32_1d(
        actual_seq_lengths_kv, "actual_seq_lengths_kv", select_idx.device
    )
    if actual_q.numel() != actual_kv.numel():
        raise ValueError("actual_seq_lengths and actual_seq_lengths_kv must have the same [B]")

    select_idx, stride = _normalize_select_idx(
        select_idx, layout, int(actual_q.numel()), q_seq_pad
    )
    select_idx = _apply_select_num(select_idx, select_num_idx)

    if layout == "TND":
        q_token_lens = actual_q
        q_id_stride = 0
    else:
        q_token_lens = torch.full_like(actual_q, stride)
        q_id_stride = stride
        select_idx = _mask_padding_tokens(select_idx, q_token_lens, actual_q)

    if index_mode == "global":
        select_idx = _global_to_batch_local(select_idx, q_token_lens, actual_kv, block_size)

    return _build_csr_from_q2k(
        select_idx, q_token_lens, actual_kv, block_size, q_id_stride
    )
