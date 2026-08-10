#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK metadata-first adapter for the installed quant_block_sparse_attn API."""

import math
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch

_CUSTOM_OPS_LOADED = False


def _ensure_cann_ops():
    global _CUSTOM_OPS_LOADED
    if not _CUSTOM_OPS_LOADED:
        pytest_dir = str(Path(__file__).resolve().parents[1] / "pytest")
        if pytest_dir not in sys.path:
            sys.path.insert(0, pytest_dir)
        import custom_ops  # noqa: F401 - loads .so + registers torch.ops.custom namespace

        _CUSTOM_OPS_LOADED = True


_FP32_BYTES = 4


def _pack_combined_kv(key, value, k_descale):
    """Pack separate PA key/value/k_descale into a combined KV storage.

    Layout per physical block: [key_segment (FP8)] [value_segment (FP8)] [k_scale_segment (FP32)]
    Returns 4D PA views whose shape matches the operator interface and whose block stride
    points at the full physical KV-cache block.
    """
    num_blocks, n2, block_size, d = (
        int(key.shape[0]),
        int(key.shape[1]),
        int(key.shape[2]),
        int(key.shape[3]),
    )
    dv = int(value.shape[-1])

    key_seg = n2 * block_size * d
    value_seg = n2 * block_size * dv
    k_scale_seg = n2 * block_size * _FP32_BYTES
    pa_block_stride = key_seg + value_seg + k_scale_seg

    storage = torch.zeros(
        (num_blocks * pa_block_stride,), dtype=torch.uint8, device=key.device
    )
    fp8_storage = storage.view(torch.float8_e4m3fn)
    fp32_storage = storage.view(torch.float32)

    key_view = torch.as_strided(
        fp8_storage, key.shape, (pa_block_stride, block_size * d, d, 1), 0
    )
    value_view = torch.as_strided(
        fp8_storage, value.shape, (pa_block_stride, block_size * dv, dv, 1), key_seg
    )
    k_scale_shape = tuple(k_descale.shape) if k_descale.dim() == 4 else tuple(k_descale.shape) + (1,)
    k_scale_view = torch.as_strided(
        fp32_storage,
        k_scale_shape,
        (pa_block_stride // _FP32_BYTES, block_size, 1, 1),
        (key_seg + value_seg) // _FP32_BYTES,
    )

    key_view.copy_(key)
    value_view.copy_(value)
    k_scale_view.copy_(k_descale if k_descale.dim() == 4 else k_descale.unsqueeze(-1))

    return key_view, value_view, k_scale_view


def _make_fp8_block_table(
    batch, seq_len, block_size, blocknum, pattern, seed, device, rng=None
):
    """Match the FP8 input generator's logical-to-physical PA mapping."""
    width = math.ceil(seq_len / block_size)
    needed_block_num = batch * width
    if blocknum <= 0:
        raise ValueError(f"blocknum must be positive, got {blocknum}")

    rng = rng if rng is not None else random.Random(int(seed))
    if blocknum >= needed_block_num:
        block_table = torch.empty((batch, width), dtype=torch.int32, device=device)
        for batch_idx in range(batch):
            ids = list(range(batch_idx * width, (batch_idx + 1) * width))
            if pattern == "reverse":
                ids.reverse()
            elif pattern == "random":
                rng.shuffle(ids)
            elif pattern != "sequential":
                raise ValueError(f"unsupported block_table_pattern: {pattern}")
            block_table[batch_idx] = torch.tensor(
                ids, dtype=torch.int32, device=device
            )
        return block_table

    if blocknum < width:
        shared_cols = width - blocknum + 1
        if shared_cols <= 0 or shared_cols > width:
            raise ValueError(f"blocknum={blocknum} out of range [1, {width}]")
        block_table = torch.zeros(
            (batch, width), dtype=torch.int32, device=device
        )
        for batch_idx in range(batch):
            for block_idx in range(shared_cols, width):
                block_table[batch_idx, block_idx] = block_idx - shared_cols + 1
        return block_table

    block_table = torch.empty((batch, width), dtype=torch.int32, device=device)
    shared_count = needed_block_num - blocknum
    full_shared_cols = shared_count // (batch - 1)
    partial_share = shared_count % (batch - 1)
    next_id = 0
    col = 0
    for _ in range(full_shared_cols):
        block_table[:, col] = next_id
        next_id += 1
        col += 1
    if partial_share > 0:
        shared_id = next_id
        next_id += 1
        block_table[: partial_share + 1, col] = shared_id
        for batch_idx in range(partial_share + 1, batch):
            block_table[batch_idx, col] = next_id
            next_id += 1
        col += 1
    for batch_idx in range(batch):
        for block_idx in range(col, width):
            block_table[batch_idx, block_idx] = next_id
            next_id += 1
    if next_id != blocknum:
        raise AssertionError(
            f"block_table generated {next_id} physical blocks, expected {blocknum}"
        )
    return block_table


def _auto_generate_params(
    query,
    key,
    sparse_indices,
    seqused_q,
    seqused_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    block_table,
    max_seqlen_q,
    max_seqlen_kv,
    layout_q,
    sparse_kv_block_size,
    block_table_pattern,
    seed,
):
    """Auto-generate missing seqused/cu_seqlens/block_table from available info."""
    B = int(sparse_indices.shape[0]) if sparse_indices is not None else 1

    if max_seqlen_q <= 0:
        if layout_q == "BSND":
            max_seqlen_q = int(query.shape[1])
        elif layout_q == "NTD":
            max_seqlen_q = int(query.shape[1])
        else:
            max_seqlen_q = int(query.shape[0])

    if max_seqlen_kv <= 0:
        if key.dim() == 4:
            max_seqlen_kv = int(key.shape[0]) * int(key.shape[2])
        else:
            max_seqlen_kv = int(key.shape[0])

    if seqused_q is None:
        seqused_q = torch.full(
            (B,), max_seqlen_q, dtype=torch.int32, device=query.device
        )
    if seqused_kv is None:
        seqused_kv = torch.full(
            (B,), max_seqlen_kv, dtype=torch.int32, device=query.device
        )
    if cu_seqlens_q is None:
        cu = torch.zeros(B + 1, dtype=torch.int32, device=query.device)
        for i in range(B):
            cu[i + 1] = cu[i] + int(seqused_q[i].item())
        cu_seqlens_q = cu
    if cu_seqlens_kv is None:
        cu = torch.zeros(B + 1, dtype=torch.int32, device=query.device)
        for i in range(B):
            cu[i + 1] = cu[i] + int(seqused_kv[i].item())
        cu_seqlens_kv = cu
    if block_table is None:
        blocknum = int(key.shape[0]) if key.dim() == 4 else B * math.ceil(
            max_seqlen_kv / sparse_kv_block_size
        )
        block_table = _make_fp8_block_table(
            B,
            max_seqlen_kv,
            sparse_kv_block_size,
            blocknum,
            block_table_pattern,
            seed,
            query.device,
        )

    return seqused_q, seqused_kv, cu_seqlens_q, cu_seqlens_kv, block_table


def _auto_generate_mxfp8_params(
    query,
    key,
    sparse_indices,
    seqused_q,
    seqused_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    block_table,
    layout_q,
    page_block_size,
    actual_seq_q=None,
    actual_seq_kv=None,
    empty_actual_seq=False,
    block_table_pattern="sequential",
    seed=0,
    sparse_q_block_size=128,
    sparse_kv_block_size=128,
    sparse_count=0,
    sparse_pattern="sequential",
    mask_mode=3,
    blocknum=None,
):
    """Generate MXFP8 sequence tensors and sparse-aware PA block mapping."""
    B = int(sparse_indices.shape[0]) if sparse_indices is not None else 1
    if layout_q != "TND":
        raise ValueError(f"MXFP8 requires layout_q='TND', got {layout_q!r}")
    if B <= 0:
        raise ValueError(f"MXFP8 requires a positive batch size, got B={B}")
    if actual_seq_q is None or actual_seq_kv is None:
        raise ValueError(
            "MXFP8 requires explicit actual_seq_q and actual_seq_kv values"
        )
    q_lengths = [int(value) for value in actual_seq_q]
    kv_lengths = [int(value) for value in actual_seq_kv]
    if len(q_lengths) != B or len(kv_lengths) != B:
        raise ValueError(
            f"actual sequence lengths must match B={B}, got "
            f"q={len(q_lengths)}, kv={len(kv_lengths)}"
        )
    if any(length < 0 for length in q_lengths + kv_lengths):
        raise ValueError("MXFP8 sequence lengths must be non-negative")
    if sum(q_lengths) != int(query.shape[0]):
        raise ValueError(
            f"sum(actual_seq_q) must equal query T={int(query.shape[0])}, "
            f"got {sum(q_lengths)}"
        )

    if seqused_q is None:
        seqused_q = torch.tensor(
            [] if empty_actual_seq else q_lengths,
            dtype=torch.int32,
            device=query.device,
        )
    if seqused_kv is None:
        seqused_kv = torch.tensor(kv_lengths, dtype=torch.int32, device=query.device)
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor(
            [0] + [sum(q_lengths[: index + 1]) for index in range(B)],
            dtype=torch.int32,
            device=query.device,
        )
    if cu_seqlens_kv is None:
        cu_seqlens_kv = torch.tensor(
            [0] + [sum(kv_lengths[: index + 1]) for index in range(B)],
            dtype=torch.int32,
            device=query.device,
        )
    if block_table is None:
        block_counts = [math.ceil(length / page_block_size) for length in kv_lengths]
        max_block_count = max(block_counts, default=0)
        block_table = torch.full(
            (B, max_block_count), -1, dtype=torch.int32, device=query.device
        )
        total_logical_blocks = sum(block_counts)
        total_physical_blocks = (
            int(blocknum)
            if blocknum is not None and int(blocknum) > 0
            else total_logical_blocks
        )
        if total_physical_blocks <= 0:
            raise ValueError(
                "physical block count derived from blocknum/actual_seq_kv must be > 0"
            )

        rng = random.Random(int(seed))
        if layout_q == "BSND":
            num_heads_q = int(query.shape[2])
        elif layout_q == "NTD":
            num_heads_q = int(query.shape[0])
        else:
            num_heads_q = int(query.shape[1])
        q_block_max = math.ceil(max(q_lengths) / int(sparse_q_block_size))
        if (
            sparse_kv_block_size <= 0
            or page_block_size % int(sparse_kv_block_size) != 0
        ):
            raise ValueError(
                "sparse_kv_block_size must be a positive divisor of page block size"
            )
        sparse_blocks_per_page = page_block_size // int(sparse_kv_block_size)

        needed_logical_ids = [set() for _ in range(B)]
        for batch_idx in range(B):
            q_block_count = math.ceil(
                q_lengths[batch_idx] / int(sparse_q_block_size)
            )
            kv_block_count = math.ceil(
                kv_lengths[batch_idx] / int(sparse_kv_block_size)
            )
            for q_block_idx in range(q_block_max):
                if int(mask_mode) == 0:
                    allowed = list(range(kv_block_count))
                elif int(mask_mode) == 3:
                    max_token = (
                        (q_block_idx + 1) * int(sparse_q_block_size)
                        - 1
                        + kv_lengths[batch_idx]
                        - q_lengths[batch_idx]
                    )
                    max_sparse_block = (
                        -1
                        if max_token < 0
                        else min(
                            kv_block_count - 1,
                            max_token // int(sparse_kv_block_size),
                        )
                    )
                    allowed = list(range(max_sparse_block + 1))
                else:
                    raise ValueError(f"unsupported mask_mode: {mask_mode}")

                for _ in range(num_heads_q):
                    if q_block_idx >= q_block_count:
                        selected = []
                    elif sparse_count <= 0 or not allowed or sparse_pattern == "empty":
                        selected = []
                    elif sparse_pattern in ("sequential", "dense", "causal"):
                        selected = allowed[: min(int(sparse_count), len(allowed))]
                    elif sparse_pattern == "reverse":
                        count = min(int(sparse_count), len(allowed))
                        selected = list(reversed(allowed[-count:]))
                    elif sparse_pattern == "tail":
                        count = min(int(sparse_count), len(allowed))
                        selected = allowed[: max(0, count - 1)]
                        if allowed[-1] not in selected:
                            selected.append(allowed[-1])
                        selected = selected[: int(sparse_count)]
                    elif sparse_pattern == "random":
                        selected = allowed[:]
                        rng.shuffle(selected)
                        selected = selected[: min(int(sparse_count), len(selected))]
                    else:
                        raise ValueError(
                            f"unsupported sparse_pattern: {sparse_pattern}"
                        )
                    needed_logical_ids[batch_idx].update(
                        block_id // sparse_blocks_per_page for block_id in selected
                    )

        physical_ids = list(range(total_physical_blocks))
        if block_table_pattern == "reverse":
            physical_ids.reverse()
        elif block_table_pattern == "random":
            rng.shuffle(physical_ids)
        elif block_table_pattern != "sequential":
            raise ValueError(
                f"unsupported block_table_pattern: {block_table_pattern}"
            )
        next_physical_id = 0
        for batch_idx, block_count in enumerate(block_counts):
            needed = sorted(needed_logical_ids[batch_idx])
            if not needed:
                needed = list(range(block_count))
            for logical_id in needed:
                block_table[batch_idx, logical_id] = physical_ids[
                    next_physical_id % total_physical_blocks
                ]
                next_physical_id += 1

    return seqused_q, seqused_kv, cu_seqlens_q, cu_seqlens_kv, block_table


def _generate_params_for_mode(
    query,
    key,
    sparse_indices,
    seqused_q,
    seqused_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    block_table,
    *,
    quant_mode,
    max_seqlen_q,
    max_seqlen_kv,
    layout_q,
    page_block_size,
    actual_seq_q,
    actual_seq_kv,
    empty_actual_seq,
    block_table_pattern,
    seed,
    sparse_q_block_size,
    sparse_kv_block_size,
    sparse_count,
    sparse_pattern,
    mask_mode,
    blocknum,
):
    if int(quant_mode) == 2:
        return _auto_generate_mxfp8_params(
            query,
            key,
            sparse_indices,
            seqused_q,
            seqused_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            block_table,
            layout_q,
            page_block_size,
            actual_seq_q,
            actual_seq_kv,
            empty_actual_seq,
            block_table_pattern,
            seed,
            sparse_q_block_size,
            sparse_kv_block_size,
            sparse_count,
            sparse_pattern,
            mask_mode,
            blocknum,
        )
    return _auto_generate_params(
        query,
        key,
        sparse_indices,
        seqused_q,
        seqused_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        block_table,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        sparse_kv_block_size,
        block_table_pattern,
        seed,
    )


class QuantBlockSparseAttnMetadataBuilder:
    """Derive and create the metadata consumed by the main operator."""

    @staticmethod
    def max_seq(cu_seqlens, seqused, fallback):
        if seqused is not None and seqused.numel() > 0:
            return int(seqused.detach().cpu().max().item())
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            values = cu_seqlens.detach().cpu()
            return int((values[1:] - values[:-1]).max().item())
        return int(fallback)

    @staticmethod
    def batch_size(q, layout_q, cu_seqlens_q, seqused_q, sparse_indices=None):
        if layout_q == "BSND":
            return int(q.shape[0])
        if seqused_q is not None:
            return int(seqused_q.numel())
        if cu_seqlens_q is not None:
            return max(int(cu_seqlens_q.numel()) - 1, 0)
        if sparse_indices is not None:
            return int(sparse_indices.shape[0])
        return 1

    @classmethod
    def build(
        cls,
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        *,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=None,
        block_table=None,
        softmax_scale=1.0,
        sparse_q_block_size=128,
        sparse_kv_block_size=128,
        layout_q="BSND",
        layout_kv="PA_BNSD",
        layout_sparse_indices="B_N_Qb_Kb",
        mask_mode=3,
        quant_mode=1,
        max_seqlen_q=0,
        max_seqlen_kv=0,
    ):
        _ensure_cann_ops()
        if layout_q == "BSND":
            num_heads_q = int(query.shape[2])
            q_fallback = int(query.shape[1])
        elif layout_q == "NTD":
            num_heads_q = int(query.shape[0])
            q_fallback = int(query.shape[1])
        else:
            num_heads_q = int(query.shape[1])
            q_fallback = int(query.shape[0])
        num_heads_kv = int(key.shape[1])
        kv_fallback = (
            int(key.shape[0] * key.shape[2]) if key.dim() == 4 else int(key.shape[0])
        )
        metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
            sparse_seq_len,
            num_heads_q,
            num_heads_kv,
            int(query.shape[-1]),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            batch_size=cls.batch_size(
                query, layout_q, cu_seqlens_q, seqused_q, sparse_indices
            ),
            sparse_block_size_q=int(sparse_q_block_size),
            sparse_block_size_k=int(sparse_kv_block_size),
            quant_mode=int(quant_mode),
            mask_mode=int(mask_mode),
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
        )
        if hasattr(metadata, "to") and metadata.device != query.device:
            metadata = metadata.to(query.device)
        return metadata


def _synchronize_npu(tensor):
    """Finish eager preparation before ACLGraph starts capturing."""
    if tensor.device.type != "npu":
        return
    import torch_npu

    torch_npu.npu.synchronize()


def _none_if_empty(tensor):
    return None if tensor is None or int(tensor.numel()) == 0 else tensor


def _validate_shared_inputs(
    query,
    key,
    sparse_indices,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    *,
    batch_size,
    num_heads_q,
    num_heads_kv,
    head_dim,
    blocksize,
    layout_q,
):
    if cu_seqlens_kv is not None and int(cu_seqlens_kv.numel()) != 0:
        raise ValueError("cu_seqlens_kv must be an empty tensor")
    if seqused_q is not None and int(seqused_q.numel()) != 0:
        raise ValueError("seqused_q must be an empty tensor")
    if int(sparse_indices.shape[0]) != int(batch_size):
        raise ValueError("batch_size does not match sparse_indices")
    query_heads = int(query.shape[0] if layout_q == "NTD" else query.shape[1])
    if query_heads != int(num_heads_q):
        raise ValueError("num_heads_q does not match query")
    if int(key.shape[1]) != int(num_heads_kv):
        raise ValueError("num_heads_kv does not match key")
    if int(query.shape[-1]) != int(head_dim):
        raise ValueError("head_dim does not match query")
    if int(key.shape[2]) != int(blocksize):
        raise ValueError("blocksize does not match key")
    if int(cu_seqlens_q.numel()) != int(batch_size) + 1:
        raise ValueError("cu_seqlens_q must contain batch_size + 1 values")
    if int(seqused_kv.numel()) != int(batch_size):
        raise ValueError("seqused_kv must contain batch_size values")


def _prepare_operator_inputs(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    block_table,
    metadata,
    *,
    quant_mode,
    softmax_scale,
    mask_mode,
    blocksize,
    sparse_block_size_q,
    sparse_block_size_kv,
    layout_q,
    layout_kv,
    layout_sparse_indices,
    batch_size,
    num_heads_q,
    num_heads_kv,
    head_dim,
):
    _validate_shared_inputs(
        query,
        key,
        sparse_indices,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        blocksize=blocksize,
        layout_q=layout_q,
    )
    p_scale = _none_if_empty(p_scale)
    if int(quant_mode) == 1 and p_scale is None:
        raise ValueError("FP8 requires a non-empty p_scale tensor")
    if metadata is None or int(metadata.numel()) < 584:
        metadata = QuantBlockSparseAttnMetadataBuilder.build(
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            atten_mask,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            block_table=block_table,
            softmax_scale=softmax_scale,
            sparse_q_block_size=sparse_block_size_q,
            sparse_kv_block_size=sparse_block_size_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
            mask_mode=mask_mode,
            quant_mode=quant_mode,
        )
    if int(quant_mode) == 1:
        key, value, k_descale = _pack_combined_kv(key, value, k_descale)
    return key, value, k_descale, p_scale, metadata


class QuantBlockSparseAttnGraph(torch.nn.Module):
    """Build metadata before ACLGraph capture and capture only the main operator."""

    def __init__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_seq_len: torch.Tensor,
        p_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        seqused_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        block_table: torch.Tensor,
        atten_mask: torch.Tensor,
        metadata: torch.Tensor,
        *,
        quant_mode: int = 1,
        softmax_scale: float = 1.0,
        mask_mode: int = 3,
        blocksize: int = 0,
        sparse_block_size_q: int = 128,
        sparse_block_size_kv: int = 128,
        layout_q: str = "TND",
        layout_kv: str = "PA_BNSD",
        layout_out: str = "TND",
        layout_sparse_indices: str = "B_N_Qb_Kb",
        return_softmax_lse: bool = False,
        batch_size: int = 0,
        num_heads_q: int = 0,
        num_heads_kv: int = 0,
        head_dim: int = 0,
    ):
        super().__init__()
        _ensure_cann_ops()
        _synchronize_npu(query)
        key, value, k_descale, p_scale, metadata = _prepare_operator_inputs(
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            atten_mask,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
            block_table,
            metadata,
            quant_mode=quant_mode,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            blocksize=blocksize,
            sparse_block_size_q=sparse_block_size_q,
            sparse_block_size_kv=sparse_block_size_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
            batch_size=batch_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
        )
        _synchronize_npu(query)
        for name, tensor in (
            ("query", query),
            ("key", key),
            ("value", value),
            ("q_descale", q_descale),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
            ("p_scale", p_scale),
            ("sparse_indices", sparse_indices),
            ("sparse_seq_len", sparse_seq_len),
            ("cu_seqlens_q", cu_seqlens_q),
            ("seqused_kv", seqused_kv),
            ("block_table", block_table),
            ("atten_mask", atten_mask),
            ("metadata", metadata),
        ):
            self.register_buffer(name, tensor, persistent=False)
        self.softmax_scale = float(softmax_scale)
        self.sparse_block_size_q = int(sparse_block_size_q)
        self.sparse_block_size_kv = int(sparse_block_size_kv)
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.layout_sparse_indices = layout_sparse_indices
        self.quant_mode = int(quant_mode)
        self.mask_mode = int(mask_mode)
        self.return_softmax_lse = bool(return_softmax_lse)

    def forward(self):
        return torch.ops.custom.npu_quant_block_sparse_attn(
            self.query,
            self.key,
            self.value,
            self.q_descale,
            self.k_descale,
            self.v_descale,
            self.p_scale,
            self.sparse_indices,
            self.sparse_seq_len,
            self.atten_mask,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=self.seqused_kv,
            block_table=self.block_table,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            sparse_q_block_size=self.sparse_block_size_q,
            sparse_kv_block_size=self.sparse_block_size_kv,
            layout_kv=self.layout_kv,
            layout_q=self.layout_q,
            layout_sparse_indices=self.layout_sparse_indices,
            layout_out=self.layout_out,
            quant_mode=self.quant_mode,
            mask_mode=self.mask_mode,
            return_softmax_lse=self.return_softmax_lse,
        )


def quant_block_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_seq_len: torch.Tensor,
    p_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    seqused_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    block_table: torch.Tensor,
    atten_mask: torch.Tensor,
    metadata: torch.Tensor,
    *,
    quant_mode: int = 1,
    softmax_scale: float = 1.0,
    mask_mode: int = 3,
    blocksize: int = 0,
    sparse_block_size_q: int = 128,
    sparse_block_size_kv: int = 128,
    layout_q: str = "TND",
    layout_kv: str = "PA_BNSD",
    layout_out: str = "TND",
    layout_sparse_indices: str = "B_N_Qb_Kb",
    return_softmax_lse: bool = False,
    batch_size: int = 0,
    num_heads_q: int = 0,
    num_heads_kv: int = 0,
    head_dim: int = 0,
):
    """Adapt the shared CSV inputs and call the installed QBSA operator."""
    _ensure_cann_ops()
    key, value, k_descale, p_scale, metadata = _prepare_operator_inputs(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_table,
        metadata,
        quant_mode=quant_mode,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        blocksize=blocksize,
        sparse_block_size_q=sparse_block_size_q,
        sparse_block_size_kv=sparse_block_size_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_sparse_indices=layout_sparse_indices,
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    )
    return torch.ops.custom.npu_quant_block_sparse_attn(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        block_table=block_table,
        metadata=metadata,
        softmax_scale=float(softmax_scale),
        sparse_q_block_size=int(sparse_block_size_q),
        sparse_kv_block_size=int(sparse_block_size_kv),
        layout_kv=layout_kv,
        layout_q=layout_q,
        layout_sparse_indices=layout_sparse_indices,
        layout_out=layout_out,
        quant_mode=int(quant_mode),
        mask_mode=int(mask_mode),
        return_softmax_lse=bool(return_softmax_lse),
    )
