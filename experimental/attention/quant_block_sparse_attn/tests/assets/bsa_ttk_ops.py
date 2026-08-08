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


def _auto_generate_params(
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
    """Auto-generate missing seqused/cu_seqlens/block_table from available info."""
    B = int(sparse_indices.shape[0]) if sparse_indices is not None else 1

    if layout_q == "BSND":
        q_fallback = int(query.shape[1])
    elif layout_q == "NTD":
        q_fallback = int(query.shape[1]) // B
    else:
        q_fallback = int(query.shape[0]) // B
    kv_fallback = int(sparse_indices.shape[-1]) * int(sparse_kv_block_size)

    def normalize_lengths(values, fallback, name):
        if values is None:
            return [int(fallback)] * B
        lengths = [int(value) for value in values]
        if len(lengths) != B:
            raise ValueError(f"{name} must contain B={B} values, got {len(lengths)}")
        return lengths

    q_lengths = normalize_lengths(actual_seq_q, q_fallback, "actual_seq_q")
    kv_lengths = normalize_lengths(actual_seq_kv, kv_fallback, "actual_seq_kv")

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
            total_logical_blocks if blocknum is None else int(blocknum)
        )
        if total_physical_blocks <= 0:
            raise ValueError(
                f"blocknum must be > 0 or None, got {blocknum}"
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


class QuantBlockSparseAttnMetadataBuilder:
    """Derive and create the metadata consumed by the main operator."""

    @staticmethod
    def batch_size(q, layout_q, cu_seqlens_q, seqused_q, sparse_indices=None):
        if layout_q == "BSND":
            return int(q.shape[0])
        if seqused_q is not None and seqused_q.numel() > 0:
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
    ):
        _ensure_cann_ops()
        if layout_q == "BSND":
            num_heads_q = int(query.shape[2])
        elif layout_q == "NTD":
            num_heads_q = int(query.shape[0])
        else:
            num_heads_q = int(query.shape[1])
        num_heads_kv = int(key.shape[1])
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


class QuantBlockSparseAttnGraph(torch.nn.Module):
    """TTK graph adapter that prepares host-dependent inputs before capture.

    TTK instantiates a registered ``torch_graph`` module before calling
    ``torch.compile``.  QBSA metadata generation and the PA combined-KV packing
    must happen at that point: both paths can copy data between host and device,
    which ACLGraph GLOBAL capture rejects.  ``forward`` therefore contains only
    the graph-safe main operator call.
    """

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
        atten_mask: torch.Tensor,
        p_scale: Optional[torch.Tensor] = None,
        *,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_kv: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        softmax_scale: float = 1.0,
        sparse_q_block_size: int = 128,
        sparse_kv_block_size: int = 128,
        layout_kv: str = "PA_BNSD",
        layout_q: str = "BSND",
        layout_sparse_indices: str = "B_N_Qb_Kb",
        layout_out: str = "TND",
        quant_mode: int = 1,
        mask_mode: int = 3,
        return_softmax_lse: bool = False,
        actual_seq_q: Optional[List[int]] = None,
        actual_seq_kv: Optional[List[int]] = None,
        empty_actual_seq: bool = False,
        block_table_pattern: str = "sequential",
        sparse_count: int = 0,
        sparse_pattern: str = "sequential",
        blocknum: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__()
        _ensure_cann_ops()

        if seqused_q is None and cu_seqlens_q is None:
            seqused_q, seqused_kv, cu_seqlens_q, cu_seqlens_kv, block_table = (
                _auto_generate_params(
                    query,
                    key,
                    sparse_indices,
                    seqused_q,
                    seqused_kv,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    block_table,
                    layout_q,
                    int(key.shape[2]) if key.dim() == 4 else sparse_kv_block_size,
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
            )

        # Metadata reads the eagerly-created sequence tensors.  Synchronize both
        # sides of that dependency before torch.compile starts ACLGraph capture.
        _synchronize_npu(query)
        if metadata is None or metadata.numel() < 584:
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
                cu_seqlens_kv=cu_seqlens_kv,
                seqused_q=seqused_q,
                seqused_kv=seqused_kv,
                block_table=block_table,
                softmax_scale=softmax_scale,
                sparse_q_block_size=sparse_q_block_size,
                sparse_kv_block_size=sparse_kv_block_size,
                layout_q=layout_q,
                layout_kv=layout_kv,
                layout_sparse_indices=layout_sparse_indices,
                mask_mode=mask_mode,
                quant_mode=quant_mode,
            )

        if key.dim() == 4 and int(quant_mode) != 2:
            key, value, k_descale = _pack_combined_kv(key, value, k_descale)
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
            ("atten_mask", atten_mask),
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_kv", cu_seqlens_kv),
            ("seqused_q", seqused_q),
            ("seqused_kv", seqused_kv),
            ("block_table", block_table),
            ("metadata", metadata),
        ):
            self.register_buffer(name, tensor, persistent=False)

        self.softmax_scale = float(softmax_scale)
        self.sparse_q_block_size = int(sparse_q_block_size)
        self.sparse_kv_block_size = int(sparse_kv_block_size)
        self.layout_kv = layout_kv
        self.layout_q = layout_q
        self.layout_sparse_indices = layout_sparse_indices
        self.layout_out = layout_out
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
            # The current MXFP8 operator contract requires both inputs to be None.
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=self.seqused_kv,
            block_table=self.block_table,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            sparse_q_block_size=self.sparse_q_block_size,
            sparse_kv_block_size=self.sparse_kv_block_size,
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
    atten_mask: torch.Tensor,
    p_scale: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    sparse_q_block_size: int = 128,
    sparse_kv_block_size: int = 128,
    layout_kv: str = "PA_BNSD",
    layout_q: str = "BSND",
    layout_sparse_indices: str = "B_N_Qb_Kb",
    layout_out: str = "TND",
    quant_mode: int = 1,
    mask_mode: int = 3,
    return_softmax_lse: bool = False,
    actual_seq_q: Optional[List[int]] = None,
    actual_seq_kv: Optional[List[int]] = None,
    empty_actual_seq: bool = False,
    block_table_pattern: str = "sequential",
    sparse_count: int = 0,
    sparse_pattern: str = "sequential",
    blocknum: Optional[int] = None,
    seed: int = 0,
):
    """Compute metadata first, then call the real QBSA operator.

    The TTK-facing signature keeps required tensors before the optional
    ``p_scale`` tensor.  The real operator call below restores its native
    argument order, where ``p_scale`` precedes the sparse-index tensors.
    """
    _ensure_cann_ops()

    if seqused_q is None and cu_seqlens_q is None:
        seqused_q, seqused_kv, cu_seqlens_q, cu_seqlens_kv, block_table = (
            _auto_generate_params(
                query,
                key,
                sparse_indices,
                seqused_q,
                seqused_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                block_table,
                layout_q,
                int(key.shape[2]) if key.dim() == 4 else sparse_kv_block_size,
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
        )

    if metadata is None or metadata.numel() < 584:
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
            sparse_q_block_size=sparse_q_block_size,
            sparse_kv_block_size=sparse_kv_block_size,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
            mask_mode=mask_mode,
            quant_mode=quant_mode,
        )

    if key.dim() == 4 and int(quant_mode) != 2:
        key, value, k_descale = _pack_combined_kv(key, value, k_descale)
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
        sparse_q_block_size=int(sparse_q_block_size),
        sparse_kv_block_size=int(sparse_kv_block_size),
        layout_kv=layout_kv,
        layout_q=layout_q,
        layout_sparse_indices=layout_sparse_indices,
        layout_out=layout_out,
        quant_mode=int(quant_mode),
        mask_mode=int(mask_mode),
        return_softmax_lse=bool(return_softmax_lse),
    )
