#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Shared workload + helper module for the `block_sparse_attention` benchmark
and correctness gate. Contains everything needed to:

- Generate per-row block-sparse selection patterns (frame + random) at any
  Q/KV block granularity.
- Build dense `[1, H, S, S]` bool masks from the block patterns (for
  reference kernels that take fine-grained masks).
- Pack the block selection into `sabi` (uint16, padded `0xFFFF`) — the
  format BSA's torch interface accepts — or into `block_sparse_mask`
  (int8 `[B, H, q_blocks, kv_blocks]`) — the format the
  `block_sparse_attention` kernel accepts.
- Time + verify a kernel against `npu_fusion_attention` /
  `npu_fused_infer_attention_score`.

Originally lifted verbatim from BSA's
`experimental/attention/blitz_sparse_attention/benchmark/benchmark.py`
(commit 9eb67134, branch `128x128-disruptive`). Decoupled here so the
`attention/block_sparse_attention/` directory can ship as a self-contained
feature — no path manipulation, no stubs, no cross-tree imports.
"""

import math
import random
import itertools
import logging
from collections import namedtuple
from typing import Callable, List, Tuple

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

FrameWidths = namedtuple('FrameWidths', ['left_cols', 'right_cols', 'top_rows', 'bottom_rows'])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global Definitions
DEVICE = 'npu'
DTYPE = torch.bfloat16
INPUT_LAYOUT = "BNSD"  # [batch_size, num_heads, seq_len, head_dim]

# Parameter sweeps (adjust as needed)
B_VALS = [1]
H_VALS = [3]
S_VALS = [118_806]  # s_q = s_kv
D_VALS = [128]   # head dimension

N_REPEATS = 10
N_WARMUP = 2

# Fraction of attention blocks that is skipped (0 = dense, 0.9 = 90% of blocks skipped)
SPARSITY_VALS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Block-sparse mask granularity (q-axis × kv-axis, in tokens).
BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 128
BLOCK_MASK_SEED = 1234

# A "frame" forces a fixed set of blocks to always be kept regardless of the
# random sparse selection — the left/right border columns are kept in every
# row, and the top/bottom border rows are kept dense. The forced blocks still
# count toward the sparsity budget; widths are in BLOCK units (cols in
# kv-block lengths, rows in q-block lengths). Set any width to 0 to disable
# that side of the frame.
#
# Why a frame matters in practice (sink + text in video generation):
#   Multimodal video-generation workloads commonly use a "sink + text" prompt
#   layout — a handful of leading tokens (BOS / system / attention-sink
#   anchors) and a chunk of text-prompt tokens that EVERY video query token
#   must attend to. Those tokens live in the leftmost KV columns, so the
#   left frame columns model them: their attention scores are non-negligible
#   for output quality and dropping them in the sparse selection causes
#   visible quality regressions. The right column mirrors this for any
#   trailing anchor tokens; the top/bottom dense rows reserve a small set of
#   query positions (e.g. boundary frames or registers) that need full
#   context. The default below is calibrated for that workload class.
FRAME = FrameWidths(left_cols=29, right_cols=1, top_rows=29, bottom_rows=1)

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False
# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = False  # True <--> enables accuracy compariston
# Which reference kernel to compare/benchmark against. One of:
#   "custom"                          - our pythonic ref_attention_fp32
#   "npu_fusion_attention"            - torch_npu.npu_fusion_attention (PFA)
#   "npu_fused_infer_attention_score" - torch_npu.npu_fused_infer_attention_score (FIAS)
TORCH_REFERENCE = "npu_fusion_attention"

torch.set_printoptions(
    threshold=100_000_000,
    linewidth=400,
    edgeitems=3,
    precision=4,
    sci_mode=False
)


def ref_attention_launcher(
    torch_reference: str,
    pfa_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, torch.Tensor, str],
    force_dense_sm: bool,
    return_lse: bool = False,
):
    """
    runs a reference attention kernel, for correctness comparisons and for baseline time measurements.

    torch_reference: which reference to launch
      "custom"                          - launch our pythonic model `ref_attention_fp32`
      "npu_fusion_attention"            - launch `torch_npu.npu_fusion_attention` (PFA)
      "npu_fused_infer_attention_score" - launch `torch_npu.npu_fused_infer_attention_score` (FIAS)
    force_dense_sm - only consulted for the npu_* references:
      True  - apply sparse_mode=0 (dense) and don't use the provided atten_mask
      False - apply sparse_mode=1 (sparse token mask) and use the provided atten_mask
    return_lse - if True, return (attention_out, softmax_lse[B,H,S]).  Only the FIAS
                 reference natively produces softmax_lse; "custom" and
                 "npu_fusion_attention" raise ValueError when asked for it.
                 If False (default), return only attention_out.
    """
    q, k, v, head_num, scale, atten_mask, input_layout = pfa_inputs
    if torch_reference == "custom":
        if return_lse:
            raise ValueError("return_lse=True is not supported by the 'custom' reference "
                             "(ref_attention_fp32 does not compute softmax_lse).")
        return ref_attention_fp32(q, k, v, scale, atten_mask=atten_mask)
    if torch_reference == "npu_fusion_attention":
        if return_lse:
            raise ValueError("return_lse=True is not supported by 'npu_fusion_attention'; "
                             "use 'npu_fused_infer_attention_score' instead.")
        return torch_npu.npu_fusion_attention(q, k, v, head_num=head_num, input_layout=input_layout,
                                              scale=scale, pre_tockens=0, next_tockens=0,
                                              atten_mask=atten_mask,
                                              sparse_mode=0 if force_dense_sm else 1)[0]
    if torch_reference == "npu_fused_infer_attention_score":
        # FIAS is only supported by this launcher in dense mode (sparse_mode=0).
        # Reason: FIAS does not accept our atten_mask layout — its sparse_mode=1
        # path requires a [B, 1, S, S] mask (H broadcast), whereas this launcher's
        # `pfa_inputs` carries an atten_mask shaped per the caller's needs (often
        # [B, H, S, S]) and would silently mis-apply or hang the kernel. Callers
        # that need a sparse FIAS reference must invoke
        # torch_npu.npu_fused_infer_attention_score(...) directly so they own the
        # mask-shape constraint at the call site.
        if not force_dense_sm:
            raise ValueError(
                "torch_reference='npu_fused_infer_attention_score' requires "
                "force_dense_sm=True. FIAS does not support our atten_mask layout "
                "(sparse_mode=1 needs [B, 1, S, S]); call "
                "torch_npu.npu_fused_infer_attention_score(...) directly for sparse runs."
            )
        # FIAS expects per-batch actual sequence lengths; derive them from q/k shape according to input_layout.
        if input_layout == "BNSD":
            B, S_q, S_kv = q.shape[0], q.shape[2], k.shape[2]
        elif input_layout in ("BSND", "BSH"):
            B, S_q, S_kv = q.shape[0], q.shape[1], k.shape[1]
        else:
            raise ValueError(f"Unsupported input_layout for FIAS reference: {input_layout!r}")
        actual_seq_q = [S_q] * B
        actual_seq_kv = [S_kv] * B
        out, lse = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=head_num, scale=scale,
            input_layout=input_layout, num_key_value_heads=head_num,
            actual_seq_lengths=actual_seq_q, actual_seq_lengths_kv=actual_seq_kv,
            softmax_lse_flag=return_lse,
        )
        if return_lse:
            # FIAS emits lse as [B, H, S, 1]; squeeze to [B, H, S].
            return out, lse.squeeze(-1)
        return out
    raise ValueError(
        f"Unknown torch_reference value: {torch_reference!r}. "
        f"Expected one of: 'custom', 'npu_fusion_attention', 'npu_fused_infer_attention_score'."
    )


def generate_sparse_blocks_by_row(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    seed: int,
    frame: FrameWidths,
    pad_value: int = -1,
    sort_within_row: bool = True,
) -> list[list[int]]:
    """
    Builds the sabi blocks (sparse blocks selection) of a single head, with a "frame" of
    always-included blocks and the rest of the blocks randomly selected to meet the target
    sparsity.

    The frame models the sink + text-prompt tokens common to video-generation
    workloads (see FRAME comment above): a fixed set of leading KV-block
    columns + a handful of trailing columns are kept in every query row, and
    the top/bottom rows are kept fully dense. The pattern that emerges is a
    sparse interior with a "picture frame" of always-on attention — without
    it, sparse runs lose enough sink/text information to visibly degrade
    output quality.

    Note: blocks from the frame + sparsely selected blocks are both taken into account when computing the sparsity.

    Constraints (post-sparsity selection):
      - first `frame.left_cols` block-cols of each row are always included (if they exist)
      - first `frame.top_rows` block-rows are forced dense
      - last `frame.bottom_rows` block-rows are forced dense
      - each row always includes the last `frame.right_cols` block-cols
    Output:
      - shape [n_block_rows][n_block_cols]
      - each row: selected cols, then pad_value to the end. Cols are sorted ascending
        when sort_within_row=True (default); otherwise shuffled deterministically per seed
    """
    sparsity = max(0.0, min(1.0, float(sparsity)))

    n_block_rows = math.ceil(s_q / block_size_q)
    n_block_cols = math.ceil(s_kv / block_size_kv)
    n_total_blocks = n_block_rows * n_block_cols
    if n_block_rows == 0 or n_block_cols == 0:
        raise ValueError(f"Invalid block configuration: {s_q=}, {s_kv=}, {block_size_q=}, {block_size_kv=}")

    if frame is not None:
        if (frame.left_cols > n_block_cols or frame.right_cols > n_block_cols or
            frame.top_rows > n_block_rows or frame.bottom_rows > n_block_rows):
            raise ValueError(f"Frame configuration exceeds block dimensions: {frame=}, {n_block_rows=}, {n_block_cols=}")

        # Forced column indices by the frame (left and right borders) for EVERY sparse row:
        sparse_forced_left = set(range(min(frame.left_cols, n_block_cols)))
        sparse_forced_right = set(range(max(n_block_cols - frame.right_cols, 0), n_block_cols))
        sparse_forced_col_ids = sparse_forced_left | sparse_forced_right
        sparse_candidate_col_ids = sorted(set(range(n_block_cols)) - sparse_forced_col_ids)

        # Rows indices which are forced to be dense by the frame (top and bottom borders):
        dense_top = set(range(min(frame.top_rows, n_block_rows)))
        dense_bottom = set(range(max(n_block_rows - frame.bottom_rows, 0), n_block_rows))
        dense_row_ids = dense_top | dense_bottom
    else:
        sparse_forced_col_ids = set()
        sparse_candidate_col_ids = list(range(n_block_cols))
        dense_row_ids = set()

    n_dense_rows = len(dense_row_ids)
    n_sparse_rows = max(n_block_rows - n_dense_rows, 0)
    n_frame_forced_blocks = len(sparse_forced_col_ids) * n_sparse_rows + n_block_cols * n_dense_rows
    if n_frame_forced_blocks / n_total_blocks > (1.0 - sparsity):
        raise ValueError(f"Target sparsity violation: "
                        f"Frame forces {n_frame_forced_blocks} blocks out of {n_total_blocks} total attention blocks. "
                        f"Implying a mimimum density of {n_frame_forced_blocks / n_total_blocks:.2%}, which violates the target "
                        f"sparsity of {(sparsity):.2%}. And this is even before allocating the random blocks inside "
                        "the frame. Consider adjusting frame sizes or target sparsity.")

    # Make sure that allocating "n_kept_row_blocks" in every sparsable row (inside the borders of the frame) will not
    # violate the required sparsity
    n_max_kept_blocks = int(math.floor(n_total_blocks * (1.0 - sparsity)))
    n_max_kept_blocks_excluding_frame =  n_max_kept_blocks - n_frame_forced_blocks
    n_kept_row_blocks_per_row =  math.ceil(n_max_kept_blocks_excluding_frame / n_sparse_rows) if n_sparse_rows > 0 else 0

    # Construct the "sabi" blocks of the current head by choosing the sparse blocks randomly
    rows: list[list[int]] = []
    rng = random.Random(seed)
    for r in range(n_block_rows):
        if r in dense_row_ids:
            cols = list(range(n_block_cols))
        else:
            if n_kept_row_blocks_per_row <= 0:
                cols = sorted(sparse_forced_col_ids)
            else:
                sparse_selected_col_ids = rng.sample(sparse_candidate_col_ids, n_kept_row_blocks_per_row)
                cols = sorted(list(sparse_forced_col_ids) + sparse_selected_col_ids)
            cols = cols + [pad_value] * (n_block_cols - len(cols))
        if not sort_within_row:
            kept = [c for c in cols if c != pad_value]
            rng.shuffle(kept)
            cols = kept + [pad_value] * (len(cols) - len(kept))
        rows.append(cols)

    return rows


def make_block_mask(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparse_blocks_by_row: list[list[int]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds a dense boolean mask [1, 1, s_q, s_kv] from sparse block columns.

    True  = masked
    False = allowed
    """
    n_block_rows = math.ceil(s_q / block_size_q)
    n_block_cols = math.ceil(s_kv / block_size_kv)

    mask = torch.ones((s_q, s_kv), dtype=torch.bool, device=device)

    rows_to_process = min(n_block_rows, len(sparse_blocks_by_row))
    for r in range(rows_to_process):
        row_start = r * block_size_q
        row_end = min(row_start + block_size_q, s_q)

        for c in sparse_blocks_by_row[r]:
            if 0 <= c < n_block_cols:
                col_start = c * block_size_kv
                col_end = min(col_start + block_size_kv, s_kv)
                mask[row_start:row_end, col_start:col_end] = False

    return mask.unsqueeze(0).unsqueeze(0)


def generate_sparse_blocks_by_row_per_head(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    num_heads: int,
    frame: FrameWidths,
    base_seed: int,
    sort_within_row: bool = True,
) -> list[list[list[int]]]:
    """
    Returns sparse blocks per head.

    Output shape:
      [num_heads][n_block_rows][num_kept_elems]
    """
    return [
        generate_sparse_blocks_by_row(
            s_q=s_q,
            s_kv=s_kv,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            sparsity=sparsity,
            seed=base_seed + h,
            frame=frame,
            sort_within_row=sort_within_row,
        )
        for h in range(num_heads)
    ]


def make_block_mask_per_head(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparse_blocks_by_row_per_head: list[list[list[int]]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds a dense boolean mask [1, num_heads, s_q, s_kv] from sparse block columns.

    True  = masked
    False = allowed
    """
    head_masks = [
        make_block_mask(
            s_q=s_q,
            s_kv=s_kv,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            sparse_blocks_by_row=sparse_blocks_by_row,
            device=device,
        ).squeeze(0).squeeze(0)  # [s_q, s_kv]
        for sparse_blocks_by_row in sparse_blocks_by_row_per_head
    ]

    return torch.stack(head_masks, dim=0).unsqueeze(0)


def block_allclose_map(
    out: torch.Tensor,
    ref: torch.Tensor,
    block_h: int,
    block_w: int,
    rtol: float = 0.01,
    atol: float = 0.01,
    print_map: bool = False,
) -> torch.Tensor:
    """
    Compare two 4D tensors [dim0, dim1, dim2, dim3] in (block_h x block_w) blocks over the last two dims.
    Returns a boolean tensor of shape [dim0, dim1, nby, nbx] where each entry indicates whether the
    entire block is allclose. Optionally prints a 0/1 block matrix per (batch, head).

    - First two dims: batch and head (iterated and printed as headers)
    - Last two dims: "drawn" dimensions (split into blocks)

    Blocks at the edges can be smaller if dim2 or dim3 is not divisible by block sizes.
    """
    if out.shape != ref.shape:
        raise ValueError(f"Shape mismatch: out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    if out.ndim != 4:
        raise ValueError(f"Expected 4D tensors [dim0, dim1, dim2, dim3], got out.ndim={out.ndim}")

    if block_h <= 0 or block_w <= 0:
        raise ValueError("block_h and block_w must be positive integers")

    dim0, dim1, dim2, dim3 = out.shape
    nby = (dim2 + block_h - 1) // block_h
    nbx = (dim3 + block_w - 1) // block_w

    # Elementwise closeness map: [dim0, dim1, dim2, dim3] boolean
    close = torch.isclose(out, ref, rtol=rtol, atol=atol)

    # Blockwise result: [dim0, dim1, nby, nbx]
    block_ok = close.new_empty((dim0, dim1, nby, nbx), dtype=torch.bool)

    for b, h, by, bx in itertools.product(range(dim0), range(dim1), range(nby), range(nbx)):
        y0, x0 = by * block_h, bx * block_w
        y1, x1 = min(y0 + block_h, dim2), min(x0 + block_w, dim3)
        block_ok[b, h, by, bx] = close[b, h, y0:y1, x0:x1].all()

    if print_map:
        for b, h in itertools.product(range(dim0), range(dim1)):
            logger.info(f"batch={b}, head={h}")
            # Print as 0/1 grid
            grid = block_ok[b, h].to(dtype=torch.int32)
            for by in range(nby):
                row = " ".join(str(int(v)) for v in grid[by].tolist())
                logger.info(row)
            logger.info()  # blank line between heads

    return block_ok


def ref_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_value: float,
    atten_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reference implementation for correctness comparisons.
    Scaled dot-product attention in float32 throughout.

    Assumes:
    - input_layout == "BNSD"
    - atten_mask (if provided) is bool broadcastable
      to [batch_size, num_heads, s_q, s_kv], where True means "masked out".
    """

    # Convert to fp32 for computation
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    #    [batch_size, num_heads, s_q, head_dim] x [batch_size, num_heads, head_dim, s_kv]
    # -> [batch_size, num_heads, s_q, s_kv]
    attn_scores = torch.matmul(q_f, k_f.transpose(-1, -2))
    attn_scores = attn_scores * scale_value

    if atten_mask is not None:
        if atten_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(
                atten_mask, torch.finfo(attn_scores.dtype).min
            )
        elif atten_mask.dtype in (torch.int8, torch.uint8):
            attn_scores = attn_scores.masked_fill(
                atten_mask == 1, torch.finfo(attn_scores.dtype).min
            )

    # Softmax in fp32
    attn_probs = torch.softmax(attn_scores, dim=-1)

    #    [batch_size, num_heads, s_q, s_kv] x [batch_size, num_heads, s_kv, head_dim]
    # -> [batch_size, num_heads, s_q, head_dim]
    out = torch.matmul(attn_probs, v_f)

    # Cast back to original dtype for comparison
    return out.to(dtype=q.dtype)


def gen_pfa_inputs(
    batch_size: int, num_heads: int, s_q: int, s_kv: int, head_dim: int,
    device: str = "npu:0", dtype: torch.dtype = DTYPE
):
    """
    Generate common attention inputs (BNSD layout).
    """
    q = torch.randn(batch_size, num_heads, s_q, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, s_kv, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, s_kv, head_dim, dtype=dtype, device=device)

    # For now we assume all sequences are full-length
    actseqlen = [s_q] * batch_size
    actseqlenkv = [s_kv] * batch_size

    return q, k, v, actseqlen, actseqlenkv


def create_attention_mask(b, h, s_q, s_kv, d, sparsity,
                          block_size_q: int, block_size_kv: int, frame: FrameWidths,
                          device: str, emit_atten_mask: bool = True):
    """
    Build the block-sparse selection (`sabi`) and, optionally, the equivalent
    dense token-level mask used by the dense reference kernel.

    Returns:
        pfa_atten_mask: bool tensor `[B, H, s_q, s_kv]` (True = masked), or
                        None when `emit_atten_mask` is False or `sparsity == 0`.
        sabi: uint16 tensor `[B, H, n_block_rows, n_block_cols]` of kept-block
              column indices, padded with `0xFFFF`.
        scale: `1.0 / sqrt(d)`.
    """
    scale = 1.0 / math.sqrt(float(d))
    per_batch_head_block_indices = [
        generate_sparse_blocks_by_row_per_head(s_q, s_kv, block_size_q, block_size_kv, sparsity, h,
                                               frame, BLOCK_MASK_SEED + bidx,
                                               sort_within_row=True)
        for bidx in range(b)
    ]
    sabi = torch.tensor(per_batch_head_block_indices, dtype=torch.uint16, device=device)
    pfa_atten_mask = None
    if emit_atten_mask and sparsity > 0:
        pfa_atten_mask = torch.cat([
            make_block_mask_per_head(s_q, s_kv, block_size_q, block_size_kv, bi, device=device)
            for bi in per_batch_head_block_indices
        ], dim=0)
    return pfa_atten_mask, sabi, scale


def _run_timed(kernel_fn: Callable, input_sets: List, n_warmup: int, n_repeat: int):
    """
    Utility function to run a kernel multiple times on the provided inputs
    and report the average latency in microseconds. Warmup is appplied first (not timed).
    """
    for args in input_sets[:n_warmup]:
        kernel_fn(*args)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for args in input_sets[n_warmup:]:
        kernel_fn(*args)
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / n_repeat * 1000.0  # ms -> μs


def _check_correctness(our_fn: Callable, ref_fn: Callable, b: int, h: int, s_q: int, s_kv: int, d: int, device: str
    ) -> str:
    """Run both implementations on shared inputs and return 'yes'/'no'."""
    q, k, v, seq, seqkv = gen_pfa_inputs(b, h, s_q, s_kv, d, device=device, dtype=DTYPE)
    out_our = our_fn(q, k, v, seq, seqkv).cpu()
    out_ref = ref_fn(q, k, v, seq, seqkv).cpu()

    if PRINT_OUTPUTS:
        logger.info("OURS: ", out_our.shape)
        logger.info(out_our)
        logger.info("REF:  ", out_ref.shape)
        logger.info(out_ref)

    equal = torch.allclose(out_our, out_ref, rtol=0.02, atol=0.02)
    if not equal and PRINT_BLOCK_EQUALITY:
        block_allclose_map(out_our, out_ref,
                           block_h=PRINT_HEIGHT, block_w=PRINT_WIDTH,
                           rtol=0.02, atol=0.02, print_map=True)
    return "yes" if equal else "no"


def _fmt_or_na(value, width, spec=".2f"):
    """Format a number with given width/spec, or right-align 'N/A' if None."""
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:{width}{spec}}"


# ──────────────────────────────────────────────────────────────────────────────
# block_sparse_attention-specific helpers
# ──────────────────────────────────────────────────────────────────────────────


def sabi_to_block_sparse_mask(sabi: torch.Tensor, b: int, h: int,
                              q_blocks: int, kv_blocks: int,
                              device: str = "npu") -> torch.Tensor:
    """Convert sabi (uint16 keeper-indices, padded 0xFFFF) into the
    block_sparse_attention kernel's int8 `[B, H, q_blocks, kv_blocks]` mask.

    1 = block kept (will be matmul'd); 0 = block skipped (gather drops it).
    """
    # NPU's aclnnNeScalar / scatter don't support uint16 directly. Cast to
    # int32 first; the operation set on int32 is fully supported.
    sabi_i32 = sabi.to(torch.int32)
    valid = (sabi_i32 != 0xFFFF)
    sabi_long = sabi_i32.long()
    # For padding entries set the index to 0; src=0 there so scatter_add is a no-op.
    sabi_safe = torch.where(valid, sabi_long, torch.zeros_like(sabi_long))

    mask_int = torch.zeros(b, h, q_blocks, kv_blocks,
                           dtype=torch.int32, device=sabi.device)
    mask_int.scatter_add_(3, sabi_safe, valid.to(torch.int32))
    mask = (mask_int > 0).to(torch.int8)
    if str(mask.device) != device:
        mask = mask.to(device)
    return mask.contiguous()
