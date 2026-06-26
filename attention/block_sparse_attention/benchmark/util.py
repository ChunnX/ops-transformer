#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Shared workload + helper module for the `block_sparse_attention` benchmark
and correctness gate. Contains everything needed to:

- Generate per-row block-sparse selection patterns (frame + random) at any
  Q/KV block granularity.
- Build dense `[1, H, S, S]` bool masks from the block patterns (for
  reference kernels that take fine-grained masks).
- Pack the block selection into `sabi` — the sparse attention block indices
  tensor (uint16, padded `0xFFFF`) — or into `block_sparse_mask`
  (int8 `[B, H, q_blocks, kv_blocks]`), the format the
  `block_sparse_attention` kernel accepts.
- Time + verify a kernel against `npu_fusion_attention` /
  `npu_fused_infer_attention_score`.

"""

import math
import random
import itertools
import logging
from collections import namedtuple
from typing import Callable, List, Tuple

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)


def is_arch35_soc(device_name=None):
    """True on Ascend 950 (arch35); False on Atlas A2 (910B*) / A3 (910_93*).

    These two families need different kernel parameters (the 950 supports FP8 +
    BSND and *requires* innerPrecise=4; A2/A3 are FP16/BF16 + TND/BNSD only and
    require innerPrecise ∈ {0, 1}). Detection reuses the torch front-end's guard
    (`_is_arch220_soc`) so the test harness and the op agree on one definition.
    """
    if device_name is None:
        try:
            device_name = torch_npu.npu.get_device_name(torch_npu.npu.current_device())
        except Exception:
            return False
    # Imported lazily: importing the front-end module triggers a JIT kernel build,
    # which util.py callers (mask helpers) should not pay for unless they ask for SoC info.
    from cann_ops_transformer.ops.block_sparse_attention import (
        _is_arch220_soc,
    )
    return not _is_arch220_soc(device_name)


def bsa_inner_precise(device_name=None):
    """The only `inner_precise` the kernel accepts on the current chip
    (see block_sparse_attention_tiling.cpp): 950 → 4 (low/high mixed softmax),
    910 / 910_93 → 0 (fp32 softmax). Lets one test run unchanged on both.
    """
    return 4 if is_arch35_soc(device_name) else 0


FrameWidths = namedtuple('FrameWidths', ['left_cols', 'right_cols', 'top_rows', 'bottom_rows'])
BlockGrid = namedtuple('BlockGrid', ['s_q', 's_kv', 'block_size_q', 'block_size_kv'])
# Workload identity for one benchmark row / one mask build. Carries the full
# spatial workload so individual helpers can take a single arg instead of
# spreading 5–7 scalars across their signatures.
WorkloadShape = namedtuple(
    'WorkloadShape', ['b', 'h', 's_q', 's_kv', 'd', 'block_size_q', 'block_size_kv'])

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

# Padding sentinel for the per-row kept-column lists once packed into the uint16
# `sabi` tensor. Stored explicitly as 65535 (not -1) so packing never relies on
# negative→unsigned wraparound, whose behaviour is not guaranteed across torch
# versions. Consumers detect padding by comparing against this value.
SABI_PAD = 0xFFFF  # 65535 — must stay > any valid kv-block index (uint16 range)

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
# For printing tensor differences in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = False  # True <--> enables accuracy comparison
# Which reference kernel to compare/benchmark against. One of:
#   "custom"                          - our pythonic ref_attention_fp32
#   "npu_fusion_attention"            - torch_npu's npu_fusion_attention (PFA)
#   "npu_fused_infer_attention_score" - torch_npu's npu_fused_infer_attention_score (FIAS)
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
            b, s_q, s_kv = q.shape[0], q.shape[2], k.shape[2]
        elif input_layout in ("BSND", "BSH"):
            b, s_q, s_kv = q.shape[0], q.shape[1], k.shape[1]
        else:
            raise ValueError(f"Unsupported input_layout for FIAS reference: {input_layout!r}")
        actual_seq_q = [s_q] * b
        actual_seq_kv = [s_kv] * b
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
        grid: BlockGrid, sparsity: float, seed: int,
        frame: FrameWidths, pad_value: int = SABI_PAD) -> list[list[int]]:
    """Per-row block selection for one head: frame-forced cols/rows + random picks
    chosen to hit the target sparsity. See FRAME comment above for the sink+text
    motivation. Frame-forced and random blocks both count toward the sparsity budget.

    Returns `[n_rows][n_cols]`; each row is the kept cols (sorted ascending)
    padded with `pad_value` (default `SABI_PAD` = 0xFFFF, the uint16 padding
    sentinel; `make_block_mask` rejects it via its `0 <= c < n_block_cols` guard).
    """
    sparsity = max(0.0, min(1.0, float(sparsity)))
    n_rows = math.ceil(grid.s_q / grid.block_size_q)
    n_cols = math.ceil(grid.s_kv / grid.block_size_kv)
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"Invalid block config: {grid=}")
    n_total = n_rows * n_cols
    forced_cols, candidate_cols, dense_rows = set(), list(range(n_cols)), set()
    if frame is not None:
        if max(frame.left_cols, frame.right_cols) > n_cols or max(frame.top_rows, frame.bottom_rows) > n_rows:
            raise ValueError(f"Frame exceeds block dims: {frame=}, {n_rows=}, {n_cols=}")
        forced_cols = set(range(min(frame.left_cols, n_cols))) | set(range(max(n_cols - frame.right_cols, 0), n_cols))
        candidate_cols = sorted(set(range(n_cols)) - forced_cols)
        dense_rows = set(range(min(frame.top_rows, n_rows))) | set(range(max(n_rows - frame.bottom_rows, 0), n_rows))
    n_sparse_rows = max(n_rows - len(dense_rows), 0)
    n_forced = len(forced_cols) * n_sparse_rows + n_cols * len(dense_rows)
    if n_forced / n_total > (1.0 - sparsity):
        raise ValueError(f"Frame forces {n_forced}/{n_total} blocks ({n_forced / n_total:.2%}) "
                         f"— exceeds target sparsity {sparsity:.2%}. Adjust frame or sparsity.")
    n_max_kept = int(math.floor(n_total * (1.0 - sparsity)))
    n_kept_per_row = math.ceil((n_max_kept - n_forced) / n_sparse_rows) if n_sparse_rows > 0 else 0
    rng = random.Random(seed)
    rows: list[list[int]] = []
    for r in range(n_rows):
        if r in dense_rows:
            cols = list(range(n_cols))
        elif n_kept_per_row <= 0:
            cols = sorted(forced_cols) + [pad_value] * (n_cols - len(forced_cols))
        else:
            picked = rng.sample(candidate_cols, n_kept_per_row)
            cols = sorted(list(forced_cols) + picked) + [pad_value] * (n_cols - len(forced_cols) - n_kept_per_row)
        rows.append(cols)
    return rows


def make_block_mask(grid: BlockGrid, sparse_blocks_by_row: list[list[int]],
                    device: str = "cpu") -> torch.Tensor:
    """Build a dense boolean mask `[1, 1, s_q, s_kv]` from sparse block columns.

    True = masked, False = allowed.
    """
    n_block_rows = math.ceil(grid.s_q / grid.block_size_q)
    n_block_cols = math.ceil(grid.s_kv / grid.block_size_kv)
    mask = torch.ones((grid.s_q, grid.s_kv), dtype=torch.bool, device=device)
    rows_to_process = min(n_block_rows, len(sparse_blocks_by_row))
    for r in range(rows_to_process):
        row_start = r * grid.block_size_q
        row_end = min(row_start + grid.block_size_q, grid.s_q)
        for c in sparse_blocks_by_row[r]:
            if 0 <= c < n_block_cols:
                col_start = c * grid.block_size_kv
                col_end = min(col_start + grid.block_size_kv, grid.s_kv)
                mask[row_start:row_end, col_start:col_end] = False
    return mask.unsqueeze(0).unsqueeze(0)


def generate_sparse_blocks_by_row_per_head(
        grid: BlockGrid, sparsity: float, num_heads: int,
        frame: FrameWidths, base_seed: int) -> list[list[list[int]]]:
    """Returns sparse blocks per head, shape `[num_heads][n_block_rows][n_block_cols]`."""
    return [
        generate_sparse_blocks_by_row(grid, sparsity, base_seed + h, frame)
        for h in range(num_heads)
    ]


def make_block_mask_per_head(grid: BlockGrid, sparse_blocks_by_row_per_head: list[list[list[int]]],
                              device: str = "cpu") -> torch.Tensor:
    """Build a dense boolean mask `[1, num_heads, s_q, s_kv]` from sparse block columns.

    True = masked, False = allowed.
    """
    head_masks = [
        make_block_mask(grid, sparse_blocks_by_row, device).squeeze(0).squeeze(0)
        for sparse_blocks_by_row in sparse_blocks_by_row_per_head
    ]
    return torch.stack(head_masks, dim=0).unsqueeze(0)


def block_allclose_map(out: torch.Tensor, ref: torch.Tensor,
                        block_shape: tuple, tol: tuple = (0.01, 0.01),
                        print_map: bool = False) -> torch.Tensor:
    """Compare two 4D tensors `[dim0, dim1, dim2, dim3]` in (block_h x block_w) tiles over the
    last two dims. Returns a boolean tensor `[dim0, dim1, nby, nbx]` indicating per-tile
    closeness. Optionally prints a 0/1 grid per (batch, head).

    `block_shape`: `(block_h, block_w)`; `tol`: `(rtol, atol)`.
    First two dims are batch/head (iteration axes); the last two are tiled.
    """
    block_h, block_w = block_shape
    rtol, atol = tol
    if out.shape != ref.shape:
        raise ValueError(f"Shape mismatch: out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    if out.ndim != 4:
        raise ValueError(f"Expected 4D tensors [dim0, dim1, dim2, dim3], got out.ndim={out.ndim}")
    if block_h <= 0 or block_w <= 0:
        raise ValueError("block_h and block_w must be positive integers")

    dim0, dim1, dim2, dim3 = out.shape
    nby = (dim2 + block_h - 1) // block_h
    nbx = (dim3 + block_w - 1) // block_w
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
            logger.info("")  # blank line between heads

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


def gen_pfa_inputs(shape: WorkloadShape, device: str = "npu:0",
                   dtype: torch.dtype = DTYPE):
    """Generate common attention inputs (BNSD layout) from a `WorkloadShape`."""
    q = torch.randn(shape.b, shape.h, shape.s_q, shape.d, dtype=dtype, device=device)
    k = torch.randn(shape.b, shape.h, shape.s_kv, shape.d, dtype=dtype, device=device)
    v = torch.randn(shape.b, shape.h, shape.s_kv, shape.d, dtype=dtype, device=device)
    actseqlen = [shape.s_q] * shape.b
    actseqlenkv = [shape.s_kv] * shape.b
    return q, k, v, actseqlen, actseqlenkv


def create_attention_mask(shape: WorkloadShape, sparsity: float, frame: FrameWidths,
                          device: str, emit_atten_mask: bool = True):
    """Build the block-sparse selection (`sabi`) and, optionally, the equivalent
    dense token-level mask used by the dense reference kernel.

    Returns:
        pfa_atten_mask: bool tensor `[B, H, s_q, s_kv]` (True = masked), or
                        None when `emit_atten_mask` is False or `sparsity == 0`.
        sabi: uint16 tensor `[B, H, n_block_rows, n_block_cols]` of kept-block
              column indices, padded with `0xFFFF`.
        scale: `1.0 / sqrt(d)`.
    """
    scale = 1.0 / math.sqrt(float(shape.d))
    grid = BlockGrid(shape.s_q, shape.s_kv, shape.block_size_q, shape.block_size_kv)
    per_batch_head_block_indices = [
        generate_sparse_blocks_by_row_per_head(grid, sparsity, shape.h, frame, BLOCK_MASK_SEED + bidx)
        for bidx in range(shape.b)
    ]
    sabi = torch.tensor(per_batch_head_block_indices, dtype=torch.uint16, device=device)
    pfa_atten_mask = None
    if emit_atten_mask and sparsity > 0:
        pfa_atten_mask = torch.cat([
            make_block_mask_per_head(grid, bi, device=device)
            for bi in per_batch_head_block_indices
        ], dim=0)
    return pfa_atten_mask, sabi, scale


def _run_timed(kernel_fn: Callable, input_sets: List, n_warmup: int):
    """
    Utility function to run a kernel multiple times on the provided inputs
    and report the average latency in microseconds. Warmup is applied first (not timed).

    The average is taken over the iterations actually timed
    (`len(input_sets) - n_warmup`), so the result is correct regardless of how the
    caller sized `input_sets` — it does not depend on a separately-passed repeat count.
    """
    n_timed = len(input_sets) - n_warmup
    if n_timed <= 0:
        raise ValueError(
            f"input_sets has {len(input_sets)} entries but n_warmup={n_warmup}; "
            "need at least one timed iteration after warmup"
        )
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

    return start.elapsed_time(end) / n_timed * 1000.0  # ms -> μs


def _check_correctness(our_fn: Callable, ref_fn: Callable, shape: WorkloadShape,
                       device: str) -> str:
    """Run both implementations on shared inputs and return 'yes'/'no'."""
    q, k, v, seq, seqkv = gen_pfa_inputs(shape, device=device, dtype=DTYPE)
    out_our = our_fn(q, k, v, seq, seqkv).cpu()
    out_ref = ref_fn(q, k, v, seq, seqkv).cpu()

    if PRINT_OUTPUTS:
        logger.info(f"OURS: {out_our.shape}")
        logger.info(out_our)
        logger.info(f"REF:  {out_ref.shape}")
        logger.info(out_ref)

    equal = torch.allclose(out_our, out_ref, rtol=0.02, atol=0.02)
    if not equal and PRINT_BLOCK_EQUALITY:
        block_allclose_map(out_our, out_ref,
                           block_shape=(PRINT_HEIGHT, PRINT_WIDTH),
                           tol=(0.02, 0.02), print_map=True)
    return "yes" if equal else "no"


def _fmt_or_na(value, width, spec=".2f"):
    """Format a number with given width/spec, or right-align 'N/A' if None."""
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:{width}{spec}}"


# ──────────────────────────────────────────────────────────────────────────────
# block_sparse_attention-specific helpers
# ──────────────────────────────────────────────────────────────────────────────


def sabi_to_block_sparse_mask(sabi: torch.Tensor, mask_shape: tuple,
                              device: str = "npu") -> torch.Tensor:
    """Convert sabi (sparse attention block indices: uint16 keeper-indices,
    padded 0xFFFF) into the block_sparse_attention kernel's int8
    `[B, H, q_blocks, kv_blocks]` mask.

    `mask_shape`: `(b, h, q_blocks, kv_blocks)`. 1 = block kept; 0 = skipped.
    """
    # NPU's aclnnNeScalar / scatter don't support uint16 directly. Cast to
    # int32 first; the operation set on int32 is fully supported.
    sabi_i32 = sabi.to(torch.int32)
    valid = (sabi_i32 != SABI_PAD)
    sabi_long = sabi_i32.long()
    # For padding entries set the index to 0; src=0 there so scatter_add is a no-op.
    # Use torch.zeros (not zeros_like) so torch_npu doesn't try to inherit the
    # source's internal memory format and warn under allow_internal_format=False.
    zeros = torch.zeros(sabi_long.shape, dtype=sabi_long.dtype, device=sabi_long.device)
    sabi_safe = torch.where(valid, sabi_long, zeros)

    mask_int = torch.zeros(*mask_shape, dtype=torch.int32, device=sabi.device)
    mask_int.scatter_add_(3, sabi_safe, valid.to(torch.int32))
    mask = (mask_int > 0).to(torch.int8)
    if str(mask.device) != device:
        mask = mask.to(device)
    return mask.contiguous()
