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
Benchmark driver for `attention/block_sparse_attention` via the
`bsa_npu` python extension.

Key design choices:
- All workload helpers (`gen_pfa_inputs`, `create_attention_mask`,
  `sabi_to_block_sparse_mask`, timing, frame policy, …) live in the local
  `util.py`. No cross-tree imports.
- Converts the per-row block selection (sabi: uint16, padded with 0xFFFF)
  into block_sparse_attention's int8 `block_sparse_mask[B, H, q_blocks,
  kv_blocks]` (1 = kept, 0 = skipped).
- atten_mask is deliberately NOT forwarded to block_sparse_attention
  (the kernel does not support intra-block fine-grained masking on CANN
  8.5.0-beta.1). Sparsity 0 numerical comparison against the dense
  reference (npu_fusion_attention) is therefore expected to deviate;
  the priority sparsity window (0.5/0.8/0.9) is where masking is
  block-aligned and the comparison is meaningful.

Run as:
    cd attention/block_sparse_attention/benchmark
    ASCEND_RT_VISIBLE_DEVICES=5 python benchmark.py
"""

import os
import logging
import itertools
from typing import Callable, List

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import bsa_npu

# All shared workload helpers live in the local `util.py` (autonomous, no
# cross-tree imports).
from util import (
    DEVICE, DTYPE, INPUT_LAYOUT,
    B_VALS, H_VALS, S_VALS, D_VALS,
    N_REPEATS, N_WARMUP,
    ATTENTION_MATRIX, SPARSITY_VALS,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_KV as _DEFAULT_BLOCK_SIZE_KV,
    SABI_SORTED,
    FRAME as _DEFAULT_FRAME,
    PRINT_MASK,
    TORCH_REFERENCE,
    FrameWidths,
    gen_pfa_inputs,
    create_attention_mask,
    ref_attention_launcher,
    _run_timed,
    _check_correctness,
    _fmt_or_na,
    sabi_to_block_sparse_mask,
)

BLOCK_SIZE_KV = int(os.environ.get("BSA_BLOCK_SIZE_KV", _DEFAULT_BLOCK_SIZE_KV))

# Allow overriding the frame widths from the environment.
# Format: "left,right,top,bottom" (e.g. BSA_FRAME=8,1,29,1).
# Frame widths are in BLOCK units. When the KV block size grows the same
# left/right "block count" covers more KV tokens — so to keep the per-row
# token coverage comparable across granularities, scale left/right
# inversely with block_size_kv (top/bottom are Q-axis at fixed 128).
_BSA_FRAME_ENV = os.environ.get("BSA_FRAME")
if _BSA_FRAME_ENV is not None:
    _parts = [int(x.strip()) for x in _BSA_FRAME_ENV.split(",")]
    assert len(_parts) == 4, f"BSA_FRAME must be 4 comma-separated ints, got {_BSA_FRAME_ENV!r}"
    FRAME = FrameWidths(left_cols=_parts[0], right_cols=_parts[1],
                        top_rows=_parts[2], bottom_rows=_parts[3])
else:
    FRAME = _DEFAULT_FRAME

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUN_REFERENCE = False  # set True to enable per-row correctness comparison


def _make_our_fn_block_sparse(sabi: torch.Tensor, h: int, scale: float,
                               b: int, s_q: int, s_kv: int):
    """Build the closure that the benchmark timing harness invokes.

    sabi is converted to the kernel's int8 block mask ONCE outside the timed
    closure; the timed call only invokes the kernel.
    """
    q_blocks = (s_q + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    kv_blocks = (s_kv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV

    block_sparse_mask = sabi_to_block_sparse_mask(
        sabi, b, h, q_blocks, kv_blocks, device=DEVICE
    )

    def fn(q, k, v, seq, seqkv):
        out, _ = bsa_npu.block_sparse_attention(
            q, k, v,
            block_sparse_mask=block_sparse_mask,
            block_shape=[BLOCK_SIZE_Q, BLOCK_SIZE_KV],
            actual_seq_lengths=seq,
            actual_seq_lengths_kv=seqkv,
            num_heads=h,
            num_key_value_heads=h,
            q_input_layout=INPUT_LAYOUT,
            kv_input_layout=INPUT_LAYOUT,
            mask_type=0,
            scale_value=float(scale),
            inner_precise=0,         # fp32 softmax (high precision)
            block_size=BLOCK_SIZE_Q,
            # 2^31-1 means "no causal window" — kernel example uses these
            pre_tokens=2147483647,
            next_tokens=2147483647,
            softmax_lse_flag=False,
        )
        return out
    return fn


def _make_ref_fn(h, scale, atten_mask, run_ref_sparsity_0):
    def fn(q, k, v, seq, seqkv):
        return ref_attention_launcher(
            torch_reference=TORCH_REFERENCE,
            pfa_inputs=(q, k, v, h, scale, atten_mask, INPUT_LAYOUT),
            force_dense_sm=run_ref_sparsity_0,
        )
    return fn


def benchmark_block_sparse_attention():
    run_our = True
    run_ref = RUN_REFERENCE
    n_warmup = N_WARMUP
    n_repeat = N_REPEATS

    if not run_our and not run_ref:
        logger.info("Nothing to run.")
        return

    # Header — same format as BSA's benchmark.py for direct comparison
    logger.info("=" * 90)
    logger.info(f"  KERNEL=block_sparse_attention  {DTYPE=}  {INPUT_LAYOUT=}  {ATTENTION_MATRIX=}")
    logger.info(f"  BLOCK_SIZE_Q={BLOCK_SIZE_Q}  BLOCK_SIZE_KV={BLOCK_SIZE_KV}  {SABI_SORTED=}"
                + (f"  (KV overridden via BSA_BLOCK_SIZE_KV={BLOCK_SIZE_KV})"
                   if BLOCK_SIZE_KV != _DEFAULT_BLOCK_SIZE_KV else ""))
    logger.info(f"  {FRAME}"
                + ("  (overridden via BSA_FRAME)" if _BSA_FRAME_ENV is not None else ""))
    logger.info(f"  {TORCH_REFERENCE=}")
    logger.info(f"  (atten_mask=nullptr — intra-block masking dropped; sparsity 0 row will diverge from ref)")
    logger.info("=" * 90)
    logger.info(
        f"{'H':>3} {'B':>3} {'s_q':>6} {'s_kv':>6} {'D':>4} {'sparsity':>9} {'Outputs_equal':>15} "
        f"{'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18}"
    )
    logger.info("-" * 90)

    for b, h, s_kv, d, sparsity in itertools.product(B_VALS, H_VALS, S_VALS, D_VALS, SPARSITY_VALS):
        s_q = s_kv

        # Reuse BSA's mask + sabi generator. Infeasible (sparsity, frame, S, block_size)
        # combinations raise inside generate_sparse_blocks_by_row — catch + skip.
        try:
            pfa_atten_mask, bsa_atten_mask, sabi, sm, scale, pre_tok, post_tok = create_attention_mask(
                b, h, s_q, s_kv, d, sparsity, ATTENTION_MATRIX, frame=FRAME,
                block_size_q=BLOCK_SIZE_Q, block_size_kv=BLOCK_SIZE_KV,
                device=DEVICE, emit_atten_mask=run_ref,
            )
        except ValueError as e:
            logger.info(
                f"{h:>3} {b:>3} {s_q:>6} {s_kv:>6} {d:>4} {sparsity:>9.2f} {'INFEASIBLE':>15} "
                f"{'N/A':>18} {'N/A':>18}"
            )
            continue

        if PRINT_MASK and pfa_atten_mask is not None:
            logger.info(pfa_atten_mask.int())
            logger.info(pfa_atten_mask.shape)

        run_ref_sparsity_0 = sparsity == 0

        our_fn = _make_our_fn_block_sparse(sabi, h, scale, b, s_q, s_kv)
        ref_fn = _make_ref_fn(h, scale, pfa_atten_mask, run_ref_sparsity_0)

        # Optional correctness check
        are_equal_ref = "N/A"
        if (run_our and run_ref) or run_ref_sparsity_0:
            are_equal_ref = _check_correctness(our_fn, ref_fn, b, h, s_q, s_kv, d, device=DEVICE)

        # Timing
        inputs = [gen_pfa_inputs(b, h, s_q, s_kv, d, device=DEVICE, dtype=DTYPE)
                  for _ in range(n_warmup + n_repeat)]

        our_duration = None
        if run_our:
            our_duration = _run_timed(our_fn, inputs, n_warmup, n_repeat)

        ref_duration = None
        if run_ref or run_ref_sparsity_0:
            ref_duration = _run_timed(ref_fn, inputs, n_warmup, n_repeat)

        logger.info(
            f"{h:>3} {b:>3} {s_q:>6} {s_kv:>6} {d:>4} {sparsity:>9.2f} {are_equal_ref:>15} "
            f"{_fmt_or_na(ref_duration, 18)} {_fmt_or_na(our_duration, 18)}"
        )

    logger.info("=" * 90)


if __name__ == "__main__":
    benchmark_block_sparse_attention()
