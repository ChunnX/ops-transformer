#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Benchmark driver for `attention/block_sparse_attention` via
`cann_ops_transformer.ops.block_sparse_attention` (provided by the
`torch_extension/cann_ops_transformer/` torch extension package).

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
import cann_ops_transformer

# All shared workload helpers live in the local `util.py` (autonomous, no
# cross-tree imports).
from util import (
    DEVICE, DTYPE, INPUT_LAYOUT,
    B_VALS, H_VALS, S_VALS, D_VALS,
    N_REPEATS, N_WARMUP,
    SPARSITY_VALS,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_KV as _DEFAULT_BLOCK_SIZE_KV,
    FRAME as _DEFAULT_FRAME,
    PRINT_MASK,
    TORCH_REFERENCE,
    FrameWidths,
    WorkloadShape,
    gen_pfa_inputs,
    create_attention_mask,
    ref_attention_launcher,
    _run_timed,
    _check_correctness,
    _fmt_or_na,
    sabi_to_block_sparse_mask,
    bsa_inner_precise,
)

BLOCK_SIZE_KV = int(os.environ.get("BSA_BLOCK_SIZE_KV", _DEFAULT_BLOCK_SIZE_KV))

# inner_precise is SoC-gated: 950 → 4 (low/high mixed), 910/910_93 → 0 (fp32 softmax).
INNER_PRECISE = bsa_inner_precise()

# Allow overriding the frame widths via the BSA_FRAME env var.
# Format: four comma-separated integers — left,right,top,bottom (example "8,1,29,1").
# Frame widths are in BLOCK units. When the KV block size grows the same
# left/right "block count" covers more KV tokens — so to keep the per-row
# token coverage comparable across granularities, scale left/right
# inversely with block_size_kv (top/bottom are Q-axis at fixed 128).
_BSA_FRAME_ENV = os.environ.get("BSA_FRAME")
if _BSA_FRAME_ENV is not None:
    _parts = [int(x.strip()) for x in _BSA_FRAME_ENV.split(",")]
    if len(_parts) != 4:
        raise ValueError(f"BSA_FRAME must be 4 comma-separated ints, got {_BSA_FRAME_ENV!r}")
    FRAME = FrameWidths(left_cols=_parts[0], right_cols=_parts[1],
                        top_rows=_parts[2], bottom_rows=_parts[3])
else:
    FRAME = _DEFAULT_FRAME

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUN_REFERENCE = False  # set True to enable per-row correctness comparison


def _make_our_fn_block_sparse(block_sparse_mask: torch.Tensor, h: int, scale: float):
    """Build the closure that the benchmark timing harness invokes.

    The int8 block mask is built ONCE by the caller (outside the timed
    closure); the timed call here only invokes the kernel.
    """
    def fn(q, k, v, seq, seqkv):
        out, _ = cann_ops_transformer.ops.block_sparse_attention(
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
            inner_precise=INNER_PRECISE,  # 910: 0 (fp32 sm); 950: 4 (low/high mixed)
            block_size=0,            # paged cache not yet supported by kernel; must be 0
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


def _print_header():
    kv_note = (f"  (KV overridden via BSA_BLOCK_SIZE_KV={BLOCK_SIZE_KV})"
               if BLOCK_SIZE_KV != _DEFAULT_BLOCK_SIZE_KV else "")
    frame_note = "  (overridden via BSA_FRAME)" if _BSA_FRAME_ENV is not None else ""
    logger.info("=" * 90)
    logger.info(f"  KERNEL=block_sparse_attention  {DTYPE=}  {INPUT_LAYOUT=}")
    logger.info(f"  BLOCK_SIZE_Q={BLOCK_SIZE_Q}  BLOCK_SIZE_KV={BLOCK_SIZE_KV}{kv_note}")
    logger.info(f"  {FRAME}{frame_note}")
    logger.info(f"  {TORCH_REFERENCE=}")
    logger.info("=" * 90)
    logger.info(
        f"{'B':>3} {'H':>3} {'s_q':>6} {'s_kv':>6} {'D':>4} {'sparsity':>9} {'Outputs_equal':>15} "
        f"{'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18}"
    )
    logger.info("-" * 90)


def _print_row(shape: WorkloadShape, sparsity, equal, ref_us, our_us):
    logger.info(
        f"{shape.b:>3} {shape.h:>3} {shape.s_q:>6} {shape.s_kv:>6} {shape.d:>4} "
        f"{sparsity:>9.2f} {equal:>15} {_fmt_or_na(ref_us, 18)} {_fmt_or_na(our_us, 18)}"
    )


def benchmark_block_sparse_attention():
    run_ref = RUN_REFERENCE
    n_warmup = N_WARMUP
    n_repeat = N_REPEATS

    _print_header()

    for b, h, s_kv, d, sparsity in itertools.product(B_VALS, H_VALS, S_VALS, D_VALS, SPARSITY_VALS):
        s_q = s_kv
        shape = WorkloadShape(b, h, s_q, s_kv, d, BLOCK_SIZE_Q, BLOCK_SIZE_KV)

        try:
            pfa_atten_mask, sabi, scale = create_attention_mask(
                shape, sparsity, FRAME, DEVICE, emit_atten_mask=run_ref,
            )
        except ValueError:
            # Infeasible (sparsity, frame, S, block_size) — frame forces a higher
            # density than the target sparsity allows.
            _print_row(shape, sparsity, "INFEASIBLE", None, None)
            continue

        if PRINT_MASK and pfa_atten_mask is not None:
            logger.info(pfa_atten_mask.int())
            logger.info(pfa_atten_mask.shape)

        run_ref_sparsity_0 = sparsity == 0
        q_blocks = (s_q + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
        kv_blocks = (s_kv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
        block_sparse_mask = sabi_to_block_sparse_mask(
            sabi, (b, h, q_blocks, kv_blocks), device=DEVICE,
        )

        our_fn = _make_our_fn_block_sparse(block_sparse_mask, h, scale)
        ref_fn = _make_ref_fn(h, scale, pfa_atten_mask, run_ref_sparsity_0)

        are_equal_ref = "N/A"
        if run_ref or run_ref_sparsity_0:
            are_equal_ref = _check_correctness(our_fn, ref_fn, shape, DEVICE)

        inputs = [gen_pfa_inputs(shape, device=DEVICE, dtype=DTYPE)
                  for _ in range(n_warmup + n_repeat)]
        our_duration = _run_timed(our_fn, inputs, n_warmup)
        ref_duration = None
        if run_ref or run_ref_sparsity_0:
            ref_duration = _run_timed(ref_fn, inputs, n_warmup)

        _print_row(shape, sparsity, are_equal_ref, ref_duration, our_duration)

    logger.info("=" * 90)


if __name__ == "__main__":
    benchmark_block_sparse_attention()
