#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Torch extension test for stem_oam_prep_varlen_q.

Tests the cann_ops_transformer torch extension wrapper against a pure
PyTorch CPU reference implementation.

Usage:
    python test_torch_stem_oam_prep_varlen_q.py
    python test_torch_stem_oam_prep_varlen_q.py --device-id 0
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch_npu


def find_repo_root():
    for path in Path(__file__).resolve().parents:
        if (path / "MindSpeed").exists() or (path / "build.sh").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
TORCH_EXTENSION_PATH = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_PATH.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_PATH))

import cann_ops_transformer  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import


DIM_QK = 128
ATOL = 1e-2
RTOL = 1e-2


def prep_q_reference(q_fp8, q_scale, q_seq_lens, cu_seqlens_q,
                     stem_block_size=128, stem_stride=16):
    """Pure PyTorch CPU reference matching the Ascend stem_oam_prep_varlen_q kernel.

    Computes the weighted group sum of Q x q_scale, flattened in natural order.

    Args:
        q_fp8:          [total_tokens, H_q, D]  FP8_E4M3FN
        q_scale:        [total_tokens, H_q]     FP32 (per-token-per-head scale)
        q_seq_lens:     [batch]                 INT32 tensor
        cu_seqlens_q:   [batch+1]               INT32 tensor
        stem_block_size: int (default 128)
        stem_stride:    int (default 16)

    Returns:
        qflat: [batch, H_q, max_Qb, stem_stride * D]  BF16
    """
    num_head_q = q_fp8.shape[1]
    dim_qk = q_fp8.shape[2]
    num_batch = q_seq_lens.shape[0]
    R = stem_block_size // stem_stride

    q_padded_lens = (
        (q_seq_lens.to(torch.int64) + stem_block_size - 1)
        // stem_block_size * stem_block_size
    )
    max_q_padded = q_padded_lens.max().item()
    max_Qb = max_q_padded // stem_block_size

    qflat_out = torch.zeros(
        num_batch, num_head_q, max_Qb, stem_stride * dim_qk,
        dtype=torch.bfloat16,
    )

    total_tokens = q_fp8.shape[0]

    for b in range(num_batch):
        q_len = q_seq_lens[b].item()
        q_padded = q_padded_lens[b].item()
        num_q_blocks = q_padded // stem_block_size
        cu_off = cu_seqlens_q[b].item()

        q_dense = torch.zeros(q_padded, num_head_q, dim_qk, dtype=torch.float32)
        q_dense[:q_len] = q_fp8[cu_off:cu_off + q_len].float()

        for h in range(num_head_q):
            q_h = q_dense[:num_q_blocks * stem_block_size, h, :]
            q_blocks = q_h.reshape(num_q_blocks, R, stem_stride, dim_qk)

            if q_scale is not None:
                positions = torch.arange(
                    num_q_blocks * stem_block_size, dtype=torch.long,
                ).reshape(num_q_blocks, R, stem_stride)
                global_positions = cu_off + positions
                valid_mask = global_positions < q_scale.shape[0]
                scales = torch.zeros_like(global_positions, dtype=torch.float32)
                scales[valid_mask] = q_scale[global_positions[valid_mask], h]
                q_group_sum = (q_blocks * scales.unsqueeze(-1)).sum(dim=1)
            else:
                q_group_sum = q_blocks.sum(dim=1)

            qflat_out[b, h, :num_q_blocks] = q_group_sum.reshape(
                num_q_blocks, stem_stride * dim_qk,
            ).to(torch.bfloat16)

    return qflat_out


def compare_result(output, golden, atol=ATOL, rtol=RTOL):
    out_f = output.float()
    gold_f = golden.float()
    diff = (out_f - gold_f).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    passed = torch.allclose(out_f, gold_f, atol=atol, rtol=rtol)
    return passed, max_abs, mean_abs


def run_test(name, q_fp8, q_scale, q_seq_lens, cu_seqlens_q,
             stem_block_size=128, stem_stride=16, device='npu:0'):
    op_fn = torch.ops.cann_ops_transformer.stem_oam_prep_varlen_q

    q_npu = q_fp8.to(device)
    qscale_npu = q_scale.to(device) if q_scale is not None else None

    qflat = op_fn(q_npu, q_seq_lens, cu_seqlens_q,
                  qScale=qscale_npu,
                  stemBlockSize=stem_block_size, stemStride=stem_stride)
    torch.npu.synchronize()

    golden = prep_q_reference(q_fp8, q_scale, q_seq_lens, cu_seqlens_q,
                              stem_block_size, stem_stride)

    passed, max_abs, mean_abs = compare_result(qflat.cpu(), golden)
    print(f"  {name:45s}  {'PASSED' if passed else 'FAILED'}  "
          f"max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}")
    return name, passed, max_abs


def main():
    parser = argparse.ArgumentParser(description="Torch extension test for stem_oam_prep_varlen_q")
    parser.add_argument('--device-id', type=int, default=0, help='NPU device id')
    args = parser.parse_args()

    torch_npu.npu.set_device(args.device_id)
    device = f'npu:{args.device_id}'

    B = 128
    S = 16

    results = []

    print("=" * 90)
    print("  Torch Extension Test: stem_oam_prep_varlen_q")
    print(f"  device={device}  atol={ATOL}  rtol={RTOL}")
    print("=" * 90)

    # P1: Single block, 1 head, scales=1.0
    q_len = 128
    q_fp8 = (torch.randn(q_len, 1, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.ones(q_len, 1, dtype=torch.float32)
    q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    results.append(run_test("P1 single_block_1head_scale1", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, B, S, device))

    # P2: Single block, 1 head, random scales
    q_fp8 = (torch.randn(q_len, 1, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(q_len, 1, dtype=torch.float32).abs() + 0.5
    results.append(run_test("P2 single_block_1head_rand_scale", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, B, S, device))

    # P3: Two blocks, 2 heads, scales=1.0
    q_len = 256
    q_fp8 = (torch.randn(q_len, 2, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.ones(q_len, 2, dtype=torch.float32)
    q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    results.append(run_test("P3 two_blocks_2heads_scale1", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, B, S, device))

    # P4: Varlen, 2 batches, 2 heads, random scales
    q_lens = [100, 200]
    total = sum(q_lens)
    q_fp8 = (torch.randn(total, 2, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(total, 2, dtype=torch.float32).abs() + 0.5
    q_seq_lens = torch.tensor(q_lens, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_lens[0], total], dtype=torch.int32)
    results.append(run_test("P4 varlen_2batch_2head_rand", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, B, S, device))

    # P5: Varlen, 3 batches (unaligned to B), 4 heads, random scales
    q_lens = [50, 130, 300]
    total = sum(q_lens)
    q_fp8 = (torch.randn(total, 4, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(total, 4, dtype=torch.float32).abs() + 0.5
    q_seq_lens = torch.tensor(q_lens, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_lens[0], q_lens[0] + q_lens[1], total], dtype=torch.int32)
    results.append(run_test("P5 varlen_3batch_4head_unaligned", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, B, S, device))

    # P6: stemBlockSize=256, stemStride=32, 2 heads, random scales
    q_len = 512
    q_fp8 = (torch.randn(q_len, 2, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(q_len, 2, dtype=torch.float32).abs() + 0.5
    q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    results.append(run_test("P6 B256_S32_single_batch", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, 256, 32, device))

    # P7: stemBlockSize=64, stemStride=16, 2 batches, 2 heads
    q_lens = [30, 100]
    total = sum(q_lens)
    q_fp8 = (torch.randn(total, 2, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(total, 2, dtype=torch.float32).abs() + 0.5
    q_seq_lens = torch.tensor(q_lens, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_lens[0], total], dtype=torch.int32)
    results.append(run_test("P7 B64_S16_varlen_2batch", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, 64, 16, device))

    # P8: stemBlockSize=128, stemStride=64, single batch, 1 head
    q_len = 256
    q_fp8 = (torch.randn(q_len, 1, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(q_len, 1, dtype=torch.float32).abs() + 0.5
    q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    results.append(run_test("P8 B128_S64_single_batch", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, 128, 64, device))

    # P9: Large varlen, 4 batches, 8 heads
    q_lens = [128, 256, 512, 1000]
    total = sum(q_lens)
    q_fp8 = (torch.randn(total, 8, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(total, 8, dtype=torch.float32).abs() + 0.5
    q_seq_lens = torch.tensor(q_lens, dtype=torch.int32)
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32)
    results.append(run_test("P9 large_varlen_4batch_8head", q_fp8, q_scale,
                            q_seq_lens, cu_seqlens_q, B, S, device))

    # Summary
    total_tests = len(results)
    passed_count = sum(r[1] for r in results)
    failed_count = total_tests - passed_count
    print(f"{'=' * 90}")
    print(f"Total: {total_tests}, Passed: {passed_count}, Failed: {failed_count}")
    print(f"Status: {'PASSED' if failed_count == 0 else 'FAILED'}")
    print(f"{'=' * 90}")
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
