#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

"""Graph mode test for stem_oam_prep_varlen_q.

Verifies that the custom op works correctly under torch.compile graph mode
on NPU, comparing eager vs graph outputs and against a CPU golden.

Usage:
    python test_graph_stem_oam_prep_varlen_q.py
    python test_graph_stem_oam_prep_varlen_q.py --device-id 0
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch_npu


def find_repo_root():
    for path in Path(__file__).resolve().parents:
        if (path / "torch_extension").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
TORCH_EXTENSION_PATH = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_PATH.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_PATH))

import cann_ops_transformer  # noqa: E402,F401

DIM_QK = 128
ATOL = 1e-2
RTOL = 1e-2


def prep_q_reference(q, q_scale, q_seq_lens, cu_seqlens_q,
                     stem_block_size=128, stem_stride=16):
    num_head_q = q.shape[1]
    dim_qk = q.shape[2]
    num_batch = len(q_seq_lens)
    R = stem_block_size // stem_stride

    q_padded_lens = [
        ((ql + stem_block_size - 1) // stem_block_size) * stem_block_size
        for ql in q_seq_lens
    ]
    max_q_padded = max(q_padded_lens)
    max_Qb = max_q_padded // stem_block_size

    qflat_out = torch.zeros(
        num_batch, num_head_q, max_Qb, stem_stride * dim_qk,
        dtype=torch.bfloat16,
    )

    for b in range(num_batch):
        q_len = q_seq_lens[b]
        q_padded = q_padded_lens[b]
        num_q_blocks = q_padded // stem_block_size
        cu_off = cu_seqlens_q[b]

        q_dense = torch.zeros(q_padded, num_head_q, dim_qk, dtype=torch.float32)
        q_dense[:q_len] = q[cu_off:cu_off + q_len].float()

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
    passed = torch.allclose(out_f, gold_f, atol=atol, rtol=rtol)
    return passed, max_abs


def run_graph_test(name, q, q_scale, q_seq_lens, cu_seqlens_q,
                   stem_block_size=128, stem_stride=16, device='npu:0'):
    op_fn = torch.ops.cann_ops_transformer.stem_oam_prep_varlen_q

    q_npu = q.to(device)
    qscale_npu = q_scale.to(device) if q_scale is not None else None

    # Eager mode
    qflat_eager = op_fn(q_npu, q_seq_lens, cu_seqlens_q,
                        q_scale=qscale_npu,
                        stem_block_size=stem_block_size, stem_stride=stem_stride)
    torch.npu.synchronize()

    # Graph mode via torch.compile
    def call_op(q_in):
        return op_fn(q_in, q_seq_lens, cu_seqlens_q,
                     q_scale=qscale_npu,
                     stem_block_size=stem_block_size, stem_stride=stem_stride)

    compiled_fn = torch.compile(call_op, backend="npu", dynamic=False)
    qflat_graph = compiled_fn(q_npu)
    torch.npu.synchronize()

    # CPU golden
    golden = prep_q_reference(q, q_scale, q_seq_lens, cu_seqlens_q,
                              stem_block_size, stem_stride)

    # Compare eager vs golden
    passed_eager, max_abs_eager = compare_result(qflat_eager.cpu(), golden)

    # Compare graph vs golden
    passed_graph, max_abs_graph = compare_result(qflat_graph.cpu(), golden)

    # Compare eager vs graph (consistency)
    passed_consist, max_abs_consist = compare_result(qflat_eager.cpu(), qflat_graph.cpu())

    all_passed = passed_eager and passed_graph and passed_consist
    status = "PASSED" if all_passed else "FAILED"
    print(f"  {name:50s}  {status}  "
          f"eager={max_abs_eager:.4e}  graph={max_abs_graph:.4e}  "
          f"diff={max_abs_consist:.4e}")
    return name, all_passed


def main():
    parser = argparse.ArgumentParser(description="Graph mode test for stem_oam_prep_varlen_q")
    parser.add_argument('--device-id', type=int, default=0, help='NPU device id')
    args = parser.parse_args()

    torch_npu.npu.set_device(args.device_id)
    device = f'npu:{args.device_id}'

    results = []

    print("=" * 100)
    print("  Graph Mode Test: stem_oam_prep_varlen_q")
    print(f"  device={device}  atol={ATOL}  rtol={RTOL}")
    print("=" * 100)

    # G1: FP8, 1 head, 1 batch, exact block
    q_len = 128
    q = (torch.randn(q_len, 1, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.ones(q_len, 1, dtype=torch.float32)
    results.append(run_graph_test("G1 fp8_H1_B1_blk128_s16_exact",
                   q, q_scale, [128], [0, 128], 128, 16, device))

    # G2: FP8, 2 heads, 2 batches, partial blocks
    q_lens = [100, 200]
    total = sum(q_lens)
    q = (torch.randn(total, 2, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(total, 2, dtype=torch.float32).abs() + 0.5
    results.append(run_graph_test("G2 fp8_H2_B2_blk128_s16_partial",
                   q, q_scale, q_lens, [0, 100, 300], 128, 16, device))

    # G3: FP8, 4 heads, 3 batches, unaligned
    q_lens = [50, 130, 300]
    total = sum(q_lens)
    q = (torch.randn(total, 4, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(total, 4, dtype=torch.float32).abs() + 0.5
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    results.append(run_graph_test("G3 fp8_H4_B3_blk128_s16_unaligned",
                   q, q_scale, q_lens, cu, 128, 16, device))

    # G4: BF16, 2 heads, 1 batch, no qScale
    q_len = 256
    q = (torch.randn(q_len, 2, DIM_QK) * 0.1).to(torch.bfloat16)
    results.append(run_graph_test("G4 bf16_H2_B1_blk128_s16_noscale",
                   q, None, [256], [0, 256], 128, 16, device))

    # G5: BF16, 4 heads, 4 batches, large
    q_lens = [128, 256, 512, 1000]
    total = sum(q_lens)
    q = (torch.randn(total, 4, DIM_QK) * 0.1).to(torch.bfloat16)
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    results.append(run_graph_test("G5 bf16_H4_B4_blk128_s16_large",
                   q, None, q_lens, cu, 128, 16, device))

    # G6: FP16, 8 heads, 2 batches, blk256_s64
    q_lens = [256, 512]
    total = sum(q_lens)
    q = (torch.randn(total, 8, DIM_QK) * 0.1).to(torch.float16)
    results.append(run_graph_test("G6 fp16_H8_B2_blk256_s64",
                   q, None, q_lens, [0, 256, 768], 256, 64, device))

    # G7: FP8, 1 head, 1 batch, smallest input (1 token)
    q = (torch.randn(1, 1, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.ones(1, 1, dtype=torch.float32)
    results.append(run_graph_test("G7 fp8_H1_B1_tiny_1token",
                   q, q_scale, [1], [0, 1], 32, 16, device))

    # G8: FP8, 8 heads, 8 batches, varied lengths
    q_lens = [64, 128, 32, 256, 96, 192, 128, 64]
    total = sum(q_lens)
    q = (torch.randn(total, 8, DIM_QK) * 0.1).to(torch.float8_e4m3fn)
    q_scale = torch.randn(total, 8, dtype=torch.float32).abs() + 0.5
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    results.append(run_graph_test("G8 fp8_H8_B8_blk64_s32_varied",
                   q, q_scale, q_lens, cu, 64, 32, device))

    total_tests = len(results)
    passed_count = sum(r[1] for r in results)
    failed_count = total_tests - passed_count
    print(f"{'=' * 100}")
    print(f"Total: {total_tests}, Passed: {passed_count}, Failed: {failed_count}")
    print(f"Status: {'PASSED' if failed_count == 0 else 'FAILED'}")
    print(f"{'=' * 100}")
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
