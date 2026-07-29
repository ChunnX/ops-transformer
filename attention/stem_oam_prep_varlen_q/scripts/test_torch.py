# ============================================================================
# PyTorch pathway test for stem_oam_prep_varlen_q
# ============================================================================

import sys
import os
import math

import torch
import torch_npu

OP_NAME = "stem_oam_prep_varlen_q"
ATOL = 1e-2
RTOL = 1e-2

B = 128   # STEM_BLOCK_SIZE
S = 16    # STEM_STRIDE
D = 128   # DIM_QK


def stem_oam_prep_varlen_q_golden(q_fp8, q_scale, q_seq_lens, cu_seqlens_q, B=128, S=16):
    """Pure PyTorch CPU reference matching the Ascend stem_oam_prep_varlen_q kernel."""
    num_head_q = q_fp8.shape[1]
    dim_qk = q_fp8.shape[2]
    num_batch = len(q_seq_lens)
    R = B // S

    q_padded_lens = [
        ((ql + B - 1) // B) * B for ql in q_seq_lens
    ]
    max_q_padded = max(q_padded_lens)
    max_Qb = max_q_padded // B

    qflat_out = torch.zeros(
        num_batch, num_head_q, max_Qb, S * dim_qk, dtype=torch.bfloat16)

    for b in range(num_batch):
        q_len = q_seq_lens[b]
        q_padded = q_padded_lens[b]
        num_q_blocks = q_padded // B
        cu_off = cu_seqlens_q[b]

        q_dense = torch.zeros(q_padded, num_head_q, dim_qk, dtype=torch.float32)
        q_dense[:q_len] = q_fp8[cu_off:cu_off + q_len].float()

        for h in range(num_head_q):
            q_h = q_dense[:num_q_blocks * B, h, :]
            q_blocks = q_h.reshape(num_q_blocks, R, S, dim_qk)

            if q_scale is not None:
                positions = torch.arange(
                    num_q_blocks * B, dtype=torch.long,
                ).reshape(num_q_blocks, R, S)
                global_positions = cu_off + positions
                valid_mask = global_positions < q_scale.shape[0]
                scales = torch.zeros_like(global_positions, dtype=torch.float32)
                scales[valid_mask] = q_scale[global_positions[valid_mask], h]
                q_group_sum = (q_blocks * scales.unsqueeze(-1)).sum(dim=1)
            else:
                q_group_sum = q_blocks.sum(dim=1)

            qflat_out[b, h, :num_q_blocks] = q_group_sum.reshape(
                num_q_blocks, S * dim_qk).to(torch.bfloat16)

    return qflat_out


def run_test(name, q_fp8, qscale, q_seq_lens, cu_seqlens_q):
    """Run a single test case, return (name, passed, max_diff)"""
    op_fn = getattr(torch.ops.cann_ops_transformer, OP_NAME)
    qflat = op_fn(q_fp8.npu(), q_seq_lens, cu_seqlens_q,
                  q_scale=qscale.npu(), stem_block_size=B, stem_stride=S)
    torch.npu.synchronize()

    golden = stem_oam_prep_varlen_q_golden(
        q_fp8.cpu(), qscale.cpu(), q_seq_lens, cu_seqlens_q, B=B, S=S)

    max_diff = torch.max(torch.abs(qflat.float() - golden.float())).item()
    passed = torch.allclose(qflat.cpu().float(), golden.float(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    results = []

    # P1: Single block, FP8, all scales=1.0
    q_len = 128
    q_fp32 = torch.randn(q_len, 1, D) * 0.1
    q_fp8 = q_fp32.to(torch.float8_e4m3fn)
    qscale = torch.ones(q_len, 1, dtype=torch.float32)
    q_seq_lens = [q_len]
    cu_seqlens_q = [0, q_len]
    results.append(run_test("P1 single_block", q_fp8, qscale, q_seq_lens, cu_seqlens_q))

    # P2: Single block, random scales
    q_fp32 = torch.randn(q_len, 1, D) * 0.1
    q_fp8 = q_fp32.to(torch.float8_e4m3fn)
    qscale = torch.randn(q_len, 1, dtype=torch.float32).abs() + 0.5
    q_seq_lens = [q_len]
    cu_seqlens_q = [0, q_len]
    results.append(run_test("P2 random_scales", q_fp8, qscale, q_seq_lens, cu_seqlens_q))

    # P3: Two blocks, 2 heads
    q_len = 256  # 2 blocks
    q_fp32 = torch.randn(q_len, 2, D) * 0.1
    q_fp8 = q_fp32.to(torch.float8_e4m3fn)
    qscale = torch.ones(q_len, 2, dtype=torch.float32)
    q_seq_lens = [q_len]
    cu_seqlens_q = [0, q_len]
    results.append(run_test("P3 two_blocks_2heads", q_fp8, qscale, q_seq_lens, cu_seqlens_q))

    # Summary
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff:.6f})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
