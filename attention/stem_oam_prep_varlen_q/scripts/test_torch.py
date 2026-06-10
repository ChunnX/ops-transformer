# ============================================================================
# PyTorch pathway test for stem_oam_prep_varlen_q
# ============================================================================

import sys
import os
import math

import torch
import torch_npu

from golden import stem_oam_prep_varlen_q_golden

# Operator configuration
SO_NAME = "libstem_oam_prep_varlen_q_ops.so"
OP_NAME = "stem_oam_prep_varlen_q"
ATOL = 1e-2
RTOL = 1e-2

B = 128   # STEM_BLOCK_SIZE
S = 16    # STEM_STRIDE
D = 128   # DIM_QK


def run_test(name, q, qscale, kscale, q_seq_lens, cu_seqlens_q):
    """Run a single test case, return (name, passed, max_diff)"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    qflat = op_fn(q.npu(), qscale.npu(), kscale, q_seq_lens, cu_seqlens_q)

    golden = stem_oam_prep_varlen_q_golden(
        q.cpu(), qscale.cpu(), kscale, q_seq_lens.cpu(), cu_seqlens_q.cpu(), B=B, S=S)
    golden_npu = golden.npu()

    max_diff = torch.max(torch.abs(qflat.float() - golden_npu.float())).item()
    passed = torch.allclose(qflat.cpu().float(), golden.cpu().float(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    # Load operator library
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # P1: Single block, BF16, all scales=1.0
    q_len = 128
    q = torch.randn(q_len, 1, D, dtype=torch.bfloat16)
    qscale = torch.ones(1, 1, q_len, dtype=torch.float32)
    kscale = 1.0
    q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    results.append(run_test("P1 single_block", q, qscale, kscale, q_seq_lens, cu_seqlens_q))

    # P2: Single block, random scales
    q = torch.randn(q_len, 1, D, dtype=torch.bfloat16)
    qscale = torch.randn(1, 1, q_len, dtype=torch.float32).abs() + 0.5
    kscale = 1.5
    q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    results.append(run_test("P2 random_scales", q, qscale, kscale, q_seq_lens, cu_seqlens_q))

    # P3: Two blocks, 2 heads
    q_len = 256  # 2 blocks
    q = torch.randn(q_len, 2, D, dtype=torch.bfloat16)
    qscale = torch.ones(1, 2, q_len, dtype=torch.float32)
    kscale = 1.0
    q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    results.append(run_test("P3 two_blocks_2heads", q, qscale, kscale, q_seq_lens, cu_seqlens_q))

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