"""
Test script for lightning_indexer_kl_loss.

Compares the NPU operator output with a pure-PyTorch CPU reference
across multiple shapes, dtypes, and edge cases.
"""

import sys
from pathlib import Path

import torch
import torch_npu

# Ensure the torch_extension package is importable
TORCH_EXTENSION_PATH = Path(__file__).resolve().parent
if str(TORCH_EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(TORCH_EXTENSION_PATH))

from cann_ops_transformer.ops import lightning_indexer_kl_loss  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import


# ---------------------------------------------------------------------------
# Reference implementation (纯 PyTorch, float 精度基准)
# ---------------------------------------------------------------------------
def lightning_indexer_kl_loss_ref(target_score, index_probs, eps=1e-9):
    """PyTorch reference for KL divergence used in Lightning Indexer loss.

    P = y / sum(y, dim=-1, keepdim=True)      # teacher: normalize target_score
    log_P = log(clamp_min(P, eps))             # clamp_min 防止 log(0) 产生 NaN
    log_Y = log(Y + eps)                       # student: +eps 保住 Y→0 处的梯度
    loss = sum((log_P - log_Y) * y)            # weighted by original y
    """
    y = target_score
    Y = index_probs
    reduce_target = torch.sum(y, dim=-1, keepdim=True)
    norm_target = y / (reduce_target + eps)
    log_p = torch.log(torch.clamp_min(norm_target, eps))
    log_Y = torch.log(Y + eps)
    tmp = log_p - log_Y
    result = tmp * y
    return torch.sum(result)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def _softmax_last_dim(x):
    """Apply softmax along last dim. Works for 2D and 3D tensors."""
    xmax = x.amax(dim=-1, keepdim=True)
    exp_x = torch.exp(x - xmax)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def gen_data_2d(b, d, dtype=torch.float32, device="cpu"):
    """2-D case: (B, D)."""
    target_score = torch.randn(b, d, dtype=dtype, device=device)
    index_probs = _softmax_last_dim(torch.randn(b, d, dtype=dtype, device=device))
    return target_score, index_probs


def gen_data_3d(b, h, d, dtype=torch.float32, device="cpu"):
    """3-D case: (B, H, D)."""
    target_score = torch.randn(b, h, d, dtype=dtype, device=device)
    index_probs = _softmax_last_dim(torch.randn(b, h, d, dtype=dtype, device=device))
    return target_score, index_probs


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def data_compare(npu_output, cpu_output, diff_thd=0.001, pct_thd=0.05, max_diff_hd=0.1):
    """Compare NPU and CPU scalar outputs with configurable thresholds.

    Args:
        npu_output: scalar float — NPU operator result.
        cpu_output: scalar float — reference result.
        diff_thd: relative error threshold (fraction).
        pct_thd: max allowed failing element ratio (only meaningful for tensors).
        max_diff_hd: any element's absolute error must not exceed this.

    Returns:
        True if all checks pass, False otherwise.
    """
    ref_v = float(cpu_output)
    npu_v = float(npu_output)

    abs_diff = abs(npu_v - ref_v)
    rel_diff = abs_diff / max(abs(ref_v), 1e-12)

    passed = True
    if abs_diff > max_diff_hd:
        print(f"  FAIL: abs_diff={abs_diff:.6e} > max_diff_hd={max_diff_hd}")
        passed = False
    if rel_diff > diff_thd:
        print(f"  FAIL: rel_diff={rel_diff:.6e} > diff_thd={diff_thd}")
        passed = False

    print(f"  ref={ref_v:.10f}  npu={npu_v:.10f}  abs_diff={abs_diff:.6e}  rel_diff={rel_diff:.6e}  {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_basic_2d():
    """2-D (B, D) basic case, float32."""
    print("\n=== test_basic_2d: (B, D) float32 ===")
    b, d = 4, 128
    y, Y = gen_data_2d(b, d)
    y_npu, Y_npu = y.npu(), Y.npu()

    ref = lightning_indexer_kl_loss_ref(y, Y)
    npu_out = lightning_indexer_kl_loss(y_npu, Y_npu)

    return data_compare(npu_out.cpu(), ref, diff_thd=0.001, max_diff_hd=0.1)


def test_basic_3d():
    """3-D (B, H, D) basic case, float32."""
    print("\n=== test_basic_3d: (B, H, D) float32 ===")
    b, h, d = 4, 10, 128
    y, Y = gen_data_3d(b, h, d)
    y_npu, Y_npu = y.npu(), Y.npu()

    ref = lightning_indexer_kl_loss_ref(y, Y)
    npu_out = lightning_indexer_kl_loss(y_npu, Y_npu)

    return data_compare(npu_out.cpu(), ref, diff_thd=0.001, max_diff_hd=0.1)


def test_fp16():
    """Half-precision (float16) 3-D case."""
    print("\n=== test_fp16: (B, H, D) float16 ===")
    b, h, d = 4, 10, 128
    y, Y = gen_data_3d(b, h, d, dtype=torch.float16)
    y_npu, Y_npu = y.npu(), Y.npu()

    ref = lightning_indexer_kl_loss_ref(y.float(), Y.float())
    npu_out = lightning_indexer_kl_loss(y_npu, Y_npu)

    return data_compare(npu_out.cpu(), ref, diff_thd=0.005, max_diff_hd=0.1)


def test_bf16():
    """Bfloat16 3-D case."""
    print("\n=== test_bf16: (B, H, D) bfloat16 ===")
    b, h, d = 4, 10, 128
    y, Y = gen_data_3d(b, h, d, dtype=torch.bfloat16)
    y_npu, Y_npu = y.npu(), Y.npu()

    ref = lightning_indexer_kl_loss_ref(y.float(), Y.float())
    npu_out = lightning_indexer_kl_loss(y_npu, Y_npu)

    return data_compare(npu_out.cpu(), ref, diff_thd=0.005, max_diff_hd=0.1)


def test_custom_eps():
    """Custom eps value."""
    print("\n=== test_custom_eps: (B, H, D) float32, eps=1e-6 ===")
    b, h, d = 2, 4, 64
    y, Y = gen_data_3d(b, h, d)
    y_npu, Y_npu = y.npu(), Y.npu()

    eps = 1e-6
    ref = lightning_indexer_kl_loss_ref(y, Y, eps=eps)
    npu_out = lightning_indexer_kl_loss(y_npu, Y_npu, eps=eps)

    return data_compare(npu_out.cpu(), ref, diff_thd=0.001, max_diff_hd=0.1)


def test_large_d():
    """Large D dimension to stress the UB tile path."""
    print("\n=== test_large_d: (B, H, 512) float32 ===")
    b, h, d = 2, 4, 512
    y, Y = gen_data_3d(b, h, d)
    y_npu, Y_npu = y.npu(), Y.npu()

    ref = lightning_indexer_kl_loss_ref(y, Y)
    npu_out = lightning_indexer_kl_loss(y_npu, Y_npu)

    return data_compare(npu_out.cpu(), ref, diff_thd=0.001, max_diff_hd=0.1)


def test_edge_zero_target():
    """Edge case: target_score all zeros — loss should be finite."""
    print("\n=== test_edge_zero_target: all-zero target_score ===")
    b, h, d = 2, 4, 128
    device = torch.device("npu:0")
    y = torch.zeros(b, h, d, dtype=torch.float32, device=device)
    Y = _softmax_last_dim(torch.randn(b, h, d, dtype=torch.float32, device=device))

    npu_out = lightning_indexer_kl_loss(y, Y)
    npu_val = float(npu_out.cpu())
    passed = not (torch.isfinite(npu_out).any() is False or torch.isnan(npu_out).any())
    # When target is all zeros: P = 0/(0+eps)=0, log(clamp_min(0,eps))=log(eps), loss=0*(log(eps)-log(Y+eps))=0
    # So expected is 0
    print(f"  zero_target_loss={npu_val:.10f}  expected≈0.0  {'PASS' if passed else 'FAIL'}")
    return passed


def test_edge_zero_index_probs():
    """Edge case: index_probs all zeros — loss should be finite (log(Y+eps)-based)."""
    print("\n=== test_edge_zero_index_probs: all-zero index_probs ===")
    b, h, d = 2, 4, 128
    device = torch.device("npu:0")
    y = torch.randn(b, h, d, dtype=torch.float32, device=device)
    Y = torch.zeros(b, h, d, dtype=torch.float32, device=device)

    npu_out = lightning_indexer_kl_loss(y, Y)
    npu_val = float(npu_out.cpu())
    passed = torch.isfinite(npu_out).item()
    print(f"  zero_index_probs_loss={npu_val:.10f}  isfinite={'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch_npu.npu.set_device(0)

    tests = [
        ("test_basic_2d", test_basic_2d),
        ("test_basic_3d", test_basic_3d),
        ("test_fp16", test_fp16),
        ("test_bf16", test_bf16),
        ("test_custom_eps", test_custom_eps),
        ("test_large_d", test_large_d),
        ("test_edge_zero_target", test_edge_zero_target),
        ("test_edge_zero_index_probs", test_edge_zero_index_probs),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        try:
            ok = func()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed:
        sys.exit(1)
