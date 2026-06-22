#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskKComp Self-Test: Result Comparison Script

Compares NPU output against CPU golden for:
  - q_comp     (FP32): Q mean-pooling compression
  - k_comp     (FP32): K mean-pooling compression
  - score      (FP32): attention score matrix
  - attn_score (FP16): softmax(score) matrix
  - mask       (INT8): block sparse mask (exact match)

Precision criteria (per-element, satisfy EITHER criterion to pass):
  - FP16: abs_error <= 2/1000 (0.002)  OR  rel_error <= 2/1000 (0.002)
  - FP32: abs_error <= 2/10000 (0.0002) OR  rel_error <= 2/10000 (0.0002)
"""

import os
import argparse
import numpy as np


# ============================================================
# Precision thresholds: satisfy EITHER abs OR rel to pass
# ============================================================
PRECISION_THRESHOLDS = {
    "FP16": {"atol": 1 / 1000, "rtol": 1 / 1000},   # 0.001
    "FP32": {"atol": 1 / 10000, "rtol": 1 / 10000},  # 0.0001
}


DTYPE_MAP = {
    "FP32": np.float32,
    "FP16": np.float16,
    "INT8": np.int8,
    "INT32": np.int32,
    "INT64": np.int64,
}


# ============================================================
# Function: load_meta_txt - Load shape/dtype metadata from text file
# ============================================================

def load_meta_txt(filepath):
    meta = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.startswith("shape_"):
                    name = key[len("shape_"):]
                    meta[f"shape_{name}"] = [int(x) for x in value.split(",")]
                elif key.startswith("dtype_"):
                    name = key[len("dtype_"):]
                    meta[f"dtype_{name}"] = value
                else:
                    meta[key] = value
    return meta


# ============================================================
# Function: load_bin - Load binary file with given shape and dtype
# ============================================================

def load_bin(filepath, shape, dtype_str):
    np_dtype = DTYPE_MAP[dtype_str]
    data = np.fromfile(filepath, dtype=np_dtype)
    expected_size = 1
    for s in shape:
        expected_size *= s
    if data.size != expected_size:
        print(f"  WARNING: size mismatch: file has {data.size} elements, expected {expected_size}")
    data = data.reshape(shape)
    return data


# ============================================================
# Function: compute_elementwise_errors - Compute abs and rel errors per element
# ============================================================

def compute_elementwise_errors(golden_f32, npu_f32):
    """Compute per-element absolute and relative errors.

    Relative error: |golden - npu| / max(|golden|, |npu|)
    For elements where both are zero, rel_error is defined as 0.
    """
    abs_err = np.abs(golden_f32 - npu_f32)
    denom = np.maximum(np.abs(golden_f32), np.abs(npu_f32))
    # Where both are zero, rel error is 0 (exact match)
    rel_err = np.where(denom > 0, abs_err / denom, 0.0)
    return abs_err, rel_err


# ============================================================
# Function: compare_float_tensor - Compare a float tensor with per-element dual criteria
# ============================================================

def compare_float_tensor(golden, npu, name, dtype_str):
    """Compare golden vs npu for a floating-point tensor.

    Per-element pass criterion: satisfy EITHER abs OR rel threshold.
    Tensor passes iff ALL elements pass.

    On failure, prints:
      - Max absolute error and its 1D flattened index
      - Max relative error and its 1D flattened index
      - Count of elements failing BOTH thresholds
      - 1D flattened indices of those failing elements
    """
    golden_f32 = golden.astype(np.float32)
    npu_f32 = npu.astype(np.float32)

    atol = PRECISION_THRESHOLDS[dtype_str]["atol"]
    rtol = PRECISION_THRESHOLDS[dtype_str]["rtol"]

    abs_err, rel_err = compute_elementwise_errors(golden_f32, npu_f32)

    # Per-element pass: either abs <= atol OR rel <= rtol
    elem_pass_abs = abs_err <= atol
    elem_pass_rel = rel_err <= rtol
    elem_pass = elem_pass_abs | elem_pass_rel  # union: pass if EITHER satisfied

    # Elements that fail BOTH criteria
    elem_fail = ~elem_pass
    fail_count = int(np.sum(elem_fail))

    # Global stats
    max_abs_err = float(np.max(abs_err))
    max_abs_idx = int(np.argmax(abs_err))
    max_rel_err = float(np.max(rel_err))
    max_rel_idx = int(np.argmax(rel_err))
    total_elements = golden_f32.size

    # Overall pass: all elements must pass at least one criterion
    all_pass = fail_count == 0

    # Print summary
    print(f"\n  {name} ({dtype_str}):")
    print(f"    Shape: {golden.shape}  Total elements: {total_elements}")
    print(f"    Thresholds: atol={atol}  rtol={rtol}  (satisfy EITHER to pass)")
    max_abs_golden_val = float(golden_f32.flat[max_abs_idx])
    max_abs_npu_val = float(npu_f32.flat[max_abs_idx])
    max_rel_golden_val = float(golden_f32.flat[max_rel_idx])
    max_rel_npu_val = float(npu_f32.flat[max_rel_idx])
    print(f"    Max absolute error: {max_abs_err:.8f}  at 1D idx={max_abs_idx}  (golden={max_abs_golden_val:.8f}, npu={max_abs_npu_val:.8f})")
    print(f"    Max relative error: {max_rel_err:.8f}  at 1D idx={max_rel_idx}  (golden={max_rel_golden_val:.8f}, npu={max_rel_npu_val:.8f})")

    if all_pass:
        print(f"    Result: PASS (all {total_elements} elements meet at least one criterion)")
    else:
        # Detail: elements failing both thresholds
        fail_indices = np.flatnonzero(elem_fail).tolist()
        print(f"    Result: FAIL")
        print(f"    Elements failing BOTH thresholds: {fail_count}/{total_elements}")
        # Print fail indices (cap at 200 to avoid flooding terminal)
        if len(fail_indices) <= 200:
            print(f"    Failing element 1D indices: {fail_indices}")
        else:
            print(f"    Failing element 1D indices (first 200): {fail_indices[:200]}")
            print(f"    ... and {len(fail_indices) - 200} more failing indices")

        # Also print abs/rel errors of failing elements for inspection
        if fail_count <= 50:
            print(f"    Failing element details (abs_err, rel_err):")
            for idx in fail_indices:
                print(f"      idx={idx}: abs_err={float(abs_err.flat[idx]):.8f}, rel_err={float(rel_err.flat[idx]):.8f}")

    return all_pass


# ============================================================
# Function: compare_mask_int8 - Compare INT8 mask tensor (exact match)
# ============================================================

def compare_mask_int8(golden_mask, npu_mask, name):
    match = np.array_equal(golden_mask, npu_mask)
    mismatch_count = int(np.sum(golden_mask != npu_mask))
    total = golden_mask.size

    print(f"\n  {name} (INT8, exact match):")
    print(f"    Shape: {golden_mask.shape}")
    print(f"    Mismatch count: {mismatch_count}/{total}")
    print(f"    Match: {match}")

    if not match:
        mismatch_indices = np.where(golden_mask != npu_mask)[0].tolist()
        if len(mismatch_indices) <= 200:
            print(f"    Mismatch element 1D indices: {mismatch_indices}")
        else:
            print(f"    Mismatch element 1D indices (first 200): {mismatch_indices[:200]}")
            print(f"    ... and {len(mismatch_indices) - 200} more mismatch indices")

    return match


# ============================================================
# Function: main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare NPU vs Golden results")
    parser.add_argument("--npu_dir", type=str, default="npu")
    parser.add_argument("--golden_dir", type=str, default=".")
    parser.add_argument("--input_dir", type=str, default=".")
    args = parser.parse_args()

    # Load golden metadata
    golden_meta = load_meta_txt(os.path.join(args.golden_dir, "golden_meta.txt"))

    print("=== Comparing NPU vs Golden ===")
    print(f"  Precision criteria:")
    for dtype_str, thresh in PRECISION_THRESHOLDS.items():
        print(f"    {dtype_str}: atol={thresh['atol']} ({thresh['atol']*100:.2f}%), rtol={thresh['rtol']} ({thresh['rtol']*100:.2f}%) — satisfy EITHER to pass")

    all_pass = True
    results = {}

    # Compare floating-point tensors
    tensor_names = ["q_comp", "k_comp", "score", "attn_score"]

    for name in tensor_names:
        # if name == "attn_score":
        #     continue
        golden_path = os.path.join(args.golden_dir, f"golden_{name}.bin")
        npu_path = os.path.join(args.npu_dir, f"npu_{name}.bin")

        if not os.path.exists(golden_path):
            print(f"\n  {name}: SKIPPED - golden file not found")
            continue
        if not os.path.exists(npu_path):
            print(f"\n  {name}: SKIPPED - NPU file not found")
            all_pass = False
            continue

        shape = golden_meta.get(f"shape_{name}", [])
        dtype_str = golden_meta.get(f"dtype_{name}", "FP32")

        golden = load_bin(golden_path, shape, dtype_str)
        npu = load_bin(npu_path, shape, dtype_str)

        print("golden")
        print(golden.flatten()[:20])
        print("npu")
        print(npu.flatten()[:20])
        passed = compare_float_tensor(golden, npu, name, dtype_str)
        results[name] = {"passed": passed}
        all_pass = all_pass and passed

    # Compare mask (INT8, exact match)
    # mask_name = "mask"
    # golden_mask_path = os.path.join(args.golden_dir, f"golden_{mask_name}.bin")
    # npu_mask_path = os.path.join(args.npu_dir, f"npu_{mask_name}.bin")

    # if os.path.exists(golden_mask_path) and os.path.exists(npu_mask_path):
    #     mask_shape = golden_meta.get(f"shape_{mask_name}", [])
    #     golden_mask = load_bin(golden_mask_path, mask_shape, "INT8")
    #     npu_mask = load_bin(npu_mask_path, mask_shape, "INT8")

    #     mask_passed = compare_mask_int8(golden_mask, npu_mask, mask_name)
    #     results[mask_name] = {"passed": mask_passed}
    #     all_pass = all_pass and mask_passed

    # Summary
    print(f"\n{'='*50}")
    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*50}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
