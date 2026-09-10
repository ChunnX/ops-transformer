#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Analyze TTK npy dump for cla_gate_quant.

Example:
    python3 tests/assets/analyze_ttk_dump.py \
        /home/g00940185/ops-test-kit/ClaGateQuant_bf16_ocp_random
"""
import sys
from pathlib import Path

import numpy as np
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn


def load_tensor(path):
    """Load a TTK-dumped npy that may be stored as raw bytes (void dtype)."""
    raw = np.load(path, allow_pickle=False)
    return raw.tobytes(), raw.shape, raw.dtype


def to_typed(raw, dtype):
    return np.frombuffer(raw, dtype=dtype)


def compare_pair(prefix, idx, first_n=20):
    out_path = Path(f"{prefix}_dyn_output_{idx}.npy")
    gold_path = Path(f"{prefix}_golden_{idx}.npy")
    if not out_path.exists() or not gold_path.exists():
        print(f"[skip] missing output/golden for idx={idx}")
        return

    dtype = float8_e4m3fn if idx in (0, 2) else float8_e8m0
    out_raw, out_shape, out_dtype = load_tensor(out_path)
    gold_raw, gold_shape, gold_dtype = load_tensor(gold_path)

    out = to_typed(out_raw, dtype)
    gold = to_typed(gold_raw, dtype)
    # Golden npy keeps the logical shape; output npy may be flat.
    if len(gold_shape) > 1:
        gold = gold.reshape(gold_shape)
        out = out.reshape(gold_shape)
    else:
        # Fallback to known full shapes.
        shapes = [(1024, 24576), (1024, 384, 2), (1024, 24576), (16, 24576, 2)]
        gold = gold.reshape(shapes[idx])
        out = out.reshape(shapes[idx])

    if out.shape != gold.shape:
        print(f"[warn] shape mismatch idx={idx}: out={out.shape} gold={gold.shape}")

    mism = np.argwhere(out != gold)
    total = gold.size
    print(f"\n==== output {idx} ====")
    print(f"shape: {gold.shape}  dtype: {dtype.__name__}")
    print(f"mismatch: {mism.shape[0]} / {total}  ({mism.shape[0] / total * 100:.6f}%)")

    for row in mism[:first_n]:
        idx_tuple = tuple(row)
        print(f"Index: {idx_tuple}  Expected: {gold[idx_tuple]}  Actual: {out[idx_tuple]}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    prefix = sys.argv[1]
    first_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    for i in range(4):
        compare_pair(prefix, i, first_n)


if __name__ == "__main__":
    main()
