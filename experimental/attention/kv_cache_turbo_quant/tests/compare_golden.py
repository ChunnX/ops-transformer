#!/usr/bin/env python3
"""Compare a kernel dump with torch_npu golden.py. Any mismatch is a failure."""

import sys
from pathlib import Path

import numpy as np

import golden

CASES = [
    ("gqa_decode_b1", 3),
    ("gqa_decode_b64", 3),
    ("gqa_prefill_t2048", 3),
    ("gqa_mse2_t2048", 2),
    ("gqa_mse4_t2048", 4),
    ("gqa_heads4_t32", 3),
    ("gqa_heads32_t8", 3),
]


def bits_of(case_dir: Path, default: int) -> int:
    path = case_dir / "bits.txt"
    if not path.exists():
        return default
    raw = path.read_bytes()
    if len(raw) >= 4:
        return int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    return default


def check_case(dump: Path, name: str, bits: int) -> int:
    case_dir = dump / name
    x = np.fromfile(case_dir / "x.bin", dtype=np.float32).reshape(-1, 128)
    rotation = np.fromfile(case_dir / "H.bin", dtype=np.float32).reshape(128, 128)
    qjl_matrix = np.fromfile(case_dir / "S.bin", dtype=np.float32).reshape(128, 128)
    idx = np.fromfile(case_dir / "idx.bin", dtype=np.uint8)
    qjl = np.fromfile(case_dir / "qjl.bin", dtype=np.uint8)
    norm = np.fromfile(case_dir / "norm.bin", dtype=np.uint16)
    gamma = np.fromfile(case_dir / "gamma.bin", dtype=np.uint16)
    g_idx, g_qjl, g_norm, g_gamma = golden.calc_expect_func(x, rotation, qjl_matrix, bits_of(case_dir, bits))
    misses = {
        "idx": int((idx != np.asarray(g_idx).reshape(-1)).sum()),
        "qjl": int((qjl != np.asarray(g_qjl).reshape(-1)).sum()),
        "norm": int((norm != np.asarray(g_norm).view(np.uint16).reshape(-1)).sum()),
        "gamma": int((gamma != np.asarray(g_gamma).view(np.uint16).reshape(-1)).sum()),
    }
    total = sum(misses.values())
    print(
        f"[golden {name}] idx_miss={misses['idx']} qjl_miss={misses['qjl']} "
        f"norm_miss={misses['norm']} gamma_miss={misses['gamma']} "
        f"{'PASS' if total == 0 else 'FAIL'}"
    )
    return total


def main() -> int:
    dump = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/kvtq_dump")
    failed = 0
    for name, bits in CASES:
        failed += check_case(dump, name, bits)
    print("GOLDEN PASSED" if failed == 0 else "GOLDEN FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
