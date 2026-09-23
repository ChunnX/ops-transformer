"""核对任务书 5 个 case 的打包长度，以及零向量在 3-bit 下的码本下标。

不使用任务书缺失的 rotation_matrix.bin / qjl_matrix.bin，输入用固定种子生成。
本脚本只证明形状和 golden.py 的比较方向，不代替逐字节验收。
"""

from __future__ import annotations

import sys

import numpy as np

from cpu_golden import encode, pack_bits

CASES = [
    ("gqa_decode_b1", 1, 8, 128, 3, 48),
    ("gqa_decode_b64", 64, 8, 128, 3, 48),
    ("gqa_prefill_t2048", 2048, 8, 128, 3, 48),
    ("gqa_mse2_t2048", 2048, 8, 128, 2, 32),
    ("gqa_mse4_t2048", 2048, 8, 128, 4, 64),
]


def main() -> int:
    passed = 0
    rng = np.random.default_rng(20260923)
    rotation = rng.standard_normal((128, 128), dtype=np.float32)
    qjl = rng.standard_normal((128, 128), dtype=np.float32)

    word = pack_bits(np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8), 3)
    assert word.tolist() == [[1, 0, 0]], word.tolist()
    signs = pack_bits(np.ones((1, 8), dtype=np.uint8), 1)
    assert signs.tolist() == [[255]], signs.tolist()
    passed += 2
    print("PASS pack_bits 已知码型 2/2")

    for name, tokens, heads, dim, bits, idx_last in CASES:
        kv = rng.uniform(-2, 2, size=(tokens, heads, dim)).astype(np.float32)
        idx, qjl_out, norm, gamma = encode(kv, rotation, qjl, bits)
        expect = {
            "quant_idx": (tokens, heads, idx_last),
            "quant_qjl": (tokens, heads, 16),
            "quant_norm": (tokens, heads),
            "quant_gamma": (tokens, heads),
        }
        got = {
            "quant_idx": idx.shape,
            "quant_qjl": qjl_out.shape,
            "quant_norm": norm.shape,
            "quant_gamma": gamma.shape,
        }
        if got != expect:
            print(f"FAIL {name}: {got} != {expect}")
            return 1
        if not np.isfinite(norm).all() or not np.isfinite(gamma).all():
            print(f"FAIL {name}: norm/gamma 出现非有限值")
            return 1
        passed += 1
        print(f"PASS {name} idx{idx.shape} qjl{qjl_out.shape} norm{norm.shape}")

    zero = np.zeros((1, 1, 128), dtype=np.float32)
    eye = np.eye(128, dtype=np.float32)
    idx, qjl_out, norm, gamma = encode(zero, eye, eye, 3)
    if float(norm.reshape(-1)[0]) != 0.0 or float(gamma.reshape(-1)[0]) != 0.0:
        print("FAIL zero norm/gamma", norm, gamma)
        return 1
    # 每 8 个下标 3（0b011）打进 3 字节：word = 3 * (1+8+64+512+4096+32768+262144+2097152)
    # 3 << (3*lane) 求和
    word = 0
    for lane in range(8):
        word |= 3 << (3 * lane)
    expect_bytes = [(word >> (8 * b)) & 0xFF for b in range(3)]
    if idx.reshape(-1)[:3].tolist() != expect_bytes:
        print("FAIL zero-vector 3-bit 下标", idx.reshape(-1)[:3].tolist(), expect_bytes)
        return 1
    passed += 1
    print(f"PASS zero_vector mse3 下标字节 {expect_bytes}")

    print(f"ALL PASS {passed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
