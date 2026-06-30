#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_dispatch.py —— DispatchTeardown(mxfp8) npu 输出 vs golden 比对。
用法: compare_dispatch.py <bs> <topk> <h> <E> <R> <datadir>
expand_x(fp8)/expand_scale(e8m0) 是纯字节重排 → 按字节(uint8) bit 精确；
int 输出(send_counts/expand_idx/expert_token_nums)要求完全相等。
"""
import sys, os
import numpy as np


def cmp_int(name, npu, gold):
    ok = np.array_equal(npu, gold)
    if ok:
        print(f"  [PASS] {name} exact-equal (n={npu.size})")
    else:
        ndiff = int((npu != gold).sum())
        print(f"  [FAIL] {name} differ {ndiff}/{npu.size}")
        print(f"         npu ={npu.reshape(-1)[:32].tolist()}")
        print(f"         gold={gold.reshape(-1)[:32].tolist()}")
    return ok


def main():
    bs, topk, h, E, R = (int(sys.argv[i]) for i in range(1, 6))
    d = sys.argv[6]
    GE = R * E
    minKE = min(topk, E)
    A = R * bs * minKE
    scaleH = h // 32

    def rd(name, dtype, shape):
        return np.fromfile(os.path.join(d, name), dtype=dtype).reshape(shape)

    allok = True
    allok &= cmp_int("expand_x(fp8 bytes)",
                     rd("expand_x.npu.bin", np.uint8, (A, h)),
                     rd("expand_x.golden.bin", np.uint8, (A, h)))
    allok &= cmp_int("expand_scale(e8m0 bytes)",
                     rd("expand_scale.npu.bin", np.uint8, (A, scaleH)),
                     rd("expand_scale.golden.bin", np.uint8, (A, scaleH)))
    allok &= cmp_int("send_counts",
                     rd("send_counts.npu.bin", np.int32, (GE,)),
                     rd("send_counts.golden.bin", np.int32, (GE,)))
    allok &= cmp_int("expand_idx",
                     rd("expand_idx.npu.bin", np.int32, (bs, topk)),
                     rd("expand_idx.golden.bin", np.int32, (bs, topk)))
    allok &= cmp_int("expert_token_nums",
                     rd("expert_token_nums.npu.bin", np.int32, (E,)),
                     rd("expert_token_nums.golden.bin", np.int32, (E,)))

    print("=== DISPATCH_TEARDOWN " + ("PASS" if allok else "FAIL") + " ===")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
