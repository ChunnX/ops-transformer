# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""通过 PTA 接口验证 LastTranspose NDDMA kernel 的精度。

参考值：torch 原生 x.transpose(-1,-2) == permute[0,1,3,2]。
纯搬运，逐元素 bitwise 相等。
用法:
  python3 validate_pta.py [dim0 dim1 dim2 dim3 dtype]  # 单 case 或默认 6 个
  python3 validate_pta.py --all                        # transpose_testcases.md 全量 L_0~39 (dim0=56) + B8
"""

import sys
import torch
import torch_npu  # noqa: F401  注册 NPU 后端
from pta_nddma import nddma_transpose

DTYPE_MAP = {0: torch.float16, 1: torch.float32, 2: torch.int8}
DT_NAME = {0: "fp16", 1: "fp32", 2: "int8"}

# transpose_testcases.md test_L_0~19 的 20 组 (d1,d2,d3)，fp16/fp32/int8 复用同一组 shape。
BASE_SHAPES = [
    (4, 256, 256), (16, 256, 128), (32, 256, 64), (64, 512, 32), (64, 1024, 16),
    (4, 256, 255), (16, 256, 127), (32, 256, 63), (64, 512, 31), (64, 1024, 15),
    (4, 256, 256), (16, 128, 256), (32, 64, 256), (64, 32, 512), (64, 16, 1024),
    (4, 255, 256), (16, 127, 256), (32, 63, 256), (64, 31, 512), (64, 15, 1024),
]

# DB 双缓冲单块容量上限（elems）：halfCap = ubCapElems/2。block=d2*d3 超过则跳过。
HALF_CAP = {0: 65536, 1: 16384, 2: 65536}


def make_input(d0, d1, d2, d3, dt):
    total = d0 * d1 * d2 * d3
    mod = 127 if dt == 2 else 2048
    flat = (torch.arange(1, total + 1, dtype=torch.float32) % mod)
    return flat.reshape(d0, d1, d2, d3).to(DTYPE_MAP[dt])


def run_case(d0, d1, d2, d3, dt):
    x = make_input(d0, d1, d2, d3, dt).npu()      # NPU device 张量
    y = nddma_transpose(x, is_mte3=1)             # PTA kernel 输出（原生 device 路径）
    ref = x.transpose(-1, -2).contiguous()        # torch 参考（NPU 上）
    ok = torch.equal(y.cpu(), ref.cpu())
    tag = f"({d0},{d1},{d2},{d3}) {DTYPE_MAP[dt]}"
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:32s} shape_out={tuple(y.shape)}")
    if not ok:
        diff = (y.cpu().to(torch.float32) != ref.cpu().to(torch.float32))
        print(f"    mismatched elems: {int(diff.sum())}/{y.numel()}")
    return ok


def run_all():
    d0 = 56
    all_ok = True
    for dt in (0, 1, 2):
        cap = HALF_CAP[dt]
        n_pass = n_run = n_skip = 0
        print(f"\n=== {DT_NAME[dt]} (dim0={d0}, halfCap={cap}) ===")
        for d1, d2, d3 in BASE_SHAPES:
            block = d2 * d3
            if block > cap:
                print(f"[SKIP] ({d0},{d1},{d2},{d3}) block={block}>{cap} 超 DB 单块容量")
                n_skip += 1
                continue
            ok = run_case(d0, d1, d2, d3, dt)
            n_run += 1
            n_pass += int(ok)
            all_ok &= ok
        print(f"--- {DT_NAME[dt]}: PASS {n_pass}/{n_run} run, {n_skip} skipped ---")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":
        all_ok = run_all()
        print("\ntest pass!" if all_ok else "\ntest FAILED")
        sys.exit(0 if all_ok else 1)
    if len(sys.argv) >= 6:
        cases = [tuple(int(a) for a in sys.argv[1:6])]
    else:
        cases = [
            (56, 4, 256, 256, 0),   # L_0 fp16 square
            (56, 16, 256, 128, 0),  # L_1 fp16 wide
            (56, 4, 256, 255, 0),   # L_5 fp16 unaligned d3
            (56, 32, 256, 64, 1),   # L_22 fp32
            (56, 4, 256, 256, 2),   # B8 square
            (56, 16, 256, 127, 2),  # B8 unaligned
        ]
    all_ok = all(run_case(*c) for c in cases)
    print("test pass!" if all_ok else "test FAILED")
    sys.exit(0 if all_ok else 1)
