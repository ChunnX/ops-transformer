# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""带宽数据采集驱动：按固定 (1 预热 + N 测量) 模式顺序跑一组代表性 case，
供单次 msprof 采集。op_summary 内 task 无 shape 字段，只能按执行顺序映射，
故本驱动同时打印 manifest（顺序、shape、dtype、iters、字节数），供解析脚本对齐。

用法: msprof --output=<dir> --task-time=on --ai-core=on --ascendcl=on python3 prof_pta_bw.py [iters]
默认 iters=5（每个 case 额外 1 次预热，共 iters+1 个 task）。
"""

import sys
import torch
import torch_npu  # noqa: F401
from pta_nddma import nddma_transpose

DTYPE_MAP = {0: torch.float16, 1: torch.float32, 2: torch.int8}
ELEM = {0: 2, 1: 4, 2: 1}
DT_NAME = {0: "fp16", 1: "fp32", 2: "int8"}
D0 = 56

# (d1, d2, d3, dt) 代表性集合：square/wide/long/unaligned/transposed × fp16/fp32/int8。
CASES = [
    (4, 256, 256, 0),    # fp16 square sweet
    (16, 256, 128, 0),   # fp16 wide
    (64, 512, 32, 0),    # fp16 many-batch small-d3
    (64, 1024, 16, 0),   # fp16 extreme long d2
    (4, 256, 255, 0),    # fp16 unaligned d3
    (16, 128, 256, 0),   # fp16 transposed direction
    (32, 256, 64, 1),    # fp32 block=16384
    (64, 512, 32, 1),    # fp32 block=16384
    (64, 1024, 16, 1),   # fp32 block=16384
    (32, 64, 256, 1),    # fp32 transposed
    (4, 256, 256, 2),    # int8 square
    (64, 512, 32, 2),    # int8 many-batch
    (4, 256, 255, 2),    # int8 unaligned
]


def make_input(d1, d2, d3, dt):
    total = D0 * d1 * d2 * d3
    mod = 127 if dt == 2 else 2048
    flat = (torch.arange(1, total + 1, dtype=torch.float32) % mod)
    return flat.reshape(D0, d1, d2, d3).to(DTYPE_MAP[dt])


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
    print(f"# MANIFEST order={len(CASES)} iters={iters} (每 case 1 预热 + {iters} 测量)")
    print(f"# {'idx':>3} {'shape(d0,d1,d2,d3)':>22} {'dt':>5} {'numel':>12} {'RW_MB':>9}")
    for i, (d1, d2, d3, dt) in enumerate(CASES):
        numel = D0 * d1 * d2 * d3
        rw_mb = 2 * numel * ELEM[dt] / 1e6
        print(f"# {i:>3} {f'({D0},{d1},{d2},{d3})':>22} {DT_NAME[dt]:>5} {numel:>12} {rw_mb:>9.2f}")
    for d1, d2, d3, dt in CASES:
        x = make_input(d1, d2, d3, dt).npu()
        _ = nddma_transpose(x, is_mte3=1)          # 预热
        for _ in range(iters):
            _ = nddma_transpose(x, is_mte3=1)
    print("done")
