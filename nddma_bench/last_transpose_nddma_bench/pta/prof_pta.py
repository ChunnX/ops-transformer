# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""msprof 性能采集驱动：通过 PTA 接口跑单一代表性 case，供 msprof 采集 kernel task 时间。

用法: msprof --output=<dir> python3 prof_pta.py [dim0 dim1 dim2 dim3 dtype iters]
默认: 56 4 256 256 0（fp16 256x256 甜点场景），iters=5。
"""

import sys
import torch
import torch_npu  # noqa: F401  注册 NPU 后端
from pta_nddma import last_transpose

DTYPE_MAP = {0: torch.float16, 1: torch.float32, 2: torch.int8}


def make_input(d0, d1, d2, d3, dt):
    total = d0 * d1 * d2 * d3
    mod = 127 if dt == 2 else 2048
    flat = (torch.arange(1, total + 1, dtype=torch.float32) % mod)
    return flat.reshape(d0, d1, d2, d3).to(DTYPE_MAP[dt])


if __name__ == "__main__":
    argv = sys.argv[1:]
    d0, d1, d2, d3, dt = (int(a) for a in (argv + ["56", "4", "256", "256", "0"])[:5])
    iters = int(argv[5]) if len(argv) >= 6 else 5

    x = make_input(d0, d1, d2, d3, dt).npu()   # 上设备一次，循环内零拷贝直调
    # 预热 1 次（触发 JIT/首个 kernel 编译发射路径）
    _ = last_transpose(x, is_mte3=1)
    for _ in range(iters):
        _ = last_transpose(x, is_mte3=1)
    print(f"done: ({d0},{d1},{d2},{d3}) {DTYPE_MAP[dt]} x{iters} iters")
