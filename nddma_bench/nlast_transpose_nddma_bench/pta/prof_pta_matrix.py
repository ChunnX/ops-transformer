# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""NLastTranspose [0,1,3,2,4] 全用例时延/带宽采集驱动。

分核维 dim0=56（= 物理 vector core 数，单波均衡；偏离 transpose_testcases.md
原始 dim0=64 规格，为对齐核数的 bench 变体：dim0=56 / blockNum=56 → per=1,rem=0，
每核正好 1 个 i0，消除不整除导致的第二波拖尾）。d1/d2/d3/d4 仍取 test_NL_0~47。

单次 msprof 会话内按固定 (1 预热 + N 测量) 顺序跑当前 .so 变体（DB 或 SB）下所有
可容纳的用例。op_summary task 无 shape，按执行顺序对齐本驱动打印的 RUN manifest。

用法: msprof ... python3 prof_pta_matrix.py <db|sb> <isMte3:0|1> [iters]
  db/sb : 与当前 build/libnddma_transpose_launch.so 编译的 USE_PINGPONG 一致（决定容量跳过）
  isMte3: 1=读写混合(MTE3 写回) 0=纯读(去 MTE3)
"""

import os
import sys
import torch
import torch_npu  # noqa: F401
from pta_nddma import nddma_transpose

DTYPE_MAP = {0: torch.float16, 1: torch.float32, 2: torch.int8}
ELEM = {0: 2, 1: 4, 2: 1}
DT_NAME = {0: "fp16", 1: "fp32", 2: "int8"}
# 分核维 dim0=56（= 物理 vector core 数，单波均衡 per=1/rem=0），固定不变。
D0 = 56

# (label, d2, d3, d4, dt, dim1) —— transpose_testcases.md NLastTranspose test_NL_0~47 的
# d2/d3/d4/dtype，外加 per-case dim1（round_004 引入）。
#
# dim1 是 kernel 核内 UB for 循环维（每核循环 gCount = i0Count*dim1 次，dim0=56 时 i0Count=1，
# 故每核循环 dim1 次），每次一条 NDDMA 搬一个 blockElems=d2*d3*d4 块。增大 dim1 = 增加核内串行
# NDDMA 块数 = 线性增大数据量与时延，**不改单块容量**（SKIP 只看 block=d2*d3*d4，与 dim1 无关）。
#
# round_004 规则：对 round_003 (dim1=8) 时**所有可运行配置里最快(min)时延 <50us** 的用例，按
# `dim1 = 8*ceil(58/min_us)` 向上放大到整数，使其所有配置时延都 >50us（瞄准 ~55~70us 余量）；
# min 已 ≥50us 的用例保持 dim1=8。dim1 逐用例不同。经实测校验（见 round_004 归档）。
CASES = [
    # FP16 A大B小 对齐 / 非对齐
    ("NL_0", 128, 8, 16, 0, 32), ("NL_1", 128, 8, 32, 0, 24), ("NL_2", 64, 8, 64, 0, 24),
    ("NL_3", 64, 8, 128, 0, 8), ("NL_4", 32, 8, 256, 0, 8), ("NL_5", 16, 8, 512, 0, 8),
    ("NL_6", 128, 8, 15, 0, 32), ("NL_7", 128, 8, 31, 0, 16), ("NL_8", 64, 8, 63, 0, 16),
    ("NL_9", 64, 8, 127, 0, 8), ("NL_10", 32, 8, 255, 0, 8), ("NL_11", 16, 8, 511, 0, 8),
    # FP16 A小B大 对齐 / 非对齐
    ("NL_12", 8, 128, 16, 0, 24), ("NL_13", 8, 128, 32, 0, 24), ("NL_14", 8, 64, 64, 0, 24),
    ("NL_15", 8, 64, 128, 0, 8), ("NL_16", 8, 32, 256, 0, 8), ("NL_17", 8, 16, 512, 0, 8),
    ("NL_18", 8, 128, 15, 0, 32), ("NL_19", 8, 128, 31, 0, 24), ("NL_20", 8, 64, 63, 0, 24),
    ("NL_21", 8, 64, 127, 0, 16), ("NL_22", 8, 32, 255, 0, 8), ("NL_23", 8, 16, 511, 0, 8),
    # FP32 A大B小 对齐 / 非对齐
    ("NL_24", 128, 8, 16, 1, 24), ("NL_25", 128, 8, 32, 1, 24), ("NL_26", 64, 8, 64, 1, 24),
    ("NL_27", 64, 8, 128, 1, 8), ("NL_28", 32, 8, 256, 1, 8), ("NL_29", 16, 8, 512, 1, 8),
    ("NL_30", 128, 8, 15, 1, 16), ("NL_31", 128, 8, 31, 1, 8), ("NL_32", 64, 8, 63, 1, 16),
    ("NL_33", 64, 8, 127, 1, 8), ("NL_34", 32, 8, 255, 1, 8), ("NL_35", 16, 8, 511, 1, 8),
    # FP32 A小B大 对齐 / 非对齐
    ("NL_36", 8, 128, 16, 1, 16), ("NL_37", 8, 128, 32, 1, 24), ("NL_38", 8, 64, 64, 1, 24),
    ("NL_39", 8, 64, 128, 1, 8), ("NL_40", 8, 32, 256, 1, 8), ("NL_41", 8, 16, 512, 1, 8),
    ("NL_42", 8, 128, 15, 1, 32), ("NL_43", 8, 128, 31, 1, 16), ("NL_44", 8, 64, 63, 1, 16),
    ("NL_45", 8, 64, 127, 1, 8), ("NL_46", 8, 32, 255, 1, 8), ("NL_47", 8, 16, 511, 1, 8),
    # B8 (int8)：对标 FP16 NL_0~23 的 d2/d3/d4/dim1 shape 集，dtype=2（设计 §1.2.1）。
    # B8 elemSize=1，带宽 R=numel×1；elem 容量=FP16(131072)，RUN/SKIP 与 FP16 逐例一致。
    ("NL_B8_0", 128, 8, 16, 2, 32), ("NL_B8_1", 128, 8, 32, 2, 24), ("NL_B8_2", 64, 8, 64, 2, 24),
    ("NL_B8_3", 64, 8, 128, 2, 8), ("NL_B8_4", 32, 8, 256, 2, 8), ("NL_B8_5", 16, 8, 512, 2, 8),
    ("NL_B8_6", 128, 8, 15, 2, 32), ("NL_B8_7", 128, 8, 31, 2, 16), ("NL_B8_8", 64, 8, 63, 2, 16),
    ("NL_B8_9", 64, 8, 127, 2, 8), ("NL_B8_10", 32, 8, 255, 2, 8), ("NL_B8_11", 16, 8, 511, 2, 8),
    ("NL_B8_12", 8, 128, 16, 2, 24), ("NL_B8_13", 8, 128, 32, 2, 24), ("NL_B8_14", 8, 64, 64, 2, 24),
    ("NL_B8_15", 8, 64, 128, 2, 8), ("NL_B8_16", 8, 32, 256, 2, 8), ("NL_B8_17", 8, 16, 512, 2, 8),
    ("NL_B8_18", 8, 128, 15, 2, 32), ("NL_B8_19", 8, 128, 31, 2, 24), ("NL_B8_20", 8, 64, 63, 2, 24),
    ("NL_B8_21", 8, 64, 127, 2, 16), ("NL_B8_22", 8, 32, 255, 2, 8), ("NL_B8_23", 8, 16, 511, 2, 8),
]

# 单块容量上限（elems）。DB=每半区 ubCapElems/2；SB=整块 ubCapElems。block=d2*d3*d4。
CAP = {
    "db": {0: 65536, 1: 16384, 2: 65536},
    "sb": {0: 131072, 1: 32768, 2: 131072},
}


def make_input(d1, d2, d3, d4, dt):
    total = D0 * d1 * d2 * d3 * d4
    if dt == 2:  # int8: 全值域随机构造，纯搬运走 bitwise（带宽采集与数据分布无关）
        return torch.randint(-128, 128, (D0, d1, d2, d3, d4), dtype=torch.int8)
    flat = (torch.arange(1, total + 1, dtype=torch.float32) % 2048)
    return flat.reshape(D0, d1, d2, d3, d4).to(DTYPE_MAP[dt])


if __name__ == "__main__":
    mode = sys.argv[1]
    is_mte3 = int(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    cap = CAP[mode]

    # NDDMA_B8_ONLY=1 → 仅采 int8(dtype=2) 用例（B8 带宽矩阵专用）。
    cases = [c for c in CASES if c[4] == 2] if os.environ.get("NDDMA_B8_ONLY") == "1" else CASES

    # 容量 SKIP 只看 block=d2*d3*d4，与 dim1 无关 → 名单与 dim1=8 时完全一致。
    run = [c for c in cases if c[1] * c[2] * c[3] <= cap[c[4]]]
    skip = [c for c in cases if c[1] * c[2] * c[3] > cap[c[4]]]

    print(f"# MANIFEST mode={mode} isMte3={is_mte3} iters={iters} "
          f"run={len(run)} skip={len(skip)} (每 case 1 预热+{iters} 测量) "
          f"[per-case dim1, dim0={D0}]")
    print(f"# {'ord':>3} {'label':>6} {'shape(d0,d1,d2,d3,d4)':>26} {'dt':>5} "
          f"{'block':>7} {'numel':>12} {'R_MB':>8} {'RW_MB':>8}")
    for i, (lb, d2, d3, d4, dt, d1) in enumerate(run):
        numel = D0 * d1 * d2 * d3 * d4
        r_mb = numel * ELEM[dt] / 1e6
        print(f"# {i:>3} {lb:>6} {f'({D0},{d1},{d2},{d3},{d4})':>26} {DT_NAME[dt]:>5} "
              f"{d2*d3*d4:>7} {numel:>12} {r_mb:>8.2f} {2*r_mb:>8.2f}")
    for lb, d2, d3, d4, dt, d1 in skip:
        print(f"# SKIP {lb} ({D0},{d1},{d2},{d3},{d4}) {DT_NAME[dt]} "
              f"block={d2*d3*d4}>{cap[dt]}")

    for lb, d2, d3, d4, dt, d1 in run:
        x = make_input(d1, d2, d3, d4, dt).npu()
        _ = nddma_transpose(x, is_mte3=is_mte3)       # 预热
        for _ in range(iters):
            _ = nddma_transpose(x, is_mte3=is_mte3)
    print("done")
