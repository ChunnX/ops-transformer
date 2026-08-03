# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""LastTranspose 全用例时延/带宽采集驱动（transpose_testcases.md test_L_0~39, dim0=56 单波）。

分核维 dim0=56（= 物理 vector core 数 Ascend950PR_9579.vector_core_cnt，单波均衡 per=1/rem=0，
消除 dim0=64 在 56 核上非整除 rem=8 的第二波拖尾；符合 DESIGN §0.4 用例设计准则 3）。

单次 msprof 会话内按固定 (1 预热 + N 测量) 顺序跑当前 .so 变体（DB 或 SB）下所有
可容纳的用例。op_summary task 无 shape，按执行顺序对齐本驱动打印的 RUN manifest。

per-case dim1（DESIGN §0.4 准则 1+2）：dim1 是 kernel 核内 UB for 循环维（每核循环 dim1 次
搬 dim1 个 block），放大 dim1 = 线性增大核内串行 NDDMA 块数与时延，**不改单块容量**
（SKIP 只看 block=d2*d3，与 dim1 无关）。对 dim0=56 下基线 min 时延 ≤40us 的用例按
`new_dim1 = base*ceil(target/min)`（target≈45~50us）放大到 >40us；≥40us 的保持基线 dim1。
kernel 无需改（运行期读 attr->dim1），仅改本驱动 per-case dim1。

用法: msprof ... python3 prof_pta_matrix.py <db|sb> <isMte3:0|1> [iters]
  db/sb : 与当前 build/liblast_transpose_launch.so 编译的 USE_PINGPONG 一致（决定容量跳过）
  isMte3: 1=读写混合(MTE3 写回) 0=纯读(去 MTE3)
"""

import os
import sys
import torch
import torch_npu  # noqa: F401
from pta_nddma import last_transpose

DTYPE_MAP = {0: torch.float16, 1: torch.float32, 2: torch.int8}
ELEM = {0: 2, 1: 4, 2: 1}
DT_NAME = {0: "fp16", 1: "fp32", 2: "int8"}
# 分核维 dim0=56（= 物理 vector core 数，单波均衡 per=1/rem=0），固定不变（DESIGN §0.4 准则 3）。
D0 = 56

# (label, d1, d2, d3, dt) —— transpose_testcases.md LastTranspose test_L_0~39 的 (d2,d3,dtype)，
# d1 为 per-case dim1（dim0=56 重采后按准则 1+2 逐用例放大，见上文 docstring）。
CASES = [
    # dim0=56 基线实测：d1=4 用例（L_0/5/10/15、L_B8_0/5/10/15）min 时延 40~52us（余量不足，
    # L_B8_0/10 db_r 仅 40.3/40.5us）→ 按准则 2 放大 d1 4→8（ceil(48/min)=2），拉到 ~80us 余量足；
    # d1≥16 的用例基线 min 均 >80us，保持原 d1。fp32 L_20/25/30/35 虽 SKIP 亦同步 d1=8 保持一致。
    ("L_0", 8, 256, 256, 0), ("L_1", 16, 256, 128, 0), ("L_2", 32, 256, 64, 0),
    ("L_3", 64, 512, 32, 0), ("L_4", 64, 1024, 16, 0), ("L_5", 8, 256, 255, 0),
    ("L_6", 16, 256, 127, 0), ("L_7", 32, 256, 63, 0), ("L_8", 64, 512, 31, 0),
    ("L_9", 64, 1024, 15, 0), ("L_10", 8, 256, 256, 0), ("L_11", 16, 128, 256, 0),
    ("L_12", 32, 64, 256, 0), ("L_13", 64, 32, 512, 0), ("L_14", 64, 16, 1024, 0),
    ("L_15", 8, 255, 256, 0), ("L_16", 16, 127, 256, 0), ("L_17", 32, 63, 256, 0),
    ("L_18", 64, 31, 512, 0), ("L_19", 64, 15, 1024, 0),
    ("L_20", 8, 256, 256, 1), ("L_21", 16, 256, 128, 1), ("L_22", 32, 256, 64, 1),
    ("L_23", 64, 512, 32, 1), ("L_24", 64, 1024, 16, 1), ("L_25", 8, 256, 255, 1),
    ("L_26", 16, 256, 127, 1), ("L_27", 32, 256, 63, 1), ("L_28", 64, 512, 31, 1),
    ("L_29", 64, 1024, 15, 1), ("L_30", 8, 256, 256, 1), ("L_31", 16, 128, 256, 1),
    ("L_32", 32, 64, 256, 1), ("L_33", 64, 32, 512, 1), ("L_34", 64, 16, 1024, 1),
    ("L_35", 8, 255, 256, 1), ("L_36", 16, 127, 256, 1), ("L_37", 32, 63, 256, 1),
    ("L_38", 64, 31, 512, 1), ("L_39", 64, 15, 1024, 1),
    # B8 (int8)：对标 FP16 L_0~19 的 (d2,d3) shape 集，dtype=2（设计 §1.1.1 L_B8_*）。
    # B8 elemSize=1，带宽 R=numel×1；elem 容量=FP16(131072)，block=d2*d3 判据同 → RUN/SKIP 与 FP16 逐例一致。
    ("L_B8_0", 8, 256, 256, 2), ("L_B8_1", 16, 256, 128, 2), ("L_B8_2", 32, 256, 64, 2),
    ("L_B8_3", 64, 512, 32, 2), ("L_B8_4", 64, 1024, 16, 2), ("L_B8_5", 8, 256, 255, 2),
    ("L_B8_6", 16, 256, 127, 2), ("L_B8_7", 32, 256, 63, 2), ("L_B8_8", 64, 512, 31, 2),
    ("L_B8_9", 64, 1024, 15, 2), ("L_B8_10", 8, 256, 256, 2), ("L_B8_11", 16, 128, 256, 2),
    ("L_B8_12", 32, 64, 256, 2), ("L_B8_13", 64, 32, 512, 2), ("L_B8_14", 64, 16, 1024, 2),
    ("L_B8_15", 8, 255, 256, 2), ("L_B8_16", 16, 127, 256, 2), ("L_B8_17", 32, 63, 256, 2),
    ("L_B8_18", 64, 31, 512, 2), ("L_B8_19", 64, 15, 1024, 2),
]

# 单块容量上限（elems）。DB=每半区 ubCapElems/2；SB=整块 ubCapElems。
# int8: UB_CAP_ELEMS_B8=131072 → DB halfCap=65536, SB=131072（与 FP16 elem 容量数值相同）。
CAP = {
    "db": {0: 65536, 1: 16384, 2: 65536},
    "sb": {0: 131072, 1: 32768, 2: 131072},
}


def make_input(d1, d2, d3, dt):
    total = D0 * d1 * d2 * d3
    if dt == 2:  # int8: 全值域随机构造，纯搬运走 bitwise（带宽采集与数据分布无关）
        return torch.randint(-128, 128, (D0, d1, d2, d3), dtype=torch.int8)
    flat = (torch.arange(1, total + 1, dtype=torch.float32) % 2048)
    return flat.reshape(D0, d1, d2, d3).to(DTYPE_MAP[dt])


if __name__ == "__main__":
    mode = sys.argv[1]
    is_mte3 = int(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    cap = CAP[mode]

    # NDDMA_B8_ONLY=1 → 仅采 int8(dtype=2) 用例（B8 带宽矩阵专用）。
    cases = [c for c in CASES if c[4] == 2] if os.environ.get("NDDMA_B8_ONLY") == "1" else CASES

    run = [(lb, d1, d2, d3, dt) for (lb, d1, d2, d3, dt) in cases if d2 * d3 <= cap[dt]]
    skip = [(lb, d1, d2, d3, dt) for (lb, d1, d2, d3, dt) in cases if d2 * d3 > cap[dt]]

    print(f"# MANIFEST mode={mode} isMte3={is_mte3} iters={iters} "
          f"run={len(run)} skip={len(skip)} (每 case 1 预热+{iters} 测量)")
    print(f"# {'ord':>3} {'label':>6} {'shape(d0,d1,d2,d3)':>22} {'dt':>5} "
          f"{'block':>7} {'numel':>11} {'R_MB':>8} {'RW_MB':>8}")
    for i, (lb, d1, d2, d3, dt) in enumerate(run):
        numel = D0 * d1 * d2 * d3
        r_mb = numel * ELEM[dt] / 1e6
        print(f"# {i:>3} {lb:>6} {f'({D0},{d1},{d2},{d3})':>22} {DT_NAME[dt]:>5} "
              f"{d2*d3:>7} {numel:>11} {r_mb:>8.2f} {2*r_mb:>8.2f}")
    for lb, d1, d2, d3, dt in skip:
        print(f"# SKIP {lb} ({D0},{d1},{d2},{d3}) {DT_NAME[dt]} block={d2*d3}>{cap[dt]}")

    for lb, d1, d2, d3, dt in run:
        x = make_input(d1, d2, d3, dt).npu()
        _ = last_transpose(x, is_mte3=is_mte3)       # 预热
        for _ in range(iters):
            _ = last_transpose(x, is_mte3=is_mte3)
    print("done")
