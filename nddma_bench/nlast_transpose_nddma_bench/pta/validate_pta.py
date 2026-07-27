# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""通过 PTA 接口验证 NLastTranspose [0,1,3,2,4] NDDMA kernel 的精度。

参考值：torch 原生 x.permute(0,1,3,2,4).contiguous()（交换中间两轴 dim2/dim3，保留内轴 dim4）。
纯搬运，逐元素 bitwise 相等。
用法:
  python3 validate_pta.py [dim0 dim1 dim2 dim3 dim4 dtype]  # 单 case 或默认若干
  python3 validate_pta.py --all                             # transpose_testcases.md 全量 NL_0~47
"""

import sys
import torch
import torch_npu  # noqa: F401  注册 NPU 后端
from pta_nddma import nlast_transpose

DTYPE_MAP = {0: torch.float16, 1: torch.float32, 2: torch.int8}
DT_NAME = {0: "fp16", 1: "fp32", 2: "int8"}

# transpose_testcases.md test_NL_0~47 的 dim2/dim3/dim4/dtype，permute [0,1,3,2,4]。
# 分核维 dim0=56（= 物理 vector core 数，与 bench 同 shape：dim0/blockNum=56/56 单波均衡，
# per=1/rem=0；偏离原始 dim0=64 规格，为对齐核数的 bench 变体），固定不变。
# per-case dim1（round_004 引入）与 prof_pta_matrix.CASES 同源，精度须同 shape 校验。
# 每项 (label, dim2, dim3, dim4, dtype, dim1)；dtype 0=fp16(NL_0~23) / 1=fp32(NL_24~47)。
D0 = 56
CASES = [
    # ---- FP16 A大B小 对齐 (NL_0~5) ----
    ("NL_0",  128,   8,  16, 0, 32), ("NL_1",  128,   8,  32, 0, 24),
    ("NL_2",   64,   8,  64, 0, 24), ("NL_3",   64,   8, 128, 0,  8),
    ("NL_4",   32,   8, 256, 0,  8), ("NL_5",   16,   8, 512, 0,  8),
    # ---- FP16 A大B小 非对齐 (NL_6~11) ----
    ("NL_6",  128,   8,  15, 0, 32), ("NL_7",  128,   8,  31, 0, 16),
    ("NL_8",   64,   8,  63, 0, 16), ("NL_9",   64,   8, 127, 0,  8),
    ("NL_10",  32,   8, 255, 0,  8), ("NL_11",  16,   8, 511, 0,  8),
    # ---- FP16 A小B大 对齐 (NL_12~17) ----
    ("NL_12",   8, 128,  16, 0, 24), ("NL_13",   8, 128,  32, 0, 24),
    ("NL_14",   8,  64,  64, 0, 24), ("NL_15",   8,  64, 128, 0,  8),
    ("NL_16",   8,  32, 256, 0,  8), ("NL_17",   8,  16, 512, 0,  8),
    # ---- FP16 A小B大 非对齐 (NL_18~23) ----
    ("NL_18",   8, 128,  15, 0, 32), ("NL_19",   8, 128,  31, 0, 24),
    ("NL_20",   8,  64,  63, 0, 24), ("NL_21",   8,  64, 127, 0, 16),
    ("NL_22",   8,  32, 255, 0,  8), ("NL_23",   8,  16, 511, 0,  8),
    # ---- FP32 A大B小 对齐 (NL_24~29) ----
    ("NL_24", 128,   8,  16, 1, 24), ("NL_25", 128,   8,  32, 1, 24),
    ("NL_26",  64,   8,  64, 1, 24), ("NL_27",  64,   8, 128, 1,  8),
    ("NL_28",  32,   8, 256, 1,  8), ("NL_29",  16,   8, 512, 1,  8),
    # ---- FP32 A大B小 非对齐 (NL_30~35) ----
    ("NL_30", 128,   8,  15, 1, 16), ("NL_31", 128,   8,  31, 1,  8),
    ("NL_32",  64,   8,  63, 1, 16), ("NL_33",  64,   8, 127, 1,  8),
    ("NL_34",  32,   8, 255, 1,  8), ("NL_35",  16,   8, 511, 1,  8),
    # ---- FP32 A小B大 对齐 (NL_36~41) ----
    ("NL_36",   8, 128,  16, 1, 16), ("NL_37",   8, 128,  32, 1, 24),
    ("NL_38",   8,  64,  64, 1, 24), ("NL_39",   8,  64, 128, 1,  8),
    ("NL_40",   8,  32, 256, 1,  8), ("NL_41",   8,  16, 512, 1,  8),
    # ---- FP32 A小B大 非对齐 (NL_42~47) ----
    ("NL_42",   8, 128,  15, 1, 32), ("NL_43",   8, 128,  31, 1, 16),
    ("NL_44",   8,  64,  63, 1, 16), ("NL_45",   8,  64, 127, 1,  8),
    ("NL_46",   8,  32, 255, 1,  8), ("NL_47",   8,  16, 511, 1,  8),
    # ---- B8 (int8) 用例集：对标 FP16 NL_0~23 的 d2/d3/d4/dim1 shape 集，dtype=2 (设计 §1.2.1)。
    #      B8 elem 容量 = FP16 (131072)，SKIP 判据 (block=d2*d3*d4) 与 FP16 逐例一致；
    #      golden 用 torch.int8（int8 无浮点舍入，走 bitwise 全等）。 ----
    # B8 A大B小 对齐 (对标 NL_0~5)
    ("NL_B8_0",  128,   8,  16, 2, 32), ("NL_B8_1",  128,   8,  32, 2, 24),
    ("NL_B8_2",   64,   8,  64, 2, 24), ("NL_B8_3",   64,   8, 128, 2,  8),
    ("NL_B8_4",   32,   8, 256, 2,  8), ("NL_B8_5",   16,   8, 512, 2,  8),
    # B8 A大B小 非对齐 (对标 NL_6~11)
    ("NL_B8_6",  128,   8,  15, 2, 32), ("NL_B8_7",  128,   8,  31, 2, 16),
    ("NL_B8_8",   64,   8,  63, 2, 16), ("NL_B8_9",   64,   8, 127, 2,  8),
    ("NL_B8_10",  32,   8, 255, 2,  8), ("NL_B8_11",  16,   8, 511, 2,  8),
    # B8 A小B大 对齐 (对标 NL_12~17)
    ("NL_B8_12",   8, 128,  16, 2, 24), ("NL_B8_13",   8, 128,  32, 2, 24),
    ("NL_B8_14",   8,  64,  64, 2, 24), ("NL_B8_15",   8,  64, 128, 2,  8),
    ("NL_B8_16",   8,  32, 256, 2,  8), ("NL_B8_17",   8,  16, 512, 2,  8),
    # B8 A小B大 非对齐 (对标 NL_18~23)
    ("NL_B8_18",   8, 128,  15, 2, 32), ("NL_B8_19",   8, 128,  31, 2, 24),
    ("NL_B8_20",   8,  64,  63, 2, 24), ("NL_B8_21",   8,  64, 127, 2, 16),
    ("NL_B8_22",   8,  32, 255, 2,  8), ("NL_B8_23",   8,  16, 511, 2,  8),
]

# 单块容量上限（elems）：blockElems = dim2*dim3*dim4 超过则跳过（RUN/SKIP 与设计 §2.2 同源）。
# 必须与当前 build/libnlast_transpose_launch.so 的 USE_PINGPONG 编译变体一致：
#   DB（USE_PINGPONG=true，默认）→ halfCap=ubCapElems/2；SB（false）→ 整块 ubCapElems。
# 通过环境变量 NDDMA_MODE=db|sb 选择（默认 db，对应默认 .so 构建）。
import os as _os
_MODE = _os.environ.get("NDDMA_MODE", "db").lower()
_CAP = {
    "db": {0: 65536, 1: 16384, 2: 65536},
    "sb": {0: 131072, 1: 32768, 2: 131072},
}
CAP_SB = _CAP.get(_MODE, _CAP["db"])


def make_input(d0, d1, d2, d3, d4, dt):
    total = d0 * d1 * d2 * d3 * d4
    if dt == 2:  # int8: 全值域 [-128,127) 随机构造，纯搬运走 bitwise 全等（设计 §3.2）
        return torch.randint(-128, 128, (d0, d1, d2, d3, d4), dtype=torch.int8)
    flat = (torch.arange(1, total + 1, dtype=torch.float32) % 2048)
    return flat.reshape(d0, d1, d2, d3, d4).to(DTYPE_MAP[dt])


def run_case(label, d2, d3, d4, dt, d1):
    x = make_input(D0, d1, d2, d3, d4, dt).npu()          # NPU device 5D 张量
    y = nlast_transpose(x, is_mte3=1)                     # PTA kernel 输出（原生 device 路径）
    ref = x.permute(0, 1, 3, 2, 4).contiguous()          # torch 参考（NPU 上）
    ok = torch.equal(y.cpu(), ref.cpu())
    tag = f"{label} ({D0},{d1},{d2},{d3},{d4}) {DT_NAME[dt]}"
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:36s} shape_out={tuple(y.shape)}")
    if not ok:
        diff = (y.cpu().to(torch.float32) != ref.cpu().to(torch.float32))
        print(f"    mismatched elems: {int(diff.sum())}/{y.numel()}")
    return ok


def run_all():
    all_ok = True
    n_pass = n_run = n_skip = 0
    for label, d2, d3, d4, dt, d1 in CASES:
        block = d2 * d3 * d4
        cap = CAP_SB[dt]
        if block > cap:
            print(f"[SKIP] {label} ({D0},{d1},{d2},{d3},{d4}) "
                  f"block={block}>{cap} 超 {DT_NAME[dt]} {_MODE.upper()} 单块容量")
            n_skip += 1
            continue
        ok = run_case(label, d2, d3, d4, dt, d1)
        n_run += 1
        n_pass += int(ok)
        all_ok &= ok
    print(f"\n--- NL_0~47: PASS {n_pass}/{n_run} run, {n_skip} skipped ---")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":
        all_ok = run_all()
        print("\ntest pass!" if all_ok else "\ntest FAILED")
        sys.exit(0 if all_ok else 1)
    if len(sys.argv) >= 7:
        # 单 case 直接用命令行完整 5D shape（dim1 由命令行给定，与 per-case dim1 无关）。
        d0, d1, d2, d3, d4, dt = (int(a) for a in sys.argv[1:7])
        x = make_input(d0, d1, d2, d3, d4, dt).npu()
        y = nlast_transpose(x, is_mte3=1)
        ref = x.permute(0, 1, 3, 2, 4).contiguous()
        ok = torch.equal(y.cpu(), ref.cpu())
        print(f"[{'PASS' if ok else 'FAIL'}] CLI ({d0},{d1},{d2},{d3},{d4}) "
              f"{DT_NAME[dt]} shape_out={tuple(y.shape)}")
        print("test pass!" if ok else "test FAILED")
        sys.exit(0 if ok else 1)
    # 默认精选若干（对齐/非对齐、FP16/FP32、A大B小/A小B大）；dim1 用 per-case 值。
    default = [
        ("NL_0",  128,   8,  16, 0, 32),   # FP16 对齐 A大B小
        ("NL_5",   16,   8, 512, 0,  8),   # FP16 对齐 最大 block=65536=halfCap
        ("NL_11",  16,   8, 511, 0,  8),   # FP16 非对齐尾
        ("NL_17",   8,  16, 512, 0,  8),   # FP16 A小B大 最大 block
        ("NL_24", 128,   8,  16, 1, 24),   # FP32 对齐 A大B小
        ("NL_36",   8, 128,  16, 1, 16),   # FP32 A小B大
    ]
    all_ok = all(run_case(*c) for c in default)
    print("test pass!" if all_ok else "test FAILED")
    sys.exit(0 if all_ok else 1)
