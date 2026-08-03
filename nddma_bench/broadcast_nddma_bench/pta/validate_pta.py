# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""通过 PTA 接口验证 Broadcast NDDMA kernel 的精度——3D NDDMA 方案（DESIGN §9，权威）。

参考值：torch 原生 x_src.broadcast_to((d0,d1,d2,d3,d4)).contiguous()（沿广播轴复制）。
纯搬运（复制读），逐元素 bitwise 相等（torch.equal，error ratio=0）。
★ 精度用例强制 dim0>=2 且 dim1>=2（PLAN §12.4）：覆盖 SrcOff/DstOff 跨块 g* 基址前进
  （dim0 分核 × dim1 for，§9.2 注），探针恒单核单块无法覆盖，块间前进仅由本对拍坐实。

用法:
  python3 validate_pta.py            # 5D 全集（12 例 fp32 + 4 dtype RUN + 1 fp16 SKIP 对照）—— 默认唯一分支
  python3 validate_pta.py --5d       # 同上（显式）

⚠️ 变体一致性（S1，重要）：`NDDMA_MODE=sb` 只切换 Python 侧容量表，**不改 .so 编译变体**。
   SB 变体精度复跑**必须**先 `USE_PINGPONG=false` 重编 `build/libnddma_broadcast_launch.so`
   （并清 torch ext 缓存）才正确；直接用 DB 编的 .so 跑 `NDDMA_MODE=sb` 会因 tile>DB halfCap
   在半区溢出而误判 FAIL。切换方法见 collect_matrix_5d.sh（备份→toggle→build→还原）。
"""

import os
import sys
import torch
import torch_npu  # noqa: F401  注册 NPU 后端

DTYPE_MAP = {0: torch.float16, 1: torch.float32, 2: torch.int8}
DT_NAME = {0: "fp16", 1: "fp32", 2: "int8"}

_MODE = os.environ.get("NDDMA_MODE", "db").lower()

# ============================================================================
# 5D 广播场景（DESIGN §9.6/§9.4）：AB/BA/BAB/ABA 四场景 bitwise 精度对拍——3D NDDMA 方案。
#   golden = x_src.broadcast_to((d0,d1,d2,d3,d4)).contiguous()（§9.6）。
#   dim0>=2 分核 × dim1>=2 核内 for（§12.4，覆盖跨块 g* 基址前进）；DB+SB 双变体各跑，合并覆盖每唯一 shape。
#   RUN/SKIP 两判据 AND（§9.4）：tile=dim2*dim3*dim4>cap（单条 3D 目的块）或 dim4*elem%8!=0（inner 8B 对齐）即 SKIP。
# 每项 (label, scene, dim2, dim3, dim4, dtype)；单条 3D NDDMA（dim1 唯一 for，dim2/dim3/dim4 为 loopInfo 三轴）。
#   dtype 默认 1=fp32（12 例）；§9.6 补 4 个变体保留 fp16/int8 覆盖 + 1 fp16 SKIP 对照。
# ============================================================================
from pta_nddma import bcast_5d, SCENE_AB, SCENE_BA, SCENE_BAB, SCENE_ABA  # noqa: E402

SCENE_NM = {0: "AB", 1: "BA", 2: "BAB", 3: "ABA"}
# tile=dim2*dim3*dim4（单条 3D 目的块，§9.4），据 dtype 约束 dim4（8B 对齐 §9.0）。
# ★ 精度用例强制 dim0>=2 且 dim1>=2（PLAN §12.4）：覆盖 SrcOff/DstOff 的跨块 g* 基址前进
#   （dim0 分核 × dim1 for，§9.2 注）——探针恒单核单块（dim0/dim1=1），块间前进只由此对拍覆盖。
D0_5D = 2   # dim0>=2（跨核前进）
D1_5D = 2   # dim1>=2（核内 for 前进）
# elemSize(byte) by dtype code：0=fp16(2)/1=fp32(4)/2=int8(1)，用于 8B 对齐判据 dim4*elem%8==0。
ELEM_SZ = {0: 2, 1: 4, 2: 1}
CASES_5D_PREC = [
    # ---- §9.6 的 12 例（tile=d2*d3*d4，dim4 全对齐 fp32：dim4*4%8==0；默认 fp32=dtype 1）----
    ("AB_0",  SCENE_AB,  8,  8,  16, 1), ("AB_1",  SCENE_AB,  4, 16,  64, 1),
    ("AB_2",  SCENE_AB,  2, 16, 256, 1),
    ("BA_0",  SCENE_BA,  8, 16,   8, 1), ("BA_1",  SCENE_BA,  4, 64,  16, 1),
    ("BA_2",  SCENE_BA,  2, 256, 16, 1),
    ("BAB_0", SCENE_BAB, 8,  8,   8, 1), ("BAB_1", SCENE_BAB, 16, 8,  16, 1),
    ("BAB_2", SCENE_BAB, 32, 4,  16, 1),
    ("ABA_0", SCENE_ABA, 8,  8,   8, 1), ("ABA_1", SCENE_ABA, 16, 8,   8, 1),
    ("ABA_2", SCENE_ABA, 8, 16,  16, 1),
    # ---- §9.6 dtype 变体：AB/BA 各补 1 个 fp16 + 1 个 int8（dim4 满足对应 dtype 8B 对齐）----
    ("AB_0_fp16", SCENE_AB, 8, 8, 16, 0), ("BA_0_fp16", SCENE_BA, 8, 16, 8, 0),
    ("AB_0_i8",   SCENE_AB, 8, 8, 16, 2), ("BA_0_i8",   SCENE_BA, 8, 16, 8, 2),
    # ---- §9.6.0 方案 B 对齐敏感 SKIP 对照例：fp16 dim4=6（6*2%8=4≠0 非对齐 → 必 SKIP，不发 kernel）----
    #   probe run_dtype_align.sh fp16 dim4=6 上板 FAIL 坐实；替代 §8.3「非对齐奇数尾轴上板 RUN」覆盖意图。
    ("AB_align_fp16", SCENE_AB, 8, 8, 6, 0),
]
# 单条 3D 目的块 tile 容量（elems），按 dtype：DB=ubCapElems/2，SB=ubCapElems（§9.4）。
#   fp32 DB halfCap=16384/SB=32768；fp16/int8 DB halfCap=65536/SB=131072。
CAP_5D = {
    "db": {0: 65536, 1: 16384, 2: 65536},
    "sb": {0: 131072, 1: 32768, 2: 131072},
}


def run_all_5d():
    """5D 四场景 bitwise 精度（dim0>=2 且 dim1>=2，覆盖跨块前进）：tile=d2*d3*d4，两判据 AND SKIP。"""
    cap = CAP_5D.get(_MODE, CAP_5D["db"])
    n_pass = n_run = n_skip = 0
    all_ok = True
    for lb, scene, d2, d3, d4, dt in CASES_5D_PREC:
        tile = d2 * d3 * d4                                   # §9.4：单条 3D 目的块
        aligned = (d4 * ELEM_SZ[dt]) % 8 == 0                 # §9.0/§9.4 判据 2：inner 8B 对齐
        # 两判据 AND（§9.4）：容量 tile<=halfCap 且 dim4*elem%8==0，任一不满足即 SKIP（不发 kernel）。
        if tile > cap[dt] or not aligned:
            why = (f"tile=d2*d3*d4={tile}>{cap[dt]}" if tile > cap[dt]
                   else f"dim4*elem={d4}*{ELEM_SZ[dt]}%8={d4*ELEM_SZ[dt]%8}!=0")
            print(f"[SKIP] {lb} {SCENE_NM[scene]} {DT_NAME[dt]} {why} ({_MODE.upper()})")
            n_skip += 1
            continue
        # §11.3.1 问题 21：golden 与 kernel 输入共享同一 dtype 张量（bcast_5d 内 x_src 先按 dtype 生成，
        #   再由同一张量走 broadcast_to().contiguous() 出 golden、并喂同一张量给 kernel，无 fp32 中间 cast）。
        y, golden = bcast_5d(D0_5D, D1_5D, d2, d3, d4, scene,
                             dtype=DTYPE_MAP[dt], is_mte3=1)
        ok = torch.equal(y.cpu(), golden.cpu())
        tag = f"{lb} {SCENE_NM[scene]} ({D0_5D},{D1_5D},{d2},{d3},{d4}) {DT_NAME[dt]}"
        print(f"[{'PASS' if ok else 'FAIL'}] {tag:46s} out={tuple(y.shape)}")
        if not ok:
            diff = (y.cpu().to(torch.float32) != golden.cpu().to(torch.float32))
            print(f"    mismatched elems: {int(diff.sum())}/{y.numel()}")
        n_run += 1
        n_pass += int(ok)
        all_ok &= ok
    print(f"\n--- CASES_5D ({_MODE.upper()}): PASS {n_pass}/{n_run} run, {n_skip} skipped ---")
    return all_ok


if __name__ == "__main__":
    # 5D 为默认唯一分支（无参也走 5D 全集）；--5d 显式等价。
    all_ok = run_all_5d()
    print("\ntest pass!" if all_ok else "\ntest FAILED")
    sys.exit(0 if all_ok else 1)
