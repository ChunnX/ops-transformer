# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Broadcast NDDMA 全用例时延/带宽采集驱动——3D NDDMA 方案（DESIGN §9，权威）。

分核维 dim0=56（= 物理 vector core 数，单波均衡 per=1/rem=0：dim0=56/blockNum=56，
每核正好 1 个 i0，消除不整除第二波拖尾）。用例取 DESIGN §9.6 的 12 例 5D 场景集
（AB/BA/BAB/ABA 各 3，默认 fp32）+ §9.6 的 4 个 fp16/int8 dtype 变体 + 1 个 fp16 SKIP 对照。

输出恒 (dim0,dim1,dim2,dim3,dim4)；tile=单条 3D 目的块 dim2*dim3*dim4（RUN/SKIP 判据，
dim1 唯一核内 for，dim2/dim3/dim4 为单条 NdDmaLoopInfo<3> 三轴）。读写不对称（DESIGN §9.6/§7.7）：
  numel_dst = dim0*dim1*dim2*dim3*dim4（满口径）；R_logical = numel_dst × elemSize。
  isMte3=1 -> R_logical + W；isMte3=0 -> R_logical。

RUN/SKIP 两判据 AND（§9.4）：tile=dim2*dim3*dim4<=halfCap[dtype] 且 dim4*elemSize%8==0
（inner 连续块 8B 对齐，§9.0 上板坐实）。SKIP 主防线在 host：任一不满足即不发 kernel。

单次 msprof 会话内按固定 (1 预热 + N 测量) 顺序跑当前 .so 变体（DB 或 SB）下所有可容纳用例。

用法: msprof ... python3 prof_pta_matrix.py <db|sb> <isMte3:0|1> [iters]
  db/sb : 与当前 build/libnddma_broadcast_launch.so 编译的 USE_PINGPONG 一致（决定容量跳过）
  isMte3: 1=读写混合(MTE3 写回) 0=纯 NDDMA 复制读
  NDDMA_CORESCAN=1 → 核数扫描（4 代表 case × blockNum 列表，DESIGN §9.6/§7.6）。
"""

import os
import sys
import torch
import torch_npu  # noqa: F401

DTYPE_MAP = {0: torch.float16, 1: torch.float32, 2: torch.int8}
ELEM = {0: 2, 1: 4, 2: 1}
DT_NAME = {0: "fp16", 1: "fp32", 2: "int8"}
# 分核维 dim0=56（= 物理 vector core 数，单波均衡 per=1/rem=0），固定不变。
D0 = 56

# ============================================================================
# 5D 广播场景采集集（DESIGN §9.6）：4 场景 × 3 形态（默认 fp32）+ 4 dtype 变体 + 1 fp16 SKIP 对照。
#   输出恒 (dim0,dim1,dim2,dim3,dim4)；tile=单条 3D 目的块 dim2*dim3*dim4（RUN/SKIP 两判据 AND，§9.4）。
#   numel_dst = dim0*dim1*dim2*dim3*dim4（满口径，§7.7）；R_logical=numel_dst×elemSize。
# 每项 (label, scene, dim2, dim3, dim4, dim1, dtype)。dim0=56 固定；per-case dim1 上板后按 min>40us 放大。
#   dim4 全部满足对应 dtype 8B 对齐（fp32 dim4%2==0 / fp16 dim4%4==0 / int8 dim4%8==0，§9.4 判据 2）。
#   AB_align_fp16（dim4=6）刻意非对齐（6*2%8=4≠0）→ host 判据 2 拦为 SKIP，回归 SKIP 判据本身（§9.6.0 方案 B）。
# ============================================================================
SCENE_NAME = {0: "AB", 1: "BA", 2: "BAB", 3: "ABA"}
CASES_5D = [
    # per-case dim1 上板后按 round_005 采集（DB 纯读 min = 最快配置 = floor）回填放大，目标最快配置 min>40us
    #   （只增 dim1 = 数据量 + 时延，tile/N/scene/dtype 全不改，§9.6 采集口径）。
    #   两点线性拟合 lat=ovh(~1.5us)+slope·dim1（base8 + 中间放大点两轮实测）→ 反解 dim1@48us；投影 floor≈48us
    #   （保 min>40us 冗余，实测 db_r/sb_r floor 全 >40us）。
    # AB (dim2/dim3 保留, dim4 广播) —— 小/中/大 tile、A小B大
    ("AB_0",  0,  8,   8,  16, 117, 1), # 小 tile·dim4 对齐·last 轴广播  tile=1024
    ("AB_1",  0,  4,  16,  64,  63, 1), # 中 tile·大 dim4 广播倍          tile=4096
    ("AB_2",  0,  2,  16, 256,  47, 1), # 大 tile·A小(d2=2)B大·逼近半区   tile=8192
    # BA (dim2/dim4 保留, dim3 广播)
    ("BA_0",  1,  8,  16,   8, 116, 1), # 小 tile·dim3 广播·dim4 保留连续  tile=1024
    ("BA_1",  1,  4,  64,  16,  63, 1), # 中 tile·大广播倍(d3=64)          tile=4096
    ("BA_2",  1,  2, 256,  16,  47, 1), # 大 tile·A小B大·外轴大广播        tile=8192
    # BAB (dim3 保留, dim2+dim4 广播) —— 双置零
    ("BAB_0", 2,  8,   8,   8, 132, 1), # 双置零·小 tile                   tile=512
    ("BAB_1", 2, 16,   8,  16,  96, 1), # 双置零·中 tile·dim2 大广播        tile=2048
    ("BAB_2", 2, 32,   4,  16,  96, 1), # 双置零·dim2 极大广播·A小(d3=4)    tile=2048
    # ABA (dim2/dim4 保留, dim3 广播) —— 中轴置零（≡ BA，§9.1 同构，语义标注）
    ("ABA_0", 3,  8,   8,   8, 129, 1), # 中轴置零·小 tile                 tile=512
    ("ABA_1", 3, 16,   8,   8, 117, 1), # 中轴置零·dim2 保留大             tile=1024
    ("ABA_2", 3,  8,  16,  16,  96, 1), # 中轴置零·中 tile·dim3 大广播      tile=2048
    # §9.6 dtype 覆盖变体：AB/BA 各补 1 个 fp16 + 1 个 int8（dim4 满足对应 dtype 8B 对齐）
    ("AB_0_fp16", 0, 8, 8, 16, 119, 0), ("BA_0_fp16", 1, 8, 16, 8, 119, 0),
    ("AB_0_i8",   0, 8, 8, 16, 119, 2), ("BA_0_i8",   1, 8, 16, 8, 119, 2),
    # §9.6.0 方案 B 对齐敏感 SKIP 对照：fp16 dim4=6（6*2%8=4≠0）→ host 判据 2 拦 SKIP，不发 kernel。
    ("AB_align_fp16", 0, 8, 8, 6, 8, 0),
]

# 单条 3D 目的块 tile 容量（elems），按 dtype：DB=ubCapElems/2，SB=ubCapElems（§9.4）。
CAP5D = {
    "db": {0: 65536, 1: 16384, 2: 65536},
    "sb": {0: 131072, 1: 32768, 2: 131072},
}
# 核数扫描代表 case（每场景一个）与 blockNum 列表（§9.6/§7.6）。
CORESCAN_LABELS = ["AB_0", "BA_0", "BAB_0", "ABA_0"]
CORESCAN_BLOCKNUMS = [1, 2, 4, 8, 16, 28, 56]


def _aligned(d4, dt):
    """§9.4 判据 2：inner 连续块 8B 对齐 dim4*elemSize%8==0（fp16%4/fp32%2/int8%8==0）。"""
    return (d4 * ELEM[dt]) % 8 == 0


def _make_src_5d_prof(dim0, dim1, dim2, dim3, dim4, scene, dt):
    """按 scene 构造 5D 输入张量（广播轴长度=1，单条 3D 整块，§9.2 与 pta_nddma._make_src_5d 同源）。
      AB : (d0,d1,dim2,dim3,1)   BA/ABA : (d0,d1,dim2,1,dim4)   BAB : (d0,d1,1,dim3,1)"""
    if scene == 0:             # AB：src(d2,d3,1) 整块 d2*d3
        shp = (dim0, dim1, dim2, dim3, 1)
    elif scene == 2:           # BAB：src(1,d3,1) 整块 d3
        shp = (dim0, dim1, 1, dim3, 1)
    else:                      # BA / ABA：src(d2,1,d4) 整块 d2*d4（BA≡ABA §9.1）
        shp = (dim0, dim1, dim2, 1, dim4)
    total = 1
    for s in shp:
        total *= s
    if dt == 2:
        return torch.randint(-128, 128, shp, dtype=torch.int8)
    flat = (torch.arange(1, total + 1, dtype=torch.float32) % 2048)
    return flat.reshape(shp).to(DTYPE_MAP[dt])


def run_matrix_5d(mode, is_mte3, iters):
    """5D 采集集四配置采集（同核 dim0=56）：RUN/SKIP 两判据 AND（tile=d2*d3*d4<=cap 且 dim4 对齐，§9.4）。"""
    from pta_nddma import nddma_broadcast_5d
    cap = CAP5D[mode]
    run, skip = [], []
    for lb, scene, d2, d3, d4, dim1, dt in CASES_5D:
        tile = d2 * d3 * d4
        # host 主防线：容量 AND 对齐，任一不满足即 SKIP（不发 kernel，§9.6.0/§12.3）。
        if tile <= cap[dt] and _aligned(d4, dt):
            run.append((lb, scene, d2, d3, d4, dim1, dt, tile))
        else:
            skip.append((lb, scene, d2, d3, d4, dim1, dt, tile))
    print(f"# MANIFEST5D mode={mode} isMte3={is_mte3} iters={iters} "
          f"run={len(run)} skip={len(skip)} dim0={D0} (每 case 1 预热+{iters} 测量)")
    print(f"# {'ord':>3} {'label':>10} {'scene':>5} {'(d2,d3,d4)':>16} {'dt':>5} "
          f"{'tile':>7} {'src_elems':>9} {'bcastX':>7} {'numel_dst':>12} {'Rlog_MB':>8}")
    for i, (lb, scene, d2, d3, d4, dim1, dt, tile) in enumerate(run):
        assert tile <= cap[dt] and _aligned(d4, dt), f"{lb} {mode} tile/align fail"
        numel_dst = D0 * dim1 * d2 * d3 * d4
        # GM 实际读 src 整块：AB=d2*d3 / BA·ABA=d2*d4 / BAB=d3（§9.6 带宽口径澄清）。
        if scene == 0:
            src_elems = d2 * d3
        elif scene == 2:
            src_elems = d3
        else:
            src_elems = d2 * d4
        bcastx = tile / src_elems                       # 广播倍数（tile/src_elems），BAB/ABA 高广播倍
        print(f"# {i:>3} {lb:>10} {SCENE_NAME[scene]:>5} {f'({d2},{d3},{d4})':>16} "
              f"{DT_NAME[dt]:>5} {tile:>7} {src_elems:>9} {bcastx:>7.1f} {numel_dst:>12} "
              f"{numel_dst*ELEM[dt]/1e6:>8.2f}")
    for lb, scene, d2, d3, d4, dim1, dt, tile in skip:
        why = (f"tile=d2*d3*d4={tile}>{cap[dt]}" if tile > cap[dt]
               else f"dim4*elem={d4}*{ELEM[dt]}%8={d4*ELEM[dt]%8}!=0")
        print(f"# SKIP5D {lb} scene={SCENE_NAME[scene]} {DT_NAME[dt]} {why} ({mode.upper()})")

    for lb, scene, d2, d3, d4, dim1, dt, tile in run:
        x = _make_src_5d_prof(D0, dim1, d2, d3, d4, scene, dt).npu()
        _ = nddma_broadcast_5d(x, D0, dim1, d2, d3, d4, scene, is_mte3=is_mte3)      # 预热
        for _ in range(iters):
            _ = nddma_broadcast_5d(x, D0, dim1, d2, d3, d4, scene, is_mte3=is_mte3)
    print("done")


def run_corescan_5d(mode, is_mte3, iters):
    """核数扫描（§9.6/§7.6）：4 代表 case × blockNum={1,2,4,8,16,28,56}（全 fp32）。
      1 核护栏：预估 1 核总时延 dim0*dim1*t_nddma>T_max(缺省 5ms) → 降 dim1（3D 下无 dim2-for，只 dim1 for）。
      尾核不均衡：BW 分母取最慢核 t（整卡墙钟），分子恒满 numel_dst；不整除记录 imbalance=ceil/floor。
    """
    from pta_nddma import nddma_broadcast_5d
    t_max_ms = float(os.environ.get("NDDMA_TIMEOUT_MS", "5"))
    t_nddma_us = float(os.environ.get("NDDMA_TNDDMA_US", "1.0"))  # 单条 3D NDDMA 估计时延（首轮回填）
    reps = {lb: (sc, d2, d3, d4, dim1, dt)
            for lb, sc, d2, d3, d4, dim1, dt in CASES_5D}
    print(f"# MANIFEST5D_CORESCAN mode={mode} isMte3={is_mte3} iters={iters} "
          f"labels={CORESCAN_LABELS} blockNums={CORESCAN_BLOCKNUMS} "
          f"T_max={t_max_ms}ms t_nddma={t_nddma_us}us")
    print(f"# {'label':>6} {'scene':>5} {'blockNum':>8} {'dim1':>5} {'numBlk':>6} "
          f"{'imbalance':>9} {'numel_dst':>12} {'note':>18}")
    for lb in CORESCAN_LABELS:
        scene, d2, d3, d4, dim1, dt = reps[lb]
        for bn in CORESCAN_BLOCKNUMS:
            numblk = D0                                  # numBlocks_=dim0
            per = numblk // bn
            rem = numblk % bn
            ceil_b = per + (1 if rem else 0)
            imbalance = (ceil_b / per) if per > 0 else float("inf")
            # 1 核护栏：预估总时延 = dim0*dim1*t_nddma（3D 下每拍一条 3D NDDMA，仅 dim1 for，无 dim2-for）。
            eff_dim1 = dim1
            note = ""
            if bn == 1:
                est_us = D0 * dim1 * t_nddma_us
                if est_us > t_max_ms * 1000:
                    new_dim1 = max(1, int((t_max_ms * 1000) // (D0 * t_nddma_us)))
                    if new_dim1 < dim1:
                        eff_dim1 = new_dim1
                        note = f"1core降dim1->{new_dim1}"
                    else:
                        note = "1core估时延OK"
            numel_dst = D0 * eff_dim1 * d2 * d3 * d4
            print(f"# {lb:>6} {SCENE_NAME[scene]:>5} {bn:>8} {eff_dim1:>5} {numblk:>6} "
                  f"{imbalance:>9.3f} {numel_dst:>12} {note:>18}")
            x = _make_src_5d_prof(D0, eff_dim1, d2, d3, d4, scene, dt).npu()
            _ = nddma_broadcast_5d(x, D0, eff_dim1, d2, d3, d4, scene,
                                   is_mte3=is_mte3, block_num=bn)                 # 预热
            for _ in range(iters):
                _ = nddma_broadcast_5d(x, D0, eff_dim1, d2, d3, d4, scene,
                                       is_mte3=is_mte3, block_num=bn)
    print("done")


if __name__ == "__main__":
    mode = sys.argv[1]
    is_mte3 = int(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) >= 4 else 5

    # 5D 为默认唯一分支；NDDMA_CORESCAN=1 → §9.6/§7.6 核数扫描（4 代表 × blockNum 列表）。
    if os.environ.get("NDDMA_CORESCAN") == "1":
        run_corescan_5d(mode, is_mte3, iters)
    else:
        run_matrix_5d(mode, is_mte3, iters)
