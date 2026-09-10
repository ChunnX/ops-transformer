# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""片上内存布局 —— 对应 AscendC arch35 的 ``cube_api/mutex_buffer*.h``。

AscendC 侧用 ``MutexBufferManager`` 在运行期从一个 pool 里 bump 分配，再由
``MutexBuffersPolicy*`` 决定几路 buffer 轮转；pypto 的 tile 地址必须是 trace
期常量，所以同样的职责在这里退化成**按 ``d_align`` 分档算好的地址表**。

本模块是这些地址与 mutex id 的**唯一出处**：kernel 主体不再出现裸地址。
每档的容量上限由 ``_build_*`` 里的 assert 守住（L1 512KB / L0C 256KB /
UB 248KB），改布局时 assert 会先于上板失败。
"""

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM,
    CUBE_BASEN,
    D_CHUNK_MAXW,
    D_NCHUNKS,
    VECTOR_BASEM,
    VECTOR_BASEN,
)
# --- end generated imports ---


# ==================================================================
#  L1 (512KB)
# ==================================================================

# ---- L1 地址 (MutexBufferManager<BufferType::L1>，总量 512KB) ----
# dSL1Buf / pL1Buf 对应 kernel_base.h InitCVCommonBuffer 中的同名 buffer
L1_Q       = 0x00000   # [128,128] fp16 = 32KB
L1_K       = 0x08000
L1_V       = 0x10000
L1_DO      = 0x18000
L1_DS      = 0x20000   # dSL1Buf : [M,N] 供 MM3 左矩阵
L1_P       = 0x28000   # pL1Buf

# 0x30000-0x3FFFF 空闲：dS^T/P^T 不再单独存储(改为 L1_DS/L1_P 的 ZN 视图)
# MM3(dQ = dS@K) 的右矩阵要非转置的 K[S2,D]，必须独占一块 L1：
# 同一地址上叠加两种 shape/layout 视图(ZN 的 [D,S2] 与 ND 的 [S2,D])
# 能编过但结果静默错误。
L1_K_ND    = 0x40000

# L1_PRELOAD：Q/K/V/dO/K_ND 各再开一块，构成 double buffer。
# tile_group 填两个 addrs + 两个 mutex_id 后，框架会自动插入核内同步，
# 使下一轮的 load 与本轮的 matmul 重叠(对应 AscendC 的 IS_L1_PRELOAD)。
# 合计 14 块 x 32KB = 448KB <= 512KB。
# Q/dO 各 3 个槽位(而非 2)：cube 领先 vector 两个 task，mm2 在 task N
# 载入 Q_N，mm4 要的是 Q_{N-2}。3 槽位时 N%3 != (N-2)%3，两者不会互相
# 覆写，于是 mm4/mm5 可以直接复用 mm2/mm1 搬好的数据，无需再读 GM。
L1_Q2      = 0x48000
L1_Q3      = 0x50000
L1_DO2     = 0x58000
L1_DO3     = 0x60000
L1_Q4      = 0x68000
L1_DO4     = 0x70000



# ---- 按 d_align 分档的 L1 布局 ----
# 64/128 档共用上面那套写死的 32KB 槽位(物理宽都是 128)。
# 192 档每块是 48KB 而不是 32KB，512KB 装不下 14 块，故 Q/dO 各留 3 槽
# (mm4/mm5 复用 mm2/mm1 数据所需的最小槽位数，见 q_l1_dk 的说明)：
#   q x3 + dO x3 + k + k_nd + v = 9 x 48KB = 432KB，再加 ds/p 各 32KB
#   (它们是 [M,N]，与 D 无关) = 496KB <= 512KB。
# dict 查表在 addrs= 位置可用，且槽位数允许随档不同(已实测)。
# 地址表在**模块级**按档算好，kernel 里只做 `TABLE[d_align]` 这一步下标取值：
# 不能把 dict 赋给 kernel 内的局部变量(parser 报 "Unsupported closure
# variable type: dict"，只接受 int/float/bool/list/tuple)，但 dict 下标取出的
# list/int 可以直接用在 addrs= / mutex_ids= 位置(已实测，且槽位数可随档不同)。
def _build_l1(width):
    """铺排一档的 L1，返回 (q槽位, dO槽位, k, k_nd, v, ds, p)。"""
    if width in (64, 128):
        # 64/128 物理宽相同，沿用写死地址，保证 128 档生成代码零回归
        return ([L1_Q, L1_Q2, L1_Q3, L1_Q4], [L1_DO, L1_DO2, L1_DO3, L1_DO4],
                L1_K, L1_K_ND, L1_V, L1_DS, L1_P)
    blk = CUBE_BASEM * width * 2           # 192档 48KB
    dsp = CUBE_BASEM * CUBE_BASEN * 2      # ds/p 是 [M,N]，与 D 无关 = 32KB
    nq = 3
    a = 0
    qs = []
    for _ in range(nq):
        qs.append(a)
        a += blk
    dos = []
    for _ in range(nq):
        dos.append(a)
        a += blk
    k = a
    a += blk
    k_nd = a
    a += blk
    v = a
    a += blk
    ds = a
    a += dsp
    p = a
    a += dsp
    assert a <= 512 * 1024, f"L1 overflow for d_align={width}: {a} bytes"
    return (qs, dos, k, k_nd, v, ds, p)



_L1_TAB = {w: _build_l1(w) for w in (64, 128, 192)}
L1_Q_ADDRS = {w: _L1_TAB[w][0] for w in _L1_TAB}
L1_DO_ADDRS = {w: _L1_TAB[w][1] for w in _L1_TAB}
L1_K_ADDR = {w: _L1_TAB[w][2] for w in _L1_TAB}
L1_KND_ADDR = {w: _L1_TAB[w][3] for w in _L1_TAB}
L1_V_ADDR = {w: _L1_TAB[w][4] for w in _L1_TAB}
L1_DS_ADDR = {w: _L1_TAB[w][5] for w in _L1_TAB}
L1_P_ADDR = {w: _L1_TAB[w][6] for w in _L1_TAB}
# mutex_id 跟着槽位数走。64/128 档 4 槽，192 档 3 槽。
L1_Q_MUTEX = {64: [4, 31, 21, 18], 128: [4, 31, 21, 18], 192: [4, 31, 21]}
L1_DO_MUTEX = {64: [7, 15, 22, 19], 128: [7, 15, 22, 19], 192: [7, 15, 22]}



# ==================================================================
#  L0A / L0B (各 64KB) 与 L0C (256KB)
# ==================================================================

# ---- L0A / L0B / L0C ----
# L0A/L0B 各 64KB，[128,128]fp16 占 32KB，故各放 2 块做双缓冲：
# 5 个 matmul 串行复用同一块 L0A/L0B 时，pl.move 填充期间 mac 只能干等
# (实测 mte1 占用率 0.309 vs ASC 0.584)。双缓冲让下一个 matmul 的
# L1->L0 搬运与当前 matmul 的计算重叠。
L0A_ADDR   = 0x0000
L0A_ADDR2  = 0x8000
L0B_ADDR   = 0x0000
L0B_ADDR2  = 0x8000
L0C_ADDR   = 0x0000

# L0C 驻留累加(IS_DKV_RESIDENT_L0C=true)：dK/dV 各占一块独立 L0C，
# 在同一 s2 块内跨 s1 累加，s2 切换时才 fixpipe 出 GM。
# acc 64KB + dk 64KB + dv 64KB = 192KB <= 256KB
L0C_DK     = 0x10000
L0C_DV     = 0x20000

# acc 的第二块(L0C 共 256KB: acc x2 + dk + dv = 4 x 64KB 正好用满)。
# mm2/mm1/mm3 共用 acc，单缓冲时 fixpipe 把结果写出期间下一个 matmul
# 只能干等(实测 fixpipe 0.458 而 mac 仅 0.783)。
L0C_ADDR2  = 0x30000



# ---- 按 d_align 分档的 L0 布局 ----
# L0A/L0B 按「最宽的一个 D 分块」声明，故 192 档是 [128,96]=24KB 而不是
# 48KB，双缓冲(2x24=48KB<=64KB)得以保留 —— 这正是切 D 轴的首要目的。
def _build_l0ab(width):
    """L0A/L0B 的两个槽位地址(双缓冲)。

    间距必须按 **tile 实际声明的宽度** ACC_COLS 算，而不是 D 分块宽度：
    L0A/L0B 要同时装下 MM3/MM4/MM5 的 [M,N] 左矩阵，故声明成 ACC_COLS 列。
    若按分块宽 96 算间距(24KB)，两个槽位会重叠(每块实际占 32KB)。
    """
    if width in (64, 128):
        return [L0A_ADDR, L0A_ADDR2]
    blk = CUBE_BASEM * max(CUBE_BASEN, D_CHUNK_MAXW[width]) * 2
    return [0x0000, blk]



def _build_l0c(width):
    """返回 (acc 槽位, dk 各分块地址, dv 各分块地址)。

    dK/dV 要在同一 s2 块内跨 s1 轮次累加(IS_DKV_RESIDENT_L0C)，故每个 D
    分块都得有自己的常驻 L0C —— 不能像 acc 那样复用。
      128 档: acc x2 + dk x1 + dv x1 = 4 x 64KB = 256KB(与改动前一致)
      192 档: acc x1 + dk x2 + dv x2 = 5 x 48KB = 240KB <= 256KB
              (dv 也按 D 的分块数分块；Dv<=D 时后一块可能整块无效，
               由运行期宽度判断跳过，见 dv_chunk_w)
    """
    if width in (64, 128):
        return ([L0C_ADDR, L0C_ADDR2], [L0C_DK], [L0C_DV])
    n = D_NCHUNKS[width]
    # acc 声明成 ACC_COLS 列(要装 MM2/MM1 的 [M,s2] 输出)，故按它算大小；
    # dK/dV 只装 D 分块，按分块宽即可。间距一律用「tile 实际声明的宽度」，
    # 否则槽位会重叠。
    accBlk = CUBE_BASEM * max(CUBE_BASEN, D_CHUNK_MAXW[width]) * 4
    dkvBlk = CUBE_BASEN * D_CHUNK_MAXW[width] * 4
    a = 0
    acc = [a]
    a += accBlk
    dk = []
    for _ in range(n):
        dk.append(a)
        a += dkvBlk
    dv = []
    for _ in range(n):
        dv.append(a)
        a += dkvBlk
    assert a <= 256 * 1024, f"L0C overflow for d_align={width}: {a} bytes"
    return (acc, dk, dv)



L0AB_ADDRS = {w: _build_l0ab(w) for w in (64, 128, 192)}
_L0C_TAB = {w: _build_l0c(w) for w in (64, 128, 192)}
L0C_ACC_ADDRS = {w: _L0C_TAB[w][0] for w in _L0C_TAB}
L0C_ACC_MUTEX = {64: [14, 28], 128: [14, 28], 192: [14]}
L0C_DK_ADDRS = {w: _L0C_TAB[w][1] for w in _L0C_TAB}
L0C_DV_ADDRS = {w: _L0C_TAB[w][2] for w in _L0C_TAB}
# 每个分块单独取一个标量地址表。不能在 kernel 里写 `TABLE[d_align][-1]`：
# parser 不支持负下标(报 "GetItemExpr index -1 out of bounds")。
L0C_DK0_ADDR = {w: L0C_DK_ADDRS[w][0] for w in _L0C_TAB}
L0C_DV0_ADDR = {w: L0C_DV_ADDRS[w][0] for w in _L0C_TAB}
# 第二个分块的地址/mutex。单块档位下这些 tile 永不被真的使用(调用路径被
# trace 期的 `if D_NCHUNKS > 1` 整块剪掉)，故让它别名到第 0 块即可。
L0C_DK1_ADDR = {w: L0C_DK_ADDRS[w][D_NCHUNKS[w] - 1] for w in _L0C_TAB}
L0C_DV1_ADDR = {w: L0C_DV_ADDRS[w][D_NCHUNKS[w] - 1] for w in _L0C_TAB}
L0C_DK1_MUTEX = {64: 16, 128: 16, 192: 29}
L0C_DV1_MUTEX = {64: 17, 128: 17, 192: 25}
# acc/L0A/L0B 的列宽：MM2/MM1 输出 [M,s2]，MM3 输出 dQ 的一个 D 分块。
# 取两者的较大值，一块 tile 同时够用。128 档等于 128(与改动前一致)。
ACC_COLS = {w: max(CUBE_BASEN, D_CHUNK_MAXW[w]) for w in (64, 128, 192)}



# ==================================================================
#  UB (248KB 可用)
# ==================================================================

# ---- pre/post 阶段的 UB ----
# 这两个阶段与主循环靠 sync_all 完全隔开，故直接从 0 起借用整块 UB
# (mm1/mm2 结果区等)，不与主循环争空间；mutex_ids 同样可复用。
#
# pre/post 按 ASC 一样走一维元素，不按 [行, D] 切。usable=248KB
# （256KB 预留 8KB 给 UB[0] tiling 中转；BN2 无 pre/sync_all，不留会
# 和 cube fixpipe 抢 UB[0]）。
#   pre  单槽 fp32：248KB/4  = 63488 elems（192 档 63360，整除 192）
#   post depth-2：  248KB/12 = 21120 elems（64 档 20480，整除 64）
# 三档都吃满 UB。行区间已由 metadata 按 nAiv 均分，块大不再空核。
# 每档 elems 都能被 128 整除，full-tile 的 VF 段数是编译期常量。
UB_HEAD_RESERVE = 0x1000
PRE_ELEMS = {64: 63488, 128: 63488, 192: 63360}
UB_PRE_ZERO = UB_HEAD_RESERVE

POST_ELEMS = {64: 20480, 128: 21120, 192: 21120}
UB_POST_F32_0 = UB_HEAD_RESERVE



def post_ub_addrs(nElems):
    """按 1D 元素个数算 post 的四个 UB 槽位地址。nElems 是 tilingkey 常量。"""
    f32Bytes = nElems * 4
    f16Bytes = nElems * 2
    f32_1 = UB_POST_F32_0 + f32Bytes
    f16_0 = f32_1 + f32Bytes
    return UB_POST_F32_0, f32_1, f16_0, f16_0 + f16Bytes


# 与其它片上地址一样做成按档查好的表：tile 的 addrs= 只接受 trace 期常量，
# 在 kernel 里做元组解包会撞上 codegen 的 "Array tuple assignment" 限制。
_POST_TAB = {w: post_ub_addrs(POST_ELEMS[w]) for w in (64, 128, 192)}
POST_F32_ADDRS = {w: [_POST_TAB[w][0], _POST_TAB[w][1]] for w in _POST_TAB}
POST_F16_ADDRS = {w: [_POST_TAB[w][2], _POST_TAB[w][3]] for w in _POST_TAB}



# ---- UB 地址，硬上限 248KB；实测总占用 242.5KB ----
# mm1ResBuf[2] / mm2ResBuf[2] 与 AscendC 同名同尺寸(VECTOR_BASEM*VECTOR_BASEN*4)
UB_MM1_0   = UB_HEAD_RESERVE + 0x00000   # fp32[64,128] 32KB
UB_MM1_1   = UB_HEAD_RESERVE + 0x08000
UB_MM2_0   = UB_HEAD_RESERVE + 0x10000
UB_MM2_1   = UB_HEAD_RESERVE + 0x18000
# V1 一趟 VECTOR_BASEM 行、按行 reduce，不再开 prod/tmp。列宽是
# V1_W = 64 if d_align==64 else dv_align（y/dy 是 Dv 宽）。
# 地址按 V1_W 分档排，见 _build_ub。
UB_Y       = UB_HEAD_RESERVE + 0x20000   # fp16[64,v1w] V1 前向输出 y



def _build_ub(v1w):
    """按 V1 列宽铺排 y/dy/nzp/sfmg/lse。已删 prod/tmp。

    返回 (y, dy, nzp, sfmg, lse0, lse1)。
    attenmask 复用 y/dy 地址（ASC attenMaskOrYInQue / pseOrDyInQue）。
    uint8[64,128]=8KB；y/dy 在 v1w>=64 时至少 8KB。NZP 仍是 [65,128]。
    """
    a = UB_HEAD_RESERVE + 0x20000
    y = a
    a += VECTOR_BASEM * v1w * 2
    dy = a
    a += VECTOR_BASEM * v1w * 2
    nzp = a
    a += (VECTOR_BASEM + 1) * VECTOR_BASEN * 2
    a = (a + 0x1FF) & ~0x1FF
    sfmg = a
    a += 0x200
    lse0 = a
    lse1 = a + 0x200
    a += 0x400
    assert a <= 248 * 1024, f"UB overflow for v1w={v1w}: {a} bytes"
    return (y, dy, nzp, sfmg, lse0, lse1)



# V1 UB 按列宽分档：64（d_align=64）、128、192。
_UB_TAB = {w: _build_ub(w) for w in (64, 128, 192)}
UB_Y_A = {w: _UB_TAB[w][0] for w in _UB_TAB}
UB_DY_A = {w: _UB_TAB[w][1] for w in _UB_TAB}
UB_NZP_A = {w: _UB_TAB[w][2] for w in _UB_TAB}
UB_SFMG_A = {w: _UB_TAB[w][3] for w in _UB_TAB}
UB_LSE0_A = {w: _UB_TAB[w][4] for w in _UB_TAB}
UB_LSE1_A = {w: _UB_TAB[w][5] for w in _UB_TAB}
# attenmask 复用 y/dy 地址（vec1 之后 vec2 再用，与 ASC 队列复用同构）。
UB_MASK_A = UB_Y_A
UB_MASK_PRE_A = UB_DY_A



# ==================================================================
#  BN2 模板专用的 UB 槽位
# ==================================================================

# BN2 的 V1 与 GS1S2 共用 softmax_grad_front_row_vf。地址从
# UB_HEAD_RESERVE 起，避免与 codegen 的 UB[0] 中转打架。
BN2_V1_ROWS = VECTOR_BASEM
BN2_UB_Y = UB_HEAD_RESERVE + 0x20000
BN2_UB_DY = UB_HEAD_RESERVE + 0x24000
BN2_UB_NZP = UB_HEAD_RESERVE + 0x28000
BN2_UB_SFMG = UB_HEAD_RESERVE + 0x2C200
BN2_UB_LSE_0 = UB_HEAD_RESERVE + 0x2C400
BN2_UB_LSE_1 = UB_HEAD_RESERVE + 0x2C600
# attenmask 复用 y/dy。v5/v6 cast 仍单独一块。
BN2_CAST_W = VECTOR_BASEN
BN2_UB_CAST = UB_HEAD_RESERVE + 0x2C800


# ==================================================================
#  mutex id 分配表
# ==================================================================
# pypto 用 mutex id 而不是显式 event 来排 buffer 的读写序，framework 据此
# 在同一 id 的前后访问之间插同步。id 空间是 0..31，**按核类型各算一套**：
# cube section 里的 id 与 vector section 里的同号 id 互不相干。
#
# 把它们集中命名在这里，是为了让「哪些同号是有意复用、哪些只是撞号」这件事
# 可查 —— 散在 make_tile_group 的字面量里时，这是审不出来的。
#
# 三类有意的同号，都不是 bug：
#   1) reader 视图与 writer 视图共用 id —— q_l1_dk/do_l1_dv 必须与
#      q_l1/do_l1 同 id，否则 mm2 会在 mm4 读完前覆写(见 fag_buffers)；
#   2) mask 复用 y/dy —— 同地址同 id，v1 用完 v2 才用；
#   3) pre/post 复用主循环的 id —— 被 sync_all 完全隔开，不会同时活跃。
#
# ---- CV 共享 UB ----
MTX_MM1_RES = [0, 1]
MTX_MM2_RES = [2, 3]
# ---- cube: L1 ----
# Q / dO 是多槽位，按 d_align 分档，见 L1_Q_MUTEX / L1_DO_MUTEX (4/31/21/18,
# 7/15/22/19)。
MTX_L1_K = [5]
MTX_L1_V = [6]
MTX_L1_DS = [8]
MTX_L1_P = [9]
MTX_L1_DS_T = [10]      # 与 MTX_L1_DS 同址、不同 id：ZN 视图有独立游标
MTX_L1_P_T = [11]
MTX_L1_K_ND = [30]
# ---- cube: L0 ----
MTX_L0A = [12, 26]
MTX_L0B = [13, 27]
# acc 按 d_align 分档，见 L0C_ACC_MUTEX (14/28)。
MTX_L0C_DK0 = [16]
MTX_L0C_DV0 = [17]
# 第二个 D 分块见 L0C_DK1_MUTEX / L0C_DV1_MUTEX。
# ---- vector: UB ----
MTX_UB_Y = [15]         # attenmask 复用
MTX_UB_DY = [16]        # attenmask(pre) 复用
MTX_UB_NZP = [19]
MTX_UB_CAST = [20]      # 仅 BN2
MTX_UB_SFMG = [23]
MTX_UB_LSE = [24, 25]
# ---- vector: pre / post (与主循环靠 sync_all 隔开，故复用上面的 id) ----
MTX_UB_ZERO = [19]
MTX_UB_POST_F32 = [19, 20]
MTX_UB_POST_F16 = [21, 22]
