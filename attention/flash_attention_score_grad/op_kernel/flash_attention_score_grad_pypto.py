# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttentionScoreGrad kernel (PyPTO Pro).

一比一复刻 AscendC 实现 arch35/flash_attention_score_grad_kernel.h，
本文件对应 SPLIT_AXIS == BN2GS1S2 模板，且为下述场景裁剪：

  非 TND(layoutType == BSNGD)、无 sparse / attenMask / pse / dropout /
  rope / sink / 确定性计算(DETER_SPARSE_TYPE == NO_DETER)、输入 FP16。

在该配置下 AscendC 的编译期谓词取值为：
  IS_DQ_WRITE_UB = IS_DK_WRITE_UB = IS_DV_WRITE_UB = false
      -> dQ/dK/dV 均由 fixpipe 经 GM workspace 做 atomicAdd 累加
  IS_DKV_RESIDENT_L0C = false (CUBE_BASEN=128, HEAD_DIM_ALIGN=128)
      -> dK/dV 不驻留 L0C，故 enPartialSum=false、isFixpOut=true
  IS_DROP = false
      -> ComputeDqkvBn2gs1s2 走 ProcessBn2gs1s2LastVec 的 else 分支
  IsValid() 恒为 true(无 attenMask)
      -> GetNextValidIdx 退化为边界判断，无需扫描
本文件实现 Process() 的两槽 taskId ping-pong；PRELOAD_TIMES=3 的
ProcessPreloadTwoTimes 预取环是 D<=128 的独立特化，后续再叠加。

CV 通信全部使用手动 set/wait cross-core，且正反向成对：
  正向 C->V : SYNC_C2_TO_V2_FLAG[2] / SYNC_C1_TO_V2_FLAG[2]  (mm2/mm1 结果就绪)
  正向 V->C : SYNC_V3_TO_C3_FLAG / SYNC_V4_TO_C5_FLAG        (dS/P 已落 L1)
  反向 C->V : SYNC_C4_TO_V3_FLAG / SYNC_C5_TO_V4_FLAG        (L1 已被 cube 读完，可覆写)
反向同步由 needSyncDkMM 控制：首轮 L1 尚无人读，无需等待。
AscendC 中 AIC 侧对每个 V->C flag 额外等待 `16 + flag`(两个 vector 子核各发一次)，
在 pypto 中等价于 sync_mode=INTRA_BLOCK 的单次 wait。

数据流与 AscendC 对齐：
  MM2(Q@K^T)->UB -> V2(softmax->P) -> V4(P cast+ND2NZ->L1)
  MM1(dO@V^T)->UB -> V3(dS=(dP-sfmg)*P cast+ND2NZ->L1)
  -> MM5(P^T@dO->dV) -> MM3(dS@K->dQ) -> MM4(dS^T@Q->dK)
V1 计算 softmaxGradFront sfmg = rowsum(dy * y)，取自前向输出 y，
对应 vector_api/cast_softmax_grad.h 的 CopyInSoftmaxGrad + MySoftmaxGradFrontCast。

MM4/MM5 的左矩阵需要转置(dS^T / P^T)。AscendC 通过 L1->L0A 载入时置
isLeftTranspose=true 免费得到；pypto 无此能力(NZ 版本静默算错、ZZ 版本
无法编译，L0A 要求非 row-major fractal)，因此显式走
transpose(UB) -> ND2NZ -> insert 到 [N,M] NZ 的 L1，再常规载入 L0A。
"""

from dataclasses import dataclass

import pypto_pro.language as pl

# `vf` 由 parser 按语法识别(_call_parser.py: attrs[0] == "vf")，无需 import


# ---- 基本块尺寸 (对应 s1/s2/dTemplateType) ----
CUBE_BASEM = 128          # s1TemplateType::Aligned128
CUBE_BASEN = 128          # s2TemplateType::Aligned128
HEAD_DIM_ALIGN = 128      # dTemplateType::Aligned128
CV_CORE_RATIO = 2
VECTOR_BASEM = CUBE_BASEM // CV_CORE_RATIO   # 64，每个 vector 子核处理半个 S1
VECTOR_BASEN = CUBE_BASEN

# 单个 VF 寄存器的 fp32 lane 数
VF_LANES = 64

# V1 的 sfmg 分段行数：prod/tmp 两块 fp32 缓冲按此尺寸开，控制 UB 占用
SFMG_LOOP_SIZE = VECTOR_BASEM // 2           # 32

# 兼容测试脚本导出的别名
TILE_M = CUBE_BASEM
TILE_N = CUBE_BASEN
TILE_D = HEAD_DIM_ALIGN

# ---- cross core sync flags，取值与 flash_attention_score_grad_common.h 一致 ----
SYNC_C1_TO_V2_FLAG = [0, 1]   # mm1(dO@V^T) 结果就绪，按 taskId&1 ping-pong
SYNC_C2_TO_V2_FLAG = [2, 3]   # mm2(Q@K^T)  结果就绪，按 taskId&1 ping-pong
SYNC_V3_TO_C3_FLAG = 4        # dS 已由 UB 落入 L1，dq/dk 可以开算
SYNC_V4_TO_C5_FLAG = 5        # P  已由 UB 落入 L1，dv 可以开算
SYNC_C4_TO_V3_FLAG = 8        # 反向：dqk 已读完 dS L1，vector 可覆写
SYNC_C5_TO_V4_FLAG = 9        # 反向：dv  已读完 P  L1，vector 可覆写

# ---- L1 地址 (MutexBufferManager<BufferType::L1>，总量 512KB) ----
# dSL1Buf / pL1Buf 对应 kernel_base.h InitCVCommonBuffer 中的同名 buffer
L1_Q       = 0x00000   # [128,128] fp16 = 32KB
L1_K       = 0x08000
L1_V       = 0x10000
L1_DO      = 0x18000
L1_DS      = 0x20000   # dSL1Buf : [M,N] 供 MM3 左矩阵
L1_P       = 0x28000   # pL1Buf
L1_DS_T    = 0x30000   # dS^T [N,M] 供 MM4 左矩阵
L1_P_T     = 0x38000   # P^T  [N,M] 供 MM5 左矩阵
# MM3(dQ = dS@K) 的右矩阵要非转置的 K[S2,D]，必须独占一块 L1：
# 同一地址上叠加两种 shape/layout 视图(ZN 的 [D,S2] 与 ND 的 [S2,D])
# 能编过但结果静默错误。
L1_K_ND    = 0x40000

# ---- L0A / L0B / L0C ----
L0A_ADDR   = 0x0000
L0B_ADDR   = 0x0000
L0C_ADDR   = 0x0000

# ---- UB 地址，硬上限 248KB；实测总占用 242.5KB ----
# mm1ResBuf[2] / mm2ResBuf[2] 与 AscendC 同名同尺寸(VECTOR_BASEM*VECTOR_BASEN*4)
UB_MM1_0   = 0x00000   # fp32[64,128] 32KB
UB_MM1_1   = 0x08000
UB_MM2_0   = 0x10000
UB_MM2_1   = 0x18000
# V1 按 SFMG_LOOP_SIZE 行分段，避免 prod+tmp 两块 [64,128] fp32 撑爆 UB
UB_Y       = 0x20000   # fp16[32,128]  8KB  V1 前向输出 y
UB_DY      = 0x22000   # fp16[32,128]  8KB  V1 的 dy
UB_PROD    = 0x24000   # fp32[32,128] 16KB  dy*y 的逐元素积
UB_TMP     = 0x28000   # fp32[32,128] 16KB  pl.sum 要求的同尺寸 workspace
UB_ND      = 0x2C000   # fp16[64,128] 16KB  cast 结果(ND)
UB_NZ      = 0x30000   # fp16[64,128] 16KB  ND2NZ 结果
UB_TND     = 0x34000   # fp16[128,64] 16KB  transpose 结果(ND)
UB_TNZ     = 0x38000   # fp16[128,64] 16KB  转置后的 ND2NZ 结果
# sfmg 两个分段结果地址连续，故可用一个 [64,1] 视图整体读取
UB_SFMG    = 0x3C000   # fp32[64,1]  256B  softmaxGradResBuf
UB_SFMG_HI = 0x3C080   # 后 32 行 (32*4 = 128B 偏移)
UB_MAX_0   = 0x3C200   # maxSumQue[2] 双缓冲
UB_MAX_1   = 0x3C400
UB_SUM_0   = 0x3C600
UB_SUM_1   = 0x3C800
# 末尾 0x3CA00 = 243KB，未超 248KB 上限


@dataclass
class FagTiling:
    b: int
    n: int
    s1: int
    s2: int
    d: int
    scale_value: float
    # 参与分核的物理核数。必须由 host 传入而非用 get_block_num() 推导：
    # AIC/AIV 两侧的 block 计数口径不同(AIV 按 vector 子核计数)，
    # 两侧算出的 [blockStart, blockEnd) 一旦不一致，set/wait 就无法配对。
    core_num: int


# ==================================================================
#  SetConstInfo —— 对应 kernel_base.h::SetConstInfo
# ==================================================================
def set_const_info(tiling):
    s1Outer = (tiling.s1 + CUBE_BASEM - 1) // CUBE_BASEM
    s2Outer = (tiling.s2 + CUBE_BASEN - 1) // CUBE_BASEN
    return pl.struct(
        "FagConstInfo",
        bSize=tiling.b,
        n2Size=tiling.n,
        s1Size=tiling.s1,
        s2Size=tiling.s2,
        dSize=tiling.d,
        scaleValue=tiling.scale_value,
        s1Outer=s1Outer,
        s2Outer=s2Outer,
        s1CvTail=tiling.s1 - (s1Outer - 1) * CUBE_BASEM,
        s2CvTail=tiling.s2 - (s2Outer - 1) * CUBE_BASEN,
        # BSND: [B,S,N,D] 的行距，mm2Ka/mm2Kb 即 AscendC 同名字段
        n2D=tiling.n * tiling.d,
        s1oS2o=s1Outer * s2Outer,
        n2GS1oS2o=tiling.n * s1Outer * s2Outer,
        totalBlockNum=tiling.b * tiling.n * s1Outer * s2Outer,
        coreNum=tiling.core_num,
    )


def make_run_infos():
    """FagRunInfo runInfos[2]，用于 cv ping pong。

    必须以字面量构造后再逐字段赋值：struct 字段按 int64_t 生成，
    直接用派生表达式(uint64_t)初始化会触发 -Wc++11-narrowing。
    """
    return pl.struct_array(
        2,
        "FagRunInfo",
        boIdx=0,
        n2oIdx=0,
        s1oIdx=0,
        s2oIdx=0,
        s1RealSize=0,
        s2RealSize=0,
        halfS1RealSize=0,
        firstHalfS1RealSize=0,
        taskIdMod2=0,
        queryOffset=0,
        keyOffset=0,
        maxSumOffset=0,
        isS2IdxNoChange=0,
    )


# ==================================================================
#  IsValid + GetNextValidIdx —— 无 attenMask 时 IsValid 恒真，
#  故仅剩 blockEnds 边界判断（对应 kernel_base.h 的 else 分支）
# ==================================================================
def get_next_valid_idx(constInfo, blockInnerIdx, blockEnd):
    # 无 attenMask 时 IsValid 恒真，无需扫描：块索引已越界即无效。
    # AscendC 用返回 -1 表示无效；此处返回 clamp 后的索引，
    # 由调用方用 `blockInnerIdx >= blockEnd` 判断是否为最后一轮
    # (codegen 不支持 IfStmt 返回值的 phi 合并，也不允许 bool 参与算术)
    return pl.min(blockInnerIdx, blockEnd)


# ==================================================================
#  SetRunInfo —— 对应 kernel_base.h::SetRunInfo 非 TND 分支 + SetAxisRunInfo
#  以 mutate 方式写 runInfo，与 AscendC 的引用传参语义一致
# ==================================================================
def set_run_info(constInfo, runInfo, taskId, index, subIdx):
    # index -> (boIdx, n2oIdx, s2oIdx, s1oIdx)，S1 为最快轴
    bDimTail = index % constInfo.n2GS1oS2o
    gDimTail = index % constInfo.s1oS2o
    runInfo.boIdx = index // constInfo.n2GS1oS2o
    runInfo.n2oIdx = bDimTail // constInfo.s1oS2o
    runInfo.s2oIdx = gDimTail // constInfo.s1Outer
    runInfo.s1oIdx = gDimTail % constInfo.s1Outer

    # 尾块真实大小。不能用分支赋值(IfStmt 无法 phi 合并)，也不能用比较结果
    # 参与算术(bool dtype 不被 mul 接受)，故写成 min 截断的形式
    s1Real = pl.min(CUBE_BASEM, constInfo.s1Size - runInfo.s1oIdx * CUBE_BASEM)
    s2Real = pl.min(CUBE_BASEN, constInfo.s2Size - runInfo.s2oIdx * CUBE_BASEN)
    runInfo.s1RealSize = s1Real
    runInfo.s2RealSize = s2Real
    runInfo.taskIdMod2 = taskId % 2

    # vector 侧两个子核各承担半个 S1：
    # subIdx=0 取 firstHalf，subIdx=1 取剩余部分
    firstHalf = (s1Real + 1) // 2
    runInfo.firstHalfS1RealSize = firstHalf
    runInfo.halfS1RealSize = s1Real - subIdx * firstHalf - (1 - subIdx) * (s1Real - firstHalf)

    # BSND 布局的 GM 偏移
    runInfo.queryOffset = runInfo.s1oIdx * CUBE_BASEM
    runInfo.keyOffset = runInfo.s2oIdx * CUBE_BASEN
    # softmax_max / softmax_sum 排布 [B,N,S1]
    runInfo.maxSumOffset = runInfo.boIdx * constInfo.n2Size + runInfo.n2oIdx

    # isS2IdxNoChange 决定 dK/dV 能否在 L0C 上累加；本配置
    # IS_DKV_RESIDENT_L0C=false，仅保留字段以对齐结构。
    # 由 index-1 直接推出上一块的 s2oIdx，避免跨迭代的可变量
    # (循环里带条件赋值的标量会触发 IfStmt phi 合并失败)
    prevS2oIdx = (index - 1) % constInfo.s1oS2o // constInfo.s1Outer
    runInfo.isS2IdxNoChange = runInfo.s2oIdx == prevS2oIdx


# ==================================================================
#  IterateMmQK —— mm2: S = Q @ K^T   (block_cube.h::IterateMmQK)
# ==================================================================
def iterate_mm_qk(constInfo, runInfo, tensor_q, tensor_k,
                  q_l1, k_l1, left, right, acc, mm2_ub):
    q_t = q_l1.current()
    k_t = k_l1.current()
    # BSND -> tile[S,D]: order=[1,3] 取 (S,D) 两轴
    pl.load(q_t, tensor_q, [runInfo.boIdx, runInfo.queryOffset, runInfo.n2oIdx, 0],
            order=[1, 3])
    # 右矩阵转置(isRightTranspose=true)：K[S2,D] 以 [D,S2] 载入
    pl.load(k_t, tensor_k, [runInfo.boIdx, runInfo.keyOffset, runInfo.n2oIdx, 0],
            order=[3, 1])
    lt = left.current()
    rt = right.current()
    at = acc.current()
    pl.move(lt, q_t)
    pl.move(rt, k_t)
    pl.matmul(at, lt, rt)
    # Fixpipe L0C->UB，dualDstCtl=1：按 M 切分写入两个 vector 子核
    pl.move(mm2_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


# ==================================================================
#  IterateMmDyV —— mm1: dP = dO @ V^T  (block_cube.h::IterateMmDyV)
# ==================================================================
def iterate_mm_dyv(constInfo, runInfo, tensor_dy, tensor_v,
                   do_l1, v_l1, left, right, acc, mm1_ub):
    do_t = do_l1.current()
    v_t = v_l1.current()
    pl.load(do_t, tensor_dy, [runInfo.boIdx, runInfo.queryOffset, runInfo.n2oIdx, 0],
            order=[1, 3])
    pl.load(v_t, tensor_v, [runInfo.boIdx, runInfo.keyOffset, runInfo.n2oIdx, 0],
            order=[3, 1])
    lt = left.current()
    rt = right.current()
    at = acc.current()
    pl.move(lt, do_t)
    pl.move(rt, v_t)
    pl.matmul(at, lt, rt)
    pl.move(mm1_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


# ==================================================================
#  IterateMmDsK —— mm3: dQ = dS @ K    (block_cube.h::IterateMmDsK)
#  IS_DQ_WRITE_UB=false -> fixpipe 直出 GM，needAtomic=true(BN2GS1S2)
# ==================================================================
def iterate_mm_dsk(constInfo, runInfo, tensor_k, tensor_dq_ws,
                   ds_l1, k_l1, left, right, acc):
    k_t = k_l1.current()
    # 右矩阵不转置(isRightTranspose=false)：K 以 [S2,D] 载入
    pl.load(k_t, tensor_k, [runInfo.boIdx, runInfo.keyOffset, runInfo.n2oIdx, 0],
            order=[1, 3])
    lt = left.current()
    rt = right.current()
    at = acc.current()
    pl.move(lt, ds_l1)
    pl.move(rt, k_t)
    pl.matmul(at, lt, rt)
    pl.store(tensor_dq_ws, at,
             [runInfo.boIdx, runInfo.queryOffset, runInfo.n2oIdx, 0],
             order=[1, 3], atomic=pl.AtomicType.AtomicAdd)


# ==================================================================
#  IterateMmDsQ —— mm4: dK = dS^T @ Q  (block_cube.h::IterateMmDsQ)
#  左矩阵取已在 vector 侧转置好的 dS^T L1
# ==================================================================
def iterate_mm_dsq(constInfo, runInfo, tensor_q, tensor_dk_ws,
                   ds_t_l1, q_l1, left, right, acc):
    q_t = q_l1.current()
    pl.load(q_t, tensor_q, [runInfo.boIdx, runInfo.queryOffset, runInfo.n2oIdx, 0],
            order=[1, 3])
    lt = left.current()
    rt = right.current()
    at = acc.current()
    pl.move(lt, ds_t_l1)
    pl.move(rt, q_t)
    pl.matmul(at, lt, rt)
    pl.store(tensor_dk_ws, at,
             [runInfo.boIdx, runInfo.keyOffset, runInfo.n2oIdx, 0],
             order=[1, 3], atomic=pl.AtomicType.AtomicAdd)


# ==================================================================
#  IterateMmPDy —— mm5: dV = P^T @ dO  (block_cube.h::IterateMmPDy)
# ==================================================================
def iterate_mm_pdy(constInfo, runInfo, tensor_dy, tensor_dv_ws,
                   p_t_l1, do_l1, left, right, acc):
    do_t = do_l1.current()
    pl.load(do_t, tensor_dy, [runInfo.boIdx, runInfo.queryOffset, runInfo.n2oIdx, 0],
            order=[1, 3])
    lt = left.current()
    rt = right.current()
    at = acc.current()
    pl.move(lt, p_t_l1)
    pl.move(rt, do_t)
    pl.matmul(at, lt, rt)
    pl.store(tensor_dv_ws, at,
             [runInfo.boIdx, runInfo.keyOffset, runInfo.n2oIdx, 0],
             order=[1, 3], atomic=pl.AtomicType.AtomicAdd)


# ==================================================================
#  CopyUB2L1 —— 对应 block_vec.h::CopyUB2L1
#  pypto 无 Vec->Mat 的 move，必须 cast -> ND2NZ(Vec->Vec) -> insert
#  insert 的源必须已是 NZ，否则能编过但结果错
# ==================================================================
def copy_ub2l1(l1_tile, nd_ub, nz_ub, row_offset):
    pl.move(nz_ub, nd_ub)                 # Vec->Vec 完成 ND->NZ transdata
    pl.insert(l1_tile, nz_ub, [row_offset, 0])


# ==================================================================
#  ProcessVec1 —— v1: sfmg = rowsum(dy * y)
#  对应 vector_api/cast_softmax_grad.h::CopyInSoftmaxGrad +
#  MySoftmaxGradFrontCast。y 为前向输出，按 SFMG_LOOP_SIZE 分段。
# ==================================================================
@pl.vector_function
def softmax_grad_front_mul_vf(prod_ub, y_ub, dy_ub, srcM: pl.DT_INT64):
    """prod = cast(dy) * cast(y)，逐行做逐元素乘（不含 reduce）。

    对应 AscendC CastAligned256F16VF128 的乘法部分：fp16 一次载入 128 个数，
    按 Even/Odd 拆成两个 64 lane 的 fp32 寄存器分别相乘。
    行内求和交给 tile 级 pl.sum：VF 的 store 目标地址不能随运行期循环变量
    变化（实测 507035），而逐行标量写回正需要这种寻址。
    注意 astype(ZERO)/astype(ONE) 得到的是「去交错」的偶/奇两半，
    所以两个乘积寄存器要按 even/odd 半区写回，保持与源同序。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, m * HEAD_DIM_ALIGN)
        vregDyHalf = vf.load_align(dy_ub, m * HEAD_DIM_ALIGN)
        vregY = vf.astype(vregYHalf, pregFullExeB16, dtype=pl.DT_FP32,
                          layout=pl.CastLayout.ZERO)
        vregY1 = vf.astype(vregYHalf, pregFullExeB16, dtype=pl.DT_FP32,
                           layout=pl.CastLayout.ONE)
        vregDy = vf.astype(vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32,
                           layout=pl.CastLayout.ZERO)
        vregDy1 = vf.astype(vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32,
                            layout=pl.CastLayout.ONE)
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        # 按显式行偏移写回，两半分别落在行内前/后 64 个位置。
        # 不能用 post_update：它跨迭代累加地址，实测得到量级失控的脏数据。
        # 偏移必须是寄存器对齐的(m*128、+64 都对齐)；逐元素求和对元素
        # 顺序不敏感，故偶/奇两半无需还原交错顺序。
        vf.store_align(prod_ub + m * HEAD_DIM_ALIGN, vregMul, pregFullExe)
        vf.store_align(prod_ub + m * HEAD_DIM_ALIGN + VF_LANES, vregMul2,
                       pregFullExe)


def process_vec1_chunk(constInfo, runInfo, s1Base, rowOff, rows,
                       tensor_y, tensor_dy, y_ub, dy_ub, prod_ub, tmp_ub,
                       sfmg_chunk_ub):
    pl.load(y_ub, tensor_y,
            [runInfo.boIdx, s1Base + rowOff, runInfo.n2oIdx, 0], order=[1, 3])
    pl.load(dy_ub, tensor_dy,
            [runInfo.boIdx, s1Base + rowOff, runInfo.n2oIdx, 0], order=[1, 3])
    softmax_grad_front_mul_vf(prod_ub, y_ub, dy_ub, rows)
    # dim=0：沿最后一轴(D)做行向 reduce，得到 [rows, 1]
    pl.sum(sfmg_chunk_ub, prod_ub, tmp_ub, dim=0)


def process_vec1(constInfo, runInfo, subIdx, tensor_y, tensor_dy,
                 y_ub, dy_ub, prod_ub, tmp_ub, sfmg_lo_ub, sfmg_hi_ub):
    """softmaxGradFront: sfmg = rowsum(dy * y)。

    按 SFMG_LOOP_SIZE 行分两段：两段的 sfmg 结果地址连续，
    上层用一个 [VECTOR_BASEM,1] 视图整体参与后续 broadcast。
    分段数是编译期常量，但 kernel 内 for 必须用 pl.range，
    故直接展开两次调用。
    """
    s1Base = runInfo.queryOffset + subIdx * runInfo.firstHalfS1RealSize
    rowsLo = pl.min(SFMG_LOOP_SIZE, runInfo.halfS1RealSize)
    rowsHi = runInfo.halfS1RealSize - rowsLo
    process_vec1_chunk(constInfo, runInfo, s1Base, 0, rowsLo,
                       tensor_y, tensor_dy, y_ub, dy_ub, prod_ub, tmp_ub,
                       sfmg_lo_ub)
    process_vec1_chunk(constInfo, runInfo, s1Base, SFMG_LOOP_SIZE, rowsHi,
                       tensor_y, tensor_dy, y_ub, dy_ub, prod_ub, tmp_ub,
                       sfmg_hi_ub)


# ==================================================================
#  ProcessVec2 —— v2: P = simpleSoftmax(S * scale, max, sum)
#  对应 vector_api/pse_atten_mask_muls_simple_softmax.h
#  无 pse / attenMask，仅 muls + simpleSoftmax
# ==================================================================
@pl.vector_function
def muls_simple_softmax_vf(mm2_ub, max_ub, sum_ub, scale: pl.DT_FP32,
                           srcM: pl.DT_INT64):
    """P = exp(S * scale - max) / sum，逐行处理。

    对应 AscendC MulsSelVF 去掉 pse/attenMask 后的主体：max/sum 用
    BRC_B32 广播载入，S 按 VECTOR_BASEN=128 分两个 64 lane 寄存器。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregMax = vf.load_align(max_ub, m, dist=pl.LoadDist.BRC_B32)
        vregSum = vf.load_align(sum_ub, m, dist=pl.LoadDist.BRC_B32)
        for n in pl.range(0, VECTOR_BASEN // VF_LANES):
            vregSrc = vf.load_align(mm2_ub, m * VECTOR_BASEN + n * VF_LANES)
            vregMuls = vf.muls(vregSrc, scale, pregFullExe)
            vregExp = vf.exp_sub(vregMuls, vregMax, pregFullExe)
            vregDiv = vf.div(vregExp, vregSum, pregFullExe)
            vf.store_align(mm2_ub + m * VECTOR_BASEN + n * VF_LANES,
                           vregDiv, pregFullExe)


def process_vec2(constInfo, runInfo, subIdx, tensor_smax, tensor_ssum,
                 mm2_ub, max_ub, sum_ub):
    s1Base = runInfo.queryOffset + subIdx * runInfo.firstHalfS1RealSize
    # CopyMaxSum: 前向的 max/sum 直接复用，无需再求极值
    pl.load(max_ub, tensor_smax, [runInfo.maxSumOffset, s1Base], order=[1, 0])
    pl.load(sum_ub, tensor_ssum, [runInfo.maxSumOffset, s1Base], order=[1, 0])
    muls_simple_softmax_vf(mm2_ub, max_ub, sum_ub, constInfo.scaleValue,
                           runInfo.halfS1RealSize)


# ==================================================================
#  ProcessVec3 —— v3: dS = (dP - sfmg) * P，随后 cast + ND2NZ 落 L1
#  对应 block_vec.h::ProcessVec3 的 BroadcastSubMul +
#  CastTransdataDeconflict + CopyUBToL1Vec3
#  同时产出 dS^T L1，供 MM4 使用
# ==================================================================
@pl.vector_function
def broadcast_sub_mul_cast_vf(nd_ub, mm1_ub, mm2_ub, sfmg_ub,
                              scale: pl.DT_FP32, srcM: pl.DT_INT64):
    """dS = (dP - broadcast(sfmg)) * P，并 cast 成 fp16。

    对应 AscendC BroadcastSubMulVF128 + CastTransdataDeconflict 的算术部分：
    sfmg 按行 BRC_B32 广播，128 列拆成两个 64 lane 寄存器；
    两个 fp32 寄存器 cast 后交错写回一个 fp16 行。
    注意此处不乘 scale，与 AscendC 一致(scale 在 dQ/dK 后处理施加)。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for m in pl.range(0, srcM):
        vregSfmg = vf.load_align(sfmg_ub, m, dist=pl.LoadDist.BRC_B32)
        # 用 DINTLV_B32 一次载入 128 个 fp32 并拆成 even/odd，
        # 使后续 Even/Odd cast + Or 的合并顺序正确；
        # sub/mul 为逐元素运算，只要 dP/P 拆分方式一致即可
        vregDpEven, vregDpOdd = vf.load_align(mm1_ub, m * VECTOR_BASEN,
                                              dist=pl.LoadDist.DINTLV_B32)
        vregPEven, vregPOdd = vf.load_align(mm2_ub, m * VECTOR_BASEN,
                                            dist=pl.LoadDist.DINTLV_B32)
        vregSub0 = vf.sub(vregDpEven, vregSfmg, pregFullExe)
        vregDs0 = vf.mul(vregSub0, vregPEven, pregFullExe)
        vregSub1 = vf.sub(vregDpOdd, vregSfmg, pregFullExe)
        vregDs1 = vf.mul(vregSub1, vregPOdd, pregFullExe)
        # dS 乘 scale：AscendC 在 workspace 后处理里对 dQ/dK(非 dV) 做
        # Muls(scaleValue)(block_vec.h:828 `MM_IDX != DV_IDX`)，
        # 而 fp32 输入路径则直接在 ProcessVec3 里乘 mm1Res(block_vec.h:619)。
        # dS 只流向 dQ/dK，故在此乘等价且省掉一趟后处理。
        vregMul0 = vf.muls(vregDs0, scale, pregFullExe)
        vregMul1 = vf.muls(vregDs1, scale, pregFullExe)
        # fp32 -> fp16：Even/Odd 分别 cast 进 b16 寄存器的两个半区再 Or 合并，
        # 与 vf_cast_transdata_deconflict.h 一致（probe 验证 MAXDIFF 0.0）
        vregCastEven = vf.astype(vregMul0, pregFullExeB16, dtype=pl.DT_FP16,
                                 layout=pl.CastLayout.ZERO)
        vregCastOdd = vf.astype(vregMul1, pregFullExeB16, dtype=pl.DT_FP16,
                                layout=pl.CastLayout.ONE)
        vregCastRes = vf.or_(vregCastEven, vregCastOdd, pregFullExeB16)
        vf.store_align(nd_ub + m * VECTOR_BASEN, vregCastRes, pregFullExeB16)


def process_vec3(constInfo, runInfo, subIdx, ds_l1, ds_t_l1,
                 mm1_ub, mm2_ub, sfmg_ub, nd_ub, nz_ub, tnd_ub, tnz_ub):
    rowOff = subIdx * runInfo.firstHalfS1RealSize
    broadcast_sub_mul_cast_vf(nd_ub, mm1_ub, mm2_ub, sfmg_ub,
                              constInfo.scaleValue, runInfo.halfS1RealSize)
    copy_ub2l1(ds_l1, nd_ub, nz_ub, rowOff)
    # MM4 左矩阵 dS^T：pypto 无 isLeftTranspose，显式转置
    pl.transpose(tnd_ub, nd_ub)
    pl.move(tnz_ub, tnd_ub)
    pl.insert(ds_t_l1, tnz_ub, [0, rowOff])


# ==================================================================
#  ProcessVec4 —— v4: P cast + ND2NZ 落 L1 (block_vec.h::ProcessVec4)
#  同时产出 P^T L1，供 MM5 使用
# ==================================================================
@pl.vector_function
def cast_p_vf(nd_ub, mm2_ub, srcM: pl.DT_INT64):
    """P: fp32 -> fp16，对应 AscendC ProcessVec4 中的 CastTransdataDeconflict。"""
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for m in pl.range(0, srcM):
        vregPEven, vregPOdd = vf.load_align(mm2_ub, m * VECTOR_BASEN,
                                            dist=pl.LoadDist.DINTLV_B32)
        vregCastEven = vf.astype(vregPEven, pregFullExeB16, dtype=pl.DT_FP16,
                                 layout=pl.CastLayout.ZERO)
        vregCastOdd = vf.astype(vregPOdd, pregFullExeB16, dtype=pl.DT_FP16,
                                layout=pl.CastLayout.ONE)
        vregCastRes = vf.or_(vregCastEven, vregCastOdd, pregFullExeB16)
        vf.store_align(nd_ub + m * VECTOR_BASEN, vregCastRes, pregFullExeB16)


def process_vec4(constInfo, runInfo, subIdx, p_l1, p_t_l1,
                 mm2_ub, nd_ub, nz_ub, tnd_ub, tnz_ub):
    rowOff = subIdx * runInfo.firstHalfS1RealSize
    cast_p_vf(nd_ub, mm2_ub, runInfo.halfS1RealSize)
    copy_ub2l1(p_l1, nd_ub, nz_ub, rowOff)
    # MM5 左矩阵 P^T：同 dS^T，显式转置
    pl.transpose(tnd_ub, nd_ub)
    pl.move(tnz_ub, tnd_ub)
    pl.insert(p_t_l1, tnz_ub, [0, rowOff])


@pl.jit(auto_mutex=True)
def flash_attention_score_grad(
    query: pl.Ptr[pl.DT_UINT8],
    key: pl.Ptr[pl.DT_UINT8],
    value: pl.Ptr[pl.DT_UINT8],
    dy: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    softmax_max: pl.Ptr[pl.DT_UINT8],
    softmax_sum: pl.Ptr[pl.DT_UINT8],
    dq_workspace: pl.Ptr[pl.DT_UINT8],
    dk_workspace: pl.Ptr[pl.DT_UINT8],
    dv_workspace: pl.Ptr[pl.DT_UINT8],
    dbg_sfmg: pl.Ptr[pl.DT_UINT8],
    dbg_ident: pl.Ptr[pl.DT_UINT8],
    dbg_ds: pl.Ptr[pl.DT_UINT8],
    tiling: FagTiling,
):
    constInfo = set_const_info(tiling)

    s1nd = constInfo.s1Size * constInfo.n2D
    s2nd = constInfo.s2Size * constInfo.n2D

    # 输入按 BSND [B,S,N,D] 建 tensor
    tensor_q = pl.make_tensor(query, [constInfo.bSize, constInfo.s1Size, constInfo.n2Size, HEAD_DIM_ALIGN],
                              [s1nd, constInfo.n2D, HEAD_DIM_ALIGN, 1], dtype=pl.DT_FP16)
    tensor_k = pl.make_tensor(key, [constInfo.bSize, constInfo.s2Size, constInfo.n2Size, HEAD_DIM_ALIGN],
                              [s2nd, constInfo.n2D, HEAD_DIM_ALIGN, 1], dtype=pl.DT_FP16)
    tensor_v = pl.make_tensor(value, [constInfo.bSize, constInfo.s2Size, constInfo.n2Size, HEAD_DIM_ALIGN],
                              [s2nd, constInfo.n2D, HEAD_DIM_ALIGN, 1], dtype=pl.DT_FP16)
    tensor_dy = pl.make_tensor(dy, [constInfo.bSize, constInfo.s1Size, constInfo.n2Size, HEAD_DIM_ALIGN],
                               [s1nd, constInfo.n2D, HEAD_DIM_ALIGN, 1], dtype=pl.DT_FP16)
    tensor_y = pl.make_tensor(y, [constInfo.bSize, constInfo.s1Size, constInfo.n2Size, HEAD_DIM_ALIGN],
                              [s1nd, constInfo.n2D, HEAD_DIM_ALIGN, 1], dtype=pl.DT_FP16)
    # softmax_max / softmax_sum : [B,N,S1]
    tensor_smax = pl.make_tensor(softmax_max, [constInfo.bSize * constInfo.n2Size, constInfo.s1Size],
                                 [constInfo.s1Size, 1], dtype=pl.DT_FP32)
    tensor_ssum = pl.make_tensor(softmax_sum, [constInfo.bSize * constInfo.n2Size, constInfo.s1Size],
                                 [constInfo.s1Size, 1], dtype=pl.DT_FP32)
    # dq/dk/dv workspace 为 fp32，供 atomicAdd 累加
    tensor_dq_ws = pl.make_tensor(dq_workspace, [constInfo.bSize, constInfo.s1Size, constInfo.n2Size, HEAD_DIM_ALIGN],
                                  [s1nd, constInfo.n2D, HEAD_DIM_ALIGN, 1], dtype=pl.DT_FP32)
    tensor_dk_ws = pl.make_tensor(dk_workspace, [constInfo.bSize, constInfo.s2Size, constInfo.n2Size, HEAD_DIM_ALIGN],
                                  [s2nd, constInfo.n2D, HEAD_DIM_ALIGN, 1], dtype=pl.DT_FP32)
    tensor_dv_ws = pl.make_tensor(dv_workspace, [constInfo.bSize, constInfo.s2Size, constInfo.n2Size, HEAD_DIM_ALIGN],
                                  [s2nd, constInfo.n2D, HEAD_DIM_ALIGN, 1], dtype=pl.DT_FP32)
    # 调试用：sfmg 落回 GM。刻意排成 [S1, B*N]，使 [64,1] 的 tile 按
    # 默认(升序)轴映射即可写入 —— load/store 的 order 都必须升序，
    # 不能用 order=[1,0] 做转置写。
    tensor_dbg_sfmg = pl.make_tensor(dbg_sfmg,
                                     [constInfo.s1Size, constInfo.bSize * constInfo.n2Size],
                                     [constInfo.bSize * constInfo.n2Size, 1],
                                     dtype=pl.DT_FP32)

    # cBlockIdx 必须按物理核取：AIV 侧 get_block_idx() 按 vector 子核计数
    # (0..2*coreNum-1)，AIC 侧按物理核计数。两侧口径不一致会算出不同的
    # [blockStart, blockEnd)，循环次数不同 -> set/wait 配不上 -> 死锁。
    # 核数直接用 tiling 传入，避免再依赖 get_block_num 的口径。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum

    # 本核负责的 [blockStart, blockEnd)，对应 s1s2BNGS1S2BlockNumList
    blockPerCore = (constInfo.totalBlockNum + coreNum - 1) // coreNum
    blockStart = cBlockIdx * blockPerCore
    blockEnd = pl.min(blockStart + blockPerCore, constInfo.totalBlockNum)

    # ---- CV 共享 UB：mm1ResBuf[2] / mm2ResBuf[2]，按 taskIdMod2 ping-pong ----
    mm1_res = pl.make_tile_group(
        type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[UB_MM1_0, UB_MM1_1], mutex_ids=[0, 1])
    mm2_res = pl.make_tile_group(
        type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[UB_MM2_0, UB_MM2_1], mutex_ids=[2, 3])

    # ---- CV 共享 L1：dSL1Buf / pL1Buf 及其转置副本 ----
    # 对应 kernel_base.h 的「CV核间共享Buffer」，vector 侧写、cube 侧读，
    # 故必须声明在两个 section 之外
    ds_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_BASEM, CUBE_BASEN], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1),
        addrs=[L1_DS], mutex_ids=[8])
    p_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_BASEM, CUBE_BASEN], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1),
        addrs=[L1_P], mutex_ids=[9])
    ds_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_BASEN, CUBE_BASEM], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1),
        addrs=[L1_DS_T], mutex_ids=[10])
    p_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_BASEN, CUBE_BASEM], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1),
        addrs=[L1_P_T], mutex_ids=[11])

    with pl.section_cube():
        q_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, HEAD_DIM_ALIGN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=[L1_Q], mutex_ids=[4])
        k_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_DIM_ALIGN, CUBE_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=[L1_K], mutex_ids=[5])
        # MM3 的右矩阵：非转置 K[S2,D]，独占地址与 mutex。
        # 早前与 ZN 的 k_l1 共用 L1_K/mutex 5，两种 layout 叠在同一地址上，
        # 编译通过但 dQ 结果静默错误。
        k_l1_n = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, HEAD_DIM_ALIGN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=[L1_K_ND], mutex_ids=[30])
        v_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_DIM_ALIGN, CUBE_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=[L1_V], mutex_ids=[6])
        do_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, HEAD_DIM_ALIGN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=[L1_DO], mutex_ids=[7])
        left = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, CUBE_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Left, compact=1),
            addrs=[L0A_ADDR], mutex_ids=[12])
        right = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, HEAD_DIM_ALIGN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Right, compact=1),
            addrs=[L0B_ADDR], mutex_ids=[13])
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, CUBE_BASEN], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[L0C_ADDR], mutex_ids=[14])

        runInfos = make_run_infos()
        # ---------------- Process() 主循环 (AIC 侧) ----------------
        taskId = 0
        blockInnerIdx = get_next_valid_idx(constInfo, blockStart, blockEnd)
        for _ in pl.range(0, blockEnd - blockStart + 1):
            isLastLoop = blockInnerIdx >= blockEnd
            # 无条件轮转，保证与 vector 侧步调一致（末轮不做 mm 也要转）
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            if not isLastLoop:
                curInfo = runInfos[taskId % 2]
                set_run_info(constInfo, curInfo, taskId, blockInnerIdx, 0)

                # ---- mm2: Q@K^T ----
                iterate_mm_qk(constInfo, curInfo, tensor_q, tensor_k,
                              q_l1, k_l1, left, right, acc, mm2_cur)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                         event_id=SYNC_C2_TO_V2_FLAG[taskId % 2])
                # ---- mm1: dO@V^T ----
                iterate_mm_dyv(constInfo, curInfo, tensor_dy, tensor_v,
                               do_l1, v_l1, left, right, acc, mm1_cur)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                         event_id=SYNC_C1_TO_V2_FLAG[taskId % 2])

            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                # 正向：等 vector 把 P / dS 写入 L1
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                          event_id=SYNC_V4_TO_C5_FLAG)
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                          event_id=SYNC_V3_TO_C3_FLAG)
                # ---- mm5: dV = P^T@dO ----
                iterate_mm_pdy(constInfo, prevInfo, tensor_dy, tensor_dv_ws,
                               p_t_l1.current(), do_l1, left, right, acc)
                # 反向：P 的 L1 已读完，放行 vector 覆写
                pl.system.set_cross_core(pipe=pl.PipeType.MTE1,
                                         event_id=SYNC_C5_TO_V4_FLAG)
                # ---- mm3: dQ = dS@K ----
                iterate_mm_dsk(constInfo, prevInfo, tensor_k, tensor_dq_ws,
                               ds_l1.current(), k_l1_n, left, right, acc)
                # ---- mm4: dK = dS^T@Q ----
                iterate_mm_dsq(constInfo, prevInfo, tensor_q, tensor_dk_ws,
                               ds_t_l1.current(), q_l1, left, right, acc)
                # 反向：dS 的 L1 已读完
                pl.system.set_cross_core(pipe=pl.PipeType.MTE1,
                                         event_id=SYNC_C4_TO_V3_FLAG)

            if isLastLoop:
                break
            taskId = taskId + 1
            blockInnerIdx = get_next_valid_idx(constInfo, blockInnerIdx + 1, blockEnd)

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        y_ub = pl.make_tile_group(
            type=pl.TileType(shape=[SFMG_LOOP_SIZE, HEAD_DIM_ALIGN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_Y], mutex_ids=[15])
        dy_ub = pl.make_tile_group(
            type=pl.TileType(shape=[SFMG_LOOP_SIZE, HEAD_DIM_ALIGN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_DY], mutex_ids=[16])
        prod_ub = pl.make_tile_group(
            type=pl.TileType(shape=[SFMG_LOOP_SIZE, HEAD_DIM_ALIGN], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_PROD], mutex_ids=[17])
        tmp_ub = pl.make_tile_group(
            type=pl.TileType(shape=[SFMG_LOOP_SIZE, HEAD_DIM_ALIGN], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_TMP], mutex_ids=[18])
        nd_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_ND], mutex_ids=[19])
        nz_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[UB_NZ], mutex_ids=[20])
        tnd_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEN, VECTOR_BASEM], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_TND], mutex_ids=[21])
        tnz_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEN, VECTOR_BASEM], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=[UB_TNZ], mutex_ids=[22])
        # sfmg 整体视图([64,1])供 V3 broadcast；lo/hi 两个分段视图供 V1 写入
        sfmg_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[UB_SFMG], mutex_ids=[23])
        sfmg_lo_ub = pl.make_tile_group(
            type=pl.TileType(shape=[SFMG_LOOP_SIZE, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[UB_SFMG], mutex_ids=[28])
        sfmg_hi_ub = pl.make_tile_group(
            type=pl.TileType(shape=[SFMG_LOOP_SIZE, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[UB_SFMG_HI], mutex_ids=[29])
        max_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[UB_MAX_0, UB_MAX_1], mutex_ids=[24, 25])
        sum_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[UB_SUM_0, UB_SUM_1], mutex_ids=[26, 27])

        runInfos = make_run_infos()
        # ---------------- Process() 主循环 (AIV 侧) ----------------
        taskId = 0
        blockInnerIdx = get_next_valid_idx(constInfo, blockStart, blockEnd)
        for _ in pl.range(0, blockEnd - blockStart + 1):
            isLastLoop = blockInnerIdx >= blockEnd
            # 无条件轮转，与 cube 侧一一对应
            mm2_res.next()
            mm1_res.next()
            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                # ---- v1: softmaxGrad sfmg = rowsum(dy*y) ----
                process_vec1(constInfo, prevInfo, subIdx, tensor_y, tensor_dy,
                             y_ub.current(), dy_ub.current(),
                             prod_ub.current(), tmp_ub.current(),
                             sfmg_lo_ub.current(), sfmg_hi_ub.current())
                # 调试：sfmg 落回 GM [S1, B*N]，tile[64,1] 直接按升序轴映射
                dbgCol = prevInfo.boIdx * constInfo.n2Size + prevInfo.n2oIdx
                dbgRow = (prevInfo.queryOffset
                          + subIdx * prevInfo.firstHalfS1RealSize)
                pl.store(tensor_dbg_sfmg, sfmg_ub.current(), [dbgRow, dbgCol])

            if not isLastLoop:
                curInfo = runInfos[taskId % 2]
                set_run_info(constInfo, curInfo, taskId, blockInnerIdx, subIdx)
                # CopyMaxSum：max/sum 双缓冲随 taskId 轮转
                max_ub.next()
                sum_ub.next()

            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                # 消费上一轮 cube 写入的槽位
                mm1_t = mm1_res.previous()
                mm2_t = mm2_res.previous()
                # ---- ComputeDqkvBn2gs1s2 ----
                # 正向：等 mm2 结果
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C2_TO_V2_FLAG[(taskId + 1) % 2])
                process_vec2(constInfo, prevInfo, subIdx, tensor_smax, tensor_ssum,
                             mm2_t, max_ub.previous(), sum_ub.previous())
                # 正向：等 mm1 结果
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C1_TO_V2_FLAG[(taskId + 1) % 2])

                # ---- ProcessBn2gs1s2LastVec (非 IS_DROP 分支) ----
                # 反向：等 cube 读完上一轮 P L1 再覆写。
                # AscendC 用 needSyncDkMM 标记，其语义等价于「已经历过一轮
                # ComputeDqkv」，即 taskId > 1；此处直接用条件表达以避免
                # 跨迭代可变标量
                needSyncDkMM = taskId > 1
                if needSyncDkMM:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE3,
                                              event_id=SYNC_C5_TO_V4_FLAG)
                process_vec4(constInfo, prevInfo, subIdx, p_l1.current(), p_t_l1.current(),
                             mm2_t, nd_ub.current(), nz_ub.current(),
                             tnd_ub.current(), tnz_ub.current())
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                         event_id=SYNC_V4_TO_C5_FLAG)
                # 反向：等 cube 读完上一轮 dS L1
                if needSyncDkMM:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE3,
                                              event_id=SYNC_C4_TO_V3_FLAG)
                process_vec3(constInfo, prevInfo, subIdx, ds_l1.current(), ds_t_l1.current(),
                             mm1_t, mm2_t, sfmg_ub.current(),
                             nd_ub.current(), nz_ub.current(),
                             tnd_ub.current(), tnz_ub.current())
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                         event_id=SYNC_V3_TO_C3_FLAG)

            if isLastLoop:
                break
            taskId = taskId + 1
            blockInnerIdx = get_next_valid_idx(constInfo, blockInnerIdx + 1, blockEnd)
