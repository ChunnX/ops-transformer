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

  非 TND(layoutType == BSNGD)、FP16。mask_mode 0=dense，3=causal，4=band。
  tilingkey `mask_mode` 取值 0/3/4（无 mask / causal / band），`swizzle`
  区分线性 IsValid skip 与 packed 有效块分核。BN2 支持 mask，禁止 BN2+swizzle。

在该配置下 AscendC 的编译期谓词取值为：
  IS_DQ_WRITE_UB = IS_DK_WRITE_UB = IS_DV_WRITE_UB = false
      -> dQ/dK/dV 均由 fixpipe 经 GM workspace 做 atomicAdd 累加
  IS_DKV_RESIDENT_L0C = true (mm4+mm5=131072，max(mm1..3)=65536，
      合计 196608 <= 262144 字节 L0C)
      -> AscendC 下 dK/dV 可驻留 L0C 累加。本文件当前仍走
         「fixpipe 直出 GM + atomicAdd」，L0C 驻留作为后续优化单独叠加
  IS_DROP = false
      -> ComputeDqkvBn2gs1s2 走 ProcessBn2gs1s2LastVec 的 else 分支
  IsValid() 按 tilingkey mask_mode 静态分支：0 恒真，3=CheckIsValidBlock，
      4=band 区间。GetNextValidIdx 线性路径 skip 无效块。
本文件实现 Process() 的两槽 taskId ping-pong。PRELOAD_TIMES=3 的
ProcessPreloadTwoTimes 预取环按需求暂不实现。

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

每个 GM tensor 只推导出一种 Layout：tensor_k 因 MM2 的转置读(order=[3,1])
被定为 Layout::DN，MM3 需要非转置的 K，若共用同一 tensor，plain 读会被按 DN
解释而静默拿到 K^T(dQ 幅度正确但数值全错)。故 MM3 用独立的 tensor_k_nt 视图
指向同一 buffer，各自推导 layout。平台仅支持
ND2NZ/DN2NZ/ND2ND/DN2DN/NZ2NZ/DN2ZN，不支持 ND2ZN。

MM4/MM5 的左矩阵需要转置(dS^T / P^T)。AscendC 通过 L1->L0A 载入时置
isLeftTranspose=true 免费得到；pypto 无此能力(NZ 版本静默算错、ZZ 版本
无法编译，L0A 要求非 row-major fractal)，因此显式走
transpose(UB) -> ND2NZ -> insert 到 [N,M] NZ 的 L1，再常规载入 L0A。
"""

from dataclasses import dataclass

import os
import sys

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
from fag_common import (
    ATTEN_MASK_COMPRESS,
    CUBE_BASEM,
    CUBE_BASEN,
    D_CHUNK_MAXW,
    D_CHUNK_OFF0,
    D_CHUNK_OFF1,
    D_CHUNK_W0,
    D_CHUNK_W1,
    D_NCHUNKS,
    D_PHYS_W,
    HEAD_DIM_ALIGN,
    NZP_ROWS,
    PRELOAD_TIMES,
    SYNC_C1_TO_V2_FLAG,
    SYNC_C2_TO_V2_FLAG,
    SYNC_C3_TO_V5_FLAG,
    SYNC_C4_TO_V3_FLAG,
    SYNC_C4_TO_V6_FLAG,
    SYNC_C5_TO_V4_FLAG,
    SYNC_DETER_FIX_FLAG,
    SYNC_V2_TO_C1_FLAG,
    SYNC_V2_TO_C2_FLAG,
    SYNC_V3_TO_C3_FLAG,
    SYNC_V4_TO_C5_FLAG,
    VECTOR_BASEM,
    VECTOR_BASEN,
    VF_LANES,
    VF_LANES_FP16,
    make_run_infos,
    set_const_info,
)
from fag_mem import (
    ACC_COLS,
    BN2_CAST_W,
    BN2_UB_CAST,
    BN2_UB_DY,
    BN2_UB_LSE_0,
    BN2_UB_LSE_1,
    BN2_UB_NZP,
    BN2_UB_SFMG,
    BN2_UB_Y,
    BN2_V1_ROWS,
    L0AB_ADDRS,
    L0A_ADDR,
    L0A_ADDR2,
    L0B_ADDR,
    L0B_ADDR2,
    L0C_ACC_ADDRS,
    L0C_ACC_MUTEX,
    L0C_ADDR,
    L0C_ADDR2,
    L0C_DK,
    L0C_DK0_ADDR,
    L0C_DK1_ADDR,
    L0C_DK1_MUTEX,
    L0C_DV,
    L0C_DV0_ADDR,
    L0C_DV1_ADDR,
    L0C_DV1_MUTEX,
    L1_DO,
    L1_DO2,
    L1_DO3,
    L1_DO4,
    L1_DO_ADDRS,
    L1_DO_MUTEX,
    L1_DS,
    L1_DS_ADDR,
    L1_K,
    L1_KND_ADDR,
    L1_K_ADDR,
    L1_K_ND,
    L1_P,
    L1_P_ADDR,
    L1_Q,
    L1_Q2,
    L1_Q3,
    L1_Q4,
    L1_Q_ADDRS,
    L1_Q_MUTEX,
    L1_V,
    L1_V_ADDR,
    POST_ELEMS,
    PRE_ELEMS,
    UB_DY_A,
    UB_LSE0_A,
    UB_LSE1_A,
    UB_MM1_0,
    UB_MM1_1,
    UB_MM2_0,
    UB_MM2_1,
    UB_NZP_A,
    UB_PRE_ZERO,
    UB_SFMG_A,
    UB_Y_A,
    post_ub_addrs,
)
from fag_tiling import FlashAttnGradTilingData, FlashAttnGradTilingKey
from fag_vector_api import (
    broadcast_sub_mul_cast_vf,
    cast_p_vf,
    fill_zero_fp16_vf,
    fill_zero_vf,
    muls_cast_vf,
    softmax_grad_front_row192_vf,
    softmax_grad_front_row_vf,
)
from fag_schedule import (
    META_FAG_BLOCK_ENDS_OFFSET,
    META_FAG_BLOCK_STARTS_OFFSET,
    META_FAG_DKV_ROW_ENDS,
    META_FAG_DKV_ROW_STARTS,
    META_FAG_DQ_ROW_ENDS,
    META_FAG_DQ_ROW_STARTS,
    META_HEAD_FAG_START_INDEX,
    META_VIEW_LEN,
    apply_s2_idx_no_change,
    block_range_of,
    next_work_gidx,
    set_run_info,
    skip_invalid,
)
from fag_block_cube import (
    flush_dkv,
    iterate_mm_dsk,
    iterate_mm_dsk_bn2,
    iterate_mm_dsq,
    iterate_mm_dsq_bn2,
    iterate_mm_dyv,
    iterate_mm_dyv_bn2,
    iterate_mm_pdy,
    iterate_mm_pdy_bn2,
    iterate_mm_qk,
    iterate_mm_qk_bn2,
)
from fag_block_vec import (
    copy_ub2l1,
    process_dq_multiblk,
    process_muls_cast_dk_bn2,
    process_muls_cast_dq_bn2,
    process_post,
    process_pre_from_meta,
    process_vec1,
    process_vec1_bn2,
    process_vec2,
    process_vec3,
    process_vec3_bn2,
    process_vec4,
)
from fag_host_mirror import (
    decide_bn2_multiblk,
    decide_d_align,
    decide_dv_align,
    decide_sparse_type,
    decide_swizzle,
    decide_template,
    total_per_batch_num,
)
from attenmask import (
    copy_in_atten_mask,
    muls_sel_simple_softmax_vf,
)

# `vf` 由 parser 按语法识别(_call_parser.py: attrs[0] == "vf")，无需 import


# cast+ND2NZ 融合 VF 的输出缓冲：声明 65 行而非 64。
# VF 用 block_stride=65 写出，每个分形列比紧凑排布多垫一个 32B block，
# 把相邻分形列错开到不同 bank(对应 ASC vf_cast_transdata_deconflict.h 的 +1)。
# lseQue[2] 双缓冲。原先是 max/sum 两组共四块，改用 LSE 后合成一组：
# lse = max + log(sum(exp(s-max)))，前向已把两者合进一个张量。


# 注意：CV 共享的 dS/P L1(NZ) 一律不设 valid_shape。
# NZ 是分形排布，缩小行数会改变分形打包的跨距，导致 vector 侧 insert 与
# cube 侧读取对不上 —— 实测 S2 尾块下 dQ(按列裁剪的 ds_l1)正确，
# 而 dK/dV(按行裁剪的 ds_t_l1/p_t_l1)错误(max 0.68/0.90)。
# 矩阵乘的实际计算范围由 L0A/L0B/acc 的 valid_shape 决定，输出再由 store
# 按 s1Real/s2Real 裁剪，故界外脏数据不会流入 GM，L1 侧无需裁剪。


# ==================================================================
#  Pre 阶段 —— dq workspace 清零 (presfmg_regbase.h:258 InitOutput)
# ==================================================================
# dQ 用 atomicAdd across s2Outer 累加，故进主循环前必须清零。
# ASC 只在 s2Outer>1 时清(同一处判断)，因为 s2Outer==1 时每个位置只写一次。
# dK/dV 走 L0C 驻留累加、每个 (b,n2,s2) 组只 store 一次，但仍用 atomicAdd
# (多核可能分到同一 s2 块的不同 s1 区间)，故同样需要清零。


# ==================================================================
#  Post 阶段 —— dQ/dK/dV 的 scale + cast (post_regbase.h:238-241)
# ==================================================================
# scale 必须在矩阵乘之后、fp32 结果上施加：先乘会让 fp16 舍入发生在
# 不同量级上，训练中误差会累积(用户明确要求)。
# 只有 dQ/dK 乘 scale，dV 不乘 —— ASC 的 `qkvIdx < 2` 判断。


def flash_attn_grad_bn2gs1s2(
    # 13 个输入，顺序严格对齐 op_host/flash_attn_grad_def.cpp 的 Input() 声明
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    # 以下 6 个为可选输入。cu_seqlens/seqused(TND)、sinks 本轮不读。
    # attn_mask：mask_mode 3/4 时按 [2048,2048] uint8 压缩矩阵搬 UB。
    # metadata：FAG 区含分核 [start,end) 与 pre/post 行区间（412 int32）。
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    # 3 个输出
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    # 单块 workspace，内部再切三份 fp32 累加区（见下方 make_ptr 分区）
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
    dv_align,
    swizzle,
):
    # coreNum 从内建 API 取而非 tiling：TilingData 无此字段。
    # get_block_num() 是 host SetBlockDim 启动的 cube 数（dense 下等于
    # metadata blockOuter），cube/vector 两侧同一口径，set/wait 才能配对。
    coreNum = pl.get_block_num()
    # swizzle 来自 tilingkey，是编译期常量 —— 每种分核各生成一份二进制，
    # `if constInfo.enableSwizzle` 的两路在编译期折叠，运行期没有判断开销。
    constInfo = set_const_info(tiling, coreNum, swizzle)

    # ---- workspace 分区 ----
    # host(GetWorkspaceSizes) 给的是单块连续内存，这里按 fp32 元素切成
    # dq/dk/dv 三段累加区。三段大小必须与 host 的 CalcWorkSpace 逐一对齐。
    # dq 段按 n1(=n2*G) 个 head，dk/dv 段按 n2 —— 与 host 的 dqElems/dkvElems
    # 一致。GQA 下若这里误用 n2，dk/dv 的起始偏移就会算错。
    # host 只在三段之后多留一行 128 对齐，不改这里的基地址。
    wsRowsQ = tiling.b * tiling.s1 * tiling.n1
    wsRowsKV = tiling.b * tiling.s2 * tiling.n2
    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    dq_workspace = ws_ptr
    # 段宽按各自的 head dim：dq/dk 是 D，dv 是 Dv(可以短于 D)。
    # 必须与 host 的 dqElems/dkElems/dvElems 逐一对齐。
    dk_workspace = pl.addptr(dq_workspace, wsRowsQ * tiling.d)
    dv_workspace = pl.addptr(dk_workspace, wsRowsKV * tiling.d)

    # ---- 统一四维视图 [viewD0, S, viewD2, D] ----
    # BSND 与 BNSD 共用同一套 make_tensor / pl.load(order=...) 代码，
    # 布局差异只体现在 viewD0/viewD2 与索引系数上(见 set_run_info 的 idx0/idx2)：
    #   BSND: [B,   S, N, D]，idx = [b,     s, n, 0]
    #   BNSD: [B*N, S, 1, D]，idx = [b*N+n, s, 0, 0]
    # 两者都是紧凑排布 —— 这一点是必须的：make_tensor 对非紧凑 stride 会静默
    # 忽略并按紧凑跨距去读(不报错)，所以不能用「倒置 stride」表达 BNSD。
    # Q 侧(n1 个 head)与 KV 侧(n2 个 head)各一套视图参数
    # 最内轴宽度取各张量**真实的** head dim，不能用分配宽度 d_align：
    # 这是 GM 上的实际排布，Q/K/dq/dk 侧是 D，V/dO/attn_out/dv 侧是 Dv。
    # (片上 tile 才按 d_align 分配、按真实宽度填。)
    dQ = constInfo.dSize
    dV = constInfo.dvSize
    viewD2Q = constInfo.viewD2Q
    d2dQ = viewD2Q * dQ
    s1ndQ = constInfo.s1Size * d2dQ
    viewD2KV = constInfo.viewD2KV
    d2dKV = viewD2KV * dQ
    s2ndKV = constInfo.s2Size * d2dKV
    # Dv 宽的那几个张量另有一套行距
    d2dQv = viewD2Q * dV
    s1ndQv = constInfo.s1Size * d2dQv
    d2dKVv = viewD2KV * dV
    s2ndKVv = constInfo.s2Size * d2dKVv

    tensor_q = pl.make_tensor(q, [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dQ],
                              [s1ndQ, d2dQ, dQ, 1], dtype=pl.DT_FP16)
    tensor_k = pl.make_tensor(k, [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
                              [s2ndKV, d2dKV, dQ, 1], dtype=pl.DT_FP16)
    # MM3(dQ = dS@K) 需要「未转置」的 K，而 MM2 需要 K^T。
    # 每个 GM tensor 只会推导出一种 Layout：tensor_k 因 MM2 的转置读被定成
    # Layout::DN，MM3 若共用它，plain 读也会被按 DN 解释，实测等价于拿到 K^T
    # (partial 与 K[s2o]^T 吻合到 1e-4)。故为 MM3 单独建一份同 buffer 的
    # tensor 视图，让它独立推导 layout。
    tensor_k_nt = pl.make_tensor(k, [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
                                 [s2ndKV, d2dKV, dQ, 1], dtype=pl.DT_FP16)
    # V/dO/attn_out 是 Dv 宽
    tensor_v = pl.make_tensor(v, [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dV],
                              [s2ndKVv, d2dKVv, dV, 1], dtype=pl.DT_FP16)
    tensor_dy = pl.make_tensor(dout, [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dV],
                               [s1ndQv, d2dQv, dV, 1], dtype=pl.DT_FP16)
    tensor_y = pl.make_tensor(attn_out, [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dV],
                              [s1ndQv, d2dQv, dV, 1], dtype=pl.DT_FP16)
    # softmax_lse : [B,N1,S1] —— 按 Q 的 head 数(n1 = n2*G)存，因为每个 Q head
    # 的 softmax 归一化因子不同，G 个 Q head 不能共享同一份 lse。
    # 对齐 ASC 的 softmax_max/sum：offset = ((b*n2 + n2o)*G + g)*S1。
    n1Size = constInfo.n2Size * constInfo.gSize
    tensor_lse = pl.make_tensor(
        softmax_lse, [constInfo.bSize, n1Size, constInfo.s1Size],
        [n1Size * constInfo.s1Size, constInfo.s1Size, 1],
        dtype=pl.DT_FP32)
    if mask_mode != 0:
        tensor_mask = pl.make_tensor(
            attn_mask, [ATTEN_MASK_COMPRESS, ATTEN_MASK_COMPRESS],
            [ATTEN_MASK_COMPRESS, 1], dtype=pl.DT_UINT8)
    # dq/dk/dv workspace 为 fp32，供 atomicAdd 累加
    # dq/dk 是 D 宽，dv 是 Dv 宽 —— 与 workspace 的分段宽度一致
    tensor_dq_ws = pl.make_tensor(dq_workspace, [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dQ],
                                  [s1ndQ, d2dQ, dQ, 1], dtype=pl.DT_FP32)
    tensor_dk_ws = pl.make_tensor(dk_workspace, [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
                                  [s2ndKV, d2dKV, dQ, 1], dtype=pl.DT_FP32)
    tensor_dv_ws = pl.make_tensor(dv_workspace, [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dV],
                                  [s2ndKVv, d2dKVv, dV, 1], dtype=pl.DT_FP32)

    # ---- pre/post：ASC 同款一维视图 ----
    # InitOutput / Muls+Cast 都按 B*S*N*D 个元素线性切，不看行宽。
    # dq 侧 head 数是 n1(=n2*G)，dk/dv 侧是 n2。
    dqRows = constInfo.bSize * constInfo.s1Size * n1Size
    dkvRows = constInfo.bSize * constInfo.s2Size * constInfo.n2Size
    dqElems = dqRows * dQ
    dkElems = dkvRows * dQ
    dvElems = dkvRows * dV
    tensor_dq_ws_flat = pl.make_tensor(dq_workspace, [1, dqElems], [dqElems, 1],
                                       dtype=pl.DT_FP32)
    tensor_dk_ws_flat = pl.make_tensor(dk_workspace, [1, dkElems], [dkElems, 1],
                                       dtype=pl.DT_FP32)
    tensor_dv_ws_flat = pl.make_tensor(dv_workspace, [1, dvElems], [dvElems, 1],
                                       dtype=pl.DT_FP32)
    tensor_dq_out = pl.make_tensor(dq, [1, dqElems], [dqElems, 1], dtype=pl.DT_FP16)
    tensor_dk_out = pl.make_tensor(dk, [1, dkElems], [dkElems, 1], dtype=pl.DT_FP16)
    tensor_dv_out = pl.make_tensor(dv, [1, dvElems], [dvElems, 1], dtype=pl.DT_FP16)
    # V1 列宽：64 档收到 64；其它档跟 dv_align（y/dy 是 Dv 宽）。
    v1w = 64 if d_align == 64 else dv_align

    # cBlockIdx 必须按物理核取：AIV 侧 get_block_idx() 按 vector 子核计数
    # (0..2*coreNum-1)，AIC 侧按物理核计数。两侧口径不一致会算出不同的
    # numBlocks 不同 -> 循环次数不同 -> set/wait 配不上 -> 死锁。
    # 核数直接用 tiling 传入，避免再依赖 get_block_num 的口径。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum

    # 分核区间取自 metadata。swizzle 支仍走公式，原因见 block_range_of。
    meta = pl.make_tensor(metadata, [META_VIEW_LEN], [1], dtype=pl.DT_INT32)
    fagOff = pl.getval(meta, META_HEAD_FAG_START_INDEX)
    blockStart = pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET + cBlockIdx)
    blockEnd = pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET + cBlockIdx)
    numBlocks = block_range_of(constInfo, cBlockIdx, coreNum,
                               blockStart, blockEnd)

    # ---- CV 共享 UB：mm1ResBuf[2] / mm2ResBuf[2]，按 taskId % 2 ping-pong ----
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
        addrs=[L1_DS_ADDR[d_align]], mutex_ids=[8])
    p_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_BASEM, CUBE_BASEN], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1),
        addrs=[L1_P_ADDR[d_align]], mutex_ids=[9])
    # dS^T / P^T 不再单独存一份：[M,N] 按 NZ 存放的字节布局，与 [N,M] 按 ZN
    # 存放完全一致，所以对 ds_l1/p_l1 的同一地址再建一个 ZN 视图，就等于
    # 零指令拿到转置。vector 侧只 insert 一次(NZ 视图)，cube 侧 mm4/mm5
    # 直接按 ZN 视图 move 到 L0A —— shape 与 L0A 一致，TMov 通过。
    #
    # 这替掉了原先的 pl.transpose：实测单次 24.9us(44864 cycles)，
    # 整核估算 5672us，占 8192 场景实测 6011us 的 94%，是性能差距主因。
    # 注意 L1->L0 无法随路转置(TMov static_assert 要求 shape 相同)，
    # Python validator 里「允许 2D 转置」的注释能过 parse 但 codegen 失败。
    ds_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_BASEN, CUBE_BASEM], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
        addrs=[L1_DS_ADDR[d_align]], mutex_ids=[10])
    p_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_BASEN, CUBE_BASEM], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
        addrs=[L1_P_ADDR[d_align]], mutex_ids=[11])

    # ==== Pre 阶段：清零 dq/dk/dv workspace ====
    # 三个 workspace 都用 atomicAdd 累加，进主循环前必须为 0。
    # 对应 ASC presfmg_regbase.h:258 的 InitOutput / Duplicate。
    # 清完用 sync_all 隔开：主循环的 atomicAdd 必须看到已清零的 GM。
    with pl.section_vector():
        # 单缓冲。mutex_ids 复用主循环的 id：pre 与主循环被 sync_all
        # 完全隔开，不会同时活跃。列宽/行数按 d_align 查表。
        zero_ub = pl.make_tile_group(
            type=pl.TileType(shape=[1, PRE_ELEMS[d_align]],
                             dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[UB_PRE_ZERO], mutex_ids=[19])
        process_pre_from_meta(meta, fagOff, tensor_dq_ws_flat, tensor_dk_ws_flat,
                              tensor_dv_ws_flat, dQ, dV, zero_ub.current())
    pl.system.sync_all()

    with pl.section_cube():
        # 地址/mutex 表按档下标取(表在模块级算好，见 _build_l1)。
        # 128 档取到的就是原有那套写死的值，故该档生成代码不变。
        q_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, D_PHYS_W[d_align]], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=L1_Q_ADDRS[d_align], mutex_ids=L1_Q_MUTEX[d_align])
        k_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[D_PHYS_W[d_align], CUBE_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=[L1_K_ADDR[d_align]], mutex_ids=[5])
        # MM3 的右矩阵：非转置 K[S2,D]，默认(ND) layout，独占地址与 mutex。
        # 源必须是 ND layout 的 tensor_k_nt(ND2ND)；若源是 DN 的 tensor_k，
        # 载入会按 DN 解释而静默得到 K^T。ZN 版本虽能编过(DN2ZN 合法)但
        # 同样不对：平台只支持 ND2NZ/DN2NZ/ND2ND/DN2DN/NZ2NZ/DN2ZN。
        k_l1_n = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, D_PHYS_W[d_align]], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=[L1_KND_ADDR[d_align]], mutex_ids=[30])
        # V/dO 是 Dv 宽，但 tile 仍按 d_align 分配(Dv<=D)：多出来的列是
        # padding，由 load/matmul 的 valid_shape 挡掉。这样两侧共用一套
        # 地址步长，L1 布局不必再分 D/Dv 两种块大小。
        v_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[D_PHYS_W[d_align], CUBE_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=[L1_V_ADDR[d_align]], mutex_ids=[6])
        do_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, D_PHYS_W[d_align]], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=L1_DO_ADDRS[d_align], mutex_ids=L1_DO_MUTEX[d_align])
        # mm4/mm5 用的 Q/dO 视图：地址与 mutex_id 和上面完全相同(共享同一
        # 块 L1 与同一套同步跟踪)，但游标独立。它从 taskId==2 才开始
        # next()，因此永远落后两个 task —— current() 恰好指向 task N-2
        # 载入的槽位，正是 mm4/mm5 需要的那一份。
        # 这样 Q/dO 每个 (b,n,s1) 块只从 GM 读一次，mm4/mm5 直接复用。
        # mutex_ids 必须与 writer 视图完全相同：这对应 ASC 的
        # qL1Buf.Get() / qL1Buf.GetPre() —— 同一个 MutexBuffer 的两个访问器，
        # 靠同一套 LockProd/UnlockProd 与 LockCons/UnlockCons 把「mm2 写」
        # 与「mm4 读」排序。若给 reader 另一套 id，这层互锁就没了，
        # mm2(task N)会在 mm4 读完前覆写，实测 dK/dV 错(0.61/0.57)。
        q_l1_dk = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, D_PHYS_W[d_align]], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=L1_Q_ADDRS[d_align], mutex_ids=L1_Q_MUTEX[d_align])
        do_l1_dv = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, D_PHYS_W[d_align]], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=L1_DO_ADDRS[d_align], mutex_ids=L1_DO_MUTEX[d_align])
        # ---- L0A/L0B/L0C ----
        # 一律按「最宽的一个 D 分块」声明：192 档是 96 列而不是 192，
        # 于是 L0A/L0B 每块 24KB、双缓冲 48KB<=64KB 得以保留。
        # 128 档分块宽度就是 128，与改动前完全一致。
        # 左矩阵最宽的一维是 ACC_COLS = max(CUBE_BASEN, 分块宽)：MM2/MM1 的
        # 左矩阵是 Q/dO 的 D 分块，MM3/MM4/MM5 的左矩阵是 dS/P([M,N])。
        left = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, ACC_COLS[d_align]],
                             dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Left, compact=1),
            addrs=L0AB_ADDRS[d_align], mutex_ids=[12, 26])
        right = pl.make_tile_group(
            type=pl.TileType(shape=[ACC_COLS[d_align], CUBE_BASEN],
                             dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Right, compact=1),
            addrs=L0AB_ADDRS[d_align], mutex_ids=[13, 27])
        # acc 的列宽取 ACC_COLS：MM2/MM1 输出 [M,s2]，MM3 输出 dQ 的一个
        # D 分块。
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, ACC_COLS[d_align]],
                             dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=L0C_ACC_ADDRS[d_align], mutex_ids=L0C_ACC_MUTEX[d_align])
        # L0C 驻留的 dK/dV 累加器，跨 s1 轮次保持不变。
        # 每个 D 分块**各建一个独立的 tile_group**(而不是一个多槽位 group)：
        # 这些累加器必须在同一 s2 块内持续存在，访问方式是 current() 而非
        # next()。若做成多槽位 group 再靠 next() 轮换，D=128 档(单块)的
        # 游标语义就和改动前不同了，零回归就不成立。
        dk_acc0 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[L0C_DK0_ADDR[d_align]], mutex_ids=[16])
        dv_acc0 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[L0C_DV0_ADDR[d_align]], mutex_ids=[17])
        # 第二块只在 192 档存在。为了让下面的调用点无需分支，单块档位下把
        # 它指向第 0 块的同一地址/同一 mutex —— 那些代码路径被 trace 期的
        # `if D_NCHUNKS > 1` 完全剪掉，永远不会真的用到这个别名。
        dk_acc1 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[L0C_DK1_ADDR[d_align]], mutex_ids=[L0C_DK1_MUTEX[d_align]])
        dv_acc1 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[L0C_DV1_ADDR[d_align]], mutex_ids=[L0C_DV1_MUTEX[d_align]])

        runInfos = make_run_infos()
        # ---------------- Process() 主循环 (AIC 侧) ----------------
        taskId = 0
        localIdx = 0
        denseCursor = blockStart
        isLastLoop = 0
        drainDone = 0
        nFetched = 0
        # cube 领先 vector 两个 task。ASC Process：while + GetNextValidIdx==-1 排空。
        while True:
            # 无条件轮转，保证与 vector 侧步调一致（末轮不做 mm 也要转）
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            # L1_PRELOAD：Q/dO 的 double buffer 无条件轮转，供本轮 mm2/mm1。
            # mm4/mm5 用 q_l1_dk/do_l1_dv —— 同地址、同 mutex_ids 的延后
            # 游标视图，直接复用这里载入的数据(见其声明处说明)。
            q_cur = q_l1.next()
            do_cur = do_l1.next()
            if isLastLoop == 0:
                gIdx = next_work_gidx(constInfo, cBlockIdx, coreNum,
                                      blockStart, blockEnd, localIdx,
                                      denseCursor, numBlocks)
                if gIdx < 0:
                    isLastLoop = 1
                else:
                    if swizzle == 0:
                        denseCursor = gIdx + 1
                    curInfo = runInfos[taskId % PRELOAD_TIMES]
                    set_run_info(constInfo, curInfo, taskId, gIdx, 0, localIdx)
                    nFetched = nFetched + 1
                    if mask_mode != 0:
                        apply_s2_idx_no_change(runInfos, curInfo, taskId, PRELOAD_TIMES)

                    # ---- mm2: Q@K^T ----
                    # 反向等待：mm2 只有 2 个 UB 槽位，task N 复用 task N-2 的
                    # 那块，必须等 vector 读完(ASC ProcessPreloadTwoTimes 里
                    # taskId>=1 时 wait SYNC_V2_TO_C2_FLAG)。
                    # taskId>=2 才有 N-2 存在；首两轮槽位是干净的，直接发。
                    # 这就是「首轮连发四次 matmul」：taskId 0/1 两轮各发
                    # mm2+mm1，累计 4 次，把两个槽位填满后 vector 才开始消费，
                    # 于是第 N 组 cube 计算与第 N-2 组 vector 计算重叠。
                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_V2_TO_C2_FLAG[taskId % 2])
                    iterate_mm_qk(constInfo, curInfo, tensor_q, tensor_k,
                                  q_cur, k_l1, left, right, acc, mm2_cur, curInfo.loadKV)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                             event_id=SYNC_C2_TO_V2_FLAG[taskId % 2])
                    # ---- mm1: dO@V^T ----
                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_V2_TO_C1_FLAG[taskId % 2])
                    iterate_mm_dyv(constInfo, curInfo, tensor_dy, tensor_v,
                                   do_cur, v_l1, left, right, acc, mm1_cur, curInfo.loadKV)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                             event_id=SYNC_C1_TO_V2_FLAG[taskId % 2])

            # mm3/mm4/mm5 消费的是 vector 刚产出的 dS/P，对应 task N-2
            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # 正向：等 vector 把 P / dS 写入 L1
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                          event_id=SYNC_V4_TO_C5_FLAG)
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                          event_id=SYNC_V3_TO_C3_FLAG)
                # ---- mm5: dV = P^T@dO (L0C 驻留累加) ----
                # dkvAcc=1: 与上一轮同一个 s2 块 -> 在 L0C 上累加
                # prevInfo 对应 task N-2 的 block 下标。
                # 排空阶段不再发新块，用 taskId-2 对应当年那一拍。
                prevIdx = taskId - 2
                # 延后游标在此推进：taskId>1 才走到这里，故它恒落后两拍
                q_prev = q_l1_dk.next()
                do_prev = do_l1_dv.next()
                iterate_mm_pdy(constInfo, prevInfo,
                               p_t_l1.current(), do_prev,
                               left, right, dv_acc0, dv_acc1, prevInfo.dkvAcc)
                # 反向：P 的 L1 已读完，放行 vector 覆写
                pl.system.set_cross_core(pipe=pl.PipeType.MTE1,
                                         event_id=SYNC_C5_TO_V4_FLAG)
                # ---- mm3: dQ = dS@K (仍直出 GM，dQ 按 s1 分块无法驻留) ----
                iterate_mm_dsk(constInfo, prevInfo, tensor_k_nt, tensor_dq_ws,
                               prevInfo.loadKV,
                               ds_l1.current(), k_l1_n, left, right, acc)

                # ---- mm4: dK = dS^T@Q (L0C 驻留累加) ----
                iterate_mm_dsq(constInfo, prevInfo,
                               ds_t_l1.current(), q_prev,
                               left, right, dk_acc0, dk_acc1, prevInfo.dkvAcc)

                # s2 块即将切换(或已是最后一轮)时，把 L0C 上的 dK/dV 落 GM。
                # 核可能在列中途结束，末块必须无条件落盘。
                if prevIdx >= nFetched - 1 or prevInfo.flushDkv == 1:
                    flush_dkv(constInfo, prevInfo,
                              tensor_dk_ws, tensor_dv_ws,
                              dk_acc0, dk_acc1, dv_acc0, dv_acc1)
                # 反向：dS 的 L1 已读完
                pl.system.set_cross_core(pipe=pl.PipeType.MTE1,
                                         event_id=SYNC_C4_TO_V3_FLAG)

            if isLastLoop == 1:
                drainDone = drainDone + 1
                if drainDone >= 2:
                    break
            taskId = taskId + 1
            localIdx = localIdx + 1

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        # y/dy 宽度是 V1_W（d_align=64 时 64，否则 dv_align）。
        y_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, v1w], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_Y_A[v1w]], mutex_ids=[15])
        dy_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, v1w], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec),
            addrs=[UB_DY_A[v1w]], mutex_ids=[16])
        # 融合 VF 的唯一输出缓冲：65 行 NZ，多出的一行是 bank 错开填充。
        # valid_shape 动态设置：insert 前收窄到 rows(框架据此算 65-rows
        # 的源跨距)，insert 后必须恢复成 65(见 copy_ub2l1)。
        nzp_ub = pl.make_tile_group(
            type=pl.TileType(shape=[NZP_ROWS, VECTOR_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=2),
            addrs=[UB_NZP_A[v1w]], mutex_ids=[19])
        sfmg_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[UB_SFMG_A[v1w]], mutex_ids=[23])
        lse_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[UB_LSE0_A[v1w], UB_LSE1_A[v1w]], mutex_ids=[24, 25])
        mask_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[UB_Y_A[v1w]], mutex_ids=[15])
        mask_pre_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[UB_DY_A[v1w]], mutex_ids=[16])

        runInfos = make_run_infos()
        # ---------------- Process() 主循环 (AIV 侧) ----------------
        taskId = 0
        localIdx = 0
        denseCursor = blockStart
        isLastLoop = 0
        drainDone = 0
        while True:
            # 无条件轮转，与 cube 侧一一对应
            mm2_res.next()
            mm1_res.next()
            # vector 落后 cube 两个 task：消费 runInfos[(taskId+1)%3] 即 N-2
            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # ---- v1: softmaxGrad sfmg = rowsum(dy*y) ----
                process_vec1(constInfo, prevInfo, subIdx, tensor_y, tensor_dy,
                             y_ub.current(), dy_ub.current(),
                             sfmg_ub.current(), v1w)

            if isLastLoop == 0:
                gIdx = next_work_gidx(constInfo, cBlockIdx, coreNum,
                                      blockStart, blockEnd, localIdx,
                                      denseCursor, numBlocks)
                if gIdx < 0:
                    isLastLoop = 1
                else:
                    if swizzle == 0:
                        denseCursor = gIdx + 1
                    curInfo = runInfos[taskId % PRELOAD_TIMES]
                    set_run_info(constInfo, curInfo, taskId, gIdx, subIdx, localIdx)
                    if mask_mode != 0:
                        apply_s2_idx_no_change(runInfos, curInfo, taskId, PRELOAD_TIMES)
                    lse_ub.next()

            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # 消费 cube 在 task N-2 写入的槽位。mm1/mm2 只有 2 槽，
                # task N 与 N-2 同奇偶 -> 同一块物理槽位，故这里要取
                # current()(与 cube 侧 next() 返回的同一块)，
                # 而不是 lag=1 时用的 previous()
                mm1_t = mm1_res.current()
                mm2_t = mm2_res.current()
                # ---- ComputeDqkvBn2gs1s2 ----
                # 正向：等 mm2 结果
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C2_TO_V2_FLAG[taskId % 2])
                if mask_mode != 0:
                    process_vec2(constInfo, prevInfo, subIdx, tensor_lse,
                                 mm2_t, lse_ub.previous(),
                                 tensor_mask, mask_ub.current(), mask_pre_ub.current())
                else:
                    process_vec2(constInfo, prevInfo, subIdx, tensor_lse,
                                 mm2_t, lse_ub.previous(),
                                 tensor_lse, mask_ub.current(), mask_pre_ub.current())
                # 正向：等 mm1 结果
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C1_TO_V2_FLAG[taskId % 2])

                # ---- ProcessBn2gs1s2LastVec (非 IS_DROP 分支) ----
                # 反向：等 cube 读完上一轮 P L1 再覆写。
                # AscendC 用 needSyncDkMM 标记，其语义等价于「已经历过一轮
                # ComputeDqkv」，即 taskId > 1；此处直接用条件表达以避免
                # 跨迭代可变标量
                needSyncDkMM = taskId > 2
                if needSyncDkMM:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE3,
                                              event_id=SYNC_C5_TO_V4_FLAG)
                process_vec4(constInfo, prevInfo, subIdx, p_l1.current(),
                             mm2_t, nzp_ub.current())
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                         event_id=SYNC_V4_TO_C5_FLAG)
                # 反向：等 cube 读完上一轮 dS L1
                if needSyncDkMM:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE3,
                                              event_id=SYNC_C4_TO_V3_FLAG)
                process_vec3(constInfo, prevInfo, subIdx, ds_l1.current(),
                             mm1_t, mm2_t, sfmg_ub.current(),
                             nzp_ub.current())
                # 反向：mm1/mm2 的 UB 槽位已读完，放行 cube 覆写。
                # 下标与刚消费的 task N-2 一致，cube 侧 task N 等的正是它
                # (ASC kernel.h:709-710 同一位置 set 这两个 flag)。
                pl.system.set_cross_core(pipe=pl.PipeType.V,
                                         event_id=SYNC_V2_TO_C1_FLAG[taskId % 2])
                pl.system.set_cross_core(pipe=pl.PipeType.V,
                                         event_id=SYNC_V2_TO_C2_FLAG[taskId % 2])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                         event_id=SYNC_V3_TO_C3_FLAG)

            if isLastLoop == 1:
                drainDone = drainDone + 1
                if drainDone >= 2:
                    break
            taskId = taskId + 1
            localIdx = localIdx + 1

    # ==== Post 阶段：dQ/dK/dV 的 scale + cast fp32->fp16 ====
    # scale 在矩阵乘之后、fp32 结果上施加(而非提前乘进 dS)：先乘会让
    # fp16 舍入发生在不同量级上，网络训练中误差会累积。
    # 只有 dQ/dK 乘 scale，dV 不乘 —— 对应 ASC 的 `qkvIdx < 2`
    # (post_regbase.h:238)。dV 传 1.0，fp32 乘 1.0 精确。
    # 进 post 前必须 sync_all：要等所有核的 atomicAdd 都落盘。
    pl.system.sync_all()

    with pl.section_vector():
        vBlockIdxP = pl.get_block_idx()
        dqRowStartP = pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdxP)
        dqRowEndP = pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdxP)
        dkvRowStartP = pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdxP)
        dkvRowEndP = pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdxP)
        # depth 2：两个 f32 槽 + 两个 f16 槽，各自独立 mutex_ids，
        # 框架据此判定「下一块的 load」与「本块的 cast/store」无冲突，
        # 从而让 MTE2 / V / MTE3 三条流水重叠。
        postF32_0, postF32_1, postF16_0, postF16_1 = post_ub_addrs(
            POST_ELEMS[d_align])
        post_f32 = pl.make_tile_group(
            type=pl.TileType(shape=[1, POST_ELEMS[d_align]],
                             dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[postF32_0, postF32_1], mutex_ids=[19, 20])
        post_f16 = pl.make_tile_group(
            type=pl.TileType(shape=[1, POST_ELEMS[d_align]],
                             dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[postF16_0, postF16_1], mutex_ids=[21, 22])
        process_post(tensor_dq_ws_flat, tensor_dq_out, post_f32, post_f16,
                     constInfo.scaleValue, dqRowStartP * dQ,
                     (dqRowEndP - dqRowStartP) * dQ)
        process_post(tensor_dk_ws_flat, tensor_dk_out, post_f32, post_f16,
                     constInfo.scaleValue, dkvRowStartP * dQ,
                     (dkvRowEndP - dkvRowStartP) * dQ)
        process_post(tensor_dv_ws_flat, tensor_dv_out, post_f32, post_f16,
                     1.0, dkvRowStartP * dV,
                     (dkvRowEndP - dkvRowStartP) * dV)


def flash_attn_grad_bn2(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
):
    """BN2 模板：一个核独占整个 (b, n2) head。

    单块（S1/S2<=128）：没有 pre、没有 post、没有 sync_all，也不读
    fp32 workspace。主循环对齐 ComputeDqkvBn2（1-ahead ping-pong）。

    MultiBlk（is_bn2_multiblk）：默认仍无 pre/post/sync_all。dQ 对齐 ASC
    IterateMmDsKFixpout：cube 首有效 s2 覆盖 workspace，其余 AtomicAdd；
    vector 只在 isLastS1Outer 从 GM Muls+Cast。dK/dV 在 L0C 累加到换 s2
    再写回。

    bn2_need_zero==1（整行或整列无效 tile）：pre 按 metadata 行区间把
    dq/dk/dv 三个 out 全部清零，再 sync_all。不分行/列。不清 workspace。
    """
    coreNum = pl.get_block_num()
    # BN2 合轴分核，kernel 内不启用 swizzle
    constInfo = set_const_info(tiling, coreNum, 0)

    viewD2 = constInfo.viewD2Q
    dSize = constInfo.dSize
    d2d = viewD2 * dSize
    s1nd = constInfo.s1Size * d2d
    s2nd = constInfo.s2Size * d2d

    tensor_q = pl.make_tensor(q, [constInfo.viewD0Q, constInfo.s1Size, viewD2, dSize],
                              [s1nd, d2d, dSize, 1], dtype=pl.DT_FP16)
    tensor_k = pl.make_tensor(k, [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
                              [s2nd, d2d, dSize, 1], dtype=pl.DT_FP16)
    # MM3 需要未转置的 K，必须与 MM2 的 K^T 视图分开(layout 各自推导)
    tensor_k_nt = pl.make_tensor(k, [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
                                 [s2nd, d2d, dSize, 1], dtype=pl.DT_FP16)
    tensor_v = pl.make_tensor(v, [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
                              [s2nd, d2d, dSize, 1], dtype=pl.DT_FP16)
    tensor_dy = pl.make_tensor(dout, [constInfo.viewD0Q, constInfo.s1Size, viewD2, dSize],
                               [s1nd, d2d, dSize, 1], dtype=pl.DT_FP16)
    tensor_y = pl.make_tensor(attn_out, [constInfo.viewD0Q, constInfo.s1Size, viewD2, dSize],
                              [s1nd, d2d, dSize, 1], dtype=pl.DT_FP16)
    tensor_lse = pl.make_tensor(
        softmax_lse, [constInfo.bSize, constInfo.n2Size, constInfo.s1Size],
        [constInfo.n2Size * constInfo.s1Size, constInfo.s1Size, 1],
        dtype=pl.DT_FP32)
    if mask_mode != 0:
        tensor_mask = pl.make_tensor(
            attn_mask, [ATTEN_MASK_COMPRESS, ATTEN_MASK_COMPRESS],
            [ATTEN_MASK_COMPRESS, 1], dtype=pl.DT_UINT8)
    # 输出直接就是 fp16：单块路径没有 fp32 workspace，也就没有 pre/post
    tensor_dq_out = pl.make_tensor(dq, [constInfo.viewD0Q, constInfo.s1Size, viewD2, dSize],
                                   [s1nd, d2d, dSize, 1], dtype=pl.DT_FP16)
    tensor_dk_out = pl.make_tensor(dk, [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
                                   [s2nd, d2d, dSize, 1], dtype=pl.DT_FP16)
    tensor_dv_out = pl.make_tensor(dv, [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
                                   [s2nd, d2d, dSize, 1], dtype=pl.DT_FP16)

    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    # workspace 视图两边都建：单块 iterate_mm_dsk_bn2 被 tilingkey 剪掉 store，
    # 不能把 UB 或 fp16 out 冒充成 fp32 GM 参数。
    alignS1 = constInfo.s1Outer * CUBE_BASEM
    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    tensor_dq_ws = pl.make_tensor(
        ws_ptr, [constInfo.coreNum * alignS1, dSize], [dSize, 1],
        dtype=pl.DT_FP32)
    coreNum = constInfo.coreNum
    # ---- 分核区间直接取自 metadata（AICPU DoFagBn2DenseSplit 的结果）----
    # BN2 下 s1Outer=s2Outer=1，一个块就是一个 (b,n2) head，与 metadata
    # 的块编号一一对应。host 按同一公式 SetBlockDim(blockOuter)；若仍有
    # 多启的核，拿到 start==end 则 numBlocks=0，安全空转。
    # cube/vector 两侧读的是同一块 GM，故 numBlocks 天然一致(set/wait 才配得上)。
    meta = pl.make_tensor(metadata, [META_VIEW_LEN], [1], dtype=pl.DT_INT32)
    fagOff = pl.getval(meta, META_HEAD_FAG_START_INDEX)
    blockStart = pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET + cBlockIdx)
    blockEnd = pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET + cBlockIdx)
    numBlocks = pl.max(blockEnd - blockStart, 0)

    mm1_res = pl.make_tile_group(
        type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[UB_MM1_0, UB_MM1_1], mutex_ids=[0, 1])
    mm2_res = pl.make_tile_group(
        type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[UB_MM2_0, UB_MM2_1], mutex_ids=[2, 3])
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
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
        addrs=[L1_DS], mutex_ids=[10])
    p_t_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_BASEN, CUBE_BASEM], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
        addrs=[L1_P], mutex_ids=[11])

    # skip_invalid 不写整行/整列无效 tile，out GM 会留脏数据。本编译位
    # 下 pre 清 dq/dk/dv out（三个都清，不分行/列），再 sync_all。
    # 行区间与 GS1S2 共用 metadata FagSplitRows，不在核内按元素 ceil 切。
    # 不清 MultiBlk workspace：cube 首写仍是 AtomicNone。
    if bn2_need_zero == 1:
        with pl.section_vector():
            n1Size = constInfo.n2Size * constInfo.gSize
            dqElems = (constInfo.bSize * constInfo.s1Size *
                       n1Size * constInfo.dSize)
            dkElems = (constInfo.bSize * constInfo.s2Size *
                       constInfo.n2Size * constInfo.dSize)
            dvElems = (constInfo.bSize * constInfo.s2Size *
                       constInfo.n2Size * constInfo.dvSize)
            tensor_dq_flat = pl.make_tensor(
                dq, [1, dqElems], [dqElems, 1], dtype=pl.DT_FP16)
            tensor_dk_flat = pl.make_tensor(
                dk, [1, dkElems], [dkElems, 1], dtype=pl.DT_FP16)
            tensor_dv_flat = pl.make_tensor(
                dv, [1, dvElems], [dvElems, 1], dtype=pl.DT_FP16)
            zero_ub = pl.make_tile_group(
                type=pl.TileType(shape=[1, PRE_ELEMS[d_align]],
                                 dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Vec,
                                 valid_shape=[-1, -1]),
                addrs=[UB_PRE_ZERO], mutex_ids=[19])
            process_pre_from_meta(
                meta, fagOff, tensor_dq_flat, tensor_dk_flat, tensor_dv_flat,
                constInfo.dSize, constInfo.dvSize, zero_ub.current())
        pl.system.sync_all()

    with pl.section_cube():
        q_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, HEAD_DIM_ALIGN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, compact=1),
            addrs=[L1_Q, L1_Q2, L1_Q3, L1_Q4], mutex_ids=[4, 31, 21, 18])
        k_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[HEAD_DIM_ALIGN, CUBE_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.ZN, compact=1),
            addrs=[L1_K], mutex_ids=[5])
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
            addrs=[L1_DO, L1_DO2, L1_DO3, L1_DO4], mutex_ids=[7, 15, 22, 19])
        left = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, CUBE_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Left, compact=1),
            addrs=[L0A_ADDR, L0A_ADDR2], mutex_ids=[12, 26])
        right = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, HEAD_DIM_ALIGN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Right, compact=1),
            addrs=[L0B_ADDR, L0B_ADDR2], mutex_ids=[13, 27])
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEM, CUBE_BASEN], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[L0C_ADDR, L0C_ADDR2], mutex_ids=[14, 28])
        dk_acc = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, HEAD_DIM_ALIGN], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[L0C_DK], mutex_ids=[16])
        dv_acc = pl.make_tile_group(
            type=pl.TileType(shape=[CUBE_BASEN, HEAD_DIM_ALIGN], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=1),
            addrs=[L0C_DV], mutex_ids=[17])

        runInfos = make_run_infos()
        taskId = 0
        localIdx = 0
        denseCursor = blockStart
        isLastLoop = 0
        drainDone = 0
        # 1-ahead：ASC ComputeDqkvBn2 while + GetNextValidIdx==-1 排空 1 轮
        while True:
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            q_cur = q_l1.next()
            do_cur = do_l1.next()
            if isLastLoop == 0:
                gIdx = next_work_gidx(constInfo, cBlockIdx, coreNum,
                                      blockStart, blockEnd, localIdx,
                                      denseCursor, numBlocks)
                if gIdx < 0:
                    isLastLoop = 1
                else:
                    denseCursor = gIdx + 1
                    curInfo = runInfos[taskId % 2]
                    set_run_info(constInfo, curInfo, taskId, gIdx, 0, localIdx)
                    if mask_mode != 0:
                        apply_s2_idx_no_change(runInfos, curInfo, taskId, 2)

                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_DETER_FIX_FLAG)
                    iterate_mm_qk_bn2(constInfo, curInfo, tensor_q, tensor_k,
                                  q_cur, k_l1, left, right, acc, mm2_cur, curInfo.loadKV)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                             event_id=SYNC_C2_TO_V2_FLAG[taskId % 2])

                    iterate_mm_dyv_bn2(constInfo, curInfo, tensor_dy, tensor_v,
                                   do_cur, v_l1, left, right, acc, mm1_cur, curInfo.loadKV)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                             event_id=SYNC_C1_TO_V2_FLAG[taskId % 2])

            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                mm1_prev = mm1_res.previous()
                mm2_prev = mm2_res.previous()
                q_prev = q_l1.previous()
                do_prev = do_l1.previous()
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                          event_id=SYNC_V3_TO_C3_FLAG)
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                          event_id=SYNC_V4_TO_C5_FLAG)

                iterate_mm_dsk_bn2(constInfo, prevInfo, tensor_k_nt,
                                   ds_l1, k_l1_n, left, right, acc,
                                   mm1_prev, tensor_dq_ws, cBlockIdx,
                                   prevInfo.loadKV)
                if is_bn2_multiblk == 1:
                    if prevInfo.dqLast == 1:
                        pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                                 event_id=SYNC_C3_TO_V5_FLAG)
                    iterate_mm_dsq_bn2(constInfo, prevInfo,
                                       ds_t_l1.current(), q_prev,
                                       left, right, dk_acc, prevInfo.dkvAcc)
                    if prevInfo.flushDkv == 1:
                        pl.set_validshape(mm2_prev, [VECTOR_BASEM, VECTOR_BASEN])
                        pl.move(mm2_prev, dk_acc.current(),
                                acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                        pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                                 event_id=SYNC_C4_TO_V6_FLAG)
                    iterate_mm_pdy_bn2(constInfo, prevInfo,
                                       p_t_l1.current(), do_prev,
                                       left, right, dv_acc, prevInfo.dkvAcc)
                    if prevInfo.flushDkv == 1:
                        pl.store(tensor_dv_out, dv_acc.current(),
                                 [prevInfo.idx0KV, prevInfo.keyOffset,
                                  prevInfo.idx2KV, 0],
                                 order=[1, 3], atomic=pl.AtomicType.AtomicNone)
                else:
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                             event_id=SYNC_C3_TO_V5_FLAG)
                    iterate_mm_dsq_bn2(constInfo, prevInfo,
                                   ds_t_l1.current(), q_prev,
                                   left, right, dk_acc, 0)
                    pl.set_validshape(mm2_prev, [VECTOR_BASEM, VECTOR_BASEN])
                    pl.move(mm2_prev, dk_acc.current(),
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                             event_id=SYNC_C4_TO_V6_FLAG)
                    iterate_mm_pdy_bn2(constInfo, prevInfo,
                                   p_t_l1.current(), do_prev,
                                   left, right, dv_acc, 0)
                    pl.store(tensor_dv_out, dv_acc.current(),
                             [prevInfo.idx0KV, prevInfo.keyOffset, prevInfo.idx2KV, 0],
                             order=[1, 3], atomic=pl.AtomicType.AtomicNone)
                pl.system.set_cross_core(pipe=pl.PipeType.MTE1,
                                         event_id=SYNC_C5_TO_V4_FLAG)
                pl.system.set_cross_core(pipe=pl.PipeType.MTE1,
                                         event_id=SYNC_C4_TO_V3_FLAG)

            if isLastLoop == 1:
                drainDone = drainDone + 1
                if drainDone >= 1:
                    break
            taskId = taskId + 1
            localIdx = localIdx + 1

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        y_ub = pl.make_tile_group(
            type=pl.TileType(shape=[BN2_V1_ROWS, d_align], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[BN2_UB_Y], mutex_ids=[15])
        dy_ub = pl.make_tile_group(
            type=pl.TileType(shape=[BN2_V1_ROWS, d_align], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[BN2_UB_DY], mutex_ids=[16])
        nzp_ub = pl.make_tile_group(
            type=pl.TileType(shape=[NZP_ROWS, VECTOR_BASEN], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
                             valid_shape=[-1, -1], compact=2),
            addrs=[BN2_UB_NZP], mutex_ids=[19])
        sfmg_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[BN2_UB_SFMG], mutex_ids=[23])
        lse_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, 1], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=[BN2_UB_LSE_0, BN2_UB_LSE_1], mutex_ids=[24, 25])
        mask_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[BN2_UB_Y], mutex_ids=[15])
        mask_pre_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, VECTOR_BASEN], dtype=pl.DT_UINT8,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[BN2_UB_DY], mutex_ids=[16])
        cast_ub = pl.make_tile_group(
            type=pl.TileType(shape=[VECTOR_BASEM, BN2_CAST_W], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1]),
            addrs=[BN2_UB_CAST], mutex_ids=[20])

        runInfos = make_run_infos()
        taskId = 0
        localIdx = 0
        denseCursor = blockStart
        isLastLoop = 0
        drainDone = 0
        while True:
            mm2_res.next()
            mm1_res.next()
            lse_ub.next()
            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                process_vec1_bn2(constInfo, prevInfo, subIdx, tensor_y, tensor_dy,
                             y_ub.current(), dy_ub.current(),
                             sfmg_ub.current(), d_align)

            if isLastLoop == 0:
                gIdx = next_work_gidx(constInfo, cBlockIdx, coreNum,
                                      blockStart, blockEnd, localIdx,
                                      denseCursor, numBlocks)
                if gIdx < 0:
                    isLastLoop = 1
                else:
                    denseCursor = gIdx + 1
                    curInfo = runInfos[taskId % 2]
                    set_run_info(constInfo, curInfo, taskId, gIdx, subIdx, localIdx)
                    if mask_mode != 0:
                        apply_s2_idx_no_change(runInfos, curInfo, taskId, 2)

            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                mm1_t = mm1_res.previous()
                mm2_t = mm2_res.previous()
                lse_t = lse_ub.previous()
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V,
                    event_id=SYNC_C2_TO_V2_FLAG[(taskId + 1) % 2])
                if mask_mode != 0:
                    process_vec2(constInfo, prevInfo, subIdx, tensor_lse,
                                 mm2_t, lse_t,
                                 tensor_mask, mask_ub.current(), mask_pre_ub.current())
                else:
                    process_vec2(constInfo, prevInfo, subIdx, tensor_lse,
                                 mm2_t, lse_t,
                                 tensor_lse, mask_ub.current(), mask_pre_ub.current())
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V,
                    event_id=SYNC_C1_TO_V2_FLAG[(taskId + 1) % 2])
                needSyncDkMM = taskId > 1
                if needSyncDkMM:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE3,
                                              event_id=SYNC_C4_TO_V3_FLAG)
                process_vec3_bn2(constInfo, prevInfo, subIdx, ds_l1.current(),
                                 mm1_t, mm2_t, sfmg_ub.current(),
                                 nzp_ub.current())
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                         event_id=SYNC_V3_TO_C3_FLAG)
                if needSyncDkMM:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE3,
                                              event_id=SYNC_C5_TO_V4_FLAG)
                process_vec4(constInfo, prevInfo, subIdx, p_l1.current(),
                             mm2_t, nzp_ub.current())
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                         event_id=SYNC_V4_TO_C5_FLAG)
                if is_bn2_multiblk == 1:
                    if prevInfo.dqLast == 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE2, event_id=SYNC_C3_TO_V5_FLAG)
                        process_dq_multiblk(
                            tensor_dq_out, tensor_dq_ws, mm1_t,
                            cast_ub.current(), constInfo, prevInfo, subIdx,
                            cBlockIdx)
                    if prevInfo.flushDkv == 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG)
                        process_muls_cast_dk_bn2(
                            tensor_dk_out, mm2_t, cast_ub.current(),
                            constInfo, prevInfo, subIdx)
                    pl.system.set_cross_core(pipe=pl.PipeType.V,
                                             event_id=SYNC_DETER_FIX_FLAG)
                else:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.V, event_id=SYNC_C3_TO_V5_FLAG)
                    process_muls_cast_dq_bn2(
                        tensor_dq_out, mm1_t, cast_ub.current(),
                        constInfo, prevInfo, subIdx)
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG)
                    process_muls_cast_dk_bn2(
                        tensor_dk_out, mm2_t, cast_ub.current(),
                        constInfo, prevInfo, subIdx)
                    pl.system.set_cross_core(pipe=pl.PipeType.V,
                                             event_id=SYNC_DETER_FIX_FLAG)

            if isLastLoop == 1:
                drainDone = drainDone + 1
                if drainDone >= 1:
                    break
            taskId = taskId + 1
            localIdx = localIdx + 1


@pl.jit(
    auto_mutex=True,
    tiling_key=FlashAttnGradTilingKey,
    datatype={
        "q": "input_dtype",
    },
)
def flash_attn_grad(
    # 13 个输入，顺序严格对齐 op_host/flash_attn_grad_def.cpp 的 Input() 声明
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
):
    """入口只做编译期分发：template 来自 tilingkey。

    0=BN2GS1S2，1=BN2（含 MultiBlk）。3 预留给 BN2S2，本轮 is_valid 裁掉。
    BN2 不接 GQA / D=192。
    """
    if template == 1:
        flash_attn_grad_bn2(q, k, v, dout, attn_out, softmax_lse, attn_mask,
                            metadata, dq, dk, dv, workspace, tiling, d_align)
    else:
        flash_attn_grad_bn2gs1s2(q, k, v, dout, attn_out, softmax_lse,
                                 cu_seqlens_q, cu_seqlens_kv, seqused_q,
                                 seqused_kv, sinks, attn_mask, metadata,
                                 dq, dk, dv, workspace, tiling,
                                 d_align, dv_align, swizzle)
