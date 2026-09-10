/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file matmul_fusion_kernel.h
 * \brief 第四阶段融合 kernel 的 device 侧组件：MatmulTilingData + Blaze MXFP8 类型链 + MatmulProcess + AicDirectPollProcess
 *
 * 被 ccu_all_gather_matmul_kernel.asc 中的 all_gather_matmul_kernel 调用。
 * - MatmulProcess：AIC 核执行 Blaze MXFP8 matmul（FP8 E4M3FN A/B + E8M0 scale → FP32 累加 → BF16 C）
 * - AicDirectPollProcess：AIC 直轮询
 *   · block 0 一次性预写 (commTurn+1) 组参数桥（组0=scale, 组1..commTurn=data）+ 写 wake（eventVa[0]=1）
 *   · 所有核轮询 eventVa[1]==1（scale AllGather done）
 *   · 循环 commTurn 次：轮询 eventVa[d+2]==1（data round d done）→ MXFP8 matmul tile d
 *     ←── 与 CCU round d+1 通信并行
 *   · CCU kernel 仅 round 0 EventWait(events[0])，之后串行自动进入下一 round（无需再唤醒）
 *   · GemmUniversal 的 Run 内置 if ASCEND_IS_AIV { return; }，天然纯 AIC，无需 KERNEL_TYPE_MIX_AIC_1_1
 */

#pragma once

// 与 Blaze block_mmad_qbmm_mx.h 一致的条件 include，确保 fp8_e4m3fn_t / fp8_e8m0_t 可用
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

// ---- Blaze MXFP8 matmul 组件 ----
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "platform/platform_ascendc.h"

// ============================================================================
// MatmulTilingData POD 结构（第四阶段：MXFP8，参照 QuantMatmulTilingData 风格）
// 每 round 一组，M = rankSize * tileM（AllGather 后单 round 的总行数）
// ============================================================================
#pragma pack(push, 8)
struct alignas(8) MatmulTilingData {
    uint32_t m{0};          // = rankSize * tileM（单 round AllGather 后总行数，matmul 的 A 行数）
    uint32_t n{0};          // = N
    uint32_t k{0};          // = K
    uint32_t baseM{0};      // = 128 或 256
    uint32_t baseN{0};      // = 128 或 256
    uint32_t baseK{0};      // = 64（MXFP8 需对齐 MXFP_DIVISOR_SIZE=64）
    uint32_t scaleKL1{0};   // L1 中一次 scale staging 的 K 长度（= baseK * stepK，对齐 64）
    uint32_t mL1{0};        // = baseM
    uint32_t nL1{0};        // = baseN
    uint32_t kL1{0};        // = baseK * stepK
    // scheduler tail 字段（BlockSchedulerQuantBatchMatmulV3::Params）
    uint32_t mTailTile{1};
    uint32_t nTailTile{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};
    // launch + buffer 参数
    uint32_t usedCoreNum{0};
    uint8_t  stepK{0};      // L1 rolling buffer 步进（kL1 = baseK * stepK）
    uint8_t  nBufferNum{0}; // L1 rolling buffer 数（=2 double buffer）
    uint8_t  dbL0c{0};      // L0C double buffer（=1 或 2）
    uint8_t  isHf32{0};     // 保留字段
};
#pragma pack(pop)

// ============================================================================
// Blaze MXFP8 matmul 类型链
// FP8 E4M3FN A(ND) / FP8 E4M3FN B(DN) + E8M0 scaleA/scaleB → FP32 累加 → BF16 C
// 使用 MatmulWithScaleMx dispatch policy + BlockSchedulerQuantBatchMatmulV3 scheduler
// GemmUniversal 的 MX 特化（kernel_qbmm_mx.h）内置 if ASCEND_IS_AIV { return; }，天然纯 AIC
// ============================================================================
using MmTypeA = fp8_e4m3fn_t;
using MmTypeB = fp8_e4m3fn_t;
using MmTypeC = bfloat16_t;
using MmBiasType = float;
using MmLayoutA = AscendC::Te::NDExtLayoutPtn;  // ND row-major
using MmLayoutB = AscendC::Te::DNExtLayoutPtn;  // DN col-major (transB)
using MmLayoutC = AscendC::Te::NDExtLayoutPtn;
using MmProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

// Tiling 常量
struct MmTilingConfig {
    static constexpr int L1_BUFFER_NUM = 2;
    static constexpr int L0C_DB = 1;
};

// ============================================================================
// MatmulProcess — 单次 MXFP8 matmul 执行（AIC 核内）
//
// 角色：被 AicDirectPollProcess 的 commTurn 循环体调用，每 round 执行一次。
// 输入：A = AllGather 后的 FP8 E4M3FN 数据（recvBuf 的对应 tile 切片）
//       scaleA = AllGather 后的 E8M0 scale（recvScaleBuf 的对应 tile 切片）
//       B / scaleB = 本 rank 的 FP8 weight / E8M0 weight scale（不通信，直接从 weightBuf 读）
// 输出：C = BF16 matmul 结果（outputBuf 的对应 tile 切片）
//
// Blaze 类型链组装（四组件）：
//   1) DispatchPolicy = MatmulWithScaleMx<0,0>  → MXFP8 量化 dispatch policy
//   2) BlockMmad       → 量化矩阵乘 block（含 scaleAGmAddr/scaleBGmAddr 字段）
//   3) BlockEpilogue   → 空后处理（不做 bias/activation，C 直接从 L0C 输出到 GM）
//   4) BlockScheduler  → QuantBatchMatmulV3（按 GetBlockIdx 多核切分 M/N 维）
//
// 关键：GemmUniversal 的 MX 特化（kernel_qbmm_mx.h）Run() 内置
//       if ASCEND_IS_AIV { return; }，所以本函数天然只跑在 AIC 核上，
//       无需 KERNEL_TYPE_MIX_AIC_1_1 混合核声明。
//
// 多核切分：BlockSchedulerQuantBatchMatmulV3 按 GetBlockIdx() 将 (M, N) 矩阵
//       切成 baseM×baseN 的 tile 分配给各 AIC 核，所有 AIC 核并行执行同一 kernel。
// ============================================================================
__aicore__ inline void MatmulProcess(__gm__ void *aGM, __gm__ void *aScaleGM,
                                     __gm__ void *bGM, __gm__ void *bScaleGM,
                                     __gm__ void *cGM, MatmulTilingData tilingData)
{
    // ---- 组装 Blaze MXFP8 matmul 类型链（四组件）----
    // DispatchPolicy：MatmulWithScaleMx 启用 MXFP8 量化路径（带 per-64-element scale）
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
    // BlockMmad：核心矩阵乘 block，模板参数依次为
    //   DispatchPolicy, A类型, A布局, B类型, B布局, C类型, C布局, Bias类型, Bias布局
    //   A/B = fp8_e4m3fn_t(1B), C = bfloat16_t(2B), Bias = float（不使用但模板需要）
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, MmTypeA, MmLayoutA, MmTypeB, MmLayoutB,
        MmTypeC, MmLayoutC, MmBiasType, MmLayoutC>;
    // BlockEpilogue：空后处理，C 从 L0C 直接搬到 GM，不做 bias / activation
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    // BlockScheduler：多核切分器，按 GetBlockIdx() 将 (M,N) 切成 baseM×baseN tile
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        MmProblemShape, 0, MmLayoutA, MmLayoutB, MmTypeA>;
    // MatmulKernel：GemmUniversal 的 MX 特化，Run() 内置 AIV early-return
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
        MmProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    // ---- __gm__ void* → GM_ADDR ----
    // Blaze BlockMmadParams 的地址字段类型是 GM_ADDR（uint64_t 别名），
    // 需要 reinterpret_cast 转换后才能赋值
    GM_ADDR aGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(aGM));
    GM_ADDR aScaleGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(aScaleGM));
    GM_ADDR bGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(bGM));
    GM_ADDR bScaleGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(bScaleGM));
    GM_ADDR cGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(cGM));

    // ---- 从 tilingData 解包参数构造 Params ----
    // Params 是 MatmulKernel 的参数聚合体，包含 4 个子参数块：
    //   problemShape  → (M, N, K, batch=1)
    //   mmadParams    → A/B/C/scaleA/scaleB 的 GM 地址
    //   l1Params      → L1 rolling buffer 配置（kL1, scaleKL1, bufferNum）
    //   schParams     → scheduler 切分参数（baseM, baseN, tail 处理）
    //   qbmmParams    → batch matmul tiling（本场景 batch=1，baseM/baseN/baseK 控制切分）
    using Params = typename MatmulKernel::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using QBMMTiling = typename MatmulKernel::QBMMTiling;

    Params params;
    // problemShape: M=单 round AllGather 后总行数, N=N, K=K, batch=1
    params.problemShape = MmProblemShape{
        static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n),
        static_cast<int64_t>(tilingData.k), 1};

    // mmadParams: A/B/C/Bias 的 GM 地址 + scaleA/scaleB 的 GM 地址
    // biasGmAddr=nullptr（不使用 bias）
    params.mmadParams = BlockMmadParams{
        aGmAddr, bGmAddr, cGmAddr, nullptr,    // aGmAddr, bGmAddr, cGmAddr, biasGmAddr
        aScaleGmAddr, bScaleGmAddr};           // scaleAGmAddr, scaleBGmAddr

    // l1Params: kL1=baseK*stepK（L1 K方向 staging 长度）,
    //           scaleKL1=scale 的 L1 K方向长度（对齐 64）,
    //           bufferNum=2（L1 double buffer）
    params.l1Params = L1Params{
        static_cast<uint64_t>(tilingData.kL1),
        static_cast<uint64_t>(tilingData.scaleKL1),
        static_cast<uint64_t>(MmTilingConfig::L1_BUFFER_NUM)};

    // schParams: scheduler 多核切分参数
    //   baseM/baseN = 主块大小，mTailTile/nTailTile 等 = 尾块处理参数
    params.schParams = BlockSchedulerParams{
        static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN),
        static_cast<int64_t>(tilingData.mTailTile), static_cast<int64_t>(tilingData.nTailTile),
        static_cast<int64_t>(tilingData.mBaseTailSplitCnt), static_cast<int64_t>(tilingData.nBaseTailSplitCnt),
        static_cast<int64_t>(tilingData.mTailMain), static_cast<int64_t>(tilingData.nTailMain)};

    // qbmmParams: batch matmul tiling
    //   batchA1-A4 / batchB1-B4 / batchC1-C4 = 全 1（单 batch，非 group matmul）
    //   biasThreeDim=0, isBias=0（不使用 bias）
    //   baseM/baseN/baseK = Cube tile 形状
    //   dbL0c = L0C double buffer 数（1 或 2）
    //   bMustHitL2 = 1（B 矩阵命中 L2 cache）
    params.qbmmParams = QBMMTiling{
        1, 1, 1, 1,    // batchA1-A4
        1, 1, 1, 1,    // batchB1-B4
        1, 1, 1, 1,    // batchC1-C4
        0,             // biasThreeDim
        tilingData.baseM, tilingData.baseN, tilingData.baseK,
        0,             // isBias
        tilingData.dbL0c > 1 ? 2U : 1U,
        1U};           // bMustHitL2

    // ---- 执行 matmul ----
    // 构造 kernel 对象并以 params 调用 operator()，Blaze 内部自动完成：
    //   GM → L1 → L0A/L0B → Cube → L0C → GM 的完整数据搬运和计算流水
    MatmulKernel kernel;
    kernel(params);
}

// ============================================================================
// AicDirectPollProcess — AIC 侧直轮询主逻辑（第四阶段 MXFP8 核心函数）
//
// 整体执行流程（3 步）：
//   步骤 0 (block 0 only): 预写 (commTurn+1) 组参数桥到 GM Variable 区，然后写 wake event
//   步骤 1 (all blocks):   轮询 eventVa[1]==1，等待 Scale AllGather 完成
//   步骤 2 (all blocks):   循环 commTurn 次，每次轮询 eventVa[d+2]==1 等 Data round d 完成，
//                          然后执行 MXFP8 matmul tile d（与 CCU round d+1 通信并行 overlap）
//
// ─── 双流时序示意 ───
//
//   streamCcu (CCU):    [wake]→ ScaleAG → DataAG₀ → DataAG₁ → ...
//                                   ↓ev[1]     ↓ev[2]     ↓ev[3]
//   streamAivAic (AIC):  [writeWake]→ waitScale → Mmul₀ → Mmul₁ → ...
//                                          ↑ev[1]    ↑ev[2]   ↑ev[3]
//
//   关键 overlap：AIC 执行 Mmul₀ 时，CCU 正在执行 DataAG₁，通算并行。
//
// ─── 参数桥 Variable 布局 ───
//   组 0 (scale round): startVa[0][0..9]
//     [0]=sendScaleBuf, [1]=recvScaleBuf, [2]=scaleToken,
//     [3]=scaleInputOff, [4]=scaleOutputOff, [5]=scaleSliceSize,
//     [6..9]=scaleGoSize[0..3]
//   组 1..commTurn (data round d): startVa[d+1][0..9]
//     [0]=sendBuf, [1]=recvBuf, [2]=token,
//     [3]=inputOff[d], [4]=outputOff[d], [5]=dataSliceSize,
//     [6..9]=goSize[0..3]
//
// ─── Event 布局 ───
//   eventVa[0]     = wake   （AIC block0 → CCU，仅 scale round 前，唯一唤醒）
//   eventVa[1]     = scale AllGather done（CCU → AIC，所有核轮询）
//   eventVa[d+2]   = data round d AllGather done（CCU → AIC），d = 0..commTurn-1
//
// ─── MXFP8 matmul tile 偏移计算 ───
//   FP8 data = 1 byte/element, E8M0 scale = 1 byte, BF16 C = 2 bytes
//   每 tile 行数 = rankSize * tileM（AllGather 后单 round 的总行数）
//
//   AllGather 后 recvBuf 中各 rank 数据按 rankId 拼接，每 round 汇聚一组：
//     round d 汇聚区 = [rank0 tileM 行][rank1 tileM 行]...[rankR tileM 行]
//
//   round d 的 matmul 读取地址：
//     A      = recvBuf      + d * tileABytes      (= d * rankSize * tileM * K * 1)
//     scaleA = recvScaleBuf + d * tileScaleABytes (= d * rankSize * tileM * scaleBytesPerMRow)
//     C      = outputBuf    + d * tileCBytes      (= d * rankSize * tileM * N * 2)
//
//   详见函数体内"AllGather 后的数据排布"注释块
// ============================================================================
__aicore__ inline void AicDirectPollProcess(
    __gm__ void *sendBuf, __gm__ void *recvBuf,
    __gm__ void *sendScaleBuf, __gm__ void *recvScaleBuf,
    __gm__ void *weightBuf, __gm__ void *weightScaleBuf, __gm__ void *outputBuf,
    CcuMappedAddrs mappedAddrs,
    uint64_t token, uint64_t scaleToken,
    uint64_t dataSliceSize, uint64_t scaleSliceSize,
    uint64_t goSize0, uint64_t goSize1, uint64_t goSize2, uint64_t goSize3,
    uint64_t scaleGoSize0, uint64_t scaleGoSize1, uint64_t scaleGoSize2, uint64_t scaleGoSize3,
    MatmulTilingData tilingData, uint32_t rankSize, uint32_t tileM, uint32_t commTurn,
    __gm__ const uint64_t *inputOffsets, __gm__ const uint64_t *outputOffsets,
    __gm__ const uint64_t *scaleInputOffsets, __gm__ const uint64_t *scaleOutputOffsets)
{
    // ========================================================================
    // MXFP8 偏移计算常量
    // MXFP8 格式：每 64 个 FP8 元素共享 1 个 E8M0 scale（但 scale 存储按
    //   MXFP_MULTI=2 分组，即 ceil(K/64)*2 个 scale byte per M row）
    // ========================================================================
    constexpr uint64_t FP8_BYTES = 1;  // sizeof(fp8_e4m3fn_t)
    constexpr uint64_t BF16_BYTES = 2; // sizeof(bfloat16_t)
    constexpr uint64_t MXFP_DIVISOR = 64;
    constexpr uint64_t MXFP_MULTI = 2;

    // scaleKLen: 每行 scale 的元素数（byte 数，因为 E8M0 = 1 byte）
    //   = ceil(K / 64) * 2，例如 K=256 → ceil(256/64)*2 = 4*2 = 8
    uint64_t scaleKLen = (tilingData.k + MXFP_DIVISOR - 1) / MXFP_DIVISOR * MXFP_MULTI;
    // 每行 FP8 data 的字节数 = K * 1
    uint64_t dataBytesPerMRow = static_cast<uint64_t>(tilingData.k) * FP8_BYTES;
    // 每行 scale 的字节数 = scaleKLen * 1（E8M0 = 1 byte）
    uint64_t scaleBytesPerMRow = scaleKLen * FP8_BYTES;  // E8M0 = 1 byte
    // 每行 C 输出的字节数 = N * 2（BF16）
    uint64_t cBytesPerM = static_cast<uint64_t>(tilingData.n) * BF16_BYTES;

    // ========================================================================
    // AllGather 后的数据排布（recvBuf / recvScaleBuf / outputBuf）
    //
    // ─── recvBuf（FP8 data，AllGather 后）───
    //
    //   每 round 汇聚 rankSize * tileM 行，各 rank 的 tileM 行按 rankId 拼接：
    //
    //   round 0: [rank0 tileM 行][rank1 tileM 行]...[rankR tileM 行]
    //   round 1: [rank0 tileM 行][rank1 tileM 行]...[rankR tileM 行]
    //   ...
    //
    //   每 tile 的 A 字节数 = rankSize * tileM * K * 1(FP8)
    //   round d 的 A 起始偏移 = d * tileABytes
    //
    // ─── recvScaleBuf（E8M0 scaleA，AllGather 后）───
    //
    //   scale 只 1 轮全量 AllGather，汇聚 rankSize * M 行：
    //   [rank0 的 M 行 scale][rank1 的 M 行 scale]...[rankR 的 M 行 scale]
    //
    //   但 matmul 按 round 切分读取，每 round 取 rankSize * tileM 行 scale：
    //   round d 的 scaleA 起始偏移 = d * tileScaleABytes
    //     = d * rankSize * tileM * scaleBytesPerMRow
    //
    // ─── outputBuf（BF16 C 输出）───
    //
    //   每 round 的 matmul 输出 rankSize * tileM 行 × N 列 × 2(BF16)：
    //   round 0: C[0 : rankSize*tileM*N*2]
    //   round 1: C[tileCBytes : 2*tileCBytes]
    //   round d 的 C 起始偏移 = d * tileCBytes
    //
    // ─── 具体示例（2 rank, M=512, tileM=256, K=256, N=256, commTurn=2）───
    //
    //   recvBuf:  round 0: [r0 rows 0-255][r1 rows 0-255]    = 512 行 × 256 × 1B
    //             round 1: [r0 rows 256-511][r1 rows 256-511] = 512 行 × 256 × 1B
    //   recvScaleBuf: [r0 scale 512行][r1 scale 512行] = 1024 行 × 8 × 1B
    //             round 0 取前 512 行 scale, round 1 取后 512 行 scale
    //   outputBuf: round 0: C[0 : 512*256*2B]     = 256 KB
    //              round 1: C[256KB : 512KB]
    // ========================================================================
    // 每 tile（单 round AllGather 后）的 A / scaleA / C 字节大小
    // tile 行数 = rankSize * tileM（所有 rank 的 tileM 行 AllGather 汇聚）
    uint64_t tileABytes = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(tileM) * dataBytesPerMRow;
    uint64_t tileScaleABytes = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(tileM) * scaleBytesPerMRow;
    uint64_t tileCBytes = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(tileM) * cBytesPerM;

    // ========================================================================
    // 步骤 0：block 0 一次性预写 (commTurn+1) 组参数桥 + 写 wake event
    //
    // 为什么由 AIC block 0 来写参数桥？
    //   CCU kernel 的 Variable 区是 GM 上的共享内存，AIC 和 CCU 都能访问。
    //   AIC block 0 负责"装弹"——把 host 传入的地址/偏移/大小等参数写入
    //   Variable 区的对应槽位，CCU kernel 运行时从这些槽位读取参数。
    //   只需 block 0 做一次（其他 block 跳过），避免重复写。
    //
    // 为什么写完所有 round 的参数桥后才写 wake？
    //   CCU kernel 是串行循环 totalRounds 次，只需唤醒一次（eventVa[0]=1）。
    //   CCU round 0 被 wake 唤醒后，后续 round 自动串行进入，无需再唤醒。
    //   因此必须在 CCU 开始执行前，把所有 round 的参数都准备好。
    // ========================================================================
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) == 0) {
        // ---- 组 0: scale round 参数桥 ----
        // 将 scale round 所需的 10 个参数写入 Variable 区 startVa[0][0..9]
        // CCU kernel round 0 会从 vars[0..9] 读取这些参数执行 scale AllGather
        {
            // scale round 的输入/输出偏移（scale 只 1 轮，全量通信）
            uint64_t scaleInputOff = scaleInputOffsets[0];
            uint64_t scaleOutputOff = scaleOutputOffsets[0];

            // 10 个 Variable 槽位的 GM 后背地址（startVa[0][0..9]）
            __gm__ uint64_t *args0 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][0]);
            __gm__ uint64_t *args1 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][1]);
            __gm__ uint64_t *args2 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][2]);
            __gm__ uint64_t *args3 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][3]);
            __gm__ uint64_t *args4 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][4]);
            __gm__ uint64_t *args5 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][5]);
            __gm__ uint64_t *args6 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][6]);
            __gm__ uint64_t *args7 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][7]);
            __gm__ uint64_t *args8 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][8]);
            __gm__ uint64_t *args9 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0][9]);

            // WriteGmByPassDCache: 绕过 DCache 直写 GM，确保 CCU 核能立即看到最新值
            // 槽位映射：[0]=sendScaleBuf, [1]=recvScaleBuf, [2]=scaleToken,
            //           [3]=scaleInputOff, [4]=scaleOutputOff, [5]=scaleSliceSize,
            //           [6..9]=scaleGoSize[0..3]（CCU 本地拷贝 loop 参数）
            AscendC::WriteGmByPassDCache<uint64_t>(args0, reinterpret_cast<uint64_t>(sendScaleBuf));
            AscendC::WriteGmByPassDCache<uint64_t>(args1, reinterpret_cast<uint64_t>(recvScaleBuf));
            AscendC::WriteGmByPassDCache<uint64_t>(args2, scaleToken);
            AscendC::WriteGmByPassDCache<uint64_t>(args3, scaleInputOff);
            AscendC::WriteGmByPassDCache<uint64_t>(args4, scaleOutputOff);
            AscendC::WriteGmByPassDCache<uint64_t>(args5, scaleSliceSize);
            AscendC::WriteGmByPassDCache<uint64_t>(args6, scaleGoSize0);
            AscendC::WriteGmByPassDCache<uint64_t>(args7, scaleGoSize1);
            AscendC::WriteGmByPassDCache<uint64_t>(args8, scaleGoSize2);
            AscendC::WriteGmByPassDCache<uint64_t>(args9, scaleGoSize3);
        }

        // ---- 组 1..commTurn: data round 参数桥 ----
        // 为每个 data round d 写入 10 个参数到 Variable 区 startVa[d+1][0..9]
        // CCU kernel round d+1 会从 vars[(d+1)*10 .. (d+1)*10+9] 读取执行 data AllGather
        for (uint32_t d = 0; d < commTurn; d++) {
            // data round d 的输入/输出偏移（从 device GM 数组读取）
            uint64_t inputOff = inputOffsets[d];
            uint64_t outputOff = outputOffsets[d];

            // 10 个 Variable 槽位的 GM 后背地址（startVa[d+1][0..9]）
            uint32_t base = (d + 1) * HCOMM_HOST_TASK_ARG_NUM;
            __gm__ uint64_t *args0 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][0]);
            __gm__ uint64_t *args1 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][1]);
            __gm__ uint64_t *args2 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][2]);
            __gm__ uint64_t *args3 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][3]);
            __gm__ uint64_t *args4 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][4]);
            __gm__ uint64_t *args5 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][5]);
            __gm__ uint64_t *args6 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][6]);
            __gm__ uint64_t *args7 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][7]);
            __gm__ uint64_t *args8 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][8]);
            __gm__ uint64_t *args9 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d + 1][9]);

            // 槽位映射：[0]=sendBuf, [1]=recvBuf, [2]=token,
            //           [3]=inputOff[d], [4]=outputOff[d], [5]=dataSliceSize,
            //           [6..9]=goSize[0..3]（CCU 本地拷贝 loop 参数）
            AscendC::WriteGmByPassDCache<uint64_t>(args0, reinterpret_cast<uint64_t>(sendBuf));
            AscendC::WriteGmByPassDCache<uint64_t>(args1, reinterpret_cast<uint64_t>(recvBuf));
            AscendC::WriteGmByPassDCache<uint64_t>(args2, token);
            AscendC::WriteGmByPassDCache<uint64_t>(args3, inputOff);
            AscendC::WriteGmByPassDCache<uint64_t>(args4, outputOff);
            AscendC::WriteGmByPassDCache<uint64_t>(args5, dataSliceSize);
            AscendC::WriteGmByPassDCache<uint64_t>(args6, goSize0);
            AscendC::WriteGmByPassDCache<uint64_t>(args7, goSize1);
            AscendC::WriteGmByPassDCache<uint64_t>(args8, goSize2);
            AscendC::WriteGmByPassDCache<uint64_t>(args9, goSize3);
            // DataSyncBarrier: 确保上述写操作对 CCU 核可见后才进入下一 round 的写
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        }

        // ---- 写 wake event（eventVa[0]=1）----
        // 这是唯一的 AIC→CCU 唤醒信号。CCU kernel round 0 通过 EventWait(events[0])
        // 阻塞等待此信号。唤醒后 CCU 串行执行所有 round，无需再次唤醒。
        __gm__ uint64_t *wakeAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
        AscendC::WriteGmByPassDCache<uint64_t>(wakeAddr, 1);
        // 确保 wake 写操作对所有 CCU 核可见
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    }

    // ========================================================================
    // 步骤 1：所有核轮询 Scale AllGather 完成（eventVa[1]==1）
    //
    // CCU kernel 完成 scale round 后执行 EventRecord(events[1])，将 eventVa[1] 置 1。
    // 所有 AIC 核在此 spin-wait 直到读到 1，确保 scale 数据全部就绪后再进入 data 循环。
    //
    // 轮询三件套：
    //   ReadGmByPassDCache — 绕过 DCache 直读 GM 最新值
    //   DataSyncBarrier    — 确保读操作完成
    //   Nop<800>           — 插入 800 个 nop 降低轮询频率，减少 GM 读带宽占用
    // ========================================================================
    __gm__ uint64_t *scaleDoneAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[1]);
    while (1) {
        uint64_t waitCke = AscendC::ReadGmByPassDCache(scaleDoneAddr);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        AscendC::Nop<800>();
        if (waitCke == 1) {
            break;
        }
    }

    // ========================================================================
    // 步骤 2：循环 commTurn 次 — 轮询 data round d done + 执行 MXFP8 matmul tile d
    //
    // 这是通算并行的核心循环：
    //   - AIC 等待 CCU 完成 data round d 的 AllGather（eventVa[d+2]==1）
    //   - 数据就绪后立即执行 matmul tile d
    //   - 同时 CCU 已开始 round d+1 的通信（overlap）
    //
    // matmul tile d 的地址计算：
    //   A      = recvBuf      + d * tileABytes      （AllGather 后的 FP8 data）
    //   scaleA = recvScaleBuf + d * tileScaleABytes （AllGather 后的 E8M0 scaleA）
    //   B      = weightBuf                         （本 rank FP8 weight，不通信）
    //   scaleB = weightScaleBuf                     （本 rank E8M0 weight scale）
    //   C      = outputBuf    + d * tileCBytes      （BF16 输出）
    // ========================================================================
    for (uint32_t d = 0; d < commTurn; d++) {
        // ---- 轮询 data round d 完成（eventVa[d+2]==1）----
        // CCU kernel round d+1 完成后执行 EventRecord(events[d+2])，将 eventVa[d+2] 置 1
        __gm__ uint64_t *doneAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[d + 2]);
        while (1) {
            uint64_t waitCke = AscendC::ReadGmByPassDCache(doneAddr);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
            AscendC::Nop<800>();
            if (waitCke == 1) {
                break;
            }
        }

        // ---- 执行 MXFP8 matmul tile d ----
        // 计算 tile d 在 recvBuf / recvScaleBuf / outputBuf 中的字节偏移
        // 偏移 = round index d × 每 tile 字节数（tile 行数 = rankSize * tileM）
        __gm__ uint8_t *recvBufBytes = reinterpret_cast<__gm__ uint8_t *>(recvBuf);
        __gm__ uint8_t *recvScaleBufBytes = reinterpret_cast<__gm__ uint8_t *>(recvScaleBuf);
        __gm__ uint8_t *outputBufBytes = reinterpret_cast<__gm__ uint8_t *>(outputBuf);
        uint64_t aOffset = static_cast<uint64_t>(d) * tileABytes;
        uint64_t scaleAOffset = static_cast<uint64_t>(d) * tileScaleABytes;
        uint64_t cOffset = static_cast<uint64_t>(d) * tileCBytes;
        // 调用 Blaze MXFP8 matmul：A/B = FP8 E4M3FN, scaleA/scaleB = E8M0, C = BF16
        MatmulProcess(recvBufBytes + aOffset,
                      recvScaleBufBytes + scaleAOffset,
                      weightBuf,
                      weightScaleBuf,
                      outputBufBytes + cOffset,
                      tilingData);
    }
}
