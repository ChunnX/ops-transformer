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

#include "kernel_operator.h"

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
using MmTypeA = AscendC::fp8_e4m3fn_t;
using MmTypeB = AscendC::fp8_e4m3fn_t;
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
// AIC 分支：MatmulProcess（Blaze MXFP8 matmul）
// 所有 AIC 核都会调用本函数（BlockSchedulerQuantBatchMatmulV3 按 GetBlockIdx 切分多核任务）
// GemmUniversal（kernel_qbmm_mx.h）的 Run 内置 if ASCEND_IS_AIV { return; }
// ============================================================================
__aicore__ inline void MatmulProcess(__gm__ void *aGM, __gm__ void *aScaleGM,
                                     __gm__ void *bGM, __gm__ void *bScaleGM,
                                     __gm__ void *cGM, MatmulTilingData tilingData)
{
    // ---- 组装 Blaze MXFP8 matmul 类型链 ----
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, MmTypeA, MmLayoutA, MmTypeB, MmLayoutB,
        MmTypeC, MmLayoutC, MmBiasType, MmLayoutC>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        MmProblemShape, 0, MmLayoutA, MmLayoutB, MmTypeA>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
        MmProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    // ---- __gm__ void* → GM_ADDR（Blaze BlockMmadParams 字段类型为 GM_ADDR）----
    GM_ADDR aGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(aGM));
    GM_ADDR aScaleGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(aScaleGM));
    GM_ADDR bGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(bGM));
    GM_ADDR bScaleGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(bScaleGM));
    GM_ADDR cGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(cGM));

    // ---- 从 tilingData 解包参数构造 Params ----
    using Params = typename MatmulKernel::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using QBMMTiling = typename MatmulKernel::QBMMTiling;

    Params params;
    params.problemShape = MmProblemShape{
        static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n),
        static_cast<int64_t>(tilingData.k), 1};

    params.mmadParams = BlockMmadParams{
        aGmAddr, bGmAddr, cGmAddr, nullptr,    // aGmAddr, bGmAddr, cGmAddr, biasGmAddr
        aScaleGmAddr, bScaleGmAddr};           // scaleAGmAddr, scaleBGmAddr

    params.l1Params = L1Params{
        static_cast<uint64_t>(tilingData.kL1),
        static_cast<uint64_t>(tilingData.scaleKL1),
        static_cast<uint64_t>(MmTilingConfig::L1_BUFFER_NUM)};

    params.schParams = BlockSchedulerParams{
        static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN),
        static_cast<int64_t>(tilingData.mTailTile), static_cast<int64_t>(tilingData.nTailTile),
        static_cast<int64_t>(tilingData.mBaseTailSplitCnt), static_cast<int64_t>(tilingData.nBaseTailSplitCnt),
        static_cast<int64_t>(tilingData.mTailMain), static_cast<int64_t>(tilingData.nTailMain)};

    params.qbmmParams = QBMMTiling{
        1, 1, 1, 1,    // batchA1-A4
        1, 1, 1, 1,    // batchB1-B4
        1, 1, 1, 1,    // batchC1-C4
        0,             // biasThreeDim
        tilingData.baseM, tilingData.baseN, tilingData.baseK,
        0,             // isBias
        tilingData.dbL0c > 1 ? 2U : 1U,
        1U};           // bMustHitL2

    MatmulKernel kernel;
    kernel(params);
}

// ============================================================================
// 第四阶段：AIC 直轮询（MXFP8 版本）
//
// 通信分两阶段：
//   阶段 4.1: Scale AllGather（1 轮，全量 M 行 scale）
//   阶段 4.2: Data AllGather（commTurn 轮，每轮 tileM 行 FP8 data）
//
// 参数桥 Variable 布局：
//   组 0 (scale round): startVa[0][0..9] — sendScaleBuf, recvScaleBuf, token, scaleOffsets, scaleSliceSize, goSize...
//   组 1..commTurn (data round d): startVa[d+1][0..9] — sendBuf, recvBuf, token, dataOffsets, dataSliceSize, goSize...
//
// Event 布局：
//   eventVa[0] = wake（AIC→CCU，仅 scale round 前）
//   eventVa[1] = scale AllGather done（CCU→AIC）
//   eventVa[d+2] = data round d AllGather done（CCU→AIC），d = 0..commTurn-1
//
// matmul tile 偏移（MXFP8，FP8 data = 1 byte, BF16 C = 2 bytes）：
//   A = recvBuf，每 tile 行数 = rankSize*tileM
//   round d: A=recvBuf[d*rankSize*tileM*K*1 : ...]
//            scaleA=recvScaleBuf[d*rankSize*tileM*scaleBytesPerMRow : ...]
//            C=outputBuf[d*rankSize*tileM*N*2 : ...]
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
    // MXFP8 偏移计算常量
    constexpr uint64_t FP8_BYTES = 1;  // sizeof(fp8_e4m3fn_t)
    constexpr uint64_t BF16_BYTES = 2; // sizeof(bfloat16_t)
    constexpr uint64_t MXFP_DIVISOR = 64;
    constexpr uint64_t MXFP_MULTI = 2;

    uint64_t scaleKLen = (tilingData.k + MXFP_DIVISOR - 1) / MXFP_DIVISOR * MXFP_MULTI;
    uint64_t dataBytesPerMRow = static_cast<uint64_t>(tilingData.k) * FP8_BYTES;
    uint64_t scaleBytesPerMRow = scaleKLen * FP8_BYTES;  // E8M0 = 1 byte
    uint64_t cBytesPerM = static_cast<uint64_t>(tilingData.n) * BF16_BYTES;

    // 每 tile 的 A/scaleA/C 字节偏移
    uint64_t tileABytes = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(tileM) * dataBytesPerMRow;
    uint64_t tileScaleABytes = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(tileM) * scaleBytesPerMRow;
    uint64_t tileCBytes = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(tileM) * cBytesPerM;

    // ============ block 0 预写 (commTurn+1) 组参数桥 + 写 wake ============
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) == 0) {
        // ---- 组 0: scale round 参数桥 ----
        {
            uint64_t scaleInputOff = scaleInputOffsets[0];
            uint64_t scaleOutputOff = scaleOutputOffsets[0];

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
        for (uint32_t d = 0; d < commTurn; d++) {
            uint64_t inputOff = inputOffsets[d];
            uint64_t outputOff = outputOffsets[d];

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
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        }

        // 写 wake（eventVa[0]=1，唯一 wake）——唤醒 CCU scale round
        __gm__ uint64_t *wakeAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
        AscendC::WriteGmByPassDCache<uint64_t>(wakeAddr, 1);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    }

    // ============ step 1: 所有核轮询 scale AllGather done（eventVa[1]==1）============
    __gm__ uint64_t *scaleDoneAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[1]);
    while (1) {
        uint64_t waitCke = AscendC::ReadGmByPassDCache(scaleDoneAddr);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        AscendC::Nop<800>();
        if (waitCke == 1) {
            break;
        }
    }

    // ============ step 2: 循环 commTurn 次：轮询 data done + MXFP8 matmul ============
    for (uint32_t d = 0; d < commTurn; d++) {
        // 轮询 data round d done（eventVa[d+2]==1）
        __gm__ uint64_t *doneAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[d + 2]);
        while (1) {
            uint64_t waitCke = AscendC::ReadGmByPassDCache(doneAddr);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
            AscendC::Nop<800>();
            if (waitCke == 1) {
                break;
            }
        }

        // MXFP8 matmul tile d
        // A=recvBuf[d*tileABytes : ...], scaleA=recvScaleBuf[d*tileScaleABytes : ...]
        // B=weightBuf, scaleB=weightScaleBuf → C=outputBuf[d*tileCBytes : ...]
        __gm__ uint8_t *recvBufBytes = reinterpret_cast<__gm__ uint8_t *>(recvBuf);
        __gm__ uint8_t *recvScaleBufBytes = reinterpret_cast<__gm__ uint8_t *>(recvScaleBuf);
        __gm__ uint8_t *outputBufBytes = reinterpret_cast<__gm__ uint8_t *>(outputBuf);
        uint64_t aOffset = static_cast<uint64_t>(d) * tileABytes;
        uint64_t scaleAOffset = static_cast<uint64_t>(d) * tileScaleABytes;
        uint64_t cOffset = static_cast<uint64_t>(d) * tileCBytes;
        MatmulProcess(recvBufBytes + aOffset,
                      recvScaleBufBytes + scaleAOffset,
                      weightBuf,
                      weightScaleBuf,
                      outputBufBytes + cOffset,
                      tilingData);
    }
}
