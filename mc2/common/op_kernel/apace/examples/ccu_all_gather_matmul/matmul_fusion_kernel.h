/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file matmul_fusion_kernel.h
 * \brief 第三阶段融合 kernel 的 device 侧组件：MatmulTilingData + Blaze 类型链 + MatmulProcess + AicDirectPollProcess
 *
 * 被 ccu_all_gather_matmul_kernel.asc 中的 all_gather_matmul_kernel（MIX_AIC_1_1）调用。
 * - MatmulProcess：AIC 核执行 Blaze 纯 matmul（FP16 A/B → FP32 累加 → BF16 C），不再等 AIV 唤醒
 * - AicDirectPollProcess：AIC 直轮询（替代第二阶段 AIV SyncBridgeProcess）
 *   · block 0 一次性预写 commTurn 组参数桥（startVa[d][0..9]）+ 写 wake（eventVa[0]=1，唯一 wake）
 *   · 所有 AIC 核循环 commTurn 次：轮询 eventVa[d+1]==1（round d done）→ matmul tile d
 *     ←── 与 CCU round d+1 通信并行
 *   · CCU kernel 仅 round 0 EventWait(events[0])，之后串行自动进入下一 round（无需再唤醒）
 *   · 每轮 done 用独立 Event 地址（eventVa[d+1]），AIC 不会读到残留值，无需重置
 *   · 不使用 SyncAll（AIV 已 return，SyncAll<true> 会死锁）；各核独立轮询 CKE 的 GM 后背地址
 */

#pragma once

#include "kernel_operator.h"

// ---- Blaze matmul 组件（参照 ops-tensor examples/mat_mul/mat_mul_basic）----
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "platform/platform_ascendc.h"

// ============================================================================
// MatmulTilingData POD 结构（host/device 共用，参照 apace quant_matmul_tiling_data.h 风格）
// 每 round 一组，M = rankSize * tileM（AllGather 后单 round 的总行数）
// ============================================================================
#pragma pack(push, 8)
struct alignas(8) MatmulTilingData {
    uint32_t m{0};          // = rankSize * tileM（单 round AllGather 后总行数，matmul 的 A 行数）
    uint32_t n{0};          // = N
    uint32_t k{0};          // = K
    uint32_t baseM{0};      // = 256
    uint32_t baseN{0};      // = 256
    uint32_t baseK{0};      // = 64（mat_mul_basic 默认）
    uint32_t mL1{0};        // = baseM
    uint32_t nL1{0};        // = baseN
    uint32_t kL1{0};        // = baseK * 2
    uint32_t mTailCnt{1};   // 尾块拆分
    uint32_t nTailCnt{1};
    uint32_t usedCoreNum{0};// AIC 核数（MIX 1:1 下为 AIV+AIC 对数）
    uint8_t  isHf32{0};     // FP16 输入时 = 0（FP32 累加）
};
#pragma pack(pop)

// ============================================================================
// Blaze matmul 类型链（严格参照 mat_mul_basic.cpp L281-297，纯 matmul，不混入 QBMM 组件）
// ============================================================================
using MmTypeA = half;
using MmTypeB = half;
using MmTypeC = bfloat16_t;
using MmBiasType = float;
using MmLayoutA = AscendC::Te::NDExtLayoutPtn;  // ND row-major
using MmLayoutB = AscendC::Te::NDExtLayoutPtn;
using MmLayoutC = AscendC::Te::NDExtLayoutPtn;
using MmProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

// Tiling 常量（参照 mat_mul_basic.cpp TilingConfig）
struct MmTilingConfig {
    static constexpr int L1_BUFFER_NUM = 2;
    static constexpr int L0C_DB = 1;
};

// ============================================================================
// AIC 分支：MatmulProcess（Blaze matmul，去 CrossCoreWaitFlag——不再等 AIV 唤醒）
// 第三阶段：AIC 直轮询模式，matmul 由 AicDirectPollProcess 在轮询通过后直接调用
// 所有 AIC 核都会调用本函数（Blaze BlockSchedulerMatmulBasic 按 GetBlockIdx 切分多核任务）
// 严格参照 mat_mul_basic.cpp L284-312 的类型链与 Params 组装方式
// ============================================================================
__aicore__ inline void MatmulProcess(__gm__ void *aGM, __gm__ void *bGM, __gm__ void *cGM, MatmulTilingData tilingData)
{
    // ---- 组装 Blaze matmul 类型链（参照 mat_mul_basic.cpp L293-297）----
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, MmTypeA, MmLayoutA, MmTypeB, MmLayoutB,
                                                     MmTypeC, MmLayoutC, MmBiasType, MmLayoutC>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<MmProblemShape>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<MmProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    // ---- __gm__ void* → GM_ADDR（Blaze BlockMmadParams 字段类型为 GM_ADDR）----
    // 参照 apace all_gather_mx_matmul_udma_impl.h L235: reinterpret_cast<GM_ADDR>(addrVal)
    // __gm__ void* 先转 uint64_t（阶段一 comm_sync 已验证），再转 GM_ADDR
    GM_ADDR aGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(aGM));
    GM_ADDR bGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(bGM));
    GM_ADDR cGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(cGM));

    // ---- 从 tilingData 解包参数构造 Params（参照 mat_mul_basic.cpp L299-308）----
    using Params = typename MatmulKernel::Params;
    Params params = {
        {static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n),
         static_cast<int64_t>(tilingData.k), 1},                                    // ProblemShape
        {aGmAddr, bGmAddr, cGmAddr, nullptr, nullptr, nullptr,                     // BlockMmadParams
         static_cast<uint64_t>(tilingData.mL1), static_cast<uint64_t>(tilingData.nL1),
         static_cast<uint64_t>(tilingData.kL1),
         tilingData.baseM, tilingData.baseN, tilingData.baseK,
         static_cast<uint32_t>(MmTilingConfig::L1_BUFFER_NUM),
         static_cast<uint16_t>(MmTilingConfig::L0C_DB)},
        {},                                                                         // BlockEpilogueParams (empty)
        {tilingData.mL1, tilingData.nL1, tilingData.kL1,                            // BlockSchedulerParams
         tilingData.baseM, tilingData.baseN, tilingData.baseK,
         tilingData.mTailCnt, tilingData.nTailCnt,
         1, 1, 1, 1, tilingData.isHf32, Blaze::Gemm::L2_CACHE_DEFAULT}
    };

    MatmulKernel kernel;
    kernel(params);
}

// ============================================================================
// 第三阶段新增：AIC 直轮询（替代第二阶段 AIV SyncBridgeProcess）
//
// commTurn 通用设计（新方案：1 组 Variable + 1 组 Event，CCU kernel 内循环）：
//   1. block 0 预写 commTurn 组参数桥（startVa[d][0..9]）+ 写 wake（eventVa[0]=1，唯一 wake）
//      CCU kernel 仅 round 0 EventWait(events[0])，round d 完成后自动串行进入 round d+1（无需再唤醒）
//   2. 循环 commTurn 次：所有核轮询该 round done（eventVa[d+1]==1）→ matmul tile d
//      round d 的 matmul 与 CCU round d+1 通信并行 overlap
//   每轮 done 用独立 Event 地址（eventVa[d+1]），AIC 不会读到残留值，无需重置
//
// 参数桥写入逻辑复用第二阶段 SyncBridgeProcess（WriteGmByPassDCache 写 taskArgs[0..9] → GM）。
// 每 round 的 inputOffset/outputOffset 从 device GM 数组读取（d_inputOffsets[d]/d_outputOffsets[d]）。
//
// matmul tile 偏移：A = recvBuf，每 tile 行数 = rankSize*tileM
//   round d: A=recvBuf[d*rankSize*tileM*K*2 : ...], C=outputBuf[d*rankSize*tileM*N*2 : ...]
// ============================================================================
__aicore__ inline void AicDirectPollProcess(
    __gm__ void *sendBuf, __gm__ void *recvBuf, __gm__ void *weightBuf, __gm__ void *outputBuf,
    CcuMappedAddrs mappedAddrs,
    uint64_t token, uint64_t sliceSize,
    uint64_t goSize0, uint64_t goSize1, uint64_t goSize2, uint64_t goSize3,
    MatmulTilingData tilingData, uint32_t rankSize, uint32_t tileM, uint32_t commTurn,
    __gm__ const uint64_t *inputOffsets, __gm__ const uint64_t *outputOffsets)
{
    // 每 tile 的 A/C 字节偏移（FP16 A / BF16 C，均 2 bytes）
    uint64_t tileABytes = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(tileM)
                        * static_cast<uint64_t>(tilingData.k) * 2;
    uint64_t tileCBytes = static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(tileM)
                        * static_cast<uint64_t>(tilingData.n) * 2;

    // ============ block 0 预写 commTurn 组参数桥 + 写 wake（eventVa[0]=1，唯一 wake）============
    // 一次性预写所有 round 的 taskArgs，让 CCU kernel 在 streamCcu 内串行循环时每轮读到对应 Variable 组。
    // CCU kernel 仅 round 0 EventWait(events[0])，round d 完成后自动进入 round d+1（无需再唤醒）。
    // round d 的 matmul 与 CCU round d+1 的通信真正 overlap。
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) == 0) {
        for (uint32_t d = 0; d < commTurn; d++) {
            uint64_t inputOffset = inputOffsets[d];
            uint64_t outputOffset = outputOffsets[d];

            __gm__ uint64_t *args0 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][0]);
            __gm__ uint64_t *args1 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][1]);
            __gm__ uint64_t *args2 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][2]);
            __gm__ uint64_t *args3 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][3]);
            __gm__ uint64_t *args4 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][4]);
            __gm__ uint64_t *args5 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][5]);
            __gm__ uint64_t *args6 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][6]);
            __gm__ uint64_t *args7 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][7]);
            __gm__ uint64_t *args8 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][8]);
            __gm__ uint64_t *args9 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[d][9]);

            AscendC::WriteGmByPassDCache<uint64_t>(args0, reinterpret_cast<uint64_t>(sendBuf));
            AscendC::WriteGmByPassDCache<uint64_t>(args1, reinterpret_cast<uint64_t>(recvBuf));
            AscendC::WriteGmByPassDCache<uint64_t>(args2, token);
            AscendC::WriteGmByPassDCache<uint64_t>(args3, inputOffset);
            AscendC::WriteGmByPassDCache<uint64_t>(args4, outputOffset);
            AscendC::WriteGmByPassDCache<uint64_t>(args5, sliceSize);
            AscendC::WriteGmByPassDCache<uint64_t>(args6, goSize0);
            AscendC::WriteGmByPassDCache<uint64_t>(args7, goSize1);
            AscendC::WriteGmByPassDCache<uint64_t>(args8, goSize2);
            AscendC::WriteGmByPassDCache<uint64_t>(args9, goSize3);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        }

        // 写 wake（eventVa[0]=1，唯一 wake）——唤醒 CCU round 0
        // CCU kernel 内部串行循环，round d 完成后自动进入 round d+1，不需要再唤醒
        __gm__ uint64_t *wakeAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
        AscendC::WriteGmByPassDCache<uint64_t>(wakeAddr, 1);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    }

    // ============ 循环 commTurn 次：轮询 done + matmul ============
    for (uint32_t d = 0; d < commTurn; d++) {

        // ============ step 2：所有核轮询该 round done（eventVa[d+1]==1）============
        // 每轮 done 用独立 Event 地址，AIC 不会读到残留值，无需重置
        __gm__ uint64_t *doneAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[d + 1]);
        while (1) {
            uint64_t waitCke = AscendC::ReadGmByPassDCache(doneAddr);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
            AscendC::Nop<800>();
            if (waitCke == 1) {
                break;
            }
        }

        // ============ step 3：所有核 matmul tile d ============
        // A=recvBuf[d*tileABytes : ...], B=weightBuf → C=outputBuf[d*tileCBytes : ...]
        // ←── CCU round d+1 通信并行（若 d+1 < commTurn）
        // MatmulProcess 参数为 __gm__ void*（阶段二修复：与 kernel 入参类型一致，内部转 GM_ADDR）
        __gm__ uint8_t *recvBufBytes = reinterpret_cast<__gm__ uint8_t *>(recvBuf);
        __gm__ uint8_t *outputBufBytes = reinterpret_cast<__gm__ uint8_t *>(outputBuf);
        uint64_t aOffset = static_cast<uint64_t>(d) * tileABytes;
        uint64_t cOffset = static_cast<uint64_t>(d) * tileCBytes;
        MatmulProcess(recvBufBytes + aOffset,
                      weightBuf,
                      outputBufBytes + cOffset,
                      tilingData);
    }
}
