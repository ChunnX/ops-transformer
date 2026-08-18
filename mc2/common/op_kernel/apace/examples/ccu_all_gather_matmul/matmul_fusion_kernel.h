/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_fusion_kernel.h
 * \brief 第二阶段融合 kernel 的 device 侧组件：MatmulTilingData + Blaze 类型链 + MatmulProcess + SyncBridgeProcess
 *
 * 被 ccu_all_gather_matmul_kernel.asc 中的 all_gather_matmul_kernel（MIX_AIC_1_1）调用。
 * - MatmulProcess：AIC 核执行 Blaze 纯 matmul（FP16 A/B → FP32 累加 → BF16 C）
 * - SyncBridgeProcess：AIV 核执行参数桥 + 唤醒 CCU + 轮询 + CrossCoreSetFlag 唤醒 AIC
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
// ============================================================================
#pragma pack(push, 8)
struct alignas(8) MatmulTilingData {
    uint32_t m{0};          // = rankSize * M（AllGather 后总行数，matmul 的 A 行数）
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
// AIC 分支：MatmulProcess（Blaze matmul + CrossCoreWaitFlag）
// 严格参照 mat_mul_basic.cpp L284-312 的类型链与 Params 组装方式
// ============================================================================
__aicore__ inline void MatmulProcess(__gm__ void *aGM, __gm__ void *bGM, __gm__ void *cGM, MatmulTilingData tilingData)
{
    // ---- 等待 AIV 唤醒：AIV 在 SyncBridgeProcess 末尾 CrossCoreSetFlag<0x2, PIPE_MTE3> ----
    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);

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
// AIV 分支：SyncBridgeProcess（复用第一阶段 comm_sync 参数桥+唤醒+轮询，末尾加 CrossCoreSetFlag）
// 多 AIV 核并发时，仅 block 0 执行参数桥/唤醒/轮询，其余 AIV 核等待 SyncAll 后一起 SetFlag
// ============================================================================
__aicore__ inline void SyncBridgeProcess(__gm__ void *src, __gm__ void *recvBuf, CcuMappedAddrs mappedAddrs,
                                          uint64_t token, uint64_t currentRankSliceInputOffset,
                                          uint64_t currentRankSliceOutputOffset, uint64_t sliceSize,
                                          uint64_t goSize0, uint64_t goSize1, uint64_t goSize2, uint64_t goSize3)
{
    // 仅 block 0 执行参数桥/唤醒/轮询（参照 apace UDMA L251 if(GetBlockIdx() < rankSize_) 门控）
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) == 0) {
        // ============ 参数桥：写 taskArgs[0..9] → GM（逐字节复用第一阶段 comm_sync）============
        __gm__ uint64_t *taskArgs0 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0]);
        __gm__ uint64_t *taskArgs1 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[1]);
        __gm__ uint64_t *taskArgs2 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[2]);
        __gm__ uint64_t *taskArgs3 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[3]);
        __gm__ uint64_t *taskArgs4 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[4]);
        __gm__ uint64_t *taskArgs5 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[5]);
        __gm__ uint64_t *taskArgs6 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[6]);
        __gm__ uint64_t *taskArgs7 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[7]);
        __gm__ uint64_t *taskArgs8 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[8]);
        __gm__ uint64_t *taskArgs9 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[9]);

        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs0, reinterpret_cast<uint64_t>(src));
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs1, reinterpret_cast<uint64_t>(recvBuf));
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs2, token);
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs3, currentRankSliceInputOffset);
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs4, currentRankSliceOutputOffset);
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs5, sliceSize);
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs6, goSize0);
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs7, goSize1);
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs8, goSize2);
        AscendC::WriteGmByPassDCache<uint64_t>(taskArgs9, goSize3);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

        // ============ 唤醒 CCU：写 ckeAddr0 = 1（逐字节复用第一阶段 comm_sync）============
        __gm__ uint64_t *ckeAddr0 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
        __gm__ uint64_t *ckeAddr1 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[1]);
        AscendC::WriteGmByPassDCache<uint64_t>(ckeAddr0, 1);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

        // ============ 轮询等待 CCU 通信完成（逐字节复用第一阶段 comm_sync）============
        while (1) {
            uint64_t waitCke = AscendC::ReadGmByPassDCache(ckeAddr1);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
            AscendC::Nop<800>();
            if (waitCke == 1) {
                break;
            }
        }
    }

    // 所有 AIV 核同步（参照 apace UDMA L256 SyncAll<true>）
    AscendC::SyncAll<true>();

    // ============ 第二阶段新增：唤醒 AIC 核执行 matmul ============
    // 参照 apace UDMA all_gather_mx_matmul_udma_impl.h L248：CrossCoreSetFlag<0x2, PIPE_MTE3>
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
}
