/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISTA_BLAZE_MX_QUANT_MATMUL_H
#define DISTA_BLAZE_MX_QUANT_MATMUL_H

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "../../block/blaze_ext/gemm/policy/dispatch_policy_alltoall.h"
#include "../../block/blaze_ext/gemm/kernel/kernel_qbmm_mx_alltoall.h"
#include "../../block/blaze_ext/gemm/block/block_mmad_qbmm_mx_alltoall.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "../../utils/qbmm_tiling_data.h"

namespace Dista {
namespace Kernel {

struct BlazeMXQuantMMBaseGmAddrs {
    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR cGM;
    uint64_t aOffset;
    uint64_t bOffset;
    uint64_t cOffset;
};

struct BlazeMXQuantMMAdditionalData {
    GM_ADDR x1Scale;
    GM_ADDR x2Scale;
    uint64_t x1ScaleOffset;
    uint32_t rankDim;
    uint32_t rankN;
    uint32_t rankM;
    uint32_t scaleKPerRank;
};

template <typename TilingDataType>
struct BlazeMXQuantMMContext {
    BlazeMXQuantMMBaseGmAddrs baseData;
    BlazeMXQuantMMAdditionalData additionalData;
    TilingDataType *tilingDataPtr;
};

template <typename AType, typename BType, typename CType>
class BlazeMXQuantMatmul {
public:
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMxAlltoAll<0, false>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                     float, LayoutC>;
    using GemmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<Blaze::Gemm::KernelMmadWithScaleMxAlltoAll,
                                                           typename BlockMmad::DispatchPolicy::ScheduleType>>>;

    using GemmParams = typename GemmKernel::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using SchParams = typename BlockScheduler::Params;
    using QBMMTiling = typename GemmKernel::QBMMTiling;
    using AlltoAllParams = typename BlockMmad::AlltoAllParams;

    using TilingDataType = QuantBatchMatmulTilingData;
    using ContextType = BlazeMXQuantMMContext<TilingDataType>;

    __aicore__ inline BlazeMXQuantMatmul() {}
    __aicore__ inline void Init() {}
    __aicore__ inline ContextType *GetContextPtr() { return &context_; }
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void End() {}

private:
    ContextType context_;
};

template <typename AType, typename BType, typename CType>
__aicore__ inline void BlazeMXQuantMatmul<AType, BType, CType>::Process(uint32_t taskIndex)
{
    GM_ADDR aGM = context_.baseData.aGM + taskIndex * context_.baseData.aOffset;
    GM_ADDR bGM = context_.baseData.bGM + taskIndex * context_.baseData.bOffset;
    GM_ADDR cGM = context_.baseData.cGM + taskIndex * context_.baseData.cOffset;
    GM_ADDR x1Scale = context_.additionalData.x1Scale + taskIndex * context_.additionalData.x1ScaleOffset;
    GM_ADDR x2Scale = context_.additionalData.x2Scale;

    const auto &mmTile = *context_.tilingDataPtr;
    uint32_t rankDim = context_.additionalData.rankDim;
    uint64_t rankKPerRank = static_cast<uint64_t>(mmTile.k) / rankDim;
    uint64_t scaleKPerRank = context_.additionalData.scaleKPerRank;
    uint64_t aRankStride = static_cast<uint64_t>(context_.additionalData.rankM) * rankKPerRank;
    uint64_t scaleARankStride = static_cast<uint64_t>(context_.additionalData.rankM) * scaleKPerRank;

    GemmParams params;

    AscendC::Std::get<Blaze::Gemm::MNK_M>(params.problemShape) = static_cast<int64_t>(mmTile.m);
    AscendC::Std::get<Blaze::Gemm::MNK_N>(params.problemShape) = static_cast<int64_t>(mmTile.n);
    AscendC::Std::get<Blaze::Gemm::MNK_K>(params.problemShape) = static_cast<int64_t>(mmTile.k);
    AscendC::Std::get<Blaze::Gemm::MNK_B>(params.problemShape) = 1;

    BlockMmadParams mmad;
    mmad.aGmAddr = aGM;
    mmad.bGmAddr = bGM;
    mmad.cGmAddr = cGM;
    mmad.scaleAGmAddr = x1Scale;
    mmad.scaleBGmAddr = x2Scale;
    mmad.biasGmAddr = nullptr;
    params.mmadParams = mmad;

    L1Params l1;
    l1.kL1 = static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK;
    l1.scaleKL1 = mmTile.scaleKL1;
    l1.l1BufNum = mmTile.nBufferNum;
    params.l1Params = l1;

    SchParams sch;
    sch.baseM = 256;
    sch.baseN = 128;
    sch.mTailTile = mmTile.mTailTile;
    sch.nTailTile = mmTile.nTailTile;
    sch.mBaseTailSplitCnt = mmTile.mBaseTailSplitCnt;
    sch.nBaseTailSplitCnt = mmTile.nBaseTailSplitCnt;
    sch.mTailMain = mmTile.mTailMain;
    sch.nTailMain = mmTile.nTailMain;
    params.schParams = sch;

    QBMMTiling qb;
    qb.batchA1 = 1;
    qb.batchA2 = 1;
    qb.batchA3 = 1;
    qb.batchA4 = 1;
    qb.batchB1 = 1;
    qb.batchB2 = 1;
    qb.batchB3 = 1;
    qb.batchB4 = 1;
    qb.batchC1 = 1;
    qb.batchC2 = 1;
    qb.batchC3 = 1;
    qb.batchC4 = 1;
    qb.biasThreeDim = 0;
    qb.baseM = 256;
    qb.baseN = 128;
    qb.baseK = mmTile.baseK;
    qb.isBias = 0;
    qb.dbL0C = 1;
    params.qbmmParams = qb;

    AlltoAllParams alltoall;
    alltoall.rankDim = rankDim;
    alltoall.rankKPerRank = rankKPerRank;
    alltoall.scaleKPerRank = scaleKPerRank;
    alltoall.aRankStride = aRankStride;
    alltoall.scaleARankStride = scaleARankStride;
    params.alltoallParams = alltoall;

    GemmKernel gemm;
    gemm(params);
}

} // namespace Kernel
} // namespace Dista

#endif
