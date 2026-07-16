/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_to_all_mx_quant_matmul_arch35.h
 * \brief AlltoAll MX Quant Matmul — simplified pipeline (arch35).
 *
 * Pipeline design:
 *   Process()
 *     AIV: comm_.ProcessScale()  →  comm_.ProcessData<X1Type>()
 *     AIC: ProcessCompute()      →  setupComputeContext_(i) + executeGemm_(i)
 *
 * AIV side: pure communication via MxQuantAlltoAllComm (replaceable with AllGather etc).
 * AIC side: pure computation, directly invokes Blaze GemmKernel.
 */

#ifndef DISTA_ALL_TO_ALL_MX_QUANT_MATMUL_ARCH35_H
#define DISTA_ALL_TO_ALL_MX_QUANT_MATMUL_ARCH35_H

#include "../../block/ccu/ccu_alltoall_communicator.h"
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

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
class AlltoAllMxQuantMatmulArch35 {
private:
    using CommType = Dista::AivComm::Ccu::MxQuantAlltoAllComm<AlltoAllMatmulTilingDataType, IsMxFp4>;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMxAlltoAll<0, false>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC,
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

    using ComputeContextType = BlazeMXQuantMMContext<QuantBatchMatmulTilingData>;

public:
    __aicore__ inline AlltoAllMxQuantMatmulArch35(AlltoAllMatmulTilingDataType *tilingData,
                                                     AscendC::TPipe *tPipe, uint64_t x1HcclDataType)
        : comm_(tilingData),
          tilingData_(tilingData), tPipe_(tPipe),
          x1HcclDataType_(x1HcclDataType){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                 GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM);
    __aicore__ inline void Process();

private:
    CommType comm_;
    ComputeContextType computeContext_;
    AlltoAllMatmulTilingDataType *tilingData_;
    AscendC::TPipe *tPipe_;

    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR commX1ScaleGM1_;
    GM_ADDR commOutGM_;

    uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
    uint64_t x1HcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E4M3;

    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t NUM_TWO = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

    __aicore__ inline void ProcessCompute();
    __aicore__ inline void setupComputeContext_(uint32_t loopIdx);
    __aicore__ inline void executeGemm_(uint32_t taskIndex);
};

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                             AlltoAllMatmulTilingDataType, IsMxFp4>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;
    commX1ScaleGM1_ = workspaceGM;
    uint64_t x1ScaleLen = CeilDivU64(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO *
                          static_cast<uint64_t>(mc2Tiling_.rankM) * sizeof(AscendC::fp8_e8m0_t);
    x1ScaleLen = CeilDivU64(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
    commOutGM_ = workspaceGM + x1ScaleLen;

    comm_.Init(x1Scale_, commX1ScaleGM1_, x1_, commOutGM_,
               mxScaleHcclDataType_, x1HcclDataType_);
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                             AlltoAllMatmulTilingDataType, IsMxFp4>::Process()
{
    if (tilingData_ == nullptr) {
        return;
    }
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    if (mc2Tiling_.rankDim == 0) {
        return;
    }
    if (mc2Tiling_.tileCnt == 0 && mc2Tiling_.tailCnt == 0) {
        return;
    }

    if ASCEND_IS_AIV {
        comm_.ProcessScale();
        comm_.template ProcessData<X1Type>();
    }

    if ASCEND_IS_AIC {
        ProcessCompute();
    }

    comm_.End();
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                             AlltoAllMatmulTilingDataType, IsMxFp4>::ProcessCompute()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    uint32_t totalTileCnt = mc2Tiling_.tileCnt + mc2Tiling_.tailCnt;

    for (uint32_t i = 0; i < totalTileCnt; ++i) {
        AscendC::CrossCoreWaitFlag(9);
        setupComputeContext_(i);
        uint32_t localIdx = (i < mc2Tiling_.tileCnt) ? i : (i - mc2Tiling_.tileCnt);
        executeGemm_(localIdx);
    }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                             AlltoAllMatmulTilingDataType, IsMxFp4>::setupComputeContext_(uint32_t loopIdx)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    bool isHead = (loopIdx < mc2Tiling_.tileCnt);
    uint32_t curM = isHead ? mc2Tiling_.tileM : mc2Tiling_.tailM;
    uint64_t headRows = static_cast<uint64_t>(mc2Tiling_.tileCnt) * mc2Tiling_.tileM;
    uint64_t rankForComm = comm_.GetRankForComm();
    uint32_t scaleKPerRank = comm_.GetScaleKPerRank();

    auto *ctx = &computeContext_;
    if (isHead) {
        ctx->baseData.aGM = commOutGM_;
        ctx->baseData.cGM = y_;
        ctx->additionalData.x1Scale = commX1ScaleGM1_;
    } else {
        ctx->baseData.aGM = commOutGM_ + headRows * rankForComm * sizeof(X1Type);
        ctx->baseData.cGM = y_ + headRows * mc2Tiling_.rankN * sizeof(YType);
        ctx->additionalData.x1Scale =
            commX1ScaleGM1_ + headRows * scaleKPerRank * sizeof(AscendC::fp8_e8m0_t);
    }
    ctx->baseData.bGM = x2_;
    ctx->baseData.aOffset = static_cast<uint64_t>(curM) * rankForComm * sizeof(X1Type);
    ctx->baseData.bOffset = 0UL;
    ctx->baseData.cOffset = static_cast<uint64_t>(curM) * mc2Tiling_.rankN * sizeof(YType);
    ctx->additionalData.x1ScaleOffset = static_cast<uint64_t>(curM) * scaleKPerRank * sizeof(AscendC::fp8_e8m0_t);
    ctx->additionalData.x2Scale = x2Scale_;
    ctx->additionalData.rankDim = mc2Tiling_.rankDim;
    ctx->additionalData.rankN = mc2Tiling_.rankN;
    ctx->additionalData.rankM = mc2Tiling_.rankM;
    ctx->additionalData.scaleKPerRank = scaleKPerRank;
    ctx->tilingDataPtr = isHead ? &(tilingData_->blazeMmTileTilingData) : &(tilingData_->blazeMmTailTilingData);
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                             AlltoAllMatmulTilingDataType, IsMxFp4>::executeGemm_(uint32_t taskIndex)
{
    GM_ADDR aGM = computeContext_.baseData.aGM + taskIndex * computeContext_.baseData.aOffset;
    GM_ADDR bGM = computeContext_.baseData.bGM + taskIndex * computeContext_.baseData.bOffset;
    GM_ADDR cGM = computeContext_.baseData.cGM + taskIndex * computeContext_.baseData.cOffset;
    GM_ADDR x1Scale = computeContext_.additionalData.x1Scale + taskIndex * computeContext_.additionalData.x1ScaleOffset;
    GM_ADDR x2Scale = computeContext_.additionalData.x2Scale;

    const auto &mmTile = *computeContext_.tilingDataPtr;
    uint32_t rankDim = computeContext_.additionalData.rankDim;
    uint64_t rankKPerRank = static_cast<uint64_t>(mmTile.k) / rankDim;
    uint64_t scaleKPerRank = computeContext_.additionalData.scaleKPerRank;
    uint64_t aRankStride = static_cast<uint64_t>(computeContext_.additionalData.rankM) * rankKPerRank;
    uint64_t scaleARankStride = static_cast<uint64_t>(computeContext_.additionalData.rankM) * scaleKPerRank;

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
