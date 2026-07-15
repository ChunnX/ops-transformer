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
 *     AIC: ProcessCompute()      →  setupComputeContext_(i) + BlazeMXQuantMatmul::Process(i)
 *
 * AIV side: pure communication via MxQuantAlltoAllComm (replaceable with AllGather etc).
 * AIC side: pure computation, reuses BlazeMXQuantMatmul without modification.
 */

#ifndef DISTA_ALL_TO_ALL_MX_QUANT_MATMUL_ARCH35_H
#define DISTA_ALL_TO_ALL_MX_QUANT_MATMUL_ARCH35_H

#include "blaze_mx_quant_matmul.h"
#include "../../block/ccu/ccu_alltoall_communicator.h"

namespace Dista {
namespace Kernel {

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
class AlltoAllMxQuantMatmulArch35 {
private:
    using CommType = Dista::AivComm::Ccu::MxQuantAlltoAllComm<AlltoAllMatmulTilingDataType, IsMxFp4>;
    using ComputationType = BlazeMXQuantMatmul<X1Type, X2Type, YType>;

public:
    __aicore__ inline AlltoAllMxQuantMatmulArch35(AlltoAllMatmulTilingDataType *tilingData,
                                                     AscendC::TPipe *tPipe, uint64_t x1HcclDataType)
        : comm_(tilingData),
          computeStage_(), tilingData_(tilingData), tPipe_(tPipe),
          x1HcclDataType_(x1HcclDataType){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                 GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM);
    __aicore__ inline void Process();

private:
    CommType comm_;
    ComputationType computeStage_;
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
    computeStage_.Init();
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
        comm_.ProcessData<X1Type>();
    }

    if ASCEND_IS_AIC {
        ProcessCompute();
    }

    comm_.End();
    computeStage_.End();
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
        computeStage_.Process(localIdx);
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

    auto *ctx = computeStage_.GetContextPtr();
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

} // namespace Kernel
} // namespace Dista

#endif
