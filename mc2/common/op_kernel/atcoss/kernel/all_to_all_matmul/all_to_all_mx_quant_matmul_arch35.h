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
 * \brief AlltoAll MX Quant Matmul without transpose stage.
 *
 * A/ScaleA data read directly from CCU AlltoAll comm buffer by BlockMmad.
 * No AIV transpose stage — BlockMmad iterates per-rank in CopyAInL1/CopyScalesInL1.
 */

#ifndef DISTA_ALL_TO_ALL_MX_QUANT_MATMUL_ARCH35_H
#define DISTA_ALL_TO_ALL_MX_QUANT_MATMUL_ARCH35_H

#include "blaze_mx_quant_matmul.h"
#include "../../block/aiv_comm/all_to_all/ccu/hccl_communicator.h"

namespace Dista {
namespace Kernel {

__aicore__ inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
struct AlltoAllMxQuantTileContext {
    uint32_t taskCnt;
    GM_ADDR  sendBuffer;
    GM_ADDR  recvBuffer;
    uint64_t sendOffset;
    uint64_t recvOffset;
    uint64_t sendCount;
    uint64_t strideCount;
    uint64_t hcclDataType;

    GM_ADDR  aGM;
    GM_ADDR  bGM;
    GM_ADDR  cGM;
    uint64_t aOffset;
    uint64_t bOffset;
    uint64_t cOffset;
    GM_ADDR  x1Scale;
    GM_ADDR  x2Scale;
    uint64_t x1ScaleOffset;
    uint32_t computeRankDim;
    uint32_t computeRankN;
    uint32_t computeRankM;
    uint32_t computeScaleKPerRank;
};

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
class AlltoAllMxQuantMatmulArch35 {
private:
    static constexpr AscendC::HcclServerType ServerType = AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU;

    using CommunicationType = Dista::AivComm::Ccu::HcclCommunicator<
        false, ServerType, Dista::AivComm::Ccu::MC2AlltoAllContext,
        AlltoAllMatmulTilingDataType, Dista::AivComm::Ccu::MC2AlltoAllPrimitives, 0, 1>;
    using ComputationType = BlazeMXQuantMatmul<X1Type, X2Type, YType>;

public:
    __aicore__ inline AlltoAllMxQuantMatmulArch35(AlltoAllMatmulTilingDataType *tilingData,
                                                    AscendC::TPipe *tPipe, uint64_t x1HcclDataType)
        : commStage_(tilingData),
          computeStage_(), tilingData_(tilingData), tPipe_(tPipe),
          x1HcclDataType_(x1HcclDataType){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM);
    __aicore__ inline void Process();

private:
    CommunicationType commStage_;
    ComputationType computeStage_;
    AlltoAllMatmulTilingDataType *tilingData_;
    AscendC::TPipe *tPipe_;
    AlltoAllMxQuantTileContext tileCtx_;
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR commX1ScaleGM1_;
    GM_ADDR commOutGM_;
    uint64_t rankForComm_;
    uint32_t scaleKPerRank_;
    uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
    uint64_t x1HcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E4M3;

    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t NUM_TWO = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
    __aicore__ inline void ProcessScale();
    __aicore__ inline void DispatchTileLoop();
    __aicore__ inline void FillCommContext();
    __aicore__ inline void FillComputeContext(typename ComputationType::TilingDataType *tilingDataPtr);
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

    if constexpr (IsMxFp4) {
        constexpr uint32_t dataNumPerByte = 2;
        rankForComm_ = CeilDivU64(mc2Tiling_.rankK, dataNumPerByte) / mc2Tiling_.rankDim;
    } else {
        rankForComm_ = mc2Tiling_.rankK / mc2Tiling_.rankDim;
    }

    scaleKPerRank_ = static_cast<uint32_t>(
        CeilDivU64(mc2Tiling_.rankK / mc2Tiling_.rankDim, MXFP_GROUP_SIZE) * NUM_TWO);

    commStage_.Init();
    computeStage_.Init();
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                            AlltoAllMatmulTilingDataType, IsMxFp4>::FillCommContext()
{
    auto *ctx = commStage_.GetContextPtr();
    ctx->taskCnt = tileCtx_.taskCnt;
    ctx->sendBuffer = tileCtx_.sendBuffer;
    ctx->recvBuffer = tileCtx_.recvBuffer;
    ctx->sendOffset = tileCtx_.sendOffset;
    ctx->recvOffset = tileCtx_.recvOffset;
    ctx->sendCount = tileCtx_.sendCount;
    ctx->strideCount = tileCtx_.strideCount;
    ctx->hcclDataType = tileCtx_.hcclDataType;
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                            AlltoAllMatmulTilingDataType, IsMxFp4>::FillComputeContext(
    typename ComputationType::TilingDataType *tilingDataPtr)
{
    auto *ctx = computeStage_.GetContextPtr();
    ctx->baseData.aGM = tileCtx_.aGM;
    ctx->baseData.bGM = tileCtx_.bGM;
    ctx->baseData.cGM = tileCtx_.cGM;
    ctx->baseData.aOffset = tileCtx_.aOffset;
    ctx->baseData.bOffset = tileCtx_.bOffset;
    ctx->baseData.cOffset = tileCtx_.cOffset;
    ctx->additionalData.x1Scale = tileCtx_.x1Scale;
    ctx->additionalData.x1ScaleOffset = tileCtx_.x1ScaleOffset;
    ctx->additionalData.x2Scale = tileCtx_.x2Scale;
    ctx->additionalData.rankDim = tileCtx_.computeRankDim;
    ctx->additionalData.rankN = tileCtx_.computeRankN;
    ctx->additionalData.rankM = tileCtx_.computeRankM;
    ctx->additionalData.scaleKPerRank = tileCtx_.computeScaleKPerRank;
    ctx->tilingDataPtr = tilingDataPtr;
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                            AlltoAllMatmulTilingDataType, IsMxFp4>::DispatchTileLoop()
{
    commStage_.PrepareAll(tileCtx_.taskCnt);
    for (uint32_t index = 0; index < tileCtx_.taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_.Process(index);
            AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(9);
            computeStage_.Process(index);
        }
    }
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

    ProcessScale();

    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    commStage_.End();
    computeStage_.End();
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                            AlltoAllMatmulTilingDataType, IsMxFp4>::ProcessScale()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;

    uint64_t rankKdivByGroupSize = CeilDivU64(mc2Tiling_.rankK / mc2Tiling_.rankDim, MXFP_GROUP_SIZE) * NUM_TWO;

    auto *commCtx = commStage_.GetContextPtr();
    commCtx->taskCnt = 1;
    commCtx->sendBuffer = x1Scale_;
    commCtx->recvBuffer = commX1ScaleGM1_;
    commCtx->sendCount = static_cast<uint64_t>(mc2Tiling_.rankM) * rankKdivByGroupSize;
    commCtx->strideCount = commCtx->sendCount;
    commCtx->hcclDataType = mxScaleHcclDataType_;

    if ASCEND_IS_AIV {
        commStage_.PrepareAll(1);
        commStage_.Process(0);
    }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                            AlltoAllMatmulTilingDataType, IsMxFp4>::ProcessTile(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    uint64_t tileMMultiRankK = static_cast<uint64_t>(mc2Tiling_.tileM) * rankForComm_;

    tileCtx_.taskCnt = mc2Tiling_.tileCnt;
    tileCtx_.sendBuffer = x1_;
    tileCtx_.recvBuffer = commOutGM_;
    tileCtx_.sendOffset = tileMMultiRankK * sizeof(DTYPE_X1);
    tileCtx_.recvOffset = tileCtx_.sendOffset;
    tileCtx_.sendCount = tileMMultiRankK;
    tileCtx_.strideCount = static_cast<uint64_t>(mc2Tiling_.rankM) * rankForComm_;
    tileCtx_.hcclDataType = x1HcclDataType_;

    tileCtx_.aGM = commOutGM_;
    tileCtx_.bGM = x2_;
    tileCtx_.cGM = y_;
    tileCtx_.aOffset = tileMMultiRankK * sizeof(DTYPE_X1);
    tileCtx_.bOffset = 0UL;
    tileCtx_.cOffset = static_cast<uint64_t>(mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y));
    tileCtx_.x1Scale = commX1ScaleGM1_;
    tileCtx_.x1ScaleOffset = static_cast<uint64_t>(mc2Tiling_.tileM) *
        static_cast<uint64_t>(scaleKPerRank_) * sizeof(AscendC::fp8_e8m0_t);
    tileCtx_.x2Scale = x2Scale_;
    tileCtx_.computeRankDim = mc2Tiling_.rankDim;
    tileCtx_.computeRankN = mc2Tiling_.rankN;
    tileCtx_.computeRankM = mc2Tiling_.rankM;
    tileCtx_.computeScaleKPerRank = scaleKPerRank_;

    FillCommContext();
    FillComputeContext(&(tilingData_->blazeMmTileTilingData));
    DispatchTileLoop();
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<X1Type, X2Type, YType, CommDataTypeX1,
                            AlltoAllMatmulTilingDataType, IsMxFp4>::ProcessTail(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    uint64_t tailMMultiRankK = static_cast<uint64_t>(mc2Tiling_.tailM) * rankForComm_;
    uint64_t tileCntMultitileMMultiRankK =
        static_cast<uint64_t>(mc2Tiling_.tileCnt) * static_cast<uint64_t>(mc2Tiling_.tileM) * rankForComm_;

    tileCtx_.taskCnt = mc2Tiling_.tailCnt;
    tileCtx_.sendBuffer = x1_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    tileCtx_.recvBuffer = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    tileCtx_.sendOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    tileCtx_.recvOffset = tileCtx_.sendOffset;
    tileCtx_.sendCount = tailMMultiRankK;
    tileCtx_.strideCount = static_cast<uint64_t>(mc2Tiling_.rankM) * rankForComm_;
    tileCtx_.hcclDataType = x1HcclDataType_;

    tileCtx_.aGM = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    tileCtx_.bGM = x2_;
    tileCtx_.cGM = y_ + static_cast<uint64_t>(mc2Tiling_.tileCnt) * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    tileCtx_.aOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    tileCtx_.bOffset = 0UL;
    tileCtx_.cOffset = static_cast<uint64_t>(mc2Tiling_.tailM) * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    tileCtx_.x1Scale = commX1ScaleGM1_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * scaleKPerRank_ * sizeof(AscendC::fp8_e8m0_t);
    tileCtx_.x1ScaleOffset = static_cast<uint64_t>(mc2Tiling_.tailM) * scaleKPerRank_ * sizeof(AscendC::fp8_e8m0_t);
    tileCtx_.x2Scale = x2Scale_;
    tileCtx_.computeRankDim = mc2Tiling_.rankDim;
    tileCtx_.computeRankN = mc2Tiling_.rankN;
    tileCtx_.computeRankM = mc2Tiling_.rankM;
    tileCtx_.computeScaleKPerRank = scaleKPerRank_;

    FillCommContext();
    FillComputeContext(&(tilingData_->blazeMmTailTilingData));
    DispatchTileLoop();
}

} // namespace Kernel
} // namespace Dista

#endif
