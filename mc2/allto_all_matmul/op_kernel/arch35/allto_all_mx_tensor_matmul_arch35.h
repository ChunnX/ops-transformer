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
 * \file allto_all_mx_tensor_matmul_arch35.h
 * \brief Arch35 for MX-FP8 tensor_api branch (transpose removal optimization)
 *        Uses GM auto-accumulation for rank-level K-axis summation
 */

#ifndef ALLTO_ALL_MX_TENSOR_MATMUL_ARCH35_H
#define ALLTO_ALL_MX_TENSOR_MATMUL_ARCH35_H

namespace Mc2Kernel {

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
class AlltoAllMxTensorMatmulArch35 {
public:
    __aicore__ inline AlltoAllMxTensorMatmulArch35(SchedulerType *pipeLine) : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM,
                                AlltoAllMatmulTilingDataType *tilingData);
    __aicore__ inline void Process();

private:
    SchedulerType *pipeLine_;
    SchedulerContextType pipeLineContext_;
    AlltoAllMatmulTilingDataType *tilingData_;
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
private:
    static constexpr uint64_t ALIGN_NUM = 512UL;
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
    __aicore__ inline void ProcessScale();
};

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxTensorMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM, AlltoAllMatmulTilingDataType *tilingData)
{
    pipeLine_->Init();

    tilingData_ = tilingData;
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;

    commX1ScaleGM1_ = workspaceGM;
    uint64_t x1ScaleLen = mc2Tiling_.rankKScaleGrp *
                 static_cast<uint64_t>(mc2Tiling_.rankM) * sizeof(AscendC::fp8_e8m0_t);
    x1ScaleLen = CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
    commOutGM_ = workspaceGM + x1ScaleLen;

    rankForComm_ = mc2Tiling_.rankK;

    pipeLine_->GetContext(&pipeLineContext_);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxTensorMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::Process()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;

    ProcessScale();

    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxTensorMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessScale()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;

    uint64_t rankKdivByGroupSize = mc2Tiling_.rankKScaleGrp;
    pipeLineContext_.communicationContext->taskCnt = 1;
    pipeLineContext_.communicationContext->sendBuffer = x1Scale_;
    pipeLineContext_.communicationContext->recvBuffer = commX1ScaleGM1_;
    pipeLineContext_.communicationContext->sendCount =
        mc2Tiling_.rankM / mc2Tiling_.rankDim * rankKdivByGroupSize;
    pipeLineContext_.communicationContext->strideCount = pipeLineContext_.communicationContext->sendCount;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    pipeLine_->ProcessScale();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxTensorMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessTile(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    uint64_t tileMMultiRankK = static_cast<uint64_t>(mc2Tiling_.tileM * rankForComm_);

    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_;
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_;
    pipeLineContext_.communicationContext->sendOffset = tileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tileMMultiRankK;
    pipeLineContext_.communicationContext->strideCount =
        mc2Tiling_.rankM * rankForComm_ / mc2Tiling_.rankDim;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    pipeLineContext_.computationContext->baseData.aGM = commOutGM_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = y_;
    pipeLineContext_.computationContext->baseData.biasGM = bias_;
    pipeLineContext_.computationContext->baseData.aOffset = 0UL;
    pipeLineContext_.computationContext->baseData.bOffset = 0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        static_cast<uint64_t>(mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y));
    pipeLineContext_.computationContext->additionalData.x1Scale = commX1ScaleGM1_;
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset = 0UL;
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;

    pipeLineContext_.computationContext->rankDim = mc2Tiling_.rankDim;
    pipeLineContext_.computationContext->globalStride =
        static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim
        * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->tileStride =
        static_cast<uint64_t>(mc2Tiling_.tileM)
        * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->scaleGlobalStride =
        static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim
        * mc2Tiling_.rankKScaleGrp * sizeof(AscendC::fp8_e8m0_t);
    pipeLineContext_.computationContext->scaleTileStride =
        static_cast<uint64_t>(mc2Tiling_.tileM)
        * mc2Tiling_.rankKScaleGrp * sizeof(AscendC::fp8_e8m0_t);

    tilingData_->mxTensorTileTilingData.tileM = mc2Tiling_.tileM;
    tilingData_->mxTensorTileTilingData.fullK = mc2Tiling_.rankK * mc2Tiling_.rankDim;
    tilingData_->mxTensorTileTilingData.fullN = mc2Tiling_.rankN;
    tilingData_->mxTensorTileTilingData.rankK = mc2Tiling_.rankK;
    tilingData_->mxTensorTileTilingData.rankDim = mc2Tiling_.rankDim;
    tilingData_->mxTensorTileTilingData.rankKScaleGrp = mc2Tiling_.rankKScaleGrp;
    tilingData_->mxTensorTailTilingData.tileM = mc2Tiling_.tailM;
    tilingData_->mxTensorTailTilingData.fullK = mc2Tiling_.rankK * mc2Tiling_.rankDim;
    tilingData_->mxTensorTailTilingData.fullN = mc2Tiling_.rankN;
    tilingData_->mxTensorTailTilingData.rankK = mc2Tiling_.rankK;
    tilingData_->mxTensorTailTilingData.rankDim = mc2Tiling_.rankDim;
    tilingData_->mxTensorTailTilingData.rankKScaleGrp = mc2Tiling_.rankKScaleGrp;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mxTensorTileTilingData);

    pipeLine_->Process(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxTensorMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessTail(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    uint64_t tailMMultiRankK = mc2Tiling_.tailM * rankForComm_;
    uint64_t tileCntMultitileMMultiRankK =
        mc2Tiling_.tileCnt * mc2Tiling_.tileM * rankForComm_;

    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->sendOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tailMMultiRankK;

    pipeLineContext_.computationContext->baseData.aGM = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM =
        y_ + static_cast<uint64_t>(mc2Tiling_.tileCnt) * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->baseData.aOffset = 0UL;
    pipeLineContext_.computationContext->baseData.bOffset = 0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        static_cast<uint64_t>(mc2Tiling_.tailM) * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->additionalData.x1Scale = commX1ScaleGM1_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM 
    * mc2Tiling_.rankKScaleGrp * sizeof(AscendC::fp8_e8m0_t);
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset = 0UL;
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    
    pipeLineContext_.computationContext->rankDim = mc2Tiling_.rankDim;
    pipeLineContext_.computationContext->globalStride =
        static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim
        * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->tileStride =
        static_cast<uint64_t>(mc2Tiling_.tailM)
        * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->scaleGlobalStride =
        static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim
        * mc2Tiling_.rankKScaleGrp * sizeof(AscendC::fp8_e8m0_t);
    pipeLineContext_.computationContext->scaleTileStride =
        static_cast<uint64_t>(mc2Tiling_.tailM)
        * mc2Tiling_.rankKScaleGrp * sizeof(AscendC::fp8_e8m0_t);

    tilingData_->mxTensorTailTilingData.tileM = mc2Tiling_.tailM;
    tilingData_->mxTensorTailTilingData.fullK = mc2Tiling_.rankK * mc2Tiling_.rankDim;
    tilingData_->mxTensorTailTilingData.fullN = mc2Tiling_.rankN;
    tilingData_->mxTensorTailTilingData.rankK = mc2Tiling_.rankK;
    tilingData_->mxTensorTailTilingData.rankDim = mc2Tiling_.rankDim;
    tilingData_->mxTensorTailTilingData.rankKScaleGrp = mc2Tiling_.rankKScaleGrp;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mxTensorTailTilingData);

    pipeLine_->Process(taskCnt);
}

} // namespace Mc2Kernel

#endif