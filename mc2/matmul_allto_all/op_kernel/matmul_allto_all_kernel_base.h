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
 * \file matmul_allto_all_kernel_base.h
 * \brief
 */

#ifndef MATMUL_ALLTO_ALL_KERNEL_BASE_H
#define MATMUL_ALLTO_ALL_KERNEL_BASE_H

namespace Mc2Kernel {
/**
 * SchedulerType: 娴佹按绾跨被鐨勬暟鎹被鍨? * SchedulerContextType: 娴佹按绾跨被浣跨敤鐨勪笂涓嬫枃鐨勬暟鎹被鍨? * MatmulAlltoAllTilingDataType: tilingdata鐨勬暟鎹被鍨? */
template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
class MatmulAlltoAllKernelBase {
public:
    __aicore__ inline MatmulAlltoAllKernelBase(SchedulerType *pipeLine) : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR workspaceGM,
                                MatmulAlltoAllTilingDataType *tilingData, AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    SchedulerType *pipeLine_;
    SchedulerContextType pipeLineContext_;
    MatmulAlltoAllTilingDataType *tilingData_;
    // 鏁版嵁鎺у埗
    AscendC::TPipe *tPipe_;
    // 杈撳叆杈撳嚭鐭╅樀鍦板潃
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    // 涓存椂绌洪棿鐨勫湴鍧€
    GM_ADDR workspaceGM_;
    GM_ADDR tempComputeOutGM_;
    GM_ADDR transOutGM_;

private:
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
};

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void
MatmulAlltoAllKernelBase<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR workspaceGM, MatmulAlltoAllTilingDataType *tilingData,
    AscendC::TPipe *tPipe)
__aicore__ inline void
MatmulAlltoAllKernelBase<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>::Init(
    // 鑾峰彇tilingdata鏁版嵁
    tilingData_ = tilingData;
    auto &&mc2Tiling_ = tilingData_->matmulAlltoAllTilingInfo;
    // 鏁版嵁鍒濆鍖?    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    workspaceGM_ = workspaceGM;
    tempComputeOutGM_ = workspaceGM;
    transOutGM_ = (GM_ADDR)(tempComputeOutGM_ + mc2Tiling_.mmResultLen);
    // 鍒濆鍖栨祦姘寸嚎
    pipeLine_->Init();
    pipeLine_->GetContext(&pipeLineContext_);
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void
MatmulAlltoAllKernelBase<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>::Process()
{
    auto &&mc2Tiling_ = tilingData_->matmulAlltoAllTilingInfo;
    // 鍚姩涓诲潡娴佹按
    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 鍚姩灏惧潡娴佹按
    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    // 缁撴潫娴佹按绾?    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void
MatmulAlltoAllKernelBase<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>
    ::ProcessTile(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->matmulAlltoAllTilingInfo;
    // 澶嶇敤鐨勪腑闂撮噺
    uint64_t tileMMultiRankN = (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankN;
    uint64_t rankNDivRankDim = (uint64_t)mc2Tiling_.rankN / mc2Tiling_.rankDim;
    // matmul鐭╅樀涔樿绠楃殑杈撳叆杈撳嚭鍦板潃锛屽埌涓嬩竴杞绠楃殑鏁版嵁鍦板潃鐨勫亸绉?    pipeLineContext_.computationContext->baseData.aGM = x1_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = tempComputeOutGM_;
    pipeLineContext_.computationContext->baseData.biasGM = bias_;
    pipeLineContext_.computationContext->baseData.aOffset =
        (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bOffset = (uint64_t)0UL;
    pipeLineContext_.computationContext->baseData.cOffset = tileMMultiRankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2MmV3TileTilingData);

    // 杞疆鎿嶄綔鐨勮緭鍏ヨ緭鍑哄湴鍧€锛屽崟杞浆缃唴閮ㄦ暟鎹潡鐨勫亸绉伙紝鍒颁笅涓€杞浆缃暟鎹湴鍧€鐨勫亸绉?    pipeLineContext_.transposeContext->transposeSrcAddr = tempComputeOutGM_;
    pipeLineContext_.transposeContext->transposeDstAddr = transOutGM_;
    pipeLineContext_.transposeContext->transposeSrcOffset = tileMMultiRankN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->nextSrcBlockOffset = rankNDivRankDim;
    pipeLineContext_.transposeContext->nextDstBlockOffset = (uint64_t)mc2Tiling_.rankM * rankNDivRankDim;
    pipeLineContext_.transposeContext->transposeDstOffset =
        pipeLineContext_.transposeContext->transposeSrcOffset / mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->rankCnt = mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->innerAxis = rankNDivRankDim;
    pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tileM;
    pipeLineContext_.transposeContext->innerOffsetIn = (uint64_t)mc2Tiling_.rankN;
    pipeLineContext_.transposeContext->innerOffsetOut = rankNDivRankDim;

    // 閫氫俊鎿嶄綔鐨勮緭鍏ヨ緭鍑哄湴鍧€锛屽埌涓嬩竴杞殑鍦板潃鍋忕Щ
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.communicationContext->sendBuffer = pipeLineContext_.transposeContext->transposeDstAddr;
    pipeLineContext_.communicationContext->recvBuffer = y_;
    pipeLineContext_.communicationContext->sendOffset = (uint64_t)mc2Tiling_.tileM * rankNDivRankDim * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount =
        (uint64_t)mc2Tiling_.tileM * pipeLineContext_.transposeContext->innerAxis;
    pipeLineContext_.communicationContext->strideCount = pipeLineContext_.transposeContext->nextDstBlockOffset;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    pipeLine_->Process(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType>
__aicore__ inline void
MatmulAlltoAllKernelBase<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType>
    ::ProcessTail(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->matmulAlltoAllTilingInfo;
    // 澶嶇敤鐨勪腑闂撮噺
    uint64_t tileCntMultitileM = (uint64_t)mc2Tiling_.tileCnt * mc2Tiling_.tileM;
    uint64_t rankNDivRankDim = (uint64_t)mc2Tiling_.rankN / mc2Tiling_.rankDim;

    pipeLineContext_.computationContext->baseData.aGM = x1_ + tileCntMultitileM * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM =
        tempComputeOutGM_ + tileCntMultitileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->baseData.aOffset =
        (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankK * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bOffset = (uint64_t)0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2MmV3TailTilingData);

    uint64_t commonOffset = (uint64_t)mc2Tiling_.rankM * (uint64_t)mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transposeSrcAddr =
        tempComputeOutGM_ + tileCntMultitileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transposeDstAddr =
        transOutGM_ + tileCntMultitileM * rankNDivRankDim * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transposeSrcOffset =
        (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transposeDstOffset =
        (uint64_t)mc2Tiling_.tailM * rankNDivRankDim * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tailM;

    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.communicationContext->sendBuffer = pipeLineContext_.transposeContext->transposeDstAddr;
    pipeLineContext_.communicationContext->recvBuffer = y_ + tileCntMultitileM * rankNDivRankDim * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->sendOffset = (uint64_t)mc2Tiling_.tailM * rankNDivRankDim * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount =
        (uint64_t)mc2Tiling_.tailM * pipeLineContext_.transposeContext->innerAxis;

    pipeLine_->Process(taskCnt);
}
}; // namespace Mc2Kernel

#endif
