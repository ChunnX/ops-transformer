/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file fa_tiling.cpp
 * \brief GQA non-quant tiling implementation for flash_attn.
 *        Ported from ops-transformer fa_tiling.cpp, adapted
 *        for AscendOps paradigm (using ContextParamsForTiling and at::Tensor
 *        instead of CANN gert::TilingContext and FaTilingInfo).
 *        References the v1 fa_tiling.cpp logic style.
 */

#include "fa_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include <algorithm>

namespace ascend_ops {
namespace fa_host {

void FaTiling::SetPlatMemoryInfo(ContextParamsForTiling &contextKeyParams)
{
    contextKeyParamsPtr = &contextKeyParams;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    aivNum = ascendcPlatform->GetCoreNumAiv();
    aicNum = ascendcPlatform->GetCoreNumAic();
    platformInfo_.aivNum = aivNum;
    platformInfo_.aicNum = aicNum;
    platformInfo_.cvRatio = aivNum / aicNum;
    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();
}

void FaTiling::AdjustSinnerAndSouter()
{
    sOuterFactor_ = SOUTER_64;
    sInnerFactor_ = SINNER_128;
}

void FaTiling::CreateSplitInput(split_core::BaseInfo &baseInfo, split_core::SplitParam &splitParam,
                                ContextParamsForTiling &contextKeyParams)
{
    baseInfo.bSize = bSize_;
    baseInfo.n2Size = n2Size_;
    baseInfo.gSize = gSize_;
    baseInfo.s1Size = s1Size_;
    baseInfo.s2Size = s2Size_;
    baseInfo.isS1G = true;
    baseInfo.sparseMode = sparseModeVal_;
    baseInfo.attnMaskFlag = attnMaskFlag_;
    baseInfo.preToken = preToken_;
    baseInfo.nextToken = nextToken_;
    baseInfo.isAccumSeqS1 = false;
    baseInfo.isAccumSeqS2 = false;

    splitParam.mBaseSize = sOuterFactor_ * CV_RATIO;
    splitParam.s2BaseSize = sInnerFactor_;
    splitParam.gS1BaseSizeOfCombine = 8;
    splitParam.streamK = true;
}

void FaTiling::SetSplitOutput(const split_core::FAMetaData &splitRes)
{

    for (size_t i = 0; i < optiling::NPU_AIC_CORE_NUM; ++i) {
        if (i >= splitRes.usedCoreNum) {
            tilingData_.faMetaData.FAMetadata[i][optiling::FA_CORE_ENABLE_INDEX] = 0;
            continue;
        }
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_CORE_ENABLE_INDEX] = 1;
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_BN2_START_INDEX] = (i == 0) ? 0 : splitRes.bN2End[i - 1];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_M_START_INDEX] = (i == 0) ? 0 : splitRes.mEnd[i - 1];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_S2_START_INDEX] = (i == 0) ? 0 : splitRes.s2End[i - 1];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_BN2_END_INDEX] = splitRes.bN2End[i];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_M_END_INDEX] = splitRes.mEnd[i];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_S2_END_INDEX] = splitRes.s2End[i];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] =
            splitRes.firstCombineDataWorkspaceIdx[i];
    }

    const split_core::CombineResult &combineRes = splitRes.combineRes;
    for (size_t i = 0; i < optiling::NPU_AIV_CORE_NUM; ++i) {
        if (i >= combineRes.combineUsedVecNum) {
            tilingData_.faMetaData.CombineMetadata[i][optiling::COMBINE_CORE_ENABLE_INDEX] = 0;
            continue;
        }
        tilingData_.faMetaData.CombineMetadata[i][optiling::COMBINE_CORE_ENABLE_INDEX] = 1;
        uint32_t curCombineIdx = combineRes.taskIdx[i];
        tilingData_.faMetaData.CombineMetadata[i][optiling::COMBINE_BN2_IDX_INDEX] = combineRes.combineBN2Idx[curCombineIdx];
        tilingData_.faMetaData.CombineMetadata[i][optiling::COMBINE_M_IDX_INDEX] = combineRes.combineMIdx[curCombineIdx];
        tilingData_.faMetaData.CombineMetadata[i][optiling::COMBINE_S2_SPLIT_NUM_INDEX] = combineRes.combineS2SplitNum[curCombineIdx];
        tilingData_.faMetaData.CombineMetadata[i][optiling::COMBINE_WORKSPACE_IDX_INDEX] = combineRes.combineWorkspaceIdx[curCombineIdx];
        tilingData_.faMetaData.CombineMetadata[i][optiling::COMBINE_M_START_INDEX] = combineRes.mStart[i];
        tilingData_.faMetaData.CombineMetadata[i][optiling::COMBINE_M_NUM_INDEX] = combineRes.mLen[i];
    }
}

void FaTiling::SplitPolicy(ContextParamsForTiling &contextKeyParams)
{
    AdjustSinnerAndSouter();

    split_core::BaseInfo baseInfo{};
    split_core::SplitParam splitParam{};
    CreateSplitInput(baseInfo, splitParam, contextKeyParams);

    split_core::FAMetaData result{aicNum, platformInfo_.cvRatio};
    split_core::SplitCore(aicNum, baseInfo, splitParam, result);
    SetSplitOutput(result);
    CalcNumBlocks(result.usedCoreNum, contextKeyParams);
    isCombine_ = (result.combineRes.combineNum > 0);
}

void FaTiling::UpdateTilingKeyConfig()
{
    auto sOuter = sOuterFactor_ * platformInfo_.cvRatio;
    auto sInner = sInnerFactor_;

    if (sOuter == SOUTER_64 && sInner == SINNER_256) {
        tilingKeyInfo_.config = 1;
    } else if (sOuter == SOUTER_128 && sInner == SINNER_128) {
        tilingKeyInfo_.config = 3;
    }
}

void FaTiling::UpdateTilingKeyInfo(ContextParamsForTiling &contextKeyParams)
{
    tilingKeyInfo_.inputLayout = InOutLayoutType_BSH_BSH;
    UpdateTilingKeyConfig();
    tilingKeyInfo_.isCombine = isCombine_;
    tilingKeyInfo_.hasAttnMask = attnMaskFlag_;
    tilingKeyInfo_.kvLayoutType = KvLayoutType_NO_PA;
    tilingKeyInfo_.isReconstructTemp = true;
}

void FaTiling::GenTilingKey(ContextParamsForTiling &contextKeyParams)
{
    UpdateTilingKeyInfo(contextKeyParams);
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.hasAttnMask,
                                    tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isCombine,
                                    tilingKeyInfo_.isReconstructTemp);
}

void FaTiling::CalcNumBlocks(uint32_t coreNum, ContextParamsForTiling &contextKeyParams)
{
    auto usedAivNum = coreNum * platformInfo_.cvRatio;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    numBlocks_ = ascendcPlatform->CalcTschBlockDim(usedAivNum, aicNum, aivNum);
}

void FaTiling::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
}

void FaTiling::CalcWorkspaceSize()
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    uint32_t mSize = sOuterFactor_ * platformInfo_.cvRatio;
    uint32_t dSize = vHeadDim_;

    workspaceSize_ = sysWorkspaceSize;

    if (isCombine_) {
        uint32_t faTmpAttnGmSize = numBlocks_ * 2 * mSize * dSize;
        uint32_t fatmpResLseGmSize = numBlocks_ * 2 * mSize * 8;
        workspaceSize_ += (faTmpAttnGmSize + 2 * fatmpResLseGmSize) * sizeof(float);
        tilingData_.baseTiling.faWorkspaceParams.accumOutSize = faTmpAttnGmSize;
        tilingData_.baseTiling.faWorkspaceParams.logSumExpSize = fatmpResLseGmSize;
    }
}

void FaTiling::FillTiling(ContextParamsForTiling &contextKeyParams)
{
    ComputeTilingData(contextKeyParams);
    SetFATilingData(contextKeyParams);
}

void FaTiling::ComputeTilingData(ContextParamsForTiling &contextKeyParams)
{
    if (attnMaskFlag_) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = contextKeyParams.attentionMaskShape.size();
        uint64_t maskS1Size = 2048;
        uint64_t maskS2Size = 2048;
        maskS2Size = contextKeyParams.attentionMaskShape[maskDimNum - 1];
        maskS1Size = contextKeyParams.attentionMaskShape[maskDimNum - 2];
        tilingData_.baseTiling.faAttnMaskParams.attnMaskBatch = maskBatch;
        tilingData_.baseTiling.faAttnMaskParams.attnMaskS1Size = maskS1Size;
        tilingData_.baseTiling.faAttnMaskParams.attnMaskS2Size = maskS2Size;
    }
    tilingData_.baseTiling.faAttnMaskParams.sparseMode = sparseModeVal_;
}

void FaTiling::SetFATilingData(ContextParamsForTiling &contextKeyParams)
{
    tilingData_.baseTiling.faBaseParams.bSize = bSize_;
    tilingData_.baseTiling.faBaseParams.n2Size = n2Size_;
    tilingData_.baseTiling.faBaseParams.gSize = gSize_;
    tilingData_.baseTiling.faBaseParams.s1Size = s1Size_;
    tilingData_.baseTiling.faBaseParams.s2Size = s2Size_;
    tilingData_.baseTiling.faBaseParams.dSize = qkHeadDim_;
    tilingData_.baseTiling.faBaseParams.dSizeV = vHeadDim_;
    tilingData_.baseTiling.faBaseParams.softmaxScale = softmaxScale_;
    tilingData_.baseTiling.faBaseParams.coreNum = numBlocks_;

    tilingData_.baseTiling.faAttnMaskParams.preTokens = preToken_;
    tilingData_.baseTiling.faAttnMaskParams.nextTokens = nextToken_;
    tilingData_.baseTiling.faAttnMaskParams.isExistRowInvalid = 0;
}

void FaTiling::DoTiling(ContextParamsForTiling &contextKeyParams)
{
    SetPlatMemoryInfo(contextKeyParams);

    inputLayout_ = InputLayout::BSND;
    const at::IntArrayRef queryShape = contextKeyParams.queryInputShape;
    bSize_ = queryShape[0];
    s1Size_ = queryShape[1];
    qkHeadDim_ = queryShape[3];
    n2Size_ = contextKeyParams.numKeyValueHeads > 0 ? contextKeyParams.numKeyValueHeads : contextKeyParams.headsNumber;
    gSize_ = contextKeyParams.headsNumber / n2Size_;

    vHeadDim_ = qkHeadDim_;
    softmaxScale_ = contextKeyParams.softmaxScale;
    sparseModeVal_ = 3;
    attnMaskFlag_ = !contextKeyParams.attentionMaskShape.empty();

    const at::IntArrayRef keyShape = contextKeyParams.keyInputShape;
    s2Size_ = keyShape[1];
    preToken_ = static_cast<int64_t>(s1Size_);
    nextToken_ = static_cast<int64_t>(s2Size_) - static_cast<int64_t>(s1Size_);

    SplitPolicy(contextKeyParams);
    FillTiling(contextKeyParams);
    CalcScheduleMode();
    CalcWorkspaceSize();
    GenTilingKey(contextKeyParams);

    contextKeyParams.workspaceSize = workspaceSize_;
}

} // namespace fa_host
} // namespace ascend_ops
