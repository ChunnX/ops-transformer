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
 * \brief Tiling implementation — BN1->S1 two-level split, no combine.
 */

#include "fa_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include <torch/torch.h>

namespace ascend_ops {
namespace fa_host {

void FaTiling::SetPlatMemoryInfo(ContextParamsForTiling &contextKeyParams)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    aivNum = ascendcPlatform->GetCoreNumAiv();
    aicNum = ascendcPlatform->GetCoreNumAic();
    platformInfo_.aivNum = aivNum;
    platformInfo_.aicNum = aicNum;
    platformInfo_.cvRatio = aivNum / aicNum;
    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();
}

void FaTiling::CreateSplitInput(split_core::BaseInfo &baseInfo, split_core::SplitParam &splitParam,
                                ContextParamsForTiling &contextKeyParams)
{
    baseInfo.bSize = bSize_;
    baseInfo.n1Size = n1Size_;
    baseInfo.n2Size = n2Size_;
    baseInfo.gSize = gSize_;
    baseInfo.s1Size = s1Size_;
    baseInfo.s2Size = s2Size_;
    baseInfo.sparseMode = sparseModeVal_;
    baseInfo.attnMaskFlag = attnMaskFlag_;
    baseInfo.preToken = preToken_;
    baseInfo.nextToken = nextToken_;
    baseInfo.isAccumSeqS1 = false;
    baseInfo.isAccumSeqS2 = false;

    splitParam.mBaseSize = M_BASE_SIZE;
    splitParam.s2BaseSize = S2_BASE_SIZE;
}

void FaTiling::SetSplitOutput(const split_core::FAMetaData &splitRes)
{
    for (size_t i = 0; i < optiling::NPU_AIC_CORE_NUM; ++i) {
        if (i >= splitRes.usedCoreNum) {
            tilingData_.faMetaData.FAMetadata[i][optiling::FA_CORE_ENABLE_INDEX] = 0;
            continue;
        }
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_CORE_ENABLE_INDEX] = 1;
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_BN1_START_INDEX] =
            (i == 0) ? 0 : splitRes.bn1End[i - 1];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_S1_START_INDEX] =
            (i == 0) ? 0 : splitRes.s1End[i - 1];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_BN1_END_INDEX] = splitRes.bn1End[i];
        tilingData_.faMetaData.FAMetadata[i][optiling::FA_S1_END_INDEX] = splitRes.s1End[i];
    }
}

void FaTiling::SplitPolicy(ContextParamsForTiling &contextKeyParams)
{
    split_core::BaseInfo baseInfo{};
    split_core::SplitParam splitParam{};
    CreateSplitInput(baseInfo, splitParam, contextKeyParams);

    split_core::FAMetaData result{aicNum};
    split_core::SplitCore(aicNum, baseInfo, splitParam, result);
    SetSplitOutput(result);
    usedCoreNum_ = result.usedCoreNum;
}

void FaTiling::UpdateTilingKeyConfig()
{
    tilingKeyInfo_.config = 0;
}

void FaTiling::UpdateTilingKeyInfo(ContextParamsForTiling &contextKeyParams)
{
    tilingKeyInfo_.inputLayout = InOutLayoutType_BSH_BSH;
    UpdateTilingKeyConfig();
    tilingKeyInfo_.hasAttnMask = attnMaskFlag_;
    tilingKeyInfo_.kvLayoutType = KvLayoutType_NO_PA;
    tilingKeyInfo_.isReconstructTemp = true;
}

void FaTiling::GenTilingKey(ContextParamsForTiling &contextKeyParams)
{
    UpdateTilingKeyInfo(contextKeyParams);
    GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.hasAttnMask,
                       tilingKeyInfo_.kvLayoutType, false,
                       tilingKeyInfo_.isReconstructTemp);
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
    tilingData_.baseTiling.faBaseParams.n1Size = n1Size_;
    tilingData_.baseTiling.faBaseParams.n2Size = n2Size_;
    tilingData_.baseTiling.faBaseParams.gSize = gSize_;
    tilingData_.baseTiling.faBaseParams.s1Size = s1Size_;
    tilingData_.baseTiling.faBaseParams.s2Size = s2Size_;
    tilingData_.baseTiling.faBaseParams.dSize = qkHeadDim_;
    tilingData_.baseTiling.faBaseParams.dSizeV = vHeadDim_;
    tilingData_.baseTiling.faBaseParams.softmaxScale = softmaxScale_;
    tilingData_.baseTiling.faBaseParams.coreNum = usedCoreNum_;

    tilingData_.baseTiling.faAttnMaskParams.preTokens = preToken_;
    tilingData_.baseTiling.faAttnMaskParams.nextTokens = nextToken_;
    tilingData_.baseTiling.faAttnMaskParams.isExistRowInvalid = 0;
}

void FaTiling::DoTiling(ContextParamsForTiling &contextKeyParams)
{
    SetPlatMemoryInfo(contextKeyParams);

    const at::IntArrayRef queryShape = contextKeyParams.queryInputShape;
    bSize_ = queryShape[0];
    s1Size_ = queryShape[1];
    n1Size_ = contextKeyParams.headsNumber;
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
    GenTilingKey(contextKeyParams);
}

} // namespace fa_host
} // namespace ascend_ops
