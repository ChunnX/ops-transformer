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
 * \file flash_attn_tiling_impl.cpp
 * \brief
 */

#include "flash_attn_tiling_impl.h"
#include "../flash_attn_tiling.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../fa_split_core_v2.h"
#include "../flash_attn_tiling_constants.h"
#include "../flash_attn_tiling_utils.h"
#include "../../op_kernel/arch35/flash_attn_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr uint64_t PRE_LOAD_NUM_GQA_ARCH35 = 3;

void FlashAttnTilingImpl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(faInfo_->opName, "FlashAttn schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus FlashAttnTilingImpl::DoOpTiling(const FaTilingInfo *faInfo)
{
    this->faInfo_ = const_cast<optiling::FaTilingInfo*>(faInfo);
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(faInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    InitImplParam();
    SplitPolicy();
    FillTiling();
    CalcScheduleMode();
    CalcWorkspaceSize();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(faInfo_->opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.cvRatio = platformInfo_.aivNum / platformInfo_.aicNum;
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(faInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void FlashAttnTilingImpl::InitImplParam()
{
    const gert::Tensor *actSeqLenQ = faInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor *actSeqLenKV = faInfo_->opParamInfo.cuSeqlensKv.tensor;
    uint32_t actSeqLenQDims = 0;
    uint32_t actSeqLenKVDims = 0;
        actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
        actSeqLenKVDims = (actSeqLenKV != nullptr) ? actSeqLenKV->GetShapeSize() : 0;
        cuSeqLenQFlag_ =
            !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int64_t>() == nullptr));
        cuSeqLenKVFlag_ =
            !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int64_t>() == nullptr));
}

void FlashAttnTilingImpl::AdjustSinnerAndSouter()
{
    uint32_t softmaxSOuterFactor = optiling::arch35FA::SOUTER_64;
    sOuterFactor_ = optiling::arch35FA::SOUTER_64;
    sInnerFactor_ = optiling::arch35FA::SINNER_128;

    // if (faInfo_->vHeadDim <= optiling::arch35FA::DSIZE_128 && faInfo_->mlaMode != MlaMode::ROPE_COMBINE_D128) {
    if (faInfo_->vHeadDim <= optiling::arch35FA::DSIZE_128) {
        // bool checkDtype = faInfo_->quantMode == FaQuantMode::NO_QUANT;
        bool checkQueryAndValueS = faInfo_->s1Size <= optiling::arch35FA::SOUTER_64 && faInfo_->s2Size > optiling::arch35FA::SINNER_128;
        uint32_t maskMode = faInfo_->maskMode;
        int64_t  winLefts = static_cast<int64_t>(faInfo_->winLeft);
        int64_t  winRights = static_cast<int64_t>(faInfo_->winRight);
        if (maskMode == 0) {
            winLefts = (winLefts > 0) ? 0 : winLefts;
        } else if (maskMode == 4) {
            winRights = (winRights > 0) ? 0 : winRights;
        }
        bool checkmaskMode = (maskMode != 2 && winLefts + winRights > 128);
        // if (checkDtype && checkQueryAndValueS && checkmaskMode && faInfo_->mlaMode != MlaMode::ROPE_SPLIT_D128) {
        if (checkQueryAndValueS && checkmaskMode) {
            sOuterFactor_ = optiling::arch35FA::SOUTER_32;
            sInnerFactor_ = optiling::arch35FA::SINNER_256;
            softmaxSOuterFactor = optiling::arch35FA::SOUTER_32;
        } else if ((faInfo_->qLayout == FaLayout::BSND) || (faInfo_->qLayout == FaLayout::TND)) {
            sOuterFactor_ = optiling::arch35FA::SOUTER_32;
            sInnerFactor_ = optiling::arch35FA::SINNER_256;
        }
    // } else if (faInfo_->vHeadDim > optiling::arch35FA::DSIZE_128 && faInfo_->mlaMode != MlaMode::ROPE_SPLIT_D512 && faInfo_->s1Size != 1) {
    } else if (faInfo_->vHeadDim > optiling::arch35FA::DSIZE_128 && faInfo_->s1Size != 1) {
        if (((faInfo_->qLayout == FaLayout::BSND) || (faInfo_->qLayout == FaLayout::TND)) &&
            faInfo_->vHeadDim <= optiling::arch35FA::DSIZE_256) { // 256 : D size
            sOuterFactor_ = optiling::arch35FA::SOUTER_32;
            sInnerFactor_ = optiling::arch35FA::SINNER_256;
        } else {
            sOuterFactor_ = optiling::arch35FA::SOUTER_64;
            sInnerFactor_ = optiling::arch35FA::SINNER_128;
        }
        softmaxSOuterFactor = optiling::arch35FA::SOUTER_32;
    } else if (faInfo_->s1Size == 1 && faInfo_->vHeadDim > optiling::arch35FA::DSIZE_128) { // IFA VD > 128
        sOuterFactor_ = optiling::arch35FA::SOUTER_64;
        sInnerFactor_ = optiling::arch35FA::SINNER_128;
        softmaxSOuterFactor = optiling::arch35FA::SOUTER_64;
    }

    OP_LOGI(faInfo_->opName, "Souter:%u SInner:%u softmaxSOuterFactor %u", sOuterFactor_, sInnerFactor_,
            softmaxSOuterFactor);
}

void FlashAttnTilingImpl::GetWinLeftsRightUp(int64_t cuSeqLength, int64_t cuSeqLengthKV,
                                                 int64_t &winLeftsLeftUp, int64_t &winRightsLeftUp)
{

}

void FlashAttnTilingImpl::FixParamWithRowInvalid(int64_t &cuSeqLength, int64_t cuSeqLengthKV,
                                                 int64_t &winLeftsLeftUp, int64_t &winRightsLeftUp)
{
    // 若出现行无效，需要重新计算winRights，winLefts，cuseqlen，以便正确计算分核核数
    int64_t winRightsError = (winRightsLeftUp < 0) ? -winRightsLeftUp : 0;
    winRightsError = winRightsError > cuSeqLength ? cuSeqLength : winRightsError;
    int64_t winLeftsError = 0;
    winLeftsError = (cuSeqLength > cuSeqLengthKV + winLeftsLeftUp) ?
                    (cuSeqLength - cuSeqLengthKV - winLeftsLeftUp) : 0;
    winLeftsError = winLeftsError > cuSeqLength ? cuSeqLength : winLeftsError;

    // 若出现上方行无效，需要重新计算winRights，winLefts，cuseqlen
    winRightsLeftUp += winRightsError;
    winLeftsLeftUp -= winRightsError;
    cuSeqLength -= winRightsError;

    // 若出现下方行无效，需要重新计算cuseqlen
    cuSeqLength -= winLeftsError;
}

bool FlashAttnTilingImpl::CheckS1OutSplit()
{
    return false;

    // if (faInfo_->maskMode == MASK_MODE_BAND ||
    //     (faInfo_->maskMode == MASK_MODE_NO_MASK && faInfo_->attnMaskFlag)) {
    //     return false;
    // }

    // 仅支持非量化，占用2B
    const int64_t dataTypeSize = 2U;
    int64_t bnSize = std::min(faInfo_->bSize * faInfo_->n2Size, static_cast<int64_t>(platformInfo_.aicNum));

    // 当所需的L2cache资源的超过系统配置一半时，开启S1外切分核优化L2cache复用率，乘2是经验值，后续进行优化
    return bnSize * faInfo_->s2Size * (faInfo_->qkHeadDim + faInfo_->vHeadDim) * dataTypeSize * 2 >=
           platformInfo_.l2Size;
}

void FlashAttnTilingImpl::SplitOutSeq()
{
    uint32_t curCoreNum = platformInfo_.aicNum;
    uint32_t sOuterSize = sOuterFactor_ * optiling::arch35FA::CV_RATIO;
    int64_t totalSize = 0;
    for (uint32_t bIdx = 0; bIdx < faInfo_->bSize; bIdx++) {
        int64_t cuSeqLengthsTmp = cuSeqLengthsQ_[bIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t winLeftsLeftUp = 0;
        int64_t winRightsLeftUp = 0;
        GetWinLeftsRightUp(cuSeqLengthsQ_[bIdx], cuSeqLengthsKV_[bIdx],
                               winLeftsLeftUp, winRightsLeftUp);
        FixParamWithRowInvalid(cuSeqLengthsTmp, cuSeqLengthsKV_[bIdx],
                               winLeftsLeftUp, winRightsLeftUp);

        int64_t outerBlockNums =
            (cuSeqLengthsTmp + static_cast<int64_t>(sOuterSize) - 1) / static_cast<int64_t>(sOuterSize);
        totalSize += outerBlockNums * faInfo_->n1Size;
    }

    int64_t cuUsedCoreNum = std::min(totalSize, static_cast<int64_t>(curCoreNum));
    tilingData_.baseTiling.flashAttnS1OuterSplitCoreParams.totalSize = totalSize;
}

void FlashAttnTilingImpl::CreateSplitInput(fa_split_core_v2::BaseInfo &baseInfo, fa_split_core_v2::SplitParam &splitParam)
{
    baseInfo.bSize = faInfo_->bSize;
    baseInfo.n2Size = faInfo_->n2Size;
    baseInfo.gSize = faInfo_->gSize;
    baseInfo.s1Size = faInfo_->s1Size;
    baseInfo.s2Size = faInfo_->s2Size;
    baseInfo.cuLenQDims = faInfo_->cuSeqLenQDims;
    baseInfo.cuLenKvDims = faInfo_->cuSeqLenKvDims;
    baseInfo.winLeft = faInfo_->winLeft;
    baseInfo.winRight = faInfo_->winRight;
    baseInfo.isS1G = (faInfo_->qLayout == FaLayout::BSND) || (faInfo_->qLayout == FaLayout::TND);
    baseInfo.maskMode = faInfo_->maskMode;

    if (faInfo_->qLayout == FaLayout::TND) {
        baseInfo.isAccumSeqS1 = true;
        baseInfo.isAccumSeqS2 = !faInfo_->pageAttentionFlag;
    } else {
        baseInfo.isAccumSeqS1 = false;
        baseInfo.isAccumSeqS2 = false;
    }
    const gert::Tensor *actSeqLenData = faInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor *actSeqLenDataKV = faInfo_->opParamInfo.cuSeqlensKv.tensor;
    if (actSeqLenData != nullptr) {
        baseInfo.cuSeqS1Size.reserve(baseInfo.bSize);
        const int64_t *s1Ptr = actSeqLenData->GetData<int64_t>();
        for (uint32_t i = 0; i < baseInfo.bSize; i++) {
            baseInfo.cuSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (actSeqLenDataKV != nullptr) {
        baseInfo.cuSeqS2Size.reserve(baseInfo.bSize);
        const int64_t *s2Ptr = actSeqLenDataKV->GetData<int64_t>();
        for (uint32_t i = 0; i < baseInfo.bSize; i++) {
            baseInfo.cuSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }
    splitParam.mBaseSize = sOuterFactor_ * optiling::arch35FA::CV_RATIO;
    splitParam.s2BaseSize = sInnerFactor_;
    splitParam.gS1BaseSizeOfFd = 8;
    splitParam.streamK = false;
}

void FlashAttnTilingImpl::SetSplitOutput(const fa_split_core_v2::FAMetaData &splitRes)
{

}

void FlashAttnTilingImpl::SplitPolicy()
{
    AdjustSinnerAndSouter(); // 确定tiling切块

    enableS1OutSplit = CheckS1OutSplit();
    if (false) {
        SplitOutSeq();
    } else {
        CalcNumBlocks(platformInfo_.aicNum);
        flashDecodeFlag_ = true; 
    }
}

void FlashAttnTilingImpl::UpdateTilingKeyConfig()
{
    auto sOuter = sOuterFactor_ * platformInfo_.cvRatio;
    auto sInner = sInnerFactor_;
    auto dSize = faInfo_->qkHeadDim;
    auto dVsize = faInfo_->vHeadDim;

    if (dSize <= optiling::arch35FA::DSIZE_64)
        dSize = optiling::arch35FA::DSIZE_64;
    else if (dSize <= optiling::arch35FA::DSIZE_128)
        dSize = optiling::arch35FA::DSIZE_128;
    else if (dSize <= optiling::arch35FA::DSIZE_256)
        dSize = optiling::arch35FA::DSIZE_256;
    else if (dSize <= optiling::arch35FA::DSIZE_512)
        dSize = optiling::arch35FA::DSIZE_512;
    else if (dSize <= optiling::arch35FA::DSIZE_576)
        dSize = optiling::arch35FA::DSIZE_576;

    if (dVsize <= optiling::arch35FA::DSIZE_64)
        dVsize = optiling::arch35FA::DSIZE_64;
    else if (dVsize <= optiling::arch35FA::DSIZE_128)
        dVsize = optiling::arch35FA::DSIZE_128;
    else if (dVsize <= optiling::arch35FA::DSIZE_256)
        dVsize = optiling::arch35FA::DSIZE_256;
    else if (dVsize <= optiling::arch35FA::DSIZE_512)
        dVsize = optiling::arch35FA::DSIZE_512;

    if (sOuter == optiling::arch35FA::SOUTER_64 && sInner == optiling::arch35FA::SINNER_64 && dSize == optiling::arch35FA::DSIZE_256 && dVsize == optiling::arch35FA::DSIZE_256) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256;
    } else if (sOuter == optiling::arch35FA::SOUTER_64 && sInner == optiling::arch35FA::SINNER_64 && dSize == optiling::arch35FA::DSIZE_512 && dVsize == optiling::arch35FA::DSIZE_512) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512;
    } else if (sOuter == optiling::arch35FA::SOUTER_64 && sInner == optiling::arch35FA::SINNER_256 && dSize == optiling::arch35FA::DSIZE_64 && dVsize == optiling::arch35FA::DSIZE_64) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64;
    } else if (sOuter == optiling::arch35FA::SOUTER_64 && sInner == optiling::arch35FA::SINNER_256 && dSize == optiling::arch35FA::DSIZE_128 && dVsize == optiling::arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_64 && dVsize == optiling::arch35FA::DSIZE_64) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_128 && dVsize == optiling::arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_192 && dVsize == optiling::arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_256 && dVsize == optiling::arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_256 && dVsize == optiling::arch35FA::DSIZE_256) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_512 && dVsize == optiling::arch35FA::DSIZE_512) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_256 && dSize == optiling::arch35FA::DSIZE_64 && dVsize == optiling::arch35FA::DSIZE_64) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64;
    } else if (sOuter == optiling::arch35FA::SOUTER_64 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_576 && dVsize == optiling::arch35FA::DSIZE_512) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512;
    } else if (sOuter == optiling::arch35FA::SOUTER_64 && sInner == optiling::arch35FA::SINNER_256 && dSize == optiling::arch35FA::DSIZE_256 && dVsize == optiling::arch35FA::DSIZE_256) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_256 && dSize == optiling::arch35FA::DSIZE_128 && dVsize == optiling::arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128;
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_128 && dVsize == optiling::arch35FA::DSIZE_64) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned64; // qkvd不等长
    } else if (sOuter == optiling::arch35FA::SOUTER_128 && sInner == optiling::arch35FA::SINNER_128 && dSize == optiling::arch35FA::DSIZE_64 && dVsize == optiling::arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned128; // qkvd不等长
    } else if (sOuter == optiling::arch35FA::SOUTER_64 && sInner == optiling::arch35FA::SINNER_256 && dSize == optiling::arch35FA::DSIZE_128 && dVsize == optiling::arch35FA::DSIZE_64) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned64; // qkvd不等长
    } else if (sOuter == optiling::arch35FA::SOUTER_64 && sInner == optiling::arch35FA::SINNER_256 && dSize == optiling::arch35FA::DSIZE_64 && dVsize == optiling::arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned128; // qkvd不等长
    } else {
    }
}

void FlashAttnTilingImpl::UpdateTilingKeyLayout()
{
    if (faInfo_->qLayout == FaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD_BNSD;
    } else if (faInfo_->qLayout == FaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND_TND;
    } else if (faInfo_->qLayout == FaLayout::BSND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSH_BSH;
    }
}

void FlashAttnTilingImpl::UpdateTilingKeyMaskMode()
{
    tilingKeyInfo_.maskMode = 0;
}

void FlashAttnTilingImpl::UpdateTilingKeyMatmulMode()
{
    tilingKeyInfo_.matmulMode = 0;
}

void FlashAttnTilingImpl::UpdateTilingKeyInfo()
{
    // if (faInfo_->emptyTensorFlag) {
    if (false) {
        // tilingKeyInfo_.emptyTensor = faInfo_->emptyTensorFlag;
    } else {
        UpdateTilingKeyLayout();
        UpdateTilingKeyConfig();
        tilingKeyInfo_.pseMode = PSE_MODE_PSE_NONE_TYPE;
        tilingKeyInfo_.isFd = flashDecodeFlag_;
        UpdateTilingKeyMaskMode();
        UpdateTilingKeyMatmulMode();
        tilingKeyInfo_.enableKvPrefix = 0;
        tilingKeyInfo_.enableS1OutSplit = enableS1OutSplit;
    }
}

void FlashAttnTilingImpl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode,
                                    tilingKeyInfo_.quantMode, tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope,
                                    tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd, tilingKeyInfo_.emptyTensor,
                                    tilingKeyInfo_.maskMode, tilingKeyInfo_.matmulMode, tilingKeyInfo_.enableKvPrefix,
                                    tilingKeyInfo_.enableS1OutSplit);

    OP_LOGI(faInfo_->opName, "The tilingkey is %llu.", tilingKey_);
    OP_LOGI(faInfo_->opName,
            "The tilingkey param is inOutLayoutType: %llu, config: %llu, pseMode: %llu, quantMode: %llu, "
            "hasAttenMask: %llu, hasRope: %llu, kvLayoutType: %llu, isFd: %llu, emptyTensor: %llu, PFAMask: %llu, "
            "pFAMatMulType: %llu, enableKvPrefix: %llu, enableS1OutSplit: %llu.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode, tilingKeyInfo_.quantMode,
            tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd,
            tilingKeyInfo_.emptyTensor, tilingKeyInfo_.maskMode, tilingKeyInfo_.matmulMode,
            tilingKeyInfo_.enableKvPrefix, tilingKeyInfo_.enableS1OutSplit);
}


void FlashAttnTilingImpl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(faInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(faInfo_->opName, "FlashAttn block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void FlashAttnTilingImpl::CalcWorkspaceSize()
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    uint32_t mSize = sOuterFactor_ * platformInfo_.cvRatio;
    uint32_t dSize = faInfo_->vHeadDim;
    uint32_t dVBasicBlock = 0;
    if (dSize <= optiling::arch35FA::DSIZE_64) {
        dVBasicBlock = optiling::arch35FA::DSIZE_64;
    } else if (dSize <= optiling::arch35FA::DSIZE_128) {
        dVBasicBlock = optiling::arch35FA::DSIZE_128;
    } else if (dSize <= optiling::arch35FA::DSIZE_256) {
        dVBasicBlock = optiling::arch35FA::DSIZE_256;
    } else if (dSize <= optiling::arch35FA::DSIZE_512) {
        dVBasicBlock = optiling::arch35FA::DSIZE_512;
    }

    workspaceSize_ = sysWorkspaceSize;

    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if (dVBasicBlock > optiling::arch35FA::DSIZE_256) {
        bmm2ResBlockSize = optiling::arch35FA::DSIZE_512;
    }
    if ((!dnFlag_ && dSize > optiling::arch35FA::DSIZE_128) || (dnFlag_ && dSize > optiling::arch35FA::DSIZE_192)) {
        bmm2Bytes = mSize * bmm2ResBlockSize * sizeof(float);
        if (dVBasicBlock > optiling::arch35FA::DSIZE_256) {
            vec2Bytes = mSize * dVBasicBlock * sizeof(float);
        }
    }
    workspaceSize_ += (bmm2Bytes + vec2Bytes) * 3 * platformInfo_.coreNum; // 3: perload 2次 需要2+1

    if (faInfo_->pageAttentionFlag) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        workspaceSize_ += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;
    }

    if (flashDecodeFlag_) {
        uint32_t faTmpAttenGmSize = platformInfo_.coreNum * 2 * mSize * dSize; // 每个核最多有2次写到workspace
        uint32_t fatmpResLseGmSize = platformInfo_.coreNum * 2 * mSize * 8;
        workspaceSize_ += (faTmpAttenGmSize + 2 * fatmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
        tilingData_.baseTiling.flashAttnWorkspaceParams.accumOutSize = faTmpAttenGmSize;
        tilingData_.baseTiling.flashAttnWorkspaceParams.logSumExpSize = fatmpResLseGmSize;
    }

    OP_LOGI(faInfo_->opName, "Workspaces: %ld", workspaceSize_);
}

void FlashAttnTilingImpl::FillTiling()
{
    ComputeTilingData();
    SetFATilingData();
    PrintAllTilingData();
}

void FlashAttnTilingImpl::ComputeTilingData()
{
    tilingData_.baseTiling.flashAttnAttenMaskParams.sparseMode = faInfo_->maskMode;
    tilingKeyInfo_.hasAttenMask = faInfo_->maskMode == 3 ? 1 : 0;

    if (tilingKeyInfo_.hasAttenMask) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = 2048;
        uint64_t maskS2Size = 2048;
        if (maskDimNum != 2 || faInfo_->s1Size == 1) {
            maskBatch = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(0);
        }
        maskS2Size = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 1);
        maskS1Size = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 2);
        tilingData_.baseTiling.flashAttnAttenMaskParams.attenMaskS1Size = maskS1Size;
        tilingData_.baseTiling.flashAttnAttenMaskParams.attenMaskS2Size = maskS2Size;
    }

    if (faInfo_->pageAttentionFlag) {
        uint32_t keyCacheDimNum = faInfo_->opParamInfo.key.shape->GetStorageShape().GetDimNum();
        if (keyCacheDimNum == 3) { // 3: BBH
            tilingData_.baseTiling.flashAttnPageAttentionParams.paLayoutType = 1;
        } else if (keyCacheDimNum == 4) { // 4: BNBD
            tilingData_.baseTiling.flashAttnPageAttentionParams.paLayoutType = 0;
        } else if (keyCacheDimNum == 5) { // 5: PA NZ
            tilingData_.baseTiling.flashAttnPageAttentionParams.paLayoutType = 2;
        }
    }
}

void FlashAttnTilingImpl::SetFATilingData()
{
    tilingData_.baseTiling.flashAttnBaseParams.bSize = faInfo_->bSize;
    tilingData_.baseTiling.flashAttnBaseParams.t1Size = faInfo_->qTSize;
    tilingData_.baseTiling.flashAttnBaseParams.t2Size = faInfo_->kTSize;
    tilingData_.baseTiling.flashAttnBaseParams.n2Size = faInfo_->n2Size;
    tilingData_.baseTiling.flashAttnBaseParams.gSize = faInfo_->gSize;
    tilingData_.baseTiling.flashAttnBaseParams.s1Size = faInfo_->s1Size;
    tilingData_.baseTiling.flashAttnBaseParams.s2Size = faInfo_->s2Size;
    tilingData_.baseTiling.flashAttnBaseParams.dSize = faInfo_->qkHeadDim;
    tilingData_.baseTiling.flashAttnBaseParams.dSizeV = faInfo_->vHeadDim;
    tilingData_.baseTiling.flashAttnBaseParams.scaleValue = faInfo_->softmaxScale;
    tilingData_.baseTiling.flashAttnBaseParams.cuSeqLengthsQSize = faInfo_->qLayout == FaLayout::TND ? faInfo_->bSize : 0;
    tilingData_.baseTiling.flashAttnBaseParams.cuSeqLengthsKVSize = faInfo_->qLayout == FaLayout::TND ? faInfo_->bSize : 0;
    tilingData_.baseTiling.flashAttnBaseParams.isKvContinuous = true;
    tilingData_.baseTiling.flashAttnBaseParams.isSoftMaxLseEnable = faInfo_->softmaxLseFlag;
    tilingData_.baseTiling.flashAttnBaseParams.iscuSeqLengthsNull = !cuSeqLenQFlag_;
    tilingData_.baseTiling.flashAttnBaseParams.iscuSeqLengthsKVNull = !cuSeqLenKVFlag_;
    tilingData_.baseTiling.flashAttnBaseParams.coreNum = numBlocks_;

    tilingData_.baseTiling.flashAttnAttenMaskParams.winLefts = faInfo_->winLeft;
    tilingData_.baseTiling.flashAttnAttenMaskParams.winRights = faInfo_->winRight;

    tilingData_.baseTiling.flashAttnLeftPaddingParams.isQHasLeftPadding = 0;
    tilingData_.baseTiling.flashAttnLeftPaddingParams.isKVHasLeftPadding = 0;
    tilingData_.baseTiling.flashAttnPageAttentionParams.blockSize = faInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = 0;
    tilingData_.baseTiling.flashAttnPageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;

    int64_t outSize = faInfo_->opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = faInfo_->softmaxLseFlag ? faInfo_->opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    tilingData_.baseTiling.flashAttnEmptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.totalOutputSize = outSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.totalSoftMaxLseOutputSize = lseSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.needInit = false;
}

ge::graphStatus FlashAttnTilingImpl::SetTilingData(FlashAttnTilingData &tilingData)
{
    FlashAttnTilingData *tiling = context_->GetTilingData<FlashAttnTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void FlashAttnTilingImpl::PrintAllTilingData()
{
    FlashAttnNoQuantTilingArch35 &baseTiling = tilingData_.baseTiling;
    FlashAttnBaseParams &flashAttnBaseParams = baseTiling.flashAttnBaseParams;
    FlashAttnAttenMaskParams &flashAttnAttenMaskParams = baseTiling.flashAttnAttenMaskParams;
    FlashAttnPageAttentionParams &flashAttnPageAttentionParams = baseTiling.flashAttnPageAttentionParams;
    FlashAttnLeftPaddingParams &flashAttnLeftPaddingParams = baseTiling.flashAttnLeftPaddingParams;
    FlashAttnWorkspaceParams &flashAttnWorkspaceParams = baseTiling.flashAttnWorkspaceParams;
    FlashAttnS1OuterSplitCoreParams &flashAttnS1OuterSplitCoreParams = baseTiling.flashAttnS1OuterSplitCoreParams;
    FlashAttnEmptyTensorParams &flashAttnEmptyTensorParams = baseTiling.flashAttnEmptyTensorParams;
    FlashAttnMetaData &flashAttnMetaData = tilingData_.flashAttnMetaData;

    OP_LOGD(faInfo_->opName, "bSize:%d", flashAttnBaseParams.bSize);
    OP_LOGD(faInfo_->opName, "t1Size:%d", flashAttnBaseParams.t1Size);
    OP_LOGD(faInfo_->opName, "t2Size:%d", flashAttnBaseParams.t2Size);
    OP_LOGD(faInfo_->opName, "n2Size:%d", flashAttnBaseParams.n2Size);
    OP_LOGD(faInfo_->opName, "gSize:%d", flashAttnBaseParams.gSize);
    OP_LOGD(faInfo_->opName, "s1Size:%d", flashAttnBaseParams.s1Size);
    OP_LOGD(faInfo_->opName, "s2Size:%d", flashAttnBaseParams.s2Size);
    OP_LOGD(faInfo_->opName, "dSize:%d", flashAttnBaseParams.dSize);
    OP_LOGD(faInfo_->opName, "dSizeV:%d", flashAttnBaseParams.dSizeV);
    OP_LOGD(faInfo_->opName, "dSizeRope:%d", flashAttnBaseParams.dSizeRope);
    OP_LOGD(faInfo_->opName, "cuSeqLengthsQSize:%d", flashAttnBaseParams.cuSeqLengthsQSize);
    OP_LOGD(faInfo_->opName, "cuSeqLengthsKVSize:%d", flashAttnBaseParams.cuSeqLengthsKVSize);
    OP_LOGD(faInfo_->opName, "scaleValue:%f", flashAttnBaseParams.scaleValue);
    OP_LOGD(faInfo_->opName, "iscuSeqLengthsNull:%d", flashAttnBaseParams.iscuSeqLengthsNull);
    OP_LOGD(faInfo_->opName, "iscuSeqLengthsKVNull:%d", flashAttnBaseParams.iscuSeqLengthsKVNull);
    OP_LOGD(faInfo_->opName, "isKvContinuous:%d", flashAttnBaseParams.isKvContinuous);
    OP_LOGD(faInfo_->opName, "isSoftMaxLseEnable:%d", flashAttnBaseParams.isSoftMaxLseEnable);
    OP_LOGD(faInfo_->opName, "coreNum:%d", flashAttnBaseParams.coreNum);

    OP_LOGD(faInfo_->opName, "maskMode:%d", flashAttnAttenMaskParams.sparseMode);
    OP_LOGD(faInfo_->opName, "winLefts:%d", flashAttnAttenMaskParams.winLefts);
    OP_LOGD(faInfo_->opName, "winRights:%d", flashAttnAttenMaskParams.winRights);
    OP_LOGD(faInfo_->opName, "attenMaskS1Size:%d", flashAttnAttenMaskParams.attenMaskS1Size);
    OP_LOGD(faInfo_->opName, "attenMaskS2Size:%d", flashAttnAttenMaskParams.attenMaskS2Size);

    OP_LOGD(faInfo_->opName, "paLayoutType:%d", flashAttnPageAttentionParams.paLayoutType);
    OP_LOGD(faInfo_->opName, "blockSize:%d", flashAttnPageAttentionParams.blockSize);
    OP_LOGD(faInfo_->opName, "maxBlockNumPerBatch:%d", flashAttnPageAttentionParams.maxBlockNumPerBatch);


    OP_LOGD(faInfo_->opName, "accumOutSize:%d", flashAttnWorkspaceParams.accumOutSize);
    OP_LOGD(faInfo_->opName, "logSumExpSize:%d", flashAttnWorkspaceParams.logSumExpSize);

    OP_LOGD(faInfo_->opName, "totalSize:%d", flashAttnS1OuterSplitCoreParams.totalSize);

    OP_LOGD(faInfo_->opName, "singleCoreSize:%d", flashAttnEmptyTensorParams.singleCoreSize);
    OP_LOGD(faInfo_->opName, "needInit:%d", flashAttnEmptyTensorParams.needInit);
    OP_LOGD(faInfo_->opName, "totalOutputSize:%d", flashAttnEmptyTensorParams.totalOutputSize);
    OP_LOGD(faInfo_->opName, "totalSoftMaxLseOutputSize:%d", flashAttnEmptyTensorParams.totalSoftMaxLseOutputSize);

    for (int aicIdx = 0; aicIdx <= FA_AIC_CORE_NUM; ++aicIdx) {
        OP_LOGD(faInfo_->opName, "FAMetadata[%d], [0]:%d, [1]:%d, [2]:%d, [3]:%d, [4]:%d, [5]:%d, [6]:%d, [7]:%d",
                aicIdx, flashAttnMetaData.FAMetadata[aicIdx][0], flashAttnMetaData.FAMetadata[aicIdx][1],
                flashAttnMetaData.FAMetadata[aicIdx][2], flashAttnMetaData.FAMetadata[aicIdx][3],
                flashAttnMetaData.FAMetadata[aicIdx][4], flashAttnMetaData.FAMetadata[aicIdx][5],
                flashAttnMetaData.FAMetadata[aicIdx][6], flashAttnMetaData.FAMetadata[aicIdx][7]);
    }

    for (int aivIdx = 0; aivIdx <= FA_AIV_CORE_NUM; ++aivIdx) {
        OP_LOGD(faInfo_->opName, "FDMetadata[%d], [0]:%d, [1]:%d, [2]:%d, [3]:%d, [4]:%d, [5]:%d, [6]:%d, [7]:%d",
                aivIdx, flashAttnMetaData.FDMetadata[aivIdx][0], flashAttnMetaData.FDMetadata[aivIdx][1],
                flashAttnMetaData.FDMetadata[aivIdx][2], flashAttnMetaData.FDMetadata[aivIdx][3],
                flashAttnMetaData.FDMetadata[aivIdx][4], flashAttnMetaData.FDMetadata[aivIdx][5],
                flashAttnMetaData.FDMetadata[aivIdx][6], flashAttnMetaData.FDMetadata[aivIdx][7]);
    }

    int64_t cap = context_->GetRawTilingData()->GetCapacity();
    OP_LOGD(faInfo_->opName, "Tiling Data context_ GetCapacity: %lu.", cap);
}

} // namespace optiling