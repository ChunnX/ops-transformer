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
 * \file mixed_quant_flash_attn_tiling.cpp
 * \brief
 */

#include "mixed_quant_flash_attn_tiling.h"
#include "../mixed_quant_flash_attn_tiling.h"
#include "../fa_adjust_sinner_souter.h"
#include <cmath>
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../mixed_quant_flash_attn_tiling_utils.h"
#include "../../op_kernel/arch35/mixed_quant_flash_attn_template_tiling_key.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace mixed_quant_flash_attn {
constexpr uint64_t PRE_LOAD_NUM_GQA_ARCH35 = 3;

void MixedQuantFlashAttnTilingImpl::InitTilingInfo(TilingInfo *tilingInfo)
{
    faInfo_ = static_cast<FaTilingInfo *>(tilingInfo);
}

bool MixedQuantFlashAttnTilingImpl::IsCapable()
{
    return true;
}

void MixedQuantFlashAttnTilingImpl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(faInfo_->opName, "MixedQuantFlashAttn schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus MixedQuantFlashAttnTilingImpl::DoOpTiling()
{
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

ge::graphStatus MixedQuantFlashAttnTilingImpl::SetPlatMemoryInfo()
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

void MixedQuantFlashAttnTilingImpl::InitImplParam()
{
    const gert::Tensor *actSeqLenQ = faInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor *actSeqLenKV = faInfo_->opParamInfo.cuSeqlensKv.tensor;
    uint32_t actSeqLenQDims = 0;
    uint32_t actSeqLenKVDims = 0;
    actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenKV != nullptr) ? actSeqLenKV->GetShapeSize() : 0;
    cuSeqLenQFlag_ = !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int32_t>() == nullptr));
    cuSeqLenKVFlag_ =
        !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int32_t>() == nullptr));

    const gert::Tensor *seqUsedQ = faInfo_->opParamInfo.sequsedQ.tensor;
    const gert::Tensor *seqUsedKv = faInfo_->opParamInfo.sequsedKv.tensor;
    uint32_t seqUsedQDims = (seqUsedQ != nullptr) ? seqUsedQ->GetShapeSize() : 0;
    uint32_t seqUsedKvDims = (seqUsedKv != nullptr) ? seqUsedKv->GetShapeSize() : 0;
    seqUsedQFlag_ = !((seqUsedQDims == 0) || (seqUsedQ == nullptr) || (seqUsedQ->GetData<int32_t>() == nullptr));
    seqUsedKvFlag_ = !((seqUsedKvDims == 0) || (seqUsedKv == nullptr) || (seqUsedKv->GetData<int32_t>() == nullptr));
}

void MixedQuantFlashAttnTilingImpl::SplitPolicy()
{
    int64_t winLeft = faInfo_->winLeft;
    int64_t winRight = faInfo_->winRight;
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK) ||
        faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        winLeft = MASK_MODE_INT_MAX;
        winRight = MASK_MODE_INT_MAX;
    }
    fa_tiling_util::AdjustSinnerAndSouter(static_cast<uint32_t>(faInfo_->vHeadDim),
                                          static_cast<uint32_t>(faInfo_->s1Size), faInfo_->s2Size,
                                          static_cast<int32_t>(faInfo_->maskMode), winLeft, winRight,
                                          static_cast<uint32_t>(faInfo_->qLayout), sOuterFactor_, sInnerFactor_);
    CalcNumBlocks(platformInfo_.aicNum);
    flashDecodeFlag_ = true;
}

void MixedQuantFlashAttnTilingImpl::UpdateTilingKeyConfig()
{
    // config=0: D=128, sOuter=64, sInner=256 (spec 当前唯一值)
    // 后续泛化新增 D 时：在此追加 if 分支，并同步 ConfigValue[] 与 tiling key 范围
    tilingKeyInfo_.config = 0;
}

void MixedQuantFlashAttnTilingImpl::UpdateTilingKeyLayout()
{
    // inputLayoutQ: 0=BSND, 1=BNSD, 2=TND (按 qLayout 映射, spec 3值)
    if (faInfo_->qLayout == FaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD;
    } else if (faInfo_->qLayout == FaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND;
    } else {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSND;
    }
}

void MixedQuantFlashAttnTilingImpl::UpdateTilingKeyKvLayout()
{
    if (faInfo_->kvLayout == FaLayout::PA_BBND) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BBH;
    } else if (faInfo_->kvLayout == FaLayout::PA_BNBD) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BNBD;
    }
}

void MixedQuantFlashAttnTilingImpl::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyKvLayout();
    tilingKeyInfo_.hasAttenMask = (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) ? 0 : 1;
    tilingKeyInfo_.quantMode = static_cast<uint64_t>(faInfo_->faQuantMode);
}

void MixedQuantFlashAttnTilingImpl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType,
                                    tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.config,
                                    tilingKeyInfo_.quantMode);
    OP_LOGI(faInfo_->opName, "The tilingkey is %llu.", tilingKey_);
    OP_LOGI(faInfo_->opName,
            "The tilingkey param is inOutLayoutType: %llu, kvLayoutType: %llu, hasAttenMask: %llu, config: %llu.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.hasAttenMask,
            tilingKeyInfo_.config);
}


void MixedQuantFlashAttnTilingImpl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(faInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(faInfo_->opName, "MixedQuantFlashAttn block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void MixedQuantFlashAttnTilingImpl::CalcWorkspaceSize()
{
    // TODO，沿用flash_attn逻辑，kernel侧完成后需调整
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    uint32_t mSize = sOuterFactor_ * platformInfo_.cvRatio;
    uint32_t dSize = faInfo_->vHeadDim;
    uint32_t dVBasicBlock = 0;
    if (dSize <= arch35FA::DSIZE_64) {
        dVBasicBlock = arch35FA::DSIZE_64;
    } else if (dSize <= arch35FA::DSIZE_128) {
        dVBasicBlock = arch35FA::DSIZE_128;
    } else if (dSize <= arch35FA::DSIZE_256) {
        dVBasicBlock = arch35FA::DSIZE_256;
    } else if (dSize <= arch35FA::DSIZE_512) {
        dVBasicBlock = arch35FA::DSIZE_512;
    }

    workspaceSize_ = sysWorkspaceSize;

    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if ((!dnFlag_ && dSize > arch35FA::DSIZE_128) || (dnFlag_ && dSize > arch35FA::DSIZE_192)) {
        bmm2Bytes = mSize * bmm2ResBlockSize * sizeof(float);
        if (dVBasicBlock > arch35FA::DSIZE_256) {
            vec2Bytes = mSize * dVBasicBlock * sizeof(float);
        }
    }
    workspaceSize_ += (bmm2Bytes + vec2Bytes) * 3 * platformInfo_.coreNum; // 3: perload 2次 需要2+1

    if (faInfo_->pageAttentionFlag) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        workspaceSize_ += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;
    }

    if (flashDecodeFlag_) {
        uint32_t faTmpAttenGmSize = platformInfo_.coreNum * 2 * mSize * dSize;
        uint32_t fatmpResLseGmSize = platformInfo_.coreNum * 2 * mSize * 8;
        workspaceSize_ += (faTmpAttenGmSize + 2 * fatmpResLseGmSize) * sizeof(float);
        tilingData_.baseTiling.mixedQuantFlashAttnWorkspaceParams.accumOutSize = faTmpAttenGmSize;
        tilingData_.baseTiling.mixedQuantFlashAttnWorkspaceParams.logSumExpSize = fatmpResLseGmSize;
    }

    OP_LOGI(faInfo_->opName, "Workspaces: %ld", workspaceSize_);
}

void MixedQuantFlashAttnTilingImpl::FillTiling()
{
    ComputeTilingData();
    SetFATilingData();
    PrintAllTilingData();
}

void MixedQuantFlashAttnTilingImpl::ComputeTilingData()
{
    tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.sparseMode = faInfo_->maskMode;
    tilingKeyInfo_.hasAttenMask = (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) ? 0 : 1;

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
        tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.attenMaskS1Size = maskS1Size;
        tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.attenMaskS2Size = maskS2Size;
    }

    if (faInfo_->pageAttentionFlag) {
        if (faInfo_->kvLayout == FaLayout::PA_BBND) {
            tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.paLayoutType = 1;
        } else if (faInfo_->kvLayout == FaLayout::PA_BNBD) {
            tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.paLayoutType = 0;
        }
    }
}

void MixedQuantFlashAttnTilingImpl::SetFATilingData()
{
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.bSize = faInfo_->bSize;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.t1Size = faInfo_->qTSize;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.t2Size = faInfo_->kTSize;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.n2Size = faInfo_->n2Size;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.gSize = faInfo_->gSize;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.s1Size = faInfo_->s1Size;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.s2Size = faInfo_->s2Size;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.dSize = faInfo_->qkHeadDim;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.dSizeV = faInfo_->vHeadDim;
    float scaleValue = faInfo_->softmaxScale;
    if (scaleValue == 0.0f) {
        scaleValue = 1.0f / std::sqrt(static_cast<float>(faInfo_->qkHeadDim));
    }
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.scaleValue = scaleValue;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.cuSeqLensQSize = faInfo_->qLayout == FaLayout::TND ? faInfo_->bSize : 0;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.cuSeqLensKVSize = 0;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.seqUsedQSize = seqUsedQFlag_ ? faInfo_->bSize : 0;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.seqUsedKvSize = seqUsedKvFlag_ ? faInfo_->bSize : 0;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.isKvContinuous = true;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.isSoftMaxLseEnable = faInfo_->softmaxLseFlag;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.iscuSeqLengthsNull = !cuSeqLenQFlag_;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.iscuSeqLengthsKVNull = !cuSeqLenKVFlag_;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.coreNum = numBlocks_;

    tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.winLefts = faInfo_->winLeft;
    tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.winRights = faInfo_->winRight;
    if (faInfo_->winLeft == -1) {
        tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.winLefts = MASK_MODE_INT_MAX;
    }
    if (faInfo_->winRight == -1) {
        tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.winRights = MASK_MODE_INT_MAX;
    }
    tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.blockSize = faInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = 0;
    if (faInfo_->pageAttentionFlag) {
        maxBlockNumPerBatch = faInfo_->opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;
    // 以下三种情况需要刷0
    // 1、存在seqUsedQ
    // 2、存在seqUsedKv(可能导致S2小于S1出现行无效)
    // 3、带mask且S2小于S1
    if (seqUsedQFlag_ || seqUsedKvFlag_ || ((faInfo_->s1Size > faInfo_->s2Size) && tilingKeyInfo_.hasAttenMask)) {
        tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.needInitOutput = true;
    } else {
        tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.needInitOutput = false;
    }
}

ge::graphStatus MixedQuantFlashAttnTilingImpl::SetTilingData(MixedQuantFlashAttnTilingData &tilingData)
{
    MixedQuantFlashAttnTilingData *tiling = context_->GetTilingData<MixedQuantFlashAttnTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void MixedQuantFlashAttnTilingImpl::PrintAllTilingData()
{
    MixedQuantFlashAttnTilingArch35 &baseTiling = tilingData_.baseTiling;
    MixedQuantFlashAttnBaseParams &mixedQuantFlashAttnBaseParams = baseTiling.mixedQuantFlashAttnBaseParams;
    MixedQuantFlashAttnAttenMaskParams &mixedQuantFlashAttnAttenMaskParams = baseTiling.mixedQuantFlashAttnAttenMaskParams;
    MixedQuantFlashAttnPageAttentionParams &mixedQuantFlashAttnPageAttentionParams = baseTiling.mixedQuantFlashAttnPageAttentionParams;
    MixedQuantFlashAttnWorkspaceParams &mixedQuantFlashAttnWorkspaceParams = baseTiling.mixedQuantFlashAttnWorkspaceParams;
    MixedQuantFlashAttnS1OuterSplitCoreParams &mixedQuantFlashAttnS1OuterSplitCoreParams = baseTiling.mixedQuantFlashAttnS1OuterSplitCoreParams;
    MixedQuantFlashAttnMetaData &mixedQuantFlashAttnMetaData = tilingData_.mixedQuantFlashAttnMetaData;

    OP_LOGD(faInfo_->opName, "bSize:%d", mixedQuantFlashAttnBaseParams.bSize);
    OP_LOGD(faInfo_->opName, "t1Size:%d", mixedQuantFlashAttnBaseParams.t1Size);
    OP_LOGD(faInfo_->opName, "t2Size:%d", mixedQuantFlashAttnBaseParams.t2Size);
    OP_LOGD(faInfo_->opName, "n2Size:%d", mixedQuantFlashAttnBaseParams.n2Size);
    OP_LOGD(faInfo_->opName, "gSize:%d", mixedQuantFlashAttnBaseParams.gSize);
    OP_LOGD(faInfo_->opName, "s1Size:%d", mixedQuantFlashAttnBaseParams.s1Size);
    OP_LOGD(faInfo_->opName, "s2Size:%d", mixedQuantFlashAttnBaseParams.s2Size);
    OP_LOGD(faInfo_->opName, "dSize:%d", mixedQuantFlashAttnBaseParams.dSize);
    OP_LOGD(faInfo_->opName, "dSizeV:%d", mixedQuantFlashAttnBaseParams.dSizeV);
    OP_LOGD(faInfo_->opName, "dSizeRope:%d", mixedQuantFlashAttnBaseParams.dSizeRope);
    OP_LOGD(faInfo_->opName, "cuSeqLensQSize:%d", mixedQuantFlashAttnBaseParams.cuSeqLensQSize);
    OP_LOGD(faInfo_->opName, "cuSeqLensKVSize:%d", mixedQuantFlashAttnBaseParams.cuSeqLensKVSize);
    OP_LOGD(faInfo_->opName, "seqUsedQSize:%d", mixedQuantFlashAttnBaseParams.seqUsedQSize);
    OP_LOGD(faInfo_->opName, "seqUsedKvSize:%d", mixedQuantFlashAttnBaseParams.seqUsedKvSize);
    OP_LOGD(faInfo_->opName, "scaleValue:%f", mixedQuantFlashAttnBaseParams.scaleValue);
    OP_LOGD(faInfo_->opName, "iscuSeqLengthsNull:%d", mixedQuantFlashAttnBaseParams.iscuSeqLengthsNull);
    OP_LOGD(faInfo_->opName, "iscuSeqLengthsKVNull:%d", mixedQuantFlashAttnBaseParams.iscuSeqLengthsKVNull);
    OP_LOGD(faInfo_->opName, "isKvContinuous:%d", mixedQuantFlashAttnBaseParams.isKvContinuous);
    OP_LOGD(faInfo_->opName, "isSoftMaxLseEnable:%d", mixedQuantFlashAttnBaseParams.isSoftMaxLseEnable);
    OP_LOGD(faInfo_->opName, "coreNum:%d", mixedQuantFlashAttnBaseParams.coreNum);

    OP_LOGD(faInfo_->opName, "maskMode:%d", mixedQuantFlashAttnAttenMaskParams.sparseMode);
    OP_LOGD(faInfo_->opName, "winLefts:%d", mixedQuantFlashAttnAttenMaskParams.winLefts);
    OP_LOGD(faInfo_->opName, "winRights:%d", mixedQuantFlashAttnAttenMaskParams.winRights);
    OP_LOGD(faInfo_->opName, "attenMaskS1Size:%d", mixedQuantFlashAttnAttenMaskParams.attenMaskS1Size);
    OP_LOGD(faInfo_->opName, "attenMaskS2Size:%d", mixedQuantFlashAttnAttenMaskParams.attenMaskS2Size);

    OP_LOGD(faInfo_->opName, "paLayoutType:%d", mixedQuantFlashAttnPageAttentionParams.paLayoutType);
    OP_LOGD(faInfo_->opName, "blockSize:%d", mixedQuantFlashAttnPageAttentionParams.blockSize);
    OP_LOGD(faInfo_->opName, "maxBlockNumPerBatch:%d", mixedQuantFlashAttnPageAttentionParams.maxBlockNumPerBatch);

    OP_LOGD(faInfo_->opName, "accumOutSize:%d", mixedQuantFlashAttnWorkspaceParams.accumOutSize);
    OP_LOGD(faInfo_->opName, "logSumExpSize:%d", mixedQuantFlashAttnWorkspaceParams.logSumExpSize);

    OP_LOGD(faInfo_->opName, "totalSize:%d", mixedQuantFlashAttnS1OuterSplitCoreParams.totalSize);

    for (int aicIdx = 0; aicIdx < MQFA_AIC_CORE_NUM; ++aicIdx) {
        OP_LOGD(faInfo_->opName, "FAMetadata[%d], [0]:%d, [1]:%d, [2]:%d, [3]:%d, [4]:%d, [5]:%d, [6]:%d, [7]:%d",
                aicIdx, mixedQuantFlashAttnMetaData.FAMetadata[aicIdx][0], mixedQuantFlashAttnMetaData.FAMetadata[aicIdx][1],
                mixedQuantFlashAttnMetaData.FAMetadata[aicIdx][2], mixedQuantFlashAttnMetaData.FAMetadata[aicIdx][3],
                mixedQuantFlashAttnMetaData.FAMetadata[aicIdx][4], mixedQuantFlashAttnMetaData.FAMetadata[aicIdx][5],
                mixedQuantFlashAttnMetaData.FAMetadata[aicIdx][6], mixedQuantFlashAttnMetaData.FAMetadata[aicIdx][7]);
    }

    for (int aivIdx = 0; aivIdx < MQFA_AIV_CORE_NUM; ++aivIdx) {
        OP_LOGD(faInfo_->opName, "FDMetadata[%d], [0]:%d, [1]:%d, [2]:%d, [3]:%d, [4]:%d, [5]:%d, [6]:%d, [7]:%d",
                aivIdx, mixedQuantFlashAttnMetaData.FDMetadata[aivIdx][0], mixedQuantFlashAttnMetaData.FDMetadata[aivIdx][1],
                mixedQuantFlashAttnMetaData.FDMetadata[aivIdx][2], mixedQuantFlashAttnMetaData.FDMetadata[aivIdx][3],
                mixedQuantFlashAttnMetaData.FDMetadata[aivIdx][4], mixedQuantFlashAttnMetaData.FDMetadata[aivIdx][5],
                mixedQuantFlashAttnMetaData.FDMetadata[aivIdx][6], mixedQuantFlashAttnMetaData.FDMetadata[aivIdx][7]);
    }

    int64_t cap = context_->GetRawTilingData()->GetCapacity();
    OP_LOGD(faInfo_->opName, "Tiling Data context_ GetCapacity: %lu.", cap);
}

} // namespace mixed_quant_flash_attn

using mixed_quant_flash_attn::MixedQuantFlashAttnTilingImpl;

// 值越小表示优先级越高
REGISTER_TILING_TEMPLATE_FIA(MixedQuantFlashAttn, MixedQuantFlashAttnTilingImpl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 1);
} // namespace optiling
