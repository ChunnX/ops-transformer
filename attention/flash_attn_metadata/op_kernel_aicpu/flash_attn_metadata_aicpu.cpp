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
 * \file flash_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include "cust_op/cust_cpu_utils.h"
#include <cstdio>
#include <cmath>
#include "flash_attn_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t FlashAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SectionStreamKResult splitRes;
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool FlashAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs =
                GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
                GetAttrValue(ctx, "head_dim", headDim_) &&
                GetAttrValue(ctx, "soc_version", socVersion_) &&
                GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    return ParamsInit();
    // return true;
}

bool FlashAttnMetadataCpuKernel::ParamsInit()
{
    // Device info 
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
    deviceInfo.socVersion = socVersion_;
    // baseInfo
    baseInfo.batchSize = batchSize_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.headDim = headDim_;
    baseInfo.attenMaskFlag = maskMode_ == 0 ? false : true;// todo
    baseInfo.sparseMode = maskMode_;
    baseInfo.preToken = winLeft_ == -1 ? std::numeric_limits<uint32_t>::max() : winLeft_;
    baseInfo.nextToken = winRight_ == -1 ? std::numeric_limits<uint32_t>::max() : winRight_;
    baseInfo.isCumulativeQuerySeq = layoutQ_ == "TND" || layoutQ_ == "NTD";
    baseInfo.isCumulativeKvSeq = layoutKv_ == "TND" || layoutKv_ == "NTD";
    baseInfo.layoutQuery = ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16; // todo
    baseInfo.kvType = load_balance::DataType::FP16; // todo

    // actual seq size
    if (!baseInfo.isCumulativeQuerySeq && sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        auto sequsedQ = (int32_t *)sequsedQ_->GetData();
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = sequsedQ[i];
        }
    } else if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        auto cuSeqlensQ = (int32_t *)cuSeqlensQ_->GetData();
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 1; i <= batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = cuSeqlensQ[i];
        }
    }
    if (!baseInfo.isCumulativeKvSeq && sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        auto sequsedKv = (int32_t *)sequsedKv_->GetData();
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = sequsedKv[i];
        }
    } else if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        auto cuSeqlensKv = (int32_t *)cuSeqlensKv_->GetData();
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 1; i <= batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = cuSeqlensKv[i];
        }
    }

    // param
    if (numHeadsKv_ == 0) {
        numHeadsKv_ = numHeadsQ_;
        groupSize_ = 1;
    } else {
        groupSize_ = numHeadsQ_ / numHeadsKv_;
    }
    bool isTransposeLayout = (layoutQ_ != layoutOut_ || layoutQ_ == "NTD");
    bool pfaMergeFlag = !(baseInfo.attenMaskFlag || isTransposeLayout);
    if (baseInfo.headDim <= NUM_128) {
        bool checkQueryAndValueS = baseInfo.querySeqSize <= NUM_64 && baseInfo.kvSeqSize > NUM_128;
        uint32_t sparseMode = baseInfo.sparseMode;
        int32_t preTokens = baseInfo.preToken;
        int32_t nextTokens = baseInfo.nextToken;
        if (sparseMode == 0) {
            preTokens = (preTokens > 0) ? 0 : preTokens;
        } else if (sparseMode == 4) {
            nextTokens = (nextTokens > 0) ? 0 : nextTokens;
        }
        bool checkSparseMode = (sparseMode != 2 && preTokens + nextTokens > 128);
        if (checkQueryAndValueS && checkSparseMode) {
            mBaseSize_ = NUM_32;
            s2BaseSize_ = NUM_256;
        } else if (((baseInfo.layoutQuery == Layout::BSH) || (baseInfo.layoutQuery == Layout::BSND) ||
                    (baseInfo.layoutQuery == Layout::TND)) &&
                   pfaMergeFlag) {
            mBaseSize_ = NUM_32;
            s2BaseSize_ = NUM_256;
        }
    } else if (baseInfo.headDim > NUM_128 && baseInfo.querySeqSize != 1) {
        if (((baseInfo.layoutQuery == Layout::BSH) || (baseInfo.layoutQuery == Layout::BSND) ||
                    (baseInfo.layoutQuery == Layout::TND)) &&
                   pfaMergeFlag && baseInfo.headDim <= NUM_256) { // 256 : D size
            mBaseSize_ = NUM_32;
            s2BaseSize_ = NUM_256;
        } else {
            mBaseSize_ = NUM_64;
            s2BaseSize_ = NUM_128;
        }
    } else if (baseInfo.querySeqSize == 1 && baseInfo.headDim > NUM_128) { // IFA VD > 128
        mBaseSize_ = NUM_64;
        s2BaseSize_ = NUM_128;
    }
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    return true;
}

bool FlashAttnMetadataCpuKernel::BalanceSchedule(SectionStreamKResult &splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == LOAD_BALANCE_SUCCESS;
}

bool FlashAttnMetadataCpuKernel::GenMetaData(SectionStreamKResult &splitRes)
{
    uint32_t sectionNum = splitRes.sectionNum;
    detail::FaMetaData faMetadata(metaData_->GetData(), sectionNum);
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        for (uint32_t i = 0; i < aicCoreNum_; ++i) {
            // faMetadata.setFaMetadata(sectionId, i, optiling::FA_CORE_ENABLE_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
        }
        for (uint32_t i = 0; i < aivCoreNum_; ++i) {
            // faMetadata.setFdMetadata(sectionId, i, optiling::FD_CORE_ENABLE_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_BN2_IDX_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_START_INDEX, 0U);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX, 0U);
        }
        // FA Metadata Generate
        auto faSplitRes = splitRes.sectionFaResult[sectionId];
        for (uint32_t i = 0; i < faSplitRes.usedCoreNum; ++i) {
            // faMetadata.setFaMetadata(sectionId, i, optiling::FA_CORE_ENABLE_INDEX, 1U);
            // FA start
            if (i > 0) {
                faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX, faSplitRes.bN2End[i - 1]);
                faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, faSplitRes.gS1End[i - 1]);
                faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, faSplitRes.s2End[i - 1]);
            } else if (sectionId > 0) {
                auto preFaSplitRes = splitRes.sectionFaResult[sectionId - 1];
                faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX, preFaSplitRes.bN2End[preFaSplitRes.usedCoreNum - 1]);
                faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, preFaSplitRes.gS1End[preFaSplitRes.usedCoreNum - 1]);
                faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, preFaSplitRes.s2End[preFaSplitRes.usedCoreNum - 1]);
            }
            // FA end
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_END_INDEX, faSplitRes.bN2End[i]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, faSplitRes.gS1End[i]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, faSplitRes.s2End[i]);
            // FA idx
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, faSplitRes.firstFdDataWorkspaceIdx[i]);
        }
        // FD Metadata Generate
        auto fdSplitRes = splitRes.sectionFdResult[sectionId];
        for (uint32_t i = 0; i < fdSplitRes.usedVecNum; ++i) {
            // faMetadata.setFdMetadata(sectionId, i, optiling::FD_CORE_ENABLE_INDEX, 1U);
            uint32_t curTaskIdx = fdSplitRes.taskIdx[i];
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_BN2_IDX_INDEX, fdSplitRes.bN2Idx[curTaskIdx]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX, fdSplitRes.gS1Idx[curTaskIdx]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX, fdSplitRes.workspaceIdx[curTaskIdx]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX, fdSplitRes.s2SplitNum[curTaskIdx]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_START_INDEX, fdSplitRes.mStart[i]);
            faMetadata.setFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX, fdSplitRes.mLen[i]);
        }
    }
    return true;
}

namespace {
static const char *kernelType = "FlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu