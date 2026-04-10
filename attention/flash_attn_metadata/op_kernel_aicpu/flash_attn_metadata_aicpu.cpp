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
    deviceInfo.aicCoreNum = aicCoreNum_;
    deviceInfo.aivCoreNum = aivCoreNum_;
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
        for (uint32_t i = 0; i <= batchSize_; ++i) {
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
        for (uint32_t i = 0; i <= batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = cuSeqlensKv[i];
        }
    }

    // param
    groupSize_ = numHeadsQ_ / numHeadsKv_;
    if (layoutQ_ == "TND" || layoutQ_ == "NTD") {
        mBaseSize_ = NUM_512;
    } else {
        uint32_t sInnerSizeAlign_ = AlignUp(s2BaseSize_, 16U);
        if (maxSeqlenQ_ <= 16) {
            if (sInnerSizeAlign_ <= NUM_512) {
                mBaseSize_ = NUM_512;
            } else if (sInnerSizeAlign_ <= NUM_1024) {
                mBaseSize_ = NUM_256;
            } else if (sInnerSizeAlign_ <= NUM_2048) {
                mBaseSize_ = NUM_128;
            } else if (sInnerSizeAlign_ <= NUM_4096) {
                mBaseSize_ = NUM_64;
            } else { // sInnerSizeAlign_最大值为8192
                mBaseSize_ = NUM_32;
            }
        } else {
            mBaseSize_ = NUM_512;
        }
    }
    if (layoutQ_ == "TND" || layoutQ_ == "NTD") {
        s2BaseSize_ = NUM_512;
    } else {
        if (maxSeqlenQ_ <= 16) {
            s2BaseSize_ = NUM_8192; // 8192
            uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
            uint32_t idx = std::min(groupSize_ / 5U, 2U);
            s2BaseSize_ = sInnerSize[idx];
        } else {
            s2BaseSize_ = NUM_512;
        }
    }
    if (maskMode_ == 2 || maskMode_ == 3 || maskMode_ == 4) {
        s2BaseSize_ = std::min(static_cast<int64_t>(s2BaseSize_), NUM_1024); // attention mask压缩场景，基本块最大支持1024*1024
    }
    if (s2BaseSize_ > maxSeqlenKv_) {
        s2BaseSize_ = maxSeqlenKv_;
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
    uint32_t* metadataPtr = static_cast<uint32_t*>(metaData_->GetData());
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        for (size_t i = 0; i < aicCoreNum_; ++i) {
            // faMetadata.setFaMetadata(sectionId, i, optiling::FA_CORE_ENABLE_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, 0U);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
        }
        for (size_t i = 0; i < aivCoreNum_; ++i) {
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
        for (size_t i = 0; i < faSplitRes.usedCoreNum; ++i) {
            // faMetadata.setFaMetadata(sectionId, i, optiling::FA_CORE_ENABLE_INDEX, 1U);
            // FA start
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX, (i == 0) ? 0U : faSplitRes.bN2End[i - 1]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, (i == 0) ? 0U : faSplitRes.gS1End[i - 1]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, (i == 0) ? 0U : faSplitRes.s2End[i - 1]);
            // FA end
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_BN2_END_INDEX, faSplitRes.bN2End[i]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, faSplitRes.gS1End[i]);
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, faSplitRes.s2End[i]);
            // FA idx
            faMetadata.setFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, faSplitRes.firstFdDataWorkspaceIdx[i]);
        }
        // FD Metadata Generate
        auto fdSplitRes = splitRes.sectionFdResult[sectionId];
        for (size_t i = 0; i < fdSplitRes.usedVecNum; ++i) {
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