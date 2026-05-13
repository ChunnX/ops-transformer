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
#include <cstdio>
#include <cmath>
#include "flash_attn_metadata_aicpu.h"
#include "../../flash_attn/op_host/fa_adjust_sinner_souter.h"

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
}

std::vector<int64_t> FlashAttnMetadataCpuKernel::GetTensorDataAsInt64(Tensor *tensor, size_t size)
{
    std::vector<int64_t> result(size);
    if (tensor == nullptr || tensor->GetData() == nullptr || size == 0) {
        return result;
    }

    DataType dataType = tensor->GetDataType();
    void *data = tensor->GetData();

    switch (dataType) {
        case DT_INT32:
            {
                int32_t *ptr = static_cast<int32_t *>(data);
                for (size_t i = 0; i < size; ++i) {
                    result[i] = static_cast<int64_t>(ptr[i]);
                }
                break;
            }
        case DT_INT64:
            {
                int64_t *ptr = static_cast<int64_t *>(data);
                for (size_t i = 0; i < size; ++i) {
                    result[i] = ptr[i];
                }
                break;
            }
        case DT_INT16:
            {
                int16_t *ptr = static_cast<int16_t *>(data);
                for (size_t i = 0; i < size; ++i) {
                    result[i] = static_cast<int64_t>(ptr[i]);
                }
                break;
            }
        case DT_UINT32:
            {
                uint32_t *ptr = static_cast<uint32_t *>(data);
                for (size_t i = 0; i < size; ++i) {
                    result[i] = static_cast<int64_t>(ptr[i]);
                }
                break;
            }
        case DT_UINT64:
            {
                uint64_t *ptr = static_cast<uint64_t *>(data);
                for (size_t i = 0; i < size; ++i) {
                    result[i] = static_cast<int64_t>(ptr[i]);
                }
                break;
            }
        case DT_UINT16:
            {
                uint16_t *ptr = static_cast<uint16_t *>(data);
                for (size_t i = 0; i < size; ++i) {
                    result[i] = static_cast<int64_t>(ptr[i]);
                }
                break;
            }
        default:
            break;
    }
    return result;
}

bool FlashAttnMetadataCpuKernel::ParamsInit()
{
    // Device info
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
    deviceInfo.cvRadio = aivCoreNum_ / aicCoreNum_;
    // baseInfo
    // actual seq size
    baseInfo.isCumulativeQuerySeq = layoutQ_ == "TND" || layoutQ_ == "NTD";
    baseInfo.isCumulativeKvSeq = layoutKv_ == "TND" || layoutKv_ == "NTD";
    if (batchSize_ > 0) {
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        if (baseInfo.isCumulativeQuerySeq) {
            for (uint32_t i = 1; i < batchSize_; ++i) {
                baseInfo.actualQuerySeqSize[i] += baseInfo.actualQuerySeqSize[i - 1];
            }
        }
        if (baseInfo.isCumulativeKvSeq) {
            for (uint32_t i = 1; i < batchSize_; ++i) {
                baseInfo.actualKvSeqSize[i] += baseInfo.actualKvSeqSize[i - 1];
            }
        }
    }
    if (baseInfo.isCumulativeQuerySeq && cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        batchSize_ = cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
        auto cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_, batchSize_ + 1);
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = cuSeqlensQ[i + 1];
            maxSeqlenQ_ = std::max(static_cast<int64_t>(maxSeqlenQ_), cuSeqlensQ[i + 1] - cuSeqlensQ[i]);
        }
    }
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        batchSize_ = sequsedQ_->GetTensorShape()->GetDimSize(0);
        auto sequsedQ = GetTensorDataAsInt64(sequsedQ_, batchSize_);
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = sequsedQ[i];
            if (baseInfo.isCumulativeQuerySeq && (i > 0)) {
                baseInfo.actualQuerySeqSize[i] += baseInfo.actualQuerySeqSize[i - 1];
            }
            maxSeqlenQ_ = std::max(static_cast<int64_t>(maxSeqlenQ_), sequsedQ[i]);
        }
    }
    if (baseInfo.isCumulativeKvSeq && cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        batchSize_ = cuSeqlensKv_->GetTensorShape()->GetDimSize(0) - 1;
        auto cuSeqlensKv = GetTensorDataAsInt64(cuSeqlensKv_, batchSize_ + 1);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = cuSeqlensKv[i + 1];
            maxSeqlenKv_ = std::max(static_cast<int64_t>(maxSeqlenKv_), cuSeqlensKv[i + 1] - cuSeqlensKv[i]);
        }
    }
    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        batchSize_ = sequsedKv_->GetTensorShape()->GetDimSize(0);
        auto sequsedKv = GetTensorDataAsInt64(sequsedKv_, batchSize_);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = sequsedKv[i];
            if (baseInfo.isCumulativeKvSeq && (i > 0)) {
                baseInfo.actualKvSeqSize[i] += baseInfo.actualKvSeqSize[i - 1];
            }
            maxSeqlenKv_ = std::max(static_cast<int64_t>(maxSeqlenKv_), sequsedKv[i]);
        }
    }
    baseInfo.batchSize = batchSize_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.headDim = headDim_;
    baseInfo.attenMaskFlag = true;
    baseInfo.sparseMode = 0;
    baseInfo.preToken = winLeft_ == -1 ? std::numeric_limits<uint32_t>::max() : winLeft_;
    baseInfo.nextToken = winRight_ == -1 ? std::numeric_limits<uint32_t>::max() : winRight_;
    baseInfo.layoutQuery = ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16; // noquant
    baseInfo.kvType = load_balance::DataType::FP16; // noquant
    // param
    if (numHeadsKv_ == 0) {
        numHeadsKv_ = numHeadsQ_;
        groupSize_ = 1;
    } else {
        groupSize_ = numHeadsQ_ / numHeadsKv_;
    }
    uint32_t qlayout = optiling::fa_tiling_util::LAYOUT_BNSD;
    if (baseInfo.layoutQuery == Layout::BSH || baseInfo.layoutQuery == Layout::BSND) {
        qlayout = optiling::fa_tiling_util::LAYOUT_BSH;
    } else if (baseInfo.layoutQuery == Layout::TND) {
        qlayout = optiling::fa_tiling_util::LAYOUT_TND;
    }
    optiling::fa_tiling_util::AdjustSinnerAndSouter(baseInfo.headDim, baseInfo.querySeqSize, baseInfo.kvSeqSize,
                                                    baseInfo.sparseMode, baseInfo.preToken, baseInfo.nextToken,
                                                    qlayout, mBaseSize_, s2BaseSize_);
    mBaseSize_ = mBaseSize_ * deviceInfo.cvRadio; // CV_Radio
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    return true;
}

bool FlashAttnMetadataCpuKernel::BalanceSchedule(SectionStreamKResult &splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool FlashAttnMetadataCpuKernel::GenMetaData(SectionStreamKResult &splitRes)
{
    if (metaData_ == nullptr || metaData_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    detail::FaMetaData faMetadata(metaData_->GetData());
    uint32_t* ptr = (uint32_t*)metaData_->GetData();
    ptr[1] = splitRes.sectionFdResult.usedVecNum > 0 ? 1 : 0;
    ptr[2] = mBaseSize_;
    ptr[3] = s2BaseSize_;
    for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
        faMetadata.setFaMetadata(i, optiling::FA_BN2_START_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_M_START_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_S2_START_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_BN2_END_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_M_END_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_S2_END_INDEX, 0U);
        faMetadata.setFaMetadata(i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
    }
    for (uint32_t i = 0; i < AIV_CORE_NUM; ++i) {
        faMetadata.setFdMetadata(i, optiling::FD_BN2_IDX_INDEX, 0U);
        faMetadata.setFdMetadata(i, optiling::FD_M_IDX_INDEX, 0U);
        faMetadata.setFdMetadata(i, optiling::FD_WORKSPACE_IDX_INDEX, 0U);
        faMetadata.setFdMetadata(i, optiling::FD_WORKSPACE_NUM_INDEX, 0U);
        faMetadata.setFdMetadata(i, optiling::FD_M_START_INDEX, 0U);
        faMetadata.setFdMetadata(i, optiling::FD_M_NUM_INDEX, 0U);
    }
    // FA Metadata Generate
    auto faSplitRes = splitRes.sectionFaResult;
    for (uint32_t i = 0; i < faSplitRes.usedCoreNum; ++i) {
        // FA start
        if (i > 0) {
            faMetadata.setFaMetadata(i, optiling::FA_BN2_START_INDEX, faSplitRes.bN2End[i - 1]);
            faMetadata.setFaMetadata(i, optiling::FA_M_START_INDEX, faSplitRes.gS1End[i - 1]);
            faMetadata.setFaMetadata(i, optiling::FA_S2_START_INDEX, faSplitRes.s2End[i - 1]);
        }
        // FA end
        faMetadata.setFaMetadata(i, optiling::FA_BN2_END_INDEX, faSplitRes.bN2End[i]);
        faMetadata.setFaMetadata(i, optiling::FA_M_END_INDEX, faSplitRes.gS1End[i]);
        faMetadata.setFaMetadata(i, optiling::FA_S2_END_INDEX, faSplitRes.s2End[i]);
        // FA idx
        faMetadata.setFaMetadata(i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                 faSplitRes.firstFdDataWorkspaceIdx[i]);
    }
    // FD Metadata Generate
    auto fdSplitRes = splitRes.sectionFdResult;
    for (uint32_t i = 0; i < fdSplitRes.usedVecNum; ++i) {
        uint32_t curTaskIdx = fdSplitRes.taskIdx[i];
        faMetadata.setFdMetadata(i, optiling::FD_BN2_IDX_INDEX, fdSplitRes.bN2Idx[curTaskIdx]);
        faMetadata.setFdMetadata(i, optiling::FD_M_IDX_INDEX, fdSplitRes.gS1Idx[curTaskIdx]);
        faMetadata.setFdMetadata(i, optiling::FD_WORKSPACE_IDX_INDEX, fdSplitRes.workspaceIdx[curTaskIdx]);
        faMetadata.setFdMetadata(i, optiling::FD_WORKSPACE_NUM_INDEX, fdSplitRes.s2SplitNum[curTaskIdx]);
        faMetadata.setFdMetadata(i, optiling::FD_M_START_INDEX, fdSplitRes.mStart[i]);
        faMetadata.setFdMetadata(i, optiling::FD_M_NUM_INDEX, fdSplitRes.mLen[i]);
    }
    return true;
}

namespace {
static const char *kernelType = "FlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
