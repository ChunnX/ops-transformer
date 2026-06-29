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
 * \file stem_indexer_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include "stem_indexer_metadata_aicpu.h"

#define KERNEL_STATUS_OK            0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t StemIndexerMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    SectionStreamKResult result;
    success = BalanceSchedule(result);

    success = GenMetadata(result);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool StemIndexerMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    qSeqLens_ = ctx.Input(static_cast<uint32_t>(ParamId::qSeqLens));
    kvSeqLens_ = ctx.Input(static_cast<uint32_t>(ParamId::kvSeqLens));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs = GetAttrValue(ctx, "q_heads", numHeadsQ_) &&
                         GetAttrValue(ctx, "kv_heads", numHeadsKv_) &&
                         GetAttrValue(ctx, "causal", causal_) &&
                         GetAttrValue(ctx, "stem_block_size", stemBlockSize_) &&
                         GetAttrValue(ctx, "window_size", windowSize_) &&
                         GetAttrValue(ctx, "soc_version", socVersion_) &&
                         GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                         GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "dim_qkflat", headDim_);
    return true;
}

std::vector<int64_t> StemIndexerMetadataCpuKernel::GetTensorDataAsInt64(Tensor *tensor, size_t size)
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

bool StemIndexerMetadataCpuKernel::BalanceSchedule(SectionStreamKResult &result)
{
    DeviceInfo deviceInfo {};
    StemIndexerBaseInfo baseInfo {};
    load_balance::SectionStreamKParam param {};
    auto success = GenerateDeviceInfo(deviceInfo) &&
                   GenerateBaseInfo(baseInfo) &&
                   GenerateSectionStreamKParam(param);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, result) == SECTION_STREAM_K_SUCCESS;
}

bool StemIndexerMetadataCpuKernel::GenerateDeviceInfo(DeviceInfo &deviceInfo)
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = 1;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
    return true;
}

bool StemIndexerMetadataCpuKernel::GenerateBaseInfo(StemIndexerBaseInfo &baseInfo)
{
    KERNEL_CHECK_NULLPTR(qSeqLens_, false, "q_seq_len is nullptr!");
    KERNEL_CHECK_NULLPTR(kvSeqLens_, false, "kv_seq_len is nullptr!");
    KERNEL_CHECK_FALSE(qSeqLens_->NumElements() == kvSeqLens_->NumElements(), false,
        "q_seq_len(%ld) has different length with kv_seq_len(%ld)",
        qSeqLens_->NumElements(), kvSeqLens_->NumElements());

    size_t batchSize = qSeqLens_->NumElements();

    baseInfo.batchSize = static_cast<uint32_t>(batchSize);
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.querySeqSize = 0;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.kvSeqSize = 0;
    baseInfo.headDim = headDim_;
    baseInfo.attenMaskFlag = causal_;
    baseInfo.sparseMode = (causal_) ? static_cast<uint32_t>(load_balance::SparseMode::RIGHT_DOWN_CAUSAL) :
        static_cast<uint32_t>(load_balance::SparseMode::BUTT);
    baseInfo.preToken = -1;
    baseInfo.nextToken = -1;
    baseInfo.layoutQuery = load_balance::Layout::BUTT;
    baseInfo.layoutKv = load_balance::Layout::BUTT;
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    baseInfo.isCumulativeQuerySeq = false;
    baseInfo.isCumulativeKvSeq = false;
    baseInfo.actualQuerySeqSize = GetTensorDataAsInt64(qSeqLens_, batchSize);
    baseInfo.actualKvSeqSize = GetTensorDataAsInt64(kvSeqLens_, batchSize);
    for (size_t i = 0; i < batchSize; ++i) {
        baseInfo.actualQuerySeqSize[i] = load_balance::CeilDiv(baseInfo.actualQuerySeqSize[i], stemBlockSize_);
        baseInfo.actualKvSeqSize[i] = load_balance::CeilDiv(baseInfo.actualKvSeqSize[i], stemBlockSize_);
    }
    // New
    baseInfo.tailSize = windowSize_;

    return true;
}

bool StemIndexerMetadataCpuKernel::GenerateSectionStreamKParam(load_balance::SectionStreamKParam &param)
{
    param.l2Byte = 0U;
    param.mBaseSize = 96;   // 96: Fix mBaseSize
    param.s2BaseSize = 256; // 256: Fix s2BaseSize
    param.fdOn = false;
    return true;
}

bool StemIndexerMetadataCpuKernel::GenMetadata(SectionStreamKResult &result)
{
    if (metadata_ == nullptr || metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }

    optiling::detail::SliMetadata sliMetadata(metadata_->GetData());
    for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
        sliMetadata.SetFaMetadata(i, optiling::SLI_CORE_ENABLE_INDEX, 0U);
        sliMetadata.SetFaMetadata(i, optiling::SLI_BN2_START_INDEX, 0U);
        sliMetadata.SetFaMetadata(i, optiling::SLI_M_START_INDEX, 0U);
        sliMetadata.SetFaMetadata(i, optiling::SLI_S2_START_INDEX, 0U);
        sliMetadata.SetFaMetadata(i, optiling::SLI_BN2_END_INDEX, 0U);
        sliMetadata.SetFaMetadata(i, optiling::SLI_M_END_INDEX, 0U);
        sliMetadata.SetFaMetadata(i, optiling::SLI_S2_END_INDEX, 0U);
        sliMetadata.SetFaMetadata(i, optiling::SLI_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
    }
    for (uint32_t i = 0; i < AIV_CORE_NUM; ++i) {
        sliMetadata.SetFdMetadata(i, optiling::SLD_CORE_ENABLE_INDEX, 0U);
        sliMetadata.SetFdMetadata(i, optiling::SLD_BN2_IDX_INDEX, 0U);
        sliMetadata.SetFdMetadata(i, optiling::SLD_M_IDX_INDEX, 0U);
        sliMetadata.SetFdMetadata(i, optiling::SLD_WORKSPACE_IDX_INDEX, 0U);
        sliMetadata.SetFdMetadata(i, optiling::SLD_WORKSPACE_NUM_INDEX, 0U);
        sliMetadata.SetFdMetadata(i, optiling::SLD_M_START_INDEX, 0U);
        sliMetadata.SetFdMetadata(i, optiling::SLD_M_NUM_INDEX, 0U);
    }

    // FA Metadata Generate
    auto faResult = result.sectionFaResult[0];

    for (uint32_t i = 0; i < faResult.usedCoreNum; ++i) {
        sliMetadata.SetFaMetadata(i, optiling::SLI_CORE_ENABLE_INDEX, 1);
        // FA start
        if (i > 0) {
            sliMetadata.SetFaMetadata(i, optiling::SLI_BN2_START_INDEX, faResult.bN2End[i - 1]);
            sliMetadata.SetFaMetadata(i, optiling::SLI_M_START_INDEX, faResult.gS1End[i - 1]);
            sliMetadata.SetFaMetadata(i, optiling::SLI_S2_START_INDEX, faResult.s2End[i - 1]);
        }
        // FA end
        sliMetadata.SetFaMetadata(i, optiling::SLI_BN2_END_INDEX, faResult.bN2End[i]);
        sliMetadata.SetFaMetadata(i, optiling::SLI_M_END_INDEX, faResult.gS1End[i]);
        sliMetadata.SetFaMetadata(i, optiling::SLI_S2_END_INDEX, faResult.s2End[i]);
        // FA idx
        sliMetadata.SetFaMetadata(i, optiling::SLI_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                 faResult.firstFdDataWorkspaceIdx[i]);
    }
    KERNEL_LOG_ERROR("Assign FD");
    // FD Metadata Generate
    auto fdResult = result.sectionFdResult[0];
    for (uint32_t i = 0; i < fdResult.usedVecNum; ++i) {
        uint32_t curTaskIdx = fdResult.taskIdx[i];
        sliMetadata.SetFdMetadata(i, optiling::SLD_CORE_ENABLE_INDEX, 1);
        sliMetadata.SetFdMetadata(i, optiling::SLD_BN2_IDX_INDEX, fdResult.bN2Idx[curTaskIdx]);
        sliMetadata.SetFdMetadata(i, optiling::SLD_M_IDX_INDEX, fdResult.gS1Idx[curTaskIdx]);
        sliMetadata.SetFdMetadata(i, optiling::SLD_WORKSPACE_IDX_INDEX, fdResult.workspaceIdx[curTaskIdx]);
        sliMetadata.SetFdMetadata(i, optiling::SLD_WORKSPACE_NUM_INDEX, fdResult.s2SplitNum[curTaskIdx]);
        sliMetadata.SetFdMetadata(i, optiling::SLD_M_START_INDEX, fdResult.mStart[i]);
        sliMetadata.SetFdMetadata(i, optiling::SLD_M_NUM_INDEX, fdResult.mLen[i]);
    }
    return true;
}

namespace {
static const char *kernelType = "StemIndexerMetadata";
REGISTER_CPU_KERNEL(kernelType, StemIndexerMetadataCpuKernel);
} // namespace

} // namespace aicpu
