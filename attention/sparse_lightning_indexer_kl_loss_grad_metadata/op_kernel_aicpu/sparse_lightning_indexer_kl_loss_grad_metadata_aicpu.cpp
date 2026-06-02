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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.cpp
 * \brief
 */

#include <algorithm>
#include <numeric>
#include "log.h"
#include "status.h"
#include "cust_op/cust_cpu_utils.h"
#include "../../sparse_lightning_indexer_kl_loss_grad/op_kernel/sparse_lightning_indexer_kl_loss_grad_metadata.h"
#include "sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
namespace {
template <typename T>
T CeilDiv(T x, T y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

template <typename T>
T AlignUp(T x, T y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y * y;
}

bool IsTensorValid(Tensor *tensor)
{
    return tensor != nullptr && tensor->GetData() != nullptr && tensor->GetTensorShape() != nullptr;
}
} // namespace

uint32_t SparseLightningIndexerKLLossGradMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    if (!Prepare(ctx)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return BuildMetadata() ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool SparseLightningIndexerKLLossGradMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    cuSeqLensQuery_ = ctx.Input(0);
    cuSeqLensKey_ = ctx.Input(1);
    seqUsedQuery_ = ctx.Input(2);
    seqUsedKey_ = ctx.Input(3);
    cmpResidualKey_ = ctx.Input(4);
    metadata_ = ctx.Output(0);

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                         GetAttrValue(ctx, "num_heads_k", numHeadsK_) &&
                         GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    GetAttrValueOpt(ctx, "batch_size", bSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", s1Size_);
    GetAttrValueOpt(ctx, "max_seqlen_k", s2Size_);
    GetAttrValueOpt(ctx, "topk", kSize_);
    GetAttrValueOpt(ctx, "layout_q", layout_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "mask_mode", sparseMode_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);

    if (layout_ == "BSND") {
        layoutType_ = SliLayout::BSND;
    } else if (layout_ == "TND") {
        layoutType_ = SliLayout::TND;
    } else {
        KERNEL_LOG_ERROR("layout_q must be BSND or TND, but got %s", layout_.c_str());
        return false;
    }

    if (bSize_ <= 0) {
        if (IsTensorValid(seqUsedQuery_)) {
            bSize_ = seqUsedQuery_->GetTensorShape()->GetDimSize(0);
        } else if (layoutType_ == SliLayout::TND && IsTensorValid(cuSeqLensQuery_)) {
            bSize_ = cuSeqLensQuery_->GetTensorShape()->GetDimSize(0) - 1;
        }
    }

    return ParamsCheck();
}

bool SparseLightningIndexerKLLossGradMetadataCpuKernel::ParamsCheck()
{
    KERNEL_CHECK_NULLPTR(metadata_, false, "metadata output is null");
    KERNEL_CHECK_NULLPTR(metadata_->GetData(), false, "metadata data is null");
    KERNEL_CHECK_NULLPTR(metadata_->GetTensorShape(), false, "metadata shape is null");

    if (aicCoreNum_ <= 0) {
        KERNEL_LOG_ERROR("aic_core_num must be positive, but got %ld", aicCoreNum_);
        return false;
    }
    if (numHeadsQ_ <= 0 || numHeadsK_ <= 0 || headDim_ <= 0) {
        KERNEL_LOG_ERROR("num_heads_q/num_heads_k/head_dim must be positive, but got nq=%ld nk=%ld d=%ld",
                         numHeadsQ_, numHeadsK_, headDim_);
        return false;
    }
    if (bSize_ <= 0 || kSize_ <= 0) {
        KERNEL_LOG_ERROR("batch_size/topk must be positive, but got batch_size=%ld topk=%ld", bSize_, kSize_);
        return false;
    }
    if (layoutType_ == SliLayout::BSND && (s1Size_ <= 0 || s2Size_ <= 0)) {
        KERNEL_LOG_ERROR("max_seqlen_q/max_seqlen_k must be positive for BSND, but got q=%ld k=%ld",
                         s1Size_, s2Size_);
        return false;
    }
    if (s1Size_ < 0 || s2Size_ < 0) {
        KERNEL_LOG_ERROR("max_seqlen_q/max_seqlen_k must be non-negative, but got q=%ld k=%ld",
                         s1Size_, s2Size_);
        return false;
    }
    if (numHeadsQ_ % numHeadsK_ != 0) {
        KERNEL_LOG_ERROR("num_heads_q must be divisible by num_heads_k, but got nq=%ld nk=%ld",
                         numHeadsQ_, numHeadsK_);
        return false;
    }
    if (sparseMode_ != static_cast<int64_t>(SliSparseMode::NO_MASK) &&
        sparseMode_ != static_cast<int64_t>(SliSparseMode::RIGHT_DOWN_CAUSAL)) {
        KERNEL_LOG_ERROR("mask_mode only supports 0 or 3, but got %ld", sparseMode_);
        return false;
    }
    if (cmpRatio_ < 1 || cmpRatio_ > 128) {
        KERNEL_LOG_ERROR("cmp_ratio must be in [1, 128], but got %ld", cmpRatio_);
        return false;
    }
    if (layoutType_ == SliLayout::TND && (!IsTensorValid(cuSeqLensQuery_) || !IsTensorValid(cuSeqLensKey_))) {
        KERNEL_LOG_ERROR("cu_seqlens_q/cu_seqlens_k must be provided for TND metadata.");
        return false;
    }
    if (layoutType_ == SliLayout::TND &&
        (cuSeqLensQuery_->GetTensorShape()->GetDimSize(0) < bSize_ + 1 ||
         cuSeqLensKey_->GetTensorShape()->GetDimSize(0) < bSize_ + 1)) {
        KERNEL_LOG_ERROR("cu_seqlens_q/cu_seqlens_k length must be at least batch_size + 1.");
        return false;
    }
    return true;
}

int64_t SparseLightningIndexerKLLossGradMetadataCpuKernel::GetActualSeqLen(Tensor *tensor, int64_t bIdx) const
{
    if (!IsTensorValid(tensor)) {
        return 0;
    }
    auto *data = reinterpret_cast<const int32_t *>(tensor->GetData());
    return static_cast<int64_t>(data[bIdx + 1]) - static_cast<int64_t>(data[bIdx]);
}

int64_t SparseLightningIndexerKLLossGradMetadataCpuKernel::CalcTotalSize() const
{
    if (layoutType_ == SliLayout::TND) {
        auto *data = reinterpret_cast<const int32_t *>(cuSeqLensQuery_->GetData());
        return data[bSize_];
    }
    return bSize_ * s1Size_;
}

int64_t SparseLightningIndexerKLLossGradMetadataCpuKernel::GetS2RealSize(int64_t s1Size, int64_t s2Size,
                                                                         int64_t s1Idx) const
{
    int64_t s2RealSize = 0;
    if (sparseMode_ == static_cast<int64_t>(SliSparseMode::NO_MASK)) {
        s2RealSize = s2Size;
    } else if (sparseMode_ == static_cast<int64_t>(SliSparseMode::RIGHT_DOWN_CAUSAL)) {
        s2RealSize = (s2Size * cmpRatio_ - s1Size + s1Idx + 1) / cmpRatio_;
        s2RealSize = std::max<int64_t>(s2RealSize, 1);
    }
    s2RealSize = AlignUp(s2RealSize, 512L);
    return std::min(s2RealSize, kSize_);
}

bool SparseLightningIndexerKLLossGradMetadataCpuKernel::BuildSparseValidArray(std::vector<int64_t> &sparseValidArray)
{
    if (sparseValidArray.empty()) {
        KERNEL_LOG_ERROR("Sparse valid array size should be larger than 0.");
        return false;
    }
    if (layoutType_ == SliLayout::TND) {
        int64_t accumS1 = 0;
        for (int64_t bIdx = 0; bIdx < bSize_; ++bIdx) {
            int64_t actualS1 = GetActualSeqLen(cuSeqLensQuery_, bIdx);
            int64_t actualS2 = GetActualSeqLen(cuSeqLensKey_, bIdx);
            for (int64_t s1Idx = 0; s1Idx < actualS1 && accumS1 < static_cast<int64_t>(sparseValidArray.size());
                 ++s1Idx, ++accumS1) {
                sparseValidArray[accumS1] = GetS2RealSize(actualS1, actualS2, s1Idx);
            }
        }
    } else {
        int64_t accum = 0;
        for (int64_t bIdx = 0; bIdx < bSize_; ++bIdx) {
            for (int64_t s1Idx = 0; s1Idx < s1Size_; ++s1Idx, ++accum) {
                sparseValidArray[accum] = GetS2RealSize(s1Size_, s2Size_, s1Idx);
            }
        }
    }
    return true;
}

bool SparseLightningIndexerKLLossGradMetadataCpuKernel::InitLoadValue(const std::vector<int64_t> &sparseValidArray,
                                                                      const std::vector<int64_t> &sparseStartIdx,
                                                                      std::vector<int64_t> &localValue) const
{
    for (int64_t idx = 0; idx < coreNum_; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < coreNum_) ? sparseStartIdx[idx + 1] : totalSize_;
        if (start < totalSize_) {
            localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
        }
    }
    return true;
}

bool SparseLightningIndexerKLLossGradMetadataCpuKernel::BalanceLoad(const std::vector<int64_t> &sparseValidArray,
                                                                    std::vector<int64_t> &localValue,
                                                                    std::vector<int64_t> &sparseStartIdx) const
{
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());

    for (int64_t idx = 1; idx < coreNum_; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize_ && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize_) {
            break;
        }
    }
    int64_t tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    for (int64_t idx = coreNum_ - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize_) {
            if (sparseStartIdx[idx - 1] == totalSize_) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());
    return tmpMaxVal < maxVal;
}

bool SparseLightningIndexerKLLossGradMetadataCpuKernel::Balance4DLoad(std::vector<int64_t> &sparseStartIdx,
                                                                      const std::vector<int64_t> &sparseValidArray,
                                                                      int64_t balanceNum) const
{
    int64_t tmpIndex = 0;
    sparseStartIdx[tmpIndex] = 0;
    int64_t sumTmpArray = 0;
    int64_t sumTmpArrayLast = 0;
    for (int64_t idx = 0; idx < static_cast<int64_t>(sparseValidArray.size()); ++idx) {
        if (tmpIndex == coreNum_ - 1) {
            break;
        }

        sumTmpArrayLast = sumTmpArray;
        sumTmpArray += sparseValidArray[idx];
        if (sumTmpArray == balanceNum) {
            tmpIndex = std::min(tmpIndex + 1, coreNum_ - 1);
            sparseStartIdx[tmpIndex] = idx + 1;
            sumTmpArray = 0;
        } else if (sumTmpArray > balanceNum) {
            tmpIndex = std::min(tmpIndex + 1, coreNum_ - 1);
            if (balanceNum - sumTmpArrayLast >= sumTmpArray - balanceNum) {
                sparseStartIdx[tmpIndex] = idx + 1;
            } else {
                sparseStartIdx[tmpIndex] = idx;
                idx--;
            }
            sumTmpArray = 0;
        }
    }
    return true;
}

bool SparseLightningIndexerKLLossGradMetadataCpuKernel::SetSparseStartIdx(
    const std::vector<int64_t> &sparseValidArray)
{
    bS1Index_.assign(SLI_METADATA_MAX_CORE_NUM, totalSize_);
    if (layoutType_ == SliLayout::TND) {
        std::vector<int64_t> localSparseStartIdx(SLI_METADATA_MAX_CORE_NUM, totalSize_);
        for (int64_t idx = 0; idx < coreNum_; ++idx) {
            localSparseStartIdx[idx] = std::min(idx * splitFactorSize_, totalSize_);
        }

        std::vector<int64_t> localValue(coreNum_, 0);
        InitLoadValue(sparseValidArray, localSparseStartIdx, localValue);

        std::vector<int64_t> tmpLocalValue(coreNum_, 0);
        std::vector<int64_t> tmpSparseStartIdx(SLI_METADATA_MAX_CORE_NUM, totalSize_);
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
        int64_t avgVal = CeilDiv(sparseArraySum, coreNum_);

        tmpSparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < coreNum_; ++idx) {
            int64_t start = tmpSparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpSparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpSparseStartIdx[idx] < totalSize_) {
                singleLoadValue += sparseValidArray[tmpSparseStartIdx[idx]];
                tmpSparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpSparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpSparseStartIdx[idx] - 1];
                if ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) {
                    tmpSparseStartIdx[idx] -= 1;
                    singleLoadValue = redoSingleLoadValue;
                }
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDiv(sparseArraySum, coreNum_ - idx);
            }
        }

        InitLoadValue(sparseValidArray, tmpSparseStartIdx, tmpLocalValue);
        while (BalanceLoad(sparseValidArray, tmpLocalValue, tmpSparseStartIdx)) {
        }

        if ((*std::max_element(localValue.begin(), localValue.end())) >
            (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
            localSparseStartIdx.swap(tmpSparseStartIdx);
        }
        bS1Index_.swap(localSparseStartIdx);
    } else {
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
        int64_t balanceNum = CeilDiv(sparseArraySum, coreNum_);
        Balance4DLoad(bS1Index_, sparseValidArray, balanceNum);
    }

    for (int64_t idx = 1; idx < static_cast<int64_t>(SLI_METADATA_MAX_CORE_NUM); ++idx) {
        if (bS1Index_[idx] == 0) {
            bS1Index_[idx] = totalSize_;
        }
    }
    return true;
}

bool SparseLightningIndexerKLLossGradMetadataCpuKernel::BuildMetadata()
{
    totalSize_ = CalcTotalSize();
    if (totalSize_ <= 0) {
        KERNEL_LOG_ERROR("totalSize should be larger than 0, but got %ld", totalSize_);
        return false;
    }

    coreNum_ = std::min<int64_t>(std::min(totalSize_, aicCoreNum_), SLI_METADATA_MAX_CORE_NUM);
    splitFactorSize_ = CeilDiv(totalSize_, coreNum_);

    std::vector<int64_t> sparseValidArray(totalSize_, 0);
    if (!BuildSparseValidArray(sparseValidArray) || !SetSparseStartIdx(sparseValidArray)) {
        return false;
    }

    auto *metadataData = reinterpret_cast<SLI_METADATA_T *>(metadata_->GetData());
    std::fill_n(metadataData, SLI_METADATA_SIZE, static_cast<SLI_METADATA_T>(0));
    auto *metadata = reinterpret_cast<detail::SliGradKLLossMetaData *>(metadataData);
    metadata->coreNum = static_cast<int32_t>(coreNum_);
    metadata->totalSize = static_cast<int32_t>(totalSize_);
    metadata->splitFactorSize = static_cast<int32_t>(splitFactorSize_);
    for (uint32_t i = 0; i < SLI_METADATA_HEADER_SIZE - 3; ++i) {
        metadata->reserved[i] = 0;
    }
    for (uint32_t i = 0; i < SLI_METADATA_MAX_CORE_NUM; ++i) {
        metadata->bS1Index[i] = static_cast<int32_t>(bS1Index_[i]);
    }
    return true;
}

static const char *kernelType = "SparseLightningIndexerKLLossGradMetadata";
REGISTER_CPU_KERNEL(kernelType, SparseLightningIndexerKLLossGradMetadataCpuKernel);
} // namespace aicpu
