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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h
 * \brief
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {

enum class SliSparseMode : int32_t {
    NO_MASK = 0,
    RIGHT_DOWN_CAUSAL = 3,
};

enum class SliLayout {
    BSND = 0,
    TND = 1,
};

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, bool>::type GetAttrValue(CpuKernelContext &ctx,
                                                                               const std::string &name, T &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = static_cast<T>(attr->GetInt());
    return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name, std::string &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetString();
    return true;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, void>::type GetAttrValueOpt(CpuKernelContext &ctx,
                                                                                  const std::string &name, T &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = static_cast<T>(attr->GetInt());
    }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, std::string &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetString();
    }
}

class SparseLightningIndexerKLLossGradMetadataCpuKernel : public CpuKernel {
public:
    SparseLightningIndexerKLLossGradMetadataCpuKernel() = default;
    ~SparseLightningIndexerKLLossGradMetadataCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsCheck();
    bool BuildMetadata();
    bool BuildSparseValidArray(std::vector<int64_t> &sparseValidArray);
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray);
    bool CanSplitWithMaxLoad(const std::vector<int64_t> &sparseValidArray,
                             int64_t start,
                             int64_t partNum,
                             int64_t maxLoad) const;
    int64_t FindMinMaxLoad(const std::vector<int64_t> &sparseValidArray,
                           int64_t start,
                           int64_t partNum) const;
    bool BuildBalancedSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                     std::vector<int64_t> &sparseStartIdx) const;
    int64_t GetCmpResidualKey(int64_t bIdx) const;
    int64_t GetPreCompressS2Len(int64_t bIdx, int64_t s2Size) const;
    int64_t GetS2RealSize(int64_t bIdx, int64_t s1Size, int64_t s2Size, int64_t s1Idx) const;
    int64_t GetS1Load(int64_t s2RealSize) const;
    int64_t GetActualSeqLen(Tensor *tensor, int64_t bIdx) const;
    int64_t CalcTotalSize() const;

    Tensor *cuSeqLensQuery_ = nullptr;
    Tensor *cuSeqLensKey_ = nullptr;
    Tensor *seqUsedQuery_ = nullptr;
    Tensor *seqUsedKey_ = nullptr;
    Tensor *cmpResidualKey_ = nullptr;
    Tensor *metadata_ = nullptr;

    int64_t bSize_ = 0;
    int64_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t kSize_ = 0;
    int64_t numHeadsQ_ = 0;
    int64_t numHeadsK_ = 0;
    int64_t headDim_ = 0;
    int64_t sparseMode_ = 0;
    int64_t cmpRatio_ = 1;
    int64_t aicCoreNum_ = 25;
    std::string layout_ = "BSND";
    std::string layoutK_ = "BSND";
    SliLayout layoutType_ = SliLayout::BSND;

    int64_t coreNum_ = 0;
    int64_t totalSize_ = 0;
    int64_t splitFactorSize_ = 0;
    std::vector<int64_t> bS1Index_;
};

} // namespace aicpu

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H
