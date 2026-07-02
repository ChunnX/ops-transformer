/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_metadata_aicpu_arch35.h
 * \brief arch35 (Ascend950) specific implementations for AICPU metadata kernel.
 */
#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_METADATA_AICPU_ARCH35_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_METADATA_AICPU_ARCH35_H

#include "../dense_lightning_indexer_softmax_lse_metadata_aicpu.h"
#include "../../../dense_lightning_indexer_softmax_lse/op_kernel/dense_lightning_indexer_softmax_lse_metadata.h"

#include <algorithm>

using namespace optiling;

namespace aicpu {

template <typename T>
inline T CeilDiv(T x, T y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

inline int64_t DenseLightningIndexerSoftmaxLseMetadataCpuKernel::CalcTotalSize() const
{
    if (IsTensorValid(seqUsedQuery_)) {
        auto *data = reinterpret_cast<const int32_t *>(seqUsedQuery_->GetData());
        int64_t total = 0;
        for (int64_t i = 0; i < bSize_; ++i) {
            total += static_cast<int64_t>(data[i]);
        }
        return total;
    }
    if (layoutType_ == DliLayout::TND && IsTensorValid(cuSeqLensQuery_)) {
        auto *data = reinterpret_cast<const int32_t *>(cuSeqLensQuery_->GetData());
        return static_cast<int64_t>(data[bSize_]);
    }
    return bSize_ * s1Size_;
}

inline bool DenseLightningIndexerSoftmaxLseMetadataCpuKernel::BuildMetadata()
{
    int64_t totalNum = CalcTotalSize();
    if (totalNum <= 0) {
        KERNEL_LOG_ERROR("totalNum should be larger than 0, but got %ld", totalNum);
        return false;
    }

    int64_t blockSize = CeilDiv(totalNum, static_cast<int64_t>(aicCoreNum_));
    int64_t actualCoreNum = CeilDiv(totalNum, blockSize);

    auto *metadataData = reinterpret_cast<DLI_METADATA_T *>(metadata_->GetData());
    std::fill_n(metadataData, DLI_METADATA_SIZE, static_cast<DLI_METADATA_T>(0));
    auto *metadata = reinterpret_cast<detail::DenseLISoftmaxLseMetaData *>(metadataData);
    metadata->totalNum = static_cast<int32_t>(totalNum);
    metadata->blockSize = static_cast<int32_t>(blockSize);
    metadata->actualCoreNum = static_cast<int32_t>(actualCoreNum);
    return true;
}

} // namespace aicpu

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_METADATA_AICPU_ARCH35_H
