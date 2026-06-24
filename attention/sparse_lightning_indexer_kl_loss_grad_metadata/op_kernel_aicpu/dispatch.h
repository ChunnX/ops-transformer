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
 * \file dispatch.h
 * \brief Runtime dispatch for sparse_lightning_indexer_kl_loss_grad_metadata AICPU kernel
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_DISPATCH_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_DISPATCH_H

#include "sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h"
#include "arch22/sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch22.h"
#include "arch35/sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch35.h"

namespace aicpu {

class SparseLightningIndexerKLLossGradMetadataCpuKernel : public CpuKernel {
public:
    SparseLightningIndexerKLLossGradMetadataCpuKernel() = default;
    ~SparseLightningIndexerKLLossGradMetadataCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override
    {
        std::string socVersion;
        auto attr = ctx.GetAttr("soc_version");
        if (attr != nullptr) {
            socVersion = attr->GetString();
        }
        if (socVersion.find("Ascend950") != std::string::npos) {
            SparseLightningIndexerKLLossGradMetadataCpuKernelArch35 kernel;
            return kernel.Compute(ctx);
        }
        SparseLightningIndexerKLLossGradMetadataCpuKernelArch22 kernel;
        return kernel.Compute(ctx);
    }
};

} // namespace aicpu

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_DISPATCH_H
