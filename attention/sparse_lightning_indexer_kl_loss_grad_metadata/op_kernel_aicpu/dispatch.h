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
 * \brief Arch dispatch for sparse_lightning_indexer_kl_loss_grad_metadata AICPU kernel
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_DISPATCH_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_DISPATCH_H

#include "sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h"

#if defined(SLI_AICPU_ARCH22)
#include "arch22/sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch22.h"
#endif

#if defined(SLI_AICPU_ARCH35)
#include "arch35/sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch35.h"
#endif

namespace aicpu {

#if defined(SLI_AICPU_ARCH22) && defined(SLI_AICPU_ARCH35)

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

#elif defined(SLI_AICPU_ARCH22)

using SparseLightningIndexerKLLossGradMetadataCpuKernel = SparseLightningIndexerKLLossGradMetadataCpuKernelArch22;

#elif defined(SLI_AICPU_ARCH35)

using SparseLightningIndexerKLLossGradMetadataCpuKernel = SparseLightningIndexerKLLossGradMetadataCpuKernelArch35;

#else
#error "No SLI_AICPU_ARCH* defined. Check ARCH_DIRECTORY in CMake configuration."
#endif

} // namespace aicpu

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_DISPATCH_H
