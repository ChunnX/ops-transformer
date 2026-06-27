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
 * \brief Registration entry for AICPU kernel with runtime arch dispatch
 */

#include "dispatch.h"

namespace aicpu {
static const char *kernelType = "SparseLightningIndexerKLLossGradMetadata";
REGISTER_CPU_KERNEL(kernelType, SparseLightningIndexerKLLossGradMetadataCpuKernel);
} // namespace aicpu
