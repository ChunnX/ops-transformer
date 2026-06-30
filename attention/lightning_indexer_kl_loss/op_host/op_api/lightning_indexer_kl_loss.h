/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_kl_loss.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_OP_LIGHTNING_INDEXER_KL_LOSS_OP_H
#define OP_API_INC_LEVEL0_OP_LIGHTNING_INDEXER_KL_LOSS_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const std::size_t LIGHTNING_INDEXER_KL_LOSS_OUTPUT_CNT = 1;

const std::array<const aclTensor *, LIGHTNING_INDEXER_KL_LOSS_OUTPUT_CNT> LightningIndexerKLLoss(
    const aclTensor *target_score, const aclTensor *index_probs,
    float eps, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_LIGHTNING_INDEXER_KL_LOSS_OP_H
