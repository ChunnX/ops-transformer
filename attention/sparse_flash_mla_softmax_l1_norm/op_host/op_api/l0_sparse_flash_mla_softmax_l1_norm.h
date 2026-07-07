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
 * \file l0_sparse_flash_mla_softmax_l1_norm.h
 * \brief
 */
#ifndef L0_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_H
#define L0_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* SparseFlashMlaSoftmaxL1Norm(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *lse,
    const aclTensor *sparseIndicesOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *topkLengthOptional,
    const aclTensor *metadataOptional,
    double scaleValue,
    int64_t cmpRatio,
    int64_t maskMode,
    const char *layoutQ,
    const char *layoutK,
    const aclTensor *softmaxL1Norm,
    aclOpExecutor *executor);
} // namespace l0op

#endif // L0_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_H
