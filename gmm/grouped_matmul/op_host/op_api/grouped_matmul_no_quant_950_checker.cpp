/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_no_quant_950_checker.h"

using namespace gmm;

aclnnStatus AclnnGroupedMatmulNoQuant950Checker::CheckGroupedMatmulNoQuant950() const
{
    CHECK_COND(CheckEmptyTensor() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "GMM check empty tensor failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulNoQuant950Checker::CheckEmptyTensor() const
{
    for (size_t i = 0; i < gmmParams_.x->Size(); ++i) {
        auto shape = (*gmmParams_.x)[i]->GetViewShape();
        int64_t index = gmmParams_.groupType == SPLIT_K ? shape.GetDimNum() - 2 : shape.GetDimNum() - 1;
        CHECK_COND(shape.GetDim(index) != 0, ACLNN_ERR_PARAM_INVALID,
                   "GMM no quant do not support origin k equals 0.");
    }
    return ACLNN_SUCCESS;
}