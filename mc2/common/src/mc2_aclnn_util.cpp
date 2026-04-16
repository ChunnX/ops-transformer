/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_aclnn_util.h"
#include "common/op_host/op_api/matmul_util.h"

namespace MC2Aclnn {

static constexpr size_t MX_SCALE_MAX_DIM = 3;

bool IsNeedScaleTrans(const aclTensor *mxScaleTensor)
{
    if (mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        return Ops::Transformer::IsTransposeLastTwoDims(mxScaleTensor);
    }
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);     // 1:最后一个维度
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 2); // 2:倒数第二个维度
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3);  // 3:倒数第三个维度
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&   // 3:倒数第三个维度
        mxScaleTensor->GetViewStrides()[dimNum - 2] == lastDim * lastThirdDim &&  // 2:倒数第二个维度
        (lastSecondDim == 1 && lastThirdDim == 1)) {
        return true;
    }
    return false;
}
} // namespace MC2Aclnn
