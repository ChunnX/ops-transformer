/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stem_oam_prep_varlen_q.h"
#include "aclnn_stem_oam_prep_varlen_q.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_error_check.h"

namespace opapi {

aclnnStatus LaunchStemOamPrepVarlenQ(
    const aclTensor* q, const aclTensor* qSeqLens, const aclTensor* cuSeqLensQ,
    /* * optional */ const aclTensor* qScale, const aclTensor* kScale,
    int64_t stemBlockSize, int64_t stemStride,
    aclTensor* qFlat,
    aclOpExecutor* executor)
{
    return aclnnStemOamPrepVarlenQGetWorkspaceSize(
        q, qSeqLens, cuSeqLensQ, qScale, kScale,
        stemBlockSize, stemStride,
        qFlat,
        nullptr, &executor);
}

}  // namespace opapi