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
 * \file kv_compress_epilog.cpp
 * \brief L0 op API implementation for KvCompressEpilog
 */

#include "kv_compress_epilog.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(KvCompressEpilog);

const aclTensor *KvCompressEpilog(const aclTensor *kvCompressCache, const aclTensor *x, const aclTensor *slotMapping,
                                  int64_t quantGroupSize, int64_t quantMode, int64_t roundScale, float scaleAttr,
                                  int64_t padCount, aclOpExecutor *executor)
{
    L0_DFX(KvCompressEpilog, kvCompressCache, x, slotMapping, quantGroupSize, quantMode, roundScale, scaleAttr,
           padCount);

    auto output = executor->AllocTensor(kvCompressCache->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(KvCompressEpilog, OP_INPUT(kvCompressCache, x, slotMapping), OP_OUTPUT(output),
                           OP_ATTR(quantGroupSize, quantMode, roundScale, scaleAttr, padCount));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(KvCompressEpilog, OP_INPUT(kvCompressCache, x, slotMapping), OP_OUTPUT(output),
                                      OP_ATTR(quantGroupSize, quantMode, roundScale, scaleAttr, padCount));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return output;
}
} // namespace l0op
