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
 * \file aclnn_quant_all_reduce.cpp
 * \brief
 */
#include "aclnn_quant_all_reduce.h"
#include "securec.h"
#include "acl/acl.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "common/utils/hccl_util.h"
#include "aclnn_kernels/transdata.h"

namespace {

using namespace op;
using namespace l0op;

extern "C" aclnnStatus aclnnQuantAllReduceGetWorkspaceSize(const aclTensor* x, const aclTensor* scales,
                                                           const char* group, const char* reduceOp,
                                                           aclTensor* output, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor)
{
    aclnnStatus retParam = QuantAllReduceCheckParams(x, scales, group, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    int64_t worldSize = -1;
    aclnnStatus ret = aclnnInnerQuantAllReduceGetWorkspaceSize(x, scales, const_cast<char*>(group),
        const_cast<char*>(reduceOp), yDtype, worldSize, output, workspaceSize, executor);
    OP_LOGD("QuantAllReduce, aclnnGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnQuantAllReduce(void* workspace, uint64_t workspaceSize,
                                           aclOpExecutor* executor, const aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerQuantAllReduce(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "QuantAllReduce, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

} // namespace