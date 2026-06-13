/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_cast.h"

#include <initializer_list>
#include <utility>

#include "cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
using DTypePair = std::pair<DataType, DataType>;

static const std::initializer_list<DTypePair> CAST_DTYPE_PAIR_SUPPORT_LIST = {
    {DataType::DT_FLOAT16, DataType::DT_FLOAT}, {DataType::DT_FLOAT16, DataType::DT_INT32},
    {DataType::DT_FLOAT16, DataType::DT_INT8}, {DataType::DT_FLOAT16, DataType::DT_UINT8},
    {DataType::DT_FLOAT16, DataType::DT_INT16}, {DataType::DT_FLOAT16, DataType::DT_BOOL},
    {DataType::DT_FLOAT, DataType::DT_FLOAT16}, {DataType::DT_FLOAT, DataType::DT_INT32},
    {DataType::DT_FLOAT, DataType::DT_INT16}, {DataType::DT_FLOAT, DataType::DT_INT8},
    {DataType::DT_FLOAT, DataType::DT_UINT8}, {DataType::DT_FLOAT, DataType::DT_BOOL},
    {DataType::DT_INT32, DataType::DT_FLOAT}, {DataType::DT_INT32, DataType::DT_FLOAT16},
    {DataType::DT_INT32, DataType::DT_INT16}, {DataType::DT_INT32, DataType::DT_INT8},
    {DataType::DT_INT32, DataType::DT_UINT8}, {DataType::DT_INT32, DataType::DT_BOOL},
    {DataType::DT_INT8, DataType::DT_FLOAT16}, {DataType::DT_INT8, DataType::DT_FLOAT},
    {DataType::DT_INT8, DataType::DT_INT32}, {DataType::DT_INT8, DataType::DT_UINT8},
    {DataType::DT_UINT8, DataType::DT_FLOAT16}, {DataType::DT_UINT8, DataType::DT_FLOAT},
    {DataType::DT_UINT8, DataType::DT_INT32}, {DataType::DT_UINT8, DataType::DT_INT8},
    {DataType::DT_INT16, DataType::DT_FLOAT16}, {DataType::DT_INT16, DataType::DT_FLOAT},
    {DataType::DT_BOOL, DataType::DT_FLOAT16}, {DataType::DT_BOOL, DataType::DT_FLOAT},
    {DataType::DT_BOOL, DataType::DT_INT32}, {DataType::DT_BOOL, DataType::DT_UINT8},
    {DataType::DT_INT64, DataType::DT_INT32}, {DataType::DT_INT64, DataType::DT_FLOAT},
    {DataType::DT_INT64, DataType::DT_FLOAT16}, {DataType::DT_INT32, DataType::DT_INT64},
    {DataType::DT_FLOAT, DataType::DT_INT64},
    {DataType::DT_BF16, DataType::DT_FLOAT}, {DataType::DT_BF16, DataType::DT_FLOAT16},
    {DataType::DT_BF16, DataType::DT_INT32}, {DataType::DT_BF16, DataType::DT_INT8},
    {DataType::DT_BF16, DataType::DT_UINT8}, {DataType::DT_BF16, DataType::DT_BOOL},
    {DataType::DT_BF16, DataType::DT_INT16}, {DataType::DT_BF16, DataType::DT_INT64}
};

bool IsCastDtypePairSupported(DataType srcType, DataType dstType)
{
    for (const auto &pair : CAST_DTYPE_PAIR_SUPPORT_LIST) {
        if (pair.first == srcType && pair.second == dstType) {
            return true;
        }
    }
    return false;
}

aclnnStatus CheckNullptr(const aclTensor *self, const aclTensor *out, const uint64_t *workspaceSize,
                         aclOpExecutor **executor)
{
    CHECK_RET(self != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(out != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtypeValid(const aclTensor *self, aclDataType dstType, const aclTensor *out)
{
    auto srcOpType = self->GetDataType();
    auto dstOpType = op::ToOpDataType(dstType);
    auto outOpType = out->GetDataType();

    CHECK_RET(dstOpType != DataType::DT_UNDEFINED, ACLNN_ERR_PARAM_INVALID);
    if (dstOpType == DataType::DT_BF16 || outOpType == DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Cast only supports BF16 as input; BF16 output is not supported.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (outOpType != dstOpType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dstType[%s] and out dtype[%s] must be the same.",
                op::ToString(dstOpType).GetString(), op::ToString(outOpType).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsCastDtypePairSupported(srcOpType, dstOpType)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Unsupported Cast dtype pair: self[%s] to out[%s].",
                op::ToString(srcOpType).GetString(), op::ToString(dstOpType).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShape(const aclTensor *self, const aclTensor *out)
{
    auto selfShape = self->GetViewShape();
    auto outShape = out->GetViewShape();
    if (selfShape != outShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of self and out must be the same.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *self, aclDataType dstType, const aclTensor *out,
                        const uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckNullptr(self, out, workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, dstType, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor *self, aclDataType dstType, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnCast, DFX_IN(self, dstType), DFX_OUT(out));

    auto ret = CheckParams(self, dstType, out, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(selfContiguous, static_cast<int64_t>(dstType), out, uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCast);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
