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
 * \file aclnn_kv_compress_epilog.cpp
 * \brief L2 ACLNN API implementation for KvCompressEpilog
 */

#include "aclnn_kv_compress_epilog.h"

#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "kv_compress_epilog.h"
#include "aclnn_kernels/contiguous.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const uint64_t DIM_NUM_2 = 2;

static aclnnStatus CheckKvCompressEpilogParam(const aclTensor *kvCompressCache, const aclTensor *x,
                                               const aclTensor *slotMapping, const aclTensor *output,
                                               const uint64_t *workspaceSize, aclOpExecutor ** const executor)
{
    CHECK_RET(kvCompressCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(x != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(output != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const aclTensor *kvCompressCache, const aclTensor *x, const aclTensor *slotMapping)
{
    CHECK_RET(kvCompressCache->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
              "Format of kvCompressCache should be ND, current format is invalid.");
    CHECK_RET(x->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
              "Format of x should be ND, current format is invalid.");
    CHECK_RET(slotMapping->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
              "Format of slotMapping should be ND, current format is invalid.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const aclTensor *x, const aclTensor *slotMapping)
{
    auto xDtype = x->GetDataType();
    CHECK_RET(xDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
              "x dtype only support BF16, actual dtype is %s.", op::ToString(DataType(xDtype)).GetString());

    auto slotMappingDtype = slotMapping->GetDataType();
    CHECK_RET(slotMappingDtype == DataType::DT_INT32 || slotMappingDtype == DataType::DT_INT64,
              ACLNN_ERR_PARAM_INVALID,
              "slotMapping dtype only support INT32/INT64, actual dtype is %s.",
              op::ToString(DataType(slotMappingDtype)).GetString());

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const aclTensor *x, const aclTensor *slotMapping)
{
    auto xDimNum = x->GetViewShape().GetDimNum();
    CHECK_RET(xDimNum == DIM_NUM_2, ACLNN_ERR_PARAM_INVALID,
              "The dimension of x should be 2, but actual is %zu.", xDimNum);

    auto slotMappingDimNum = slotMapping->GetViewShape().GetDimNum();
    CHECK_RET(slotMappingDimNum == 1, ACLNN_ERR_PARAM_INVALID,
              "The dimension of slotMapping should be 1, but actual is %zu.", slotMappingDimNum);

    auto bs = x->GetViewShape().GetDim(0);
    auto slotMappingSize = slotMapping->GetViewShape().GetDim(0);
    CHECK_RET(slotMappingSize == bs, ACLNN_ERR_PARAM_INVALID,
              "slotMapping size must equal x first dimension, got slotMappingSize=%ld, bs=%ld.",
              slotMappingSize, bs);

    auto d = x->GetViewShape().GetDim(1);
    CHECK_RET(d > 0, ACLNN_ERR_PARAM_INVALID, "The second dimension of x must be positive, got %ld.", d);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor *kvCompressCache, const aclTensor *x, const aclTensor *slotMapping,
                               const aclTensor *output, const uint64_t *workspaceSize, aclOpExecutor ** const executor)
{
    CHECK_RET(CheckKvCompressEpilogParam(kvCompressCache, x, slotMapping, output, workspaceSize, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(kvCompressCache, x, slotMapping) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(x, slotMapping) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, slotMapping) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&kvCompressCache, const aclTensor *&x, const aclTensor *&slotMapping,
                                  aclOpExecutor *executor)
{
    kvCompressCache = l0op::Contiguous(kvCompressCache, executor);
    CHECK_RET(kvCompressCache != nullptr, ACLNN_ERR_INNER_NULLPTR);

    x = l0op::Contiguous(x, executor);
    CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);

    slotMapping = l0op::Contiguous(slotMapping, executor);
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(const aclTensor *kvCompressCache, const aclTensor *x,
                                                   const aclTensor *slotMapping, int64_t quantGroupSize,
                                                   int64_t quantMode, int64_t roundScale, float scaleAttr,
                                                   int64_t padCount, aclTensor *output, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnKvCompressEpilog,
                   DFX_IN(kvCompressCache, x, slotMapping, quantGroupSize, quantMode, roundScale, scaleAttr, padCount),
                   DFX_OUT(output));

    CHECK_RET(CheckParams(kvCompressCache, x, slotMapping, output, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(DataContiguous(kvCompressCache, x, slotMapping, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0Result = l0op::KvCompressEpilog(kvCompressCache, x, slotMapping, quantGroupSize, quantMode, roundScale,
                                           scaleAttr, padCount, l0Executor);
    CHECK_RET(l0Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(l0Result, output, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvCompressEpilog(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKvCompressEpilog);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in KvCompressEpilog launch aicore.");
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
}
#endif
