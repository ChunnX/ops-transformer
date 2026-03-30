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
 * \file aclnn_mhc_pre_sinkhorn.cpp
 * \brief aclnn_mhc_pre_sinkhorn
 */

#include "aclnn_mhc_pre_sinkhorn.h"
#include "mhc_pre_sinkhorn.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
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

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t MIN_NUMITERS = 0;
static constexpr size_t MAX_NUMITERS = 100;
static constexpr size_t SUPPORT_DIM_NUM_3 = 3;
static constexpr size_t SUPPORT_DIM_NUM_4 = 4;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;


static bool CheckNotNull(const aclTensor *x, int64_t outFlag, const aclTensor *output, const aclTensor *normOut,
                         const aclTensor *sumOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(output, return false);
    if (outFlag) {
        OP_CHECK_NULL(normOut, return false);
        OP_CHECK_NULL(sumOut, return false);
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *x, int64_t outFlag, const aclTensor *output, const aclTensor *normOut,
                            const aclTensor *sumOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);
    if (outFlag) {
        OP_CHECK_DTYPE_NOT_SUPPORT(normOut, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(sumOut, DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckFormat(const aclTensor *x, int64_t outFlag, const aclTensor *output, const aclTensor *normOut,
                        const aclTensor *sumOut)
{
    if (x->GetStorageFormat() != output->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. x [%s], output [%s].",
                ToString(x->GetStorageFormat()).GetString(), ToString(output->GetStorageFormat()).GetString());
        return false;
    }
    if (outFlag) {
        if (x->GetStorageFormat() != normOut->GetStorageFormat()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. x [%s], normOut [%s].",
                    ToString(x->GetStorageFormat()).GetString(), ToString(normOut->GetStorageFormat()).GetString());
            return false;
        }
        if (x->GetStorageFormat() != sumOut->GetStorageFormat()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. x [%s], sumOut [%s].",
                    ToString(x->GetStorageFormat()).GetString(), ToString(sumOut->GetStorageFormat()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor *x, int64_t outFlag, const aclTensor *output, const aclTensor *normOut,
                       const aclTensor *sumOut, int64_t numIters)
{
    OP_CHECK_SHAPE_NOT_EQUAL(x, output, return false);

    if (numIters <= MIN_NUMITERS || numIters > MAX_NUMITERS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numIters value error, numIters must in 1 to 100, but got numIters = %ld .",
                numIters);
        return false;
    }

    if (outFlag != 0 && outFlag != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outFlag value error, outFlag must be 0 or 1, but got outFlag = %ld .",
                outFlag);
        return false;
    }

    auto xShape = x->GetViewShape();
    auto xDim = xShape.GetDimNum();
    if (xDim != SUPPORT_DIM_NUM_3 && xDim != SUPPORT_DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, input dim must be 3 or 4, but got Dim = %ld .", xDim);
        return false;
    }

    int64_t n0 = 0;
    int64_t n1 = 0;
    if (xDim == SUPPORT_DIM_NUM_3) {
        n0 = xShape.GetDim(DIM_ONE);
        n1 = xShape.GetDim(DIM_TWO);
    } else if (xDim == SUPPORT_DIM_NUM_4) {
        n0 = xShape.GetDim(DIM_TWO);
        n1 = xShape.GetDim(DIM_THREE);
    }
    if ((n0 != n1) || (n0 != N_VALID_4 && n0 != N_VALID_6 && n0 != N_VALID_8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "n0 must equal n1, and n must be %ld or %ld or %ld, but got n0 = %ld, n1 = %ld", N_VALID_4, N_VALID_6,
                N_VALID_8, n0, n1);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor *x, int64_t outFlag, float eps, int64_t numIters,
                                      const aclTensor *output, const aclTensor *normOut, const aclTensor *sumOut)
{
    CHECK_RET(CheckNotNull(x, outFlag, output, normOut, sumOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, outFlag, output, normOut, sumOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, outFlag, output, normOut, sumOut, numIters), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(x, outFlag, output, normOut, sumOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhornGetWorkspaceSize(const aclTensor *x, float eps, int64_t numIters, int64_t outFlag,
                                                aclTensor *output, aclTensor *normOut, aclTensor *sumOut,
                                                uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPreSinkhorn, DFX_IN(x, eps, numIters, outFlag), DFX_OUT(output, normOut, sumOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t actualOutFlag = outFlag;
    if (normOut == nullptr || sumOut == nullptr) {
        Shape emptyShape({});
        normOut = (uniqueExecutor.get())->AllocTensor(emptyShape, x->GetDataType(), Format::FORMAT_ND);
        sumOut = (uniqueExecutor.get())->AllocTensor(emptyShape, x->GetDataType(), Format::FORMAT_ND);
        actualOutFlag = 0;
    }

    auto ret = CheckParams(x, actualOutFlag, eps, numIters, output, normOut, sumOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support empty input tensor.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const aclTensor *xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor *outputContiguous = const_cast<aclTensor*>(l0op::Contiguous(output, uniqueExecutor.get()));
    CHECK_RET(outputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelOut =
        l0op::MhcPreSinkhorn(xContiguous, actualOutFlag, eps, numIters, outputContiguous, normOut, sumOut,
                             uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(kernelOut, output, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreSinkhorn);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
