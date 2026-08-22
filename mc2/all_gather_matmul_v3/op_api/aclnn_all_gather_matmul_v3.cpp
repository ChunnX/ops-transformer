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
 * \file aclnn_all_gather_matmul_v3.cpp
 * \brief aclnn API implementation for AllGatherMatmulV3 (MX-quant FP8/FP4, apace UDMA path)
 */

#include "aclnnInner_all_gather_matmul_v3.h"

#include <cstring>
#include <initializer_list>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "graph/types.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/op_executor.h"

using namespace op;

namespace {

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

constexpr int64_t TWO_DIMS = 2;
constexpr int64_t SCALE_DIMS = 3;
constexpr int64_t MX_SCALE_BLOCK = 64;
constexpr int64_t SCALE_LAST_DIM = 2;

const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT4_E2M1};
const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};
const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

bool IsDtypeSupported(const aclTensor *tensor, const std::initializer_list<op::DataType> &supportTypes)
{
    return std::find(supportTypes.begin(), supportTypes.end(), tensor->GetDataType()) != supportTypes.end();
}

bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

// 用户传 [N,K].t() (viewShape [K,N]) → 翻回 [N,K] 连续视图，保留原始物理存储
static const aclTensor *TransX2Tensor(const aclTensor *x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (uint64_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }

    uint64_t viewDimsNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDims;
    viewDims.resize(viewDimsNum);
    for (uint64_t i = 0; i < viewDimsNum; i++) {
        viewDims[i] = x2->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDims[0] = x2->GetViewShape().GetDim(1);
    viewDims[1] = x2->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(viewDimsNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDims.data(), viewDimsNum, dataType, stride.data(), offset, format, storageDims.data(),
                           storageDimsNum, x2->GetTensor()->GetAddr());
}

// CheckNotNull: 校验 REQUIRED 输入/输出非空（gather_out 由 torch 侧内部构造空 tensor，不校验）
aclnnStatus CheckNotNull(const aclTensor *context, const aclTensor *x1, const aclTensor *x2,
                         const aclTensor *output, const uint64_t *workspaceSize,
                         const aclOpExecutor **executor)
{
    OP_CHECK_NULL(context, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(output, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

// CheckDtype: dtype 校验
aclnnStatus CheckDtype(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                       const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *output)
{
    if (!IsDtypeSupported(x1, X_DTYPE_SUPPORT_LIST) || !IsDtypeSupported(x2, X_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x1/x2 dtype must be "
                "FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1, got x1=%d x2=%d.",
                static_cast<int>(x1->GetDataType()), static_cast<int>(x2->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1) !=
        (x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: FP4 requires both x1 and x2 to be "
                "FLOAT4_E2M1, got x1=%d x2=%d.",
                static_cast<int>(x1->GetDataType()), static_cast<int>(x2->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // x1Scale/x2Scale 为 OPTIONAL，非空时校验 dtype
    if (x1Scale != nullptr && !IsDtypeSupported(x1Scale, SCALE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x1Scale must be FLOAT8_E8M0, got %d.",
                static_cast<int>(x1Scale->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2Scale != nullptr && !IsDtypeSupported(x2Scale, SCALE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x2Scale must be FLOAT8_E8M0, got %d.",
                static_cast<int>(x2Scale->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsDtypeSupported(output, OUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: output dtype must be BF16/FP16, got %d.",
                static_cast<int>(output->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // bias 不为空时校验 FLOAT
    if (bias != nullptr && !IsDtypeSupported(bias, BIAS_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: bias dtype must be FLOAT, got %d.",
                static_cast<int>(bias->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// CheckFormat: ND 校验（scale 兼容 NCL，同 v2 CheckScale）
aclnnStatus CheckFormat(const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                        const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *output)
{
    if (context->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: context only supports ND format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x1 only supports ND format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x2 only supports ND format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: output only supports ND format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (bias != nullptr && bias->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: bias only supports ND format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1Scale != nullptr && x1Scale->GetStorageFormat() != op::Format::FORMAT_ND &&
        x1Scale->GetStorageFormat() != op::Format::FORMAT_NCL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x1_scale only supports ND/NCL format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2Scale != nullptr && x2Scale->GetStorageFormat() != op::Format::FORMAT_ND &&
        x2Scale->GetStorageFormat() != op::Format::FORMAT_NCL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x2_scale only supports ND/NCL format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// CheckShape: gather_out 必须空 tensor；scale dim(0) 校验
aclnnStatus CheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                       const aclTensor *x2Scale, const aclTensor *output)
{
    if (x1->GetViewShape().GetDimNum() != TWO_DIMS || x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x1/x2 must be 2D, got x1=%zuD x2=%zuD.",
                x1->GetViewShape().GetDimNum(), x2->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (output->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: output must be 2D, got %zuD.",
                output->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // gather_out 由 torch 侧构造空 tensor，不校验
    // x1: [M, K]; x2: [N, K]
    int64_t mVal = x1->GetViewShape().GetDim(0);
    int64_t kX1 = x1->GetViewShape().GetDim(1);
    int64_t nVal = x2->GetViewShape().GetDim(0);
    int64_t kX2 = x2->GetViewShape().GetDim(1);
    if (kX1 != kX2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: K mismatch, x1.K=%ld != x2.K=%ld "
                "(x2 must be [N, K] layout).", kX1, kX2);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (output->GetViewShape().GetDim(1) != nVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: output.N=%ld != x2.N=%ld.",
                output->GetViewShape().GetDim(1), nVal);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // scale: x1Scale [M, Ceil(K/64), 2]; x2Scale [N, Ceil(K/64), 2]
    int64_t kg = (kX1 + MX_SCALE_BLOCK - 1) / MX_SCALE_BLOCK;
    if (x1Scale != nullptr) {
        if (x1Scale->GetViewShape().GetDimNum() != SCALE_DIMS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x1_scale must be %ldD, got %zuD.",
                    SCALE_DIMS, x1Scale->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x1Scale->GetViewShape().GetDim(0) != mVal ||
            x1Scale->GetViewShape().GetDim(1) != kg ||
            x1Scale->GetViewShape().GetDim(2) != SCALE_LAST_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnAllGatherQuantMatmulV3: x1_scale shape mismatch, expected [%ld, %ld, %ld].",
                    mVal, kg, SCALE_LAST_DIM);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (x2Scale != nullptr) {
        if (x2Scale->GetViewShape().GetDimNum() != SCALE_DIMS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnAllGatherQuantMatmulV3: x2_scale must be %ldD, got %zuD.",
                    SCALE_DIMS, x2Scale->GetViewShape().GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x2Scale->GetViewShape().GetDim(0) != nVal ||
            x2Scale->GetViewShape().GetDim(1) != kg ||
            x2Scale->GetViewShape().GetDim(2) != SCALE_LAST_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnAllGatherQuantMatmulV3: x2_scale shape mismatch, expected [%ld, %ld, %ld].",
                    nVal, kg, SCALE_LAST_DIM);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckAttrs(const char *group, const char *commMode)
{
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "aclnnAllGatherQuantMatmulV3: group is null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (commMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "aclnnAllGatherQuantMatmulV3: commMode is null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnAllGatherQuantMatmulV3GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const char *group, int64_t rankSize,
    int64_t hcclBufferSize, int64_t groupSize, const char *commMode, const aclTensor *output,
    const aclTensor *gatherOut, uint64_t *workspaceSize, aclOpExecutor **executor)  // todo: 这里就没amaxout对接torch了，可选输出暂不支持
{
    OP_LOGD("aclnnAllGatherQuantMatmulV3GetWorkspaceSize start");

    // 校验顺序：null -> dtype -> format -> attrs -> shape
    aclnnStatus ret = CheckNotNull(context, x1, x2, output, workspaceSize,
                                   const_cast<const aclOpExecutor **>(executor));
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckDtype(x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckFormat(context, x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckAttrs(group, commMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    OP_LOGD("X1 is %s.", x1->ToString().GetString());
    OP_LOGD("X2 is %s.", x2->ToString().GetString());

    constexpr bool kIsTransA = false;
    constexpr bool kIsTransB = true;
    auto transX2 = x2;
    auto transX2Scale = x2ScaleOptional;
    if (IsTransposeLastTwoDims(x2)) {
        OP_LOGI("x2 is a transposed [K,N] view, transpose back to [N,K].");
        transX2 = TransX2Tensor(x2);
        if (transX2 == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if ((x2ScaleOptional != nullptr) && MC2Aclnn::IsNeedScaleTrans(x2ScaleOptional)) {
        // printf("x2 view shape: %ld %ld %ld\n", x2ScaleOptional->GetViewShape().GetDim(0), x2ScaleOptional->GetViewShape().GetDim(1), x2ScaleOptional->GetViewShape().GetDim(2));
        transX2Scale = TransX2Tensor(x2ScaleOptional);
        if (transX2Scale == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
        OP_LOGI("x2Scale after trans: dim0=%ld dim1=%ld dim2=%ld",
                transX2Scale->GetViewShape().GetDim(0), transX2Scale->GetViewShape().GetDim(1),
                transX2Scale->GetViewShape().GetDim(2));
    }

    ret = CheckShape(x1, transX2, x1ScaleOptional, transX2Scale, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    uint64_t outDtype = static_cast<uint64_t>(output->GetDataType());
    OP_LOGD("outDtype value is: %lu.", outDtype);

    ret = aclnnInnerAllGatherMatmulV3GetWorkspaceSize(
        context, x1, transX2, biasOptional, x1ScaleOptional, transX2Scale,
        const_cast<char *>(group), hcclBufferSize,
        kIsTransA, kIsTransB, rankSize, groupSize, static_cast<int64_t>(outDtype),
        const_cast<char *>(commMode), output, gatherOut, nullptr,
        workspaceSize, executor);

    OP_LOGD("aclnnInnerAllGatherMatmulV3GetWorkspaceSize ret = %d.", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAllGatherMatmulV3GetWorkspaceSize failed, ret=%d", ret);
    }

    OP_LOGD("aclnnAllGatherQuantMatmulV3GetWorkspaceSize end: workspaceSize=%lu", *workspaceSize);
    return ret;
}

extern "C" aclnnStatus aclnnAllGatherQuantMatmulV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream)
{
    OP_LOGD("aclnnAllGatherQuantMatmulV3 start");

    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "aclnnAllGatherQuantMatmulV3: executor is null.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (NnopbaseSetHcclServerType != nullptr) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerAllGatherMatmulV3(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAllGatherMatmulV3 failed, ret=%d", ret);
        return ACLNN_ERR_INNER;
    }

    OP_LOGD("aclnnAllGatherQuantMatmulV3 end");
    return ACLNN_SUCCESS;
}
