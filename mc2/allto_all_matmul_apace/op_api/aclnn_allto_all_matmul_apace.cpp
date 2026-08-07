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
 * \file aclnn_allto_all_matmul_apace.cpp
 * \brief aclnn API implementation for AlltoAllMatmulApace (MX-quant FP8, apace UDMA path)
 */

#include "aclnn_allto_all_matmul_apace.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/op_executor.h"

using namespace op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static bool IsTransposeLastTwoDims(const aclTensor *tensor)
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

static aclTensor *TransX2Tensor(const aclTensor *x2)
{
    uint64_t storageShapeDimNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = x2->GetStorageShape().GetDim(i);
    }
    uint64_t viewShapeDimNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim(viewShapeDimNum);
    viewDim[0] = x2->GetViewShape().GetDim(1);
    viewDim[1] = x2->GetViewShape().GetDim(0);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    stride[0] = transStride[1];
    stride[1] = transStride[0];
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;
    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, x2->GetTensor()->GetAddr());
}

extern "C" aclnnStatus aclnnInnerAlltoAllMatmulApaceGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1Scale, const aclTensor *x2Scale, char *group, int64_t worldSize, int64_t hcclBufferSize,
    int64_t yDtype, int64_t x1QuantMode, int64_t x2QuantMode, bool transposeX1, bool transposeX2,
    int64_t groupSize, bool alltoallOutFlag, char *commModeOptional, int64_t precisionMode, const aclTensor *yOut,
    const aclTensor *all2allOutOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllMatmulApace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream);

extern "C" aclnnStatus aclnnAlltoAllMatmulApaceGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const char *group, int64_t worldSize,
    int64_t hcclBufferSize, int64_t x1QuantMode, int64_t x2QuantMode, bool transposeX1, bool transposeX2,
    int64_t groupSize, const char *commMode, int64_t precisionMode, const aclTensor *output,
    const aclTensor *alltoAllOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnAlltoAllMatmulApaceGetWorkspaceSize start");
    (void)biasOptional;
    (void)alltoAllOutOptional;
    (void)hcclBufferSize;

    if (x1 == nullptr || x2 == nullptr || x1ScaleOptional == nullptr || x2ScaleOptional == nullptr ||
        output == nullptr || workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "aclnnAlltoAllMatmulApace: null pointer in input parameters.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto transX2 = x2;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        bool notContiguous = IsTransposeLastTwoDims(x2);
        if (notContiguous) {
            transposeX2 = true;
            transX2 = TransX2Tensor(x2);
            if (transX2 == nullptr) {
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
    }

    constexpr int64_t kDefaultYdtype = 27; // ge::DT_BF16
    constexpr bool kDefaultAlltoallOutFlag = false;

    aclnnStatus ret = aclnnInnerAlltoAllMatmulApaceGetWorkspaceSize(
        context, x1, transX2, biasOptional, x1ScaleOptional, x2ScaleOptional, const_cast<char *>(group), worldSize,
        hcclBufferSize, kDefaultYdtype, x1QuantMode, x2QuantMode, transposeX1, transposeX2,
        groupSize, kDefaultAlltoallOutFlag, const_cast<char *>(commMode ? commMode : "ai_cpu"), precisionMode, output,
        alltoAllOutOptional, workspaceSize, executor);

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAlltoAllMatmulApaceGetWorkspaceSize failed, ret=%d", ret);
    }

    OP_LOGD("aclnnAlltoAllMatmulApaceGetWorkspaceSize end: workspaceSize=%lu", *workspaceSize);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllMatmulApace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream)
{
    OP_LOGD("aclnnAlltoAllMatmulApace start");

    NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    aclnnStatus ret = aclnnInnerAlltoAllMatmulApace(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAlltoAllMatmulApace failed, ret=%d", ret);
        return ACLNN_ERR_INNER;
    }

    OP_LOGD("aclnnAlltoAllMatmulApace end");
    return ACLNN_SUCCESS;
}
