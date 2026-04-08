/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#include <algorithm>

#include "securec.h"
#include "common/utils/op_mc2.h"
#include "acl/acl.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"
#include "common/utils/hccl_util.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "aclnn_allto_allv_quant_grouped_mat_mul.h"

namespace {
using namespace op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
};

// 需要使用的常量定义
static constexpr size_t TWO_DIMS = 2U;
static constexpr size_t THREE_DIMS = 3U;

static bool CheckNullStatus(const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                            const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                            bool permuteOutFlag, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // // 检查必选入参出参为非空
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Optional output flag does not match optional output ptr.");
        return false;
    }
    return true;
}

// 检查必要输入是否为空/quantMode为1或6，必须非空/1或6
static bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY,
                         const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, int64_t gmmXQuantMode,
                         int64_t gmmWeightQuantMode)
{
    if (gmmX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmX should not be null.");
        return false;
    }
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmWeight should not be null.");
        return false;
    }
    if (gmmY == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmY should not be null.");
        return false;
    }
    if (gmmXScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXScale should not be null.");
        return false;
    }
    if (gmmWeightScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeightScale should not be null.");
        return false;
    }
    if ((gmmXQuantMode != static_cast<int64_t>(QuantModeType::PERTENSOR_QUANT)) &&
        (gmmXQuantMode != static_cast<int64_t>(QuantModeType::MX_QUANT))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXQuantMode should be 1(pertensor quant) or 6(mx quant), but actual is %lu.", gmmXQuantMode);
        return false;
    }
    return true;
}

static bool CheckGmmWeightValid(const aclTensor *gmmWeight) {
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "In AlltoAllvQuantGroupedMatmul, input gmmWeight should not be null.");
        return false;
    }
    OP_CHECK_WRONG_DIMENSION(gmmWeight, THREE_DIMS, return false);
    return true;
}

static bool CheckMmWeightValid(const aclTensor *mmWeightOptional) {
    if (mmWeightOptional == nullptr) {
        return false;
    }
    OP_CHECK_WRONG_DIMENSION(mmWeightOptional, TWO_DIMS, return false);
    return true;
}


// 处理支持转置的tensor物理排布不连续问题（gmmWeight）
static const aclTensor *TransGmmWeightTensor(const aclTensor *gmmWeight)
{
    uint64_t storageShapeDimNum = gmmWeight->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = gmmWeight->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = gmmWeight->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = gmmWeight->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[1] = gmmWeight->GetViewShape().GetDim(2);
    viewDim[2] = gmmWeight->GetViewShape().GetDim(1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(gmmWeight, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = gmmWeight->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[1] = transStride[2];
    stride[2] = transStride[1];

    auto offset = gmmWeight->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, gmmWeight->GetTensor()->GetAddr());
}

// 处理支持转置的tensor物理排布不连续问题（mmWeightOptional）
static const aclTensor *TransMmWeightOptionalTensor(const aclTensor *mmWeightOptional)
{
    uint64_t storageShapeDimNum = mmWeightOptional->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = mmWeightOptional->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = mmWeightOptional->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = mmWeightOptional->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[0] = mmWeightOptional->GetViewShape().GetDim(1);
    viewDim[1] = mmWeightOptional->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(mmWeightOptional, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = mmWeightOptional->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = mmWeightOptional->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, mmWeightOptional->GetTensor()->GetAddr());
}

// 检查tensor是否连续
bool IsTransposeLastTwoDims(const aclTensor *tensor) {
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    // 根据stride步长判断tensor是否连续取值的
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {		// 表示tensor为1x1的大小，不存在非连续问题
            return false;
          }
        return true;
      }
    return false;
}


static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                               const aclTensor *gmmWeightScale, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, int64_t gmmXQuantMode,
                               int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                               const char *group, int64_t epWorldSize, bool permuteOutFlag, const aclTensor *gmmY,
                               const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // 检查空状态
    CHECK_RET(CheckNullStatus(sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                              mmXScaleOptional, mmWeightScaleOptional, permuteOutFlag, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查参数是否为空
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, gmmY, gmmXScale, gmmWeightScale, gmmXQuantMode, gmmWeightQuantMode),
              ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("AlltoAllvQuantGroupedMatmul checkParams success");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, const aclTensor *mmXScaleOptional, 
    const aclTensor *mmWeightScaleOptional, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    bool permuteOutFlag, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t groupSize, int64_t yDtype, int64_t mmDtype, const aclTensor *gmmY,
    const aclTensor *mmYOptional, const aclTensor *permuteOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, const aclTensor *mmXScaleOptional, 
    const aclTensor *mmWeightScaleOptional, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    bool permuteOutFlag, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t groupSize, const aclTensor *gmmY, const aclTensor *mmYOptional,
    const aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{

    int64_t yDtype = gmmY->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();

    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, gmmXScale,
        gmmWeightScale, mmXScaleOptional, mmWeightScaleOptional, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
        transMmWeight, permuteOutFlag, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, groupSize,
        yDtype, mmDtype, gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllvQuantGroupedMatmul, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    int64_t gmmXQuantMode,
    int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    int64_t groupSize, bool permuteOutFlag, aclTensor *gmmY, aclTensor *mmYOptional, aclTensor *permuteOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前支持转置的gmmWeight涉及该处理
    CHECK_RET(CheckGmmWeightValid(gmmWeight), ACLNN_ERR_PARAM_NULLPTR);	// 先检查gmmWeight是否合法，避免非法操作
    bool notContiguous = IsTransposeLastTwoDims(gmmWeight);    // notContiguous标识gmmWeight是否是非连续的，通常在pytorch经过.t()会导致gmmWeight非连续
    auto transposeGmmWeight = gmmWeight;    // 复制一个gmmWeight
    if (notContiguous && transGmmWeight) {    // 当非连续和转置同时生效时，判断为错误用法，直接报错
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight not contiguous, and set gmmWeight transpose, it is error!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {    // 只有当非连续时，才会涉及到转连续等情况
        transGmmWeight = !transGmmWeight;
        // 把非连续gmmWeight转成连续
        transposeGmmWeight = TransGmmWeightTensor(gmmWeight);
        CHECK_RET(transposeGmmWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("gmmWeight is a non-contiguous tensor. The original dim1 is %ld, and dim2 is %ld. After processing, transposeGmmWeight dim1 is %ld, and dim2 is %ld.",
            gmmWeight->GetViewShape().GetDim(1), gmmWeight->GetViewShape().GetDim(2), transposeGmmWeight->GetViewShape().GetDim(1), transposeGmmWeight->GetViewShape().GetDim(2));
    }

    // 处理非连续Tensor，目前支持转置的mmWeightOptional涉及该处理
    if (CheckMmWeightValid(mmWeightOptional)) {
        bool notContiguous = IsTransposeLastTwoDims(mmWeightOptional); // notContiguous标识mmWeightOptional是否是非连续的
        auto transMmWeightOptional = mmWeightOptional;
        if (notContiguous && transMmWeight) { // 当非连续和转置同时生效时，判断为错误用法，直接报错
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "mmWeightOptional not contiguous, and set mmWeightOptional transpose, it is error!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            transMmWeight = !transMmWeight;
            // 把非连续mmWeightOptional转成连续
            transMmWeightOptional = TransMmWeightOptionalTensor(mmWeightOptional);
            CHECK_RET(transMmWeightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            OP_LOGD("mmWeightOptional is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. After "
                    "processing, transMmWeightOptional dim0 is %ld, and dim1 is %ld.",
                    mmWeightOptional->GetViewShape().GetDim(0), mmWeightOptional->GetViewShape().GetDim(1),
                    transMmWeightOptional->GetViewShape().GetDim(0), transMmWeightOptional->GetViewShape().GetDim(1));
            mmWeightOptional = transMmWeightOptional;
        }
        
    }
    aclnnStatus ret_param = CheckParams(
        gmmX, transposeGmmWeight, gmmXScale, gmmWeightScale,
        sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional,
        mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, transposeGmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, gmmXScale,
        gmmWeightScale, mmXScaleOptional, mmWeightScaleOptional,
        group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
        transMmWeight, permuteOutFlag, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, groupSize,
        gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMul(workspace, workspaceSize, executor, stream);
    return ret;
}
} // namespace
