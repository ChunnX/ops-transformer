/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

/*!
 * \file aclnn_quant_all_reduce_base.cpp
 * \brief
 */
#include "aclnn_quant_all_reduce_base.h"
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
#include "mc2/common/op_kernel/mc2_quant_reduce_scatter_context.h"

#ifdef BUILD_OPEN_PROJECT

#include "version/hcomm_version.h"
#define HCCL_CHANNEL_SUPPORT_VERSION 89999700

#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#endif  // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

#endif  // BUILD_OPEN_PROJECT

namespace {

using namespace op;
using namespace l0op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

static constexpr size_t HCCL_GROUP_NAME_LENGTH_MAX = 128U; // group长度小于128字符

// K-G量化支持的Dtype
static const std::initializer_list<op::DataType> X_DTYPE_KG_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2
};
static const std::initializer_list<op::DataType> SCALES_DTYPE_KG_SUPPORT_LIST = {
    op::DataType::DT_FLOAT
};

// MX量化支持的Dtype
static const std::initializer_list<op::DataType> X_DTYPE_MX_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2
};
static const std::initializer_list<op::DataType> SCALES_DTYPE_MX_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E8M0
};

// output支持的Dtype
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT
};

// 检查入参是否为nullptr
static bool QuantAllReduceCheckNotNull(const aclTensor* x, const aclTensor* scales,
                                       const aclTensor* output)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(scales, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

// 检查K-G量化方案中x、scales、output的数据类型是否在算子的支持列表内
static bool QuantAllReduceCheckKGAllDtypesValid(const aclTensor* x, const aclTensor* scales,
                                                const aclTensor* output)
{
    if (CheckType(x->GetDataType(), X_DTYPE_KG_SUPPORT_LIST) &&             \
        CheckType(scales->GetDataType(), SCALES_DTYPE_KG_SUPPORT_LIST) &&   \
        CheckType(output->GetDataType(), OUTPUT_DTYPE_SUPPORT_LIST)) {
        return true;
    } else {
        return false;
    }
}

// 检查MX量化方案中x、scales、output的数据类型是否在算子的支持列表内
static bool QuantAllReduceCheckMXAllDtypesValid(const aclTensor* x, const aclTensor* scales,
                                                const aclTensor* output)
{
    if (CheckType(x->GetDataType(), X_DTYPE_MX_SUPPORT_LIST) &&             \
        CheckType(scales->GetDataType(), SCALES_DTYPE_MX_SUPPORT_LIST) &&   \
        CheckType(output->GetDataType(), OUTPUT_DTYPE_SUPPORT_LIST)) {
        return true;
    } else {
        return false;
    }
}

// 统一数据类型检查
static bool QuantAllReduceCheckAllDtypesValid(const aclTensor* x, const aclTensor* scales,
                                              const aclTensor* output)
{
    bool isAllDtypesValid = false;
    isAllDtypesValid = (QuantAllReduceCheckKGAllDtypesValid(x, scales, output) || \
                        QuantAllReduceCheckMXAllDtypesValid(x, scales, output));
    if (!isAllDtypesValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In KG quantMode, x support [DT_INT8/DT_HIFLOAT8/DT_FLOAT8_E5M2/DT_FLOAT8_E4M3FN],"
                "scales support [DT_FLOAT] and output support [DT_FLOAT16/DT_BF16/DT_FLOAT]."
                "In MX quantMode, x support [DT_FLOAT8_E5M2/DT_FLOAT8_E4M3FN], scales support [DT_FLOAT8_E8M0]"
                "and output support [DT_FLOAT16/DT_BF16/DT_FLOAT]."
                "Input tensors x: %s, scales: %s and output: %s are not simultaneously supported.",
                op::ToString(x->GetDataType()).GetString(),
                op::ToString(scales->GetDataType()).GetString(),
                op::ToString(output->GetDataType()).GetString());
    }
    return isAllDtypesValid;
}

static bool QuantAllReduceCheckAllFormatValid(const aclTensor *x, const aclTensor *scales, const aclTensor *output)
{
    if (IsPrivateFormat(x->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnQuantAllReduce, x format %s does not support Private Format.",
                op::ToString(x->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(scales->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnQuantAllReduce, scales format %s does not support Private Format.",
                op::ToString(scales->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In aclnnQuantAllReduce, output format %s does not support Private Format.",
                op::ToString(output->GetStorageFormat()).GetString());
        return false;
    }

    // 内部只处理ND格式，这里做reformat操作
    if (x->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x origin format is: %s.", op::ToString(x->GetStorageFormat()).GetString());
        x = l0op::ReFormat(x, op::Format::FORMAT_ND);
        CHECK_RET(x != nullptr, false);
    }
    if (scales->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("scales origin format is: %s.", op::ToString(scales->GetStorageFormat()).GetString());
        scales = l0op::ReFormat(scales, op::Format::FORMAT_ND);
        CHECK_RET(scales != nullptr, false);
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is: %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = l0op::ReFormat(output, op::Format::FORMAT_ND);
        CHECK_RET(output != nullptr, false);
    }

    return true;
}

static bool QuantAllReduceCheckGroupLength(const char* group)
{
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "QuantAllReduce, group is nullptr !");
        return false;
    }

    size_t groupLen = strnlen(group, HCCL_GROUP_NAME_LENGTH_MAX); // group长度≥128字符, 返回HCCL_GROUP_NAME_LENGTH_MAX
    if (groupLen >= HCCL_GROUP_NAME_LENGTH_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantAllReduce, Limit the length of the group to less than %lu characters.",
                HCCL_GROUP_NAME_LENGTH_MAX);
        return false;
    }

    return true;
}

// 参数综合校验
static aclnnStatus QuantAllReduceCheckParams(const aclTensor* x, const aclTensor* scales,
                                             const char* group, const aclTensor* output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(QuantAllReduceCheckNotNull(x, scales, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(QuantAllReduceCheckAllDtypesValid(x, scales, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查参数数据格式是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(QuantAllReduceCheckAllFormatValid(x, scales, output), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查group参数是否在要求范围之内
    CHECK_RET(QuantAllReduceCheckGroupLength(group), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

} // namespace

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

extern "C" aclnnStatus aclnnInnerQuantAllReduceV2GetWorkspaceSize(const aclTensor *context, const aclTensor* x,
                                                                  const aclTensor* scales,
                                                                  int64_t hcclBufferSize,
                                                                  const char* reduceOp, int64_t yDtype,
                                                                  int64_t worldSize, aclTensor* output,
                                                                  uint64_t* workspaceSize,
                                                                  aclOpExecutor** executor);

extern "C" aclnnStatus aclnnInnerQuantAllReduceV2(void* workspace, uint64_t workspaceSize,
                                                  aclOpExecutor* executor, const aclrtStream stream);

#else

extern "C" aclnnStatus aclnnInnerQuantAllReduceGetWorkspaceSize(const aclTensor* x, const aclTensor* scales,
                                                                const char* group, const char* reduceOp,
                                                                uint64_t yDtype, int64_t worldSize, aclTensor* output,
                                                                uint64_t* workspaceSize, aclOpExecutor** executor);

extern "C" aclnnStatus aclnnInnerQuantAllReduce(void* workspace, uint64_t workspaceSize,
                                                aclOpExecutor* executor, const aclrtStream stream);

#endif

extern "C" aclnnStatus aclnnQuantAllReduceBaseGetWorkspaceSize(const aclTensor* x, const aclTensor* scales,
                                                               const char* group, const char* reduceOp,
                                                               aclTensor* output, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor)
{
    aclnnStatus retParam = QuantAllReduceCheckParams(x, scales, group, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    int64_t worldSize = -1;

    aclnnStatus ret = ACLNN_SUCCESS;
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    aclTensor *mc2Context = nullptr;
    uint64_t hcclBuffSize = 0;
    const char *opName = "quant_all_reduce_v2";
    auto aclnnRet = Mc2Aclnn::Mc2Context::GetMc2ContextTensorForQrs(group, opName, hcclBuffSize,
                                                                    mc2Context, worldSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    ret = aclnnInnerQuantAllReduceV2GetWorkspaceSize(mc2Context, x, scales, hcclBuffSize,
                                                     reduceOp, yDtype, worldSize, output,
                                                     workspaceSize, executor);
    OP_LOGD("Execute QuantAllReduceV2, aclnnGetWorkspaceSize ret is: %d.", ret);
#else
    ret = aclnnInnerQuantAllReduceGetWorkspaceSize(x, scales, group, reduceOp, yDtype,
                                                   worldSize, output, workspaceSize, executor);
    OP_LOGD("Execute QuantAllReduce, aclnnGetWorkspaceSize ret is: %d.", ret);
#endif
    return ret;
}

extern "C" aclnnStatus aclnnQuantAllReduceBase(void* workspace, uint64_t workspaceSize,
                                               aclOpExecutor* executor, const aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }

#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    OP_LOGD("inner QuantAllReduceV2 start");
    return aclnnInnerQuantAllReduceV2(workspace, workspaceSize, executor, stream);
#else
    OP_LOGD("inner QuantAllReduce start");
    return aclnnInnerQuantAllReduce(workspace, workspaceSize, executor, stream);
#endif
}
