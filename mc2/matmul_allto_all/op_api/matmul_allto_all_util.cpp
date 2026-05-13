/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_allto_all_util.h"
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
#include "common/utils/hccl_util.h"
#include "aclnn_kernels/transdata.h"

// 量化与非量化共用的方法和常量、枚举值
namespace matmul_allto_all_check {
using namespace op;

// 校验AlltoAll和Permute数据交换的方向参数, 可以为空和{-1,-2}, 不允许为其他值
bool CheckAlltoAllAxes(const aclIntArray* alltoAllAxesOptional, bool isMatmulAlltoAll)
{
    // alltoAllAxesOptional为空时会兼容性处理，不报错
    if (alltoAllAxesOptional == nullptr) {
        OP_LOGW("The alltoAllAxesOptional is nullptr.");
        return true;
    }
    uint64_t alltoallAxesSize = 0U;  // alltoallAxes的大小
    aclGetIntArraySize(alltoAllAxesOptional, &alltoallAxesSize);
    if (alltoallAxesSize != TWO_DIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of alltoAllAxesOptional should equal to 2, but it is %zu now.", alltoallAxesSize);
        return false;
    }
    int64_t data1 = (*alltoAllAxesOptional)[0];
    int64_t data2 = (*alltoAllAxesOptional)[1];
    if (isMatmulAlltoAll) {
        OP_API_CHECK((data1 != NEG_ONE), {
          OP_LOGE(ACLNN_ERR_PARAM_INVALID,
          "The 0-axis of alltoAllAxesOptional should be -1, but it is: %ld.", data1);
          return false;
        });
        OP_API_CHECK((data2 != NEG_TWO), {
          OP_LOGE(ACLNN_ERR_PARAM_INVALID,
          "The 1-axis of alltoAllAxesOptional should be -2, but it is: %ld.", data2);
          return false;
        });
    } else {
        OP_API_CHECK((data1 != NEG_TWO), {
          OP_LOGE(ACLNN_ERR_PARAM_INVALID,
          "The 0-axis of alltoAllAxesOptional should be -2, but it is: %ld.", data1);
          return false;
        });
        OP_API_CHECK((data2 != NEG_ONE), {
          OP_LOGE(ACLNN_ERR_PARAM_INVALID,
          "The 1-axis of alltoAllAxesOptional should be -1, but it is: %ld.", data2);
          return false;
        });
    }
    return true;
}

// 校验输入的转置配置，x1不允许转置
bool CheckTransposeX1(bool transposeX1)
{
    OP_API_CHECK(transposeX1, {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1 should not be transposed, but it is transposed.");
    return false;
  });
    return true;
}

// 校验通信域名的字符串长度是否符合要求
bool CheckGroupLength(const char *group)
{
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Group should not be nullptr.");
        return false;
    }
    auto len = strnlen(group, MAX_GROUP_LEN);
    if ((len >= MAX_GROUP_LEN) || (len == ZERO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Required group name length in range (0, 128).");
        return false;
    }
    return true;
}

// 校验输入tensor的维度
bool CheckInputDimensions(const aclTensor* x1, const aclTensor* x2, const aclTensor* output,
                          const aclTensor* alltoAllOutOptional = nullptr)
{
    OP_CHECK_WRONG_DIMENSION(x1, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(output, TWO_DIMS, return false);
    if (alltoAllOutOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(alltoAllOutOptional, TWO_DIMS, return false);
    }
    return true;
}

// 校验bias的维度和shape
bool CheckBiasShape(const aclTensor* biasOptional, int64_t nVal)
{
    if (biasOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(biasOptional, ONE_DIM, return false);
        auto biasDim = biasOptional->GetViewShape().GetDim(0);
        if (biasDim != nVal) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The n-axis of x2 and bias should be same, but x2's n-axis is: %ld and bias's n-axis is: %ld.", nVal, biasDim);
            return false;
        }
    }
    return true;
}

// 校验MatmulAlltoAll和QuantMatmulAlltoAll输入属性shape
bool CheckShapeMMAA(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                bool transposeX2, const aclTensor* output)
{
    if (!CheckInputDimensions(x1, x2, output)) {
        return false;
    }
    auto kdimX1 = x1->GetViewShape().GetDim(1);
    auto kdimX2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);

    OP_LOGI("The input transposeX2 is: %d, x1 dim0 is: %ld, x1 dim1 is: %ld, x2 dim0 is: %ld, x2 dim1 is: %ld",
            transposeX2, x1->GetViewShape().GetDim(0), x1->GetViewShape().GetDim(1), x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1));

    if (kdimX1 != kdimX2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "The k-axis of x1 and x2 should be same, but x1's k-axis is: %ld and x2's k-axis is: %ld.", kdimX1, kdimX2);
        return false;
    }
    if (kdimX1 < KVALUE_MIN || kdimX1 > KVALUE_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "The k-axis should be in range[1, 65535], but it is: %ld.", kdimX1);
        return false;
    }
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    return CheckBiasShape(biasOptional, nVal);
}

// 校验AlltoAllMatmul和AlltoAllQuantMatmul输入属性shape
bool CheckShapeAAMM(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                    bool transposeX2, const aclTensor* output, const aclTensor* alltoAllOutOptional)
{
    if (!CheckInputDimensions(x1, x2, output, alltoAllOutOptional)) {
        return false;
    }
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    return CheckBiasShape(biasOptional, nVal);
}

// 检查tensor是否连续
bool IsTransposeLastTwoDims(const aclTensor *tensor) {
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        OP_LOGD("The view_shape dim is: %ld", tensor->GetViewShape().GetDimNum());
        return false;
    }

    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
    if (tensor->GetViewStrides()[dim2] == 1
        && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            OP_LOGI("The input tensor is contiguous.");
            return false;
        }
        OP_LOGI("The input tensor is not contiguous.");
        return true;
    }
    return false;
}

// 检查x2是否合法
aclnnStatus CheckX2Valid(const aclTensor* x2) {
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2 should not be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
  	if (x2->IsEmpty()) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x2 do not support empty tensor.");
    	return ACLNN_ERR_PARAM_INVALID;
  	}
    OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// 检查是否有alltoallout输出
bool IsAll2AllOut(const aclTensor *alltoAllOut)
{
    if (alltoAllOut == nullptr) {
        return false;
    }
    if (alltoAllOut->IsEmpty() && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
         OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alltoAllOutOptional do not support empty tensor.");
         return false;
    }
    return true;
}

// 处理支持转置的tensor物理排布不连续问题
aclTensor *TransX2Tensor(const aclTensor *x2)
{
    uint64_t storageShapeDimNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = x2->GetStorageShape().GetDim(i);
    }
    uint64_t viewShapeDimNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = x2->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[0] = x2->GetViewShape().GetDim(1);
    viewDim[1] = x2->GetViewShape().GetDim(0);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;
    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, x2->GetTensor()->GetAddr());
}
} // namespace matmul_allto_all_check


namespace matmul_allto_all_helper {

using namespace op;
using namespace matmul_allto_all_check;

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *output)
{
    if (x1 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x1 should not be null.");
        return false;
    }
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2 should not be null.");
        return false;
    }
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Output should not be null.");
        return false;
    }
    return true;
}

// 检查是否有空tensor
// 950 非量化场景支持x1的m轴为0，即token提示词为空
static bool CheckNotEmptyTensor(const aclTensor *x1, const aclTensor *x2, bool transposeX2)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        auto mVal = x1->GetViewShape().GetDim(0);
        OP_API_CHECK((mVal == ZERO), {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X1 is empty tensor with zero dimM, which is unsupported.");
            return false;
        });
    }
    auto kVal1 = x1->GetViewShape().GetDim(1);
    auto kVal2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    OP_API_CHECK((kVal1 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X1 is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal2 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X2 is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((nVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X2 is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    return true;
}

// 非量化模式下X支持的FP16数据类型
static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};
// 非量化模式下Output支持的FP16数据类型
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_BF16};
// 非量化模式下910B对Bias数据类型有特殊规范
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                            op::DataType::DT_FLOAT};

// 非量化场景下，校验所有输入的参数类型是否正确（A5）
static bool CheckAllDtypesValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    // biasDtype可以为输入xDtype，也可以为fp32
    if (biasOptional != nullptr) {
        if (biasOptional->GetDataType() != op::DataType::DT_FLOAT && biasOptional->GetDataType() != x1->GetDataType()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "MatmulAlltoAllV2, biasOptional dtype should be x1Dtype or float32 , but it is %s.",
                    op::ToString(biasOptional->GetDataType()).GetString());
            return false;
        }
    }
    return true;
}

// 910B在输入为bfloat16时，bias不支持bfloat16，只能为float32
static bool CheckAllDtypesValid910B(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                    const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    if (biasOptional != nullptr) {
        auto biasDtype = biasOptional->GetDataType();
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, BIAS_DTYPE_SUPPORT_LIST, return false);
        if (x1->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(x1, biasOptional, return false);
        } else if (biasDtype != op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The dtype of bias must be [DT_FLOAT] when x's dtype is [DT_BFLOAT16], but get: [%s].",
                    op::ToString(biasDtype).GetString());
            return false;
        }
    }
    return true;
}

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                        const aclTensor *output)
{
    // 输入格式不支持私有格式
    if (IsPrivateFormat(x1->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnQuantMatmulAlltoAllV2, x1 format %s does not support private format.",
                op::ToString(x1->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(x2->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnQuantMatmulAlltoAllV2, x2 format %s does not support private format.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return false;
    }
    if (biasOptional != nullptr) {
        if (IsPrivateFormat(biasOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantMatmulAlltoAllV2, biasOptional format %s does not support private format.",
                    op::ToString(biasOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAllV2, output format %s does not support private format.",
                op::ToString(output->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                          const aclTensor *output)
{
    // 内部只处理ND格式，这里做reformat操作
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1 origin format is %s.", op::ToString(x1->GetStorageFormat()).GetString());
        x1 = l0op::ReFormat(x1, op::Format::FORMAT_ND);
        CHECK_RET(x1 != nullptr, false);
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2 origin format is %s.", op::ToString(x2->GetStorageFormat()).GetString());
        x2 = l0op::ReFormat(x2, op::Format::FORMAT_ND);
        CHECK_RET(x2 != nullptr, false);
    }
    if (biasOptional != nullptr) {
        if (biasOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("bias origin format is %s.", op::ToString(biasOptional->GetStorageFormat()).GetString());
            biasOptional = l0op::ReFormat(biasOptional, op::Format::FORMAT_ND);
            CHECK_RET(biasOptional != nullptr, false);
        }
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = l0op::ReFormat(output, op::Format::FORMAT_ND);
        CHECK_RET(output != nullptr, false);
    }
    return true;
}

static aclnnStatus CheckAndHandleParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                        const aclIntArray *alltoAllAxesOptional, const char *group, bool transposeX1,
                                        bool transposeX2, const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查空tensor
    CHECK_RET(CheckNotEmptyTensor(x1, x2, transposeX2), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape
    CHECK_RET(CheckShapeMMAA(x1, x2, biasOptional, transposeX2, output), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    // bias的数据类型限制在950和910B上有所区别，这里根据芯片版本做区分
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        CHECK_RET(CheckAllDtypesValid(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckAllDtypesValid910B(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    }
    // 5. 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    // 6. 兼容性处理非ND格式
    CHECK_RET(ReFormatNotND(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    // 7. 检查alltoallAxes是否为空或者[-1,-2]
    CHECK_RET(CheckAlltoAllAxes(alltoAllAxesOptional, true), ACLNN_ERR_PARAM_INVALID);
    // 8. 检查transposeX1是否合法, 目前不能为true
    CHECK_RET(CheckTransposeX1(transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 9. 检查group长度是否小于等于128
    CHECK_RET(CheckGroupLength(group), ACLNN_ERR_PARAM_INVALID);
    // 如果所有检查都通过，且reformat也通过，输出参数检查成功
    OP_LOGD("aclnnMatmulAlltoAllV2 checkParams success");
    return ACLNN_SUCCESS;
}

static aclnnStatus DealWithEmptyTensor(uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("MatmulAlltoAllV2, dealing with empty tensor.");
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace matmul_allto_all_helper


namespace quant_matmul_allto_all_helper {

using namespace op;
using namespace matmul_allto_all_check;

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *x1Scale,
                         const aclTensor *x2Scale, const aclTensor *output)
{
    if (x1 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x1 should not be null.");
        return false;
    }
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2 should not be null.");
        return false;
    }
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
        if (bias == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input bias should not be null.");
            return false;
        }
    }
    if (x1Scale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x1Scale should not be null.");
        return false;
    }
    if (x2Scale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2Scale should not be null.");
        return false;
    }
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Output should not be null.");
        return false;
    }
    return true;
}

// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *x1, const aclTensor *x2, bool transposeX2)
{
    auto mVal = x1->GetViewShape().GetDim(0);
    auto kVal1 = x1->GetViewShape().GetDim(1);
    auto kVal2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    OP_API_CHECK((mVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "QuantMatmulAlltoAll, x1 is empty tensor with zero dimM, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal1 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "QuantMatmulAlltoAll, x1 is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal2 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "QuantMatmulAlltoAll, x2 is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((nVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "QuantMatmulAlltoAll, x2 is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    return true;
}

// 校验Scale为1D时的shape（KC量化）
static bool Check1DScaleShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                              const aclTensor *x2Scale, bool transposeX2)
{
    OP_CHECK_WRONG_DIMENSION_WITH_SCENARIO(x1Scale, ONE_DIM, KC_SCENE, return false);
    OP_CHECK_WRONG_DIMENSION_WITH_SCENARIO(x2Scale, ONE_DIM, KC_SCENE, return false);
    auto mVal = x1->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x1ScaleDim = x1Scale->GetViewShape().GetDim(0);
    if (x1ScaleDim != mVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The m-axis of x1 and x1scale should be same, but x1's m-axis is: %ld and x1Scale's is: %ld.", mVal,
                x1ScaleDim);
        return false;
    }
    auto x2ScaleDim = x2Scale->GetViewShape().GetDim(0);
    if (x2ScaleDim != nVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The n-axis of x2 and x2ScaleDim should be same, but x2's n-axis is: %ld and x2ScaleDim is: %ld.", nVal,
                x2ScaleDim);
        return false;
    }
    return true;
}

// 校验Scale为3D时的shape（MX量化）
static bool Check3DScaleShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                              const aclTensor *x2Scale, bool transposeX2)
{
    OP_CHECK_WRONG_DIMENSION_WITH_SCENARIO(x1Scale, THREE_DIMS, MX_SCENE, return false);
    OP_CHECK_WRONG_DIMENSION_WITH_SCENARIO(x2Scale, THREE_DIMS, MX_SCENE, return false);
    auto mVal = x1->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x1ScaleMVal = x1Scale->GetViewShape().GetDim(0);
    auto x2ScaleNVal = transposeX2 ? x2Scale->GetViewShape().GetDim(0) : x2Scale->GetViewShape().GetDim(1);
    auto x1ScaleKVal = x1Scale->GetViewShape().GetDim(1);
    auto x2ScaleKVal = transposeX2 ? x2Scale->GetViewShape().GetDim(1) : x2Scale->GetViewShape().GetDim(0);
    auto x1ScaleLastDim = x1Scale->GetViewShape().GetDim(2);
    auto x2ScaleLastDim = x2Scale->GetViewShape().GetDim(2);
    if (x1ScaleMVal != mVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The m-axis of x1 and x1scale should be same, but x1's m-axis is: %ld and x1Scale's is: %ld.", mVal,
                x1ScaleMVal);
        return false;
    }
    if (x2ScaleNVal != nVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The n-axis of x2 and x2ScaleDim should be same, but x2's n-axis is: %ld and x2ScaleDim is: %ld.", nVal,
                x2ScaleNVal);
        return false;
    }
    if (x1ScaleKVal != x2ScaleKVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The k-axis of x1scale and x2scale should be same, but x1scale's k-axis is: %ld and x2Scale's k-axis "
                "is: %ld.",
                x1ScaleKVal, x2ScaleKVal);
        return false;
    }
    if (x1ScaleLastDim != TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dim of x1scale should be 2, but now it is: %ld.", x1ScaleLastDim);
        return false;
    }
    if (x2ScaleLastDim != TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dim of x2scale should be 2, but now it is: %ld.", x2ScaleLastDim);
        return false;
    }
    return true;
}

// 校验输入Scaleshape
static bool CheckScaleShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                            const aclTensor *x2Scale, int64_t x1QuantMode, int64_t x2QuantMode, bool transposeX2)
{
    bool scaleShapeValid = true;
    if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::MX_QUANT &&
        static_cast<QuantModeType>(x2QuantMode) == QuantModeType::MX_QUANT) {
        OP_API_CHECK(!transposeX2, {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In the mx quantization scenario, x2 must be transposed.");
            return false;
        });
        scaleShapeValid = Check3DScaleShape(x1, x2, x1Scale, x2Scale, transposeX2);
    } else if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::PERTOKEN_QUANT &&
               static_cast<QuantModeType>(x2QuantMode) == QuantModeType::PERCHANNEL_QUANT) {
        scaleShapeValid = Check1DScaleShape(x1, x2, x1Scale, x2Scale, transposeX2);
    }
    return scaleShapeValid;
}

// 910B数据类型校验
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_X = {op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BIAS = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_Y = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_SCALE = {op::DataType::DT_FLOAT};

static bool CheckKCBiasDtypesValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                   const aclTensor *x2Scale, const aclTensor *bias, const aclTensor *y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, DTYPE_SUPPORT_LIST_Y, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1Scale, DTYPE_SUPPORT_LIST_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, DTYPE_SUPPORT_LIST_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_BIAS, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    auto biasDtype = bias->GetDataType();
    auto yDtype = y->GetDataType();
    if (biasDtype != ge::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SAME(y, bias, return false);
    } else if (yDtype != ge::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When bias' dtype is [DT_FLOAT], y's dtype must be [DT_BF16], but get [%s].",
                op::ToString(yDtype).GetString());
        return false;
    }
    return true;
}

// 950数据类型校验
// 量化模式下X支持的FP8数据类型（PerToken量化、PerChannel量化）(A5)
static const std::initializer_list<op::DataType> X_DTYPE_FP8_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT8_E4M3FN,
                                                                                op::DataType::DT_FLOAT8_E5M2};
// 量化模式下X支持的FP4和FP8数据类型（MXQuant量化）(A5)
static const std::initializer_list<op::DataType> X_DTYPE_FP4ANDFP8_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT4_E2M1};
// 量化模式下Bias支持的数据类型（KC量化、MXQuant量化）(A5)
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT};
// 量化模式下Scale支持的FP8数据类型（MXQuant量化）(A5)
static const std::initializer_list<op::DataType> SCALE_DTYPE_FP8_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT8_E8M0};
// 量化模式下Scale支持的FP32数据类型（PerToken量化、PerChannel量化）(A5)
static const std::initializer_list<op::DataType> SCALE_DTYPE_FP32_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT};
// 量化模式下Output支持的数据类型（KC量化、MXQuant量化）(A5)
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

// KC量化场景下，校验所有输入的参数类型是否正确(A5)
static bool CheckKCQuantDtypesValidA5(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                      const aclTensor *x2Scale, const aclTensor *biasOptional, const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(x1, X_DTYPE_FP8_SUPPORT_LIST_A5, KC_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(x2, X_DTYPE_FP8_SUPPORT_LIST_A5, KC_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(x1Scale, SCALE_DTYPE_FP32_SUPPORT_LIST_A5, KC_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(x2Scale, SCALE_DTYPE_FP32_SUPPORT_LIST_A5, KC_SCENE, return false);
    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(biasOptional, BIAS_DTYPE_SUPPORT_LIST_A5, KC_SCENE, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(output, OUTPUT_DTYPE_SUPPORT_LIST_A5, KC_SCENE, return false);
    return true;
}

// MX量化场景下，校验所有输入的参数类型是否正确(A5)
static bool CheckMXQuantDtypesValidA5(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                      const aclTensor *x2Scale, const aclTensor *biasOptional, const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(x1, X_DTYPE_FP4ANDFP8_SUPPORT_LIST_A5, MX_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(x2, X_DTYPE_FP4ANDFP8_SUPPORT_LIST_A5, MX_SCENE, return false);
    if (x1->GetDataType() != x2->GetDataType() &&
        (x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1 || x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In mxquant scenario, x1 and x2 must have same dtype when one is fp4_e2m1, but found x1: %s, x2: %s.",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(x1Scale, SCALE_DTYPE_FP8_SUPPORT_LIST_A5, MX_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(x2Scale, SCALE_DTYPE_FP8_SUPPORT_LIST_A5, MX_SCENE, return false);
    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(biasOptional, BIAS_DTYPE_SUPPORT_LIST_A5, MX_SCENE, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(output, OUTPUT_DTYPE_SUPPORT_LIST_A5, MX_SCENE, return false);
    return true;
}

// 校验所有场景的数据类型是否在各自的支持列表中
static bool CheckDtypesValid(const aclTensor *x1, const aclTensor *x2, const int64_t x1QuantMode,
                             const int64_t x2QuantMode, const aclTensor *x1Scale, const aclTensor *x2Scale,
                             const aclTensor *biasOptional, const aclTensor *output)
{
    bool isAllDtypesValid = false;
    // 根据量化场景和芯片型号进入不同分支判断
    if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::PERTOKEN_QUANT &&
        static_cast<QuantModeType>(x2QuantMode) == QuantModeType::PERCHANNEL_QUANT) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            isAllDtypesValid = CheckKCBiasDtypesValid(x1, x2, x1Scale, x2Scale, biasOptional, output);
        } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            isAllDtypesValid = CheckKCQuantDtypesValidA5(x1, x2, x1Scale, x2Scale, biasOptional, output);
        }
    } else if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::MX_QUANT &&
               static_cast<QuantModeType>(x2QuantMode) == QuantModeType::MX_QUANT) {
        isAllDtypesValid = CheckMXQuantDtypesValidA5(x1, x2, x1Scale, x2Scale, biasOptional, output);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The input x1QuantMode %ld and x2QuantMode %ld do not match any currently supported quantization mode "
                "scenarios.",
                x1QuantMode, x2QuantMode);
    }
    return isAllDtypesValid;
}

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                        const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *output)
{
    // 输入格式不支持私有格式
    if (IsPrivateFormat(x1->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnQuantMatmulAlltoAll, x1 format %s does not support private format.",
                op::ToString(x1->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(x2->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnQuantMatmulAlltoAll, x2 format %s does not support private format.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return false;
    }
    if (biasOptional != nullptr) {
        if (IsPrivateFormat(biasOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantMatmulAlltoAll, biasOptional format %s does not support private format.",
                    op::ToString(biasOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (IsPrivateFormat(x1Scale->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnQuantMatmulAlltoAll, x1Scale format %s does not support private format.",
                op::ToString(x1Scale->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(x2Scale->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnQuantMatmulAlltoAll, x2Scale format %s does not support private format.",
                op::ToString(x2Scale->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnQuantMatmulAlltoAll, output format %s does not support private format.",
                op::ToString(output->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                          const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *output)
{
    // 内部只处理ND格式，这里做reformat操作
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1 origin format is %s.", op::ToString(x1->GetStorageFormat()).GetString());
        x1 = l0op::ReFormat(x1, op::Format::FORMAT_ND);
        CHECK_RET(x1 != nullptr, false);
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2 origin format is %s.", op::ToString(x2->GetStorageFormat()).GetString());
        x2 = l0op::ReFormat(x2, op::Format::FORMAT_ND);
        CHECK_RET(x2 != nullptr, false);
    }
    if (biasOptional != nullptr) {
        if (biasOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("bias origin format is %s.", op::ToString(biasOptional->GetStorageFormat()).GetString());
            biasOptional = l0op::ReFormat(biasOptional, op::Format::FORMAT_ND);
            CHECK_RET(biasOptional != nullptr, false);
        }
    }
    if (x1Scale->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1Scale origin format is %s.", op::ToString(x1Scale->GetStorageFormat()).GetString());
        x1Scale = l0op::ReFormat(x1Scale, op::Format::FORMAT_ND);
        CHECK_RET(x1Scale != nullptr, false);
    }
    if (x2Scale->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2Scale origin format is %s.", op::ToString(x2Scale->GetStorageFormat()).GetString());
        x2Scale = l0op::ReFormat(x2Scale, op::Format::FORMAT_ND);
        CHECK_RET(x2Scale != nullptr, false);
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = l0op::ReFormat(output, op::Format::FORMAT_ND);
        CHECK_RET(output != nullptr, false);
    }
    return true;
}

static aclnnStatus CheckAndHandleParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                        const aclTensor *x1Scale, const aclTensor *x2Scale, const char *group,
                                        const aclIntArray *alltoAllAxesOptional, int64_t x1QuantMode,
                                        int64_t x2QuantMode, bool transposeX1, bool transposeX2,
                                        const aclTensor *output)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, biasOptional, x1Scale, x2Scale, output), ACLNN_ERR_PARAM_NULLPTR);
    // 检查空tensor
    CHECK_RET(CheckNotEmptyTensor(x1, x2, transposeX2), ACLNN_ERR_PARAM_INVALID);
    // 检查shape
    CHECK_RET(CheckShapeMMAA(x1, x2, biasOptional, transposeX2, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckScaleShape(x1, x2, x1Scale, x2Scale, x1QuantMode, x2QuantMode, transposeX2),
              ACLNN_ERR_PARAM_INVALID);
    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypesValid(x1, x2, x1QuantMode, x2QuantMode, x1Scale, x2Scale, biasOptional, output),
              ACLNN_ERR_PARAM_INVALID);
    // 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(x1, x2, biasOptional, x1Scale, x2Scale, output), ACLNN_ERR_PARAM_INVALID);
    // 兼容性处理非ND格式
    CHECK_RET(ReFormatNotND(x1, x2, biasOptional, x1Scale, x2Scale, output), ACLNN_ERR_PARAM_INVALID);
    // 检查alltoAllAxes是否为空或者[-1,-2]
    CHECK_RET(CheckAlltoAllAxes(alltoAllAxesOptional, true), ACLNN_ERR_PARAM_INVALID);
    // 检查transposeX1是否合法, 目前不能为true
    CHECK_RET(CheckTransposeX1(transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 检查group长度是否小于等于128
    CHECK_RET(CheckGroupLength(group), ACLNN_ERR_PARAM_INVALID);
    // 如果所有检查都通过，且reformat也通过，输出参数检查成功
    OP_LOGD("aclnnQuantMatmulAlltoAll checkParams success");
    return ACLNN_SUCCESS;
}
} // namespace quant_matmul_allto_all_helper
