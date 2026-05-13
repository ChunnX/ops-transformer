/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_H
#define CHECKER_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

// 量化与非量化共用的方法和常量、枚举值
namespace matmul_allto_all_check {

// 需要使用的常量定义
constexpr int64_t NEG_ONE = -1;
constexpr int64_t NEG_TWO = -2;
constexpr int64_t ZERO = 0;
constexpr int64_t TWO = 2;
constexpr size_t ONE_DIM = 1;
constexpr size_t TWO_DIMS = 2U;
constexpr size_t MAX_GROUP_LEN = 128U;
constexpr int64_t KVALUE_MIN = 1;
constexpr int64_t KVALUE_MAX = 65535;
constexpr char KC_SCENE[] = "KC quantization(x1QuantMode=3, x2QuantMode=2)";
constexpr char KCDYN_SCENE[] = "KCDYN quantization(x1QuantMode=7, x2QuantMode=2)";
constexpr char MX_SCENE[] = "MX quantization(x1QuantMode=6, x2QuantMode=6)";

// 通信域相关枚举值
enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
// 量化相关枚举值
enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
    DYN_PERTOKEN_QUANT = 7
};

// 带场景标识的校验宏，用于区分不同量化场景下的校验错误
#define OP_CHECK_WRONG_DIMENSION_WITH_SCENARIO(tensor, expectedDimNum, scenario, retExpr) \
  if (tensor->GetViewShape().GetDimNum() != expectedDimNum) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In [%s] scenario, dimension check failed: Expected %zu dimension input, but got %s with sizes %s.", \
            scenario, static_cast<size_t>(expectedDimNum), #tensor, op::ToString(tensor->GetViewShape()).GetString()); \
    retExpr; \
  }

#define OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO(tensor, supportList, scenario, retExpr) \
  if (!CheckType(tensor->GetDataType(), supportList)) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In [%s] scenario, dtype check failed: Tensor %s not implemented for %s, should be in dtype support list %s.", \
            scenario, #tensor, op::ToString(tensor->GetDataType()).GetString(), op::ToString(supportList).GetString()); \
    retExpr; \
  }

#define OP_CHECK_DTYPE_NOT_SAME_WITH_SCENARIO(tensor1, tensor2, scenario, retExpr) \
  if (tensor1->GetDataType() != tensor2->GetDataType()) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In [%s] scenario, dtype consistency check failed: Expected both tensors to have same dtype, but found %s %s and %s %s.", \
            scenario, #tensor1, op::ToString(tensor1->GetDataType()).GetString(), \
            #tensor2, op::ToString(tensor2->GetDataType()).GetString()); \
    retExpr; \
  }

// 校验AlltoAll和Permute数据交换的方向参数, 在alltoallmatmul中可以为空和{-1,-2}, 在matmulalltoall中可以为空和{-2,-1}, 不允许为其他值
bool CheckAlltoAllAxes(const aclIntArray* alltoAllAxesOptional, bool isMatmulAlltoAll);

// 校验输入的转置配置，x1不允许转置
bool CheckTransposeX1(bool transposeX1);

// 校验通信域名的字符串长度是否符合要求
bool CheckGroupLength(const char *group);

// 校验MatmulAlltoAll和QuantMatmulAlltoAll输入属性shape
bool CheckShapeMMAA(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, bool transposeX2, const aclTensor* output);
// 校验AlltoAllMatmul和AlltoAllQuantMatmul输入属性shape
bool CheckShapeAAMM(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, bool transposeX2, const aclTensor* output, const aclTensor* alltoAllOutOptional);

// 检查tensor是否连续
bool IsTransposeLastTwoDims(const aclTensor *tensor);

// 检查x2是否合法，空指针、空tensor和维度
aclnnStatus CheckX2Valid(const aclTensor* x2);

// 检查是否有alltoallout输出
bool IsAll2AllOut(const aclTensor *alltoAllOut);

// 处理支持转置的tensor物理排布不连续问题
aclTensor *TransX2Tensor(const aclTensor *x2);
} // namespace matmul_allto_all_check


namespace matmul_allto_all_helper {

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *output);

// 检查是否有空tensor
// 950 非量化场景支持x1的m轴为0，即token提示词为空
static bool CheckNotEmptyTensor(const aclTensor *x1, const aclTensor *x2, bool transposeX2);

// 非量化场景下，校验所有输入的参数类型是否正确（A5）
static bool CheckAllDtypesValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                const aclTensor *output);

// 910B在输入为bfloat16时，bias不支持bfloat16，只能为float32
static bool CheckAllDtypesValid910B(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                    const aclTensor *output);

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                        const aclTensor *output);

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                          const aclTensor *output);

static aclnnStatus CheckAndHandleParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                        const aclIntArray *alltoAllAxesOptional, const char *group, bool transposeX1,
                                        bool transposeX2, const aclTensor *output);

static aclnnStatus DealWithEmptyTensor(uint64_t *workspaceSize, aclOpExecutor **executor);
} // namespace matmul_allto_all_helper


namespace quant_matmul_allto_all_helper {

constexpr size_t THREE_DIMS = 3U;

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *x1Scale,
                         const aclTensor *x2Scale, const aclTensor *output);

// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *x1, const aclTensor *x2, bool transposeX2);

// 校验Scale为1D时的shape（KC量化）
static bool Check1DScaleShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                              const aclTensor *x2Scale, bool transposeX2);

// 校验Scale为3D时的shape（MX量化）
static bool Check3DScaleShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                              const aclTensor *x2Scale, bool transposeX2);

// 校验输入Scaleshape
static bool CheckScaleShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                            const aclTensor *x2Scale, int64_t x1QuantMode, int64_t x2QuantMode, bool transposeX2);

static bool CheckKCBiasDtypesValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                   const aclTensor *x2Scale, const aclTensor *bias, const aclTensor *y);

// KC量化场景下，校验所有输入的参数类型是否正确(A5)
static bool CheckKCQuantDtypesValidA5(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                      const aclTensor *x2Scale, const aclTensor *biasOptional, const aclTensor *output);

// MX量化场景下，校验所有输入的参数类型是否正确(A5)
static bool CheckMXQuantDtypesValidA5(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                      const aclTensor *x2Scale, const aclTensor *biasOptional, const aclTensor *output);

// 校验所有场景的数据类型是否在各自的支持列表中
static bool CheckDtypesValid(const aclTensor *x1, const aclTensor *x2, const int64_t x1QuantMode,
                             const int64_t x2QuantMode, const aclTensor *x1Scale, const aclTensor *x2Scale,
                             const aclTensor *biasOptional, const aclTensor *output);

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                        const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *output);

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                          const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *output);

static aclnnStatus CheckAndHandleParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                        const aclTensor *x1Scale, const aclTensor *x2Scale, const char *group,
                                        const aclIntArray *alltoAllAxesOptional, int64_t x1QuantMode,
                                        int64_t x2QuantMode, bool transposeX1, bool transposeX2,
                                        const aclTensor *output);
} // namespace quant_matmul_allto_all_helper

#endif // CHECKER_H
