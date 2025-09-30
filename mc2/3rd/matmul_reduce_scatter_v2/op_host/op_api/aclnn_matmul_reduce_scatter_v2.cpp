/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "aclnn_matmul_reduce_scatter_v2.h"
#include "securec.h"

#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul_util.h"
#include "hccl_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
static constexpr int64_t SCALAR = 1;
static constexpr int64_t ONE_DIM = 1;
static constexpr int64_t TWO_DIMS = 2;
static constexpr int64_t KVALUE_MIN = 256;
static constexpr int64_t KVALUE_MAX = 65535;
static constexpr int64_t LAST_AXIS = -1;
static constexpr int64_t SECOND_TO_LAST_AXIS = -2;
static constexpr int64_t DIM_NUM_THREE = 3;
static constexpr int64_t DIM_NUM_TWO = 2;
static constexpr int64_t DIM_NUM_ONE = 1;
typedef struct {
    uint32_t id;
    const char* funcName;
    bool hasReg;
} NnopbaseDfxId;

enum class NnopbaseHcclServerType : uint8_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerMatmulReduceScatterV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const aclTensor* quantScale, const char* group, const char* reduce_op, bool transposeX1, bool transposeX2,
    int64_t commTurn, int64_t rankSize, int64_t blockSize, int64_t groupSize, bool isAmaxOut, int64_t yDtype, const char* commMode,
    const aclTensor* output, const aclTensor* amaxOut, uint64_t* workspaceSize, aclOpExecutor** executor){
        return ACLNN_SUCCESS;
    };
extern aclnnStatus aclnnInnerMatmulReduceScatterV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream){
                                                    return ACLNN_SUCCESS;
                                                   };
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static inline bool IsAscend910D(void)
{
    return op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_95;
}

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* output)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}
enum class CaseOption {
    HIGH_ACCURACY = 0,
    LOW_ACCURACY_PER_TENSOR_WITHOUT_QUANT_AMAX,
    LOW_ACCURACY_PER_TENSOR_WITH_QUANT_AMAX,
    LOW_ACCURACY_PER_BLOCK,
    INVALID,
};

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2,
    op::DataType::DT_HIFLOAT8, op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> BIAS_OUTPUT_SUPPORT_TYPE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INPUT_SUPPORT_TYPE_HIGH_ACCURACY = {op::DataType::DT_FLOAT16,
                                                                                     op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> INPUT_SUPPORT_TYPE_LOW_ACCURACY = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8};

static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* output)
{
    // 检查x1、x2、bias、output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, BIAS_OUTPUT_SUPPORT_TYPE, return false);
    // 检查bias的数据类型是否在算子的支持列表内
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_OUTPUT_SUPPORT_TYPE, return false);
    }
    return true;
}
// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(int64_t streamMode)
{
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected streamMode to be %ld, but got %ld.", NUM_ACL_STOP_ON_FAILURE,
                streamMode);
        return false;
    }
    return true;
}
static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, int64_t streamMode,
                               const aclTensor* output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x1, x2, bias, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CheckShape(const aclTensor* x1, const aclTensor* x2, bool isTransA)
{
    OP_CHECK_WRONG_DIMENSION(x1, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return false);
    int64_t kVal1 = 0, kVal2 = 0;
    if (isTransA) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Does not support transpose x1 matrix.");
        return false;
    }
    kVal1 = x1->GetViewShape().GetDim(1);
    kVal2 = x2->GetViewShape().GetDim(0);
    OP_API_CHECK((kVal1 != kVal2), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The k-axis of x1 and x2 should be same, but x1's k-axis is: %ld and x2's k-axis is: %ld.", kVal1,
                kVal2);
        return false;
    });
    OP_API_CHECK((kVal1 < KVALUE_MIN || kVal1 >= KVALUE_MAX), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis should be in range[%ld, %ld), but it is: %ld.", KVALUE_MIN,
                KVALUE_MAX, kVal1);
        return false;
    });
    return true;
}

static bool CheckEmptyTensor(const aclTensor* tensor, const char* tensorName)
{
    if (IsNullptr(tensor, tensorName)) {
        return false;
    }
    if (tensor->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s should not be empty.\n", tensorName);
        return false;
    }
    return true;
}

static bool CheckEmptyOptionalTensor(const aclTensor* tensor, const char* tensorName)
{
    if (tensor == nullptr) {
        OP_LOGD("Expecting valid tensor, name %s", tensorName);
        return false;
    }
    if (tensor->IsEmpty()) {
        OP_LOGD("Expecting non-empty tensor, name %s", tensorName);
        return false;
    }
    return true;
}

static enum CaseOption CheckHighAccuracyCase(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                             const aclTensor* y, const aclTensor* amax)
{
    // 高精度场景须数据类型相同
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return CaseOption::INVALID);
    OP_CHECK_DTYPE_NOT_SAME(x1, y, return CaseOption::INVALID);
    if (CheckEmptyOptionalTensor(bias, "bias")) {
        OP_CHECK_DTYPE_NOT_SAME(x1, bias, return CaseOption::INVALID);
    }

    if (amax != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Does not support non-nullptr amaxOutOptional.");
        return CaseOption::INVALID;
    }
    return CaseOption::HIGH_ACCURACY;
}

static enum CaseOption CheckLowAccuracyCase(const aclTensor* x1, const aclTensor* x2, [[maybe_unused]] const aclTensor* bias,
                                            [[maybe_unused]] const aclTensor* y, [[maybe_unused]] const aclTensor* amax,
                                            const aclTensor* x1Scale, const aclTensor* x2Scale,
                                            [[maybe_unused]] const aclTensor* quantScale)
{
    //先x1 x2指针判空
    if(!CheckEmptyTensor(x1, "x1")||!CheckEmptyTensor(x2, "x2")){
        return CaseOption::INVALID;
    }
    // 矩阵入参为Hifloat8时，x1x2必须类型相同。
    if (x1->GetDataType() == op::DataType::DT_HIFLOAT8) {
        OP_CHECK_DTYPE_NOT_SAME(x1, x2, return CaseOption::INVALID);
    }
    
    if (!CheckEmptyTensor(x1Scale, "x1Scale") || !CheckEmptyTensor(x2Scale, "x2Scale")) {
        return CaseOption::INVALID;
    }
    if ((x1Scale->GetViewShape().GetDimNum() == DIM_NUM_ONE) ||
        ((x1Scale->GetViewShape().GetDimNum() == DIM_NUM_THREE) &&
        (x1Scale->GetDataType() == ge::DT_FLOAT8_E8M0))) {
        return CaseOption::LOW_ACCURACY_PER_TENSOR_WITH_QUANT_AMAX;
    } else if (x1Scale->GetViewShape().GetDimNum() == DIM_NUM_TWO) {
        //perblock其余判定放在tiling侧进行
        return CaseOption::LOW_ACCURACY_PER_BLOCK;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invalid scene");
        return CaseOption::INVALID;
    }
}

static enum CaseOption CheckCase(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* y,
                                 const aclTensor* amax, const aclTensor* x1Scale, const aclTensor* x2Scale,
                                 const aclTensor* quantScale)
{
    if (CheckType(x1->GetDataType(), INPUT_SUPPORT_TYPE_HIGH_ACCURACY) &&
        CheckType(x2->GetDataType(), INPUT_SUPPORT_TYPE_HIGH_ACCURACY)) {  // 高精度场景
        return CheckHighAccuracyCase(x1, x2, bias, y, amax);
    } else if (CheckType(x1->GetDataType(), INPUT_SUPPORT_TYPE_LOW_ACCURACY) &&
               CheckType(x2->GetDataType(), INPUT_SUPPORT_TYPE_LOW_ACCURACY)) {  // 低精度场景 
        return CheckLowAccuracyCase(x1, x2, bias, y, amax, x1Scale, x2Scale, quantScale);
    } 
    else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Does not match any available case.");
        return CaseOption::INVALID;
    }
}

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
    auto stride = x2->GetViewStrides();

    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDims.data(), viewDimsNum, dataType, stride.data(), offset, format, storageDims.data(),
                           storageDimsNum, x2->GetTensor()->GetAddr());
}

aclnnStatus matmulReduceScatterV2GetWorkSpaceSizeCcuMode(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                       const aclTensor* x1Scale, const aclTensor* x2Scale,
                                                       const aclTensor* quantScale, int64_t blockSize,
                                                       const char* group, const char* reduceOp, int64_t commTurn,
                                                       int64_t streamMode, int64_t groupSize, const char* commMode, aclTensor* output,
                                                       aclTensor* amaxOutOptional, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = CheckParams(x1, x2, bias, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    // 处理空tensor
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGD("MatmulReduceScatter, dealing with empty tensor.");
        // 固定写法，创建OpExecutor
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        if (workspaceSize != nullptr) {
            *workspaceSize = 0;
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    OP_LOGD("X1 is %s.", x1->ToString().GetString());
    OP_LOGD("X2 is %s.", x2->ToString().GetString());

    bool transposeX1 = IsTransposeLastTwoDims(x1);
    bool transposeX2 = IsTransposeLastTwoDims(x2);
    CHECK_RET(CheckShape(x1, x2, transposeX1), ACLNN_ERR_PARAM_INVALID);

    CaseOption caseIndex =
        CheckCase(x1, x2, bias, output, amaxOutOptional, x1Scale, x2Scale, quantScale);
    if (caseIndex >= CaseOption::INVALID) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Does not match any available case.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("Input fit case %u.", static_cast<uint32_t>(caseIndex));

    bool isAmaxOut = false;
    if (caseIndex == CaseOption::LOW_ACCURACY_PER_TENSOR_WITH_QUANT_AMAX) {
        isAmaxOut = true;
    }

    uint32_t rankSize = 0;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = ACLNN_SUCCESS;
    if (transposeX2) {
        // x2转置时将两轴shape调换
        auto transX2 = TransX2Tensor(x2);
        OP_LOGD("X2 dim0 is %ld, dim1 is %ld.", x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1));
        ret = aclnnInnerMatmulReduceScatterV2GetWorkspaceSize(
            x1, transX2, bias, x1Scale, x2Scale, quantScale, group, reduceOp, transposeX1, transposeX2, commTurn, 
            rankSize, blockSize, groupSize, isAmaxOut, yDtype, commMode, output, amaxOutOptional, workspaceSize, executor);
    } else {
        ret = aclnnInnerMatmulReduceScatterV2GetWorkspaceSize(
            x1, x2, bias, x1Scale, x2Scale, quantScale, group, reduceOp, transposeX1, transposeX2, commTurn, 
            rankSize, blockSize, groupSize, isAmaxOut, yDtype, commMode, output, amaxOutOptional, workspaceSize, executor);
    }
    
    OP_LOGD("MatmulReduceScatterV2, end ret %d.", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

static bool MatmulReduceScatterV2IsWeightNZFormat(const aclTensor* x2)
{
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    aclGetFormat(x2, &format);
    if (format == aclFormat::ACL_FORMAT_ND) {
        OP_LOGD("MatmulReduceScatterV2, Recieved weight format is ACL_FORMAT_ND");
    }
    if (format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        OP_LOGD("MatmulReduceScatterV2, Recieved weight format is ACL_FORMAT_FRACTAL_NZ");
        uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
        OP_LOGD("MatmulReduceScatterV2, Shape is %lu", storageDimsNum);
        const uint64_t transdataNzDim = 4U;
        if (storageDimsNum == transdataNzDim) {
            return true;
        }
    }
    return false;
}

aclnnStatus matmulReduceScatterV2GetWorkSpaceSizeAivMode(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                       const aclTensor* x1Scale, const aclTensor* x2Scale,
                                                       const aclTensor* quantScale, int64_t blockSize,
                                                       const char* group, const char* reduceOp, int64_t commTurn,
                                                       int64_t streamMode, int64_t groupSize, const char* commMode, aclTensor* output,
                                                       aclTensor* amaxOutOptional, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    OP_LOGD("aclnnMatmulReduceScatterV2GetWorkspaceSizeAivMode start");
    auto ret_param = CheckParams(x1, x2, bias, streamMode, output);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    bool transposeX1 = IsTransposeLastTwoDims(x1);
    bool transposeX2 = IsTransposeLastTwoDims(x2);
    uint32_t rankSize = 0;
    bool isAmaxOut = false;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    (void)MatmulReduceScatterV2IsWeightNZFormat(x2);
    CHECK_RET(CheckShape(x1, x2, transposeX1), ACLNN_ERR_PARAM_INVALID);
    aclnnStatus ret = aclnnInnerMatmulReduceScatterV2GetWorkspaceSize(x1, x2, bias, x1Scale, x2Scale, quantScale, group, reduceOp,
        transposeX1, transposeX2, commTurn, rankSize, blockSize, groupSize, isAmaxOut, yDtype, commMode, output, amaxOutOptional, workspaceSize, executor);
    OP_LOGD("MatmulReduceScatterV2AivMode, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    return ret;
}

aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                       const aclTensor* x1Scale, const aclTensor* x2Scale,
                                                       const aclTensor* quantScale, int64_t blockSize,
                                                       const char* group, const char* reduceOp, int64_t commTurn,
                                                       int64_t streamMode, int64_t groupSize, const char* commMode, aclTensor* output,
                                                       aclTensor* amaxOutOptional, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    aclnnStatus ret = ACLNN_SUCCESS;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        ret = matmulReduceScatterV2GetWorkSpaceSizeCcuMode(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group, reduceOp, commTurn,
                                                       streamMode, groupSize, commMode, output, amaxOutOptional, workspaceSize, executor);
    } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B || GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        ret = matmulReduceScatterV2GetWorkSpaceSizeAivMode(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group, reduceOp, commTurn,
                                                       streamMode, groupSize, commMode, output, amaxOutOptional, workspaceSize, executor);
    }
    return ret;
}

aclnnStatus aclnnMatmulReduceScatterV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    if ((workspace == nullptr) || (workspaceSize == 0UL)) {
        OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
        return ACLNN_SUCCESS;
    }

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B || GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        if (NnopbaseSetHcclServerType) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }

    aclnnStatus ret = aclnnInnerMatmulReduceScatterV2(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
