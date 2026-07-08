/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_weight_quant_matmul_all_reduce.h"

#include "aclnnInner_matmul_all_reduce.h"
#include "opdev/tensor_view_utils.h"
#include "matmul_all_reduce_util.h"
#include "mc2_comm_utils.h"
#include "mc2_log_compat.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t ANTIQUANT_GROUP_SIZE_MIN_VALUE = 32;

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void* executor, const size_t irIndex);

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_310P = {DataType::DT_FLOAT16};
// op::DataType不支持INT4，使用DateType
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT = {DataType::DT_INT8, DataType::DT_INT4};

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale, const aclTensor* output)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* scale, const aclTensor* offset,
    const aclTensor* x3, const aclTensor* output)
{
    const auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    const auto dtypeSupportList =
        (npuArch == NpuArch::DAV_2002) ? DTYPE_SUPPORT_LIST_310P : DTYPE_SUPPORT_LIST;

    const std::initializer_list<op::DataType> dtypeSupportListQuantA5 = {
        DataType::DT_INT8, DataType::DT_INT4, DataType::DT_FLOAT8_E4M3FN, DataType::DT_HIFLOAT8};

    const std::initializer_list<op::DataType> dtypeSupportListBiasA5 = {
        DataType::DT_FLOAT16, DataType::DT_BF16};

    const auto x2DtypeSupportList =
        (npuArch == NpuArch::DAV_3510) ? dtypeSupportListQuantA5 : DTYPE_SUPPORT_LIST_QUANT;

    const auto biasDtypeSupportList =
        (npuArch == NpuArch::DAV_3510) ? dtypeSupportListBiasA5 : dtypeSupportList;
    // 检查x1、x2、bias、scale、offset、x3、output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, dtypeSupportList, return false);
    // 对于量化来说，x2只为INT8/INT4
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, x2DtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, dtypeSupportList, return false);
    // 检查x1和scale的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x1, scale, return false);
    // 检查x1和output的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);

    // 检查bias、offset、x3的数据类型是否在算子的支持列表内
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, biasDtypeSupportList, return false);
            // 检查x1和bias的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(bias, x1, return false);
    }
    if (offset != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(offset, dtypeSupportList, return false);
        // 检查scale和offset的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(scale, offset, return false);
    }
    if (x3 != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x3, dtypeSupportList, return false);
        // 检查x1和x3的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(x3, x1, return false);
    }

    return true;
}

// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(const char* reduceOp, int64_t streamMode, int64_t antiquantGroupSize, const aclTensor* x1)
{
    if (strcmp(reduceOp, REDUCE_OP_SUM) != 0) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnWeightQuantMatmulAllReduce", "reduceOp", reduceOp, "\"sum\"");
        return false;
    }
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnWeightQuantMatmulAllReduce", "streamMode", std::to_string(streamMode).c_str(), "\"1\"");
        return false;
    }

    const size_t x1Len = x1->GetViewShape().GetDimNum();
    int64_t kLen = x1->GetViewShape().GetDim(x1Len - 1);
    // if kLen equals to 0, no need to check antiquantGroupSize for per-group
    if (kLen == 0) {
        OP_LOGD("WeightQuantMatmulAllReduce, k value is equal to 0.");
        return true;
    }

    // antiquantGroupSize为默认值0或者antiquantGroupSize%32 == 0并且antiquantGroupSize在[32, min(k-1,INT_MAX)]范围内
    if (antiquantGroupSize == 0) {
        return true;
    }
    if (antiquantGroupSize % ANTIQUANT_GROUP_SIZE_MIN_VALUE != 0 ||
        antiquantGroupSize < ANTIQUANT_GROUP_SIZE_MIN_VALUE ||
        antiquantGroupSize > std::min(static_cast<int32_t>(kLen - 1), INT32_MAX)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnWeightQuantMatmulAllReduce", "antiquantGroupSize",
            std::to_string(antiquantGroupSize).c_str(),
            std::string("The value of antiquantGroupSize must be in range [" + std::to_string(ANTIQUANT_GROUP_SIZE_MIN_VALUE) +
                ", min(" + std::to_string(kLen - 1) + ", INT_MAX)]").c_str());
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* scale, const aclTensor* offset,
    const aclTensor* x3, const aclTensor* output, int64_t antiquantGroupSize)
{
    bool isWeightNZ = MatmulAllReduceIsWeightNZFormat(x2);
    OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
    OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
    OP_LOGD("MatmulAllReduce, CheckShape isWeightNZ is %d", isWeightNZ);
    if (isWeightNZ) {
        return true;
    }
    uint64_t weightDim = TWO_DIMS;
    OP_LOGD("MatmulAllReduce, CheckShape weightDim is %lu", weightDim);
    // x2的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2可能为4维
    if (x2->GetViewShape().GetDimNum() != weightDim) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnWeightQuantMatmulAllReduce", "x2",
            (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
            ("The shape of x2 must be " + std::to_string(weightDim) + "D.").c_str());
        return false;
    }
    uint64_t x2Dim0 = QuantMatmulAllReduceIsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    uint64_t x2Dim1 = QuantMatmulAllReduceIsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x2ShapeStr = op::ToString(x2->GetViewShape());
    QuantMatmulAllReduceProcessTransposedX2(x2, x2Dim0, x2Dim1, x2ShapeStr);
    // 仅支持x2矩阵转置，x1不支持转置, x1.GetDimNum(1) == x2.GetDimNum(0)
    const size_t x1Len = x1->GetViewShape().GetDimNum();
    OP_LOGD("MatmulAllReduce, CheckShape x1Len is %lu", x1Len);
    uint64_t x1Dim1 = x1->GetViewShape().GetDim(x1Len - 1);
    if (x1Dim1 != x2Dim0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnWeightQuantMatmulAllReduce", "x1",
            op::ToString(x1->GetViewShape()).GetString(),
            std::string("The shape [last dim] of x1 must be equal to first dim of x2, but x2 shape is " + std::string(x2ShapeStr.GetString())).c_str());
        return false;
    }

    // output的最后一维与x2的最后一维相同
    op::Shape outShape = x1->GetViewShape();
    outShape.SetDim(x1Len - 1, x2Dim1);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outShape, return false);

    // 判断output是否为空tensor
    if (output->IsEmpty()) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnWeightQuantMatmulAllReduce", "output",
            op::ToString(output->GetViewShape()).GetString(), "non-empty tensor");
        return false;
    }

    // x1 shape [s,m,k], x2 shape [k,n], output shape [s,m,n], bias shape [n]
    if (bias != nullptr) {
        if (bias->GetViewShape().GetDimNum() != ONE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnWeightQuantMatmulAllReduce", "bias",
                (std::to_string(bias->GetViewShape().GetDimNum()) + "D").c_str(),
                "The shape of bias must be 1D.");
            return false;
        }
        op::Shape biasShape;
        biasShape.AppendDim(output->GetViewShape().GetDim(x1Len - 1));
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(bias, biasShape, return false);
    }

    // x3 shape [s,m,n]
    if (x3 != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(x3, output, return false);
    }

    int64_t kValue = x1->GetViewShape().GetDim(x1Len - 1);
    // if kValue equals to 0, no need to check antiquantScale/antiquantOffset for per-group
    if (kValue == 0) {
        return true;
    }
    // scale和offset为per-tensor则shape为[1]，为per-channel则shape为[n]或者[1,n]
    OP_CHECK_MIN_DIM(scale, ONE_DIM, return false);
    OP_CHECK_MAX_DIM(scale, TWO_DIMS, return false);
    if (!IsAntiquantScaleShapeValid(scale, x1, output, antiquantGroupSize)) {
        if (antiquantGroupSize == 0) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnWeightQuantMatmulAllReduce", "antiquantScale",
                op::ToString(scale->GetViewShape()).GetString(),
                std::string("[1] or [n] or [1,n], last dim should be " +
                    std::to_string(output->GetViewShape().GetDim(x1Len - 1)) + " or 1").c_str());
        }
        if (antiquantGroupSize != 0) {
            size_t kValueGroup = CeilDiv(x1->GetViewShape().GetDim(x1Len - 1), antiquantGroupSize);
            OP_LOGE_FOR_INVALID_SHAPE("aclnnWeightQuantMatmulAllReduce", "antiquantScale",
                op::ToString(scale->GetViewShape()).GetString(),
                std::string("[" + std::to_string(kValueGroup) + "," +
                    std::to_string(output->GetViewShape().GetDim(x1Len - 1)) + "] for per-group calculation").c_str());
        }
        return false;
    }
    if (offset != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(offset, scale, return false);
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffset, const aclTensor* x3, const char* reduceOp, int64_t streamMode,
    const aclTensor* output, int64_t antiquantGroupSize)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, antiquantScale, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x1, x2, bias, antiquantScale, antiquantOffset, x3, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(CheckAttr(reduceOp, streamMode, antiquantGroupSize, x1), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出shape
    CHECK_RET(
        CheckShape(x1, x2, bias, antiquantScale, antiquantOffset, x3, output, antiquantGroupSize),
        ACLNN_ERR_PARAM_INVALID);

    // 5.【A2】检查x2、antiquantScale、antiquantOffset矩阵在非转置的情况下是否连续
    if (op::GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B &&
        !MatmulAllReduceCheckValidEmptyTensor(x1, x2)) {
        CHECK_RET(MatmulAllReduceCheckValidContiguous(x2, "x2"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(MatmulAllReduceCheckValidContiguous(antiquantScale, "antiquantScale"), ACLNN_ERR_PARAM_INVALID);
        if (antiquantOffset != nullptr) {
            CHECK_RET(MatmulAllReduceCheckValidContiguous(antiquantOffset, "antiquantOffset"), ACLNN_ERR_PARAM_INVALID);
        }
    }
    // 6. 检查连续性
    CHECK_RET(CheckContiguousAffine(x2, antiquantScale, antiquantOffset,
        "aclnnWeightQuantMatmulAllReduce"), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffset, const aclTensor* x3, const char* group, const char* reduceOp, int64_t commTurn,
    int64_t streamMode, int64_t antiquantGroupSize, const aclTensor* output, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = CheckParams(
        x1, x2, bias, antiquantScale, antiquantOffset, x3, reduceOp, streamMode, output, antiquantGroupSize);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    const size_t x1DimNum = x1->GetOriginalShape().GetDimNum();
    if (x1DimNum < 1 || x1DimNum > THREE_DIMS) {
        return ACLNN_ERR_INNER;
    }

    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2) || QuantMatmulAllReduceIsAclnnPreTransposed(x2);
    aclTensor* pertokenScale = nullptr;
    aclTensor* commQuantScale1 = nullptr;
    aclTensor* commQuantScale2 = nullptr;
    aclTensor* dequantScale = nullptr;
    if(x2->GetTensor() == nullptr){
        OP_LOGE_WITH_INVALID_INPUT("aclnnWeightQuantMatmulAllReduce", "x2");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto copyX2 = QuantMatmulAllReduceCopyTensor(x2);
    auto tempX2 = MatmulAllReduceIsWeightNZFormat(x2) ? copyX2 : x2;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    const char* commModePtr = "ai_cpu";
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, tempX2, bias, x3, antiquantScale, antiquantOffset, dequantScale, pertokenScale, commQuantScale1,
        commQuantScale2, const_cast<char*>(group), const_cast<char*>(reduceOp),
        transposeX1, transposeX2, commTurn, antiquantGroupSize, 0, yDtype, 0, const_cast<char*>(commModePtr), output,
        workspaceSize, executor);
    OP_LOGD("WeightQuantMatmulAllReduce, aclnnMatmulAllReduceGetWorkspaceSize ret %d", ret);
#ifdef MC2_UT
    ret = 0;
#endif
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr && executor != nullptr && *executor != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 6U); // 6 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 8U); // 8 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 9U); // 9 is input irIndex
        }
    }
    OP_LOGD("WeightQuantMatmulAllReduce, end ret %d", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnWeightQuantMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    aclnnStatus ret = LaunchInnerMatmulAllReduceSimple(
        workspace, workspaceSize, executor, stream, "aclnnWeightQuantMatmulAllReduce");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif