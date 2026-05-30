/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quant_matmul_all_reduce_v3.cpp
 * \brief
 */
#include "aclnn_quant_matmul_all_reduce_v3.h"

#include "aclnnInner_matmul_all_reduce.h"
#include "matmul_all_reduce_util.h"
#include "mc2_comm_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void* executor, const size_t irIndex);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORTED_LIST_BIAS = {op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> DTYPE_SUPPORTED_LIST_QUANT = {op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> DTYPE_SUPPORTED_LIST_DEQUANT = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORTED_LIST_PERTOKEN = {op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORTED_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORTED_LIST_DEQUANT_310P = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> DTYPE_SUPPORTED_LIST_310P = {op::DataType::DT_FLOAT16};

// 检查入参是否为nullptr
static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2Orig, const aclTensor* dequantScale, const aclTensor* output)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2Orig, return false);
    OP_CHECK_NULL(dequantScale, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2Orig, const aclTensor* bias, const aclTensor* dequantScale,
    const aclTensor* pertokenScale, const aclTensor* x3O, const aclTensor* commQuantScale1Optional,
    const aclTensor* commQuantScale2Optional, const aclTensor* output)
{
    const auto& dequantDtypeList = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002 ?
                                          DTYPE_SUPPORTED_LIST_DEQUANT_310P :
                                          DTYPE_SUPPORTED_LIST_DEQUANT;

    const auto& outDtypeList = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002 ?
                                      DTYPE_SUPPORTED_LIST_310P :
                                      DTYPE_SUPPORTED_LIST;

    // 检查x1、x2Orig、bias、scale、bOff、x3O、output的数据类型是否在算子的支持列表内
    // 对于量化来说，x1,x2Orig只为INT8
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORTED_LIST_QUANT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Orig, DTYPE_SUPPORTED_LIST_QUANT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(dequantScale, dequantDtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, outDtypeList, return false);
    // 检查bias、bOff、x3O的数据类型是否在算子的支持列表内
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORTED_LIST_BIAS, return false);
    }
    if (x3O != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x3O, outDtypeList, return false);
        // 检查x3O和output的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(x3O, output, return false);
    }
    // 检查x1和x2Orig的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x1, x2Orig, return false);
    // BF16场景，dequantScale的dtype与output一致，为BF16
    if (output->GetDataType() == op::DataType::DT_BF16 || dequantScale->GetDataType() == op::DataType::DT_BF16) {
        OP_CHECK_DTYPE_NOT_SAME(dequantScale, output, return false);
    }
    // pertokenScale必须为float32
    if (pertokenScale != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(pertokenScale, DTYPE_SUPPORTED_LIST_PERTOKEN, return false);
        // FP16场景，dequantScale的dtype与pertokenScale一致，为float32
        if (output->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(dequantScale, pertokenScale, return false);
        }
    }
    if (commQuantScale1Optional != nullptr && commQuantScale2Optional != nullptr) {
        // 检查commQuantScale1 commQuantScale2的数据类型是否在算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(commQuantScale1Optional, DTYPE_SUPPORTED_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(commQuantScale2Optional, DTYPE_SUPPORTED_LIST, return false);
        // 检查commQuantScale1Optional commQuantScale2Optional和output的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(commQuantScale1Optional, output, return false);
        OP_CHECK_DTYPE_NOT_SAME(commQuantScale2Optional, output, return false);
    }
    return true;
}

// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(const char* rOp, int64_t streamMode)
{
    if (strcmp(rOp, REDUCE_OP_SUM) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected rOp to be sum, but got %s.", rOp);
        return false;
    }
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected streamMode to be 1, but got %ld.", streamMode);
        return false;
    }
    return true;
}

static void ProcessTransposedX2(const aclTensor* x2Orig, uint64_t& x2OrigDim0, uint64_t& x2OrigDim1, ge::AscendString& x2OrigShapeStr)
{
    op::Shape x2OrigViewShape = x2Orig->GetViewShape();
    x2OrigViewShape.SetDim(0, x2OrigDim0);
    x2OrigViewShape.SetDim(1, x2OrigDim1);
    if (QuantMatmulAllReduceIsAclnnPreTransposed(x2Orig)) {
        x2OrigShapeStr = op::ToString(x2OrigViewShape);
    }
    OP_LOGD("MatmulAllReduce, x2Orig view shape is %s", x2OrigShapeStr.GetString());
}

static bool CheckShape(
    const aclTensor* x1, const aclTensor* x2Orig, const aclTensor* bias, const aclTensor* dequantScale,
    const aclTensor* pertokenScale, const aclTensor* commQuantScale1Optional, const aclTensor* commQuantScale2Optional,
    const aclTensor* x3O, const aclTensor* output)
{
    bool isWeightNZ = MatmulAllReduceIsWeightNZFormat(x2Orig);

    OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
    OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
    if (isWeightNZ) {
        return true;
    }
    // x2Orig的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2Orig可能为4维
    OP_CHECK_WRONG_DIMENSION(x2Orig, TWO_DIMS, return false);

    uint64_t x2OrigDim0 = QuantMatmulAllReduceIsAclnnPreTransposed(x2Orig) ? x2Orig->GetViewShape().GetDim(1) : x2Orig->GetViewShape().GetDim(0);
    uint64_t x2OrigDim1 = QuantMatmulAllReduceIsAclnnPreTransposed(x2Orig) ? x2Orig->GetViewShape().GetDim(0) : x2Orig->GetViewShape().GetDim(1);
    auto x2OrigShapeStr = op::ToString(x2Orig->GetViewShape());
    ProcessTransposedX2(x2Orig, x2OrigDim0, x2OrigDim1, x2OrigShapeStr);
    // 仅支持x2Orig矩阵转置，x1不支持转置, x1.GetDimNum(1) == x2Orig.GetDimNum(0)
    const size_t x1Len = x1->GetViewShape().GetDimNum();
    if (static_cast<uint64_t>(x1->GetViewShape().GetDim(x1Len - 1)) != x2OrigDim0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected last dim of x1 to be equal to first dim of x2Orig, but got x1 shape: %s, x2Orig shape: %s.",
            op::ToString(x1->GetViewShape()).GetString(), x2OrigShapeStr.GetString());
        return false;
    }

    // output的最后一维与x2Orig的最后一维相同
    op::Shape outShape = x1->GetViewShape();
    outShape.SetDim(x1Len - 1, x2OrigDim1);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outShape, return false);

    // x1 shape [s,m,k], x2Orig shape [k,n], output shape [s,m,n], bias shape [n]
    if (bias != nullptr) {
        OP_CHECK_WRONG_DIMENSION(bias, ONE_DIM, return false);
        op::Shape biasShape;
        biasShape.AppendDim(output->GetViewShape().GetDim(x1Len - 1));
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(bias, biasShape, return false);
    }

    // x3O shape [s,m,n]
    if (x3O != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(x3O, output, return false);
    }

    // scale和bOff为per-tensor则shape为[1]，为per-channel则shape为[n]或者[1,n]
    OP_CHECK_MIN_DIM(dequantScale, ONE_DIM, return false);
    OP_CHECK_MAX_DIM(dequantScale, TWO_DIMS, return false);
    const size_t scaleLen = dequantScale->GetViewShape().GetDimNum();
    if (!(scaleLen == DIM_LEN_ONE && (dequantScale->GetViewShape().GetDim(0) == NUM_ONE ||
                                      dequantScale->GetViewShape().GetDim(0) == outShape.GetDim(x1Len - 1))) &&
        !(scaleLen == DIM_LEN_TWO && dequantScale->GetViewShape().GetDim(0) == NUM_ONE &&
          dequantScale->GetViewShape().GetDim(1) == outShape.GetDim(x1Len - 1))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected dequantScale be [1] or [n] or [1,n], last dim of dequantScale should be %ld or 1, "
            "but got dequantScale shape: %s.",
            output->GetViewShape().GetDim(x1Len - 1), op::ToString(dequantScale->GetViewShape()).GetString());
        return false;
    }

    // pertokenScale的shape为[m]
    if (pertokenScale != nullptr) {
        OP_CHECK_MIN_DIM(pertokenScale, ONE_DIM, return false);
        OP_CHECK_MAX_DIM(pertokenScale, ONE_DIM, return false);
        const size_t pertokenScaleLen = pertokenScale->GetViewShape().GetDimNum();
        int64_t x1M = 1;
        for (size_t dim = 0; dim < x1Len - 1; dim++) {
            x1M *= x1->GetViewShape().GetDim(dim);
        }
        if (!(pertokenScaleLen == DIM_LEN_ONE && pertokenScale->GetViewShape().GetDim(0) == x1M)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Expected pertokenScale be [%ld], but got pertokenScale shape: %s.", x1M,
                op::ToString(pertokenScale->GetViewShape()).GetString());
            return false;
        }
    }

    // commQuantScale1、commQuantScale2必须同时存在，且shape为[1，n]或[n]
    if ((commQuantScale1Optional != nullptr && commQuantScale2Optional == nullptr) ||
        (commQuantScale1Optional == nullptr && commQuantScale2Optional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "comm_quantScale_1 and comm_quant_scale_2 should both not be nullptr.");
        return false;
    }
    if (commQuantScale1Optional != nullptr && commQuantScale2Optional != nullptr) {
        OP_CHECK_MIN_DIM(commQuantScale1Optional, ONE_DIM, return false);
        OP_CHECK_MAX_DIM(commQuantScale1Optional, TWO_DIMS, return false);
        OP_CHECK_MIN_DIM(commQuantScale2Optional, ONE_DIM, return false);
        OP_CHECK_MAX_DIM(commQuantScale2Optional, TWO_DIMS, return false);
        const size_t commQuantScale1OptionalLen = commQuantScale1Optional->GetViewShape().GetDimNum();
        const size_t commQuantScale2OptionalLen = commQuantScale2Optional->GetViewShape().GetDimNum();
        if (commQuantScale1OptionalLen != commQuantScale2OptionalLen) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected commQuantScale1 and commQuantScale2 have the same shape, "
                "but got commQuantScale1 shape: %s, commQuantScale2 shape: %s.",
                op::ToString(commQuantScale1Optional->GetViewShape()).GetString(),
                op::ToString(commQuantScale2Optional->GetViewShape()).GetString());
            return false;
        }
        uint64_t commQuantScale1Dim0 = commQuantScale1Optional->GetViewShape().GetDim(0);
        uint64_t commQuantScale2Dim0 = commQuantScale2Optional->GetViewShape().GetDim(0);
        uint64_t commQuantScale1Dim1 = 0;
        uint64_t commQuantScale2Dim1 = 0;
        if (commQuantScale1OptionalLen == TWO_DIMS && commQuantScale2OptionalLen == TWO_DIMS) {
            commQuantScale1Dim1 = commQuantScale1Optional->GetViewShape().GetDim(1);
            commQuantScale2Dim1 = commQuantScale2Optional->GetViewShape().GetDim(1);
        }

        if ((!(commQuantScale1OptionalLen == ONE_DIM && commQuantScale1Dim0 == x2OrigDim1 &&
               commQuantScale2Dim0 == x2OrigDim1)) &&
            (!(commQuantScale1OptionalLen == TWO_DIMS && commQuantScale1Dim0 == 1 && commQuantScale2Dim0 == 1 &&
               commQuantScale1Dim1 == x2OrigDim1 && commQuantScale2Dim1 == x2OrigDim1))) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected commQuantScale1 and commQuantScale2 be [n] or [1,n], last dim should be %ld, "
                "but got commQuantScale1 shape: %s, commQuantScale2 shape: %s.",
                x2OrigDim1, op::ToString(commQuantScale1Optional->GetViewShape()).GetString(),
                op::ToString(commQuantScale2Optional->GetViewShape()).GetString());
            return false;
        }
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2Orig, const aclTensor* bias, const aclTensor* dequantScale,
    const aclTensor* pertokenScale, const aclTensor* x3O, const aclTensor* commQuantScale1Optional,
    const aclTensor* commQuantScale2Optional, const char* rOp, int64_t streamMode, const aclTensor* output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2Orig, dequantScale, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(
        CheckDtypeValid(
            x1, x2Orig, bias, dequantScale, pertokenScale, x3O, commQuantScale1Optional, commQuantScale2Optional, output),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(CheckAttr(rOp, streamMode), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出shape
    CHECK_RET(
        CheckShape(
            x1, x2Orig, bias, dequantScale, pertokenScale, commQuantScale1Optional, commQuantScale2Optional, x3O, output),
        ACLNN_ERR_PARAM_INVALID);

    // 5.【A2】检查x2Orig矩阵非连续合法性
    if (op::GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B &&
        !MatmulAllReduceCheckValidEmptyTensor(x1, x2Orig)) {
        CHECK_RET(MatmulAllReduceCheckValidContiguous(x2Orig, "x2Orig"), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static const aclTensor* CopyTensor(const aclTensor* x2Orig)
{
    uint64_t stDimsCnt = x2Orig->GetStorageShape().GetDimNum();
    std::vector<int64_t> stDims(stDimsCnt);
    for (size_t i = 0; i < stDimsCnt; i++) {
        stDims[i] = x2Orig->GetStorageShape().GetDim(i);
    }
    OP_LOGD("MatmulAllReduce, CopyTensor stDimsCnt is %lu.", stDimsCnt);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2Orig, &dataType);
    auto strideLen = x2Orig->GetViewStrides();
    auto bOff = x2Orig->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    auto stgFormat = ge::GetPrimaryFormat(x2Orig->GetStorageFormat());
    if (stgFormat == Format::FORMAT_ND) {
        OP_LOGD("MatmulAllReduce, CopyTensor format is ACL_FORMAT_ND");
        format = aclFormat::ACL_FORMAT_ND;
    } else if (stgFormat == Format::FORMAT_FRACTAL_NZ) {
        format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    return aclCreateTensor(
        stDims.data(), stDimsCnt, dataType, strideLen.data(), bOff, format, stDims.data(), stDimsCnt,
        x2Orig->GetTensor()->GetAddr());
}

aclnnStatus aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2Orig, const aclTensor* biasOptional, const aclTensor* x3OOptional,
    const aclTensor* dequantScale, const aclTensor* pertokenScaleOptional, const aclTensor* commQuantScale1Optional,
    const aclTensor* commQuantScale2Optional, const char* group, const char* rOp, int64_t commTurn,
    int64_t streamMode, const aclTensor* output, uint64_t* calcWorkSpace, aclOpExecutor** executor)
{
    uint64_t tsStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = CheckParams(
        x1, x2Orig, biasOptional, dequantScale, pertokenScaleOptional, x3OOptional, commQuantScale1Optional,
        commQuantScale2Optional, rOp, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // dequantScale转为uint64
    auto dequant = const_cast<aclTensor*>(dequantScale);
    if (dequant == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmulAllReduce, dequant is nullptr.");
        return ACLNN_ERR_INNER;
    }
    if (dequant->GetDataType() == op::DataType::DT_INT64) {
        dequant->SetDataType(op::DataType::DT_UINT64);
    }

    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2Orig) || QuantMatmulAllReduceIsAclnnPreTransposed(x2Orig);
    aclTensor* scale = nullptr;
    aclTensor* bOff = nullptr;
    int64_t antiquantGroupSize = 0;
    auto tempX2 = x2Orig;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2002 && MatmulAllReduceIsWeightNZFormat(x2Orig)) {
        if(x2Orig->GetTensor() == nullptr){
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Tensor of x2Orig is null.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        tempX2 = CopyTensor(x2Orig);
    }
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, tempX2, biasOptional, x3OOptional, scale, bOff, dequant, pertokenScaleOptional, commQuantScale1Optional,
        commQuantScale2Optional, const_cast<char*>(group), const_cast<char*>(rOp),
        transposeX1, transposeX2, commTurn, antiquantGroupSize, 0, yDtype, 0,
        output, calcWorkSpace, executor);

    OP_LOGI("Group %s, reduce op %s, trans flag %u %u, ret %d.", group, rOp, transposeX1, transposeX2, ret);
#ifdef MC2_UT
    ret = 0;
#endif
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 4U); // 4 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
        }
    }
    OP_LOGD("QuantMatmulAllReduce, end ret %d", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(tsStamp, dfxId);
    return ret;
}

aclnnStatus aclnnQuantMatmulAllReduceV3(
    void* workspace, uint64_t calcWorkSpace, aclOpExecutor* executor, const aclrtStream stream)
{
    uint64_t tsStamp = NnopbaseMsprofSysTime();
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            uint8_t commType = Mc2Comm::GetCommModeFromEnv();
            if (commType == Mc2Comm::COMM_MODE_AICPU) {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, calcWorkSpace, executor, stream);
    OP_LOGD("QuantMatmulAllReduce, aclnnQuantMatmulAllReduceV3 ret %d", ret);

    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "QuantMatmulAllReduce, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }

    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(tsStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
