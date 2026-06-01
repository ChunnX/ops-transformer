/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_checker.cc
 * \brief
 */

#include "op_host/tiling_type.h"
#include "log/log.h"
#include "quant_batch_matmul_v3_checker.h"
#include "mc2_log.h"

namespace {
constexpr uint64_t MX_GROUP_SIZE = 32;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint64_t PER_BLOCK_SIZE = 128;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t SCALE_THREE_DIM = 3;
constexpr size_t BIAS_THREE_DIM = 3;
constexpr int64_t LAST_AXIS_LIMIT = 65535;

// input index
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t OFFSET_INDEX = 3;
constexpr uint32_t BIAS_INDEX = 4;
constexpr uint32_t PERTOKEN_SCALE_INDEX = 5;
}

namespace optiling {

bool Mc2QuantBatchMatmulV3Checker::LogicXOR(bool cond1, bool cond2) const
{
    uint64_t result = static_cast<uint64_t>(cond1) ^ static_cast<uint64_t>(cond2);
    return static_cast<bool>(result);
}

bool Mc2QuantBatchMatmulV3Checker::CheckABDtypes() const
{
    OP_TILING_CHECK(
        LogicXOR((inputParams_.aDtype == ge::DT_INT8), (inputParams_.bDtype == ge::DT_INT8)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1 and x2",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            "When one input dtype is INT8, then the other input dtype must be INT8, actual x1 is %s, x2 is %s."),
        return false);
    OP_TILING_CHECK(
        LogicXOR((inputParams_.aDtype == ge::DT_HIFLOAT8), (inputParams_.bDtype == ge::DT_HIFLOAT8)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1 and x2",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            "When one input dtype is HIFLOAT8, then the other input dtype must be HIFLOAT8, actual x1 is %s, x2 is %s."),
        return false);
    OP_TILING_CHECK(
        LogicXOR((inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2),
                 (inputParams_.bDtype == ge::DT_FLOAT4_E2M1 || inputParams_.bDtype == ge::DT_FLOAT4_E1M2)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1 and x2",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            "When one input dtype is FLOAT4, then the other input dtype must be FLOAT4, actual x1 is %s, x2 is %s."),
        return false);
    OP_TILING_CHECK(
        LogicXOR((inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2),
                 (inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype == ge::DT_FLOAT8_E5M2)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1 and x2",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            "When one input dtype is FLOAT8, then the other input dtype must be FLOAT8, actual x1 is %s, x2 is %s."),
        return false);
    if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) == nullptr ||
        context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) == nullptr) {
        OP_TILING_CHECK(!(inputParams_.aDtype == ge::DT_INT8 || inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN ||
                            inputParams_.aDtype == ge::DT_FLOAT8_E5M2 || inputParams_.aDtype == ge::DT_HIFLOAT8) ||
                            !(inputParams_.bDtype == ge::DT_INT8 || inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN ||
                                inputParams_.bDtype == ge::DT_FLOAT8_E5M2 || inputParams_.bDtype == ge::DT_HIFLOAT8),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "x1 and x2",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                            "Input dtypes should be INT8/FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, actual are %s and %s."),
                        return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckScaleDtypeWithPertoken() const
{
    bool isFp4 = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
    bool isFp8 = inputParams_.aDtype == ge::DT_FLOAT8_E5M2 || inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN;
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_INT8 && inputParams_.perTokenScaleDtype != inputParams_.scaleDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "pertokenScale and scale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                        "When input is not INT8, pertokenScale and scale should have the same \
dtypes, actual pertokeScale is %s and scale is %s."),
                    return false);
    OP_TILING_CHECK(
        isFp4 && inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "scale",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
            "When input is FLOAT4 with pertokenScale, scale must be FLOAT8_E8M0, actual is %s."),
        return false);
    OP_TILING_CHECK(isFp8 &&
                    (inputParams_.scaleDtype != ge::DT_FLOAT && inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "scale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                        "When input is FLOAT8 with pertokenScale, scale must be FLOAT or FLOAT8_E8M0, actual is %s."),
                    return false);
    OP_TILING_CHECK(!(isFp4 || isFp8)  && inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "pertokenScale",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                        "When input is INT8/HIFLOAT8, pertokenScale should be FLOAT, actual dtype is %s."),
                    return false);

    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype == ge::DT_FLOAT16 && inputParams_.scaleDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                              "When input is INT8 and output is FLOAT16 with pertokenScale, scale dtype should be \
FLOAT, actual dtype is %s."),
        return false);
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype == ge::DT_BF16 &&
                        !(inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                                          "When input is INT8 and output is BFLOAT16 with pertokenScale, scale dtype \
should be FLOAT/BFLOAT16, actual dtype is %s."),
                    return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckScalesDtype() const
{
    bool isFp4 = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
    if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) != nullptr &&
        context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr) {
        OP_TILING_CHECK(!CheckScaleDtypeWithPertoken(),
            OP_LOGE(inputParams_.opName, "Check scale dtype with pertoken failed"), return false);
    } else {
        OP_TILING_CHECK(
            isFp4,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "pertokenScale", "",
                                  "When input is FLOAT4, pertokenScale should not be null."),
            return false);
        OP_TILING_CHECK(
            inputParams_.aDtype != ge::DT_INT8 &&
            !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "scale",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                "When input is not INT8 without pertokenScale, scale dtype should be UINT64/INT64, actual dtype is %s."),
            return false);
        OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 &&
                            (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16) &&
                            !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                                              "When input is INT8 and output is INT8 or FLOAT16 without pertokenScale, \
scale dtype should be UINT64/INT64, actual dtype is %s."),
                        return false);
        OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype == ge::DT_BF16 &&
                            !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_FLOAT ||
                              inputParams_.scaleDtype == ge::DT_BF16 || inputParams_.scaleDtype == ge::DT_INT64),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                                              "When input is INT8 and output is BFLOAT16 without pertokenScale, scale \
dtype should be UINT64/FLOAT/BFLOAT16/INT64, actual dtype is %s."),
                        return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckBiasDtype() const
{
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    OP_TILING_CHECK(
        biasDesc != nullptr && inputParams_.biasDtype != ge::DT_FLOAT && inputParams_.aDtype != ge::DT_INT8,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "bias",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
            "When input dtype is not INT8, bias dtype should be FLOAT, actual dtype is %s."),
        return false);

    OP_TILING_CHECK(
        biasDesc != nullptr && inputParams_.aDtype == ge::DT_INT8 &&
            (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16) &&
            (inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64) &&
            inputParams_.biasDtype != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                              ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                              "When input dtype is INT8, output is INT8/FLOAT16/BFLOAT16, and scale is UINT64/INT64, \
the bias dtype should be INT32, actual dtype is %s."),
        return false);

    OP_TILING_CHECK(biasDesc != nullptr && inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype == ge::DT_FLOAT16 &&
                        inputParams_.scaleDtype == ge::DT_FLOAT &&
                        !(inputParams_.biasDtype == ge::DT_INT32 || inputParams_.biasDtype == ge::DT_FLOAT ||
                          inputParams_.biasDtype == ge::DT_FLOAT16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                          "When input dtype is INT8, output is FLOAT16, and scale is FLOAT, bias dtype \
should be INT32/FLOAT/FLOAT16, actual dtype is %s."),
                    return false);

    OP_TILING_CHECK(biasDesc != nullptr && inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype == ge::DT_BF16 &&
                        (inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16) &&
                        !(inputParams_.biasDtype == ge::DT_INT32 || inputParams_.biasDtype == ge::DT_FLOAT ||
                          inputParams_.biasDtype == ge::DT_BF16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                          "When input dtype is INT8, output is BFLOAT16, and scale is FLOAT/BFLOAT16, \
bias dtype should be INT32/FLOAT/BFLOAT16, actual dtype is %s."),
                    return false);

    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckOutputDtype() const
{
    OP_TILING_CHECK(
        (inputParams_.aDtype == ge::DT_FLOAT8_E5M2 || inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN ||
            inputParams_.aDtype == ge::DT_FLOAT8_E8M0 || inputParams_.aDtype == ge::DT_FLOAT4_E2M1 ||
            inputParams_.aDtype == ge::DT_FLOAT4_E1M2) &&
            (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_HIFLOAT8),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                "When input dtype is FLOAT8/FLOAT4, output dtype must not be INT8/HIFLOAT8, actual is %s."),
        return false);
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_HIFLOAT8 &&
                        (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT8_E5M2 ||
                            inputParams_.cDtype == ge::DT_FLOAT8_E4M3FN),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                            ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                            "When input dtype is HIFLOAT8, output must not be INT8/FLOAT8, actual is %s."),
                    return false);
    bool isFp4 = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
    bool isFp8 = inputParams_.aDtype == ge::DT_HIFLOAT8 || inputParams_.aDtype == ge::DT_FLOAT8_E5M2 ||
                    inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E8M0;
    if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) != nullptr &&
        context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr) {
        OP_TILING_CHECK(
            (isFp8 || isFp4) && !(inputParams_.cDtype == ge::DT_FLOAT || inputParams_.cDtype == ge::DT_FLOAT16 ||
                                    inputParams_.cDtype == ge::DT_BF16),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "output",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                "When input is FLOAT8/FLOAT4 with pertokenScale, output must be FLOAT/FLOAT16/BFLOAT16, actual is %s."),
            return false);
        OP_TILING_CHECK(
            !(isFp8 || isFp4) && !(inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "output",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                "When input is INT8 with pertokenScale, output dtype should be FLOAT16/BFLOAT16 actual is %s."),
            return false);
        OP_TILING_CHECK(
            !(inputParams_.cDtype == ge::DT_FLOAT || inputParams_.cDtype == ge::DT_FLOAT16 ||
                inputParams_.cDtype == ge::DT_BF16),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                    "Output dtype should be FLOAT/FLOAT16/BFLOAT16 with pertokenScale, actual is %s."),
            return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckDtypesInRange() const
{
    static const std::vector<ge::DataType> legalInputDtypes = {
        ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
    static const std::vector<ge::DataType> legalOutputDtypes = {ge::DT_INT8,          ge::DT_FLOAT16,  ge::DT_BF16,
                                                                ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_FLOAT};
    OP_TILING_CHECK(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.aDtype) == legalInputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            "The x1 dtype must be INT8/FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1/FLOAT4_E1M2, actual is %s."),
        return false);
    OP_TILING_CHECK(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.bDtype) == legalInputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x2",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
            "The x2 dtype must be INT8/FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1/FLOAT4_E1M2, actual is %s."),
        return false);
    OP_TILING_CHECK(
        std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), inputParams_.cDtype) == legalOutputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                "Output dtype must be INT8/FLOAT16/BFLOAT16/FLOAT8_E4M3FN/HIFLOAT8/FLOAT, actual is %s."),
        return false);
    auto offsetDesc = context_->GetOptionalInputDesc(OFFSET_INDEX);
    OP_TILING_CHECK(offsetDesc != nullptr && offsetDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "offset",
                                ge::TypeUtils::DataTypeToSerialString(offsetDesc->GetDataType()).c_str(),
                                "The offset dtype should be FLOAT, actual dtype is %s."),
        return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckDtype() const
{
    if (!CheckDtypesInRange()) {
        return false;
    }
    if (!CheckABDtypes()) {
        return false;
    }
    if (!CheckScalesDtype()) {
        return false;
    }
    if (!CheckBiasDtype()) {
        return false;
    }
    if (!CheckOutputDtype()) {
        return false;
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckInputValidInPerblockMode(const gert::Shape& scaleShape,
                                                              const gert::StorageShape *pertokenShape,
                                                              const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    if (!inputParams_.isPerBlock) {
       return true;
    }
    OP_TILING_CHECK(inputParams_.hasBias,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "bias", "", "In perblock mode, bias is not supported yet."),
                    return false);
    auto scaleShapeLen = scaleShape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto &pertoken = pertokenShape->GetStorageShape();
    auto pertokenShapeLen = pertoken.GetDimNum();
    auto x1ShapeLen = x1Shape.GetDimNum();
    if (!CheckDimValidInPerblockMode(x1ShapeLen, x2ShapeLen, pertokenShapeLen, scaleShapeLen)) {
        return false;
    }
    if (!CheckBatchValidInPerblockMode(scaleShape, pertoken, x1Shape, x2Shape)) {
        return false;
    }

    if (!CheckGroupValidInPerblockMode()) {
        return false;
    }

    if (!CheckShapeValidInPerblockMode(scaleShape, pertoken, x1Shape, x2Shape)) {
        return false;
    }

    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckGroupValidInPerblockMode() const
{
    OP_TILING_CHECK(inputParams_.groupSizeM != PER_BLOCK_SIZE && inputParams_.groupSizeM != 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSizeM",
                                          std::to_string(inputParams_.groupSizeM).c_str(),
                                          "When scale dtype is not FLOAT8_E8M0, \
input or infered groupSizeM should be 128 or 1, but now is %lu, \
groupSizeM = (groupSize >> 32) & 0xFFFF."),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeK != PER_BLOCK_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSizeK",
                                          std::to_string(inputParams_.groupSizeK).c_str(),
                                          "When scale dtype is not FLOAT8_E8M0, \
input or infered groupSizeK should be 128, but now is %lu, \
groupSizeK = groupSize & 0xFFFF."),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSizeN != PER_BLOCK_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSizeN",
                                          std::to_string(inputParams_.groupSizeN).c_str(),
                                          "When scale dtype is not FLOAT8_E8M0, \
input or infered groupSizeN should be 128, but now is %lu, \
groupSizeN = (groupSize >> 16) & 0xFFFF."),
                    return false);
  return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckShapeValidInPerblockMode(const gert::Shape& scaleShape,
                                                              const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto  x1ShapeLen = x1Shape.GetDimNum();
    auto  x2ShapeLen = x2Shape.GetDimNum();
    OP_TILING_CHECK(
        (ops::CeilDiv(static_cast<uint64_t>(x2Shape.GetDim(x2ShapeLen - 2)), PER_BLOCK_SIZE) !=
             static_cast<uint64_t>(scaleShape.GetDim(x2ShapeLen - 2)) ||
         ops::CeilDiv(static_cast<uint64_t>(x2Shape.GetDim(x2ShapeLen - 1)), PER_BLOCK_SIZE) !=
             static_cast<uint64_t>(scaleShape.GetDim(x2ShapeLen - 1))),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams_.opName, "scale",
            std::to_string(scaleShape.GetDim(x2ShapeLen - 1)).c_str(),
            "In G-B or B-B quantification, the size of last two dimensions of scale should be both equal to \
the size of last two dimensions of x2 ceildivided by groupSize 128, but now, \
scaleShape[-1] is %ld, x2Shape[-1] is %ld, scaleShape[-2] is %ld, x2Shape[-2] is %ld."),
        return false);
    int64_t x1MIndex = inputParams_.transA ? (x1ShapeLen - 1) : (x1ShapeLen - 2);
    int64_t x1KIndex = inputParams_.transA ? (x1ShapeLen - 2) : (x1ShapeLen - 1);
    uint64_t x1M = x1Shape.GetDim(x1MIndex);
    uint64_t scaleX1M = pertoken.GetDim(x1MIndex);
    uint64_t x1K = x1Shape.GetDim(x1KIndex);
    uint64_t scaleX1K = pertoken.GetDim(x1KIndex);
    OP_TILING_CHECK(
        (ops::CeilDiv(x1M, inputParams_.groupSizeM) != scaleX1M),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams_.opName, "pertokenScale",
            std::to_string(scaleX1M).c_str(),
            "In G-B or B-B quantification, the m dimension size of x1 ceildivided by groupSizeM should be equal to \
the m dimension size of pertokenScale, but now, groupSizeM is %lu, \
m dimension size of pertokenScale is %lu, m dimension size of x1Shape is %lu."),
        return false);
    OP_TILING_CHECK(
        (ops::CeilDiv(x1K, inputParams_.groupSizeK) != scaleX1K),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams_.opName, "pertokenScale",
            std::to_string(scaleX1K).c_str(),
            "In G-B or B-B quantification, the k dimension size of x1 ceildivided by groupSizeK should be equal to \
the k dimension size of pertokenScale, but now, groupSizeK is %lu, \
k dimension size of pertokenScale is %lu, k dimension size of x1Shape is %lu."),
        return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckDimValidInPerblockMode(size_t x1ShapeLen, size_t x2ShapeLen,
                                                            size_t pertokenShapeLen, size_t scaleShapeLen) const
{
    OP_TILING_CHECK(scaleShapeLen != x2ShapeLen,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        inputParams_.opName, "scale", std::to_string(scaleShapeLen).c_str(),
                        "In G-B or B-B quantification, x2 dimension and scale dimension should be equal, \
but x2 dimension is: %zu, scale dimension is: %zu."),
                    return false);
    OP_TILING_CHECK(
        pertokenShapeLen != x1ShapeLen,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            inputParams_.opName, "pertokenScale", std::to_string(pertokenShapeLen).c_str(),
            "In G-B or B-B quantification, x1 dimension and pertoken dimension should be equal, \
but x1 dimension is: %zu, pertoken dimension is: %zu."),
        return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckBatchValidInPerblockMode(const gert::Shape& scaleShape,
                                                              const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                                              const gert::Shape& x2Shape) const
{
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto x1ShapeLen = x1Shape.GetDimNum();
    if (x2ShapeLen > DIM_NUM_TWO) {
        for (size_t i = 0; i < x2ShapeLen - DIM_NUM_TWO; ++i) {
            OP_TILING_CHECK(scaleShape.GetDim(i) != x2Shape.GetDim(i),
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "scale",
                                                  std::to_string(scaleShape.GetDim(i)).c_str(),
                                                  "In G-B or B-B quantification, x2 batch and scale batch should be equal,"
                                                  "but at dimension %zu, x2 batch: %ld, scale batch: %ld."), return false);
        }
    }
    if (x1ShapeLen > DIM_NUM_TWO) {
        for (size_t i = 0; i < x1ShapeLen - DIM_NUM_TWO; ++i) {
            OP_TILING_CHECK(pertoken.GetDim(i) != x1Shape.GetDim(i),
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "pertokenScale",
                                                  std::to_string(pertoken.GetDim(i)).c_str(),
                                                  "In G-B or B-B quantification, x1 batch and pertoken batch should be equal,"
                                                  "but at dimension %zu, x1 batch: %ld, pertoken batch: %ld."), return false);
        }
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::MxPertokenScaleShapeCheck(const gert::StorageShape *pertokenShape) const
{
    auto &pertoken = pertokenShape->GetStorageShape();
    auto pertokenShapeLen = pertoken.GetDimNum();
    OP_TILING_CHECK(pertokenShapeLen != SCALE_THREE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "pertokenScale",
                                          std::to_string(pertokenShapeLen).c_str(),
                                          "In mx quantification, pertokenScale's dimension must be 3, but it is %zu."),
                    return false);
    auto mDimIdx = inputParams_.transA ? 1 : 0;
    auto kDimIdx = inputParams_.transA ? 0 : 1;
    OP_TILING_CHECK(
        static_cast<uint64_t>(pertoken.GetDim(kDimIdx)) != ops::CeilDiv(inputParams_.kSize, MXFP_DIVISOR_SIZE),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "pertokenScale",
                              std::to_string(pertoken.GetDim(kDimIdx)).c_str(),
                              "In mx quantification, when x1's transposition is %s, pertokenScale's dimension %d shape \
should be equal to x1 k dimension size[%lu] ceil div 64, but actual is [%ld]."),
        return false);
    OP_TILING_CHECK(static_cast<uint64_t>(pertoken.GetDim(mDimIdx)) != inputParams_.mSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opName, "pertokenScale",
                        std::to_string(pertoken.GetDim(mDimIdx)).c_str(),
                        "In mx quantification, when x1's transposition is %s, pertokenScale's dimension %d shape \
should be equal to x1 m dimension size[%lu], but actual is [%ld]."),
                    return false);
    OP_TILING_CHECK(static_cast<uint64_t>(pertoken.GetDim(pertokenShapeLen - 1)) != MXFP_MULTI_BASE_SIZE,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "pertokenScale",
                                          std::to_string(pertoken.GetDim(pertokenShapeLen - 1)).c_str(),
                                          "In mx quantification, pertokenScale's last dimension \
should be equal to 2, but actual is [%ld]."),
                    return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::MxScaleShapeCheck(const gert::Shape &scaleShape) const
{
    auto scaleShapeLen = scaleShape.GetDimNum();
    OP_TILING_CHECK(
        scaleShapeLen != SCALE_THREE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "scale",
                                      std::to_string(scaleShapeLen).c_str(),
                              "In mx quantification, scale's dimension must be 3, but it is %zu."),
        return false);
    auto kDimIdx = inputParams_.transB ? 1 : 0;
    auto nDimIdx = inputParams_.transB ? 0 : 1;
    OP_TILING_CHECK(
        static_cast<uint64_t>(scaleShape.GetDim(kDimIdx)) != ops::CeilDiv(inputParams_.kSize, MXFP_DIVISOR_SIZE),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "scale",
                              std::to_string(scaleShape.GetDim(kDimIdx)).c_str(),
                              "In mx quantification, when x2's transposition is %s, scale's dimension %d shape should \
be equal to x2 k dimension size[%lu] ceil div 64, but actual is [%ld]."),
        return false);
    OP_TILING_CHECK(static_cast<uint64_t>(scaleShape.GetDim(nDimIdx)) != inputParams_.nSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "scale",
                                          std::to_string(scaleShape.GetDim(nDimIdx)).c_str(),
                                          "In mx quantification, when x2's transposition is %s, scale's dimension %d \
shape should be equal to x2 n dimension size[%lu], but actual is [%ld]."),
                    return false);
    OP_TILING_CHECK(static_cast<uint64_t>(scaleShape.GetDim(scaleShapeLen - 1)) != MXFP_MULTI_BASE_SIZE,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "scale",
                                          std::to_string(scaleShape.GetDim(scaleShapeLen - 1)).c_str(),
                                          "In mx quantification, scale's dimension 2 shape should be equal to 2, but \
actual is [%ld]."),
                    return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckInputValidInMxPerGroupMode(const gert::Shape& scaleShape,
                                                                const gert::StorageShape *pertokenShape,
                                                                const std::vector<int64_t> &dimValueOfMKN) const
{
    if (!inputParams_.isMxPerGroup) {
        return true;
    }
    OP_TILING_CHECK(inputParams_.groupSizeM != 1ULL || inputParams_.groupSizeN != 1ULL || inputParams_.groupSizeK != 32ULL,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize",
                                          std::to_string(inputParams_.groupSizeM).c_str(),
                                          "When scale dtype is FLOAT8_E8M0, \
input or infered [groupSizeM, groupSizeN ,groupSizeK] should be [1, 1, 32], \
but now [groupSizeM, groupSizeN ,groupSizeK] is [%lu, %lu, %lu], \
groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32."),
                    return false);
    if (pertokenShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "pertokenScale");
        return false;
    }
    OP_TILING_CHECK(!MxPertokenScaleShapeCheck(pertokenShape),
                    OP_LOGE(inputParams_.opName, "The perTokenScale shape check failed."), return false);
    OP_TILING_CHECK(!MxScaleShapeCheck(scaleShape),
                    OP_LOGE(inputParams_.opName, "The scale shape check failed."), return false);
    bool isFp4Input = (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2);
    OP_TILING_CHECK(isFp4Input && (dimValueOfMKN[2] % 2 != 0 || dimValueOfMKN[0] % 2 != 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "x1 and x2",
                                          std::to_string(dimValueOfMKN[0]).c_str(),
                                          "The x1 inner[%ld] or x2 inner[%ld] is not even \
number when input type is FLOAT4."), return false);
    // To determine if x2 k dim ceil div 32 is an even number, divide it by 2 in mxfp4 case
    OP_TILING_CHECK(
        (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ || isFp4Input) &&
            ops::CeilDiv(inputParams_.kSize, MX_GROUP_SIZE) % 2 != 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "x2",
                              std::to_string(ops::CeilDiv(inputParams_.kSize, MX_GROUP_SIZE)).c_str(),
                              "In weight nz case or FLOAT4 case, when scale dtype is FLOAT8_E8M0, x2 k dimension size \
ceil div 32 must be even, actual is %lu."), return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckShapeInRangeForOptionalInputs(const gert::Shape & scaleShape,
                                                                   const gert::StorageShape *biasShape,
                                                                   const gert::StorageShape *pertokenShape,
                                                                   const gert::StorageShape *offsetShape,
                                                                   size_t outDimNum) const
{
    if (biasShape != nullptr) {
        auto biasDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(!(biasDimNum == 1 || biasDimNum == BIAS_THREE_DIM),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "bias", std::to_string(biasDimNum).c_str(),
                                              "The bias dimension should equal to 1 or 3, but it is %zu."), return false);
        OP_TILING_CHECK(biasDimNum == BIAS_THREE_DIM && outDimNum != biasDimNum,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "output", std::to_string(outDimNum).c_str(),
                                              "When bias dimension equal to 3, output dimension must be 3, but it is %zu."), return false);
    }
    if (offsetShape != nullptr) {
        OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype != ge::DT_INT8,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "offset", "", "When inputDtype is INT8 and outputDtype is not \
INT8, offset must be null"), return false);
        OP_TILING_CHECK(inputParams_.cDtype == ge::DT_INT8 && offsetShape->GetStorageShape().GetDimNum() != 1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "offset",
                                                                   std::to_string(offsetShape->GetStorageShape().GetDimNum()).c_str(),
                                                                   "The offset shape should be 1 dimension, but it is %zu"),
                        return false);
        OP_TILING_CHECK(
            inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "offset", "", "When scaleDtype isn't UINT64 or INT64, offset must be null"),
            return false);
    }
    if (pertokenShape != nullptr) {
        auto pertokenDimNum = pertokenShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            !inputParams_.isPerBlock && !inputParams_.isMxPerGroup &&
            (pertokenDimNum != 1 || scaleShape.GetDimNum() != 1),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                inputParams_.opName, "pertokenScale", std::to_string(pertokenDimNum).c_str(),
                "When not G-B, B-B quantification or mx quantification, dimNum of pertokenScale and \
Scale must be 1, actual are [%zu] and [%zu]"), return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const
{
    int64_t mul = 1;
    int64_t mulBound = 1;
    const char* dimName = shapeIdx == X1_INDEX ? "x1" : "x2";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t curDim = shape.GetDim(i);

        OP_TILING_CHECK(i == shape.GetDimNum() - LAST_FIRST_DIM_INDEX && curDim > LAST_AXIS_LIMIT,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, dimName,
                                              std::to_string(curDim).c_str(),
                                              "The last dimension size of %s should not be \
larger than 65535 but actual is %ld"),
                        return false);

        OP_TILING_CHECK(curDim <= 0 || curDim > static_cast<int64_t>(INT32_MAX),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, dimName,
                                              std::to_string(curDim).c_str(),
                                              "Input shape must be within the range of [1,%d], \
actual %zu dimension of %s is %ld."), return false);

        mulBound = curDim * mul;
        OP_TILING_CHECK(mulBound / curDim != mul,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, dimName,
                                              std::to_string(mul).c_str(),
                                              "Multiple of %s shape dims should be in boundary of INT64_MAX."), return false);
        mul = mulBound;
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::BiasShapeCheck(const gert::Shape &biasShape, const gert::Shape &scaleShape,
                                               const gert::StorageShape *pertokenShape) const
{
    auto biasDimNum = biasShape.GetDimNum();
    // 3 dim bias case
    if (biasDimNum == BIAS_THREE_DIM) {
        auto biasFirstDim = static_cast<uint64_t>(biasShape.GetDim(0)); // using index 0 to get bias first dim value
        auto biasSecondDim = static_cast<uint64_t>(biasShape.GetDim(1)); // using index 1 to get bias second dim value
        auto biasThirdDim = static_cast<uint64_t>(biasShape.GetDim(2)); // using index 2 to get bias third dim value
        OP_TILING_CHECK(biasFirstDim != inputParams_.batchC,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                              std::to_string(biasFirstDim).c_str(),
                                              "Input bias 1st dimension shape should equal to batchC[%lu], \
but it is %zu."), return false);
        OP_TILING_CHECK(biasSecondDim != 1,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                              std::to_string(biasSecondDim).c_str(),
                                              "Input bias 2nd dimension shape should equal to 1, but it is %zu."), return false);
        OP_TILING_CHECK(biasThirdDim != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                              std::to_string(biasThirdDim).c_str(),
                                              "Input bias 3rd dimension shape should equal to inputParams_.nSize, \
but it is %zu while n is %lu."), return false);
    }
    if (biasDimNum == 1) {
        OP_TILING_CHECK(static_cast<uint64_t>(biasShape.GetDim(0)) != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                              std::to_string(biasShape.GetDim(0)).c_str(),
                                              "Input bias dimension shape should equal n, but it is %lu while n is %lu."),
                                              return false);
    }
    OP_TILING_CHECK(
        (inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2 ||
         inputParams_.aDtype == ge::DT_HIFLOAT8) &&
            inputParams_.scaleDtype == ge::DT_FLOAT && static_cast<uint64_t>(scaleShape.GetDim(0)) == inputParams_.nSize &&
            inputParams_.nSize != 1UL && pertokenShape != nullptr && inputParams_.perTokenScaleDtype == ge::DT_FLOAT &&
            pertokenShape->GetStorageShape().GetDim(0) == 1L && inputParams_.mSize != 1UL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "bias", "", "Input bias is unsupported."), return false);

    return true;
}

bool Mc2QuantBatchMatmulV3Checker::ExtraInputCheck() const
{
    bool isInt8Input = !(inputParams_.aDtype == ge::DT_HIFLOAT8 || inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN ||
                         inputParams_.aDtype == ge::DT_FLOAT8_E5M2);
    bool isFp4Input = (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2);
    OP_TILING_CHECK(
        inputParams_.isMxPerGroup && (isFp4Input || inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) &&
        (inputParams_.transA || !inputParams_.transB),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x1 and x2",
                              inputParams_.transA ? "true" : "false",
                              "When scale is FLOAT8_E8M0 and inputs are FLOAT4 or weight format is NZ, only support \
trans_a false and trans_b true, actual [%s, %s]."),
        return false);
    OP_TILING_CHECK(!isInt8Input && context_->GetOptionalInputShape(OFFSET_INDEX) != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "offset",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                          "Not support offset with input dtype(%s) yet."),
                    return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::PerTokenDimValueCheck(const gert::Shape &scaleShape,
                                                      const gert::StorageShape *pertokenShape) const
{
    auto &pertoken = pertokenShape->GetStorageShape();
    bool isFp8HiF8 = inputParams_.aDtype == ge::DT_FLOAT8_E5M2 || inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN ||
                     inputParams_.aDtype == ge::DT_HIFLOAT8;
    uint64_t perTokenDim0 = static_cast<uint64_t>(pertoken.GetDim(0));
    OP_TILING_CHECK(
        isFp8HiF8 && !inputParams_.isMxPerGroup && (perTokenDim0 != 1 && perTokenDim0 != inputParams_.mSize) &&
        pertoken.GetDimNum() == 1 && scaleShape.GetDimNum() == 1 && scaleShape.GetDim(0) == 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams_.opName, "pertokenScale",
            std::to_string(pertoken.GetDim(0)).c_str(),
            "When the quantization mode is not mx, when scale's dimension 0 size equal to 1, pertokenScale \
dimension 0 size must be equal to m[%lu] or 1, actual is [%ld]."), return false);
    OP_TILING_CHECK(
        isFp8HiF8 && !inputParams_.isMxPerGroup && inputParams_.mSize != 1 && perTokenDim0 == 1 &&
            pertoken.GetDimNum() == 1 && scaleShape.GetDimNum() == 1 &&
            (scaleShape.GetDim(0) != 1 && static_cast<uint64_t>(scaleShape.GetDim(0)) != inputParams_.nSize),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "scale",
                              std::to_string(scaleShape.GetDim(0)).c_str(),
                              "When the quantization mode is not mx, perTokenScale dimension 0 size equal to \
1, m not equal to 1, scale dimension 0 size must be equal to 1 or n[%lu], actual is [%ld]."),
        return false);
    OP_TILING_CHECK(
        pertoken.GetDimNum() == 1 && static_cast<uint64_t>(perTokenDim0) == inputParams_.mSize &&
        scaleShape.GetDimNum() == 1 &&
        (scaleShape.GetDim(0) != 1 && static_cast<uint64_t>(scaleShape.GetDim(0)) != inputParams_.nSize),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams_.opName, "scale",
            std::to_string(scaleShape.GetDim(0)).c_str(),
            "PerTokenScale dimension 0 size equal to m[%lu], scale dimension 0 size must be equal to 1 or n[%lu], \
actual is [%ld]."), return false);
    OP_TILING_CHECK(perTokenDim0 != inputParams_.mSize && inputParams_.aDtype == ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "pertokenScale",
                                            std::to_string(pertoken.GetDim(0)).c_str(),
                                            "PertokenScale dimension 0 shape should be equal to m[%lu] \
but actual is [%ld]."), return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckDimValue(const gert::Shape & scaleShape, const gert::StorageShape *biasShape,
                                              const gert::StorageShape *pertokenShape,
                                              const gert::StorageShape *offsetShape,
                                              const std::vector<int64_t> &dimValueOfMKN) const
{
    auto x2Inner = dimValueOfMKN[2]; // using index 2 to get x2Inner
    auto x2Outer = dimValueOfMKN[3]; // using index 3 to get x2Outer
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? x2Inner : x2Outer);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "x1 and x2",
                                          std::to_string(inputParams_.kSize).c_str(),
                                          "The size of k dimension of x1[%lu] is not equal to \
the size of k dimension of x2[%lu]"),
                    return false);
    if (biasShape != nullptr && !BiasShapeCheck(biasShape->GetStorageShape(), scaleShape, pertokenShape)) {
        return false;
    }
    if (offsetShape != nullptr) {
        OP_TILING_CHECK(inputParams_.cDtype == ge::DT_INT8 && !(offsetShape->GetStorageShape().GetDim(0) == 1 ||
                         static_cast<uint64_t>(offsetShape->GetStorageShape().GetDim(0)) == inputParams_.nSize),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "offset",
                                                                   std::to_string(offsetShape->GetStorageShape().GetDim(0)).c_str(),
                                                                   "The offset dimension value must be 1 or n[%lu], \
but it is %ld."),
                        return false);
    }
    auto scaleShapeLen = scaleShape.GetDimNum();
    OP_TILING_CHECK(
        scaleShapeLen != 1 &&
            (inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64 || pertokenShape == nullptr),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            inputParams_.opName, "scale", std::to_string(scaleShapeLen).c_str(),
            "When scaleDtype is UINT64 or INT64 or pertokenScale is None, scale's dimension must be 1, actually is : %zu"), return false);
    if (pertokenShape != nullptr) {
        OP_TILING_CHECK(!PerTokenDimValueCheck(scaleShape, pertokenShape),
                        OP_LOGE(inputParams_.opName, "Check perTokenShape failed"), return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Checker::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                                           const gert::StorageShape *biasShape,
                                           const gert::StorageShape *pertokenShape,
                                           const std::vector<int64_t> &dimValueOfMKN) const
{
    auto &x1Shape = *mandtoryShape[0]; // using index 0 to get x1Shape
    auto &x2Shape = *mandtoryShape[1]; // using index 1 to get x2Shape
    auto &scaleShape = *mandtoryShape[2]; // using index 2 to get scaleShape
    auto offsetShape = context_->GetOptionalInputShape(OFFSET_INDEX);
    size_t outDimNum = std::max(x1Shape.GetDimNum(), x2Shape.GetDimNum());
    if (!CheckShapeInRangeForOptionalInputs(scaleShape, biasShape, pertokenShape, offsetShape, outDimNum) ||
        !CheckDimValue(scaleShape, biasShape, pertokenShape, offsetShape, dimValueOfMKN) ||
        !CheckShapeInBoundary(x1Shape, X1_INDEX) || !CheckShapeInBoundary(x2Shape, X2_INDEX)){
        return false;
    }
    if ((!CheckInputValidInPerblockMode(scaleShape, pertokenShape, x1Shape, x2Shape) ||
         !CheckInputValidInMxPerGroupMode(scaleShape, pertokenShape, dimValueOfMKN))) {
        return false;
    }
    if (!ExtraInputCheck()) {
        return false;
    }
    return true;
}
}  // namespace optiling