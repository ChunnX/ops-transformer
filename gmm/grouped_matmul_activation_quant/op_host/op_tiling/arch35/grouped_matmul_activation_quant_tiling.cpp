/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_activation_quant_tiling.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "../../../op_kernel/arch35/grouped_matmul_activation_quant_tiling_key.h"

using namespace Ops::Transformer::OpTiling;
using namespace optiling::GmmConstant;

namespace optiling {
namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GROUP_LIST_INDEX = 1;
constexpr uint32_t WEIGHT_INDEX = 2;
constexpr uint32_t WEIGHT_SCALE_INDEX = 3;
constexpr uint32_t X_SCALE_INDEX = 5;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;
constexpr uint32_t ATTR_INDEX_ACTIVATION_TYPE = 0;
constexpr uint32_t ATTR_INDEX_TRANS_W = 1;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 2;
constexpr uint32_t ATTR_INDEX_TUNING_CONFIG = 3;
constexpr uint32_t ATTR_INDEX_QUANT_MODE = 4;
constexpr uint32_t ATTR_INDEX_Y_DTYPE = 5;
constexpr uint32_t ATTR_INDEX_ROUND_MODE = 6;
constexpr uint32_t ATTR_INDEX_SCALE_ALG = 7;
constexpr uint32_t ATTR_INDEX_DST_TYPE_MAX = 8;
constexpr uint32_t WEIGHT_NZ_DIM_NUM = 5;
constexpr uint32_t X_SCALE_M_DIM_INDEX = 0;
constexpr uint32_t X_SCALE_K_BLOCK_DIM_INDEX = 1;
constexpr uint32_t X_SCALE_PAIR_DIM_INDEX = 2;
constexpr uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 0;
constexpr uint32_t WEIGHT_SCALE_K_BLOCK_DIM_INDEX = 1;
constexpr uint32_t WEIGHT_SCALE_N_DIM_INDEX = 2;
constexpr uint32_t WEIGHT_SCALE_PAIR_DIM_INDEX = 3;
constexpr uint32_t WEIGHT_SCALE_TRANS_N_DIM_INDEX = 1;
constexpr uint32_t WEIGHT_SCALE_TRANS_K_BLOCK_DIM_INDEX = 2;
constexpr uint32_t WEIGHT_NZ_GROUP_DIM_INDEX = 0;
constexpr uint32_t WEIGHT_NZ_BLOCK1_DIM_INDEX = 1;
constexpr uint32_t WEIGHT_NZ_BLOCK2_DIM_INDEX = 2;
constexpr uint32_t WEIGHT_NZ_C0_DIM_INDEX = 3;
constexpr uint32_t WEIGHT_NZ_LAST_DIM_INDEX = 4;
constexpr uint32_t WEIGHT_NZ_LAST_DIM_FP8 = 32;
constexpr uint32_t WEIGHT_NZ_C0_DIM = 16;
constexpr uint32_t N_SIZE_ALIGN = 64;
constexpr uint32_t FP8_E4M3FN_VALUE = 36;
constexpr uint32_t FP8_E5M2_VALUE = 35;
constexpr uint32_t FP8_E5M2_PROTO_VALUE = 34;
constexpr uint8_t ROUND_MODE_RINT = 4;
constexpr uint32_t SCALE_ALG_OCP = 0;
constexpr uint32_t SCALE_ALG_CUBLAS = 1;
constexpr uint8_t ACTIVATION_TYPE_GELU_TANH = 2;
constexpr char ACTIVATION_TYPE_GELU_TANH_STR[] = "gelu_tanh";
constexpr uint64_t MAX_BASE_M = 128UL;
constexpr float DEFAULT_DST_TYPE_MAX = 0.0f;

std::string ShapeToDebugString(const gert::Shape &shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += "]";
    return result;
}
} // namespace

void GroupedMatmulActivationQuantTiling950::Reset()
{
    tilingData_ = {};
    roundMode_ = DEFAULT_ROUND_MODE_RINT;
    scaleAlg_ = DEFAULT_SCALE_ALG_OCP;
    activationType_ = DEFAULT_ACTIVATION_TYPE_GELU_TANH;
    dstTypeMax_ = DEFAULT_DST_TYPE_MAX;
    yDtype_ = ge::DT_FLOAT8_E4M3FN;
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmBasicApiTiling::GetShapeAttrsInfo();
}

bool GroupedMatmulActivationQuantTiling950::IsFp8(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

namespace {
bool IsFp8OutputDtype(ge::DataType dtype)
{
    auto dtypeValue = static_cast<uint32_t>(dtype);
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2 ||
           dtypeValue == FP8_E5M2_PROTO_VALUE;
}

} // namespace

bool GroupedMatmulActivationQuantTiling950::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams_.transA = false;
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE(context_->GetNodeName(), "group_list_type must be 0 or 1, actual is %d.",
                        inputParams_.groupListType),
                return false);

    const char *quantModePtr = attrs->GetAttrPointer<char>(ATTR_INDEX_QUANT_MODE);
    const int64_t *yDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_Y_DTYPE);
    const char *roundModePtr = attrs->GetAttrPointer<char>(ATTR_INDEX_ROUND_MODE);
    const int64_t *scaleAlgPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SCALE_ALG);
    const char *activationTypePtr = attrs->GetAttrPointer<char>(ATTR_INDEX_ACTIVATION_TYPE);
    OP_CHECK_IF(quantModePtr == nullptr || yDtypePtr == nullptr || roundModePtr == nullptr ||
                    scaleAlgPtr == nullptr || activationTypePtr == nullptr,
                OP_LOGE(context_->GetNodeName(), "quant attrs must not be nullptr."), return false);
    auto yDtype = static_cast<ge::DataType>(*yDtypePtr);
    std::string quantMode(quantModePtr);
    OP_LOGD(context_->GetNodeName(),
            "GMMAQ AnalyzeAttrs: activationType=%s, quantMode=%s, yDtype=%ld, roundMode=%s, scaleAlg=%ld, "
            "transposeWeight=%d, groupListType=%d.",
            activationTypePtr, quantMode.c_str(), *yDtypePtr, roundModePtr, *scaleAlgPtr,
            static_cast<int>(inputParams_.transB), inputParams_.groupListType);
    OP_CHECK_IF(quantMode != "MX",
                OP_LOGE(context_->GetNodeName(), "quant_mode only supports MX, actual is %s.", quantMode.c_str()),
                return false);
    OP_CHECK_IF(!IsFp8(yDtype),
                OP_LOGE(context_->GetNodeName(), "y_dtype must be FLOAT8_E4M3FN or FLOAT8_E5M2."),
                return false);
    std::string roundMode(roundModePtr);
    OP_CHECK_IF(roundMode != "rint",
                OP_LOGE(context_->GetNodeName(), "mxfp8 only supports round_mode rint."), return false);
    OP_CHECK_IF(*scaleAlgPtr != SCALE_ALG_OCP && *scaleAlgPtr != SCALE_ALG_CUBLAS,
                OP_LOGE(context_->GetNodeName(), "mxfp8 only supports scale_alg 0 or 1."), return false);
    std::string activationType(activationTypePtr);
    OP_CHECK_IF(activationType != ACTIVATION_TYPE_GELU_TANH_STR,
                OP_LOGE(context_->GetNodeName(), "activation_type only supports gelu_tanh."), return false);
    roundMode_ = ROUND_MODE_RINT;
    scaleAlg_ = static_cast<uint8_t>(*scaleAlgPtr);
    activationType_ = ACTIVATION_TYPE_GELU_TANH;
    dstTypeMax_ = DEFAULT_DST_TYPE_MAX;
    yDtype_ = yDtype;
    inputParams_.groupType = GroupedMatmul::SPLIT_M;
    return true;
}

bool GroupedMatmulActivationQuantTiling950::CheckDtype() const
{
    OP_CHECK_IF(!IsFp8(inputParams_.aDtype),
                OP_LOGE(context_->GetNodeName(), "x must be FLOAT8_E4M3FN or FLOAT8_E5M2."),
                return false);
    OP_CHECK_IF(inputParams_.bDtype != ge::DT_FLOAT8_E4M3FN,
                OP_LOGE(context_->GetNodeName(), "weight only supports FLOAT8_E4M3FN."),
                return false);
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 ||
                    inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE(context_->GetNodeName(), "x_scale and weight_scale must be FLOAT8_E8M0."),
                return false);
    OP_CHECK_IF(!IsFp8OutputDtype(inputParams_.outDataDtype),
                OP_LOGE(context_->GetNodeName(), "y must be FLOAT8_E4M3FN or FLOAT8_E5M2."), return false);
    OP_CHECK_IF(inputParams_.outScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE(context_->GetNodeName(), "y_scale must be FLOAT8_E8M0."), return false);
    return true;
}

bool GroupedMatmulActivationQuantTiling950::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    auto xScaleDesc = context_->GetInputDesc(X_SCALE_INDEX);
    auto wScaleDesc = context_->GetInputDesc(WEIGHT_SCALE_INDEX);
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    auto yScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(xDesc == nullptr || wDesc == nullptr || xScaleDesc == nullptr || wScaleDesc == nullptr ||
                    yDesc == nullptr || yScaleDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "input or output desc is nullptr."), return false);
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const int64_t *yDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_Y_DTYPE);
    OP_CHECK_IF(yDtypePtr == nullptr, OP_LOGE(context_->GetNodeName(), "y_dtype must not be nullptr."),
                return false);
    yDtype_ = static_cast<ge::DataType>(*yDtypePtr);
    inputParams_.aDtype = xDesc->GetDataType();
    inputParams_.bDtype = wDesc->GetDataType();
    inputParams_.scaleDtype = wScaleDesc->GetDataType();
    inputParams_.perTokenScaleDtype = xScaleDesc->GetDataType();
    inputParams_.outDataDtype = yDesc->GetDataType();
    inputParams_.outScaleDtype = yScaleDesc->GetDataType();
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    inputParams_.hasBias = false;
    return CheckDtype();
}

bool GroupedMatmulActivationQuantTiling950::CheckMxScaleShape(const gert::Shape &xScaleShape,
                                                            const gert::Shape &wScaleShape) const
{
    OP_CHECK_IF(xScaleShape.GetDimNum() != MXFP_PER_TOKEN_SCALE_DIM_NUM ||
                    wScaleShape.GetDimNum() != MXFP_TYPE_M_SCALE_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "x_scale must be 3D and weight_scale must be 4D."),
                return false);
    auto expectedKBlocks = static_cast<int64_t>(Ops::Base::CeilDiv(inputParams_.kSize, MXFP_BASEK_FACTOR));
    auto wScaleN =
        inputParams_.transB ? wScaleShape.GetDim(WEIGHT_SCALE_TRANS_N_DIM_INDEX) :
                              wScaleShape.GetDim(WEIGHT_SCALE_N_DIM_INDEX);
    auto wScaleK =
        inputParams_.transB ? wScaleShape.GetDim(WEIGHT_SCALE_TRANS_K_BLOCK_DIM_INDEX) :
                              wScaleShape.GetDim(WEIGHT_SCALE_K_BLOCK_DIM_INDEX);
    OP_CHECK_IF(xScaleShape.GetDim(X_SCALE_M_DIM_INDEX) != static_cast<int64_t>(inputParams_.mSize) ||
                    xScaleShape.GetDim(X_SCALE_K_BLOCK_DIM_INDEX) != expectedKBlocks ||
                    xScaleShape.GetDim(X_SCALE_PAIR_DIM_INDEX) != static_cast<int64_t>(MXFP_MULTI_BASE_SIZE),
                OP_LOGE(context_->GetNodeName(), "x_scale shape must be (M, ceil(K/64), 2)."), return false);
    OP_CHECK_IF(wScaleShape.GetDim(WEIGHT_SCALE_GROUP_DIM_INDEX) != static_cast<int64_t>(inputParams_.groupNum) ||
                    wScaleN != static_cast<int64_t>(inputParams_.nSize) || wScaleK != expectedKBlocks ||
                    wScaleShape.GetDim(WEIGHT_SCALE_PAIR_DIM_INDEX) != static_cast<int64_t>(MXFP_MULTI_BASE_SIZE),
                OP_LOGE(context_->GetNodeName(),
                        "weight_scale shape must be (E, ceil(K/64), N, 2) or (E, N, ceil(K/64), 2)."),
                return false);
    return true;
}

bool GroupedMatmulActivationQuantTiling950::CheckWeightNzShape(const gert::Shape &wStorageShape) const
{
    OP_CHECK_IF(inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(), "weight must use FRACTAL_NZ format."), return false);
    OP_CHECK_IF(wStorageShape.GetDimNum() != WEIGHT_NZ_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "NZ weight storage shape must be 5D."), return false);
    OP_CHECK_IF(wStorageShape.GetDim(WEIGHT_NZ_C0_DIM_INDEX) != WEIGHT_NZ_C0_DIM ||
                    wStorageShape.GetDim(WEIGHT_NZ_LAST_DIM_INDEX) != WEIGHT_NZ_LAST_DIM_FP8,
                OP_LOGE(context_->GetNodeName(), "FP8 NZ weight storage tail dims must be 16 and 32."),
                return false);
    auto expectedNBlock = inputParams_.transB ?
        static_cast<int64_t>(Ops::Base::CeilDiv(inputParams_.kSize, WEIGHT_NZ_LAST_DIM_FP8)) :
        static_cast<int64_t>(Ops::Base::CeilDiv(inputParams_.nSize, WEIGHT_NZ_LAST_DIM_FP8));
    auto expectedKBlock = inputParams_.transB ?
        static_cast<int64_t>(Ops::Base::CeilDiv(inputParams_.nSize, WEIGHT_NZ_C0_DIM)) :
        static_cast<int64_t>(Ops::Base::CeilDiv(inputParams_.kSize, WEIGHT_NZ_C0_DIM));
    OP_CHECK_IF(wStorageShape.GetDim(WEIGHT_NZ_GROUP_DIM_INDEX) != static_cast<int64_t>(inputParams_.groupNum) ||
                    wStorageShape.GetDim(WEIGHT_NZ_BLOCK1_DIM_INDEX) != expectedNBlock ||
                    wStorageShape.GetDim(WEIGHT_NZ_BLOCK2_DIM_INDEX) != expectedKBlock,
                OP_LOGE(context_->GetNodeName(),
                        "NZ weight storage shape mismatch, actual is %s, expected first three dims are [%lu, %ld, %ld].",
                        ShapeToDebugString(wStorageShape).c_str(), inputParams_.groupNum, expectedNBlock,
                        expectedKBlock),
                return false);
    return true;
}

bool GroupedMatmulActivationQuantTiling950::CheckCoreNum() const
{
    auto compileInfo = context_->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(inputParams_.opName, "compileInfo is nullptr."), return false);
    OP_CHECK_IF(compileInfo->aicNum == 0,
                OP_LOGE(inputParams_.opName, "aicNum should be positive integer, actual is %u.",
                        compileInfo->aicNum),
                return false);
    OP_CHECK_IF(compileInfo->aivNum != GmmConstant::CORE_RATIO * compileInfo->aicNum,
                OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:2, actual %u:%u.",
                        compileInfo->aicNum, compileInfo->aivNum),
                return false);
    return true;
}

bool GroupedMatmulActivationQuantTiling950::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto xScaleStorageShape = context_->GetInputShape(X_SCALE_INDEX);
    auto wScaleStorageShape = context_->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    OP_CHECK_IF(xStorageShape == nullptr || wStorageShape == nullptr || xScaleStorageShape == nullptr ||
                    wScaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "input shape is nullptr."), return false);
    OP_CHECK_IF(!SetGroupNum(GROUP_LIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    const auto &xShape = xStorageShape->GetOriginShape();
    const auto &wShape = wStorageShape->GetOriginShape();
    const auto &weightStorageShape = wStorageShape->GetStorageShape();
    const auto &xScaleShape = xScaleStorageShape->GetOriginShape();
    const auto &wScaleShape = wScaleStorageShape->GetOriginShape();
    const auto &weightScaleStorageShape = wScaleStorageShape->GetStorageShape();
    std::printf("[GMMAQ tiling] AnalyzeInputs before SetMKN: xOrigin=%s, xStorage=%s, weightOrigin=%s, "
                "weightStorage=%s, xScaleOrigin=%s, weightScaleOrigin=%s, weightScaleStorage=%s, transB=%d, "
                "groupNum=%lu.\n",
                ShapeToDebugString(xShape).c_str(), ShapeToDebugString(xStorageShape->GetStorageShape()).c_str(),
                ShapeToDebugString(wShape).c_str(), ShapeToDebugString(weightStorageShape).c_str(),
                ShapeToDebugString(xScaleShape).c_str(), ShapeToDebugString(wScaleShape).c_str(),
                ShapeToDebugString(weightScaleStorageShape).c_str(), static_cast<int>(inputParams_.transB),
                inputParams_.groupNum);
    OP_LOGD(context_->GetNodeName(),
            "GMMAQ AnalyzeInputs before SetMKN: xOrigin=%s, xStorage=%s, weightOrigin=%s, weightStorage=%s, "
            "xScaleOrigin=%s, weightScaleOrigin=%s, weightScaleStorage=%s, transB=%d, groupNum=%lu.",
            ShapeToDebugString(xShape).c_str(), ShapeToDebugString(xStorageShape->GetStorageShape()).c_str(),
            ShapeToDebugString(wShape).c_str(), ShapeToDebugString(weightStorageShape).c_str(),
            ShapeToDebugString(xScaleShape).c_str(), ShapeToDebugString(wScaleShape).c_str(),
            ShapeToDebugString(weightScaleStorageShape).c_str(), static_cast<int>(inputParams_.transB),
            inputParams_.groupNum);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    std::printf("[GMMAQ tiling] AnalyzeInputs after SetMKN: M=%lu, K=%lu, N=%lu.\n",
                inputParams_.mSize, inputParams_.kSize, inputParams_.nSize);
    OP_LOGD(context_->GetNodeName(), "GMMAQ AnalyzeInputs after SetMKN: M=%lu, K=%lu, N=%lu.",
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize);
    if (wShape.GetDimNum() == WEIGHT_NZ_DIM_NUM) {
        inputParams_.mSize = static_cast<uint64_t>(xShape.GetDim(0));
        inputParams_.kSize = static_cast<uint64_t>(xShape.GetDim(1));
        inputParams_.nSize = inputParams_.transB ?
            static_cast<uint64_t>(weightStorageShape.GetDim(WEIGHT_NZ_BLOCK2_DIM_INDEX) * WEIGHT_NZ_C0_DIM) :
            static_cast<uint64_t>(weightStorageShape.GetDim(WEIGHT_NZ_BLOCK1_DIM_INDEX) * WEIGHT_NZ_LAST_DIM_FP8);
        std::printf("[GMMAQ tiling] AnalyzeInputs recover MKN from WeightNZ storage: M=%lu, K=%lu, N=%lu.\n",
                    inputParams_.mSize, inputParams_.kSize, inputParams_.nSize);
        OP_LOGD(context_->GetNodeName(), "GMMAQ AnalyzeInputs recover MKN from WeightNZ storage: M=%lu, K=%lu, N=%lu.",
                inputParams_.mSize, inputParams_.kSize, inputParams_.nSize);
    }
    OP_CHECK_IF(inputParams_.nSize % N_SIZE_ALIGN != 0,
                OP_LOGE(inputParams_.opName,
                        "N must be a multiple of %u, actual is %u. xOrigin=%s, xStorage=%s, weightOrigin=%s, "
                        "weightStorage=%s, xScaleOrigin=%s, weightScaleOrigin=%s, weightScaleStorage=%s.",
                        N_SIZE_ALIGN, static_cast<uint32_t>(inputParams_.nSize),
                        ShapeToDebugString(xShape).c_str(), ShapeToDebugString(xStorageShape->GetStorageShape()).c_str(),
                        ShapeToDebugString(wShape).c_str(), ShapeToDebugString(weightStorageShape).c_str(),
                        ShapeToDebugString(xScaleShape).c_str(), ShapeToDebugString(wScaleShape).c_str(),
                        ShapeToDebugString(weightScaleStorageShape).c_str()),
                return false);
    inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
    inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;
    OP_CHECK_IF(!CheckMxScaleShape(xScaleStorageShape->GetOriginShape(), wScaleStorageShape->GetOriginShape()),
                OP_LOGE(inputParams_.opName, "CheckMxScaleShape failed."), return false);
    OP_CHECK_IF(!CheckWeightNzShape(wStorageShape->GetStorageShape()),
                OP_LOGE(inputParams_.opName, "CheckWeightNzShape failed."), return false);
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::DoOpTiling()
{
    tilingData_.gmmActivationQuantParams.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmActivationQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmActivationQuantParams.activationType = activationType_;
    tilingData_.gmmActivationQuantParams.quantDtype =
        yDtype_ == ge::DT_FLOAT8_E5M2 ? FP8_E5M2_VALUE : FP8_E4M3FN_VALUE;
    tilingData_.gmmActivationQuantParams.roundMode = roundMode_;
    tilingData_.gmmActivationQuantParams.dequantDtype = static_cast<uint8_t>(ge::DT_FLOAT);
    tilingData_.gmmActivationQuantParams.scaleAlg = scaleAlg_;
    tilingData_.gmmActivationQuantParams.dstTypeMax = dstTypeMax_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::DoLibApiTiling()
{
    // Reuse GMM common tiling helpers and keep the output aligned with GroupedQmmBasicApiTiling::DoLibApiTiling.
    // The kernel consumes compact low-level BasicAPI fields instead of TCubeTiling, so scaleKAL1 and scaleKBL1
    // must be identical.
    GroupedQmmTiling::CalBasicBlock();
    auto baseMModified = std::min(basicTiling_.baseM, MAX_BASE_M);
    basicTiling_.baseM = GroupedMatmul::CeilAlign(baseMModified, GmmConstant::CUBE_BLOCK);
    OP_CHECK_IF(GroupedQmmBasicApiTiling::CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed."), return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.m = inputParams_.mSize;
    tilingData_.mmTilingData.n = inputParams_.nSize;
    tilingData_.mmTilingData.k = inputParams_.kSize;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    tilingData_.mmTilingData.kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    uint32_t scaleKL1 = std::min(
        std::max(basicTiling_.scaleFactorA * basicTiling_.stepKa,
                 basicTiling_.scaleFactorB * basicTiling_.stepKb) *
            basicTiling_.baseK,
        inputParams_.kSize);
    tilingData_.mmTilingData.scaleKAL1 = scaleKL1;
    tilingData_.mmTilingData.scaleKBL1 = scaleKL1;
    tilingData_.mmTilingData.isBias = 0;
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulActivationQuantTiling950::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB),
                              static_cast<uint64_t>(inputParams_.transA));
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::PostTiling()
{
    return SaveTilingDataToContext(tilingData_);
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulActivationQuant, GroupedMatmulActivationQuantTiling950, 0);

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGroupedMatmulActivationQuant(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupedMatmulActivationQuantTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedMatmulActivationQuant", "tiling context is nullptr."),
                return ge::GRAPH_FAILED);
    std::vector<int32_t> registerList = {0};
    return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
}

IMPL_OP_OPTILING(GroupedMatmulActivationQuant)
    .Tiling(GroupedMatmulActivationQuantTilingFunc)
    .TilingParse<GMMCompileInfo>(TilingPrepareForGroupedMatmulActivationQuant);
} // namespace optiling
