/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file post_quant_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "post_quant_checker.h"

namespace optilingForBuildPKG {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

// CheckSingle
ge::graphStatus PostQuantChecker::CheckSingleDtype(const FaTilingInfo &faInfo)
{
    // QuantScale2 and quantOffset2 only support bf16/fp32 data type.
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.quantScale2.desc, QUANT_SCALE2_NAME)) {
        OP_LOGE(faInfo.opName, "QuantScale2 only support bf16/fp32 data type!");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.quantOffset2.desc, QUANT_OFFSET2_NAME)) {
        OP_LOGE(faInfo.opName, "QuantOffset2 only support bf16/fp32 data type!");
        return ge::GRAPH_FAILED;
    }
    // post-quantization scenarios only support int8/fp8_e4m3fn/hifloat8 output data type
    if (faInfo.isOutQuantEnable) {
        ge::DataType outputType = faInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(outputType != ge::DT_INT8 && outputType != ge::DT_FLOAT8_E4M3FN && outputType != ge::DT_HIFLOAT8,
                    OP_LOGE(faInfo.opName,
                    "The quantScale2 exists, output data type only supports int8/fp8_e4m3fn/hifloat8!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckParaExistence
ge::graphStatus PostQuantChecker::CheckExistenceQuantScale2(const FaTilingInfo &faInfo)
{
    // Post-quantization scenarios must include quantScale2.
    if (faInfo.isOutQuantEnable) {
        OP_CHECK_IF(faInfo.opParamInfo.quantScale2.tensor == nullptr,
                    OP_LOGE(faInfo.opName, "Quant_scale2 is nullptr in post quant scenario!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(faInfo.opParamInfo.quantScale2.desc == nullptr,
                    OP_LOGE(faInfo.opName, "Desc of quant_scale2 is nullptr in post quant scenario!"),
                    return ge::GRAPH_FAILED);
        int64_t quantScale2ShapeSize = faInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        OP_CHECK_IF(quantScale2ShapeSize <= 0,
                    OP_LOGE(faInfo.opName, "Shape size of quant_scale2 is nonpositive in post quant scenario!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus PostQuantChecker::CheckFeatureQueryDType(const FaTilingInfo &faInfo)
{
    // Post-quantization scale dtype must be FP32. BF16 is allowed only if the query is BF16
    if (faInfo.isOutQuantEnable) {
        const ge::DataType quantScale2Type = faInfo.opParamInfo.quantScale2.tensor->GetDataType();
        OP_CHECK_IF(faInfo.inputQType != ge::DT_BF16 && quantScale2Type != ge::DT_FLOAT,
                    OP_LOGE(faInfo.opName,
                            "When query is not bf16, the post quant scale dtype only supports float32!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureOutputEqual(const FaTilingInfo &faInfo)
{
    // In the Anti-quantization scenario,
    // only KV input and post-quantization outputs with the same data type are supported.
    if (faInfo.isOutQuantEnable) {
        ge::DataType outputType = faInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(outputType != faInfo.inputKvType,
                    OP_LOGE(faInfo.opName,
                            "The quantScale2 exists, output data type only supports that matches the KV input type!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeaturePrefix(const FaTilingInfo &faInfo)
{
    // When prefix exists, post-quantization scenarios only support int8 output data types.
    if (faInfo.isOutQuantEnable) {
        ge::DataType outputType = faInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(faInfo.sysPrefixFlag && outputType != ge::DT_INT8,
                    OP_LOGE(faInfo.opName, "When prefix exists, the output data type only supports int8!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureRowValid(const FaTilingInfo &faInfo)
{
    if (!faInfo.isOutQuantEnable) {
        return ge::GRAPH_SUCCESS;
    }
    if (faInfo.s1Size > 1) {
        bool checkPostQuantOffset =
            (faInfo.outputType == ge::DT_INT8) &&
            (faInfo.opParamInfo.quantOffset2.tensor != nullptr && faInfo.opParamInfo.quantOffset2.desc != nullptr) &&
            (faInfo.opParamInfo.quantOffset2.tensor->GetStorageShape().GetShapeSize() != 0);
        if (!faInfo.isMaxWorkspace) {
            std::vector<int64_t> actualSeqLengthsKV{};
            std::vector<int64_t> actualSeqLengths{};
            actualSeqLengthsKV.resize(faInfo.bSize);
            actualSeqLengths.resize(faInfo.bSize);

            const gert::Tensor *tempData = faInfo.opParamInfo.actualSeqLengthsQ.tensor;
            const gert::Tensor *tempDataKV = faInfo.opParamInfo.actualSeqLengths.tensor;
            const int64_t *preTokens = faInfo.opParamInfo.preToken;
            const int64_t *nextTokens = faInfo.opParamInfo.nextToken;
            uint32_t actualLenDims = (tempData != nullptr) ? tempData->GetShapeSize() : 0;
            uint32_t actualLenDimsKV = (tempDataKV != nullptr) ? tempDataKV->GetShapeSize() : 0;
            for (uint32_t i = 0; i < faInfo.bSize; i++) {
                if ((actualLenDims == 0) || (tempData == nullptr) || (tempData->GetData<int64_t>() == nullptr)) {
                    actualSeqLengths[i] = faInfo.s1Size;
                } else {
                    actualSeqLengths[i] = (actualLenDims > 1) ? static_cast<uint32_t>(tempData->GetData<int64_t>()[i]) :
                                                                static_cast<uint32_t>(tempData->GetData<int64_t>()[0]);
                }
                if ((actualLenDimsKV == 0) || (tempDataKV == nullptr) ||
                    (tempDataKV->GetData<int64_t>() == nullptr)) { // The user did not input act_seq_kv
                    if (faInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
                        actualSeqLengthsKV[i] = faInfo.s2Size;
                    } else {
                        actualSeqLengthsKV[i] = faInfo.kvListSeqLens[i];
                    }
                } else {
                    actualSeqLengthsKV[i] = (actualLenDimsKV > 1) ?
                                                static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[i]) :
                                                static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[0]);
                }
                int64_t preTokensPerbatch = 0;
                int64_t nextTokensPerbatch = 0;
                if (faInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
                    preTokensPerbatch = static_cast<int64_t>(SPARSE_MODE_INT_MAX);
                    nextTokensPerbatch =
                        actualSeqLengthsKV[i] + static_cast<int64_t>(faInfo.systemPrefixLen) - actualSeqLengths[i];
                } else if (faInfo.sparseMode == SPARSE_MODE_BAND) {
                    preTokensPerbatch = faInfo.preToken - actualSeqLengthsKV[i] -
                                        static_cast<int64_t>(faInfo.systemPrefixLen) + actualSeqLengths[i];
                    nextTokensPerbatch = faInfo.nextToken + actualSeqLengthsKV[i] +
                                         static_cast<int64_t>(faInfo.systemPrefixLen) - actualSeqLengths[i];
                } else {
                    preTokensPerbatch = faInfo.preToken;
                    nextTokensPerbatch = faInfo.nextToken;
                }
                OP_CHECK_IF((checkPostQuantOffset &&
                             ((preTokensPerbatch + actualSeqLengthsKV[i] +
                                   static_cast<int64_t>(faInfo.systemPrefixLen) - actualSeqLengths[i] <
                               0) ||
                              (nextTokensPerbatch < 0))),
                            OPS_REPORT_VECTOR_INNER_ERR(
                                faInfo.opName,
                                "When sparse mode = %d, output dtype is int8, the output's dequant offset "
                                "is not null or empty tensor, "
                                "preTokens = %ld and nextTokens = %ld, some rows of the matrix do not "
                                "participate in the calculation, "
                                "the accuracy of the final result will be incorrect. Please see the "
                                "documentation for more details.",
                                faInfo.sparseMode, *preTokens, *nextTokens),
                            return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// CheckCheckMultiPara
ge::graphStatus PostQuantChecker::CheckMultiParaQuantOffset2(const FaTilingInfo &faInfo)
{
    // Scale2 and offset2 should have same dtype and shape
    if (faInfo.isOutQuantEnable && faInfo.opParamInfo.quantOffset2.tensor != nullptr &&
        faInfo.opParamInfo.quantOffset2.desc != nullptr) {
        const ge::DataType quantScale2Type = faInfo.opParamInfo.quantScale2.tensor->GetDataType();
        int64_t quantScale2ShapeSize = faInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        size_t quantScale2Dim = faInfo.opParamInfo.quantScale2.tensor->GetStorageShape().GetDimNum();

        const ge::DataType quantOffset2Datatype = faInfo.opParamInfo.quantOffset2.tensor->GetDataType();
        int64_t quantOffset2ShapeSize = faInfo.opParamInfo.quantOffset2.tensor->GetShapeSize();
        size_t quantOffset2Dim = faInfo.opParamInfo.quantOffset2.tensor->GetStorageShape().GetDimNum();

        OP_CHECK_IF(quantOffset2Datatype != quantScale2Type,
                    OP_LOGE(faInfo.opName, "QuantScale2 and quantOffset2 should have same dtype!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(quantOffset2Dim != quantScale2Dim,
                    OP_LOGE(faInfo.opName, "QuantScale2 and quantOffset2 should have same dim!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(quantOffset2ShapeSize != quantScale2ShapeSize,
                    OP_LOGE(faInfo.opName, "QuantScale2 and quantOffset2 should have same shape size!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckMultiParaShape(const FaTilingInfo &faInfo)
{
    // For post quant per-tensor, quant scale/offset only support [1].
    // For post quant per-channel, quant scale/offset dim multiply result only support qN * vD.
    if (faInfo.isOutQuantEnable) {
        int64_t quantScale2ShapeSize = faInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        size_t quantScale2Dim = faInfo.opParamInfo.quantScale2.tensor->GetStorageShape().GetDimNum();
        std::string layoutString = faInfo.opParamInfo.layOut;
        bool isSupportedLayout = layoutString == "BSH" || layoutString == "BSND" || layoutString == "BNSD" ||
                                 layoutString == "BNSD_BSND";
        uint32_t numHeads = *(faInfo.opParamInfo.numHeads);
        uint64_t quantScale2ShapeSizePerChannel =
            static_cast<uint64_t>(numHeads) * static_cast<uint64_t>(faInfo.vHeadDim);
        // per-tensor or per-channel verification
        if (quantScale2Dim == 1) {
            if (static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel || !isSupportedLayout) {
                OP_CHECK_IF((static_cast<uint64_t>(quantScale2ShapeSize) != 1U),
                            OP_LOGE(faInfo.opName,
                                    "For post quant per-tensor, quant scale/offset only support [1], now is [%d]",
                                    quantScale2ShapeSize),
                            return ge::GRAPH_FAILED);
            }
        } else {
            if (isSupportedLayout) {
                OP_CHECK_IF((static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel),
                            OP_LOGE(faInfo.opName,
                                    "For post quant per-channel, when layout is %s, "
                                    "quantScale2/quantOffset2 dim multiply "
                                    "result only support support qN * vD(%u * %u = %lu), now is (%ld).",
                                    layoutString.c_str(), numHeads, faInfo.vHeadDim, quantScale2ShapeSizePerChannel,
                                    quantScale2ShapeSize),
                            return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(faInfo.opParamInfo.quantScale2.tensor->GetStorageShape() !=
                                gert::Shape({numHeads, faInfo.vHeadDim}),
                            OP_LOGE(faInfo.opName,
                                    "For post quant per-channel, when layout is %s, "
                                    "quantScale2/quantOffset2 expect shape is [%u, %u].",
                                    layoutString.c_str(), numHeads, faInfo.vHeadDim),
                            return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSingleDtype(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceQuantScale2(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureQueryDType(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeaturePrefix(faInfo)) {
            return ge::GRAPH_FAILED;
        }
        if (faInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
            if (ge::GRAPH_SUCCESS != CheckFeatureRowValid(faInfo)) {
                 return ge::GRAPH_FAILED;
            }
        }
    } else if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureOutputEqual(faInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckMultiParaQuantOffset2(faInfo) ||
        ge::GRAPH_SUCCESS != CheckMultiParaShape(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optilingForBuildPKG
