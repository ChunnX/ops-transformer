/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cla_gate_mx_quant_with_dual_axis_tiling_arch35.cpp
 * \brief Tiling implementation for ClaGateMxQuantWithDualAxis
 */

#include "cla_gate_mx_quant_with_dual_axis_tiling_arch35.h"
#include "op_common/op_host/util/platform_util.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/cla_gate_mx_quant_with_dual_axis_tiling_key.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace ClaGateMxQuantWithDualAxisOp;

namespace optiling {
namespace {
// Attr order must match op_def:
// dst_type(0), scale_alg(1), round_mode(2)
constexpr int64_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 1;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 2;

constexpr int64_t INPUT_GLOBAL_OUT = 0;
constexpr int64_t INPUT_LOCAL_OUT = 1;
constexpr int64_t INPUT_GLOBAL_GATE = 2;
constexpr int64_t INPUT_LOCAL_GATE = 3;
constexpr int64_t OUTPUT_ROW_DATA = 0;
constexpr int64_t OUTPUT_ROW_SCALE = 1;
constexpr int64_t OUTPUT_COL_DATA = 2;
constexpr int64_t OUTPUT_COL_SCALE = 3;

constexpr int64_t BLOCK_W = 256;
constexpr int64_t BLOCK_H = 64;
constexpr int64_t MIN_N = 4;
constexpr int64_t MAX_N = 128;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2,
                                                    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

RoundModeList GetRoundMode(const string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    }
    if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    }
    if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}

bool IsValidTNDShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 3 && shape.GetDim(0) > 0 && shape.GetDim(1) >= MIN_N &&
           shape.GetDim(1) <= MAX_N && shape.GetDim(1) % 2 == 0 &&
           (shape.GetDim(2) == 128 || shape.GetDim(2) == 256);
}

bool IsValidGateShape(const gert::Shape& shape, int64_t t, int64_t n)
{
    if (shape.GetDimNum() == 2) {
        return shape.GetDim(0) == t && shape.GetDim(1) == n;
    }
    if (shape.GetDimNum() == 3) {
        return shape.GetDim(0) == t && shape.GetDim(1) == n && shape.GetDim(2) == 1;
    }
    return false;
}

bool CheckShape(const gert::Shape& go, const gert::Shape& lo, const gert::Shape& gg, const gert::Shape& lg)
{
    return IsValidTNDShape(go) && lo == go && IsValidGateShape(gg, go.GetDim(0), go.GetDim(1)) &&
           lg == gg;
}

ge::graphStatus CheckOutputShapes(const gert::Shape& go, const gert::Shape& rowData, const gert::Shape& rowScale,
                                  const gert::Shape& colData, const gert::Shape& colScale)
{
    int64_t t = go.GetDim(0);
    int64_t k = go.GetDim(1) * go.GetDim(2);
    if (rowData.GetDimNum() != 2 || colData.GetDimNum() != 2 || rowData != colData ||
        rowData.GetDim(0) != t || rowData.GetDim(1) != k) {
        return ge::GRAPH_FAILED;
    }
    if (rowScale.GetDimNum() != 3 || rowScale.GetDim(0) != t ||
        rowScale.GetDim(1) != Ops::Base::CeilDiv(k, BLOCK_H) || rowScale.GetDim(2) != 2) {
        return ge::GRAPH_FAILED;
    }
    if (colScale.GetDimNum() != 3 || colScale.GetDim(0) != Ops::Base::CeilDiv(t, BLOCK_H) ||
        colScale.GetDim(1) != k || colScale.GetDim(2) != 2) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, ClaGateMxQuantWithDualAxisTilingParam& params)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    params.totalCoreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(params.totalCoreNum <= 0, OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    params.ubSize = static_cast<int64_t>(ubSize);
    params.workspaceSize = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAndCheckAttrs(gert::TilingContext* context, ClaGateMxQuantWithDualAxisTilingParam& params,
                                 int64_t& tilingKeyMode, int64_t& tilingKeyRound)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto roundModePtr = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, roundModePtr);
    string roundModeStr = roundModePtr;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(roundMode == RoundModeList::MODE_UNDEFINED,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "round_mode", roundModeStr.c_str(),
                                          "rint, floor or round"),
                return ge::GRAPH_FAILED);

    auto scaleAlgPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleAlgPtr);
    auto dstTypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstTypePtr);
    params.roundMode = static_cast<int64_t>(roundMode);
    params.scaleAlg = *scaleAlgPtr;
    params.dstType = *dstTypePtr;

    OP_CHECK_IF(params.scaleAlg != 0 && params.scaleAlg != 1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "scale_alg",
                                          std::to_string(params.scaleAlg).c_str(), "0 or 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(params.yDtype) == 0 || static_cast<int64_t>(params.yDtype) != params.dstType,
                OP_LOGE(context->GetNodeName(), "dst_type and output data type mismatch."),
                return ge::GRAPH_FAILED);
    // FP8 only rint
    OP_CHECK_IF(Y_SUPPORT_DTYPE_FP8_SET.count(params.yDtype) != 0 && roundMode != RoundModeList::MODE_RINT,
                OP_LOGE(context->GetNodeName(), "When output is FP8, round_mode only supports rint."),
                return ge::GRAPH_FAILED);
    // FP4 only scale_alg=0
    OP_CHECK_IF(Y_SUPPORT_DTYPE_FP4_SET.count(params.yDtype) != 0 && params.scaleAlg != 0,
                OP_LOGE(context->GetNodeName(), "When output is FP4, scale_alg must be 0."),
                return ge::GRAPH_FAILED);

    if (roundMode == RoundModeList::MODE_RINT) {
        tilingKeyRound = TPL_RINT;
    } else if (roundMode == RoundModeList::MODE_ROUND) {
        tilingKeyRound = TPL_ROUND;
    } else {
        tilingKeyRound = TPL_FLOOR;
    }
    tilingKeyMode = TPL_MODE_BLOCK;
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus TilingForClaGateMxQuantWithDualAxis(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    auto goShapePtr = context->GetInputShape(INPUT_GLOBAL_OUT);
    auto loShapePtr = context->GetInputShape(INPUT_LOCAL_OUT);
    auto ggShapePtr = context->GetInputShape(INPUT_GLOBAL_GATE);
    auto lgShapePtr = context->GetInputShape(INPUT_LOCAL_GATE);
    OP_CHECK_NULL_WITH_CONTEXT(context, goShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, loShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, ggShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, lgShapePtr);
    const auto& go = goShapePtr->GetStorageShape();
    const auto& lo = loShapePtr->GetStorageShape();
    const auto& gg = ggShapePtr->GetStorageShape();
    const auto& lg = lgShapePtr->GetStorageShape();
    OP_CHECK_IF(!CheckShape(go, lo, gg, lg),
                OP_LOGE(context->GetNodeName(),
                        "Expected global_out/local_out [T,N,D], gate [T,N] or [T,N,1]; "
                        "N in [4,128] even, D in {128,256}"),
                return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDesc(INPUT_GLOBAL_OUT)->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(dtype) == 0 ||
                    context->GetInputDesc(INPUT_LOCAL_OUT)->GetDataType() != dtype ||
                    context->GetInputDesc(INPUT_GLOBAL_GATE)->GetDataType() != dtype ||
                    context->GetInputDesc(INPUT_LOCAL_GATE)->GetDataType() != dtype,
                OP_LOGE(context->GetNodeName(), "all four inputs must have the same float16 or bfloat16 dtype"),
                return ge::GRAPH_FAILED);

    auto rowDataDesc = context->GetOutputDesc(OUTPUT_ROW_DATA);
    auto rowScaleDesc = context->GetOutputDesc(OUTPUT_ROW_SCALE);
    auto colDataDesc = context->GetOutputDesc(OUTPUT_COL_DATA);
    auto colScaleDesc = context->GetOutputDesc(OUTPUT_COL_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowDataDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowScaleDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, colDataDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, colScaleDesc);
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(rowDataDesc->GetDataType()) == 0 ||
                    colDataDesc->GetDataType() != rowDataDesc->GetDataType() ||
                    rowScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0 ||
                    colScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                OP_LOGE(context->GetNodeName(), "Invalid output dtype."),
                return ge::GRAPH_FAILED);

    auto rowDataShape = context->GetOutputShape(OUTPUT_ROW_DATA)->GetStorageShape();
    auto rowScaleShape = context->GetOutputShape(OUTPUT_ROW_SCALE)->GetStorageShape();
    auto colDataShape = context->GetOutputShape(OUTPUT_COL_DATA)->GetStorageShape();
    auto colScaleShape = context->GetOutputShape(OUTPUT_COL_SCALE)->GetStorageShape();
    OP_CHECK_IF(CheckOutputShapes(go, rowDataShape, rowScaleShape, colDataShape, colScaleShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Invalid output shapes."),
                return ge::GRAPH_FAILED);

    ClaGateMxQuantWithDualAxisTilingParam params;
    params.yDtype = rowDataDesc->GetDataType();
    params.dimM = go.GetDim(0);
    params.dimHeadNum = go.GetDim(1);
    params.dimHeadSize = go.GetDim(2);
    params.dimN = params.dimHeadNum * params.dimHeadSize;
    params.blockW = BLOCK_W;
    params.splitBlockH = BLOCK_H;

    int64_t keyMode = TPL_MODE_BLOCK;
    int64_t keyRound = TPL_RINT;
    OP_CHECK_IF(GetPlatformInfo(context, params) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAndCheckAttrs(context, params, keyMode, keyRound) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetAndCheckAttrs failed."),
                return ge::GRAPH_FAILED);

    params.dimNBlockNum = Ops::Base::CeilDiv(params.dimN, params.blockW);
    params.dimNTail = params.dimN % params.blockW == 0 ? params.blockW : params.dimN % params.blockW;
    int64_t dimMSplit = Ops::Base::CeilDiv(params.dimM, params.splitBlockH);
    int64_t totalBlocks = dimMSplit * params.dimNBlockNum;
    params.usedCoreNum = std::min(params.totalCoreNum, totalBlocks);
    context->SetBlockDim(params.usedCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(keyMode, keyRound, params.scaleAlg, TPL_NO_GROUP_INDEX));

    auto data = context->GetTilingData<ClaGateMxQuantWithDualAxisTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    *data = {};
    data->usedCoreNum = params.usedCoreNum;
    data->dstType = params.dstType;
    data->activateLeft = 1;
    data->dimM = params.dimM;
    data->dimN = params.dimN;
    data->numGroups = 1;
    data->blockW = params.blockW;
    data->splitBlockH = params.splitBlockH;
    data->dimNBlockNum = params.dimNBlockNum;
    data->dimNTail = params.dimNTail;
    data->headNum = params.dimHeadNum;
    data->headDim = params.dimHeadSize;

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = params.workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForClaGateMxQuantWithDualAxis(gert::TilingParseContext* context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ClaGateMxQuantWithDualAxis)
    .Tiling(TilingForClaGateMxQuantWithDualAxis)
    .TilingParse<ClaGateMxQuantWithDualAxisCompileInfo>(TilingPrepareForClaGateMxQuantWithDualAxis);
} // namespace optiling
