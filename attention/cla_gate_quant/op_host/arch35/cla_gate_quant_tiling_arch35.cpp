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
 * \file cla_gate_quant_tiling_arch35.cpp
 * \brief Tiling implementation for ClaGateQuant
 */

#include "cla_gate_quant_tiling_arch35.h"
#include "op_common/op_host/util/platform_util.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/cla_gate_quant_tiling_key.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace ClaGateQuantOp;

namespace optiling {
namespace {
// Attr order must match op_def:
// dst_type(0), round_mode(1), scale_alg(2), inputAttnLayout(3), dual_axis_flag(4)
constexpr int64_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 2;
constexpr int64_t INDEX_ATTR_INPUT_ATTN_LAYOUT = 3;
constexpr int64_t INDEX_ATTR_DUAL_AXIS_FLAG = 4;

constexpr int64_t INPUT_GLOBAL_ATTN = 0;
constexpr int64_t INPUT_LOCAL_ATTN = 1;
constexpr int64_t INPUT_GLOBAL_GATE = 2;
constexpr int64_t INPUT_LOCAL_GATE = 3;
constexpr int64_t OUTPUT_ROW_DATA = 0;
constexpr int64_t OUTPUT_ROW_SCALE = 1;
constexpr int64_t OUTPUT_COL_DATA = 2;
constexpr int64_t OUTPUT_COL_SCALE = 3;

constexpr int64_t BLOCK_W = 256;
constexpr int64_t BLOCK_H = 64;
constexpr int64_t MIN_N = 1;
constexpr int64_t MAX_N = 128;
constexpr int64_t FULL_ROW_BLOCK = 64;
constexpr int64_t FULL_COL_BLOCK = 256;
constexpr int64_t UB_BLOCK_BYTES = 32;
constexpr int64_t RESERVED_UB_BYTES = 2 * 1024;
constexpr int64_t DB_BUFFER_COUNT = 2;
constexpr int64_t FP32_VF_ELEMENTS = 64;

int64_t AlignUbBlock(int64_t bytes)
{
    return Ops::Base::CeilDiv(bytes, UB_BLOCK_BYTES) * UB_BLOCK_BYTES;
}

// Keep this calculation synchronized with the single-axis branch of
// CommonBase::Init.  The tail-axis kernel has no col_data/col_scale output, so
// it deliberately does not reserve the two second-axis queues.  The remaining
// colScaleReciprocalBuf_ is reused as compact max-exp scratch and therefore
// still grows with the selected batch.
int64_t CalcTailAxisUbBytes(const ClaGateQuantTilingParam &params, int64_t batchTiles)
{
    int64_t headsPerTile = FULL_COL_BLOCK / params.dimHeadSize;
    int64_t halfBufferElems = batchTiles * FULL_COL_BLOCK;
    int64_t inputFrameBytes = halfBufferElems * params.dtypeSize * 2; // global + local
    // Kernel gives each global/local gate (and sigmoid) side a complete FP32
    // VF span.  The last sigmoid iteration stores one full VF without a tail
    // mask, so accounting only the logical gate count would both misalign the
    // second side and underestimate its physical scratch footprint.
    int64_t logicalGateCount = batchTiles * headsPerTile;
    int64_t gateSideElems = Ops::Base::CeilDiv(logicalGateCount, FP32_VF_ELEMENTS) * FP32_VF_ELEMENTS;
    int64_t gateFrameBytes = gateSideElems * params.dtypeSize * 2;
    int64_t mergedBytes = halfBufferElems * params.dtypeSize;
    int64_t sigmoidBytes = gateSideElems * static_cast<int64_t>(sizeof(float)) * 2;
    int64_t rowDataFrameBytes = halfBufferElems; // current common queue stores bytes for both FP8/FP4
    int64_t rowScaleFrameBytes = batchTiles * UB_BLOCK_BYTES;
    int64_t rowReciprocalBytes = batchTiles * UB_BLOCK_BYTES;
    int64_t fixedColReciprocalBytes = FULL_COL_BLOCK * 2 * params.dtypeSize;
    constexpr int64_t B16_VF_ELEMENTS = 128;
    int64_t compactScaleCount = batchTiles * (FULL_COL_BLOCK / 32);
    int64_t compactMaxExpBytes = Ops::Base::CeilDiv(compactScaleCount, B16_VF_ELEMENTS) *
                                 B16_VF_ELEMENTS * static_cast<int64_t>(sizeof(uint16_t));
    int64_t colReciprocalBytes = std::max(fixedColReciprocalBytes, compactMaxExpBytes);

    return DB_BUFFER_COUNT * AlignUbBlock(inputFrameBytes) +
           DB_BUFFER_COUNT * AlignUbBlock(gateFrameBytes) + AlignUbBlock(mergedBytes) +
           AlignUbBlock(sigmoidBytes) + DB_BUFFER_COUNT * AlignUbBlock(rowDataFrameBytes) +
           DB_BUFFER_COUNT * AlignUbBlock(rowScaleFrameBytes) + AlignUbBlock(rowReciprocalBytes) +
           AlignUbBlock(colReciprocalBytes);
}

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN,
                                                    ge::DT_FLOAT8_E5M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP8_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

RoundModeList GetRoundMode(const string &roundMode)
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

bool IsValidTNDShape(const gert::Shape &shape)
{
    return shape.GetDimNum() == 3 && shape.GetDim(0) > 0 && shape.GetDim(1) >= MIN_N && shape.GetDim(1) <= MAX_N &&
           (shape.GetDim(2) == 128 || shape.GetDim(2) == 256);
}

bool IsValidGateShape(const gert::Shape &shape, int64_t t, int64_t n)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) == t && shape.GetDim(1) == n;
}

bool CheckShape(const gert::Shape &globalAttn, const gert::Shape &localAttn, const gert::Shape &globalGate,
                const gert::Shape &localGate)
{
    return IsValidTNDShape(globalAttn) && localAttn == globalAttn &&
           IsValidGateShape(globalGate, globalAttn.GetDim(0), globalAttn.GetDim(1)) && localGate == globalGate;
}

ge::graphStatus CheckOutputShapes(const gert::Shape &globalAttn, const gert::Shape &rowData, const gert::Shape &rowScale,
                                  const gert::Shape &colData, const gert::Shape &colScale)
{
    int64_t rowCount = globalAttn.GetDim(0);
    int64_t columnCount = globalAttn.GetDim(1) * globalAttn.GetDim(2);
    if (rowData.GetDimNum() != 2 || rowData.GetDim(0) != rowCount || rowData.GetDim(1) != columnCount) {
        return ge::GRAPH_FAILED;
    }
    if (rowScale.GetDimNum() != 3 || rowScale.GetDim(0) != rowCount ||
        rowScale.GetDim(1) != Ops::Base::CeilDiv(columnCount, BLOCK_H) || rowScale.GetDim(2) != 2) {
        return ge::GRAPH_FAILED;
    }
    bool colEmpty =
        (colData.GetDimNum() >= 1 && colData.GetDim(0) == 0) && (colScale.GetDimNum() >= 1 && colScale.GetDim(0) == 0);
    if (!colEmpty) {
        if (colData.GetDimNum() != 2 || colData.GetDim(0) != rowCount || colData.GetDim(1) != columnCount) {
            return ge::GRAPH_FAILED;
        }
        if (colScale.GetDimNum() != 3 || colScale.GetDim(0) != Ops::Base::CeilDiv(rowCount, BLOCK_H) ||
            colScale.GetDim(1) != columnCount || colScale.GetDim(2) != 2) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void CalcTailAxisTiling(ClaGateQuantTilingParam &params)
{
    int64_t rowCount = params.dimM;
    int64_t columnCount = params.dimN;

    // For the single-axis (tail-axis) path, choose the number of AIV cores by
    // the amount of input data first. Every full 4KB input chunk is treated as
    // the smallest amount of work that should occupy one core. Double buffering
    // doubles the resident input volume, so divide the 4KB-based hint by 2 as
    // well. If the data is large enough, fall back to the full core count and
    // split by T*N (256-column tiles) over those cores.
    constexpr int64_t BYTES_PER_CORE_HINT = 4096;
    int64_t inputBytes = rowCount * columnCount * params.dtypeSize;
    int64_t coreHint = inputBytes / BYTES_PER_CORE_HINT / 2;
    if (coreHint < 1) {
        coreHint = 1;
    }
    params.usedCoreNum = std::min(params.totalCoreNum, coreHint);

    // Single-axis tiling granularity: one row x one 256-column block.
    // rowBlock is always 1; only the column direction is split into 256-wide
    // blocks. If the last column block is shorter than 256, the kernel pads it
    // with zeros before the interleaved double-buffer copy/quantization.
    int64_t rowBlock = 1;
    int64_t colBlock = FULL_COL_BLOCK;
    int64_t colTileNum = Ops::Base::CeilDiv(columnCount, colBlock);
    params.rowBlockSize = rowBlock;
    params.colBlockSize = colBlock;
    params.rowTileNum = Ops::Base::CeilDiv(rowCount, rowBlock);
    params.colTileNum = colTileNum;
    params.rowTailSize = 1;
    params.colTailSize = columnCount - (colTileNum - 1) * colBlock;
    if (params.colTailSize <= 0) {
        params.colTailSize = colBlock;
    }

    // Derive the maximum resident 1x256 tile count from the actual UB size and
    // the buffers allocated by CommonBase::Init.  This may be smaller or larger
    // than 64 on a different product/configuration; the final batch is allowed
    // to be smaller than the selected capacity.
    int64_t totalTiles = params.rowTileNum * params.colTileNum;
    int64_t maxTilesPerCore = Ops::Base::CeilDiv(totalTiles, params.usedCoreNum);
    int64_t availableUb = std::max<int64_t>(0, params.ubSize - RESERVED_UB_BYTES);
    int64_t ubBatchSize = 1;
    for (int64_t candidate = 1; candidate <= maxTilesPerCore; ++candidate) {
        if (CalcTailAxisUbBytes(params, candidate) > availableUb) {
            break;
        }
        ubBatchSize = candidate;
    }
    int64_t selectedBatchSize = std::min(maxTilesPerCore, ubBatchSize);
    // Number of gate scalars carried by one full-width 1x256 tile.
    int64_t headsPerTile = FULL_COL_BLOCK / params.dimHeadSize;
    // A repeated full batch should contain complete 64-scalar sigmoid VFs.
    // One 256-column tile contains 256 / headDim gate scalars, so the optimal
    // tile quantum is 64 / headsPerTile: 64 tiles for D=256 and 32 for D=128.
    // For example, D=256 with capacity 67 would execute two Exp/Div VFs per
    // batch while using only three lanes in the second VF.
    // If all work fits in one batch, keep the exact tile count to avoid an
    // unnecessary split; otherwise round the UB-derived capacity down to the
    // largest sigmoid tile-quantum multiple.  This also satisfies the smaller
    // OCP (16-tile) and cuBLAS (8-tile) scale-vector granularities.
    int64_t sigmoidTileQuantum = FP32_VF_ELEMENTS / headsPerTile;
    if (maxTilesPerCore > ubBatchSize && ubBatchSize >= sigmoidTileQuantum) {
        selectedBatchSize = (ubBatchSize / sigmoidTileQuantum) * sigmoidTileQuantum;
    }
    params.batchSize = std::max<int64_t>(1, selectedBatchSize);
    params.batchCount = Ops::Base::CeilDiv(maxTilesPerCore, params.batchSize);
}

ge::graphStatus GetPlatformInfo(gert::TilingContext *context, ClaGateQuantTilingParam &params)
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

ge::graphStatus GetAndCheckAttrs(gert::TilingContext *context, ClaGateQuantTilingParam &params,
                                 int64_t &tilingKeyDualAxisFlag, int64_t &tilingKeyRound)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto roundModePtr = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, roundModePtr);
    string roundModeStr = roundModePtr;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        roundMode == RoundModeList::MODE_UNDEFINED,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "round_mode", roundModeStr.c_str(), "rint, floor or round"),
        return ge::GRAPH_FAILED);

    auto scaleAlgPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleAlgPtr);
    auto dstTypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstTypePtr);
    auto dualAxisFlagPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_DUAL_AXIS_FLAG);
    auto inputAttnLayoutPtr = attrs->GetAttrPointer<char>(INDEX_ATTR_INPUT_ATTN_LAYOUT);
    params.roundMode = static_cast<int64_t>(roundMode);
    params.scaleAlg = *scaleAlgPtr;
    params.dstType = *dstTypePtr;
    params.dualAxisFlag = (dualAxisFlagPtr != nullptr && *dualAxisFlagPtr) ? 1 : 0;
    string inputAttnLayout = (inputAttnLayoutPtr != nullptr) ? string(inputAttnLayoutPtr) : string("TND");
    OP_CHECK_IF(inputAttnLayout != "TND",
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "inputAttnLayout", inputAttnLayout.c_str(), "TND"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(params.scaleAlg != 0 && params.scaleAlg != 1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "scale_alg", std::to_string(params.scaleAlg).c_str(),
                                          "0 or 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Y_SUPPORT_DTYPE_SET.count(params.yDtype) == 0 || static_cast<int64_t>(params.yDtype) != params.dstType,
                OP_LOGE(context->GetNodeName(), "dst_type and output data type mismatch."), return ge::GRAPH_FAILED);
    // FP8 only rint
    OP_CHECK_IF(Y_SUPPORT_DTYPE_FP8_SET.count(params.yDtype) != 0 && roundMode != RoundModeList::MODE_RINT,
                OP_LOGE(context->GetNodeName(), "When output is FP8, round_mode only supports rint."),
                return ge::GRAPH_FAILED);
    // FP4 only scale_alg=0
    OP_CHECK_IF(Y_SUPPORT_DTYPE_FP4_SET.count(params.yDtype) != 0 && params.scaleAlg != 0,
                OP_LOGE(context->GetNodeName(), "When output is FP4, scale_alg must be 0."), return ge::GRAPH_FAILED);
    // FP4 requires K = N*D to be a multiple of 4 (two packed FP4 values per byte)
    OP_CHECK_IF(Y_SUPPORT_DTYPE_FP4_SET.count(params.yDtype) != 0 && (params.dimN % 4) != 0,
                OP_LOGE(context->GetNodeName(), "When output is FP4, K=N*D must be divisible by 4."),
                return ge::GRAPH_FAILED);

    if (roundMode == RoundModeList::MODE_RINT) {
        tilingKeyRound = TPL_RINT;
    } else if (roundMode == RoundModeList::MODE_ROUND) {
        tilingKeyRound = TPL_ROUND;
    } else {
        tilingKeyRound = TPL_FLOOR;
    }
    tilingKeyDualAxisFlag = (params.dualAxisFlag != 0) ? TPL_DUAL_AXIS : TPL_SINGLE_AXIS;
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus TilingForClaGateQuant(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    auto globalAttnShapePtr = context->GetInputShape(INPUT_GLOBAL_ATTN);
    auto localAttnShapePtr = context->GetInputShape(INPUT_LOCAL_ATTN);
    auto globalGateShapePtr = context->GetInputShape(INPUT_GLOBAL_GATE);
    auto localGateShapePtr = context->GetInputShape(INPUT_LOCAL_GATE);
    OP_CHECK_NULL_WITH_CONTEXT(context, globalAttnShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, localAttnShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, globalGateShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, localGateShapePtr);
    const auto &globalAttnShape = globalAttnShapePtr->GetStorageShape();
    const auto &localAttnShape = localAttnShapePtr->GetStorageShape();
    const auto &globalGateShape = globalGateShapePtr->GetStorageShape();
    const auto &localGateShape = localGateShapePtr->GetStorageShape();
    OP_CHECK_IF(!CheckShape(globalAttnShape, localAttnShape, globalGateShape, localGateShape),
                OP_LOGE(context->GetNodeName(), "Expected global_attn/local_attn [T,N,D], gate [T,N]; "
                                                "N in [1,128], D in {128,256}"),
                return ge::GRAPH_FAILED);

    auto inputDtype = context->GetInputDesc(INPUT_GLOBAL_ATTN)->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(inputDtype) == 0 ||
                    context->GetInputDesc(INPUT_LOCAL_ATTN)->GetDataType() != inputDtype ||
                    context->GetInputDesc(INPUT_GLOBAL_GATE)->GetDataType() != inputDtype ||
                    context->GetInputDesc(INPUT_LOCAL_GATE)->GetDataType() != inputDtype,
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
                OP_LOGE(context->GetNodeName(), "Invalid output dtype."), return ge::GRAPH_FAILED);

    auto rowDataShape = context->GetOutputShape(OUTPUT_ROW_DATA)->GetStorageShape();
    auto rowScaleShape = context->GetOutputShape(OUTPUT_ROW_SCALE)->GetStorageShape();
    auto colDataShape = context->GetOutputShape(OUTPUT_COL_DATA)->GetStorageShape();
    auto colScaleShape = context->GetOutputShape(OUTPUT_COL_SCALE)->GetStorageShape();
    OP_CHECK_IF(CheckOutputShapes(globalAttnShape, rowDataShape, rowScaleShape, colDataShape, colScaleShape) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Invalid output shapes."), return ge::GRAPH_FAILED);

    ClaGateQuantTilingParam params;
    params.yDtype = rowDataDesc->GetDataType();
    params.dimM = globalAttnShape.GetDim(0);
    params.dimHeadNum = globalAttnShape.GetDim(1);
    params.dimHeadSize = globalAttnShape.GetDim(2);
    params.dimN = params.dimHeadNum * params.dimHeadSize;
    params.dtypeSize = 2; // fp16/bf16 inputs are both 2 bytes per element.
    params.blockW = BLOCK_W;
    params.splitBlockH = BLOCK_H;

    int64_t keyDualAxisFlag = TPL_DUAL_AXIS;
    int64_t keyRound = TPL_RINT;
    OP_CHECK_IF(GetPlatformInfo(context, params) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAndCheckAttrs(context, params, keyDualAxisFlag, keyRound) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetAndCheckAttrs failed."), return ge::GRAPH_FAILED);

    params.dualAxisFlag = params.dualAxisFlag == 0 ? 0 : 1;

    params.dimNBlockNum = Ops::Base::CeilDiv(params.dimN, params.blockW);
    params.dimNTail = params.dimN % params.blockW == 0 ? params.blockW : params.dimN % params.blockW;

    if (params.dualAxisFlag == 0) {
        CalcTailAxisTiling(params);
    } else {
        int64_t rowSplitCount = Ops::Base::CeilDiv(params.dimM, params.splitBlockH);
        int64_t totalTaskCount = rowSplitCount * params.dimNBlockNum;
        params.usedCoreNum = std::min(params.totalCoreNum, totalTaskCount);
    }
    context->SetBlockDim(params.usedCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(keyDualAxisFlag, keyRound, params.scaleAlg));

    auto tilingData = context->GetTilingData<ClaGateQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = {};
    tilingData->usedCoreNum = params.usedCoreNum;
    tilingData->dstType = params.dstType;
    tilingData->dimM = params.dimM;
    tilingData->dimN = params.dimN;
    tilingData->blockW = params.blockW;
    tilingData->splitBlockH = params.splitBlockH;
    tilingData->dimNBlockNum = params.dimNBlockNum;
    tilingData->dimNTail = params.dimNTail;
    tilingData->headNum = params.dimHeadNum;
    tilingData->headDim = params.dimHeadSize;
    tilingData->dualAxisFlag = params.dualAxisFlag;
    tilingData->rowTileNum = params.rowTileNum;
    tilingData->colTileNum = params.colTileNum;
    tilingData->rowBlockSize = params.rowBlockSize;
    tilingData->rowTailSize = params.rowTailSize;
    tilingData->colBlockSize = params.colBlockSize;
    tilingData->colTailSize = params.colTailSize;
    tilingData->batchCount = params.batchCount;
    tilingData->batchSize = params.batchSize;

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = params.workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForClaGateQuant(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ClaGateQuant)
    .Tiling(TilingForClaGateQuant)
    .TilingParse<ClaGateQuantCompileInfo>(TilingPrepareForClaGateQuant);
} // namespace optiling
