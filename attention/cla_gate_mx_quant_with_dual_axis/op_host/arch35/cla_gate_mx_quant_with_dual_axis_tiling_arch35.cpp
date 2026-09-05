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
constexpr int64_t INPUT_GLOBAL_OUT = 0;
constexpr int64_t INPUT_LOCAL_OUT = 1;
constexpr int64_t INPUT_GLOBAL_GATE = 2;
constexpr int64_t INPUT_LOCAL_GATE = 3;
constexpr int64_t OUTPUT_ROW_DATA = 0;
constexpr int64_t OUTPUT_ROW_SCALE = 1;
constexpr int64_t OUTPUT_COL_DATA = 2;
constexpr int64_t OUTPUT_COL_SCALE = 3;
constexpr int64_t EXPECTED_S = 256;
constexpr int64_t EXPECTED_B = 4;
constexpr int64_t EXPECTED_N = 96;
constexpr int64_t EXPECTED_D = 256;
constexpr int64_t DIM_M = EXPECTED_S * EXPECTED_B;
constexpr int64_t DIM_K = EXPECTED_N * EXPECTED_D;
constexpr int64_t BLOCK_W = 256;
constexpr int64_t BLOCK_H = 64;
constexpr int64_t ATTR_SCALE_ALG = 0;
constexpr int64_t ATTR_DST_TYPE = 1;

bool IsShape(const gert::Shape& shape, std::initializer_list<int64_t> expected)
{
    if (shape.GetDimNum() != expected.size()) return false;
    size_t i = 0;
    for (int64_t dim : expected) {
        if (shape.GetDim(i++) != dim) return false;
    }
    return true;
}
} // namespace

ge::graphStatus TilingForClaGateMxQuantWithDualAxis(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto globalOut = context->GetInputShape(INPUT_GLOBAL_OUT);
    auto localOut = context->GetInputShape(INPUT_LOCAL_OUT);
    auto globalGate = context->GetInputShape(INPUT_GLOBAL_GATE);
    auto localGate = context->GetInputShape(INPUT_LOCAL_GATE);
    OP_CHECK_NULL_WITH_CONTEXT(context, globalOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, localOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, globalGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, localGate);
    const auto& go = globalOut->GetStorageShape();
    const auto& lo = localOut->GetStorageShape();
    const auto& gg = globalGate->GetStorageShape();
    const auto& lg = localGate->GetStorageShape();
    OP_CHECK_IF(!IsShape(go, {EXPECTED_S, EXPECTED_B, EXPECTED_N, EXPECTED_D}) || lo != go,
                OP_LOGE(context->GetNodeName(), "global_out/local_out must both be [256,4,96,256]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(IsShape(gg, {EXPECTED_S, EXPECTED_B, EXPECTED_N}) ||
                  IsShape(gg, {EXPECTED_S, EXPECTED_B, EXPECTED_N, 1})) || lg != gg,
                OP_LOGE(context->GetNodeName(), "gate logits must both be [256,4,96] or [256,4,96,1]"),
                return ge::GRAPH_FAILED);

    ge::DataType dtype = context->GetInputDesc(INPUT_GLOBAL_OUT)->GetDataType();
    OP_CHECK_IF((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) ||
                    context->GetInputDesc(INPUT_LOCAL_OUT)->GetDataType() != dtype ||
                    context->GetInputDesc(INPUT_GLOBAL_GATE)->GetDataType() != dtype ||
                    context->GetInputDesc(INPUT_LOCAL_GATE)->GetDataType() != dtype,
                OP_LOGE(context->GetNodeName(), "all four inputs must have the same float16 or bfloat16 dtype"),
                return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto dstType = attrs->GetAttrPointer<int64_t>(ATTR_DST_TYPE);
    auto scaleAlg = attrs->GetAttrPointer<int64_t>(ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstType);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleAlg);
    OP_CHECK_IF(*dstType != 36 || (*scaleAlg != 0 && *scaleAlg != 1),
                OP_LOGE(context->GetNodeName(), "only dst_type=36 and scale_alg=0/1 are supported"),
                return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = platform.GetCoreNumAiv();
    int64_t totalBlocks = (DIM_M / BLOCK_H) * (DIM_K / BLOCK_W);
    int64_t usedCoreNum = std::min(coreNum, totalBlocks);
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(TPL_MODE_BLOCK, TPL_RINT, *scaleAlg, TPL_NO_GROUP_INDEX));

    auto data = context->GetTilingData<ClaGateMxQuantWithDualAxisTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    *data = {};
    data->usedCoreNum = usedCoreNum;
    data->dstType = *dstType;
    data->dimM = DIM_M;
    data->dimN = DIM_K;
    data->numGroups = 1;
    data->blockW = BLOCK_W;
    data->splitBlockH = BLOCK_H;
    data->dimNBlockNum = DIM_K / BLOCK_W;
    data->dimNTail = BLOCK_W;
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = platform.GetLibApiWorkSpaceSize();
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
