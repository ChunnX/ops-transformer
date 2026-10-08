/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_grad_tiling.cpp
 * \brief CausalConv1dGrad Tiling 实现（arch35）
 *
 * 切分策略：仅沿 D（通道）维切分为 baseDim 大小的 tile，核间按 tile 轮转。
 * 每个核独占若干完整通道，在核内即可完整累加 dweight/dbias，无需跨核 reduce。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/causal_conv1d_grad_tiling_data.h"
#include "../../op_kernel/arch35/causal_conv1d_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;

constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;  // 系统预留 workspace
constexpr int64_t BASE_DIM = 128;                      // 通道 tile 大小（32B 对齐）

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CausalConv1dGradTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. shape 信息
    auto gradYShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYShapePtr);
    auto weightShapePtr = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShapePtr);
    auto gradYShape = gradYShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();

    OP_CHECK_IF(gradYShape.GetDimNum() != 3,
                OP_LOGE(context, "grad_y must be 3D [B,T,D], got dimNum=%zu", gradYShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightShape.GetDimNum() != 2,
                OP_LOGE(context, "weight must be 2D [kW,D], got dimNum=%zu", weightShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    int64_t batch = gradYShape.GetDim(0);
    int64_t seqLen = gradYShape.GetDim(1);
    int64_t dim = gradYShape.GetDim(2);
    int64_t kW = weightShape.GetDim(0);

    OP_CHECK_IF(weightShape.GetDim(1) != dim,
                OP_LOGE(context, "weight dim mismatch: weight[1]=%ld, grad_y[2]=%ld", weightShape.GetDim(1), dim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dim <= 0 || batch <= 0 || seqLen <= 0 || kW <= 0,
                OP_LOGE(context, "invalid shape: B=%ld T=%ld D=%ld kW=%ld", batch, seqLen, dim, kW),
                return ge::GRAPH_FAILED);

    // 3. dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16,
                OP_LOGE(context, "invalid dtype %d (only float16/bfloat16)", static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);

    // 4. workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 5. 切分计算：沿 D 维按 BASE_DIM 切 tile
    int64_t baseDim = BASE_DIM;
    int64_t dimTileCnt = CeilDiv(dim, baseDim);
    int64_t usedCores = (dimTileCnt < coreNum) ? dimTileCnt : coreNum;

    // 6. 填充 TilingData
    CausalConv1dGradTilingData* tiling = context->GetTilingData<CausalConv1dGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CausalConv1dGradTilingData), 0, sizeof(CausalConv1dGradTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->batch = batch;
    tiling->seqLen = seqLen;
    tiling->dim = dim;
    tiling->kW = kW;
    tiling->baseDim = baseDim;
    tiling->dimTileCnt = dimTileCnt;
    tiling->usedCores = usedCores;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    // 7. TilingKey（dtype 维度）
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCausalConv1dGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CausalConv1dGradCompileInfo {};

IMPL_OP_OPTILING(CausalConv1dGrad)
    .Tiling(CausalConv1dGradTilingFunc)
    .TilingParse<CausalConv1dGradCompileInfo>(TilingParseForCausalConv1dGrad);

} // namespace optiling
