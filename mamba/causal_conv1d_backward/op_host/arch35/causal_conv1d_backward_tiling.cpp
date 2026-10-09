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
 * \file causal_conv1d_backward_tiling.cpp
 * \brief CausalConv1dBackward Tiling 实现（arch35）
 *
 * 切分策略：仅沿 D（通道）维切分为 baseDim 大小的 tile，核间按 tile 轮转。
 * 每个核独占若干完整通道，在核内即可完整累加 grad_weight，无需跨核 reduce。
 * token 轴按 query_start_loc 划分的序列段独立处理（None 时整轴一段）。
 */

#include <cstring>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/causal_conv1d_backward_tiling_data.h"
#include "../../op_kernel/arch35/causal_conv1d_backward_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;  // 系统预留 workspace
constexpr int64_t BASE_DIM = 128;                      // 通道 tile 大小（32B 对齐）

constexpr uint32_t IDX_GRAD_Y = 0;
constexpr uint32_t IDX_X = 1;
constexpr uint32_t IDX_WEIGHT = 2;
constexpr uint32_t IDX_QSL = 3;
constexpr uint32_t ATTR_ACTIVATION_MODE = 0;

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

static int64_t ParseActivationModeStr(const char* str)
{
    if (str == nullptr || std::strcmp(str, "") == 0 || std::strcmp(str, "silu") == 0) {
        return 1;
    }
    if (std::strcmp(str, "none") == 0) {
        return 0;
    }
    return -1;
}

// 解析 query_start_loc（可选输入）。返回值：-1 表示未提供，>0 表示 qslSize。
// 2D 变长输入必须提供；3D 输入可选（提供时按段切分，未提供时整轴一段）。
static int64_t ParseQueryStartLoc(gert::TilingContext* context, bool& qslExist)
{
    qslExist = false;
    auto qslShapePtr = context->GetOptionalInputShape(IDX_QSL);
    if (qslShapePtr == nullptr) {
        return -1;
    }
    const auto& qslStorageShape = qslShapePtr->GetStorageShape();
    const int64_t qslDimNum = qslStorageShape.GetDimNum();
    if (qslDimNum == 0 || (qslDimNum == 1 && qslStorageShape.GetDim(0) <= 0)) {
        return -1;
    }
    OP_CHECK_IF(qslDimNum != 1,
                OP_LOGE(context, "query_start_loc must be 1D, got dimNum=%ld", qslDimNum),
                return -2);
    int64_t qslSize = qslStorageShape.GetDim(0);
    OP_CHECK_IF(qslSize < 2,
                OP_LOGE(context, "query_start_loc size must be >= 2 (num_seq+1), got %ld", qslSize),
                return -2);
    auto qslDesc = context->GetOptionalInputDesc(IDX_QSL);
    OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc);
    OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(context, "query_start_loc dtype must be int32, got %d",
                        static_cast<int>(qslDesc->GetDataType())),
                return -2);
    qslExist = true;
    return qslSize;
}

static ge::graphStatus CausalConv1dBackwardTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 属性：activation_mode
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* activationModeStr = attrs->GetAttrPointer<char>(ATTR_ACTIVATION_MODE);
    int64_t activationMode = ParseActivationModeStr(activationModeStr);
    OP_CHECK_IF(activationMode < 0,
                OP_LOGE(context, "activation_mode only supports \"silu\" or \"none\", got %s",
                        activationModeStr == nullptr ? "null" : activationModeStr),
                return ge::GRAPH_FAILED);

    // 3. shape 信息
    auto gradYShapePtr = context->GetInputShape(IDX_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYShapePtr);
    auto weightShapePtr = context->GetInputShape(IDX_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShapePtr);
    auto gradYShape = gradYShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();

    int64_t tokenCnt = 0;
    int64_t seqCnt = 1;
    if (gradYShape.GetDimNum() == 2) {
        // 2D varlen [cu_seq_len, D]：query_start_loc 必须提供
        tokenCnt = gradYShape.GetDim(0);
        bool qslExist = false;
        int64_t qslSize = ParseQueryStartLoc(context, qslExist);
        OP_CHECK_IF(qslSize < 0,
                    OP_LOGE(context, "2D grad_y (varlen) requires query_start_loc to be provided"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tokenCnt <= 0,
                    OP_LOGE(context, "2D grad_y cu_seq_len must be > 0, got %ld", tokenCnt),
                    return ge::GRAPH_FAILED);
        seqCnt = qslSize - 1;
    } else if (gradYShape.GetDimNum() == 3) {
        // 3D batch [B, L, D]：token 轴为展平的 B*L；query_start_loc 可选
        int64_t batch = gradYShape.GetDim(0);
        int64_t seqLen = gradYShape.GetDim(1);
        tokenCnt = 0;
        OP_CHECK_IF(__builtin_mul_overflow(batch, seqLen, &tokenCnt),
                    OP_LOGE(context, "batch * seqlen overflow: B=%ld L=%ld", batch, seqLen),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tokenCnt <= 0,
                    OP_LOGE(context, "invalid shape: B=%ld L=%ld", batch, seqLen),
                    return ge::GRAPH_FAILED);
        bool qslExist = false;
        int64_t qslSize = ParseQueryStartLoc(context, qslExist);
        OP_CHECK_IF(qslSize == -2,
                    OP_LOGE(context, "ParseQueryStartLoc error"), return ge::GRAPH_FAILED);
        seqCnt = qslExist ? (qslSize - 1) : 1;
    } else {
        OP_LOGE(context, "grad_y must be 2D [cu_seq_len, D] or 3D [B, L, D], got dimNum=%zu",
                gradYShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int64_t dim = gradYShape.GetDimNum() == 2 ? gradYShape.GetDim(1) : gradYShape.GetDim(2);
    int64_t kW = weightShape.GetDim(0);
    OP_CHECK_IF(weightShape.GetDimNum() != 2,
                OP_LOGE(context, "weight must be 2D [kW,D], got dimNum=%zu", weightShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightShape.GetDim(1) != dim,
                OP_LOGE(context, "weight dim mismatch: weight[1]=%ld, grad_y dim=%ld",
                        weightShape.GetDim(1), dim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kW < 2 || kW > 4,
                OP_LOGE(context, "kernel width W only supports [2, 3, 4], got %ld", kW),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dim <= 0,
                OP_LOGE(context, "invalid dim=%ld", dim),
                return ge::GRAPH_FAILED);
    // 32B 对齐（与正向 CausalConv1d 同款）：保证所有通道 tile（含尾 tile）的
    // UB 平面行距为 32B 整数倍，逐行兜底路径的 count 形式向量调用地址安全
    auto dimDesc = context->GetInputDesc(IDX_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimDesc);
    int64_t dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(dimDesc->GetDataType()));
    OP_CHECK_IF((dim * dtypeSize) % 32 != 0,
                OP_LOGE(context, "dim=%ld must satisfy (dim * dtypeSize) %% 32 == 0 "
                                 "(32-byte aligned, dtypeSize=%ld)", dim, dtypeSize),
                return ge::GRAPH_FAILED);

    // 4. dtype
    auto inputDesc = context->GetInputDesc(IDX_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16,
                OP_LOGE(context, "invalid dtype %d (only float16/bfloat16)", static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);

    // 5. workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 6. 切分计算：沿 D 维按 BASE_DIM 切 tile
    int64_t baseDim = BASE_DIM;
    int64_t dimTileCnt = CeilDiv(dim, baseDim);
    int64_t usedCores = (dimTileCnt < coreNum) ? dimTileCnt : coreNum;

    // 7. 填充 TilingData
    CausalConv1dBackwardTilingData* tiling = context->GetTilingData<CausalConv1dBackwardTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CausalConv1dBackwardTilingData), 0, sizeof(CausalConv1dBackwardTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->tokenCnt = tokenCnt;
    tiling->seqCnt = seqCnt;
    tiling->dim = dim;
    tiling->kW = kW;
    tiling->activationMode = activationMode;
    tiling->baseDim = baseDim;
    tiling->dimTileCnt = dimTileCnt;
    tiling->usedCores = usedCores;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    // 8. TilingKey（dtype 维度）
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCausalConv1dBackward([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CausalConv1dBackwardCompileInfo {};

IMPL_OP_OPTILING(CausalConv1dBackward)
    .Tiling(CausalConv1dBackwardTilingFunc)
    .TilingParse<CausalConv1dBackwardCompileInfo>(TilingParseForCausalConv1dBackward);

} // namespace optiling
