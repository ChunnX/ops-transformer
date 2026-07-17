/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_moe_small_bs_w4a16_tiling.cpp
 * \brief
 */

#include <set>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/fused_moe_small_bs_w4a16_tiling_data.h"
#include "../op_kernel/fused_moe_small_bs_w4a16_tiling_key.h"

namespace optiling {

// 与 kernel 侧保持一致的常量
constexpr int64_t BLOCK_SIZE = 128;          // kernel 一次处理的 K 方向块大小
constexpr int64_t GROUP_SIZE = 32;           // 量化 group 大小
constexpr int64_t PACK_RATIO = 8;            // 一个 int32 打包 8 个 int4
constexpr int64_t W13_OUT_FACTOR = 2;        // W13 输出为 2 * InterSize (gate | value)
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U; // 系统 workspace

constexpr uint32_t IDX_X = 0;
constexpr uint32_t IDX_W13_WEIGHT = 1;
constexpr uint32_t IDX_W13_SCALES = 2;
constexpr uint32_t IDX_W2_WEIGHT = 3;
constexpr uint32_t IDX_W2_SCALES = 4;
constexpr uint32_t IDX_EXPERT_IDS = 5;
constexpr uint32_t IDX_TOPK_WEIGHTS = 6;

struct FusedMoeSmallBsW4a16CompileInfo {
};

// 获取平台信息 coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, int64_t &coreNum)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool CheckDim(const gert::Shape *shape, size_t dimNum)
{
    return shape != nullptr && shape->GetDimNum() == dimNum;
}

// 获取并校验 shape、dtype 信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext *context, FusedMoeSmallBsW4a16TilingData *tiling,
                                         ge::DataType &dataType)
{
    // x: [BatchSize, HiddenSize]
    auto xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    OP_CHECK_IF(!CheckDim(&xStorageShape, 2), OP_LOGE(context, "x must be 2D [BatchSize, HiddenSize]"),
                return ge::GRAPH_FAILED);
    int64_t batchSize = xStorageShape.GetDim(0);
    int64_t hiddenSize = xStorageShape.GetDim(1);
    OP_CHECK_IF(hiddenSize % BLOCK_SIZE != 0,
                OP_LOGE(context, "hiddenSize %ld must be multiple of %ld", hiddenSize, BLOCK_SIZE),
                return ge::GRAPH_FAILED);

    // expert_ids: [BatchSize, TopK]
    auto expertIdsShape = context->GetInputShape(IDX_EXPERT_IDS);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);
    auto expertIdsStorageShape = expertIdsShape->GetStorageShape();
    OP_CHECK_IF(!CheckDim(&expertIdsStorageShape, 2),
                OP_LOGE(context, "expert_ids must be 2D [BatchSize, TopK]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertIdsStorageShape.GetDim(0) != batchSize,
                OP_LOGE(context, "expert_ids batch size mismatch with x"), return ge::GRAPH_FAILED);
    int64_t topK = expertIdsStorageShape.GetDim(1);

    // topk_weights: [BatchSize, TopK]
    auto topkWeightsShape = context->GetInputShape(IDX_TOPK_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsShape);
    auto topkWeightsStorageShape = topkWeightsShape->GetStorageShape();
    OP_CHECK_IF(!CheckDim(&topkWeightsStorageShape, 2) ||
                    topkWeightsStorageShape.GetDim(0) != batchSize || topkWeightsStorageShape.GetDim(1) != topK,
                OP_LOGE(context, "topk_weights must be 2D [BatchSize, TopK] and match expert_ids"),
                return ge::GRAPH_FAILED);

    // w13_weight: [NumExperts, HiddenSize, 2 * InterSize / 8]
    auto w13WeightShape = context->GetInputShape(IDX_W13_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, w13WeightShape);
    auto w13WeightStorageShape = w13WeightShape->GetStorageShape();
    OP_CHECK_IF(!CheckDim(&w13WeightStorageShape, 3),
                OP_LOGE(context, "w13_weight must be 3D [NumExperts, HiddenSize, 2 * InterSize / 8]"),
                return ge::GRAPH_FAILED);
    int64_t numExperts = w13WeightStorageShape.GetDim(0);
    OP_CHECK_IF(w13WeightStorageShape.GetDim(1) != hiddenSize,
                OP_LOGE(context, "w13_weight dim1 %ld must equal hiddenSize %ld", w13WeightStorageShape.GetDim(1),
                        hiddenSize),
                return ge::GRAPH_FAILED);
    int64_t w13OutSize = w13WeightStorageShape.GetDim(2) * PACK_RATIO;
    OP_CHECK_IF(w13OutSize % W13_OUT_FACTOR != 0,
                OP_LOGE(context, "w13_weight packed out dim %ld invalid", w13WeightStorageShape.GetDim(2)),
                return ge::GRAPH_FAILED);
    int64_t interSize = w13OutSize / W13_OUT_FACTOR;
    OP_CHECK_IF(interSize % BLOCK_SIZE != 0,
                OP_LOGE(context, "interSize %ld must be multiple of %ld", interSize, BLOCK_SIZE),
                return ge::GRAPH_FAILED);

    // w13_scales: [NumExperts, HiddenSize / 32, 2 * InterSize]
    auto w13ScalesShape = context->GetInputShape(IDX_W13_SCALES);
    OP_CHECK_NULL_WITH_CONTEXT(context, w13ScalesShape);
    auto w13ScalesStorageShape = w13ScalesShape->GetStorageShape();
    OP_CHECK_IF(!CheckDim(&w13ScalesStorageShape, 3) || w13ScalesStorageShape.GetDim(0) != numExperts ||
                    w13ScalesStorageShape.GetDim(1) != hiddenSize / GROUP_SIZE ||
                    w13ScalesStorageShape.GetDim(2) != w13OutSize,
                OP_LOGE(context, "w13_scales must be [NumExperts, HiddenSize / 32, 2 * InterSize]"),
                return ge::GRAPH_FAILED);

    // w2_weight: [NumExperts, InterSize, HiddenSize / 8]
    auto w2WeightShape = context->GetInputShape(IDX_W2_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, w2WeightShape);
    auto w2WeightStorageShape = w2WeightShape->GetStorageShape();
    OP_CHECK_IF(!CheckDim(&w2WeightStorageShape, 3) || w2WeightStorageShape.GetDim(0) != numExperts ||
                    w2WeightStorageShape.GetDim(1) != interSize ||
                    w2WeightStorageShape.GetDim(2) * PACK_RATIO != hiddenSize,
                OP_LOGE(context, "w2_weight must be [NumExperts, InterSize, HiddenSize / 8]"),
                return ge::GRAPH_FAILED);

    // w2_scales: [NumExperts, InterSize / 32, HiddenSize]
    auto w2ScalesShape = context->GetInputShape(IDX_W2_SCALES);
    OP_CHECK_NULL_WITH_CONTEXT(context, w2ScalesShape);
    auto w2ScalesStorageShape = w2ScalesShape->GetStorageShape();
    OP_CHECK_IF(!CheckDim(&w2ScalesStorageShape, 3) || w2ScalesStorageShape.GetDim(0) != numExperts ||
                    w2ScalesStorageShape.GetDim(1) != interSize / GROUP_SIZE ||
                    w2ScalesStorageShape.GetDim(2) != hiddenSize,
                OP_LOGE(context, "w2_scales must be [NumExperts, InterSize / 32, HiddenSize]"),
                return ge::GRAPH_FAILED);

    // dtype 校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype of x, only fp16/bf16 supported");
        return ge::GRAPH_FAILED;
    }

    tiling->batchSize = static_cast<int32_t>(batchSize);
    tiling->hiddenSize = static_cast<int32_t>(hiddenSize);
    tiling->interSize = static_cast<int32_t>(interSize);
    tiling->numExperts = static_cast<int32_t>(numExperts);
    tiling->topK = static_cast<int32_t>(topK);
    return ge::GRAPH_SUCCESS;
}

// 计算 workspace：W13 输出(FP32) + SwiGLU 输出(T) + W2 输出(FP32) + 系统 workspace
static ge::graphStatus GetWorkspaceSize(gert::TilingContext *context, const FusedMoeSmallBsW4a16TilingData *tiling)
{
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    int64_t totalTokens = static_cast<int64_t>(tiling->batchSize) * tiling->topK;
    size_t typeSize = sizeof(uint16_t); // x 仅支持 fp16/bf16，均为 2 字节
    size_t w13OutSize = static_cast<size_t>(totalTokens) * W13_OUT_FACTOR * tiling->interSize * sizeof(float);
    size_t w2InSize = static_cast<size_t>(totalTokens) * tiling->interSize * typeSize;
    size_t w2OutSize = static_cast<size_t>(tiling->batchSize) * tiling->hiddenSize * sizeof(float);
    currentWorkspace[0] = w13OutSize + w2InSize + w2OutSize + WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus FusedMoeSmallBsW4a16TilingFunc(gert::TilingContext *context)
{
    // 1、获取平台运行信息
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);

    // 2、获取shape、属性信息并填充tiling数据
    FusedMoeSmallBsW4a16TilingData *tiling = context->GetTilingData<FusedMoeSmallBsW4a16TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(FusedMoeSmallBsW4a16TilingData), 0, sizeof(FusedMoeSmallBsW4a16TilingData)) !=
                    EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, tiling, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3、计算WorkspaceSize
    OP_CHECK_IF(GetWorkspaceSize(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4、设置blockDim与tilingKey
    context->SetBlockDim(coreNum);
    if (dataType == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_FP16));
    } else if (dataType == ge::DT_BF16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_BF16));
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFusedMoeSmallBsW4a16(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(FusedMoeSmallBsW4a16)
    .Tiling(FusedMoeSmallBsW4a16TilingFunc)
    .TilingParse<FusedMoeSmallBsW4a16CompileInfo>(TilingParseForFusedMoeSmallBsW4a16);
} // namespace optiling
