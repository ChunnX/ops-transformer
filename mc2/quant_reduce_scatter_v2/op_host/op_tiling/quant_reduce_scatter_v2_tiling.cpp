/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

/*!
 * \file quant_reduce_scatter_v2_tiling.cpp
 * \brief host侧tiling实现
 */
#include <register/op_def_registry.h>
#include "mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_tiling_key.h"
#include "mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_tiling_data.h"
#include "mc2/quant_reduce_scatter/op_host/op_tiling/common/quant_reduce_scatter_util_tiling.h"

namespace MC2Tiling {

using namespace AscendC;
using namespace ge;

/**
 * @brief 设置tilingData，给各成员变量赋值
 */
static void SetTilingData(gert::TilingContext *context, QuantReduceScatterTilingData &tilingData,
                          const QuantReduceScatterConfig& config)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // set tilingData
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum));
    tilingData.quantReduceScatterTilingInfo.aivNum = aivNum;
    uint64_t xValueBS = context->GetInputShape(config.X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueH = context->GetInputShape(config.X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scalesValueH = context->GetInputShape(config.SCALES_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    // 3d场景，context->GetInputShape在函数CheckInputTensorDim中已经校验
    if (context->GetInputShape(config.X_INDEX)->GetStorageShape().GetDimNum() == THREE_DIMS) {
        xValueBS = xValueBS * context->GetInputShape(config.X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
        xValueH = context->GetInputShape(config.X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        scalesValueH = context->GetInputShape(config.SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    }
    tilingData.quantReduceScatterTilingInfo.bs = xValueBS;
    tilingData.quantReduceScatterTilingInfo.hiddenSize = xValueH;
    tilingData.quantReduceScatterTilingInfo.scaleHiddenSize = scalesValueH;
    tilingData.quantReduceScatterTilingInfo.totalWinSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    tilingData.quantReduceScatterTilingInfo.isMc2Context = config.isMc2Context;
}

/**
 * @brief 基于 TARGET_ITER 公式计算 host 推荐的 xPerBlock（先除 rankSize 再反推），写入 tilingData
 */
static void SetXPerBlock(QuantReduceScatterTilingData &tilingData, const TilingRunInfo &runInfo)
{
    constexpr uint32_t TARGET_ITER = 3U;   // 与 QAR 对称
    constexpr uint32_t MIN_BLOCK = 2048U;  // QRS comm-bound：per_core 太小公式失效，MIN 兜到 2048
    constexpr uint32_t ALIGN_BLOCK = 1024U;  // 与 kernel X_BLOCK_ALIGN_NUM 对齐
    uint64_t xNums = tilingData.quantReduceScatterTilingInfo.bs * tilingData.quantReduceScatterTilingInfo.hiddenSize;
    uint64_t aivNum = tilingData.quantReduceScatterTilingInfo.aivNum;
    uint64_t rankSize = static_cast<uint64_t>(runInfo.rankSize);
    uint64_t xSliceSizeNums = xNums / rankSize;
    uint64_t perCoreElem = (xSliceSizeNums + aivNum - 1U) / aivNum;
    uint64_t xPerBlock = (perCoreElem + TARGET_ITER - 1U) / TARGET_ITER;
    xPerBlock = std::max<uint64_t>(xPerBlock, MIN_BLOCK);
    xPerBlock = (xPerBlock / ALIGN_BLOCK) * ALIGN_BLOCK;
    tilingData.quantReduceScatterTilingInfo.xPerBlock = static_cast<uint32_t>(xPerBlock);
    tilingData.quantReduceScatterTilingInfo.alignBlock = ALIGN_BLOCK;
}

static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 设置tilingKey模板参数
    const uint64_t tilingKey = GET_TPL_TILING_KEY(MTE_COMM);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in quant_reduce_scatter_v2.", tilingKey);
}

/**
 * @brief 校验attr context
 */
static ge::graphStatus CheckMc2Context(gert::TilingContext *context, const char *nodeName,
                                       const QuantReduceScatterConfig &config)
{
    const gert::StorageShape *ctxStorageShape = context->GetInputShape(config.CONTEXT_INDEX);
    OP_TILING_CHECK(ctxStorageShape == nullptr,
                    OP_LOGE(nodeName, "The context shape is null."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(ctxStorageShape->GetStorageShape().GetDimNum() != 1,
                    OP_LOGE(nodeName,
                            "The context shape dim must be 1, but current actual value is: %lu.",
                            ctxStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    int64_t ctxDim0 = ctxStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "The context dim0 is: %ld.", ctxDim0);

    auto ctxDesc = context->GetInputDesc(config.CONTEXT_INDEX);
    OP_TILING_CHECK(ctxDesc == nullptr,
                    OP_LOGE(nodeName, "The context desc is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ctxDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName,
                            "The context dataType is invalid, dataType should be int32, but actual value is: %s.",
                            Ops::Base::ToString(ctxDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(ctxDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE(nodeName, "The context format is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief quant_reduce_scatter_v2算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus QuantReduceScatterV2TilingFunc(gert::TilingContext *context)
{
    OP_LOGD("quant_reduce_scatter_v2", "Enter QuantReduceScatterV2TilingFunc.");

    OP_TILING_CHECK(context == nullptr,
                    OP_LOGE("quant_reduce_scatter_v2", "failed to get tiling context in quant_reduce_scatter_v2."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
                    OP_LOGE("quant_reduce_scatter_v2", "failed to get nodeName in quant_reduce_scatter_v2."),
                    return ge::GRAPH_FAILED);

    QuantReduceScatterTilingData *tilingData = context->GetTilingData<QuantReduceScatterTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr in quant_reduce_scatter_v2."),
                    return ge::GRAPH_FAILED);

    TilingRunInfo runInfo = {};
    OP_TILING_CHECK(QuantReduceScatterUtilTiling::CheckNpuArch(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "NpuArch is invalid in quant_reduce_scatter_v2."), return ge::GRAPH_FAILED);

    QuantReduceScatterConfig config;
    config.X_INDEX = 1;
    config.SCALES_INDEX = 2;
    config.isMc2Context = true;
    OP_TILING_CHECK(CheckMc2Context(context, nodeName, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "context check failed in quant_reduce_scatter_v2."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(QuantReduceScatterUtilTiling::CheckTilingFunc(context, runInfo,
                    OpType::OP_QUANT_REDUCE_SCATTER, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tiling check failed in quant_reduce_scatter_v2."), return ge::GRAPH_FAILED);

    SetTilingData(context, *tilingData, config);
    SetXPerBlock(*tilingData, runInfo);
    SetTilingKey(context);
    return ge::GRAPH_SUCCESS;
}

struct QuantReduceScatterV2CompileInfo {};
 
ge::graphStatus TilingParseForQuantReduceScatterV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantReduceScatterV2)
    .Tiling(QuantReduceScatterV2TilingFunc)
    .TilingParse<QuantReduceScatterV2CompileInfo>(TilingParseForQuantReduceScatterV2);

}  // namespace MC2Tiling
