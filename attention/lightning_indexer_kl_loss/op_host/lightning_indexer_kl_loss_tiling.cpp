/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/lightning_indexer_kl_loss_tiling_data.h"
#include "../op_kernel/lightning_indexer_kl_loss_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

struct LightningIndexerKlLossCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t ComputeTileLength(int64_t K, int64_t ubSize, int64_t dtypeSize)
{
    // UB 划分为 7 块连续 buffer:
    // ub_target_score_in: tileLen * K * dtypeSize
    // ub_index_probs_in:  tileLen * K * dtypeSize
    // ub_reduce_sum:      tileLen * 8 * dtypeSize  (8 元素对齐)
    // ub_log_P:           tileLen * K * dtypeSize
    // ub_log_Y:           tileLen * K * dtypeSize
    // ub_out:             8 * dtypeSize
    // tmp_ub:             tileLen * K * dtypeSize
    // 总元素数 = tileLen * (5K + 8) + 8
    // 总内存占用 <= ubSize
    int64_t totalElems = ubSize / dtypeSize;
    int64_t maxTileLen = (totalElems - 8) / (5 * K + 8);
    return (maxTileLen > 1) ? maxTileLen : 1;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, bool deterministic)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    if (deterministic) {
        // 确定性模式需要 workspace 存放各 Core 独立结果: coreNum * sizeof(float)
        int64_t coreNum = GetBlockDim();
        currentWorkspace[0] = static_cast<size_t>(coreNum) * sizeof(float);
    }
    else {
        currentWorkspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LightningIndexerKlLossTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 获取输入 shape: (..., D) -> M = totalElements / D
    const auto* targetScoreShape = context->GetInputShape(INDEXZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetScoreShape);
    int64_t dimNum = static_cast<int64_t>(targetScoreShape->GetDimNum());
    int64_t D = targetScoreShape->GetDim(dimNum - 1);
    int64_t totalElements = 1;
    for (int64_t i = 0; i < dimNum; i++) {
        totalElements *= targetScoreShape->GetDim(i);
    }
    int64_t M = totalElements / D; // 总行数

    auto attrs = context_->GetAttrs();
    float eps = *attrs->GetFloat(0);
    bool deterministic =  *attrs->GetBool(1) || (context_->GetDeterministic() == 1);

    // 获取数据类型
    auto dataType = context->GetInputDesc(INDEXZERO)->GetDataType();
    int64_t dtypeSize = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) ? 2 : 4;
    // UB 统一按 fp32 划分
    int64_t fp32Size = sizeof(float);
    int64_t tileLength = ComputeTileLength(D, ubSize, fp32Size);

    // 计算 tile 块总数
    int64_t totalTileNum = (M + tileLength - 1) / tileLength;

    // round-robin 分核
    int64_t formerNum = totalTileNum % coreNum;
    int64_t formerTileNum = (totalTileNum + coreNum - 1) / coreNum;
    int64_t tailTileNum = totalTileNum / coreNum;

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    LightningIndexerKlLossTilingData* tiling = context->GetTilingData<LightningIndexerKlLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LightningIndexerKlLossTilingData), 0, sizeof(LightningIndexerKlLossTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalLength = M;
    tiling->K = D;
    tiling->KAligned = (D + 7) / 8 * 8;
    tiling->formerNum = formerNum;
    tiling->formerTileNum = formerTileNum;
    tiling->tailTileNum = tailTileNum;
    tiling->tileLength = tileLength;
    tiling->eps = 1e-9f;
    tiling->coreNum = coreNum;

    context->SetBlockDim(coreNum);

    // TilingKey: [bit 0: deterministic] | [bit 1: isHalf]
    auto dataType = context->GetInputDesc(INDEXZERO)->GetDataType();
    bool isHalf = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16);
    ASCENDC_TPL_SEL_PARAM(context, deterministic, isHalf);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLightningIndexerKlLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LightningIndexerKlLoss).Tiling(LightningIndexerKlLossTilingFunc).TilingParse<LightningIndexerKlLossCompileInfo>(TilingParseForLightningIndexerKlLoss);
} // namespace optiling
