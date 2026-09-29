/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "slig_host_checks.h"
#include "mlp_sparse_lightning_indexer_grad_kl_loss_tiling.h"

namespace optiling {
namespace {
constexpr size_t QUERY_INPUT = 0;
constexpr size_t KEY_INPUT = 1;
constexpr size_t SPARSE_INDICES_INPUT = 5;
constexpr size_t LAYOUT_ATTR = 1;
constexpr size_t SPARSE_MODE_ATTR = 2;
constexpr size_t DETERMINISTIC_ATTR = 6;
constexpr uint64_t SYNC_ONLY_TILING_KEY = 0;
constexpr size_t SYSTEM_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

} // namespace

ge::graphStatus TilingMlpSparseLightningIndexerGradKLLoss(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MlpSparseLightningIndexerGradKLLoss", "context is null"),
                return ge::GRAPH_FAILED);
    const auto *compileInfo = context->GetCompileInfo<MlpSparseLightningIndexerGradKLLossCompileInfo>();
    uint32_t aicCoreNum = compileInfo == nullptr ? 0 : compileInfo->aicCoreNum;
    if (aicCoreNum == 0 && context->GetPlatformInfo() != nullptr) {
        platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        aicCoreNum = platform.GetCoreNumAic();
        if (aicCoreNum == 0) {
            // Some 910B runtime platform descriptions expose only the two
            // vector sub-cores of each mixed core.
            aicCoreNum = platform.GetCoreNumAiv() / 2;
        }
    }
    OP_CHECK_IF(aicCoreNum == 0, OP_LOGE(context, "no mixed AIC/AIV core available"),
                return ge::GRAPH_FAILED);

    const gert::StorageShape *queryStorageShape = context->GetInputShape(QUERY_INPUT);
    const gert::StorageShape *keyStorageShape = context->GetInputShape(KEY_INPUT);
    const gert::StorageShape *indicesStorageShape = context->GetInputShape(SPARSE_INDICES_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesStorageShape);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *layout = attrs->GetAttrPointer<char>(LAYOUT_ATTR);
    const int64_t *sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR);
    const bool *deterministic = attrs->GetAttrPointer<bool>(DETERMINISTIC_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, layout);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseMode);
    OP_CHECK_NULL_WITH_CONTEXT(context, deterministic);
    OP_CHECK_IF(!*deterministic, OP_LOGE(context, "sync-only kernel requires deterministic=true"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*sparseMode != 3, OP_LOGE(context, "sync-only kernel only supports sparse_mode=3"),
                return ge::GRAPH_FAILED);

    const gert::Shape &queryShape = queryStorageShape->GetStorageShape();
    const gert::Shape &keyShape = keyStorageShape->GetStorageShape();
    const gert::Shape &indicesShape = indicesStorageShape->GetStorageShape();
    int64_t totalSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    if (std::strcmp(layout, "TND") == 0) {
        OP_CHECK_IF(queryShape.GetDimNum() != 3 || keyShape.GetDimNum() != 3 || indicesShape.GetDimNum() != 3,
                    OP_LOGE(context, "TND inputs must be rank 3"), return ge::GRAPH_FAILED);
        totalSize = queryShape.GetDim(0);
        // The extracted synchronization graph models the original B=1
        // reproduction case. Sequence arrays remain in the public ABI, while
        // the graph dimensions come directly from the TND tensor lengths.
        s1Size = totalSize;
        s2Size = keyShape.GetDim(0);
    } else if (std::strcmp(layout, "BSND") == 0) {
        OP_CHECK_IF(queryShape.GetDimNum() != 4 || keyShape.GetDimNum() != 4 || indicesShape.GetDimNum() != 4,
                    OP_LOGE(context, "BSND inputs must be rank 4"), return ge::GRAPH_FAILED);
        totalSize = queryShape.GetDim(0) * queryShape.GetDim(1);
        s1Size = queryShape.GetDim(1);
        s2Size = keyShape.GetDim(1);
    } else {
        OP_LOGE(context, "layout must be TND or BSND, got %s", layout);
        return ge::GRAPH_FAILED;
    }

    const int64_t kSize = indicesShape.GetDim(indicesShape.GetDimNum() - 1);
    OP_CHECK_IF(totalSize <= 0 || s1Size <= 0 || s2Size < s1Size,
                OP_LOGE(context, "invalid sequence sizes: total=%ld, s1=%ld, s2=%ld", totalSize, s1Size, s2Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kSize <= 0 || kSize > 8192 || kSize % 128 != 0,
                OP_LOGE(context, "topk must be in (0, 8192] and aligned to 128, got %ld", kSize),
                return ge::GRAPH_FAILED);

    MlpSparseLightningIndexerGradKLLossTilingData tilingData;
    tilingData.set_totalSize(totalSize);
    tilingData.set_s1Size(s1Size);
    tilingData.set_s2Size(s2Size);
    tilingData.set_kSize(kSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetTilingKey(SYNC_ONLY_TILING_KEY);
    context->SetBlockDim(aicCoreNum);
    context->SetScheduleMode(1); // Match before: batchmode for deterministic mode-0 synchronization.
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = SYSTEM_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareMlpSparseLightningIndexerGradKLLoss(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MlpSparseLightningIndexerGradKLLoss", "parse context is null"),
                return ge::GRAPH_FAILED);
    auto *compileInfo = context->GetCompiledInfo<MlpSparseLightningIndexerGradKLLossCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->aicCoreNum = platform.GetCoreNumAic();
    if (compileInfo->aicCoreNum == 0) {
        compileInfo->aicCoreNum = platform.GetCoreNumAiv() / 2;
    }
    return compileInfo->aicCoreNum == 0 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MlpSparseLightningIndexerGradKLLoss)
    .Tiling(TilingMlpSparseLightningIndexerGradKLLoss)
    .TilingParse<MlpSparseLightningIndexerGradKLLossCompileInfo>(
        TilingPrepareMlpSparseLightningIndexerGradKLLoss);
} // namespace optiling
