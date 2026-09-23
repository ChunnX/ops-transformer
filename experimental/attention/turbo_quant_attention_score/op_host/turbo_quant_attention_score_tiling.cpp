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
 * \file turbo_quant_attention_score_tiling.cpp
 * \brief TurboQuantAttentionScore tiling：(q_token, kv_head, kv_block) 三级展平满核均分，
 *        tilingkey 按 mse_bits 特化
 */
#include "turbo_quant_attention_score_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

using namespace ge;

namespace optiling {

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_IDX_INDEX = 2;
constexpr uint32_t BLOCK_TABLE_INDEX = 6;
constexpr size_t ATTR_MSE_BITS_INDEX = 0;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t MAX_GROUP = 8;         // kernel UB 常驻 query 缓冲上限
constexpr int64_t MAX_KV_HEADS = 16;     // kernel norm/gamma 平面 UB 常驻上限
constexpr int64_t MAX_BT_ENTRIES = 8192; // kernel block_table UB 缓冲上限（batch * maxBlocksPerSeq）
constexpr int64_t MAX_BATCH = 1024;      // kernel seq_lens UB 缓冲上限

// tilingkey = 100 + mse_bits（与设计文档一致：102/103/104）
constexpr uint64_t TILING_KEY_MSE2 = 102;
constexpr uint64_t TILING_KEY_MSE3 = 103;
constexpr uint64_t TILING_KEY_MSE4 = 104;

static uint64_t GetTilingKeyByBits(int64_t bits)
{
    switch (bits) {
        case 2:
            return TILING_KEY_MSE2;
        case 4:
            return TILING_KEY_MSE4;
        default:
            return TILING_KEY_MSE3;
    }
}

ge::graphStatus TurboQuantAttentionScoreTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // ---- query shape: [num_q_tokens, num_q_heads, 128] ----
    const gert::StorageShape *qShape = context->GetInputShape(QUERY_INDEX);
    if (qShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = qShape->GetStorageShape();
    if (shape.GetDimNum() != 3 || shape.GetDim(2) != HEAD_DIM) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numTokens = shape.GetDim(0);
    const int64_t numQHeads = shape.GetDim(1);
    if (numTokens <= 0 || numQHeads <= 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- key_cache_idx shape: [num_blocks, 128, num_kv_heads, idx_bytes] ----
    const gert::StorageShape *idxShape = context->GetInputShape(KEY_IDX_INDEX);
    if (idxShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &ishape = idxShape->GetStorageShape();
    if (ishape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numKvHeads = ishape.GetDim(2);
    if (numKvHeads <= 0 || numKvHeads > MAX_KV_HEADS || numQHeads % numKvHeads != 0 ||
        numQHeads / numKvHeads > MAX_GROUP) {
        return ge::GRAPH_FAILED;
    }

    // ---- block_table shape: [batch, max_blocks_per_seq] ----
    const gert::StorageShape *btShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    if (btShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &bshape = btShape->GetStorageShape();
    if (bshape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = bshape.GetDim(0);
    const int64_t maxBlocks = bshape.GetDim(1);
    if (batch <= 0 || maxBlocks <= 0 || batch > MAX_BATCH || batch * maxBlocks > MAX_BT_ENTRIES) {
        return ge::GRAPH_FAILED;
    }
    // batch > 1 时按 decode 语义：每个请求恰好 1 个 query token（与 golden 一致）
    if (batch != 1 && batch != numTokens) {
        return ge::GRAPH_FAILED;
    }

    // ---- 属性 mse_bits ----
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *pBits = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS_INDEX);
    const int64_t mseBits = (pBits != nullptr) ? *pBits : 3;
    if (mseBits != 2 && mseBits != 3 && mseBits != 4) {
        return ge::GRAPH_FAILED;
    }

    // ---- 分核：work item = (q_token, kv_head, kv_block) 展平后尽量均分 ----
    const int64_t totalItems = numTokens * numKvHeads * maxBlocks;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t availableCoreNum = ascendcPlatform.GetCoreNum();
    if (availableCoreNum == 0) {
        availableCoreNum = 1;
    }
    const uint32_t usedCoreNum =
        (totalItems < static_cast<int64_t>(availableCoreNum)) ? static_cast<uint32_t>(totalItems) : availableCoreNum;
    const uint32_t itemsPerCore = static_cast<uint32_t>(totalItems / usedCoreNum);
    const uint32_t formerCoreNum = static_cast<uint32_t>(totalItems % usedCoreNum);

    auto *td = context->GetTilingData<TurboQuantAttentionScore::TurboQuantAttentionScoreTilingData>();
    if (td == nullptr) {
        return ge::GRAPH_FAILED;
    }
    td->numTokens = numTokens;
    td->totalItems = totalItems;
    td->numQHeads = static_cast<uint32_t>(numQHeads);
    td->numKvHeads = static_cast<uint32_t>(numKvHeads);
    td->groupSize = static_cast<uint32_t>(numQHeads / numKvHeads);
    td->batch = static_cast<uint32_t>(batch);
    td->maxBlocksPerSeq = static_cast<uint32_t>(maxBlocks);
    td->maxKvLen = static_cast<uint32_t>(maxBlocks * BLOCK_SIZE);
    td->usedCoreNum = usedCoreNum;
    td->formerCoreNum = formerCoreNum;
    td->itemsPerCore = itemsPerCore;
    td->mseBits = static_cast<uint32_t>(mseBits);

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(GetTilingKeyByBits(mseBits));

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForTurboQuantAttentionScore(gert::TilingParseContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto compileInfo = context->GetCompiledInfo<TurboQuantAttentionScoreCompileInfo>();
    if (compileInfo != nullptr) {
        compileInfo->coreNum = ascendcPlatform.GetCoreNum();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        compileInfo->ubSizePlatform = ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantAttentionScore)
    .Tiling(TurboQuantAttentionScoreTilingFunc)
    .TilingParse<TurboQuantAttentionScoreCompileInfo>(TilingPrepareForTurboQuantAttentionScore);

} // namespace optiling
