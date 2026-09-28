/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "turbo_quant_attention_score_tiling.h"

namespace optiling {
namespace {
constexpr size_t IN_QUERY_ROTATED = 0;
constexpr size_t IN_QUERY_QJL = 1;
constexpr size_t IN_KEY_CACHE_IDX = 2;
constexpr size_t IN_KEY_CACHE_QJL = 3;
constexpr size_t IN_KEY_CACHE_NORM = 4;
constexpr size_t IN_KEY_CACHE_GAMMA = 5;
constexpr size_t IN_BLOCK_TABLE = 6;
constexpr size_t IN_SEQ_LENS = 7;
constexpr size_t OUT_SCORES = 0;
constexpr size_t ATTR_MSE_BITS = 0;
constexpr uint32_t MAX_KV_TILE = 128;
constexpr float PI = 3.14159265358979323846F;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0U) {
        return value;
    }
    return (value + alignment - 1U) / alignment * alignment;
}

uint32_t PackedIdxBytes(uint32_t headDim, uint32_t mseBits)
{
    return (headDim * mseBits + 7U) / 8U;
}

uint32_t PackedQjlBytes(uint32_t headDim)
{
    return (headDim + 7U) / 8U;
}

uint64_t TilingKeyFor(uint32_t mseBits)
{
    if (mseBits == 2U) {
        return TQAS_TILING_KEY_MSE2;
    }
    if (mseBits == 4U) {
        return TQAS_TILING_KEY_MSE4;
    }
    return TQAS_TILING_KEY_MSE3;
}

struct RuntimeParams {
    uint32_t numTokens = 0;
    uint32_t numQHeads = 0;
    uint32_t numKvHeads = 0;
    uint32_t headDim = 0;
    uint32_t blockSize = TQAS_PAGED_BLOCK_SIZE;
    uint32_t numBlocks = 0;
    uint32_t idxPackedBytes = 0;
    uint32_t qjlPackedBytes = 0;
    uint32_t maxBatch = 0;
    uint32_t maxBlocksPerSeq = 0;
    uint32_t maxKvLen = 0;
    uint32_t kvTile = 0;
    uint32_t usedCoreNum = 0;
    uint32_t mseBits = 3;
    uint32_t groups = 0;
    uint32_t ubBlockSize = 0;
    float qjlCoeff = 0.0F;
};

ge::graphStatus ParseRuntimeParams(gert::TilingContext *context, platform_ascendc::PlatformAscendC &platform,
                                   RuntimeParams &params)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TurboQuantAttentionScore", "tiling context is nullptr"),
                return ge::GRAPH_FAILED);

    const auto queryShape = context->GetInputShape(IN_QUERY_ROTATED);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const auto queryQjlShape = context->GetInputShape(IN_QUERY_QJL);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryQjlShape);
    const auto idxShape = context->GetInputShape(IN_KEY_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    const auto qjlShape = context->GetInputShape(IN_KEY_CACHE_QJL);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlShape);
    const auto normShape = context->GetInputShape(IN_KEY_CACHE_NORM);
    OP_CHECK_NULL_WITH_CONTEXT(context, normShape);
    const auto gammaShape = context->GetInputShape(IN_KEY_CACHE_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    const auto tableShape = context->GetInputShape(IN_BLOCK_TABLE);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableShape);
    const auto seqShape = context->GetInputShape(IN_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqShape);
    const auto outShape = context->GetOutputShape(OUT_SCORES);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    const auto &query = queryShape->GetStorageShape();
    const auto &queryQjl = queryQjlShape->GetStorageShape();
    const auto &idx = idxShape->GetStorageShape();
    const auto &qjl = qjlShape->GetStorageShape();
    const auto &norm = normShape->GetStorageShape();
    const auto &gamma = gammaShape->GetStorageShape();
    const auto &table = tableShape->GetStorageShape();
    const auto &seq = seqShape->GetStorageShape();
    const auto &out = outShape->GetStorageShape();

    OP_CHECK_IF(query.GetDimNum() != 3, OP_LOGE(context, "query_rotated rank must be 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryQjl.GetDimNum() != 3, OP_LOGE(context, "query_qjl rank must be 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDimNum() != 4, OP_LOGE(context, "key_cache_idx rank must be 4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qjl.GetDimNum() != 4, OP_LOGE(context, "key_cache_qjl rank must be 4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(norm.GetDimNum() != 3, OP_LOGE(context, "key_cache_norm rank must be 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma.GetDimNum() != 3, OP_LOGE(context, "key_cache_gamma rank must be 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(table.GetDimNum() != 2, OP_LOGE(context, "block_table rank must be 2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seq.GetDimNum() != 1, OP_LOGE(context, "seq_lens rank must be 1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(out.GetDimNum() != 3, OP_LOGE(context, "attn_scores rank must be 3"), return ge::GRAPH_FAILED);

    for (size_t i = 0; i < 3; ++i) {
        OP_CHECK_IF(query.GetDim(i) <= 0, OP_LOGE(context, "invalid query_rotated dim %zu", i),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(query.GetDim(i) != queryQjl.GetDim(i),
                    OP_LOGE(context, "query_rotated/query_qjl dim %zu mismatch", i), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(idx.GetDim(0) <= 0, OP_LOGE(context, "num_blocks must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDim(0) != qjl.GetDim(0), OP_LOGE(context, "num_blocks mismatch idx/qjl"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDim(0) != norm.GetDim(0), OP_LOGE(context, "num_blocks mismatch idx/norm"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDim(0) != gamma.GetDim(0), OP_LOGE(context, "num_blocks mismatch idx/gamma"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDim(1) != static_cast<int64_t>(TQAS_PAGED_BLOCK_SIZE),
                OP_LOGE(context, "paged block_size must be %u", TQAS_PAGED_BLOCK_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qjl.GetDim(1) != static_cast<int64_t>(TQAS_PAGED_BLOCK_SIZE),
                OP_LOGE(context, "paged block_size must be %u", TQAS_PAGED_BLOCK_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(norm.GetDim(1) != static_cast<int64_t>(TQAS_PAGED_BLOCK_SIZE),
                OP_LOGE(context, "paged block_size must be %u", TQAS_PAGED_BLOCK_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma.GetDim(1) != static_cast<int64_t>(TQAS_PAGED_BLOCK_SIZE),
                OP_LOGE(context, "paged block_size must be %u", TQAS_PAGED_BLOCK_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDim(2) <= 0, OP_LOGE(context, "num_kv_heads must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDim(2) != qjl.GetDim(2), OP_LOGE(context, "num_kv_heads mismatch idx/qjl"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDim(2) != norm.GetDim(2), OP_LOGE(context, "num_kv_heads mismatch idx/norm"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx.GetDim(2) != gamma.GetDim(2), OP_LOGE(context, "num_kv_heads mismatch idx/gamma"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(table.GetDim(0) <= 0, OP_LOGE(context, "batch must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(table.GetDim(0) != seq.GetDim(0), OP_LOGE(context, "block_table/seq_lens batch mismatch"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(table.GetDim(1) <= 0, OP_LOGE(context, "max_blocks_per_seq must be positive"), return ge::GRAPH_FAILED);

    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *mseBitsAttr = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS);
    OP_CHECK_NULL_WITH_CONTEXT(context, mseBitsAttr);
    OP_CHECK_IF(*mseBitsAttr != 2 && *mseBitsAttr != 3 && *mseBitsAttr != 4,
                OP_LOGE(context, "mse_bits must be 2, 3, or 4, got %ld", *mseBitsAttr), return ge::GRAPH_FAILED);

    params.mseBits = static_cast<uint32_t>(*mseBitsAttr);
    params.numTokens = static_cast<uint32_t>(query.GetDim(0));
    params.numQHeads = static_cast<uint32_t>(query.GetDim(1));
    params.headDim = static_cast<uint32_t>(query.GetDim(2));
    params.numKvHeads = static_cast<uint32_t>(idx.GetDim(2));
    params.numBlocks = static_cast<uint32_t>(idx.GetDim(0));
    params.blockSize = TQAS_PAGED_BLOCK_SIZE;
    params.maxBatch = static_cast<uint32_t>(table.GetDim(0));
    params.maxBlocksPerSeq = static_cast<uint32_t>(table.GetDim(1));
    params.idxPackedBytes = static_cast<uint32_t>(idx.GetDim(3));
    params.qjlPackedBytes = static_cast<uint32_t>(qjl.GetDim(3));

    OP_CHECK_IF(params.headDim == 0U, OP_LOGE(context, "head_dim must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(params.numKvHeads == 0U, OP_LOGE(context, "num_kv_heads must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(params.numQHeads % params.numKvHeads != 0U,
                OP_LOGE(context, "GQA requires num_q_heads %% num_kv_heads == 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(params.maxBatch != 1U && params.numTokens != params.maxBatch,
                OP_LOGE(context, "when batch>1, num_tokens must equal batch"), return ge::GRAPH_FAILED);

    const uint32_t expectIdx = PackedIdxBytes(params.headDim, params.mseBits);
    const uint32_t expectQjl = PackedQjlBytes(params.headDim);
    OP_CHECK_IF(params.idxPackedBytes != expectIdx,
                OP_LOGE(context, "idx packed bytes mismatch: got %u expect %u", params.idxPackedBytes, expectIdx),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(params.qjlPackedBytes != expectQjl,
                OP_LOGE(context, "qjl packed bytes mismatch: got %u expect %u", params.qjlPackedBytes, expectQjl),
                return ge::GRAPH_FAILED);

    params.groups = params.numQHeads / params.numKvHeads;
    params.qjlCoeff = std::sqrt(PI / 2.0F) / static_cast<float>(params.headDim);

    const uint32_t tableCap = params.maxBlocksPerSeq * params.blockSize;
    const uint32_t outKvLen = static_cast<uint32_t>(out.GetDim(2));
    OP_CHECK_IF(out.GetDim(0) != static_cast<int64_t>(params.numTokens),
                OP_LOGE(context, "attn_scores token dim mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(out.GetDim(1) != static_cast<int64_t>(params.numQHeads),
                OP_LOGE(context, "attn_scores head dim mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outKvLen == 0U || outKvLen > tableCap, OP_LOGE(context, "attn_scores kv_len invalid"),
                return ge::GRAPH_FAILED);
    params.maxKvLen = outKvLen;

    params.ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(params.ubBlockSize == 0U, OP_LOGE(context, "ubBlockSize is zero"), return ge::GRAPH_FAILED);

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    OP_CHECK_IF(ubBytes == 0U, OP_LOGE(context, "UB size is zero"), return ge::GRAPH_FAILED);
    const uint64_t alignBytes = static_cast<uint64_t>(params.ubBlockSize);
    const uint64_t budget = ubBytes * 4U / 5U;
    const uint64_t headBytes = AlignUp(static_cast<uint64_t>(params.headDim) * sizeof(float), alignBytes);
    const uint64_t headBf16 = AlignUp(static_cast<uint64_t>(params.headDim) * sizeof(uint16_t), alignBytes);
    const bool isPrefill = (params.numTokens > 1U && params.maxBatch == 1U);
    constexpr uint32_t K_SHARE = 4U;
    const uint64_t fixedBytes = 2U * headBf16 + (2U * K_SHARE + 4U) * headBytes +
                                AlignUp(16U * sizeof(float), alignBytes) + AlignUp(24U * sizeof(float), alignBytes) +
                                AlignUp(static_cast<uint64_t>(params.headDim) * sizeof(uint32_t), alignBytes) +
                                AlignUp(256U * 8U * sizeof(float), alignBytes);
    uint64_t perKv =
        (static_cast<uint64_t>(K_SHARE) + 1U) * sizeof(float) +
        static_cast<uint64_t>(params.numKvHeads) * (static_cast<uint64_t>(params.idxPackedBytes) +
                                                    params.qjlPackedBytes + 2U * sizeof(uint16_t) + 2U * sizeof(float));
    if (isPrefill) {
        // dequant+signs cache, norm/gamma rows, and batched-dot scratch (1x headDim per kv).
        perKv += 3U * static_cast<uint64_t>(params.headDim) * sizeof(float) + 2U * sizeof(float) +
                 8U * sizeof(float); // BlockReduceSum tv scratch
    }
    OP_CHECK_IF(budget <= fixedBytes + perKv, OP_LOGE(context, "UB size %lu insufficient for tiling", ubBytes),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(perKv == 0U, OP_LOGE(context, "perKv budget is zero"), return ge::GRAPH_FAILED);
    params.kvTile = static_cast<uint32_t>(
        std::min<uint64_t>(std::min<uint64_t>((budget - fixedBytes) / perKv, MAX_KV_TILE), params.maxKvLen));
    OP_CHECK_IF(params.kvTile == 0U, OP_LOGE(context, "calculated kvTile is zero"), return ge::GRAPH_FAILED);
    // Cap tile for A2 192KB UB loop overhead (decode/prefill share the same upper bound).
    params.kvTile = std::min(params.kvTile, 64U);

    uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0U, OP_LOGE(context, "AIV core num is zero"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(params.kvTile == 0U, OP_LOGE(context, "kvTile is zero before numKvTiles"), return ge::GRAPH_FAILED);
    const uint64_t numKvTiles = (static_cast<uint64_t>(params.maxKvLen) + params.kvTile - 1U) / params.kvTile;
    if (isPrefill) {
        const uint64_t totalUnits = static_cast<uint64_t>(params.numKvHeads) * numKvTiles;
        params.usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(totalUnits, coreNum));
    } else {
        const uint64_t totalRows = static_cast<uint64_t>(params.numTokens) * params.numQHeads;
        const uint64_t totalUnits = totalRows * numKvTiles;
        params.usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(totalUnits, coreNum));
    }
    OP_CHECK_IF(params.usedCoreNum == 0U, OP_LOGE(context, "usedCoreNum is zero"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TurboQuantAttentionScoreTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TurboQuantAttentionScore", "tiling context is nullptr"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);

    RuntimeParams params;
    OP_CHECK_IF(ParseRuntimeParams(context, platform, params) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to parse TurboQuantAttentionScore tiling parameters"),
                return ge::GRAPH_FAILED);

    TurboQuantAttentionScoreTilingData tiling;
    tiling.set_numTokens(params.numTokens);
    tiling.set_numQHeads(params.numQHeads);
    tiling.set_numKvHeads(params.numKvHeads);
    tiling.set_headDim(params.headDim);
    tiling.set_blockSize(params.blockSize);
    tiling.set_numBlocks(params.numBlocks);
    tiling.set_idxPackedBytes(params.idxPackedBytes);
    tiling.set_qjlPackedBytes(params.qjlPackedBytes);
    tiling.set_maxBatch(params.maxBatch);
    tiling.set_maxBlocksPerSeq(params.maxBlocksPerSeq);
    tiling.set_maxKvLen(params.maxKvLen);
    tiling.set_kvTile(params.kvTile);
    tiling.set_usedCoreNum(params.usedCoreNum);
    tiling.set_mseBits(params.mseBits);
    tiling.set_groups(params.groups);
    tiling.set_ubBlockSize(params.ubBlockSize);
    tiling.set_qjlCoeff(params.qjlCoeff);

    context->SetBlockDim(params.usedCoreNum);
    context->SetTilingKey(TilingKeyFor(params.mseBits));
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;

    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForTurboQuantAttentionScore(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TurboQuantAttentionScore", "tiling parse context is nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<TurboQuantAttentionScoreCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivCoreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(compileInfo->aivCoreNum <= 0,
                OP_LOGE(context, "aivCoreNum must be positive, got %ld", compileInfo->aivCoreNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfo->ubSize == 0U, OP_LOGE(context, "ubSize must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfo->ubBlockSize == 0U, OP_LOGE(context, "ubBlockSize must be positive"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantAttentionScore)
    .Tiling(TurboQuantAttentionScoreTilingFunc)
    .TilingParse<TurboQuantAttentionScoreCompileInfo>(TilingPrepareForTurboQuantAttentionScore);
} // namespace optiling
