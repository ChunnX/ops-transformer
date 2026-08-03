/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file msa_index_score_tiling.cpp
 * \brief MsaIndexScore Tiling 实现。
 */

#include "msa_index_score_tiling.h"
#include "../op_kernel/msa_index_score_common.h"

using namespace ge;
using namespace MsaIndexScoreNs;

namespace optiling {
namespace {
constexpr uint32_t MSA_QUERY_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_CACHE_DIM_NUM = 4;
constexpr uint32_t MSA_BLOCK_TABLE_DIM_NUM = 2;

inline uint32_t RoundUpU32(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

ge::graphStatus ParseAndCheck(gert::TilingContext *context, MsaIndexScoreInfo &info)
{
    info.platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(info.platformInfo == nullptr, OP_LOGE(context, "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    const gert::StorageShape *queryShape = context->GetInputShape(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::StorageShape *keyCacheShape = context->GetInputShape(MSA_IDX_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyCacheShape);
    const gert::StorageShape *blockTableShape = context->GetInputShape(MSA_IDX_BLOCK_TABLE);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTableShape);

    const gert::Shape &q = queryShape->GetStorageShape();
    const gert::Shape &kc = keyCacheShape->GetStorageShape();
    const gert::Shape &bt = blockTableShape->GetStorageShape();

    OP_CHECK_IF(q.GetDimNum() != MSA_QUERY_DIM_NUM,
                OP_LOGE(context, "query must be TND(3 dims), got %zu.", q.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_CACHE_DIM_NUM,
                OP_LOGE(context, "key_cache must be BSND(4 dims), got %zu.", kc.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(bt.GetDimNum() != MSA_BLOCK_TABLE_DIM_NUM,
                OP_LOGE(context, "block_table must have 2 dims, got %zu.", bt.GetDimNum()), return ge::GRAPH_FAILED);

    info.totalQ = static_cast<uint32_t>(q.GetDim(0));      // 0: T
    info.numQHeads = static_cast<uint32_t>(q.GetDim(1));   // 1: numQHeads
    info.headDim = static_cast<uint32_t>(q.GetDim(2));     // 2: headDim
    info.blockSize = static_cast<uint32_t>(kc.GetDim(1));  // 1: blockSize
    info.numKvHeads = static_cast<uint32_t>(kc.GetDim(2)); // 2: numKvHeads
    info.batch = static_cast<uint32_t>(bt.GetDim(0));      // 0: batch
    info.maxBlocksPerBatch = static_cast<uint32_t>(bt.GetDim(1)); // 1: maxBlocksPerBatch

    OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(3)) != info.headDim, // 3: headDim
                OP_LOGE(context, "key_cache headDim(%ld) must equal query headDim(%u).", kc.GetDim(3), info.headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.blockSize != MSA_BLOCK_SIZE,
                OP_LOGE(context, "only blockSize == %u is supported, got %u.", MSA_BLOCK_SIZE, info.blockSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.headDim == 0 || (info.headDim % MSA_SCORE_STRIDE_ALIGN) != 0,
                OP_LOGE(context, "headDim(%u) must be a positive multiple of %u.", info.headDim,
                        MSA_SCORE_STRIDE_ALIGN),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.headDim > MSA_K_TILE,
                OP_LOGE(context, "headDim(%u) larger than %u is not supported yet.", info.headDim, MSA_K_TILE),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.numKvHeads == 0 || (info.numQHeads % info.numKvHeads) != 0,
                OP_LOGE(context, "numQHeads(%u) must be divisible by numKvHeads(%u).", info.numQHeads,
                        info.numKvHeads),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.batch == 0 || info.totalQ == 0 || info.maxBlocksPerBatch == 0,
                OP_LOGE(context, "batch/totalQ/maxBlocksPerBatch must be positive."), return ge::GRAPH_FAILED);

    info.hasScale = (context->GetOptionalInputShape(MSA_IDX_SCALE) != nullptr);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *blockSizeAttr = attrs->GetInt(MSA_ATTR_BLOCK_SIZE);
    if (blockSizeAttr != nullptr) {
        OP_CHECK_IF(static_cast<uint32_t>(*blockSizeAttr) != info.blockSize,
                    OP_LOGE(context, "attr block_size(%ld) mismatches key_cache dim1(%u).", *blockSizeAttr,
                            info.blockSize),
                    return ge::GRAPH_FAILED);
    }
    const bool *fillTailAttr = attrs->GetBool(MSA_ATTR_FILL_INVALID_TAIL);
    info.fillInvalidTail = (fillTailAttr == nullptr) ? true : *fillTailAttr;

    const auto *queryDesc = context->GetInputDesc(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    info.queryDtype = queryDesc->GetDataType();
    OP_CHECK_IF(info.queryDtype != ge::DT_BF16 && info.queryDtype != ge::DT_FLOAT16,
                OP_LOGE(context, "query dtype must be bf16 or fp16."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DoTiling(gert::TilingContext *context, const MsaIndexScoreInfo &info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(info.platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0, OP_LOGE(context, "aic core num is 0."), return ge::GRAPH_FAILED);

    const uint32_t scoreBlockStride = RoundUpU32(info.maxBlocksPerBatch, MSA_SCORE_STRIDE_ALIGN);

    MsaIndexScoreTilingData tilingData;
    tilingData.set_batch(info.batch);
    tilingData.set_totalQ(info.totalQ);
    tilingData.set_numQHeads(info.numQHeads);
    tilingData.set_numKvHeads(info.numKvHeads);
    tilingData.set_gqaGroup(info.numQHeads / info.numKvHeads);
    tilingData.set_headDim(info.headDim);
    tilingData.set_blockSize(info.blockSize);
    tilingData.set_maxBlocksPerBatch(info.maxBlocksPerBatch);
    tilingData.set_scoreBlockStride(scoreBlockStride);
    tilingData.set_usedCoreNum(aicNum);
    tilingData.set_hasScale(info.hasScale ? 1U : 0U);
    tilingData.set_fillInvalidTail(info.fillInvalidTail ? 1U : 0U);

    // 输入经 AutoContiguous 归一化，按连续布局推导 stride（元素为单位）。
    tilingData.set_strideQt(info.numQHeads * info.headDim);
    tilingData.set_strideQn(info.headDim);
    tilingData.set_strideKvBlock(info.blockSize * info.numKvHeads * info.headDim);
    tilingData.set_strideKvToken(info.numKvHeads * info.headDim);
    tilingData.set_strideScaleT(info.numQHeads);
    tilingData.set_strideScaleN(1U);
    tilingData.set_strideOutHead(info.totalQ * scoreBlockStride);
    tilingData.set_strideOutToken(scoreBlockStride);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aicNum, aicNum, aivNum));

    // AIC -> AIV 的 S 中间结果轮转缓冲：每个 AIC 私有 MSA_WORKSPACE_STAGES 个 slot。
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize() +
                    static_cast<size_t>(aicNum) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM * sizeof(float);

    context->SetTilingKey(info.queryDtype == ge::DT_BF16 ? MSA_TILING_KEY_BF16 : MSA_TILING_KEY_FP16);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingPrepareForMsaIndexScore(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForMsaIndexScore(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("MsaIndexScore", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    MsaIndexScoreInfo info;
    if (ParseAndCheck(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return DoTiling(context, info);
}

IMPL_OP_OPTILING(MsaIndexScore)
    .Tiling(TilingForMsaIndexScore)
    .TilingParse<MsaIndexScoreCompileInfo>(TilingPrepareForMsaIndexScore);

} // namespace optiling
