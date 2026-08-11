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
 * \file msa_index_score_tiling.cpp
 * \brief MsaIndexScore Tiling 实现（A2/A3 P0：PA BBND + sparse_mode 0/3）。
 */

#include "msa_index_score_tiling.h"
#include "../op_kernel/msa_index_score_common.h"

using namespace ge;
using namespace MsaIndexScoreNs;

namespace optiling {
namespace {
constexpr uint32_t MSA_QUERY_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_BBND_DIM_NUM = 4;
constexpr uint32_t MSA_BLOCK_TABLE_DIM_NUM = 2;
constexpr uint32_t MSA_SCALE_DIM_NUM = 3;
constexpr uint32_t MSA_ATTEN_MASK_DIM_NUM = 2;

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
    const gert::StorageShape *keyShape = context->GetInputShape(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    // P0：PageAttention 必须传 block_table
    const gert::StorageShape *blockTableShape = context->GetOptionalInputShape(MSA_IDX_BLOCK_TABLE);
    OP_CHECK_IF(blockTableShape == nullptr, OP_LOGE(context, "PageAttention requires block_table."),
                return ge::GRAPH_FAILED);
    const gert::StorageShape *actualSeqQlenShape = context->GetOptionalInputShape(MSA_IDX_ACTUAL_SEQ_QLEN);
    OP_CHECK_IF(actualSeqQlenShape == nullptr, OP_LOGE(context, "TND query requires actual_seq_qlen."),
                return ge::GRAPH_FAILED);
    const gert::StorageShape *actualSeqKlenShape = context->GetOptionalInputShape(MSA_IDX_ACTUAL_SEQ_KLEN);
    OP_CHECK_IF(actualSeqKlenShape == nullptr, OP_LOGE(context, "PageAttention requires actual_seq_klen."),
                return ge::GRAPH_FAILED);
    const gert::StorageShape *startLocShape = context->GetInputShape(MSA_IDX_START_LOC);
    OP_CHECK_NULL_WITH_CONTEXT(context, startLocShape);

    const gert::Shape &q = queryShape->GetStorageShape();
    const gert::Shape &kc = keyShape->GetStorageShape();
    const gert::Shape &bt = blockTableShape->GetStorageShape();

    OP_CHECK_IF(q.GetDimNum() != MSA_QUERY_DIM_NUM,
                OP_LOGE(context, "query must be TND(3 dims), got %zu.", q.GetDimNum()), return ge::GRAPH_FAILED);
    // P0：仅 BBND [block_num, block_size, N2, D]
    OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_BBND_DIM_NUM,
                OP_LOGE(context, "P0 only supports key BBND(4 dims), got %zu.", kc.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(bt.GetDimNum() != MSA_BLOCK_TABLE_DIM_NUM,
                OP_LOGE(context, "block_table must have 2 dims, got %zu.", bt.GetDimNum()), return ge::GRAPH_FAILED);

    info.totalQ = static_cast<uint32_t>(q.GetDim(0));
    info.numQHeads = static_cast<uint32_t>(q.GetDim(1));
    info.headDim = static_cast<uint32_t>(q.GetDim(2));
    info.numPages = static_cast<uint32_t>(kc.GetDim(0));
    info.blockSize = static_cast<uint32_t>(kc.GetDim(1));
    info.numKvHeads = static_cast<uint32_t>(kc.GetDim(2));
    info.batch = static_cast<uint32_t>(bt.GetDim(0));
    info.maxBlocksPerBatch = static_cast<uint32_t>(bt.GetDim(1));

    OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(3)) != info.headDim,
                OP_LOGE(context, "key headDim(%ld) must equal query headDim(%u).", kc.GetDim(3), info.headDim),
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
    OP_CHECK_IF(info.batch == 0 || info.totalQ == 0 || info.maxBlocksPerBatch == 0 || info.numPages == 0,
                OP_LOGE(context, "batch/totalQ/maxBlocksPerBatch/numPages must be positive."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint32_t>(startLocShape->GetStorageShape().GetDim(0)) != info.batch,
                OP_LOGE(context, "start_loc size must equal batch."), return ge::GRAPH_FAILED);

    const auto *queryDesc = context->GetInputDesc(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    info.queryDtype = queryDesc->GetDataType();
    OP_CHECK_IF(info.queryDtype != ge::DT_BF16 && info.queryDtype != ge::DT_FLOAT16,
                OP_LOGE(context, "query dtype must be bf16 or fp16 on A2/A3."), return ge::GRAPH_FAILED);

    const auto *keyDesc = context->GetInputDesc(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    info.keyDtype = keyDesc->GetDataType();
    info.isQuant = (info.keyDtype == ge::DT_INT8);
    if (info.isQuant) {
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT16,
                    OP_LOGE(context, "int8 key currently requires fp16 query."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(info.keyDtype != info.queryDtype,
                    OP_LOGE(context, "non-quant key dtype must match query dtype."), return ge::GRAPH_FAILED);
    }

    const gert::StorageShape *scaleShape = context->GetOptionalInputShape(MSA_IDX_SCALE);
    if (info.isQuant) {
        OP_CHECK_IF(scaleShape == nullptr,
                    OP_LOGE(context, "int8 key requires scale of shape [NP, N_kv, block_size]."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &sc = scaleShape->GetStorageShape();
        OP_CHECK_IF(sc.GetDimNum() != MSA_SCALE_DIM_NUM,
                    OP_LOGE(context, "dequant scale must be 3D [NP, N_kv, block_size], got rank %zu.", sc.GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(sc.GetDim(0)) != info.numPages ||
                        static_cast<uint32_t>(sc.GetDim(1)) != info.numKvHeads ||
                        static_cast<uint32_t>(sc.GetDim(2)) != info.blockSize,
                    OP_LOGE(context, "dequant scale shape must be [%u, %u, %u], got [%ld, %ld, %ld].", info.numPages,
                            info.numKvHeads, info.blockSize, sc.GetDim(0), sc.GetDim(1), sc.GetDim(2)),
                    return ge::GRAPH_FAILED);
        const auto *scaleDesc = context->GetOptionalInputDesc(MSA_IDX_SCALE);
        OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
        OP_CHECK_IF(scaleDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context, "dequant scale dtype must be float32."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(scaleShape != nullptr, OP_LOGE(context, "non-quant path must not pass scale."),
                    return ge::GRAPH_FAILED);
    }

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *sparseModeAttr = attrs->GetInt(MSA_ATTR_SPARSE_MODE);
    info.sparseMode = (sparseModeAttr == nullptr) ? MSA_SPARSE_MODE_RIGHT_DOWN
                                                  : static_cast<uint32_t>(*sparseModeAttr);
    OP_CHECK_IF(info.sparseMode != MSA_SPARSE_MODE_DEFAULT && info.sparseMode != MSA_SPARSE_MODE_RIGHT_DOWN,
                OP_LOGE(context, "sparse_mode must be 0 or 3, got %u.", info.sparseMode), return ge::GRAPH_FAILED);

    const gert::StorageShape *attenMaskShape = context->GetOptionalInputShape(MSA_IDX_ATTEN_MASK);
    if (info.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN) {
        OP_CHECK_IF(attenMaskShape == nullptr,
                    OP_LOGE(context, "sparse_mode=3 requires atten_mask of shape [2048, 2048]."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &am = attenMaskShape->GetStorageShape();
        OP_CHECK_IF(am.GetDimNum() != MSA_ATTEN_MASK_DIM_NUM ||
                        static_cast<uint32_t>(am.GetDim(0)) != MSA_ATTEN_MASK_SIZE ||
                        static_cast<uint32_t>(am.GetDim(1)) != MSA_ATTEN_MASK_SIZE,
                    OP_LOGE(context, "atten_mask must be [2048, 2048], got rank=%zu dims=[%ld,%ld].", am.GetDimNum(),
                            am.GetDimNum() > 0 ? am.GetDim(0) : -1, am.GetDimNum() > 1 ? am.GetDim(1) : -1),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(attenMaskShape != nullptr,
                    OP_LOGE(context, "sparse_mode=0 must not pass atten_mask."), return ge::GRAPH_FAILED);
    }

    info.initBlocks = MSA_DEFAULT_INIT_BLOCKS;
    info.localBlocks = MSA_DEFAULT_LOCAL_BLOCKS;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DoTiling(gert::TilingContext *context, const MsaIndexScoreInfo &info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(info.platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0, OP_LOGE(context, "aic core num is 0."), return ge::GRAPH_FAILED);

    const uint32_t scoreBlockStride = RoundUpU32(info.maxBlocksPerBatch, MSA_SCORE_STRIDE_ALIGN);
    const uint32_t sWsElems = aicNum * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
    const uint32_t kScratchBytes = info.isQuant ? (aicNum * MSA_K_SCRATCH_ELEM_NUM * sizeof(uint16_t)) : 0U;
    const uint32_t kScratchOffsetElems = sWsElems;

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
    tilingData.set_isQuant(info.isQuant ? 1U : 0U);
    tilingData.set_sparseMode(info.sparseMode);
    tilingData.set_initBlocks(info.initBlocks);
    tilingData.set_localBlocks(info.localBlocks);
    tilingData.set_numPages(info.numPages);

    tilingData.set_strideQt(info.numQHeads * info.headDim);
    tilingData.set_strideQn(info.headDim);
    tilingData.set_strideKvBlock(info.blockSize * info.numKvHeads * info.headDim);
    tilingData.set_strideKvToken(info.numKvHeads * info.headDim);
    tilingData.set_strideScalePage(info.numKvHeads * info.blockSize);
    tilingData.set_strideScaleHead(info.blockSize);
    tilingData.set_strideOutHead(info.totalQ * scoreBlockStride);
    tilingData.set_strideOutToken(scoreBlockStride);
    tilingData.set_kScratchOffsetElems(kScratchOffsetElems);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aicNum, aicNum, aivNum));

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize() +
                    static_cast<size_t>(sWsElems) * sizeof(float) + static_cast<size_t>(kScratchBytes);

    uint64_t tilingKey = MSA_TILING_KEY_FP16;
    if (info.isQuant) {
        tilingKey = MSA_TILING_KEY_FP16_INT8;
    } else {
        tilingKey = (info.queryDtype == ge::DT_BF16) ? MSA_TILING_KEY_BF16 : MSA_TILING_KEY_FP16;
    }
    context->SetTilingKey(tilingKey);
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
