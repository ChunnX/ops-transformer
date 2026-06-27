/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "sparse_attention_score_prefill_tiling.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <vector>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int BLOCK_TABLE_INDEX = 3;
constexpr int K2Q_ROW_PTR_INDEX = 4;
constexpr int K2Q_Q_INDICES_INDEX = 5;
constexpr int K2Q_SLOT_INDICES_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 7;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 8;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;

constexpr int ATTR_NUM_KV_HEADS_INDEX = 0;
constexpr int ATTR_SCALE_VALUE_INDEX = 1;
constexpr int ATTR_BLOCK_SIZE_INDEX = 2;
constexpr int ATTR_TOP_K_INDEX = 3;
constexpr int ATTR_INNER_PRECISE_INDEX = 4;

namespace optiling {

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) return 0;
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (align != 0) ? ((value + align - 1) / align * align) : value;
}

ge::graphStatus SASAPrefillTiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    blockDim_ = aicNum_;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    kvHeads_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_NUM_KV_HEADS_INDEX));
    scaleValue_ = static_cast<float>(*attrs->GetFloat(ATTR_SCALE_VALUE_INDEX));
    blockSize_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_BLOCK_SIZE_INDEX));
    topK_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_TOP_K_INDEX));
    innerPrecise_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_INNER_PRECISE_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::ParseInputTensors(gert::TilingContext *context)
{
    printf("[SASAPrefillTiling] Enter ParseInputTensors\n");
    // Query: [total_q_tokens, num_q_heads, D]
    auto qShape = context->GetInputShape(QUERY_INDEX);
    if (qShape == nullptr) { printf("[SASAPrefillTiling] qShape is null!\n"); return ge::GRAPH_FAILED; }
    totalQTokens_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_T));
    numHeads_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_D));
    printf("[SASAPrefillTiling] totalQTokens=%u numHeads=%u D=%u\n", totalQTokens_, numHeads_, embeddingSize_);

    // KV: [num_physical_blocks, blockSize, kvHeads, D]
    auto kShape = context->GetInputShape(KEY_INDEX);
    if (kShape == nullptr) { printf("[SASAPrefillTiling] kShape is null!\n"); return ge::GRAPH_FAILED; }
    if (kvHeads_ == 0) {
        kvHeads_ = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BLOCKED_KV_DIM_KV_HEAD));
    }
    groupSize_ = (kvHeads_ > 0) ? (numHeads_ / kvHeads_) : 1;
    printf("[SASAPrefillTiling] kvHeads=%u groupSize=%u\n", kvHeads_, groupSize_);

    // block_table: [batch_size, max_num_blocks]
    auto btShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    if (btShape == nullptr) { printf("[SASAPrefillTiling] btShape is null!\n"); return ge::GRAPH_FAILED; }
    batch_ = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(0));
    maxBlocksPerBatch_ = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(1));
    printf("[SASAPrefillTiling] batch=%u maxBlocksPerBatch=%u\n", batch_, maxBlocksPerBatch_);

    // Data type
    auto qTensor = context->GetInputDesc(QUERY_INDEX);
    dataType_ = qTensor->GetDataType();

    printf("[SASAPrefillTiling] ParseInputTensors done\n");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::ParseSeqlens(gert::TilingContext *context)
{
    printf("[SASAPrefillTiling] Enter ParseSeqlens\n");
    fflush(stdout);

    auto k2qRowPtrShape = context->GetInputShape(K2Q_ROW_PTR_INDEX);
    if (k2qRowPtrShape == nullptr) {
        printf("[SASAPrefillTiling] k2qRowPtr shape is null!\n");
        return ge::GRAPH_FAILED;
    }
    if (k2qRowPtrShape->GetStorageShape().GetDimNum() < 2) {
        printf("[SASAPrefillTiling] k2qRowPtr must be rank-2 [kvHeads, totalRows + 1]\n");
        return ge::GRAPH_FAILED;
    }
    numKvBlocks_ = static_cast<uint32_t>(k2qRowPtrShape->GetStorageShape().GetDim(1)) - 1;
    uint32_t rowPtrDim0 = static_cast<uint32_t>(k2qRowPtrShape->GetStorageShape().GetDim(0));
    printf("[SASAPrefillTiling] totalKvRows=%u k2qRowPtrHeads=%u kvHeads=%u\n",
           numKvBlocks_, rowPtrDim0, kvHeads_);
    fflush(stdout);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::CalculateReverseIndexMeta(gert::TilingContext *context)
{
    // k2qQIndices / k2qSlotIndices: [kvHeads, totalQ * topK]
    auto qIndicesShape = context->GetInputShape(K2Q_Q_INDICES_INDEX);
    if (qIndicesShape != nullptr && qIndicesShape->GetStorageShape().GetDimNum() >= 2) {
        k2qNnzUpperBound_ = static_cast<uint32_t>(qIndicesShape->GetStorageShape().GetDim(1));
    } else {
        k2qNnzUpperBound_ = totalQTokens_ * topK_;
    }
    if (k2qNnzUpperBound_ == 0) {
        k2qNnzUpperBound_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::CalculateTaskSplit(gert::TilingContext *context)
{
    (void)context;
    uint32_t totalTaskP1 = numKvBlocks_ * kvHeads_;
    uint32_t totalTaskP2 = totalQTokens_ * kvHeads_;
    uint32_t totalTaskMax = std::max(totalTaskP1, totalTaskP2);
    blockDim_ = std::min(totalTaskMax, std::max(aicNum_, aivNum_));
    if (blockDim_ == 0) {
        blockDim_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::CalculateWorkSpace(gert::TilingContext *context)
{
    (void)context;
    uint64_t oBytes = static_cast<uint64_t>(groupSize_) * embeddingSize_ * sizeof(float);
    partialSize_ = oBytes + static_cast<uint64_t>(groupSize_) * sizeof(float);

    // User workspace only stores Phase1 partials consumed by Phase2:
    // [totalQTokens, kvHeads, topK, partialSize]. Reverse-index tensors are
    // independent GM inputs and must not be counted here.
    uint64_t userWorkspaceSize = static_cast<uint64_t>(totalQTokens_) * kvHeads_ * topK_ * partialSize_;
    workSpaceSize_ = libapiSize_ + userWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::FillTilingData(gert::TilingContext *context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_groupSize(groupSize_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_numKvBlocks(numKvBlocks_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_k2qNnzUpperBound(k2qNnzUpperBound_);
    tilingData_->set_totalTaskNumP1(numKvBlocks_ * kvHeads_);
    tilingData_->set_totalTaskNumP2(totalQTokens_ * kvHeads_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_innerPrecise(innerPrecise_);
    tilingData_->set_partialSize(partialSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_tilingKey(SASA_PREFILL_BF16_D128_TILING);
    context->SetTilingKey(SASA_PREFILL_BF16_D128_TILING);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::GetTiling(gert::TilingContext *context,
    SparseAttentionScorePrefillTilingData &tilingData)
{
    tilingData_ = &tilingData;

    auto ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = ParseAttrs(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = ParseInputTensors(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = ParseSeqlens(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = CalculateReverseIndexMeta(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTiling::SetTilingData(gert::TilingContext *context,
    SparseAttentionScorePrefillTilingData &tilingData)
{
    // Set block dim computed from both Phase1 and Phase2 task counts.
    context->SetBlockDim(blockDim_);

    // Set workspace size
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(workSpaceSize_);

    // Set tiling data
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingSparseAttentionScorePrefill(gert::TilingContext* context)
{
    printf("[SASAPrefillTiling] ===== TilingSparseAttentionScorePrefill ENTER =====\n");
    if (context == nullptr) { printf("[SASAPrefillTiling] context is null!\n"); return ge::GRAPH_FAILED; }
    SparseAttentionScorePrefillTilingData tilingData;
    SASAPrefillTiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        printf("[SASAPrefillTiling] GetTiling success, calling SetTilingData\n");
        tiling.SetTilingData(context, tilingData);
        printf("[SASAPrefillTiling] ===== TilingSparseAttentionScorePrefill DONE =====\n");
        return ge::GRAPH_SUCCESS;
    }
    printf("[SASAPrefillTiling] GetTiling FAILED!\n");
    OP_LOGE(context->GetNodeName(), "GetTiling failed");
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForSparseAttentionScorePrefill(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseAttentionScorePrefill)
    .Tiling(TilingSparseAttentionScorePrefill)
    .TilingParse<SparseAttentionScorePrefillCompileInfo>(TilingPrepareForSparseAttentionScorePrefill);

}  // namespace optiling
