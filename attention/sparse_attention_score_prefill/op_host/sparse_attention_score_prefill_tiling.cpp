/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
/*
 * =========================================================================================
 * sparse_attention_score_prefill —— Tiling (切分策略) 实现
 * =========================================================================================
 *
 * 【本文件职责】
 *   1. SASAPrefillTilingBase 公共方法: 解析属性/shape/任务切分/GetTiling 编排/SetTilingData
 *   2. SASAPrefillTilingA5 架构实现: A5(ascend950) workspace 计算 + tiling 填充
 *   3. 入口函数: 架构检测 → 分发到 A5/A2 子类
 *
 * 【A2 专用 tiling 逻辑】
 *   位于 sparse_attention_score_prefill_tiling_a2.cpp, 与本文件完全独立
 *
 * 【Tiling 流程】
 *   GetNpuInfo → ParseAttrs → ParseInputTensors → ParseSeqlens
 *   → CalculateReverseIndexMeta → CalculateTaskSplit → CalculateWorkSpace(virtual) → FillTilingData(virtual)
 *
 * 【A5 Workspace 布局】 (顺序固定, kernel 侧按偏移读取)
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ libapiWorkspace (libapiSize_)                                   │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ accumOut    [totalQ * kvHeads * topK * groupSize * D]           │
 *   │   fp32 (innerPrecise!=1) 或 bf16 (innerPrecise==1)              │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ softmaxMax  [totalQ * kvHeads * topK * groupSize] fp32          │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ softmaxSum  [totalQ * kvHeads * topK * groupSize] fp32          │
 *   └─────────────────────────────────────────────────────────────────┘
 *   A5 无 GM S 中转 workspace (L0C→UB 直接 Fixpipe)
 * =========================================================================================
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

constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

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

// =================================================================================
// SASAPrefillTilingBase —— 公共方法实现
// =================================================================================

// 获取 NPU 硬件信息: AIV/AIC 核数、UB/L1 大小、libapi workspace 大小, 并检测架构 (A2/A5)
ge::graphStatus SASAPrefillTilingBase::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    blockDim_ = aicNum_;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    // 架构检测: A2 (ascend910b) UB=192KB, A5 (ascend950) UB=256KB
    // A2 的 L0C→UB 需经 GM 中转 (L0C→GM→UB), 需要 per-core GM S staging workspace;
    // A5 直接 L0C→UB Fixpipe, 无需额外 GM S workspace.
    isA2_ = (ubSize_ < 256U * 1024U);
    printf("[SASAPrefillTiling] NPU info: aivNum=%u aicNum=%u ubSize=%lu l1Size=%lu libapiSize=%lu arch=%s\n",
           aivNum_, aicNum_, ubSize_, l1Size_, libapiSize_, isA2_ ? "A2(ascend910b)" : "A5(ascend950)");
    return ge::GRAPH_SUCCESS;
}

// 解析算子属性: kvHeads, scaleValue, blockSize, topK, innerPrecise
ge::graphStatus SASAPrefillTilingBase::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    kvHeads_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_NUM_KV_HEADS_INDEX));
    scaleValue_ = static_cast<float>(*attrs->GetFloat(ATTR_SCALE_VALUE_INDEX));
    blockSize_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_BLOCK_SIZE_INDEX));
    topK_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_TOP_K_INDEX));
    innerPrecise_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_INNER_PRECISE_INDEX));
    printf("[SASAPrefillTiling] Attrs: kvHeads=%u scaleValue=%f blockSize=%u topK=%u innerPrecise=%u\n",
           kvHeads_, scaleValue_, blockSize_, topK_, innerPrecise_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTilingBase::ParseInputTensors(gert::TilingContext *context)
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

ge::graphStatus SASAPrefillTilingBase::ParseSeqlens(gert::TilingContext *context)
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

// 计算反向索引元数据: k2qNnzUpperBound (CSR 数据区每 head 上界)
ge::graphStatus SASAPrefillTilingBase::CalculateReverseIndexMeta(gert::TilingContext *context)
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
    printf("[SASAPrefillTiling] k2qNnzUpperBound=%u\n", k2qNnzUpperBound_);
    return ge::GRAPH_SUCCESS;
}

// 计算任务划分和核数分配
// blockDim = min(max(P1任务数, P2任务数), AIC核数)
// P1 和 P2 在同一个 kernel 内执行, 必须取两者最大值
ge::graphStatus SASAPrefillTilingBase::CalculateTaskSplit(gert::TilingContext *context)
{
    (void)context;
    uint32_t totalTaskP1 = numKvBlocks_ * kvHeads_;      // Phase1: KV block × KV head
    uint32_t totalTaskP2 = totalQTokens_ * kvHeads_;     // Phase2: Q token × KV head
    uint32_t totalTaskMax = std::max(totalTaskP1, totalTaskP2);
    // The A2 kernel uses SyncAll between QK, softmax, PV, and Phase2.  Every
    // AIC must therefore be launched even when it has no assigned task.
    blockDim_ = aicNum_;
    printf("[SASAPrefillTiling] TaskSplit: totalTaskP1=%u totalTaskP2=%u blockDim=%u (aicNum=%u, SyncAll)\n",
           totalTaskP1, totalTaskP2, blockDim_, aicNum_);
    return ge::GRAPH_SUCCESS;
}

// GetTiling 编排: 顺序调用各步骤, CalculateWorkSpace/FillTilingData 为 virtual, 由子类实现
ge::graphStatus SASAPrefillTilingBase::GetTiling(gert::TilingContext *context,
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

    printf("[SASAPrefillTiling] Enter CalculateReverseIndexMeta\n");
    fflush(stdout);
    ret = CalculateReverseIndexMeta(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    printf("[SASAPrefillTiling] Enter CalculateTaskSplit\n");
    fflush(stdout);
    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    printf("[SASAPrefillTiling] Enter CalculateWorkSpace\n");
    fflush(stdout);
    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    printf("[SASAPrefillTiling] Enter FillTilingData\n");
    fflush(stdout);
    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    return ge::GRAPH_SUCCESS;
}

// 设置 TilingData 到 context: schedule mode, blockDim, workspace, tiling buffer
ge::graphStatus SASAPrefillTilingBase::SetTilingData(gert::TilingContext *context,
    SparseAttentionScorePrefillTilingData &tilingData)
{
    printf("[SASAPrefillTiling] SetTilingData: begin\n");
    fflush(stdout);
    // 使用 SyncAll 需要设置为 batchmode 模式 (所有核同时启动)
    // 多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    printf("[SASAPrefillTiling] SetTilingData: schedule mode set\n");
    fflush(stdout);

    // 设置 blockDim (Phase1 和 Phase2 共用, 取 max(P1, P2) 和 AIC 核数的较小值)
    context->SetBlockDim(blockDim_);
    printf("[SASAPrefillTiling] SetTilingData: block dim set\n");
    fflush(stdout);

    // 设置 workspace 大小
    size_t *workspaces = context->GetWorkspaceSizes(1);
    printf("[SASAPrefillTiling] SetTilingData: workspace ptr acquired\n");
    fflush(stdout);
    workspaces[0] = static_cast<size_t>(workSpaceSize_);

    // 将 tiling data 序列化到 context 的 raw tiling buffer
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    printf("[SASAPrefillTiling] SetTilingData: raw buffer saved\n");
    fflush(stdout);
    printf("[SASAPrefillTiling] SetTilingData: dataSize=%zu capacity=%zu\n",
           tilingData.GetDataSize(), context->GetRawTilingData()->GetCapacity());
    fflush(stdout);
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    printf("[SASAPrefillTiling] SetTilingData: data size set\n");
    fflush(stdout);
    printf("[SASAPrefillTiling] SetTilingData: scheduleMode=BATCH blockDim=%u workspaceSize=%lu\n",
           blockDim_, workSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

// =================================================================================
// SASAPrefillTilingA5 —— A5 (ascend950) 架构专用 tiling
//
// A5 特点: L0C→UB 支持 Fixpipe 直接搬运, 无需 GM S 中转 workspace
// Workspace 布局: libapi | accumOut | softmaxMax | softmaxSum (无 GM S staging)
// =================================================================================

ge::graphStatus SASAPrefillTilingA5::CalculateWorkSpace(gert::TilingContext *context)
{
    (void)context;
    uint64_t slotOElems = static_cast<uint64_t>(groupSize_) * embeddingSize_;  // 每 slot O 元素数
    uint64_t slotStatElems = static_cast<uint64_t>(groupSize_);                // 每 slot stat 元素数
    uint64_t taskSlots = static_cast<uint64_t>(totalQTokens_) * kvHeads_ * topK_;

    // accumOut: [totalQ, kvHeads, topK, groupSize, D] fp32 或 bf16
    //   innerPrecise==1: bf16 (PV fixpipe F322BF16 + Phase2 regbase cast)
    //   innerPrecise!=1: fp32 (默认路径)
    // softmaxMax/Sum: [totalQ, kvHeads, topK, groupSize] fp32 (始终)
    accumOutSize_ = taskSlots * slotOElems;
    lseStatSize_ = taskSlots * slotStatElems;
    uint64_t accumOutBytes = accumOutSize_ *
        (innerPrecise_ == 1U ? sizeof(uint16_t) : sizeof(float));
    uint64_t userWorkspaceSize = accumOutBytes + (lseStatSize_ * 2U) * sizeof(float);

    // A5: L0C→UB 直接 Fixpipe, 无需 GM S/P 中转 workspace
    gmSWorkspaceSize_ = 0U;
    gmPWorkspaceSize_ = 0U;

    workSpaceSize_ = libapiSize_ + userWorkspaceSize;
    printf("[SASAPrefillTilingA5] Workspace: accumOutSize=%lu lseStatSize=%lu "
           "accumOutBytes=%lu userWS=%lu totalWS=%lu (innerPrecise=%u, %s path, arch=A5)\n",
           accumOutSize_, lseStatSize_, accumOutBytes, userWorkspaceSize, workSpaceSize_,
           innerPrecise_, innerPrecise_ == 1U ? "bf16" : "fp32");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTilingA5::FillTilingData(gert::TilingContext *context)
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
    tilingData_->set_accumOutSize(accumOutSize_);
    tilingData_->set_lseStatSize(lseStatSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_gmSWorkspaceSize(gmSWorkspaceSize_);  // A5: 始终为 0
    tilingData_->set_gmPWorkspaceSize(gmPWorkspaceSize_);  // A5: 始终为 0
    // 根据 innerPrecise 选择 TilingKey, 决定 kernel 编译路径:
    //   innerPrecise==1 → SASA_PREFILL_BF16_D128_INNER_LOW_TILING (bf16 O_partial)
    //   innerPrecise!=1 → SASA_PREFILL_BF16_D128_TILING (fp32 O_partial)
    uint64_t tilingKeyVal = (innerPrecise_ == 1U)
        ? SASA_PREFILL_BF16_D128_INNER_LOW_TILING
        : SASA_PREFILL_BF16_D128_TILING;
    tilingData_->set_tilingKey(tilingKeyVal);
    context->SetTilingKey(tilingKeyVal);
    printf("[SASAPrefillTilingA5] FillTilingData: tilingKey=%lu (%s path), gmSWorkspaceSize=0 (A5 no GM S staging)\n",
           tilingKeyVal, innerPrecise_ == 1U ? "bf16 INNER_LOW" : "fp32");
    return ge::GRAPH_SUCCESS;
}

// =================================================================================
// 入口: 架构检测 → 分发到 A5/A2 子类
// =================================================================================

ASCENDC_EXTERN_C ge::graphStatus TilingSparseAttentionScorePrefill(gert::TilingContext* context)
{
    printf("[SASAPrefillTiling] ===== TilingSparseAttentionScorePrefill ENTER =====\n");
    if (context == nullptr) { printf("[SASAPrefillTiling] context is null!\n"); return ge::GRAPH_FAILED; }

    // 架构检测: A2 (ascend910b) UB=192KB, A5 (ascend950) UB=256KB
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    bool isA2 = (ubSize < 256U * 1024U);
    printf("[SASAPrefillTiling] Arch dispatch: ubSize=%lu → %s tiling\n",
           ubSize, isA2 ? "A2(ascend910b)" : "A5(ascend950)");

    SparseAttentionScorePrefillTilingData tilingData;

    // 根据架构选择对应的 tiling 子类, A2/A5 逻辑完全独立
    SASAPrefillTilingA5 tilingA5;
    SASAPrefillTilingA2 tilingA2;
    SASAPrefillTilingBase *tiling = isA2
        ? static_cast<SASAPrefillTilingBase *>(&tilingA2)
        : static_cast<SASAPrefillTilingBase *>(&tilingA5);

    if (tiling->GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        printf("[SASAPrefillTiling] GetTiling success, calling SetTilingData\n");
        tiling->SetTilingData(context, tilingData);
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
