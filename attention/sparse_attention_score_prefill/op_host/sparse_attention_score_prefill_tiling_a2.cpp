/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
/*
 * =========================================================================================
 * sparse_attention_score_prefill —— A2 (ascend910b) 专用 Tiling 实现
 * =========================================================================================
 *
 * 【本文件职责】
 *   SASAPrefillTilingA2 架构实现: A2(ascend910b) workspace 计算 + tiling 填充
 *   与 A5 (sparse_attention_score_prefill_tiling.cpp) 完全独立, 互不影响
 *
 * 【A2 vs A5 核心差异】
 *   A5 (ascend950): L0C→UB 支持 Fixpipe 直接搬运, 无需 GM 中转
 *   A2 (ascend910b): L0C→UB 无法直接 Fixpipe, 需经 GM staging:
 *     QK L0C → Fixpipe → GM S staging → DataCopyPad → UB
 *   因此 A2 需要额外的 per-core GM S staging workspace
 *
 * 【A2 Workspace 布局】 (顺序固定, kernel 侧按偏移读取)
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ libapiWorkspace (libapiSize_)                                   │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ accumOut    [totalQ * kvHeads * topK * groupSize * D]           │
 *   │   fp32 (innerPrecise!=1) 或 bf16 (innerPrecise==1)              │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ softmaxMax  [totalQ * kvHeads * topK * groupSize] fp32          │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ softmaxSum  [totalQ * kvHeads * topK * groupSize] fp32          │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ GM S staging [blockDim * 2 * L0_TILE_M * blockSize * sizeof(bf16)]│
 *   │   per-core: 2 * 128 * 128 * 2 = 65536 bytes = 64KB (双缓冲)     │
 *   │   A2 专用, A5 无此区域                                          │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * 【GM S staging 用途】
 *   QK matmul 结果在 L0C 中, A2 无法直接 Fixpipe 到 UB
 *   流程: L0C → (DataCopy+SetFixpipeNz2ndFlag) → GM S staging → DataCopyPad → UB
 *   双缓冲 (2 级) 用于 QK/softmax/PV 三级流水线, 前一级写 GM S[0], 后一级读 GM S[1], 交替进行
 * =========================================================================================
 */

#include "sparse_attention_score_prefill_tiling.h"
#include <cstdint>
#include <cstdio>

namespace optiling {

// =================================================================================
// SASAPrefillTilingA2 —— A2 (ascend910b) 架构专用 tiling
//
// A2 特点: L0C→UB 需经 GM 中转, 需额外 per-core GM S staging workspace
// Workspace 布局: libapi | accumOut | softmaxMax | softmaxSum | GM S staging (per-core)
// =================================================================================

ge::graphStatus SASAPrefillTilingA2::CalculateWorkSpace(gert::TilingContext *context)
{
    (void)context;
    uint64_t slotOElems = static_cast<uint64_t>(groupSize_) * embeddingSize_;  // 每 slot O 元素数
    // A2 GM DMA uses 32-byte transactions.  Keep each max/sum slot at least
    // eight FP32 values wide so a groupSize=4 scatter cannot overwrite its
    // neighbouring sparse-selection slot.
    uint64_t slotStatElems = static_cast<uint64_t>((groupSize_ + 7U) & ~7U);
    uint64_t taskSlots = static_cast<uint64_t>(totalQTokens_) * kvHeads_ * topK_;

    // === 公共部分: accumOut + softmaxMax + softmaxSum (与 A5 相同) ===
    // accumOut: [totalQ, kvHeads, topK, groupSize, D] fp32 或 bf16
    //   innerPrecise==1: bf16 (PV fixpipe F322BF16 + Phase2 regbase cast)
    //   innerPrecise!=1: fp32 (默认路径)
    // softmaxMax/Sum: [totalQ, kvHeads, topK, roundUp(groupSize, 8)] fp32.
    accumOutSize_ = taskSlots * slotOElems;
    lseStatSize_ = taskSlots * slotStatElems;
    uint64_t accumOutBytes = accumOutSize_ *
        (innerPrecise_ == 1U ? sizeof(uint16_t) : sizeof(float));
    uint64_t userWorkspaceSize = accumOutBytes + (lseStatSize_ * 2U) * sizeof(float);
    // Align GM staging region to 32 bytes (Nd2Nz DataCopy GM→L1 requires 32-byte aligned src)
    userWorkspaceSize = (userWorkspaceSize + 31U) & ~static_cast<uint64_t>(31U);

    // === A2 专用: GM S staging workspace ===
    // A2 的 L0C→UB 无法直接 Fixpipe (A5 才支持), 需经 GM staging:
    //   QK L0C → Fixpipe → GM S staging → DataCopyPad → UB
    // 每个 core 需要 2 级 (双缓冲) GM S buffer:
    //   gmSPerCoreBytes = 2 * L0_TILE_M * blockSize * sizeof(bf16)
    //   = 2 * 128 * 128 * 2 = 65536 bytes = 64KB (blockSize=128)
    // 总 A2 GM S workspace = blockDim * gmSPerCoreBytes
    // Keep the GM staging capacity consistent with BlockMmadQK::L0_TILE_M.
    // A 64-row allocation overflows whenever a batched group has M > 64.
    constexpr uint32_t A2_L0_TILE_M = 64U;
    uint64_t gmSStageElems = static_cast<uint64_t>(A2_L0_TILE_M) * blockSize_;
    uint64_t gmSPerCoreBytes = 2U * gmSStageElems * sizeof(uint16_t);  // ElementS = bf16
    gmSWorkspaceSize_ = gmSPerCoreBytes;
    uint64_t totalGmSBytes = static_cast<uint64_t>(blockDim_) * gmSPerCoreBytes;
    userWorkspaceSize += totalGmSBytes;

    // === A2 专用: GM P staging workspace ===
    // VEC 计算 P=softmax(S) 后写入 GM (UB→GM DataCopyPad), CUBE 再从 GM 读取到 L1 (Nd2Nz)。
    // A2 不支持 VEC UB→L1 直写 (A5-only feature), 必须经 GM 中转。
    // P remains live for PRE_LAUNCH=2 batches before Cube consumes it, so
    // it requires three GM stages.  Two stages let softmax(batch+2) overwrite
    // P while Cube is still loading P(batch).
    uint64_t gmPStageElems = static_cast<uint64_t>(A2_L0_TILE_M) * blockSize_;
    uint64_t gmPPerCoreBytes = 3U * gmPStageElems * sizeof(uint16_t);  // ElementP = bf16
    gmPWorkspaceSize_ = gmPPerCoreBytes;
    uint64_t totalGmPBytes = static_cast<uint64_t>(blockDim_) * gmPPerCoreBytes;
    userWorkspaceSize += totalGmPBytes;

    workSpaceSize_ = libapiSize_ + userWorkspaceSize;
    printf("[SASAPrefillTilingA2] Workspace: accumOutSize=%lu lseStatSize=%lu "
           "accumOutBytes=%lu userWS=%lu totalWS=%lu (innerPrecise=%u, %s path, arch=A2)\n",
           accumOutSize_, lseStatSize_, accumOutBytes, userWorkspaceSize, workSpaceSize_,
           innerPrecise_, innerPrecise_ == 1U ? "bf16" : "fp32");
    printf("[SASAPrefillTilingA2] GM S staging: perCore=%lu bytes, total=%lu bytes (blockDim=%u)\n",
           gmSPerCoreBytes, totalGmSBytes, blockDim_);
    printf("[SASAPrefillTilingA2] GM P staging: perCore=%lu bytes, total=%lu bytes (blockDim=%u)\n",
           gmPPerCoreBytes, totalGmPBytes, blockDim_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASAPrefillTilingA2::FillTilingData(gert::TilingContext *context)
{
    printf("[SASAPrefillTilingA2] FillTilingData: begin\n");
    fflush(stdout);
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
    printf("[SASAPrefillTilingA2] FillTilingData: basic fields written\n");
    fflush(stdout);
    tilingData_->set_innerPrecise(innerPrecise_);
    tilingData_->set_accumOutSize(accumOutSize_);
    tilingData_->set_lseStatSize(lseStatSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_gmSWorkspaceSize(gmSWorkspaceSize_);  // A2: per-core GM S staging 字节数 (非零)
    tilingData_->set_gmPWorkspaceSize(gmPWorkspaceSize_);  // A2: per-core GM P staging 字节数 (非零)
    // 根据 innerPrecise 选择 TilingKey, 决定 kernel 编译路径:
    //   innerPrecise==1 → SASA_PREFILL_BF16_D128_INNER_LOW_TILING (bf16 O_partial)
    //   innerPrecise!=1 → SASA_PREFILL_BF16_D128_TILING (fp32 O_partial)
    uint64_t tilingKeyVal = (innerPrecise_ == 1U)
        ? SASA_PREFILL_BF16_D128_INNER_LOW_TILING
        : SASA_PREFILL_BF16_D128_TILING;
    printf("[SASAPrefillTilingA2] FillTilingData: workspace fields written\n");
    fflush(stdout);
    tilingData_->set_tilingKey(tilingKeyVal);
    context->SetTilingKey(tilingKeyVal);
    printf("[SASAPrefillTilingA2] FillTilingData: tiling key set\n");
    fflush(stdout);
    printf("[SASAPrefillTilingA2] FillTilingData: tilingKey=%lu (%s path), gmSWorkspaceSize=%lu, gmPWorkspaceSize=%lu (A2 GM S/P staging)\n",
           tilingKeyVal, innerPrecise_ == 1U ? "bf16 INNER_LOW" : "fp32", gmSWorkspaceSize_, gmPWorkspaceSize_);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
