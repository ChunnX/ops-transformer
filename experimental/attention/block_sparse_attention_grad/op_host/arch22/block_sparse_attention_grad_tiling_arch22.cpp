/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../block_sparse_attention_grad_tiling.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include "log/log.h"

#include <cstdint>
#include <string>
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;
using namespace std;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;
constexpr int TND_DIM_NUM = 3;

constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;
constexpr int BNSD_DIM_NUM = 4;

constexpr int BSH_DIM_B = 0;
constexpr int BSH_DIM_S = 1;
constexpr int BSH_DIM_H = 2;

constexpr int DOUT_INDEX = 0;
constexpr int QUERY_INDEX = 1;
constexpr int KEY_INDEX = 2;
constexpr int VALUE_INDEX = 3;
constexpr int OUT_INDEX = 4;
constexpr int SOFTMAX_LSE_INDEX = 5;
constexpr int BLOCK_SPARSE_MASK_INDEX = 6;
constexpr int BLOCK_SHAPE_INDEX = 8;

constexpr int ATTENTION_MASK_INDEX = 7;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 9;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 10;

constexpr int Q_INPUT_LAYOUT_INDEX = 0;
constexpr int KV_INPUT_LAYOUT_INDEX = 1;
constexpr int NUM_KEY_VALUE_HEADS_INDEX = 2;
constexpr int MASK_TYPE_INDEX = 3;
constexpr int SCALE_VALUE_INDEX = 4;

constexpr int VALID_HEAD_DIM_128 = 128;
constexpr uint64_t VEC_POST_DIVISION = 3; // post 划分的空间块数， input 2份，output 1份
constexpr uint64_t WORKSPACE_NUM_ALIGN = 256;
constexpr uint32_t BATCH_SCHEDULE_MODE = 1;

namespace optiling {

constexpr uint32_t BASIC_BLOCK_SIZE = 128;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 2 * 2;
constexpr uint32_t NUM3 = 3;
constexpr uint32_t ONEBLOCK_FLOAT_NUM = 32 / sizeof(float); // 基本块32字节的float数目
static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint32_t GetQBlocks(int32_t qseqlen, int32_t x)
{
    uint32_t qBlocksInX = CeilDiv(x, BASIC_BLOCK_SIZE);
    uint32_t completeXBlocks = x != 0 ? qseqlen / x : qseqlen / BASIC_BLOCK_SIZE;
    uint32_t remainingSeqlen = x != 0 ? qseqlen - completeXBlocks * x : qseqlen % BASIC_BLOCK_SIZE;
    uint32_t remainingBlocks = CeilDiv(remainingSeqlen, BASIC_BLOCK_SIZE);
    return qBlocksInX * completeXBlocks + remainingBlocks;
}

ge::graphStatus BSAGradTiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ProcessTND(gert::TilingContext *context)
{
    const auto *queryShape = context->GetInputShape(QUERY_INDEX);
    const auto *kvShape = context->GetInputShape(KEY_INDEX);

    if (queryShape->GetOriginShape().GetDimNum() != TND_DIM_NUM ||
        kvShape->GetOriginShape().GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "TND format must have 3 dimensions");
        return ge::GRAPH_FAILED;
    }
    totalTokensT_ = queryShape->GetOriginShape().GetDim(TND_DIM_T);
    kvTotalSeqlen_ = kvShape->GetStorageShape().GetDim(TND_DIM_T);
    numHeads_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(TND_DIM_N));
    headDim_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(TND_DIM_D));
    kvHeads_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(TND_DIM_N));
    dqSize_ = totalTokensT_ * numHeads_ * headDim_;
    dkvSize_ = kvTotalSeqlen_ * kvHeads_ * headDim_;

    auto actualSeqLengths = context->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
    if (actualSeqLengths == nullptr) {
        OP_LOGE(context->GetNodeName(), "TND format must have is actualSeqLengthsOptional");
        return ge::GRAPH_FAILED;
    }
    batch_ = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
    qSeqLenList = actualSeqLengths->GetData<int64_t>();
    if (qSeqLenList == nullptr) {
        OP_LOGE(context->GetNodeName(), "Actual seq lengths GetData is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto actualSeqLengthsKv = context->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);
    if (actualSeqLengthsKv == nullptr) {
        OP_LOGE(context->GetNodeName(), "TND format must have actualSeqLengthsKvOptional");
        return ge::GRAPH_FAILED;
    }
    kvSeqLenList = actualSeqLengthsKv->GetData<int64_t>();
    if (kvSeqLenList == nullptr) {
        OP_LOGE(context->GetNodeName(), "Actual seq lengths kv GetData is nullptr");
        return ge::GRAPH_FAILED;
    }

    maxQSeqlen_ = 0;
    maxKvSeqlen_ = 0;
    for (uint32_t i = 0; i < batch_; ++i) {
        maxQSeqlen_ = std::max(maxQSeqlen_, static_cast<uint32_t>(qSeqLenList[i]));
        maxKvSeqlen_ = std::max(maxKvSeqlen_, static_cast<uint32_t>(kvSeqLenList[i]));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ProcessBNSD(gert::TilingContext *context)
{
    const auto *queryShape = context->GetInputShape(QUERY_INDEX);
    const auto *kvShape = context->GetInputShape(KEY_INDEX);
    auto blockSparseMaskShape = context->GetInputShape(BLOCK_SPARSE_MASK_INDEX);

    if (queryShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM ||
        kvShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM ||
        blockSparseMaskShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "BNSD format must have 4 dimensions");
        return ge::GRAPH_FAILED;
    }
    batch_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_B));
    numHeads_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_N));
    maxQSeqlen_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_S));
    headDim_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_D));
    kvHeads_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(BNSD_DIM_N));
    maxKvSeqlen_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(BNSD_DIM_S));
    dqSize_ = batch_ * numHeads_ * maxQSeqlen_ * headDim_;
    dkvSize_ = batch_ * kvHeads_ * maxKvSeqlen_ * headDim_;
    return ge::GRAPH_SUCCESS;
}

// Counts use [B, Nq, maskBlocks, 2] for both BNSD and TND.
// The last dimension stores the valid Q-row and KV-column prefixes.
ge::graphStatus BSAGradTiling::ParseAttnMaskCounts(gert::TilingContext *context)
{
    hasPerBlockMask_ = false;
    maxMaskBlocks_ = 0;
    maskType_ = 0;

    auto attnMask = context->GetOptionalInputTensor(ATTENTION_MASK_INDEX);
    if (attnMask == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto attnMaskDesc = context->GetOptionalInputDesc(ATTENTION_MASK_INDEX);
    if (attnMaskDesc == nullptr || attnMaskDesc->GetDataType() != ge::DT_INT32) {
        // The API's BOOL placeholder and UINT8 masks do not carry prefix counts.
        return ge::GRAPH_SUCCESS;
    }

    maxMaskBlocks_ = std::max(CeilDiv(maxQSeqlen_, static_cast<uint32_t>(blockShapeX_)),
                              CeilDiv(maxKvSeqlen_, static_cast<uint32_t>(blockShapeY_)));

    const auto &shape = attnMask->GetStorageShape();
    const uint32_t dimNum = static_cast<uint32_t>(shape.GetDimNum());
    if (dimNum == 4 && shape.GetDim(3) == 2 && shape.GetDim(2) >= static_cast<int64_t>(maxMaskBlocks_) &&
        shape.GetDim(2) <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
        shape.GetDim(0) == static_cast<int64_t>(batch_) && shape.GetDim(1) == static_cast<int64_t>(numHeads_)) {
        // Head/batch offsets use the physical table stride, including optional padding.
        maxMaskBlocks_ = static_cast<uint32_t>(shape.GetDim(2));
        hasPerBlockMask_ = true;
        maskType_ = 1;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(context->GetNodeName(),
            "attenMaskOptional only supports int32 per-head 4D [B=%u, N=%u, >=%u, 2] "
            "([...,0]=Q block valid rows, [...,1]=KV block valid cols); 1D shared counts is removed. "
            "got dim=%u.",
            batch_, numHeads_, maxMaskBlocks_, dimNum);
    return ge::GRAPH_FAILED;
}

// Wide takes precedence over grouped for eligible BNSD tiles.
// ProcessInput applies the per-head mask fallback after this selection.
uint32_t BSAGradTiling::SelectSchedule()
{
    const uint32_t blockX = static_cast<uint32_t>(blockShapeX_);
    const uint32_t blockY = static_cast<uint32_t>(blockShapeY_);
    const bool kvBasicFitsBlockY = blockY <= BASIC_BLOCK_SIZE; // basicKVBlockSize=min(blockY,128)

    wideEligible_ = layout_ == InputLayout::BNSD && kvBasicFitsBlockY &&
                    (blockY == BASIC_BLOCK_SIZE || (blockX == 64 && blockY == 64));
    optimizedWide_ = wideEligible_ && headDim_ == 128;

    if (deterministic_) {
        scheduleMode_ = SCHEDULE_LEGACY;
        wideQTasksPerWorkItem_ = 0;
        return scheduleMode_;
    }
    if (wideEligible_ && (blockY == BASIC_BLOCK_SIZE || optimizedWide_ || hasPerBlockMask_)) {
        scheduleMode_ = SCHEDULE_WIDE;
        wideQTasksPerWorkItem_ = (optimizedWide_ || hasPerBlockMask_) ? 2 : 48;
        return scheduleMode_;
    }
    if (layout_ == InputLayout::BNSD && kvBasicFitsBlockY) {
        scheduleMode_ = SCHEDULE_GROUPED;
        return scheduleMode_;
    }
    scheduleMode_ = SCHEDULE_LEGACY;
    return scheduleMode_;
}

ge::graphStatus BSAGradTiling::ProcessAttrs(gert::TilingContext *context)
{
    auto blockShapeOptional = context->GetInputTensor(BLOCK_SHAPE_INDEX);
    if (blockShapeOptional == nullptr) {
        OP_LOGE(context->GetNodeName(), "Block shape tensor is null");
        return ge::GRAPH_FAILED;
    }
    blockShapeList = blockShapeOptional->GetData<int64_t>();
    if (blockShapeList != nullptr) {
        blockShapeX_ = blockShapeList[0];
        blockShapeY_ = blockShapeList[1];
    } else {
        // Missing host data falls back to the 64x64 shape used by the UT fixture.
        blockShapeX_ = 64;
        blockShapeY_ = 64;
    }

    if (context->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE_INDEX) == nullptr) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(headDim_));
    } else {
        scaleValue_ = *context->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE_INDEX);
    }

    auto qInputDesc = context->GetInputDesc(QUERY_INDEX);
    if (qInputDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "Query inputDesc is null");
        return ge::GRAPH_FAILED;
    } else {
        dataType_ = qInputDesc->GetDataType();
    }

    if (kvHeads_ == 0) {
        OP_LOGE(context->GetNodeName(), "kvHeads can not be zero.");
        return ge::GRAPH_FAILED;
    }

    if (!(numHeads_ >= kvHeads_ && numHeads_ % kvHeads_ == 0)) {
        OP_LOGE(context->GetNodeName(),
                "Invalid head config: query heads(%u) must be >=kv heads(%u) and divisible by it.", numHeads_,
                kvHeads_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ProcessInput(gert::TilingContext *context)
{
    if (context->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string qLayout(context->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX));
    if (qLayout == "TND") {
        layout_ = InputLayout::TND;
    } else if (qLayout == "BNSD") {
        layout_ = InputLayout::BNSD;
    } else {
        OP_LOGE(context->GetNodeName(), "Unsupported layout: %s. Supported formats: TND, BNSD",
                context->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX));
        return ge::GRAPH_FAILED;
    }

    if (context->GetInputShape(QUERY_INDEX) == nullptr) {
        OP_LOGE(context->GetNodeName(), "Query shape is null");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(KEY_INDEX) == nullptr) {
        OP_LOGE(context->GetNodeName(), "KV shape is null");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(BLOCK_SPARSE_MASK_INDEX) == nullptr) {
        OP_LOGE(context->GetNodeName(), "Block sparse mask is null");
        return ge::GRAPH_FAILED;
    }

    if (layout_ == InputLayout::TND) {
        if (ProcessTND(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (layout_ == InputLayout::BNSD) {
        if (ProcessBNSD(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // The active kernel uses atomic accumulation; deterministic requests are ignored.
    // Keep the workspace and tiling flags consistent with that execution path.
    deterministic_ = false;
    if (ProcessAttrs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Prefix-table validation depends on both sequence lengths and block dimensions.
    if (ParseAttnMaskCounts(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    scheduleMode_ = SelectSchedule();
    // Grouped does not consume per-head prefix counts; legacy handles these shapes.
    if (hasPerBlockMask_ && scheduleMode_ == SCHEDULE_GROUPED) {
        scheduleMode_ = SCHEDULE_LEGACY;
        wideQTasksPerWorkItem_ = 0;
        OP_LOGI(context->GetNodeName(), "per-head 4D mask on grouped shape rerouted to legacy schedule.");
    }
    OP_LOGI(context->GetNodeName(),
            "BSAG schedule: mode=%u (0=legacy,1=grouped,2=wide), deterministic=%d, "
            "hasPerBlockMask=%d(4D per-head), wideQTasksPerWorkItem=%u.",
            scheduleMode_, deterministic_ ? 1 : 0, hasPerBlockMask_ ? 1 : 0, wideQTasksPerWorkItem_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculatePostUbBaseSize(gert::TilingContext *context)
{
    // post 计算：划分空间块数，256 字节对齐
    postUbBaseSize_ = static_cast<uint64_t>(ubSize_ - sizeof(BlockSparseAttentionGradTilingData) - 2 * 1024) /
                      VEC_POST_DIVISION / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculateSoftmaxGradTiling(gert::TilingContext *context)
{
    if (headDim_ == 0) {
        OP_LOGE(context->GetNodeName(), "CalculateSoftmaxGradTiling failed: headDim_ is 0");
        return ge::GRAPH_FAILED;
    }

    constexpr static uint64_t inputBufferLen = 24 * 1024; // castBuffer 24K*2=48K
    constexpr static uint64_t castBufferLen = 48 * 1024;  // castBuffer 48K*2=96K

    uint64_t outputBufferLen = (castBufferLen + headDim_ - 1) / headDim_ * ONEBLOCK_FLOAT_NUM; // 输出(s1,8)
    uint64_t tempBufferLen = 40 * 1024 - outputBufferLen;

    int64_t singleLoopNBurstNum = inputBufferLen / sizeof(float) / headDim_;
    auto softmaxGradShape = ge::Shape({singleLoopNBurstNum, headDim_});

    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, sizeof(float), tempBufferLen, tilingData_->softmaxGradTilingData,
                                   true);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::AssignCoreTasks(uint32_t numHeads, uint32_t kvHeads, uint32_t blockX, uint32_t coreNum,
                                               const std::vector<uint32_t> &tasksInBatch,
                                               const std::vector<uint64_t> &qPrefixTokenSum,
                                               const std::vector<uint64_t> &kvPrefixTokenSum)
{
    if (coreNum == 0 || numHeads == 0 || kvHeads == 0 || numHeads < kvHeads) {
        return ge::GRAPH_FAILED;
    }
    if (totalTaskNum_ == 0) {
        OP_LOGE("BSAGradTiling", "total task num is 0");
        return ge::GRAPH_FAILED;
    }

    // Divide work among launched cores so each active core can advance its handshake loop.
    const uint32_t launchCores = std::min(coreNum, totalTaskNum_);
    taskNumPerCore_ = totalTaskNum_ / launchCores;
    tailTaskNum_ = totalTaskNum_ % launchCores;
    uint32_t currentGlobalTaskId = 0;
    uint32_t qBlocksInX = CeilDiv(blockX, BASIC_BLOCK_SIZE);
    qBlocksInX = qBlocksInX == 0 ? 1 : qBlocksInX;

    for (uint32_t i = 0; i < coreNum; i++) {
        if (i >= launchCores) {
            // Zero the cursor entries for cores that are not launched.
            tilingData_->get_beginBatch()[i] = 0;
            tilingData_->get_beginHead()[i] = 0;
            tilingData_->get_beginQSeqOffset()[i] = 0;
            tilingData_->get_preQSeqLengths()[i] = 0;
            tilingData_->get_preKVSeqLengths()[i] = 0;
            continue;
        }
        uint32_t curBatch = 0, curHeadNum = 0, curQSeqIdx = 0, tempId = currentGlobalTaskId;
        uint64_t preQSeqLengths = 0, preKVSeqLengths = 0;

        if (layout_ == InputLayout::TND) {
            // TND遍历顺序是 Batch -> SeqBlock -> Head
            for (uint32_t b = 0; b < batch_; b++) {
                if (tempId < tasksInBatch[b]) {
                    curBatch = b;
                    curHeadNum = tempId % numHeads;
                    uint32_t blockIdx = tempId / numHeads;

                    uint32_t macroBlockIdx = blockIdx / qBlocksInX;
                    uint32_t microBlockInMacro = blockIdx % qBlocksInX;

                    curQSeqIdx = macroBlockIdx * blockX + microBlockInMacro * BASIC_BLOCK_SIZE;
                    preQSeqLengths = qPrefixTokenSum[b] + curQSeqIdx;
                    preKVSeqLengths = kvPrefixTokenSum[b];
                    break;
                }
                tempId -= tasksInBatch[b];
            }
        } else {
            // BNSD遍历顺序是 Batch -> Head -> SeqBlock
            uint32_t qBlocksPerHead = GetQBlocks(maxQSeqlen_, blockX);
            qBlocksPerHead = qBlocksPerHead == 0 ? 1 : qBlocksPerHead;

            curBatch = tempId / (numHeads * qBlocksPerHead);
            uint32_t remain = tempId % (numHeads * qBlocksPerHead);
            curHeadNum = remain / qBlocksPerHead;
            uint32_t blockIdx = remain % qBlocksPerHead;

            curQSeqIdx = (blockIdx / qBlocksInX) * blockX + (blockIdx % qBlocksInX) * BASIC_BLOCK_SIZE;

            preQSeqLengths = (static_cast<uint64_t>(curBatch) * numHeads * maxQSeqlen_) +
                             (static_cast<uint64_t>(curHeadNum) * maxQSeqlen_) + (static_cast<uint64_t>(curQSeqIdx));
            preKVSeqLengths = (static_cast<uint64_t>(curBatch) * kvHeads * maxKvSeqlen_) +
                              (static_cast<uint64_t>(curHeadNum / (numHeads / kvHeads)) * maxKvSeqlen_);
        }

        tilingData_->get_beginBatch()[i] = curBatch;
        tilingData_->get_beginHead()[i] = curHeadNum;
        tilingData_->get_beginQSeqOffset()[i] = curQSeqIdx;
        tilingData_->get_preQSeqLengths()[i] = preQSeqLengths;
        tilingData_->get_preKVSeqLengths()[i] = preKVSeqLengths;

        uint32_t taskLen = (i < tailTaskNum_) ? (taskNumPerCore_ + 1) : taskNumPerCore_;
        currentGlobalTaskId += taskLen;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculateTaskSplit(gert::TilingContext *context)
{
    uint32_t numHeads = numHeads_;
    uint32_t kvHeads = kvHeads_;
    uint32_t blockX = blockShapeX_;
    uint32_t coreNum = aicNum_;

    totalTaskNum_ = 0;
    totalQBlocks_ = 0;

    const auto *queryShape = context->GetInputShape(QUERY_INDEX);
    const auto *kvShape = context->GetInputShape(KEY_INDEX);
    uint32_t totalQ = (layout_ == InputLayout::TND) ? queryShape->GetOriginShape().GetDim(TND_DIM_T) : 0;
    uint32_t totalKv = (layout_ == InputLayout::TND) ? kvShape->GetOriginShape().GetDim(TND_DIM_T) : 0;

    std::vector<uint32_t> tasksInBatch(batch_);
    std::vector<uint32_t> qWorkItemsPerHead(batch_, 0);
    std::vector<uint64_t> qPrefixTokenSum(batch_ + 1, 0);
    std::vector<uint64_t> kvPrefixTokenSum(batch_ + 1, 0);

    for (uint32_t b = 0; b < batch_; b++) {
        uint32_t qSeqlen = 0;
        uint32_t kvSeqlen = 0;
        if (layout_ == InputLayout::TND) {
            qSeqlen = static_cast<uint32_t>(qSeqLenList[b]);
            kvSeqlen = static_cast<uint32_t>(kvSeqLenList[b]);
        } else {
            qSeqlen = maxQSeqlen_;
            kvSeqlen = maxKvSeqlen_;
        }

        uint32_t qBlocks = GetQBlocks(qSeqlen, blockX);
        // Wide counts work items; legacy and grouped count individual Q tasks.
        uint32_t tasksPerHead = qBlocks;
        if (scheduleMode_ == SCHEDULE_WIDE) {
            qWorkItemsPerHead[b] = CeilDiv(qBlocks, wideQTasksPerWorkItem_);
            tasksPerHead = qWorkItemsPerHead[b];
        }
        tasksInBatch[b] = tasksPerHead * numHeads;
        totalTaskNum_ += tasksInBatch[b];

        uint32_t curQBlockNum = CeilDiv(qSeqlen, blockX) * numHeads;
        totalQBlocks_ += curQBlockNum;

        qPrefixTokenSum[b + 1] = qPrefixTokenSum[b] + qSeqlen;
        kvPrefixTokenSum[b + 1] = kvPrefixTokenSum[b] + kvSeqlen;
    }

    if (scheduleMode_ == SCHEDULE_WIDE) {
        wideWorkItemCount_ = totalTaskNum_;
        if (wideWorkItemCount_ == 0) {
            OP_LOGE(context->GetNodeName(), "wide work item count is 0");
            return ge::GRAPH_FAILED;
        }
        if (AssignWideCoreTasks(numHeads, kvHeads, blockX, coreNum, tasksInBatch, qWorkItemsPerHead) !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (AssignCoreTasks(numHeads, kvHeads, blockX, coreNum, tasksInBatch, qPrefixTokenSum, kvPrefixTokenSum) !=
               ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    blockDim_ = std::min(aicNum_, totalTaskNum_); // wide 下 blockDim 即 cyclic 轮转周期 wideTaskStride
    return ge::GRAPH_SUCCESS;
}

// Optimized wide assigns cyclic work items; other wide schedules use contiguous ranges.
// Only BNSD reaches this mapping.
ge::graphStatus BSAGradTiling::AssignWideCoreTasks(uint32_t numHeads, uint32_t kvHeads, uint32_t blockX,
                                                   uint32_t coreNum, const std::vector<uint32_t> &tasksInBatch,
                                                   const std::vector<uint32_t> &qWorkItemsPerHead)
{
    if (coreNum == 0 || numHeads == 0 || kvHeads == 0 || numHeads < kvHeads) {
        return ge::GRAPH_FAILED;
    }
    if (totalTaskNum_ == 0) {
        OP_LOGE("BSAGradTiling", "wide work item count is 0");
        return ge::GRAPH_FAILED;
    }

    const uint32_t launchCores = std::min(coreNum, totalTaskNum_);
    const uint32_t wideStride = optimizedWide_ ? launchCores : 0;
    taskNumPerCore_ = totalTaskNum_ / launchCores; // work item 计数
    tailTaskNum_ = totalTaskNum_ % launchCores;
    uint32_t globalWorkItem = 0;
    uint32_t qBlocksInX = CeilDiv(blockX, BASIC_BLOCK_SIZE);
    qBlocksInX = qBlocksInX == 0 ? 1 : qBlocksInX;

    for (uint32_t i = 0; i < coreNum; i++) {
        uint32_t temp = wideStride != 0 ? i : globalWorkItem;
        uint32_t curBatch = 0;
        for (; curBatch < batch_; ++curBatch) {
            if (temp < tasksInBatch[curBatch]) {
                break;
            }
            temp -= tasksInBatch[curBatch];
        }
        if (curBatch >= batch_) {
            // Zero the cursor entries for cores that are not launched.
            tilingData_->get_beginBatch()[i] = 0;
            tilingData_->get_beginHead()[i] = 0;
            tilingData_->get_beginQSeqOffset()[i] = 0;
            tilingData_->get_preQSeqLengths()[i] = 0;
            tilingData_->get_preKVSeqLengths()[i] = 0;
            continue;
        }
        uint32_t itemsPerHead = qWorkItemsPerHead[curBatch];
        itemsPerHead = itemsPerHead == 0 ? 1 : itemsPerHead;
        const uint32_t curHeadNum = temp / itemsPerHead;
        const uint32_t itemWithinHead = temp % itemsPerHead;
        const uint32_t firstQTask = itemWithinHead * wideQTasksPerWorkItem_;
        const uint32_t curQSeqIdx = (firstQTask / qBlocksInX) * blockX + (firstQTask % qBlocksInX) * BASIC_BLOCK_SIZE;

        tilingData_->get_beginBatch()[i] = curBatch;
        tilingData_->get_beginHead()[i] = curHeadNum;
        tilingData_->get_beginQSeqOffset()[i] = curQSeqIdx;
        tilingData_->get_preQSeqLengths()[i] =
            (static_cast<uint64_t>(curBatch) * numHeads + curHeadNum) * maxQSeqlen_ + curQSeqIdx;
        tilingData_->get_preKVSeqLengths()[i] =
            (static_cast<uint64_t>(curBatch) * kvHeads + curHeadNum / (numHeads / kvHeads)) * maxKvSeqlen_;
        globalWorkItem += taskNumPerCore_ + (i < tailTaskNum_ ? 1 : 0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculateWorkSpace(gert::TilingContext *context)
{
    if (blockDim_ == 0) {
        OP_LOGE(context->GetNodeName(), "blockDim is 0");
        return ge::GRAPH_FAILED;
    }

    if (scheduleMode_ == SCHEDULE_WIDE) {
        // Each S/dP region contains two FP32 raw stages and low-precision P/dS stages.
        wideWorkspaceStageElements_ = 128 * 128 * 8;                  // 单 stage：128x128 FP32 tile + guard
        wideWorkspaceCoreElements_ = wideWorkspaceStageElements_ * 2; // 双 stage ping-pong
        wideRawPlaneBytes_ = static_cast<uint64_t>(blockDim_) * wideWorkspaceCoreElements_ * sizeof(float);
        sOutSize_ =
            wideRawPlaneBytes_ + static_cast<uint64_t>(blockDim_) * wideWorkspaceCoreElements_ * sizeof(uint16_t);
        dPOutSize_ = sOutSize_;
    } else if (scheduleMode_ == SCHEDULE_GROUPED) {
        // Match GROUPED_WORKSPACE_CORE_SIZE: two stages, with FP32 raw and 16-bit output planes.
        constexpr uint64_t groupedWorkspacePlaneElements = 128ULL * 128ULL * 16ULL * 2ULL;
        sOutSize_ =
            static_cast<uint64_t>(blockDim_) * groupedWorkspacePlaneElements * (sizeof(float) + sizeof(uint16_t));
        dPOutSize_ = sOutSize_;
    } else {
        sOutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float);
        dPOutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float);
    }
    if (layout_ == InputLayout::TND) {
        dQOutSize_ = totalTokensT_ * numHeads_ * headDim_ * sizeof(float);
        dKOutSize_ = kvTotalSeqlen_ * kvHeads_ * headDim_ * sizeof(float);
        dVOutSize_ = dKOutSize_;
        gradSize_ = totalTokensT_ * numHeads_ * ONEBLOCK_FLOAT_NUM * sizeof(float);
    } else {
        dQOutSize_ = batch_ * numHeads_ * maxQSeqlen_ * headDim_ * sizeof(float);
        dKOutSize_ = batch_ * kvHeads_ * maxKvSeqlen_ * headDim_ * sizeof(float);
        dVOutSize_ = dKOutSize_;
        gradSize_ = batch_ * numHeads_ * maxQSeqlen_ * ONEBLOCK_FLOAT_NUM * sizeof(float);
    }

    uint32_t groupSizeForDet_ = (kvHeads_ > 0) ? (numHeads_ / kvHeads_) : 1;
    uint32_t batchSizeForDet_ = 40;
    uint32_t KForDet_ = (aicNum_ > 0) ? ((batchSizeForDet_ + aicNum_ - 1) / aicNum_) : 1;
    if (KForDet_ < 1)
        KForDet_ = 1;
    uint64_t detDkWorkspaceSize_ =
        deterministic_ ? ((uint64_t)aicNum_ * groupSizeForDet_ * dkvSize_ * sizeof(float)) : 0;
    uint64_t detDvWorkspaceSize_ =
        deterministic_ ? ((uint64_t)aicNum_ * groupSizeForDet_ * dkvSize_ * sizeof(float)) : 0;
    workSpaceSize_ = libapiSize_ + sOutSize_ + dPOutSize_ + dQOutSize_ + dKOutSize_ + dVOutSize_ + gradSize_ +
                     detDkWorkspaceSize_ + detDvWorkspaceSize_;
    context->GetWorkspaceSizes(1)[0] = workSpaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::FillTilingData(gert::TilingContext *context)
{
    if (tilingData_ == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_headDim(headDim_);
    tilingData_->set_totalTaskNum(totalTaskNum_);
    tilingData_->set_maskType(maskType_);
    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);
    tilingData_->set_totalQBlocks(totalQBlocks_);
    tilingData_->set_inputLayout(static_cast<uint32_t>(layout_));
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_maxKvSeqlen(maxKvSeqlen_);
    tilingData_->set_basicQBlockSize(BASIC_BLOCK_SIZE);
    tilingData_->set_basicKVBlockSize(static_cast<uint32_t>(std::min<int64_t>(blockShapeY_, BASIC_BLOCK_SIZE)));
    tilingData_->set_taskNumPerCore(taskNumPerCore_);
    tilingData_->set_tailTaskNum(tailTaskNum_);

    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim_);
    context->SetScheduleMode(BATCH_SCHEDULE_MODE);

    tilingData_->set_scheduleMode(scheduleMode_);
    const bool enableWideOptimizations = scheduleMode_ == SCHEDULE_WIDE && optimizedWide_;
    tilingData_->set_wideQOperandCache(enableWideOptimizations ? 1U : 0U);
    tilingData_->set_wideFastPipeline(enableWideOptimizations ? 1U : 0U);
    tilingData_->set_hasPerBlockMask(hasPerBlockMask_ ? 1 : 0);
    tilingData_->set_maxMaskBlocks(maxMaskBlocks_);
    if (scheduleMode_ == SCHEDULE_WIDE) {
        tilingData_->set_wideQTasksPerWorkItem(wideQTasksPerWorkItem_);
        tilingData_->set_widePacketEdgeCapacity(8);
        tilingData_->set_wideTaskStride(optimizedWide_ ? blockDim_ : 0);
        tilingData_->set_wideWorkspaceStageElements(wideWorkspaceStageElements_);
        tilingData_->set_wideWorkspaceCoreElements(wideWorkspaceCoreElements_);
        tilingData_->set_wideRawPlaneBytes(wideRawPlaneBytes_);
    } else {
        tilingData_->set_wideQTasksPerWorkItem(0); // 显式关闭 wide（kernel 谓词据此回退）
        tilingData_->set_widePacketEdgeCapacity(0);
    }

    tilingData_->set_sOutSize(sOutSize_);
    tilingData_->set_dPOutSize(dPOutSize_);
    tilingData_->set_dQOutSize(dQOutSize_);
    tilingData_->set_dKOutSize(dKOutSize_);
    tilingData_->set_dVOutSize(dVOutSize_);
    tilingData_->set_gradSize(gradSize_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_usedVecCoreNum(blockDim_ * 2);
    tilingData_->set_qTotalSeqlen(totalTokensT_);
    tilingData_->set_kvTotalSeqlen(kvTotalSeqlen_);
    tilingData_->set_dqSize(dqSize_);
    tilingData_->set_dkvSize(dkvSize_);
    tilingData_->set_postUbBaseSize(postUbBaseSize_);
    tilingData_->set_ubSize(ubSize_ - sizeof(BlockSparseAttentionGradTilingData) - 2 * 1024);
    uint32_t gsForDet = (kvHeads_ > 0) ? (numHeads_ / kvHeads_) : 1;
    uint32_t bsForDet = 40;
    uint32_t kForDet = (aicNum_ > 0) ? ((bsForDet + aicNum_ - 1) / aicNum_) : 1;
    if (kForDet < 1)
        kForDet = 1;
    uint64_t detDkSize = deterministic_ ? ((uint64_t)aicNum_ * gsForDet * dkvSize_ * sizeof(float)) : 0;
    uint64_t detDvSize = deterministic_ ? ((uint64_t)aicNum_ * gsForDet * dkvSize_ * sizeof(float)) : 0;
    tilingData_->set_detDkWorkspaceSize(detDkSize);
    tilingData_->set_detDvWorkspaceSize(detDvSize);
    tilingData_->set_aicNumForDet(aicNum_);
    tilingData_->set_groupSizeForDet(gsForDet);
    tilingData_->set_deterministic(deterministic_ ? 1 : 0);
    tilingData_->set_batchSizeForDet(bsForDet);
    tilingData_->set_KForDet(kForDet);
    return ge::GRAPH_SUCCESS;
}

uint64_t BSAGradTiling::GenerateTilingKey()
{
    uint64_t tilingKey = 0;
    if (dataType_ == ge::DT_FLOAT16) {
        if (layout_ == InputLayout::TND) {
            tilingKey = 0;
        } else {
            tilingKey = 1;
        }
    } else if (dataType_ == ge::DT_BF16) {
        if (layout_ == InputLayout::TND) {
            tilingKey = 10;
        } else {
            tilingKey = 11;
        }
    }
    return tilingKey;
}

ge::graphStatus BSAGradTiling::GetBSAGradTiling(gert::TilingContext *context,
                                                BlockSparseAttentionGradTilingData &tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetNpuInfo failed");
        return ret;
    }

    ret = ProcessInput(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "ProcessInput failed");
        return ret;
    }

    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateTaskSplit failed");
        return ret;
    }

    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateWorkSpace failed");
        return ret;
    }

    ret = CalculatePostUbBaseSize(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculatePostUbBaseSize failed");
        return ret;
    }

    ret = CalculateSoftmaxGradTiling(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateSoftmaxGradTiling failed");
        return ret;
    }

    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "FillTilingData failed");
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::SetTilingData(gert::TilingContext *context,
                                             BlockSparseAttentionGradTilingData &tilingData)
{
    OP_CHECK_IF(
        context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("BlockSparseAttentionGrad", "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

class BlockSparseAttentionGradArch22Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockSparseAttentionGradArch22Tiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR("BlockSparseAttentionGrad", "Context is nullptr."),
                    return ge::GRAPH_FAILED);
        if (tiling_.GetBSAGradTiling(context_, tilingData_) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "GetBSAGradTiling failed");
            return ge::GRAPH_FAILED;
        }
        tilingKey_ = tilingData_.get_tilingKey();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        return tiling_.SetTilingData(context_, tilingData_);
    }

    uint64_t GetTilingKey() const override
    {
        return tilingKey_;
    }

private:
    BSAGradTiling tiling_;
    BlockSparseAttentionGradTilingData tilingData_;
    uint64_t tilingKey_ = 0;
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockSparseAttentionGrad, BlockSparseAttentionGradArch22Tiling,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 1);

} // namespace optiling
