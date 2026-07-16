/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_mla_softmax_l1_norm_tiling2.cpp
 * \brief 补充完整的 tiling 实现，支持 BSND 和 TND 布局
 */
#include <string>
#include "sparse_flash_mla_softmax_l1_norm_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;

namespace optiling {

static ge::graphStatus SparseFlashMlaSoftmaxL1NormTilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // 获取属性
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 属性索引根据 def.cpp 定义：
    // 0: scale_value, 1: max_seqlen_k, 2: cmp_ratio, 3: mask_mode, 4: layout_q_optional, 5: layout_k_optional
    const int64_t* maxSeqlenKPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* cmpRatioPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* maskModePtr = attrs->GetAttrPointer<int64_t>(3);
    const char* layoutQPtr = attrs->GetAttrPointer<char>(4);

    int64_t maxSeqlenK = (maxSeqlenKPtr != nullptr) ? *maxSeqlenKPtr : 0;
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : 1;
    int64_t maskMode = (maskModePtr != nullptr) ? *maskModePtr : 0;
    std::string layoutQ = (layoutQPtr != nullptr) ? std::string(layoutQPtr) : "BSND";

    bool isTnd = (layoutQ == "TND");

    auto queryShape = context->GetInputShape(0);
    auto keyShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    int64_t bSize = 0, sqSize = 0, skSize = 0, gSize = 0, dSize = 0;
    int64_t t1 = 0, t2 = 0;

    if (isTnd) {
        // TND 布局: query shape 为 [T1, G, D], key shape 为 [T2, 1, D]
        t1 = queryShape->GetStorageShape().GetDim(0);
        gSize = queryShape->GetStorageShape().GetDim(1);
        dSize = queryShape->GetStorageShape().GetDim(2);

        t2 = keyShape->GetStorageShape().GetDim(0);

        // TND 下 B 的大小需要从 cu_seqlens 或 seqused 获取
        auto cuSeqQInput = context->GetOptionalInput(4); // cu_seqlens_q_optional
        if (cuSeqQInput != nullptr) {
            bSize = context->GetInputShape(4)->GetStorageShape().GetDim(0) - 1;
        } else if (context->GetOptionalInput(6) != nullptr) { // seqused_q_optional
            bSize = context->GetInputShape(6)->GetStorageShape().GetDim(0);
        } else {
            bSize = 1; // 默认 1
        }
    } else {
        // BSND 布局: query shape 为 [B, Sq, G, D], key shape 为 [B, Sk, 1, D]
        bSize = queryShape->GetStorageShape().GetDim(0);
        sqSize = queryShape->GetStorageShape().GetDim(1);
        gSize = queryShape->GetStorageShape().GetDim(2);
        dSize = queryShape->GetStorageShape().GetDim(3);
        skSize = keyShape->GetStorageShape().GetDim(1);
    }

    // 获取可选输入是否存在
    // 输入索引：3:sparse_indices, 4:cu_seqlens_q, 5:cu_seqlens_k, 6:seqused_q, 7:seqused_k, 8:cmp_residual_k, 9:topk_length
    bool has_seqused_q = (context->GetOptionalInput(6) != nullptr);
    bool has_seqused_k = (context->GetOptionalInput(7) != nullptr);
    bool has_topk_length = (context->GetOptionalInput(9) != nullptr);

    // 获取 sparse 相关信息
    auto sparse_indices_input = context->GetOptionalInput(3);
    bool isSparse = (sparse_indices_input != nullptr);
    int64_t kLength = 0;
    if (isSparse) {
        auto sparseShape = context->GetInputShape(3);
        kLength = sparseShape->GetStorageShape().GetDim(sparseShape->GetDimNum() - 1);
    } else {
        kLength = maxSeqlenK;
    }

    // 填充 TilingData
    OpTiling *tilingData = context->GetTilingData<OpTiling>();
    tilingData->b = bSize;
    tilingData->sq = sqSize;
    tilingData->sk = skSize;
    tilingData->g = gSize;
    tilingData->d = dSize;
    tilingData->t1 = t1;
    tilingData->t2 = t2;
    tilingData->max_seqlen_k = maxSeqlenK;
    tilingData->k_length = kLength;
    tilingData->cmp_ratio = cmpRatio;
    tilingData->has_seqused_q = has_seqused_q;
    tilingData->has_seqused_k = has_seqused_k;
    tilingData->has_topk_length = has_topk_length;

    // 计算 TilingKey
    // SmlaTilingKey: SPARSE_MODE(bits 0-1), IS_TND(bit 2), IS_SPARSE(bit 3)
    uint64_t tilingKey = 0;
    tilingKey |= (static_cast<uint64_t>(maskMode) & 0x3);
    tilingKey |= (static_cast<uint64_t>(isTnd ? 1 : 0) << 2);
    tilingKey |= (static_cast<uint64_t>(isSparse ? 1 : 0) << 3);
    context->SetTilingKey(tilingKey);

    // Workspace: system reserve + qk buffer [G, Sk] FP32.
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (!isTnd) {
        workspaceSize += static_cast<size_t>(gSize) * static_cast<size_t>(skSize) * sizeof(float);
    } else {
        workspaceSize += static_cast<size_t>(gSize) * static_cast<size_t>(t2) * sizeof(float);
    }
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(), "SparseFlashMlaSoftmaxL1Norm tiling2 completed: blockDim=%u, B=%ld, Sq=%ld, Sk=%ld, G=%ld, D=%ld, t1=%ld, t2=%ld, max_seqlen_k=%ld, k_length=%ld, cmp_ratio=%ld, isTnd=%d, isSparse=%d.",
            blockDim, bSize, sqSize, skSize, gSize, dSize, t1, t2, maxSeqlenK, kLength, cmpRatio, isTnd, isSparse);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSparseFlashMlaSoftmaxL1Norm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseFlashMlaSoftmaxL1Norm)
    .Tiling(SparseFlashMlaSoftmaxL1NormTilingFunc)
    .TilingParse<SparseFlashMlaSoftmaxL1NormCompileInfo>(TilingParseForSparseFlashMlaSoftmaxL1Norm);
} // namespace optiling