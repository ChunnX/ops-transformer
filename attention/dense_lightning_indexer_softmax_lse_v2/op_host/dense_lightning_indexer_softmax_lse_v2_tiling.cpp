/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_v2_tiling.cpp
 * \brief
 */

#include "dense_lightning_indexer_softmax_lse_v2_tiling.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

static ge::graphStatus ParamCheck(gert::TilingContext* context, int64_t layout, int64_t maskMode,
                                   int64_t cmpRatio, const std::string& layoutQStr, const std::string& layoutKStr,
                                   int64_t bSize, int64_t s1Size, int64_t s2Size, int64_t n1Size, int64_t dSize)
{
    // Constraint: layout_q must equal layout_k
    OP_CHECK_IF(layoutQStr != layoutKStr,
                OP_LOGE(context->GetNodeName(), "layoutQ and layoutK must be the same, but got %s and %s.",
                        layoutQStr.c_str(), layoutKStr.c_str()),
                return ge::GRAPH_PARAM_INVALID);

    // Constraint: mask_mode only supports 0 or 3
    OP_CHECK_IF(maskMode != 0 && maskMode != 3,
                OP_LOGE(context->GetNodeName(), "mask_mode only supports 0 or 3, but got %ld.", maskMode),
                return ge::GRAPH_PARAM_INVALID);

    // Constraint: cmp_ratio in [1, 128]
    OP_CHECK_IF(cmpRatio < 1 || cmpRatio > 128,
                OP_LOGE(context->GetNodeName(), "cmp_ratio must be in [1, 128], but got %ld.", cmpRatio),
                return ge::GRAPH_PARAM_INVALID);

    // Constraint: D must be 128
    OP_CHECK_IF(dSize != 128,
                OP_LOGE(context->GetNodeName(), "head_dim must be 128, but got %ld.", dSize),
                return ge::GRAPH_PARAM_INVALID);

    // Constraint: N1 in [1, 128]
    OP_CHECK_IF(n1Size < 1 || n1Size > 128,
                OP_LOGE(context->GetNodeName(), "num_heads_q must be in [1, 128], but got %ld.", n1Size),
                return ge::GRAPH_PARAM_INVALID);

    // Constraint: B > 0, S1 > 0, S2 > 0
    OP_CHECK_IF(bSize <= 0 || s1Size <= 0 || s2Size <= 0,
                OP_LOGE(context->GetNodeName(), "B, S1, S2 must be positive, but got B=%ld, S1=%ld, S2=%ld.",
                        bSize, s1Size, s2Size),
                return ge::GRAPH_PARAM_INVALID);

    // Constraint: cmp_residual_k values must be less than cmp_ratio
    auto cmpResidualKTensor = context->GetOptionalInputTensor(7);
    if (cmpResidualKTensor != nullptr) {
        const int32_t* cmpResidualKData = cmpResidualKTensor->GetData<int32_t>();
        if (cmpResidualKData != nullptr) {
            auto cmpResidualKShape = context->GetOptionalInputShape(7);
            int64_t cmpResidualKNum = cmpResidualKShape->GetStorageShape().GetShapeSize();
            for (int64_t i = 0; i < cmpResidualKNum; ++i) {
                OP_CHECK_IF(static_cast<int64_t>(cmpResidualKData[i]) >= cmpRatio,
                            OP_LOGE(context->GetNodeName(), "cmp_residual_k[%ld]=%d must be less than cmp_ratio=%ld.",
                                    i, cmpResidualKData[i], cmpRatio),
                            return ge::GRAPH_PARAM_INVALID);
            }
        }
    }

    // Constraint: seqused_q <= S1 (BSND) or <= seqlens_q[b] (TND)
    auto seqUsedQTensor = context->GetOptionalInputTensor(5);
    if (seqUsedQTensor != nullptr) {
        const int32_t* seqUsedQData = seqUsedQTensor->GetData<int32_t>();
        if (seqUsedQData != nullptr) {
            if (layout == 1) {
                auto cuSeqLensQTensor = context->GetOptionalInputTensor(3);
                if (cuSeqLensQTensor != nullptr) {
                    const int32_t* cuSeqLensQData = cuSeqLensQTensor->GetData<int32_t>();
                    if (cuSeqLensQData != nullptr) {
                        for (int64_t i = 0; i < bSize; ++i) {
                            int64_t seqlenQ = static_cast<int64_t>(cuSeqLensQData[i + 1]) - static_cast<int64_t>(cuSeqLensQData[i]);
                            OP_CHECK_IF(static_cast<int64_t>(seqUsedQData[i]) > seqlenQ,
                                        OP_LOGE(context->GetNodeName(), "seqused_q[%ld]=%d exceeds seqlens_q[%ld]=%ld.",
                                                i, seqUsedQData[i], i, seqlenQ),
                                        return ge::GRAPH_PARAM_INVALID);
                        }
                    }
                }
            } else {
                for (int64_t i = 0; i < bSize; ++i) {
                    OP_CHECK_IF(static_cast<int64_t>(seqUsedQData[i]) > s1Size,
                                OP_LOGE(context->GetNodeName(), "seqused_q[%ld]=%d exceeds max_seqlen_q=%ld.",
                                        i, seqUsedQData[i], s1Size),
                                return ge::GRAPH_PARAM_INVALID);
                }
            }
        }
    }

    // Constraint: seqused_k <= S2 (BSND) or <= seqlens_k[b] (TND)
    auto seqUsedKTensor = context->GetOptionalInputTensor(6);
    if (seqUsedKTensor != nullptr) {
        const int32_t* seqUsedKData = seqUsedKTensor->GetData<int32_t>();
        if (seqUsedKData != nullptr) {
            if (layout == 1) {
                auto cuSeqLensKTensor = context->GetOptionalInputTensor(4);
                if (cuSeqLensKTensor != nullptr) {
                    const int32_t* cuSeqLensKData = cuSeqLensKTensor->GetData<int32_t>();
                    if (cuSeqLensKData != nullptr) {
                        for (int64_t i = 0; i < bSize; ++i) {
                            int64_t seqlenK = static_cast<int64_t>(cuSeqLensKData[i + 1]) - static_cast<int64_t>(cuSeqLensKData[i]);
                            OP_CHECK_IF(static_cast<int64_t>(seqUsedKData[i]) > seqlenK,
                                        OP_LOGE(context->GetNodeName(), "seqused_k[%ld]=%d exceeds seqlens_k[%ld]=%ld.",
                                                i, seqUsedKData[i], i, seqlenK),
                                        return ge::GRAPH_PARAM_INVALID);
                        }
                    }
                }
            } else {
                for (int64_t i = 0; i < bSize; ++i) {
                    OP_CHECK_IF(static_cast<int64_t>(seqUsedKData[i]) > s2Size,
                                OP_LOGE(context->GetNodeName(), "seqused_k[%ld]=%d exceeds max_seqlen_k=%ld.",
                                        i, seqUsedKData[i], s2Size),
                                return ge::GRAPH_PARAM_INVALID);
                }
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DenseLightningIndexerSoftmaxLseV2TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // Extract layout attr
    auto attrs = context->GetAttrs();
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t layout = 0;  // 0=BSND, 1=TND
    std::string layoutQStr = "BSND";
    std::string layoutKStr = "BSND";
    if (attrs != nullptr) {
        auto maskModePtr = attrs->GetAttrPointer<int64_t>(2);
        auto cmpRatioPtr = attrs->GetAttrPointer<int64_t>(3);
        if (maskModePtr != nullptr) {
            maskMode = *maskModePtr;
        }
        if (cmpRatioPtr != nullptr) {
            cmpRatio = *cmpRatioPtr;
        }
        // Attr indices: 0=layoutQ(String), 1=layoutK(String), 2=maskMode(Int), 3=cmpRatio(Int)
        auto layoutQPtr = attrs->GetAttrPointer<char>(0);
        if (layoutQPtr != nullptr) {
            layoutQStr = std::string(layoutQPtr);
            if (layoutQStr == "TND") {
                layout = 1;
            }
        }
        auto layoutKPtr = attrs->GetAttrPointer<char>(1);
        if (layoutKPtr != nullptr) {
            layoutKStr = std::string(layoutKPtr);
        }
    }

    // Extract runtime dimensions from input shapes.
    auto queryShape = context->GetInputShape(0);
    auto keyShape = context->GetInputShape(1);
    OP_CHECK_IF(queryShape == nullptr || keyShape == nullptr,
                OP_LOGE(context->GetNodeName(), "query_index or key_index shape is null."),
                return ge::GRAPH_PARAM_INVALID);
    int64_t bSize, s1Size, n1Size, dSize, s2Size;

    if (layout == 1) {
        // TND: query [T1, N1, D], key [T2, 1, D]
        s1Size = queryShape->GetStorageShape().GetDim(0);
        n1Size = queryShape->GetStorageShape().GetDim(1);
        dSize = queryShape->GetStorageShape().GetDim(2);
        s2Size = keyShape->GetStorageShape().GetDim(0);
        // B from cu_seq_lens_q input (index 3, shape [B+1])
        auto cuSeqLensQShape = context->GetOptionalInputShape(3);
        if (cuSeqLensQShape != nullptr) {
            bSize = cuSeqLensQShape->GetStorageShape().GetDim(0) - 1;
        } else {
            bSize = 1;
        }
    } else {
        // BSND: query [B, S1, N1, D], key [B, S2, 1, D]
        bSize = queryShape->GetStorageShape().GetDim(0);
        s1Size = queryShape->GetStorageShape().GetDim(1);
        n1Size = queryShape->GetStorageShape().GetDim(2);
        dSize = queryShape->GetStorageShape().GetDim(3);
        s2Size = keyShape->GetStorageShape().GetDim(1);
    }

    // Unified parameter validation
    auto ret = ParamCheck(context, layout, maskMode, cmpRatio, layoutQStr, layoutKStr,
                          bSize, s1Size, s2Size, n1Size, dSize);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ParamCheck failed."),
                return ge::GRAPH_PARAM_INVALID);

    // OpTiling comes from the pypto kernel (.py dataclass) via force-include.
    OpTiling *tilingData = context->GetTilingData<OpTiling>();
    tilingData->b = bSize;
    tilingData->s1 = s1Size;
    tilingData->s2 = s2Size;
    tilingData->n1 = n1Size;
    tilingData->d = dSize;

    // Determine if S2 exceeds reduce_sum_vec UB capacity.
    constexpr int64_t TKV = 128;
    constexpr int64_t TKV_HALF = TKV / 2;
    constexpr int64_t KV_CACHE_NUM = 46832;
    int64_t s2Tiles = (s2Size + TKV - 1) / TKV;
    int64_t maxUbS2Tiles = KV_CACHE_NUM / TKV_HALF;
    int64_t isLongS2 = (s2Tiles > maxUbS2Tiles) ? 1 : 0;
    tilingData->isLongS2 = isLongS2;

    tilingData->mask_mode = maskMode;
    tilingData->cmp_ratio = cmpRatio;
    tilingData->layout = layout;

    // Check if optional inputs seq_used_q (index 5) and seq_used_k (index 6) are provided
    int64_t hasSeqUsedQ = (context->GetOptionalInputTensor(5) != nullptr) ? 1 : 0;
    int64_t hasSeqUsedK = (context->GetOptionalInputTensor(6) != nullptr) ? 1 : 0;
    tilingData->has_seqused_q = hasSeqUsedQ;
    tilingData->has_seqused_k = hasSeqUsedK;

    // TilingKey: bit0=isLongS2, bit1=hasSeqUsedQ, bit2=hasSeqUsedK
    // 顺序与 pypto-pro TilingKeyField 声明顺序一致
    uint64_t tilingKey = static_cast<uint64_t>(isLongS2)
                       | (static_cast<uint64_t>(hasSeqUsedQ) << 1)
                       | (static_cast<uint64_t>(hasSeqUsedK) << 2);
    context->SetTilingKey(tilingKey);

    // Workspace: system reserve + workspace_max [1,72] FP32 (288 bytes) + workspace_sum [1,72] FP32 (288 bytes)
    //            + workspace [36, S2] FP32 (for long S2 reduce_sum spill).
    constexpr size_t WS_MAX_BYTES = 288;
    constexpr int64_t WS_MAX_CORES = 36;
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += WS_MAX_BYTES * 2;
    workspaceSize += static_cast<size_t>(WS_MAX_CORES) * static_cast<size_t>(s2Size) * sizeof(float);
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(), "DenseLightningIndexerSoftmaxLseV2 tiling completed: blockDim=%u, aivNum=%u, aicNum=%u, layout=%ld, B=%ld, S1=%ld, S2=%ld, N1=%ld, D=%ld, isLongS2=%ld, maskMode=%ld, cmpRatio=%ld, hasSeqUsedQ=%ld, hasSeqUsedK=%ld, workspaceSize=%zu.",
            blockDim, aivNum, aicNum, layout, bSize, s1Size, s2Size, n1Size, dSize, isLongS2, maskMode, cmpRatio, hasSeqUsedQ, hasSeqUsedK, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDenseLightningIndexerSoftmaxLseV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DenseLightningIndexerSoftmaxLseV2)
    .Tiling(DenseLightningIndexerSoftmaxLseV2TilingFunc)
    .TilingParse<DenseLISoftmaxLseV2CompileInfo>(TilingParseForDenseLightningIndexerSoftmaxLseV2);

} // namespace optiling
