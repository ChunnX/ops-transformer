/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stem_oam_prep_varlen_q_tiling.h"
#include "../../op_kernel/arch35/stem_oam_prep_varlen_q_tiling_data.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include <algorithm>

namespace optiling {

// ===========================================================================
// Constants (avoid magic numbers)
// ===========================================================================

// Hardware
static constexpr int64_t  DIM_QK_HOST    = 128;
static constexpr uint32_t MAX_BATCH_SIZE = 1024;

// Input tensor indices
static constexpr uint32_t INPUT_Q         = 0;
static constexpr uint32_t INPUT_QSEQLENS  = 1;
static constexpr uint32_t INPUT_CUSEQLENS = 2;
static constexpr uint32_t INPUT_QSCALE    = 3;

// Attribute indices
static constexpr uint32_t ATTR_STEM_BLOCK_SIZE = 0;
static constexpr uint32_t ATTR_STEM_STRIDE     = 1;

// Expected tensor ranks
static constexpr uint32_t Q_RANK         = 3;  // [total_tokens, H_q, D]
static constexpr uint32_t QSEQLENS_RANK  = 1;  // [batch]
static constexpr uint32_t CUSEQLENS_RANK = 1;  // [batch+1]
static constexpr uint32_t QSCALE_RANK    = 2;  // [total_tokens, H_q]
static constexpr uint32_t QFLAT_RANK     = 4;  // [batch, H_q, max_Qb, kflat_dim]

// stemBlockSize constraints
static constexpr int64_t STEM_BLOCK_SIZE_ALIGN = 32;
static constexpr int64_t STEM_BLOCK_SIZE_MAX   = 256;

// stemStride constraints
static constexpr int64_t STEM_STRIDE_ALIGN = 16;
static constexpr int64_t STEM_STRIDE_MAX   = 64;

// UB sizing
static constexpr uint32_t UB_SIZE_BYTES           = 248 * 1024;
static constexpr uint32_t VARLEN_BUF_SIZE_BYTES   = MAX_BATCH_SIZE * sizeof(int32_t) * 2;

// Buffer element sizes (for per-block UB accounting)
static constexpr uint32_t SIZEOF_FP8         = 1;
static constexpr uint32_t SIZEOF_FLOAT32     = 4;
static constexpr uint32_t SIZEOF_BFLOAT16    = 2;

// Tiling key (single template)
static constexpr uint32_t DEFAULT_TILING_KEY = 0;

static void PrintTilingData(const char* nodeName, const StemPrepQTilingData* tiling)
{
    OP_LOGD(nodeName, "========== StemPrepQTilingData ==========");
    OP_LOGD(nodeName, "batch_size: %u", tiling->batch_size);
    OP_LOGD(nodeName, "num_q_heads: %u", tiling->num_q_heads);
    OP_LOGD(nodeName, "dim_qk: %u", tiling->dim_qk);
    OP_LOGD(nodeName, "stem_block_size: %u", tiling->stem_block_size);
    OP_LOGD(nodeName, "stem_stride: %u", tiling->stem_stride);
    OP_LOGD(nodeName, "R_val: %u", tiling->R_val);
    OP_LOGD(nodeName, "kflat_dim: %u", tiling->kflat_dim);
    OP_LOGD(nodeName, "max_Qb: %u", tiling->max_Qb);
    OP_LOGD(nodeName, "used_core_num: %u", tiling->used_core_num);
    OP_LOGD(nodeName, "blocks_per_core_base: %u", tiling->blocks_per_core_base);
    OP_LOGD(nodeName, "blocks_remainder: %u", tiling->blocks_remainder);
}

// ===========================================================================
// 1. ValidateAttrs: Validate compile info + stem attr constraints
// ===========================================================================
static ge::graphStatus ValidateAttrs(gert::TilingContext* context,
                                      int64_t& stemBlockSize,
                                      int64_t& stemStride)
{
    const char* nodeName = context->GetNodeName();

    auto compile_info = static_cast<const StemPrepQCompileInfo*>(context->GetCompileInfo());
    if (compile_info == nullptr || compile_info->core_num == 0) {
        OP_LOGE(nodeName, "Invalid compile info or core_num is 0");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    const int64_t *stemBlockSizePtr = attrs->GetInt(ATTR_STEM_BLOCK_SIZE);
    const int64_t *stemStridePtr = attrs->GetInt(ATTR_STEM_STRIDE);
    if (stemBlockSizePtr == nullptr || stemStridePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "stemBlockSize/stemStride");
        return ge::GRAPH_FAILED;
    }
    stemBlockSize = *stemBlockSizePtr;
    stemStride = *stemStridePtr;

    if (stemBlockSize % STEM_BLOCK_SIZE_ALIGN != 0 ||
        stemBlockSize > STEM_BLOCK_SIZE_MAX || stemBlockSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "stemBlockSize",
            std::to_string(stemBlockSize).c_str(),
            "must be positive, <= %ld, and multiple of %ld");
        return ge::GRAPH_FAILED;
    }
    if (stemStride % STEM_STRIDE_ALIGN != 0 ||
        stemStride > STEM_STRIDE_MAX || stemStride <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "stemStride",
            std::to_string(stemStride).c_str(),
            "must be positive, <= %ld, and multiple of %ld");
        return ge::GRAPH_FAILED;
    }
    if (stemStride > stemBlockSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "stemStride",
            std::to_string(stemStride).c_str(),
            ("must be <= stemBlockSize (" + std::to_string(stemBlockSize) + ")").c_str());
        return ge::GRAPH_FAILED;
    }
    if (stemBlockSize % stemStride != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "stemBlockSize",
            std::to_string(stemBlockSize).c_str(),
            "must be a multiple of stemStride");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// 2. ValidateInputShapes: Validate tensor ranks, basic shapes, and sizes
// ===========================================================================
static ge::graphStatus ValidateInputShapes(gert::TilingContext* context,
                                            int64_t& batch,
                                            int64_t& numQHeads)
{
    const char* nodeName = context->GetNodeName();

    auto qShape = context->GetInputShape(INPUT_Q);
    auto qSeqLensShape = context->GetInputShape(INPUT_QSEQLENS);
    auto cuSeqLensQShape = context->GetInputShape(INPUT_CUSEQLENS);
    if (qShape == nullptr || qSeqLensShape == nullptr || cuSeqLensQShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "q/qSeqLens/cuSeqLensQ");
        return ge::GRAPH_FAILED;
    }

    if (qShape->GetStorageShape().GetDimNum() != Q_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "q",
            std::to_string(qShape->GetStorageShape().GetDimNum()).c_str(),
            std::to_string(Q_RANK).c_str());
        return ge::GRAPH_FAILED;
    }
    if (qSeqLensShape->GetStorageShape().GetDimNum() != QSEQLENS_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "qSeqLens",
            std::to_string(qSeqLensShape->GetStorageShape().GetDimNum()).c_str(),
            std::to_string(QSEQLENS_RANK).c_str());
        return ge::GRAPH_FAILED;
    }
    if (cuSeqLensQShape->GetStorageShape().GetDimNum() != CUSEQLENS_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "cuSeqLensQ",
            std::to_string(cuSeqLensQShape->GetStorageShape().GetDimNum()).c_str(),
            std::to_string(CUSEQLENS_RANK).c_str());
        return ge::GRAPH_FAILED;
    }

    batch = qSeqLensShape->GetStorageShape().GetDim(0);
    int64_t qSeqLensSize = qSeqLensShape->GetStorageShape().GetDimNum() > 0 ?
        qSeqLensShape->GetStorageShape().GetDim(0) : 0;
    int64_t cuSeqLensSize = cuSeqLensQShape->GetStorageShape().GetDimNum() > 0 ?
        cuSeqLensQShape->GetStorageShape().GetDim(0) : 0;

    if (qSeqLensSize != batch) {
        OP_LOGE(nodeName, "qSeqLens size mismatch: expected batch=%ld, got %ld",
                batch, qSeqLensSize);
        return ge::GRAPH_FAILED;
    }
    if (cuSeqLensSize != batch + 1) {
        OP_LOGE(nodeName, "cuSeqLensQ size mismatch: expected batch+1=%ld, got %ld",
                batch + 1, cuSeqLensSize);
        return ge::GRAPH_FAILED;
    }

    numQHeads = qShape->GetStorageShape().GetDim(1);
    int64_t D = qShape->GetStorageShape().GetDim(2);
    if (D != DIM_QK_HOST) {
        OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(nodeName, INPUT_Q,
            std::to_string(D).c_str(), std::to_string(DIM_QK_HOST).c_str());
        return ge::GRAPH_FAILED;
    }

    if (batch < 0 || batch > MAX_BATCH_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "batch",
            std::to_string(batch).c_str(),
            ("batch must be in [0, " + std::to_string(MAX_BATCH_SIZE) + "]").c_str());
        return ge::GRAPH_FAILED;
    }
    if (numQHeads < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "numQHeads",
            std::to_string(numQHeads).c_str(),
            "numQHeads must be >= 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// 3. ValidateDtypes: Validate tensor data types
// ===========================================================================
static ge::graphStatus ValidateDtypes(gert::TilingContext* context,
                                       ge::DataType& qDtype)
{
    const char* nodeName = context->GetNodeName();

    auto qDesc = context->GetInputDesc(INPUT_Q);
    auto qSeqLensDesc = context->GetInputDesc(INPUT_QSEQLENS);
    auto cuSeqLensQDesc = context->GetInputDesc(INPUT_CUSEQLENS);
    if (qDesc == nullptr || qSeqLensDesc == nullptr || cuSeqLensQDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "q/qSeqLens/cuSeqLensQ descriptor");
        return ge::GRAPH_FAILED;
    }

    qDtype = qDesc->GetDataType();
    if (qDtype != ge::DT_FLOAT8_E4M3FN &&
        qDtype != ge::DT_BF16 &&
        qDtype != ge::DT_FLOAT16) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "q",
            std::to_string(static_cast<int>(qDtype)).c_str(),
            "DT_FLOAT8_E4M3FN/DT_BF16/DT_FLOAT16");
        return ge::GRAPH_FAILED;
    }

    if (qSeqLensDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "qSeqLens",
            std::to_string(static_cast<int>(qSeqLensDesc->GetDataType())).c_str(),
            "DT_INT32");
        return ge::GRAPH_FAILED;
    }
    if (cuSeqLensQDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "cuSeqLensQ",
            std::to_string(static_cast<int>(cuSeqLensQDesc->GetDataType())).c_str(),
            "DT_INT32");
        return ge::GRAPH_FAILED;
    }

    // Optional qScale: required for FP8, must be null for BF16/FP16
    auto qScaleDesc = context->GetOptionalInputDesc(INPUT_QSCALE);
    if (qDtype == ge::DT_FLOAT8_E4M3FN) {
        if (qScaleDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "qScale is required when q is DT_FLOAT8_E4M3FN");
            return ge::GRAPH_FAILED;
        }
        if (qScaleDesc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "qScale",
                std::to_string(static_cast<int>(qScaleDesc->GetDataType())).c_str(),
                "DT_FLOAT");
            return ge::GRAPH_FAILED;
        }
        auto qScaleShape = context->GetInputShape(INPUT_QSCALE);
        if (qScaleShape == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "qScale");
            return ge::GRAPH_FAILED;
        }
        if (qScaleShape->GetStorageShape().GetDimNum() != QSCALE_RANK) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "qScale",
                std::to_string(qScaleShape->GetStorageShape().GetDimNum()).c_str(),
                std::to_string(QSCALE_RANK).c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (qScaleDesc != nullptr) {
            OP_LOGE(nodeName, "qScale must be null when q is %s (not DT_FLOAT8_E4M3FN)",
                (qDtype == ge::DT_BF16 ? "DT_BF16" : "DT_FLOAT16"));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// 4. ValidateConsistency: Cross-tensor consistency + empty input detection
// ===========================================================================
static ge::graphStatus ValidateConsistency(gert::TilingContext* context,
                                            int64_t batch,
                                            int64_t numQHeads,
                                            ge::DataType qDtype)
{
    const char* nodeName = context->GetNodeName();

    auto qShape = context->GetInputShape(INPUT_Q);
    int64_t total_tokens = qShape->GetStorageShape().GetDim(0);

    // qScale shape consistency: [total_tokens, H_q]
    if (qDtype == ge::DT_FLOAT8_E4M3FN) {
        auto qScaleShape = context->GetInputShape(INPUT_QSCALE);
        int64_t scaleTokens = qScaleShape->GetStorageShape().GetDim(0);
        int64_t scaleHeads = qScaleShape->GetStorageShape().GetDim(1);
        if (scaleTokens != total_tokens) {
            OP_LOGE(nodeName, "qScale shape dim 0 mismatch: expected total_tokens=%ld, got %ld",
                    total_tokens, scaleTokens);
            return ge::GRAPH_FAILED;
        }
        if (scaleHeads != numQHeads) {
            OP_LOGE(nodeName, "qScale shape dim 1 mismatch: expected H_q=%ld, got %ld",
                    numQHeads, scaleHeads);
            return ge::GRAPH_FAILED;
        }
    }

    if (batch == 0 || numQHeads == 0) {
        OP_LOGI(nodeName, "Empty input: batch=%ld, numQHeads=%ld, skip computation",
                batch, numQHeads);
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcSequenceInfo(gert::TilingContext* context,
                                         StemPrepQTilingData* tiling,
                                         uint32_t B,
                                         uint32_t& totalBlocks)
{
    const char* nodeName = context->GetNodeName();

    auto qShape = context->GetInputShape(INPUT_Q);
    if (qShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "q");
        return ge::GRAPH_FAILED;
    }

    uint32_t H_q = tiling->num_q_heads;
    uint32_t batchU32 = tiling->batch_size;

    auto qSeqLensTensor = context->GetInputTensor(INPUT_QSEQLENS);
    const int32_t *qSeqLensData = (qSeqLensTensor != nullptr) ?
        qSeqLensTensor->GetData<int32_t>() : nullptr;

    if (qSeqLensData != nullptr) {
        auto cuSeqLensTensor = context->GetInputTensor(INPUT_CUSEQLENS);
        const int32_t *cuSeqLensData = (cuSeqLensTensor != nullptr) ?
            cuSeqLensTensor->GetData<int32_t>() : nullptr;

        if (cuSeqLensData != nullptr) {
            if (cuSeqLensData[0] != 0) {
                OP_LOGE_WITH_INVALID_INPUT(nodeName, "cuSeqLensQ[0] must be 0");
                return ge::GRAPH_FAILED;
            }
            for (uint32_t b = 0; b < batchU32; b++) {
                if (cuSeqLensData[b + 1] - cuSeqLensData[b] != qSeqLensData[b]) {
                    OP_LOGE(nodeName, "cuSeqLensQ prefix sum mismatch at batch %u: "
                        "cuSeqLensQ[%u]-cuSeqLensQ[%u]=%d, qSeqLens[%u]=%d",
                        b, b + 1, b,
                        cuSeqLensData[b + 1] - cuSeqLensData[b], b, qSeqLensData[b]);
                    return ge::GRAPH_FAILED;
                }
            }
            // Validate final prefix sum: cuSeqLensQ[batch] == sum(qSeqLens)
            int32_t sum_qSeqLens = 0;
            for (uint32_t b = 0; b < batchU32; b++) {
                sum_qSeqLens += qSeqLensData[b];
            }
            if (cuSeqLensData[batchU32] != sum_qSeqLens) {
                OP_LOGE(nodeName, "cuSeqLensQ[%u]=%d does not equal sum(qSeqLens)=%d",
                    batchU32, cuSeqLensData[batchU32], sum_qSeqLens);
                return ge::GRAPH_FAILED;
            }
        }

        uint32_t max_Qb = 0;
        for (uint32_t b = 0; b < batchU32; b++) {
            uint32_t q_len_b = static_cast<uint32_t>(qSeqLensData[b]);
            uint32_t q_padded_b = ((q_len_b + B - 1) / B) * B;
            uint32_t num_Qb_b = q_padded_b / B;
            max_Qb = std::max(max_Qb, num_Qb_b);
        }
        tiling->max_Qb = max_Qb;
        totalBlocks = H_q * max_Qb * batchU32;
    } else {
        uint32_t totalTokens = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));
        uint32_t max_q_len = totalTokens / batchU32;
        uint32_t max_q_padded = ((max_q_len + B - 1) / B) * B;
        tiling->max_Qb = max_q_padded / B;
        totalBlocks = H_q * tiling->max_Qb * batchU32;

        OP_LOGD(nodeName, "qSeqLens data not available, using conservative estimate: max_q_len=%u",
                max_q_len);
    }

    OP_LOGI(nodeName, "CalcSequenceInfo: max_Qb=%u, totalBlocks=%u",
            tiling->max_Qb, totalBlocks);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateOutputShape(gert::TilingContext* context,
                                            const StemPrepQTilingData* tiling,
                                            uint32_t S)
{
    const char* nodeName = context->GetNodeName();

    auto qFlatShape = context->GetOutputShape(0);
    auto qFlatDesc = context->GetOutputDesc(0);
    if (qFlatShape == nullptr || qFlatDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "qFlat output");
        return ge::GRAPH_FAILED;
    }

    auto& shape = qFlatShape->GetStorageShape();
    if (shape.GetDimNum() != QFLAT_RANK) {
        OP_LOGE(nodeName, "qFlat output must be 4D [batch, H_q, max_Qb, kflat_dim], got %ld dims",
                static_cast<long>(shape.GetDimNum()));
        return ge::GRAPH_FAILED;
    }

    uint32_t expected_batch = tiling->batch_size;
    uint32_t expected_heads = tiling->num_q_heads;
    uint32_t expected_maxQb = tiling->max_Qb;
    uint32_t expected_kflat = S * DIM_QK_HOST;

    int64_t actual_dim0 = shape.GetDim(0);
    int64_t actual_dim1 = shape.GetDim(1);
    int64_t actual_dim2 = shape.GetDim(2);
    int64_t actual_dim3 = shape.GetDim(3);

    if (actual_dim0 != static_cast<int64_t>(expected_batch)) {
        OP_LOGE(nodeName, "qFlat shape mismatch dim 0: expected batch=%u, got %ld",
                expected_batch, actual_dim0);
        return ge::GRAPH_FAILED;
    }
    if (actual_dim1 != static_cast<int64_t>(expected_heads)) {
        OP_LOGE(nodeName, "qFlat shape mismatch dim 1: expected H_q=%u, got %ld",
                expected_heads, actual_dim1);
        return ge::GRAPH_FAILED;
    }
    if (actual_dim2 < static_cast<int64_t>(expected_maxQb)) {
        OP_LOGE(nodeName, "qFlat shape dim 2 too small: need max_Qb=%u, got %ld",
                expected_maxQb, actual_dim2);
        return ge::GRAPH_FAILED;
    }
    if (actual_dim3 < static_cast<int64_t>(expected_kflat)) {
        OP_LOGE(nodeName, "qFlat shape dim 3 too small: need kflat_dim=%u (S*%d=%u*%d), got %ld",
                expected_kflat, static_cast<int>(DIM_QK_HOST), S, static_cast<int>(DIM_QK_HOST), actual_dim3);
        return ge::GRAPH_FAILED;
    }

    // Validate output dtype
    auto qDtype = context->GetInputDesc(INPUT_Q)->GetDataType();
    auto outDtype = qFlatDesc->GetDataType();
    ge::DataType expectedOutDtype = (qDtype == ge::DT_FLOAT16) ? ge::DT_FLOAT16 : ge::DT_BF16;
    if (outDtype != expectedOutDtype) {
        OP_LOGE(nodeName, "qFlat output dtype mismatch: expected %s for input %s, got %s",
                (expectedOutDtype == ge::DT_BF16 ? "DT_BF16" : "DT_FLOAT16"),
                (qDtype == ge::DT_FLOAT8_E4M3FN ? "DT_FLOAT8_E4M3FN" :
                 (qDtype == ge::DT_BF16 ? "DT_BF16" : "DT_FLOAT16")),
                (outDtype == ge::DT_BF16 ? "DT_BF16" :
                 (outDtype == ge::DT_FLOAT16 ? "DT_FLOAT16" : "other")));
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(nodeName, "qFlat output shape validated: [%u, %u, %u(max_Qb=%u), %ld(kflat=%u)]",
            expected_batch, expected_heads,
            static_cast<uint32_t>(actual_dim2), expected_maxQb,
            actual_dim3, expected_kflat);

    return ge::GRAPH_SUCCESS;
}

static void CalcCoreDistribution(gert::TilingContext* context,
                                  StemPrepQTilingData* tiling,
                                  uint32_t totalBlocks,
                                  uint32_t N)
{
    const char* nodeName = context->GetNodeName();
    
    if (totalBlocks == 0) {
        tiling->used_core_num = 0;
        tiling->blocks_per_core_base = 0;
        tiling->blocks_remainder = 0;
        context->SetTilingKey(DEFAULT_TILING_KEY);
        context->SetBlockDim(1);
        OP_LOGI(nodeName, "No work to do, totalBlocks=0");
        return;
    }

    tiling->used_core_num = std::min(totalBlocks, N);
    tiling->blocks_per_core_base = totalBlocks / tiling->used_core_num;
    tiling->blocks_remainder = totalBlocks % tiling->used_core_num;

    context->SetTilingKey(DEFAULT_TILING_KEY);
    context->SetBlockDim(tiling->used_core_num);
    context->SetScheduleMode(1);  // Batch mode
    
    OP_LOGI(nodeName, "Tiling: totalBlocks=%u, used_core_num=%u, blocks_per_core_base=%u, blocks_remainder=%u",
            totalBlocks, tiling->used_core_num, tiling->blocks_per_core_base, tiling->blocks_remainder);
}

ge::graphStatus Tiling4StemOamPrepVarlenQ(gert::TilingContext* context) {
    const char* nodeName = context->GetNodeName();
    int64_t stemBlockSize = 0;
    int64_t stemStride = 0;
    int64_t batch = 0;
    int64_t numQHeads = 0;
    ge::DataType qDtype = ge::DT_UNDEFINED;

    if (ValidateAttrs(context, stemBlockSize, stemStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateInputShapes(context, batch, numQHeads) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateDtypes(context, qDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Empty input early return (before tiling data is populated beyond basic fields)
    if (batch == 0 || numQHeads == 0) {
        // Still need to set tiling key and blockDim to avoid runtime issues
        context->SetTilingKey(DEFAULT_TILING_KEY);
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }
    if (ValidateConsistency(context, batch, numQHeads, qDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto compile_info = static_cast<const StemPrepQCompileInfo*>(context->GetCompileInfo());
    uint32_t N = static_cast<uint32_t>(compile_info->core_num);

    StemPrepQTilingData* tiling = context->GetTilingData<StemPrepQTilingData>();
    if (tiling == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tiling");
        return ge::GRAPH_FAILED;
    }

    uint32_t B = static_cast<uint32_t>(stemBlockSize);
    uint32_t S = static_cast<uint32_t>(stemStride);

    tiling->stem_block_size = B;
    tiling->stem_stride = S;
    tiling->R_val = B / S;
    tiling->kflat_dim = S * DIM_QK_HOST;
    tiling->dim_qk = DIM_QK_HOST;
    tiling->batch_size = static_cast<uint32_t>(batch);
    tiling->num_q_heads = static_cast<uint32_t>(numQHeads);

    uint32_t totalBlocks = 0;
    if (CalcSequenceInfo(context, tiling, B, totalBlocks) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ValidateOutputShape(context, tiling, S) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    CalcCoreDistribution(context, tiling, totalBlocks, N);

    // Calculate ub_factor: how many blocks can fit in UB at once
    uint32_t perBlockUB = B * DIM_QK_HOST * SIZEOF_FLOAT32     // qBlockFP32Buf (float)
                        + B * DIM_QK_HOST * SIZEOF_FP8         // qBlockBuf (fp8)
                        + B * SIZEOF_FLOAT32                   // scalesBuf (float)
                        + S * DIM_QK_HOST * SIZEOF_FLOAT32     // groupSumBuf (float)
                        + S * DIM_QK_HOST * SIZEOF_BFLOAT16;   // outQueueQflat (bf16)
    uint32_t availableUB = UB_SIZE_BYTES - VARLEN_BUF_SIZE_BYTES;
    uint32_t ubFactor = availableUB / perBlockUB;
    tiling->ub_factor = ubFactor;

    OP_LOGI(nodeName, "ub_factor=%u (perBlock=%u, available=%u)",
            ubFactor, perBlockUB, availableUB);

    // Calculate workspace size
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        workspaceSizes[0] = workspaceSize;
    }
    OP_LOGI(nodeName, "Workspace size: %zu bytes", workspaceSize);

    PrintTilingData(nodeName, tiling);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4StemOamPrepVarlenQ(gert::TilingParseContext* context) {
    auto compileInfo = context->GetCompiledInfo<StemPrepQCompileInfo>();
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StemOamPrepVarlenQ)
    .Tiling(Tiling4StemOamPrepVarlenQ)
    .TilingParse<StemPrepQCompileInfo>(TilingPrepare4StemOamPrepVarlenQ);

}  // namespace optiling
