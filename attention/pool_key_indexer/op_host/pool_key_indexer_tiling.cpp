/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_key_indexer_tiling.cpp
 * \brief PoolKeyIndexer tiling implementation: param parsing, validation,
 *        s1BaseSize/mBaseSizeMax/s2BaseSize/trunkLen computation,
 *        workspace offset calculation, SplitCore base-block logic.
 *        Design ref: docs/design.md §5.2/§5.3/§5.4/§5.6/§6.3
 */

#include "pool_key_indexer_tiling.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

using namespace ge;
namespace optiling {

static uint32_t AlignUp(uint32_t val, uint32_t align)
{
    return (align == 0) ? val : (val + align - 1) / align * align;
}

static uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

static uint64_t RoundUp64(uint64_t val, uint64_t align)
{
    return (align == 0) ? val : (val + align - 1) / align * align;
}

ge::graphStatus PoolKeyIndexerTiling::ParseAndCheckParams(PoolKeyIndexerTilingInfo &info)
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *topkPtr = attrs->GetInt(PKI_ATTR_TOPK);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topkPtr);
    int64_t topkVal = *topkPtr;
    bool topkValid = (topkVal >= 1 && topkVal <= 2048) ||
                     topkVal == 3072 || topkVal == 4096 || topkVal == 5120 ||
                     topkVal == 6144 || topkVal == 7168 || topkVal == 8192;
    OP_CHECK_IF(!topkValid,
                OP_LOGE(context_, "topk(%ld) must be in [1, 2048] or one of {3072, 4096, 5120, 6144, 7168, 8192}",
                        topkVal),
                return ge::GRAPH_FAILED);
    info.topk = static_cast<uint32_t>(topkVal);

    const int64_t *poolSizePtr = attrs->GetInt(PKI_ATTR_POOL_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolSizePtr);
    int64_t poolSizeVal = *poolSizePtr;
    OP_CHECK_IF(poolSizeVal < 1 || poolSizeVal > PKI_POOL_SIZE_LIMIT,
                OP_LOGE(context_, "pool_size(%ld) must be in [1, %u]", poolSizeVal, PKI_POOL_SIZE_LIMIT),
                return ge::GRAPH_FAILED);
    info.poolSize = static_cast<uint32_t>(poolSizeVal);

    const char *layoutQPtr = attrs->GetAttrPointer<char>(PKI_ATTR_LAYOUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutQPtr);
    std::string layoutQStr(layoutQPtr);
    if (layoutQStr == "BSND") {
        info.layoutQ = PkiDataLayout::BSND;
    } else if (layoutQStr == "TND") {
        info.layoutQ = PkiDataLayout::TND;
    } else {
        OP_LOGE(context_, "layout_q must be BSND or TND, got %s", layoutQStr.c_str());
        return ge::GRAPH_FAILED;
    }

    const char *layoutKPtr = attrs->GetAttrPointer<char>(PKI_ATTR_LAYOUT_K);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutKPtr);
    std::string layoutKStr(layoutKPtr);
    if (layoutKStr == "BSND") {
        info.layoutK = PkiDataLayout::BSND;
    } else if (layoutKStr == "TND") {
        info.layoutK = PkiDataLayout::TND;
    } else if (layoutKStr == "PA_BBND") {
        info.layoutK = PkiDataLayout::PA_BBND;
        info.pageAttention = true;
    } else {
        OP_LOGE(context_, "layout_k must be BSND, TND or PA_BBND, got %s", layoutKStr.c_str());
        return ge::GRAPH_FAILED;
    }

    // ---- 1. layout_q and layout_k must match unless PA ----
    if (!info.pageAttention && info.layoutQ != info.layoutK) {
        OP_LOGE(context_, "layout_q(%s) and layout_k(%s) must match in non-PA scenario",
                layoutQStr.c_str(), layoutKStr.c_str());
        return ge::GRAPH_FAILED;
    }

    // ---- 2/3. act_seq / block_table presence rules ----
    const gert::Tensor *actSeqQ = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_Q_INDEX);
    const gert::Tensor *actSeqK = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_K_INDEX);
    const gert::Tensor *blockTable = context_->GetOptionalInputTensor(PKI_BLOCK_TABLE_INDEX);

    if (info.pageAttention) {
        // PA scenario: actual_seq_k and block_table must be non-null
        OP_CHECK_IF(actSeqK == nullptr,
                    OP_LOGE(context_, "PA scenario, actual_seq_k must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(blockTable == nullptr,
                    OP_LOGE(context_, "PA scenario, block_table must not be null"),
                    return ge::GRAPH_FAILED);
        if (info.layoutQ == PkiDataLayout::BSND) {
            // PA + BSND: actual_seq_q must be null
            OP_CHECK_IF(actSeqQ != nullptr,
                        OP_LOGE(context_, "PA + BSND, actual_seq_q must be null"),
                        return ge::GRAPH_FAILED);
        } else {
            // PA + TND: actual_seq_q must be non-null
            OP_CHECK_IF(actSeqQ == nullptr,
                        OP_LOGE(context_, "PA + TND, actual_seq_q must not be null"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // non-PA: block_table must be null
        OP_CHECK_IF(blockTable != nullptr,
                    OP_LOGE(context_, "non-PA scenario, block_table must be null"),
                    return ge::GRAPH_FAILED);
        if (info.layoutQ == PkiDataLayout::BSND) {
            // BSND: actual_seq_q and actual_seq_k must be null
            OP_CHECK_IF(actSeqQ != nullptr,
                        OP_LOGE(context_, "BSND scenario, actual_seq_q must be null"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(actSeqK != nullptr,
                        OP_LOGE(context_, "BSND scenario, actual_seq_k must be null"),
                        return ge::GRAPH_FAILED);
        } else {
            // TND: actual_seq_q and actual_seq_k must be non-null
            OP_CHECK_IF(actSeqQ == nullptr,
                        OP_LOGE(context_, "TND scenario, actual_seq_q must not be null"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(actSeqK == nullptr,
                        OP_LOGE(context_, "TND scenario, actual_seq_k must not be null"),
                        return ge::GRAPH_FAILED);
        }
    }

    const int64_t *maskModePtr = attrs->GetInt(PKI_ATTR_MASK_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskModePtr);
    info.maskMode = static_cast<int32_t>(*maskModePtr);
    if (info.maskMode != 0 && info.maskMode != 3) {
        OP_LOGE(context_, "mask_mode must be 0 or 3, got %d", info.maskMode);
        return ge::GRAPH_FAILED;
    }

    const int64_t *quantModePtr = attrs->GetInt(PKI_ATTR_QUANT_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantModePtr);
    info.quantMode = static_cast<int32_t>(*quantModePtr);
    if (info.quantMode != -1 && info.quantMode != 0 && info.quantMode != 1) {
        OP_LOGE(context_, "quant_mode must be -1, 0 or 1, got %d", info.quantMode);
        return ge::GRAPH_FAILED;
    }

    const bool *returnValuePtr = attrs->GetAttrPointer<bool>(PKI_ATTR_RETURN_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, returnValuePtr);
    info.returnValue = *returnValuePtr;

    // topk % poolSize == 0
    if (info.poolSize == 0 || info.topk % info.poolSize != 0) {
        OP_LOGE(context_, "topk(%u) must be divisible by poolSize(%u)", info.topk, info.poolSize);
        return ge::GRAPH_FAILED;
    }
    info.sparseCount = info.topk / info.poolSize;

    // Parse shapes
    const gert::Shape *queryShape = context_->GetInputShape(PKI_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    const gert::Shape *keyShape = context_->GetInputShape(PKI_POOL_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);

    info.queryDtype = context_->GetInputDesc(PKI_QUERY_INDEX)->GetDataType();
    info.keyDtype = context_->GetInputDesc(PKI_POOL_KEY_INDEX)->GetDataType();
    info.weightsDtype = context_->GetInputDesc(PKI_WEIGHTS_INDEX)->GetDataType();

    // ---- Required inputs non-null and non-empty ----
    const gert::Tensor *queryTensor = context_->GetInputTensor(PKI_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryTensor);
    const gert::Tensor *poolKeyTensor = context_->GetInputTensor(PKI_POOL_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolKeyTensor);
    const gert::Tensor *weightsTensor = context_->GetInputTensor(PKI_WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsTensor);
    const gert::Tensor *poolTailKTensor = context_->GetInputTensor(PKI_POOL_TAIL_K_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolTailKTensor);
    OP_CHECK_IF(queryShape->GetShapeSize() == 0,
                OP_LOGE(context_, "query must not be empty tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape->GetShapeSize() == 0,
                OP_LOGE(context_, "pool_key must not be empty tensor"),
                return ge::GRAPH_FAILED);
    const gert::Shape *weightsShape = context_->GetInputShape(PKI_WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsShape);
    OP_CHECK_IF(weightsShape->GetShapeSize() == 0,
                OP_LOGE(context_, "weights must not be empty tensor"),
                return ge::GRAPH_FAILED);
    const gert::Shape *poolTailKShape = context_->GetInputShape(PKI_POOL_TAIL_K_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolTailKShape);
    OP_CHECK_IF(poolTailKShape->GetShapeSize() == 0,
                OP_LOGE(context_, "pool_tail_k must not be empty tensor"),
                return ge::GRAPH_FAILED);

    // ---- A2/A3 does not support FP8 ----
    if (info.npuArch != NpuArch::DAV_3510 &&
        info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(context_, "FP8 is only supported on Ascend950 (DAV_3510)");
        return ge::GRAPH_FAILED;
    }

    // ---- query / pool_key dtype must match ----
    if (info.queryDtype != info.keyDtype) {
        OP_LOGE(context_, "query dtype(%d) must match pool_key dtype(%d)",
                static_cast<int32_t>(info.queryDtype), static_cast<int32_t>(info.keyDtype));
        return ge::GRAPH_FAILED;
    }

    // ---- quantMode-driven dtype + descale validation ----
    const gert::Tensor *qDescaleTensor = context_->GetOptionalInputTensor(PKI_Q_DESCALE_INDEX);
    const gert::Tensor *kDescaleTensor = context_->GetOptionalInputTensor(PKI_K_DESCALE_INDEX);

    if (info.quantMode == PKI_QUANT_NONE) {
        // quantMode=-1: no quantization
        // descale must be null
        OP_CHECK_IF(qDescaleTensor != nullptr,
                    OP_LOGE(context_, "quant_mode=-1, q_descale must be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDescaleTensor != nullptr,
                    OP_LOGE(context_, "quant_mode=-1, k_descale must be null"),
                    return ge::GRAPH_FAILED);
        // q/k must be FP16 or BF16
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT16 && info.queryDtype != ge::DT_BF16,
                    OP_LOGE(context_, "quant_mode=-1, query dtype must be FP16 or BF16, got %d",
                            static_cast<int32_t>(info.queryDtype)),
                    return ge::GRAPH_FAILED);
        // weights must match query
        OP_CHECK_IF(info.weightsDtype != info.queryDtype,
                    OP_LOGE(context_, "quant_mode=-1, weights dtype must match query dtype"),
                    return ge::GRAPH_FAILED);
    } else if (info.quantMode == PKI_QUANT_FP8_PER_TOKEN) {
        // quantMode=0: FP8 per-token-head
        // descale must be non-null
        OP_CHECK_IF(qDescaleTensor == nullptr,
                    OP_LOGE(context_, "quant_mode=0, q_descale must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDescaleTensor == nullptr,
                    OP_LOGE(context_, "quant_mode=0, k_descale must not be null"),
                    return ge::GRAPH_FAILED);
        // q/k must be FP8_E4M3FN
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT8_E4M3FN,
                    OP_LOGE(context_, "quant_mode=0, query dtype must be FP8_E4M3FN, got %d",
                            static_cast<int32_t>(info.queryDtype)),
                    return ge::GRAPH_FAILED);
        // descale must be FLOAT
        ge::DataType qScaleDtype = context_->GetOptionalInputDesc(PKI_Q_DESCALE_INDEX)->GetDataType();
        ge::DataType kScaleDtype = context_->GetOptionalInputDesc(PKI_K_DESCALE_INDEX)->GetDataType();
        OP_CHECK_IF(qScaleDtype != ge::DT_FLOAT,
                    OP_LOGE(context_, "quant_mode=0, q_descale dtype must be FLOAT, got %d",
                            static_cast<int32_t>(qScaleDtype)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kScaleDtype != ge::DT_FLOAT,
                    OP_LOGE(context_, "quant_mode=0, k_descale dtype must be FLOAT, got %d",
                            static_cast<int32_t>(kScaleDtype)),
                    return ge::GRAPH_FAILED);
        // weights must be FP16 or BF16
        OP_CHECK_IF(info.weightsDtype != ge::DT_FLOAT16 && info.weightsDtype != ge::DT_BF16,
                    OP_LOGE(context_, "quant_mode=0, weights must be FP16 or BF16, got %d",
                            static_cast<int32_t>(info.weightsDtype)),
                    return ge::GRAPH_FAILED);
    } else if (info.quantMode == PKI_QUANT_MXFP8) {
        // quantMode=1: mxFP8
        // descale must be non-null
        OP_CHECK_IF(qDescaleTensor == nullptr,
                    OP_LOGE(context_, "quant_mode=1, q_descale must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDescaleTensor == nullptr,
                    OP_LOGE(context_, "quant_mode=1, k_descale must not be null"),
                    return ge::GRAPH_FAILED);
        // q/k must be FP8_E4M3FN
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT8_E4M3FN,
                    OP_LOGE(context_, "quant_mode=1, query dtype must be FP8_E4M3FN, got %d",
                            static_cast<int32_t>(info.queryDtype)),
                    return ge::GRAPH_FAILED);
        // descale must be FLOAT8_E8M0
        ge::DataType qScaleDtype = context_->GetOptionalInputDesc(PKI_Q_DESCALE_INDEX)->GetDataType();
        ge::DataType kScaleDtype = context_->GetOptionalInputDesc(PKI_K_DESCALE_INDEX)->GetDataType();
        OP_CHECK_IF(qScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(context_, "quant_mode=1, q_descale dtype must be FLOAT8_E8M0, got %d",
                            static_cast<int32_t>(qScaleDtype)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(context_, "quant_mode=1, k_descale dtype must be FLOAT8_E8M0, got %d",
                            static_cast<int32_t>(kScaleDtype)),
                    return ge::GRAPH_FAILED);
        // weights must be FP16 or BF16
        OP_CHECK_IF(info.weightsDtype != ge::DT_FLOAT16 && info.weightsDtype != ge::DT_BF16,
                    OP_LOGE(context_, "quant_mode=1, weights must be FP16 or BF16, got %d",
                            static_cast<int32_t>(info.weightsDtype)),
                    return ge::GRAPH_FAILED);
    }

    // N1, N2, D from query shape
    if (info.layoutQ == PkiDataLayout::BSND) {
        OP_CHECK_IF(queryShape->GetDimNum() != 4,
                    OP_LOGE(context_, "BSND query must be 4D, got %zu", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        info.bSize = queryShape->GetDim(0);
        info.s1Size = queryShape->GetDim(1);
        info.n1Size = queryShape->GetDim(2);
        info.headDim = queryShape->GetDim(3);
    } else {
        OP_CHECK_IF(queryShape->GetDimNum() != 3,
                    OP_LOGE(context_, "TND query must be 3D, got %zu", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        info.bSize = 0; // determined from actual_seq_q
        info.s1Size = queryShape->GetDim(0); // T1
        info.n1Size = queryShape->GetDim(1);
        info.headDim = queryShape->GetDim(2);
    }

    info.n2Size = PKI_N2_FIXED;
    if (info.n1Size > PKI_N1_LIMIT) {
        OP_LOGE(context_, "N1(%u) must be <= %u", info.n1Size, PKI_N1_LIMIT);
        return ge::GRAPH_FAILED;
    }
    if (info.headDim != PKI_HEAD_DIM) {
        OP_LOGE(context_, "headDim(%u) must be 128", info.headDim);
        return ge::GRAPH_FAILED;
    }
    info.gSize = info.n1Size / info.n2Size;

    // S2 from key shape
    if (info.pageAttention) {
        OP_CHECK_IF(keyShape->GetDimNum() != 4,
                    OP_LOGE(context_, "PA_BBND key must be 4D (blockNum, blockSize, N2, D)"),
                    return ge::GRAPH_FAILED);
        info.blockSize = keyShape->GetDim(1);
        if (info.blockSize == 0 || info.blockSize % PKI_BLOCK_SIZE_FACTOR != 0 ||
            info.blockSize > PKI_BLOCK_SIZE_LIMIT) {
            OP_LOGE(context_, "blockSize(%u) must be multiple of %u and <= %u",
                    info.blockSize, PKI_BLOCK_SIZE_FACTOR, PKI_BLOCK_SIZE_LIMIT);
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(static_cast<uint32_t>(keyShape->GetDim(2)) != info.n2Size,
                    OP_LOGE(context_, "PA key N2(%ld) must be %u", keyShape->GetDim(2), info.n2Size),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(keyShape->GetDim(3)) != info.headDim,
                    OP_LOGE(context_, "PA key D(%ld) must be %u(128)", keyShape->GetDim(3), info.headDim),
                    return ge::GRAPH_FAILED);
        // keyStride0 = blockSize * N2 * D (elements per physical block, used for PA address calc)
        uint64_t keyStride = static_cast<uint64_t>(info.blockSize) * info.n2Size * info.headDim;
        OP_CHECK_IF(keyStride > UINT32_MAX,
                    OP_LOGE(context_, "keyStride0(%lu) exceeds uint32 range", keyStride),
                    return ge::GRAPH_FAILED);
        info.keyStride0 = static_cast<uint32_t>(keyStride);
        // S2 computed at runtime from actual_seq_k
        info.s2Size = 0;
    } else if (info.layoutK == PkiDataLayout::BSND) {
        OP_CHECK_IF(keyShape->GetDimNum() != 4,
                    OP_LOGE(context_, "BSND key must be 4D"),
                    return ge::GRAPH_FAILED);
        info.s2Size = keyShape->GetDim(1);
        if (info.n2Size != keyShape->GetDim(2)) {
            OP_LOGE(context_, "N2 mismatch: query N2=%u, key N2=%lu",
                    info.n2Size, keyShape->GetDim(2));
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(static_cast<uint32_t>(keyShape->GetDim(3)) != info.headDim,
                    OP_LOGE(context_, "BSND key D(%ld) must be %u(128)", keyShape->GetDim(3), info.headDim),
                    return ge::GRAPH_FAILED);
        // keyStride0 = S2 * N2 * D (B-stride for BSND address calc)
        uint64_t keyStride = static_cast<uint64_t>(info.s2Size) * info.n2Size * info.headDim;
        OP_CHECK_IF(keyStride > UINT32_MAX,
                    OP_LOGE(context_, "keyStride0(%lu) exceeds uint32 range", keyStride),
                    return ge::GRAPH_FAILED);
        info.keyStride0 = static_cast<uint32_t>(keyStride);
    } else {
        OP_CHECK_IF(keyShape->GetDimNum() != 3,
                    OP_LOGE(context_, "TND key must be 3D"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(keyShape->GetDim(1)) != info.n2Size,
                    OP_LOGE(context_, "TND key N2(%ld) must be %u", keyShape->GetDim(1), info.n2Size),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(keyShape->GetDim(2)) != info.headDim,
                    OP_LOGE(context_, "TND key D(%ld) must be %u(128)", keyShape->GetDim(2), info.headDim),
                    return ge::GRAPH_FAILED);
        info.s2Size = 0; // determined from actual_seq_k
        // keyStride0 = N2 * D (S2-stride for TND address calc)
        info.keyStride0 = info.n2Size * info.headDim;
    }

    // block_table for PA
    if (info.pageAttention) {
        const gert::Tensor *blockTableTensor = context_->GetOptionalInputTensor(PKI_BLOCK_TABLE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, blockTableTensor);
        const gert::Shape *btShape = context_->GetOptionalInputShape(PKI_BLOCK_TABLE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, btShape);
        OP_CHECK_IF(btShape->GetDimNum() != 2,
                    OP_LOGE(context_, "block_table must be 2D (B, maxBlockNumPerSeq)"),
                    return ge::GRAPH_FAILED);
        info.maxBlockNumPerBatch = btShape->GetDim(1);
    }

    // ---- Tensor shape validation ----

    // weights shape: BSND (B,S1,N1) 3D / TND (T1,N1) 2D
    {
        const gert::Shape *wShape = context_->GetInputShape(PKI_WEIGHTS_INDEX);
        if (info.layoutQ == PkiDataLayout::BSND) {
            OP_CHECK_IF(wShape->GetDimNum() != 3,
                        OP_LOGE(context_, "BSND weights must be 3D (B,S1,N1), got %zu",
                                wShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetDim(0) != info.bSize,
                        OP_LOGE(context_, "weights B(%ld) must match query B(%u)",
                                wShape->GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetDim(1) != info.s1Size,
                        OP_LOGE(context_, "weights S1(%ld) must match query S1(%u)",
                                wShape->GetDim(1), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetDim(2) != info.n1Size,
                        OP_LOGE(context_, "weights N1(%ld) must match query N1(%u)",
                                wShape->GetDim(2), info.n1Size),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(wShape->GetDimNum() != 2,
                        OP_LOGE(context_, "TND weights must be 2D (T1,N1), got %zu",
                                wShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetDim(0) != info.s1Size,
                        OP_LOGE(context_, "weights T1(%ld) must match query T1(%u)",
                                wShape->GetDim(0), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetDim(1) != info.n1Size,
                        OP_LOGE(context_, "weights N1(%ld) must match query N1(%u)",
                                wShape->GetDim(1), info.n1Size),
                        return ge::GRAPH_FAILED);
        }
    }

    // pool_tail_k shape: (B,) 1D
    {
        const gert::Shape *ptkShape = context_->GetInputShape(PKI_POOL_TAIL_K_INDEX);
        OP_CHECK_IF(ptkShape->GetDimNum() != 1,
                    OP_LOGE(context_, "pool_tail_k must be 1D (B,), got %zu",
                            ptkShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        // B count checked in value-range validation below
    }

    // actual_seq_q shape (non-null, TND only): (B,) 1D prefix sum
    {
        const gert::Tensor *actSeqQT = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_Q_INDEX);
        if (actSeqQT != nullptr) {
            const gert::Shape *asqShape = context_->GetOptionalInputShape(PKI_ACTUAL_SEQ_Q_INDEX);
            OP_CHECK_NULL_WITH_CONTEXT(context_, asqShape);
            OP_CHECK_IF(asqShape->GetDimNum() != 1,
                        OP_LOGE(context_, "actual_seq_q must be 1D, got %zu",
                                asqShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
        }
    }

    // actual_seq_k shape (non-null, TND/PA): (B,) 1D
    {
        const gert::Tensor *actSeqKT = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_K_INDEX);
        if (actSeqKT != nullptr) {
            const gert::Shape *askShape = context_->GetOptionalInputShape(PKI_ACTUAL_SEQ_K_INDEX);
            OP_CHECK_NULL_WITH_CONTEXT(context_, askShape);
            OP_CHECK_IF(askShape->GetDimNum() != 1,
                        OP_LOGE(context_, "actual_seq_k must be 1D, got %zu",
                                askShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
        }
    }

    // q_descale shape (non-null, quantMode>=0): BSND (B,S1,N1) 3D / TND (T1,N1) 2D
    if (qDescaleTensor != nullptr) {
        const gert::Shape *qdsShape = context_->GetOptionalInputShape(PKI_Q_DESCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, qdsShape);
        if (info.layoutQ == PkiDataLayout::BSND) {
            OP_CHECK_IF(qdsShape->GetDimNum() != 3,
                        OP_LOGE(context_, "BSND q_descale must be 3D (B,S1,N1), got %zu",
                                qdsShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetDim(0) != info.bSize,
                        OP_LOGE(context_, "q_descale B(%ld) must match query B(%u)",
                                qdsShape->GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetDim(1) != info.s1Size,
                        OP_LOGE(context_, "q_descale S1(%ld) must match query S1(%u)",
                                qdsShape->GetDim(1), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetDim(2) != info.n1Size,
                        OP_LOGE(context_, "q_descale N1(%ld) must match query N1(%u)",
                                qdsShape->GetDim(2), info.n1Size),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(qdsShape->GetDimNum() != 2,
                        OP_LOGE(context_, "TND q_descale must be 2D (T1,N1), got %zu",
                                qdsShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetDim(0) != info.s1Size,
                        OP_LOGE(context_, "q_descale T1(%ld) must match query T1(%u)",
                                qdsShape->GetDim(0), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetDim(1) != info.n1Size,
                        OP_LOGE(context_, "q_descale N1(%ld) must match query N1(%u)",
                                qdsShape->GetDim(1), info.n1Size),
                        return ge::GRAPH_FAILED);
        }
    }

    // k_descale shape (non-null, quantMode>=0): BSND (B,S2,N2) 3D / TND (T2,N2) 2D / PA (blockNum,blockSize,N2) 3D
    if (kDescaleTensor != nullptr) {
        const gert::Shape *kdsShape = context_->GetOptionalInputShape(PKI_K_DESCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kdsShape);
        if (info.pageAttention) {
            OP_CHECK_IF(kdsShape->GetDimNum() != 3,
                        OP_LOGE(context_, "PA k_descale must be 3D (blockNum,blockSize,N2), got %zu",
                                kdsShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetDim(1) != info.blockSize,
                        OP_LOGE(context_, "k_descale blockSize(%ld) must match key blockSize(%u)",
                                kdsShape->GetDim(1), info.blockSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetDim(2) != info.n2Size,
                        OP_LOGE(context_, "k_descale N2(%ld) must match key N2(%u)",
                                kdsShape->GetDim(2), info.n2Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetDim(0) != keyShape->GetDim(0),
                        OP_LOGE(context_, "PA k_descale blockNum(%ld) must match key blockNum(%ld)",
                                kdsShape->GetDim(0), keyShape->GetDim(0)),
                        return ge::GRAPH_FAILED);
        } else if (info.layoutK == PkiDataLayout::BSND) {
            OP_CHECK_IF(kdsShape->GetDimNum() != 3,
                        OP_LOGE(context_, "BSND k_descale must be 3D (B,S2,N2), got %zu",
                                kdsShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetDim(0) != info.bSize,
                        OP_LOGE(context_, "k_descale B(%ld) must match query B(%u)",
                                kdsShape->GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetDim(1) != info.s2Size,
                        OP_LOGE(context_, "BSND k_descale S2(%ld) must match key S2(%ld)",
                                kdsShape->GetDim(1), info.s2Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetDim(2) != info.n2Size,
                        OP_LOGE(context_, "k_descale N2(%ld) must match key N2(%u)",
                                kdsShape->GetDim(2), info.n2Size),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(kdsShape->GetDimNum() != 2,
                        OP_LOGE(context_, "TND k_descale must be 2D (T2,N2), got %zu",
                                kdsShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetDim(0) != keyShape->GetDim(0),
                        OP_LOGE(context_, "TND k_descale T2(%ld) must match key T2(%ld)",
                                kdsShape->GetDim(0), keyShape->GetDim(0)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetDim(1) != info.n2Size,
                        OP_LOGE(context_, "k_descale N2(%ld) must match key N2(%u)",
                                kdsShape->GetDim(1), info.n2Size),
                        return ge::GRAPH_FAILED);
        }
    }

    // sparse_indices_out shape: BSND (B,S1,topk+poolSize-1) 3D / TND (T1,topk+poolSize-1) 2D
    {
        const gert::Shape *idxOutShape = context_->GetOutputShape(PKI_SPARSE_INDICES_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, idxOutShape);
        int64_t outLastDim = static_cast<int64_t>(info.topk) + info.poolSize - 1;
        if (info.layoutQ == PkiDataLayout::BSND) {
            OP_CHECK_IF(idxOutShape->GetDimNum() != 3,
                        OP_LOGE(context_, "BSND sparse_indices_out must be 3D, got %zu",
                                idxOutShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetDim(0) != info.bSize,
                        OP_LOGE(context_, "sparse_indices_out B(%ld) must match query B(%u)",
                                idxOutShape->GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetDim(1) != info.s1Size,
                        OP_LOGE(context_, "sparse_indices_out S1(%ld) must match query S1(%u)",
                                idxOutShape->GetDim(1), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetDim(2) != outLastDim,
                        OP_LOGE(context_, "sparse_indices_out last dim(%ld) must be topk+poolSize-1(%ld)",
                                idxOutShape->GetDim(2), outLastDim),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(idxOutShape->GetDimNum() != 2,
                        OP_LOGE(context_, "TND sparse_indices_out must be 2D, got %zu",
                                idxOutShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetDim(0) != info.s1Size,
                        OP_LOGE(context_, "sparse_indices_out T1(%ld) must match query T1(%u)",
                                idxOutShape->GetDim(0), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetDim(1) != outLastDim,
                        OP_LOGE(context_, "sparse_indices_out last dim(%ld) must be topk+poolSize-1(%ld)",
                                idxOutShape->GetDim(1), outLastDim),
                        return ge::GRAPH_FAILED);
        }
    }

    // sparse_values_out shape (returnValue=true): BSND (B,S1,topk//poolSize) 3D / TND (T1,topk//poolSize) 2D
    if (info.returnValue) {
        const gert::Shape *valOutShape = context_->GetOutputShape(PKI_SPARSE_VALUES_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, valOutShape);
        int64_t valLastDim = static_cast<int64_t>(info.sparseCount);
        if (info.layoutQ == PkiDataLayout::BSND) {
            OP_CHECK_IF(valOutShape->GetDimNum() != 3,
                        OP_LOGE(context_, "BSND sparse_values_out must be 3D, got %zu",
                                valOutShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetDim(0) != info.bSize,
                        OP_LOGE(context_, "sparse_values_out B(%ld) must match query B(%u)",
                                valOutShape->GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetDim(1) != info.s1Size,
                        OP_LOGE(context_, "sparse_values_out S1(%ld) must match query S1(%u)",
                                valOutShape->GetDim(1), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetDim(2) != valLastDim,
                        OP_LOGE(context_, "sparse_values_out last dim(%ld) must be topk/poolSize(%ld)",
                                valOutShape->GetDim(2), valLastDim),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(valOutShape->GetDimNum() != 2,
                        OP_LOGE(context_, "TND sparse_values_out must be 2D, got %zu",
                                valOutShape->GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetDim(0) != info.s1Size,
                        OP_LOGE(context_, "sparse_values_out T1(%ld) must match query T1(%u)",
                                valOutShape->GetDim(0), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetDim(1) != valLastDim,
                        OP_LOGE(context_, "sparse_values_out last dim(%ld) must be topk/poolSize(%ld)",
                                valOutShape->GetDim(1), valLastDim),
                        return ge::GRAPH_FAILED);
        }
    }

    // ---- Value-range validation (requires ValueDepend + support_aclnn) ----

    // #5/#6: actual_seq_q prefix-sum (TND): determines B, validates non-decreasing + last==T1
    if (info.layoutQ == PkiDataLayout::TND) {
        const gert::Tensor *asqTensor = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_Q_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, asqTensor);
        const int32_t *asqData = asqTensor->GetData<int32_t>();
        OP_CHECK_IF(asqData == nullptr,
                    OP_LOGE(context_, "TND: actual_seq_q data is null, tiling sink not supported"),
                    return ge::GRAPH_FAILED);
        int64_t asqCount = asqTensor->GetShapeSize();
        info.bSize = static_cast<uint32_t>(asqCount);
        int32_t prev = 0;
        for (int64_t i = 0; i < asqCount; ++i) {
            OP_CHECK_IF(asqData[i] < 0,
                        OP_LOGE(context_, "actual_seq_q[%ld]=%d must be >= 0", i, asqData[i]),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(asqData[i] < prev,
                        OP_LOGE(context_, "actual_seq_q must be non-decreasing, [%ld]=%d < prev=%d",
                                i, asqData[i], prev),
                        return ge::GRAPH_FAILED);
            prev = asqData[i];
        }
        if (asqCount > 0) {
            OP_CHECK_IF(asqData[asqCount - 1] != static_cast<int32_t>(info.s1Size),
                        OP_LOGE(context_, "actual_seq_q last(%d) must equal query T1(%u)",
                                asqData[asqCount - 1], info.s1Size),
                        return ge::GRAPH_FAILED);
        }
    }

    // #5: actual_seq_k values: TND prefix-sum / PA non-prefix-sum
    if (info.layoutK == PkiDataLayout::TND || info.pageAttention) {
        const gert::Tensor *askTensor = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_K_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, askTensor);
        const int32_t *askData = askTensor->GetData<int32_t>();
        OP_CHECK_IF(askData == nullptr,
                    OP_LOGE(context_, "actual_seq_k data is null, tiling sink not supported"),
                    return ge::GRAPH_FAILED);
        int64_t askCount = askTensor->GetShapeSize();
        OP_CHECK_IF(static_cast<uint32_t>(askCount) != info.bSize,
                    OP_LOGE(context_, "actual_seq_k count(%ld) must equal B(%u)", askCount, info.bSize),
                    return ge::GRAPH_FAILED);

        if (info.layoutK == PkiDataLayout::TND) {
            int32_t prev = 0;
            for (int64_t i = 0; i < askCount; ++i) {
                OP_CHECK_IF(askData[i] < 0,
                            OP_LOGE(context_, "actual_seq_k[%ld]=%d must be >= 0", i, askData[i]),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(askData[i] < prev,
                            OP_LOGE(context_, "actual_seq_k must be non-decreasing, [%ld]=%d < prev=%d",
                                    i, askData[i], prev),
                            return ge::GRAPH_FAILED);
                prev = askData[i];
            }
            if (askCount > 0) {
                OP_CHECK_IF(askData[askCount - 1] != static_cast<int32_t>(keyShape->GetDim(0)),
                            OP_LOGE(context_, "actual_seq_k last(%d) must equal key T2(%ld)",
                                    askData[askCount - 1], keyShape->GetDim(0)),
                            return ge::GRAPH_FAILED);
            }
        } else {
            // PA: non-prefix-sum, each value is current batch token count
            uint32_t maxTokens = 0;
            for (int64_t i = 0; i < askCount; ++i) {
                OP_CHECK_IF(askData[i] < 0,
                            OP_LOGE(context_, "actual_seq_k[%ld]=%d must be >= 0", i, askData[i]),
                            return ge::GRAPH_FAILED);
                if (static_cast<uint32_t>(askData[i]) > maxTokens) {
                    maxTokens = static_cast<uint32_t>(askData[i]);
                }
            }
            // #3: block_table dim(1) >= maxBlockNumPerSeq
            uint32_t maxBlockNumPerSeq = CeilDiv(maxTokens, info.blockSize);
            OP_CHECK_IF(info.maxBlockNumPerBatch < maxBlockNumPerSeq,
                        OP_LOGE(context_, "block_table dim(1)=%u must be >= maxBlockNumPerSeq=%u",
                                info.maxBlockNumPerBatch, maxBlockNumPerSeq),
                        return ge::GRAPH_FAILED);
        }
    }

    // #2: block_table dim(0) == B (PA, after B determined)
    if (info.pageAttention) {
        const gert::Shape *btShape = context_->GetOptionalInputShape(PKI_BLOCK_TABLE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, btShape);
        OP_CHECK_IF(static_cast<uint32_t>(btShape->GetDim(0)) != info.bSize,
                    OP_LOGE(context_, "block_table dim(0)(%ld) must equal B(%u)",
                            btShape->GetDim(0), info.bSize),
                    return ge::GRAPH_FAILED);
    }

    // #4: pool_tail_k values: each must be in [0, pool_size-1]
    {
        const gert::Tensor *ptkTensor = context_->GetInputTensor(PKI_POOL_TAIL_K_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, ptkTensor);
        OP_CHECK_IF(static_cast<uint32_t>(ptkTensor->GetShapeSize()) != info.bSize,
                    OP_LOGE(context_, "pool_tail_k count(%ld) must equal B(%u)",
                            ptkTensor->GetShapeSize(), info.bSize),
                    return ge::GRAPH_FAILED);
        const int32_t *ptkData = ptkTensor->GetData<int32_t>();
        if (ptkData != nullptr) {
            int64_t ptkCount = ptkTensor->GetShapeSize();
            for (int64_t i = 0; i < ptkCount; ++i) {
                OP_CHECK_IF(ptkData[i] < 0 || ptkData[i] >= static_cast<int32_t>(info.poolSize),
                            OP_LOGE(context_, "pool_tail_k[%ld]=%d must be in [0, %u)",
                                    i, ptkData[i], info.poolSize),
                            return ge::GRAPH_FAILED);
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PoolKeyIndexerTiling::CalcTilingParams(PoolKeyIndexerTilingInfo &info)
{
    // s1BaseSize / mBaseSizeMax selection (ref design §5.2)
    if (info.sparseCount > PKI_TOPK_6K) {
        tilingData_.s1BaseSize = PKI_S1_BASE_SIZE_SMALL;
        tilingData_.mBaseSizeMax = PKI_M_BASE_SIZE_SMALL;
    } else {
        tilingData_.s1BaseSize = PKI_S1_BASE_SIZE;
        tilingData_.mBaseSizeMax = PKI_M_BASE_SIZE;
    }
    tilingData_.mBaseSize = tilingData_.s1BaseSize * info.gSize;
    tilingData_.s2BaseSize = PKI_S2_BASE_SIZE;
    tilingData_.trunkLen = GetTrunkLen(info.sparseCount);

    // Core count from platform
    tilingData_.usedCoreNum = info.aicNum;
    info.usedCoreNum = info.aicNum;

    return ge::GRAPH_SUCCESS;
}

uint32_t PoolKeyIndexerTiling::GetTrunkLen(uint32_t sparseCount)
{
    if (sparseCount <= PKI_TOPK_2K) return PKI_TRUNK_LEN_16K;
    if (sparseCount <= PKI_TOPK_3K) return PKI_TRUNK_LEN_12K;
    if (sparseCount <= PKI_TOPK_4K) return PKI_TRUNK_LEN_8K;
    if (sparseCount <= PKI_TOPK_5K) return PKI_TRUNK_LEN_4K;
    if (sparseCount <= PKI_TOPK_6K) return PKI_TRUNK_LEN_2K;
    return PKI_TRUNK_LEN_12K;
}

ge::graphStatus PoolKeyIndexerTiling::CalcWorkspaceSize(PoolKeyIndexerTilingInfo &info,
                                                         uint64_t &workspaceSize)
{
    uint64_t offset = 0;
    uint32_t aicNum = info.aicNum;
    uint32_t blockNum = info.aivNum; // GetBlockNum equivalent
    uint32_t s1BaseSize = tilingData_.s1BaseSize;
    uint32_t s2BaseSize = tilingData_.s2BaseSize;
    uint32_t sparseCount = info.sparseCount;
    uint32_t topkCountAlign16 = AlignUp(sparseCount, 16U);

    // S2 max for scoreGm sizing (use kSeqSize from shape or actual_seq_k)
    uint64_t kSeqSize = (info.s2Size > 0) ? static_cast<uint64_t>(info.s2Size) : 8192;
    uint64_t kSeqSizeAlign = RoundUp64(kSeqSize, s2BaseSize);

    // scoreGm: per-core, s1BaseSize × kSeqSizeAlign × sizeof(SCORE_T)
    uint64_t singleCoreScoreSize = static_cast<uint64_t>(s1BaseSize) * kSeqSizeAlign * PKI_SCORE_T_SIZE;
    uint64_t scoreGmSize = RoundUp64(static_cast<uint64_t>(aicNum) * singleCoreScoreSize, PKI_GM_ALIGN_BYTES);
    tilingData_.wsOffScore = offset;
    offset += scoreGmSize;

    // LD score: GetBlockNum × s1BaseSize × topkCountAlign16 × 2 × sizeof(SCORE_T)
    uint64_t ldScoreSize = RoundUp64(
        static_cast<uint64_t>(blockNum) * s1BaseSize * topkCountAlign16 * 2 * PKI_SCORE_T_SIZE, 32);
    tilingData_.wsOffLdScore = offset;
    offset += ldScoreSize;

    // LD index: GetBlockNum × s1BaseSize × topkCountAlign16 × 2 × sizeof(int32_t)
    uint64_t ldIdxSize = RoundUp64(
        static_cast<uint64_t>(blockNum) * s1BaseSize * topkCountAlign16 * 2 * sizeof(int32_t), 32);
    tilingData_.wsOffLdIdx = offset;
    offset += ldIdxSize;

    workspaceSize = offset;
    OP_LOGI(context_->GetNodeName(), "PoolKeyIndexer workspace: scoreGm=%lu, ldScore=%lu, ldIdx=%lu, total=%lu",
            scoreGmSize, ldScoreSize, ldIdxSize, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

void PoolKeyIndexerTiling::SetTilingData(PoolKeyIndexerTilingInfo &info)
{
    tilingData_.bSize = info.bSize;
    tilingData_.gSize = info.gSize;
    tilingData_.s1Size = info.s1Size;
    tilingData_.s2Size = info.s2Size;
    tilingData_.sparseCount = info.sparseCount;
    tilingData_.topk = info.topk;
    tilingData_.poolSize = info.poolSize;
    tilingData_.maskMode = static_cast<uint32_t>(info.maskMode);
    tilingData_.quantMode = static_cast<uint32_t>(info.quantMode);
    tilingData_.returnValue = info.returnValue ? 1 : 0;
    tilingData_.layoutQ = static_cast<uint32_t>(info.layoutQ);
    tilingData_.layoutK = static_cast<uint32_t>(info.layoutK);
    tilingData_.blockSize = info.blockSize;
    tilingData_.maxBlockNumPerBatch = info.maxBlockNumPerBatch;
    tilingData_.keyStride0 = info.keyStride0;
    tilingData_.keyDequantScaleStride0 = info.keyDequantScaleStride0;
}

ge::graphStatus PoolKeyIndexerTiling::DoTiling(PoolKeyIndexerTilingInfo *tilingInfo)
{
    auto &info = *tilingInfo;
    info.opName = context_->GetNodeName();

    // Get platform info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    info.npuArch = ascendcPlatform.GetNpuArch();
    ascendcPlatform.GetCoreNumPerAicore(info.aivNum, info.aicNum);
    info.ubSize = ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB);

    // Parse and validate
    auto ret = ParseAndCheckParams(info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Compute tiling params
    ret = CalcTilingParams(info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Compute workspace
    uint64_t workspaceSize = 0;
    ret = CalcWorkspaceSize(info, workspaceSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Set tiling data fields
    SetTilingData(info);

    // Write tiling data
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // Set workspace size
    size_t curWsSize = 0;
    if (context_->GetWorkspaceSize(curWsSize) != ge::GRAPH_SUCCESS) {
        curWsSize = 0;
    }
    size_t finalWsSize = std::max(curWsSize, static_cast<size_t>(workspaceSize));
    context_->SetWorkspaceSize(finalWsSize);

    // Set blockDim (1:2 AIC:AIV)
    uint32_t blockDim = platform_ascendc::CalcTschBlockDim(info.aivNum, info.aicNum, info.aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOGI(info.opName, "PoolKeyIndexer DoTiling: blockDim=%u, aicNum=%u, aivNum=%u, wsSize=%lu",
            blockDim, info.aicNum, info.aivNum, finalWsSize);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
