/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generic_lightning_indexer_tiling.h"
#include <algorithm>
#include <string>
#include "register/op_impl_registry.h"
#include "../op_kernel/generic_lightning_indexer_template_tiling_key.h"

using namespace ge;
namespace optiling {

static uint64_t RoundUp64(uint64_t val, uint64_t align)
{
    return (align == 0) ? val : (val + align - 1) / align * align;
}

static bool IsTopkValid(int64_t topk)
{
    return (topk >= 1 && topk <= 2048) || topk == 3072 || topk == 4096 || topk == 5120 || topk == 6144 ||
           topk == 7168 || topk == 8192;
}

static uint32_t QuantModeToTpl(int32_t hostQuant)
{
    if (hostQuant < 0) {
        return GLI_TPL_QUANT_NONE;
    }
    if (hostQuant == 0 || hostQuant == 1) {
        return GLI_TPL_QUANT_FP8_PTH;
    }
    return static_cast<uint32_t>(hostQuant + 1);
}

static uint32_t DtypeToTpl(ge::DataType dt)
{
    if (dt == ge::DT_BF16) {
        return GLI_TPL_BF16;
    }
    if (dt == ge::DT_FLOAT8_E4M3FN) {
        return GLI_TPL_FP8;
    }
    if (dt == ge::DT_INT8) {
        return GLI_TPL_INT8;
    }
    if (dt == ge::DT_HIFLOAT8) {
        return GLI_TPL_HIFP8;
    }
    if (dt == ge::DT_FLOAT4_E2M1) {
        return GLI_TPL_FP4;
    }
    if (dt == ge::DT_FLOAT8_E5M2) {
        return GLI_TPL_FP8_E5M2;
    }
    return GLI_TPL_FP16;
}

static bool IsPaLayoutK(const std::string &layoutK)
{
    return layoutK == "PA_BBND" || layoutK == "PA_BSND" || layoutK == "BBND" || layoutK == "BNBD";
}

static bool IsHostVisibleValue(const gert::Tensor *tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto placement = tensor->GetPlacement();
    return (placement == gert::kOnHost) || (placement == gert::kFollowing);
}

static int64_t SeqElem(const gert::Tensor *tensor, int64_t idx)
{
    if (!IsHostVisibleValue(tensor) || idx < 0) {
        return -1;
    }
    if (tensor->GetDataType() == ge::DT_INT64) {
        const int64_t *data = tensor->GetData<int64_t>();
        return (data == nullptr) ? -1 : data[idx];
    }
    const int32_t *data = tensor->GetData<int32_t>();
    return (data == nullptr) ? -1 : static_cast<int64_t>(data[idx]);
}

static uint32_t SeqDtypeBit(const gert::Tensor *tensor, uint32_t bit)
{
    return (tensor != nullptr && tensor->GetDataType() == ge::DT_INT64) ? bit : 0U;
}

// seqused / start_loc / block_table 给出 B；否则读 ValueDepend：首元素为 0 且末元素等于 T 视为 cu_seqlens [B+1]。
static void InferSeqBatch(const gert::Tensor *act, const gert::Tensor *seqused, const gert::Tensor *startLoc,
                          const gert::StorageShape *btShape, int64_t tLen, uint32_t &bSize, uint32_t &cuFormat)
{
    cuFormat = 0;
    auto dim0 = [](const gert::Tensor *t) -> int64_t { return t->GetStorageShape().GetDim(0); };
    if (seqused != nullptr) {
        bSize = static_cast<uint32_t>(dim0(seqused));
        if (act != nullptr && dim0(act) == static_cast<int64_t>(bSize) + 1) {
            cuFormat = 1;
        }
        return;
    }
    if (startLoc != nullptr) {
        bSize = static_cast<uint32_t>(dim0(startLoc));
        if (act != nullptr && dim0(act) == static_cast<int64_t>(bSize) + 1) {
            cuFormat = 1;
        }
        return;
    }
    if (btShape != nullptr) {
        bSize = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(0));
        if (act != nullptr && act->GetStorageShape().GetDim(0) == static_cast<int64_t>(bSize) + 1) {
            cuFormat = 1;
        }
        return;
    }
    if (act == nullptr) {
        bSize = 1;
        return;
    }
    const int64_t n = dim0(act);
    const int64_t v0 = SeqElem(act, 0);
    const int64_t vlast = (n > 0) ? SeqElem(act, n - 1) : -1;
    if (n >= 2 && v0 == 0 && (tLen <= 0 || vlast == tLen)) {
        cuFormat = 1;
        bSize = static_cast<uint32_t>(n - 1);
        return;
    }
    bSize = static_cast<uint32_t>(n);
}

ge::graphStatus GenericLightningIndexerTiling::ParseAndCheckParams(GenericLightningIndexerTilingInfo &info)
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *topkPtr = attrs->GetInt(GLI_ATTR_TOPK);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topkPtr);
    OP_CHECK_IF(!IsTopkValid(*topkPtr), OP_LOGE(context_, "topk(%ld) invalid", *topkPtr), return ge::GRAPH_FAILED);
    info.topk = static_cast<uint32_t>(*topkPtr);
    info.sparseCount = info.topk;

    const int64_t *cmpPtr = attrs->GetInt(GLI_ATTR_CMP_RATIO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cmpPtr);
    OP_CHECK_IF(*cmpPtr < 1 || *cmpPtr > GLI_CMP_RATIO_MAX, OP_LOGE(context_, "cmp_ratio(%ld) not in [1,128]", *cmpPtr),
                return ge::GRAPH_FAILED);
    info.cmpRatio = static_cast<uint32_t>(*cmpPtr);

    const char *layoutQPtr = attrs->GetAttrPointer<char>(GLI_ATTR_LAYOUT_Q);
    const char *layoutKPtr = attrs->GetAttrPointer<char>(GLI_ATTR_LAYOUT_K);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutQPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutKPtr);
    const std::string layoutQStr(layoutQPtr);
    const std::string layoutKStr(layoutKPtr);
    if (layoutQStr == "BSND") {
        info.layoutQ = GliDataLayout::BSND;
    } else if (layoutQStr == "TND") {
        info.layoutQ = GliDataLayout::TND;
    } else {
        OP_LOGE(context_, "layout_q must be BSND or TND, got %s", layoutQStr.c_str());
        return ge::GRAPH_FAILED;
    }
    if (layoutKStr == "BSND") {
        info.layoutK = GliDataLayout::BSND;
    } else if (layoutKStr == "TND") {
        info.layoutK = GliDataLayout::TND;
    } else if (IsPaLayoutK(layoutKStr)) {
        info.layoutK = GliDataLayout::PA_BBND;
        info.pageAttention = true;
        info.paBnbd = (layoutKStr == "BNBD");
    } else {
        OP_LOGE(context_, "layout_k must be BSND, TND, PA_BBND, PA_BSND, BBND or BNBD, got %s", layoutKStr.c_str());
        return ge::GRAPH_FAILED;
    }
    if (!info.pageAttention && info.layoutQ != info.layoutK) {
        OP_LOGE(context_, "layout_q and layout_k must match unless PA_BBND");
        return ge::GRAPH_FAILED;
    }

    const int64_t *sparsePtr = attrs->GetInt(GLI_ATTR_SPARSE_MODE);
    const int64_t *actPtr = attrs->GetInt(GLI_ATTR_ACT_TYPE);
    const bool *weightedPtr = attrs->GetAttrPointer<bool>(GLI_ATTR_HAS_WEIGHTED_SUM);
    const bool *expandPtr = attrs->GetAttrPointer<bool>(GLI_ATTR_EXPAND_INDICES);
    const bool *appendPtr = attrs->GetAttrPointer<bool>(GLI_ATTR_APPEND_TAIL);
    const int64_t *quantPtr = attrs->GetInt(GLI_ATTR_QUANT_MODE);
    const bool *retValPtr = attrs->GetAttrPointer<bool>(GLI_ATTR_RETURN_VALUE);
    const int64_t *selectPtr = attrs->GetInt(GLI_ATTR_SELECT_MODE);
    const int64_t *keyReducePtr = attrs->GetInt(GLI_ATTR_KEY_REDUCE_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparsePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, actPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightedPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, appendPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, retValPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, selectPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyReducePtr);

    info.sparseMode = static_cast<int32_t>(*sparsePtr);
    info.actType = static_cast<int32_t>(*actPtr);
    info.hasWeightedSum = *weightedPtr;
    info.expandIndices = *expandPtr;
    info.appendTail = *appendPtr;
    info.quantMode = static_cast<int32_t>(*quantPtr);
    info.returnValue = *retValPtr;
    info.selectMode = static_cast<int32_t>(*selectPtr);
    info.keyReduceMode = static_cast<int32_t>(*keyReducePtr);
    const int64_t *initBlocksPtr = attrs->GetInt(GLI_ATTR_INIT_BLOCKS);
    const int64_t *localBlocksPtr = attrs->GetInt(GLI_ATTR_LOCAL_BLOCKS);
    info.initBlocks = (initBlocksPtr == nullptr) ? 0 : static_cast<uint32_t>(*initBlocksPtr);
    info.localBlocks = (localBlocksPtr == nullptr) ? 0 : static_cast<uint32_t>(*localBlocksPtr);
    const float *scoreScalePtr = attrs->GetAttrPointer<float>(GLI_ATTR_SCORE_SCALE);
    info.scoreScale = (scoreScalePtr == nullptr) ? 1.0f : *scoreScalePtr;
    const int64_t *maxSeqQPtr = attrs->GetInt(GLI_ATTR_MAX_SEQLEN_Q);
    info.maxSeqlenQ = (maxSeqQPtr == nullptr) ? -1 : static_cast<int32_t>(*maxSeqQPtr);

    OP_CHECK_IF(info.sparseMode != 0 && info.sparseMode != 3 && info.sparseMode != 4,
                OP_LOGE(context_, "sparse_mode must be 0/3/4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.appendTail && !info.expandIndices, OP_LOGE(context_, "append_tail requires expand_indices"),
                return ge::GRAPH_FAILED);

    const bool isMsa = (info.selectMode == 1 || info.keyReduceMode == 1 || info.actType == 1);
    if (isMsa) {
        OP_CHECK_IF(info.selectMode != 1 || info.keyReduceMode != 1 || info.actType != 1,
                    OP_LOGE(context_, "MSA requires act_type=1, key_reduce_mode=1, select_mode=1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(info.hasWeightedSum, OP_LOGE(context_, "MSA requires has_weighted_sum=false"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(info.expandIndices, OP_LOGE(context_, "MSA forbids expand_indices"), return ge::GRAPH_FAILED);
        info.returnValue = true;
    } else {
        OP_CHECK_IF(info.actType != 0, OP_LOGE(context_, "non-MSA act_type must be 0 (relu)"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(info.selectMode != 0, OP_LOGE(context_, "non-MSA select_mode must be 0"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(info.keyReduceMode != 0, OP_LOGE(context_, "non-MSA key_reduce_mode must be 0"),
                    return ge::GRAPH_FAILED);
    }

    info.expandMode = 0;
    if (info.expandIndices) {
        info.expandMode = info.appendTail ? 2 : 1;
    }

    const gert::StorageShape *qShape = context_->GetInputShape(GLI_QUERY_INDEX);
    const gert::StorageShape *kShape = context_->GetInputShape(GLI_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kShape);

    const gert::Tensor *weights = context_->GetOptionalInputTensor(GLI_WEIGHTS_INDEX);
    if (info.hasWeightedSum) {
        OP_CHECK_IF(weights == nullptr, OP_LOGE(context_, "has_weighted_sum=true requires weights"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(weights != nullptr, OP_LOGE(context_, "has_weighted_sum=false forbids weights"),
                    return ge::GRAPH_FAILED);
    }

    auto qDesc = context_->GetInputDesc(GLI_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qDesc);
    info.queryDtype = qDesc->GetDataType();
    auto kDesc = context_->GetInputDesc(GLI_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kDesc);
    info.keyDtype = kDesc->GetDataType();
    const bool mixedFp16Int8 = isMsa && info.queryDtype == ge::DT_FLOAT16 && info.keyDtype == ge::DT_INT8;
    const bool qFp = (info.queryDtype == ge::DT_FLOAT16 || info.queryDtype == ge::DT_BF16);
    const bool mixedFp8Key = qFp && info.keyDtype == ge::DT_FLOAT8_E4M3FN;
    OP_CHECK_IF(info.keyDtype != info.queryDtype && !mixedFp16Int8 && !mixedFp8Key,
                OP_LOGE(context_, "query and key dtype must match, except MSA FP16×INT8 and Qwen FP8 key"),
                return ge::GRAPH_FAILED);
    if (info.quantMode < 0 && mixedFp8Key) {
        info.quantMode = 0;
    }
    if (info.quantMode < 0 && (info.queryDtype == ge::DT_INT8 || info.keyDtype == ge::DT_INT8)) {
        info.quantMode = 2;
    }
    if (info.quantMode < 0 && info.queryDtype == ge::DT_HIFLOAT8 && !isMsa) {
        info.quantMode = 4;
    }
    if (info.quantMode < 0 && info.queryDtype == ge::DT_FLOAT4_E2M1) {
        info.quantMode = 5;
    }

    const gert::Tensor *actQ = context_->GetOptionalInputTensor(GLI_ACTUAL_SEQ_Q_INDEX);
    const gert::Tensor *actK = context_->GetOptionalInputTensor(GLI_ACTUAL_SEQ_K_INDEX);
    const gert::Tensor *sequsedQ = context_->GetOptionalInputTensor(GLI_SEQUSED_Q_INDEX);
    const gert::Tensor *sequsedK = context_->GetOptionalInputTensor(GLI_SEQUSED_K_INDEX);
    const gert::Tensor *startLoc = context_->GetOptionalInputTensor(GLI_START_LOC_INDEX);
    const gert::Tensor *residualTensor = context_->GetOptionalInputTensor(GLI_CMP_RESIDUAL_K_INDEX);

    if (info.layoutQ == GliDataLayout::BSND) {
        OP_CHECK_IF(qShape->GetStorageShape().GetDimNum() != 4, OP_LOGE(context_, "BSND query must be 4D"),
                    return ge::GRAPH_FAILED);
        info.bSize = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));
        info.s1Size = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(1));
        info.n1Size = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(2));
        info.headDim = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(3));
    } else {
        OP_CHECK_IF(qShape->GetStorageShape().GetDimNum() != 3, OP_LOGE(context_, "TND query must be 3D"),
                    return ge::GRAPH_FAILED);
        info.s1Size = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));
        info.n1Size = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(1));
        info.headDim = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(2));
        OP_CHECK_IF(actQ == nullptr, OP_LOGE(context_, "TND requires actual_seq_q"), return ge::GRAPH_FAILED);
        InferSeqBatch(actQ, sequsedQ, startLoc, nullptr, static_cast<int64_t>(info.s1Size), info.bSize,
                      info.seqQCuFormat);
    }
    info.msaScoreNT = (isMsa && info.layoutQ == GliDataLayout::TND) ? 1 : 0;
    info.seqDtypeFlags = SeqDtypeBit(actQ, GLI_SEQ_Q_I64) | SeqDtypeBit(actK, GLI_SEQ_K_I64) |
                         SeqDtypeBit(sequsedQ, GLI_SEQUSED_Q_I64) | SeqDtypeBit(sequsedK, GLI_SEQUSED_K_I64) |
                         SeqDtypeBit(residualTensor, GLI_RESIDUAL_I64) | SeqDtypeBit(startLoc, GLI_START_LOC_I64);
    OP_CHECK_IF(info.headDim != GLI_HEAD_DIM, OP_LOGE(context_, "headDim must be 128"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.n1Size == 0 || info.n1Size > GLI_N1_LIMIT, OP_LOGE(context_, "N1 must be in [1,64]"),
                return ge::GRAPH_FAILED);
    info.gSize = info.n1Size;

    if (info.pageAttention) {
        OP_CHECK_IF(kShape->GetStorageShape().GetDimNum() != 4, OP_LOGE(context_, "PA key must be 4D"),
                    return ge::GRAPH_FAILED);
        uint32_t n2 = 0;
        if (info.paBnbd) {
            n2 = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(1));
            info.blockSize = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(2));
        } else {
            info.blockSize = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(1));
            n2 = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(2));
        }
        OP_CHECK_IF(info.blockSize == 0 || (info.blockSize % GLI_BLOCK_SIZE_FACTOR) != 0 ||
                        info.blockSize > GLI_BLOCK_SIZE_LIMIT,
                    OP_LOGE(context_, "PA block_size invalid"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(n2 != GLI_N2_FIXED, OP_LOGE(context_, "key N2 must be 1"), return ge::GRAPH_FAILED);
        const gert::StorageShape *btShape = context_->GetOptionalInputShape(GLI_BLOCK_TABLE_INDEX);
        OP_CHECK_IF(btShape == nullptr, OP_LOGE(context_, "PA requires block_table"), return ge::GRAPH_FAILED);
        info.maxBlockNumPerBatch = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(1));
        info.s2Size = static_cast<int64_t>(info.maxBlockNumPerBatch) * info.blockSize;
        uint32_t bFromPa = 0;
        uint32_t seqKCu = 0;
        InferSeqBatch(actK, sequsedK, nullptr, btShape, -1, bFromPa, seqKCu);
        info.seqKCuFormat = seqKCu;
        if (info.layoutQ == GliDataLayout::BSND) {
            OP_CHECK_IF(bFromPa != info.bSize, OP_LOGE(context_, "PA batch from block_table/seq must match query B"),
                        return ge::GRAPH_FAILED);
        }
        const int64_t *ks0 = attrs->GetAttrPointer<int64_t>(GLI_ATTR_KEY_STRIDE0);
        if (ks0 != nullptr && *ks0 >= 0) {
            info.keyStride0 = static_cast<uint32_t>(*ks0);
        } else if (info.paBnbd) {
            info.keyStride0 = n2 * info.blockSize * info.headDim;
        } else {
            info.keyStride0 = info.blockSize * GLI_N2_FIXED * info.headDim;
        }
        const int64_t *kds0 = attrs->GetAttrPointer<int64_t>(GLI_ATTR_K_DESCALE_STRIDE0);
        if (kds0 != nullptr && *kds0 >= 0) {
            info.keyDequantScaleStride0 = static_cast<uint32_t>(*kds0);
        } else if (info.paBnbd) {
            info.keyDequantScaleStride0 = n2 * info.blockSize;
        } else {
            info.keyDequantScaleStride0 = info.blockSize * GLI_N2_FIXED;
        }
        OP_CHECK_IF(actK == nullptr && sequsedK == nullptr, OP_LOGE(context_, "PA requires actual_seq_k or seqused_k"),
                    return ge::GRAPH_FAILED);
    } else if (info.layoutK == GliDataLayout::BSND) {
        OP_CHECK_IF(kShape->GetStorageShape().GetDimNum() != 4, OP_LOGE(context_, "BSND key must be 4D"),
                    return ge::GRAPH_FAILED);
        info.s2Size = kShape->GetStorageShape().GetDim(1);
        OP_CHECK_IF(static_cast<uint32_t>(kShape->GetStorageShape().GetDim(2)) != GLI_N2_FIXED,
                    OP_LOGE(context_, "key N2 must be 1"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(kShape->GetStorageShape().GetDimNum() != 3, OP_LOGE(context_, "TND key must be 3D"),
                    return ge::GRAPH_FAILED);
        info.s2Size = kShape->GetStorageShape().GetDim(0);
        OP_CHECK_IF(actK == nullptr, OP_LOGE(context_, "TND requires actual_seq_k"), return ge::GRAPH_FAILED);
        uint32_t bFromK = 0;
        InferSeqBatch(actK, sequsedK, nullptr, nullptr, info.s2Size, bFromK, info.seqKCuFormat);
        OP_CHECK_IF(bFromK != info.bSize, OP_LOGE(context_, "TND actual_seq_q/k batch mismatch"),
                    return ge::GRAPH_FAILED);
    }

    // sparse_mode=4 的尾部由 query 下标推出，不依赖 cmp_residual_k。
    const bool needResidual = (info.cmpRatio > 1 && info.sparseMode == 3 && info.keyReduceMode == 0);
    const bool residualForbidden = (info.cmpRatio <= 1 || info.sparseMode == 0 || info.keyReduceMode != 0);
    OP_CHECK_IF(needResidual && residualTensor == nullptr,
                OP_LOGE(context_, "cmp_residual_k required when cmp_ratio!=1 and sparse_mode!=0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(residualForbidden && residualTensor != nullptr,
                OP_LOGE(context_, "cmp_residual_k must be omitted unless cmp_ratio!=1 and sparse_mode is 3 or 4"),
                return ge::GRAPH_FAILED);

    const bool needQuantScale = (info.quantMode >= 0);
    const gert::Tensor *qScale = context_->GetOptionalInputTensor(GLI_Q_DESCALE_INDEX);
    const gert::Tensor *kScale = context_->GetOptionalInputTensor(GLI_K_DESCALE_INDEX);
    if (needQuantScale) {
        OP_CHECK_IF(kScale == nullptr, OP_LOGE(context_, "quant_mode>=0 requires k_descale"), return ge::GRAPH_FAILED);
        const bool keyOnlyScale =
            mixedFp16Int8 || ((info.queryDtype == ge::DT_FLOAT16 || info.queryDtype == ge::DT_BF16) &&
                              info.keyDtype == ge::DT_FLOAT8_E4M3FN);
        OP_CHECK_IF(!keyOnlyScale && qScale == nullptr, OP_LOGE(context_, "quant_mode>=0 requires q_descale"),
                    return ge::GRAPH_FAILED);
    }

    info.hasOutputIdxOffset = (context_->GetOptionalInputTensor(GLI_OUTPUT_IDX_OFFSET_INDEX) != nullptr) ? 1 : 0;

    if (isMsa) {
        int64_t nBlocks = 1;
        if (info.pageAttention) {
            nBlocks = static_cast<int64_t>(info.maxBlockNumPerBatch);
        } else if (info.cmpRatio > 0) {
            nBlocks = (info.s2Size + static_cast<int64_t>(info.cmpRatio) - 1) / static_cast<int64_t>(info.cmpRatio);
        }
        if (nBlocks <= 0) {
            nBlocks = 1;
        }
        // sparseCount 复用为 MSA 输出末维 Align16(ceil(S2/r))，供 kernel 写 sparse_values 行宽
        info.sparseCount = static_cast<uint32_t>(((nBlocks + 15) / 16) * 16);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenericLightningIndexerTiling::CalcTilingParams(GenericLightningIndexerTilingInfo &info)
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platformInfo is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    info.aicNum = ascendcPlatform.GetCoreNumAic();
    info.aivNum = ascendcPlatform.GetCoreNumAiv();
    info.usedCoreNum = std::max(info.aicNum, 1u);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenericLightningIndexerTiling::CalcWorkspaceSize(GenericLightningIndexerTilingInfo &info,
                                                                 uint64_t &workspaceSize)
{
    constexpr uint64_t BASE_TOPK = 2048;
    constexpr uint64_t TOPK_MAX_SIZE = 8192;
    constexpr uint64_t SPARSE_COUNT_8K = 8192;
    constexpr uint64_t S2_BASE_SIZE_ARCH22 = 512;
    constexpr uint64_t BLOCK_CUBE = 16;
    constexpr uint64_t WS_DOUBLE = 2;
    constexpr uint64_t LD_PARAM_NUM = 16;

    uint64_t offset = 0;
    uint32_t aicNum = info.aicNum;
    uint64_t s1BaseSize = (info.sparseCount > BASE_TOPK) ? (SPARSE_COUNT_8K / info.sparseCount) * 2 : 8;
    uint64_t mBaseSizeAlign = RoundUp64(s1BaseSize * info.gSize, BLOCK_CUBE);

    uint64_t scoreGmSize =
        RoundUp64(static_cast<uint64_t>(aicNum) * WS_DOUBLE * mBaseSizeAlign * S2_BASE_SIZE_ARCH22 * GLI_SCORE_T_SIZE,
                  GLI_GM_ALIGN_BYTES);
    uint64_t s1BaseSize35 = (info.sparseCount > BASE_TOPK) ? GLI_S1_BASE_SIZE_SMALL : GLI_S1_BASE_SIZE;
    uint64_t s2Size35Align =
        (static_cast<uint64_t>(std::max(info.s2Size, static_cast<int64_t>(1))) + GLI_S2_BASE_SIZE - 1) /
        GLI_S2_BASE_SIZE * GLI_S2_BASE_SIZE;
    uint64_t scoreHeads = (info.keyReduceMode == 1) ? static_cast<uint64_t>(std::max(info.gSize, 1u)) : 1ULL;
    uint64_t scoreGmSize35 =
        RoundUp64(static_cast<uint64_t>(aicNum) * s1BaseSize35 * scoreHeads * s2Size35Align * GLI_SCORE_T_SIZE,
                  GLI_GM_ALIGN_BYTES);
    scoreGmSize = std::max(scoreGmSize, scoreGmSize35);
    tilingData_.set_wsOffScore(offset);
    offset += scoreGmSize;

    uint64_t ldScoreSize = RoundUp64(
        static_cast<uint64_t>(aicNum) * s1BaseSize * WS_DOUBLE * WS_DOUBLE * TOPK_MAX_SIZE * GLI_SCORE_T_SIZE, 32);
    tilingData_.set_wsOffLdScore(offset);
    offset += ldScoreSize;

    uint64_t ldIdxSize =
        RoundUp64(static_cast<uint64_t>(aicNum) * s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t), 32);
    tilingData_.set_wsOffLdIdx(offset);
    offset += ldIdxSize;

    uint64_t kCastSize = 0;
    const bool castKey = (info.queryDtype == ge::DT_FLOAT16 || info.queryDtype == ge::DT_BF16) &&
                         (info.keyDtype == ge::DT_INT8 || info.keyDtype == ge::DT_FLOAT8_E4M3FN);
    if (castKey) {
        kCastSize = RoundUp64(static_cast<uint64_t>(aicNum) * GLI_S2_BASE_SIZE * GLI_HEAD_DIM * sizeof(uint16_t),
                              GLI_GM_ALIGN_BYTES);
    }
    tilingData_.set_wsOffKCast(offset);
    offset += kCastSize;

    workspaceSize = offset;
    return ge::GRAPH_SUCCESS;
}

void GenericLightningIndexerTiling::SetTilingData(const GenericLightningIndexerTilingInfo &info)
{
    tilingData_.set_bSize(info.bSize);
    tilingData_.set_gSize(info.gSize);
    tilingData_.set_s1Size(info.s1Size);
    tilingData_.set_s2Size(info.s2Size);
    tilingData_.set_sparseCount(info.sparseCount);
    tilingData_.set_topk(info.topk);
    tilingData_.set_poolSize(info.cmpRatio);
    tilingData_.set_usedCoreNum(info.usedCoreNum);
    uint32_t s1Base = (info.sparseCount > 2048) ? GLI_S1_BASE_SIZE_SMALL : GLI_S1_BASE_SIZE;
    tilingData_.set_s1BaseSize(s1Base);
    tilingData_.set_mBaseSize(GLI_M_BASE_SIZE);
    tilingData_.set_mBaseSizeMax(GLI_M_BASE_SIZE);
    tilingData_.set_s2BaseSize(GLI_S2_BASE_SIZE);
    tilingData_.set_trunkLen(GLI_TRUNK_LEN_8K);
    tilingData_.set_maskMode(static_cast<uint32_t>(info.sparseMode));
    tilingData_.set_quantMode(QuantModeToTpl(info.quantMode));
    tilingData_.set_returnValue(info.returnValue ? 1 : 0);
    tilingData_.set_layoutQ(static_cast<uint32_t>(info.layoutQ));
    tilingData_.set_layoutK(static_cast<uint32_t>(info.layoutK));
    tilingData_.set_blockSize(info.blockSize);
    tilingData_.set_maxBlockNumPerBatch(info.maxBlockNumPerBatch);
    tilingData_.set_keyStride0(info.keyStride0);
    tilingData_.set_keyDequantScaleStride0(info.keyDequantScaleStride0);
    tilingData_.set_qkScale(info.scoreScale);
    tilingData_.set_expandMode(info.expandMode);
    tilingData_.set_hasWeightedSum(info.hasWeightedSum ? 1 : 0);
    tilingData_.set_actType(static_cast<uint32_t>(info.actType));
    tilingData_.set_selectMode(static_cast<uint32_t>(info.selectMode));
    tilingData_.set_keyReduceMode(static_cast<uint32_t>(info.keyReduceMode));
    tilingData_.set_initBlocks(info.initBlocks);
    tilingData_.set_localBlocks(info.localBlocks);
    tilingData_.set_hasOutputIdxOffset(info.hasOutputIdxOffset);
    tilingData_.set_maxSeqlenQ(info.maxSeqlenQ);
    tilingData_.set_seqQCuFormat(info.seqQCuFormat);
    tilingData_.set_seqKCuFormat(info.seqKCuFormat);
    tilingData_.set_seqDtypeFlags(info.seqDtypeFlags);
    tilingData_.set_msaScoreNT(info.msaScoreNT);
}

ge::graphStatus GenericLightningIndexerTiling::DoTiling(GenericLightningIndexerTilingInfo *tilingInfo)
{
    OP_CHECK_IF(context_ == nullptr || tilingInfo == nullptr, OP_LOGE("GenericLightningIndexer", "null context"),
                return ge::GRAPH_FAILED);
    tilingInfo->opName = context_->GetNodeName();
    OP_CHECK_IF(ParseAndCheckParams(*tilingInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParseAndCheckParams failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcTilingParams(*tilingInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CalcTilingParams failed"),
                return ge::GRAPH_FAILED);

    uint64_t workspaceSize = 0;
    OP_CHECK_IF(CalcWorkspaceSize(*tilingInfo, workspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "CalcWorkspaceSize failed"), return ge::GRAPH_FAILED);

    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize += ascendcPlatform.GetLibApiWorkSpaceSize();

    SetTilingData(*tilingInfo);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_, "workspace sizes is nullptr"), return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(workspaceSize);

    uint32_t tplDtQ = DtypeToTpl(tilingInfo->queryDtype);
    uint32_t tplDtK = DtypeToTpl(tilingInfo->keyDtype);
    uint32_t headReduce = GLI_TPL_HEAD_WEIGHTED;
    if (tilingInfo->keyReduceMode == 1) {
        headReduce = GLI_TPL_HEAD_NONE;
    } else if (!tilingInfo->hasWeightedSum) {
        headReduce = GLI_TPL_HEAD_SUM;
    }
    uint64_t tilingKey =
        GET_TPL_TILING_KEY(tplDtQ, tplDtK, static_cast<uint32_t>(GLI_TPL_INT32),
                           static_cast<uint32_t>(tilingInfo->layoutQ), static_cast<uint32_t>(tilingInfo->layoutK),
                           static_cast<uint32_t>(tilingInfo->sparseMode), QuantModeToTpl(tilingInfo->quantMode),
                           static_cast<uint32_t>(tilingInfo->returnValue ? 1 : 0), headReduce, tilingInfo->expandMode);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(tilingInfo->aivNum, tilingInfo->aicNum, tilingInfo->aivNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGenericLightningIndexer(gert::TilingContext *context)
{
    GenericLightningIndexerTiling tiling(context);
    GenericLightningIndexerTilingInfo info;
    return tiling.DoTiling(&info);
}

static ge::graphStatus TilingPrepareForGenericLightningIndexer(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GenericLightningIndexer", "TilingParseContext is nullptr"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platformInfo is nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<GenericLightningIndexerCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenericLightningIndexer)
    .Tiling(TilingGenericLightningIndexer)
    .TilingParse<GenericLightningIndexerCompileInfo>(TilingPrepareForGenericLightningIndexer);

} // namespace optiling
