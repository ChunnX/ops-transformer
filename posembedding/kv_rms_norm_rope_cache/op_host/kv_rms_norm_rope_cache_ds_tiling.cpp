/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file kv_rms_norm_rope_cache_ds_tiling.cpp
 * \brief
 */
#include "kv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t RMS_NORM_LENGTHS[2] = {512, 192};
int64_t RMS_NORM_LENGTH = 512;
constexpr int64_t ROPE_LENGTH = 64;
constexpr int64_t MAX_BLOCK_DIM = 65535;
constexpr int64_t V_LENGTH = 128;
constexpr int64_t D_LENGTH = 576;

constexpr int8_t STATIC_QUANT_NONE = 0;
constexpr int8_t STATIC_QUANT_SYM = 1;
constexpr int8_t STATIC_QUANT_ASYM = 2;

constexpr uint64_t TLING_KEY_5011 = 5011;
constexpr uint64_t TLING_KEY_5010 = 5010;
constexpr uint64_t TLING_KEY_5000 = 5000;
constexpr uint64_t TLING_KEY_4011 = 4011;
constexpr uint64_t TLING_KEY_4001 = 4001;
constexpr uint64_t TLING_KEY_4010 = 4010;
constexpr uint64_t TLING_KEY_4000 = 4000;
constexpr uint64_t TLING_KEY_5001 = 5001;
constexpr uint64_t TLING_KEY_3001 = 3001;
constexpr uint64_t TLING_KEY_3000 = 3000;
constexpr uint64_t TLING_KEY_3010 = 3010;
constexpr uint64_t TLING_KEY_2000 = 2000;
constexpr uint64_t TLING_KEY_2001 = 2001;
constexpr uint64_t TLING_KEY_1000 = 1000;
constexpr uint64_t TLING_KEY_1010 = 1010;
constexpr uint64_t TLING_KEY_1001 = 1001;
constexpr uint64_t TLING_KEY_1011 = 1011;

bool KvRmsNormRopeCacheTilingDs::CheckQuantTermShape(const gert::TilingContext *context, size_t inputIdx,
                                                     int64_t headSize, bool &hasShape)
{
    bool isValid = true;
    auto termShape = context->GetOptionalInputShape(inputIdx);
    hasShape = (termShape != nullptr);
    if (hasShape) {
        auto dimNum = termShape->GetStorageShape().GetDimNum();
        isValid = (dimNum > 0 && dimNum <= DIM_TWO);
        if (isValid) {
            if (dimNum == DIM_ONE) {
                isValid = (termShape->GetStorageShape().GetDim(0) == headSize);
            } else {
                const int64_t dimFor2D = (methodMode_ == 0) ? DIM_ONE : tilingData_.get_numHead();
                isValid = (termShape->GetStorageShape().GetDim(0) == dimFor2D) &&
                          (termShape->GetStorageShape().GetDim(1) == headSize);
            }
        }
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingDs::CheckCacheValid(const gert::TilingContext *context, int64_t batchSize, int64_t numHead,
                                                 int64_t cacheLen, int64_t headSize, size_t cacheIndex,
                                                 const char *cacheName)
{
    auto cacheShapeTuple = GetShapeTuple(context, cacheIndex);
    int64_t cacheB = std::get<SHAPE_IDX_B>(cacheShapeTuple);
    int64_t cacheN = std::get<SHAPE_IDX_N>(cacheShapeTuple);
    int64_t cacheS = std::get<SHAPE_IDX_S>(cacheShapeTuple);
    int64_t cacheD = std::get<SHAPE_IDX_D>(cacheShapeTuple);
    // Batch and headnum of cache Must be consist with input kv
    if (cacheB != batchSize) {
        OP_LOGE(context->GetNodeName(), "In CacheMode::Norm, inconsist B dimension %ld of %s with kv %ld.", cacheB,
                cacheName, batchSize);
        return false;
    }
    if (cacheN != numHead) {
        OP_LOGE(context->GetNodeName(), "In CacheMode::Norm, inconsist N dimension %ld of %s with kv %ld.", cacheN,
                cacheName, numHead);
        return false;
    }
    if (cacheD != headSize) {
        OP_LOGE(context->GetNodeName(), "In CacheMode::Norm, inconsist D dimension %ld of %s with kv %ld.", cacheD,
                cacheName, headSize);
        return false;
    }
    if (cacheS < cacheLen) {
        // 'cacheLen' here is seqlen of the input kv
        OP_LOGE(context->GetNodeName(), "In CacheMode::Norm, Scache %ld of %s is smaller than kv %ld.", cacheS,
                cacheName, cacheLen);
        return false;
    }

    return true;
}

ge::graphStatus KvRmsNormRopeCacheTilingDs::ResolveQuantConfig(size_t scaleIdx, size_t offsetIdx, int64_t headSize,
                                                               const char *cacheName, int8_t &quantType)
{
    bool hasQuantScale = false;
    OP_CHECK_IF(!CheckQuantTermShape(context_, scaleIdx, headSize, hasQuantScale),
                OP_LOGE(context_->GetNodeName(), "Invalid quant scale shape for %s.", cacheName),
                return ge::GRAPH_FAILED);
    quantType = hasQuantScale ? STATIC_QUANT_SYM : STATIC_QUANT_NONE;

    if (quantType == STATIC_QUANT_SYM) {
        // try to validate quant offset if available
        bool hasQuantOffset = false;
        OP_CHECK_IF(!CheckQuantTermShape(context_, offsetIdx, headSize, hasQuantOffset),
                    OP_LOGE(context_->GetNodeName(), "Invalid quant offset shape for %s.", cacheName),
                    return ge::GRAPH_FAILED);
        quantType = hasQuantOffset ? STATIC_QUANT_ASYM : STATIC_QUANT_SYM;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvRmsNormRopeCacheTilingDs::ResolveQuantComponents()
{
    int8_t kQuantType = STATIC_QUANT_NONE;
    int8_t vQuantType = STATIC_QUANT_NONE;

    ge::graphStatus componentStatus = ge::GRAPH_SUCCESS;
    componentStatus = ResolveQuantConfig(K_ROPE_SCALE_IDX, K_ROPE_OFFSET_IDX, hDimKCache, "k_cache", kQuantType);
    if (componentStatus == ge::GRAPH_SUCCESS) {
        componentStatus = ResolveQuantConfig(C_KV_SCALE_IDX, C_KV_OFFSET_IDX, hDimVCache, "ckv_cache", vQuantType);
    }

    // Update overall quant mode
    if (componentStatus == ge::GRAPH_SUCCESS) {
        tilingData_.set_isKQuant(kQuantType);
        tilingData_.set_isVQuant(vQuantType);
        staticQuantType = std::max(kQuantType, vQuantType);
        OP_LOGI(context_->GetNodeName(), "Resolved static quant mode of Atlas 2/3 is %ld.", staticQuantType);
    }

    return componentStatus;
}

bool KvRmsNormRopeCacheTilingDs::validateQuantCrossTermShapes()
{
    auto scale1Shape = context_->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context_->GetOptionalInputShape(C_KV_SCALE_IDX);
    auto offset1Shape = context_->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
    auto offset2Shape = context_->GetOptionalInputShape(C_KV_OFFSET_IDX);
    bool isValid = true;
    if ((scale1Shape != nullptr) && (scale2Shape != nullptr)) {
        isValid = isValid && (scale1Shape->GetStorageShape().GetDimNum() == scale2Shape->GetStorageShape().GetDimNum());
    }
    if ((offset1Shape != nullptr) && (offset2Shape != nullptr)) {
        isValid =
            isValid && (offset1Shape->GetStorageShape().GetDimNum() == offset2Shape->GetStorageShape().GetDimNum());
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingDs::validateCacheDatatypes()
{
    bool isValid = true;
    // Validate cache datatypes
    auto kcacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    auto vcacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
    if (kcacheDesc == nullptr || vcacheDesc == nullptr) {
        // Caches are required
        return false;
    }

    ge::DataType kcacheDtype = kcacheDesc->GetDataType();
    ge::DataType vcacheDtype = vcacheDesc->GetDataType();

    int8_t kQuantType = tilingData_.get_isKQuant();
    int8_t vQuantType = tilingData_.get_isVQuant();

    if (kQuantType > STATIC_QUANT_NONE) {
        isValid = (kcacheDtype == ge::DT_INT8);
    } else {
        isValid = (kcacheDtype == kvDtype_);
    }

    if (vQuantType > STATIC_QUANT_NONE) {
        isValid = isValid && (vcacheDtype == ge::DT_INT8);
    } else {
        if (methodMode_ == 0) {
            isValid = isValid && (vcacheDtype == kvDtype_);
        } else {
            isValid = isValid && (vcacheDtype == vDtype_);
        }
    }

    return isValid;
}

bool KvRmsNormRopeCacheTilingDs::IsCapable()
{
    // Block invalid socs from further tiling steps for Atlas 2 and 3
    return !isRegbase_;
}

void KvRmsNormRopeCacheTilingDs::DoOpTilingPaBlkNz()
{
    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLength();
    int64_t numHead = tilingData_.get_numHead();
    int64_t bns = batchSize * numHead * seqLen;
    int64_t blockFactor = (bns + coreNum_ - 1) / coreNum_;
    int64_t numBlocks = (bns + blockFactor - 1) / blockFactor;
    tilingData_.set_blockFactor(blockFactor);
    tilingData_.set_numBlocks(numBlocks);

    int64_t maxUbFactor = (methodMode_ == 1) ? 32 : 16;
    constexpr static int64_t needUbSize = static_cast<int64_t>(170) * static_cast<int64_t>(1024);
    if (static_cast<int64_t>(ubSize_) >= static_cast<int64_t>(needUbSize)) {
        tilingData_.set_ubFactor(maxUbFactor);
    } else {
        tilingData_.set_ubFactor(1);
    }
}

ge::graphStatus KvRmsNormRopeCacheTilingDs::DoOpTiling()
{
    RMS_NORM_LENGTH = RMS_NORM_LENGTHS[methodMode_];
    auto kvShapeTuple = GetShapeTuple(context_, KV_INDEX);
    int64_t batchSize = std::get<SHAPE_IDX_B>(kvShapeTuple);
    int64_t numHead = std::get<SHAPE_IDX_N>(kvShapeTuple);
    int64_t seqLength = std::get<SHAPE_IDX_S>(kvShapeTuple);

    if (methodMode_ == 0) {
        hDimKCache = ROPE_LENGTH;     // k_cache hDim
        hDimVCache = RMS_NORM_LENGTH; // ckv_cache hDim
    } else if (methodMode_ == 1) {
        hDimKCache = RMS_NORM_LENGTH;
        hDimVCache = V_LENGTH;
    } else {
        OP_LOGE(context_->GetNodeName(), "Unsupported hDim size configuration.");
        return ge::GRAPH_FAILED;
    }

    // Validate cache shapes
    if (currentCacheMode_ == CacheMode::Norm) {
        OP_CHECK_IF(!CheckKCacheValid(context_, batchSize, numHead, seqLength, hDimKCache),
                    OP_LOGE(context_->GetNodeName(), "k_cache shape invalid."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!CheckVCacheValid(context_, batchSize, numHead, seqLength, hDimVCache),
                    OP_LOGE(context_->GetNodeName(), "ckv_cache shape invalid."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(!CheckCacheValidPA(context_, hDimKCache, K_CACHE_INDEX, "k_cache"),
                    OP_LOGE(context_->GetNodeName(), "k_cache shape invalid."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!CheckCacheValidPA(context_, hDimVCache, V_CACHE_INDEX, "v_cache"),
                    OP_LOGE(context_->GetNodeName(), "ckv_cache shape invalid."), return ge::GRAPH_FAILED);
    }

    tilingData_.set_batchSize(batchSize);
    tilingData_.set_numHead(numHead);
    tilingData_.set_seqLength(seqLength);
    tilingData_.set_cacheLength(cacheLength_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_methodMode(methodMode_);

    if (isOutputKv_) {
        tilingData_.set_isOutputKv(1);
    } else {
        tilingData_.set_isOutputKv(0);
    }
    // Extract quant types for k_cache and v_cache
    OP_CHECK_IF(!validateQuantCrossTermShapes(),
                OP_LOGE(context_->GetNodeName(), "Mismatched dimNums between quant terms of caches!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((ResolveQuantComponents() == ge::GRAPH_FAILED),
                OP_LOGE(context_->GetNodeName(), "Invalid cache quant inputs configuration, check shapes!"),
                return ge::GRAPH_FAILED);

    // validate datatype combination of caches according to component quant mode
    OP_CHECK_IF(!validateCacheDatatypes(),
                OP_LOGE(context_->GetNodeName(), "Invalid cache datatypes for current quant configurations!"),
                return ge::GRAPH_FAILED);

    if ((dk_ != ROPE_LENGTH)) {
        auto cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
        std::string reasonMsg = "The D-dimension of input cos must be equal to " + std::to_string(ROPE_LENGTH) +
                                ", where D is the last axis of cos";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((dv_ != RMS_NORM_LENGTH)) {
        auto gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
        std::string reasonMsg = "The 0th axis of input gamma must be equal to " + std::to_string(RMS_NORM_LENGTH);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gamma", ToString(gammaShape).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (methodMode_ == 0) {
        OP_CHECK_IF((currentCacheMode_ == CacheMode::Norm) && (quantMode_ == QUANT_MODE),
                    OP_LOGE(context_->GetNodeName(), "CacheMode::Norm do not support quant!"), return ge::GRAPH_FAILED);

        if ((kv_ != D_LENGTH)) {
            auto kvShape = context_->GetInputShape(KV_INDEX)->GetStorageShape();
            std::string reasonMsg = "The D-dimension of input kv must be equal to " + std::to_string(D_LENGTH) +
                                    ", where D is the last axis of kv";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kv", ToString(kvShape).c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if ((vlen_ != V_LENGTH)) {
            auto vShape = context_->GetOptionalInputShape(V_IDX)->GetStorageShape();
            std::string reasonMsg = "The D-dimension of input v must be equal to " + std::to_string(V_LENGTH) +
                                    ", where D is the last axis of v";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "v", ToString(vShape).c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF((quantMode_ != NON_QUANT_MODE && quantMode_ != QUANT_MODE),
                    OP_LOGE(context_->GetNodeName(), "Only Support QUANT or NON_QUANT."), return ge::GRAPH_FAILED);
    }

    if (currentCacheMode_ == CacheMode::PA && quantMode_ != NON_QUANT_MODE) {
        DoOpTilingPaBlkNz();
        tilingKey_ = TLING_KEY_5011;
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_BLK_BNSD) {
        DoOpTilingPaBlkNz();
        if (quantMode_ != NON_QUANT_MODE) {
            tilingKey_ = TLING_KEY_5010;
        } else {
            tilingKey_ = TLING_KEY_5000;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_NZ) {
        DoOpTilingPaBlkNz();
        if (quantMode_ != NON_QUANT_MODE) {
            tilingKey_ = TLING_KEY_4011;
        } else {
            tilingKey_ = TLING_KEY_4001;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_BLK_NZ) {
        DoOpTilingPaBlkNz();
        if (quantMode_ != NON_QUANT_MODE) {
            tilingKey_ = TLING_KEY_4010;
        } else {
            tilingKey_ = TLING_KEY_4000;
        }
        return ge::GRAPH_SUCCESS;
    }

    int8_t outputKvValue = tilingData_.get_isOutputKv();
    bool outputKv = outputKvValue == 0 ? false : true;
    if (outputKv && currentCacheMode_ == CacheMode::PA) {
        DoOpTilingPaBlkNz();
        tilingKey_ = TLING_KEY_5001;
        return ge::GRAPH_SUCCESS;
    }

    if (IsB1SD(context_)) {
        int64_t bns = batchSize * numHead * seqLength;
        int64_t blockFactor = (bns + coreNum_ - 1) / coreNum_;
        int64_t numBlocks = (bns + blockFactor - 1) / blockFactor;
        tilingData_.set_blockFactor(blockFactor);
        tilingData_.set_numBlocks(numBlocks);

        int64_t maxUbFactor = (methodMode_ == 1) ? 32 : 16;
        constexpr static int64_t needUbSize = static_cast<int64_t>(170) * static_cast<int64_t>(1024);
        if (static_cast<int64_t>(ubSize_) >= static_cast<int64_t>(needUbSize)) {
            tilingData_.set_ubFactor(maxUbFactor);
        } else {
            tilingData_.set_ubFactor(1);
        }
    } else {
        if (batchSize % BATCHES_FOR_EACH_CORE == 0) {
            tilingData_.set_numBlocks(batchSize / BATCHES_FOR_EACH_CORE);
            tilingData_.set_rowsPerBlock(BATCHES_FOR_EACH_CORE);
        } else {
            tilingData_.set_numBlocks(batchSize);
            tilingData_.set_rowsPerBlock(1);
        }
        // Check numBlocks <= MAX_BLOCK_DIM
        OP_CHECK_IF(tilingData_.get_numBlocks() > MAX_BLOCK_DIM,
                    OP_LOGE(context_->GetNodeName(), "numBlocks must be smaller than 65535."), return ge::GRAPH_FAILED);
    }

    if (methodMode_ == 1) {
        if (!isPagedAttention_ && quantMode_ == NON_QUANT_MODE) {
            tilingData_.set_isOutputKv(0);
        }
        if (IsB1SD(context_)) {
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_3001;
            } else if (!isPagedAttention_ && quantMode_ == NON_QUANT_MODE) {
                tilingKey_ = TLING_KEY_3000;
            } else {
                tilingKey_ = TLING_KEY_3010;
            }
        } else {
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_2000;
            } else if (!isPagedAttention_ && quantMode_ == NON_QUANT_MODE) {
                tilingKey_ = TLING_KEY_1000;
            } else if (!isPagedAttention_ && quantMode_ > NON_QUANT_MODE) {
                tilingKey_ = TLING_KEY_1010;
            }
            if (isMTP_) {
                tilingKey_ += 1;
            }
        }
    } else {
        if (IsB1SD(context_)) {
            if (!isPagedAttention_ && quantMode_ == NON_QUANT_MODE) {
                tilingData_.set_isOutputKv(0);
            }
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_3001;
            } else {
                tilingKey_ = TLING_KEY_3000;
            }
        } else {
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_2000;
            } else {
                tilingKey_ = TLING_KEY_1000;
            }
            if (isMTP_) {
                tilingKey_ += 1;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvRmsNormRopeCacheTilingDs::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_.get_numBlocks());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(KvRmsNormRopeCache, KvRmsNormRopeCacheTilingDs, TEMPLATE_DS_PRIORITY);
} // namespace optiling
