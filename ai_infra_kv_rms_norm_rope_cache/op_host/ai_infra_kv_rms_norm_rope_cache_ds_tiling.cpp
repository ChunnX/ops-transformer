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
 * \file ai_infra_kv_rms_norm_rope_cache_ds_tiling.cpp
 * \brief
 */
#include "ai_infra_kv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
// #include "util/math_util.h"
#include "../../../common/include/tiling_base/tiling_math_util.h"

namespace{
    template <typename T>
    inline T CeilDiv(T num1, T num2)
    {
        if (num2 == 0) {
            return 0;
        }
        return (num1 + num2 - 1) / num2;
    }
    
}

namespace optiling {
constexpr int64_t RMS_NORM_LENGTHS[2] = {512, 192};
int64_t RMS_NORM_LENGTH = 512;
constexpr int64_t ROPE_LENGTH = 64;
constexpr int64_t MAX_BLOCK_DIM = 65535;
constexpr int64_t V_LENGTH = 128;
constexpr int64_t D_LENGTH = 576;
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

bool AiInfraKvRmsNormRopeCacheTilingDs::IsB1SD(const gert::TilingContext* context)
{
    // Tiling strategy decision between 'CutT' and 'CutB'.
    auto kvShapeTuple = GetShapeTuple(context, KV_INDEX);
    int64_t seqLen = std::get<SHAPE_IDX_S>(kvShapeTuple);
    
    bool SAligned = (seqLen == this->ropeSeqLength);

    if (SAligned) {
        // CutT is available as the most effective tiling method.
        // SAligned: Each single batch (B) has a unique rope factor sequence, where 'S_kv = S_rope'.   
        return true;
    } else {
        // No need to reload rope factors upon processing a new batch, where parallel procession among multiple batches with same S index is available.
        
        // MTP definition could be valid ONLY when 'Skv>Srope = 1'.
        this->isMTP_ = (seqLen > 1);
        return false;
    }
    
}

bool AiInfraKvRmsNormRopeCacheTilingDs::CheckScaleValid(const gert::TilingContext* context)
{
    auto scale1Shape = context->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context->GetOptionalInputShape(C_KV_SCALE_IDX);

    bool isValid = true;
    isValid = isValid && ((scale1Shape != nullptr) || (scale2Shape != nullptr));
    if ((scale1Shape != nullptr) && (scale2Shape != nullptr)) {
        isValid = isValid && (scale1Shape->GetStorageShape().GetDimNum() == scale2Shape->GetStorageShape().GetDimNum());
    }
    isValid = isValid && (((scale1Shape != nullptr) && (scale1Shape->GetStorageShape().GetDimNum() <= DIM_TWO)) ||
                          ((scale2Shape != nullptr) && (scale2Shape->GetStorageShape().GetDimNum() <= DIM_TWO)));
    if (!isValid) {
        return false;
    }
    if(methodMode_ == 0){
        if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == ROPE_LENGTH);
        } else if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == DIM_ONE);
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(1) == ROPE_LENGTH);
        }

        if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
        } else if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == DIM_ONE);
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH);
        }
    }
    else {
        if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
        } else if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == tilingData_.get_numHead());
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH);
        }

        if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == V_LENGTH);
        } else if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == tilingData_.get_numHead());
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(1) == V_LENGTH);
        }
    }

    return isValid;
}


bool AiInfraKvRmsNormRopeCacheTilingDs::CheckOffsetValid(const gert::TilingContext* context)
{
    auto offset1Shape = context->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
    auto offset2Shape = context->GetOptionalInputShape(C_KV_OFFSET_IDX);

    bool isValid = true;
    isValid = isValid && ((offset1Shape != nullptr) || (offset2Shape != nullptr));
    if ((offset1Shape != nullptr) && (offset2Shape != nullptr)) {
        isValid = isValid && (offset1Shape->GetStorageShape().GetDimNum() == offset2Shape->GetStorageShape().GetDimNum());
    }
    isValid = isValid && (((offset1Shape != nullptr) && (offset1Shape->GetStorageShape().GetDimNum() <= DIM_TWO)) ||
                          ((offset2Shape != nullptr) && (offset2Shape->GetStorageShape().GetDimNum() <= DIM_TWO)));
    if (!isValid) {
        return false;
    }
    if(methodMode_ == 0){
        if (offset1Shape != nullptr && offset1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(0) == ROPE_LENGTH);
        } else if (offset1Shape != nullptr && offset1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(0) == DIM_ONE);
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(1) == ROPE_LENGTH); 
        }

        if (offset2Shape != nullptr && offset2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
        } else if (offset2Shape != nullptr && offset2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(0) == DIM_ONE);
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH);
        }
    }
    else {
        if (offset1Shape != nullptr && offset1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
        } else if (offset1Shape != nullptr && offset1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(0) == tilingData_.get_numHead());
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH); 
        }

        if (offset2Shape != nullptr && offset2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(0) == V_LENGTH);
        } else if (offset2Shape != nullptr && offset2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(0) == tilingData_.get_numHead());
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(1) == V_LENGTH);
        }
    }
    return isValid;
}

ge::graphStatus AiInfraKvRmsNormRopeCacheTilingDs::ValidateStaticQuantMode(const gert::TilingContext* context){
    #if 0
        if (inputQuantMode_ == QuantClass::None) {
            auto kRopeScaleShape = context_->GetOptionalInputShape(K_ROPE_SCALE_IDX);
            OP_CHECK_IF(kRopeScaleShape != nullptr, OP_LOGE(context_, "k_rope_scale is not None"), return ge::GRAPH_FAILED);

            auto cKvScaleShape = context_->GetOptionalInputShape(C_KV_SCALE_IDX);
            OP_CHECK_IF(cKvScaleShape != nullptr, OP_LOGE(context_, "c_kv_scale is not None"), return ge::GRAPH_FAILED);

            auto kRopeOffsetShape = context_->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
            OP_CHECK_IF(kRopeOffsetShape != nullptr, OP_LOGE(context_, "k_rope_offset is not None"), return ge::GRAPH_FAILED);

            auto cKvOffsetShape = context_->GetOptionalInputShape(C_KV_OFFSET_IDX);
            OP_CHECK_IF(cKvOffsetShape != nullptr, OP_LOGE(context_, "c_kv_offset is not None"), return ge::GRAPH_FAILED);
        }
    #endif
    // Validate Quant mode availability
    if(methodMode_ == 0){
        OP_CHECK_IF(
        (currentCacheMode_ == CacheMode::Norm) && (quantMode_ != NON_QUANT_MODE),
        OP_LOGE(context_->GetNodeName(), "Static Quant is NOT available for CacheMode::Norm!"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
        (this->inputQuantMode_ != QuantClass::None && quantMode_ > QUANT_MODE),
        OP_LOGE(context_->GetNodeName(), "Only Support SYMMETRIC_QUANT or NON_QUANT."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            !(quantMode_ >= NON_QUANT_MODE && quantMode_ <= ASYM_QUANT_MODE),
            OP_LOGE(context_->GetNodeName(), "Only Support SYMMETRIC_QUANT, ASYM_QUANT_MODE or NON_QUANT."), return ge::GRAPH_FAILED);
    }

    // Validate quant tensors
    if (quantMode_ == QUANT_MODE) {
        OP_CHECK_IF(
            !CheckScaleValid(context_), OP_LOGE(context_->GetNodeName(), "Static Quant: scale shape check failed."),
            return ge::GRAPH_FAILED);
    }

    if ((quantMode_ > QUANT_MODE)) {
        OP_CHECK_IF(
            !CheckOffsetValid(context_), OP_LOGE(context_->GetNodeName(), "Static Quant: offset shape check failed."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

bool AiInfraKvRmsNormRopeCacheTilingDs::IsCapable()
{
    return !isRegbase_;
}

ge::graphStatus AiInfraKvRmsNormRopeCacheTilingDs::SetupRotaryMode(const gert::TilingContext* context){
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* tmpRotaryMode = attrs->GetStr(ROTARY_MODE_IDX);
    if (tmpRotaryMode != nullptr) {
        std::string rotaryMode = tmpRotaryMode;
        std::unordered_map<std::string, RotaryMode> rotaryModeMap = {
            {"interleave-half", RotaryMode::INTER_HALF},
            {"half", RotaryMode::HALF}};
        auto getRotaryMode = [&rotaryModeMap](const std::string& mode) -> RotaryMode {
            auto it = rotaryModeMap.find(mode);
            return (it != rotaryModeMap.end()) ? it->second : RotaryMode::INTER_HALF;
        };
        rotaryMode_ = getRotaryMode(rotaryMode);
    } else {
        rotaryMode_ = RotaryMode::INTER_HALF;
    }
    tilingData_.set_rotaryMode(this->rotaryMode_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraKvRmsNormRopeCacheTilingDs::SetupQuantClass(const gert::TilingContext* context){
    // Update genuine static quant mode 
    this->quantMode_ = (this->kStaticQuantMode > this->vStaticQuantMode) ? this->kStaticQuantMode : this->vStaticQuantMode;

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    
    const char* tmpInputQuantMode = attrs->GetStr(INPUT_QUANT_MODE_IDX);
    if (tmpInputQuantMode != nullptr) {
        std::string inputQuantMode = tmpInputQuantMode;
        std::unordered_map<std::string, QuantClass> inputQuantModeMap = {
            {"none", QuantClass::None},
            {"static", QuantClass::Static},
            {"pertile128", QuantClass::Pertile_128}};
        auto getInputQuantMode = [&inputQuantModeMap](const std::string& mode) -> QuantClass {
            auto it = inputQuantModeMap.find(mode);
            return (it != inputQuantModeMap.end()) ? it->second : QuantClass::Static;
        };
        inputQuantMode_ = getInputQuantMode(inputQuantMode);
    } else {
        inputQuantMode_ = QuantClass::None;
    }

    // Validate: when quant_mode="none", scale/offset must be null
    if (inputQuantMode_ == QuantClass::None) {
        auto kRopeScaleShape = context_->GetOptionalInputShape(K_ROPE_SCALE_IDX);
        OP_CHECK_IF(kRopeScaleShape != nullptr, OP_LOGE(context_, "k_rope_scale is not None"), return ge::GRAPH_FAILED);

        auto cKvScaleShape = context_->GetOptionalInputShape(C_KV_SCALE_IDX);
        OP_CHECK_IF(cKvScaleShape != nullptr, OP_LOGE(context_, "c_kv_scale is not None"), return ge::GRAPH_FAILED);

        auto kRopeOffsetShape = context_->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
        OP_CHECK_IF(kRopeOffsetShape != nullptr, OP_LOGE(context_, "k_rope_offset is not None"), return ge::GRAPH_FAILED);

        auto cKvOffsetShape = context_->GetOptionalInputShape(C_KV_OFFSET_IDX);
        OP_CHECK_IF(cKvOffsetShape != nullptr, OP_LOGE(context_, "c_kv_offset is not None"), return ge::GRAPH_FAILED);
    }

    // Quant branch compatibility
    if (inputQuantMode_ == QuantClass::Static && this->quantMode_ == NON_QUANT_MODE){
        // Reduce static quant with no available quant terms into QuantClass::None
        inputQuantMode_ = QuantClass::None;
    }
    #if 0 // Equivalence
        else if (inputQuantMode_ == QuantClass::Static) {
            auto kRopeScaleShape = context_->GetOptionalInputShape(K_ROPE_SCALE_IDX);
            auto cKvScaleShape = context_->GetOptionalInputShape(C_KV_SCALE_IDX);
            auto kRopeOffsetShape = context_->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
            auto cKvOffsetShape = context_->GetOptionalInputShape(C_KV_OFFSET_IDX);

            if (kRopeScaleShape == nullptr && cKvScaleShape == nullptr && kRopeOffsetShape == nullptr && cKvOffsetShape == nullptr) {
                inputQuantMode_ = QuantClass::None;
            }
        }
    #endif
    
    // Allocate quantmode index
    tilingData_.set_quantClass(this->inputQuantMode_);
    // Static QuantModes in : 0(None), 1(static symmetric), 2(static asymmetric)
    tilingData_.set_isKQuant(this->kStaticQuantMode);
    tilingData_.set_isVQuant(this->vStaticQuantMode);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraKvRmsNormRopeCacheTilingDs::SetupCacheLayout(const gert::TilingContext* context){
    // Setup Default Contiguous Stride
    // A standard contiguous block stride = BlockSize * NumHead * HeadDim
    int64_t numHead = tilingData_.get_numHead();
    auto v_shape_ptr = context_->GetInputShape(V_CACHE_INDEX);
    int64_t defaultVStride = blockSize_ * numHead * dv_;
    if (v_shape_ptr != nullptr) {
        defaultVStride = blockSize_ * numHead * v_shape_ptr->GetOriginShape().GetDim(SHAPE_IDX_D);
    }
    int64_t defaultKStride = blockSize_ * numHead * dk_;
    
    int64_t vStride_ = defaultVStride;
    int64_t kStride_ = 0;
    int64_t cacheLayOut = CacheLayout::CONTIGUOUS;

    // Overwrite Stride and Determine Layout if in PA Mode
    if (isPagedAttention_ && methodMode_ == 0) {
        // Fetch the actual physical stride pointer from the framework
        const gert::Stride* v_stride_ptr = context_->GetInputStride(V_CACHE_INDEX);
        
        // Check if the pointer is valid and has at least 1 dimension
        if (v_stride_ptr != nullptr && v_stride_ptr->GetDimNum() > 0) {
            vStride_ = v_stride_ptr->GetStride(0); // Dim 0 stride is the jump between blocks
        }

        OP_CHECK_IF(vStride_ < defaultVStride, OP_LOGE(context_, "vStride_ is less than defaultVStride"), return ge::GRAPH_FAILED);

        // 2. Check if K cache (ckv_cache) was actually provided
        auto k_shape_ptr = context_->GetInputShape(K_CACHE_INDEX);

        if (k_shape_ptr == nullptr || k_shape_ptr->GetOriginShape().GetDimNum() == 1) {
            // ----------------------------------------
            // K_CACHE is EMPTY -> SFA Mode (Fused K & V)
            // ----------------------------------------
            // Check the quantMode (assuming 0 is non-quantized, adjust if your enum differs)
            if (inputQuantMode_ == QuantClass::None) {
                cacheLayOut = CacheLayout::SFA_NO_QUANT;
            } else if (inputQuantMode_ == QuantClass::Pertile_128) {
                tilingData_.set_pertileSize(128);
                cacheLayOut = CacheLayout::SFA_C8;
            }
        } else {
            // ----------------------------------------
            // K_CACHE is PROVIDED -> SWA Mode (Separate K & V)
            // ----------------------------------------
            const gert::Stride* k_stride_ptr = context_->GetInputStride(K_CACHE_INDEX);
            if (k_stride_ptr != nullptr && k_stride_ptr->GetDimNum() > 0) {
                kStride_ = k_stride_ptr->GetStride(0);
            } else {
                kStride_ = defaultKStride;
            }

            OP_CHECK_IF(kStride_ < defaultKStride, OP_LOGE(context_, "kStride_ is less than defaultKStride"), return ge::GRAPH_FAILED);

            if (vStride_ > defaultVStride || kStride_ > defaultKStride) {
            // Check if stride matches the SFA merged cache size (Dk + Dv)
                cacheLayOut = CacheLayout::SWA;
            }

            // todo check
        }
    }

    tilingData_.set_vStride(vStride_);
    tilingData_.set_kStride(kStride_);
    tilingData_.set_cacheLayOut(cacheLayOut);
    return ge::GRAPH_SUCCESS;
}

void AiInfraKvRmsNormRopeCacheTilingDs::DoOpTilingPaBlkNz()
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

uint64_t AiInfraKvRmsNormRopeCacheTilingDs::TilingKeyAttachQuantStat(uint64_t tilingBase, QuantClass quantClass){
    // Attach quant class index label onto tilingkey
    uint64_t quantClassIdx = static_cast<uint64_t>(quantClass);
    return (tilingBase + quantClassIdx * static_cast<uint64_t>(NUM_DECEM));
}

ge::graphStatus AiInfraKvRmsNormRopeCacheTilingDs::DoOpTiling()
{
    RMS_NORM_LENGTH = RMS_NORM_LENGTHS[methodMode_];
    auto kvShapeTuple = GetShapeTuple(context_, KV_INDEX);
    tilingData_.set_batchSize(std::get<SHAPE_IDX_B>(kvShapeTuple));
    tilingData_.set_numHead(std::get<SHAPE_IDX_N>(kvShapeTuple));
    tilingData_.set_seqLength(std::get<SHAPE_IDX_S>(kvShapeTuple));
    tilingData_.set_cacheLength(cacheLength_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_methodMode(methodMode_);

    // Setup cache layout (stride, SFA/SWA detection)
    auto retLayout = SetupCacheLayout(context_);
    if (retLayout != ge::GRAPH_SUCCESS) {
        return retLayout;
    }

    if (isOutputKv_) {
        tilingData_.set_isOutputKv(1);
    } else {
        tilingData_.set_isOutputKv(0);
    }

    // Validate quant mode and check scale/offset shapes
    auto retValidate = ValidateStaticQuantMode(context_);
    if (retValidate != ge::GRAPH_SUCCESS) {
        return retValidate;
    }
    OP_CHECK_IF(
        (!isRegbase_) && (dk_ != ROPE_LENGTH), OP_LOGE(context_->GetNodeName(), "rope last dim only support 64."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!isRegbase_) && (dv_ != RMS_NORM_LENGTH),
        OP_LOGE(context_->GetNodeName(), "rms_norm last dim only support 512 or 192."), return ge::GRAPH_FAILED);
    if(methodMode_ == 0){
        OP_CHECK_IF(
        (!isRegbase_) && (kv_ != D_LENGTH),
        OP_LOGE(context_->GetNodeName(), "rms_norm last dim only support 576."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (!isRegbase_) && (vlen_ != V_LENGTH),
            OP_LOGE(context_->GetNodeName(), "Tensor v last dim only support 128."), return ge::GRAPH_FAILED);
    }

    if (currentCacheMode_ == CacheMode::PA && inputQuantMode_ != QuantClass::None) {
        DoOpTilingPaBlkNz();
        tilingKey_ = TilingKeyAttachQuantStat(TLING_KEY_5001, this->inputQuantMode_); // TLING_KEY_50X1, X>0
        return ge::GRAPH_SUCCESS;
    }

    // Expansion Quantmodes and Cache Layouts are ONLY available for PA cache for now, therefore other cache modes do not need to adopt 'TilingKeyAttachQuantStat'.
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

    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLength();   // Skv
    int64_t numHead = tilingData_.get_numHead();

    // Decide exact tiling strategy and computation template applied to current input
    this->isB1SD_ = this->IsB1SD(context_);
    // Choose template with the largest throughput, token * head per core, as the tiling strategy. 
    this->isCutT_ = this->isB1SD_;  // When '&& seqLen>1' is removed, deafult tiling strategy for Skv=Srope=1 is set to CutT.
    
    if(!this->isCutT_){
        // This is a temporary lazy design to accommodate MTP tiling defect that DOES NOT support tailcore.
        // Redirect to CutT.
        this->isCutT_ = (BATCHES_FOR_EACH_CORE * coreNum_ < batchSize);
    }

    // Final tiling decision
    if (isCutT_) {
        // B, S axes of kv and rope terms are aligned, therefore 'CutBS' can be reduced to 'CutT', whose procession collapses from B-S loops into token traversal.
        int64_t bns = batchSize * numHead * seqLen;                 // total token-head to process
        int64_t blockFactor = (bns + coreNum_ - 1) / coreNum_;      // token-head per core
        int64_t numBlocks = (bns + blockFactor - 1) / blockFactor;  // tailcore index

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
        // Switch to CutB tiling when S_rope = 1, predict parallel tokens in multiple batches.
        if (batchSize % BATCHES_FOR_EACH_CORE == 0) {
            tilingData_.set_numBlocks(batchSize / BATCHES_FOR_EACH_CORE);
            tilingData_.set_rowsPerBlock(BATCHES_FOR_EACH_CORE);
        } else {
            tilingData_.set_numBlocks(batchSize);
            tilingData_.set_rowsPerBlock(1);
        }
        // Check numBlocks <= MAX_BLOCK_DIM
        OP_CHECK_IF(
            tilingData_.get_numBlocks() > MAX_BLOCK_DIM,
            OP_LOGE(context_->GetNodeName(), "numBlocks must be smaller than 65535."), return ge::GRAPH_FAILED);
    }

    // Redirect quant mode to tilingkey 

    // Check Point: QuantClass shall be matched with sub quantmodes.
    OP_LOGD(context_->GetNodeName(), "TilingKey Decision Start: isPagedAttention_ = %d, inputQuantMode_ = %lld, StaticQuantModes[KV, K, V]=[%lld, %lld, %lld].\n", isPagedAttention_ ? 1 : 0, static_cast<int64_t>(inputQuantMode_), this->quantMode_, this->kStaticQuantMode, this->vStaticQuantMode); 

    if (!isPagedAttention_ && inputQuantMode_== QuantClass::None) {
        // isOutputKv is viable once output layout is different from the input or quant mode enabled.
        tilingData_.set_isOutputKv(0);
    }

    if (methodMode_ == DIM_ONE) {
        if (this->isCutT_) {
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_3001;
            } else {    // !isPagedAttention_
                if (quantMode_ > NON_QUANT_MODE) {
                    tilingKey_ = TLING_KEY_3010;
                } else {
                    // Redirect Expansion quantmodes not supported to NoneQuant
                    tilingKey_ = TLING_KEY_3000;
                } 
            }
        } else {
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_2000;
            } else {    // !isPagedAttention_
                // STP is not adopted to expansion quant modes or cache layouts.
                if (quantMode_ > NON_QUANT_MODE) {
                    tilingKey_ = TLING_KEY_1010;
                } else {
                    // Redirect Expansion quantmodes not supported to NoneQuant
                    tilingKey_ = TLING_KEY_1000;
                }
            }
        }
    } else {
        // Quant modes are not available in V1
        if (this->isCutT_) {
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
        }
    }

    if (!this->isCutT_ && isMTP_) {
        // Shared rope terms among multiple batches and seqLen>1, Multiple Token Prediction is available.
        tilingKey_ += 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraKvRmsNormRopeCacheTilingDs::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_.get_numBlocks());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiInfraKvRmsNormRopeCacheTilingDs::GetShapeAttrsInfo()
{
    auto ret = AiInfraKvRmsNormRopeCacheTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto retRotary = SetupRotaryMode(context_);
    if (retRotary != ge::GRAPH_SUCCESS) {
        return retRotary;
    }

    auto retQuant = SetupQuantClass(context_);
    if (retQuant != ge::GRAPH_SUCCESS) {
        return retQuant;
    }

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AiInfraKvRmsNormRopeCache, AiInfraKvRmsNormRopeCacheTilingDs, TEMPLATE_DS_PRIORITY);
} // namespace optiling