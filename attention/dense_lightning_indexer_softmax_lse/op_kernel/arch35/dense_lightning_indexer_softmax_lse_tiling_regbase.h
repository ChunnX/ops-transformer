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
 * \file dense_lightning_indexer_softmax_lse_tiling_regbase.h
 * \brief Arch35 tiling data definitions
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_REGBASE_TILING_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_REGBASE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {

const uint32_t DENSE_LI_SOFTMAX_LSE_MAX_AIC_CORE_NUM_REGBASE = 36;
constexpr uint32_t MAX_CORE_NUM_REGBASE = 36;

class DenseLISoftmaxLseBaseParamsRegbase {
public:
    int32_t bSize;
    int32_t n2Size;
    int32_t gSize;
    int32_t s1Size;
    int32_t s2Size;
    int32_t dSize;
    int32_t kSize;
    int32_t sparseMode;
    int32_t s1BaseSize;
    int32_t s2BaseSize;
    int32_t mBaseSize;
    int64_t preTokens;
    int64_t nextTokens;
    float scaleValue;
    uint8_t layoutType;

    int32_t get_bSize() const {return bSize;}
    void set_bSize(int32_t bSizeParam) {this->bSize = bSizeParam;}

    int32_t get_n2Size() const {return n2Size;}
    void set_n2Size(int32_t n2SizeParam) {this->n2Size = n2SizeParam;}

    int32_t get_gSize() const {return gSize;}
    void set_gSize(int32_t gSizeParam) {this->gSize = gSizeParam;}

    int32_t get_s1Size() const {return s1Size;}
    void set_s1Size(int32_t s1SizeParam) {this->s1Size = s1SizeParam;}

    int32_t get_s2Size() const {return s2Size;}
    void set_s2Size(int32_t s2SizeParam) {this->s2Size = s2SizeParam;}

    int32_t get_dSize() const {return dSize;}
    void set_dSize(int32_t dSizeParam) {this->dSize = dSizeParam;}

    int32_t get_kSize() const {return kSize;}
    void set_kSize(int32_t kSizeParam) {this->kSize = kSizeParam;}

    int32_t get_sparseMode() const {return sparseMode;}
    void set_sparseMode(int32_t sparseModeParam) {this->sparseMode = sparseModeParam;}

    int32_t get_s1BaseSize() const {return s1BaseSize;}
    void set_s1BaseSize(int32_t s1BaseSizeParam) {this->s1BaseSize = s1BaseSizeParam;}

    int32_t get_s2BaseSize() const {return s2BaseSize;}
    void set_s2BaseSize(int32_t s2BaseSizeParam) {this->s2BaseSize = s2BaseSizeParam;}

    int32_t get_mBaseSize() const {return mBaseSize;}
    void set_mBaseSize(int32_t mBaseSizeParam) {this->mBaseSize = mBaseSizeParam;}

    int64_t get_preTokens() const {return preTokens;}
    void set_preTokens(int64_t preTokensParam) {this->preTokens = preTokensParam;}

    int64_t get_nextTokens() const {return nextTokens;}
    void set_nextTokens(int64_t nextTokensParam) {this->nextTokens = nextTokensParam;}

    float get_scaleValue() const {return scaleValue;}
    void set_scaleValue(float scaleValueParam) {this->scaleValue = scaleValueParam;}

    uint8_t get_layoutType() const {return layoutType;}
    void set_layoutType(uint8_t layoutTypeParam) {this->layoutType = layoutTypeParam;}
};

class DenseLISoftmaxLseMultiCoreParamsRegbase {
public:
    uint32_t coreNum;
    uint32_t usedCoreNum;
    int64_t splitFactorSize;
    int64_t totalSize;
    int64_t bS1Index[MAX_CORE_NUM_REGBASE];

    int32_t get_coreNum() const {return coreNum;}
    void set_coreNum(uint32_t coreNumParam) {this->coreNum = coreNumParam;}

    int32_t get_usedCoreNum() const {return usedCoreNum;}
    void set_usedCoreNum(uint32_t usedCoreNumParam) {this->usedCoreNum = usedCoreNumParam;}

    int64_t get_splitFactorSize() const {return splitFactorSize;}
    void set_splitFactorSize(int64_t splitFactorSizeParam) {this->splitFactorSize = splitFactorSizeParam;}

    int64_t get_totalSize() const {return totalSize;}
    void set_totalSize(int64_t totalSizeParam) {this->totalSize = totalSizeParam;}

    int64_t *get_bS1Ptr() {return bS1Index;}
};

class DenseLISoftmaxLseInitOutputParamsRegbase {
public:
    uint32_t singleCoreSize;
    uint32_t rsvd;
    int64_t totalOutputSize;

    uint32_t get_singleCoreSize() const {return singleCoreSize;}
    void set_singleCoreSize(uint32_t singleCoreSizeParam) {this->singleCoreSize = singleCoreSizeParam;}

    int64_t get_totalOutputSize() const {return totalOutputSize;}
    void set_totalOutputSize(int64_t totalOutputSizeParam) {this->totalOutputSize = totalOutputSizeParam;}
};

class DenseLISoftmaxLseVecApiParamsRegbase {
public:
    SoftMaxTiling softmaxTilingData;
};

class DenseLightningIndexerSoftmaxLseRegBaseTilingData {
public:
    DenseLISoftmaxLseBaseParamsRegbase baseParams;
    DenseLISoftmaxLseMultiCoreParamsRegbase multiCoreParams;
    DenseLISoftmaxLseInitOutputParamsRegbase initOutputParams;
    DenseLISoftmaxLseVecApiParamsRegbase vectorParams;
};

} // namespace optiling

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_REGBASE_TILING_H