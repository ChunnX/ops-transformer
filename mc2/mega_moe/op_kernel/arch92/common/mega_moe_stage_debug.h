/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH92_STAGE_DEBUG_H
#define MEGA_MOE_ARCH92_STAGE_DEBUG_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

using namespace AscendC;

namespace StageDebug {

constexpr unsigned long PRE_DISPATCH_MARKER = 910000UL;
constexpr unsigned long DISPATCH_OUTPUT_MARKER = 910001UL;
constexpr unsigned long GMM1_OUTPUT_MARKER = 910002UL;
constexpr unsigned long SWIGLU_OUTPUT_MARKER = 910003UL;
constexpr unsigned long GMM2_OUTPUT_MARKER = 910004UL;
constexpr unsigned long COMBINE_OUTPUT_MARKER = 910005UL;
constexpr unsigned long SWIGLU_QUANT_OUTPUT_MARKER = 910006UL;
constexpr unsigned long GMM2_INPUT_MARKER = 910007UL;
constexpr unsigned long GMM2_CONFIG_MARKER = 910012UL;
constexpr unsigned long UINT_BIAS = 1000000UL;
constexpr uint32_t DATA_SAMPLE_COUNT = 20U;
constexpr uint32_t SCALE_SAMPLE_COUNT = 8U;
constexpr uint32_t BF16_FORMAT = 0U;
constexpr uint32_t BF16_PAIR_FORMAT = 1U;
constexpr uint32_t FP8_E4M3_FORMAT = 4U;
constexpr uint32_t FP8_E5M2_FORMAT = 5U;
constexpr uint32_t CONFIG_FORMAT = 6U;

__aicore__ inline void LogHeader(unsigned long marker, uint32_t format, uint32_t rankIdx,
                                 uint32_t expertIdx, uint32_t rowIdx)
{
    set_cond(marker);
    set_cond(UINT_BIAS + static_cast<unsigned long>(format));
    set_cond(UINT_BIAS + static_cast<unsigned long>(AscendC::GetBlockIdx()));
    set_cond(UINT_BIAS + static_cast<unsigned long>(rankIdx));
    set_cond(UINT_BIAS + static_cast<unsigned long>(expertIdx));
    set_cond(UINT_BIAS + static_cast<unsigned long>(rowIdx));
}

__aicore__ inline void LogGmm2Config(
    uint32_t problemM, uint32_t problemN, uint32_t problemK, uint32_t scaleK,
    uint32_t actualM, uint32_t actualN, uint32_t actualK, uint32_t tileM, uint32_t tileN,
    uint64_t kL1, uint64_t scaleKL1, uint32_t weightLayout, uint32_t mLoc, uint32_t nLoc, uint32_t kLoc,
    uint32_t rankIdx, uint32_t expertIdx)
{
    LogHeader(GMM2_CONFIG_MARKER, CONFIG_FORMAT, rankIdx, expertIdx, 0U);
    set_cond(UINT_BIAS + static_cast<unsigned long>(problemM));
    set_cond(UINT_BIAS + static_cast<unsigned long>(problemN));
    set_cond(UINT_BIAS + static_cast<unsigned long>(problemK));
    set_cond(UINT_BIAS + static_cast<unsigned long>(scaleK));
    set_cond(UINT_BIAS + static_cast<unsigned long>(actualM));
    set_cond(UINT_BIAS + static_cast<unsigned long>(actualN));
    set_cond(UINT_BIAS + static_cast<unsigned long>(actualK));
    set_cond(UINT_BIAS + static_cast<unsigned long>(tileM));
    set_cond(UINT_BIAS + static_cast<unsigned long>(tileN));
    set_cond(UINT_BIAS + static_cast<unsigned long>(kL1));
    set_cond(UINT_BIAS + static_cast<unsigned long>(kL1));
    set_cond(UINT_BIAS + static_cast<unsigned long>(scaleKL1));
    set_cond(UINT_BIAS + static_cast<unsigned long>(weightLayout));
    set_cond(UINT_BIAS + static_cast<unsigned long>(mLoc));
    set_cond(UINT_BIAS + static_cast<unsigned long>(nLoc));
    set_cond(UINT_BIAS + static_cast<unsigned long>(kLoc));
}
__aicore__ inline void LogRawBf16(GM_ADDR address, uint32_t elementOffset, uint32_t count)
{
    GM_ADDR elementAddress = address + static_cast<uint64_t>(elementOffset) * sizeof(uint16_t);
    __gm__ int32_t *words = reinterpret_cast<__gm__ int32_t *>(elementAddress);
    for (uint32_t elementIdx = 0U; elementIdx < count; elementIdx += 2U) {
        uint32_t packed = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(words + elementIdx / 2U));
        set_cond(static_cast<unsigned long>(packed & 0xFFFFU));
        if (elementIdx + 1U < count) {
            set_cond(static_cast<unsigned long>((packed >> 16U) & 0xFFFFU));
        }
    }
}

__aicore__ inline void LogRawBytes(GM_ADDR address, uint32_t count)
{
    __gm__ int32_t *words = reinterpret_cast<__gm__ int32_t *>(address);
    for (uint32_t elementIdx = 0U; elementIdx < count; elementIdx += sizeof(uint32_t)) {
        uint32_t packed = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(words + elementIdx / sizeof(uint32_t)));
        uint32_t remaining = count - elementIdx;
        uint32_t validBytes = remaining < sizeof(uint32_t) ? remaining : sizeof(uint32_t);
        for (uint32_t byteIdx = 0U; byteIdx < validBytes; ++byteIdx) {
            set_cond(static_cast<unsigned long>((packed >> (byteIdx * 8U)) & 0xFFU));
        }
    }
}

__aicore__ inline void LogBf16(unsigned long marker, GM_ADDR dataAddr, uint32_t rankIdx,
                               uint32_t expertIdx, uint32_t rowIdx)
{
    LogHeader(marker, BF16_FORMAT, rankIdx, expertIdx, rowIdx);
    LogRawBf16(dataAddr, 0U, DATA_SAMPLE_COUNT);
}

__aicore__ inline void LogBf16Pair(unsigned long marker, GM_ADDR firstAddr, GM_ADDR secondAddr,
                                   uint32_t rankIdx, uint32_t expertIdx, uint32_t rowIdx)
{
    LogHeader(marker, BF16_PAIR_FORMAT, rankIdx, expertIdx, rowIdx);
    LogRawBf16(firstAddr, 0U, DATA_SAMPLE_COUNT);
    LogRawBf16(secondAddr, 0U, DATA_SAMPLE_COUNT);
}

template <typename DataType>
__aicore__ inline void LogFp8Scale(unsigned long marker, GM_ADDR dataAddr, GM_ADDR scaleAddr,
                                   uint32_t rankIdx, uint32_t expertIdx, uint32_t rowIdx)
{
    constexpr uint32_t fp8Format =
        AscendC::IsSameType<DataType, fp8_e5m2_t>::value ? FP8_E5M2_FORMAT : FP8_E4M3_FORMAT;
    LogHeader(marker, fp8Format, rankIdx, expertIdx, rowIdx);
    LogRawBytes(dataAddr, DATA_SAMPLE_COUNT);
    LogRawBytes(scaleAddr, SCALE_SAMPLE_COUNT);
}

template <typename Tensor>
__aicore__ inline void LogLocalBf16(unsigned long marker, Tensor data, uint32_t rankIdx,
                                    uint32_t expertIdx, uint32_t rowIdx)
{
    LogHeader(marker, BF16_FORMAT, rankIdx, expertIdx, rowIdx);
    auto raw = data.template ReinterpretCast<uint16_t>();
    for (uint32_t elementIdx = 0U; elementIdx < DATA_SAMPLE_COUNT; ++elementIdx) {
        set_cond(static_cast<unsigned long>(raw.GetValue(elementIdx)));
    }
}

template <typename DataType, typename DataTensor, typename ScaleTensor>
__aicore__ inline void LogLocalFp8Scale(unsigned long marker, DataTensor data, ScaleTensor scale,
                                        uint32_t rankIdx, uint32_t expertIdx, uint32_t rowIdx)
{
    constexpr uint32_t fp8Format =
        AscendC::IsSameType<DataType, fp8_e5m2_t>::value ? FP8_E5M2_FORMAT : FP8_E4M3_FORMAT;
    LogHeader(marker, fp8Format, rankIdx, expertIdx, rowIdx);
    auto rawData = data.template ReinterpretCast<uint8_t>();
    auto rawScale = scale.template ReinterpretCast<uint8_t>();
    for (uint32_t elementIdx = 0U; elementIdx < DATA_SAMPLE_COUNT; ++elementIdx) {
        set_cond(static_cast<unsigned long>(rawData.GetValue(elementIdx)));
    }
    for (uint32_t scaleIdx = 0U; scaleIdx < SCALE_SAMPLE_COUNT; ++scaleIdx) {
        set_cond(static_cast<unsigned long>(rawScale.GetValue(scaleIdx)));
    }
}

} // namespace StageDebug
} // namespace MegaMoeImpl

#endif // MEGA_MOE_ARCH92_STAGE_DEBUG_H
