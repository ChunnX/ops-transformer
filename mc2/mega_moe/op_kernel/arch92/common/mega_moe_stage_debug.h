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
constexpr unsigned long POST_DISPATCH_MARKER = 910001UL;
constexpr unsigned long GMM1_OUTPUT_MARKER = 910002UL;
constexpr unsigned long GMM2_INPUT_MARKER = 910003UL;
constexpr unsigned long GMM2_OUTPUT_MARKER = 910004UL;
constexpr unsigned long GMM1_WEIGHT_GATE_MARKER = 910005UL;
constexpr unsigned long GMM1_WEIGHT_UP_MARKER = 910006UL;
constexpr unsigned long GMM1_CONFIG_MARKER = 910007UL;
constexpr unsigned long GMM2_WEIGHT_MARKER = 910008UL;
constexpr unsigned long GMM2_CONFIG_MARKER = 910009UL;
constexpr unsigned long UINT_BIAS = 1000000UL;
constexpr uint32_t DATA_SAMPLE_COUNT = 20U;
constexpr uint32_t SCALE_SAMPLE_COUNT = 8U;
constexpr uint32_t FP8_E4M3_FORMAT = 4U;
constexpr uint32_t FP8_E5M2_FORMAT = 5U;
constexpr uint32_t BF16_FORMAT = 0U;
constexpr uint32_t CONFIG_FORMAT = 6U;
constexpr uint32_t WEIGHT_LAYOUT_DN = 0U;
constexpr uint32_t WEIGHT_LAYOUT_ZN = 1U;
constexpr uint32_t FNV1A_OFFSET_BASIS = 2166136261U;
constexpr uint32_t FNV1A_PRIME = 16777619U;

struct MmadConfigRecord {
    uint32_t problemM;
    uint32_t problemN;
    uint32_t problemK;
    uint32_t scaleK;
    uint32_t actualM;
    uint32_t actualN;
    uint32_t actualK;
    uint32_t tileM;
    uint32_t tileN;
    uint64_t kAL1;
    uint64_t kBL1;
    uint64_t scaleKL1;
    uint32_t weightLayout;
    uint32_t mLoc;
    uint32_t nLoc;
    uint32_t kLoc;
};

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

__aicore__ inline void LogUint(uint64_t value)
{
    set_cond(UINT_BIAS + static_cast<unsigned long>(value));
}

template <typename Fp8Type>
__aicore__ inline uint32_t GetFp8Format()
{
    static_assert(Std::IsSame<Fp8Type, fp8_e4m3fn_t>::value || Std::IsSame<Fp8Type, fp8_e5m2_t>::value,
                  "stage debug only supports fp8_e4m3fn_t/fp8_e5m2_t");
    return Std::IsSame<Fp8Type, fp8_e4m3fn_t>::value ? FP8_E4M3_FORMAT : FP8_E5M2_FORMAT;
}

__aicore__ inline void LogRawBytes(GM_ADDR address, uint32_t count)
{
    __gm__ int32_t *words = reinterpret_cast<__gm__ int32_t *>(address);
    for (uint32_t byteOffset = 0U; byteOffset < count; byteOffset += sizeof(int32_t)) {
        uint32_t packed = static_cast<uint32_t>(
            AscendC::ReadGmByPassDCache(words + byteOffset / sizeof(int32_t)));
        uint32_t bytesInWord = count - byteOffset < sizeof(int32_t) ? count - byteOffset : sizeof(int32_t);
        for (uint32_t byteIdx = 0U; byteIdx < bytesInWord; ++byteIdx) {
            set_cond(static_cast<unsigned long>((packed >> (byteIdx * 8U)) & 0xFFU));
        }
    }
}

__aicore__ inline uint32_t HashRawBytes(GM_ADDR address, uint64_t count)
{
    __gm__ int32_t *words = reinterpret_cast<__gm__ int32_t *>(address);
    uint32_t hash = FNV1A_OFFSET_BASIS;
    for (uint64_t byteOffset = 0U; byteOffset < count; byteOffset += sizeof(int32_t)) {
        uint32_t packed = static_cast<uint32_t>(
            AscendC::ReadGmByPassDCache(words + byteOffset / sizeof(int32_t)));
        uint32_t bytesInWord = count - byteOffset < sizeof(int32_t) ?
                                   static_cast<uint32_t>(count - byteOffset) :
                                   static_cast<uint32_t>(sizeof(int32_t));
        for (uint32_t byteIdx = 0U; byteIdx < bytesInWord; ++byteIdx) {
            hash ^= (packed >> (byteIdx * 8U)) & 0xFFU;
            hash *= FNV1A_PRIME;
        }
    }
    return hash;
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

template <typename Fp8Type>
__aicore__ inline void LogFp8AndScale(unsigned long marker, GM_ADDR dataAddr, GM_ADDR scaleAddr,
                                      uint32_t rankIdx, uint32_t expertIdx, uint32_t rowIdx)
{
    LogHeader(marker, GetFp8Format<Fp8Type>(), rankIdx, expertIdx, rowIdx);
    LogRawBytes(dataAddr, DATA_SAMPLE_COUNT);
    LogRawBytes(scaleAddr, SCALE_SAMPLE_COUNT);
}

template <typename Fp8Type>
__aicore__ inline void LogFp8AndScaleWithHash(unsigned long marker, GM_ADDR dataAddr,
                                              GM_ADDR scaleAddr, uint64_t dataByteCount,
                                              uint64_t scaleByteCount, uint32_t rankIdx,
                                              uint32_t expertIdx, uint32_t rowIdx)
{
    LogFp8AndScale<Fp8Type>(marker, dataAddr, scaleAddr, rankIdx, expertIdx, rowIdx);
    LogUint(dataByteCount);
    LogUint(scaleByteCount);
    set_cond(static_cast<unsigned long>(HashRawBytes(dataAddr, dataByteCount)));
    set_cond(static_cast<unsigned long>(HashRawBytes(scaleAddr, scaleByteCount)));
}

template <typename Fp8Type, typename TensorData, typename TensorScale>
__aicore__ inline void LogFp8TensorAndScaleWithHash(
    unsigned long marker, const TensorData &dataTensor, const TensorScale &scaleTensor,
    uint64_t dataByteCount, uint64_t scaleByteCount, uint32_t rankIdx,
    uint32_t expertIdx, uint32_t rowIdx)
{
    LogFp8AndScaleWithHash<Fp8Type>(
        marker, reinterpret_cast<GM_ADDR>(dataTensor.Data().Get()),
        reinterpret_cast<GM_ADDR>(scaleTensor.Data().Get()), dataByteCount,
        scaleByteCount, rankIdx, expertIdx, rowIdx);
}

__aicore__ inline void LogMmadConfig(unsigned long marker, const MmadConfigRecord &record,
                                     uint32_t rankIdx, uint32_t expertIdx, uint32_t rowIdx)
{
    LogHeader(marker, CONFIG_FORMAT, rankIdx, expertIdx, rowIdx);
    LogUint(record.problemM);
    LogUint(record.problemN);
    LogUint(record.problemK);
    LogUint(record.scaleK);
    LogUint(record.actualM);
    LogUint(record.actualN);
    LogUint(record.actualK);
    LogUint(record.tileM);
    LogUint(record.tileN);
    LogUint(record.kAL1);
    LogUint(record.kBL1);
    LogUint(record.scaleKL1);
    LogUint(record.weightLayout);
    LogUint(record.mLoc);
    LogUint(record.nLoc);
    LogUint(record.kLoc);
}

__aicore__ inline void LogBf16Pair(unsigned long marker, GM_ADDR dataAddr, uint32_t secondHalfOffset,
                                   uint32_t rankIdx, uint32_t expertIdx, uint32_t rowIdx)
{
    LogHeader(marker, BF16_FORMAT, rankIdx, expertIdx, rowIdx);
    LogRawBf16(dataAddr, 0U, DATA_SAMPLE_COUNT);
    LogRawBf16(dataAddr, secondHalfOffset, DATA_SAMPLE_COUNT);
}

__aicore__ inline void LogBf16(unsigned long marker, GM_ADDR dataAddr, uint32_t rankIdx,
                               uint32_t expertIdx, uint32_t rowIdx)
{
    LogHeader(marker, BF16_FORMAT, rankIdx, expertIdx, rowIdx);
    LogRawBf16(dataAddr, 0U, DATA_SAMPLE_COUNT);
}

} // namespace StageDebug
} // namespace MegaMoeImpl

#endif // MEGA_MOE_ARCH92_STAGE_DEBUG_H
