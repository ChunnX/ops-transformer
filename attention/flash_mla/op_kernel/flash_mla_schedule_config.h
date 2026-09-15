/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_SCHEDULE_CONFIG_H
#define FLASH_MLA_SCHEDULE_CONFIG_H

#include <cstdint>
#include <cstddef>

// Host/AICPU also use this header; their C++ compiler has no AICore qualifier.
#if !defined(__CCE_AICORE__) && !defined(__aicore__)
#define __aicore__
#endif

namespace FlashMla {
enum class MlaLayout : uint32_t {
    BSH = 0,
    BSND = 1,
    BNSD = 2,
    TND = 3,
    NZ = 4
};
enum class MlaFamily : uint32_t {
    NO_SCHEDULE = 0,
    ARCH22_MLA = 1,
    LEGACY_MLA = 2
};
enum class MlaStatus : uint32_t {
    INVALID = 0,
    SUCCESS = 1,
    INVALID_ARGUMENT = 2,
    INVALID_SEQ = 3,
    UNSUPPORTED = 4,
    OVERFLOW = 5,
    CAPACITY_EXCEEDED = 6,
    ABI_MISMATCH = 7,
    SIGNATURE_MISMATCH = 8,
    INVALID_PATCH = 9
};
constexpr uint32_t kMlaArch22 = 2201;
constexpr uint32_t kMlaFp16 = 1;
constexpr uint32_t kMlaBf16 = 27;
constexpr uint32_t kMlaInt8 = 2;
constexpr uint32_t kMlaNoQuant = 0;
constexpr uint32_t kMlaInt8FullQuant = 1;
constexpr uint32_t kMlaMaxAicNum = 25;
constexpr uint32_t kMlaMetadataBytes = 1920;
constexpr uint32_t kMlaMetadataElements = kMlaMetadataBytes / sizeof(int32_t);
constexpr uint32_t kMlaMetadataAlignment = 64;
constexpr uint32_t kMlaFlagFd = 1U;
constexpr uint32_t kMlaFlagBalance = 2U;
constexpr uint32_t kMlaDynamicFlagsMask = kMlaFlagFd | kMlaFlagBalance;

// Descriptors only: these structures never contain Host or device pointers.
struct MlaScheduleConfig {
    uint32_t numHeads = 0;
    uint32_t batchSize = 0;
    uint64_t qSeqSize = 0;
    uint64_t kvSeqSize = 0;
    MlaLayout layoutQ = MlaLayout::BSH;
    MlaLayout layoutKv = MlaLayout::BSH;
    uint32_t blockSize = 0;
    uint32_t sparseMode = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    uint32_t hasAttenMask = 0;
    uint32_t queryDataType = kMlaFp16;
    uint32_t seqQPresent = 0;
    uint32_t seqKvPresent = 0;
    uint64_t seqQCount = 0;
    uint64_t seqKvCount = 0;
};
struct MlaPlatformInfo {
    // Legacy platform fields are retained for source compatibility; scheduling uses arch and cores.
    uint32_t socVersion = 0;
    uint32_t architecture = kMlaArch22;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t sysWorkspaceSize = 0;
};
struct MlaSeqView {
    const int64_t *data = nullptr;
    uint64_t count = 0;
};
struct MlaSignature {
    uint64_t lo;
    uint64_t hi;
};

__aicore__ inline bool MlaCheckedAdd(uint64_t a, uint64_t b, uint64_t &out)
{
    if (b > UINT64_MAX - a) {
        return false;
    }
    out = a + b;
    return true;
}
__aicore__ inline bool MlaCheckedMul(uint64_t a, uint64_t b, uint64_t &out)
{
    // c220 has no __multi3 runtime. A division-bound check followed by a*b is
    // folded into a 128-bit multiply by the device compiler, so use 32-bit limbs.
    constexpr uint64_t lowMask = UINT32_MAX;
    const uint64_t aLow = a & lowMask;
    const uint64_t aHigh = a >> 32;
    const uint64_t bLow = b & lowMask;
    const uint64_t bHigh = b >> 32;
    if (aHigh != 0 && bHigh != 0) {
        return false;
    }
    const uint64_t low = aLow * bLow;
    const uint64_t crossA = aHigh * bLow;
    const uint64_t crossB = aLow * bHigh;
    uint64_t high = low >> 32;
    if (crossA > lowMask - high) {
        return false;
    }
    high += crossA;
    if (crossB > lowMask - high) {
        return false;
    }
    high += crossB;
    out = (high << 32) | (low & lowMask);
    return true;
}
// Hash explicit little-endian scalar bytes, never native padding or pointer values.
__aicore__ inline uint64_t MlaHashWord(uint64_t hash, uint64_t word)
{
    for (uint32_t i = 0; i < 8; ++i) {
        hash = (hash ^ (word & 255U)) * UINT64_C(1099511628211);
        word >>= 8;
    }
    return hash;
}
__aicore__ inline uint64_t MlaPlatformSignature(const MlaPlatformInfo &p)
{
    uint64_t h = UINT64_C(14695981039346656037);
    const uint64_t words[] = {p.architecture, p.aicNum, p.aivNum};
    for (auto word : words) {
        h = MlaHashWord(h, word);
    }
    return h;
}
__aicore__ inline uint32_t MlaQuantMode(const MlaScheduleConfig &c)
{
    return c.queryDataType == kMlaInt8 ? kMlaInt8FullQuant : kMlaNoQuant;
}
__aicore__ inline MlaLayout MlaScheduleKvLayout(const MlaScheduleConfig &c)
{
    // Dense KV follows Q. PA scheduling uses absolute lengths independently of
    // the cache's storage layout; the INT8 value tiler requires its NZ marker.
    return c.blockSize == 0 ? c.layoutQ : (MlaQuantMode(c) == kMlaInt8FullQuant ? MlaLayout::NZ : MlaLayout::BSH);
}
__aicore__ inline MlaSignature MlaScheduleSignature(const MlaScheduleConfig &c, const MlaPlatformInfo &p)
{
    MlaSignature h{UINT64_C(14695981039346656037), UINT64_C(7809847782465536322)};
    const uint64_t words[] = {c.numHeads,
                              c.batchSize,
                              c.qSeqSize,
                              c.kvSeqSize,
                              static_cast<uint32_t>(c.layoutQ),
                              static_cast<uint32_t>(MlaScheduleKvLayout(c)),
                              c.blockSize,
                              c.sparseMode,
                              static_cast<uint64_t>(c.preTokens),
                              static_cast<uint64_t>(c.nextTokens),
                              c.hasAttenMask,
                              MlaQuantMode(c),
                              c.seqQPresent,
                              c.seqKvPresent,
                              c.seqQCount,
                              c.seqKvCount,
                              MlaPlatformSignature(p)};
    for (auto word : words) {
        h.lo = MlaHashWord(h.lo, word);
        h.hi = MlaHashWord(h.hi, word ^ UINT64_C(0x9e3779b97f4a7c15));
    }
    return h;
}
#if !defined(__CCE_AICORE__)
inline bool MlaStringEqual(const char *a, const char *b)
{
    if (a == nullptr || b == nullptr) {
        return false;
    }
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}
inline bool ParseMlaLayout(const char *text, MlaLayout &layout)
{
    const char *names[] = {"BSH", "BSND", "BNSD", "TND", "NZ"};
    for (uint32_t i = 0; i < 5; ++i) {
        if (MlaStringEqual(text, names[i])) {
            layout = static_cast<MlaLayout>(i);
            return true;
        }
    }
    return false;
}
#endif
// The supported MLA subset has no padding or antiquant-query special cases.
// Preserve GetPreNextToken's update, clamp, then no-mask override order before
// applying CheckTokens to the effective window, as in the original FIA path.
__aicore__ inline MlaStatus NormalizeMlaScheduleConfig(MlaScheduleConfig &c)
{
    constexpr int64_t maxToken = INT32_MAX;
    if (c.sparseMode == 1) {
        c.preTokens = maxToken;
        c.nextTokens = maxToken;
    } else if (c.sparseMode == 2 || c.sparseMode == 3 || c.sparseMode == 9) {
        c.preTokens = maxToken;
        c.nextTokens = 0;
    }
    c.preTokens = c.preTokens > maxToken ? maxToken : (c.preTokens < -maxToken ? -maxToken : c.preTokens);
    c.nextTokens = c.nextTokens > maxToken ? maxToken : (c.nextTokens < -maxToken ? -maxToken : c.nextTokens);
    if (c.sparseMode == 0 && c.hasAttenMask == 0) {
        c.preTokens = maxToken;
        c.nextTokens = maxToken;
    }
    if ((c.sparseMode == 0 && c.hasAttenMask != 0) || c.sparseMode == 4) {
        // CheckTokens requires pre + next >= 0. The original checker receives
        // these clamped values; checking raw int64 attributes would reject
        // windows that the original parser normalizes to a legal boundary.
        if ((c.preTokens < 0 && c.nextTokens < 0) || (c.nextTokens < 0 && c.nextTokens < -c.preTokens) ||
            (c.preTokens < 0 && c.preTokens < -c.nextTokens)) {
            return MlaStatus::INVALID_ARGUMENT;
        }
    }
    return MlaStatus::SUCCESS;
}
__aicore__ inline MlaFamily SelectMlaFamily(const MlaScheduleConfig &c)
{
    return c.queryDataType == kMlaInt8 ? MlaFamily::LEGACY_MLA : MlaFamily::ARCH22_MLA;
}
// Shared by the metadata producer and consumer. These checks need descriptors
// only; TND per-batch query lengths are checked after reading the sequence values.
__aicore__ inline MlaStatus ValidateMlaSparseConfig(const MlaScheduleConfig &c)
{
    const bool quant = c.queryDataType == kMlaInt8;
    if (c.sparseMode != 0 && c.sparseMode != 3 && c.sparseMode != 9 && (quant || c.sparseMode != 4)) {
        return MlaStatus::INVALID_ARGUMENT;
    }
    if (c.hasAttenMask == 0 && c.sparseMode != 0) {
        return MlaStatus::INVALID_ARGUMENT;
    }
    if (quant && c.layoutQ != MlaLayout::TND && c.qSeqSize != 0 &&
        ((c.qSeqSize == 1 && (c.sparseMode != 0 || c.hasAttenMask != 0)) ||
         (c.qSeqSize > 1 && (c.sparseMode == 0 || c.hasAttenMask == 0)))) {
        return MlaStatus::INVALID_ARGUMENT;
    }
    if (quant && c.layoutQ == MlaLayout::TND && c.sparseMode == 0 && c.hasAttenMask != 0) {
        return MlaStatus::INVALID_ARGUMENT;
    }
    return MlaStatus::SUCCESS;
}
__aicore__ inline uint64_t MakeMlaStaticLaunchKey(MlaFamily family, uint32_t dtype, MlaLayout q, MlaLayout kv, bool pa,
                                                  bool tree, uint32_t cvRatio)
{
    const uint64_t typeCode = dtype == kMlaFp16 ? 0 : (dtype == kMlaBf16 ? 1 : 2);
    if (family == MlaFamily::NO_SCHEDULE) {
        return UINT64_C(0x100000) + (typeCode << 12);
    }
    return UINT64_C(0x100000) + (static_cast<uint64_t>(family) << 16) + (typeCode << 12) +
           (static_cast<uint64_t>(q) << 8) + (static_cast<uint64_t>(kv) << 4) + (static_cast<uint64_t>(pa) << 3) +
           (static_cast<uint64_t>(tree) << 2) + cvRatio - 1;
}
} // namespace FlashMla
#endif
