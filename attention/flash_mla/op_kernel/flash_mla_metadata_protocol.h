/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_METADATA_H
#define FLASH_MLA_METADATA_H

#include "flash_mla_schedule_config.h"
#include "flash_mla_tiling_fields.h"

namespace FlashMla {
constexpr uint32_t kMlaMetadataMagic = 0x34414c4dU; // "MLA4" in little endian
constexpr uint32_t kMlaHeaderBytes = 128;
constexpr uint32_t kMlaNonQuantPayloadBytes = 1600;
constexpr uint32_t kMlaLegacyPayloadBytes = 1784;
constexpr uint32_t kMlaMaxPayloadBytes = kMlaLegacyPayloadBytes;
// ABI IDs bind the baseline field offsets, field ownership, and scalar widths.
constexpr uint64_t kMlaNonQuantPayloadAbiId = UINT64_C(0xfeb6263e01010640);
constexpr uint64_t kMlaLegacyPayloadAbiId = UINT64_C(0xfeb6263e000206f8);

struct alignas(8) MlaMetadataHeader {
    uint32_t magic;
    MlaStatus status;
    MlaFamily family;
    uint32_t headerBytes;
    uint32_t payloadBytes;
    uint32_t totalBytes;
    uint32_t dynamicFlags;
    uint32_t errorDetail;
    uint64_t payloadAbiId;
    uint64_t platformSignature;
    uint64_t scheduleSignatureLo;
    uint64_t scheduleSignatureHi;
    uint64_t workspaceNeed; // User workspace bytes; system reservation is owned by Host tiling.
    uint32_t originalBlockDim;
    uint32_t cvRatio;
    uint32_t aicNum;
    uint32_t aivNum;
    uint64_t maxQ;
    uint64_t maxKv;
    uint64_t qSeqSquareSum;
    uint32_t reserved[4];
};
struct alignas(8) MlaValuePatch {
    MlaMetadataHeader header;
    uint8_t payload[kMlaMaxPayloadBytes];
};
struct alignas(8) MlaMetadata {
    MlaMetadataHeader header;
    uint8_t payload[kMlaMetadataBytes - kMlaHeaderBytes];
};
struct alignas(8) MlaStaticLaunchInfo {
    MlaFamily family = MlaFamily::NO_SCHEDULE;
    uint32_t physicalBlockDim = 0;
    uint32_t cvRatio = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t queryDataType = 0;
    MlaLayout layoutQ = MlaLayout::BSH;
    MlaLayout layoutKv = MlaLayout::BSH;
    uint32_t pageAttention = 0;
    uint32_t treeMask = 0;
    uint64_t workspaceCap = 0;
    uint64_t sysWorkspaceSize = 0;
    uint64_t payloadAbiId = 0;
    uint64_t platformSignature = 0;
    uint64_t scheduleSignatureLo = 0;
    uint64_t scheduleSignatureHi = 0;
    uint64_t staticLaunchKey = 0;
    uint32_t maskRank = 0;
    uint32_t softmaxLseFlag = 0;
    uint64_t maskDim0 = 0;
    uint64_t maskDim1 = 0;
    uint64_t maskDim2 = 0;
    uint64_t maskBatchStride = 0;
    uint64_t maskBatchStrideAlt = 0;
    uint64_t qStorageSize = 0;
    uint64_t kvStorageSize = 0;
    uint64_t outputElements = 0;
    uint64_t lseElements = 0;
};
struct alignas(8) MlaStaticTiling {
    MlaStaticLaunchInfo launch;
    uint8_t payload[kMlaMaxPayloadBytes];
};
static_assert(sizeof(MlaMetadataHeader) == kMlaHeaderBytes, "metadata header ABI");
static_assert(sizeof(MlaMetadata) == kMlaMetadataBytes, "metadata tensor ABI");
static_assert(offsetof(MlaMetadata, payload) == kMlaHeaderBytes, "payload offset");
static_assert(offsetof(MlaMetadataHeader, magic) == 0, "magic offset");
static_assert(offsetof(MlaMetadataHeader, status) == 4, "status offset");
static_assert(offsetof(MlaMetadataHeader, family) == 8, "family offset");
static_assert(offsetof(MlaMetadataHeader, headerBytes) == 12, "headerBytes offset");
static_assert(offsetof(MlaMetadataHeader, payloadBytes) == 16, "payloadBytes offset");
static_assert(offsetof(MlaMetadataHeader, totalBytes) == 20, "totalBytes offset");
static_assert(offsetof(MlaMetadataHeader, dynamicFlags) == 24, "dynamicFlags offset");
static_assert(offsetof(MlaMetadataHeader, errorDetail) == 28, "errorDetail offset");
static_assert(offsetof(MlaMetadataHeader, payloadAbiId) == 32, "payloadAbiId offset");
static_assert(offsetof(MlaMetadataHeader, platformSignature) == 40, "platformSignature offset");
static_assert(offsetof(MlaMetadataHeader, scheduleSignatureLo) == 48, "scheduleSignatureLo offset");
static_assert(offsetof(MlaMetadataHeader, scheduleSignatureHi) == 56, "scheduleSignatureHi offset");
static_assert(offsetof(MlaMetadataHeader, workspaceNeed) == 64, "workspaceNeed offset");
static_assert(offsetof(MlaMetadataHeader, originalBlockDim) == 72, "originalBlockDim offset");
static_assert(offsetof(MlaMetadataHeader, cvRatio) == 76, "cvRatio offset");
static_assert(offsetof(MlaMetadataHeader, aicNum) == 80, "aicNum offset");
static_assert(offsetof(MlaMetadataHeader, aivNum) == 84, "aivNum offset");
static_assert(offsetof(MlaMetadataHeader, maxQ) == 88, "maxQ offset");
static_assert(offsetof(MlaMetadataHeader, maxKv) == 96, "maxKv offset");
static_assert(offsetof(MlaMetadataHeader, qSeqSquareSum) == 104, "qSeqSquareSum offset");
static_assert(offsetof(MlaMetadataHeader, reserved) == 112, "reserved offset");

__aicore__ inline uint32_t MlaPayloadBytes(MlaFamily family)
{
    return family == MlaFamily::ARCH22_MLA ? kMlaNonQuantPayloadBytes :
                                             (family == MlaFamily::LEGACY_MLA ? kMlaLegacyPayloadBytes : 0);
}
__aicore__ inline uint64_t MlaPayloadAbiId(MlaFamily family)
{
    return family == MlaFamily::ARCH22_MLA ? kMlaNonQuantPayloadAbiId :
                                             (family == MlaFamily::LEGACY_MLA ? kMlaLegacyPayloadAbiId : 0);
}
__aicore__ inline void MlaCopyBytes(void *destination, const void *source, uint32_t bytes)
{
#if !defined(__CCE_AICORE__)
    // All callers copy disjoint POD storage; keep empty copies valid without addresses.
    if (bytes != 0) {
        __builtin_memcpy(destination, source, bytes);
    }
#else
    auto *dst = static_cast<uint8_t *>(destination);
    const auto *src = static_cast<const uint8_t *>(source);
    // Copies are between disjoint POD storage. Fixed-size memcpy preserves
    // unaligned access and aliasing semantics while using word loads/stores on
    // AICore, whose scalar compiler otherwise retains the byte loop.
    uint32_t i = 0;
    for (; bytes - i >= sizeof(uint32_t); i += sizeof(uint32_t)) {
        uint32_t word;
        __builtin_memcpy(&word, src + i, sizeof(word));
        __builtin_memcpy(dst + i, &word, sizeof(word));
    }
    for (; i < bytes; ++i) {
        dst[i] = src[i];
    }
#endif
}
__aicore__ inline void MlaZeroBytes(void *destination, uint32_t bytes)
{
#if !defined(__CCE_AICORE__)
    if (bytes != 0) {
        __builtin_memset(destination, 0, bytes);
    }
#else
    auto *dst = static_cast<uint8_t *>(destination);
    for (uint32_t i = 0; i < bytes; ++i) {
        dst[i] = 0;
    }
#endif
}
__aicore__ inline void InitializeMlaPatch(const MlaScheduleConfig &config, const MlaPlatformInfo &platform,
                                          MlaFamily family, MlaValuePatch &patch)
{
    MlaZeroBytes(&patch, sizeof(patch));
    auto &h = patch.header;
    h.magic = kMlaMetadataMagic;
    h.status = MlaStatus::INVALID;
    h.family = family;
    h.headerBytes = kMlaHeaderBytes;
    h.payloadBytes = MlaPayloadBytes(family);
    h.totalBytes = kMlaMetadataBytes;
    h.payloadAbiId = MlaPayloadAbiId(family);
    h.platformSignature = MlaPlatformSignature(platform);
    auto signature = MlaScheduleSignature(config, platform);
    h.scheduleSignatureLo = signature.lo;
    h.scheduleSignatureHi = signature.hi;
    h.aicNum = platform.aicNum;
    h.aivNum = platform.aivNum;
    h.cvRatio = platform.aicNum == 0 ? 0 : platform.aivNum / platform.aicNum;
}
// Copy the contiguous dynamic regions; all other bytes keep their Host values.
__aicore__ inline void CopyMlaDynamicFields(MlaFamily family, uint8_t *dst, const uint8_t *src)
{
    const auto *ranges = family == MlaFamily::ARCH22_MLA ? kMlaNonQuantDynamicRanges : kMlaLegacyDynamicRanges;
    const uint32_t count =
        family == MlaFamily::ARCH22_MLA ?
            sizeof(kMlaNonQuantDynamicRanges) / sizeof(*ranges) :
            (family == MlaFamily::LEGACY_MLA ? sizeof(kMlaLegacyDynamicRanges) / sizeof(*ranges) : 0);
    for (uint32_t i = 0; i < count; ++i) {
        const auto &range = ranges[i];
        MlaCopyBytes(dst + range.begin, src + range.begin, range.end - range.begin);
    }
}
__aicore__ inline bool MlaPayloadRangeIsZero(const uint8_t *payload, uint32_t begin, uint32_t end)
{
    if (begin >= end) {
        return true;
    }
    uint32_t nonzero = 0;
    // memcpy gives defined unaligned/aliasing behavior for the gap following
    // the one-byte isEmptyBatchOverHalf field, as well as ordinary uint32 gaps.
    for (; end - begin >= sizeof(uint32_t); begin += sizeof(uint32_t)) {
        uint32_t word;
        __builtin_memcpy(&word, payload + begin, sizeof(word));
        nonzero |= word;
    }
    for (; begin < end; ++begin) {
        nonzero |= payload[begin];
    }
    return nonzero == 0;
}
__aicore__ inline bool MlaPayloadIsCanonical(MlaFamily family, const uint8_t *payload, uint32_t bytes)
{
    if (bytes > kMlaMetadataBytes - kMlaHeaderBytes || (bytes != 0 && payload == nullptr)) {
        return false;
    }
    // Check the complement of the copied ranges, including padding and tail.
    const auto *ranges = family == MlaFamily::ARCH22_MLA ? kMlaNonQuantDynamicRanges : kMlaLegacyDynamicRanges;
    const uint32_t count =
        family == MlaFamily::ARCH22_MLA ?
            sizeof(kMlaNonQuantDynamicRanges) / sizeof(*ranges) :
            (family == MlaFamily::LEGACY_MLA ? sizeof(kMlaLegacyDynamicRanges) / sizeof(*ranges) : 0);
    uint32_t previousEnd = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const auto &range = ranges[i];
        if (!MlaPayloadRangeIsZero(payload, previousEnd, bytes < range.begin ? bytes : range.begin)) {
            return false;
        }
        previousEnd = range.end;
    }
    return MlaPayloadRangeIsZero(payload, previousEnd, bytes);
}
#if !defined(__CCE_AICORE__)
// Invalidate as soon as a validated output is bound, before parsing attrs/seq.
inline void InvalidateMlaMetadata(void *output)
{
    auto *metadata = static_cast<MlaMetadata *>(output);
    __atomic_store_n(reinterpret_cast<uint32_t *>(&metadata->header.status), static_cast<uint32_t>(MlaStatus::INVALID),
                     __ATOMIC_RELEASE);
}
inline MlaStatus WriteMlaMetadata(void *output, uint64_t capacity, const MlaValuePatch &patch)
{
    if (output == nullptr || capacity < kMlaMetadataBytes ||
        (reinterpret_cast<uintptr_t>(output) & (kMlaMetadataAlignment - 1U)) != 0) {
        return MlaStatus::CAPACITY_EXCEEDED;
    }
    auto *metadata = static_cast<MlaMetadata *>(output);
    InvalidateMlaMetadata(output);
    MlaZeroBytes(metadata, kMlaMetadataBytes);
    auto header = patch.header;
    header.status = MlaStatus::INVALID;
    MlaCopyBytes(&metadata->header, &header, kMlaHeaderBytes);
    CopyMlaDynamicFields(patch.header.family, metadata->payload, patch.payload);
    // CPU kernel completion on the same stream is the cross-engine publication fence.
    __atomic_store_n(reinterpret_cast<uint32_t *>(&metadata->header.status), static_cast<uint32_t>(patch.header.status),
                     __ATOMIC_RELEASE);
    return patch.header.status;
}
#endif
__aicore__ inline MlaStatus ValidateMlaMetadata(const MlaMetadata &metadata, const MlaStaticLaunchInfo &launch)
{
    const auto &h = metadata.header;
    if (h.status != MlaStatus::SUCCESS) {
        return h.status == MlaStatus::INVALID ? MlaStatus::INVALID_PATCH : h.status;
    }
    if (h.magic != kMlaMetadataMagic || h.headerBytes != kMlaHeaderBytes ||
        h.totalBytes != kMlaMetadataBytes || h.family != launch.family || h.payloadBytes != MlaPayloadBytes(h.family) ||
        h.payloadAbiId != MlaPayloadAbiId(h.family) || h.payloadAbiId != launch.payloadAbiId || h.reserved[0] != 0 ||
        h.reserved[1] != 0 || h.reserved[2] != 0 || h.reserved[3] != 0 || h.errorDetail != 0) {
        return MlaStatus::ABI_MISMATCH;
    }
    if (h.platformSignature != launch.platformSignature || h.scheduleSignatureLo != launch.scheduleSignatureLo ||
        h.scheduleSignatureHi != launch.scheduleSignatureHi || h.aicNum != launch.aicNum || h.aivNum != launch.aivNum ||
        h.cvRatio != launch.cvRatio) {
        return MlaStatus::SIGNATURE_MISMATCH;
    }
    if ((h.dynamicFlags & ~kMlaDynamicFlagsMask) != 0 ||
        !MlaPayloadIsCanonical(h.family, metadata.payload, sizeof(metadata.payload))) {
        return MlaStatus::INVALID_PATCH;
    }
    if (launch.sysWorkspaceSize > launch.workspaceCap ||
        h.workspaceNeed > launch.workspaceCap - launch.sysWorkspaceSize) {
        return MlaStatus::CAPACITY_EXCEEDED;
    }
    return MlaStatus::SUCCESS;
}
__aicore__ inline MlaStatus ApplyMlaValuePatch(const MlaMetadata &metadata, const MlaStaticLaunchInfo &launch,
                                               void *tiling, uint32_t tilingBytes)
{
    const auto status = ValidateMlaMetadata(metadata, launch);
    if (status != MlaStatus::SUCCESS) {
        return status;
    }
    if (tiling == nullptr || tilingBytes != MlaPayloadBytes(launch.family)) {
        return MlaStatus::CAPACITY_EXCEEDED;
    }
    CopyMlaDynamicFields(launch.family, static_cast<uint8_t *>(tiling), metadata.payload);
    return MlaStatus::SUCCESS;
}
} // namespace FlashMla
#endif
