/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef STEP4_PACKED_REGION_GATHER_ARCH35_HPP
#define STEP4_PACKED_REGION_GATHER_ARCH35_HPP

#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"

namespace NpuArch::Gemm::Block {

static constexpr uint32_t STEP4_GATHER_MAX_SEG = 32;
// One 256-tile GM slot holds two QK L1 halves (mm1L1TileN=128). Cube issues each half
// directly from GM (no PackedSegList spill). Host workspace stays 200 words / 256-tile.
static constexpr uint32_t STEP4_SEG_LIST_WORDS = 200;
static constexpr uint32_t STEP4_SEG_HALF_WORDS = 100;
static constexpr uint32_t STEP4_SEG_LIST_BYTES = STEP4_SEG_LIST_WORDS * sizeof(uint32_t);
static constexpr uint16_t STEP4_LIST_READY_FLAG0 = 11;
static constexpr uint16_t STEP4_LIST_READY_FLAG1 = 12;
static constexpr uint16_t STEP4_LIST_READY_FLAG0_V1 = 27;
static constexpr uint16_t STEP4_LIST_READY_FLAG1_V1 = 28;
static constexpr uint32_t STEP4_SEG_META_NDNUM_MASK = 0xFFu;
static constexpr uint32_t STEP4_PREFILL_PAIR_N = 8;
static constexpr uint32_t STEP4_PREFILL_HDR_TAIL_SHIFT = 16;

static constexpr uint32_t STEP4_PACKED_REGION_ID_MASK = 0x00FFFFFFu;
static constexpr uint32_t STEP4_PACKED_VALID_SHIFT = 24;
static constexpr uint32_t STEP4_PACKED_PREFILL_SKIP = 0xFFFFFFFFu;
static constexpr uint32_t STEP4_DECODE_START_MASK = 0xFFFFFFFFu;

template <bool IS_DECODE, class PackedT>
__aicore__ inline void ParsePackedSlot(PackedT meta, uint32_t regionSize, int32_t kvSeqlen, uint32_t &phys,
                                       uint32_t &valid)
{
    if constexpr (IS_DECODE) {
        uint64_t word = static_cast<uint64_t>(meta);
        uint32_t start = static_cast<uint32_t>(word);
        phys = static_cast<uint32_t>(word >> 32);
        int64_t vis = static_cast<int64_t>(kvSeqlen) - static_cast<int64_t>(start);
        if (vis < 0) {
            vis = 0;
        }
        if (vis > static_cast<int64_t>(regionSize)) {
            vis = static_cast<int64_t>(regionSize);
        }
        valid = static_cast<uint32_t>(vis);
        (void)start;
    } else {
        // Prefill valid sits in bits [31:24]. Skip is only the sentinel 0xFFFFFFFF (-1);
        // do not treat any negative INT32 as skip (high valid bits share the sign bit).
        uint32_t word = static_cast<uint32_t>(meta);
        if (word == STEP4_PACKED_PREFILL_SKIP) {
            phys = 0;
            valid = 0;
            return;
        }
        phys = word & STEP4_PACKED_REGION_ID_MASK;
        valid = word >> STEP4_PACKED_VALID_SHIFT;
    }
}

// TGD GM row of region `phys`'s first token. pageSize is the paged block (blocksize);
// regionSize stays 8. On a cache flattened in region-id order this equals phys * regionSize:
//   (phys / rpp) * pageSize + (phys % rpp) * regionSize, rpp = pageSize / regionSize.
__aicore__ inline uint32_t PackedRegionGmStart(uint32_t phys, uint32_t regionSize, uint32_t pageSize)
{
    if (regionSize == 0) {
        return 0;
    }
    if (pageSize == 0 || (pageSize % regionSize) != 0) {
        return phys * regionSize;
    }
    uint32_t rpp = pageSize / regionSize;
    return (phys / rpp) * pageSize + (phys % rpp) * regionSize;
}

// One packed slot clipped to [gatheredStart, targetEnd). slotRef is the first not-fully-consumed
// slot; compactPosRef is that slot's compact start. A tile that ends mid-slot leaves both so the
// next tile resumes via copyStartInSlot.
template <bool IS_DECODE, class PackedT>
__aicore__ inline uint32_t PackedNextSlot(AscendC::GlobalTensor<PackedT> &gPacked, uint32_t packedRowStart,
                                          uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                          uint32_t gatheredStart, uint32_t targetEnd, int32_t kvSeqlen,
                                          uint32_t &compactPosRef, uint32_t &slotRef, uint32_t &segOriStart)
{
    while (slotRef < packedCount) {
        PackedT meta = gPacked.GetValue(packedRowStart + slotRef);
        uint32_t phys = 0;
        uint32_t valid = 0;
        ParsePackedSlot<IS_DECODE>(meta, regionSize, kvSeqlen, phys, valid);
        if (valid == 0) {
            ++slotRef;
            continue;
        }
        uint32_t slotStart = compactPosRef;
        compactPosRef += valid;
        if (compactPosRef <= gatheredStart) {
            ++slotRef;
            continue;
        }
        if (slotStart >= targetEnd) {
            compactPosRef = slotStart;
            break;
        }
        uint32_t copyStartInSlot = (gatheredStart > slotStart) ? (gatheredStart - slotStart) : 0;
        uint32_t copyEndInSlot = valid;
        uint32_t slotCap = targetEnd - slotStart;
        if (slotCap < copyEndInSlot) {
            copyEndInSlot = slotCap;
        }
        uint32_t segLen = copyEndInSlot - copyStartInSlot;
        if (segLen == 0) {
            compactPosRef = slotStart;
            ++slotRef;
            continue;
        }
        segOriStart = PackedRegionGmStart(phys, regionSize, pageSize) + copyStartInSlot;
        if (copyEndInSlot < valid) {
            compactPosRef = slotStart;
            return segLen;
        }
        ++slotRef;
        return segLen;
    }
    return 0;
}

// Same walk as PackedNextSlot, but packed already lives in UB (mask2idx-style MTE2 burst).
template <bool IS_DECODE, class PackedT>
__aicore__ inline uint32_t PackedNextSlotUb(AscendC::LocalTensor<PackedT> ubPacked, AscendC::LocalTensor<int32_t> ubGm,
                                            bool hasGm, uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                            uint32_t gatheredStart, uint32_t targetEnd, int32_t kvSeqlen,
                                            uint32_t &compactPosRef, uint32_t &slotRef, uint32_t &segOriStart,
                                            uint32_t ubOff)
{
    while (slotRef < packedCount) {
        PackedT meta = ubPacked.GetValue(slotRef + ubOff);
        uint32_t phys = 0;
        uint32_t valid = 0;
        ParsePackedSlot<IS_DECODE>(meta, regionSize, kvSeqlen, phys, valid);
        if (valid == 0) {
            ++slotRef;
            continue;
        }
        uint32_t slotStart = compactPosRef;
        compactPosRef += valid;
        if (compactPosRef <= gatheredStart) {
            ++slotRef;
            continue;
        }
        if (slotStart >= targetEnd) {
            compactPosRef = slotStart;
            break;
        }
        uint32_t copyStartInSlot = (gatheredStart > slotStart) ? (gatheredStart - slotStart) : 0;
        uint32_t copyEndInSlot = valid;
        uint32_t slotCap = targetEnd - slotStart;
        if (slotCap < copyEndInSlot) {
            copyEndInSlot = slotCap;
        }
        uint32_t segLen = copyEndInSlot - copyStartInSlot;
        if (segLen == 0) {
            compactPosRef = slotStart;
            ++slotRef;
            continue;
        }
        uint32_t gmBase = hasGm ? static_cast<uint32_t>(ubGm.GetValue(slotRef + ubOff)) :
                                  PackedRegionGmStart(phys, regionSize, pageSize);
        segOriStart = gmBase + copyStartInSlot;
        if (copyEndInSlot < valid) {
            compactPosRef = slotStart;
            return segLen;
        }
        ++slotRef;
        return segLen;
    }
    return 0;
}

__aicore__ inline uint32_t PackedRegionLog2Pow2(uint32_t value)
{
    uint32_t logv = 0;
    uint32_t tmp = value;
    while (tmp > 1) {
        tmp >>= 1;
        ++logv;
    }
    return logv;
}

// Prefill INT32: phys = word & 0x00FFFFFF, gm = (phys/rpp)*page + (phys%rpp)*region, all shifts when pow2.
__aicore__ inline bool PackedRegionVecPrefillGm(AscendC::LocalTensor<int32_t> ubPacked,
                                                AscendC::LocalTensor<int32_t> ubPhys,
                                                AscendC::LocalTensor<int32_t> ubGm,
                                                AscendC::LocalTensor<int32_t> ubTmp, uint32_t nAlign,
                                                uint32_t regionSize, uint32_t pageSize)
{
    if (regionSize == 0 || pageSize == 0 || (pageSize % regionSize) != 0) {
        return false;
    }
    uint32_t rpp = pageSize / regionSize;
    if (((rpp & (rpp - 1)) != 0) || ((pageSize & (pageSize - 1)) != 0) || ((regionSize & (regionSize - 1)) != 0)) {
        return false;
    }
    uint32_t logRpp = PackedRegionLog2Pow2(rpp);
    uint32_t logPage = PackedRegionLog2Pow2(pageSize);
    uint32_t logRs = PackedRegionLog2Pow2(regionSize);
    AscendC::Duplicate(ubTmp, static_cast<int32_t>(STEP4_PACKED_REGION_ID_MASK), nAlign);
    AscendC::And(ubPhys, ubPacked, ubTmp, nAlign);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(ubGm, ubPhys, static_cast<int32_t>(logRpp), nAlign);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftLeft(ubGm, ubGm, static_cast<int32_t>(logPage), nAlign);
    AscendC::Duplicate(ubTmp, static_cast<int32_t>(rpp - 1), nAlign);
    AscendC::And(ubTmp, ubPhys, ubTmp, nAlign);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftLeft(ubTmp, ubTmp, static_cast<int32_t>(logRs), nAlign);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(ubGm, ubGm, ubTmp, nAlign);
    AscendC::PipeBarrier<PIPE_V>();
    return true;
}

struct PackedSegList {
    uint32_t gm[STEP4_GATHER_MAX_SEG];
    uint32_t dst[STEP4_GATHER_MAX_SEG];
    uint32_t nValue[STEP4_GATHER_MAX_SEG];
    uint32_t ndNum[STEP4_GATHER_MAX_SEG];
    uint64_t srcStride[STEP4_GATHER_MAX_SEG];
    uint32_t nSeg;
};

__aicore__ inline uint32_t PackedSegListTokenCount(PackedSegList const &list)
{
    uint32_t tokens = 0;
    uint32_t n = list.nSeg;
    if (n > STEP4_GATHER_MAX_SEG) {
        n = STEP4_GATHER_MAX_SEG;
    }
    for (uint32_t i = 0; i < n; ++i) {
        tokens += list.nValue[i] * list.ndNum[i];
    }
    return tokens;
}

__aicore__ inline uint32_t PackedSegStoreWords(uint32_t ndNum)
{
    return (ndNum == 2) ? 4u : 2u;
}

__aicore__ inline uint32_t PackedSegPackMeta(uint32_t nValue, uint32_t ndNum)
{
    return (nValue << 8) | (ndNum & STEP4_SEG_META_NDNUM_MASK);
}

__aicore__ inline void PackedSegUnpackMeta(uint32_t meta, uint32_t &nValue, uint32_t &ndNum)
{
    ndNum = meta & STEP4_SEG_META_NDNUM_MASK;
    nValue = meta >> 8;
}

template <class ListBuf>
__aicore__ inline uint32_t PackedSegWriteRecordAt(ListBuf list, uint32_t base, uint32_t rel, uint32_t gm,
                                                  uint32_t nValue, uint32_t ndNum, uint64_t srcStride,
                                                  uint32_t relLimit = STEP4_SEG_LIST_WORDS)
{
    uint32_t need = PackedSegStoreWords(ndNum);
    if (rel + need > relLimit) {
        return rel;
    }
    list.SetValue(base + rel, gm);
    list.SetValue(base + rel + 1, PackedSegPackMeta(nValue, ndNum));
    if (ndNum == 2) {
        list.SetValue(base + rel + 2, static_cast<uint32_t>(srcStride));
        list.SetValue(base + rel + 3, static_cast<uint32_t>(srcStride >> 32));
    }
    return rel + need;
}

__aicore__ inline void StorePackedSegListToUb(AscendC::LocalTensor<uint32_t> ubList, PackedSegList const &list)
{
    uint32_t n = list.nSeg;
    if (n > STEP4_GATHER_MAX_SEG) {
        n = STEP4_GATHER_MAX_SEG;
    }
    uint32_t rel = 1;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t next = PackedSegWriteRecordAt(ubList, 0, rel, list.gm[i], list.nValue[i], list.ndNum[i],
                                              list.srcStride[i]);
        if (next == rel) {
            n = i;
            break;
        }
        rel = next;
    }
    ubList.SetValue(0, n);
}

__aicore__ inline void StorePackedSegList(AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff,
                                          PackedSegList const &list)
{
    uint32_t n = list.nSeg;
    if (n > STEP4_GATHER_MAX_SEG) {
        n = STEP4_GATHER_MAX_SEG;
    }
    uint32_t rel = 1;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t next = PackedSegWriteRecordAt(gList, wordOff, rel, list.gm[i], list.nValue[i], list.ndNum[i],
                                              list.srcStride[i]);
        if (next == rel) {
            n = i;
            break;
        }
        rel = next;
    }
    gList.SetValue(wordOff, n);
}

__aicore__ inline void LoadPackedSegList(AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff, PackedSegList &list)
{
    uint32_t n = gList.GetValue(wordOff);
    if (n > STEP4_GATHER_MAX_SEG) {
        n = STEP4_GATHER_MAX_SEG;
    }
    uint32_t rel = 1;
    uint32_t compactDst = 0;
    uint32_t stored = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (rel + 2u > STEP4_SEG_LIST_WORDS) {
            break;
        }
        list.gm[i] = gList.GetValue(wordOff + rel);
        uint32_t meta = gList.GetValue(wordOff + rel + 1);
        rel += 2;
        PackedSegUnpackMeta(meta, list.nValue[i], list.ndNum[i]);
        if (list.ndNum[i] == 2) {
            if (rel + 2u > STEP4_SEG_LIST_WORDS) {
                break;
            }
            uint32_t lo = gList.GetValue(wordOff + rel);
            uint32_t hi = gList.GetValue(wordOff + rel + 1);
            rel += 2;
            list.srcStride[i] = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
        } else {
            list.ndNum[i] = 1;
            list.srcStride[i] = 0;
        }
        list.dst[i] = compactDst;
        compactDst += list.nValue[i] * list.ndNum[i];
        stored = i + 1;
    }
    list.nSeg = stored;
}

// l1Split keeps Nd2Nz pairs and single segs inside one QK L1 (128). PV issues the same
// list over the full 256-tile, so a split only adds extra segs; it does not change tokens.
template <bool IS_DECODE, bool SUPPORT_PAIR, class PackedT>
__aicore__ inline void PackedRegionDecodeToList(AscendC::GlobalTensor<PackedT> gPacked, uint32_t packedRowStart,
                                                uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                                uint32_t gatheredStart, uint32_t gatheredLen, int32_t kvSeqlen,
                                                uint32_t &compactPos, uint32_t &slot, uint64_t tokenStride,
                                                uint32_t eleNumPerC0, PackedSegList &list, uint32_t l1Split)
{
    list.nSeg = 0;
    if (gatheredLen == 0) {
        return;
    }
    uint32_t tileEnd = gatheredStart + gatheredLen;
    uint32_t dealtLenAccum = 0;
    while (slot < packedCount && dealtLenAccum < gatheredLen && list.nSeg < STEP4_GATHER_MAX_SEG) {
        uint32_t targetEnd = tileEnd;
        if (l1Split > 0) {
            uint32_t nextSplit = ((dealtLenAccum / l1Split) + 1) * l1Split;
            uint32_t splitAbs = gatheredStart + nextSplit;
            if (splitAbs < targetEnd) {
                targetEnd = splitAbs;
            }
        }
        uint32_t oriStart = 0;
        uint32_t segLen = PackedNextSlot<IS_DECODE>(gPacked, packedRowStart, packedCount, regionSize, pageSize,
                                                    gatheredStart, targetEnd, kvSeqlen, compactPos, slot, oriStart);
        if (segLen == 0) {
            break;
        }
        bool storedPair = false;
        if constexpr (SUPPORT_PAIR) {
            uint32_t compactPos2 = compactPos;
            uint32_t slot2 = slot;
            uint32_t oriStart2 = 0;
            uint32_t segLen2 = PackedNextSlot<IS_DECODE>(gPacked, packedRowStart, packedCount, regionSize, pageSize,
                                                         gatheredStart, targetEnd, kvSeqlen, compactPos2, slot2,
                                                         oriStart2);
            if (segLen2 == segLen && oriStart2 != oriStart) {
                uint32_t gmLo = (oriStart < oriStart2) ? oriStart : oriStart2;
                uint32_t gmHi = (oriStart < oriStart2) ? oriStart2 : oriStart;
                uint64_t srcStride = static_cast<uint64_t>(gmHi - gmLo) * tokenStride;
                if (srcStride % eleNumPerC0 == 0 && dealtLenAccum + segLen + segLen2 <= gatheredLen) {
                    uint32_t i = list.nSeg;
                    list.gm[i] = gmLo;
                    list.dst[i] = dealtLenAccum;
                    list.nValue[i] = segLen;
                    list.ndNum[i] = 2;
                    list.srcStride[i] = srcStride;
                    list.nSeg = i + 1;
                    dealtLenAccum += segLen + segLen2;
                    compactPos = compactPos2;
                    slot = slot2;
                    storedPair = true;
                }
            }
        }
        if (!storedPair) {
            uint32_t i = list.nSeg;
            list.gm[i] = oriStart;
            list.dst[i] = dealtLenAccum;
            list.nValue[i] = segLen;
            list.ndNum[i] = 1;
            list.srcStride[i] = 0;
            list.nSeg = i + 1;
            dealtLenAccum += segLen;
        }
    }
}

template <bool IS_DECODE, bool SUPPORT_PAIR, class PackedT>
__aicore__ inline void PackedRegionDecodeToListUb(AscendC::LocalTensor<PackedT> ubPacked,
                                                  AscendC::LocalTensor<int32_t> ubGm, bool hasGm, uint32_t packedCount,
                                                  uint32_t regionSize, uint32_t pageSize, uint32_t gatheredStart,
                                                  uint32_t gatheredLen, int32_t kvSeqlen, uint32_t &compactPos,
                                                  uint32_t &slot, uint64_t tokenStride, uint32_t eleNumPerC0,
                                                  PackedSegList &list, uint32_t l1Split, uint32_t ubOff)
{
    list.nSeg = 0;
    if (gatheredLen == 0) {
        return;
    }
    uint32_t tileEnd = gatheredStart + gatheredLen;
    uint32_t dealtLenAccum = 0;
    while (slot < packedCount && dealtLenAccum < gatheredLen && list.nSeg < STEP4_GATHER_MAX_SEG) {
        uint32_t targetEnd = tileEnd;
        if (l1Split > 0) {
            uint32_t nextSplit = ((dealtLenAccum / l1Split) + 1) * l1Split;
            uint32_t splitAbs = gatheredStart + nextSplit;
            if (splitAbs < targetEnd) {
                targetEnd = splitAbs;
            }
        }
        uint32_t oriStart = 0;
        uint32_t segLen = PackedNextSlotUb<IS_DECODE>(ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize,
                                                     gatheredStart, targetEnd, kvSeqlen, compactPos, slot, oriStart,
                                                     ubOff);
        if (segLen == 0) {
            break;
        }
        bool storedPair = false;
        if constexpr (SUPPORT_PAIR) {
            uint32_t compactPos2 = compactPos;
            uint32_t slot2 = slot;
            uint32_t oriStart2 = 0;
            uint32_t segLen2 = PackedNextSlotUb<IS_DECODE>(ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize,
                                                          gatheredStart, targetEnd, kvSeqlen, compactPos2, slot2,
                                                          oriStart2, ubOff);
            if (segLen2 == segLen && oriStart2 != oriStart) {
                uint32_t gmLo = (oriStart < oriStart2) ? oriStart : oriStart2;
                uint32_t gmHi = (oriStart < oriStart2) ? oriStart2 : oriStart;
                uint64_t srcStride = static_cast<uint64_t>(gmHi - gmLo) * tokenStride;
                if (srcStride % eleNumPerC0 == 0 && dealtLenAccum + segLen + segLen2 <= gatheredLen) {
                    uint32_t i = list.nSeg;
                    list.gm[i] = gmLo;
                    list.dst[i] = dealtLenAccum;
                    list.nValue[i] = segLen;
                    list.ndNum[i] = 2;
                    list.srcStride[i] = srcStride;
                    list.nSeg = i + 1;
                    dealtLenAccum += segLen + segLen2;
                    compactPos = compactPos2;
                    slot = slot2;
                    storedPair = true;
                }
            }
        }
        if (!storedPair) {
            uint32_t i = list.nSeg;
            list.gm[i] = oriStart;
            list.dst[i] = dealtLenAccum;
            list.nValue[i] = segLen;
            list.ndNum[i] = 1;
            list.srcStride[i] = 0;
            list.nSeg = i + 1;
            dealtLenAccum += segLen;
        }
    }
}

__aicore__ inline void PackedRegionStoreUbListTile(AscendC::LocalTensor<uint32_t> ubList, __gm__ uint32_t *listPtr,
                                                   uint32_t wordOff)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::GlobalTensor<uint32_t> gTile;
    gTile.SetGlobalBuffer(listPtr + wordOff);
    AscendC::DataCopy(gTile, ubList, STEP4_SEG_LIST_WORDS);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// Prefill rs=8: skip -1, each live slot is a full region. 0=end, 1=gm, 2=bad valid.
template <class PackedT>
__aicore__ inline uint32_t PackedPrefill8NextLiveGm(AscendC::LocalTensor<PackedT> ubPacked, uint32_t packedCount,
                                                    uint32_t regionSize, uint32_t pageSize, uint32_t ubOff,
                                                    uint32_t &slot, uint32_t &compactPos, uint32_t &gm)
{
    while (slot < packedCount) {
        uint32_t word = static_cast<uint32_t>(ubPacked.GetValue(slot + ubOff));
        ++slot;
        if (word == STEP4_PACKED_PREFILL_SKIP) {
            continue;
        }
        uint32_t valid = word >> STEP4_PACKED_VALID_SHIFT;
        if (valid == 0) {
            continue;
        }
        if (valid != STEP4_PREFILL_PAIR_N) {
            return 2;
        }
        uint32_t phys = word & STEP4_PACKED_REGION_ID_MASK;
        compactPos += valid;
        gm = (pageSize != 0 && (pageSize % regionSize) == 0) ? (phys * regionSize) :
                                                               PackedRegionGmStart(phys, regionSize, pageSize);
        return 1;
    }
    return 0;
}

template <bool SUPPORT_PAIR, class PackedT>
__aicore__ inline void PackedRegionDecodePrefill8ToUbList(AscendC::LocalTensor<PackedT> ubPacked,
                                                          AscendC::LocalTensor<int32_t> ubGm, bool hasGm,
                                                          uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                                          uint32_t gatheredStart, uint32_t gatheredLen,
                                                          uint32_t &compactPos, uint32_t &slot, uint64_t tokenStride,
                                                          uint32_t eleNumPerC0, uint32_t ubOff,
                                                          AscendC::LocalTensor<uint32_t> ubList, uint32_t listBase,
                                                          uint32_t relLimit)
{
    (void)ubGm;
    (void)hasGm;
    uint32_t nPair = 0;
    uint32_t nTail = 0;
    uint32_t rel = 1;
    uint32_t dealtLenAccum = 0;
    bool ok = true;
    while (ok && compactPos < gatheredStart) {
        uint32_t skipGm = 0;
        uint32_t st = PackedPrefill8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff, slot, compactPos,
                                              skipGm);
        (void)skipGm;
        if (st != 1) {
            ok = false;
        }
    }
    if (ok && compactPos != gatheredStart) {
        ok = false;
    }
    while (ok && slot < packedCount && dealtLenAccum < gatheredLen) {
        uint32_t gm0 = 0;
        uint32_t st0 = PackedPrefill8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff, slot, compactPos,
                                               gm0);
        if (st0 != 1) {
            ok = (st0 == 0 && dealtLenAccum == gatheredLen);
            break;
        }
        bool storedPair = false;
        if constexpr (SUPPORT_PAIR) {
            if (dealtLenAccum + 16u <= gatheredLen && rel + 3u <= relLimit) {
                uint32_t compactPos2 = compactPos;
                uint32_t slot2 = slot;
                uint32_t gm1 = 0;
                uint32_t st1 = PackedPrefill8NextLiveGm(ubPacked, packedCount, regionSize, pageSize, ubOff, slot2,
                                                       compactPos2, gm1);
                if (st1 == 1 && gm1 != gm0) {
                    uint32_t gmLo = (gm0 < gm1) ? gm0 : gm1;
                    uint32_t gmHi = (gm0 < gm1) ? gm1 : gm0;
                    uint64_t stride = static_cast<uint64_t>(gmHi - gmLo) * tokenStride;
                    if (stride % eleNumPerC0 == 0) {
                        ubList.SetValue(listBase + rel, gmLo);
                        ubList.SetValue(listBase + rel + 1, static_cast<uint32_t>(stride));
                        ubList.SetValue(listBase + rel + 2, static_cast<uint32_t>(stride >> 32));
                        rel += 3;
                        compactPos = compactPos2;
                        slot = slot2;
                        dealtLenAccum += 16;
                        nPair += 1;
                        storedPair = true;
                    }
                }
            }
        }
        if (!storedPair) {
            if (dealtLenAccum + STEP4_PREFILL_PAIR_N < gatheredLen || rel + 1u > relLimit) {
                ok = false;
                break;
            }
            ubList.SetValue(listBase + rel, gm0);
            rel += 1;
            dealtLenAccum += STEP4_PREFILL_PAIR_N;
            nTail += 1;
        }
    }
    uint32_t hdr = nPair | (nTail << STEP4_PREFILL_HDR_TAIL_SHIFT);
    ubList.SetValue(listBase, (ok && dealtLenAccum == gatheredLen) ? hdr : 0);
}

template <bool IS_DECODE, bool SUPPORT_PAIR, class PackedT>
__aicore__ inline void PackedRegionDecodeTileToUbList(AscendC::LocalTensor<PackedT> ubPacked,
                                                      AscendC::LocalTensor<int32_t> ubGm, bool hasGm,
                                                      uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                                      uint32_t gatheredStart, uint32_t gatheredLen, int32_t kvSeqlen,
                                                      uint32_t &compactPos, uint32_t &slot, uint64_t tokenStride,
                                                      uint32_t eleNumPerC0, uint32_t ubOff,
                                                      AscendC::LocalTensor<uint32_t> ubList, uint32_t listBase,
                                                      uint32_t relLimit)
{
    if constexpr (!IS_DECODE) {
        if (regionSize == STEP4_PREFILL_PAIR_N) {
            PackedRegionDecodePrefill8ToUbList<SUPPORT_PAIR>(
                ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize, gatheredStart, gatheredLen, compactPos, slot,
                tokenStride, eleNumPerC0, ubOff, ubList, listBase, relLimit);
            return;
        }
    }
    uint32_t nSeg = 0;
    uint32_t rel = 1;
    uint32_t dealtLenAccum = 0;
    uint32_t tileEnd = gatheredStart + gatheredLen;
    while (slot < packedCount && dealtLenAccum < gatheredLen && nSeg < STEP4_GATHER_MAX_SEG) {
        uint32_t oriStart = 0;
        uint32_t segLen = PackedNextSlotUb<IS_DECODE>(ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize,
                                                     gatheredStart, tileEnd, kvSeqlen, compactPos, slot, oriStart,
                                                     ubOff);
        if (segLen == 0) {
            break;
        }
        uint32_t ndNum = 1;
        uint32_t gmStart = oriStart;
        uint64_t srcStride = 0;
        if constexpr (SUPPORT_PAIR) {
            uint32_t compactPos2 = compactPos;
            uint32_t slot2 = slot;
            uint32_t oriStart2 = 0;
            uint32_t segLen2 = PackedNextSlotUb<IS_DECODE>(ubPacked, ubGm, hasGm, packedCount, regionSize, pageSize,
                                                          gatheredStart, tileEnd, kvSeqlen, compactPos2, slot2,
                                                          oriStart2, ubOff);
            if (segLen2 == segLen && oriStart2 != oriStart) {
                uint32_t gmLo = (oriStart < oriStart2) ? oriStart : oriStart2;
                uint32_t gmHi = (oriStart < oriStart2) ? oriStart2 : oriStart;
                uint64_t stride = static_cast<uint64_t>(gmHi - gmLo) * tokenStride;
                if (stride % eleNumPerC0 == 0 && dealtLenAccum + segLen + segLen2 <= gatheredLen) {
                    ndNum = 2;
                    gmStart = gmLo;
                    srcStride = stride;
                    compactPos = compactPos2;
                    slot = slot2;
                    dealtLenAccum += segLen + segLen2;
                }
            }
        }
        if (ndNum == 1) {
            dealtLenAccum += segLen;
        }
        uint32_t next = PackedSegWriteRecordAt(ubList, listBase, rel, gmStart, segLen, ndNum, srcStride, relLimit);
        if (next == rel) {
            break;
        }
        rel = next;
        nSeg += 1;
    }
    ubList.SetValue(listBase, (dealtLenAccum < gatheredLen) ? 0 : nSeg);
}

template <bool IS_DECODE, class PackedT>
__aicore__ inline void PackedRegionDecodeOneRowToGm(AscendC::GlobalTensor<PackedT> gPacked,
                                                    AscendC::GlobalTensor<int32_t> gCounts,
                                                    AscendC::GlobalTensor<int32_t> gSeqlens, uint32_t qTokens,
                                                    uint32_t totalTaskNum, uint32_t topK, uint32_t regionSize,
                                                    uint32_t pageSize, uint32_t kvBaseTile, uint64_t tokenStride,
                                                    uint32_t eleNumPerC0, uint32_t l1Split, uint32_t maxTiles,
                                                    uint32_t packedAlign, uint32_t packedRow,
                                                    AscendC::LocalTensor<PackedT> ubPacked,
                                                    AscendC::LocalTensor<int32_t> ubGm,
                                                    AscendC::LocalTensor<uint32_t> ubList, __gm__ uint32_t *listPtr)
{
    constexpr uint32_t ALIGN_ELEMS = static_cast<uint32_t>(32 / sizeof(PackedT));
    uint32_t totalPackedElems = totalTaskNum * topK;
    uint32_t count = gCounts.GetValue(packedRow);
    int32_t kvSeqlen = 0;
    if constexpr (IS_DECODE) {
        uint32_t qIdx = (qTokens == 0) ? 0 : (packedRow - (packedRow / qTokens) * qTokens);
        kvSeqlen = gSeqlens.GetValue(qIdx);
    }
    uint32_t packedRowStart = packedRow * topK;
    uint32_t rowListBase = packedRow * maxTiles * STEP4_SEG_LIST_WORDS;
    uint32_t gatheredTokens = count * regionSize;
    if constexpr (IS_DECODE) {
        gatheredTokens = 0;
        for (uint32_t s = 0; s < count; ++s) {
            PackedT meta = gPacked.GetValue(packedRowStart + s);
            uint32_t phys = 0;
            uint32_t valid = 0;
            ParsePackedSlot<IS_DECODE>(meta, regionSize, kvSeqlen, phys, valid);
            gatheredTokens += valid;
            (void)phys;
        }
    }
    if (count == 0 || gatheredTokens == 0) {
        return;
    }
    uint32_t left = packedRowStart % ALIGN_ELEMS;
    uint32_t alignedStart = packedRowStart - left;
    if (alignedStart >= totalPackedElems) {
        return;
    }
    uint32_t remain = totalPackedElems - alignedStart;
    uint32_t copyCount = (remain / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (copyCount > packedAlign) {
        copyCount = packedAlign;
    }
    if (copyCount < left + count) {
        return;
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::DataCopy(ubPacked, gPacked[alignedStart], copyCount);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

    uint32_t slot = 0;
    uint32_t compactPos = 0;
    uint32_t nTiles = 1;
    if (kvBaseTile > 0) {
        nTiles = (gatheredTokens + kvBaseTile - 1) / kvBaseTile;
    }
    if (nTiles > maxTiles) {
        nTiles = maxTiles;
    }
    for (uint32_t t = 0; t < nTiles; ++t) {
        uint32_t start = t * kvBaseTile;
        if (start >= gatheredTokens) {
            break;
        }
        uint32_t len = kvBaseTile;
        if (start + len > gatheredTokens) {
            len = gatheredTokens - start;
        }
        uint32_t half = (l1Split > 0) ? l1Split : len;
        ubList.SetValue(STEP4_SEG_HALF_WORDS, 0);
        uint32_t nHalf = 1;
        if (half > 0 && len > half) {
            nHalf = 2;
        }
        for (uint32_t h = 0; h < nHalf; ++h) {
            uint32_t halfStart = start + h * half;
            if (halfStart >= start + len) {
                break;
            }
            uint32_t halfLen = half;
            if (halfStart + halfLen > start + len) {
                halfLen = start + len - halfStart;
            }
            PackedRegionDecodeTileToUbList<IS_DECODE, true>(
                ubPacked, ubGm, false, count, regionSize, pageSize, halfStart, halfLen, kvSeqlen, compactPos, slot,
                tokenStride, eleNumPerC0, left, ubList, h * STEP4_SEG_HALF_WORDS, STEP4_SEG_HALF_WORDS);
        }
        PackedRegionStoreUbListTile(ubList, listPtr, rowListBase + t * STEP4_SEG_LIST_WORDS);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
}

// All-AIV pre-pass like BSA mask2idx: WaitFlag V_MTE2, DataCopyPad from a 32B-aligned
// GM origin, scalar compact on UB, DataCopyPad UB->GM. No PIPE_ALL.
template <bool IS_DECODE, class PackedT>
__aicore__ inline void PackedRegionDecodeAllRowsToGm(AscendC::GlobalTensor<PackedT> gPacked,
                                                     AscendC::GlobalTensor<int32_t> gCounts,
                                                     AscendC::GlobalTensor<int32_t> gSeqlens, uint32_t qTokens,
                                                     uint32_t totalTaskNum, uint32_t topK, uint32_t regionSize,
                                                     uint32_t pageSize, uint32_t kvBaseTile, uint64_t tokenStride,
                                                     uint32_t eleNumPerC0, uint32_t l1Split, uint32_t maxTiles,
                                                     uint32_t packedAlign, AscendC::LocalTensor<PackedT> ubPacked,
                                                     AscendC::LocalTensor<int32_t> ubPhys,
                                                     AscendC::LocalTensor<int32_t> ubGm,
                                                     AscendC::LocalTensor<int32_t> ubTmp,
                                                     AscendC::LocalTensor<uint32_t> ubList, __gm__ uint32_t *listPtr)
{
    (void)ubPhys;
    (void)ubTmp;
    uint32_t nSub = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    uint32_t subIdx = AscendC::GetBlockIdx();
    constexpr uint32_t ALIGN_ELEMS = static_cast<uint32_t>(32 / sizeof(PackedT));
    uint32_t totalPackedElems = totalTaskNum * topK;
    bool wroteList = false;
    for (uint32_t packedRow = subIdx; packedRow < totalTaskNum; packedRow += nSub) {
        uint32_t count = gCounts.GetValue(packedRow);
        int32_t kvSeqlen = 0;
        if constexpr (IS_DECODE) {
            uint32_t qIdx = (qTokens == 0) ? 0 : (packedRow - (packedRow / qTokens) * qTokens);
            kvSeqlen = gSeqlens.GetValue(qIdx);
        }
        uint32_t packedRowStart = packedRow * topK;
        uint32_t rowListBase = packedRow * maxTiles * STEP4_SEG_LIST_WORDS;
        uint32_t gatheredTokens = count * regionSize;
        if constexpr (IS_DECODE) {
            gatheredTokens = 0;
            for (uint32_t s = 0; s < count; ++s) {
                PackedT meta = gPacked.GetValue(packedRowStart + s);
                uint32_t phys = 0;
                uint32_t valid = 0;
                ParsePackedSlot<IS_DECODE>(meta, regionSize, kvSeqlen, phys, valid);
                gatheredTokens += valid;
                (void)phys;
            }
        }
        if (count == 0 || gatheredTokens == 0) {
            continue;
        }
        uint32_t left = packedRowStart % ALIGN_ELEMS;
        uint32_t alignedStart = packedRowStart - left;
        if (alignedStart >= totalPackedElems) {
            continue;
        }
        uint32_t remain = totalPackedElems - alignedStart;
        uint32_t copyCount = (remain / ALIGN_ELEMS) * ALIGN_ELEMS;
        if (copyCount > packedAlign) {
            copyCount = packedAlign;
        }
        if (copyCount < left + count) {
            continue;
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy(ubPacked, gPacked[alignedStart], copyCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        uint32_t slot = 0;
        uint32_t compactPos = 0;
        uint32_t nTiles = 1;
        if (kvBaseTile > 0) {
            nTiles = (gatheredTokens + kvBaseTile - 1) / kvBaseTile;
        }
        if (nTiles > maxTiles) {
            nTiles = maxTiles;
        }
        for (uint32_t t = 0; t < nTiles; ++t) {
            uint32_t start = t * kvBaseTile;
            if (start >= gatheredTokens) {
                break;
            }
            uint32_t len = kvBaseTile;
            if (start + len > gatheredTokens) {
                len = gatheredTokens - start;
            }
            uint32_t half = (l1Split > 0) ? l1Split : len;
            ubList.SetValue(STEP4_SEG_HALF_WORDS, 0);
            uint32_t nHalf = 1;
            if (half > 0 && len > half) {
                nHalf = 2;
            }
            for (uint32_t h = 0; h < nHalf; ++h) {
                uint32_t halfStart = start + h * half;
                if (halfStart >= start + len) {
                    break;
                }
                uint32_t halfLen = half;
                if (halfStart + halfLen > start + len) {
                    halfLen = start + len - halfStart;
                }
                PackedRegionDecodeTileToUbList<IS_DECODE, true>(
                    ubPacked, ubGm, false, count, regionSize, pageSize, halfStart, halfLen, kvSeqlen, compactPos, slot,
                    tokenStride, eleNumPerC0, left, ubList, h * STEP4_SEG_HALF_WORDS, STEP4_SEG_HALF_WORDS);
            }
            PackedRegionStoreUbListTile(ubList, listPtr, rowListBase + t * STEP4_SEG_LIST_WORDS);
            wroteList = true;
        }
    }
    if (wroteList) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    }
}

template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline void PackedRegionIssuePair8(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                              uint32_t gm, uint64_t srcStride, uint32_t embed, uint32_t l1Dst,
                                              uint32_t dstStride)
{
    auto l1Tile = tla::GetTile(l1Tensor, tla::MakeCoord(l1Dst, 0), tla::MakeShape(STEP4_PREFILL_PAIR_N, embed));
    auto gTile = tla::GetTile(gTensor, tla::MakeCoord(gm, 0), tla::MakeShape(STEP4_PREFILL_PAIR_N, embed));
    copyGmToL1(l1Tile, gTile, 2, srcStride, dstStride);
}

template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline void PackedRegionIssueSingle8(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                uint32_t gm, uint32_t embed, uint32_t l1Dst)
{
    auto l1Tile = tla::GetTile(l1Tensor, tla::MakeCoord(l1Dst, 0), tla::MakeShape(STEP4_PREFILL_PAIR_N, embed));
    auto gTile = tla::GetTile(gTensor, tla::MakeCoord(gm, 0), tla::MakeShape(STEP4_PREFILL_PAIR_N, embed));
    copyGmToL1(l1Tile, gTile);
}

// Prefill rs=8: hdr = nPair | (nTail << 16), then nPair * (gm, stride_lo, stride_hi), then nTail * gm.
// DrainEvery: 0 = one barrier at the end (K). V 256-row zN NaNs if 8 ndNum=2 stay in flight.
template <uint32_t DrainEvery = 0, class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline uint32_t PackedRegionIssueFromGmPrefill8(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                           AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff,
                                                           uint32_t embed, uint32_t l1DstBase)
{
    uint32_t hdr = gList.GetValue(wordOff);
    if (hdr == 0) {
        return 0;
    }
    uint32_t nPair = hdr & 0xFFu;
    uint32_t nTail = hdr >> STEP4_PREFILL_HDR_TAIL_SHIFT;
    const uint32_t dstStride = STEP4_PREFILL_PAIR_N * tla::get<0, 0>(l1Tensor.stride());
    uint32_t rel = 1;
    uint32_t issued = 0;
    uint32_t nIssuedCopies = 0;
    if (nPair == 8 && nTail == 0) {
        for (uint32_t i = 0; i < 8; ++i) {
            uint32_t gm = gList.GetValue(wordOff + rel);
            uint32_t lo = gList.GetValue(wordOff + rel + 1);
            uint32_t hi = gList.GetValue(wordOff + rel + 2);
            rel += 3;
            uint64_t srcStride = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
            PackedRegionIssuePair8(copyGmToL1, gTensor, l1Tensor, gm, srcStride, embed, l1DstBase + (i << 4),
                                   dstStride);
            ++nIssuedCopies;
            if constexpr (DrainEvery > 0) {
                if ((nIssuedCopies % DrainEvery) == 0) {
                    AscendC::PipeBarrier<PIPE_MTE2>();
                }
            }
        }
        issued = 128;
    } else {
        uint32_t localDst = l1DstBase;
        for (uint32_t i = 0; i < nPair; ++i) {
            uint32_t gm = gList.GetValue(wordOff + rel);
            uint32_t lo = gList.GetValue(wordOff + rel + 1);
            uint32_t hi = gList.GetValue(wordOff + rel + 2);
            rel += 3;
            uint64_t srcStride = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
            PackedRegionIssuePair8(copyGmToL1, gTensor, l1Tensor, gm, srcStride, embed, localDst, dstStride);
            ++nIssuedCopies;
            if constexpr (DrainEvery > 0) {
                if ((nIssuedCopies % DrainEvery) == 0) {
                    AscendC::PipeBarrier<PIPE_MTE2>();
                }
            }
            localDst += 16;
            issued += 16;
        }
        for (uint32_t t = 0; t < nTail; ++t) {
            uint32_t gm = gList.GetValue(wordOff + rel);
            rel += 1;
            PackedRegionIssueSingle8(copyGmToL1, gTensor, l1Tensor, gm, embed, localDst);
            ++nIssuedCopies;
            if constexpr (DrainEvery > 0) {
                if ((nIssuedCopies % DrainEvery) == 0) {
                    AscendC::PipeBarrier<PIPE_MTE2>();
                }
            }
            localDst += STEP4_PREFILL_PAIR_N;
            issued += STEP4_PREFILL_PAIR_N;
        }
    }
    if (issued > 0 && (DrainEvery == 0 || (nIssuedCopies % DrainEvery) != 0)) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    return issued;
}

// Same Prefill8 list as K, but each pair is two nValue=8 copies (no ndNum=2) for V L1.
template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline uint32_t PackedRegionIssueFromGmPrefill8Singles(CopyGmToL1 &copyGmToL1, TensorGM &gTensor,
                                                                   TensorL1 &l1Tensor,
                                                                   AscendC::GlobalTensor<uint32_t> gList,
                                                                   uint32_t wordOff, uint32_t embed, uint32_t l1DstBase)
{
    uint32_t hdr = gList.GetValue(wordOff);
    if (hdr == 0) {
        return 0;
    }
    uint32_t nPair = hdr & 0xFFu;
    uint32_t nTail = hdr >> STEP4_PREFILL_HDR_TAIL_SHIFT;
    uint64_t tokenStride = static_cast<uint64_t>(tla::get<0>(gTensor.stride()));
    uint32_t rel = 1;
    uint32_t localDst = l1DstBase;
    uint32_t issued = 0;
    for (uint32_t i = 0; i < nPair; ++i) {
        uint32_t gm = gList.GetValue(wordOff + rel);
        uint32_t lo = gList.GetValue(wordOff + rel + 1);
        uint32_t hi = gList.GetValue(wordOff + rel + 2);
        rel += 3;
        uint64_t srcStride = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
        PackedRegionIssueSingle8(copyGmToL1, gTensor, l1Tensor, gm, embed, localDst);
        localDst += STEP4_PREFILL_PAIR_N;
        issued += STEP4_PREFILL_PAIR_N;
        uint32_t gmHi = gm;
        if (tokenStride > 0) {
            gmHi = gm + static_cast<uint32_t>(srcStride / tokenStride);
        }
        PackedRegionIssueSingle8(copyGmToL1, gTensor, l1Tensor, gmHi, embed, localDst);
        localDst += STEP4_PREFILL_PAIR_N;
        issued += STEP4_PREFILL_PAIR_N;
    }
    for (uint32_t t = 0; t < nTail; ++t) {
        uint32_t gm = gList.GetValue(wordOff + rel);
        rel += 1;
        PackedRegionIssueSingle8(copyGmToL1, gTensor, l1Tensor, gm, embed, localDst);
        localDst += STEP4_PREFILL_PAIR_N;
        issued += STEP4_PREFILL_PAIR_N;
    }
    if (issued > 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    return issued;
}

template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline uint32_t PackedRegionIssueFromGmGeneric(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                          AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff,
                                                          uint32_t embed, uint32_t l1DstBase)
{
    uint32_t n = gList.GetValue(wordOff);
    if (n == 0 || n > STEP4_GATHER_MAX_SEG) {
        return 0;
    }
    const uint32_t eleNumPerC0 = tla::get<0, 0>(l1Tensor.stride());
    uint32_t rel = 1;
    uint32_t localDst = 0;
    uint32_t issued = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (rel + 2u > STEP4_SEG_HALF_WORDS) {
            break;
        }
        uint32_t gm = gList.GetValue(wordOff + rel);
        uint32_t meta = gList.GetValue(wordOff + rel + 1);
        rel += 2;
        uint32_t nValue = 0;
        uint32_t ndNum = 0;
        PackedSegUnpackMeta(meta, nValue, ndNum);
        uint64_t srcStride = 0;
        if (ndNum == 2) {
            if (rel + 2u > STEP4_SEG_HALF_WORDS) {
                break;
            }
            uint32_t lo = gList.GetValue(wordOff + rel);
            uint32_t hi = gList.GetValue(wordOff + rel + 1);
            rel += 2;
            srcStride = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
        } else {
            ndNum = 1;
        }
        auto l1Tile =
            tla::GetTile(l1Tensor, tla::MakeCoord(l1DstBase + localDst, 0), tla::MakeShape(nValue, embed));
        auto gTile = tla::GetTile(gTensor, tla::MakeCoord(gm, 0), tla::MakeShape(nValue, embed));
        if (ndNum == 2) {
            copyGmToL1(l1Tile, gTile, 2, srcStride, nValue * eleNumPerC0);
        } else {
            copyGmToL1(l1Tile, gTile);
        }
        localDst += nValue * ndNum;
        issued += nValue * ndNum;
    }
    if (issued > 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    return issued;
}

template <bool IS_DECODE, class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline uint32_t PackedRegionIssueFromGm(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                   AscendC::GlobalTensor<uint32_t> gList, uint32_t wordOff,
                                                   uint32_t embed, uint32_t l1DstBase, uint32_t regionSize)
{
    if constexpr (!IS_DECODE) {
        if (regionSize == STEP4_PREFILL_PAIR_N) {
            return PackedRegionIssueFromGmPrefill8(copyGmToL1, gTensor, l1Tensor, gList, wordOff, embed, l1DstBase);
        }
    }
    return PackedRegionIssueFromGmGeneric(copyGmToL1, gTensor, l1Tensor, gList, wordOff, embed, l1DstBase);
}

template <class CopyGmToL1, class TensorGM, class TensorL1>
__aicore__ inline uint32_t PackedRegionIssueFromList(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                                     PackedSegList const &list, uint32_t embed, uint32_t dstBegin,
                                                     uint32_t dstLen)
{
    uint32_t dstEnd = dstBegin + dstLen;
    const uint32_t eleNumPerC0 = tla::get<0, 0>(l1Tensor.stride());
    uint32_t issued = 0;
    for (uint32_t i = 0; i < list.nSeg; ++i) {
        uint32_t nTok = list.nValue[i] * list.ndNum[i];
        uint32_t lo = list.dst[i];
        uint32_t hi = lo + nTok;
        if (lo < dstBegin || hi > dstEnd) {
            continue;
        }
        uint32_t l1Dst = lo - dstBegin;
        auto l1Tile = tla::GetTile(l1Tensor, tla::MakeCoord(l1Dst, 0), tla::MakeShape(list.nValue[i], embed));
        auto gTile = tla::GetTile(gTensor, tla::MakeCoord(list.gm[i], 0), tla::MakeShape(list.nValue[i], embed));
        if (list.ndNum[i] == 2) {
            copyGmToL1(l1Tile, gTile, 2, list.srcStride[i], list.nValue[i] * eleNumPerC0);
        } else {
            copyGmToL1(l1Tile, gTile);
        }
        issued += nTok;
    }
    if (issued > 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    return issued;
}

// Compact live tokens from packed slots into a contiguous L1 tile.
// Prefill INT32: phys_region | (valid_tokens << 24); skip sentinel is -1 (0xFFFFFFFF).
// Decode INT64: start_token | (phys_region << 32); valid = clamp(kv_seqlen - start, 0, regionSize).
// One slot per segment; two equal-length slots use Nd2Nz ndNum=2. compactPos/slot are inout
// cursors. Issue copies without a per-segment PipeBarrier; wait once after the tile.
template <bool IS_DECODE, class CopyGmToL1, class TensorGM, class TensorL1, class PackedT>
__aicore__ inline void PackedRegionGatherToL1(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                              AscendC::GlobalTensor<PackedT> gPacked, uint32_t packedRowStart,
                                              uint32_t packedCount, uint32_t regionSize, uint32_t pageSize,
                                              uint32_t gatheredStart, uint32_t gatheredLen, uint32_t embed,
                                              int32_t kvSeqlen, uint32_t &compactPos, uint32_t &slot)
{
    constexpr bool kSupportPair = tla::detail::isRowMajor<typename TensorGM::Layout>::value &&
                                  !AscendC::IsSameType<typename TensorGM::Element, fp4x2_e2m1_t>::value;
    uint32_t targetEnd = gatheredStart + gatheredLen;
    uint32_t dealtLenAccum = 0;
    while (slot < packedCount && dealtLenAccum < gatheredLen) {
        uint32_t oriStart = 0;
        uint32_t segLen = PackedNextSlot<IS_DECODE>(gPacked, packedRowStart, packedCount, regionSize, pageSize,
                                                    gatheredStart, targetEnd, kvSeqlen, compactPos, slot, oriStart);
        if (segLen == 0) {
            break;
        }
        auto l1Tile = tla::GetTile(l1Tensor, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(segLen, embed));
        auto gTile = tla::GetTile(gTensor, tla::MakeCoord(oriStart, 0), tla::MakeShape(segLen, embed));
        bool paired = false;
        if constexpr (kSupportPair) {
            uint32_t compactPos2 = compactPos;
            uint32_t slot2 = slot;
            uint32_t oriStart2 = 0;
            uint32_t segLen2 = PackedNextSlot<IS_DECODE>(gPacked, packedRowStart, packedCount, regionSize, pageSize,
                                                         gatheredStart, targetEnd, kvSeqlen, compactPos2, slot2,
                                                         oriStart2);
            if (segLen2 == segLen && oriStart2 != oriStart) {
                uint32_t gmLo = (oriStart < oriStart2) ? oriStart : oriStart2;
                uint32_t gmHi = (oriStart < oriStart2) ? oriStart2 : oriStart;
                uint64_t srcStride = static_cast<uint64_t>(gmHi - gmLo) *
                                     static_cast<uint64_t>(tla::get<0>(gTensor.stride()));
                const uint32_t eleNumPerC0 = tla::get<0, 0>(l1Tensor.stride());
                const uint32_t dstStride = segLen * eleNumPerC0;
                if (srcStride % eleNumPerC0 == 0 && dealtLenAccum + segLen + segLen2 <= gatheredLen) {
                    auto gPair = tla::GetTile(gTensor, tla::MakeCoord(gmLo, 0), tla::MakeShape(segLen, embed));
                    copyGmToL1(l1Tile, gPair, 2, srcStride, dstStride);
                    dealtLenAccum += segLen + segLen2;
                    compactPos = compactPos2;
                    slot = slot2;
                    paired = true;
                }
            }
        }
        if (!paired) {
            copyGmToL1(l1Tile, gTile);
            dealtLenAccum += segLen;
        }
    }
    if (dealtLenAccum > 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
}

} // namespace NpuArch::Gemm::Block

#endif
