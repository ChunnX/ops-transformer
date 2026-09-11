/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef STEP4_PACKED_REGION_GATHER_ARCH35_HPP
#define STEP4_PACKED_REGION_GATHER_ARCH35_HPP

#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"

namespace NpuArch::Gemm::Block {

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
            if (segLen2 == segLen && oriStart2 > oriStart) {
                uint64_t srcStride = static_cast<uint64_t>(oriStart2 - oriStart) *
                                     static_cast<uint64_t>(tla::get<0>(gTensor.stride()));
                const uint32_t eleNumPerC0 = tla::get<0, 0>(l1Tensor.stride());
                const uint32_t dstStride = segLen * eleNumPerC0;
                if (srcStride % eleNumPerC0 == 0 &&
                    dealtLenAccum + segLen + segLen2 <= gatheredLen) {
                    copyGmToL1(l1Tile, gTile, 2, srcStride, dstStride);
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
