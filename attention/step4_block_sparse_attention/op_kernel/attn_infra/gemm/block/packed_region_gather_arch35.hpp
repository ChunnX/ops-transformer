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
        int32_t word = static_cast<int32_t>(meta);
        if (word < 0) {
            phys = 0;
            valid = 0;
            return;
        }
        phys = static_cast<uint32_t>(word) & STEP4_PACKED_REGION_ID_MASK;
        valid = static_cast<uint32_t>(word) >> STEP4_PACKED_VALID_SHIFT;
    }
}

// Compact live tokens from packed slots into a contiguous L1 tile.
// Prefill INT32: phys_region | (valid_tokens << 24); meta < 0 skipped.
// Decode INT64: start_token | (phys_region << 32); valid = clamp(kv_seqlen - start, 0, regionSize).
template <bool IS_DECODE, class CopyGmToL1, class TensorGM, class TensorL1, class PackedT>
__aicore__ inline void PackedRegionGatherToL1(CopyGmToL1 &copyGmToL1, TensorGM &gTensor, TensorL1 &l1Tensor,
                                              AscendC::GlobalTensor<PackedT> gPacked, uint32_t packedRowStart,
                                              uint32_t packedCount, uint32_t regionSize, uint32_t gatheredStart,
                                              uint32_t gatheredLen, uint32_t embed, int32_t kvSeqlen)
{
    uint32_t targetEnd = gatheredStart + gatheredLen;
    uint32_t compactPos = 0;
    uint32_t dealtLenAccum = 0;
    for (uint32_t slot = 0; slot < packedCount && dealtLenAccum < gatheredLen; ++slot) {
        PackedT meta = gPacked.GetValue(packedRowStart + slot);
        uint32_t phys = 0;
        uint32_t valid = 0;
        ParsePackedSlot<IS_DECODE>(meta, regionSize, kvSeqlen, phys, valid);
        if (valid == 0) {
            continue;
        }
        uint32_t slotStart = compactPos;
        compactPos += valid;
        if (compactPos <= gatheredStart) {
            continue;
        }
        if (slotStart >= targetEnd) {
            break;
        }
        uint32_t copyStartInSlot = (gatheredStart > slotStart) ? (gatheredStart - slotStart) : 0;
        uint32_t copyEndInSlot = valid;
        uint32_t slotCap = targetEnd - slotStart;
        if (slotCap < copyEndInSlot) {
            copyEndInSlot = slotCap;
        }
        uint32_t curDealtLen = copyEndInSlot - copyStartInSlot;
        if (curDealtLen == 0) {
            continue;
        }
        uint32_t oriStart = phys * regionSize + copyStartInSlot;
        uint32_t remain = gatheredLen - dealtLenAccum;
        uint32_t probe = slot + 1;
        while (probe < packedCount && curDealtLen < remain) {
            PackedT nextMeta = gPacked.GetValue(packedRowStart + probe);
            uint32_t nextPhys = 0;
            uint32_t nextValid = 0;
            ParsePackedSlot<IS_DECODE>(nextMeta, regionSize, kvSeqlen, nextPhys, nextValid);
            if (nextValid == 0) {
                ++probe;
                continue;
            }
            if (nextPhys * regionSize != oriStart + curDealtLen) {
                break;
            }
            uint32_t take = nextValid;
            uint32_t left = remain - curDealtLen;
            if (take > left) {
                take = left;
            }
            curDealtLen += take;
            compactPos += nextValid;
            slot = probe;
            ++probe;
            if (take < nextValid) {
                break;
            }
        }
        auto l1Tile = tla::GetTile(l1Tensor, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
        auto gTile = tla::GetTile(gTensor, tla::MakeCoord(oriStart, 0), tla::MakeShape(curDealtLen, embed));
        copyGmToL1(l1Tile, gTile);
        AscendC::PipeBarrier<PIPE_MTE2>();
        dealtLenAccum += curDealtLen;
    }
}

} // namespace NpuArch::Gemm::Block

#endif
