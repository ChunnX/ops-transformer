/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_QK_ARCH35_MXFP4_HPP
#define GEMM_BLOCK_QK_ARCH35_MXFP4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "block_mmad_arch35_utils.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0_mx_a5.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_gm_to_l1_mx_scale_dn2nz_a5.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Block {

template <
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5BsaQK<true>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_>
{
public:
    using DispatchPolicy = MmadAtlasA5BsaQK<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});

    // Q(M) 方向统一 pad 到 128，满足 Fixpipe nSize 32B 倍数约束
    static constexpr uint32_t Q_M_PAD = 128;

    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    // Event ID：QK 用 2-3，L1 K 复用 MXFP4::KV_EVENT0..3（与 PV 共享）
    static constexpr uint32_t Q_EVENT0 = 2;
    static constexpr uint32_t Q_EVENT1 = 3;
    static constexpr uint32_t QK_L0AB_EVENT0 = 2;
    static constexpr uint32_t QK_L0AB_EVENT1 = 3;
    static constexpr uint32_t QK_L0C_EVENT0 = 2;
    static constexpr uint32_t QK_L0C_EVENT1 = 3;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t& kvBufId) : kBufId(kvBufId) {
        for (uint32_t i = 0; i < MXFP4::L1_Q_BUF_CNT; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(MXFP4::L1_Q_BUF_OFFSET + MXFP4::L1_Q_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L1_Q_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP4::L1_Q_DESCALE_BUF_OFFSET + MXFP4::L1_Q_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(MXFP4::L1_KV_BUF_OFFSET + MXFP4::L1_KV_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L1_KV_DESCALE_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP4::L1_KV_DESCALE_BUF_OFFSET + MXFP4::L1_KV_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L0A_QK_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementB>(MXFP4::L0A_QK_BUF_OFFSET + MXFP4::L0A_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L0B_QK_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(MXFP4::L0B_QK_BUF_OFFSET + MXFP4::L0B_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::L0C_QK_BUF_CNT; i++) {
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<float>(
                MXFP4::L0C_QK_BUF_OFFSET + MXFP4::L0C_QK_BUF_SIZE * i);
        }

        AllocEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline ~BlockMmadTla() {}

    __aicore__ inline uint32_t CeilDivision(uint32_t numerator, uint32_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
    {
        return (a < b) ? a : b;
    }

    // Q 稠密 GM → L1（RowMajor → zN）
    template <class TensorQ, class TensorL1Q>
    __aicore__ inline void CopyQGmToL1(uint32_t qsActBaseTile,
                                       uint32_t embed,
                                       TensorQ gQTensorTla,
                                       TensorL1Q l1QTensorTla)
    {
        auto l1QTileTla = GetTile(l1QTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));
        // BSND: GetTile 确保 dValue = embed 而非 qHeadMul * embed
        auto gQTileTla = GetTile(gQTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));

        using CopyGmToL1Q = Tile::TileCopyTla<ArchTag, decltype(gQTileTla), decltype(l1QTensorTla)>;
        CopyGmToL1Q copyGmToL1Q;
        copyGmToL1Q(l1QTileTla, gQTileTla);
    }

    // Q-scale 稠密 GM → L1（Host 沿 D Dn2Nz）
    __aicore__ inline void CopyQScaleGmToL1(uint32_t qsActBaseTile,
                                            uint32_t scaleK,
                                            AscendC::GlobalTensor<uint8_t> &gQDequantScale,
                                            uint32_t qBufId,
                                            uint32_t qHeadMul)
    {
        copyGmToL1MxScaleDn2Nz(l1BScaleTensor[qBufId], gQDequantScale, qsActBaseTile, scaleK,
                               0, 0, qHeadMul);
    }

    // K 数据稀疏 gather：按 gSparseBlockIdx 逐段搬 GM → L1
    template <class TensorK, class TensorL1K>
    __aicore__ inline
    void SparseKBaseTileL1FullLoad(
        TensorK gKTensorTla,
        TensorL1K l1KTensorTla,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY,
        uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t kvsActBaseTile, uint32_t embed)
    {
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset   = baseTileStartOffset + kvsActBaseTile;
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx   = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStart    = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
        uint32_t dealtLenAccum  = 0;
        using CopyGmToL1K = Tile::TileCopyTla<ArchTag, decltype(gKTensorTla), decltype(l1KTensorTla)>;
        CopyGmToL1K copyGmToL1K;

        while (dealtLenAccum < kvsActBaseTile &&
               gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval &&
               oriStartOffset < static_cast<uint32_t>(kvSeqlen)) {
            uint32_t curYBlockSize = (oriYBlockIdx == yBlockNumAval - 1)
                                     ? static_cast<uint32_t>(kvSeqlen - oriYBlockIdx * blockShapeY)
                                     : blockShapeY;
            uint32_t gatheredEndOffset = MinU32(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                                                baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) { break; }

            auto gKTile = GetTile(
                gKTensorTla, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            auto l1KTile = GetTile(
                l1KTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1K(l1KTile, gKTile);

            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStart   = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < kvsActBaseTile) {
                oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            }
        }
    }

    // K-scale 稀疏 gather：按 gSparseBlockIdx 逐段搬 GM → L1（Host 沿 D Dn2Nz）
    __aicore__ inline
    void SparseKScaleBaseTileL1FullLoad(
        AscendC::GlobalTensor<uint8_t> gKScale,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY,
        uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t kvsActBaseTile, uint32_t embed,
        uint32_t kBufId, uint32_t kvHeadMul)
    {
        uint32_t scaleK = CeilDivision(embed, MXFP4::FP4_C0_ELEMS) * 2;
        uint32_t scaleRowHalf = scaleK / 2;

        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset   = baseTileStartOffset + kvsActBaseTile;
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx   = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStart    = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
        uint32_t dealtLenAccum  = 0;

        while (dealtLenAccum < kvsActBaseTile && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < static_cast<uint32_t>(kvSeqlen)) {
            uint32_t curYBlockSize = (oriYBlockIdx == yBlockNumAval - 1)
                                     ? static_cast<uint32_t>(kvSeqlen - oriYBlockIdx * blockShapeY)
                                     : blockShapeY;
            uint32_t gatheredEndOffset = MinU32(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                                                baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) { break; }

            uint32_t srcB16Off = oriStartOffset * kvHeadMul * scaleRowHalf;
            uint32_t dstB16Off = dealtLenAccum * scaleRowHalf;
            copyGmToL1MxScaleDn2Nz(
                l1AScaleTensor[kBufId], gKScale, curDealtLen, scaleK, srcB16Off, dstB16Off,
                kvHeadMul);

            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStart   = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < kvsActBaseTile) {
                oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            }
        }
    }

    // Q → L0B（非转置，带 mx scale）
    __aicore__ inline void LoadQToL0B(uint32_t qsActBaseTileAlign16,
                                      uint32_t embed,
                                      uint32_t qsActBaseTileAlign16L0,
                                      uint32_t scaleMAlign16,
                                      uint32_t scaleK,
                                      uint32_t qBufId,
                                      uint32_t qkL0abBufId)
    {
        copyL1ToL0BMxQk(
            l0BTensor[qkL0abBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1BTensor[qBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1BScaleTensor[qBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
            qsActBaseTileAlign16, embed, qsActBaseTileAlign16L0, scaleMAlign16, scaleK);
    }

    // K → L0A（非转置，带 mx scale，n 方向子切分）
    __aicore__ inline void LoadKToL0A(uint32_t nSubRowStart,
                                      uint32_t nCur,
                                      uint32_t embed,
                                      uint32_t kvsActBaseTileAlign16,
                                      uint32_t scaleK,
                                      uint32_t kBufId,
                                      uint32_t qkL0abBufId)
    {
        copyL1ToL0AMxQk(
            l0ATensor[qkL0abBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1ATensor[kBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1AScaleTensor[kBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
            nSubRowStart, nCur, embed, kvsActBaseTileAlign16, scaleK);
    }

    // C = K · Qᵀ，每 nSub 独立计算，initC 恒 true
    __aicore__ inline void MatmulQK(uint32_t n,
                                    uint32_t qsActBaseTileAlign16,
                                    uint32_t embed,
                                    uint32_t qkL0abBufId,
                                    uint32_t qkL0cBufId)
    {
        auto l0ALayout = tla::MakeLayout<ElementB, LayoutTagL0A>(n, embed);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[qkL0abBufId], l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<ElementA, LayoutTagL0B>(embed, qsActBaseTileAlign16);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[qkL0abBufId], l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(n, qsActBaseTileAlign16);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[qkL0cBufId], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, n, qsActBaseTileAlign16, embed, true);
    }

    // L0C → UB fixpipe（NO_SPLIT 模式，subBlockId 选目标 AIV）
    template <class TensorUB>
    __aicore__ inline void FixpipeMm1(uint32_t qsActBaseTileAlign16, uint32_t n,
                                      TensorUB &ubSTensorTla, bool subBlockId, uint32_t qkL0cBufId)
    {
        auto l0CTensorTla = tla::MakeTensor(
            l0CTensor[qkL0cBufId], tla::MakeLayoutL0C(n, qsActBaseTileAlign16), Arch::PositionL0C{});

        using CopyL0CToUB = Tile::CopyL0CToUBTla<
            ArchTag, decltype(l0CTensorTla), TensorUB,
            Tile::CopyL0CToUBMode::NO_SPLIT,
            Tile::ScaleGranularity::NO_QUANT, false>;
        CopyL0CToUB copyL0CToUB;
        copyL0CToUB(ubSTensorTla, l0CTensorTla, subBlockId);
    }

    template <class TensorQ, class TensorK, class TensorUB, class TaskInfoT, class TileInfoT>
    __aicore__ inline
    void operator()(TensorQ &gQTensorTla,
                    TensorK &gKTensorTla,
                    AscendC::GlobalTensor<uint8_t> gQDequantScale,
                    AscendC::GlobalTensor<uint8_t> gKDequantScale,
                    AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                    TensorUB &ubSTensorTla,
                    TaskInfoT &curTaskInfo,
                    TileInfoT &curTileInfo,
                    uint32_t embed,
                    uint32_t kvSBaseTile, uint32_t blockShapeY,
                    uint32_t subBlockIdx,
                    uint32_t kvHeadMul,
                    uint32_t qHeadMul)
    {
        uint32_t qsActBaseTile = curTaskInfo.qsActBaseTile;
        int64_t kvSeqlen = curTaskInfo.kvSeqlen;
        uint32_t yBlockNumAval = curTaskInfo.yBlockNumAval;
        uint32_t yBlockNumRsvd = curTaskInfo.yBlockNumRsvd;
        uint32_t qsActBaseTileAlign16 = curTaskInfo.qsActBaseTileAlign16;
        uint32_t qsActBaseTileAlign128 = curTaskInfo.qsActBaseTileAlign128;

        uint32_t kvsActBaseTile = curTileInfo.kvsActBaseTile;
        uint32_t gatheredKvSTileIdx = curTileInfo.pvGatheredKvSTileIdx;
        bool isFirstKvsTile = curTileInfo.isFirstKvsTile;
        bool isLastKvsTile = curTileInfo.isLastKvsTile;
        uint32_t kvsActBaseTileAlign16 = curTileInfo.kvsActBaseTileAlign16;

        uint32_t scaleK = CeilDivision(embed, MXFP4::FP4_C0_ELEMS) * 2;
        uint32_t l1KEventId = MXFP4::KV_EVENT0 + kBufId;
        uint32_t l1QEventId = Q_EVENT0 + qBufId;

        // Q/QScale 仅在 task 首 KV tile 搬 GM → L1，后续 tile 复用
        if (isFirstKvsTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            auto l1QLayout = tla::MakeLayout<ElementA, layout::zN>(qsActBaseTileAlign16, embed);
            auto l1QTensorTla = tla::MakeTensor(l1BTensor[qBufId], l1QLayout, Arch::PositionL1{});
            CopyQGmToL1(qsActBaseTile, embed, gQTensorTla, l1QTensorTla);
            CopyQScaleGmToL1(qsActBaseTile, scaleK, gQDequantScale, qBufId, qHeadMul);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
        }

        // 每 tile 稀疏 gather K / K-scale 到 L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        auto l1KLayout = tla::MakeLayout<ElementA, layout::zN>(kvsActBaseTile, embed);
        auto l1KTensorTla = tla::MakeTensor(l1ATensor[kBufId], l1KLayout, Arch::PositionL1{});
        SparseKBaseTileL1FullLoad(
            gKTensorTla, l1KTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, kvsActBaseTile, embed);
        SparseKScaleBaseTileL1FullLoad(
            gKDequantScale, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, kvsActBaseTile, embed, kBufId, kvHeadMul);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);

        // n 方向按 L0_TILE_N 子切分
        uint32_t nLoopNum = CeilDivision(kvsActBaseTile, L0_TILE_N);
        for (uint32_t nSub = 0; nSub < nLoopNum; ++nSub) {
            uint32_t l0ABEventId = QK_L0AB_EVENT0 + qkL0abBufId;
            uint32_t l0CEventId = QK_L0C_EVENT0 + qkL0cBufId;
            uint32_t nSubRowStart = nSub * L0_TILE_N;
            uint32_t nCur = MinU32(L0_TILE_N, kvsActBaseTile - nSubRowStart);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);

            {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
                LoadQToL0B(qsActBaseTileAlign16, embed, qsActBaseTileAlign16, qsActBaseTileAlign16, scaleK, qBufId, qkL0abBufId);
                LoadKToL0A(nSubRowStart, nCur, embed, kvsActBaseTileAlign16, scaleK, kBufId, qkL0abBufId);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
                MatmulQK(nCur, qsActBaseTileAlign16, embed, qkL0cBufId, qkL0abBufId);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
                qkL0abBufId = (qkL0abBufId + 1) % MXFP4::L0A_QK_BUF_CNT;
            }

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            // nCur 偶数对齐用于 Fixpipe mSize
            uint32_t nCurEven = (nCur + 1) >> 1 << 1;
            auto ubSSubTla = GetTile(
                ubSTensorTla, tla::MakeCoord(nSubRowStart, 0), tla::MakeShape(nCurEven, qsActBaseTileAlign16));
            FixpipeMm1(qsActBaseTileAlign16, nCur, ubSSubTla, subBlockIdx != 0, qkL0cBufId);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
            qkL0cBufId = (qkL0cBufId + 1) % MXFP4::L0C_QK_BUF_CNT;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        kBufId = (kBufId + 1) % MXFP4::L1_KV_BUF_CNT;

        if (unlikely(isLastKvsTile)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            qBufId = (qBufId + 1) % MXFP4::L1_Q_BUF_CNT;
        }
    }

protected:
    AscendC::LocalTensor<ElementB> l1ATensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementA> l1BTensor[MXFP4::L1_Q_BUF_CNT];
    AscendC::LocalTensor<ElementB> l0ATensor[MXFP4::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<ElementA> l0BTensor[MXFP4::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[MXFP4::L0C_QK_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[MXFP4::L1_Q_BUF_CNT];

    Tile::CopyL1ToL0AMxQKA5 copyL1ToL0AMxQk;
    Tile::CopyL1ToL0BMxQKA5 copyL1ToL0BMxQk;
    TileMmad tileMmad;
    Tile::CopyGmToL1MxScaleDn2NzA5 copyGmToL1MxScaleDn2Nz;

    uint32_t qBufId = 0;
    uint32_t& kBufId;
    uint32_t qkL0abBufId = 0;
    uint32_t qkL0cBufId = 0;
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_QK_ARCH35_MXFP4_HPP
