/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_QK_ARCH35_MXFP8_HPP
#define GEMM_BLOCK_QK_ARCH35_MXFP8_HPP

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
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_gm_to_l1_mx_scale_dn2nz_a5.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

namespace NpuArch::Gemm::Block {

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
          class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaQKMxfp8<true>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_,
                    ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaQKMxfp8<true>;
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

    // Event ID：QK 用 2-3，L1 K 复用 MXFP8::KV_EVENT0..3（与 PV 共享）
    static constexpr uint32_t Q_EVENT0 = 2;
    static constexpr uint32_t Q_EVENT1 = 3;
    static constexpr uint32_t QK_L0AB_EVENT0 = 2;
    static constexpr uint32_t QK_L0AB_EVENT1 = 3;
    static constexpr uint32_t QK_L0C_EVENT0 = 2;
    static constexpr uint32_t QK_L0C_EVENT1 = 3;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t &kvBufId, uint64_t softmaxScale)
        : kBufId(kvBufId)
    {
        softmaxScale_ = softmaxScale;
        for (uint32_t i = 0; i < MXFP8::L1_Q_BUF_CNT; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(MXFP8::L1_Q_BUF_OFFSET + MXFP8::L1_Q_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L1_Q_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP8::L1_Q_DESCALE_BUF_OFFSET +
                                                                                 MXFP8::L1_Q_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(MXFP8::L1_KV_BUF_OFFSET + MXFP8::L1_KV_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L1_KV_DESCALE_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP8::L1_KV_DESCALE_BUF_OFFSET +
                                                                                 MXFP8::L1_KV_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L0A_QK_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementB>(MXFP8::L0A_QK_BUF_OFFSET +
                                                                              MXFP8::L0A_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L0B_QK_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(MXFP8::L0B_QK_BUF_OFFSET +
                                                                              MXFP8::L0B_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L0C_QK_BUF_CNT; i++) {
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<float>(MXFP8::L0C_QK_BUF_OFFSET + MXFP8::L0C_QK_BUF_SIZE * i);
        }

        AllocEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT3);
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

    // qs64：L1 Q / Q-scale 按 128 行布局，先清零再只搬有效 64 行，避免 pad 行脏数据进 S。
    __aicore__ inline void PadQL1Zero(uint32_t qBufId)
    {
        constexpr uint32_t BLOCK = 32;
        AscendC::InitConstValueParams<uint16_t> qFill(1, static_cast<uint16_t>(MXFP8::L1_Q_BUF_SIZE / BLOCK), 0,
                                                      MXFP8::ZERO_FILL_PATTERN);
        AscendC::Fill(l1BTensor[qBufId].template ReinterpretCast<uint16_t>(), qFill);
        AscendC::InitConstValueParams<uint16_t> qScaleFill(
            1, static_cast<uint16_t>(MXFP8::L1_Q_DESCALE_BUF_SIZE / BLOCK), 0, MXFP8::ZERO_FILL_PATTERN);
        AscendC::Fill(l1BScaleTensor[qBufId].template ReinterpretCast<uint16_t>(), qScaleFill);
    }

    // Q 稠密 GM → L1（RowMajor → zN）
    template <class TensorQ, class TensorL1Q>
    __aicore__ inline void CopyQGmToL1(uint32_t qsActBaseTile, uint32_t embed, TensorQ gQTensorTla,
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
    __aicore__ inline void CopyQScaleGmToL1(uint32_t qsActBaseTile, uint32_t scaleK,
                                            AscendC::GlobalTensor<uint8_t> &gQDequantScale, uint32_t qBufId,
                                            uint32_t qHeadMul)
    {
        copyGmToL1MxScaleDn2Nz(l1BScaleTensor[qBufId], gQDequantScale, qsActBaseTile, scaleK, 0, 0, qHeadMul);
    }

    // K 数据 + K-scale 融合稀疏 gather：按 gSparseBlockIdx 逐段搬 GM → L1。
    // 两者的分段边界必然一致，故只遍历一遍 index，每段发 K(Nd2Nz) / K-scale(Dn2Nz) 两条 copy。
    // ori 连续递增即代表 GM 上物理相邻，向前偷看合并成单条 copy。
    // 段推进天然落在 y-block 边界上（仅首段 innerStart 可能非 0），故循环内无除法/取模。
    // firstYBlockIdx / firstOriYBlockIdx 由调用方提前预取，避免首个 index 的阻塞访存落在此处。
    template <class TensorK, class TensorL1K>
    __aicore__ inline void SparseKFusedBaseTileL1FullLoad(
        TensorK gKTensorTla, TensorL1K l1KTensorTla, AscendC::GlobalTensor<uint8_t> gKScale,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd, uint32_t dataRows,
        uint32_t scaleRows, uint32_t embed, uint32_t kBufId, uint32_t scaleBufId, uint32_t kvHeadMul,
        uint32_t firstYBlockIdx, uint32_t firstOriYBlockIdx)
    {
        using CopyGmToL1K = Tile::TileCopyTla<ArchTag, decltype(gKTensorTla), decltype(l1KTensorTla)>;
        CopyGmToL1K copyGmToL1K;
        (void)kBufId;

        uint32_t scaleK = CeilDivision(embed, MXFP8::MX_GROUP_ELEMS);
        uint32_t scaleRowHalf = scaleK / 2;
        uint32_t walkRows = (scaleRows > dataRows) ? scaleRows : dataRows;

        const uint32_t kvSeqlenU = static_cast<uint32_t>(kvSeqlen);
        uint32_t gatheredYBlockIdx = firstYBlockIdx;
        uint32_t yBlockInnerStart = gatheredKvSTileIdx * kvSBaseTile - firstYBlockIdx * blockShapeY;
        uint32_t oriYBlockIdx = firstOriYBlockIdx;
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < walkRows && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval) {
            uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            if (oriStartOffset >= kvSeqlenU) {
                break;
            }

            uint32_t curYBlockSize =
                (oriYBlockIdx == yBlockNumAval - 1) ? (kvSeqlenU - oriYBlockIdx * blockShapeY) : blockShapeY;
            if (curYBlockSize <= yBlockInnerStart) {
                break;
            }
            uint32_t availLen = curYBlockSize - yBlockInnerStart;
            // 尾部不满块之后 gather 流即终止（与原实现的 curDealtLen==0 退出等价）
            bool runEndsShort = (curYBlockSize < blockShapeY);

            // 向前偷看：ori 连续则 GM 地址连续，拼进同一条 copy
            uint32_t peekYBlockIdx = gatheredYBlockIdx + 1;
            uint32_t peekOriIdx = oriYBlockIdx + 1;
            while (!runEndsShort && dealtLenAccum + availLen < walkRows && peekYBlockIdx < yBlockNumRsvd &&
                   peekOriIdx < yBlockNumAval &&
                   static_cast<uint32_t>(gSparseBlockIdx.GetValue(peekYBlockIdx)) == peekOriIdx) {
                uint32_t peekBlockSize =
                    (peekOriIdx == yBlockNumAval - 1) ? (kvSeqlenU - peekOriIdx * blockShapeY) : blockShapeY;
                availLen += peekBlockSize;
                runEndsShort = (peekBlockSize < blockShapeY);
                ++peekYBlockIdx;
                ++peekOriIdx;
            }

            uint32_t curDealtLen = MinU32(availLen, walkRows - dealtLenAccum);
            uint32_t dataLen = 0;
            if (dealtLenAccum < dataRows) {
                dataLen = MinU32(curDealtLen, dataRows - dealtLenAccum);
            }
            uint32_t scaleLen = 0;
            if (dealtLenAccum < scaleRows) {
                scaleLen = MinU32(curDealtLen, scaleRows - dealtLenAccum);
            }

            uint32_t pairLen = 0;
            uint32_t pairOriStart = 0;
            bool pairEndsShort = false;
            uint32_t pairPeekEnd = peekYBlockIdx;
#if BSA_MXFP8_NDNUM2
            if (!runEndsShort && dataLen > 0 && dataLen == curDealtLen && dataLen == scaleLen &&
                dealtLenAccum + curDealtLen < walkRows && peekYBlockIdx < yBlockNumRsvd) {
                uint32_t nextOri = static_cast<uint32_t>(gSparseBlockIdx.GetValue(peekYBlockIdx));
                if (nextOri < yBlockNumAval) {
                    uint32_t nextStart = nextOri * blockShapeY;
                    uint32_t nextSize =
                        (nextOri == yBlockNumAval - 1) ? (kvSeqlenU - nextOri * blockShapeY) : blockShapeY;
                    uint32_t remain = walkRows - dealtLenAccum - curDealtLen;
                    uint32_t nextDealt = MinU32(nextSize, remain);
                    if (nextDealt == dataLen && nextStart > oriStartOffset) {
                        pairLen = nextDealt;
                        pairOriStart = nextStart;
                        pairEndsShort = (nextSize < blockShapeY);
                        pairPeekEnd = peekYBlockIdx + 1;
                    }
                }
            }
#endif
            uint32_t srcNdStride = 0;
            uint32_t dstNzStride = 0;
            bool fusedNd2 = false;
            if (pairLen > 0) {
                auto gKTile0 = GetTile(gKTensorTla, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(dataLen, embed));
                auto l1KTile0 = GetTile(l1KTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(dataLen, embed));
                auto gKTile1 = GetTile(gKTensorTla, tla::MakeCoord(pairOriStart, 0), tla::MakeShape(dataLen, embed));
                auto l1KTile1 =
                    GetTile(l1KTensorTla, tla::MakeCoord(dealtLenAccum + dataLen, 0), tla::MakeShape(dataLen, embed));
                fusedNd2 = MXFP8::Mxfp8U32Delta(static_cast<int64_t>(gKTile1.layout()(gKTile1.coord())) -
                                                    static_cast<int64_t>(gKTile0.layout()(gKTile0.coord())),
                                                srcNdStride) &&
                           MXFP8::Mxfp8U32Delta(static_cast<int64_t>(l1KTile1.layout()(l1KTile1.coord())) -
                                                    static_cast<int64_t>(l1KTile0.layout()(l1KTile0.coord())),
                                                dstNzStride);
                if (fusedNd2) {
                    copyGmToL1K(l1KTile0, gKTile0, 2, srcNdStride, dstNzStride);
                }
            }
            if (!fusedNd2 && dataLen > 0) {
                auto gKTile = GetTile(gKTensorTla, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(dataLen, embed));
                auto l1KTile = GetTile(l1KTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(dataLen, embed));
                copyGmToL1K(l1KTile, gKTile);
            }
            if (scaleLen > 0) {
                uint32_t srcB16Off = oriStartOffset * kvHeadMul * scaleRowHalf;
                uint32_t dstB16Off = dealtLenAccum * scaleRowHalf;
                copyGmToL1MxScaleDn2Nz(l1AScaleTensor[scaleBufId], gKScale, scaleLen, scaleK, srcB16Off, dstB16Off,
                                       kvHeadMul);
                if (fusedNd2) {
                    uint32_t srcB16Off2 = pairOriStart * kvHeadMul * scaleRowHalf;
                    uint32_t dstB16Off2 = (dealtLenAccum + dataLen) * scaleRowHalf;
                    copyGmToL1MxScaleDn2Nz(l1AScaleTensor[scaleBufId], gKScale, pairLen, scaleK, srcB16Off2, dstB16Off2,
                                           kvHeadMul);
                }
#if BSA_MXFP8_MTE2_INSTR_PROBE
                // 探针 v1：再发一遍相同 K-scale copy。dst 相同。
                if (scaleBufId < MXFP8::L1_KV_DESCALE_BUF_CNT) {
                    copyGmToL1MxScaleDn2Nz(l1AScaleTensor[scaleBufId], gKScale, scaleLen, scaleK, srcB16Off, dstB16Off,
                                           kvHeadMul);
                }
#endif
            }

            dealtLenAccum += curDealtLen + (fusedNd2 ? pairLen : 0);
            if (runEndsShort || (fusedNd2 && pairEndsShort)) {
                break;
            }
            gatheredYBlockIdx = fusedNd2 ? pairPeekEnd : peekYBlockIdx;
            yBlockInnerStart = 0;
            if (dealtLenAccum < walkRows) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
            }
        }
    }

    // Q → L0B（非转置，带 mx scale；kStart/kCur 沿 D 切段）
    __aicore__ inline void LoadQToL0B(uint32_t qsActBaseTileAlign16, uint32_t kCur, uint32_t qsActBaseTileAlign16L0,
                                      uint32_t scaleMAlign16, uint32_t scaleK, uint32_t qBufId, uint32_t qkL0abBufId,
                                      uint32_t kStart)
    {
        copyL1ToL0BMxQk(l0BTensor[qkL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                        l1BTensor[qBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                        l1BScaleTensor[qBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), qsActBaseTileAlign16,
                        kCur, qsActBaseTileAlign16L0, scaleMAlign16, scaleK, kStart);
    }

    // K → L0A（非转置，带 mx scale，n 方向子切分 + D 方向切段）
    __aicore__ inline void LoadKToL0A(uint32_t nSubRowStart, uint32_t nCur, uint32_t kCur,
                                      uint32_t kvsActBaseTileAlign16, uint32_t scaleK, uint32_t kBufId,
                                      uint32_t scaleBufId, uint32_t scaleRowStart, uint32_t qkL0abBufId,
                                      uint32_t kStart)
    {
        copyL1ToL0AMxQk(l0ATensor[qkL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                        l1ATensor[kBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                        l1AScaleTensor[scaleBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), nSubRowStart, nCur,
                        kCur, kvsActBaseTileAlign16, scaleK, kStart, scaleRowStart);
    }

    // C = K · Qᵀ；默认整段 D，initC 由调用方传入
    __aicore__ inline void MatmulQK(uint32_t n, uint32_t qsActBaseTileAlign16, uint32_t kCur, uint32_t qkL0aBufId,
                                    uint32_t qkL0bBufId, uint32_t qkL0cBufId, bool initC)
    {
        // FIA MatmulFullMX: Mmad 操作数必须是 mx_fp8_e4m3_t（与 LoadData L0 dest 一致），
        // 否则 cube 按普通 e4m3 乘加，S 在 fp16 fixpipe 溢出成 inf。
        auto l0ALayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0A>(n, kCur);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[qkL0aBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                                            l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0B>(kCur, qsActBaseTileAlign16);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[qkL0bBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                                            l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(n, qsActBaseTileAlign16);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[qkL0cBufId], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, n, qsActBaseTileAlign16, kCur, initC);
    }

    // L0C → UB fixpipe（NO_SPLIT 模式，subBlockId 选目标 AIV）
    template <class TensorUB>
    __aicore__ inline void FixpipeMm1(uint32_t qsActBaseTileAlign16, uint32_t n, TensorUB &ubSTensorTla,
                                      bool subBlockId, uint32_t qkL0cBufId)
    {
        auto l0CTensorTla =
            tla::MakeTensor(l0CTensor[qkL0cBufId], tla::MakeLayoutL0C(n, qsActBaseTileAlign16), Arch::PositionL0C{});

        using CopyL0CToUB =
            Tile::CopyL0CToUBTla<ArchTag, decltype(l0CTensorTla), TensorUB, Tile::CopyL0CToUBMode::NO_SPLIT,
                                 Tile::ScaleGranularity::PER_TENSOR, false>;
        CopyL0CToUB copyL0CToUB;
        copyL0CToUB(ubSTensorTla, l0CTensorTla, softmaxScale_, subBlockId);
    }

    // 发一次 nSub 的 fixpipe（含 M_FIX / FIX_M）。同 mxfp4：延后一拍，插在下一轮 Wait MTE1_M 前。
    template <class TensorUB>
    __aicore__ inline void EmitFixpipeMm1(uint32_t l0cBufId, uint32_t nSubRowStart, uint32_t nCur, uint32_t qsMmadN,
                                          TensorUB &ubSTensorTla, uint32_t subBlockIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(QK_L0C_EVENT0 + l0cBufId);
        uint32_t nCurEven = (nCur + 1) >> 1 << 1;
        auto ubSSubTla = GetTile(ubSTensorTla, tla::MakeCoord(nSubRowStart, 0), tla::MakeShape(nCurEven, qsMmadN));
        FixpipeMm1(qsMmadN, nCur, ubSSubTla, subBlockIdx != 0, l0cBufId);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0 + l0cBufId);
    }

    // P0b：只发 GM→L1（Q 首 tile + 每 tile K），不 Wait MTE2、不占 L0、不推进 kBufId。
    // kernel 在 Wait(V1_C1) 之前调用，让 K 的 MTE2 和 AIV softmax 重叠。
    template <class TensorQ, class TensorK, class TaskInfoT, class TileInfoT>
    __aicore__ inline void PrefetchK(TensorQ &gQTensorTla, TensorK &gKTensorTla,
                                     AscendC::GlobalTensor<uint8_t> gQDequantScale,
                                     AscendC::GlobalTensor<uint8_t> gKDequantScale,
                                     AscendC::GlobalTensor<int32_t> gSparseBlockIdx, TaskInfoT &curTaskInfo,
                                     TileInfoT &curTileInfo, uint32_t embed, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                     uint32_t kvHeadMul, uint32_t qHeadMul)
    {
        uint32_t qsActBaseTile = curTaskInfo.qsActBaseTile;
        uint32_t qsActBaseTileAlign128 = curTaskInfo.qsActBaseTileAlign128;
        uint32_t qsMmadN = (qsActBaseTileAlign128 != 0) ? qsActBaseTileAlign128 : curTaskInfo.qsActBaseTileAlign16;
        int64_t kvSeqlen = curTaskInfo.kvSeqlen;
        uint32_t yBlockNumAval = curTaskInfo.yBlockNumAval;
        uint32_t yBlockNumRsvd = curTaskInfo.yBlockNumRsvd;

        uint32_t kvsActBaseTile = curTileInfo.kvsActBaseTile;
        uint32_t gatheredKvSTileIdx = curTileInfo.pvGatheredKvSTileIdx;
        bool isFirstKvsTile = curTileInfo.isFirstKvsTile;
        bool isLastKvsTile = curTileInfo.isLastKvsTile;

        uint32_t scaleK = CeilDivision(embed, MXFP8::MX_GROUP_ELEMS);
        uint32_t l1QEventId = Q_EVENT0 + qBufId;

#if BSA_MXFP8_K_PAIR256
        // 奇 tile 复用偶 tile 已搬进 L1 的下半 128 行，不再发 MTE2。
        const bool pairOdd = (gatheredKvSTileIdx & 1u) != 0u;
        if (pairOdd) {
            return;
        }
        const bool pairEven = !isLastKvsTile;
        uint32_t gatherRows = kvsActBaseTile;
        if (pairEven) {
            gatherRows += MXFP8::Mxfp8NextKvsTileRows(gatheredKvSTileIdx, kvSBaseTile, curTaskInfo.gatheredKvSeqlen);
        }
        pairKSlot_ = kBufId;
        pairKL1Align16_ = MXFP8::Mxfp8Align16(gatherRows);
#else
        (void)isLastKvsTile;
        uint32_t gatherRows = kvsActBaseTile;
        pairKSlot_ = kBufId;
        pairKL1Align16_ = curTileInfo.kvsActBaseTileAlign16;
#endif
#if BSA_MXFP8_SCALE_BATCH4
        uint32_t scaleRows = 0;
        if ((gatheredKvSTileIdx % MXFP8::SCALE_BATCH_TILES) == 0u) {
            scaleRows = MXFP8::Mxfp8GatheredTilesRows(gatheredKvSTileIdx, MXFP8::SCALE_BATCH_TILES, kvSBaseTile,
                                                      curTaskInfo.gatheredKvSeqlen);
            pairKScaleSlot_ = pairKSlot_;
        }
#else
        uint32_t scaleRows = gatherRows;
        pairKScaleSlot_ = pairKSlot_;
#endif
        uint32_t l1KEventId = MXFP8::KV_EVENT0 + pairKSlot_;

        // 首个 sparse index 是阻塞式 GM 标量读，读回来之前本拍一条 MTE2 都发不出去。
        // 这里只取上一拍存下的值；真正的 GetValue 挪到 copy 发出之后（见函数末尾）。
        // task 切换时 gSparseBlockIdx 基址与 tile 序号一起重置，缓存作废，就地阻塞读一次。
        uint32_t firstYBlockIdx = (gatheredKvSTileIdx * kvSBaseTile) / blockShapeY;
#if BSA_MXFP8_IDX_PREFETCH
        uint32_t firstOriYBlockIdx =
            isFirstKvsTile ? static_cast<uint32_t>(gSparseBlockIdx.GetValue(firstYBlockIdx)) : nextOriYBlockIdx_;
#else
        uint32_t firstOriYBlockIdx = static_cast<uint32_t>(gSparseBlockIdx.GetValue(firstYBlockIdx));
#endif

        // Q 仍只在 task 首 KV tile 搬；一次 ~16KB，保持 copy 后立刻 Wait，不改 Q 双槽。
        if (isFirstKvsTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            if (qsActBaseTile < Q_M_PAD) {
                PadQL1Zero(qBufId);
            }
            auto l1QLayout = tla::MakeLayout<ElementA, layout::zN>(qsMmadN, embed);
            auto l1QTensorTla = tla::MakeTensor(l1BTensor[qBufId], l1QLayout, Arch::PositionL1{});
            CopyQGmToL1(qsActBaseTile, embed, gQTensorTla, l1QTensorTla);
            CopyQScaleGmToL1(qsActBaseTile, scaleK, gQDequantScale, qBufId, qHeadMul);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        auto l1KLayout = tla::MakeLayout<ElementA, layout::zN>(gatherRows, embed);
        auto l1KTensorTla = tla::MakeTensor(l1ATensor[pairKSlot_], l1KLayout, Arch::PositionL1{});
        SparseKFusedBaseTileL1FullLoad(gKTensorTla, l1KTensorTla, gKDequantScale, gSparseBlockIdx, gatheredKvSTileIdx,
                                       kvSeqlen, kvSBaseTile, blockShapeY, yBlockNumAval, yBlockNumRsvd, gatherRows,
                                       scaleRows, embed, pairKSlot_, pairKScaleSlot_, kvHeadMul, firstYBlockIdx,
                                       firstOriYBlockIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);
#if BSA_MXFP8_IDX_PREFETCH
        // K 的 copy 已在飞。P1 下次 gather 是下一偶 tile；标量读与 MTE2 重叠，
        // 剩余延迟压在紧接着的 ComputeQK（LoadQ/LoadK/mmad）底下。
        uint32_t nextTile = MXFP8::Mxfp8NextGatherTileIdx(gatheredKvSTileIdx);
        uint64_t nextStart = static_cast<uint64_t>(nextTile) * static_cast<uint64_t>(kvSBaseTile);
        if (nextStart < static_cast<uint64_t>(curTaskInfo.gatheredKvSeqlen)) {
            uint32_t nextYBlockIdx = static_cast<uint32_t>(nextStart / static_cast<uint64_t>(blockShapeY));
            nextOriYBlockIdx_ = static_cast<uint32_t>(gSparseBlockIdx.GetValue(nextYBlockIdx));
        }
#endif
#if BSA_MXFP8_K_PAIR256
        // 槽已占用到奇 tile LoadK 结束；立刻推进 kvBufId，让随后的 PV 用下一槽，避免和跨 softmax 的 K 对撞。
        kBufId = (kBufId + 1) % MXFP8::L1_KV_BUF_CNT;
#endif
    }

    // P0b：等本拍 K 的 MTE2 完成后 LoadK/mmad。最后一次 LoadK 立刻放 L1，不把槽占到 fixpipe。
    template <class TensorUB, class TaskInfoT, class TileInfoT>
    __aicore__ inline void ComputeQK(TensorUB &ubSTensorTla, TaskInfoT &curTaskInfo, TileInfoT &curTileInfo,
                                     uint32_t embed, uint32_t subBlockIdx)
    {
        uint32_t qsActBaseTileAlign128 = curTaskInfo.qsActBaseTileAlign128;
        uint32_t qsMmadN = (qsActBaseTileAlign128 != 0) ? qsActBaseTileAlign128 : curTaskInfo.qsActBaseTileAlign16;
        uint32_t kvsActBaseTile = curTileInfo.kvsActBaseTile;
        bool isLastKvsTile = curTileInfo.isLastKvsTile;
        uint32_t gatheredKvSTileIdx = curTileInfo.pvGatheredKvSTileIdx;

        uint32_t scaleK = CeilDivision(embed, MXFP8::MX_GROUP_ELEMS);
        uint32_t l1QEventId = Q_EVENT0 + qBufId;

#if BSA_MXFP8_K_PAIR256
        const bool pairOdd = (gatheredKvSTileIdx & 1u) != 0u;
        const bool pairEven = ((gatheredKvSTileIdx & 1u) == 0u) && !isLastKvsTile;
        uint32_t kSlot = pairKSlot_;
        uint32_t l1KAlign16 = pairKL1Align16_;
        uint32_t pairRowOff = pairOdd ? MXFP8::QK_HALF_KVS : 0;
#else
        const bool pairOdd = false;
        const bool pairEven = false;
        uint32_t kSlot = kBufId;
        uint32_t l1KAlign16 = curTileInfo.kvsActBaseTileAlign16;
        uint32_t pairRowOff = 0;
#endif
#if BSA_MXFP8_SCALE_BATCH4
        uint32_t scaleSlot = pairKScaleSlot_;
        uint32_t scaleRowStart = (gatheredKvSTileIdx % MXFP8::SCALE_BATCH_TILES) * MXFP8::QK_HALF_KVS;
        const bool scaleTail =
            ((gatheredKvSTileIdx % MXFP8::SCALE_BATCH_TILES) == (MXFP8::SCALE_BATCH_TILES - 1u)) || isLastKvsTile;
#else
        uint32_t scaleSlot = kSlot;
        uint32_t scaleRowStart = pairRowOff;
        const bool scaleTail = true;
#endif
        uint32_t l1KEventId = MXFP8::KV_EVENT0 + kSlot;

        if (!pairOdd) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);
        }

        // 一拍 QK 128×128（L0_TILE_N=128，满块 nLoopNum=1）；尾块 kvs<128 仍一次 mmad。
        uint32_t nLoopNum = CeilDivision(kvsActBaseTile, L0_TILE_N);
        bool fixPending = false;
        uint32_t pendL0cBufId = 0;
        uint32_t pendNSubRowStart = 0;
        uint32_t pendNCur = 0;
        constexpr uint32_t L0B_Q_SLOT = 0;
        for (uint32_t nSub = 0; nSub < nLoopNum; ++nSub) {
            uint32_t l0CEventId = QK_L0C_EVENT0 + qkL0cBufId;
            uint32_t nSubRowStart = nSub * L0_TILE_N + pairRowOff;
            uint32_t nCur = MinU32(L0_TILE_N, kvsActBaseTile - nSub * L0_TILE_N);
            uint32_t l0ABEventId = QK_L0AB_EVENT0 + qkL0abBufId;

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
            LoadQToL0B(qsMmadN, embed, qsMmadN, qsMmadN, scaleK, qBufId, L0B_Q_SLOT, 0);
            LoadKToL0A(nSubRowStart, nCur, embed, l1KAlign16, scaleK, kSlot, scaleSlot, scaleRowStart, qkL0abBufId, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
            // MTE1 队列：LoadK 完成后再放 L1，与 Cube Wait(MTE1_M) 同序，mmad 只用 L0。
            // P1 偶 tile 不放槽，留给奇 tile 的下半 LoadK。
            if (nSub + 1 == nLoopNum && !pairEven) {
                bool holdForScale = false;
#if BSA_MXFP8_SCALE_BATCH4
                holdForScale = (kSlot == scaleSlot) && !scaleTail;
#endif
                if (!holdForScale) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
#if !BSA_MXFP8_K_PAIR256
                    kBufId = (kBufId + 1) % MXFP8::L1_KV_BUF_CNT;
#endif
                }
#if BSA_MXFP8_SCALE_BATCH4
                if (scaleTail && scaleSlot != kSlot) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT0 + scaleSlot);
                }
#endif
            }

            if (fixPending) {
                EmitFixpipeMm1(pendL0cBufId, pendNSubRowStart, pendNCur, qsMmadN, ubSTensorTla, subBlockIdx);
                fixPending = false;
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
            MatmulQK(nCur, qsMmadN, embed, qkL0abBufId, L0B_Q_SLOT, qkL0cBufId, true);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
            qkL0abBufId = (qkL0abBufId + 1) % MXFP8::L0A_QK_BUF_CNT;

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            pendL0cBufId = qkL0cBufId;
            pendNSubRowStart = nSub * L0_TILE_N;
            pendNCur = nCur;
            fixPending = true;
            qkL0cBufId = (qkL0cBufId + 1) % MXFP8::L0C_QK_BUF_CNT;
        }
        if (fixPending) {
            EmitFixpipeMm1(pendL0cBufId, pendNSubRowStart, pendNCur, qsMmadN, ubSTensorTla, subBlockIdx);
        }

        if (unlikely(isLastKvsTile)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            qBufId = (qBufId + 1) % MXFP8::L1_Q_BUF_CNT;
        }
    }

    // 回退路径：Wait(V1_C1) 之后整段调用，等价于 PrefetchK + ComputeQK。
    template <class TensorQ, class TensorK, class TensorUB, class TaskInfoT, class TileInfoT>
    __aicore__ inline void operator()(TensorQ &gQTensorTla, TensorK &gKTensorTla,
                                      AscendC::GlobalTensor<uint8_t> gQDequantScale,
                                      AscendC::GlobalTensor<uint8_t> gKDequantScale,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, TensorUB &ubSTensorTla,
                                      TaskInfoT &curTaskInfo, TileInfoT &curTileInfo, uint32_t embed,
                                      uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t subBlockIdx,
                                      uint32_t kvHeadMul, uint32_t qHeadMul)
    {
        PrefetchK(gQTensorTla, gKTensorTla, gQDequantScale, gKDequantScale, gSparseBlockIdx, curTaskInfo, curTileInfo,
                  embed, kvSBaseTile, blockShapeY, kvHeadMul, qHeadMul);
        ComputeQK(ubSTensorTla, curTaskInfo, curTileInfo, embed, subBlockIdx);
    }

protected:
    AscendC::LocalTensor<ElementB> l1ATensor[MXFP8::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementA> l1BTensor[MXFP8::L1_Q_BUF_CNT];
    AscendC::LocalTensor<ElementB> l0ATensor[MXFP8::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<ElementA> l0BTensor[MXFP8::L0B_QK_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[MXFP8::L0C_QK_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[MXFP8::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[MXFP8::L1_Q_BUF_CNT];

    Tile::CopyL1ToL0AMxFp8QKA5 copyL1ToL0AMxQk;
    Tile::CopyL1ToL0BMxFp8QKA5 copyL1ToL0BMxQk;
    TileMmad tileMmad;
    Tile::CopyGmToL1MxScaleDn2NzA5 copyGmToL1MxScaleDn2Nz;

    uint32_t qBufId = 0;
    uint32_t &kBufId;
    // 上一拍 PrefetchK 末尾预读的、同 task 内下次 gather 的首个 sparse index
    uint32_t nextOriYBlockIdx_ = 0;
    uint32_t qkL0abBufId = 0;
    uint32_t qkL0cBufId = 0;
    uint64_t softmaxScale_;
    // P1：偶 tile gather 所用的 L1 槽和整槽行对齐，奇 tile LoadK 复用。
    uint32_t pairKSlot_ = 0;
    uint32_t pairKL1Align16_ = 0;
    uint32_t pairKScaleSlot_ = 0;
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_QK_ARCH35_MXFP8_HPP
