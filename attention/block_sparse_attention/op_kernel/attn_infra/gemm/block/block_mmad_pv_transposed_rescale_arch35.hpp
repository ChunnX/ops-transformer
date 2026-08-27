/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief matmul implementation for single transposed p&v base tile, with rescale fused on CUBE
 * This implementation is designed for the following senario (DN + FP8, rescaleOnCube on cube):
 * A full p base tile is loaded to L1 from UB(softmax), no workspace transit
 * A full v base tile is loaded to L1 from GM，relavent instructions launched before p base tile crossCore wait
 * PV is computed in transposed form: C' = (V)^T * (P)^T, C' is [D, qS] on L0C (split n by subcore: [D, qS/2])
 * rescale is fused as: C' += (GO_prev)^T * diag(dm·vS_{t-1}/vS_t), accumulated on the same L0C tile
 */
#ifndef GEMM_BLOCK_PV_TRANSPOSED_RESCALE_ARCH35_HPP
#define GEMM_BLOCK_PV_TRANSPOSED_RESCALE_ARCH35_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_arch35_utils.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad_rescale.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

// partial specialization for performing (V)^T * (P)^T = C' [D, qS], fused with rescale
template <
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementS_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5BsaPV<true, true>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementS_,
    ElementBias_,
    TileCopy_,
    TileMmad_>
{
public:
    using DispatchPolicy = MmadAtlasA5BsaPV<true, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementS = ElementS_;

    using TileMmad = TileMmad_;
    using TileMmadRescale = Gemm::Tile::TileMmadRescaleTla<ArchTag>;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;      // fp8 V^T
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;      // fp8 P^T

    using TileCopyRescale = Gemm::Tile::PackedTileCopyTlaToUB<
        NpuArch::Arch::AtlasA5, float, NpuArch::layout::zN, float, NpuArch::layout::zN,
        float, NpuArch::layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
        Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using CopyL1ToL0ARescale = typename TileCopyRescale::CopyL1ToL0A;  // float GO^T
    using CopyL1ToL0BRescale = typename TileCopyRescale::CopyL1ToL0B;  // float DIAGDM

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_BUF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_HALF_BUF_SIZE / L0_STAGES;

    static constexpr uint32_t DIAG_BUF_NUM = 3;
    static constexpr uint32_t SUB_BLOCK_NUM = 2;
    static constexpr uint32_t DIAG_MAX_ROW_NUM_PER_SUB_BLOCK = 64;
    static constexpr uint32_t DIAG_PACKED_COL_NUM = 16;
    static constexpr uint32_t DIAG_ELE_NUM_PER_SUB_BLOCK =
        DIAG_MAX_ROW_NUM_PER_SUB_BLOCK * DIAG_PACKED_COL_NUM;
    static constexpr uint32_t DIAG_PACKED_ROW_NUM_PER_STAGE =
        SUB_BLOCK_NUM * DIAG_MAX_ROW_NUM_PER_SUB_BLOCK;
    static constexpr uint32_t DIAG_MMAD_TILE_SIZE = 16;
    static constexpr uint32_t DIAG_MMAD_TILE_NUM_PER_SUB_BLOCK =
        DIAG_MAX_ROW_NUM_PER_SUB_BLOCK / DIAG_MMAD_TILE_SIZE;
    static constexpr bool rescaleOnCube = DispatchPolicy::rescaleOnCube;
    // GO^T 的 L1 写完成事件（FIX_M 的空闲 id，l0C 释放用 0-3）：
    // 中间块 DataCopy（CO1→C1）后 set、非首块 RE 读 GO^T 前 wait——严格交替，单 id 即可。
    // DataCopy（CO1→C1）与 Fixpipe 同属 CO1 搬出引擎（FIX 域，共用 Fixpipe 寄存器体系），
    // 完成事件仍用 FIX_MTE1（FIX→MTE1：CO1 写 L1 完成 → MTE1 可读 L1）
    static constexpr uint32_t GO_READY_EVENT_ID = 4;

    __aicore__ inline
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart,
                 uint32_t l1DiagAddrStart,
                 Mm2L1TileHelper &mm2L1TileHelper)
    {
        l1ATileM = mm2L1TileHelper.mm2L1TileN;
        l1BTileN = mm2L1TileHelper.mm2L1TileM;
        l1ATileK = mm2L1TileHelper.mm2L1TileKRight;
        l1BTileK = mm2L1TileHelper.mm2L1TileKLeft;
        l1ABufNum = mm2L1TileHelper.vL1BufNum;
        l1BBufNum = mm2L1TileHelper.pL1BufNum;

        uint32_t l1VAddrStart = l1BufAddrStart +
            mm2L1TileHelper.mm2L1TileM * mm2L1TileHelper.mm2L1TileKLeft * sizeof(ElementB) * mm2L1TileHelper.pL1BufNum;
        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1VAddrStart + l1ATileK * l1ATileM * sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        for (uint32_t i = 0; i < DIAG_BUF_NUM; i++) {
            l1DiagTensor[i] = resource.l1Buf.template GetBufferByByte<float>(
                l1DiagAddrStart +
                SUB_BLOCK_NUM * DIAG_ELE_NUM_PER_SUB_BLOCK * sizeof(float) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(
                L0B_PINGPONG_BUF_SIZE * i);
            l0ARescaleTensor[i] = resource.l0ABuf.template GetBufferByByte<float>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0BRescaleTensor[i] = resource.l0BBuf.template GetBufferByByte<float>(
                L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_HALF_BUF_SIZE + L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline
    ~BlockMmadTla() {}

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    __aicore__ inline
    uint32_t GetCurLoopCounter(uint32_t outterLoopItr, uint32_t curLoopNum, uint32_t curLoopItr)
    {
        return outterLoopItr * curLoopNum + curLoopItr;
    }

    template <class TensorB, class TensorL1B>
    __aicore__ inline
    void SparseVBaseTileL1FullLoad(TensorB &gBTensor, TensorL1B &l1BTensorTla,
                                   AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                   uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                                   uint32_t kvSBaseTile, uint32_t blockShapeY,
                                   uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                                   uint32_t curBaseTileSize, uint32_t embed)
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorB>;
        CopyGmToL1A copyGmToL1A;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset = baseTileStartOffset + curBaseTileSize;
        // 稀疏情况下对实际选中的部分gather后进行基本块切分
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = blockShapeY;
            if (oriYBlockIdx == yBlockNumAval - 1) {
                curYBlockSize = kvSeqlen - oriYBlockIdx * blockShapeY;
            }
            uint32_t gatheredEndOffset =
                min(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                    baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) {
                break;
            }
            auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                tla::MakeCoord(0, dealtLenAccum), tla::MakeShape(embed, curDealtLen));
            auto gBTensorTlaTile = GetTile(gBTensor,
                tla::MakeCoord(0, oriStartOffset), tla::MakeShape(embed, curDealtLen));
            copyGmToL1A(l1BTensorTlaTile, gBTensorTlaTile);
            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < curBaseTileSize) {
                // 防止最后一次循环之后对GM的访问越界引发硬件error
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
            }
        }
    }

    template <class TensorB, class TensorC, class TensorGo, class TensorGoC1>
    __aicore__ inline
    void operator()(TensorB &gBTensor, TensorC &ubCTensor, TensorGo &l1GoTensorTla, TensorGoC1 &l1GoTensorTlaC1,
                    AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                    GemmCoord actualOriShape,
                    uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                    uint32_t kvSBaseTile, uint32_t blockShapeY,
                    uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                    uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
                    Arch::CrossCoreFlag smToMm2Flag, Arch::CrossCoreFlag mm2ToReFlag,
                    bool isFirstKvSTile, bool isLastKvSTile,
                    uint64_t deqScalar = 0)
    {
        uint32_t rowNum = actualOriShape[0];        // qS
        uint32_t embed = actualOriShape[1];         // D
        uint32_t curBaseTileSize = actualOriShape[2];  // kvS

        uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;
        uint32_t l1BBufId = gatheredKvSTileIdx % l1BBufNum;
        uint32_t l1VEventId = l1ABufId + 3;
        uint32_t l1DiagBufId = gatheredKvSTileIdx % DIAG_BUF_NUM;

        // V gather 落 L1：ColumnMajor [D, kvS]（nZ@A1，GM->L1 用 ColumnMajor->nZ 特化，
        // 与 block_mmad_k_qt_st_arch35.hpp 的 Q 加载同构）；A=(V)^T 直接读同一 nZ 视图
        // 注：文件内布局 tag 全部硬编码，不依赖 TileCopy 的 L1A/L1B 推导（interface 保持原样）
        auto l1ALayoutTla = tla::MakeLayout<ElementA, NpuArch::layout::nZ>(embed, curBaseTileSize);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1VEventId);
        SparseVBaseTileL1FullLoad(
            gBTensor, l1ATensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embed);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1VEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1VEventId);
        // fwd crossCoreSync from online sm to mm2
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        // B = (P)^T [kvS, qS]：P 在 L1 从 [kvS,qS] 视角就是 zN，声明 zN 走 copyL1ToL0B zN→nZ 特化（与 B=V 同构）
        auto l1BLayoutTla = tla::MakeLayout<ElementB, NpuArch::layout::zN>(curBaseTileSize, rowNum);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});

        // GO^T [D, qS] fp32（kernel 构造的 zN TLA 传入，与末块 go 的 TLA 传入模式一致）
        // diag 双子核槽位：块 0 在 0、块 1 在 64×64 float 偏移处。槽位容量固定，但槽位内
        // 实际数据由 S_trans 按 RoundUp(实际子核列数, 16)×同宽的 zN 布局紧凑写入。

        // PV^T 与 RE 共用的 L0 分块数：M=D、N=qS、K=kvS（RE 的 GO 沿 M 方向同 D 切分）
        uint32_t mL0LoopNum = CeilDiv(embed, L0_TILE_M);
        uint32_t nL0LoopNum = CeilDiv(rowNum, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_K);
        uint32_t l0CBufId = gatheredKvSTileIdx % L0_STAGES;
        uint32_t l0CEventId = l0CBufId + 2;
        // L0C：声明为 [D, qS] 整块（CUBE 单实例 mmad 全量计算）
        auto l0CLayoutTla = tla::MakeLayoutL0C(embed, rowNum);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});

        // PV^T：C [D, qS] = (V)^T [D, kvS] * (P)^T [kvS, qS]，M=D、N=qS、K=kvS 三重 L0 分块（全量，无子核分摊）
        for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
            uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (embed - mL0Itr * L0_TILE_M) : L0_TILE_M;
            for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (rowNum - nL0Itr * L0_TILE_N) : L0_TILE_N;
                // l0C 同一 buffer 内子块（bias 保留给 RE 累加，不按 chunk 换 buffer）
                auto l0CTensorTlaTile = GetTile(l0CTensorTla,
                    tla::MakeCoord(mL0Itr * L0_TILE_M, nL0Itr * L0_TILE_N), tla::MakeShape(l0TileMAct, l0TileNAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                    uint32_t l0TileKAct = (kL0Itr == kL0LoopNum - 1) ?
                        (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ALoopCounter = prefixSumL0AStages + GetCurLoopCounter(mL0Itr, kL0LoopNum, kL0Itr);
                    uint32_t l0BLoopCounter = prefixSumL0BStages + GetCurLoopCounter(nL0Itr, kL0LoopNum, kL0Itr);
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;

                    // A 块 = (V)^T [D, kvS] 的 [D 块, kvS 块]：ColumnMajor 视图的 M/K 方向分块
                    auto l1ATensorTlaTile = GetTile(l1ATensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K), tla::MakeShape(l0TileMAct, l0TileKAct));
                    auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1VEventId);
                    }

                    // B 块 = (P)^T [kvS, qS] 的 [kvS 块, qS 块]：zN 视图的 K/N 方向分块
                    auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                        tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N), tla::MakeShape(l0TileKAct, l0TileNAct));
                    auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    // reverse crossCoreSync for P：最后一次 copyL1ToL0B 已消费 L1 P 缓冲，VEC 才能覆写
                    // （与旧 PV 的 "L1 P buffer free" 方向对齐；softmax epilogue 在 P→L1 DataCopy 前
                    // wait 同一 flag，缺此 set 会死锁）
                    if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                    }

                    // M 侧：等 MTE1 数据就绪（MTE1_M）再 mmad——与旧 PV/QK 惯例一致
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    if ((mL0Itr == 0) && (nL0Itr == 0) && (kL0Itr == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(
                        l0CTensorTlaTile,
                        l0ATensorTla,
                        l0BTensorTla,
                        RoundUp(l0TileMAct, 16),
                        l0TileNAct,
                        l0TileKAct,
                        (kL0Itr == 0));
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                }
            }
        }

        // rescale：GO^T 按双子核负责的 Q 行数分批搬到 L0A；diag 从双子核 L1 槽位中抽取 16×16
        // 对角块，紧凑打包到 L0B，避免 qS>8 时把 64×64 父布局 stride 当成紧凑子矩阵 stride。
        // mmad 按 16 列块分开乘（K=16 不浪费计算；diag 块对角 → 批间交叉项为 0）
        // 首块（含单块）不涉及 rescale
        if (!isFirstKvSTile) {
            // GO^T 由上一块 FixPipe(CFG_NZ→L1) 写入：等 L1 写完成后再读（独立 event id，与 l0C 释放的 FIX_M 区分）
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(GO_READY_EVENT_ID);
            // diag 就绪（双向 token 的 ready 方向）：等 VEC 把本轮 diag(dm/vScale) 写完再读，
            // 避免读到旧一轮或半成品；同一 flag 在 RE 末尾再 set 回 consumed 给 VEC 覆写前 wait
            Arch::CrossCoreFlag diagFlag(6 + l1DiagBufId);
            WaitCrossCoreSync<4, PIPE_MTE1>(diagFlag);
            // 尾块 qS 非 16 倍数时向上取整：最后一块使用实际列数（mmad 的 k/n 用实际值，与 PV^T 的
            // K 尾块同惯例；k 对齐填充读的 B 行在 C 的列区内为 0——对角矩阵保证填充贡献恒为 0，无污染）
            // 使用与 S_trans::nCopyOffset 完全相同的双子核切分，保证每个 subblock 与 L1 diag 槽位一一对应。
            uint32_t qRowNumPerSubBlock = RoundUp(rowNum, 64) / SUB_BLOCK_NUM;
            uint32_t activeSubBlockNum = CeilDiv(rowNum, qRowNumPerSubBlock);  // 1 或 2
            // diag 的 L0B stage 固定 1 次：最多 2×4 个紧凑 16×16 块，两批共用。
            uint32_t l0BLoopCounter = prefixSumL0BStages + nL0LoopNum * kL0LoopNum;
            uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
            uint32_t l0BEventId = l0BBufId + 2;
            // Copy the complete fixed 8KB stage in one LoadData. Unused tile slots are
            // transferred as well, but the MMAD loop only selects valid group/tile indices.
            auto l1DiagStageLayoutTla = tla::MakeLayout<float, NpuArch::layout::zN>(
                DIAG_PACKED_ROW_NUM_PER_STAGE, DIAG_PACKED_COL_NUM);
            auto l1DiagStageTensorTla = tla::MakeTensor(
                l1DiagTensor[l1DiagBufId], l1DiagStageLayoutTla,
                tla::MakeCoord(0U, 0U), Arch::PositionL1{});
            auto l0BDiagStageLayoutTla = tla::MakeLayout<float, LayoutTagL0B>(
                DIAG_PACKED_ROW_NUM_PER_STAGE, DIAG_PACKED_COL_NUM);
            auto l0BDiagStageTensorTla = tla::MakeTensor(
                l0BRescaleTensor[l0BBufId], l0BDiagStageLayoutTla,
                tla::MakeCoord(0U, 0U), Arch::PositionL0B{});

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
            CopyL1ToL0BRescale copyL1ToL0BRescale;
            copyL1ToL0BRescale(l0BDiagStageTensorTla, l1DiagStageTensorTla);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

            for (uint32_t diagSubBlockIdx = 0; diagSubBlockIdx < activeSubBlockNum; diagSubBlockIdx++) {
                uint32_t subBlockColStart = diagSubBlockIdx * qRowNumPerSubBlock;
                uint32_t subBlockColEnd = (diagSubBlockIdx == activeSubBlockNum - 1) ?
                    rowNum : (diagSubBlockIdx + 1) * qRowNumPerSubBlock;
                uint32_t subBlockColNum = subBlockColEnd - subBlockColStart;
                uint32_t diagTileNum = CeilDiv(subBlockColNum, DIAG_MMAD_TILE_SIZE);
                // A 大搬（先行循环，mL0 流水预取）：(GO_prev)^T 的 [D 块, 批列宽] 连续 → L0A zN 恒等（qS=128 时 32KB 满）
                for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                    uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (embed - mL0Itr * L0_TILE_M) : L0_TILE_M;
                    // RE 的 stage 接在 PV^T 之后：A 每 (diagSubBlockIdx, mL0) 一次（PV^T A 总 = mL0×kL0）
                    uint32_t l0ALoopCounter = prefixSumL0AStages +
                        mL0LoopNum * kL0LoopNum + diagSubBlockIdx * mL0LoopNum + mL0Itr;
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    auto l1GoTensorTlaTile = GetTile(l1GoTensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, subBlockColStart),
                        tla::MakeShape(l0TileMAct, subBlockColNum));
                    auto l0ALayoutTla = tla::MakeLayout<float, LayoutTagL0A>(l0TileMAct, subBlockColNum);
                    auto l0ATensorTla = tla::MakeTensor(l0ARescaleTensor[l0ABufId],
                        l0ALayoutTla, Arch::PositionL0A{});
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    CopyL1ToL0ARescale copyL1ToL0ARescale;
                    copyL1ToL0ARescale(l0ATensorTla, l1GoTensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                }
                // MMAD still consumes one 16x16 diagonal tile at a time.
                for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                    uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (embed - mL0Itr * L0_TILE_M) : L0_TILE_M;
                    uint32_t l0ALoopCounter = prefixSumL0AStages +
                        mL0LoopNum * kL0LoopNum + diagSubBlockIdx * mL0LoopNum + mL0Itr;
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    auto l0ALayoutTla = tla::MakeLayout<float, LayoutTagL0A>(l0TileMAct, subBlockColNum);
                    auto l0ATensorTla = tla::MakeTensor(l0ARescaleTensor[l0ABufId],
                        l0ALayoutTla, Arch::PositionL0A{});
                    // M 侧：等 MTE1 数据就绪（MTE1_M）再 mmad——与旧 PV/QK 惯例一致
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    if ((diagSubBlockIdx == 0) && (mL0Itr == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    }
                    for (uint32_t diagTileIdx = 0; diagTileIdx < diagTileNum; diagTileIdx++) {
                        uint32_t diagTileColNum = (diagTileIdx == diagTileNum - 1) ?
                            (subBlockColNum - diagTileIdx * DIAG_MMAD_TILE_SIZE) : DIAG_MMAD_TILE_SIZE;
                        // C columns are global and A columns are group-local. B is selected
                        // from the packed L0B stage parent by its fixed group-slot index.
                        auto l0CRescaleTile = GetTile(l0CTensorTla,
                            tla::MakeCoord(mL0Itr * L0_TILE_M,
                                subBlockColStart + diagTileIdx * DIAG_MMAD_TILE_SIZE),
                            tla::MakeShape(l0TileMAct, diagTileColNum));
                        auto l0AGoTile = GetTile(l0ATensorTla,
                            tla::MakeCoord(mL0Itr * L0_TILE_M, diagTileIdx * DIAG_MMAD_TILE_SIZE),
                            tla::MakeShape(l0TileMAct, diagTileColNum));
                        uint32_t diagPackedTileIdx =
                            diagSubBlockIdx * DIAG_MMAD_TILE_NUM_PER_SUB_BLOCK + diagTileIdx;
                        auto l0BDiagTile = GetTile(l0BDiagStageTensorTla,
                            tla::MakeCoord(diagPackedTileIdx * DIAG_MMAD_TILE_SIZE, 0U),
                            tla::MakeShape(DIAG_MMAD_TILE_SIZE, DIAG_MMAD_TILE_SIZE));
                        tileMmadRescale(
                            l0CRescaleTile,
                            l0AGoTile,
                            l0BDiagTile,
                            RoundUp(l0TileMAct, 16),
                            diagTileColNum,
                            diagTileColNum,
                            false);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    if ((diagSubBlockIdx == activeSubBlockNum - 1) && (mL0Itr == mL0LoopNum - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    }
                    // diag 消费完成（softmax 覆写同 buffer 前 wait）：最后一块 mmad 后 set（读已完成，单实例单发）
                    if ((diagSubBlockIdx == activeSubBlockNum - 1) && (mL0Itr == mL0LoopNum - 1)) {
                        // 同一 flag set 回 consumed：VEC 下一轮覆写同 buffer 前 wait
                        SetCrossCoreSync<4, PIPE_MTE1>(diagFlag);
                    }
                }
            }
        }

        // fixpipe
        // 注：FIX_M 的 L0C 释放等待已在 mmad 循环（首个 chunk (mL0Itr==0 && nL0Itr==0 && kL0Itr==0)）执行，
        // 此处不重复 wait（双 wait 会消费两个 token 而死锁）
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
        if (isLastKvSTile) {
            // 末块（含单块）：NZ2DN 反量化搬 UB go，倒置再倒置回 ND [qS, D]，VEC 做 ÷gl/cast/写 GM
            // 单目标（dualDstCtl=0）：每子核手动调一次，搬自己的列半区 [D, qS/2]（双目标模式不支持随路量化）
            // ubCTensor 由 kernel 构造为 ColumnMajor（DN）布局的 TLA tensor，
            // CopyL0CToDst<TensorC> 按 dst 类型里的 layout tag 匹配 NZ2DN 特化（与现有 PV 同模式）
            using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
            // reverse crossCoreSync, do fixPipe only after ubCTensor is fully released
            WaitCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            // 单目标（dualDstCtl=0）显式两次调用（Sub0/Sub1，与现有 MM1/PV 的 FixPipe 模式一致）：
            // 各搬 [D, qS/2] 半区到对应 AIV 的 UB（subBlockId 0/1 路由），双目标模式不支持随路量化
            // 半区 = Q 行数按 64 对齐后 /2；qS=128 时为 64，与 kernel go UB [embed_, 64] 一致。
            uint32_t qRowNumPerSubBlock = RoundUp(rowNum, 64) / SUB_BLOCK_NUM;
            auto l0CTensorTlaTileSub0 = GetTile(l0CTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(embed, qRowNumPerSubBlock));
            auto l0CTensorTlaTileSub1 = GetTile(l0CTensorTla,
                tla::MakeCoord(0, qRowNumPerSubBlock), tla::MakeShape(embed, qRowNumPerSubBlock));
            CopyL0CToDst copyL0CToDst;
            copyL0CToDst(ubCTensor, l0CTensorTlaTileSub0, deqScalar, 0);
            copyL0CToDst(ubCTensor, l0CTensorTlaTileSub1, deqScalar, 1);
            // crossCoreSync after all fixPipe move
            SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
        } else {
            // 首/中间块：NZ 保形落 L1 作为下一块 GO^T（倒置状态保持，零转换，raw 域——比值记账在 softmax 侧）
            // 通路：DataCopy（CO1→C1，channelSplit）——950 的 L0C fractal = 16×16 元素（fp32 64B/C0），
            // L1 标准 NZ = 32B/C0（fp32 16×8）；Fixpipe 的 channelSplit 仅 L0C→GM 有效，L0C→L1 无法做
            // 该 16×16→16×8 切分；DataCopyCO12Dst 是唯一正确通路（dst 位置标签必须为 C1）。
            // 写侧 C1 视图（kernel 传入的 l1GoTensorTlaC1，与 RE 读侧 A1 视图同一物理 L1，零拷贝双视图）
            using CopyL0CToL1 = typename TileCopy::template CopyL0CToL1<TensorGoC1>;
            CopyL0CToL1 copyL0CToL1;
            // deqScalar 在 CO1→C1 通路不生效（保形，比值记账在 softmax 侧），不传
            copyL0CToL1(l1GoTensorTlaC1, l0CTensorTla);
            // GO^T L1 写完成 → 非首块 RE 读前 wait（只能在此分支 set：末块 NZ2DN→UB 与 GO^T 无关，
            // 若统一在尾部 set，跨 task 时 RE 会 wait 到无关的 UB 写完成而提前通过）
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(GO_READY_EVENT_ID);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];   // V 段（gather 写入，A=(V)^T 读）
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];   // P 段（softmax 写入，B=(P)^T 读）
    AscendC::LocalTensor<float> l1DiagTensor[DIAG_BUF_NUM];    // packed zN [128,16], 8KB per stage
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    // RE 段：GO^T/diag 为 fp32，需 float 的 L0 buffer（PV^T 的 fp8 buffer 元素类型不匹配 float 布局）；
    // 物理复用同一 L0A/L0B 内存（RE 块 ≤ PV^T 块大小，同 stage 不同时使用）
    AscendC::LocalTensor<float> l0ARescaleTensor[L0_STAGES];
    AscendC::LocalTensor<float> l0BRescaleTensor[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES];

    TileMmad tileMmad;
    TileMmadRescale tileMmadRescale;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;
};
////////////////////////////////////////////////////////////////////

}  // namespace NpuArch::Gemm::Block
#endif
