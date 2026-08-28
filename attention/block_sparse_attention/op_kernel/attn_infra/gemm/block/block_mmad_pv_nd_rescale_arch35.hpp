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
 * @brief ND P*V matmul with historical-output rescaling on CUBE.
 */
#ifndef GEMM_BLOCK_PV_ND_RESCALE_ARCH35_HPP
#define GEMM_BLOCK_PV_ND_RESCALE_ARCH35_HPP

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
    MmadAtlasA5BsaPV<false, true>,
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
    using DispatchPolicy = MmadAtlasA5BsaPV<false, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementS = ElementS_;

    using TileMmad = TileMmad_;
    using TileMmadRescale = Gemm::Tile::TileMmadRescaleTla<ArchTag>;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileCopyRescale = Gemm::Tile::PackedTileCopyTlaToUB<
        NpuArch::Arch::AtlasA5, float, NpuArch::layout::zN, float, NpuArch::layout::zN,
        float, NpuArch::layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
        Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using CopyL1ToL0ARescale = typename TileCopyRescale::CopyL1ToL0A;
    using CopyL1ToL0BRescale = typename TileCopyRescale::CopyL1ToL0B;

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
    static constexpr bool rescaleOnCube = DispatchPolicy::rescaleOnCube;

    static constexpr uint32_t DIAG_BUF_NUM = 3;
    static constexpr uint32_t SUB_BLOCK_NUM = 2;
    static constexpr uint32_t DIAG_ROWS_PER_SUB_BLOCK = 64;
    static constexpr uint32_t DIAG_PACKED_COLS = 16;
    static constexpr uint32_t DIAG_STAGE_PACKED_ROWS = SUB_BLOCK_NUM * DIAG_ROWS_PER_SUB_BLOCK;
    static constexpr uint32_t DIAG_MMAD_TILE_SIZE = 16;
    static constexpr uint32_t DIAG_MMAD_TILES_PER_SUB_BLOCK =
        DIAG_ROWS_PER_SUB_BLOCK / DIAG_MMAD_TILE_SIZE;
    static constexpr uint32_t GO_READY_EVENT_ID = 4;

    __aicore__ inline
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart, uint32_t l1DiagAddrStart,
                 Mm2L1TileHelper &mm2L1TileHelper)
    {
        l1ATileM = mm2L1TileHelper.mm2L1TileM;
        l1BTileN = mm2L1TileHelper.mm2L1TileN;
        l1ATileK = mm2L1TileHelper.mm2L1TileKLeft;
        l1BTileK = mm2L1TileHelper.mm2L1TileKRight;
        l1ABufNum = mm2L1TileHelper.pL1BufNum;
        l1BBufNum = mm2L1TileHelper.vL1BufNum;
        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1BufAddrStart + l1ATileM * l1ATileK * sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum +
                l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(
                L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_HALF_BUF_SIZE + L0C_PINGPONG_BUF_SIZE * i);
            l0ARescaleTensor[i] = resource.l0ABuf.template GetBufferByByte<float>(L0A_PINGPONG_BUF_SIZE * i);
            l0BRescaleTensor[i] = resource.l0BBuf.template GetBufferByByte<float>(L0B_PINGPONG_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < DIAG_BUF_NUM; i++) {
            l1DiagTensor[i] = resource.l1Buf.template GetBufferByByte<float>(
                l1DiagAddrStart + DIAG_STAGE_PACKED_ROWS * DIAG_PACKED_COLS * sizeof(float) * i);
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
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset = baseTileStartOffset + curBaseTileSize;
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
                tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            auto gBTensorTlaTile = GetTile(gBTensor,
                tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < curBaseTileSize) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
            }
        }
    }

    template <class TensorB, class TensorC, class TensorGo, class TensorGoC1>
    __aicore__ inline
    void operator()(TensorB &gBTensor, TensorC &ubCTensor, TensorGo &l1GoTensorTla,
             TensorGoC1 &l1GoTensorTlaC1,
             AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
             GemmCoord actualOriShape,
             uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
             uint32_t kvSBaseTile, uint32_t blockShapeY,
             uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
             uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
             Arch::CrossCoreFlag smToMm2Flag, Arch::CrossCoreFlag mm2ToReFlag,
             bool isFirstKvSTile, bool isLastKvSTile, uint64_t deqScalar = 0)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;

        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t curBaseTileSize = actualOriShape[2];

        uint32_t l1BBufId = gatheredKvSTileIdx % l1BBufNum;
        uint32_t l1BEventId = l1BBufId + 3;
        uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;

        auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(curBaseTileSize, embed);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, curBaseTileSize);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
        // load V full base tile to L1 before crossCoreSync
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
        SparseVBaseTileL1FullLoad(
            gBTensor, l1BTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embed);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        // fwd crossCoreSync from online sm to mm2
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        // P full base tile already on L1
        uint32_t mL0LoopNum = CeilDiv(rowNum, L0_TILE_M);
        uint32_t nL0LoopNum = CeilDiv(embed, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_K);
        // while splitting the base tile OTmp to 2 AIVs,
        // the order of the elements in each column is expected to be preserved,
        // which means a column in l0C cannot be chunked and processed by dualMode FixPipe seperately.
        // therefore, FixPipe won't launch until each portion(chunked only by columns, based on nbuffer strategy)
        // of the base tile is ready on l0C
        for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
            uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (embed - nL0Itr * L0_TILE_N) : L0_TILE_N;
            uint32_t nLoopCounter = GetCurLoopCounter(gatheredKvSTileIdx, nL0LoopNum, nL0Itr);
            // l0C nbuffer chunked only in n loop
            uint32_t l0CLoopCounter = nLoopCounter;
            uint32_t l0CBufId = l0CLoopCounter % L0_STAGES;
            // uint32_t l0CEventId = l0CBufId;
            uint32_t l0CEventId = l0CBufId + 2;
            auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, l0TileNAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});
            for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                // uint32_t mLoopCounter = GetCurLoopCounter(gatheredKvSTileIdx, mL0LoopNum, mL0Itr);
                // different m chunks will be concated in the same piece of l0C buffer
                auto l0CTensorTlaTile = GetTile(l0CTensorTla,
                    tla::MakeCoord(mL0Itr * L0_TILE_M, 0), tla::MakeShape(l0TileMAct, l0TileNAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                    uint32_t l0ALoopCounter = prefixSumL0AStages + GetCurLoopCounter(mL0Itr, kL0LoopNum, kL0Itr);
                    uint32_t l0BLoopCounter = prefixSumL0BStages + GetCurLoopCounter(nL0Itr, kL0LoopNum, kL0Itr);
                    uint32_t l0TileKAct = (kL0Itr == kL0LoopNum - 1) ?
                        (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    // l0ABufId = (mLoopCounter % 2 == 0) ? (1 - l0ABufId) : l0ABufId;
                    uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;
                    // when L0B buffers wouldn't be reused across the k loop
                    // redundant L0B load caused by m loop can be avoided
                    auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                        tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N), tla::MakeShape(l0TileKAct, l0TileNAct));
                    auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                    if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                    }

                    auto l1ATensorTlaTile = GetTile(l1ATensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K), tla::MakeShape(l0TileMAct, l0TileKAct));
                    auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    // reverse crossCoreSync for P
                    if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                    }

                    bool initMmad = (kL0Itr == 0);
                    uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(
                        l0CTensorTlaTile,
                        l0ATensorTla,
                        l0BTensorTla,
                        l0TileMAligned,
                        l0TileNAct,
                        l0TileKAct,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                }
            }
            if (!isFirstKvSTile) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(GO_READY_EVENT_ID);
                    uint32_t l1DiagBufId = gatheredKvSTileIdx % DIAG_BUF_NUM;
                    Arch::CrossCoreFlag diagFlag(6 + l1DiagBufId);
                    WaitCrossCoreSync<4, PIPE_MTE1>(diagFlag);

                    uint32_t diagL0ALoopCounter =
                        prefixSumL0AStages + mL0LoopNum * kL0LoopNum;
                    uint32_t diagL0ABufId = diagL0ALoopCounter % L0_STAGES;
                    uint32_t diagL0AEventId = diagL0ABufId;
                    auto l1DiagLayoutTla = tla::MakeLayout<float, NpuArch::layout::zN>(
                        DIAG_STAGE_PACKED_ROWS, DIAG_PACKED_COLS);
                    auto l1DiagTensorTla = tla::MakeTensor(
                        l1DiagTensor[l1DiagBufId], l1DiagLayoutTla, Arch::PositionL1{});
                    auto l0ADiagLayoutTla = tla::MakeLayout<float, LayoutTagL0A>(
                        DIAG_STAGE_PACKED_ROWS, DIAG_PACKED_COLS);
                    auto l0ADiagTensorTla = tla::MakeTensor(
                        l0ARescaleTensor[diagL0ABufId], l0ADiagLayoutTla, Arch::PositionL0A{});
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(diagL0AEventId);
                    CopyL1ToL0ARescale copyL1ToL0ARescale;
                    copyL1ToL0ARescale(l0ADiagTensorTla, l1DiagTensorTla);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(diagL0AEventId);

                    uint32_t rowsPerSubBlock = RoundUp(rowNum, 64) / 2;
                    uint32_t subBlockNumAct = CeilDiv(rowNum, rowsPerSubBlock);
                    for (uint32_t subBlockIdx = 0; subBlockIdx < subBlockNumAct; subBlockIdx++) {
                        uint32_t subBlockRowStart = subBlockIdx * rowsPerSubBlock;
                        uint32_t subBlockRowEnd =
                            min(rowNum, subBlockRowStart + rowsPerSubBlock);
                        uint32_t subBlockRowNum = subBlockRowEnd - subBlockRowStart;
                        uint32_t diagTileNum = CeilDiv(subBlockRowNum, DIAG_MMAD_TILE_SIZE);
                        uint32_t goL0BLoopCounter =
                            prefixSumL0BStages + nL0LoopNum * kL0LoopNum + subBlockIdx;
                        uint32_t goL0BBufId = goL0BLoopCounter % L0_STAGES;
                        uint32_t goL0BEventId = goL0BBufId + 2;
                        auto l1GoTile = GetTile(l1GoTensorTla,
                            tla::MakeCoord(subBlockRowStart, nL0Itr * L0_TILE_N),
                            tla::MakeShape(subBlockRowNum, l0TileNAct));
                        auto l0BGoLayoutTla = tla::MakeLayout<float, LayoutTagL0B>(
                            subBlockRowNum, l0TileNAct);
                        auto l0BGoTensorTla = tla::MakeTensor(
                            l0BRescaleTensor[goL0BBufId], l0BGoLayoutTla, Arch::PositionL0B{});
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(goL0BEventId);
                        CopyL1ToL0BRescale copyL1ToL0BRescale;
                        copyL1ToL0BRescale(l0BGoTensorTla, l1GoTile);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(goL0BEventId);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(goL0BEventId);
                        if (subBlockIdx == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(diagL0AEventId);
                        }
                        for (uint32_t j = 0; j < diagTileNum; j++) {
                            uint32_t nAct = min(DIAG_MMAD_TILE_SIZE,
                                subBlockRowNum - j * DIAG_MMAD_TILE_SIZE);
                            uint32_t diagTileIdx =
                                subBlockIdx * DIAG_MMAD_TILES_PER_SUB_BLOCK + j;
                            auto cTile = GetTile(l0CTensorTla,
                                tla::MakeCoord(subBlockRowStart + j * DIAG_MMAD_TILE_SIZE, 0),
                                tla::MakeShape(nAct, l0TileNAct));
                            auto diagTile = GetTile(l0ADiagTensorTla,
                                tla::MakeCoord(diagTileIdx * DIAG_MMAD_TILE_SIZE, 0U),
                                tla::MakeShape(DIAG_MMAD_TILE_SIZE, DIAG_MMAD_TILE_SIZE));
                            auto goTile = GetTile(l0BGoTensorTla,
                                tla::MakeCoord(j * DIAG_MMAD_TILE_SIZE, 0),
                                tla::MakeShape(nAct, l0TileNAct));
                            tileMmadRescale(cTile, diagTile, goTile,
                                DIAG_MMAD_TILE_SIZE, l0TileNAct, nAct, false);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(goL0BEventId);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(diagL0AEventId);
                    SetCrossCoreSync<4, PIPE_MTE1>(diagFlag);
            }

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            if (isLastKvSTile) {
                WaitCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
                uint32_t mPerSubCore = RoundUp(rowNum, 64) / 2;
                uint32_t nFixPAligned8 = RoundUp(l0TileNAct, 8);
                auto ubCTensorTlaTile = GetTile(ubCTensor,
                    tla::MakeCoord(0, nL0Itr * L0_TILE_N),
                    tla::MakeShape(mPerSubCore, nFixPAligned8));
                auto l0CTensorTlaTileSub0 = GetTile(l0CTensorTla,
                    tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, l0TileNAct));
                auto l0CTensorTlaTileSub1 = GetTile(l0CTensorTla,
                    tla::MakeCoord(mPerSubCore, 0), tla::MakeShape(mPerSubCore, l0TileNAct));
                CopyL0CToDst copyL0CToDstSub0;
                CopyL0CToDst copyL0CToDstSub1;
                copyL0CToDstSub0(ubCTensorTlaTile, l0CTensorTlaTileSub0, deqScalar, false);
                copyL0CToDstSub1(ubCTensorTlaTile, l0CTensorTlaTileSub1, deqScalar, true);
                SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            } else {
                using CopyL0CToL1 = typename TileCopy::template CopyL0CToL1<TensorGoC1>;
                CopyL0CToL1 copyL0CToL1;
                copyL0CToL1(l1GoTensorTlaC1, l0CTensorTla);
                AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(GO_READY_EVENT_ID);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<float> l1DiagTensor[DIAG_BUF_NUM];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
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


