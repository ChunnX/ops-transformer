/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H

#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/catlass.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/resource.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/dispatch_policy.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/gemm_type.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/matrix_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/layout.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/detail/callback.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/copy_gm_to_ub.hpp"

namespace Catlass::Epilogue::Block {

template <
    class ArchTag_,
    class CType_,
    class ScaleType_,
    class PerTokenScaleType_,
    class BiasType_,
    class DType_,
    class TileRowBroadcastMul_,
    class TileRowBroadcastAdd_,
    class TileBroadcastOneBlk_,
    class TileOneBlkColumnBroadcastMul_,
    class TileCopy_,
    class EpilogueTileSwizzle_
>
class BlockEpilogue <
    ArchTag_,
    CType_,
    ScaleType_,
    PerTokenScaleType_,
    BiasType_,
    DType_,
// block epilogue marker
    TileRowBroadcastMul_,
    TileRowBroadcastAdd_,
    TileBroadcastOneBlk_,
    TileOneBlkColumnBroadcastMul_,
    TileCopy_,
    EpilogueTileSwizzle_
> {
public:
    static constexpr uint32_t UB_STAGES = 2;

    // Data infos
    using ArchTag = ArchTag_;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementScale = typename ScaleType_::Element;
    using LayoutScale = typename ScaleType_::Layout;
    using ElementPerTokenScale = typename PerTokenScaleType_::Element;
    using LayoutPerTokenScale = typename PerTokenScaleType_::Layout;
    using ElementBias = typename BiasType_::Element;
    using LayoutBias = typename BiasType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(
        std::is_same_v<ElementC, int32_t> && (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>) &&
            std::is_same_v<ElementScale, float> && std::is_same_v<ElementPerTokenScale, float>,
        "The element type template parameters of BlockEpilogue are wrong"
    );
    static_assert(
        std::is_same_v<LayoutC, layout::RowMajor> && std::is_same_v<LayoutScale, layout::VectorLayout> &&
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && std::is_same_v<LayoutD, layout::RowMajor>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileRowBroadcastAdd = TileRowBroadcastAdd_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbX;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbY;
    using CopyGmToUbBias = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, BiasType_>;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyGmToUbD = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<ElementD, layout::RowMajor>>;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileRowBroadcastMul::TileShape;

    static_assert(
        TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH &&
        std::is_same_v<TileShape, typename TileOneBlkColumnBroadcastMul::TileShape>,
        "TileShape must be consistent for all tile compute ops"
    );

    static_assert(
        (UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COLUMN * sizeof(ElementScale) +
         TileShape::ROW * sizeof(ElementPerTokenScale) + TileShape::COLUMN * sizeof(ElementScale) +
         TileShape::COUNT * sizeof(ElementD)) +
         (TileShape::COUNT + TileShape::COUNT) * sizeof(float) + TileShape::ROW * BYTE_PER_BLK)
        <= ArchTag::UB_SIZE,
        "TileShape is too large to fit in UB"
    );

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource)
    {
        size_t ubOff_ = 0;
        int32_t evtVMTE2_ = 0;
        int32_t evtMTE2V_ = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubC_[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOff_);
            ubOff_ += TileShape::COUNT * sizeof(ElementC);
            ubScaleBuf_[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOff_);
            ubOff_ += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenBuf_[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOff_);
            ubOff_ += TileShape::ROW * sizeof(ElementPerTokenScale);
            ubBias_[i] = resource.ubBuf.template GetBufferByByte<ElementBias>(ubOff_);
            ubOff_ += TileShape::COLUMN * sizeof(ElementBias);
            ubD_[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOff_);
            ubOff_ += TileShape::COUNT * sizeof(ElementD);

            evtCVMTE2_[i] = evtVMTE2_++;
            eventUbBiasVMTE2List[i] = evtVMTE2_++;
            evtCMTE2V_[i] = evtMTE2V_++;
            eventUbBiasMTE2VList[i] = evtMTE2V_++;
            evtScaleMTE2V_[i] = evtMTE2V_++;
            evtPerTokenMTE2V_[i] = evtMTE2V_++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;
        }
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOff_);
        ubOff_ += TileShape::COUNT * sizeof(float);
        if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
            AscendC::IsSameType<ElementBias, half>::value) {
            ubBiasFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOff_);
            ubOff_ += TileShape::COLUMN * sizeof(float);
        }
        ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOff_);
        ubOff_ += TileShape::COUNT * sizeof(float);
        ubPerTokenScaleBrcb = resource.ubBuf.template GetBufferByByte<float>(ubOff_);
        ubOff_ += TileShape::ROW * BYTE_PER_BLK;
        ubPerTokenMul = ubMul;
        ubBiasAdd = ubMul;
    }

    CATLASS_DEVICE
    void WaitFlag()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void InitFlag()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2_[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    // perChannel、perToken
    CATLASS_DEVICE
    void operator() (__gm__ ElementScale *ptrScale, LayoutScale layoutScale,
                     __gm__ ElementPerTokenScale *ptrPerTokenScale, LayoutPerTokenScale layoutPerTokenScale,
                     __gm__ ElementBias *ptrBias, LayoutBias layoutBias,
                     __gm__ ElementC *ptrC, LayoutC layoutC, __gm__ ElementD *ptrD, LayoutD layoutD,
                     GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = problemShape.GetCoordMN();

        gmScale.SetGlobalBuffer(ptrScale);
        gmPerTokenScale.SetGlobalBuffer(ptrPerTokenScale);
        gmC.SetGlobalBuffer(ptrC);
        gmD.SetGlobalBuffer(ptrD);
        if (ptrBias != nullptr) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)ptrBias);
        }

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            auto gmTileCMain = gmC[layoutC.GetOffset(tileOffset)];
            auto layoutGmTileCMain = layoutC.GetTileLayout(actualTileShape);

            auto &ubC = ubC_[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};

            // 把 C 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2_[ubListId]);
            copyGmToUbC(ubC, gmTileCMain, layoutUbC, layoutGmTileCMain);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtCMTE2V_[ubListId]);

            auto scaleTileOff = tileOffset.template GetCoordByAxis<1>();
            auto scaleTileShp = actualTileShape.template GetCoordByAxis<1>();

            auto gmTileScaleMain = gmScale[layoutScale.GetOffset(scaleTileOff)];
            auto layoutGmTileScaleMain = layoutScale.GetTileLayout(scaleTileShp);

            auto &ubScale = ubScaleBuf_[ubListId];
            auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShp);

            // 把 scale 从GM拷贝到UB
            copyGmToUbScale(ubScale, gmTileScaleMain, layoutUbScale, layoutGmTileScaleMain);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtScaleMTE2V_[ubListId]);

            auto ptsTileOffset = tileOffset.template GetCoordByAxis<0>();
            auto ptsTileShape = actualTileShape.template GetCoordByAxis<0>();

            auto gmTilePTS = gmPerTokenScale[layoutPerTokenScale.GetOffset(ptsTileOffset)];
            auto layoutGmTilePTS = layoutPerTokenScale.GetTileLayout(ptsTileShape);

            auto &ubPerTokenScale = ubPerTokenBuf_[ubListId];
            auto layoutUbPerTokenScale = LayoutScale::template MakeLayoutInUb<ElementPerTokenScale>(
                ptsTileShape);

            // 把 perTokenScale 从GM拷贝到UB
            copyGmToUbPerTokenScale(ubPerTokenScale, gmTilePTS, layoutUbPerTokenScale,
                layoutGmTilePTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtPerTokenMTE2V_[ubListId]);
            // 只有当bias不为nullptr时，才处理bias相关操作
            if (ptrBias != nullptr) {
                auto biasTileOffset = tileOffset.template GetCoordByAxis<1>();
                auto biasTileShape = actualTileShape.template GetCoordByAxis<1>();

                auto gmTileBias = gmBias[layoutBias.GetOffset(biasTileOffset)];
                auto layoutGmTileBias = layoutBias.GetTileLayout(biasTileShape);

                auto &ubBias = ubBias_[ubListId];
                auto layoutUbBias = LayoutBias::template MakeLayoutInUb<ElementBias>(biasTileShape);

                // 把bias 从GM拷贝到UB
                copyGmToUbBias(ubBias, gmTileBias, layoutUbBias, layoutGmTileBias);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);

                if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                    AscendC::IsSameType<ElementBias, half>::value) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
                    AscendC::Cast(ubBiasFp32, ubBias, AscendC::RoundMode::CAST_NONE, TileShape::COLUMN);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
                }
            }
            // 在UB上把C cast到FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtCMTE2V_[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2_[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtScaleMTE2V_[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtPerTokenMTE2V_[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastMul(ubMul, ubCFp32, ubScale);
            tileBroadcastOneBlk(ubPerTokenScaleBrcb, ubPerTokenScale);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubMul, ubPerTokenScaleBrcb);
            AscendC::PipeBarrier<PIPE_V>();
            // 选择要cast的源tensor：如果有bias，使用ubBiasAdd；否则直接使用ubPerTokenMul
            AscendC::LocalTensor_<float> &castSrc = ptrBias != nullptr ? ubBiasAdd : ubPerTokenMul;

            // 只有当bias不为nullptr时，才执行bias加法
            if (ptrBias != nullptr) {
                if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                    AscendC::IsSameType<ElementBias, half>::value) {
                    tileRowBroadcastAdd(ubBiasAdd, ubPerTokenMul, ubBiasFp32);
                } else {
                    auto &ubBias = ubBias_[ubListId];
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
                    tileRowBroadcastAdd(ubBiasAdd, ubPerTokenMul, ubBias);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            auto &ubD = ubD_[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};

            // 将乘法结果从UB cast到D
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, castSrc, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            auto gmTileDMain = gmD[layoutD.GetOffset(tileOffset)];
            auto layoutGmTileDMain = layoutD.GetTileLayout(actualTileShape);

            // 把乘法结果从UB拷贝到GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileDMain, ubD, layoutGmTileDMain, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

    // perChannel
    CATLASS_DEVICE
    void operator() (__gm__ ElementScale *ptrScale, LayoutScale layoutScale, __gm__ ElementBias *ptrBias,
                     LayoutBias layoutBias, __gm__ ElementC *ptrC, LayoutC layoutC, __gm__ ElementD *ptrD,
                     LayoutD layoutD, GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord blockCoordMN = problemShape.GetCoordMN();

        gmScale.SetGlobalBuffer(ptrScale);
        gmC.SetGlobalBuffer(ptrC);
        gmD.SetGlobalBuffer(ptrD);
        if (ptrBias != nullptr) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)ptrBias);
        }
        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueSwizzle(blockCoordMN, tileShape);
        uint32_t loopCount = epilogueSwizzle.GetLoops();
        uint32_t subBlkIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlkNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t iterIdx = subBlkIdx; iterIdx < loopCount; iterIdx += subBlkNum) {
            auto tileCoord = epilogueSwizzle.GetTileCoord(iterIdx);
            auto actualTileShape = epilogueSwizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            auto gmBlkC = gmC[layoutC.GetOffset(tileOffset)];
            auto layoutGmBlkC = layoutC.GetTileLayout(actualTileShape);

            auto &ubC_2 = ubC_2__2[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};

            // 把 C 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2_[ubListId]);
            copyGmToUbC(ubC_2, gmBlkC, layoutUbC, layoutGmBlkC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtCMTE2V_[ubListId]);

            auto scaleBlkOffset = tileOffset.template GetCoordByAxis<1>();
            auto scaleBlkShape = actualTileShape.template GetCoordByAxis<1>();

            auto gmBlkScale = gmScale[layoutScale.GetOffset(scaleBlkOffset)];
            auto layoutGmBlkScale = layoutScale.GetTileLayout(scaleBlkShape);

            auto &ubS_2cale = ubS_2caleBuf_[ubListId];
            auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleBlkShape);

            // 把 scale 从GM拷贝到UB
            copyGmToUbScale(ubS_2cale, gmBlkScale, layoutUbScale, layoutGmBlkScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtScaleMTE2V_[ubListId]);

            // 只有当bias不为nullptr时，才处理bias相关操作
            if (ptrBias != nullptr) {
                auto biasBlkOffset = tileOffset.template GetCoordByAxis<1>();
                auto biasBlkShape = actualTileShape.template GetCoordByAxis<1>();

                auto gmBlkBias = gmBias[layoutBias.GetOffset(biasBlkOffset)];
                auto layoutGmBlkBias = layoutBias.GetTileLayout(biasBlkShape);

                auto &ubBias_2 = ubBias_2_[ubListId];
                auto layoutUbBias = LayoutBias::template MakeLayoutInUb<ElementBias>(biasBlkShape);

                // 把bias 从GM拷贝到UB

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
                copyGmToUbBias(ubBias_2, gmBlkBias, layoutUbBias, layoutGmBlkBias);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);

                if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                    AscendC::IsSameType<ElementBias, half>::value) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
                    AscendC::Cast(ubBias_2Fp32, ubBias_2, AscendC::RoundMode::CAST_NONE, TileShape::COLUMN);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
                }
            }
            // 在UB上把C cast到FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtCMTE2V_[ubListId]);
            AscendC::Cast(ubC_2Fp32, ubC_2, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2_[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtScaleMTE2V_[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastMul(ubMul, ubC_2Fp32, ubS_2cale);
            AscendC::PipeBarrier<PIPE_V>();

            // 选择要cast的源tensor：如果有bias，使用ubBias_2Add；否则直接使用ubMul
            AscendC::LocalTensor_<float> &castSrc = ptrBias != nullptr ? ubBias_2Add : ubMul;

            // 只有当bias不为nullptr时，才执行bias加法
            if (ptrBias != nullptr) {
                if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                    AscendC::IsSameType<ElementBias, half>::value) {
                    tileRowBroadcastAdd(ubBias_2Add, ubMul, ubBias_2Fp32);
                } else {
                    auto &ubBias_2 = ubBias_2_[ubListId];
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
                    tileRowBroadcastAdd(ubBias_2Add, ubMul, ubBias_2);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            auto &ubD = ubD_[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};

            // 将乘法结果从UB cast到D
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, castSrc, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            auto gmBlkD = gmD[layoutD.GetOffset(tileOffset)];
            auto layoutGmBlkD = layoutD.GetTileLayout(actualTileShape);

            // 把乘法结果从UB拷贝到GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmBlkD, ubD, layoutGmBlkD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

    // perToken
    CATLASS_DEVICE
    void operator() (__gm__ ElementPerTokenScale *ptrPerTokenScale, LayoutPerTokenScale layoutPerTokenScale,
                     __gm__ ElementBias *ptrBias, LayoutBias layoutBias,
                     __gm__ ElementD *ptrIn, LayoutD layoutIn, __gm__ ElementD *ptrOut, LayoutD layoutOut,
                     GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord coordMN = problemShape.GetCoordMN();

        AscendC::GlobalTensor<ElementD> gmIn;
        AscendC::GlobalTensor<ElementD> gmOut;
        gmPerTokenScale.SetGlobalBuffer(ptrPerTokenScale);
        gmIn.SetGlobalBuffer(ptrIn);
        gmOut.SetGlobalBuffer(ptrOut);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle tileSwizzle(coordMN, tileShape);
        uint32_t numLoops = tileSwizzle.GetLoops();
        uint32_t sblkIdx = AscendC::GetSubBlockIdx();
        uint32_t sblkNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t stepIdx = sblkIdx; stepIdx < numLoops; stepIdx += sblkNum) {
            auto tileCoord = tileSwizzle.GetTileCoord(stepIdx);
            auto actualTileShape = tileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            auto gmBlkIn = gmIn[layoutIn.GetOffset(tileOffset)];
            auto gmBlkOut = gmOut[layoutOut.GetOffset(tileOffset)];
            auto layoutGmBlkIn = layoutIn.GetTileLayout(actualTileShape);
            auto layoutGmBlkOut = layoutOut.GetTileLayout(actualTileShape);

            auto &ubIn = ubD_[ubListId];
            LayoutD layoutUbIn{actualTileShape, ubTileStride};

            // 把 D 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2_[ubListId]);
            copyGmToUbD(ubIn, gmBlkIn, layoutUbIn, layoutGmBlkIn);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtCMTE2V_[ubListId]);

            auto ptsBlkOffset = tileOffset.template GetCoordByAxis<0>();
            auto ptsBlkShape = actualTileShape.template GetCoordByAxis<0>();

            auto gmBlkPTS = gmPerTokenScale[layoutPerTokenScale.GetOffset(ptsBlkOffset)];
            auto layoutGmBlkPTS = layoutPerTokenScale.GetTileLayout(ptsBlkShape);

            auto &ubPerTokenScale = ubPerTokenBuf_[ubListId];
            auto layoutUbPerTokenScale = LayoutScale::template MakeLayoutInUb<ElementPerTokenScale>(
                ptsBlkShape);

            // 把 perTokenScale 从GM拷贝到UB
            copyGmToUbPerTokenScale(ubPerTokenScale, gmBlkPTS, layoutUbPerTokenScale,
                layoutGmBlkPTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtPerTokenMTE2V_[ubListId]);

            // 只有当bias不为nullptr时，才处理bias相关操作
            if (ptrBias != nullptr) {
                auto biasBlkOffset = tileOffset.template GetCoordByAxis<1>();
                auto biasBlkShape = actualTileShape.template GetCoordByAxis<1>();

                auto gmBlkBias = gmBias[layoutBias.GetOffset(biasBlkOffset)];
                auto layoutGmBlkBias = layoutBias.GetTileLayout(biasBlkShape);

                auto &ubBias_3 = ubBias_3_[ubListId];
                auto layoutUbBias = LayoutBias::template MakeLayoutInUb<ElementBias>(biasBlkShape);

                // 把bias 从GM拷贝到UB

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
                copyGmToUbBias(ubBias_3, gmBlkBias, layoutUbBias, layoutGmBlkBias);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);

                if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                    AscendC::IsSameType<ElementBias, half>::value) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
                    AscendC::Cast(ubBias_3Fp32, ubBias_3, AscendC::RoundMode::CAST_NONE, TileShape::COLUMN);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
                }
            }

            // 在UB上把D cast到FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtCMTE2V_[ubListId]);
            AscendC::Cast(ubC_3Fp32, ubIn, AscendC::RoundMode::CAST_NONE, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2_[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtPerTokenMTE2V_[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileBroadcastOneBlk(ubPerTokenScaleBrcb, ubPerTokenScale);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubC_3Fp32, ubPerTokenScaleBrcb);
            AscendC::PipeBarrier<PIPE_V>();

            // 选择要cast的源tensor：如果有bias，使用ubBias_3Add；否则直接使用ubPerTokenMul
            AscendC::LocalTensor_<float> &castSrc = ptrBias != nullptr ? ubBias_3Add : ubPerTokenMul;

            // 只有当bias不为nullptr时，才执行bias加法
            if (ptrBias != nullptr) {
                if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                    AscendC::IsSameType<ElementBias, half>::value) {
                    tileRowBroadcastAdd(ubBias_3Add, ubPerTokenMul, ubBias_3Fp32);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    auto &ubBias_3 = ubBias_3_[ubListId];
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
                    tileRowBroadcastAdd(ubBias_3Add, ubPerTokenMul, ubBias_3);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
                }
            }
            // 将乘法结果从UB cast到D
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubIn, castSrc, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            // 把乘法结果从UB拷贝到GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmBlkOut, ubIn, layoutGmBlkOut, layoutUbIn);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

private:
    AscendC::LocalTensor_<ElementC> ubC_3__3[UB_STAGES];
    AscendC::LocalTensor_<ElementScale> ubS_3caleBuf_[UB_STAGES];
    AscendC::LocalTensor_<ElementPerTokenScale> ubPerTokenBuf_[UB_STAGES];
    AscendC::LocalTensor_<ElementBias> ubBias_3_[UB_STAGES];
    AscendC::LocalTensor_<ElementD> ubD_[UB_STAGES];

    int32_t evtCVMTE2_[UB_STAGES];
    int32_t evtCMTE2V_[UB_STAGES];
    int32_t evtScaleMTE2V_[UB_STAGES];
    int32_t evtPerTokenMTE2V_[UB_STAGES];
    int32_t eventUbBiasVMTE2List[UB_STAGES];
    int32_t eventUbBiasMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor_<float> ubC_3Fp32;
    AscendC::LocalTensor_<float> ubBias_3Fp32;
    AscendC::LocalTensor_<float> ubMul;
    AscendC::LocalTensor_<float> ubPerTokenScaleBrcb;
    AscendC::LocalTensor_<float> ubPerTokenMul;
    AscendC::LocalTensor_<float> ubBias_3Add;

    TileRowBroadcastMul tileRowBroadcastMul;
    TileRowBroadcastAdd tileRowBroadcastAdd;
    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;

    CopyGmToUbC copyGmToUbC;
    CopyGmToUbScale copyGmToUbScale;
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
    CopyGmToUbBias copyGmToUbBias;
    CopyUbToGmD copyUbToGmD;
    CopyGmToUbD copyGmToUbD;

    AscendC::GlobalTensor<ElementScale> gmScale;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementD> gmD;
    AscendC::GlobalTensor<ElementBias> gmBias;
};
}  // namespace Catlass::Epilogue::Block

#endif  // MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H
