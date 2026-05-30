/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/resource.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/gemm_type.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/matrix_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/layout.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/detail/callback.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/block/block_epilogue.hpp"

namespace Catlass::Epilogue::Block {

template <uint32_t UB_STAGES_, class CType_, class ScaleType_, class PerTokenScaleType_, class BiasType_, class DType_,
          class TileRowBroadcastMul_, class TileRowBroadcastAdd_, class TileBroadcastOneBlk_,
          class TileOneBlkColumnBroadcastMul_, class TileCopy_, class EpilogueTileSwizzle_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequant<UB_STAGES_>, CType_, ScaleType_, PerTokenScaleType_, BiasType_,
                    DType_, TileRowBroadcastMul_, TileRowBroadcastAdd_, TileBroadcastOneBlk_,
                    TileOneBlkColumnBroadcastMul_, TileCopy_, EpilogueTileSwizzle_> {
public:
    // Data infos
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
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

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileRowBroadcastAdd = TileRowBroadcastAdd_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbScale;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbPerTokenScale;
    using CopyGmToUbBias = typename TileCopy_::CopyGmToUbBias;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileRowBroadcastMul::TileShape;

    static_assert((UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COLUMN * sizeof(ElementScale) +
                                TileShape::ROW * sizeof(ElementPerTokenScale) +
                                TileShape::COLUMN * sizeof(ElementBias) + TileShape::COUNT * sizeof(ElementD)) +
                   (TileShape::COUNT + TileShape::COUNT) * sizeof(float) + TileShape::ROW * BYTE_PER_BLK) <=
                      ArchTag::UB_SIZE,
                  "TileShape is too large to fit in UB");

    struct Params {
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        GM_ADDR ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        GM_ADDR ptrBias;
        LayoutBias layoutBias;

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(GM_ADDR ptrScale_, LayoutScale layoutScale_, GM_ADDR ptrPerTokenScale_,
               LayoutPerTokenScale layoutPerTokenScale_, GM_ADDR ptrBias_, LayoutBias layoutBias_)
            : ptrScale(ptrScale_), layoutScale(layoutScale_), ptrPerTokenScale(ptrPerTokenScale_),
              layoutPerTokenScale(layoutPerTokenScale_), ptrBias(ptrBias_), layoutBias(layoutBias_)
        {
        }
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource)
    {
        size_t ubOff = 0;
        int32_t evtVMTE2 = 0;
        int32_t evtMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCBuf[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOff);
            ubOff += TileShape::COUNT * sizeof(ElementC);
            ubScaleBuf[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOff);
            ubOff += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenBuf[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOff);
            ubOff += TileShape::ROW * sizeof(ElementPerTokenScale);
            ubBiasBuf[i] = resource.ubBuf.template GetBufferByByte<ElementBias>(ubOff);
            ubOff += TileShape::COLUMN * sizeof(ElementBias);
            ubDBuf[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOff);
            ubOff += TileShape::COUNT * sizeof(ElementD);

            evtCVMTE2[i] = evtVMTE2++;
            evtCMTE2V[i] = evtMTE2V++;
            evtScaleVMTE2[i] = evtVMTE2++;
            evtScaleMTE2V[i] = evtMTE2V++;
            evtPerTokenVMTE2[i] = evtVMTE2++;
            evtPerTokenMTE2V[i] = evtMTE2V++;
            eventUbBiasVMTE2List[i] = evtVMTE2++;
            eventUbBiasMTE2VList[i] = evtMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtScaleVMTE2[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtPerTokenVMTE2[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOff);
        ubOff += TileShape::COUNT * sizeof(float);
        if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                      AscendC::IsSameType<ElementBias, half>::value) {
            ubBiasFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOff);
            ubOff += TileShape::COLUMN * sizeof(float);
        }
        ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOff);
        ubOff += TileShape::COUNT * sizeof(float);
        ubPerTokenScaleBrcb = resource.ubBuf.template GetBufferByByte<float>(ubOff);
        ubOff += TileShape::ROW * BYTE_PER_BLK;
        ubPerTokenMul = ubMul;
        ubBiasAdd = ubMul;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtScaleVMTE2[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtPerTokenVMTE2[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    // perChannel、perToken
    CATLASS_DEVICE
    void operator()(MatrixCoord const &blockOffset, GemmCoord const &actualBlockShapeMNK,
                    AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutBlockC,
                    AscendC::GlobalTensor<ElementD> const &gmBlockD, LayoutD const &layoutBlockD,
                    Callback &&callback = Callback{})
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }
        callback();

        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();

        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer((__gm__ ElementScale *)params.ptrScale);
        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer((__gm__ ElementPerTokenScale *)params.ptrPerTokenScale);
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffset = blockOffset + tileOffsetInBlock;

            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffsetInBlock)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);

            auto &ubC = ubCBuf[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};
            // 把 C 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtCMTE2V[ubListId]);

            auto scaleTileOffset = tileOffset.template GetCoordByAxis<1>();
            auto scaleTileShape = actualTileShape.template GetCoordByAxis<1>();

            auto gmTileScale = gmScale[params.layoutScale.GetOffset(scaleTileOffset)];
            auto layoutGmTileScale = params.layoutScale.GetTileLayout(scaleTileShape);

            auto &ubScale = ubScaleBuf[ubListId];

            auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShape);

            // 把 scale 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtScaleVMTE2[ubListId]);
            copyGmToUbScale(ubScale, gmTileScale, layoutUbScale, layoutGmTileScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtScaleMTE2V[ubListId]);

            auto perTokenScaleTileOffset = tileOffset.template GetCoordByAxis<0>();
            auto perTokenScaleTileShape = actualTileShape.template GetCoordByAxis<0>();

            auto gmTilePerTokenScale = gmPerTokenScale[params.layoutPerTokenScale.GetOffset(perTokenScaleTileOffset)];
            auto layoutGmTilePerTokenScale = params.layoutPerTokenScale.GetTileLayout(perTokenScaleTileShape);

            auto &ubPerTokenScale = ubPerTokenBuf[ubListId];
            auto layoutUbPerTokenScale =
                LayoutScale::template MakeLayoutInUb<ElementPerTokenScale>(perTokenScaleTileShape);

            // 把 perTokenScale 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtPerTokenVMTE2[ubListId]);
            copyGmToUbPerTokenScale(ubPerTokenScale, gmTilePerTokenScale, layoutUbPerTokenScale,
                                    layoutGmTilePerTokenScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtPerTokenMTE2V[ubListId]);
            auto biasTileOffset = tileOffset.template GetCoordByAxis<1>();
            auto biasTileShape = actualTileShape.template GetCoordByAxis<1>();

            auto gmTileBias = gmBias[params.layoutBias.GetOffset(biasTileOffset)];
            auto layoutGmTileBias = params.layoutBias.GetTileLayout(biasTileShape);

            auto &ubBias = ubBiasBuf[ubListId];
            auto layoutUbBias = LayoutBias::template MakeLayoutInUb<ElementBias>(biasTileShape);

            // 把bias 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
            copyGmToUbBias(ubBias, gmTileBias, layoutUbBias, layoutGmTileBias);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);

            // 在UB上把C cast到FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtCMTE2V[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtCVMTE2[ubListId]);

            if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                          AscendC::IsSameType<ElementBias, half>::value) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
                AscendC::Cast(ubBiasFp32, ubBias, AscendC::RoundMode::CAST_NONE, TileShape::COLUMN);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
            }

            // 在UB上做广播乘法
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtScaleMTE2V[ubListId]);
            tileRowBroadcastMul(ubMul, ubCFp32, ubScale);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtScaleVMTE2[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtPerTokenMTE2V[ubListId]);
            tileBroadcastOneBlk(ubPerTokenScaleBrcb, ubPerTokenScale);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtPerTokenVMTE2[ubListId]);

            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubMul, ubPerTokenScaleBrcb);
            AscendC::PipeBarrier<PIPE_V>();

            if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                          AscendC::IsSameType<ElementBias, half>::value) {
                tileRowBroadcastAdd(ubBiasAdd, ubPerTokenMul, ubBiasFp32);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
                tileRowBroadcastAdd(ubBiasAdd, ubPerTokenMul, ubBias);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
            }

            auto &ubD = ubDBuf[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};

            // 将乘法结果从UB cast到D
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, ubBiasAdd, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            auto gmTileD = gmBlockD[layoutBlockD.GetOffset(tileOffsetInBlock)];
            auto layoutGmTileD = layoutBlockD.GetTileLayout(actualTileShape);

            // 把乘法结果从UB拷贝到GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCBuf[UB_STAGES];
    AscendC::LocalTensor<ElementScale> ubScaleBuf[UB_STAGES];
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenBuf[UB_STAGES];
    AscendC::LocalTensor<ElementBias> ubBiasBuf[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDBuf[UB_STAGES];

    int32_t evtCVMTE2[UB_STAGES];
    int32_t evtCMTE2V[UB_STAGES];
    int32_t evtScaleVMTE2[UB_STAGES];
    int32_t evtScaleMTE2V[UB_STAGES];
    int32_t evtPerTokenVMTE2[UB_STAGES];
    int32_t evtPerTokenMTE2V[UB_STAGES];
    int32_t eventUbBiasVMTE2List[UB_STAGES];
    int32_t eventUbBiasMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32;
    AscendC::LocalTensor<float> ubBiasFp32;
    AscendC::LocalTensor<float> ubMul;
    AscendC::LocalTensor<float> ubPerTokenScaleBrcb;
    AscendC::LocalTensor<float> ubPerTokenMul;
    AscendC::LocalTensor<float> ubBiasAdd;

    TileRowBroadcastMul tileRowBroadcastMul;
    TileRowBroadcastAdd tileRowBroadcastAdd;
    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;

    CopyGmToUbC copyGmToUbC;
    CopyGmToUbScale copyGmToUbScale;
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
    CopyGmToUbBias copyGmToUbBias;
    CopyUbToGmD copyUbToGmD;
};
} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP