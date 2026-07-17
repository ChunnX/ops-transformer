/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_A2_BF16_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_A2_BF16_HPP

#include "../template_linear_algebra_v2/catlass.hpp"
#include "../template_linear_algebra_v2/arch/resource.hpp"
#include "../template_linear_algebra_v2/epilogue/dispatch_policy.hpp"
#include "../template_linear_algebra_v2/gemm_coord.hpp"
#include "../template_linear_algebra_v2/matrix_coord.hpp"
#include "../template_linear_algebra_v2/layout/layout.hpp"
#include "../template_linear_algebra_v2/detail/callback.hpp"
#include "../utils/gated_activation.hpp"

namespace Catlass::Epilogue::Block {

// A2 BF16/FP16 gated activation. The input row is split into fixed gate/up
// tiles, making the UB footprint independent of N.
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileElemWiseMuls_,
          class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileElemWiseMuls_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using Activation = typename DispatchPolicy::Activation;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t TILE_LENGTH = GATED_ACTIVATION_TILE_LENGTH;
    static constexpr uint32_t MAX_SCALE_COUNT = 4096;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    static constexpr size_t BUFFER_SIZE =
        UB_STAGES *
            (2 * TILE_LENGTH * sizeof(ElementC) + TILE_LENGTH * sizeof(ElementD) + 3 * TILE_LENGTH * sizeof(float)) +
        MAX_SCALE_COUNT * sizeof(float);
    static_assert(BUFFER_SIZE < ArchTag::UB_SIZE, "The tiled activation buffers exceed UB capacity");

    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyUbToGmDequantScale =
        Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>>;

    struct Params {
        CATLASS_DEVICE Params(){};
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, int32_t n, Params const &params = Params{})
    {
        (void)n;
        (void)params;
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += 2 * TILE_LENGTH * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(ElementD);
            ubGateFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);
            ubUpFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);
            ubActivationList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }

        ubPerTokenScaleOutput = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += MAX_SCALE_COUNT * sizeof(float);
        sharedTmpBuffer = resource.ubBuf.template GetBufferByByte<uint8_t>(ubOffset);
    }

    CATLASS_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE ~BlockEpilogue()
    {
    }

    // Per-token dequant + activation + per-token dynamic quant. This overload
    // is kept for interface compatibility with the unified kernel.
    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmC, MatrixCoord const &shapeC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale1,
                    AscendC::GlobalTensor<ElementD> const &gmD,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale2, uint32_t epilogueCoreNum = 40,
                    uint32_t gmmOutPreRowStride = 1, Callback &&callback = Callback{})
    {
        callback();
        uint32_t blockM = shapeC.row();
        uint32_t branchLength = shapeC.column() / 2;
        uint32_t loopStartIdx = 0;
        uint32_t tasksForIdx = 0;
        if (!GetTaskRange(blockM, epilogueCoreNum, loopStartIdx, tasksForIdx)) {
            return;
        }

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * gmmOutPreRowStride];
            auto gmTileD = gmD[loopIdx * branchLength];
            ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx);
            ElementPerTokenScale rowAbsMax = 0.0f;

            for (uint32_t tileOffset = 0; tileOffset < branchLength; tileOffset += TILE_LENGTH) {
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                ComputeTile(gmTileC, branchLength, tileOffset, tileLength, gmmOutPreRowStride, perTokenScale, true,
                            0.0f);
                auto &ubAbs = ubGateFp32List[ubListId];
                auto &ubReduceMax = ubUpFp32List[ubListId];
                AscendC::Abs(ubAbs, ubActivationList[ubListId], tileLength);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceMax<float>(ubReduceMax, ubAbs, ubReduceMax, tileLength, false);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                ElementPerTokenScale tileAbsMax = ubReduceMax.GetValue(0);
                rowAbsMax = rowAbsMax > tileAbsMax ? rowAbsMax : tileAbsMax;
                AdvanceStage();
            }

            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            ubPerTokenScaleOutput.SetValue(loopIdx - loopStartIdx, rowAbsMax / 127.0f);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            ElementPerTokenScale quantMultiplier = rowAbsMax > 0.0f ? 127.0f / rowAbsMax : 0.0f;

            for (uint32_t tileOffset = 0; tileOffset < branchLength; tileOffset += TILE_LENGTH) {
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                ComputeTile(gmTileC, branchLength, tileOffset, tileLength, gmmOutPreRowStride, perTokenScale, true,
                            0.0f);
                QuantizeAndCopyTile(gmTileD[tileOffset], tileLength, quantMultiplier);
                AdvanceStage();
            }
        }

        if (tasksForIdx > 0) {
            LayoutPerTokenScale scaleLayout{tasksForIdx};
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            copyUbToGmDequantScale(gmPerTokenScale2[loopStartIdx], ubPerTokenScaleOutput, scaleLayout, scaleLayout);
        }
    }

    // Non-quantized BF16/FP16 path.
    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmC, MatrixCoord const &shapeC,
                    AscendC::GlobalTensor<ElementD> const &gmD, uint32_t epilogueCoreNum = 40,
                    float activationClamp = 0.0f, uint32_t gmmOutPreRowStride = 1, Callback &&callback = Callback{})
    {
        callback();
        uint32_t blockM = shapeC.row();
        uint32_t branchLength = shapeC.column() / 2;
        uint32_t loopStartIdx = 0;
        uint32_t tasksForIdx = 0;
        if (!GetTaskRange(blockM, epilogueCoreNum, loopStartIdx, tasksForIdx)) {
            return;
        }

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * gmmOutPreRowStride];
            auto gmTileD = gmD[loopIdx * branchLength];
            for (uint32_t tileOffset = 0; tileOffset < branchLength; tileOffset += TILE_LENGTH) {
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                ComputeTile(gmTileC, branchLength, tileOffset, tileLength, gmmOutPreRowStride, 1.0f, false,
                            activationClamp);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
                AscendC::Cast(ubDList[ubListId], ubActivationList[ubListId], AscendC::RoundMode::CAST_RINT, tileLength);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
                LayoutD tileLayout{1, tileLength};
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
                copyUbToGmD(gmTileD[tileOffset], ubDList[ubListId], tileLayout, tileLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
                AdvanceStage();
            }
        }
    }

private:
    CATLASS_DEVICE
    bool GetTaskRange(uint32_t blockM, uint32_t epilogueCoreNum, uint32_t &loopStartIdx, uint32_t &tasksForIdx)
    {
        uint32_t subblockIdx = get_block_idx() + get_subblockid() * get_block_num();
        uint32_t moveDataCoreNum = get_block_num() * 2 - epilogueCoreNum;
        if (subblockIdx < moveDataCoreNum) {
            return false;
        }
        uint32_t epilogueCoreIdx = subblockIdx - moveDataCoreNum;
        uint32_t perCoreData = blockM / epilogueCoreNum;
        uint32_t remainderData = blockM % epilogueCoreNum;
        tasksForIdx = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
        loopStartIdx =
            epilogueCoreIdx * perCoreData + (epilogueCoreIdx < remainderData ? epilogueCoreIdx : remainderData);
        return true;
    }

    CATLASS_DEVICE
    static uint32_t TileLength(uint32_t branchLength, uint32_t tileOffset)
    {
        uint32_t remaining = branchLength - tileOffset;
        return remaining < TILE_LENGTH ? remaining : TILE_LENGTH;
    }

    CATLASS_DEVICE
    void ComputeTile(AscendC::GlobalTensor<ElementC> const &gmTileC, uint32_t branchLength, uint32_t tileOffset,
                     uint32_t tileLength, uint32_t gmRowStride, ElementPerTokenScale perTokenScale,
                     bool applyPerTokenScale, float activationClamp)
    {
        auto &ubC = ubCList[ubListId];
        auto &ubGate = ubGateFp32List[ubListId];
        auto &ubUp = ubUpFp32List[ubListId];
        LayoutC tileLayout{1, tileLength};
        LayoutC gmTileLayout{1, tileLength, gmRowStride};

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
        copyGmToUbC(ubC, gmTileC[tileOffset], tileLayout, gmTileLayout);
        copyGmToUbC(ubC[TILE_LENGTH], gmTileC[branchLength + tileOffset], tileLayout, gmTileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
        AscendC::Cast(ubGate, ubC, AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::Cast(ubUp, ubC[TILE_LENGTH], AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

        if (applyPerTokenScale) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubGate, ubGate, perTokenScale, tileLength);
            AscendC::Muls(ubUp, ubUp, perTokenScale, tileLength);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Activation::Compute(ubActivationList[ubListId], ubGate, ubUp, sharedTmpBuffer, tileLength, activationClamp);
    }

    CATLASS_DEVICE
    void QuantizeAndCopyTile(AscendC::GlobalTensor<ElementD> const &gmTileD, uint32_t tileLength,
                             ElementPerTokenScale quantMultiplier)
    {
        auto &ubQuantTmp = ubGateFp32List[ubListId];
        auto ubQuantS32 = ubQuantTmp.template ReinterpretCast<int32_t>();
        auto ubQuantF16 = ubQuantTmp.template ReinterpretCast<half>();
        AscendC::Muls(ubQuantTmp, ubActivationList[ubListId], quantMultiplier, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ubQuantS32, ubQuantTmp, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetDeqScale(static_cast<half>(1.0));
        AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
        AscendC::Cast(ubDList[ubListId], ubQuantF16, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
        LayoutD tileLayout{1, tileLength};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
        copyUbToGmD(gmTileD, ubDList[ubListId], tileLayout, tileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
    }

    CATLASS_DEVICE
    void AdvanceStage()
    {
        ubListId = (ubListId + 1 < UB_STAGES) ? ubListId + 1 : 0;
    }

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubGateFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubUpFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubActivationList[UB_STAGES];
    AscendC::LocalTensor<float> ubPerTokenScaleOutput;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];
    uint32_t ubListId{0};

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    CopyUbToGmDequantScale copyUbToGmDequantScale;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_A2_BF16_HPP
