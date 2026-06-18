/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_BLOCK_EPILOGUE_RESCALE_O_HPP
#define FUSED_BLOCK_EPILOGUE_RESCALE_O_HPP

#include <limits>
#include "../../../attn_infra/arch/fused_resource.hpp"
#include "../../../attn_infra/epilogue/fused_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/fused_epilogue_tile_copy.hpp"
#include "../../../attn_infra/fused_gemm_coord.hpp"
#include "../../../attn_infra/fused_matrix_coord.hpp"
#include "../../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class OutputType_,
    class InputType_,
    class UpdateType_,
    class LseType_,
    LseMode LSE_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2RescaleO<LSE_MODE_, float>,
    OutputType_,
    InputType_,
    UpdateType_,
    LseType_>
{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2RescaleO<LSE_MODE_, float>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementInput = typename InputType_::Element;
    using ElementUpdate = typename UpdateType_::Element;
    using ElementLse = typename LseType_::Element;
    using ElementOutput = typename OutputType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutUpdate = typename UpdateType_::Layout;
    using LayoutLse = typename LseType_::Layout;

    static constexpr LseMode LSE_MODE = DispatchPolicy::LSE_MODE;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;
    static constexpr uint32_t MULTIPLIER = 2;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr float LSE_OUT_INI = std::numeric_limits<float>::infinity();
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t NUM4 = 4;
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr uint32_t SIZE_OF_16BIT = 2;

    struct SplitKVParams {
        bool isSplitkv = false;
        AscendC::GlobalTensor<ElementLse> gCombineLse;
        AscendC::GlobalTensor<ElementLse> gCombineo;
        const LayoutLse* layoutgmLse = nullptr;
        const LayoutInput* layoutgmLo = nullptr;
    };

    __aicore__ inline
    BlockEpilogue() {}

    #include "fused_block_epilogue_rescale_o/fused_block_epilogue_rescale_o_util.inc.hpp"
    #include "fused_block_epilogue_rescale_o/fused_block_epilogue_rescale_o_subcore.inc.hpp"
    #include "fused_block_epilogue_rescale_o/fused_block_epilogue_rescale_o_subcore_ext.inc.hpp"

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        GemmCoord actualBlockShape,
        uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t kvNBlockSize,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        uint32_t isNew)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        uint32_t embedRoundV = (layoutInput.stride(0) == 0) ? BLOCK_SIZE : layoutInput.stride(0);
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embedRoundV;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t kvNSplitSubBlock = kvNBlockSize / subBlockNum;
        uint32_t kvNThisSubBlock = (kvNBlockSize == 1U) ? 0
                                   : (subBlockIdx == 1U) ? (kvNBlockSize - kvNSplitSubBlock)
                                                        : kvNSplitSubBlock;

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? 0
                                 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                      : qNSplitSubBlock)
            : (kvNThisSubBlock * qNBlockSize);

        uint32_t inRowSplitSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock)) :
            (qSBlockSize * qNBlockSize * kvNSplitSubBlock);

        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;

        uint32_t outRowOffsetThisSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0) : 0;
        uint32_t outColOffsetThisSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? 0 : (subBlockIdx * qNSplitSubBlock * embed)) :
            (subBlockIdx * kvNSplitSubBlock * qNBlockSize * embed);

        uint32_t qSThisSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize) : qSBlockSize;

        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        uint32_t outLseRowOffsetThisSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0) : 0;
        uint32_t outLseColOffsetThisSubBlock = (kvNBlockSize == 1U) ?
            ((qNBlockSize == 1U) ? 0 : (subBlockIdx * qNSplitSubBlock)) :
            (subBlockIdx * kvNSplitSubBlock * qNBlockSize);

        int64_t offsetLse =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));

        auto gLseThisSubBlock = gLse[offsetLse];
        auto layoutOutLseThisSubBlock = layoutLse;

        if (inRowActualThisSubBlock > 0U) {
            uint32_t rowLoop = NpuArch::Detail::Alignment::CeilDiv(inRowActualThisSubBlock, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;

            uint32_t epiTokenNum = 0;
            uint32_t integralHeadNum = 0;
            uint32_t proTokenIndex = 0;
            uint32_t proTokenIdxPre = 0;
            uint32_t proTokenNum = 0;
            uint32_t qSRemian = qSThisSubBlock;

            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                int64_t offsetOutput =
                    static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + outOffsetSubBlock;
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput;
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gInputCurLoop = gInput[offsetInput];

                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));
                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                proTokenIndex = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIndex)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                SubCoreCompute(
                    gOutputCurLoop,
                    gInputCurLoop,
                    gUpdateCurLoop,
                    gLseThisSubBlock,
                    layoutOutputCurLoop,
                    layoutInputCurLoop,
                    layoutUpdateCurLoop,
                    layoutOutLseThisSubBlock,
                    qNThisSubBlock,
                    qSThisSubBlock,
                    inRowActualThisSubBlock,
                    isFirstStackTile,
                    isLastStackTile,
                    curStackTileMod,
                    needRowLoop,
                    (rowLoopIdx == rowLoop - 1U),
                    rowOffsetLoop,
                    proTokenIndex,
                    proTokenNum,
                    epiTokenNum,
                    integralHeadNum);
            }
        }
    }

    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLseOutput,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        GemmCoord actualBlockShape,
        uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        int32_t delStartRow, int32_t delEndRow, uint32_t qSeqlen, uint32_t qSBlockIdx, uint32_t curQNBlockTile,
        const SplitKVParams& splitParams = SplitKVParams())
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embedV = actualBlockShape.n();
        uint32_t embedRoundV = (layoutInput.stride(0) == 0) ? BLOCK_SIZE : layoutInput.stride(0);
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embedRoundV;
        uint32_t rowNumTile = NpuArch::Detail::Alignment::RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0
                                : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                    : qNSplitSubBlock;
        uint32_t inRowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embedV;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        int64_t gmlooutOffsetSubBlock = 0;
        if (splitParams.isSplitkv) {
            gmlooutOffsetSubBlock =
                splitParams.layoutgmLo->GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));
        }

        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ?
            inRowOffsetThisSubBlock : 0;
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ?
            0 : subBlockIdx * qNSplitSubBlock;
        int64_t offsetLse =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLseOutput[offsetLse];

        auto layoutOutLseThisSubBlock = layoutLse;

        int64_t gmLseoffsetLse = 0;
        if (splitParams.isSplitkv) {
            gmLseoffsetLse =
                splitParams.layoutgmLse->GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        }

        // Prepare block params for SubCoreCompute
        SplitKVParams blockParams = splitParams;
        if (splitParams.isSplitkv) {
            blockParams.gCombineLse = splitParams.gCombineLse[gmLseoffsetLse];
        }

        if (inRowActualThisSubBlock > 0U) {
            uint32_t rowLoop = NpuArch::Detail::Alignment::CeilDiv(inRowActualThisSubBlock, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;

            // The rows of each cycle consist of multiple heads with several tokens.
            uint32_t epiTokenNum = 0;      // the token num of the epilogue part
            uint32_t integralHeadNum = 0;  // the number of integral heads within a cycle
            uint32_t qSRemian = qSThisSubBlock;
            uint32_t proTokenIdx = 0;      // the token idx of the start token of the prologue part
            uint32_t proTokenIdxPre = 0;   // the token idx of the start token of the pre prologue part
            uint32_t proTokenNum = 0;      // the token num of the prologue part
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                int64_t offsetOutput =
                    static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embedV) + outOffsetSubBlock;

                int64_t gmloffset = 0;
                if (splitParams.isSplitkv) {
                    gmloffset =
                        static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embedV) + gmlooutOffsetSubBlock;
                    blockParams.gCombineo = splitParams.gCombineo[gmloffset];
                }

                auto gOutputCurLoop = gOutput[offsetOutput];
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gInputCurLoop = gInput[offsetInput];
                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embedV));

                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embedV));
                auto layoutOutputCurLoop = layoutOutput;

                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                SubCoreCompute(
                    gOutputCurLoop,
                    gInputCurLoop,
                    gUpdateCurLoop,
                    gLseThisSubBlock,
                    layoutOutputCurLoop,
                    layoutInputCurLoop,
                    layoutUpdateCurLoop,
                    layoutOutLseThisSubBlock,
                    qNThisSubBlock,
                    qSThisSubBlock,
                    inRowActualThisSubBlock,
                    isFirstStackTile,
                    isLastStackTile,
                    curStackTileMod,
                    needRowLoop,
                    (rowLoopIdx == rowLoop - 1U),
                    rowOffsetLoop,
                    proTokenIdx,
                    proTokenNum,
                    epiTokenNum,
                    integralHeadNum,
                    rowOffsetCurLoop,
                    delStartRow,
                    delEndRow,
                    qSeqlen,
                    qSBlockIdx,
                    rowNum,
                    inRowOffsetThisSubBlock,
                    blockParams,
                    curQNBlockTile);
            }
        }
    }


private:
    AscendC::LocalTensor<float> loUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> hmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> tvUbTensor;
    AscendC::LocalTensor<ElementOutput> goUbTensor16;
    AscendC::LocalTensor<float> goUbTensor32;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> lse32_ubuf_tensor;
};
}

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_HPP
