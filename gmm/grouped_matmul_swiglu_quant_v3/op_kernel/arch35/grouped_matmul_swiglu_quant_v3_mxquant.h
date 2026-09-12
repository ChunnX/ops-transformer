/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Cgmct entry for the Ascend 950 MXFP8 weight-NZ V3 path.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V3_MXQUANT_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V3_MXQUANT_H

#include "cgmct/kernel/kernel_gmm_swiglu_mxquant.h"
#include "cgmct/block/block_mx_mm_aic_to_aiv_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

namespace Cgmct {
namespace Gemm {
namespace Tile {

// Mode 1 consumes adjacent (even, odd) MM outputs.  The weight-NZ matrix is
// laid out by N rows, so one strided GM->L1 copy selects every other row and
// packs the selected rows into the ordinary NZ tile expected by MMAD.
template <class ArchTag, const auto &COPY_CFG>
struct SwigluMode1CopyGmToB1 {
    template <typename TransT, typename SrcT>
    __aicore__ inline void operator()(const AscendC::LocalTensor<TransT> &dst,
                                      AscendC::GlobalTensor<SrcT> srcNz, int curRow, int curCol, int tileHeight,
                                      int tileWidth, int baseHeight, int baseWidth, int orgHeight, int orgWidth,
                                      bool iskRowDirec)
    {
        (void)orgWidth;
        (void)iskRowDirec;
        constexpr int32_t c0Size = AscendC::AuxGetC0Size<SrcT>();
        const int32_t alignedGRow = AscendC::Ceil(orgHeight, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
        const int32_t row = curRow * baseHeight;
        const int32_t col = curCol * baseWidth;
        const int32_t alignHeight = AscendC::Ceil(tileHeight, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
        const int32_t rowBytes = alignHeight * c0Size * static_cast<int32_t>(sizeof(TransT));
        const int32_t blockLen = rowBytes / AscendC::ONE_BLK_SIZE;
        const int32_t srcStride = (2 * alignedGRow - alignHeight) * c0Size * static_cast<int32_t>(sizeof(TransT)) /
                                  AscendC::ONE_BLK_SIZE;
        const int64_t srcOffset = static_cast<int64_t>(row) * c0Size +
                                  static_cast<int64_t>(2 * col) * alignedGRow;

        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(AscendC::Ceil(tileWidth, c0Size));
        params.blockLen = static_cast<uint16_t>(blockLen);
        params.srcStride = static_cast<uint16_t>(srcStride);
        params.dstStride = 0;
        AscendC::DataCopy(dst, srcNz[srcOffset], params);
    }
};

struct SwigluMode1TileCopy {
    using BaseTileCopy = TileCopy<Arch::DAV3510, CopyInAndCopyOutSplitMWithParams>;

    template <class InputType, const auto &COPY_CFG>
    using CopyGmToA1 = typename BaseTileCopy::template CopyGmToA1<InputType, COPY_CFG>;

    template <class InputType, const auto &COPY_CFG>
    using CopyGmToB1 = SwigluMode1CopyGmToB1<Arch::DAV3510, COPY_CFG>;

    template <class InputType, class OutputType, typename T = void>
    using CopyCo1ToOut = typename BaseTileCopy::template CopyCo1ToOut<InputType, OutputType, T>;
};

} // namespace Tile
} // namespace Gemm
} // namespace Cgmct

template <typename layoutA, typename layoutB, int64_t SWIGLU_MODE>
__aicore__ inline void GmmSwigluAswtV3(
    GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
    GM_ADDR smoothScale, GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMSwigluQuantV3TilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMSwigluQuantV3TilingDataParams, mmTilingData, mmTilingData_, tiling);

    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using WeightScaleType = AscendC::fp8_e8m0_t;
    using BiasType = float;
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    using C1Type = float;
    using BlockEpilogue =
        Block::BlockEpilogueSwigluQuant<L0TileShape, CType, C1Type, WeightScaleType, WeightScaleType, true>;
    using ProblemShape = MatmulShape;
    using DefaultTileCopy = Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>;
    using TileCopy = AscendC::Std::conditional_t<SWIGLU_MODE == 1, Tile::SwigluMode1TileCopy, DefaultTileCopy>;
    using BlockMmad =
        Block::BlockMxMmAicToAivBuilder<AType, LayoutA, BType, LayoutB, BiasType, C1Type, LayoutC, L1TileShape,
                                        L0TileShape, BlockScheduler, QuantMatmulWithTileMultiBlock<>,
                                        TileCopy>;
    using QGmmKernel = Kernel::KernelGmmSwiGluMixOnlineDynamic<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    using EpilogueArguments = typename QGmmKernel::BlockEpilogueArguments;

    GMMTiling gmmParams{gmmSwigluQuantParams_.groupNum,
                        gmmSwigluQuantParams_.groupListType,
                        mmTilingData_.baseM,
                        mmTilingData_.baseN,
                        mmTilingData_.baseK,
                        gmmSwigluQuantParams_.isMxWeightNzMultiTensor};
    gmmParams.matmulTiling = &mmTilingData_;

    EpilogueArguments epilogueArgs{y,
                                   yScale,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   static_cast<uint32_t>(mmTilingData_.baseM),
                                   static_cast<uint32_t>(mmTilingData_.baseN),
                                   gmmSwigluQuantParams_.swigluMode,
                                   gmmSwigluQuantParams_.activateLeft != 0,
                                   gmmSwigluQuantParams_.clampLimit,
                                   gmmSwigluQuantParams_.gluAlpha,
                                   gmmSwigluQuantParams_.gluBias};
    Params params = {{1, 1, 1, 1},
                     {x, weight, weightScale, xScale, y, groupList},
                     epilogueArgs,
                     gmmParams};
    QGmmKernel op;
    op(params);
}

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V3_MXQUANT_H
