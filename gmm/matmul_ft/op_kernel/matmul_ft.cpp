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
 * \file matmul_ft.cpp
 * \brief
 */

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "matmul_ft.h"
#include "matmul_ft_tiling_key.h"
#include "lib/matmul_intf.h"

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemv/block/block_gemv.hpp"

#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemv/tile/tile_fault_sum.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/tile/tile_matmul_elem_add.hpp"
#include "catlass/gemv/tile/tile_threshold.hpp"      // TileThreCalc
#include "catlass/gemv/tile/tile_std_estimate.hpp"   // TileStdEstRobust
#include "catlass/gemm_coord.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/gemm/dispatch_policy.hpp"

#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"

#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/gemv/helper.hpp"

using GemmInTypeC = DTYPE_A;
using GemmInTypeN = DTYPE_A;

using GemmOutTypeC = float;
using GemmOutTypeN = float;

using GemvInTypeCforCE = DTYPE_X;
using GemvInTypeNforCE = DTYPE_X;

using GemvInTypeCforAB = DTYPE_XV;
using GemvInTypeNforAB = DTYPE_XV;


using GemvOutTypeC = float;
using GemvOutTypeN = float;

using ScalarTypeC = float;
using ScalarTypeN = float;

using ScalarType = float;

constexpr bool enableUnitFlag = false;
constexpr bool enableShuffleK = true;

template <uint32_t schMode>
__global__ __aicore__ void matmul_ft(
    GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR xv, GM_ADDR vxForAe,
    GM_ADDR zRow, GM_ADDR zCol, GM_ADDR dRow, GM_ADDR dCol,
    GM_ADDR compRow, GM_ADDR compCol, GM_ADDR c, GM_ADDR threshold,
    GM_ADDR bMeanAbs, GM_ADDR bMeanSquare, GM_ADDR bVar,
    GM_ADDR be, GM_ADDR beForAiv, GM_ADDR bMaxSlice, GM_ADDR bMinSlice,
    GM_ADDR aMax, GM_ADDR aMean, GM_ADDR aMin,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MatmulFtTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulFtTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);

#if (ORIG_DTYPE_A == DT_FLOAT16)
    using TilingPolicy = MatmulFtFp16TilingPolicy;
#elif (ORIG_DTYPE_A == DT_FLOAT)
    using TilingPolicy = MatmulFtFp32TilingPolicy;
#else
    using TilingPolicy = MatmulFtBf16TilingPolicy;
#endif
    using L1TileShape =
        Catlass::GemmShape<128, 256, TilingPolicy::L1TileShape_K>;
    using L0TileShape =
        Catlass::GemmShape<128, 256, TilingPolicy::L0TileShape_K>;

    using LayoutX = Catlass::layout::VectorLayout;
    using LayoutY = Catlass::layout::VectorLayout;
    using LayoutCOMP = Catlass::layout::VectorLayout;

    using LayoutA = Catlass::layout::RowMajor;
    using LayoutACol = Catlass::layout::ColumnMajor;

    using LayoutB = Catlass::layout::RowMajor;
    using LayoutBCol = Catlass::layout::ColumnMajor;

    using LayoutC = Catlass::layout::RowMajor;
    using LayoutCCol = Catlass::layout::ColumnMajor;

    using LayoutZ = Catlass::layout::VectorLayout;
    using FT_COMP_TYPE = FTSelf::Gemv::helper::FT_COMP_TYPE;


    using ArchTag = Catlass::Arch::AtlasA2;
    using FT_REDUCE_TYPE = FTSelf::Gemv::helper::FT_REDUCE_TYPE;

    using LayoutMY = Catlass::layout::RowMajor;
    using LayoutMX = Catlass::layout::RowMajor;

    using FT_ENC_TYPE = FTSelf::Gemv::helper::FT_ENC_TYPE;
    using FT_RCE_THRE_TYPE = FTSelf::Gemv::helper::FT_RCE_THRE_TYPE;
    using FT_L02L1_TYPE = FTSelf::Gemv::helper::FT_L02L1_TYPE;
    using FT_AIC_BE_SCHEME = FTSelf::Gemv::helper::FT_AIC_BE_SCHEME;

    #if (ORIG_DTYPE_A == DT_FLOAT16 || ORIG_DTYPE_A == DT_FLOAT)
        constexpr FT_AIC_BE_SCHEME BlockFTGemvAIC_BE_SCHEME = FT_AIC_BE_SCHEME::ROWCOMPLETE;
        constexpr FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE BlockFTSum_A_B = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED;
    #else
        constexpr FT_AIC_BE_SCHEME BlockFTGemvAIC_BE_SCHEME = FT_AIC_BE_SCHEME::ROWCOMPLETE_BF;
        constexpr FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE BlockFTSum_A_B = FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_MIXED_BF;
    #endif


    // using L1TileShapeC = GemvShape<32, 512>;
    // using L0TileShapeC = GemvShape<32, 256>;

    using L1TileShapeAB = Catlass::GemmShape<128, 128, 128>;
    using L0TileShapeAB = Catlass::GemmShape<128, 128, 128>;

    using L1TileShapeBE =
        Catlass::GemvShape<TilingPolicy::L1TileShapeBE_M, 256>;
    using L0TileShapeBE =
        Catlass::GemvShape<TilingPolicy::L0TileShapeBE_M, 64>;

    using UBBlockShapeBE = Catlass::GemvShape<L1TileShapeBE::M*2, L1TileShapeBE::N*1>;

    using AType = Catlass::Gemm::GemmType<GemmInTypeN, LayoutA>;
    using BType = Catlass::Gemm::GemmType<GemmInTypeN, LayoutB>;

    using CType = Catlass::Gemm::GemmType<GemmOutTypeN, LayoutC>;

    using XTypeAIC = Catlass::Gemm::GemmType<GemvInTypeNforAB, LayoutX>;
    using YTypeBEAIC = Catlass::Gemm::GemmType<GemvInTypeNforAB, LayoutMY>;
    using BiasType = void;

    using XTypeAIV = Catlass::Gemm::GemmType<GemvInTypeNforCE, LayoutX>;

    using GemvDispatchPolicy = Catlass::Gemm::GemvAtlasA2;
    using COMPDispatchPolicy = Catlass::Gemm::GemvAtlasA2;
    using BeAICDispatchPolicy = Catlass::Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using GEMVAICDispatchPolicy = Catlass::Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using TileCopyGemvAic = FTSelf::Gemv::Tile::TileCopyGemvAic<typename BeAICDispatchPolicy::ArchTag, BType, XTypeAIC, YTypeBEAIC, BiasType>;
    using TileMmadGemvAic = Catlass::Gemm::Tile::TileMmad<typename  BeAICDispatchPolicy::ArchTag, XTypeAIC, BType, BiasType>;

    using BlockFTGemvAIC = FTSelf::Gemv::Block::BlockFTGemvBe<BeAICDispatchPolicy,
        BlockFTGemvAIC_BE_SCHEME,
        UBBlockShapeBE, L1TileShapeBE, L0TileShapeBE,
        BType, XTypeAIC, YTypeBEAIC, BiasType, TileCopyGemvAic, TileMmadGemvAic>;

    using ZType = Catlass::Gemm::GemmType<GemvOutTypeN, LayoutZ>;

    static constexpr FT_AIC_BE_SCHEME BE_SCHEME = BlockFTGemvAIC::BE_SCHEME;

    constexpr uint32_t computeLength = 8192;

    using YType = Catlass::Gemm::GemmType<GemvInTypeNforCE, LayoutY>;

    using BRedType = Catlass::Gemm::GemmType<GemvInTypeNforCE, LayoutB>;
    using TileVmuls = Catlass::Gemv::Tile::TileVmuls<ArchTag, XTypeAIV>;

    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;


    using L1TileShapeFirst = Catlass::GemmShape<
        256, 256, TilingPolicy::L1TileShapeFirst_K>;
    using L0TileShapeFirst = Catlass::GemmShape<
        256, 256, TilingPolicy::L0TileShapeFirst_K>;

    using L1TileShapeforFT = Catlass::GemmShape<L1TileShapeFirst::M, 16, L1TileShapeFirst::K>;
    using L0TileShapeforFT = Catlass::GemmShape<L0TileShapeFirst::M, 16, L0TileShapeFirst::K>;

    using MXType = Catlass::Gemm::GemmType<GemvInTypeNforAB, LayoutMX>;
    using MYType = Catlass::Gemm::GemmType<GemvOutTypeN, LayoutMY>;

    using BlockMmadABe = FTSelf::Gemm::Block::BlockMmadSpecABeNoSplitKRobust<
        MmadDispatchPolicy,
        L1TileShapeforFT,
        L0TileShapeforFT,
        AType, BType, CType, MXType, MYType, BiasType>;

    using BlockMmadPreload = FTSelf::Gemm::Block::BlockMmadPreload<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using BlockSchedulerFirst = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    using TileFaultCopyRedAiv = FTSelf::Gemv::Tile::TileCopyFTRedAiv<ArchTag,
        AType, BType, YType, ZType>;

    using UBTileShapeforB =
        Catlass::GemvShape<TilingPolicy::UBTileShapeforB_M, L0TileShape::N>;
    using UBBlockShapeforB = Catlass::GemvShape<UBTileShapeforB::M*2, UBTileShapeforB::N*2>;

    using UBTileShapeforA =
        Catlass::GemvShape<TilingPolicy::UBTileShapeforA_M, 256>;

    using ARedType = Catlass::Gemm::GemmType<GemvInTypeNforCE, LayoutA>;
    using TileFaultSum = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::MAX_MIN, ARedType, ZType>;

    using BlockFTSum = FTSelf::Gemv::Block::BlockFTSumNoSplitK<
        GemvDispatchPolicy,
        FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        BlockFTSum_A_B,
        UBTileShapeforB, UBBlockShapeforB, UBTileShapeforA,
        L1TileShape, AType, BType, YType, ZType, void,
        TileFaultCopyRedAiv, TileFaultSum>;

    using SliceSumDispatchPolicy = Catlass::Gemm::GemvAtlasA2;

    using SliceSumUBTileShape = Catlass::GemvShape<8, 256>;
    using TileMatrixAddforABEReduce = FTSelf::Gemv::Tile::TileMatmulAdd<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>;
    using TileCopyMatrixAddforABEReduce = FTSelf::Gemv::Tile::TileCopyMatrixAddAiv<
        typename SliceSumDispatchPolicy::ArchTag, MYType, MYType, void>;

    using BlockSliceSum = FTSelf::Gemv::Block::BlockSliceKMNSum<SliceSumDispatchPolicy,
        FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::ABE_FUSED_THRE,
        SliceSumUBTileShape, MYType, MYType, void, TileCopyMatrixAddforABEReduce, TileMatrixAddforABEReduce>;

    using MeanMaxTileVmuls = Catlass::Gemv::Tile::TileVmuls<typename GemvDispatchPolicy::ArchTag, ZType>;
    using UBTileShapeforBRed =
        Catlass::GemvShape<TilingPolicy::UBTileShapeforBRed_M, TilingPolicy::UBTileShapeforBRed_N>;
    using TileFaultSumBReduce = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM_MAX, BRedType, ZType>;
    using TileFaultCopyBReduce = FTSelf::Gemv::Tile::TileCopyGemvAiv<ArchTag, BRedType, YType, ZType>;

    using BlockSliceRed = FTSelf::Gemv::Block::BlockSliceKMNSum<
        SliceSumDispatchPolicy,
        FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::A_B_ROBUST,
        SliceSumUBTileShape,
        UBTileShapeforBRed,
        MYType, BRedType, ZType,
        MYType, void,
        TileCopyMatrixAddforABEReduce,
        TileFaultCopyBReduce,
        TileMatrixAddforABEReduce,
        TileFaultSumBReduce,
        MeanMaxTileVmuls>;

    using UBTileShapeCE = Catlass::GemvShape<64, L1TileShape::N>;
    using UBBlockShapeCE = Catlass::GemvShape<L1TileShape::M, UBTileShapeCE::N>;

    using COMPZType = Catlass::Gemm::GemmType<uint8_t, LayoutZ>;
    using TileFaultCopyCE = FTSelf::Gemv::Tile::TileCopyGemvThreCompFusedAiv<ArchTag,
        CType, CType, XTypeAIV, ZType, COMPZType, void>;

    using ThreCalcDispatchPolicy = Catlass::Gemm::GemvAtlasA2;
    using TileThreCalc = FTSelf::Gemv::Tile::TileThreCalc<
        typename ThreCalcDispatchPolicy::ArchTag,
        FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        CType, ZType, ZType, BiasType>;

    using TileStdEst = FTSelf::Gemv::Tile::TileStdEstRobust<
        typename ThreCalcDispatchPolicy::ArchTag,
        ZType,
        ZType
    >;

    using TileFaultSumCSum = FTSelf::Gemv::Tile::TileFaultSum<ArchTag, FT_REDUCE_TYPE::SUM, CType, ZType>;

    using BlockFTGemvAIV = FTSelf::Gemv::Block::BlockFTGemvCENoSplitKPreload<
        GemvDispatchPolicy,
        FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
        FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE::THRE_FUSED,
        FTSelf::Gemv::helper::FT_ENC_TYPE::RCE,
        FTSelf::Gemv::helper::FT_COMP_TYPE::RSUB,
        FTSelf::Gemv::helper::FT_ABE_TYPE::CENTRAL_BLOCK,
        UBTileShapeCE, UBBlockShapeCE, L1TileShape,
        CType, XTypeAIV, ZType, COMPZType, void,
        TileFaultCopyCE, TileFaultSumCSum,
        TileThreCalc, TileStdEst>;

    using UBTileShape = Catlass::GemmShape<L1TileShapeAB::M, L1TileShapeAB::N, L1TileShapeAB::K>;
    using TileFaultCopy = FTSelf::Gemv::Tile::TileCopyGemvAiv<ArchTag, AType, XTypeAIV, ZType>;
    using TileFaultVmad = FTSelf::Gemv::Tile::TileVmad<ArchTag, AType, XTypeAIV, ZType>;

    using BlockSumGemv = FTSelf::Gemv::Block::BlockSumGemv<GemvDispatchPolicy, UBTileShape, AType, XTypeAIV, ZType, void, TileFaultCopy, TileFaultVmad, TileVmuls>;

    using BlockConfig = MatfulFTKernel::BlockConfig<
        BlockMmadABe, BlockMmadPreload, BlockSchedulerFirst, BlockScheduler,
        BlockFTGemvAIC, BlockFTSum, BlockFTGemvAIV,
        BlockSliceRed, BlockSliceSum>;

    using MatmulFTKernel = MatfulFTKernel::MatmulAsVarABonAicSplitRobustPreload<BlockConfig>;

#if (ORIG_DTYPE_A == DT_FLOAT16) || (ORIG_DTYPE_A == DT_BF16) || (ORIG_DTYPE_A == DT_FLOAT)
    if constexpr (schMode == MATMUL_FT_TPL_SCH_MODE_BF16) {
        typename MatmulFTKernel::Arguments arguments{
            Catlass::GemmCoord{
                tilingData.compute.problemGemmShape.m,
                tilingData.compute.problemGemmShape.n,
                tilingData.compute.problemGemmShape.k},
            Catlass::GemvCoord{tilingData.compute.problemShape.m, tilingData.compute.problemShape.n},
            sizeof(GemmInTypeN),
            x,
            xv,
            a,
            b,
            c,
            zRow,
            zCol,
            dRow,
            dCol,
            compRow,
            compCol,
            be,
            beForAiv,
            bMaxSlice,
            bMinSlice,
            bMeanAbs,
            bMeanSquare,
            bVar,
            vxForAe,
            aMean,
            aMax,
            aMin,
            threshold,
            FT_ENC_TYPE::RCE,
            1,
            static_cast<bool>(tilingData.compute.outputWorkspace),
            0.0f,
            0.0f,
            tilingData.compute.eMax,
            static_cast<bool>(tilingData.compute.outputThre),
            static_cast<bool>(tilingData.compute.outputCE),
            tilingData.compute.splitKNum
        };

        typename MatmulFTKernel::Params params =
            MatmulFTKernel::ToUnderlyingArguments(arguments, userWorkspace, tilingData.compute);
        MatmulFTKernel op;
        op(params);
    }
#endif
}
