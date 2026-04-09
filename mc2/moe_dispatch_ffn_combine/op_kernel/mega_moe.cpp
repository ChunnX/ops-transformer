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
 * \file mega_moe.cpp
 * \brief
 */
#include "mega_moe.h"
#include "kernel_operator.h"

using namespace MegaMoeImpl;

template <uint32_t M0, uint32_t N0, uint32_t K0>
__aicore__ inline void mega_moe_impl(
            GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR weight1, GM_ADDR weightScales1,
            GM_ADDR weight2, GM_ADDR weightScales2, GM_ADDR probs, GM_ADDR y, GM_ADDR workspace,
            const MegaMoeTilingData *tilingData) 
{
    uint32_t m = tilingData->m;
    uint32_t k = tilingData->k;
    uint32_t n = tilingData->n;
    uint32_t expertPerRank = tilingData->expertPerRank;
    uint8_t groupListType = tilingData->groupListType;
    uint32_t mxScaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    // 定义矩阵的类型和布局
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e4m3fn_t;
    using CType = fp8_e4m3fn_t;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutMMadOut = layout::RowMajor;
    
    using xScaleType = AscendC::fp8_e8m0_t;
    using weightScaleType = AscendC::fp8_e8m0_t;
    using MMadOutType = bfloat16_t; // 由float-->float16
    auto layoutMxScaleA = tla::MakeMxScaleLayout<xScaleType, LayoutA, false>(m, mxScaleK); // bool: is scaleB
    auto layoutMxScaleB = tla::MakeMxScaleLayout<weightScaleType, LayoutB, true>(mxScaleK, n);

    // 定义scheduler类型
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;
    using L1TileShape = tla::Shape<Int<128>, Int<256>, Int<512>>;
    using L0TileShape = tla::Shape<Int<128>, Int<256>, Int<128>>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = BlockEpilogueSwigluQuant<L1TileShape, CType, MMadOutType, weightScaleType, weightScaleType, true>;
    
    using TileCopy = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, AType, LayoutA, BType, LayoutB, weightScaleType, decltype(layoutMxScaleA), weightScaleType,
        decltype(layoutMxScaleB), MMadOutType, LayoutMMadOut, void>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, AType, BType, MMadOutType, void, TileCopy>;
    
    // gmm2
    uint32_t mx2ScaleN = CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    using B2Type = fp8_e4m3fn_t;
    using MMadOut2Type = bfloat16_t;
    using LayoutB2 = layout::ColumnMajor;
    using LayoutMMadOut2 = layout::RowMajor;
    using x2ScaleType = AscendC::fp8_e8m0_t;
    using weight2ScaleType = AscendC::fp8_e8m0_t;
    
    auto layoutMx2ScaleA = tla::MakeMxScaleLayout<x2ScaleType, LayoutMMadOut, false>(m, mx2ScaleN);
    auto layoutMx2ScaleB = tla::MakeMxScaleLayout<weight2ScaleType, LayoutB2, true>(mx2ScaleN, k);

    using L1TileShape2 = tla::Shape<Int<128>, Int<256>, Int<512>>;
    using L0TileShape2 = tla::Shape<Int<128>, Int<256>, Int<128>>;

    using TileCopy2 = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, CType, LayoutMMadOut, B2Type, LayoutB2, weight2ScaleType, decltype(layoutMx2ScaleA), weight2ScaleType,
        decltype(layoutMx2ScaleB), MMadOut2Type, LayoutMMadOut2, void>;
    using BlockMmad2 = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape2, L0TileShape2, CType, B2Type, MMadOut2Type, void, TileCopy2>;

    using QGmmKernel = KernelGmmSwiGluMixOnlineDynamic<BlockMmad, BlockMmad2, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
    GM_ADDR buff[HCCL_MAX_RANK_SIZE];
    for (int i = 0; i < tilingData->EP; i++) {
        buff[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }
    
    Params params = {
        .aGmAddr = x,
        .expertIdxGmAddr = expertIdx,
        .bGmAddr = weight1,
        .bScaleGmAddr = weightScales1,
        .b2GmAddr = weight2,
        .b2ScaleGmAddr = weightScales2,
        .y2GmAddr = y,
        .probsGmAddr = probs,
        .workspaceInfo = WorkspaceInfo(workspace, tilingData),
        .peermemInfo = PeermemInfo(buff[tilingData->rank], tilingData),
        .shmem = HcclShmem(buff, tilingData->rank, tilingData->EP),
        .tilingData = tilingData
    };

    QGmmKernel op;
    op(params);
};



extern "C" __global__ __aicore__ void mega_moe(
  GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
  GM_ADDR weight1, GM_ADDR weightScales1, GM_ADDR weight2, GM_ADDR weightScales2,
  GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
  REGISTER_TILING_DEFAULT(MegaMoeTilingData);

  GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);
  MegaMoe op;

  uint32_t m0 = tilingData.m0;
  uint32_t k0 = tilingData.k0;
  uint32_t n0 = tilingData.n0;
  if (m0 == NUM_128 && n0 == NUM_256 && k0 == NUM_512) {
      mega_moe_impl<NUM_128, NUM_256, NUM_512>(mc2Context, x, expertIdx, weight1,
        weightScales1, weight2, weightScales2, expertScales, y, workspaceGM, &tilingData);
  } else if (m0 == NUM_256 && n0 == NUM_256 && k0 == NUM_384) {
      mega_moe_impl<NUM_256, NUM_256, NUM_384>(mc2Context, x, expertIdx, weight1,
        weightScales1, weight2, weightScales2, expertScales, y, workspaceGM, &tilingData);
  }

  // op.Init();
  // op.Process();
}
