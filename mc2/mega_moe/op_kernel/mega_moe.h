/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mega_moe.h
 * \brief
 */
#ifndef MEGA_MOE_H
#define MEGA_MOE_H


#include "gmmsq_base.h"
#include "kernel_gmm_swiglu_mxquant.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "tile_copy.hpp"
#include "mega_moe_tiling_def.h"
#include "hccl_shmem.h"
#include "workspace_info.h"

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

using namespace Catlass;
using namespace tla;

namespace MegaMoeImpl {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
};

class MegaMoe {
public:
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR weight1, GM_ADDR weightScales1,
                                GM_ADDR weight2, GM_ADDR weightScales2, GM_ADDR probs, GM_ADDR y, GM_ADDR workspace,
                                const MegaMoeTilingData *tilingData); // TODO: buff[LCOC_MAX_RANK_SIZE] 要改成context
    __aicore__ inline void Process();
};

__aicore__ inline void MegaMoe::Init(
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
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;
    using L1TileShape = tla::Shape<Int<256>, Int<256>, Int<448>>;
    using L0TileShape = tla::Shape<Int<256>, Int<256>, Int<128>>;

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

    using L1TileShape2 = tla::Shape<Int<256>, Int<256>, Int<448>>;
    using L0TileShape2 = tla::Shape<Int<256>, Int<256>, Int<128>>;

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


__aicore__ inline void MegaMoe::Process() {};
  // TODO: 代码结构要改
}  // namespace MegaMoeImpl
#endif