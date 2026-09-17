/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_ALL_TO_ALLV_GROUPED_MAT_MUL_MTE_DEVICE_H
#define OPS_TRANSFORMER_ALL_TO_ALLV_GROUPED_MAT_MUL_MTE_DEVICE_H

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block_mte.hpp"
#include "catccos/comm/block/comm_block_scheduler_alltoallv_allgather.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_alltoallv_allgather.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/ascend950_alltoallv_grouped_matmul_mte.hpp"

namespace OpsTransformer::CatccosMte {

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
          uint32_t M0, uint32_t N0, uint32_t K0, bool ExportAllToAllV = false,
          uint32_t WorkspaceStages = 2, uint32_t MSplit = 1>
struct AllToAllVGroupedMatMulMteConfig {
    static_assert(WorkspaceStages == 2 || WorkspaceStages == 3,
                  "Ascend950 BF16 MTE MoE supports two or three workspace stages");
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr uint32_t kM0 = M0;
    static constexpr uint32_t kN0 = N0;
    static constexpr uint32_t kK0 = K0;
    static constexpr uint32_t kWorkspaceStages = WorkspaceStages;
    static constexpr uint32_t kMSplit = MSplit;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<64>>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    using BlockComm = Catccos::Comm::Block::MteCommBlock<
        ElementA, Catccos::Comm::MteCopyDirection::Put>;
    using BlockExport = Catccos::Comm::Block::MteCommBlock<
        ElementA, Catccos::Comm::MteCopyDirection::Local>;

    using MoeConstraints = Catccos::DGemm::MoeConstraints<1, 8, 64>;
    using BlockMmadScheduler =
        Catccos::DGemm::Block::BlockMmadSchedulerAllToAllVAllGather<MoeConstraints, MSplit>;
    using BlockScheduler = Catccos::Comm::Block::BlockCommSchedulerAllToAllVAllGather<MoeConstraints>;
    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    using Kernel = Catccos::DGemm::Kernel::Ascend950AllToAllVGroupedMatmulMte<
        ProblemShape, BlockMmad, BlockComm, BlockMmadScheduler, BlockScheduler,
        WorkspaceStages, ExportAllToAllV, BlockExport>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

template <uint32_t M0, uint32_t N0, uint32_t K0,
          bool ExportAllToAllV = false, uint32_t WorkspaceStages = 2,
          uint32_t MSplit = 1>
using Bf16AllToAllVGroupedMatMulMteConfig = AllToAllVGroupedMatMulMteConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    M0, N0, K0, ExportAllToAllV, WorkspaceStages, MSplit>;

using AllToAllVGroupedMatMulMteDefaultConfig = AllToAllVGroupedMatMulMteConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    128, 256, 256>;

using AllToAllVGroupedMatMulMteM0_256Config = AllToAllVGroupedMatMulMteConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    256, 128, 256>;

using AllToAllVGroupedMatMulMteDefaultExportConfig = AllToAllVGroupedMatMulMteConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    128, 256, 256, true>;

using AllToAllVGroupedMatMulMteM0_256ExportConfig = AllToAllVGroupedMatMulMteConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    256, 128, 256, true>;

} // namespace OpsTransformer::CatccosMte

#endif // OPS_TRANSFORMER_ALL_TO_ALLV_GROUPED_MAT_MUL_MTE_DEVICE_H
