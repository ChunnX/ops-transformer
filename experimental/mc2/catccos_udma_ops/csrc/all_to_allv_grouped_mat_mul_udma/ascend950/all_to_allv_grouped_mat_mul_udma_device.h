/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_DEVICE_H
#define OPS_TRANSFORMER_ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_DEVICE_H

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
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_mte_udma.hpp"
#include "catccos/comm/block/comm_block_scheduler_alltoallv_allgather.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/comm/tile/tile_remote_copy_udma_2d.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_alltoallv_allgather.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/ascend950_alltoallv_grouped_matmul.hpp"

namespace OpsTransformer::CatccosUdma {

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
          uint32_t M0, uint32_t N0, uint32_t K0>
struct AllToAllVGroupedMatMulUdmaConfig {
    using ArchTag = Catlass::Arch::Ascend950;
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

    static constexpr bool kIsDynamic = true;
    static constexpr uint32_t kUbStages = 2;
    static constexpr uint32_t kWorkspaceStages = 2;
    using TileLocalCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, kIsDynamic, AType, AType, void, Catccos::detail::CopyDirect::Put,
        Catccos::detail::CopyTransport::Mte>;
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopyUdma2D<
        ArchTag, AType, AType, Catccos::detail::CopyDirect::Put>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using LocalDispatch = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, kUbStages, kIsDynamic>;
    using RemoteDispatch = Catccos::Comm::AtlasCommUdmaRemoteCopy<ArchTag, kUbStages>;
    using BlockLocalComm = Catccos::Comm::Block::CommBlock<
        LocalDispatch, AType, AType, void, TileLocalCopy, TileScheduler>;
    using BlockRemoteComm = Catccos::Comm::Block::CommBlock<
        RemoteDispatch, AType, AType, TileRemoteCopy>;
    using BlockComm = Catccos::Comm::Block::CommBlockMteUdma<BlockLocalComm, BlockRemoteComm>;

    using MoeConstraints = Catccos::DGemm::MoeConstraints<1, 8, 64>;
    using BlockMmadScheduler = Catccos::DGemm::Block::BlockMmadSchedulerAllToAllVAllGather<MoeConstraints>;
    using BlockScheduler = Catccos::Comm::Block::BlockCommSchedulerAllToAllVAllGather<MoeConstraints>;
    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    using Kernel = Catccos::DGemm::Kernel::Ascend950AllToAllVGroupedMatmul<
        ProblemShape, BlockMmad, BlockComm, BlockMmadScheduler, BlockScheduler, kWorkspaceStages, true>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

using AllToAllVGroupedMatMulUdmaDefaultConfig = AllToAllVGroupedMatMulUdmaConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    128, 256, 256>;

} // namespace OpsTransformer::CatccosUdma

#endif // OPS_TRANSFORMER_ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_DEVICE_H
