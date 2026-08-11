/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_H
#define OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_H

#include "info.h"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_mte_udma.hpp"
#include "catccos/comm/block/comm_block_scheduler_reducescatter_alltoallv.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/comm/tile/tile_remote_copy_udma_2d.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_reducescatter_alltoallv.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/grouped_matmul_alltoallv_tla.hpp"

namespace OpsTransformer::CatccosUdma {

// The delivery source commit makes the symmetric data-window size a kernel
// template argument. The pinned official CatCCOS base still hard-codes the
// legacy 1000 MiB sync offset, so this thin derived kernel reproduces that
// ToUnderlyingArguments change without patching the third-party checkout.
template <class BaseKernel, uint64_t SymmetricDataBytes>
class CompactGroupedMatMulAllToAllVKernel : public BaseKernel {
public:
    using Arguments = typename BaseKernel::Arguments;
    using Params = typename BaseKernel::Params;
    using ProblemShape = typename BaseKernel::ProblemShape;
    using LayoutTagA = typename BaseKernel::LayoutTagA;
    using LayoutTagB = typename BaseKernel::LayoutTagB;
    using LayoutTagC = typename BaseKernel::LayoutTagC;
    using BlockComm = typename BaseKernel::BlockComm;
    using BlockCommParams = typename BaseKernel::BlockCommParams;
    using BlockCommSchedulerParams = typename BaseKernel::BlockCommSchedulerParams;

    static Params ToUnderlyingArguments(Arguments const &arguments, uint8_t *workspace = nullptr)
    {
        (void)workspace;
        LayoutTagA layoutTagA{arguments.gemmShape.m(), arguments.gemmShape.k()};
        LayoutTagB layoutTagB{arguments.gemmShape.k(), arguments.gemmShape.n()};
        LayoutTagC layoutTagC{arguments.gemmShape.m(), arguments.gemmShape.n()};
        ProblemShape problemShape{
            arguments.gemmShape, arguments.rankSize, arguments.rankIdx,
            arguments.epSize, arguments.expertNum,
            arguments.ptrLocalTokensPerExpert,
            arguments.ptrGlobalTokensPerLocalExpert};
        auto layoutA = tla::MakeLayoutFromTag(layoutTagA);
        auto layoutB = tla::MakeLayoutFromTag(layoutTagB);
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);
        auto gmSymmetric = arguments.ptrSymmetric;
        auto syncMmadFinish = arguments.ptrSymmetric + SymmetricDataBytes;
        auto syncCommFinish = syncMmadFinish + SYNC_UNIT_SIZE;
        typename BlockComm::TileRemoteCopy::Params tileParams{arguments.commTileShape};
        BlockCommParams blockCommParams{arguments.commBlockShape, tileParams};
        BlockCommSchedulerParams blockCommSchedulerParams{arguments.commCoreSplit};
        return Params(
            problemShape, arguments.commInterval, layoutTagC,
            arguments.ptrA, layoutA, arguments.ptrB, layoutB,
            arguments.ptrC, layoutC, gmSymmetric,
            syncMmadFinish, syncCommFinish,
            blockCommParams, blockCommSchedulerParams);
    }
};

template <class ElementA, class LayoutA, class ElementB, class LayoutB,
          class ElementC, class LayoutC, uint32_t M0, uint32_t N0, uint32_t K0>
struct GroupedMatMulAllToAllVUdmaConfig {
    using ArchTag = Catlass::Arch::Ascend950;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<
        ArchTag, true, false, 1, false, 2, 2, 2, 2>;
    using L1TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<K0>>;
    using L0TileShape = tla::Shape<tla::Int<M0>, tla::Int<N0>, tla::Int<64>>;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape,
        ElementA, ElementB, ElementC, void, TileCopy>;

    using MoeConstraints = Catccos::DGemm::MoeConstraints<1, 8, 32>;
    static constexpr bool kIsDynamic = true;
    static constexpr uint32_t kUbStages = 2;
    static constexpr uint32_t kWorkspaceStages = 2;
    static constexpr uint64_t kSymmetricDataBytes = 200UL * 1024 * 1024 * sizeof(uint16_t);
    using BlockMmadScheduler =
        Catccos::DGemm::Block::BlockMmadSchedulerReduceScatterAllToAllV<MoeConstraints>;
    using BlockCommScheduler =
        Catccos::Comm::Block::BlockCommSchedulerReduceScatterAllToAllV<
            MoeConstraints, kIsDynamic, void>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using TileLocalCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, kIsDynamic, CType, CType, void,
        Catccos::detail::CopyDirect::Get, Catccos::detail::CopyTransport::Mte>;
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopyUdma2D<
        ArchTag, CType, CType, Catccos::detail::CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using LocalDispatch = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, kUbStages, kIsDynamic>;
    using RemoteDispatch = Catccos::Comm::AtlasCommUdmaRemoteCopy<ArchTag, kUbStages>;
    using BlockLocalComm = Catccos::Comm::Block::CommBlock<
        LocalDispatch, CType, CType, void, TileLocalCopy, TileScheduler>;
    using BlockRemoteComm = Catccos::Comm::Block::CommBlock<
        RemoteDispatch, CType, CType, TileRemoteCopy>;
    using BlockComm = Catccos::Comm::Block::CommBlockMteUdma<BlockLocalComm, BlockRemoteComm>;

    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    using BaseKernel = Catccos::DGemm::Kernel::GroupedMatmulAllToAllVTla<
        ProblemShape, BlockMmad, BlockComm, BlockMmadScheduler,
        BlockCommScheduler, kWorkspaceStages>;
    using Kernel = CompactGroupedMatMulAllToAllVKernel<BaseKernel, kSymmetricDataBytes>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

using GroupedMatMulAllToAllVUdmaDefaultConfig = GroupedMatMulAllToAllVUdmaConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    128, 256, 256>;

} // namespace OpsTransformer::CatccosUdma

#endif // OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_H
