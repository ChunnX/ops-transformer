/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_H
#define OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_H

#include <type_traits>

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
#include "catccos/comm/block/comm_block_scheduler_gmm_alltoallv_target_order.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/comm/tile/tile_remote_copy_udma_2d.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_gmm_alltoallv_target_order.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/grouped_matmul_alltoallv_tla.hpp"

namespace OpsTransformer::CatccosUdma {

// The target-order kernel gets its dynamic block/tile shape from the local-MTE
// block Params.  CatCCOS' official UDMA CommBlock does not expose a Params
// alias because its TileRemoteCopy builds and completes the WQE internally.
// This adapter supplies only that host/device construction ABI; the actual
// block operator, UDMA GET and quiet remain the official CatCCOS code path.
template <class OfficialBlock_, class ParamsBlock_>
class OfficialUdmaBlockAdapter : public OfficialBlock_ {
public:
    using Base = OfficialBlock_;
    // Keep the same host-side dynamic block/tile parameter ABI as the local
    // MTE block.  The official UDMA block itself has no Params alias and its
    // TileRemoteCopy owns a different Params type, but its device constructor
    // accepts an arbitrary parameter object.
    using Params = typename ParamsBlock_::Params;
    using TileRemoteCopy = typename ParamsBlock_::TileRemoteCopy;
    using ElementSrc = typename Base::ElementSrc;
    using LayoutSrc = typename Base::LayoutSrc;
    using ElementDst = typename Base::ElementDst;
    using LayoutDst = typename Base::LayoutDst;

    template <class ResourceArch>
    CATLASS_DEVICE
    OfficialUdmaBlockAdapter(
        Catlass::Arch::Resource<ResourceArch> &resource,
        Params const &params)
        : Base(resource, params)
    {
    }
};

// Our overlay CommBlock supports deferred checked quiet for the custom tiles.
// The official TileRemoteCopy instead performs quiet inside operator(); expose
// that existing contract to the overlay without changing the official body.
template <class OfficialTile_>
class OfficialUdmaTileAdapter : public OfficialTile_ {
public:
    static constexpr bool DEFER_QUIET = false;
};

template <class ElementA, class LayoutA, class ElementB, class LayoutB,
          class ElementC, class LayoutC, uint32_t M0, uint32_t N0, uint32_t K0,
          bool UseMte3Udma = false, bool UseOfficialUdmaBlock = false>
struct GroupedMatMulAllToAllVUdmaConfig {
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr uint32_t kM0 = M0;
    static constexpr uint32_t kN0 = N0;
    static constexpr uint32_t kK0 = K0;
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
        Catccos::DGemm::Block::BlockMmadSchedulerGmmAllToAllVTargetOrder<MoeConstraints>;
    using BlockCommScheduler =
        Catccos::CommEpilogue::Block::BlockCommSchedulerGmmAllToAllVTargetOrder<
            MoeConstraints, kIsDynamic>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using TileLocalCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, kIsDynamic, CType, CType, void,
        Catccos::detail::CopyDirect::Get, Catccos::detail::CopyTransport::Mte>;
    using TileRemoteCopy = std::conditional_t<
        UseMte3Udma,
        Catccos::Comm::Tile::TileRemoteCopyUdma2DWithScratch<
            ArchTag, CType, CType, Catccos::detail::CopyDirect::Get>,
        Catccos::Comm::Tile::TileRemoteCopyUdma2D<
            ArchTag, CType, CType, Catccos::detail::CopyDirect::Get>>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using LocalDispatch = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, kUbStages, kIsDynamic>;
    using RemoteDispatch = Catccos::Comm::AtlasCommUdmaRemoteCopy<ArchTag, kUbStages>;
    using BlockLocalComm = Catccos::Comm::Block::CommBlock<
        LocalDispatch, CType, CType, void, TileLocalCopy, TileScheduler>;
    using BlockRemoteComm = Catccos::Comm::Block::CommBlock<
        RemoteDispatch, CType, CType, TileRemoteCopy>;
    using HybridBlockComm =
        Catccos::Comm::Block::CommBlockMteUdma<BlockLocalComm, BlockRemoteComm>;
    using OfficialTile = OfficialUdmaTileAdapter<
        Catccos::Comm::Tile::TileRemoteCopy<
            ArchTag, kIsDynamic, CType, CType, void,
            Catccos::detail::CopyDirect::Get,
            Catccos::detail::CopyTransport::Udma>>;
    using OfficialBlockBase = Catccos::Comm::Block::CommBlock<
        RemoteDispatch, CType, CType, OfficialTile>;
    using OfficialBlockComm = OfficialUdmaBlockAdapter<
        OfficialBlockBase, BlockLocalComm>;
    using BlockComm = std::conditional_t<
        UseOfficialUdmaBlock, OfficialBlockComm, HybridBlockComm>;

    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    using Kernel = Catccos::DGemm::Kernel::GroupedMatmulAllToAllVTla<
        ProblemShape, BlockMmad, BlockComm, BlockMmadScheduler,
        BlockCommScheduler, kWorkspaceStages, kSymmetricDataBytes, true,
        UseOfficialUdmaBlock>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

using GroupedMatMulAllToAllVUdmaDefaultConfig = GroupedMatMulAllToAllVUdmaConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    128, 256, 256>;

using GroupedMatMulAllToAllVUdmaM0_256Config = GroupedMatMulAllToAllVUdmaConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    256, 128, 256>;

// Validation-only variants. They keep the same GMM and communication tiling,
// but publish UDMA GET WQEs through the official PIPE_MTE3 + UB scratch path.
using GroupedMatMulAllToAllVUdmaMte3Config = GroupedMatMulAllToAllVUdmaConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    128, 256, 256, true>;

using GroupedMatMulAllToAllVUdmaM0_256Mte3Config = GroupedMatMulAllToAllVUdmaConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    256, 128, 256, true>;

// Validation-only variants using CatCCOS' official UDMA block call. The
// official TileRemoteCopy creates its own TPipe/TBuf WQE scratch, submits one
// UDMA GET and performs its void quiet inside operator().
using GroupedMatMulAllToAllVUdmaOfficialBlockConfig =
    GroupedMatMulAllToAllVUdmaConfig<
        bfloat16_t, Catlass::layout::RowMajor,
        bfloat16_t, Catlass::layout::RowMajor,
        bfloat16_t, Catlass::layout::RowMajor,
        128, 256, 256, false, true>;

using GroupedMatMulAllToAllVUdmaM0_256OfficialBlockConfig =
    GroupedMatMulAllToAllVUdmaConfig<
        bfloat16_t, Catlass::layout::RowMajor,
        bfloat16_t, Catlass::layout::RowMajor,
        bfloat16_t, Catlass::layout::RowMajor,
        256, 128, 256, false, true>;

} // namespace OpsTransformer::CatccosUdma

#endif // OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_UDMA_DEVICE_H
