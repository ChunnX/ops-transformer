/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_MTE_DEVICE_H
#define OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_MTE_DEVICE_H

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
#include "catccos/comm/block/comm_block_mte.hpp"
#include "catccos/comm/block/comm_block_scheduler_gmm_alltoallv_target_order.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/block/block_scheduler_gmm_alltoallv_target_order.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/grouped_matmul_alltoallv_mte.hpp"

namespace OpsTransformer::CatccosMte {

template <class ElementA, class LayoutA, class ElementB, class LayoutB,
          class ElementC, class LayoutC, uint32_t M0, uint32_t N0, uint32_t K0>
struct GroupedMatMulAllToAllVMteConfig {
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
    static constexpr uint32_t kWorkspaceStages = 2;
    static constexpr uint64_t kSymmetricDataBytes = 200UL * 1024 * 1024 * sizeof(uint16_t);
    using BlockMmadScheduler =
        Catccos::DGemm::Block::BlockMmadSchedulerGmmAllToAllVTargetOrder<MoeConstraints>;
    using BlockCommScheduler =
        Catccos::CommEpilogue::Block::BlockCommSchedulerGmmAllToAllVTargetOrder<
            MoeConstraints, kIsDynamic>;
    using BlockComm = Catccos::Comm::Block::MteCommBlock<
        ElementC, Catccos::Comm::MteCopyDirection::Get>;

    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    using Kernel = Catccos::DGemm::Kernel::GroupedMatmulAllToAllVMte<
        ProblemShape, BlockMmad, BlockComm, BlockMmadScheduler,
        BlockCommScheduler, kWorkspaceStages, kSymmetricDataBytes>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

using GroupedMatMulAllToAllVMteDefaultConfig = GroupedMatMulAllToAllVMteConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    128, 256, 256>;

using GroupedMatMulAllToAllVMteM0_256Config = GroupedMatMulAllToAllVMteConfig<
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    bfloat16_t, Catlass::layout::RowMajor,
    256, 128, 256>;

} // namespace OpsTransformer::CatccosMte

#endif // OPS_TRANSFORMER_GROUPED_MAT_MUL_ALL_TO_ALLV_MTE_DEVICE_H
