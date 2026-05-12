/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_mx_kernel_a_full_load.h
 * \brief Kernel-side SWAT MX implementation for the A-full-load path.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "kernel_utils/common_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "tensor.h"

#include "../block/quant_matmul_mx_block_mmad_a_full_load.h"
#include "../block/quant_matmul_mx_block_scheduler_a_full_load.h"
#include "../utils/quant_matmul_constant.h"
#include "adv_api/hccl/hccl.h"
#include "../../common/op_kernel/moe_distribute_base.h"

namespace Kernel {
// Keep the class template parameter list in one place so the declaration and
// out-of-line member definitions stay perfectly aligned.
#define QBMM_MX_KERNEL_A_FULL_LOAD_CLASS_TEM_PARAMS template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_A_FULL_LOAD_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;

// HCCL window layout constants: data zone layout inside the 0/1 ping-pong
// windows. These match the values used by the host-side HCCL configuration.
constexpr static uint64_t WIN_DATA_INNER_OFFSET = 30UL * 1024UL * 1024UL;
constexpr static uint64_t SINGLE_ZERONE_DATA_ZONE_SIZE = 100UL * 1024UL * 1024UL;
constexpr static uint64_t UB_ALIGN_BYTES = 32UL;

// Sub-tile M dimension for AIV UB processing. Each scheduler tile is split
// into sub-tiles of this many rows; V0 and V1 alternate sub-tile indices.
constexpr static int64_t AIV_UB_TILE_M = 32;

QBMM_MX_KERNEL_A_FULL_LOAD_CLASS_TEM_PARAMS
class QuantMatmulMxKernelAFullLoad {
public:
    __aicore__ inline QuantMatmulMxKernelAFullLoad()
    {}
    __aicore__ inline ~QuantMatmulMxKernelAFullLoad()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    // The kernel layer does not own a scheduling policy itself; it selects the
    // concrete scheduler from the block pipeline traits.
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, transA, transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA = AscendC::Te::NDLayoutFormat<AType>;
    using MakeLayoutB = AscendC::Te::DNLayoutFormat<BType>;
    using MakeLayoutScaleA = AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutScaleB = AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;

    struct QBMMTiling {
        // Base tile shape and L0C buffering mode selected by host tiling.
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint8_t dbL0C;
    };

    // Communication parameters for AIV alltoall path.
    struct CommParams {
        // Size of one TP domain = m / tpWorldSize.
        int64_t tpSizeM;
        // Number of TP ranks participating in the alltoall.
        int32_t tpWorldSize;
    };

    struct Params {
        // Translate the serialized tiling payload once at the launcher
        // boundary, then keep the kernel, scheduler, and block MMAD layers on
        // their own typed parameter bundles.
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        CommParams commParams;
    };

public:
    // Launch one SWAT kernel instance on the current AIC block.
    __aicore__ inline void operator()(const Params& params, TPipe* pipe = nullptr);

private:
    // Bind raw GM addresses to typed kernel pointers once per launch.
    __aicore__ inline void ResetGmAddr(const Params& params);

    // Wrap the full GM tensors and repeatedly slice them into per-block views
    // according to the scheduler output.
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);

    // `ProblemShape` stays generic at the template boundary, but the block
    // MMAD operator consumes a fixed 3D tuple internally.
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

    // AIV: read bf16 workspace and write to remote HCCL windows (alltoall).
    __aicore__ inline void CommProcess(
        int64_t mPos, int64_t nPos, const BlockShape& singleShape, const Params& params);

    // HCCL window helpers.
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId);

private:
    BlockMmad mmadOp_;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ fp8_e8m0_t* scaleAGmAddr_;
    __gm__ fp8_e8m0_t* scaleBGmAddr_;
    __gm__ HcclOpResParam *winContext_{nullptr};
    TPipe *tpipe_{nullptr};
    uint32_t rankId_{0};
    uint32_t coreVid_{0};
    // VECIN queue for DataCopyPad GM→VECIN→GM (handles stride and non-aligned tail)
    TQue<QuePosition::VECIN, 1> inQueue_;
};

QBMM_MX_KERNEL_A_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelAFullLoad<QBMM_MX_KERNEL_A_FULL_LOAD_FUN_TEM_PARAMS>::operator()(
    const Params& params, TPipe* pipe)
{
    if ASCEND_IS_AIC {
        ResetGmAddr(params);
        BlockSchedulerOp bs(params.problemShape, params.schParams);
        TupleShape problemShape_ = ToShapeTuple(params.problemShape);
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
        bool enableL0cPingPong = (params.qbmmParams.dbL0C > 1);
        mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, enableL0cPingPong);
        Process(params, bs);
        return;
    }

    if ASCEND_IS_AIV {
        tpipe_ = pipe;
        if (tpipe_ != nullptr) {
            uint32_t paddedRowBytes = ::CeilDiv(
                static_cast<uint64_t>(params.qbmmParams.baseN) * sizeof(bfloat16_t),
                UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
            tpipe_->InitBuffer(inQueue_, 1, AIV_UB_TILE_M * paddedRowBytes);
        }
        winContext_ = (__gm__ HcclOpResParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        rankId_ = winContext_->localUsrRankId;
        coreVid_ = AscendC::GetBlockIdx();
        // V核使用与AIC相同的调度器,blockIdx_在MIX_AIC_1_2模式下
        // 通过GetTaskRation归一化,使得同组AIC+AIV的blockIdx_相同,
        // 从而产出相同的(mPos,nPos)序列。
        BlockSchedulerOp bs(params.problemShape, params.schParams);
        BlockCoord blockIdx;
        while (bs.GetTileIdx(blockIdx)) {
            int64_t mPos = Get<MNK_M>(blockIdx);
            int64_t nPos = Get<MNK_N>(blockIdx);
            BlockShape singleShape = bs.GetBlockShape(blockIdx);
            if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
                return;
            }
            CrossCoreWaitFlag(9);
            CommProcess(mPos, nPos, singleShape, params);
        }
        return;
    }
}

QBMM_MX_KERNEL_A_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelAFullLoad<QBMM_MX_KERNEL_A_FULL_LOAD_FUN_TEM_PARAMS>::ResetGmAddr(
    const Params& params)
{
    // The sample launcher passes raw GM addresses through `Params`; convert
    // them here once so the hot loop works with typed pointers only.
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
}

QBMM_MX_KERNEL_A_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelAFullLoad<QBMM_MX_KERNEL_A_FULL_LOAD_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
    // Build full-GM tensor views once, then slice them per scheduled block.
    int64_t kScaleSize =
        ::CeilDiv(static_cast<int64_t>(params.problemShape.k), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
    auto layoutScaleA = MakeLayoutScaleA{}(params.problemShape.m, kScaleSize);
    auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
    auto layoutScaleB = MakeLayoutScaleB{}(kScaleSize, params.problemShape.n);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(params.problemShape.m, params.problemShape.n);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(aGmAddr_), layoutA);
    auto gmScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleAGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(bGmAddr_), layoutB);
    auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleBGmAddr_), layoutScaleB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    constexpr int64_t kPos = 0L;

    bool isBMatrixOnlyNeedLoadOnce = params.problemShape.m <= params.qbmmParams.baseM && (params.problemShape.k & 0xff) == 0;
    if (isBMatrixOnlyNeedLoadOnce) {
        gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    }
    while (bs.GetTileIdx(blockIdx)) {
        // The scheduler packs GM origin into M/N and retains logical tile
        // indices in K/B so shape reconstruction still works.
        int64_t mPos = Get<MNK_M>(blockIdx);
        int64_t nPos = Get<MNK_N>(blockIdx);
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            // Tail splitting can create empty logical slices; ignore them and
            // stop the current block once no useful work remains.
            return;
        }

        // `blockIdx` now carries both GM origin and logical tile metadata.
        auto gmBlockA =
            gmA(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), params.problemShape.k));
        auto gmBlockScaleA =
            gmScaleA(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), kScaleSize));
        auto gmBlockB =
            gmB(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(params.problemShape.k, Get<MNK_N>(singleShape)));
        auto gmBlockScaleB =
            gmScaleB(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(kScaleSize, Get<MNK_N>(singleShape)));
        auto gmBlockC =
            gmC(AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));

        // The block MMAD layer owns all data movement below GM granularity and
        // performs the actual accumulation for this scheduled tile.
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
        CrossCoreSetFlag<0x2, PIPE_FIX>(9);
    }
}

QBMM_MX_KERNEL_A_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline GM_ADDR QuantMatmulMxKernelAFullLoad<QBMM_MX_KERNEL_A_FULL_LOAD_FUN_TEM_PARAMS>::GetWindAddrByRankId(const int32_t rankId)
{
    if (rankId == rankId_) {
        return (GM_ADDR)(winContext_->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn);   // 先找到某个rank的首地址，然后再偏移到具体的处理data的地方
}

QBMM_MX_KERNEL_A_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void
QuantMatmulMxKernelAFullLoad<QBMM_MX_KERNEL_A_FULL_LOAD_FUN_TEM_PARAMS>::CommProcess(
    int64_t mPos, int64_t nPos, const BlockShape& singleShape, const Params& params)
{
    int64_t singleM = Get<MNK_M>(singleShape);
    int64_t singleN = Get<MNK_N>(singleShape);
    if (singleM <= 0 || singleN <= 0) {
        return;
    }

    int64_t m = params.problemShape.m;
    int64_t n = params.problemShape.n;
    int64_t tpSizeM = params.commParams.tpSizeM;
    int32_t tpWorldSize = params.commParams.tpWorldSize;

    // AIV pair index within a MIX_AIC_1_2 group: coreVid_=1,4,7... → 1 (odd),
    // coreVid_=2,5,8... → 0 (even). Either way each sub-tile is covered
    // exactly once by the alternating assignment below.
    int64_t aivIdx = coreVid_ % 2;

    // Split the scheduler tile into sub-tiles of AIV_UB_TILE_M rows in M
    // direction. The two V cores alternate assignment by sub-tile index:
    //   V0 (aivIdx=0): sub-tiles 0, 2, 4, ...
    //   V1 (aivIdx=1): sub-tiles 1, 3, 5, ...
    // This also handles tail tiles (singleM not multiple of AIV_UB_TILE_M).
    int64_t subTileCnt = ::CeilDiv(singleM, AIV_UB_TILE_M);

    __gm__ CType* workspaceAddr =
        reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    GlobalTensor<bfloat16_t> workspaceGm;
    workspaceGm.SetGlobalBuffer(workspaceAddr);

    // DataCopyPad uses byte-level blockLen/stride, handling both aligned and
    // non-aligned tail tiles seamlessly (32B-padding in VECIN auto-discarded).
    uint32_t blkLenBytes = static_cast<uint32_t>(singleN * sizeof(bfloat16_t));
    uint32_t rowPaddedBytes = ::CeilDiv(blkLenBytes, UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    uint32_t rowInVECINElems = rowPaddedBytes / sizeof(bfloat16_t);

    for (int64_t st = aivIdx; st < subTileCnt; st += 2) {
        int64_t subTileM = Min(AIV_UB_TILE_M, singleM - st * AIV_UB_TILE_M);
        if (subTileM <= 0) {
            continue;
        }
        int64_t globalBaseM = mPos + st * AIV_UB_TILE_M;

        // 1) 整块读子 tile 到 VECIN（一次性 GM→VECIN）
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        LocalTensor<bfloat16_t> batchLocal = inQueue_.AllocTensor<bfloat16_t>();

        DataCopyExtParams readParams{
            static_cast<uint16_t>(subTileM),
            blkLenBytes,
            static_cast<uint32_t>((n - singleN) * sizeof(bfloat16_t)),
            0,
            0
        };
        DataCopyPad(batchLocal, workspaceGm[globalBaseM * n + nPos], readParams, padParams);
        inQueue_.EnQue(batchLocal);
        batchLocal = inQueue_.DeQue<bfloat16_t>();

        // 2) 按 TP domain 分批次写远端窗口（每批一次 VECIN→GM）
        int64_t processed = 0;
        while (processed < subTileM) {
            int64_t globalRow = globalBaseM + processed;
            int32_t dstRankId = static_cast<int32_t>(globalRow / tpSizeM);
            if (dstRankId >= tpWorldSize) {
                break;
            }
            int64_t tpBoundary = (dstRankId + 1) * tpSizeM;
            int64_t rowsInBatch = Min(subTileM - processed, tpBoundary - globalRow);
            if (rowsInBatch <= 0) {
                break;
            }

            GM_ADDR remoteWinAddr = GetWindAddrByRankId(dstRankId);
            GlobalTensor<bfloat16_t> remoteGm;
            remoteGm.SetGlobalBuffer((__gm__ bfloat16_t*)remoteWinAddr);
            uint64_t dstBaseOffset = static_cast<uint64_t>(rankId_) * tpSizeM * n +
                                     (globalRow % tpSizeM) * n + nPos;

            DataCopyExtParams writeParams{
                static_cast<uint16_t>(rowsInBatch),
                blkLenBytes,
                0,
                static_cast<uint32_t>((n - singleN) * sizeof(bfloat16_t)),
                0
            };
            DataCopyPad(remoteGm[dstBaseOffset], batchLocal[processed * rowInVECINElems], writeParams);

            processed += rowsInBatch;
        }

        inQueue_.FreeTensor(batchLocal);
    }
}

} // namespace Kernel

#undef QBMM_MX_KERNEL_A_FULL_LOAD_CLASS_TEM_PARAMS
#undef QBMM_MX_KERNEL_A_FULL_LOAD_FUN_TEM_PARAMS
