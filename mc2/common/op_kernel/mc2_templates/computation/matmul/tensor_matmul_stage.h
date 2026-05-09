/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * 基于参考 BlockMmad + BlockScheduler 的矩阵乘计算阶段
 */

#ifndef TENSOR_MATMUL_STAGE_H
#define TENSOR_MATMUL_STAGE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul_base.h"
#include "../../../../../allto_all_matmul_apt/op_kernel/arch35/tensor_matmul_tiling_data.h"
#include "tensor.h"
#include "../../../../matmul_ref/block/matmul_block_mmad.h"
#include "../../../../matmul_ref/block/matmul_block_scheduler.h"

namespace Mc2Kernel {

template <typename TilingType>
class TensorMatmulStage {
public:
    using AType = DTYPE_X1;
    using BType = DTYPE_X2;
    using CType = DTYPE_Y;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using DispatchPolicy = MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>;
    using BlockOp = Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC>;
    using ProblemShape = MatmulShape;
    using BlockScheduler = MatmulSwatScheduler<NO_FULL_LOAD_MODE>;
    using SchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, false, false>::SchedulerOp;

    __aicore__ inline TensorMatmulStage(AscendC::TPipe *tPipe) : tPipePtr_(tPipe) {}
    __aicore__ inline void Init() {}
    __aicore__ inline MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::FpMMAdditionalData, TilingType> *GetContextPtr() { return &MMcontext_; }
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void End() {}

private:
    MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::FpMMAdditionalData, TilingType> MMcontext_;
    AscendC::TPipe *tPipePtr_;
};

template <typename TilingType>
__aicore__ inline void TensorMatmulStage<TilingType>::Process(uint32_t taskIndex)
{
    if ASCEND_IS_AIV { return; }

    auto& base = MMcontext_.baseData;
    GM_ADDR aGM = base.aGM + taskIndex * base.aOffset;
    GM_ADDR bGM = base.bGM + taskIndex * base.bOffset;
    GM_ADDR cGM = base.cGM + taskIndex * base.cOffset;
    tPipePtr_->Reset();

    auto* tiling = MMcontext_.tilingDataPtr;
    uint32_t m_ = tiling->tileM, k_ = tiling->fullK, n_ = tiling->fullN;
    uint32_t baseM_ = tiling->baseM, baseN_ = tiling->baseN, baseK_ = tiling->baseK, l1k = tiling->kL1;

    BlockOp mmadOp_;
    mmadOp_.Init(typename BlockOp::TupleShape{static_cast<int64_t>(m_), static_cast<int64_t>(n_), static_cast<int64_t>(k_)},
                 typename BlockOp::BlockShape{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_), static_cast<int64_t>(baseK_), 0},
                 typename BlockOp::L1Params{l1k}, tiling->dbL0c > 1);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ AType*>(aGM)), AscendC::Te::MakeNDLayout<AType>(m_, k_));
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BType*>(bGM)), AscendC::Te::MakeDNLayout<BType>(k_, n_));
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(cGM)), AscendC::Te::MakeNDLayout<CType>(m_, n_));

    using SchParams = typename SchedulerOp::Params;
    SchedulerOp bs(ProblemShape{m_, n_, k_, 1L},
                   SchParams{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_),
                             static_cast<int64_t>(tiling->mTailTile), static_cast<int64_t>(tiling->nTailTile),
                             static_cast<int64_t>(tiling->mBaseTailSplitCnt), static_cast<int64_t>(tiling->nBaseTailSplitCnt),
                             static_cast<int64_t>(tiling->mTailMain), static_cast<int64_t>(tiling->nTailMain)});

    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    BlockCoord blockIdx;
    constexpr int64_t kPos = 0L;
    while (bs.GetTileIdx(blockIdx)) {
        int64_t mPos = AscendC::Std::get<MNK_M>(blockIdx), nPos = AscendC::Std::get<MNK_N>(blockIdx);
        auto ss = bs.GetBlockShape(blockIdx);
        if (Get<MNK_M>(ss) <= 0 || Get<MNK_N>(ss) <= 0) break;
        mmadOp_(gmA(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(ss), k_)),
                gmB(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(k_, Get<MNK_N>(ss))),
                gmC(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(Get<MNK_M>(ss), Get<MNK_N>(ss))),
                ss);
    }
}

} // namespace Mc2Kernel
#endif
