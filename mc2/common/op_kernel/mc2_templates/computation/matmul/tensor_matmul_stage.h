/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * 基于参考 BlockMmad + BlockScheduler 的矩阵乘计算阶段
 * 替换 MC2FpMatmul，提供 Init / GetContextPtr / Process / End 接口
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

    static constexpr uint32_t BASE_M = 128;
    static constexpr uint32_t BASE_N = 128;
    static constexpr uint32_t BASE_K = 128;

    __aicore__ inline TensorMatmulStage(AscendC::TPipe *tPipe) : tPipePtr_(tPipe) {}

    __aicore__ inline void Init() {}

    __aicore__ inline MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::FpMMAdditionalData, TilingType> *GetContextPtr()
    {
        return &MMcontext_;
    }

    __aicore__ inline void Process(uint32_t taskIndex);

    __aicore__ inline void End() {}

private:
    MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::FpMMAdditionalData, TilingType> MMcontext_;
    AscendC::TPipe *tPipePtr_;
    BlockOp mmadOp_;
};


template <typename TilingType>
__aicore__ inline void TensorMatmulStage<TilingType>::Process(uint32_t taskIndex)
{
    if ASCEND_IS_AIV {
        return;
    }

    auto& base = MMcontext_.baseData;
    GM_ADDR aGM = base.aGM + taskIndex * base.aOffset;
    GM_ADDR bGM = base.bGM + taskIndex * base.bOffset;
    GM_ADDR cGM = base.cGM + taskIndex * base.cOffset;

    tPipePtr_->Reset();

    auto* tiling = MMcontext_.tilingDataPtr;
    uint32_t m_ = tiling->tileM;
    uint32_t k_ = tiling->fullK;
    uint32_t n_ = tiling->fullN;
    uint32_t baseM_ = tiling->baseM;
    uint32_t baseN_ = tiling->baseN;
    uint32_t baseK_ = tiling->baseK;
    uint32_t l1k   = tiling->kL1;

    typename BlockOp::L1Params l1Params{l1k};
    typename BlockOp::TupleShape problemShape{static_cast<int64_t>(m_), static_cast<int64_t>(n_), static_cast<int64_t>(k_)};
    typename BlockOp::BlockShape l0TileShape{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_), static_cast<int64_t>(baseK_), 0};
    mmadOp_.Init(problemShape, l0TileShape, l1Params, (m_ > baseM_) || (n_ > baseN_));

    auto layoutA = AscendC::Te::MakeNDLayout<AType>(m_, k_);
    auto layoutB = AscendC::Te::MakeNDLayout<BType>(k_, n_);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(m_, n_);
    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ AType*>(aGM)), layoutA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BType*>(bGM)), layoutB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(cGM)), layoutC);

    ProblemShape problem{m_, n_, k_, 1L};
    using SchParams = typename SchedulerOp::Params;
    SchParams schParams{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_), 0, 0, 0, 0, 0, 0};
    SchedulerOp bs(problem, schParams);

    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    BlockCoord blockIdx;
    constexpr int64_t kPos = 0L;
    while (bs.GetTileIdx(blockIdx)) {
        int64_t mPos = AscendC::Std::get<MNK_M>(blockIdx);
        int64_t nPos = AscendC::Std::get<MNK_N>(blockIdx);
        auto singleShape = bs.GetBlockShape(blockIdx);

        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            break;
        }

        auto gmBlockA = gmA(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), k_));
        auto gmBlockB = gmB(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(k_, Get<MNK_N>(singleShape)));
        auto gmBlockC = gmC(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));

        mmadOp_(gmBlockA, gmBlockB, gmBlockC, singleShape);
    }
}

} // namespace Mc2Kernel

#endif
