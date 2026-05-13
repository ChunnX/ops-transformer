/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * 基于 matmul_story 参考 MX BlockMmad + MX BlockScheduler 的 MX-FP8 矩阵乘计算阶段
 * 替换 MC2MXQuantMatmul，提供 Init / GetContextPtr / Process / End 接口
 */

#ifndef TENSOR_MX_MATMUL_STAGE_H
#define TENSOR_MX_MATMUL_STAGE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul_base.h"
#include "mc2_mx_quant_matmul.h"
#include "../../../../../allto_all_matmul_apt/op_kernel/arch35/tensor_matmul_tiling_data.h"
#include "tensor.h"
#include "../../../../matmul_ref/mx_block_mmad_swat.h"
#include "../../../../matmul_ref/block/matmul_block_scheduler.h"
#include "../../../../matmul_ref/block/quant_matmul_mx_block_scheduler_swat.h"

namespace Mc2Kernel {

template <typename TilingType, bool X2TRANSPOSE>
class TensorMXQuantMatmulStage {
public:
    using AType = DTYPE_X1;
    using BType = DTYPE_X2;
    using ScaleAType = AscendC::fp8_e8m0_t;
    using ScaleBType = AscendC::fp8_e8m0_t;
    using CType = DTYPE_Y;

    using LayoutA = AscendC::Te::NDLayoutFormat<AType>;
    using LayoutSA = AscendC::Te::ScaleANDLayoutFormat<ScaleAType>;
    using LayoutB = AscendC::Te::NDLayoutFormat<BType>;
    using LayoutSB = AscendC::Te::ScaleBDNLayoutFormat<ScaleBType>;

    using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, 2>;
    using BlockOp = Block::BlockMmad<
        DispatchPolicy,
        AscendC::Std::tuple<AType, ScaleAType>, AscendC::Std::tuple<LayoutA, LayoutSA>,
        AscendC::Std::tuple<BType, ScaleBType>, AscendC::Std::tuple<LayoutB, LayoutSB>,
        CType, AscendC::Te::NDLayoutFormat<CType>>;
    using ProblemShape = MatmulShape;
    using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
    using SchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, false, false>::SchedulerOp;

    static constexpr uint32_t BASE_M = 128;
    static constexpr uint32_t BASE_N = 128;
    static constexpr uint32_t BASE_K = 128;

    __aicore__ inline TensorMXQuantMatmulStage(AscendC::TPipe *tPipe) : tPipePtr_(tPipe) {}
    __aicore__ inline void Init() {}
    __aicore__ inline MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::MXQuantMMAdditionalData, TilingType> *GetContextPtr() { return &MMcontext_; }
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void End() {}

private:
    MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::MXQuantMMAdditionalData, TilingType> MMcontext_;
    AscendC::TPipe *tPipePtr_;
};


template <typename TilingType, bool X2TRANSPOSE>
__aicore__ inline void TensorMXQuantMatmulStage<TilingType, X2TRANSPOSE>::Process(uint32_t taskIndex)
{
    if ASCEND_IS_AIV { return; }

    auto& base = MMcontext_.baseData;
    auto& add  = MMcontext_.additionalData;

    uint32_t rankDim = MMcontext_.rankDim;
    uint64_t globalStride = MMcontext_.globalStride;
    uint64_t tileStride = MMcontext_.tileStride;
    uint64_t scaleGlobalStride = MMcontext_.scaleGlobalStride;
    uint64_t scaleTileStride = MMcontext_.scaleTileStride;

    tPipePtr_->Reset();

    auto* tiling = MMcontext_.tilingDataPtr;
    uint32_t m_ = tiling->tileM;
    uint32_t k_ = tiling->fullK;
    uint32_t n_ = tiling->fullN;
    uint32_t baseM_ = tiling->baseM;
    uint32_t baseN_ = tiling->baseN;
    uint32_t baseK_ = tiling->baseK;
    uint32_t l1k    = tiling->kL1;
    uint32_t skl1   = tiling->scaleKL1;

    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint16_t SYNC_AIC_RANK_FLAG = 10;

    GM_ADDR cGM_base = base.cGM + taskIndex * base.cOffset;
    GM_ADDR bGM = base.bGM;
    GM_ADDR x2ScaleGM = add.x2Scale;

    for (uint32_t rankIdx = 0; rankIdx < rankDim; rankIdx++) {
        GM_ADDR aGM_rank = base.aGM + rankIdx * globalStride + taskIndex * tileStride;
        GM_ADDR x1ScaleGM_rank = add.x1Scale + rankIdx * scaleGlobalStride + taskIndex * scaleTileStride;

        if (rankIdx >= 1) {
            AscendC::SetAtomicAdd<CType>();
        }

        BlockOp mxMmadOp_;
        typename BlockOp::L1Params l1Params{l1k, skl1};
        typename BlockOp::TupleShape problemShape{static_cast<int64_t>(m_), static_cast<int64_t>(n_), static_cast<int64_t>(k_)};
        typename BlockOp::BlockShape l0TileShape{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_), static_cast<int64_t>(baseK_), 0};
        mxMmadOp_.Init(problemShape, l0TileShape, l1Params, tiling->dbL0c > 1);

        uint32_t kScaleGrp = (k_ + 32 - 1) / 32;
        auto layoutA = AscendC::Te::MakeNDLayout<AType>(m_, k_);
        auto layoutB = AscendC::Te::MakeNDLayout<BType>(n_, k_);
        auto layoutC = AscendC::Te::MakeNDLayout<CType>(m_, n_);
        auto layoutSA = AscendC::Te::MakeScaleANDLayout<ScaleAType>(m_, kScaleGrp);
        auto layoutSB = AscendC::Te::MakeScaleBDNLayout<ScaleBType>(kScaleGrp, n_);

        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ AType*>(aGM_rank)), layoutA);
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BType*>(bGM)), layoutB);
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(cGM_base)), layoutC);
        auto gmScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ ScaleAType*>(x1ScaleGM_rank)), layoutSA);
        auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ ScaleBType*>(x2ScaleGM)), layoutSB);

        ProblemShape problem{m_, n_, k_, 1L};
        using SchParams = typename SchedulerOp::Params;
        SchParams schParams{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_),
                            static_cast<int64_t>(tiling->mTailTile), static_cast<int64_t>(tiling->nTailTile),
                            static_cast<int64_t>(tiling->mBaseTailSplitCnt), static_cast<int64_t>(tiling->nBaseTailSplitCnt),
                            static_cast<int64_t>(tiling->mTailMain), static_cast<int64_t>(tiling->nTailMain)};
        SchedulerOp bs(problem, schParams);

        using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
        BlockCoord blockIdx;
        constexpr int64_t kPos = 0L;

        uint32_t loopCnt = 0;
        const uint32_t MAX_LOOP = (m_ / baseM_ + 1) * (n_ / baseN_ + 1) + 1;
        while (bs.GetTileIdx(blockIdx)) {
            if (++loopCnt > MAX_LOOP) {
                if (AscendC::GetBlockIdx() == 0) {
                }
                break;
            }
            int64_t mPos = AscendC::Std::get<MNK_M>(blockIdx);
            int64_t nPos = AscendC::Std::get<MNK_N>(blockIdx);
            auto singleShape = bs.GetBlockShape(blockIdx);

            if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
                break;
            }

            auto gmBlockA = gmA(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), k_));
            auto gmBlockB = gmB(AscendC::Te::MakeCoord(nPos, kPos), AscendC::Te::MakeShape(Get<MNK_N>(singleShape), k_));
            auto gmBlockC = gmC(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
            auto gmBlockScaleA = gmScaleA(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), kScaleGrp));
            auto gmBlockScaleB = gmScaleB(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(kScaleGrp, Get<MNK_N>(singleShape)));

            mxMmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
        }

        AscendC::CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_RANK_FLAG);
        AscendC::CrossCoreWaitFlag(SYNC_AIC_RANK_FLAG);
    }

    AscendC::SetAtomicNone();
}

} // namespace Mc2Kernel

#endif
