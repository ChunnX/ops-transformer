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

    __aicore__ inline TensorMXQuantMatmulStage() {}
    __aicore__ inline void Init() {}
    __aicore__ inline MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::MXQuantMMAdditionalData, TilingType> *GetContextPtr() { return &MMcontext_; }
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void End() {}

private:
    MC2KernelTemplate::MC2MMContext<
        MC2KernelTemplate::MXQuantMMAdditionalData, TilingType> MMcontext_;
};


template <typename TilingType, bool X2TRANSPOSE>
__aicore__ inline void TensorMXQuantMatmulStage<TilingType, X2TRANSPOSE>::Process(uint32_t taskIndex)
{
    if ASCEND_IS_AIV { return; }

    auto* tiling = MMcontext_.tilingDataPtr;
    uint32_t m_ = tiling->tileM;
    uint32_t k_ = tiling->fullK;
    uint32_t n_ = tiling->fullN;
    uint32_t baseM_ = tiling->baseM;
    uint32_t baseN_ = tiling->baseN;
    uint32_t baseK_ = tiling->baseK;
    uint32_t l1k    = tiling->kL1;
    uint32_t skl1   = tiling->scaleKL1;

    uint32_t rankDim = tiling->rankDim;
    uint32_t rankK_ = tiling->rankK;
    uint32_t scaleKPerRank = tiling->rankKScaleGrp;
    uint32_t kScaleGrp = scaleKPerRank * rankDim;

    auto& base = MMcontext_.baseData;
    auto& add  = MMcontext_.additionalData;

    uint64_t globalStride = MMcontext_.globalStride;
    uint64_t tileStride = MMcontext_.tileStride;
    uint64_t scaleGlobalStride = MMcontext_.scaleGlobalStride;
    uint64_t scaleTileStride = MMcontext_.scaleTileStride;

    GM_ADDR cGM_base = base.cGM + taskIndex * base.cOffset;
    GM_ADDR bGM = base.bGM;
    GM_ADDR x2ScaleGM = add.x2Scale;

    constexpr uint32_t MAX_RANK = 8;
    GM_ADDR aGmBases[MAX_RANK];
    GM_ADDR scaleAGmBases[MAX_RANK];
    for (uint32_t r = 0; r < rankDim; r++) {
        aGmBases[r] = base.aGM + r * globalStride + taskIndex * tileStride;
        scaleAGmBases[r] = add.x1Scale + r * scaleGlobalStride + taskIndex * scaleTileStride;
    }

    auto layoutB = AscendC::Te::MakeDNLayout<BType>(k_, n_);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(m_, n_);
    auto layoutSB = AscendC::Te::MakeScaleBDNLayout<ScaleBType>(kScaleGrp, n_);

    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BType*>(bGM)), layoutB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(cGM_base)), layoutC);
    auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ ScaleBType*>(x2ScaleGM)), layoutSB);

    BlockOp mxMmadOp_;
    typename BlockOp::L1Params l1Params{l1k, skl1};
    typename BlockOp::TupleShape problemShape{static_cast<int64_t>(m_), static_cast<int64_t>(n_), static_cast<int64_t>(k_)};
    typename BlockOp::BlockShape l0TileShape{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_), static_cast<int64_t>(baseK_), 0};
    mxMmadOp_.Init(problemShape, l0TileShape, l1Params, tiling->dbL0c > 1);

    ProblemShape problem{m_, n_, k_, 1L};
    using SchParams = typename SchedulerOp::Params;
    SchParams schParams{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_),
                        1L, 1L, 1L, 1L, 0L, 0L};
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

        GM_ADDR aGmAddrs[MAX_RANK];
        GM_ADDR scaleAGmAddrs[MAX_RANK];
        uint64_t mPosU64 = static_cast<uint64_t>(mPos);
        uint64_t mStrideA = static_cast<uint64_t>(rankK_) * sizeof(AType);
        uint64_t mStrideScaleA = static_cast<uint64_t>(scaleKPerRank) * sizeof(ScaleAType);
        for (uint32_t r = 0; r < rankDim; r++) {
            aGmAddrs[r] = aGmBases[r] + mPosU64 * mStrideA;
            scaleAGmAddrs[r] = scaleAGmBases[r] + mPosU64 * mStrideScaleA;
        }
        mxMmadOp_.SetMultiRankInfo(rankDim, rankK_, aGmAddrs, scaleAGmAddrs);

        auto layoutA_ph = AscendC::Te::MakeNDLayout<AType>(Get<MNK_M>(singleShape), rankK_);
        auto gmBlockA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ AType*>(aGmAddrs[0])), layoutA_ph);
        auto gmBlockB = gmB(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(k_, Get<MNK_N>(singleShape)));
        auto gmBlockC = gmC(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));

        auto layoutSA_ph = AscendC::Te::MakeScaleANDLayout<ScaleAType>(Get<MNK_M>(singleShape), scaleKPerRank);
        auto gmBlockScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ ScaleAType*>(scaleAGmAddrs[0])), layoutSA_ph);
        auto gmBlockScaleB = gmScaleB(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(kScaleGrp, Get<MNK_N>(singleShape)));

        mxMmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
    }
}

} // namespace Mc2Kernel

#endif
