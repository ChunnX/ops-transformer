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
 * \file kernel_gmm_swiglu_hif4quant.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_GMM_SWIGLU_HIF4QUANT_H
#define MATMUL_KERNEL_KERNEL_GMM_SWIGLU_HIF4QUANT_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "./semaphore.h"

#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "../epilogue/block_epilogue_swiglu_hif4_quant.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"

namespace Cgmct {
namespace Gemm {
namespace Kernel {

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;
constexpr uint8_t  SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint64_t SWIGLU_N_HALF = 2L;
}  // namespace

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_,
    typename Enable_ = void>
class KernelGmmSwiGluHif4Quant {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
        "KernelGmmSwiGluHif4Quant is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class KernelGmmSwiGluHif4Quant<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>> {
public:
    __aicore__ inline KernelGmmSwiGluHif4Quant()
    {}

    __aicore__ inline ~KernelGmmSwiGluHif4Quant()
    {}

    using BlockEpilogue = BlockEpilogue_;
    using BlockMmadOp = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    static constexpr bool transA = BlockMmadOp::transA;
    static constexpr bool transB = BlockMmadOp::transB;
    static constexpr auto formatA = BlockMmadOp::formatA;
    static constexpr auto formatB = BlockMmadOp::formatB;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadOp::L1TileShape,
            typename BlockMmadOp::L0TileShape, BlockScheduler, transA, transB>::SchedulerOp;
    // mmadOp
    using BlockMmadArguments = typename BlockMmadOp::Params;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadOp::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using DataTypeOut = typename BlockEpilogue::DataTypeOut;
    using AType = typename BlockMmadOp::AType;
    using BType = typename BlockMmadOp::BType;
    using CType = typename BlockMmadOp::CType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using MmShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using TupleShapeTemp = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y, yscale
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass = Coordinate<transA, transB, formatA, formatB, BlockMmadOp::formatC>;

    // attribute
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<AscendC::hif4_scale> x1ScaleGlobal_;
    AscendC::GlobalTensor<AscendC::hif4_scale> x2ScaleGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // shape
    TupleShapeTemp problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0, 0};
    uint64_t preOffset_ = 0;
    int64_t perGroupBOffset_ = 0;
    int64_t bRightOffsetStep_ = 0;
    int64_t x2ScaleRightOffsetStep_ = 0;
    static constexpr int32_t c0Size = MATMUL_MNK_ALIGN_INT4;
    BlockMmadOp mmadOp_;
    BlockEpilogue epilogueOp_;
    AscendC::LocalTensor<CType> l0cOutUbFirst_;
    AscendC::LocalTensor<CType> l0cOutUbSecond_;
    bool isVecSetSyncCom_ = false;
    bool isBias_{false};

    struct GMMTiling {
        uint32_t groupNum;
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t scaleKAL1;
        uint32_t scaleKBL1;
        uint8_t isBias;
        uint8_t dbL0C;
        uint8_t groupListType;
        __aicore__ GMMTiling()
        {}
        __aicore__ GMMTiling(uint32_t groupNum_, uint32_t m_, uint32_t n_, uint32_t k_, uint32_t baseM_,
                             uint32_t baseN_, uint32_t baseK_, uint32_t kAL1_, uint32_t kBL1_, uint32_t scaleKAL1_,
                             uint32_t scaleKBL1_, uint8_t isBias_, uint8_t dbL0C_, uint8_t groupListType_)
            : groupNum(groupNum_), m(m_), n(n_), k(k_), baseM(baseM_), baseN(baseN_), baseK(baseK_), kAL1(kAL1_),
              kBL1(kBL1_), scaleKAL1(scaleKAL1_), scaleKBL1(scaleKBL1_), isBias(isBias_), dbL0C(dbL0C_),
              groupListType(groupListType_)
        {
        }
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        GMMTiling gmmArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GMMTiling gmmParams;
        Params() = default;
    };

    __aicore__ inline void NotifyCube()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
    }
    __aicore__ inline void WaitForVector()
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
    }
    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
    }
    __aicore__ inline void WaitForCube()
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForVector();
            }
        }
    }

    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
    {
        int32_t splitValue = 0;
        if (groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline void UpdateGlobalBuffer(const Params &params)
    {
        if ASCEND_IS_AIC {
            aGlobal_.SetGlobalBuffer((__gm__ AType *)params.mmadParams.aGmAddr + Get<IDX_A_OFFSET>(baseOffset_));
            bGlobal_.SetGlobalBuffer(
                GetTensorAddr<BType>(0, params.mmadParams.bGmAddr) + Get<IDX_B_OFFSET>(baseOffset_));
            x1ScaleGlobal_.SetGlobalBuffer(
                (__gm__ AscendC::hif4_scale *)params.mmadParams.x1ScaleGmAddr + Get<IDX_X1SCALE_OFFSET>(baseOffset_));
            x2ScaleGlobal_.SetGlobalBuffer(GetTensorAddr<AscendC::hif4_scale>(0, params.mmadParams.x2ScaleGmAddr) +
                                           Get<IDX_X2SCALE_OFFSET>(baseOffset_));
        }
        if ASCEND_IS_AIV {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        // baseOffset is 0 when groupIdx = 0
        if (groupIdx == 0) {
            return;
        }
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);

        Get<IDX_A_OFFSET>(baseOffset_) = ((preOffset_ - m) * k) >> 1;
        Get<IDX_B_OFFSET>(baseOffset_) = (perGroupBOffset_ * static_cast<int64_t>(groupIdx)) >> 1;

        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE);
        // scaleAAxisBaseOffset (m, ceil(k,64), 1)
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) = (preOffset_ - m) * scaleK;
        // scaleBAxisBaseOffset (g, n, ceil(k,64), 1) or (g, ceil(k,64), n, 1)
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) = static_cast<int64_t>(groupIdx) * n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) = ((preOffset_ - m) * (n / SWIGLU_N_HALF)) >> 1;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) = (preOffset_ - m) * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE);
    }

    __aicore__ inline bool UpdateGroupParams(const Params &params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<M_VALUE>(problemShape_) = splitValue;
        // split_m，when m=0, skip
        if (Get<M_VALUE>(problemShape_) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void InitParamsAndTensor(const Params &params)
    {
        Get<N_VALUE>(problemShape_) = params.gmmParams.n;
        Get<K_VALUE>(problemShape_) = params.gmmParams.k;
        perGroupBOffset_ = Get<N_VALUE>(problemShape_) * Get<K_VALUE>(problemShape_);
        groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params.mmadParams.groupListGmAddr));
        int64_t resN = Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF;  // SwiGLU: N -> N/2
        if constexpr (transB) {
            bRightOffsetStep_ = resN * Get<K_VALUE>(problemShape_);
        } else {
            bRightOffsetStep_ = resN;
        }

        if constexpr (transB) {
            x2ScaleRightOffsetStep_ = resN * CeilDiv(Get<K_VALUE>(problemShape_), MXFP_DIVISOR_SIZE);
        } else {
            x2ScaleRightOffsetStep_ = resN;
        }
    }

    __aicore__ inline void SetL2CacheDisableIfNeeded(int64_t mSize, int64_t curBaseM, int64_t baseN)
    {
        if constexpr (formatB != CubeFormat::ND) {
            if (curBaseM >= mSize) {
                bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                x2ScaleGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            } else {
                bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
                x2ScaleGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
            }
        } else {
            if constexpr (transB) {
                if (curBaseM >= mSize && (Get<K_VALUE>(problemShape_) & 0xff) == 0) {
                    bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    x2ScaleGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
                    x2ScaleGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
                }
            } else {
                if (curBaseM >= mSize && (Get<N_VALUE>(problemShape_) & 0xff) == 0 && (baseN & 0xff) == 0) {
                    bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    x2ScaleGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
                    x2ScaleGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
                }
            }
        }
    }

    __aicore__ inline void ProcessSingleGroup(const Params &params, BlockSchedulerOp &bs, uint32_t groupIdx)
    {
        int64_t m = Get<M_VALUE>(problemShape_);
        int64_t n = Get<N_VALUE>(problemShape_);
        int64_t k = Get<K_VALUE>(problemShape_);
        CoordClass coord(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockCoord tileIdx;
        if (!bs.GetTileIdx(tileIdx)) {
            return;
        }
        UpdateOffset(groupIdx);
        UpdateGlobalBuffer(params);
        SetL2CacheDisableIfNeeded(Get<M_VALUE>(problemShape_),
            static_cast<int64_t>(params.gmmParams.baseM),
            static_cast<int64_t>(params.gmmParams.baseN));
        do {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
                return;
            }
            blockOffset_ = coord.template GetQuantIOOffset<GroupedMatmul::QuantMode::HIF4_PEGROUP_MODE>(
                Get<IDX_M_TILEIDX>(tileIdx),
                Get<IDX_N_TILEIDX>(tileIdx),
                Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
                Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape),
                c0Size);
            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForVector();
                }
                AscendC::Std::tuple<int64_t, int64_t, int64_t> mmSingleShape{
                    Get<M_VALUE>(singleShape), Get<N_VALUE>(singleShape), k};
                // left block
                mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)],
                    bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
                    x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
                    x2ScaleGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)],
                    l0cOutUbFirst_,
                    mmSingleShape);
                // right block
                int64_t bRightOffset = Get<IDX_B_OFFSET>(blockOffset_) + bRightOffsetStep_;
                int64_t x2ScaleRightOffset = Get<IDX_X2SCALE_OFFSET>(blockOffset_) + x2ScaleRightOffsetStep_;
                mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)],
                    bGlobal_[bRightOffset],
                    x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
                    x2ScaleGlobal_[x2ScaleRightOffset],
                    l0cOutUbSecond_,
                    mmSingleShape);
                NotifyVector();
            }
            isVecSetSyncCom_ = true;
            if ASCEND_IS_AIV {
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
                    Get<M_VALUE>(singleShape), Get<N_VALUE>(singleShape), 0, 0};
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    Get<IDX_C_OFFSET>(blockOffset_),
                    Get<IDX_C_SCALE_OFFSET>(blockOffset_),
                    Get<IDX_X2SCALE_OFFSET>(blockOffset_),
                    Get<IDX_X1SCALE_OFFSET>(blockOffset_),
                    Get<IDX_BIAS_OFFSET>(blockOffset_),
                };
                WaitForCube();
                epilogueOp_(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        } while (bs.GetTileIdx(tileIdx));
    }

    __aicore__ inline void operator()(const Params &params)
    {
        // Init mmadOp epilogue
        const auto &gmmParams = params.gmmParams;
        isBias_ = gmmParams.isBias != 0;
        TupleShapeTemp l0Shape{static_cast<int64_t>(gmmParams.baseM),
            static_cast<int64_t>(gmmParams.baseN),
            static_cast<int64_t>(gmmParams.baseK)};
        typename BlockMmadOp::L1Params l1Params{static_cast<uint64_t>(gmmParams.kAL1),
            static_cast<uint64_t>(gmmParams.kBL1),
            static_cast<uint64_t>(gmmParams.scaleKAL1),
            DOUBLE_BUFFER_COUNT};
        mmadOp_.Init(problemShape_, l0Shape, l1Params, isBias_, gmmParams.dbL0C == DOUBLE_BUFFER_COUNT);
        InitParamsAndTensor(params);
        epilogueOp_.Init(params.epilogueParams);
        uint32_t groupNum = params.gmmParams.groupNum;
        BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        if constexpr (!transA) {
            if constexpr (transB) {
                bs.SetTailAlign(1, AscendC::BLOCK_CUBE);
            } else {
                bs.SetTailAlign(1, c0Size);
            }
        }
        l0cOutUbFirst_ = epilogueOp_.GetFirstL0c2UbTensor();
        l0cOutUbSecond_ = epilogueOp_.GetSecondL0c2UbTensor();
        for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            TupleShape resProblemShape{
                Get<M_VALUE>(problemShape_), Get<N_VALUE>(problemShape_) >> 1, Get<K_VALUE>(problemShape_), 0};
            bs.UpdateNextProblem(resProblemShape);
            epilogueOp_.UpdateNextProblem(resProblemShape);
            if ASCEND_IS_AIC {
                MmShape mmProblemShape{
                    Get<M_VALUE>(problemShape_), Get<N_VALUE>(problemShape_) >> 1, Get<K_VALUE>(problemShape_)};
                mmadOp_.UpdateParamsForNextProblem(mmProblemShape);
            }
            ProcessSingleGroup(params, bs, groupIdx);
        }
        End();
    }
};
}  // namespace Kernel
}  // namespace Gemm
}  // namespace Cgmct
#endif  // MATMUL_KERNEL_KERNEL_GMM_SWIGLU_HIF4QUANT_H