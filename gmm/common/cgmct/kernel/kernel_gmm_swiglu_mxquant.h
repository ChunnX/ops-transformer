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
 * \file kernel_gmm_swiglu_mxquant.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_GMM_SWIGLU_MXQUANT_H
#define MATMUL_KERNEL_KERNEL_GMM_SWIGLU_MXQUANT_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "./semaphore.h"

#include "../block/block_mx_mm_aic_to_aiv_builder.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "../epilogue/block_epilogue_swiglu_mx_quant.h"

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
constexpr uint64_t GROUP_LIST_TYPE_SPARSE = 2UL;
constexpr uint64_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2UL;
constexpr uint64_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1UL;
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
constexpr int64_t MATRIX_INNER_DIM_LIMIT_SIZE_V35 = 2097151L; // 21bits
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint64_t SWIGLU_N_HALF = 2;
} // namespace

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelGmmSwiGluMixOnlineDynamic {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGmmSwiGluMixOnlineDynamic is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelGmmSwiGluMixOnlineDynamic<
    ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>> {
public:
    __aicore__ inline KernelGmmSwiGluMixOnlineDynamic() {}

    __aicore__ inline ~KernelGmmSwiGluMixOnlineDynamic() {}

    using BlockEpilogue = BlockEpilogue_;
    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    static constexpr auto formatA = BlockMmadBuilder::formatA;
    static constexpr auto formatB = BlockMmadBuilder::formatB;
    static constexpr int64_t l1M = BlockMmadBuilder::l1M;
    static constexpr int64_t l1N = BlockMmadBuilder::l1N;
    static constexpr int64_t l1K = BlockMmadBuilder::l1K;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
                                               typename BlockMmadBuilder::L0TileShape, BlockScheduler, transA,
                                               transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using DataTypeOut = typename BlockEpilogue::DataTypeOut;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    static constexpr bool IS_FP4 =
        AscendC::IsSameTypeV<AType, fp4x2_e2m1_t> || AscendC::IsSameTypeV<AType, fp4x2_e1m2_t>;
    static constexpr int32_t c0Size = IS_FP4 ? MATMUL_MNK_ALIGN_INT4 : MATMUL_MNK_ALIGN_INT8;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y, yscale
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass =
        Coordinate<transA, transB, formatA, formatB, BlockMmadBuilder::formatC>;

    // attribute
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x1ScaleGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x2ScaleGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // shape
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0, 0};
    int64_t preOffset_ = 0;
    int32_t curBaseM_ = 0;
    BlockMmadOp mmadOp_;
    BlockEpilogue epilogueOp_;
    AscendC::LocalTensor<CType> l0cOutUbFirst_;
    AscendC::LocalTensor<CType> l0cOutUbSecond_;
    bool isVecSetSyncCom_ = false;

    struct GMMTiling {
        uint32_t groupNum;
        uint8_t groupListType;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        const TCubeTiling* __restrict matmulTiling;
        __aicore__ GMMTiling() {}
        __aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_, int32_t baseN_,
                             int32_t baseK_) :
            groupNum(groupNum_), groupListType(groupListType_), baseM(baseM_), baseN(baseN_), baseK(baseK_)
        {}
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
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
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

    __aicore__ inline void BaseMBalance(BlockSchedulerOp& bs, int64_t m, int64_t baseM)
    {
        if constexpr (!transA) {
            int64_t mCnt = CeilDiv(m, baseM);
            curBaseM_ = Align16(CeilDiv(m, mCnt));
            bs.UpdateBaseM(curBaseM_);
        }
    }

    __aicore__ inline bool IfNeedSplit(const BlockSchedulerOp& bs)
    {
        return (bs.GetEndBlockIdx() + 1) <= AscendC::GetBlockNum() / SWIGLU_N_HALF;
    }

    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
    {
        int32_t splitValue = 0;
        if (groupListType == 0) {
            int64_t offset = groupListGm_.GetValue(groupIdx);
            splitValue = static_cast<int32_t>(offset - preOffset_);
            preOffset_ = offset;
        } else if (groupListType == 1) {
            splitValue = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
            preOffset_ += splitValue;
        } else {
            splitValue = static_cast<int32_t>(
                groupListGm_.GetValue(groupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE + SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET));
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline void UpdateGlobalBuffer(const Params& params)
    {
        if ASCEND_IS_AIC {
            aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr + Get<IDX_A_OFFSET>(baseOffset_));
            bGlobal_.SetGlobalBuffer(GetTensorAddr<BType>(0, params.mmadParams.bGmAddr) + Get<IDX_B_OFFSET>(baseOffset_));
            x1ScaleGlobal_.SetGlobalBuffer((__gm__ AscendC::fp8_e8m0_t*)params.mmadParams.x1ScaleGmAddr + Get<IDX_X1SCALE_OFFSET>(baseOffset_));
            x2ScaleGlobal_.SetGlobalBuffer(GetTensorAddr<AscendC::fp8_e8m0_t>(0, params.mmadParams.x2ScaleGmAddr) + Get<IDX_X2SCALE_OFFSET>(baseOffset_));
        }
        if ASCEND_IS_AIV {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        int64_t m = Get<M_VALUE>(problemShape_);
        int64_t n = Get<N_VALUE>(problemShape_);
        int64_t k = Get<K_VALUE>(problemShape_);
        int64_t mOffset = preOffset_ - m;

        if constexpr (IS_FP4) {
            Get<IDX_A_OFFSET>(baseOffset_) = (mOffset * k) >> 1;
        } else {
            Get<IDX_A_OFFSET>(baseOffset_) = mOffset * k;
        }

        if constexpr (formatB == CubeFormat::NZ) {
            if constexpr (transB) {
                Get<IDX_B_OFFSET>(baseOffset_) =
                    Align16(n) * (IS_FP4 ? Align64(k) : Align32(k)) * static_cast<int64_t>(groupIdx);
            } else {
                Get<IDX_B_OFFSET>(baseOffset_) =
                    (IS_FP4 ? Align64(n) : Align32(n)) * Align16(k) * static_cast<int64_t>(groupIdx);
            }
        } else {
            if constexpr (IS_FP4) {
                Get<IDX_B_OFFSET>(baseOffset_) = (n * k * static_cast<int64_t>(groupIdx)) >> 1;
            } else {
                Get<IDX_B_OFFSET>(baseOffset_) = n * k * static_cast<int64_t>(groupIdx);
            }
        }

        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // scaleAAxisBaseOffset (m, ceil(k,64), 2)
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) = mOffset * scaleK;
        // scaleBAxisBaseOffset (g, n, ceil(k,64), 2) or (g, ceil(k,64), n, 2)
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) = static_cast<int64_t>(groupIdx) * n * scaleK;
        if (AscendC::IsSameTypeV<DataTypeOut, fp4x2_e2m1_t> || AscendC::IsSameTypeV<DataTypeOut, fp4x2_e1m2_t>) {
            Get<IDX_C_OFFSET>(baseOffset_) = (mOffset * n / SWIGLU_N_HALF) >> 1;
        } else {
            Get<IDX_C_OFFSET>(baseOffset_) = mOffset * n / SWIGLU_N_HALF;
        }
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) =
            mOffset * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    }

    __aicore__ inline bool UpdateGroupParams(const Params& params, uint32_t loopIdx, uint32_t groupIdx)
    {
        int32_t splitValue = GetSplitValueFromGroupList(loopIdx, params.gmmParams.groupListType);
        Get<M_VALUE>(problemShape_) = splitValue;
        UpdateOffset(groupIdx);
        // split_m，when m=0, skip
        if (Get<M_VALUE>(problemShape_) <= 0 || Get<K_VALUE>(problemShape_) <= 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void InitParamsAndTensor(const Params& params)
    {
        Get<N_VALUE>(problemShape_) = params.gmmParams.matmulTiling->N;
        Get<K_VALUE>(problemShape_) = params.gmmParams.matmulTiling->Ka;
        curBaseM_ = params.gmmParams.baseM;
        groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.mmadParams.groupListGmAddr));
    }

    __aicore__ inline void ComputeOffset(int64_t &bRightOffset, int64_t &x2ScaleRightOffset, int64_t n, int64_t k)
    {
        int64_t resN = n / 2; // SwiGLU: N -> N/2

        if constexpr (formatB == CubeFormat::NZ) {
            if constexpr (transB) {
                bRightOffset += resN * c0Size;
            } else {
                bRightOffset += resN * CeilDiv(k, MATMUL_MNK_ALIGN) * MATMUL_MNK_ALIGN;
            }
        } else {
            if constexpr (transB) {
                bRightOffset += resN * k;
            } else {
                bRightOffset += resN;
            }
        }

        if constexpr (transB) {
            x2ScaleRightOffset += resN * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        } else {
            x2ScaleRightOffset += resN * MXFP_MULTI_BASE_SIZE;
        }
    }

    template <bool isLastGroupAndNeedSplit>
    __aicore__ inline void ProcessSingleGroup(const Params& params, BlockSchedulerOp& bs, uint32_t)
    {
        int64_t m = Get<M_VALUE>(problemShape_);
        int64_t n = Get<N_VALUE>(problemShape_);
        int64_t k = Get<K_VALUE>(problemShape_);
        UpdateGlobalBuffer(params);
        if constexpr (!isLastGroupAndNeedSplit) {
            SetL2CacheDisableIfNeeded(m, static_cast<int64_t>(curBaseM_), static_cast<int64_t>(params.gmmParams.baseN));
        }
        CoordClass coord(m, n, k, curBaseM_, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockCoord tileIdx;
        while (bs.GetTileIdx(tileIdx)) {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            if (Get<M_VALUE>(singleShape) <= 0 || Get<N_VALUE>(singleShape) <= 0) {
                return;
            }
            // isMx = true
            blockOffset_ = coord.template GetQuantIOOffset<GroupedMatmul::QuantMode::MX_PERGROUP_MODE>(
                Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
                Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
            if constexpr (formatB == CubeFormat::NZ && transB) {
                int64_t nOffset =
                    Get<IDX_N_TILEIDX>(tileIdx) * static_cast<int64_t>(params.gmmParams.baseN) +
                    Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape);
                Get<IDX_B_OFFSET>(blockOffset_) = nOffset * c0Size;
            }
            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForVector();
                }
                AscendC::Std::tuple<int32_t, int32_t, int32_t> mmSingleShape{Get<M_VALUE>(singleShape),
                                                                             Get<N_VALUE>(singleShape), k};
                // left block
                mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
                        x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
                        x2ScaleGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)], l0cOutUbFirst_, mmSingleShape, transA,
                        transB);
                // right block
                int64_t bRightOffset = Get<IDX_B_OFFSET>(blockOffset_);
                int64_t x2ScaleRightOffset = Get<IDX_X2SCALE_OFFSET>(blockOffset_);
                ComputeOffset(bRightOffset, x2ScaleRightOffset, n, k);
                mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[bRightOffset],
                        x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)], x2ScaleGlobal_[x2ScaleRightOffset],
                        l0cOutUbSecond_, mmSingleShape, transA, transB);
                NotifyVector();
            }
            isVecSetSyncCom_ = true;
            if ASCEND_IS_AIV {
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(singleShape),
                                                                                      Get<N_VALUE>(singleShape), 0, 0};
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    Get<IDX_C_OFFSET>(blockOffset_), Get<IDX_C_SCALE_OFFSET>(blockOffset_),
                    Get<IDX_X2SCALE_OFFSET>(blockOffset_), Get<IDX_X1SCALE_OFFSET>(blockOffset_),
                    Get<IDX_BIAS_OFFSET>(blockOffset_),
                };
                WaitForCube();
                epilogueOp_(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }

    __aicore__ inline void operator()(const Params& params)
    {
        // Init mmadOp epilogue
        mmadOp_.Init(const_cast<TCubeTiling* __restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
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
        if (groupNum == 0) {
            End();
            return;
        }

        for (uint32_t loopIdx = 0; loopIdx < groupNum - 1; loopIdx++) {
            uint32_t groupIdx = loopIdx;
            if (params.gmmParams.groupListType == GROUP_LIST_TYPE_SPARSE) {
                groupIdx = static_cast<uint32_t>(groupListGm_.GetValue(loopIdx * SPARSE_GROUP_LIST_ITEM_STRIDE));
            }
            if (!UpdateGroupParams(params, loopIdx, groupIdx)) {
                if (params.gmmParams.groupListType == GROUP_LIST_TYPE_SPARSE && Get<M_VALUE>(problemShape_) <= 0) {
                    break;
                }
                continue;
            }
            BaseMBalance(bs, Get<M_VALUE>(problemShape_), params.gmmParams.baseM);
            TupleShape resProblemShape{Get<M_VALUE>(problemShape_), Get<N_VALUE>(problemShape_) >> 1,
                                       Get<K_VALUE>(problemShape_)};
            bs.UpdateNextProblem(resProblemShape);
            epilogueOp_.UpdateNextProblem(resProblemShape);
            ProcessSingleGroup<false>(params, bs, groupIdx);
        }

        uint32_t loopIdx = groupNum - 1;
        uint32_t groupIdx = loopIdx;
        if (params.gmmParams.groupListType == GROUP_LIST_TYPE_SPARSE) {
            groupIdx = static_cast<uint32_t>(groupListGm_.GetValue(loopIdx * SPARSE_GROUP_LIST_ITEM_STRIDE));
        }
        if (UpdateGroupParams(params, loopIdx, groupIdx)) {
            BaseMBalance(bs, Get<M_VALUE>(problemShape_), params.gmmParams.baseM);
            TupleShape resProblemShape{Get<M_VALUE>(problemShape_), Get<N_VALUE>(problemShape_) >> 1,
                                       Get<K_VALUE>(problemShape_)};
            bs.UpdateNextProblem(resProblemShape);
            epilogueOp_.UpdateNextProblem(resProblemShape);
            if (IfNeedSplit(bs)) {
                bs.UpdateTailTile();
                ProcessSingleGroup<true>(params, bs, groupIdx);
            } else {
                ProcessSingleGroup<false>(params, bs, groupIdx);
            }
        }
        End();
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Cgmct
#endif // AOT_KERNEL_ASWT_GROUPED_MATMUL_H
