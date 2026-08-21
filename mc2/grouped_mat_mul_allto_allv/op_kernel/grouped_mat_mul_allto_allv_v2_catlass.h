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
 * \file grouped_mat_mul_allto_allv_v2_catlass.h
 * \brief CATLASS GroupedMatmulSliceM adapter used by the AIV memory-semantics path.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_V2_CATLASS_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_V2_CATLASS_H

#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "grouped_mat_mul_allto_allv_v2_tiling.h"

namespace MC2KernelTemplate {

template <typename T, bool WeightTrans, bool IsSharedExpert>
class CatlassGroupedMatmulOp {
public:
    __aicore__ inline CatlassGroupedMatmulOp() = default;

    template <typename TilingDataType, typename GmmTilingDataType, typename GmmArrayType>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR weightScale, GM_ADDR output,
                                GM_ADDR workspace, const TilingDataType *tilingData,
                                const GmmTilingDataType *gmmTilingData, GmmArrayType *gmmArray,
                                AscendC::TPipe *pipe, bool isA2avGmm = false)
    {
        (void)xScale;
        (void)weightScale;
        (void)gmmTilingData;
        (void)gmmArray;
        (void)pipe;
        (void)isA2avGmm;
        if ASCEND_IS_AIV {
            return;
        }
        x_ = x;
        weight_ = weight;
        output_ = output;
        (void)workspace;
        taskTilingInfo_ = &tilingData->taskTilingInfo;
    }

    __aicore__ inline void Process(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Correctness-first AIV path: a single AIC owns the CATLASS resource
        // and executes all group tiles. This avoids event-resource aliasing
        // between multiple AIC instances in the MIX_AIC_1_2 kernel.
        if (AscendC::GetBlockIdx() != 0U) {
            return;
        }
        if constexpr (IsSharedExpert) {
            if (startExpertIdx != 0U || expertNum != 1U) {
                return;
            }
            Run(taskTilingInfo_->BS, taskTilingInfo_->H2, taskTilingInfo_->N2, 0U, 1U, true);
        } else {
            if (startExpertIdx >= taskTilingInfo_->e || expertNum == 0U) {
                return;
            }
            uint32_t actualExpertNum = expertNum;
            if (startExpertIdx + actualExpertNum > taskTilingInfo_->e) {
                actualExpertNum = taskTilingInfo_->e - startExpertIdx;
            }
            Run(taskTilingInfo_->A, taskTilingInfo_->H1, taskTilingInfo_->N1,
                startExpertIdx, actualExpertNum, false);
        }
    }

    __aicore__ inline void Process(uint32_t expertIdx)
    {
        Process(expertIdx, 1U);
    }

    __aicore__ inline void End() {}

private:
    __aicore__ inline uint32_t GetExpertTokens(uint32_t expertIdx) const
    {
        uint32_t tokens = 0U;
        for (uint32_t rank = 0U; rank < taskTilingInfo_->epWorldSize; ++rank) {
            tokens += taskTilingInfo_->sendCnt[rank * taskTilingInfo_->e + expertIdx];
        }
        return tokens;
    }

    __aicore__ inline void Run(uint32_t m, uint32_t k, uint32_t n, uint32_t startExpertIdx,
                               uint32_t expertNum, bool isSharedExpert)
    {
        using namespace Catlass;
        // Match CATLASS' validated Pingpong configuration. Only AIC0 owns this
        // resource, so its unit-flag lifecycle cannot race another CATLASS block.
        constexpr bool enableUnitFlag = true;

        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
        // Correctness-first fixed tiles; host-side tuning is intentionally deferred.
        using L1TileShape = GemmShape<Gmma2avV2Tiling::L1_TILE_M, Gmma2avV2Tiling::L1_TILE_N,
                                      Gmma2avV2Tiling::L1_TILE_K>;
        using L0TileShape = GemmShape<Gmma2avV2Tiling::L0_TILE_M, Gmma2avV2Tiling::L0_TILE_N,
                                      Gmma2avV2Tiling::L0_TILE_K>;
        using LayoutA = layout::RowMajor;
        using LayoutB = std::conditional_t<WeightTrans, layout::ColumnMajor, layout::RowMajor>;
        using LayoutC = layout::RowMajor;
        using AType = Gemm::GemmType<T, LayoutA>;
        using BType = Gemm::GemmType<T, LayoutB>;
        using CType = Gemm::GemmType<T, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<1, 0>;

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        BlockScheduler blockScheduler;
        uint64_t groupOffsetA = 0UL;
        uint64_t groupOffsetB = static_cast<uint64_t>(startExpertIdx) * k * n;
        uint64_t groupOffsetC = 0UL;

        if (!isSharedExpert) {
            for (uint32_t expert = 0U; expert < startExpertIdx; ++expert) {
                uint32_t tokens = GetExpertTokens(expert);
                groupOffsetA += static_cast<uint64_t>(tokens) * k;
                groupOffsetC += static_cast<uint64_t>(tokens) * n;
            }
        }

        for (uint32_t localExpert = 0U; localExpert < expertNum; ++localExpert) {
            uint32_t expertIdx = startExpertIdx + localExpert;
            uint32_t currentM = isSharedExpert ? m : GetExpertTokens(expertIdx);
            if (currentM == 0U) {
                groupOffsetB += static_cast<uint64_t>(k) * n;
                continue;
            }

            GemmCoord problemShape{currentM, n, k};
            LayoutA layoutA{currentM, k};
            LayoutB layoutB{k, n};
            LayoutC layoutC{currentM, n};
            blockScheduler.Update(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            AscendC::GlobalTensor<T> gmA;
            AscendC::GlobalTensor<T> gmB;
            AscendC::GlobalTensor<T> gmC;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x_) + groupOffsetA);
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight_) + groupOffsetB);
            gmC.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output_) + groupOffsetC);

            for (uint32_t loopIdx = 0U; loopIdx < coreLoops; ++loopIdx) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, 0};
                MatrixCoord offsetB{0, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                blockMmad(gmA[layoutA.GetOffset(offsetA)], layoutA,
                          gmB[layoutB.GetOffset(offsetB)], layoutB,
                          gmC[layoutC.GetOffset(offsetC)], layoutC, actualBlockShape);
            }

            groupOffsetA += static_cast<uint64_t>(currentM) * k;
            groupOffsetB += static_cast<uint64_t>(k) * n;
            groupOffsetC += static_cast<uint64_t>(currentM) * n;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    GM_ADDR x_{nullptr};
    GM_ADDR weight_{nullptr};
    GM_ADDR output_{nullptr};
    const TaskTilingInfo *taskTilingInfo_{nullptr};
};

} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_V2_CATLASS_H
