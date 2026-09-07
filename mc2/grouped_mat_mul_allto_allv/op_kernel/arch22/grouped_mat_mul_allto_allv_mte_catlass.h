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
 * \file grouped_mat_mul_allto_allv_mte_catlass.h
 * \brief CATLASS GroupedMatmulSliceM adapter used by the AIV-driven MTE path.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_MTE_CATLASS_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_MTE_CATLASS_H

#include <type_traits>

#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_arch.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_dispatch_policy.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/tla_gemm_kernel_basic_matmul.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "grouped_mat_mul_allto_allv_mte_tiling.h"

namespace MC2KernelTemplate {

template <typename T, bool WeightTrans, bool IsSharedExpert>
class CatlassGroupedMatmulOp {
public:
    __aicore__ inline CatlassGroupedMatmulOp() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR output,
                                const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo)
    {
        if ASCEND_IS_AIV {
            return;
        }
        x_ = x;
        weight_ = weight;
        output_ = output;
        taskTilingInfo_ = taskTilingInfo;
    }

    __aicore__ inline void Process(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIV {
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
        // Every AIC owns an independent CATLASS resource and processes its
        // strided subset of L1 tiles.
        // Match the arch22 AllToAllMatmul/MatmulAlltoAll compute path.  This
        // grouped kernel publishes one complete output chunk rather than
        // consuming per-tile unit events, so FIX_M/M_FIX must guard L0C reuse
        // and GM visibility before the AIV ready event is emitted.
        constexpr bool enableUnitFlag = false;

        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
        // Correctness-first fixed tiles; host-side tuning is intentionally deferred.
        using L1TileShape = GemmShape<Gmma2avMteTiling::L1_TILE_M, Gmma2avMteTiling::L1_TILE_N,
                                      Gmma2avMteTiling::L1_TILE_K>;
        using L0TileShape = GemmShape<Gmma2avMteTiling::L0_TILE_M, Gmma2avMteTiling::L0_TILE_N,
                                      Gmma2avMteTiling::L0_TILE_K>;
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
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t aicCoreNum = AscendC::GetBlockNum();
        uint32_t startCoreIdx = 0U;

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
            // Keep the same small-M weight-load policy as CATLASS
            // GroupedMatmulSliceM.  Every expert in the current model cases
            // occupies one M tile; bypassing L2 avoids reusing a stale weight
            // line when allocator-backed expert buffers are recycled between
            // invocations.
            if (CeilDiv(currentM, L1TileShape::M) == 1U) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            // Continue the tile-to-core sequence across expert boundaries.
            // This is the allocation used by CATLASS GroupedMatmulSliceM;
            // restarting every small-M expert at core 0 leaves most AICs idle
            // and does not follow the resource/event progression expected by
            // the grouped kernel.
            const uint32_t startLoopIdx = coreIdx < startCoreIdx ?
                coreIdx + aicCoreNum - startCoreIdx : coreIdx - startCoreIdx;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
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
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }

        // Drain all asynchronous L1/L0/FIX stages before the scheduler
        // publishes this HCCL window to peer readers.
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

    }

private:
    GM_ADDR x_{nullptr};
    GM_ADDR weight_{nullptr};
    GM_ADDR output_{nullptr};
    const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo_{nullptr};
};

} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_MTE_CATLASS_H
