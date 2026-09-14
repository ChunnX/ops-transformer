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

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "grouped_mat_mul_allto_allv_mte_tiling.h"

namespace MC2KernelTemplate {

template <typename T, bool WeightTrans, bool IsSharedExpert>
class CatlassGroupedMatmulOp {
private:
    struct IgnoreWaveCompletion {
        __aicore__ inline void operator()(uint32_t) const {}
    };

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
        IgnoreWaveCompletion callback;
        ProcessRange<false>(startExpertIdx, expertNum, callback);
    }

    // Keep a single CATLASS resource lifetime across experts. Each local
    // wave contains at most M_GROUPS_PER_WAVE complete M tiles (all N tiles).
    // The completion callback publishes a monotonic GM boundary after FIX_S;
    // AIV0 can stage that wave while AIC computes the following wave.
    template <typename WaveCompletion>
    __aicore__ inline void ProcessExperts(uint32_t startExpertIdx, uint32_t expertNum,
                                          WaveCompletion &&onWaveComplete)
    {
        ProcessRange<true>(startExpertIdx, expertNum, onWaveComplete);
    }

    __aicore__ inline void Process(uint32_t expertIdx)
    {
        Process(expertIdx, 1U);
    }

    __aicore__ inline void End() {}

private:
    template <bool NotifyWaves, typename WaveCompletion>
    __aicore__ inline void ProcessRange(uint32_t startExpertIdx, uint32_t expertNum, WaveCompletion &onWaveComplete)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if (taskTilingInfo_ == nullptr || expertNum == 0U) {
            return;
        }
        if constexpr (IsSharedExpert) {
            if (startExpertIdx != 0U || expertNum != 1U) {
                return;
            }
            Run<NotifyWaves>(taskTilingInfo_->BS, taskTilingInfo_->H2, taskTilingInfo_->N2, 0U, 1U, true,
                                    onWaveComplete);
        } else {
            const uint32_t totalExpertNum = static_cast<uint32_t>(taskTilingInfo_->e);
            if (startExpertIdx >= totalExpertNum) {
                return;
            }
            uint32_t actualExpertNum = expertNum;
            if (actualExpertNum > totalExpertNum - startExpertIdx) {
                actualExpertNum = totalExpertNum - startExpertIdx;
            }
            Run<NotifyWaves>(taskTilingInfo_->A, taskTilingInfo_->H1, taskTilingInfo_->N1, startExpertIdx,
                                    actualExpertNum, false, onWaveComplete);
        }
    }

    __aicore__ inline uint32_t GetExpertTokens(uint32_t expertIdx) const
    {
        uint32_t tokens = 0U;
        for (uint32_t rank = 0U; rank < taskTilingInfo_->epWorldSize; ++rank) {
            tokens += taskTilingInfo_->sendCnt[rank * taskTilingInfo_->e + expertIdx];
        }
        return tokens;
    }

    template <bool NotifyWaves, typename WaveCompletion>
    __aicore__ inline void Run(uint32_t m, uint32_t k, uint32_t n, uint32_t startExpertIdx, uint32_t expertNum,
                               bool isSharedExpert, WaveCompletion &onWaveComplete)
    {
        using namespace Catlass;
        // Every AIC owns an independent CATLASS resource and processes its
        // strided subset of L1 tiles.
        // Match the arch22 AllToAllMatmul/MatmulAlltoAll compute path.  This
        // grouped kernel publishes GM wave progress instead of consuming
        // per-tile unit events. FIX_M/M_FIX still guard L0C reuse; the
        // completion callback orders FIX stores before scalar publication.
        constexpr bool enableUnitFlag = false;

        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
        // Correctness-first fixed tiles; host-side tuning is intentionally deferred.
        using L1TileShape =
            GemmShape<Gmma2avMteTiling::L1_TILE_M, Gmma2avMteTiling::L1_TILE_N, Gmma2avMteTiling::L1_TILE_K>;
        using L0TileShape =
            GemmShape<Gmma2avMteTiling::L0_TILE_M, Gmma2avMteTiling::L0_TILE_N, Gmma2avMteTiling::L0_TILE_K>;
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
                if (tokens != 0U) {
                    // A sub-range must use the same first-core rotation as a
                    // full [0, E) invocation. The rotation advances by the
                    // expert's M-tile count (loopsM), matching the M-row
                    // striped ownership below.
                    const uint32_t loopsM = (tokens + L1TileShape::M - 1U) / L1TileShape::M;
                    startCoreIdx = (startCoreIdx + loopsM) % aicCoreNum;
                }
            }
        }

        for (uint32_t localExpert = 0U; localExpert < expertNum; ++localExpert) {
            uint32_t expertIdx = startExpertIdx + localExpert;
            uint32_t currentM = isSharedExpert ? m : GetExpertTokens(expertIdx);
            uint32_t loopsM = 0U;
            if (currentM != 0U) {
                GemmCoord problemShape{currentM, n, k};
                LayoutA layoutA{currentM, k};
                LayoutB layoutB{k, n};
                LayoutC layoutC{currentM, n};
                blockScheduler.Update(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
                loopsM = blockScheduler.loopsMN.row();
                const uint32_t loopsN = blockScheduler.loopsMN.column();
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

                // M-row striped ownership: this AIC computes complete output
                // rows (all N tiles) of the M tiles whose index satisfies
                // mIdx === (coreIdx - startCoreIdx) mod aicCoreNum. Unlike a
                // flat per-tile stripe, no N tile of a row is ever produced
                // by another AIC, so the paired AIV can stage the row as soon
                // as this core publishes the wave complete. The AIC/AIV pair
                // derives the same ownership from the same rotation state.
                const uint32_t startMIdx =
                    coreIdx < startCoreIdx ? coreIdx + aicCoreNum - startCoreIdx : coreIdx - startCoreIdx;
                uint32_t ownedTileCount = 0U;
                for (uint32_t mIdx = startMIdx; mIdx < loopsM; mIdx += aicCoreNum) {
                    if constexpr (NotifyWaves) { onWaveComplete.MmBegin(expertIdx, loopsN); }
                    for (uint32_t nIdx = 0U; nIdx < loopsN; ++nIdx) {
                        const uint32_t loopIdx = mIdx * loopsN + nIdx;
                        GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                        GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                        MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, 0};
                        MatrixCoord offsetB{0, blockCoord.n() * L1TileShape::N};
                        MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                        blockMmad(gmA[layoutA.GetOffset(offsetA)], layoutA, gmB[layoutB.GetOffset(offsetB)], layoutB,
                                  gmC[layoutC.GetOffset(offsetC)], layoutC, actualBlockShape);
                    }
                    if constexpr (NotifyWaves) {
                        onWaveComplete.MmEnd();
                        ++ownedTileCount;
                        if (ownedTileCount % Gmma2avMteTiling::M_GROUPS_PER_WAVE == 0U ||
                            mIdx + aicCoreNum >= loopsM) {
                            onWaveComplete.NotifyWaveReady();
                        }
                    }
                }
            }

            groupOffsetA += static_cast<uint64_t>(currentM) * k;
            groupOffsetB += static_cast<uint64_t>(k) * n;
            groupOffsetC += static_cast<uint64_t>(currentM) * n;
            startCoreIdx = (startCoreIdx + loopsM) % aicCoreNum;


        }

        if constexpr (!NotifyWaves) {
            // The optional shared MM has no expert-ready callback; preserve
            // its final-drain semantics. The
            // synchronous ping-pong policy is drained by BlockMmad's
            // destructor, while an asynchronous policy needs this flush.
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }
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
