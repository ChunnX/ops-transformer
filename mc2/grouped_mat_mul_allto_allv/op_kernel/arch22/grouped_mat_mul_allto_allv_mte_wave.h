/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Bounded compute waves and independent Stage/Comm role loops, adapted from mega_moe A5.
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_MTE_WAVE_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_MTE_WAVE_H

#include "kernel_operator.h"
#include "grouped_mat_mul_allto_allv_mte_tiling.h"

namespace MC2KernelTemplate {

template <typename CommOpType, typename ComputationOpType, typename SharedComputationOpType, bool IsNeedMM>
class GmmA2avMteScheduler {
private:
    using TimerType = typename CommOpType::Timer;
    // arch22-compatible counterpart of A5 NotifyWaveGmm2Ready. FIX_S orders
    // the L0C->GM stores before publishing progress. Flush the AIC scalar
    // store through DCache before AIV reads it through MTE2 on arch22.
    // No L0C->UB instruction and no reverse-credit dependency on AIV1.
    struct AicWaveReadyCallback {
        __gm__ int64_t *slot;
        int64_t sequence{0};
        TimerType *timer;
        uint32_t expert{0};
        __aicore__ inline void MmBegin(uint32_t e, uint32_t n) {
            expert = e;
            timer->Record(16, e);
            timer->AddCounter(14, n);
        }
        __aicore__ inline void MmEnd() { timer->Record(17, expert); }

        __aicore__ inline void NotifyWaveReady()
        {
            timer->Record(18, expert);
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
            AscendC::GlobalTensor<int64_t> ready;
            ready.SetGlobalBuffer(slot);
            ready.SetValue(0, ++sequence);
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                             AscendC::DcciDst::CACHELINE_OUT>(ready);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            timer->Record(19, expert);
        }
    };

public:
    static __aicore__ inline void ProcessAic(ComputationOpType &computeOp, SharedComputationOpType &sharedComputeOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo,
                                             GM_ADDR cumsumBuffer, TimerType *aicTimer, uint32_t timerEnabled)
    {
        if ASCEND_IS_AIC {
            TimerType nullTimer;
            if (aicTimer == nullptr) { aicTimer = &nullTimer; }
            aicTimer->Record(14, 0U);
            auto progress = reinterpret_cast<__gm__ int64_t *>(cumsumBuffer +
                Gmma2avMteTiling::ComputeProgressOffset(taskTilingInfo->epWorldSize, taskTilingInfo->e,
                                                       taskTilingInfo->aicCoreNum));
            AicWaveReadyCallback callback{progress + static_cast<uint64_t>(AscendC::GetBlockIdx()) *
                (Gmma2avMteTiling::STAGE_PROGRESS_SLOT_BYTES / sizeof(int64_t)), 0, aicTimer};
            aicTimer->Record(42, 0U);
            AscendC::SyncAll<false>();
            aicTimer->Record(43, 0U);
            computeOp.ProcessExperts(0U, static_cast<uint32_t>(taskTilingInfo->e), callback);
            if constexpr (IsNeedMM) {
                if (timerEnabled == 0U) { sharedComputeOp.Process(0U, 1U); }
            }
            aicTimer->Record(20, 0U);
            AscendC::SyncAll<false>();
            aicTimer->Record(21, 0U);
            aicTimer->Record(15, 0U);
            aicTimer->Finish();
        }
    }

    static __aicore__ inline void ProcessAiv(CommOpType &commOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo)
    {
        if ASCEND_IS_AIV {
            commOp.BeginExpertPipeline();
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            if (commOp.IsStageAiv()) {
                // Stage advances on local compute waves only. Its workspace
                // rows remain valid until the invocation finishes, so a later
                // monotonic progress value also satisfies an earlier wait.
                for (uint32_t expertIdx = 0U; expertIdx < expertNum; ++expertIdx) {
                    const bool hasWork = commOp.HasExpertWork(expertIdx);
                    commOp.StageExpertPayload(expertIdx, expertIdx + 1U);
                    if (hasWork) {
                        commOp.PublishStageProgress(expertIdx);
                    }
                }
            } else {
                // A5-style dedicated communication role: no compute-wave
                // flag waits, no return credit, no all-AIV barrier in this loop.
                // Cross-rank exchange uses complete experts because each rank
                // can have a different local wave count.
                for (uint32_t expertIdx = 0U; expertIdx < expertNum; ++expertIdx) {
                    commOp.CommExpertChunk(expertIdx, expertIdx + 1U);
                }
            }
            commOp.End();
        }
    }
};


} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_MTE_WAVE_H
