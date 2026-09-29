/*
 * Deterministic synchronization-only extraction of the original kernel.
 * Numerical work, data movement and local-memory allocation are deliberately
 * absent; the original cross-core synchronization topology is preserved.
 */
#ifndef MLP_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SYNC_ONLY_H
#define MLP_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SYNC_ONLY_H

#include "kernel_operator.h"
#include "mlp_sparse_lightning_indexer_grad_kl_loss_sync_only_config.h"

namespace MlpSliSyncOnly {
constexpr uint8_t SYNC_V0_TO_C1_P_SY_FLAG[2] = {0, 1};
constexpr uint8_t SYNC_C1_TO_V1_P_FLAG[2] = {4, 5};
constexpr uint8_t SYNC_C1_TO_V1_SY_FLAG[2] = {6, 7};
constexpr uint8_t SYNC_V1_TO_V1_PSY_FLAG = SLI_SYNC_ONLY_VV_EVENT_ID;
constexpr uint8_t SYNC_V1_TO_C2_DW_FLAG[2] = {8, 9};
constexpr uint8_t SYNC_C2_TO_V2_SA_FLAG[2] = {10, 3};
#if SLI_SYNC_ONLY_ISOLATE_V2_EVENT == 1
constexpr uint8_t SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[2] = {15, 15};
#elif SLI_SYNC_ONLY_ISOLATE_V2_EVENT == 2
constexpr uint8_t SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[2] = {15, 1};
#elif SLI_SYNC_ONLY_ISOLATE_V2_EVENT == 3
constexpr uint8_t SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[2] = {0, 15};
#else
constexpr uint8_t SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[2] = {0, 1};
#endif
constexpr uint8_t SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0 = 2;
constexpr uint8_t SYNC_PIPELINE_ROUND_FLAG = 14;
constexpr int64_t S2_BASE_SIZE = 1024;

struct RunInfo {
    uint32_t taskId = 0;
    int64_t globalIndex = 0;
    bool isValid = false;
};

__aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

__aicore__ inline int64_t MinValue(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t MixBits(uint32_t value)
{
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

__aicore__ inline void Nop(int32_t count)
{
    AscendC::PipeBarrier<PIPE_ALL>();
#pragma unroll
    for (int32_t i = 0; i < count; ++i) {
        asm volatile("nop");
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

class DeterministicSyncKernel {
public:
    __aicore__ inline void Init(const MlpSparseLightningIndexerGradKLLossTilingData *tilingData)
    {
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        totalSize_ = tilingData->totalSize;
        kSize_ = tilingData->kSize;
        s1Size_ = tilingData->s1Size;
        s2Size_ = tilingData->s2Size;
        if ASCEND_IS_AIV {
            const uint32_t aivIdx = AscendC::GetBlockIdx();
            aicIdx_ = static_cast<int64_t>(aivIdx / 2U);
            subBlockIdx_ = aivIdx & 1U;
        } else if ASCEND_IS_AIC {
            aicIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
            subBlockIdx_ = 0;
        }
    }

    __aicore__ inline void Process()
    {
        AscendC::SyncAll<false>();
        DeterProcess();
        AscendC::SyncAll<false>();
    }

private:
    __aicore__ inline int64_t KRealSize(int64_t globalIndex) const
    {
        int64_t valid = s2Size_ - s1Size_ + globalIndex + 1;
        if (valid < 0) {
            valid = 0;
        }
        return MinValue(kSize_, valid);
    }

    __aicore__ inline void BuildRunInfo(int64_t localTaskId, int64_t globalIndex, RunInfo &runInfo)
    {
        runInfo.taskId = static_cast<uint32_t>(localTaskId);
        runInfo.globalIndex = globalIndex;
        runInfo.isValid = globalIndex < totalSize_ && KRealSize(globalIndex) > 0;
    }

    __aicore__ inline bool SelectDelay(uint32_t taskId) const
    {
#if !SLI_SYNC_ONLY_DELAY_ENABLE
        (void)taskId;
        return false;
#else
        if (SLI_SYNC_ONLY_DELAY_TARGET_AIC >= 0 &&
            aicIdx_ != static_cast<int64_t>(SLI_SYNC_ONLY_DELAY_TARGET_AIC)) {
            return false;
        }
        if (SLI_SYNC_ONLY_DELAY_TARGET_AIV >= 0 &&
            subBlockIdx_ != static_cast<uint32_t>(SLI_SYNC_ONLY_DELAY_TARGET_AIV)) {
            return false;
        }
        if (taskId < static_cast<uint32_t>(SLI_SYNC_ONLY_DELAY_FIRST_TASK)) {
            return false;
        }
        const uint32_t relative = taskId - static_cast<uint32_t>(SLI_SYNC_ONLY_DELAY_FIRST_TASK);
        if (relative % static_cast<uint32_t>(SLI_SYNC_ONLY_DELAY_TASK_STRIDE) != 0) {
            return false;
        }
        const uint32_t hit = relative / static_cast<uint32_t>(SLI_SYNC_ONLY_DELAY_TASK_STRIDE);
        return SLI_SYNC_ONLY_DELAY_MAX_HITS == 0 ||
               hit < static_cast<uint32_t>(SLI_SYNC_ONLY_DELAY_MAX_HITS);
#endif
    }

    __aicore__ inline void Delay(uint32_t taskId)
    {
        if (!SelectDelay(taskId)) {
            return;
        }
        uint32_t count = static_cast<uint32_t>(SLI_SYNC_ONLY_DELAY_COUNT);
#if SLI_SYNC_ONLY_DELAY_MODE == 2
        uint32_t state = static_cast<uint32_t>(SLI_SYNC_ONLY_RANDOM_SEED);
        state ^= (static_cast<uint32_t>(aicIdx_) + 1U) * 0x9E3779B9U;
        state ^= (subBlockIdx_ + 1U) * 0x85EBCA6BU;
        state ^= (taskId + 1U) * 0xC2B2AE35U;
        state = MixBits(state);
        count = count == 0 ? 0 : state % (count + 1U);
#endif
        Nop(static_cast<int32_t>(count));
    }

    __aicore__ inline void ProcessVector0(const RunInfo &runInfo)
    {
        const int64_t s2LoopTimes = CeilDiv(KRealSize(runInfo.globalIndex), S2_BASE_SIZE);
        for (int64_t i = 0; i < s2LoopTimes; ++i) {
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V0_TO_C1_P_SY_FLAG[i & 1]);
        }
    }

    __aicore__ inline void ProcessCube1(const RunInfo &runInfo)
    {
        const int64_t s2LoopTimes = CeilDiv(KRealSize(runInfo.globalIndex), S2_BASE_SIZE);
        for (int64_t i = 0; i < s2LoopTimes; ++i) {
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_V0_TO_C1_P_SY_FLAG[i & 1]);
        }
        const uint32_t parity = runInfo.taskId & 1U;
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_TO_V1_P_FLAG[parity]);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_TO_V1_SY_FLAG[parity]);
    }

    __aicore__ inline void ProcessVector1(const RunInfo &runInfo)
    {
        const uint32_t parity = runInfo.taskId & 1U;
        AscendC::CrossCoreWaitFlag(SYNC_C1_TO_V1_P_FLAG[parity]);
        AscendC::CrossCoreWaitFlag(SYNC_C1_TO_V1_SY_FLAG[parity]);
        AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(SYNC_V1_TO_V1_PSY_FLAG);
        Delay(runInfo.taskId);
        AscendC::CrossCoreWaitFlag<1, PIPE_MTE2>(SYNC_V1_TO_V1_PSY_FLAG);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_TO_C2_DW_FLAG[parity]);
    }

    __aicore__ inline void ProcessCube2(const RunInfo &runInfo)
    {
        const uint32_t parity = runInfo.taskId & 1U;
        AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_V1_TO_C2_DW_FLAG[parity]);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_TO_V2_SA_FLAG[parity]);
    }

    __aicore__ inline void ProcessDeterVector2(const RunInfo &runInfo)
    {
        const uint32_t parity = runInfo.taskId & 1U;
        AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C2_TO_V2_SA_FLAG[parity]);
        AscendC::CrossCoreSetFlag<0, PIPE_MTE2>(SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[parity]);
        AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>(SYNC_C2_TO_V2_DETER_SA_FLAG_MOD0[parity]);
        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);

        for (int32_t idx = 0; idx < static_cast<int32_t>(coreNum_); ++idx) {
            const int64_t globalIndex = static_cast<int64_t>(runInfo.taskId) * coreNum_ + idx;
            if (globalIndex >= totalSize_) {
                continue;
            }
            if ((idx & 1) == 0) {
                AscendC::CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            }
            if ((idx & 1) == 1 || globalIndex == totalSize_ - 1) {
                AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
            }
        }
        AscendC::CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_V2_TO_V2_DETER_SA_FLAG_MOD0);
    }

    __aicore__ inline void DeterProcess()
    {
        constexpr int64_t extraLoopTimes = 2;
        RunInfo runInfos[3] = {};
        int64_t taskId = 0;
        for (int64_t globalIndex = aicIdx_;
             globalIndex < totalSize_ + extraLoopTimes * coreNum_;
             globalIndex += coreNum_) {
            RunInfo &runInfoNeg2 = runInfos[(taskId + 1) % 3];
            RunInfo &runInfoNeg1 = runInfos[(taskId + 2) % 3];
            RunInfo &runInfo0 = runInfos[taskId % 3];
            BuildRunInfo(taskId, globalIndex, runInfo0);

            if ASCEND_IS_AIV {
                AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_PIPELINE_ROUND_FLAG);
            } else if ASCEND_IS_AIC {
                AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_PIPELINE_ROUND_FLAG);
            }
            if (runInfo0.isValid) {
                if ASCEND_IS_AIV {
                    ProcessVector0(runInfo0);
                }
            }
            if (runInfoNeg1.isValid) {
                if ASCEND_IS_AIC {
                    ProcessCube1(runInfoNeg1);
                }
                if ASCEND_IS_AIV {
                    ProcessVector1(runInfoNeg1);
                }
                if ASCEND_IS_AIC {
                    ProcessCube2(runInfoNeg1);
                }
            }
            if (runInfoNeg2.isValid) {
                if ASCEND_IS_AIV {
                    ProcessDeterVector2(runInfoNeg2);
                    runInfoNeg2.isValid = false;
                }
            }
            ++taskId;
        }

        if (aicIdx_ + 1 > totalSize_ % coreNum_) {
            RunInfo tailRun;
            tailRun.taskId = static_cast<uint32_t>(taskId - extraLoopTimes);
            tailRun.globalIndex = static_cast<int64_t>(tailRun.taskId) * coreNum_ + aicIdx_;
            tailRun.isValid = true;
            if ASCEND_IS_AIC {
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_TO_V2_SA_FLAG[tailRun.taskId & 1U]);
            }
            if ASCEND_IS_AIV {
                ProcessDeterVector2(tailRun);
            }
        }
    }

    int64_t coreNum_ = 0;
    int64_t totalSize_ = 0;
    int64_t kSize_ = 0;
    int64_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t aicIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
};
} // namespace MlpSliSyncOnly

#endif
