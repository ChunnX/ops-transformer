#ifndef MEGA_MOE_ASCEND_TIMER_V2_DEVICE_HPP
#define MEGA_MOE_ASCEND_TIMER_V2_DEVICE_HPP

#include "AscendTimerV2.hpp"

struct TimerOverwriteTag {};
struct TimerAccumulateTag {};

class AscendTimerDevice {
public:
    __aicore__ inline AscendTimerDevice() = default;
    __aicore__ inline void Init(GM_ADDR timerAddr)
    {
        timerAddr_ = timerAddr;
        int blockId = AscendC::GetBlockIdx();
        groupId_ = blockId;
        subBlockId_ = 0;
        if ASCEND_IS_AIV {
            groupId_ = blockId / AscendC::GetTaskRation();
            subBlockId_ = AscendC::GetSubBlockIdx();
            coreId_ = groupId_ * 3 + subBlockId_ + 1;
        } else {
            coreId_ = groupId_ * 3;
        }
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamicStart_[i] = 0;
            dynamicIter_[i] = -1;
        }
    }
    __aicore__ inline void SetDynamicActualIter(AscendTimer::DynamicTimingType type, int64_t iter)
    {
        if (timerAddr_ == nullptr || coreId_ < 0 || coreId_ >= AscendTimer::N_CORE_COUNT ||
            !MatchesDynamicCore(type)) {
            return;
        }
        int32_t idx = AscendTimer::N_TIMING_COUNTER +
            coreId_ * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + static_cast<int32_t>(type);
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(timerAddr_));
        timer(idx) = iter;
    }
    __aicore__ inline bool IsAic() const
    {
        if ASCEND_IS_AIV {
            return false;
        }
        return true;
    }
    __aicore__ inline bool IsAiv0() const
    {
        if ASCEND_IS_AIV {
            return subBlockId_ == 0;
        }
        return false;
    }
    __aicore__ inline bool IsAiv1() const
    {
        if ASCEND_IS_AIV {
            return subBlockId_ == 1;
        }
        return false;
    }
    __aicore__ inline void Tik()
    {
        if (timerAddr_ == nullptr) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        start_ = AscendC::GetSystemCycle();
    }
    __aicore__ inline void Tik(AscendTimer::DynamicTimingType type, int iter)
    {
        int typeIdx = static_cast<int>(type);
        if (timerAddr_ == nullptr || typeIdx < 0 || typeIdx >= AscendTimer::DYNAMIC_TYPE_COUNT ||
            iter < 0 || iter >= AscendTimer::MAX_DYNAMIC_ITER || !MatchesDynamicCore(type)) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        dynamicStart_[typeIdx] = AscendC::GetSystemCycle();
        dynamicIter_[typeIdx] = iter;
    }
    template <typename ModeTag = TimerOverwriteTag>
    __aicore__ inline void Tok(int32_t index)
    {
        if (timerAddr_ == nullptr || index < 0 || index >= AscendTimer::N_TIMING_ITEM_PER_CORE ||
            !MatchesFixedCore(index)) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        Write(index, start_, AscendC::GetSystemCycle(), ModeTag{});
    }
    template <typename ModeTag = TimerOverwriteTag>
    __aicore__ inline void Tok(AscendTimer::DynamicTimingType type, int iter)
    {
        int typeIdx = static_cast<int>(type);
        if (timerAddr_ == nullptr || typeIdx < 0 || typeIdx >= AscendTimer::DYNAMIC_TYPE_COUNT ||
            iter < 0 || iter >= AscendTimer::MAX_DYNAMIC_ITER || dynamicIter_[typeIdx] != iter ||
            !MatchesDynamicCore(type)) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int index = AscendTimer::FIXED_TIMING_COUNT + typeIdx * AscendTimer::MAX_DYNAMIC_ITER + iter;
        Write(index, dynamicStart_[typeIdx], AscendC::GetSystemCycle(), ModeTag{});
        dynamicIter_[typeIdx] = -1;
    }
private:
    __aicore__ inline bool MatchesFixedCore(int32_t index) const
    {
        if (index == AscendTimer::AIC_WAIT_ALLGATHER) {
            return IsAic();
        }
        return index >= AscendTimer::AIV_INIT_PARAM && index < AscendTimer::AIC_WAIT_ALLGATHER &&
               (IsAiv0() || IsAiv1());
    }
    __aicore__ inline bool MatchesDynamicCore(AscendTimer::DynamicTimingType type) const
    {
        switch (type) {
            case AscendTimer::AIV_DISPATCH:
            case AscendTimer::AIV_SWIGLU_WAIT:
            case AscendTimer::AIV_SWIGLU:
            case AscendTimer::AIV_COMBINE_WAIT:
            case AscendTimer::AIV_COMBINE:
                return IsAiv1();
            case AscendTimer::AIV_GMM1_PROLOGUE:
            case AscendTimer::AIV_GMM2_PROLOGUE:
                return IsAiv0();
            case AscendTimer::AIC_GMM1_WAIT:
            case AscendTimer::AIC_GMM1:
            case AscendTimer::AIC_GMM2_WAIT:
            case AscendTimer::AIC_GMM2:
                return IsAic();
            default:
                return false;
        }
    }
    __aicore__ inline int32_t PhysicalIndex(int32_t logicalIndex) const
    {
        int32_t coreBase = groupId_ * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3;
        if ASCEND_IS_AIV {
            coreBase += AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * (1 + subBlockId_);
        }
        return coreBase + logicalIndex * 2;
    }
    __aicore__ inline void Write(int32_t index, int64_t start, int64_t end, TimerOverwriteTag)
    {
        int32_t offset = PhysicalIndex(index);
        if (offset + 1 >= AscendTimer::N_TIMING_COUNTER) {
            return;
        }
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(timerAddr_));
        timer(offset) = start;
        timer(offset + 1) = end;
    }
    __aicore__ inline void Write(int32_t index, int64_t start, int64_t end, TimerAccumulateTag)
    {
        int32_t offset = PhysicalIndex(index);
        if (offset + 1 >= AscendTimer::N_TIMING_COUNTER) {
            return;
        }
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(timerAddr_));
        int64_t oldStart = timer(offset);
        int64_t oldEnd = timer(offset + 1);
        timer(offset) = oldStart == 0 ? start : oldStart;
        timer(offset + 1) = oldEnd == 0 ? end : oldEnd + (end - start);
    }
    GM_ADDR timerAddr_ = nullptr;
    int64_t start_ = 0;
    int64_t dynamicStart_[AscendTimer::DYNAMIC_TYPE_COUNT]{};
    int dynamicIter_[AscendTimer::DYNAMIC_TYPE_COUNT]{};
    int groupId_ = 0;
    int subBlockId_ = 0;
    int coreId_ = 0;
};

#endif
