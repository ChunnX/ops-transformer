#ifndef MC2_KERNEL_TIMING_H
#define MC2_KERNEL_TIMING_H

#include "kernel_operator.h"

namespace Mc2Kernel {

// KernelTiming — 累计打点计时工具（无全局变量，全部为类成员）
//
// 与覆盖式版本不同，本版本对同一 slot 多次 Start/Stop 会【累加】duration，
// 适用于循环体内累计耗时分析。每个 slot 记录：
//   - lastStart / lastStop : 最后一次 Start/Stop 的原始 GetSystemCycle() 时间戳
//   - accumDuration        : 所有 Start/Stop 配对的 (stop-start) 累加值（cycle）
//   - callCount            : Start/Stop 配对次数
//
// 内存布局（写入输出 GM 头部）：
//   每个核 16 slots × 4 个 uint64 = 512 字节
//   layout: [coreId * 16 * 4 + slot * 4 + 0] = lastStart
//           [coreId * 16 * 4 + slot * 4 + 1] = lastStop
//           [coreId * 16 * 4 + slot * 4 + 2] = accumDuration
//           [coreId * 16 * 4 + slot * 4 + 3] = callCount
class KernelTiming {
public:
    static constexpr int32_t MAX_SLOTS = 16;
    static constexpr int32_t VALS_PER_SLOT = 4; // lastStart, lastStop, accumDuration, callCount

    __aicore__ inline KernelTiming() : gmOutputAddr_(nullptr), currentId_(0) {}

    __aicore__ inline void Init(GM_ADDR outputAddr)
    {
        gmOutputAddr_ = outputAddr;
        currentId_ = 0;
        for (int32_t i = 0; i < MAX_SLOTS; i++) {
            lastStart_[i] = 0;
            lastStop_[i] = 0;
            accumDuration_[i] = 0;
            callCount_[i] = 0;
        }
    }

    // 开始计时。slotId >= 0 显式指定 slot；否则自增。记录 lastStart。
    __aicore__ inline void StartTiming(int32_t slotId = -1)
    {
        if (slotId >= 0) {
            currentId_ = slotId;
        } else {
            currentId_++;
        }
        if (currentId_ < 0 || currentId_ >= MAX_SLOTS) {
            return;
        }
        lastStart_[currentId_] = AscendC::GetSystemCycle();
    }

    // 停止计时。累加 (now - lastStart) 到 accumDuration，callCount++。
    __aicore__ inline void StopTiming(int32_t slotId = -1)
    {
        if (slotId >= 0) {
            currentId_ = slotId;
        }
        if (currentId_ < 0 || currentId_ >= MAX_SLOTS) {
            return;
        }
        uint64_t now = AscendC::GetSystemCycle();
        lastStop_[currentId_] = now;
        if (now >= lastStart_[currentId_]) {
            accumDuration_[currentId_] += (now - lastStart_[currentId_]);
        }
        callCount_[currentId_]++;
    }

    __aicore__ inline int32_t GetCurrentSlotId() const
    {
        return currentId_;
    }

    // 将累计 timing 数据写入输出 GM 头部。必须在 Process 最末尾调用。
    // isAicActivated=true 用于 AIC+AIV 混合模式（SyncAll<false> 同步全部核），
    // AIV 核的 coreId 会自动加上 aicNum 偏移。
    __aicore__ inline void WriteTimingToOutput(int32_t coreId, bool isAicActivated = false, int32_t aicNum = 0)
    {
        PipeBarrier<PIPE_ALL>();
        if (isAicActivated) {
            SyncAll<false>();
        } else {
            SyncAll<true>();
        }

        GlobalTensor<uint64_t> output;
        output.SetGlobalBuffer((__gm__ uint64_t *)gmOutputAddr_);

        if (isAicActivated && g_coreType == AscendC::AIV) {
            coreId = coreId + aicNum;
        }
        for (int32_t i = 0; i < MAX_SLOTS; i++) {
            int32_t base = coreId * MAX_SLOTS * VALS_PER_SLOT + i * VALS_PER_SLOT;
            output.SetValue(base + 0, lastStart_[i]);
            output.SetValue(base + 1, lastStop_[i]);
            output.SetValue(base + 2, accumDuration_[i]);
            output.SetValue(base + 3, callCount_[i]);
        }
    }

private:
    uint64_t lastStart_[MAX_SLOTS];
    uint64_t lastStop_[MAX_SLOTS];
    uint64_t accumDuration_[MAX_SLOTS];
    uint64_t callCount_[MAX_SLOTS];
    int32_t  currentId_;
    GM_ADDR  gmOutputAddr_;
};

}  // namespace Mc2Kernel

#endif  // MC2_KERNEL_TIMING_H
