/*
 * 昇腾 NPU 设备端高精度打点计时器 (CocTimerDevice)
 * 作用：用于在 AIC/AIV 算子执行期间，精确记录各阶段的 System Cycle 并写回 Global Memory。
 */

#ifndef COC_TIMER_DEVICE_H
#define COC_TIMER_DEVICE_H

#include "coc_timer.hpp"

// 整个 NPU 上用于存储所有核心计时数据 + 动态迭代次数的全局缓冲区总大小（单位：int64_t 个数）
#define COC_TIMER_TOTAL_BUFFER_SIZE (CocTimer::N_TIMING_COUNTER + CocTimer::DYNAMIC_ITER_TOTAL_SIZE)

// ========== 1. 编译期路由标签 ==========
// 用于在 Tok 时通过模板参数决定是"累加耗时"还是"覆盖耗时"
typedef struct AccumulateTag_ Accumulate;
struct AccumulateTag_ {};

typedef struct OverwriteTag_ Overwrite;
struct OverwriteTag_ {};

class CocTimerDevice
{
public:
    // ========== 2. 成员变量 ==========
    GM_ADDR t_addr;                                                 // 指向全局计时缓冲区的 GM 地址 (Host 侧分配)
    int64_t start_cycle;                                            // 基础计时段的起点 (GetSystemCycle)
    int64_t dynamic_start_cycles[CocTimer::DYNAMIC_TYPE_COUNT];  // 每个动态类型的专属起点数组
    int16_t iter_counters[CocTimer::DYNAMIC_TYPE_COUNT];         // 记录各动态类型当前的迭代次数

    // 用于 start/end 存储的索引
    int32_t last_timing_idx;                                        // 最近一次 Tik 计算的 timing_idx
    bool type_valid[CocTimer::DYNAMIC_TYPE_COUNT];               // 校验配对

    // 核心身份标识
    int block_id;   // 当前 Block 索引
    int r;          // 任务分片比例
    int group_id;   // 组 ID
    int subblockid; // 子块 ID (仅 AIV 有效)
    int core_id;    // 映射后的全局物理核心 ID

    // ========== 3. 构造函数 ==========
    __aicore__ inline CocTimerDevice()
    {
        t_addr = nullptr;
        start_cycle = 0;
        block_id = 0;
        r = 0;
        group_id = 0;
        subblockid = 0;
        core_id = 0;
        last_timing_idx = 0;
        for (int i = 0; i < CocTimer::DYNAMIC_TYPE_COUNT; i++) {
            iter_counters[i] = 0;
            dynamic_start_cycles[i] = 0;
            type_valid[i] = false;  // 初始化每个类型的有效位
        }
    }

    __aicore__ inline CocTimerDevice(GM_ADDR timeAddr)
    {
        Init(timeAddr); // 复用 Init 逻辑，拒绝代码冗余
    }

    // ========== 4. 拷贝构造 ==========
    __aicore__ inline CocTimerDevice(const CocTimerDevice &other)
    {
        t_addr = other.t_addr;
        start_cycle = other.start_cycle;
        block_id = other.block_id;
        r = other.r;
        group_id = other.group_id;
        subblockid = other.subblockid;
        core_id = other.core_id;
    }

    // ========== 5. 生命周期初始化 (Init) ==========
    __aicore__ inline void Init(const GM_ADDR timeAddr)
    {
        t_addr = (GM_ADDR)timeAddr;
        block_id = AscendC::GetBlockIdx();
        r = AscendC::GetTaskRation();
        group_id = block_id;
        core_id = block_id;
        last_timing_idx = 0;

        // 异构核心身份计算
        if ASCEND_IS_AIV
        {
            group_id = group_id / r;
            subblockid = AscendC::GetSubBlockIdx();
            core_id = group_id * 3 + subblockid + 1;
        }
        else
        {
            subblockid = 0;
            core_id = group_id * 3;
        }

        // 初始化内部状态
        for (int i = 0; i < CocTimer::DYNAMIC_TYPE_COUNT; i++) {
            iter_counters[i] = 0;
            dynamic_start_cycles[i] = 0;
            type_valid[i] = false; // 重置有效位
        }

        // 内部自动执行显存清理，实现开箱即用
        zero();
        zeroDynamicActualIterAll();
    }

    __aicore__ inline void Init(const GM_ADDR timeAddr, int64_t startCycle)
    {
        Init(timeAddr);
        start_cycle = startCycle;
    }

    // ========== 6. 基础状态接口 ==========
    __aicore__ __inline__ GM_ADDR GetTAddr() { return t_addr; }
    __aicore__ __inline__ int64_t GetStartCycle() { return start_cycle; }
    __aicore__ __inline__ int GetCoreId() { return core_id; }

    // ========== 7. 内存清理接口 (Zero) ==========
    __aicore__ __inline__ void zero(int32_t timing_idx)
    {
        if (timing_idx * 2 >= CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN) {
            return;
        }
        int32_t adjusted_idx = calculate_index(timing_idx);
        if (adjusted_idx > 0 || timing_idx == 0) {
            AscendC::GlobalTensor<int64_t> timeTensor;
            timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
            timeTensor(adjusted_idx) = 0;       // 清空 start 槽位
            timeTensor(adjusted_idx + 1) = 0;   // 清空 end 槽位
        }
    }

    __aicore__ __inline__ void zero()
    {
        zero(CocTimer::KERNEL_TIMING_IDX);
    }

    __aicore__ __inline__ void zeroDynamicActualIterAll()
    {
        if (core_id < 0 || core_id >= CocTimer::N_CORE_COUNT) {
            return;
        }
        int32_t coreOffset = CocTimer::N_TIMING_COUNTER + core_id * CocTimer::DYNAMIC_ITER_PER_CORE_ALIGN;

        for (int type = 0; type < CocTimer::DYNAMIC_TYPE_COUNT; ++type) {
            int32_t idx = coreOffset + type;
            if (idx >= COC_TIMER_TOTAL_BUFFER_SIZE) break;

            AscendC::GlobalTensor<int64_t> iterTensor;
            iterTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
            iterTensor(idx) = 0;
        }
    }

    // ========== 8. 计时起点 (Tik) ==========
    __aicore__ __inline__ void Tik()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        start_cycle = AscendC::GetSystemCycle();
    }

    __aicore__ __inline__ void Tik(int64_t start)
    {
        start_cycle = start;
    }

    // 动态类型 Tik
    __aicore__ __inline__ void Tik(CocTimer::DynamicTimingType type)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t start_ts = AscendC::GetSystemCycle();

        int current_iter = iter_counters[(int)type];
        int timing_idx = CocTimer::FIXED_TIMING_COUNT + (int)type * CocTimer::MAX_DYNAMIC_ITER + current_iter;

        last_timing_idx = timing_idx;
        type_valid[(int)type] = true;

        int32_t adjusted_idx = calculate_index(timing_idx);
        if (adjusted_idx > 0 || timing_idx == 0) {
            AscendC::GlobalTensor<int64_t> timeTensor;
            timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
            timeTensor(adjusted_idx) = start_ts;
        }
        dynamic_start_cycles[(int)type] = start_ts;
    }

    // ========== 9. 计时终点 (Tok) ==========
    template <typename ModeTag = Accumulate>
    __aicore__ __inline__ void Tok(int32_t timing_idx)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();
        int64_t elapsed = end_cycle - start_cycle;

        int32_t adjusted_idx = calculate_index(timing_idx);

        if (adjusted_idx < 0 || (adjusted_idx == 0 && timing_idx != 0)) {
            return;
        }

        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);

        if constexpr (std::is_same_v<ModeTag, Accumulate>) {
            int64_t current_start = timeTensor(adjusted_idx);
            int64_t current_end = timeTensor(adjusted_idx + 1);

            if (current_start == 0 && current_end == 0) {
                timeTensor(adjusted_idx) = start_cycle;
                timeTensor(adjusted_idx + 1) = start_cycle + elapsed;
            } else {
                timeTensor(adjusted_idx + 1) = current_end + elapsed;
            }
        } else {
            timeTensor(adjusted_idx) = start_cycle;
            timeTensor(adjusted_idx + 1) = end_cycle;
        }
    }

    // 全局默认 Tok (映射到 KERNEL_TIMING)
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok()
    {
        Tok<ModeTag>(CocTimer::KERNEL_TIMING_IDX);
    }

    // 动态细粒度 Tok
    //  Overwrite  (默认): 每次迭代独立记录，CSV 中出现 TYPE_0, TYPE_1, TYPE_2, ...
    //  Accumulate:      所有迭代累加到 TYPE_0 一个 slot，写入负数 Iter 标识
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok(CocTimer::DynamicTimingType type)
    {
        if (!type_valid[(int)type]) return;

        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();
        int64_t elapsed = end_cycle - dynamic_start_cycles[(int)type];
        int current_iter = iter_counters[(int)type];

        if (current_iter >= CocTimer::MAX_DYNAMIC_ITER) {
            type_valid[(int)type] = false;
            return;
        }

        if constexpr (std::is_same_v<ModeTag, Accumulate>) {
            // ── Accumulate: 累加到迭代0的slot，写入负数代表累加模式 ──
            int timing_idx = CocTimer::FIXED_TIMING_COUNT + (int)type * CocTimer::MAX_DYNAMIC_ITER;
            int32_t adjusted_idx = calculate_index(timing_idx);
            if (adjusted_idx > 0 || timing_idx == 0) {
                AscendC::GlobalTensor<int64_t> timeTensor;
                timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
                if (current_iter == 0) {
                    timeTensor(adjusted_idx)     = 0;       // 强行把起点设为 0
                    timeTensor(adjusted_idx + 1) = elapsed; // 终点直接填本次耗时
                } else {
                    int64_t current_sum = timeTensor(adjusted_idx + 1);
                    timeTensor(adjusted_idx + 1) = current_sum + elapsed; // 持续累加纯耗时
                }
            }

            iter_counters[(int)type] = current_iter + 1;
            int32_t iter_idx = CocTimer::N_TIMING_COUNTER + core_id * CocTimer::DYNAMIC_ITER_PER_CORE_ALIGN + (int)type;
            if (iter_idx < COC_TIMER_TOTAL_BUFFER_SIZE) {
                AscendC::GlobalTensor<int64_t> iterTensor;
                iterTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
                iterTensor(iter_idx) = -(current_iter + 1);
            }
        } else {
            int timing_idx = CocTimer::FIXED_TIMING_COUNT + (int)type * CocTimer::MAX_DYNAMIC_ITER + current_iter;
            int32_t adjusted_idx = calculate_index(timing_idx);
            if (adjusted_idx > 0 || timing_idx == 0) {
                AscendC::GlobalTensor<int64_t> timeTensor;
                timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
                timeTensor(adjusted_idx + 1) = end_cycle;
            }

            iter_counters[(int)type] = current_iter + 1;
            int32_t iter_idx = CocTimer::N_TIMING_COUNTER + core_id * CocTimer::DYNAMIC_ITER_PER_CORE_ALIGN + (int)type;
            if (iter_idx < COC_TIMER_TOTAL_BUFFER_SIZE) {
                AscendC::GlobalTensor<int64_t> iterTensor;
                iterTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
                iterTensor(iter_idx) = current_iter + 1;
            }
        }

        type_valid[(int)type] = false;
    }

    // ========== 10. 调试信息 ==========
    __aicore__ __inline__ void print()
    {
        AscendC::printf("CocTimer: core_id=%d, block_id=%d, group_id=%d, subblockid=%d, t_addr=%p\n",
                        core_id, block_id, group_id, subblockid, t_addr);
    }

private:
    // ========== 11. 内部辅助函数 ==========

    // 计算当前核心在 Global Memory 中的实际偏移量
    __aicore__ __inline__ int32_t calculate_index(int32_t idx)
    {
        if (idx * 2 >= CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN) return 0;

        int32_t base_element_offset = 0;
        if ASCEND_IS_AIV
        {
            base_element_offset = group_id * CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 + CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * (1 + subblockid);
        }
        else
        {
            base_element_offset = group_id * CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3;
        }

        int32_t final_idx = base_element_offset + idx * 2;
        return (final_idx >= CocTimer::N_TIMING_COUNTER) ? 0 : final_idx;
    }

    __aicore__ __inline__ void accumulate_time(int32_t idx, int64_t time)
    {
        int32_t adjusted_idx = calculate_index(idx);
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        timeTensor(adjusted_idx) += time;
    }

    __aicore__ __inline__ void write_time(int32_t idx, int64_t time)
    {
        int32_t adjusted_idx = calculate_index(idx);
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        timeTensor(adjusted_idx) = time;
    }
};

#endif // COC_TIMER_DEVICE_H
