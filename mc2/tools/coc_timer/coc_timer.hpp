#ifndef COC_TIMER_COMMON_H
#define COC_TIMER_COMMON_H

#include <stdint.h>


namespace CocTimer
{
// ========== 1. 枚举定义（CPU/NPU共用） ==========
// 固定计时项
enum FixedTiming
{
    KERNEL_TIMING_IDX = 0,       // 整个 Kernel 的总耗时起点（通常是 kernel 入口）
    FIXED_TIMING_COUNT           // 哨兵值：固定计时项总数
};

enum DynamicTimingType
{
    AIC,
    AIV,
    DYNAMIC_TYPE_COUNT_ENUM,
    //继续添加....
};

// ========== 2. 公共常量定义（CPU/NPU共用） ==========
// 所有动态项默认最大迭代次数
// 注意：存储 start/end 分开，因此每个事件需要 2 个存储 slot
static constexpr int MAX_DYNAMIC_ITER = 150;

// CPU端：对齐相关常量（128字节(16元素)对齐，避免多核写一个cache line）
// 每核原始计数器数 = 固定项*2(START/END) + 动态类型*最大迭代*2(START/END)
static constexpr int N_TIMING_COUNTER_PER_CORE = FIXED_TIMING_COUNT * 2 + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER * 2;
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;                // 16倍对齐后每核计数器数
static constexpr int N_CORE_COUNT = 60;                                                                           // 核心总数 AIC+AIV核心数量 A2机器是这个数字 A3需要变化 可能也需要感知一下core_num参数
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;                           // 总基础计数器数

// 动态迭代次数存储区对齐配置（每核动态项存储区16元素对齐）
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;                         // 动态项类型总数
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;    // 每核动态迭代次数区16对齐
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT; // 动态迭代次数总存储区大小
}
// ========== 4. 计时开关宏 ==========
// #define ENABLE_TIMER 1 // 计时开关，因为计时会打乱流水，开关验证计时器对流水的影响。
#ifdef ENABLE_TIMER
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
        code ;            \
    } while (0)
#else
#define TIMER_BLOCK(code) \
    do                    \
    {                     \
    } while (0)
#endif

#endif // ASCEND_TIMER_COMMON_H