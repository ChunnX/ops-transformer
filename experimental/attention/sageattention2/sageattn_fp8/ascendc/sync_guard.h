#ifndef SAGE_ATTN_FP8_ASCENDC_SYNC_GUARD_H
#define SAGE_ATTN_FP8_ASCENDC_SYNC_GUARD_H
#pragma once

#include <cstdint>
#include "tiling.h"

/**
 * bisheng -xasc 设备编译标记（与 online_softmax.h 同一约定，此处独立成守卫以保持头文件自洽）。
 */


#if SAGE_ATTN_KERNEL_API
#include "kernel_operator.h"
// 官方 CATLASS 跨核同步组件（catlass/arch/cross_core_sync.hpp）：CrossCoreFlag
// 类型化旗标 + CrossCoreSetFlag/WaitFlag 薄封装（forceinline 直译 AscendC 原语，
// 指令流与裸调用一致）。宿主门禁（BiSheng 宿主模式，KERNEL_API=0）不引入。
#include "catlass/arch/cross_core_sync.hpp"
#endif

namespace sage_attn {

/**
 * 跨核同步方向与信号量定义 (1C2V 架构)
 */
enum class CrossCoreDirection : uint32_t {
    CUBE_TO_VEC0 = 0,  // Cube 核心 -> Vector 0 (量化 prologue 核)
    CUBE_TO_VEC1 = 1,  // Cube 核心 -> Vector 1 (softmax 核)
    VEC0_TO_CUBE = 2,  // Vector 0 -> Cube 核心 (K/V 就位)
    VEC1_TO_CUBE = 3   // Vector 1 -> Cube 核心 (softmax 结果 S 槽就绪/缓冲归还信号)
};

/**
 * 跨核 flag 语义表（生产者在前、消费者在后；task/02_code.md B-2 按旗标族分治）：
 * S 槽 FREE 跨任务波次进域预置/退域回收配平；KV/DOT/PARTIAL FREE 首拍缓冲区天然
 * 为空即许可，生产者跳过首拍等待、不预置不回收（避免跨核伪预置首拍死锁）。
 * 生命周期 = 单个任务（DecodeTask 一次解码到 Finalize 输出），
 * 逐 flag 配平见 op_kernel/sageattention2.cpp 各 Process 入口与出口。
 */
namespace CrossCoreFlags {
    constexpr uint32_t FLAG_Q_READY              = 0;  // Q 量化常驻已就位 (V0 set → Cube wait)
    constexpr uint32_t FLAG_KV_PING_READY        = 1;  // Ping 块 KV 量化就位 (V0 set → Cube wait)
    constexpr uint32_t FLAG_KV_PONG_READY        = 2;  // Pong 块 KV 量化就位 (V0 set → Cube wait)
    constexpr uint32_t FLAG_QK_P_READY           = 3;  // QK 矩阵乘完成，P 块就绪 (Cube set → V1 wait)
    constexpr uint32_t FLAG_SOFTMAX_S_PING_READY = 4;  // Ping 槽 S 就绪 (V1 set → Cube wait)
    constexpr uint32_t FLAG_SOFTMAX_S_PONG_READY = 5;  // Pong 槽 S 就绪 (V1 set → Cube wait)
    constexpr uint32_t FLAG_PV_O_DONE            = 6;  // PV 累加与退休队列清空完成 (Cube set → V1 wait)
    // 缓冲回收握手：消费者消费完成后向生产者归还缓冲许可，杜绝深流水下未消费块被覆写
    constexpr uint32_t FLAG_KV_PING_FREE         = 7;  // Ping 槽 KV 已消费，V0 可覆写 (Cube set → V0 wait)
    constexpr uint32_t FLAG_KV_PONG_FREE         = 8;  // Pong 槽 KV 已消费，V0 可覆写 (Cube set → V0 wait)
    constexpr uint32_t FLAG_SOFTMAX_S_PING_FREE  = 9;  // Ping 槽 S 已消费，V1 可覆写 (Cube set → V1 wait)
    constexpr uint32_t FLAG_SOFTMAX_S_PONG_FREE  = 10; // Pong 槽 S 已消费，V1 可覆写 (Cube set → V1 wait)
    // A7 PV 部分和交接（task/12_device_data_path.md 第三节；2026-10-03；R17-2 后交接槽
    // 定址 V1 UB A7_PV_PARTIAL 区 [160K,224K)，Vector1UbBudget 唯一定义点）：
    // Cube 每块 PV L0C → UB 交接槽后 set，V1 合并进 O 主累加器后 set FREE
    constexpr uint32_t FLAG_PV_PARTIAL_READY     = 11; // 块 PV 部分和已落 UB 交接槽 (Cube set → V1 wait)
    constexpr uint32_t FLAG_PV_PARTIAL_FREE      = 12; // 交接槽已合并，Cube 可覆写 (V1 set → Cube wait)
    // A7 QK dot32 单缓冲读保护：V1 完成 block k 反量化读取后 set，Cube 方可在
    // beat k+1 覆写 dot32 区（单 L0C 要求 QK fp 先于 PV Mmad，无法借道 S(k) 排序；
    // R17-2 后 dot32 区定址 [96K,160K)，Vector1UbBudget 唯一定义点）
    constexpr uint32_t FLAG_DOT_FREE             = 13; // dot32 区已读毕，Cube 可覆写 (V1 set → Cube wait)
}

/**
 * 跨核同步原语封装 (1C2V 硬件跨核信号灯)
 *
 * 委托官方 CATLASS 组件（catlass/arch/cross_core_sync.hpp，一处定义）：
 *   Catlass::Arch::CrossCoreSetFlag<modeId, pipe>(Catlass::Arch::CrossCoreFlag&)
 *   Catlass::Arch::CrossCoreWaitFlag<modeId, pipe>(Catlass::Arch::CrossCoreFlag&)
 * modeId 4 = 块内 (intra-block) 同步：
 * 在 Atlas 950PR 1C2V 硬件中，Cube 与 Vector 间 intra-block flag 分为两半：
 * Vector 0 对应 0~15，Vector 1 对应 16~31。AIV 执行时硬件做局部映射，
 * Cube (AIC) 在与 Vector 1 同步时必须将 flag_id 加上 16。
 * PIPE_S 为全局作用域常量，不可加名字空间限定。
 */
__aicore__ inline __attribute__((always_inline)) void CrossCoreSet(CrossCoreDirection dir, uint32_t flag_id) {
#if SAGE_ATTN_KERNEL_API
    uint32_t hw_flag_id = flag_id;
    if ASCEND_IS_AIC {
        if (dir == CrossCoreDirection::CUBE_TO_VEC1 || dir == CrossCoreDirection::VEC1_TO_CUBE) {
            hw_flag_id += 16;
        }
    }
    Catlass::Arch::CrossCoreFlag flag{static_cast<Catlass::Arch::FlagID>(hw_flag_id)};
    Catlass::Arch::CrossCoreSetFlag<4, PIPE_S>(flag);
#else
    (void)dir;
    (void)flag_id;
#endif
}

__aicore__ inline __attribute__((always_inline)) void CrossCoreWait(CrossCoreDirection dir, uint32_t flag_id) {
#if SAGE_ATTN_KERNEL_API
    uint32_t hw_flag_id = flag_id;
    if ASCEND_IS_AIC {
        if (dir == CrossCoreDirection::CUBE_TO_VEC1 || dir == CrossCoreDirection::VEC1_TO_CUBE) {
            hw_flag_id += 16;
        }
    }
    Catlass::Arch::CrossCoreFlag flag{static_cast<Catlass::Arch::FlagID>(hw_flag_id)};
    Catlass::Arch::CrossCoreWaitFlag<4, PIPE_S>(flag);
#else
    (void)dir;
    (void)flag_id;
#endif
}

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_SYNC_GUARD_H
