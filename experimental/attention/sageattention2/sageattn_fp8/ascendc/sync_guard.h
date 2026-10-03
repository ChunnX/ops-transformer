/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SAGE_ATTN_FP8_ASCENDC_SYNC_GUARD_H
#define SAGE_ATTN_FP8_ASCENDC_SYNC_GUARD_H

#include <cstdint>

/**
 * bisheng -xasc 设备编译标记（与 online_softmax.h 同一约定，此处独立成守卫以保持头文件自洽）。
 */
#ifndef SAGE_ATTN_DEVICE_COMPILE
#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
#define SAGE_ATTN_DEVICE_COMPILE 1
#else
#define SAGE_ATTN_DEVICE_COMPILE 0
#endif
#endif

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

#if SAGE_ATTN_DEVICE_COMPILE || defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#include "kernel_operator.h"
#endif

namespace sage_attn {

/**
 * 跨核同步方向与信号量定义 (1C2V 架构)
 */
enum class CrossCoreDirection : uint32_t {
    CUBE_TO_VEC0 = 0,  // Cube 核心 -> Vector 0 (量化 prologue 核)
    CUBE_TO_VEC1 = 1,  // Cube 核心 -> Vector 1 (softmax 核)
    VEC0_TO_CUBE = 2,  // Vector 0 -> Cube 核心 (K/V 就位)
    VEC1_TO_CUBE = 3,  // Vector 1 -> Cube 核心 (softmax 结果与 skip-scale 信号)
    VEC0_TO_VEC1 = 4,  // Vector 0 -> Vector 1
    VEC1_TO_VEC0 = 5   // Vector 1 -> Vector 0
};

/**
 * 跨核 flag 语义表（每个 flag 的 set/wait 均须 1:1 配对，符合同步规范）
 * 生命周期 = 单个任务（DecodeTask 一次解码到 Finalize 输出），跨任务波次的
 * 进域预置 / 退域回收纪律见 op_kernel/sageattention2.cpp 各 Process 入口与出口。
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
    // A7 PV 部分和交接（task/12_device_data_path.md 第三节；2026-10-03）：
    // Cube 每块 PV L0C → UB 交接槽（TWO_LEVEL_REDUCE 区）后 set，V1 合并进 O 主累加器后 set FREE
    constexpr uint32_t FLAG_PV_PARTIAL_READY     = 11; // 块 PV 部分和已落 UB 交接槽 (Cube set → V1 wait)
    constexpr uint32_t FLAG_PV_PARTIAL_FREE      = 12; // 交接槽已合并，Cube 可覆写 (V1 set → Cube wait)
    // A7 QK dot32 单缓冲读保护：V1 完成 block k 反量化读取后 set，Cube 方可在
    // beat k+1 覆写 dot32 区（单 L0C 要求 QK fp 先于 PV Mmad，无法借道 S(k) 排序）
    constexpr uint32_t FLAG_DOT_FREE             = 13; // dot32 区已读毕，Cube 可覆写 (V1 set → Cube wait)
}

/**
 * 跨核同步原语封装 (1C2V 硬件跨核信号灯)
 *
 * 设备端映射 CANN 官方接口（dav-3510 头文件实测签名）：
 *   CrossCoreSetFlag<uint8_t modeId, pipe_t pipe>(uint16_t flagId)
 *   CrossCoreWaitFlag<uint8_t modeId = 0, pipe_t pipe = PIPE_S>(uint16_t flagId)
 * modeId 4 = 块内 (intra-block) 同步：支持全部 flagId [0,15]；
 * 1C2V 的 Cube/V0/V1 同属一个 AI Core，块内信号灯语义精确匹配。
 * PIPE_S 为全局作用域常量，不可加名字空间限定。
 */
__aicore__ inline void CrossCoreSet(CrossCoreDirection dir, uint32_t flag_id) {
#if SAGE_ATTN_DEVICE_COMPILE || defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    (void)dir;
    AscendC::CrossCoreSetFlag<4, PIPE_S>(static_cast<uint16_t>(flag_id));
#else
    (void)dir;
    (void)flag_id;
#endif
}

__aicore__ inline void CrossCoreWait(CrossCoreDirection dir, uint32_t flag_id) {
#if SAGE_ATTN_DEVICE_COMPILE || defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    (void)dir;
    AscendC::CrossCoreWaitFlag<4, PIPE_S>(static_cast<uint16_t>(flag_id));
#else
    (void)dir;
    (void)flag_id;
#endif
}

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_SYNC_GUARD_H
