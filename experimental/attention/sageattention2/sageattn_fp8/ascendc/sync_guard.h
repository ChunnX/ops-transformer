/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
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
#include <utility>

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

namespace sage_attn {

/**
 * 硬件管线标识（对应 DaVinci 硬件微架构）
 */
enum class PipeId : uint32_t {
    PIPE_MTE1 = 0,  // L1 到 L0 搬运管线 (Cube 侧数据加载)
    PIPE_MTE2 = 1,  // GM 到片上 (L1/UB) 搬入管线
    PIPE_MTE3 = 2,  // 片上到 GM 搬出管线
    PIPE_M    = 3,  // MMAD 矩阵乘计算管线 (Cube)
    PIPE_V    = 4,  // 向量计算管线 (Vector SIMD/SIMT)
    PIPE_FIX  = 5,  // FixPipe 搬运与随路量化/排布管线
    PIPE_S    = 6,  // 标量计算管线
    PIPE_ALL  = 7   // 全流水线屏障
};

/**
 * 片上流水 BufferID 映射（白皮书 4.1.6 节互斥锁语义缓冲标识）
 */
namespace BufferIds {
    // Vector 核心 UB 缓冲 ID
    constexpr uint32_t VEC_RAW_KV_PING          = 0;
    constexpr uint32_t VEC_RAW_KV_PONG          = 1;
    constexpr uint32_t VEC_QUANT_WORKSPACE      = 2;
    constexpr uint32_t VEC_NZ_KV_PING           = 3;
    constexpr uint32_t VEC_NZ_KV_PONG           = 4;
    constexpr uint32_t VEC_SOFTMAX_INTERMEDIATE = 5;
    constexpr uint32_t VEC_O_ACCUM              = 6;

    // Cube 核心 L0 缓冲 ID
    constexpr uint32_t CUBE_L0A_Q               = 0;
    constexpr uint32_t CUBE_L0B_KV_PING         = 1;
    constexpr uint32_t CUBE_L0B_KV_PONG         = 2;
    constexpr uint32_t CUBE_L0C_ACC             = 3;
}

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

namespace CrossCoreFlags {
    constexpr uint32_t FLAG_Q_READY         = 0;  // Q 量化常驻已就位
    constexpr uint32_t FLAG_KV_PING_READY   = 1;  // Ping 块 KV 量化就位
    constexpr uint32_t FLAG_KV_PONG_READY   = 2;  // Pong 块 KV 量化就位
    constexpr uint32_t FLAG_QK_P_READY      = 3;  // QK 矩阵乘完成，P 块就绪
    constexpr uint32_t FLAG_SOFTMAX_S_READY = 4;  // Softmax 完成，S 块就绪
    constexpr uint32_t FLAG_SKIP_SCALE_SIG  = 5;  // Skip-scale 跳步缩放信号
    constexpr uint32_t FLAG_PV_O_DONE       = 6;  // PV 累加完成
}

/**
 * 硬件级 BufferID 同步封装 (get_buf / rel_buf)
 * 白皮书 4.1.6 节：互斥锁语义，内聚性优于 set_flag/wait_flag，避免管线饥饿
 */
__aicore__ inline void GetBuffer(PipeId pipe, uint32_t buf_id, bool is_wait = false) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    // CANN 设备侧内置硬件 intrinsic 调用
    ::get_buf(static_cast<uint32_t>(pipe), buf_id, is_wait);
#else
    // Host 侧测试与静态校验桩
    (void)pipe;
    (void)buf_id;
    (void)is_wait;
#endif
}

__aicore__ inline void RelBuffer(PipeId pipe, uint32_t buf_id, bool is_wait = false) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    // CANN 设备侧内置硬件 intrinsic 调用
    ::rls_buf(static_cast<uint32_t>(pipe), buf_id, is_wait);
#else
    // Host 侧测试与静态校验桩
    (void)pipe;
    (void)buf_id;
    (void)is_wait;
#endif
}

/**
 * BufferID RAII 守卫：保证在作用域退出时自动安全释放 buffer，根除死锁
 */
class ScopedBufferGuard {
public:
    __aicore__ ScopedBufferGuard(PipeId pipe, uint32_t buf_id, bool is_wait = false)
        : pipe_(pipe), buf_id_(buf_id), is_wait_(is_wait), is_active_(true) {
        GetBuffer(pipe_, buf_id_, is_wait_);
    }

    __aicore__ ~ScopedBufferGuard() {
        if (is_active_) {
            RelBuffer(pipe_, buf_id_, is_wait_);
        }
    }

    __aicore__ void ReleaseEarly() {
        if (is_active_) {
            RelBuffer(pipe_, buf_id_, is_wait_);
            is_active_ = false;
        }
    }

    ScopedBufferGuard(const ScopedBufferGuard&) = delete;
    ScopedBufferGuard& operator=(const ScopedBufferGuard&) = delete;

    __aicore__ ScopedBufferGuard(ScopedBufferGuard&& other) noexcept
        : pipe_(other.pipe_), buf_id_(other.buf_id_), is_wait_(other.is_wait_), is_active_(other.is_active_) {
        other.is_active_ = false;
    }

private:
    PipeId pipe_;
    uint32_t buf_id_;
    bool is_wait_;
    bool is_active_;
};

/**
 * 跨核同步原语封装 (1C2V 硬件跨核信号灯)
 */
__aicore__ inline void CrossCoreSet(CrossCoreDirection dir, uint32_t flag_id) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    // 映射到硬件跨核寄存器或 cross_core_sync 指令
    ::CrossCoreSetFlag(static_cast<uint32_t>(dir), flag_id);
#else
    (void)dir;
    (void)flag_id;
#endif
}

__aicore__ inline void CrossCoreWait(CrossCoreDirection dir, uint32_t flag_id) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    ::CrossCoreWaitFlag(static_cast<uint32_t>(dir), flag_id);
#else
    (void)dir;
    (void)flag_id;
#endif
}

/**
 * 跨核信号 RAII 守卫：保证在析构时完成触发通知
 */
class ScopedCrossCoreSignalGuard {
public:
    __aicore__ ScopedCrossCoreSignalGuard(CrossCoreDirection dir, uint32_t flag_id)
        : dir_(dir), flag_id_(flag_id), is_active_(true) {}

    __aicore__ ~ScopedCrossCoreSignalGuard() {
        if (is_active_) {
            CrossCoreSet(dir_, flag_id_);
        }
    }

    __aicore__ void Cancel() {
        is_active_ = false;
    }

    ScopedCrossCoreSignalGuard(const ScopedCrossCoreSignalGuard&) = delete;
    ScopedCrossCoreSignalGuard& operator=(const ScopedCrossCoreSignalGuard&) = delete;

private:
    CrossCoreDirection dir_;
    uint32_t flag_id_;
    bool is_active_;
};

/**
 * 1:1 配对防护 RAII 守卫 (遵从 cannbot-skills 算子红线准则三与案例十一：生产者在前、消费者在后)
 * 严格执行 SetFlag / WaitFlag 1:1 配对，杜绝流水线挂死
 */
template <PipeId SrcPipe, PipeId DstPipe>
class ScopedPipeEventGuard {
public:
    __aicore__ explicit ScopedPipeEventGuard(uint32_t event_id)
        : event_id_(event_id), is_set_(false) {}

    __aicore__ void Signal() {
        if (!is_set_) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
            ::SetFlag<static_cast<uint32_t>(SrcPipe), static_cast<uint32_t>(DstPipe)>(event_id_);
#endif
            is_set_ = true;
        }
    }

    __aicore__ void Wait() {
        if (is_set_) {
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
            ::WaitFlag<static_cast<uint32_t>(SrcPipe), static_cast<uint32_t>(DstPipe)>(event_id_);
#endif
            is_set_ = false;
        }
    }

    __aicore__ ~ScopedPipeEventGuard() {
        // 如果已 Set 但尚未 Wait，在作用域退出时强行消费或断言，防止硬件死锁
        if (is_set_) {
            Wait();
        }
    }

    ScopedPipeEventGuard(const ScopedPipeEventGuard&) = delete;
    ScopedPipeEventGuard& operator=(const ScopedPipeEventGuard&) = delete;

private:
    uint32_t event_id_;
    bool is_set_;
};

/**
 * 便捷同步保护宏
 */
#define SAGE_GUARD_BUFFER(pipe, buf_id) \
    ::sage_attn::ScopedBufferGuard _buf_guard_##buf_id(pipe, buf_id, false)

#define SAGE_GUARD_BUFFER_WAIT(pipe, buf_id) \
    ::sage_attn::ScopedBufferGuard _buf_guard_##buf_id(pipe, buf_id, true)

#define SAGE_GUARD_CROSS_SIGNAL(dir, flag_id) \
    ::sage_attn::ScopedCrossCoreSignalGuard _cross_guard_##flag_id(dir, flag_id)

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_SYNC_GUARD_H
