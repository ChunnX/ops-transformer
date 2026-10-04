#pragma once

#include <cstdint>
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/sync_guard.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"

namespace sage_attn {

/**
 * \brief 严格对称的 1C2V 硬件 Flag 双槽流水状态机
 * 采用刚刚落地的 [A1-1] 延迟释放逻辑：
 * Cube 在拍 k+1 读完 V 后才释放 FLAG_KV_FREE，彻底杜绝多 Block L1 Bank 提前覆写。
 */
class AttentionSync {
public:
    // =========================================================================
    // 1. Cube 侧同步原语
    // =========================================================================

    /**
     * \brief 等待 V0 完成 Q 矩阵量化并常驻 L1 Bank 0
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeWaitQReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::VEC0_TO_CUBE, CrossCoreFlags::FLAG_Q_READY);
#endif
    }

    /**
     * \brief 等待 V0 拍 k 的 KV 分块量化就绪
     * 集成 CATLASS CrossCoreFlagWithReverse 硬件队深 15 级防锁死安全防护：
     * 模 15 边界（(k + 1) % MAX_CROSS_CORE_SYNC_DEPTH == 0）向 V0 发送轻量反向确认 ACK，解除单向长流累积。
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeWaitKvReady(bool is_ping, uint32_t k)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t kv_flag = is_ping ? CrossCoreFlags::FLAG_KV_PING_READY : CrossCoreFlags::FLAG_KV_PONG_READY;
        CrossCoreWait(CrossCoreDirection::VEC0_TO_CUBE, kv_flag);
        // CATLASS CrossCoreFlagWithReverse 硬件队深 15 级反向确认握手
        if ((k + 1) % Catlass::SageAttn::HardwareResourceQuota::MAX_CROSS_CORE_SYNC_DEPTH == 0) {
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC0, CrossCoreFlags::FLAG_KV_REVERSE_ACK);
        }
#endif
    }

    /**
     * \brief 拍 k > 0 等待 V1 读毕上一拍 dot32 缓冲
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeWaitDotFree(uint32_t k)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        if (k > 0) {
            CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_DOT_FREE);
        }
#endif
    }

    /**
     * \brief FIX_S 自配对排空后，通知 V1 拍 k 的 QK P 矩阵/dot32 就绪
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeSignalQkReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreSet<PIPE_FIX>(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_QK_P_READY);
#endif
    }

    /**
     * \brief 等待 V1 上一拍 Softmax 概率矩阵 S 就绪
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeWaitSoftmaxReady(bool prev_is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t s_flag = prev_is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_READY
                                       : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_READY;
        CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, s_flag);
#endif
    }

    /**
     * \brief 归还 S[prev_k] 槽许可给 V1
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeReleaseSoftmaxFree(bool prev_is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t s_free = prev_is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE
                                       : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE;
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, s_free);
#endif
    }

    /**
     * \brief [A1-1] 延迟释放：Cube 在拍 k+1 读完 V 后才释放 FLAG_KV_FREE 给 V0
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeReleaseKvFree(bool prev_is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t prev_kv_free = prev_is_ping ? CrossCoreFlags::FLAG_KV_PING_FREE
                                             : CrossCoreFlags::FLAG_KV_PONG_FREE;
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC0, prev_kv_free);
#endif
    }

    /**
     * \brief 等待 V1 释放上一块 partial 交接槽
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeWaitPartialFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
#endif
    }

    /**
     * \brief FIX_S 自配对排空后，通知 V1 PV 部分和交接槽就绪
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeSignalPartialReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreSet<PIPE_FIX>(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_PARTIAL_READY);
#endif
    }

    /**
     * \brief 通知 V1 全部 PV MMAD 计算已完成
     */
    __aicore__ inline __attribute__((always_inline)) static void CubeSignalPvDone()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_O_DONE);
#endif
    }

    // =========================================================================
    // 2. Vector 0 侧同步原语
    // =========================================================================

    /**
     * \brief 通知 Cube：Q 量化常驻已在 L1 就位
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec0SignalQReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
        CrossCoreSet(CrossCoreDirection::VEC0_TO_CUBE, CrossCoreFlags::FLAG_Q_READY);
#endif
    }

    /**
     * \brief 缓冲回收握手：k >= 2 时等待 Cube 消费完上一轮同奇偶槽 (k-2)
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec0WaitKvFree(uint32_t k, bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        if (k >= 2) {
            uint32_t kv_free = is_ping ? CrossCoreFlags::FLAG_KV_PING_FREE : CrossCoreFlags::FLAG_KV_PONG_FREE;
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC0, kv_free);
        }
#endif
    }

    /**
     * \brief MTE3_MTE2 自配对排空后，通知 Cube 拍 k 的 KV 分块量化就绪
     * 集成 CATLASS CrossCoreFlagWithReverse 硬件队深 15 级防锁死安全防护：
     * 模 15 边界（(k + 1) % MAX_CROSS_CORE_SYNC_DEPTH == 0）等待 Cube 反向确认 ACK，解除单向长流累积。
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec0SignalKvReady(bool is_ping, uint32_t k)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::Sync<HardEvent::MTE3_MTE2>();
        Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
        uint32_t kv_flag = is_ping ? CrossCoreFlags::FLAG_KV_PING_READY : CrossCoreFlags::FLAG_KV_PONG_READY;
        CrossCoreSet(CrossCoreDirection::VEC0_TO_CUBE, kv_flag);
        // CATLASS CrossCoreFlagWithReverse 硬件队深 15 级反向确认握手
        if ((k + 1) % Catlass::SageAttn::HardwareResourceQuota::MAX_CROSS_CORE_SYNC_DEPTH == 0) {
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC0, CrossCoreFlags::FLAG_KV_REVERSE_ACK);
        }
#endif
    }

    /**
     * \brief 任务退域补等：消费 Cube 末尾释放的未用 KV 槽许可，保证跨任务 Flag 净余量恒为零
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec0DrainKvPingFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC0, CrossCoreFlags::FLAG_KV_PING_FREE);
#endif
    }

    __aicore__ inline __attribute__((always_inline)) static void Vec0DrainKvPongFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC0, CrossCoreFlags::FLAG_KV_PONG_FREE);
#endif
    }

    // =========================================================================
    // 3. Vector 1 侧同步原语
    // =========================================================================

    /**
     * \brief 等待 Cube PV 部分和就绪
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec1WaitPartialReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_PARTIAL_READY);
#endif
    }

    /**
     * \brief 释放 PV partial 交接槽许可给 Cube
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec1ReleasePartialFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::Sync<HardEvent::V_S>();
        CrossCoreSet(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
#endif
    }

    /**
     * \brief 等待 Cube 释放 S 槽许可
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec1WaitSoftmaxFree(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t s_free = is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE
                                  : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE;
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, s_free);
#endif
    }

    /**
     * \brief 等待 Cube 拍 k 的 QK P / dot32 矩阵块就绪
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec1WaitQkReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_QK_P_READY);
#endif
    }

    /**
     * \brief V_S 自配对排空后，释放 dot32 单缓冲读保护给 Cube
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec1ReleaseDotFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::Sync<HardEvent::V_S>();
        CrossCoreSet(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_DOT_FREE);
#endif
    }

    /**
     * \brief 通知 Cube 拍 k 的 Softmax 概率矩阵 S 就绪
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec1SignalSoftmaxReady(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::Sync<HardEvent::V_S>();
        uint32_t s_flag = is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_READY
                                  : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_READY;
        CrossCoreSet(CrossCoreDirection::VEC1_TO_CUBE, s_flag);
#endif
    }

    /**
     * \brief 等待 Cube 全部 PV 累加收尾完成
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec1WaitPvDone()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_O_DONE);
#endif
    }

    /**
     * \brief 退域补等：按需消费 Cube 释放的 S 槽许可，保证跨任务波次余量为零
     */
    __aicore__ inline __attribute__((always_inline)) static void Vec1DrainSoftmaxPingFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE);
#endif
    }

    __aicore__ inline __attribute__((always_inline)) static void Vec1DrainSoftmaxPongFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE);
#endif
    }
};

} // namespace sage_attn
