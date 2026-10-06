#pragma once

#include <cstdint>
#include "static_slot.h"

#if SAGE_ATTN_DEVICE_COMPILE
#include "catlass/arch/cross_core_sync.hpp"
#endif

namespace sage_attn {

/**
 * \brief 1C2V 硬件跨核同步标志位定义 (Ascend 950 / dav-3510)
 *
 * 硬件寻址铁律：
 * - AIV SubBlock 0 (Vector 0)：在 AIV 视角与 AIC 视角统一使用 Flag ID 0 ~ 15；
 * - AIV SubBlock 1 (Vector 1)：在 AIV 视角使用局部 Flag ID 0 ~ 15；
 *   在 AIC (Cube) 视角在硬件总线上固定偏移 16，使用 16 + Flag ID (16 ~ 31)。
 */
namespace SyncFlags {
    // Vector 0 (SubBlock 0) 通信信道 (0 ~ 15)
    constexpr uint32_t Q_READY          = 0;  // V0 Q 写入 L1 完成 (V0 set → Cube wait)
    constexpr uint32_t KV_PING_READY    = 1;  // V0 Ping KV 写入 L1 完成 (V0 set → Cube wait)
    constexpr uint32_t KV_PONG_READY    = 2;  // V0 Pong KV 写入 L1 完成 (V0 set → Cube wait)
    constexpr uint32_t KV_PING_FREE     = 3;  // L1 Ping KV 槽空闲许可 (Cube set → V0 wait)
    constexpr uint32_t KV_PONG_FREE     = 4;  // L1 Pong KV 槽空闲许可 (Cube set → V0 wait)
    constexpr uint32_t V0_TASK_DONE     = 5;  // V0 任务全部完成 (V0 set → Cube wait)
    constexpr uint32_t V0_TASK_ACK      = 6;  // Cube 任务结束确认 (Cube set → V0 wait)

    // Vector 1 (SubBlock 1) 通信信道 (V1 视角 0 ~ 15 / AIC 视角 16 + ID)
    // 严格避让硬件保留区 [11, 14]，全部使用 0 ~ 10
    constexpr uint32_t QK_PING_READY    = 0;  // QK dot32 Ping 写入 V1 UB 完成 (Cube set → V1 wait)
    constexpr uint32_t QK_PONG_READY    = 1;  // QK dot32 Pong 写入 V1 UB 完成 (Cube set → V1 wait)
    constexpr uint32_t S_PING_READY     = 2;  // V1 Ping S 写入 L1 完成 (V1 set → Cube wait)
    constexpr uint32_t S_PONG_READY     = 3;  // V1 Pong S 写入 L1 完成 (V1 set → Cube wait)
    constexpr uint32_t S_PING_FREE      = 4;  // L1 Ping S 槽空闲许可 (Cube set → V1 wait)
    constexpr uint32_t S_PONG_FREE      = 5;  // L1 Pong S 槽空闲许可 (Cube set → V1 wait)
    constexpr uint32_t DOT_PING_FREE    = 6;  // V1 dot32 Ping 读毕释放 (V1 set → Cube wait)
    constexpr uint32_t DOT_PONG_FREE    = 7;  // V1 dot32 Pong 读毕释放 (V1 set → Cube wait)
    constexpr uint32_t V1_TASK_ACK      = 8;  // Cube 任务结束确认 (Cube set → V1 wait)
    constexpr uint32_t PV_DONE          = 9;  // Cube PV 搬出 GM 完成 (Cube set → V1 wait)
    constexpr uint32_t V1_TASK_DONE     = 10; // V1 任务全量落盘完成 (V1 set → Cube wait)
}

/**
 * \brief CATLASS 官方拍计数守卫模型的 1C2V 硬件管道精准同步控制器
 *
 * 零预置零泄漏：首拍信用来自几何（双缓冲槽位天然为空），生产者在 k < 2（或 k == 0）
 * 直接写入，消费侧自 k >= 2 起等待释放信号（守卫为标量拍计数分支，官方 950 FAI 同款）；
 * 跨核事件精准绑定专用硬件执行管道 (PIPE_FIX / PIPE_MTE3 / PIPE_MTE1 / PIPE_V)，
 * 避开全局 generic PIPE_S 标量单元暂停死锁。Flag ID 全表严格小于 11（编译期断言锁定），
 * 避让 CANN 框架保留区 [11, 14]；V0/V1 分属独立 Bank（Cube 侧以 16 + ID 访问 Bank 1）。
 */
class AttentionSync {
public:
    static constexpr uint8_t MODE = 4; // 1C2V Intra-Block 模式

    static_assert(SyncFlags::Q_READY < 11, "SyncFlags::Q_READY must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::KV_PING_READY < 11, "SyncFlags::KV_PING_READY must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::KV_PONG_READY < 11, "SyncFlags::KV_PONG_READY must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::KV_PING_FREE < 11, "SyncFlags::KV_PING_FREE must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::KV_PONG_FREE < 11, "SyncFlags::KV_PONG_FREE must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::V0_TASK_DONE < 11, "SyncFlags::V0_TASK_DONE must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::V0_TASK_ACK < 11, "SyncFlags::V0_TASK_ACK must avoid hardware reserved IDs [11, 14]");

    static_assert(SyncFlags::QK_PING_READY < 11, "SyncFlags::QK_PING_READY must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::QK_PONG_READY < 11, "SyncFlags::QK_PONG_READY must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::S_PING_READY < 11, "SyncFlags::S_PING_READY must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::S_PONG_READY < 11, "SyncFlags::S_PONG_READY must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::S_PING_FREE < 11, "SyncFlags::S_PING_FREE must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::S_PONG_FREE < 11, "SyncFlags::S_PONG_FREE must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::DOT_PING_FREE < 11, "SyncFlags::DOT_PING_FREE must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::DOT_PONG_FREE < 11, "SyncFlags::DOT_PONG_FREE must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::V1_TASK_ACK < 11, "SyncFlags::V1_TASK_ACK must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::PV_DONE < 11, "SyncFlags::PV_DONE must avoid hardware reserved IDs [11, 14]");
    static_assert(SyncFlags::V1_TASK_DONE < 11, "SyncFlags::V1_TASK_DONE must avoid hardware reserved IDs [11, 14]");

    // -------------------------------------------------------------
    // 1. 终止退域同步屏障（仅 AIV 执行 MTE3 落盘屏障，AIC 严格禁 PipeBarrier<PIPE_ALL>）
    // -------------------------------------------------------------
    __aicore__ inline static void ReleaseSyncTokens()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        if ASCEND_IS_AIV {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
#endif
    }

    // -------------------------------------------------------------
    // 3. Vector 0 <-> Cube 硬件信道原语 (PIPE_MTE3 <-> PIPE_MTE1)
    // -------------------------------------------------------------
    __aicore__ inline static void V0SignalQReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE3>(SyncFlags::Q_READY);
#endif
    }

    __aicore__ inline static void CubeWaitQReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE1>(SyncFlags::Q_READY);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void V0WaitKvFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::KV_PING_FREE : SyncFlags::KV_PONG_FREE;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE3>(flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void V0SignalKvReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::KV_PING_READY : SyncFlags::KV_PONG_READY;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE3>(flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void CubeWaitKvReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::KV_PING_READY : SyncFlags::KV_PONG_READY;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE1>(flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void CubeReleaseKvFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::KV_PING_FREE : SyncFlags::KV_PONG_FREE;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(flag);
#endif
    }

    // -------------------------------------------------------------
    // 4. Cube <-> Vector 1 硬件信道原语 (PIPE_FIX <-> PIPE_V)
    // -------------------------------------------------------------
    template <bool IS_PING>
    __aicore__ inline static void CubeWaitDotFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::DOT_PING_FREE : SyncFlags::DOT_PONG_FREE;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreWaitFlag<MODE, PIPE_FIX>(16 + flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void CubeSignalQkReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::QK_PING_READY : SyncFlags::QK_PONG_READY;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreSetFlag<MODE, PIPE_FIX>(16 + flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void V1WaitQkReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::QK_PING_READY : SyncFlags::QK_PONG_READY;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreWaitFlag<MODE, PIPE_V>(flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void V1ReleaseDotFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::DOT_PING_FREE : SyncFlags::DOT_PONG_FREE;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreSetFlag<MODE, PIPE_V>(flag);
#endif
    }

    __aicore__ inline static void CubeSignalPvDone()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_FIX>(16 + SyncFlags::PV_DONE);
#endif
    }

    __aicore__ inline static void V1WaitPvDone()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_V>(SyncFlags::PV_DONE);
#endif
    }

    __aicore__ inline static void CubeWaitDotFree() { CubeWaitDotFree<true>(); }
    __aicore__ inline static void CubeSignalQkReady() { CubeSignalQkReady<true>(); }
    __aicore__ inline static void V1WaitQkReady() { V1WaitQkReady<true>(); }
    __aicore__ inline static void V1ReleaseDotFree() { V1ReleaseDotFree<true>(); }

    // -------------------------------------------------------------
    // 5. Vector 1 <-> Cube L1 Softmax S 硬件信道 (PIPE_MTE3 <-> PIPE_MTE1)
    // -------------------------------------------------------------
    template <bool IS_PING>
    __aicore__ inline static void V1WaitSoftmaxFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::S_PING_FREE : SyncFlags::S_PONG_FREE;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE3>(flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void V1SignalSoftmaxReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::S_PING_READY : SyncFlags::S_PONG_READY;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE3>(flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void CubeWaitSoftmaxReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::S_PING_READY : SyncFlags::S_PONG_READY;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE1>(16 + flag);
#endif
    }

    template <bool IS_PING>
    __aicore__ inline static void CubeReleaseSoftmaxFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t flag = IS_PING ? SyncFlags::S_PING_FREE : SyncFlags::S_PONG_FREE;
        static_assert(flag < 11, "SyncFlags must avoid hardware reserved IDs [11, 14]");
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(16 + flag);
#endif
    }

    // -------------------------------------------------------------
    // 8. 任务退域三管精准同步屏障 (MODE=4 Intra-Block)
    // -------------------------------------------------------------
    __aicore__ inline static void V0SignalTaskDone()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE3>(SyncFlags::V0_TASK_DONE);
#endif
    }

    __aicore__ inline static void V0WaitTaskAck()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE3>(SyncFlags::V0_TASK_ACK);
#endif
    }

    __aicore__ inline static void CubeWaitV0TaskDone()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE1>(SyncFlags::V0_TASK_DONE);
#endif
    }

    __aicore__ inline static void CubeSignalV0TaskAck()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(SyncFlags::V0_TASK_ACK);
#endif
    }

    __aicore__ inline static void V1SignalTaskDone()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_V>(SyncFlags::V1_TASK_DONE);
#endif
    }

    __aicore__ inline static void V1WaitTaskAck()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_V>(SyncFlags::V1_TASK_ACK);
#endif
    }

    __aicore__ inline static void CubeWaitV1TaskDone()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_FIX>(16 + SyncFlags::V1_TASK_DONE);
#endif
    }

    __aicore__ inline static void CubeSignalV1TaskAck()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_FIX>(16 + SyncFlags::V1_TASK_ACK);
#endif
    }
};

} // namespace sage_attn
