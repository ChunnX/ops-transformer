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

    // Vector 1 (SubBlock 1) 通信信道 (V1 视角 0 ~ 15 / AIC 视角 16 + ID)
    constexpr uint32_t QK_READY         = 5;  // QK dot32 写入 V1 UB 完成 (Cube set → V1 wait)
    constexpr uint32_t DOT_FREE         = 6;  // V1 dot32 读毕释放 (V1 set → Cube wait)
    constexpr uint32_t S_PING_READY     = 7;  // V1 Ping S 写入 L1 完成 (V1 set → Cube wait)
    constexpr uint32_t S_PONG_READY     = 8;  // V1 Pong S 写入 L1 完成 (V1 set → Cube wait)
    constexpr uint32_t S_PING_FREE      = 9;  // L1 Ping S 槽空闲许可 (Cube set → V1 wait)
    constexpr uint32_t S_PONG_FREE      = 10; // L1 Pong S 槽空闲许可 (Cube set → V1 wait)
    constexpr uint32_t PV_PARTIAL_READY = 11; // PV 部分和写入 V1 UB 完成 (Cube set → V1 wait)
    constexpr uint32_t PV_PARTIAL_FREE  = 12; // V1 部分和累加完毕释放 (V1 set → Cube wait)
    constexpr uint32_t PV_DONE          = 13; // Cube 全部 PV 运算完成 (Cube set → V1 wait)
}

/**
 * \brief 基于 CATLASS Token 预置模型的 1C2V 硬件管道精准同步控制器
 *
 * 彻底消除运行时 if (k >= 2) / if (k > 0) 分支判断与脆弱退域补等；
 * 跨核事件精准绑定专用硬件执行管道 (PIPE_FIX / PIPE_MTE3 / PIPE_MTE1 / PIPE_V)，
 * 彻底杜绝全局 generic PIPE_S 标量单元暂停死锁。
 */
class AttentionSync {
public:
    static constexpr uint8_t MODE = 4; // 1C2V Intra-Block 模式

    // -------------------------------------------------------------
    // 1. 初始化预置初始 Token（Token Pre-setting in Init）
    // -------------------------------------------------------------
    __aicore__ inline static void InitSyncTokens()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        if ASCEND_IS_AIC {
            // Cube 预置 L1 KV 槽空闲 Token 给 Vector 0 (Ping & Pong)
            AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(SyncFlags::KV_PING_FREE);
            AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(SyncFlags::KV_PONG_FREE);
            // Cube 预置 L1 S 槽空闲 Token 给 Vector 1 (SubBlock 1 对应 16 + flag)
            AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(16 + SyncFlags::S_PING_FREE);
            AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(16 + SyncFlags::S_PONG_FREE);
        } else {
            if (AscendC::GetSubBlockIdx() == 1) {
                // Vector 1 预置 UB 接收槽空闲 Token 给 Cube
                AscendC::CrossCoreSetFlag<MODE, PIPE_V>(SyncFlags::DOT_FREE);
                AscendC::CrossCoreSetFlag<MODE, PIPE_V>(SyncFlags::PV_PARTIAL_FREE);
            }
        }
#endif
    }

    // -------------------------------------------------------------
    // 2. 终止退域同步屏障
    // -------------------------------------------------------------
    __aicore__ inline static void ReleaseSyncTokens()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::PipeBarrier<PIPE_ALL>();
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

    __aicore__ inline static void V0WaitKvFree(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t flag = is_ping ? SyncFlags::KV_PING_FREE : SyncFlags::KV_PONG_FREE;
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE3>(flag);
#else
        (void)is_ping;
#endif
    }

    __aicore__ inline static void V0SignalKvReady(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t flag = is_ping ? SyncFlags::KV_PING_READY : SyncFlags::KV_PONG_READY;
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE3>(flag);
#else
        (void)is_ping;
#endif
    }

    __aicore__ inline static void CubeWaitKvReady(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t flag = is_ping ? SyncFlags::KV_PING_READY : SyncFlags::KV_PONG_READY;
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE1>(flag);
#else
        (void)is_ping;
#endif
    }

    __aicore__ inline static void CubeReleaseKvFree(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t flag = is_ping ? SyncFlags::KV_PING_FREE : SyncFlags::KV_PONG_FREE;
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(flag);
#else
        (void)is_ping;
#endif
    }

    // -------------------------------------------------------------
    // 4. Cube <-> Vector 1 硬件信道原语 (PIPE_FIX <-> PIPE_V)
    // -------------------------------------------------------------
    __aicore__ inline static void CubeWaitDotFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_FIX>(16 + SyncFlags::DOT_FREE);
#endif
    }

    __aicore__ inline static void CubeSignalQkReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_FIX>(16 + SyncFlags::QK_READY);
#endif
    }

    __aicore__ inline static void V1WaitQkReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_V>(SyncFlags::QK_READY);
#endif
    }

    __aicore__ inline static void V1ReleaseDotFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_V>(SyncFlags::DOT_FREE);
#endif
    }

    // -------------------------------------------------------------
    // 5. Vector 1 <-> Cube L1 Softmax S 硬件信道 (PIPE_MTE3 <-> PIPE_MTE1)
    // -------------------------------------------------------------
    __aicore__ inline static void V1WaitSoftmaxFree(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t flag = is_ping ? SyncFlags::S_PING_FREE : SyncFlags::S_PONG_FREE;
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE3>(flag);
#else
        (void)is_ping;
#endif
    }

    __aicore__ inline static void V1SignalSoftmaxReady(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t flag = is_ping ? SyncFlags::S_PING_READY : SyncFlags::S_PONG_READY;
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE3>(flag);
#else
        (void)is_ping;
#endif
    }

    __aicore__ inline static void CubeWaitSoftmaxReady(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t flag = is_ping ? SyncFlags::S_PING_READY : SyncFlags::S_PONG_READY;
        AscendC::CrossCoreWaitFlag<MODE, PIPE_MTE1>(16 + flag);
#else
        (void)is_ping;
#endif
    }

    __aicore__ inline static void CubeReleaseSoftmaxFree(bool is_ping)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        uint32_t flag = is_ping ? SyncFlags::S_PING_FREE : SyncFlags::S_PONG_FREE;
        AscendC::CrossCoreSetFlag<MODE, PIPE_MTE1>(16 + flag);
#else
        (void)is_ping;
#endif
    }

    // -------------------------------------------------------------
    // 6. Cube <-> Vector 1 PV Partial 交接信道 (PIPE_FIX <-> PIPE_V)
    // -------------------------------------------------------------
    __aicore__ inline static void CubeWaitPartialFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_FIX>(16 + SyncFlags::PV_PARTIAL_FREE);
#endif
    }

    __aicore__ inline static void CubeSignalPartialReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_FIX>(16 + SyncFlags::PV_PARTIAL_READY);
#endif
    }

    __aicore__ inline static void V1WaitPartialReady()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreWaitFlag<MODE, PIPE_V>(SyncFlags::PV_PARTIAL_READY);
#endif
    }

    __aicore__ inline static void V1ReleasePartialFree()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::CrossCoreSetFlag<MODE, PIPE_V>(SyncFlags::PV_PARTIAL_FREE);
#endif
    }

    // -------------------------------------------------------------
    // 7. Cube -> Vector 1 任务收尾握手 (PIPE_FIX -> PIPE_V)
    // -------------------------------------------------------------
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
};

} // namespace sage_attn
