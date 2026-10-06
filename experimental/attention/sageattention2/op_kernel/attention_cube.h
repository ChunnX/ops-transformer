#pragma once

#include <cstdint>
#include <type_traits>
#include "static_slot.h"
#include "attention_sync.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"
#include "../sageattn_fp8/catlass/sage_qk_mmad.hpp"
#include "../sageattn_fp8/catlass/sage_pv_mmad.hpp"

namespace sage_attn {

/**
 * \brief 强类型分块参数结构
 * 明确区分 (rows, block_n, d)，彻底杜绝 (k, n) 传反错误。
 */
struct CubeBlockParams {
    uint32_t rows{0};     // M 维：TotalRows() = cur_heads * cur_q_len
    uint32_t block_n{0};  // N 维 (QK) / K 维 (PV)
    uint32_t d{0};        // head dim (d_pad)
};

/**
 * \brief 专职 Cube 侧 QK MMAD、PV MMAD 与 FixPipe 发射执行器
 *
 * 采用 CATLASS 拍计数守卫模型驱动的无阻塞流水：
 * - 矩阵乘与 Fixpipe 发射精准绑定 PIPE_MTE1 与 PIPE_FIX，杜绝标量等待；
 * - 槽位释放与交接严格对称，零动态 if (k >= 2) 分支。
 */
class AttentionCube {
public:
    template <uint8_t PV_CHAIN_CAP = 1>
    __aicore__ inline __attribute__((always_inline)) void Process(
        const TaskContext &ctx,
        const CubeBlockParams &params,
        uint32_t active_kv_blocks,
        uint32_t l1_q_offset)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::SageQkMmad<> qk_mmad;
        Catlass::SageAttn::SagePvMmad<> pv_mmad;

        // 1. 等待 V0 完成 Q 矩阵量化并常驻 L1 Bank 0
        AttentionSync::CubeWaitQReady();

        // 加载 Q 分块至 L0A: (rows, d)
        qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
        Catlass::SageAttn::Sync<HardEvent::MTE1_M>();

        auto StepCubeBlock = [&](uint32_t k, auto ping_tag, auto prev_ping_tag) __aicore__ {
            constexpr bool IS_PING = decltype(ping_tag)::value;
            constexpr bool PREV_IS_PING = decltype(prev_ping_tag)::value;
            constexpr uint32_t l1_kv_addr = IS_PING ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();
            constexpr uint32_t ub_dot32_offset = IS_PING ? SlotV1Dot32Ping::Offset() : SlotV1Dot32Pong::Offset();

            // 若 k >= 2，重新从 L1 Bank 0 加载 Q 分块（上一拍末尾的 PV MMAD 占用了 L0A）
            if (k >= 2) {
                qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
            }

            // 等待 V0 当前 KV 分块量化就绪
            AttentionSync::CubeWaitKvReady<IS_PING>();

            // 拍 k（Cube）：INT8 QK^T MMAD
            qk_mmad.LoadL0B(l1_kv_addr, params.d, params.block_n);
            Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
            qk_mmad.Compute(true);
            Catlass::SageAttn::Sync<HardEvent::M_FIX>();

            // QK dot32 双缓冲发射（k >= 2 时等待同槽位的前一轮释放）
            if (k >= 2) {
                AttentionSync::CubeWaitDotFree<IS_PING>();
            }
            qk_mmad.EmitL0CToUbRaw(ub_dot32_offset, params.rows, params.block_n);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
            // 立即向 Vector 1 发送 QK dot32 就绪信号，使 Vector 1 立即并行启动 Softmax 计算
            AttentionSync::CubeSignalQkReady<IS_PING>();

            // 若 k > 0，计算前一拍 k-1 的 PV MMAD 并写出 GM 工作区
            // 此时 Vector 1 并发计算拍 k 的 Softmax，Cube 并发计算拍 k-1 的 PV MMAD
            if (k > 0) {
                const uint32_t prev_k = k - 1;
                AttentionSync::CubeWaitSoftmaxReady<PREV_IS_PING>();
                StepPvMmadCore<PREV_IS_PING>(pv_mmad, params, true);
                AttentionSync::CubeReleaseSoftmaxFree<PREV_IS_PING>();
                AttentionSync::CubeReleaseKvFree<PREV_IS_PING>();
                Catlass::SageAttn::CopyL0CToGmSplitM<float>(
                    Catlass::SageAttn::HardwareResourceQuota::O_CHAIN_L0C_OFFSET,
                    ctx.CorePvWorkspaceSlot(prev_k, params.rows, params.d),
                    params.rows, params.d);
                Catlass::SageAttn::Sync<HardEvent::FIX_M>();
            }
        };

        uint32_t k = 0;
        for (; k + 1 < active_kv_blocks; k += 2) {
            StepCubeBlock(k, std::true_type{}, std::false_type{});
            StepCubeBlock(k + 1, std::false_type{}, std::true_type{});
        }
        if (k < active_kv_blocks) {
            StepCubeBlock(k, std::true_type{}, std::false_type{});
        }

        // 尾拍：收割最后一个分块的 PV MMAD 并写出 GM
        if (active_kv_blocks > 0) {
            const uint32_t last_k = active_kv_blocks - 1;
            if ((last_k & 1u) == 0u) {
                AttentionSync::CubeWaitSoftmaxReady<true>();
                StepPvMmadCore<true>(pv_mmad, params, true);
                AttentionSync::CubeReleaseSoftmaxFree<true>();
                AttentionSync::CubeReleaseKvFree<true>();
            } else {
                AttentionSync::CubeWaitSoftmaxReady<false>();
                StepPvMmadCore<false>(pv_mmad, params, true);
                AttentionSync::CubeReleaseSoftmaxFree<false>();
                AttentionSync::CubeReleaseKvFree<false>();
            }
            Catlass::SageAttn::CopyL0CToGmSplitM<float>(
                Catlass::SageAttn::HardwareResourceQuota::O_CHAIN_L0C_OFFSET,
                ctx.CorePvWorkspaceSlot(last_k, params.rows, params.d),
                params.rows, params.d);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
            AttentionSync::CubeSignalPvDone();

            // 尾拍回收：等待未等待的 dot32 释放信号，保证信道零残留
            if (active_kv_blocks == 1) {
                AttentionSync::CubeWaitDotFree<true>();
            } else {
                AttentionSync::CubeWaitDotFree<true>();
                AttentionSync::CubeWaitDotFree<false>();
            }
        }
#else
        (void)ctx;
        (void)params;
        (void)active_kv_blocks;
        (void)l1_q_offset;
#endif
#if SAGE_ATTN_DEVICE_COMPILE
        // 跨核任务同步屏障：等待 V0 与 V1 均完成当前任务工作
        AttentionSync::CubeWaitV0TaskDone();
        AttentionSync::CubeWaitV1TaskDone();

        // 向 V0 与 V1 发送任务退域确认，确保三管同频退出
        AttentionSync::CubeSignalV0TaskAck();
        AttentionSync::CubeSignalV1TaskAck();
#endif
    }

private:
#if SAGE_ATTN_DEVICE_COMPILE
    /**
     * \brief PV MMAD 裸指令核（Load L0A/L0B → Compute，L0C 结果在途）：
     * StepPvMmad 共用的最小指令序列；跨核同步（Softmax/Partial 握手）由各调用方按各自流水语义显式编排，核内零同步。
     */
    template <bool IS_PING>
    __aicore__ inline __attribute__((always_inline)) void StepPvMmadCore(
        Catlass::SageAttn::SagePvMmad<> &pv_mmad,
        const CubeBlockParams &params,
        bool initC)
    {
        constexpr uint32_t l1_v_addr = IS_PING ? SlotL1VPing::Offset() : SlotL1VPong::Offset();
        constexpr uint32_t s_slot = IS_PING ? SlotL1SoftmaxPPing::Offset() : SlotL1SoftmaxPPong::Offset();
        pv_mmad.LoadL0A(s_slot, params.rows, params.block_n);
        pv_mmad.LoadL0B(l1_v_addr, params.block_n, params.d);
        Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
        pv_mmad.Compute(initC);
        Catlass::SageAttn::Sync<HardEvent::M_FIX>();
    }
#endif
};

} // namespace sage_attn
