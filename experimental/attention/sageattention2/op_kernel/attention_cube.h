#pragma once

#include <cstdint>
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
 * FixPipe 单目的地纪律：QK/PV 两路 L0C 产出统一经 CopyL0CToUbSplitM
 * 单目的地硬件直送 V1 UB（dualDstCtl=0 + subBlockId=true，
 * sage_tile.hpp:155-156）。官方 SPLIT_M 双目的地模式（dualDstCtl=1，
 * M/2 各写一个 SubCore 的 UB）与本核 1C2V 拓扑不兼容——V0 专职量化、
 * V1 是 Cube 产出的唯一消费者，双目的地会把半幅结果覆写进 V0 工作区；
 * 完整考证证据链见 sage_pv_mmad.hpp ExecuteEpilogue 文档块。双 SubCore
 * 吞吐匹配预期由 V0 量化 / V1 Softmax 职责分工结构性达成（msprof 待证）。
 */
class AttentionCube {
public:
    __aicore__ inline __attribute__((always_inline)) void Process(
        const CubeBlockParams &params,
        uint32_t active_kv_blocks,
        uint32_t l1_q_offset,
        uint32_t ub_dot32_offset,
        uint32_t ub_pv_partial_offset)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::SageQkMmad<> qk_mmad;
        Catlass::SageAttn::SagePvMmad<> pv_mmad;

        // 1. 等待 V0 完成 Q 矩阵量化并常驻 L1 Bank 0
        AttentionSync::CubeWaitQReady();

        // 加载 Q 分块至 L0A: (rows, d)
        qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
        Catlass::SageAttn::Sync<HardEvent::MTE1_M>();

        // 延迟一拍宽流水主循环
        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t l1_kv_addr = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();

            // 1. 拍 k（Cube 并发）：先发射上一拍的 FP8 PV MMAD 与交接（当 k > 0 时）
            if (k > 0) {
                uint32_t prev_k = k - 1;
                bool prev_is_ping = ((prev_k % 2) == 0);
                StepPvMmad(pv_mmad, params, prev_is_ping, ub_pv_partial_offset, k > 1);
            }

            // 2. 等待 V0 当前 KV 分块量化就绪（CATLASS 模 15 级防锁死安全防护）
            AttentionSync::CubeWaitKvReady(is_ping, k);

            // 3. 拍 k（Cube）：INT8 QK^T MMAD
            // 若执行过 PV MMAD（k > 0），L0A 已被 Softmax P 覆写，需重新从 L1 Bank 0 加载常驻 Q
            if (k > 0) {
                qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
            }
            qk_mmad.LoadL0B(l1_kv_addr, params.block_n, params.d);
            Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
            qk_mmad.Compute(true);
            Catlass::SageAttn::Sync<HardEvent::M_FIX>();

            // 4. QK dot32 裸搬 UB
            AttentionSync::CubeWaitDotFree(k);
            qk_mmad.EmitL0CToUbRaw(ub_dot32_offset, params.rows, params.block_n);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
            AttentionSync::CubeSignalQkReady();
        }

        // Drain 拍：完成最后一个 KV 分块的 PV MMAD 与部分和搬出
        if (active_kv_blocks > 0) {
            uint32_t last_k = active_kv_blocks - 1;
            bool last_is_ping = ((last_k % 2) == 0);
            StepPvMmad(pv_mmad, params, last_is_ping, ub_pv_partial_offset, active_kv_blocks > 1);
        }
#else
        (void)params;
        (void)active_kv_blocks;
        (void)l1_q_offset;
        (void)ub_dot32_offset;
        (void)ub_pv_partial_offset;
#endif
        // 通知 V1：整个 Attention 的 PV 累加已全部完成
        AttentionSync::CubeSignalPvDone();
    }

    __aicore__ inline __attribute__((always_inline)) void TestProcessStandalone(
        const CubeBlockParams &params,
        uint32_t active_kv_blocks,
        uint32_t l1_q_offset,
        uint32_t ub_dot32_offset,
        uint32_t ub_pv_partial_offset)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::SageQkMmad<> qk_mmad;
        Catlass::SageAttn::SagePvMmad<> pv_mmad;

        // 加载 Q 分块至 L0A: (rows, d)
        qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
        Catlass::SageAttn::Sync<HardEvent::MTE1_M>();

        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t l1_kv_addr = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();

            if (k > 0) {
                uint32_t prev_k = k - 1;
                bool prev_is_ping = ((prev_k % 2) == 0);
                uint32_t l1_v_addr = (prev_is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset())
                                     + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
                uint32_t s_slot = prev_is_ping ? SlotL1SoftmaxPPing::Offset() : SlotL1SoftmaxPPong::Offset();
                pv_mmad.LoadL0A(s_slot, params.rows, params.block_n);
                pv_mmad.LoadL0B(l1_v_addr, params.block_n, params.d);
                Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
                pv_mmad.Compute(true);
                Catlass::SageAttn::Sync<HardEvent::M_FIX>();
                pv_mmad.ExecuteEpilogue(ub_pv_partial_offset);
                Catlass::SageAttn::Sync<HardEvent::FIX_M>();
            }

            if (k > 0) {
                qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
            }
            qk_mmad.LoadL0B(l1_kv_addr, params.block_n, params.d);
            Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
            qk_mmad.Compute(true);
            Catlass::SageAttn::Sync<HardEvent::M_FIX>();

            qk_mmad.EmitL0CToUbRaw(ub_dot32_offset, params.rows, params.block_n);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
        }

        if (active_kv_blocks > 0) {
            uint32_t last_k = active_kv_blocks - 1;
            bool last_is_ping = ((last_k % 2) == 0);
            uint32_t l1_v_addr = (last_is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset())
                                 + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
            uint32_t s_slot = last_is_ping ? SlotL1SoftmaxPPing::Offset() : SlotL1SoftmaxPPong::Offset();
            pv_mmad.LoadL0A(s_slot, params.rows, params.block_n);
            pv_mmad.LoadL0B(l1_v_addr, params.block_n, params.d);
            Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
            pv_mmad.Compute(true);
            Catlass::SageAttn::Sync<HardEvent::M_FIX>();
            pv_mmad.ExecuteEpilogue(ub_pv_partial_offset);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
        }
#else
        (void)params;
        (void)active_kv_blocks;
        (void)l1_q_offset;
        (void)ub_dot32_offset;
        (void)ub_pv_partial_offset;
#endif
    }

    __aicore__ inline __attribute__((always_inline)) void TestProcessWithVec0(
        const CubeBlockParams &params,
        uint32_t active_kv_blocks,
        uint32_t l1_q_offset,
        uint32_t ub_dot32_offset,
        uint32_t ub_pv_partial_offset)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        (void)ub_dot32_offset;
        (void)ub_pv_partial_offset;
        Catlass::SageAttn::SageQkMmad<> qk_mmad;

        // 1. 等待 V0 完成 Q 量化
        AttentionSync::CubeWaitQReady();
        qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
        Catlass::SageAttn::Sync<HardEvent::MTE1_M>();

        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t l1_kv_addr = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();

            // 2. 等待 V0 完成 KV 量化
            AttentionSync::CubeWaitKvReady(is_ping, k);

            qk_mmad.LoadL0B(l1_kv_addr, params.block_n, params.d);
            Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
            qk_mmad.Compute(true);
            Catlass::SageAttn::Sync<HardEvent::M_FIX>();

            // 3. 释放 KV_FREE 给 V0（在最后一块或每块）
            AttentionSync::CubeReleaseKvFree(is_ping);
        }
#else
        (void)params; (void)active_kv_blocks; (void)l1_q_offset; (void)ub_dot32_offset; (void)ub_pv_partial_offset;
#endif
    }

private:
#if SAGE_ATTN_DEVICE_COMPILE
    __aicore__ inline __attribute__((always_inline)) void StepPvMmad(
        Catlass::SageAttn::SagePvMmad<> &pv_mmad,
        const CubeBlockParams &params,
        bool is_ping,
        uint32_t ub_pv_partial_offset,
        bool wait_partial_free)
    {
        uint32_t l1_v_addr = (is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset())
                             + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
        uint32_t s_slot = is_ping ? SlotL1SoftmaxPPing::Offset() : SlotL1SoftmaxPPong::Offset();

        AttentionSync::CubeWaitSoftmaxReady(is_ping);

        pv_mmad.LoadL0A(s_slot, params.rows, params.block_n);
        pv_mmad.LoadL0B(l1_v_addr, params.block_n, params.d);
        Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
        pv_mmad.Compute(true);
        Catlass::SageAttn::Sync<HardEvent::M_FIX>();

        AttentionSync::CubeReleaseSoftmaxFree(is_ping);
        AttentionSync::CubeReleaseKvFree(is_ping);

        if (wait_partial_free) {
            AttentionSync::CubeWaitPartialFree();
        }
        pv_mmad.ExecuteEpilogue(ub_pv_partial_offset);
        Catlass::SageAttn::Sync<HardEvent::FIX_M>();
        AttentionSync::CubeSignalPartialReady();
    }
#endif
};

} // namespace sage_attn
