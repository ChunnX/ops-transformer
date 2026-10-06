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
 * 采用 CATLASS 拍计数守卫模型驱动的无阻塞流水：
 * - 矩阵乘与 Fixpipe 发射精准绑定 PIPE_MTE1 与 PIPE_FIX，杜绝标量等待；
 * - 槽位释放与交接严格对称，零动态 if (k >= 2) 分支。
 */
class AttentionCube {
public:
    __aicore__ inline __attribute__((always_inline)) void Process(
        const CubeBlockParams &params,
        uint32_t active_kv_blocks,
        uint32_t l1_q_offset,
        uint32_t ub_dot32_offset,
        uint32_t ub_pv_partial_offset,
        uint8_t pv_chain_cap = 1)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::SageQkMmad<> qk_mmad;
        Catlass::SageAttn::SagePvMmad<> pv_mmad;

        // 1. 等待 V0 完成 Q 矩阵量化并常驻 L1 Bank 0
        AttentionSync::CubeWaitQReady();

        // 加载 Q 分块至 L0A: (rows, d)
        qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
        Catlass::SageAttn::Sync<HardEvent::MTE1_M>();

        uint32_t chain_len = 0;
        bool prev_emitted = false;

        // 延迟一拍宽流水主循环
        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t l1_kv_addr = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();

            // 1. 拍 k（Cube 并发）：先发射上一拍的 FP8 PV MMAD 与交接（当 k > 0 时）
            if (k > 0) {
                uint32_t prev_k = k - 1;
                bool prev_is_ping = ((prev_k % 2) == 0);
                
                bool initC = (chain_len == 0);
                chain_len++;
                bool emit = (chain_len >= pv_chain_cap || k == active_kv_blocks);
                
                StepPvMmad(pv_mmad, params, prev_is_ping, ub_pv_partial_offset, prev_k > 0, prev_emitted, initC, emit);
                if (emit) {
                    chain_len = 0;
                }
                prev_emitted = emit;
            }

            // 2. 等待 V0 当前 KV 分块量化就绪
            AttentionSync::CubeWaitKvReady(is_ping);

            // 3. 拍 k（Cube）：INT8 QK^T MMAD
            // 若执行过 PV MMAD（k > 0），L0A 已被 Softmax P 覆写，需重新从 L1 Bank 0 加载常驻 Q
            if (k > 0) {
                qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
            }
            qk_mmad.LoadL0B(l1_kv_addr, params.d, params.block_n);
            Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
            qk_mmad.Compute(true);
            Catlass::SageAttn::Sync<HardEvent::M_FIX>();

            // 4. QK dot32 裸搬 UB（首拍 dot32 槽天然空，k >= 1 受 V1 读毕释放保护）
            if (k > 0) {
                AttentionSync::CubeWaitDotFree();
            }
            qk_mmad.EmitL0CToUbRaw(ub_dot32_offset, params.rows, params.block_n);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
            AttentionSync::CubeSignalQkReady();
        }

        // Drain 拍：完成最后一个 KV 分块的 PV MMAD 与部分和搬出
        if (active_kv_blocks > 0) {
            uint32_t last_k = active_kv_blocks - 1;
            bool last_is_ping = ((last_k % 2) == 0);
            
            bool initC = (chain_len == 0);
            chain_len++;
            bool emit = true; // drain always emits
            
            StepPvMmad(pv_mmad, params, last_is_ping, ub_pv_partial_offset, last_k > 0, prev_emitted, initC, emit);
            
            // 尾拍配平：回收 V1 释放的末拍 dot32 与 drain partial 信号，确保信道零遗留
            AttentionSync::CubeWaitDotFree();
            AttentionSync::CubeWaitPartialFree();
        }
#else
        (void)params;
        (void)active_kv_blocks;
        (void)l1_q_offset;
        (void)ub_dot32_offset;
        (void)ub_pv_partial_offset;
        (void)pv_chain_cap;
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
                StepPvMmadBare(pv_mmad, params, ((k - 1) % 2) == 0, ub_pv_partial_offset);
            }

            if (k > 0) {
                qk_mmad.LoadL0A(l1_q_offset, params.rows, params.d);
            }
            qk_mmad.LoadL0B(l1_kv_addr, params.d, params.block_n);
            Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
            qk_mmad.Compute(true);
            Catlass::SageAttn::Sync<HardEvent::M_FIX>();

            qk_mmad.EmitL0CToUbRaw(ub_dot32_offset, params.rows, params.block_n);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
        }

        if (active_kv_blocks > 0) {
            StepPvMmadBare(pv_mmad, params, ((active_kv_blocks - 1) % 2) == 0, ub_pv_partial_offset);
        }
#else
        (void)params;
        (void)active_kv_blocks;
        (void)l1_q_offset;
        (void)ub_dot32_offset;
        (void)ub_pv_partial_offset;
#endif
    }


private:
#if SAGE_ATTN_DEVICE_COMPILE
    /**
     * \brief PV MMAD 裸指令核（Load L0A/L0B → Compute，L0C 结果在途）：
     * StepPvMmad 与诊断探针 TestProcessStandalone 共用的最小序列；跨核同步
     * （Softmax/Partial 握手）由各调用方按各自流水语义显式编排，核内零同步。
     */
    __aicore__ inline __attribute__((always_inline)) void StepPvMmadCore(
        Catlass::SageAttn::SagePvMmad<> &pv_mmad,
        const CubeBlockParams &params,
        bool is_ping,
        bool initC)
    {
        uint32_t l1_v_addr = is_ping ? SlotL1VPing::Offset() : SlotL1VPong::Offset();
        uint32_t s_slot = is_ping ? SlotL1SoftmaxPPing::Offset() : SlotL1SoftmaxPPong::Offset();
        pv_mmad.LoadL0A(s_slot, params.rows, params.block_n);
        pv_mmad.LoadL0B(l1_v_addr, params.block_n, params.d);
        Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
        pv_mmad.Compute(initC);
        Catlass::SageAttn::Sync<HardEvent::M_FIX>();
    }

    __aicore__ inline __attribute__((always_inline)) void StepPvMmad(
        Catlass::SageAttn::SagePvMmad<> &pv_mmad,
        const CubeBlockParams &params,
        bool is_ping,
        uint32_t ub_pv_partial_offset,
        bool prev_valid,
        bool prev_emitted,
        bool initC,
        bool emit)
    {
        AttentionSync::CubeWaitSoftmaxReady(is_ping);

        StepPvMmadCore(pv_mmad, params, is_ping, initC);

        AttentionSync::CubeReleaseSoftmaxFree(is_ping);
        AttentionSync::CubeReleaseKvFree(is_ping);

        if (emit) {
            if (prev_valid) {
                if (prev_emitted) {
                    AttentionSync::CubeWaitPartialFree();
                } else {
                    AttentionSync::CubeWaitPvChain();
                }
            }
            pv_mmad.ExecuteEpilogue(ub_pv_partial_offset);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
            AttentionSync::CubeSignalPartialReady();
        } else {
            // When not emitting, L0C holds the partial result. 
            // V1 must grant permission to proceed to avoid overwriting or getting out of sync.
            if (prev_valid) {
                if (prev_emitted) {
                    AttentionSync::CubeWaitPartialFree();
                } else {
                    AttentionSync::CubeWaitPvChain();
                }
            }
            // Even if not emitting, we might need a delay or just wait until next time we emit.
            // But wait, ExecuteEpilogue is what takes L0C to UB.
            // If we don't ExecuteEpilogue, we don't use UB, so we don't need PartialFree/PvChain for UB,
            // BUT we still need to wait for V1 to consume the *previous* partial if it emitted.
        }
    }

    /** 诊断探针专用：裸核 + 部分和搬出（无跨核握手，Stage 单核流水观测用）。 */
    __aicore__ inline __attribute__((always_inline)) void StepPvMmadBare(
        Catlass::SageAttn::SagePvMmad<> &pv_mmad,
        const CubeBlockParams &params,
        bool is_ping,
        uint32_t ub_pv_partial_offset)
    {
        StepPvMmadCore(pv_mmad, params, is_ping, true);
        pv_mmad.ExecuteEpilogue(ub_pv_partial_offset);
        Catlass::SageAttn::Sync<HardEvent::FIX_M>();
    }
#endif
};

} // namespace sage_attn
