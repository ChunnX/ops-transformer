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
 */
class AttentionCube {
public:
    __aicore__ inline __attribute__((always_inline)) void Process(
        const CubeBlockParams &params,
        uint32_t active_kv_blocks,
        uint32_t l1_q_offset,
        uint32_t ub_softmax_offset,
        uint32_t ub_dot32_offset,
        uint32_t ub_pv_partial_offset)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::SageQkMmad<> qk_mmad;
        Catlass::SageAttn::SagePvMmad<> pv_mmad;

        // 进域预置 S 槽双缓冲许可
        AttentionSync::CubeInitPipeline();

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
            // 提至拍头立即发射 PartialReady 给 V1，彻底打破与 DotFree 的交织依赖
            if (k > 0) {
                uint32_t prev_k = k - 1;
                bool prev_is_ping = ((prev_k % 2) == 0);
                uint32_t prev_l1_v_addr = (prev_is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset())
                                          + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
                uint32_t s_slot = ub_softmax_offset + (prev_is_ping ? 0u : params.rows * params.block_n);

                AttentionSync::CubeWaitSoftmaxReady(prev_is_ping);

                // PV MMAD 契约：C[rows×d] = P[rows×block_n] × V[block_n×d]
                pv_mmad.LoadL0A(s_slot, params.rows, params.block_n);
                pv_mmad.LoadL0B(prev_l1_v_addr, params.block_n, params.d);
                Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
                pv_mmad.Compute(true);
                Catlass::SageAttn::Sync<HardEvent::M_FIX>();

                // S 槽消费完成：归还缓冲许可给 V1
                AttentionSync::CubeReleaseSoftmaxFree(prev_is_ping);

                // [A1-1] 延迟释放：prev_k 块的 K 与 V 均已在 L0 消费完毕，释放其 L1 KV 槽许可给 V0
                AttentionSync::CubeReleaseKvFree(prev_is_ping);

                // PV 部分和裸搬 UB 交接槽
                AttentionSync::CubeWaitPartialFree(k);
                pv_mmad.ExecuteEpilogue(ub_pv_partial_offset);
                Catlass::SageAttn::Sync<HardEvent::FIX_M>();
                AttentionSync::CubeSignalPartialReady();
            }

            // 2. 等待 V0 当前 KV 分块量化就绪
            AttentionSync::CubeWaitKvReady(is_ping);

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
            uint32_t last_l1_v_addr = (last_is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset())
                                      + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
            uint32_t s_slot = ub_softmax_offset + (last_is_ping ? 0u : params.rows * params.block_n);

            AttentionSync::CubeWaitSoftmaxReady(last_is_ping);

            pv_mmad.LoadL0A(s_slot, params.rows, params.block_n);
            pv_mmad.LoadL0B(last_l1_v_addr, params.block_n, params.d);
            Catlass::SageAttn::Sync<HardEvent::MTE1_M>();
            pv_mmad.Compute(true);
            Catlass::SageAttn::Sync<HardEvent::M_FIX>();

            AttentionSync::CubeReleaseSoftmaxFree(last_is_ping);
            // [A1-1] 最后一个分块 last_k 的 V 消费完毕，归还其 L1 KV 槽许可
            AttentionSync::CubeReleaseKvFree(last_is_ping);

            if (active_kv_blocks > 1) {
                AttentionSync::CubeWaitPartialFree(active_kv_blocks);
            }
            pv_mmad.ExecuteEpilogue(ub_pv_partial_offset);
            Catlass::SageAttn::Sync<HardEvent::FIX_M>();
            AttentionSync::CubeSignalPartialReady();
        }
#else
        (void)params;
        (void)active_kv_blocks;
        (void)l1_q_offset;
        (void)ub_softmax_offset;
        (void)ub_dot32_offset;
        (void)ub_pv_partial_offset;
#endif
        // 通知 V1：整个 Attention 的 PV 累加已全部完成
        AttentionSync::CubeSignalPvDone();
    }
};

} // namespace sage_attn
