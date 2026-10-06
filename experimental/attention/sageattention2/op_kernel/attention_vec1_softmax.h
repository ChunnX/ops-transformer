#pragma once

#include <cstdint>
#include "static_slot.h"
#include "attention_sync.h"
#include "sage_device_vf.h"
#include "attention_vec0_quant.h"
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/online_softmax.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"

#if !SAGE_ATTN_DEVICE_COMPILE
#include "twin_reference.h"
#include "../quant/quant_common.h"
#endif

namespace sage_attn {

/**
 * \brief 专职 Vector 1 侧 Logit 反量化、MicroAPI StoreUnAlign 非对齐流式刷写、Online Softmax 与 FP8 铸模
 */
template <uint32_t DTYPE_MODE>
class AttentionVec1Softmax {
public:
    static constexpr uint32_t kDtype = NormDtypeMode(DTYPE_MODE);

    __aicore__ inline __attribute__((always_inline)) void ResetRowStates(uint32_t rows)
    {
        for (uint32_t r = 0; r < rows && r < MAX_PACK_ROWS; ++r) {
            row_states_[r].Reset();
#if !SAGE_ATTN_DEVICE_COMPILE
            rescale_factors_[r] = 1.0f;
#endif
        }
    }

    __aicore__ inline __attribute__((always_inline)) const OnlineSoftmaxState *GetRowStates() const
    {
        return row_states_;
    }

    __aicore__ inline __attribute__((always_inline)) OnlineSoftmaxState *GetRowStates()
    {
        return row_states_;
    }

#if !SAGE_ATTN_DEVICE_COMPILE
    __aicore__ inline __attribute__((always_inline)) const float *GetRescaleFactors() const
    {
        return rescale_factors_;
    }
#endif

#if SAGE_ATTN_DEVICE_COMPILE
    /**
     * \brief A7 scale 条带桥：加载 Q scale 与 V scale (片外 L2 Cache 暂存条带)
     */
    __aicore__ inline __attribute__((always_inline)) void StageV1ScaleStrips(const TaskContext &ctx)
    {
        Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
        __gm__ uint8_t *strip = ctx.StageStrip();
        GlobalTensor<float> gm_strip;
        LocalTensor<float> ub_qs(TPosition::VECCALC, SlotV1QsRow::Offset(), STAGE_SLOT_FLOATS);
        LocalTensor<float> ub_vs(TPosition::VECCALC, SlotV1VsChan::Offset(), STAGE_SLOT_FLOATS);
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(strip + STAGE_Q_SCALE_OFFSET));
        DataCopy(ub_qs, gm_strip, STAGE_SLOT_FLOATS);
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(strip + STAGE_V_SCALE_OFFSET));
        DataCopy(ub_vs, gm_strip, STAGE_SLOT_FLOATS);
        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
    }

    /**
     * \brief A7 scale 条带桥：加载第 k 块的 K scale (片外 L2 Cache 暂存条带)
     */
    __aicore__ inline __attribute__((always_inline)) void StageKScaleStrip(const TaskContext &ctx, uint32_t k)
    {
        Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
        __gm__ uint8_t *strip = ctx.StageStrip();
        GlobalTensor<float> gm_strip;
        LocalTensor<float> ub_ks(TPosition::VECCALC, SlotV1KsRow::Offset(), STAGE_SLOT_FLOATS);
        uint32_t k_offset = STAGE_K_SCALE_BASE_OFFSET + (k % STAGE_K_SCALE_NUM_SLOTS) * STAGE_SLOT_BYTES;
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(strip + k_offset));
        DataCopy(ub_ks, gm_strip, ctx.block_n);
        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
    }
#endif

#if !SAGE_ATTN_DEVICE_COMPILE
    __aicore__ inline __attribute__((always_inline)) void StageV1ScaleStrips(const TaskContext &ctx) { (void)ctx; }
    __aicore__ inline __attribute__((always_inline)) void StageKScaleStrip(const TaskContext &ctx, uint32_t k) { (void)ctx; (void)k; }

    /**
     * \brief 宿主串行仿真重摆：恢复设备分支“V0 产出→消费”的逐拍序（幂等，与设备 can_reuse_kv 快路分支成对修改）
     */
    __aicore__ inline __attribute__((always_inline)) void RestageKvBlockHost(const TaskContext &ctx, uint32_t k)
    {
        if (k >= ctx.active_kv_blocks) {
            return;
        }
        const uint32_t cur_block_k = ctx.CurBlockK(k);
        const bool is_ping = ((k % 2) == 0);
        const uint32_t l1_dst_offset = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();
        const uint32_t l1_v_offset = is_ping ? SlotL1VPing::Offset() : SlotL1VPong::Offset();

        const uint8_t *core_ws = ctx.CoreKvWorkspacePtr();
        if (core_ws != nullptr) {
            const uint32_t d_pad_val = ctx.tiling ? ctx.tiling->d_pad : ctx.d;
            const KvPreQuantWorkspaceOffsets ws = ComputeKvPreQuantCoreOffsets(ctx.sk, d_pad_val);
            const uint64_t tile_bytes = static_cast<uint64_t>(ctx.block_n) * ctx.d;
            const uint64_t k_block_offset = ws.key_offset + static_cast<uint64_t>(k) * tile_bytes;
            const uint64_t v_block_offset = ws.val_offset + static_cast<uint64_t>(k) * tile_bytes;
            const uint64_t ks_block_offset = ws.key_scale_offset + static_cast<uint64_t>(k) * STAGE_SLOT_BYTES;

            const float *ks_src = reinterpret_cast<const float*>(core_ws + ks_block_offset);
            const uint32_t cur_k_start = k * ctx.block_n;
            const bool per_thread = (ctx.qk_quant_gran == QUANT_GRAN_PER_THREAD);
            if (ctx.sim_mem != nullptr) {
                uint32_t k_offset = STAGE_K_SCALE_BASE_OFFSET + (k % STAGE_K_SCALE_NUM_SLOTS) * STAGE_SLOT_BYTES;
                float *stage_k = reinterpret_cast<float *>(ctx.sim_mem->stage_strip + k_offset);
                for (uint32_t j = 0; j < ctx.block_n; ++j) {
                    stage_k[j] = (j < cur_block_k) ? ks_src[j] : 0.0f;
                }
                for (uint32_t j = 0; j < cur_block_k; ++j) {
                    const uint32_t token = cur_k_start + j;
                    uint32_t s_idx = per_thread ? quant::KScaleIndex<true>(token) : quant::KScaleIndex<false>(token);
                    ctx.sim_mem->k_scales[s_idx] = ks_src[j];
                }
                int8_t *l1_k = reinterpret_cast<int8_t *>(ctx.sim_mem->l1_mem + l1_dst_offset);
                const int8_t *k_src_cached = reinterpret_cast<const int8_t*>(core_ws + k_block_offset);
                for (uint32_t j = 0; j < ctx.block_n; ++j) {
                    for (uint32_t c = 0; c < ctx.d; ++c) {
                        l1_k[Catlass::SageAttn::ZnIndex(j, c, ctx.block_n)] = k_src_cached[j * ctx.d + c];
                    }
                }
                uint8_t *l1_v = ctx.sim_mem->l1_mem + l1_v_offset;
                const uint8_t *v_src_cached = reinterpret_cast<const uint8_t*>(core_ws + v_block_offset);
                for (uint32_t j = 0; j < ctx.block_n; ++j) {
                    for (uint32_t c = 0; c < ctx.d; ++c) {
                        l1_v[Catlass::SageAttn::ZnIndex(j, c, ctx.block_n)] = v_src_cached[j * ctx.d + c];
                    }
                }
            }
            return;
        }

        const int64_t b_idx = static_cast<int64_t>(ctx.cur_b);
        const int64_t hkv_idx = static_cast<int64_t>(ctx.cur_hkv);
        const int64_t sk_start = static_cast<int64_t>(k * ctx.block_n);

        const Catlass::SageAttn::GmBhsdLayout kv_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(ctx.b, ctx.hkv, ctx.sk, ctx.d, ctx.is_hnd != 0);
        const int64_t kv_gm_offset = kv_layout(tla::MakeCoord(b_idx, hkv_idx, sk_start, 0u));
        const __gm__ float *km_ptr = nullptr;
        if (ctx.smooth_k != 0) {
            km_ptr = reinterpret_cast<const __gm__ float *>(ctx.km_self);
        }

        AttentionVec0Quant<DTYPE_MODE> vec0;
        vec0.LoadAndQuantizeK(ctx, ctx.key_gm + kv_gm_offset * sizeof(uint16_t), km_ptr,
                              k, l1_dst_offset);
        vec0.QuantizeAndStageValue(ctx, ctx.value_gm + kv_gm_offset * sizeof(uint16_t),
                                   is_ping ? SlotL1VPing::Offset() : SlotL1VPong::Offset(),
                                   k, cur_block_k, kv_layout);
    }

    __aicore__ inline __attribute__((always_inline)) void ComputeBlockMaxAndSum(
        const TaskContext &ctx,
        uint32_t row, float &cur_max, float &cur_sum, uint32_t k)
    {
        const bool per_thread = (ctx.qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const uint32_t g = row / ctx.cur_q_len;
        const uint32_t token_q = ctx.cur_q_start + (row % ctx.cur_q_len);
        const float q_scale =
            ctx.sim_mem->q_scales[(ctx.cur_g0 + g) * kQScaleSlotStride + quant::QScaleIndex(token_q, per_thread)];
        float *logits_row = &ctx.sim_mem->logits[static_cast<size_t>(row) * ctx.block_n];
        const int8_t *l1_q = reinterpret_cast<const int8_t *>(
            ctx.sim_mem->l1_mem + ctx.l1_q_offset);
        const bool is_ping = ((k % 2) == 0);
        const uint32_t cur_kv_base = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();
        const int8_t *l1_k = reinterpret_cast<const int8_t *>(
            ctx.sim_mem->l1_mem + cur_kv_base);
        const uint32_t cur_k_start = k * ctx.block_n;

        for (uint32_t j = 0; j < ctx.block_n; ++j) {
            const uint32_t token_k = cur_k_start + j;
            if (token_k >= ctx.sk || (ctx.is_causal != 0 &&
                                      static_cast<int64_t>(token_k) > static_cast<int64_t>(token_q))) {
                logits_row[j] = SAGE_MASK_SENTINEL;
                continue;
            }
            int32_t dot = 0;
            for (uint32_t c = 0; c < ctx.d; ++c) {
                const int8_t q_val = l1_q[Catlass::SageAttn::ZnIndex(row, c, ctx.TotalRows())];
                const int8_t k_val = l1_k[Catlass::SageAttn::ZnIndex(j, c, ctx.block_n)];
                dot += static_cast<int32_t>(q_val) * static_cast<int32_t>(k_val);
            }
            logits_row[j] = twin::DequantQkLogit(
                dot, q_scale,
                ctx.sim_mem->k_scales[quant::KScaleIndex(token_k, per_thread)],
                ctx.sm_scale_log2e);
        }

        float row_max = SAGE_MASK_SENTINEL;
        for (uint32_t j = 0; j < ctx.block_n; ++j) {
            if (logits_row[j] > row_max) row_max = logits_row[j];
        }
        if (row_max <= SAGE_MASK_SENTINEL) {
            cur_max = SAGE_MASK_SENTINEL;
            cur_sum = 0.0f;
            return;
        }
        cur_max = row_max - S_FP8_OFFSET;
        cur_sum = twin::OnlineBlockExp2Sum(logits_row, cur_k_start,
                                           cur_k_start, cur_k_start + ctx.block_n,
                                           ctx.is_causal != 0, token_q, cur_max);
    }

    __aicore__ inline __attribute__((always_inline)) void ComputeSoftmaxProbability(
        const TaskContext &ctx,
        bool is_ping)
    {
        uint8_t *slot_ptr = ctx.sim_mem->ub_mem + ctx.SoftmaxSlotOffset(is_ping);
        const uint32_t rows = ctx.TotalRows();
        for (uint32_t r = 0; r < rows && r < MAX_PACK_ROWS; ++r) {
            const float m_prime = row_states_[r].m_scale;
            const float *logits_row = &ctx.sim_mem->logits[static_cast<size_t>(r) * ctx.block_n];
            uint8_t *slot_row = slot_ptr + static_cast<size_t>(r) * ctx.block_n;
            for (uint32_t j = 0; j < ctx.block_n; ++j) {
                slot_row[j] = twin::SoftmaxProbCode(logits_row[j], m_prime);
            }
        }
    }
#endif

    /**
     * \brief 拍 k 的 Softmax 概率矩阵生成与行状态步进
     */
    template <bool IS_CAUSAL = false, bool IS_PING = true>
    __aicore__ inline __attribute__((always_inline)) void ProcessBlock(
        const TaskContext &ctx,
        uint32_t k)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        const uint32_t rows = ctx.TotalRows();
        const uint32_t cur_k_start = k * ctx.block_n;

        LocalTensor<float> ub_qs(TPosition::VECCALC, SlotV1QsRow::Offset(), STAGE_SLOT_FLOATS);
        LocalTensor<float> ub_ks(TPosition::VECCALC, SlotV1KsRow::Offset(), ctx.block_n);
        LocalTensor<float> ub_running_max(TPosition::VECCALC, SlotV1RowMax::Offset(), MAX_PACK_ROWS);
        LocalTensor<float> ub_running_sum(TPosition::VECCALC, SlotV1RowSum::Offset(), MAX_PACK_ROWS);
        LocalTensor<float> ub_block_max(TPosition::VECCALC, SlotV1BlockMax::Offset(), MAX_PACK_ROWS);
        LocalTensor<float> ub_block_sum(TPosition::VECCALC, SlotV1BlockSum::Offset(), MAX_PACK_ROWS);
        LocalTensor<float> ub_factor(TPosition::VECCALC, SlotV1RsFactor::Offset(), MAX_PACK_ROWS);
        constexpr uint32_t dot32_offset = IS_PING ? SlotV1Dot32Ping::Offset() : SlotV1Dot32Pong::Offset();
        LocalTensor<float> ub_logits_dot(TPosition::VECCALC, dot32_offset, rows * ctx.block_n);
        LocalTensor<float8_e4m3_t> ub_out_p(TPosition::VECCALC, SlotV1FixpipeRecv::Offset(), rows * ctx.block_n);

        // M6-3 Logits 瓦片寄存器流式融合：反量化 + 三态掩码 + 行内 Max + ExpSub 指数求和
        // 单遍行流水完成（原 DequantMaskLogitsRowsVF + BlockMaxAndSumRowsVF 两趟结构退役）。
        // 对 UB Logits 的 3 次重复加载消除为 1 次流式读取（Max/Sum 中间状态完全驻留向量
        // 寄存器），Logits 仅 1 次流式写出供下游 SoftmaxProbRowsVF 按 running max 铸 FP8；
        // chunk 顺序 Max 链与 MulDstAdd 累加链严格同序，位级语义与原两趟一致。
        dev_vf::DequantMaskMaxSumRowsVF<IS_CAUSAL>(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()), rows,
                                                   reinterpret_cast<__ubuf__ float*>(ub_qs.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ float*>(ub_ks.GetPhyAddr()),
                                                   ctx.sm_scale_log2e, ctx.cur_q_start, ctx.cur_q_len,
                                                   cur_k_start, ctx.sk, ctx.is_causal, ctx.block_n,
                                                   reinterpret_cast<__ubuf__ float*>(ub_block_max.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ float*>(ub_block_sum.GetPhyAddr()));

        dev_vf::UpdateOnlineSoftmaxRowsVF(reinterpret_cast<__ubuf__ float*>(ub_running_max.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_running_sum.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_block_max.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_block_sum.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_factor.GetPhyAddr()),
                                          rows, k);

        if (ctx.block_n <= 64u) {
            dev_vf::SoftmaxProbRowsVF<1u>(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_out_p.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_running_max.GetPhyAddr()),
                                          rows, ctx.block_n);
        } else {
            dev_vf::SoftmaxProbRowsVF<2u>(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_out_p.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_running_max.GetPhyAddr()),
                                          rows, ctx.block_n);
        }

        // 通过片上 SSBUF 高速通路将 FP8 P 矩阵搬入 L1 Bank 0 专有双缓冲槽，供 Cube PV MMAD 读取
        Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
        constexpr uint32_t l1_p_addr = IS_PING ? SlotL1SoftmaxPPing::Offset() : SlotL1SoftmaxPPong::Offset();
        LocalTensor<float8_e4m3_t> l1_p(TPosition::A1, l1_p_addr, rows * ctx.block_n);
        Catlass::SageAttn::CopyUbToL1zN(l1_p, ub_out_p, rows, ctx.block_n);
        Catlass::SageAttn::Sync<HardEvent::MTE3_V>();
        Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
#else
        RestageKvBlockHost(ctx, k);

        for (uint32_t r = 0; r < ctx.TotalRows() && r < MAX_PACK_ROWS; ++r) {
            float cur_max = MIN_FLOAT_VAL;
            float cur_sum = 0.0f;
            ComputeBlockMaxAndSum(ctx, r, cur_max, cur_sum, k);
            if (cur_max <= SAGE_MASK_SENTINEL) {
                rescale_factors_[r] = 1.0f;
                continue;
            }
            const float rescale_factor =
                OnlineSoftmaxTransformer::Step(row_states_[r], cur_max, cur_sum);
            rescale_factors_[r] = rescale_factor;
        }

        float *gm_alpha = ctx.CoreAlphaWorkspaceSlot(k, ctx.TotalRows());
        if (gm_alpha != nullptr) {
            for (uint32_t r = 0; r < MAX_PACK_ROWS; ++r) {
                gm_alpha[r] = (r < ctx.TotalRows()) ? rescale_factors_[r] : 1.0f;
            }
        }

        ComputeSoftmaxProbability(ctx, IS_PING);
#endif
    }

    template <bool IS_CAUSAL = false>
    __aicore__ inline __attribute__((always_inline)) void ProcessBlock(
        const TaskContext &ctx,
        uint32_t k,
        bool is_ping)
    {
        if (is_ping) {
            ProcessBlock<IS_CAUSAL, true>(ctx, k);
        } else {
            ProcessBlock<IS_CAUSAL, false>(ctx, k);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ProcessBlock(
        const TaskContext &ctx,
        uint32_t k,
        bool is_ping)
    {
        if (ctx.is_causal != 0) {
            ProcessBlock<true>(ctx, k, is_ping);
        } else {
            ProcessBlock<false>(ctx, k, is_ping);
        }
    }

private:
    OnlineSoftmaxState row_states_[MAX_PACK_ROWS];
#if !SAGE_ATTN_DEVICE_COMPILE
    float rescale_factors_[MAX_PACK_ROWS]{1.0f};
#endif
};

} // namespace sage_attn
