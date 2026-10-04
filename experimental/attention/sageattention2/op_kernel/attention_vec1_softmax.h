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

#if SAGE_ATTN_DEVICE_COMPILE
    /**
     * \brief A7 scale 条带桥：加载 Q scale 与 V scale
     */
    __aicore__ inline __attribute__((always_inline)) void StageV1ScaleStrips(const TaskContext &ctx)
    {
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
     * \brief A7 scale 条带桥：加载第 k 块的 K scale
     */
    __aicore__ inline __attribute__((always_inline)) void StageKScaleStrip(const TaskContext &ctx, uint32_t k)
    {
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
    /**
     * \brief 宿主串行仿真重摆：恢复设备分支“V0 产出→消费”的逐拍序（幂等）
     */
    __aicore__ inline __attribute__((always_inline)) void RestageKvBlockHost(const TaskContext &ctx, uint32_t k)
    {
        if (k >= ctx.active_kv_blocks) {
            return;
        }
        const uint32_t cur_block_k = ctx.CurBlockK(k);
        const bool is_ping = ((k % 2) == 0);
        const uint32_t l1_dst_offset = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();

        const int64_t b_idx = static_cast<int64_t>(ctx.cur_b);
        const int64_t hkv_idx = static_cast<int64_t>(ctx.cur_hkv);
        const int64_t sk_start = static_cast<int64_t>(k * ctx.block_n);

        const Catlass::SageAttn::GmBhsdLayout kv_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(ctx.b, ctx.hkv, ctx.sk, ctx.d, ctx.tiling->tensor_layout == LAYOUT_HND);
        const int64_t kv_gm_offset = kv_layout(tla::MakeCoord(b_idx, hkv_idx, sk_start, 0u));
        const __gm__ float *km_ptr = nullptr;
        if (ctx.tiling->smooth_k != 0) {
            km_ptr = reinterpret_cast<const __gm__ float *>(ctx.km_self);
        }

        AttentionVec0Quant<DTYPE_MODE> vec0;
        vec0.QuantizeAndStageKey(ctx, ctx.key_gm + kv_gm_offset * sizeof(uint16_t), km_ptr, l1_dst_offset,
                                 k, cur_block_k, kv_layout);
        vec0.QuantizeAndStageValue(ctx, ctx.value_gm + kv_gm_offset * sizeof(uint16_t),
                                   l1_dst_offset + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET,
                                   k, cur_block_k, kv_layout);
    }

    __aicore__ inline __attribute__((always_inline)) void ComputeBlockMaxAndSum(
        const TaskContext &ctx,
        uint32_t row, float &cur_max, float &cur_sum, uint32_t k)
    {
        const bool per_thread = (ctx.tiling->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const uint32_t g = row / ctx.cur_q_len;
        const uint32_t token_q = ctx.cur_q_start + (row % ctx.cur_q_len);
        const float q_scale =
            ctx.sim_mem->q_scales[(ctx.cur_g0 + g) * kQScaleSlotStride + quant::QScaleIndex(token_q, per_thread)];
        float *logits_row = &ctx.sim_mem->logits[static_cast<size_t>(row) * ctx.block_n];
        const int8_t *l1_q = reinterpret_cast<const int8_t *>(
            ctx.sim_mem->l1_mem + ctx.tiling->l1_q_offset);
        const bool is_ping = ((k % 2) == 0);
        const uint32_t cur_kv_base = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();
        const int8_t *l1_k = reinterpret_cast<const int8_t *>(
            ctx.sim_mem->l1_mem + cur_kv_base);
        const uint32_t cur_k_start = k * ctx.block_n;

        for (uint32_t j = 0; j < ctx.block_n; ++j) {
            const uint32_t token_k = cur_k_start + j;
            if (token_k >= ctx.sk || (ctx.tiling->is_causal != 0 &&
                                      static_cast<int64_t>(token_k) > static_cast<int64_t>(token_q))) {
                logits_row[j] = SAGE_MASK_SENTINEL;
                continue;
            }
            const int8_t *qr = l1_q + static_cast<size_t>(row) * ctx.d;
            const int8_t *kr = l1_k + static_cast<size_t>(j) * ctx.d;
            int32_t dot = 0;
            for (uint32_t c = 0; c < ctx.d; ++c) {
                dot += qr[c] * kr[c];
            }
            logits_row[j] = twin::DequantQkLogit(
                dot, q_scale,
                ctx.sim_mem->k_scales[quant::KScaleIndex(token_k, per_thread)],
                ctx.tiling->sm_scale_log2e);
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
        cur_max = row_max;
        cur_sum = twin::OnlineBlockExp2Sum(logits_row, cur_k_start,
                                           cur_k_start, cur_k_start + ctx.block_n,
                                           ctx.tiling->is_causal != 0, token_q, row_max);
    }

    __aicore__ inline __attribute__((always_inline)) void RescaleAccumulator(
        const TaskContext &ctx,
        float rescale_factor, uint32_t row)
    {
        float *o_row = &ctx.sim_mem->o_accum[static_cast<size_t>(row) * ctx.d];
        for (uint32_t c = 0; c < ctx.d; ++c) {
            o_row[c] *= rescale_factor;
        }
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
    __aicore__ inline __attribute__((always_inline)) void ProcessBlock(
        const TaskContext &ctx,
        uint32_t k,
        bool is_ping)
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
        LocalTensor<float> ub_logits_dot(TPosition::VECCALC, SlotV1Dot32::Offset(), rows * ctx.block_n);
        LocalTensor<float> ub_o(TPosition::VECCALC, SlotV1OAccum::Offset(), rows * ctx.d);
        LocalTensor<float8_e4m3_t> ub_out_p(TPosition::VECCALC, SlotV1FixpipeRecv::Offset(), rows * ctx.block_n);

        dev_vf::DequantMaskLogitsRowsVF(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()), rows,
                                        reinterpret_cast<__ubuf__ float*>(ub_qs.GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float*>(ub_ks.GetPhyAddr()),
                                        ctx.tiling->sm_scale_log2e, ctx.cur_q_start, ctx.cur_q_len,
                                        cur_k_start, ctx.sk, ctx.tiling->is_causal, ctx.block_n);
        dev_vf::BlockMaxAndSumRowsVF(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()), rows,
                                     ctx.block_n,
                                     reinterpret_cast<__ubuf__ float*>(ub_block_max.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(ub_block_sum.GetPhyAddr()));

        dev_vf::UpdateOnlineSoftmaxRowsVF(reinterpret_cast<__ubuf__ float*>(ub_running_max.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_running_sum.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_block_max.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_block_sum.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(ub_factor.GetPhyAddr()),
                                          rows, k);

        if (k > 0) {
            dev_vf::RescaleAccumRowsVF(reinterpret_cast<__ubuf__ float*>(ub_o.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float*>(ub_factor.GetPhyAddr()),
                                       rows, ctx.d);
        }

        dev_vf::SoftmaxProbRowsVF(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_out_p.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(ub_running_max.GetPhyAddr()),
                                  rows, ctx.block_n);

        // 通过片上 SSBUF 高速通路将 FP8 P 矩阵搬入 L1 Bank 0 专有双缓冲槽，供 Cube PV MMAD 读取
        Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
        const uint32_t l1_p_addr = is_ping ? SlotL1SoftmaxPPing::Offset() : SlotL1SoftmaxPPong::Offset();
        LocalTensor<float8_e4m3_t> l1_p(TPosition::A1, l1_p_addr, rows * ctx.block_n);
        Catlass::SageAttn::CopyUbToL1(l1_p, ub_out_p, rows * ctx.block_n);
        Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
#else
        RestageKvBlockHost(ctx, k);

        for (uint32_t r = 0; r < ctx.TotalRows() && r < MAX_PACK_ROWS; ++r) {
            float cur_max = MIN_FLOAT_VAL;
            float cur_sum = 0.0f;
            ComputeBlockMaxAndSum(ctx, r, cur_max, cur_sum, k);
            if (cur_max <= SAGE_MASK_SENTINEL) {
                continue;
            }
            const float rescale_factor =
                OnlineSoftmaxTransformer::Step(row_states_[r], cur_max, cur_sum);
            if (row_states_[r].need_rescale &&
                SageFastAbs(rescale_factor - 1.0f) > SCALE_ONE_TOLERANCE) {
                RescaleAccumulator(ctx, rescale_factor, r);
            }
        }

        ComputeSoftmaxProbability(ctx, is_ping);
#endif
    }

private:
    OnlineSoftmaxState row_states_[MAX_PACK_ROWS];
};

} // namespace sage_attn
