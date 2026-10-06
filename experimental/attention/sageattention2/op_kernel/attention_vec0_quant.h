#pragma once

#include <cstdint>
#include "static_slot.h"
#include "attention_sync.h"
#include "sage_device_vf.h"
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../quant/quant_common.h"

#if !SAGE_ATTN_DEVICE_COMPILE
#include "twin_reference.h"
#endif

namespace sage_attn {

/**
 * \brief 专职 Vector 0 侧 GM 流式预取、K 减均值、INT8/FP8 量化与 SSBUF 流式发送
 * 严格遵循硬件物理总线隔离，严禁越权访问 Cube 私有通道。
 */
template <uint32_t DTYPE_MODE>
class AttentionVec0Quant {
public:
    static constexpr uint32_t kDtype = NormDtypeMode(DTYPE_MODE);
    static_assert(MAX_PACK_ROWS * 128u * sizeof(uint16_t) <= Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_BYTES,
                  "SlotV0InputPing capacity must accommodate MAX_PACK_ROWS * head_dim_max");

#if !SAGE_ATTN_DEVICE_COMPILE
    __aicore__ inline __attribute__((always_inline)) static float ReadInputBits(const uint16_t *base, int64_t idx)
    {
        return twin::detail::ReadInput(base, static_cast<size_t>(idx), kDtype);
    }
#endif

    /**
     * \brief 1. V0：GM→L1 Q 搬入与 INT8 量化
     */
    template <bool PER_THREAD, uint32_t D = 0u>
    __aicore__ inline __attribute__((always_inline)) void LoadAndQuantizeQImpl(
        const TaskContext &ctx,
        __gm__ uint8_t *q_src,
        const Catlass::SageAttn::GmBhsdLayout &q_layout)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        const uint32_t total_rows_q = ctx.TotalRows();
        const uint32_t d_step = (D != 0u) ? D : ctx.d;
        const uint32_t d = d_step;

        LocalTensor<float> ub_rowmax(TPosition::VECCALC, SlotV0RowMax::Offset(), MAX_PACK_ROWS);
        LocalTensor<float> ub_scales(TPosition::VECCALC, SlotV0Scales::Offset(), kQScaleSlotStride);
        LocalTensor<half> ub_in(TPosition::VECCALC, SlotV0InputPing::Offset(), MAX_PACK_ROWS * d);
        LocalTensor<int8_t> ub_stage_q_mat(TPosition::VECCALC, SlotV0VStageTok::Offset(), MAX_PACK_ROWS * d);
        LocalTensor<int8_t> l1_q(TPosition::A1, ctx.l1_q_offset, total_rows_q * d);

        // 窗口 scale 组清零
        const uint32_t grp_first_q = quant::QScaleIndex<PER_THREAD>(ctx.cur_q_start);
        const uint32_t grp_last_q = quant::QScaleIndex<PER_THREAD>(ctx.cur_q_start + ctx.cur_q_len - 1u);
        const uint32_t grp_first_aligned = (grp_first_q / 8u) * 8u;
        const uint32_t grp_last_aligned = ::Min(((grp_last_q + 8u) / 8u) * 8u, static_cast<uint32_t>(kQScaleSlotStride));
        const uint32_t grp_cnt_aligned = (grp_first_aligned < grp_last_aligned)
                                             ? (grp_last_aligned - grp_first_aligned)
                                             : 0u;
        __ubuf__ float *ub_scales_ptr = reinterpret_cast<__ubuf__ float*>(ub_scales.GetPhyAddr());
        for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
            dev_vf::FillF32VF(ub_scales_ptr + (ctx.cur_g0 + g) * kQScaleSlotStride + grp_first_aligned, 0.0f, grp_cnt_aligned);
        }

        // A. 行批 abs-max (利用 950 NDDMA 2D/Strided DMA 硬件指令按 Head 批量拉入 UB)
        const uint32_t act_rows_q = ::Min(total_rows_q, static_cast<uint32_t>(MAX_PACK_ROWS));
        const uint32_t stride_q = Catlass::SageAttn::CalcGm2dStride32B(ctx.hq, d, sizeof(half), ctx.tensor_layout);
        Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
        for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
            const uint32_t head_start_row = g * ctx.cur_q_len;
            if (head_start_row >= act_rows_q) break;
            const uint32_t rows_to_load = ::Min(ctx.cur_q_len, act_rows_q - head_start_row);
            const int64_t head_gm_offset = q_layout(tla::MakeCoord(0u, g, 0u, 0u)) * static_cast<int64_t>(sizeof(half));
            Catlass::SageAttn::LoadTile2DFromGm(ub_in[head_start_row * d],
                                                reinterpret_cast<__gm__ half*>(q_src + head_gm_offset),
                                                rows_to_load, d, stride_q);
        }
        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
        dev_vf::RowAbsMaxRowsVF<kDtype, false, D>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(ub_rowmax.GetPhyAddr()),
                                               nullptr, act_rows_q, d);
        Catlass::SageAttn::Sync<HardEvent::V_S>();

        // B. 组归并 + scale 定档
        for (uint32_t r = 0; r < act_rows_q; ++r) {
            const uint32_t g = r / ctx.cur_q_len;
            const uint32_t token = ctx.cur_q_start + (r % ctx.cur_q_len);
            const float amax = ub_rowmax.GetValue(r);
            const uint32_t slot = (ctx.cur_g0 + g) * kQScaleSlotStride + quant::QScaleIndex<PER_THREAD>(token);
            if (amax > ub_scales.GetValue(slot)) ub_scales.SetValue(slot, amax);
        }
        for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
            for (uint32_t s = grp_first_q; s <= grp_last_q && s < kQScaleSlotStride; ++s) {
                const float amax = ub_scales.GetValue((ctx.cur_g0 + g) * kQScaleSlotStride + s);
                ub_scales.SetValue((ctx.cur_g0 + g) * kQScaleSlotStride + s,
                                   quant::FinalizeQkScale<PER_THREAD>(amax));
            }
        }
        Catlass::SageAttn::Sync<HardEvent::S_V>();

        // C. 批量量化与 MTE3 2D DMA 分形重组写入 L1 zN 布局
        const uint32_t m_aligned_q = (act_rows_q + 15u) & ~15u;
        for (uint32_t r = 0; r < act_rows_q; ++r) {
            const uint32_t g = r / ctx.cur_q_len;
            const uint32_t i = r % ctx.cur_q_len;
            const uint32_t token = ctx.cur_q_start + i;
            __ubuf__ float* scale_ptr = ub_scales_ptr + (ctx.cur_g0 + g) * kQScaleSlotStride + quant::QScaleIndex<PER_THREAD>(token);
            dev_vf::QuantRowInt8VF<kDtype, PER_THREAD, false, D>(
                reinterpret_cast<__ubuf__ half*>(ub_in[r * d].GetPhyAddr()),
                reinterpret_cast<__ubuf__ int8_t*>(ub_stage_q_mat[r * d].GetPhyAddr()),
                scale_ptr, d);
        }
        for (uint32_t r = act_rows_q; r < m_aligned_q; ++r) {
            Duplicate(ub_stage_q_mat[r * d], int8_t(0), d);
        }
        Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
        Catlass::SageAttn::CopyUbToL1zN(l1_q, ub_stage_q_mat, act_rows_q, d);
        Catlass::SageAttn::Sync<HardEvent::MTE3_V>();

        // A7 V0：行解析 q_scale 写入片上 L1 Bank 0 专有槽位（彻底消灭 GM 往返）
        LocalTensor<float> ub_stage_q(TPosition::VECCALC, SlotV0StageStrip::Offset(), STAGE_SLOT_FLOATS);
        dev_vf::FillF32VF(reinterpret_cast<__ubuf__ float*>(ub_stage_q.GetPhyAddr()), 0.0f, STAGE_SLOT_FLOATS);
        Catlass::SageAttn::Sync<HardEvent::V_S>();
        for (uint32_t r = 0; r < act_rows_q; ++r) {
            const uint32_t g = r / ctx.cur_q_len;
            const uint32_t token = ctx.cur_q_start + (r % ctx.cur_q_len);
            ub_stage_q.SetValue(r, ub_scales.GetValue((ctx.cur_g0 + g) * kQScaleSlotStride + quant::QScaleIndex<PER_THREAD>(token)));
        }
        Catlass::SageAttn::Sync<HardEvent::S_MTE3>();
        GlobalTensor<float> gm_stage_q;
        gm_stage_q.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ctx.StageStrip() + STAGE_Q_SCALE_OFFSET));
        DataCopy(gm_stage_q, ub_stage_q, STAGE_SLOT_FLOATS);
        Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
#else
        const uint16_t *q_u16 = reinterpret_cast<const uint16_t *>(q_src);
        int8_t *l1_q = reinterpret_cast<int8_t *>(ctx.sim_mem->l1_mem + ctx.l1_q_offset);
        const uint32_t d = ctx.d;

        const uint32_t grp_first = quant::QScaleIndex<PER_THREAD>(ctx.cur_q_start);
        const uint32_t grp_last = quant::QScaleIndex<PER_THREAD>(ctx.cur_q_start + ctx.cur_q_len - 1u);
        for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
            for (uint32_t s = grp_first; s <= grp_last && s < kQScaleSlotStride; ++s) {
                ctx.sim_mem->q_scales[(ctx.cur_g0 + g) * kQScaleSlotStride + s] = 0.0f;
            }
        }

        for (uint32_t r = 0; r < ctx.TotalRows(); ++r) {
            const uint32_t g = r / ctx.cur_q_len;
            if (r >= MAX_PACK_ROWS || g >= ctx.cur_heads) break;
            const uint32_t i = r % ctx.cur_q_len;
            const uint32_t token = ctx.cur_q_start + i;
            float *scale_slot = &ctx.sim_mem->q_scales[(ctx.cur_g0 + g) * kQScaleSlotStride + quant::QScaleIndex<PER_THREAD>(token)];
            const uint16_t *row_ptr = q_u16 + q_layout(tla::MakeCoord(0u, g, i, 0u));
            for (uint32_t c = 0; c < d; ++c) {
                const float abs_v = SageFastAbs(ReadInputBits(row_ptr, c));
                if (abs_v > *scale_slot) *scale_slot = abs_v;
            }
        }

        for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
            for (uint32_t s = grp_first; s <= grp_last && s < kQScaleSlotStride; ++s) {
                const float amax = ctx.sim_mem->q_scales[(ctx.cur_g0 + g) * kQScaleSlotStride + s];
                ctx.sim_mem->q_scales[(ctx.cur_g0 + g) * kQScaleSlotStride + s] =
                    quant::FinalizeQkScale<PER_THREAD>(amax);
            }
        }

        for (uint32_t r = 0; r < ctx.TotalRows(); ++r) {
            const uint32_t g = r / ctx.cur_q_len;
            if (r >= MAX_PACK_ROWS || g >= ctx.cur_heads) break;
            const uint32_t i = r % ctx.cur_q_len;
            const uint32_t token = ctx.cur_q_start + i;
            const float scale = ctx.sim_mem->q_scales[(ctx.cur_g0 + g) * kQScaleSlotStride + quant::QScaleIndex<PER_THREAD>(token)];
            const uint16_t *row_ptr = q_u16 + q_layout(tla::MakeCoord(0u, g, i, 0u));
            for (uint32_t c = 0; c < d; ++c) {
                const float val = ReadInputBits(row_ptr, c);
                const int8_t code = PER_THREAD ? quant::RoundHalfAwayFromZero(val / scale)
                                               : quant::RoundNearestEven(val / scale);
                l1_q[Catlass::SageAttn::ZnIndex(r, c, ctx.TotalRows())] = code;
            }
        }

        if (ctx.sim_mem != nullptr) {
            float *stage_q = reinterpret_cast<float *>(ctx.sim_mem->stage_strip + STAGE_Q_SCALE_OFFSET);
            for (uint32_t r = 0; r < ctx.TotalRows() && r < MAX_PACK_ROWS; ++r) {
                const uint32_t g = r / ctx.cur_q_len;
                const uint32_t token = ctx.cur_q_start + (r % ctx.cur_q_len);
                stage_q[r] = ctx.sim_mem->q_scales[(ctx.cur_g0 + g) * kQScaleSlotStride + quant::QScaleIndex<PER_THREAD>(token)];
            }
        }
#endif
    }

    __aicore__ inline __attribute__((always_inline)) void LoadAndQuantizeQ(
        const TaskContext &ctx,
        __gm__ uint8_t *q_src,
        const Catlass::SageAttn::GmBhsdLayout &q_layout)
    {
        if (ctx.d == 64u) {
            if (ctx.qk_quant_gran == QUANT_GRAN_PER_THREAD) {
                LoadAndQuantizeQImpl<true, 64u>(ctx, q_src, q_layout);
            } else {
                LoadAndQuantizeQImpl<false, 64u>(ctx, q_src, q_layout);
            }
        } else if (ctx.d == 128u) {
            if (ctx.qk_quant_gran == QUANT_GRAN_PER_THREAD) {
                LoadAndQuantizeQImpl<true, 128u>(ctx, q_src, q_layout);
            } else {
                LoadAndQuantizeQImpl<false, 128u>(ctx, q_src, q_layout);
            }
        } else {
            if (ctx.qk_quant_gran == QUANT_GRAN_PER_THREAD) {
                LoadAndQuantizeQImpl<true, 0u>(ctx, q_src, q_layout);
            } else {
                LoadAndQuantizeQImpl<false, 0u>(ctx, q_src, q_layout);
            }
        }
    }

    /**
     * \brief 2. V0：K 减 kMean + INT8 量化 + 摆放写 L1 槽首
     */
    template <bool PER_THREAD, uint32_t D = 0u>
    __aicore__ inline __attribute__((always_inline)) void LoadAndQuantizeKImpl(
        const TaskContext &ctx,
        __gm__ uint8_t *k_src,
        const __gm__ float *km_ptr,
        uint32_t cur_k,
        uint32_t l1_dst_addr)
    {
        const uint32_t cur_k_start = cur_k * ctx.block_n;
        const uint32_t d_step = (D != 0u) ? D : ctx.d;
        const uint32_t d = d_step;
        const uint32_t block_n = ctx.block_n;
        const uint32_t block_k = ctx.CurBlockK(cur_k);

#if SAGE_ATTN_DEVICE_COMPILE
        constexpr uint32_t grp_per_blk = PER_THREAD ? 4u : 1u;
        const uint32_t blk_first = cur_k_start / quant::K_BLK;
        const uint32_t grp_lo = blk_first * grp_per_blk;
        const uint32_t grp_hi = ((cur_k_start + block_n - 1u) / quant::K_BLK + 1u) * grp_per_blk;

        LocalTensor<float> ub_rowmax(TPosition::VECCALC, SlotV0RowMax::Offset(), MAX_PACK_ROWS);
        LocalTensor<float> ub_k_scales(TPosition::VECCALC, SlotV0Scales::Offset(), kKScaleSlotStride);
        __ubuf__ float *ub_k_scales_ptr = reinterpret_cast<__ubuf__ float*>(ub_k_scales.GetPhyAddr());
        LocalTensor<float> ub_km(TPosition::VECCALC, SlotV0Km::Offset(), d);
        LocalTensor<half> ub_in(TPosition::VECCALC, SlotV0InputPing::Offset(), MAX_PACK_ROWS * d);
        LocalTensor<int8_t> ub_stage_k_mat(TPosition::VECCALC, SlotV0VStageTok::Offset(), block_n * d);
        LocalTensor<int8_t> ub_stage_k_tr(TPosition::VECCALC, SlotV0VStageTr::Offset(), d * block_n);
        LocalTensor<int8_t> l1_k(TPosition::B1, l1_dst_addr, d * block_n);
        const bool has_km = (km_ptr != nullptr);

        if (has_km) {
            GlobalTensor<float> gm_km;
            gm_km.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(const_cast<__gm__ float*>(km_ptr)));
            DataCopy(ub_km, gm_km, d);
            Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
        }

        const uint32_t grp_lo_aligned = (grp_lo / 8u) * 8u;
        const uint32_t grp_hi_aligned = ::Min(((grp_hi + 7u) / 8u) * 8u, static_cast<uint32_t>(kKScaleSlotStride));
        const uint32_t grp_fill_cnt = (grp_lo_aligned < grp_hi_aligned)
                                          ? (grp_hi_aligned - grp_lo_aligned)
                                          : 0u;
        if (grp_fill_cnt != 0u) {
            dev_vf::FillF32VF(ub_k_scales_ptr + grp_lo_aligned, 0.0f, grp_fill_cnt);
        }

        // 单条 2D Strided DMA 猝发读整个 block_k 分块拉入 UB，消灭逐行 GM 读与流水气泡
        const uint32_t stride_k = Catlass::SageAttn::CalcGm2dStride32B(ctx.hkv, d, sizeof(half), ctx.tensor_layout);
        Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
        Catlass::SageAttn::LoadTile2DFromGm(ub_in, reinterpret_cast<__gm__ half*>(k_src), block_k, d, stride_k);
        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
        if (has_km) {
            dev_vf::RowAbsMaxRowsVF<kDtype, true, D>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ float*>(ub_rowmax.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ float*>(ub_km.GetPhyAddr()),
                                                  block_k, d);
        } else {
            dev_vf::RowAbsMaxRowsVF<kDtype, false, D>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ float*>(ub_rowmax.GetPhyAddr()),
                                                   nullptr,
                                                   block_k, d);
        }
        Catlass::SageAttn::Sync<HardEvent::V_S>();

        for (uint32_t j = 0; j < block_k; ++j) {
            const uint32_t slot = quant::KScaleIndex<PER_THREAD>(cur_k_start + j);
            const float amax = ub_rowmax.GetValue(j);
            if (amax > ub_k_scales.GetValue(slot)) ub_k_scales.SetValue(slot, amax);
        }
        for (uint32_t s = grp_lo; s < grp_hi && s < kKScaleSlotStride; ++s) {
            const float amax = ub_k_scales.GetValue(s);
            const float scale = quant::FinalizeQkScale<PER_THREAD>(amax);
            ub_k_scales.SetValue(s, scale);
        }
        Catlass::SageAttn::Sync<HardEvent::S_V>();

        LocalTensor<float> ub_stage_k(TPosition::VECCALC, SlotV0StageStrip::Offset(), STAGE_SLOT_FLOATS);
        dev_vf::FillF32VF(reinterpret_cast<__ubuf__ float*>(ub_stage_k.GetPhyAddr()), 0.0f, STAGE_SLOT_FLOATS);
        Catlass::SageAttn::Sync<HardEvent::V_S>();
        // 尾块 padding 槽保持 FillF32VF 预置的 0.0f，循环只覆盖有效 token 区间
        for (uint32_t j = 0; j < block_k; ++j) {
            const uint32_t token = cur_k_start + j;
            ub_stage_k.SetValue(j, ub_k_scales.GetValue(quant::KScaleIndex<PER_THREAD>(token)));
        }
        Catlass::SageAttn::Sync<HardEvent::S_MTE3>();
        GlobalTensor<float> gm_stage_k;
        uint32_t k_offset = STAGE_K_SCALE_BASE_OFFSET + (cur_k % STAGE_K_SCALE_NUM_SLOTS) * STAGE_SLOT_BYTES;
        gm_stage_k.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ctx.StageStrip() + k_offset));
        DataCopy(gm_stage_k, ub_stage_k, STAGE_SLOT_FLOATS);
        Catlass::SageAttn::Sync<HardEvent::MTE3_S>();

        // 批量量化 INT8 矩阵 [block_n, d]
        if (has_km) {
            for (uint32_t j = 0; j < block_k; ++j) {
                const uint32_t token = cur_k_start + j;
                __ubuf__ float* scale_ptr = ub_k_scales_ptr + quant::KScaleIndex<PER_THREAD>(token);
                dev_vf::QuantRowInt8VF<kDtype, PER_THREAD, true, D>(
                    reinterpret_cast<__ubuf__ half*>(ub_in[j * d].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ int8_t*>(ub_stage_k_mat[j * d].GetPhyAddr()),
                    scale_ptr, d,
                    reinterpret_cast<__ubuf__ float*>(ub_km.GetPhyAddr()));
            }
        } else {
            for (uint32_t j = 0; j < block_k; ++j) {
                const uint32_t token = cur_k_start + j;
                __ubuf__ float* scale_ptr = ub_k_scales_ptr + quant::KScaleIndex<PER_THREAD>(token);
                dev_vf::QuantRowInt8VF<kDtype, PER_THREAD, false, D>(
                    reinterpret_cast<__ubuf__ half*>(ub_in[j * d].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ int8_t*>(ub_stage_k_mat[j * d].GetPhyAddr()),
                    scale_ptr, d,
                    nullptr);
            }
        }
        for (uint32_t j = block_k; j < block_n; ++j) {
            Duplicate(ub_stage_k_mat[j * d], int8_t(0), d);
        }
        Catlass::SageAttn::Sync<HardEvent::V_S>();

        // K 矩阵预转置 [block_n, d] -> [d, block_n] (通道主序 K^T)
        dev_vf::TransposeB8VF<D>(
            reinterpret_cast<__ubuf__ uint8_t*>(ub_stage_k_tr.GetPhyAddr()),
            reinterpret_cast<__ubuf__ const uint8_t*>(ub_stage_k_mat.GetPhyAddr()),
            block_n, d);

        Catlass::SageAttn::Sync<HardEvent::S_MTE3>();
        Catlass::SageAttn::CopyUbToL1zN(l1_k, ub_stage_k_tr, d, block_n);
        Catlass::SageAttn::Sync<HardEvent::MTE3_V>();
#else
        const uint16_t *k_u16 = reinterpret_cast<const uint16_t *>(k_src);
        int8_t *l1_k = reinterpret_cast<int8_t *>(ctx.sim_mem->l1_mem + l1_dst_addr);
        const Catlass::SageAttn::GmBhsdLayout kv_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(ctx.b, ctx.hkv, ctx.sk, ctx.d, ctx.is_hnd != 0);

        constexpr uint32_t grp_per_blk = PER_THREAD ? 4u : 1u;
        const uint32_t blk_first = cur_k_start / quant::K_BLK;
        const uint32_t blk_last = (cur_k_start + block_n - 1u) / quant::K_BLK;
        const uint32_t grp_lo = blk_first * grp_per_blk;
        const uint32_t grp_hi = (blk_last + 1u) * grp_per_blk;
        for (uint32_t s = grp_lo; s < grp_hi && s < kKScaleSlotStride; ++s) {
            ctx.sim_mem->k_scales[s] = 0.0f;
        }

        for (uint32_t j = 0; j < block_k; ++j) {
            const uint32_t token = cur_k_start + j;
            float *scale_slot = &ctx.sim_mem->k_scales[quant::KScaleIndex<PER_THREAD>(token)];
            const uint16_t *row_ptr = k_u16 + kv_layout(tla::MakeCoord(0u, 0u, j, 0u));
            for (uint32_t c = 0; c < d; ++c) {
                float val = ReadInputBits(row_ptr, c);
                val = quant::SubKm<kDtype>(val, km_ptr, c);
                const float abs_v = SageFastAbs(val);
                if (abs_v > *scale_slot) *scale_slot = abs_v;
            }
        }

        for (uint32_t s = grp_lo; s < grp_hi && s < kKScaleSlotStride; ++s) {
            const float amax = ctx.sim_mem->k_scales[s];
            const float scale = quant::FinalizeQkScale<PER_THREAD>(amax);
            ctx.sim_mem->k_scales[s] = scale;
        }

        if (ctx.sim_mem != nullptr) {
            uint32_t k_offset = STAGE_K_SCALE_BASE_OFFSET + (cur_k % STAGE_K_SCALE_NUM_SLOTS) * STAGE_SLOT_BYTES;
            float *stage_k = reinterpret_cast<float *>(ctx.sim_mem->stage_strip + k_offset);
            for (uint32_t j = 0; j < block_n; ++j) {
                if (j < block_k) {
                    const uint32_t token = cur_k_start + j;
                    stage_k[j] = ctx.sim_mem->k_scales[quant::KScaleIndex<PER_THREAD>(token)];
                } else {
                    stage_k[j] = 0.0f;
                }
            }
        }

        for (uint32_t j = 0; j < block_n; ++j) {
            if (j < block_k) {
                const uint32_t token = cur_k_start + j;
                const float scale = ctx.sim_mem->k_scales[quant::KScaleIndex<PER_THREAD>(token)];
                const uint16_t *row_ptr = k_u16 + kv_layout(tla::MakeCoord(0u, 0u, j, 0u));
                for (uint32_t c = 0; c < d; ++c) {
                    float val = ReadInputBits(row_ptr, c);
                    val = quant::SubKm<kDtype>(val, km_ptr, c);
                    const int8_t code = PER_THREAD ? quant::RoundHalfAwayFromZero(val / scale)
                                                   : quant::RoundNearestEven(val / scale);
                    l1_k[Catlass::SageAttn::ZnIndex(c, j, d)] = code;
                }
            } else {
                for (uint32_t c = 0; c < d; ++c) {
                    l1_k[Catlass::SageAttn::ZnIndex(c, j, d)] = 0;
                }
            }
        }
#endif
    }

    __aicore__ inline __attribute__((always_inline)) void LoadAndQuantizeK(
        const TaskContext &ctx,
        __gm__ uint8_t *k_src,
        const __gm__ float *km_ptr,
        uint32_t cur_k,
        uint32_t l1_dst_addr)
    {
        if (ctx.d == 64u) {
            if (ctx.qk_quant_gran == QUANT_GRAN_PER_THREAD) {
                LoadAndQuantizeKImpl<true, 64u>(ctx, k_src, km_ptr, cur_k, l1_dst_addr);
            } else {
                LoadAndQuantizeKImpl<false, 64u>(ctx, k_src, km_ptr, cur_k, l1_dst_addr);
            }
        } else if (ctx.d == 128u) {
            if (ctx.qk_quant_gran == QUANT_GRAN_PER_THREAD) {
                LoadAndQuantizeKImpl<true, 128u>(ctx, k_src, km_ptr, cur_k, l1_dst_addr);
            } else {
                LoadAndQuantizeKImpl<false, 128u>(ctx, k_src, km_ptr, cur_k, l1_dst_addr);
            }
        } else {
            if (ctx.qk_quant_gran == QUANT_GRAN_PER_THREAD) {
                LoadAndQuantizeKImpl<true, 0u>(ctx, k_src, km_ptr, cur_k, l1_dst_addr);
            } else {
                LoadAndQuantizeKImpl<false, 0u>(ctx, k_src, km_ptr, cur_k, l1_dst_addr);
            }
        }
    }

    /**
     * \brief 3. V0：V 转置 + 16-token permute + per-channel FP8 量化写 L1 槽内 V 子区
     */
    template <uint32_t D = 0u>
    __aicore__ inline __attribute__((always_inline)) void QuantizeAndStageValueImpl(
        const TaskContext &ctx,
        __gm__ uint8_t *v_src,
        uint32_t l1_dst_addr,
        uint32_t cur_k,
        uint32_t block_k,
        const Catlass::SageAttn::GmBhsdLayout &kv_layout)
    {
        const uint32_t d_step = (D != 0u) ? D : ctx.d;
        const uint32_t d = d_step;
        const uint32_t block_n = ctx.block_n;
        const float scale_max = ctx.v_scale_max;

#if SAGE_ATTN_DEVICE_COMPILE
        LocalTensor<float> ub_v_scale(TPosition::VECCALC, SlotV0VScale::Offset(), d);
        LocalTensor<float> ub_v_min(TPosition::VECCALC, SlotV0VMin::Offset(), d);
        LocalTensor<float> ub_v_sum(TPosition::VECCALC, SlotV0VSum::Offset(), d);
        LocalTensor<half> ub_in(TPosition::VECCALC, SlotV0InputPing::Offset(), MAX_PACK_ROWS * d);
        LocalTensor<float8_e4m3_t> ub_stage_tok(TPosition::VECCALC, SlotV0VStageTok::Offset(), block_n * d);
        LocalTensor<float8_e4m3_t> l1_v(TPosition::B1, l1_dst_addr, block_n * d);
        const uint32_t smooth_v = (ctx.smooth_v != 0) ? 1u : 0u;
        const uint32_t stride_v = Catlass::SageAttn::CalcGm2dStride32B(ctx.hkv, d, sizeof(half), ctx.tensor_layout);

        if (cur_k == 0) {
            bool is_head_cached = (ctx.cur_b == cached_b_ && ctx.cur_hkv == cached_hkv_ && v_scale_cached_);
            if (!is_head_cached) {
                const uint32_t fp8_padded = ::RoundUp<16u>(ctx.sk);
                if (smooth_v != 0u) {
                    Duplicate(ub_v_scale, MIN_FLOAT_VAL, d);
                    Duplicate(ub_v_min, -MIN_FLOAT_VAL, d);
                    Duplicate(ub_v_sum, 0.0f, d);
                    for (uint32_t t_start = 0; t_start < ctx.sk; t_start += MAX_PACK_ROWS) {
                        const uint32_t chunk_len = ::Min(static_cast<uint32_t>(MAX_PACK_ROWS), ctx.sk - t_start);
                        const int64_t chunk_gm_offset = kv_layout(tla::MakeCoord(0u, 0u, t_start, 0u)) * static_cast<int64_t>(sizeof(half));
                        Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
                        Catlass::SageAttn::LoadTile2DFromGm(ub_in, reinterpret_cast<__gm__ half*>(v_src + chunk_gm_offset),
                                                            chunk_len, d, stride_v);
                        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
                        dev_vf::ChannelStatsAccumTileVF<kDtype, D>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                                                reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                                                reinterpret_cast<__ubuf__ float*>(ub_v_min.GetPhyAddr()),
                                                                reinterpret_cast<__ubuf__ float*>(ub_v_sum.GetPhyAddr()),
                                                                chunk_len, d);
                    }
                    Catlass::SageAttn::Sync<HardEvent::V_S>();
                    dev_vf::FinalizeSmoothVScaleVF<D>(reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ float*>(ub_v_min.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ float*>(ub_v_sum.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                                   fp8_padded, (fp8_padded > ctx.sk) ? 1u : 0u, scale_max, d);
                    Catlass::SageAttn::Sync<HardEvent::V_S>();
                } else {
                    Duplicate(ub_v_scale, 0.0f, d);
                    for (uint32_t t_start = 0; t_start < ctx.sk; t_start += MAX_PACK_ROWS) {
                        const uint32_t chunk_len = ::Min(static_cast<uint32_t>(MAX_PACK_ROWS), ctx.sk - t_start);
                        const int64_t chunk_gm_offset = kv_layout(tla::MakeCoord(0u, 0u, t_start, 0u)) * static_cast<int64_t>(sizeof(half));
                        Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
                        Catlass::SageAttn::LoadTile2DFromGm(ub_in, reinterpret_cast<__gm__ half*>(v_src + chunk_gm_offset),
                                                            chunk_len, d, stride_v);
                        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
                        dev_vf::ChannelAmaxAccumTileVF<kDtype, D>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                                               reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                                               chunk_len, d);
                    }
                    dev_vf::FinalizeVScaleVF<D>(reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                             reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                             scale_max, d);
                }
                cached_b_ = ctx.cur_b;
                cached_hkv_ = ctx.cur_hkv;
                v_scale_cached_ = true;
            }

            Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
            GlobalTensor<float> gm_stage_v;
            gm_stage_v.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ctx.StageStrip() + STAGE_V_SCALE_OFFSET));
            DataCopy(gm_stage_v, ub_v_scale, STAGE_SLOT_FLOATS);
            if (smooth_v != 0u) {
                GlobalTensor<float> gm_vm;
                gm_vm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                    ctx.workspace_gm + ctx.workspace_vm_offset +
                    static_cast<int64_t>(ctx.cur_b) * ctx.hkv * d * sizeof(float) +
                    static_cast<int64_t>(ctx.cur_hkv) * d * sizeof(float)));
                DataCopy(gm_vm, ub_v_sum, d);
            }
            Catlass::SageAttn::Sync<HardEvent::MTE3_V>();
        }

        // 分块量化写 L1：单条 2D Strided DMA 猝发读整个 block_k 分块拉入 UB，消灭逐行 GM 读与流水气泡
        Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
        Catlass::SageAttn::LoadTile2DFromGm(ub_in, reinterpret_cast<__gm__ half*>(v_src), block_k, d, stride_v);
        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();

        // 阶段一：V 行 FP8 量化写入 tok 暂存 [block_n, d]
        if (smooth_v != 0u) {
            for (uint32_t j = 0; j < block_k; ++j) {
                dev_vf::QuantVRowFp8VF<kDtype, true, D>(
                    reinterpret_cast<__ubuf__ half*>(ub_in[j * d].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_stage_tok[j * d].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(ub_v_sum.GetPhyAddr()),
                    d, scale_max);
            }
        } else {
            for (uint32_t j = 0; j < block_k; ++j) {
                dev_vf::QuantVRowFp8VF<kDtype, false, D>(
                    reinterpret_cast<__ubuf__ half*>(ub_in[j * d].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_stage_tok[j * d].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                    nullptr,
                    d, scale_max);
            }
        }
        for (uint32_t j = block_k; j < block_n; ++j) {
            Duplicate(ub_stage_tok[j * d].ReinterpretCast<int8_t>(), int8_t(0), d);
        }

        // 阶段二：MTE3 2D DMA 分形重组写入 L1 zN 布局（硬件 L0B 加载时原生转置为 nZ）
        Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
        Catlass::SageAttn::CopyUbToL1zN(l1_v, ub_stage_tok, block_n, d);
        Catlass::SageAttn::Sync<HardEvent::MTE3_V>();
#else
        const uint16_t *v_u16 = reinterpret_cast<const uint16_t *>(v_src);
        uint8_t *l1_v = ctx.sim_mem->l1_mem + l1_dst_addr;

        if (cur_k == 0) {
            const uint16_t *v_head = v_u16 - kv_layout(tla::MakeCoord(0u, 0u, cur_k * block_n, 0u));
            const uint32_t fp8_padded = ::RoundUp<16u>(ctx.sk);
            const bool smooth_v = (ctx.smooth_v != 0);
            for (uint32_t c = 0; c < d && c < 256u; ++c) {
                if (smooth_v) {
                    float sum_v = 0.0f;
                    for (uint32_t t = 0; t < ctx.sk; ++t) {
                        sum_v += ReadInputBits(v_head, kv_layout(tla::MakeCoord(0u, 0u, t, 0u)) + c);
                    }
                    const float vm = sum_v / static_cast<float>(fp8_padded);
                    float amax = (fp8_padded > ctx.sk) ? SageFastAbs(vm) : 0.0f;
                    for (uint32_t t = 0; t < ctx.sk; ++t) {
                        const float diff =
                            ReadInputBits(v_head, kv_layout(tla::MakeCoord(0u, 0u, t, 0u)) + c) - vm;
                        const float abs_v = SageFastAbs(diff);
                        if (abs_v > amax) amax = abs_v;
                    }
                    ctx.sim_mem->v_means[c] = vm;
                    ctx.sim_mem->v_scales[c] = amax / scale_max;
                } else {
                    float amax = 0.0f;
                    for (uint32_t t = 0; t < ctx.sk; ++t) {
                        const float abs_v = SageFastAbs(
                            ReadInputBits(v_head, kv_layout(tla::MakeCoord(0u, 0u, t, 0u)) + c));
                        if (abs_v > amax) amax = abs_v;
                    }
                    ctx.sim_mem->v_scales[c] = amax / scale_max;
                }
            }
            if (ctx.sim_mem != nullptr) {
                float *stage_v = reinterpret_cast<float *>(ctx.sim_mem->stage_strip + STAGE_V_SCALE_OFFSET);
                for (uint32_t c = 0; c < d; ++c) {
                    stage_v[c] = ctx.sim_mem->v_scales[c];
                }
            }
        }

        const bool smooth_v = (ctx.smooth_v != 0);
        for (uint32_t c = 0; c < d; ++c) {
            const float scale = (c < 256u) ? ctx.sim_mem->v_scales[c] : 1.0f;
            const float vm = (smooth_v && c < 256u) ? ctx.sim_mem->v_means[c] : 0.0f;
            for (uint32_t jl = 0; jl < block_n; ++jl) {
                uint8_t code = 0;
                if (jl < block_k) {
                    const uint16_t *row_ptr = v_u16 + kv_layout(tla::MakeCoord(0u, 0u, jl, 0u));
                    float scaled = (ReadInputBits(row_ptr, c) - vm) / scale;
                    if (scaled > scale_max) scaled = scale_max;
                    if (scaled < -scale_max) scaled = -scale_max;
                    code = quant::FloatToE4m3fn(scaled);
                }
                l1_v[Catlass::SageAttn::ZnIndex(jl, c, block_n)] = code;
            }
        }
#endif
    }

    __aicore__ inline __attribute__((always_inline)) void QuantizeAndStageValue(
        const TaskContext &ctx,
        __gm__ uint8_t *v_src,
        uint32_t l1_dst_addr,
        uint32_t cur_k,
        uint32_t block_k,
        const Catlass::SageAttn::GmBhsdLayout &kv_layout)
    {
        if (ctx.d == 64u) {
            QuantizeAndStageValueImpl<64u>(ctx, v_src, l1_dst_addr, cur_k, block_k, kv_layout);
        } else if (ctx.d == 128u) {
            QuantizeAndStageValueImpl<128u>(ctx, v_src, l1_dst_addr, cur_k, block_k, kv_layout);
        } else {
            QuantizeAndStageValueImpl<0u>(ctx, v_src, l1_dst_addr, cur_k, block_k, kv_layout);
        }
    }

    /**
     * \brief Vector 0 主执行流程
     */
    __aicore__ inline __attribute__((always_inline)) void Process(const TaskContext &ctx)
    {
        int64_t b_idx = static_cast<int64_t>(ctx.cur_b);
        int64_t hkv_idx = static_cast<int64_t>(ctx.cur_hkv);
        int64_t sq_start = static_cast<int64_t>(ctx.cur_q_start);
        int64_t hq_first = hkv_idx * static_cast<int64_t>(ctx.gqa_ratio) + static_cast<int64_t>(ctx.cur_g0);

        const Catlass::SageAttn::GmBhsdLayout q_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(ctx.b, ctx.hq, ctx.sq, ctx.d, ctx.is_hnd != 0);
        const int64_t q_gm_offset = q_layout(tla::MakeCoord(b_idx, hq_first, sq_start, 0u));

        // 1. Prologue：加载当前核负责的 Q，执行 INT8 量化并写入 L1 Bank 0
        LoadAndQuantizeQ(ctx, ctx.query_gm + q_gm_offset * sizeof(uint16_t), q_layout);

        // 通知 Cube：Q 量化常驻已就位
        AttentionSync::V0SignalQReady();

        const uint32_t active_kv_blocks = ctx.active_kv_blocks;
        const Catlass::SageAttn::GmBhsdLayout kv_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(ctx.b, ctx.hkv, ctx.sk, ctx.d, ctx.is_hnd != 0);

        // 2. 流水循环：流式预取第 k 块 KV 并执行 INT8/FP8 量化
        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t l1_dst_offset = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();

            // 缓冲回收握手：首两拍 (k < 2) L1 槽天然空闲，k >= 2 起严格受 Cube 消费进度回压
            if (k >= 2) {
                AttentionSync::V0WaitKvFree(is_ping);
            }

            int64_t sk_start = static_cast<int64_t>(k) * ctx.block_n;
            uint32_t cur_block_k = ctx.CurBlockK(k);

            const int64_t kv_gm_offset = kv_layout(tla::MakeCoord(b_idx, hkv_idx, sk_start, 0u));

            const __gm__ float *km_ptr = nullptr;
#if !SAGE_ATTN_DEVICE_COMPILE
            if (ctx.smooth_k != 0) {
                km_ptr = reinterpret_cast<const __gm__ float *>(ctx.km_self);
            }
#endif

            LoadAndQuantizeK(ctx, ctx.key_gm + kv_gm_offset * sizeof(uint16_t), km_ptr,
                             k, l1_dst_offset);

            QuantizeAndStageValue(ctx, ctx.value_gm + kv_gm_offset * sizeof(uint16_t),
                                  is_ping ? SlotL1VPing::Offset() : SlotL1VPong::Offset(),
                                  k, cur_block_k, kv_layout);

            AttentionSync::V0SignalKvReady(is_ping);
        }

        // 尾拍配平：等待 Cube 释放最后两拍的 KV 缓冲，保证信道零残留并在退出前与 Cube 对齐
        if (active_kv_blocks >= 1) {
            AttentionSync::V0WaitKvFree<true>();
        }
        if (active_kv_blocks >= 2) {
            AttentionSync::V0WaitKvFree<false>();
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ProcessStandalone(const TaskContext &ctx)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        int64_t b_idx = static_cast<int64_t>(ctx.cur_b);
        int64_t hkv_idx = static_cast<int64_t>(ctx.cur_hkv);
        int64_t sq_start = static_cast<int64_t>(ctx.cur_q_start);
        int64_t hq_first = hkv_idx * static_cast<int64_t>(ctx.gqa_ratio) + static_cast<int64_t>(ctx.cur_g0);

        const Catlass::SageAttn::GmBhsdLayout q_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(ctx.b, ctx.hq, ctx.sq, ctx.d, ctx.is_hnd != 0);
        const int64_t q_gm_offset = q_layout(tla::MakeCoord(b_idx, hq_first, sq_start, 0u));

        // 1. Prologue：加载 Q，执行量化写 L1 和 StageStrip
        LoadAndQuantizeQ(ctx, ctx.query_gm + q_gm_offset * sizeof(uint16_t), q_layout);

        const uint32_t active_kv_blocks = ctx.active_kv_blocks;
        const Catlass::SageAttn::GmBhsdLayout kv_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(ctx.b, ctx.hkv, ctx.sk, ctx.d, ctx.is_hnd != 0);
        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t l1_dst_offset = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();
            int64_t sk_start = static_cast<int64_t>(k) * ctx.block_n;
            uint32_t cur_block_k = ctx.CurBlockK(k);

            const int64_t kv_gm_offset = kv_layout(tla::MakeCoord(b_idx, hkv_idx, sk_start, 0u));

            LoadAndQuantizeK(ctx, ctx.key_gm + kv_gm_offset * sizeof(uint16_t), nullptr,
                             k, l1_dst_offset);

            QuantizeAndStageValue(ctx, ctx.value_gm + kv_gm_offset * sizeof(uint16_t),
                                  is_ping ? SlotL1VPing::Offset() : SlotL1VPong::Offset(),
                                  k, cur_block_k, kv_layout);
        }
#else
        (void)ctx;
#endif
    }

private:
    uint32_t cached_b_{UINT32_MAX};
    uint32_t cached_hkv_{UINT32_MAX};
    bool v_scale_cached_{false};

public:
    __aicore__ inline __attribute__((always_inline)) void InvalidateCache() {
        cached_b_ = UINT32_MAX;
        cached_hkv_ = UINT32_MAX;
        v_scale_cached_ = false;
    }
};

} // namespace sage_attn
