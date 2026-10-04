#pragma once

#include <cstdint>
#include <memory>
#include <algorithm>
#include "static_slot.h"
#include "attention_sync.h"
#include "attention_cube.h"
#include "attention_vec0_quant.h"
#include "attention_vec1_softmax.h"
#include "sage_device_vf.h"
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
 * \brief 专职 Vector 1 侧 L0C 部分和累积、Scale 恢复、Row Rescale 与 GM 输出写回
 */
template <uint32_t DTYPE_MODE>
class AttentionVec1Output {
public:
    static constexpr uint32_t kDtype = NormDtypeMode(DTYPE_MODE);

#if !SAGE_ATTN_DEVICE_COMPILE
    __aicore__ inline __attribute__((always_inline)) static float ReadInputBits(const uint16_t *base, int64_t idx)
    {
        return twin::detail::ReadInput(base, static_cast<size_t>(idx), kDtype);
    }

    __aicore__ inline __attribute__((always_inline)) static void WriteOutputBits(uint16_t *base, int64_t idx, float val)
    {
        twin::detail::WriteOutput(base, static_cast<size_t>(idx), kDtype, val);
    }
#endif

    /**
     * \brief 复位 O 主累加器
     */
    __aicore__ inline __attribute__((always_inline)) void ResetAccumulator(const TaskContext &ctx)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        const uint32_t rows = ctx.TotalRows();
        LocalTensor<float> ub_o(TPosition::VECCALC, SlotV1OAccum::Offset(), rows * ctx.d);
        Duplicate(ub_o, 0.0f, rows * ctx.d);
#else
        for (uint32_t r = 0; r < ctx.TotalRows() && r < MAX_PACK_ROWS; ++r) {
            std::fill_n(&ctx.sim_mem->o_accum[static_cast<size_t>(r) * ctx.d], ctx.d, 0.0f);
        }
#endif
    }

    /**
     * \brief A7：pv_partial 交接槽全行合并
     */
    __aicore__ inline __attribute__((always_inline)) void MergePartialRows(const TaskContext &ctx)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        const uint32_t rows = ctx.TotalRows();
        const uint32_t fp16_round = (ctx.tiling->pv_accum_dtype == PV_ACCUM_FP32_FP16) ? 1u : 0u;
        LocalTensor<float> ub_vs(TPosition::VECCALC, SlotV1VsChan::Offset(), ctx.d);
        LocalTensor<float> ub_partial(TPosition::VECCALC, SlotV1PvPartial::Offset(), rows * ctx.d);
        LocalTensor<float> ub_o(TPosition::VECCALC, SlotV1OAccum::Offset(), rows * ctx.d);
        dev_vf::MergePvPartialRowsVF(
            reinterpret_cast<__ubuf__ float*>(ub_partial.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ub_o.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ub_vs.GetPhyAddr()),
            rows, ctx.d, fp16_round);
#else
        (void)ctx;
#endif
    }

#if !SAGE_ATTN_DEVICE_COMPILE
    /**
     * \brief 宿主/KT 标量基线：PV 累加经 twin::AccumulatePvRow 单点执行
     */
    __aicore__ inline __attribute__((always_inline)) void AccumulatePvBlockTwin(
        const TaskContext &ctx,
        uint32_t cur_k,
        uint32_t block_k,
        const OnlineSoftmaxState row_states_[MAX_PACK_ROWS])
    {
        const bool fp16_accum = (ctx.tiling->pv_accum_dtype == PV_ACCUM_FP32_FP16);
        const bool is_ping = ((cur_k % 2) == 0);
        const uint32_t cur_kv_base = is_ping ? SlotL1KvPing::Offset() : SlotL1KvPong::Offset();
        const uint8_t *l1_v = ctx.sim_mem->l1_mem + cur_kv_base +
                              Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
        const uint32_t rows = ctx.TotalRows();
        const uint32_t cur_k_start = cur_k * ctx.block_n;

        for (uint32_t r = 0; r < rows && r < MAX_PACK_ROWS; ++r) {
            const uint32_t token_q = ctx.cur_q_start + (r % ctx.cur_q_len);
            const float m_prime = row_states_[r].m_scale;
            twin::AccumulatePvRow(
                &ctx.sim_mem->o_accum[static_cast<size_t>(r) * ctx.d], ctx.d,
                &ctx.sim_mem->logits[static_cast<size_t>(r) * ctx.block_n], cur_k_start,
                cur_k_start, cur_k_start + block_k,
                ctx.tiling->is_causal != 0, token_q, m_prime,
                fp16_accum ? PV_ACCUM_FP32_FP16 : PV_ACCUM_FP32, ctx.k_step,
                [&](uint32_t j, uint32_t c) -> float {
                    const uint32_t jl = j - cur_k_start;
                    const uint32_t slot = quant::VSlotOfToken(jl);
                    const float vm = (ctx.tiling->smooth_v != 0) ? ctx.sim_mem->v_means[c] : 0.0f;
                    return quant::E4m3fnToFloat(l1_v[static_cast<size_t>(c) * ctx.block_n + slot]) *
                               ctx.sim_mem->v_scales[c] + vm;
                });
        }
    }
#endif

    /**
     * \brief 最终输出归一化与 LSE 换算写回 GM
     */
    __aicore__ inline __attribute__((always_inline)) void FinalizeAndStore(
        const TaskContext &ctx,
        const OnlineSoftmaxState row_states_[MAX_PACK_ROWS])
    {
        const int64_t b_idx = static_cast<int64_t>(ctx.cur_b);
        const int64_t hkv_idx = static_cast<int64_t>(ctx.cur_hkv);
        const int64_t sq_start = static_cast<int64_t>(ctx.cur_q_start);

        const Catlass::SageAttn::GmBhsdLayout out_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(ctx.b, ctx.hq, ctx.sq, ctx.d, ctx.tiling->tensor_layout == LAYOUT_HND);
        const Catlass::SageAttn::GmBhsLayout lse_layout =
            Catlass::SageAttn::MakeGmBhsLayout(ctx.b, ctx.hq, ctx.sq);

#if SAGE_ATTN_DEVICE_COMPILE
        const bool has_vm = (ctx.tiling->smooth_v != 0);
        LocalTensor<float> ub_vm(TPosition::VECCALC, SlotV1VmChan::Offset(), ctx.d);
        if (has_vm) {
            GlobalTensor<float> gm_vm;
            gm_vm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                ctx.workspace_gm + ctx.tiling->workspace_vm_offset +
                (b_idx * static_cast<int64_t>(ctx.hkv) + hkv_idx) * ctx.d * sizeof(float)));
            DataCopy(ub_vm, gm_vm, ctx.d);
            Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
        }
        LocalTensor<half> ub_half(TPosition::VECCALC, SlotV1FixpipeRecv::Offset(), ctx.d);
        LocalTensor<float> ub_running_sum(TPosition::VECCALC, SlotV1RowSum::Offset(), MAX_PACK_ROWS);
        LocalTensor<float> ub_running_max(TPosition::VECCALC, SlotV1RowMax::Offset(), MAX_PACK_ROWS);

        // 1. 最终输出归一化与写回 GM（纯向量流水与 MTE3 搬运，零标量计算、零 LSE 耦合）
        for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
            const int64_t hq_curr = hkv_idx * static_cast<int64_t>(ctx.gqa_ratio) + static_cast<int64_t>(ctx.cur_g0) + g;
            if (hq_curr >= ctx.hq) break;

            for (uint32_t i = 0; i < ctx.cur_q_len; ++i) {
                const uint32_t r = g * ctx.cur_q_len + i;
                if (r >= MAX_PACK_ROWS) break;

                LocalTensor<float> ub_o_row(TPosition::VECCALC,
                                            SlotV1OAccum::Offset() + r * ctx.d * sizeof(float),
                                            ctx.d);
                dev_vf::FinalizeRowVF<kDtype>(reinterpret_cast<__ubuf__ float*>(ub_o_row.GetPhyAddr()),
                                             reinterpret_cast<__ubuf__ uint16_t*>(ub_half.GetPhyAddr()),
                                             reinterpret_cast<__ubuf__ float*>(ub_running_sum.GetPhyAddr()),
                                             r,
                                             has_vm ? reinterpret_cast<__ubuf__ float*>(ub_vm.GetPhyAddr()) : nullptr,
                                             ctx.tiling->d_og);
                Catlass::SageAttn::Sync<HardEvent::V_MTE3>();

                const int64_t out_gm_offset = out_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i, 0u)) *
                                              static_cast<int64_t>(sizeof(half));
                if constexpr (kDtype == kTplDtypeBf16) {
                    GlobalTensor<bfloat16_t> gm_out;
                    gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ctx.out_gm + out_gm_offset));
                    gm_out.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    LocalTensor<bfloat16_t> ub_bf16 = ub_half.ReinterpretCast<bfloat16_t>();
                    DataCopy(gm_out, ub_bf16, ctx.tiling->d_og);
                } else {
                    GlobalTensor<half> gm_out;
                    gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ctx.out_gm + out_gm_offset));
                    gm_out.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    DataCopy(gm_out, ub_half, ctx.tiling->d_og);
                }
                Catlass::SageAttn::Sync<HardEvent::MTE3_V>();
            }
        }

        // 2. LSE 计算与写回 GM（严格收敛在 has_lse 内部，has_lse == 0 时彻底旁路，零标量开销、零同步等待）
        if (ctx.tiling->has_lse != 0 && ctx.lse_gm != nullptr) {
            Catlass::SageAttn::Sync<HardEvent::V_S>();
            for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
                const int64_t hq_curr = hkv_idx * static_cast<int64_t>(ctx.gqa_ratio) + static_cast<int64_t>(ctx.cur_g0) + g;
                if (hq_curr >= ctx.hq) break;

                const int64_t lse_elem_base = lse_layout(tla::MakeCoord(b_idx, hq_curr, sq_start));
                const uint32_t lse_len_padded = ((ctx.cur_q_len + 7u) / 8u) * 8u;
                LocalTensor<float> ub_lse_row(TPosition::VECCALC, SlotV1LseRow::Offset(), lse_len_padded);

                for (uint32_t i = 0; i < ctx.cur_q_len; ++i) {
                    const uint32_t r = g * ctx.cur_q_len + i;
                    if (r >= MAX_PACK_ROWS) break;

                    const float m_val = ub_running_max.GetValue(r);
                    const float d_val = ub_running_sum.GetValue(r);
                    const float lse_exp2 = (d_val > 0.0f)
                        ? (m_val + OnlineSoftmaxTransformer::FastLog2(d_val))
                        : MIN_FLOAT_VAL;
                    const float lse_host = OnlineSoftmaxTransformer::ConvertLseToHost(
                        lse_exp2, ctx.tiling->sm_scale_log2e * INV_LOG2E,
                        0.0f, false, S_FP8_OFFSET);
                    ub_lse_row.SetValue(i, lse_host);
                }

                Catlass::SageAttn::Sync<HardEvent::S_MTE3>();
                GlobalTensor<float> gm_lse;
                gm_lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                    ctx.lse_gm + lse_elem_base * static_cast<int64_t>(sizeof(float))));
                gm_lse.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

                const bool lse_burst_ok = (static_cast<uint64_t>(lse_elem_base) % 8u == 0u) && ((ctx.cur_q_len % 8u) == 0u);
                if (lse_burst_ok) {
                    DataCopy(gm_lse, ub_lse_row, ctx.cur_q_len);
                } else {
                    AscendC::DataCopyExtParams copyOutParams(1, static_cast<uint32_t>(ctx.cur_q_len) * sizeof(float), 0, 0, 0);
                    AscendC::DataCopyPad(gm_lse, ub_lse_row, copyOutParams);
                }
                Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
            }
        }
#else
        for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
            const int64_t hq_curr = hkv_idx * static_cast<int64_t>(ctx.gqa_ratio) + static_cast<int64_t>(ctx.cur_g0) + g;
            if (hq_curr >= ctx.hq) break;

            for (uint32_t i = 0; i < ctx.cur_q_len; ++i) {
                const uint32_t r = g * ctx.cur_q_len + i;
                if (r >= MAX_PACK_ROWS) break;
                const float inv_row = OnlineSoftmaxTransformer::GetFinalInvDenominator(row_states_[r]);

                for (uint32_t c = 0; c < ctx.tiling->d_og; ++c) {
                    const float val = ctx.sim_mem->o_accum[static_cast<size_t>(r) * ctx.d + c] * inv_row;
                    const int64_t out_elem = out_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i, c));
                    WriteOutputBits(reinterpret_cast<uint16_t *>(ctx.out_gm) + out_elem, 0, val);
                }
            }
        }

        if (ctx.tiling->has_lse != 0 && ctx.lse_gm != nullptr) {
            for (uint32_t g = 0; g < ctx.cur_heads; ++g) {
                const int64_t hq_curr = hkv_idx * static_cast<int64_t>(ctx.gqa_ratio) + static_cast<int64_t>(ctx.cur_g0) + g;
                if (hq_curr >= ctx.hq) break;

                for (uint32_t i = 0; i < ctx.cur_q_len; ++i) {
                    const uint32_t r = g * ctx.cur_q_len + i;
                    if (r >= MAX_PACK_ROWS) break;

                    float lse_corr_val = 0.0f;
                    bool has_corr = false;
                    if (ctx.tiling->smooth_k != 0) {
                        const uint16_t *q_row = reinterpret_cast<const uint16_t *>(ctx.query_gm) +
                                                out_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i, 0u));
                        lse_corr_val = quant::LseCorrDot(ctx.d, [&](uint32_t c) { return ReadInputBits(q_row, c); },
                                                         ctx.km_self);
                        has_corr = true;
                    }
                    const float lse_host = OnlineSoftmaxTransformer::ConvertLseToHost(
                        row_states_[r].lse, ctx.tiling->sm_scale_log2e * INV_LOG2E,
                        lse_corr_val, has_corr, S_FP8_OFFSET);

                    const int64_t lse_offset = lse_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i)) *
                                               static_cast<int64_t>(sizeof(float));
                    *reinterpret_cast<float *>(ctx.lse_gm + lse_offset) = lse_host;
                }
            }
        }
#endif
    }
};

/**
 * \brief Vector 1 协同流水主过程
 */
template <uint32_t DTYPE_MODE>
__aicore__ inline __attribute__((always_inline)) void ProcessVec1(
    const TaskContext &ctx,
    AttentionVec1Softmax<DTYPE_MODE> &softmax,
    AttentionVec1Output<DTYPE_MODE> &output)
{
    softmax.ResetRowStates(ctx.TotalRows());
    output.ResetAccumulator(ctx);

    const uint32_t active_kv_blocks = ctx.active_kv_blocks;
    for (uint32_t k = 0; k < active_kv_blocks; ++k) {
        bool is_ping = ((k % 2) == 0);
        uint32_t block_k = ctx.CurBlockK(k);

#if SAGE_ATTN_DEVICE_COMPILE
        if (k > 0) {
            AttentionSync::Vec1WaitPartialReady();
            output.MergePartialRows(ctx);
            AttentionSync::Vec1ReleasePartialFree();
        }
#endif

        if (k >= 2) {
            AttentionSync::Vec1WaitSoftmaxFree(is_ping);
        }
        AttentionSync::Vec1WaitQkReady();

#if SAGE_ATTN_DEVICE_COMPILE
        if (k == 0) {
            softmax.StageV1ScaleStrips(ctx);
        }
        softmax.StageKScaleStrip(ctx, k);
#endif

        softmax.ProcessBlock(ctx, k, is_ping);

#if !SAGE_ATTN_DEVICE_COMPILE
        output.AccumulatePvBlockTwin(ctx, k, block_k, softmax.GetRowStates());
#endif

        // 仅在后续拍仍有 QK dot32 时释放许可（Cube 仅等待 N-1 拍，保证跨任务净余量为零）
        if (k + 1 < active_kv_blocks) {
            AttentionSync::Vec1ReleaseDotFree();
        }
        AttentionSync::Vec1SignalSoftmaxReady(is_ping);
    }

#if SAGE_ATTN_DEVICE_COMPILE
    if (active_kv_blocks > 0) {
        AttentionSync::Vec1WaitPartialReady();
        output.MergePartialRows(ctx);
        // 注意：Drain 拍为最后一块 partial，Cube 后续不再发射 PV，不再需要释放 PartialFree 许可
    }
#endif

    AttentionSync::Vec1WaitPvDone();
    output.FinalizeAndStore(ctx, softmax.GetRowStates());
    if (active_kv_blocks >= 1) {
        AttentionSync::Vec1DrainSoftmaxPingFree();
    }
    if (active_kv_blocks >= 2) {
        AttentionSync::Vec1DrainSoftmaxPongFree();
    }
}

/**
 * \brief SageAttention2 1C2V 多核协同内核调度器
 */
template <uint32_t DTYPE_MODE>
class SageAttnKernel {
public:
    static constexpr uint32_t kDtype = NormDtypeMode(DTYPE_MODE);

    __aicore__ inline __attribute__((always_inline)) void Init(
        __gm__ uint8_t *query,
        __gm__ uint8_t *key,
        __gm__ uint8_t *value,
        __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse,
        __gm__ uint8_t *workspace,
        const SageAttention2TilingData *__restrict tilingData)
    {
        ctx_.tiling = tilingData;
        ctx_.query_gm = query;
        ctx_.key_gm = key;
        ctx_.value_gm = value;
        ctx_.out_gm = attentionOut;
        ctx_.lse_gm = softmaxLse;
        ctx_.workspace_gm = workspace;

#if SAGE_ATTN_DEVICE_COMPILE
        ctx_.core_idx = static_cast<uint32_t>(GetBlockIdx());
        if ASCEND_IS_AIV {
            ctx_.sub_core_idx = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
            ctx_.core_idx /= 2;
        } else {
            ctx_.sub_core_idx = 0u;
        }
#else
        ctx_.core_idx = 0u;
        ctx_.sub_core_idx = 0u;
#endif

        ctx_.b = ctx_.tiling->b;
        ctx_.hq = ctx_.tiling->hq;
        ctx_.hkv = ctx_.tiling->hkv;
        ctx_.sq = ctx_.tiling->sq;
        ctx_.sk = ctx_.tiling->sk;
        ctx_.d = ctx_.tiling->d_pad;
        ctx_.gqa_ratio = (ctx_.hkv > 0) ? (ctx_.hq / ctx_.hkv) : 1;

        ctx_.m_per_core = ctx_.tiling->m_per_core;
        ctx_.block_n = ctx_.tiling->block_n;
        ctx_.k_step = (ctx_.tiling->k_step != 0u) ? ctx_.tiling->k_step : DEFAULT_K_STEP;

        const uint32_t pack_cap = (ctx_.m_per_core > 0u) ? (MAX_PACK_ROWS / ctx_.m_per_core) : 1u;
        ctx_.pack_heads = (ctx_.gqa_ratio < pack_cap) ? ctx_.gqa_ratio : pack_cap;
        ctx_.gqa_chunks = (ctx_.pack_heads > 0u) ? ::CeilDiv(ctx_.gqa_ratio, ctx_.pack_heads) : 1u;

        uint32_t q_blks_per_head = ::CeilDiv(ctx_.sq, ctx_.m_per_core);
        uint32_t total_tasks = ctx_.b * ctx_.hkv * ctx_.gqa_chunks * q_blks_per_head;
        uint32_t cores = ctx_.tiling->core_num_used;
        // Host 判据 b·hkv 偶 ⇒ total_tasks 恒偶；此处防御性复核奇偶与极小域，
        // host 判据若回归则静默退化为全量边界，绝不进入丢尾任务的契约域外分支。
        bool use_die = (ctx_.tiling->enable_die_affinity != 0) && (cores % 2u == 0u) &&
                       (total_tasks >= 2u) && (total_tasks % 2u == 0u);

        task_end_ = DieTaskEnd(total_tasks, ctx_.core_idx, cores, use_die);
        if (ctx_.core_idx >= task_end_) {
            is_active_core_ = false;
            return;
        }
        is_active_core_ = true;

        uint64_t initial_task = use_die ? DieTask(ctx_.core_idx, total_tasks, cores) : ctx_.core_idx;
        DecodeTask(initial_task, q_blks_per_head);

#if !SAGE_ATTN_DEVICE_COMPILE
        if (sim_mem_ == nullptr) {
            sim_mem_ = std::make_unique<SimMemory>();
        }
        ctx_.sim_mem = sim_mem_.get();
        ctx_.km_self = km_self_;
#endif
    }

    __aicore__ inline __attribute__((always_inline)) void DecodeTask(uint64_t task_idx, uint32_t q_blks_per_head)
    {
        int64_t t = static_cast<int64_t>(task_idx);
        int64_t q_blk_id = t % q_blks_per_head;
        int64_t rem = t / q_blks_per_head;
        int64_t g_chunk = rem % ctx_.gqa_chunks;
        rem /= ctx_.gqa_chunks;
        int64_t hkv_id = rem % ctx_.hkv;
        int64_t b_id = rem / ctx_.hkv;

        ctx_.cur_b = static_cast<uint32_t>(b_id);
        ctx_.cur_hkv = static_cast<uint32_t>(hkv_id);
        ctx_.cur_q_blk = static_cast<uint32_t>(q_blk_id);

        ctx_.cur_q_start = ctx_.cur_q_blk * ctx_.m_per_core;
        ctx_.cur_q_len = (ctx_.cur_q_start + ctx_.m_per_core <= ctx_.sq) ? ctx_.m_per_core : (ctx_.sq - ctx_.cur_q_start);
        ctx_.cur_g0 = static_cast<uint32_t>(g_chunk) * ctx_.pack_heads;
        ctx_.cur_heads = (ctx_.cur_g0 + ctx_.pack_heads <= ctx_.gqa_ratio) ? ctx_.pack_heads : (ctx_.gqa_ratio - ctx_.cur_g0);

        const uint32_t total_kv_blocks = ::CeilDiv(ctx_.sk, ctx_.block_n);
        ctx_.active_kv_blocks = (ctx_.tiling->is_causal != 0)
            ? ::Min(total_kv_blocks, ::CeilDiv(ctx_.cur_q_start + ctx_.cur_q_len, ctx_.block_n))
            : total_kv_blocks;

#if !SAGE_ATTN_DEVICE_COMPILE
        if (ctx_.tiling->smooth_k != 0) {
            const Catlass::SageAttn::GmBhsdLayout km_layout =
                Catlass::SageAttn::MakeGmBhsdLayout(ctx_.b, ctx_.hkv, ctx_.sk, ctx_.d, ctx_.tiling->tensor_layout == LAYOUT_HND);
            const uint16_t *k_bits = reinterpret_cast<const uint16_t *>(ctx_.key_gm);
            const int64_t km_b = static_cast<int64_t>(ctx_.cur_b);
            const int64_t km_hkv = static_cast<int64_t>(ctx_.cur_hkv);
            for (uint32_t di = 0; di < ctx_.d; ++di) {
                km_self_[di] = quant::KeyMeanElement(ctx_.sk, kDtype, [&](uint32_t si) {
                    return AttentionVec0Quant<DTYPE_MODE>::ReadInputBits(
                        k_bits + km_layout(tla::MakeCoord(km_b, km_hkv, static_cast<int64_t>(si), 0u)), di);
                });
            }
        }
#endif
    }

    __aicore__ inline __attribute__((always_inline)) void Process()
    {
        if (!is_active_core_) {
            return;
        }

        uint32_t q_blks_per_head = ::CeilDiv(ctx_.sq, ctx_.m_per_core);
        uint32_t total_tasks = ctx_.b * ctx_.hkv * ctx_.gqa_chunks * q_blks_per_head;
        uint32_t cores = ctx_.tiling->core_num_used;
        // Host 判据 b·hkv 偶 ⇒ total_tasks 恒偶；此处防御性复核奇偶与极小域，
        // host 判据若回归则静默退化为全量边界，绝不进入丢尾任务的契约域外分支。
        bool use_die = (ctx_.tiling->enable_die_affinity != 0) && (cores % 2u == 0u) &&
                       (total_tasks >= 2u) && (total_tasks % 2u == 0u);
        const uint64_t task_end = DieTaskEnd(total_tasks, ctx_.core_idx, cores, use_die);

        for (uint64_t virtual_task = ctx_.core_idx; virtual_task < task_end; virtual_task += cores) {
            if (virtual_task > ctx_.core_idx) {
                uint64_t mapped_task = use_die ? DieTask(virtual_task, total_tasks, cores) : virtual_task;
                DecodeTask(mapped_task, q_blks_per_head);
            }

#if SAGE_ATTN_DEVICE_COMPILE
            if ASCEND_IS_AIC {
                CubeBlockParams p{ctx_.TotalRows(), ctx_.block_n, ctx_.d};
                cube_.Process(p, ctx_.active_kv_blocks, ctx_.tiling->l1_q_offset,
                              SlotV1Dot32::Offset(), SlotV1PvPartial::Offset());
            } else {
                if (ctx_.sub_core_idx == 0) {
                    vec0_.Process(ctx_);
                } else {
                    ProcessVec1(ctx_, softmax_, output_);
                }
            }
#else
            vec0_.Process(ctx_);
            CubeBlockParams p{ctx_.TotalRows(), ctx_.block_n, ctx_.d};
            cube_.Process(p, ctx_.active_kv_blocks, ctx_.tiling->l1_q_offset,
                          SlotV1Dot32::Offset(), SlotV1PvPartial::Offset());
            ProcessVec1(ctx_, softmax_, output_);
#endif
        }
    }

private:
    TaskContext ctx_;
    uint64_t task_end_{0};
    bool is_active_core_{false};

    AttentionCube cube_;
    AttentionVec0Quant<DTYPE_MODE> vec0_;
    AttentionVec1Softmax<DTYPE_MODE> softmax_;
    AttentionVec1Output<DTYPE_MODE> output_;

#if !SAGE_ATTN_DEVICE_COMPILE
    std::unique_ptr<SimMemory> sim_mem_{nullptr};
    float km_self_[128]{0.0f};
#endif
};

/**
 * \brief 算子发射器 SageAttnLaunch
 */
struct SageAttnLaunch {
    template <uint32_t DTYPE_MODE>
    static __aicore__ inline __attribute__((always_inline)) void Run(
        __gm__ uint8_t *query,
        __gm__ uint8_t *key,
        __gm__ uint8_t *value,
        __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse,
        __gm__ uint8_t *workspace,
        const SageAttention2TilingData *tiling_data)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        SageAttnKernel<DTYPE_MODE> kernel;
        kernel.Init(query, key, value, attentionOut, softmaxLse, workspace, tiling_data);
#else
        (void)workspace;
        SageAttnKernel<DTYPE_MODE> kernel;
        kernel.Init(query, key, value, attentionOut, softmaxLse, nullptr, tiling_data);
#endif
        kernel.Process();
    }

    static __aicore__ inline __attribute__((always_inline)) void RunByDtype(
        uint32_t dtype_mode,
        __gm__ uint8_t *query,
        __gm__ uint8_t *key,
        __gm__ uint8_t *value,
        __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse,
        __gm__ uint8_t *workspace,
        const SageAttention2TilingData *tiling_data)
    {
        if (NormDtypeMode(dtype_mode) == kTplDtypeBf16) {
            Run<kTplDtypeBf16>(query, key, value, attentionOut, softmaxLse, workspace, tiling_data);
        } else {
            Run<kTplDtypeFp16>(query, key, value, attentionOut, softmaxLse, workspace, tiling_data);
        }
    }
};

} // namespace sage_attn
