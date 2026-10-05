#pragma once

#include <cstdint>
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/online_softmax.h"

#if SAGE_ATTN_DEVICE_COMPILE
#ifndef ENABLE_CV_COMM_VIA_SSBUF
#define ENABLE_CV_COMM_VIA_SSBUF 1
#endif
#include "kernel_operator.h"
#endif

#include "sage_attention2_template_tiling_key.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"

namespace sage_attn {

#if SAGE_ATTN_DEVICE_COMPILE
using namespace AscendC;
#endif

/**
 * \brief 编译期安全守卫模板 StaticSlot
 * 强制在编译期约束 OFFSET 和 BYTES 满足昇腾硬件 32 字节对齐，
 * 把一切未对齐与越界错误掐死在编译期。
 */
template <uint32_t OFFSET_, uint32_t BYTES_>
struct StaticSlot {
    static constexpr uint32_t OFFSET = OFFSET_;
    static constexpr uint32_t BYTES = BYTES_;

    static_assert(OFFSET % 32 == 0, "StaticSlot OFFSET must be 32-byte aligned!");
    static_assert(BYTES % 32 == 0, "StaticSlot BYTES must be 32-byte aligned!");

    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) static constexpr uint32_t Offset() { return OFFSET; }
};

// --- Vector 0 UB 槽编译期守卫 ---
using SlotV0InputPing   = StaticSlot<Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_OFFSET, Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_BYTES>;
using SlotV0RowMax      = StaticSlot<Catlass::SageAttn::Vector0UbBudget::QUANT_ROWMAX_OFFSET, 512>;
using SlotV0Scales      = StaticSlot<Catlass::SageAttn::Vector0UbBudget::QUANT_SCALES_OFFSET, 16384>;
using SlotV0Km          = StaticSlot<Catlass::SageAttn::Vector0UbBudget::QUANT_KM_OFFSET, 512>;
using SlotV0StageStrip  = StaticSlot<Catlass::SageAttn::Vector0UbBudget::QUANT_STAGE_STRIP_OFFSET, 512>;
using SlotV0VMin        = StaticSlot<Catlass::SageAttn::Vector0UbBudget::QUANT_VMIN_OFFSET, 512>;
using SlotV0VSum        = StaticSlot<Catlass::SageAttn::Vector0UbBudget::QUANT_VSUM_OFFSET, 512>;
using SlotV0NzOutput    = StaticSlot<Catlass::SageAttn::Vector0UbBudget::NZ_OUTPUT_OFFSET, 512>;

// --- Vector 1 UB 槽编译期守卫 ---
using SlotV1OAccum      = StaticSlot<Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET, Catlass::SageAttn::Vector1UbBudget::O_ACCUM_BYTES>;
using SlotV1FixpipeRecv = StaticSlot<Catlass::SageAttn::Vector1UbBudget::FIXPIPE_RECV_OFFSET, Catlass::SageAttn::Vector1UbBudget::FIXPIPE_RECV_BYTES>;
using SlotV1Dot32       = StaticSlot<Catlass::SageAttn::Vector1UbBudget::A7_DOT32_OFFSET, Catlass::SageAttn::Vector1UbBudget::A7_DOT32_BYTES>;
using SlotV1PvPartial   = StaticSlot<Catlass::SageAttn::Vector1UbBudget::A7_PV_PARTIAL_OFFSET, Catlass::SageAttn::Vector1UbBudget::A7_PV_PARTIAL_BYTES>;
using SlotV1RowMax      = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_ROW_MAX_OFFSET, 512>;
using SlotV1RowSum      = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_ROW_SUM_OFFSET, 512>;
using SlotV1KsRow       = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_KS_ROW_OFFSET, 512>;
using SlotV1QsRow       = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_QS_ROW_OFFSET, 512>;
using SlotV1VsChan      = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_VS_CHAN_OFFSET, 512>;
using SlotV1VmChan      = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_VM_CHAN_OFFSET, 512>;
using SlotV1BlockMax    = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_CORR_ROW_OFFSET, 512>;
using SlotV1LseRow      = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_LSE_ROW_OFFSET, 512>;
using SlotV1BlockSum    = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_M_LN_ROW_OFFSET, 512>;
using SlotV1RsFactor    = StaticSlot<Catlass::SageAttn::Vector1UbBudget::V1_RS_FACTOR_ROW_OFFSET, 512>;

// --- L1 Bank 双缓冲槽编译期守卫 ---
using SlotL1KvPing      = StaticSlot<Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET, Catlass::SageAttn::L1DoubleBankContract::KV_SLOT_BYTES>;
using SlotL1KvPong      = StaticSlot<Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET, Catlass::SageAttn::L1DoubleBankContract::KV_SLOT_BYTES>;
using SlotL1SoftmaxPPing = StaticSlot<Catlass::SageAttn::L1DoubleBankContract::P_PING_OFFSET, Catlass::SageAttn::L1DoubleBankContract::P_SLOT_BYTES>;
using SlotL1SoftmaxPPong = StaticSlot<Catlass::SageAttn::L1DoubleBankContract::P_PONG_OFFSET, Catlass::SageAttn::L1DoubleBankContract::P_SLOT_BYTES>;

#if !SAGE_ATTN_DEVICE_COMPILE
/**
 * KT/宿主标量基线暂存区（SimMemory）：以固定大小数组成员承载真实硬件的 L1/UB/L0 分域
 */
struct SimMemory {
    uint8_t l1_mem[512 * 1024]{0};
    uint8_t ub_mem[512 * 1024]{0};
    float o_accum[128 * 128]{0.0f};
    float logits[128 * 128]{0.0f};
    float q_scales[MAX_GQA_RATIO * kQScaleSlotStride]{0.0f};
    float k_scales[kKScaleSlotStride]{0.0f};
    float v_scales[256]{0.0f};
    float v_means[256]{0.0f};
    uint8_t stage_strip[STAGE_CORE_STRIDE]{0};
};
#endif

#if SAGE_ATTN_DEVICE_COMPILE
using TilingPtr = const __gm__ SageAttention2TilingData *;
#else
using TilingPtr = const SageAttention2TilingData *;
#endif

/**
 * \brief 统一任务上下文结构 TaskContext
 * 承载当前分核任务的只读元数据与运行时指针，零运行时拷贝
 */
struct TaskContext {
    TilingPtr tiling{nullptr};
    __gm__ uint8_t *query_gm{nullptr};
    __gm__ uint8_t *key_gm{nullptr};
    __gm__ uint8_t *value_gm{nullptr};
    __gm__ uint8_t *out_gm{nullptr};
    __gm__ uint8_t *lse_gm{nullptr};
    __gm__ uint8_t *workspace_gm{nullptr};

    uint32_t core_idx{0};
    uint32_t sub_core_idx{0};

    uint32_t b{0};
    uint32_t hq{0};
    uint32_t hkv{0};
    uint32_t sq{0};
    uint32_t sk{0};
    uint32_t d{0};
    uint32_t gqa_ratio{1};
    uint32_t pack_heads{1};
    uint32_t gqa_chunks{1};
    uint32_t cur_g0{0};
    uint32_t cur_heads{1};

    uint32_t m_per_core{64};
    uint32_t block_n{64};
    uint32_t k_step{DEFAULT_K_STEP};

    uint32_t cur_b{0};
    uint32_t cur_hkv{0};
    uint32_t cur_q_blk{0};
    uint32_t cur_q_start{0};
    uint32_t cur_q_len{0};
    uint32_t active_kv_blocks{0};

    uint32_t is_causal{0};
    uint32_t is_hnd{1};
    uint32_t smooth_k{0};
    uint32_t smooth_v{0};
    float sm_scale_log2e{0.0f};
    uint32_t d_og{0};
    uint32_t has_lse{0};
    uint32_t qk_quant_gran{0};
    uint32_t pv_accum_dtype{0};
    uint32_t l1_q_offset{0};
    uint64_t workspace_vm_offset{0};
    uint32_t tensor_layout{0};
    float v_scale_max{0.0f};

#if !SAGE_ATTN_DEVICE_COMPILE
    SimMemory *sim_mem{nullptr};
    float *km_self{nullptr};
#endif

    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) uint32_t TotalRows() const
    {
        return cur_heads * cur_q_len;
    }

    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) uint32_t CurBlockK(uint32_t cur_k) const
    {
        const int64_t sk_start = static_cast<int64_t>(cur_k) * block_n;
        return (sk_start + block_n <= sk) ? block_n : static_cast<uint32_t>(sk - sk_start);
    }

    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) __gm__ uint8_t *StageStrip() const
    {
#if SAGE_ATTN_DEVICE_COMPILE
        return workspace_gm + static_cast<uint64_t>(core_idx) * STAGE_CORE_STRIDE;
#else
        if (workspace_gm != nullptr) {
            return workspace_gm + static_cast<uint64_t>(core_idx) * STAGE_CORE_STRIDE;
        }
        return (sim_mem != nullptr) ? sim_mem->stage_strip : nullptr;
#endif
    }

    [[nodiscard]] __aicore__ inline __attribute__((always_inline)) uint32_t SoftmaxSlotOffset(bool is_ping) const
    {
        const uint32_t tile_bytes = TotalRows() * block_n;
        return SlotV1FixpipeRecv::Offset() + (is_ping ? 0u : tile_bytes);
    }
};

} // namespace sage_attn
