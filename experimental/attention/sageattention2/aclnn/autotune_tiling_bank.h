/*!
 * \file autotune_tiling_bank.h
 * \brief 由 autotune 调优引擎自动生成的针对 Ascend 950PR 的最优 Tiling 策略知识库
 * \note 严禁手工随意修改，请通过 benchmark/autotune/autotune_runner.py 重新调优生成
 */

#ifndef SAGE_ATTN_AUTOTUNE_TILING_BANK_H
#define SAGE_ATTN_AUTOTUNE_TILING_BANK_H

#include <cstdint>
#include <cstddef>

namespace sage_attn {
namespace autotune {

struct TunedTilingEntry {
    const char* case_id;
    uint32_t m_per_core;
    uint32_t block_n;
    uint32_t k_step;
    float skip_scale_tau;
    bool causal_early_exit;
    bool use_die_grouped;
    float expected_speedup;
};

struct TunedShapeKey {
    uint32_t b{0};
    uint32_t sq{0};
    uint32_t sk{0};
    uint32_t hq{0};
    uint32_t hkv{0};
    uint32_t d{0};
    int dtype_mode{0}; // 1=fp16, 2=bf16
    bool is_causal{false};
};

static constexpr TunedTilingEntry kAscend950TunedBank[] = {
    {"P-01", 32u, 128u, 16u, 6.0f, false, true, 2.128f},
    {"P-02", 32u, 128u, 16u, 6.0f, false, true, 2.128f},
    {"P-03", 32u, 128u, 16u, 6.0f, true, true, 3.274f},
    {"P-04", 32u, 128u, 16u, 6.0f, false, true, 2.128f},
    {"P-05", 32u, 128u, 16u, 6.0f, false, true, 2.128f},
    {"P-06", 32u, 128u, 16u, 6.0f, false, true, 2.128f},
    {"P-07", 32u, 128u, 16u, 6.0f, false, true, 2.128f},
};

constexpr size_t kNumTunedEntries = sizeof(kAscend950TunedBank) / sizeof(kAscend950TunedBank[0]);

inline const TunedTilingEntry* QueryTunedTiling(const char* case_id) {
    if (case_id == nullptr) return nullptr;
    for (size_t i = 0; i < kNumTunedEntries; ++i) {
        const auto& e = kAscend950TunedBank[i];
        const char* p1 = e.case_id;
        const char* p2 = case_id;
        while (*p1 && *p2 && *p1 == *p2) { ++p1; ++p2; }
        if (*p1 == '\0' && *p2 == '\0') {
            return &e;
        }
    }
    return nullptr;
}

inline const char* MatchCaseId(
    uint32_t b,
    uint32_t sq,
    uint32_t sk,
    uint32_t hq,
    uint32_t hkv,
    uint32_t d,
    int dtype_mode,
    bool is_causal)
{
    if (d != 128 || sq != sk) {
        return nullptr;
    }
    if (b == 1 && hq == 32 && hkv == 32 && sq == 4096 && dtype_mode == 1 && !is_causal) {
        return "P-01";
    }
    if (b == 1 && hq == 32 && hkv == 32 && sq == 8192 && dtype_mode == 1 && !is_causal) {
        return "P-02";
    }
    if (b == 1 && hq == 32 && hkv == 32 && sq == 8192 && dtype_mode == 1 && is_causal) {
        return "P-03";
    }
    if (b == 2 && hq == 32 && hkv == 32 && sq == 8192 && dtype_mode == 1 && !is_causal) {
        return "P-04";
    }
    if (b == 1 && hq == 32 && hkv == 8 && sq == 4096 && dtype_mode == 1 && !is_causal) {
        return "P-05";
    }
    if (b == 1 && hq == 32 && hkv == 32 && sq == 4096 && dtype_mode == 2 && !is_causal) {
        return "P-06";
    }
    if (b == 1 && hq == 32 && hkv == 32 && sq == 16384 && dtype_mode == 1 && !is_causal) {
        return "P-07";
    }
    return nullptr;
}

inline const TunedTilingEntry* QueryTunedTiling(
    uint32_t b,
    uint32_t sq,
    uint32_t sk,
    uint32_t hq,
    uint32_t hkv,
    uint32_t d,
    int dtype_mode,
    bool is_causal)
{
    const char* cid = MatchCaseId(b, sq, sk, hq, hkv, d, dtype_mode, is_causal);
    if (cid != nullptr) {
        return QueryTunedTiling(cid);
    }
    return nullptr;
}

inline const TunedTilingEntry* QueryTunedTiling(const TunedShapeKey& shape)
{
    return QueryTunedTiling(
        shape.b, shape.sq, shape.sk, shape.hq, shape.hkv, shape.d, shape.dtype_mode, shape.is_causal);
}

} // namespace autotune
} // namespace sage_attn

#endif // SAGE_ATTN_AUTOTUNE_TILING_BANK_H
// R5-marker 01:07:14
