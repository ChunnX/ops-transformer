/*!
 * \file autotune_tiling_bank.h
 * \brief 真机实测固化的针对 Ascend 950PR 的最优 Tiling 策略知识库
 *        （P-01..P-07，a31b913f9 固化；调优会话经 build_opp.sh SAGE_AUTOTUNE_HOOKS=1
 *        的 SAGEATTN_FORCE_* / SAGEATTN_TUNE_CASE 覆盖钩子驱动，见
 *        aclnn/sage_attn_host_tiling.h 门控注释）
 * \note 生产策略契约：条目数值最终须由真机网格自动固化（apply_best_tiling.py）。
 *       当前 P-01..P-07 档位（除 GQA P-05=32u 外）m_per_core 设为 128u 为消除 75% 冗余
 *       KV 量化与 GM 重复加载之微架构推演预置候选值（状态：PRESET_PENDING_REAL_NPU_GRID），
 *       待开机执行 12_run_autotune.sh 网格搜索后自动校验并固化覆盖。
 *       仅保留消费调优值的四个字段与查询/匹配接口。
 */

#ifndef SAGE_ATTN_AUTOTUNE_TILING_BANK_H
#define SAGE_ATTN_AUTOTUNE_TILING_BANK_H
#pragma once

#include <cstdint>
#include <string_view>

namespace sage_attn {
namespace autotune {

struct TunedTilingEntry {
    const char* case_id;
    uint32_t m_per_core;
    uint32_t block_n;
    uint32_t k_step;
    bool use_die_grouped;
};

static constexpr TunedTilingEntry kAscend950TunedBank[] = {
    {"P-01", 128u, 128u, 16u, true},
    {"P-02", 128u, 128u, 16u, true},
    {"P-03", 128u, 128u, 16u, true},
    {"P-04", 128u, 128u, 16u, true},
    {"P-05", 32u, 128u, 16u, true},
    {"P-06", 128u, 128u, 16u, true},
    {"P-07", 128u, 128u, 16u, true},
};

[[nodiscard]] inline const TunedTilingEntry* QueryTunedTiling(std::string_view case_id) {
    for (const auto& e : kAscend950TunedBank) {
        if (case_id == e.case_id) {
            return &e;
        }
    }
    return nullptr;
}

[[nodiscard]] inline const TunedTilingEntry* QueryTunedTiling(const char* case_id) {
    if (case_id == nullptr) return nullptr;
    return QueryTunedTiling(std::string_view{case_id});
}

[[nodiscard]] inline const char* MatchCaseId(
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

[[nodiscard]] inline const TunedTilingEntry* QueryTunedTiling(
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

} // namespace autotune
} // namespace sage_attn

#endif // SAGE_ATTN_AUTOTUNE_TILING_BANK_H
