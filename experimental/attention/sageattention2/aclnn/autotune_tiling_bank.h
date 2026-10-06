/*!
 * \file autotune_tiling_bank.h
 * \brief 真机实测固化的针对 Ascend 950PR 的最优 Tiling 策略知识库
 *        （P-01..P-07，a31b913f9 固化；调优会话经 build_opp.sh SAGE_AUTOTUNE_HOOKS=1
 *        的 SAGEATTN_FORCE_* / SAGEATTN_TUNE_CASE 覆盖钩子驱动，见
 *        aclnn/sage_attn_host_tiling.h 门控注释）
 * \note 生产策略契约与证据链校准：
 *       条目数值最终须由真机网格自动固化（apply_best_tiling.py）。
 *       当前 P-01..P-07 档位（除 GQA P-05=32u 外）m_per_core 设为 128u 的证据链：
 *       - 剖面推断方程：两份真实 msprof 数据确立联立方程：
 *         m=32 时代 R + 128*q = 31.58ms（128 个分块任务），
 *         22.4ms 测次 R + 32*q = 12.18ms（32 个分块任务）。
 *         零标量直通消除标量流水停顿，但无法解释 aiv_vec 耗时由 31.58ms 降至 12.18ms；
 *         唯一自洽的理论解释为任务数 128 降至 32，即该测次已在 m=128 下通过真机运行，
 *         去除了 75% 的内层冗余量化开销。
 *       - 严谨性勘误：86.66ms 系 P-02（8K 序列）实测值，非 P-01 旧版数据，不可混淆。
 *       - 遗留闭环：当前实证表明 m=128 显著优于 m=32，但相对于 m=64 的全局最优性
 *         仍处于 PRESET_PENDING_REAL_NPU_GRID 状态，待真机网格重跑（12_run_autotune.sh）
 *         与全量 commit hash 解析链闭环后最终固化。
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
