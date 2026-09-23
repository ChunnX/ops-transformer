/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Private, default-off diagnostic for the fixed ORI_CMP TND attention case.
 */
#ifndef SMLA_ATTENTION_ABLATION_H
#define SMLA_ATTENTION_ABLATION_H

#include "smla_experiment_config.h"
#include "kernel_operator.h"
#include "smla_metadata_common.h"

namespace SMLAKernel {
struct AttentionAblation {
    // Bit positions: W2Q, RQ, RK, RP, RV, W3KV, W3P. One means REMOVE transfer.
    enum : uint32_t { W2Q = 0, RQ = 1, RK = 2, RP = 3, RV = 4, W3KV = 5, W3P = 6 };
    static constexpr uint32_t CONFIG = 900;
    static constexpr uint32_t OUTPUT = 928;
    static constexpr uint32_t PRIME_TASKS = 32;
    static_assert(optiling::AIC_CORE_MAX_NUM * optiling::FA_METADATA_SIZE +
                  optiling::AIV_CORE_MAX_NUM * optiling::FD_METADATA_SIZE <= CONFIG,
                  "Diagnostic must not overlap functional metadata");
#if SMLA_ATTENTION_ABLATION
    uint32_t mask = 0;
#endif
#if SMLA_ATTENTION_ABLATION && SMLA_ATTENTION_AUDIT
    uint32_t mode = 0; // 0 formal (no counters/timestamps), 1 audit, 2 sampled completion trace
    uint32_t sampleTask = 100;
    uint32_t actor = 0;
    bool selected = false;
    __gm__ uint32_t *metadata = nullptr;
    mutable uint32_t counts[16] = {};
    mutable uint64_t ticks[8] = {};
#endif

    __aicore__ inline void Init(__gm__ uint8_t *meta, uint32_t core, uint32_t sub, bool legal)
    {
#if SMLA_ATTENTION_ABLATION
        AscendC::GlobalTensor<uint32_t> cfg;
        cfg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(meta));
        if (!legal || cfg.GetValue(CONFIG) != 0x534D4C41U || cfg.GetValue(CONFIG + 7) != 0x12345678U) {
            return;
        }
        uint32_t m = cfg.GetValue(CONFIG + 1);
        uint32_t c = cfg.GetValue(CONFIG + 2);
        uint32_t t = cfg.GetValue(CONFIG + 3);
        uint32_t md = cfg.GetValue(CONFIG + 4);
        if (m > 127 || c >= 32 || t < PRIME_TASKS || t >= 600 || md > 2) {
            return;
        }
        mask = m;
#if SMLA_ATTENTION_AUDIT
        mode = md;
        sampleTask = t;
        selected = core == c;
        metadata = reinterpret_cast<__gm__ uint32_t *>(meta);
        if ASCEND_IS_AIC { actor = 0; } else { actor = 1 + sub; }
#endif
#endif
    }

    __aicore__ inline bool Transfer(int64_t task, uint32_t bit) const
    {
#if SMLA_ATTENTION_ABLATION
        bool skip = task >= PRIME_TASKS && (mask & (1U << bit)) != 0;
#if SMLA_ATTENTION_AUDIT
        if (selected && mode != 0) {
            ++counts[bit + (skip ? 8 : 0)];
        }
#endif
        return !skip;
#else
        return true;
#endif
    }

    __aicore__ inline void Tile(bool full) const
    {
#if SMLA_ATTENTION_ABLATION && SMLA_ATTENTION_AUDIT
        if (selected && mode != 0) { ++counts[full ? 7 : 15]; }
#endif
    }

    __aicore__ inline bool Sample(int64_t task) const
    {
#if SMLA_ATTENTION_ABLATION && SMLA_ATTENTION_AUDIT
        return selected && mode == 2 && task == sampleTask;
#else
        return false;
#endif
    }

    template <AscendC::HardEvent EVENT>
    __aicore__ inline void Mark(int64_t task, uint32_t point, bool join = true) const
    {
#if SMLA_ATTENTION_ABLATION && SMLA_ATTENTION_AUDIT
        if (Sample(task)) {
            // Event 7 is private to these paired scalar joins, not a cross-core flag.
            if (join) {
                AscendC::SetFlag<EVENT>(7);
                AscendC::WaitFlag<EVENT>(7);
            }
            ticks[point] = AscendC::GetSystemCycle();
        }
#endif
    }

    __aicore__ inline void Finish() const
    {
#if SMLA_ATTENTION_ABLATION && SMLA_ATTENTION_AUDIT
        if (selected && mode != 0) {
            // Each actor owns a separate aligned 128-byte line in the unused tail.
            AscendC::GlobalTensor<uint32_t> out;
            out.SetGlobalBuffer(metadata + OUTPUT + actor * 32);
            for (uint32_t i = 0; i < 16; ++i) { out.SetValue(i, counts[i]); }
            for (uint32_t i = 0; i < 8; ++i) {
                out.SetValue(16 + 2 * i, static_cast<uint32_t>(ticks[i]));
                out.SetValue(17 + 2 * i, static_cast<uint32_t>(ticks[i] >> 32));
            }
        }
#endif
    }
};
} // namespace SMLAKernel
#endif
