/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SMLA_EXPERIMENT_CONFIG_H
#define SMLA_EXPERIMENT_CONFIG_H

// Shared by the AI Core kernel and its AICPU metadata producer.
// 0 original; 1 split reference; 2 control; 3 balanced; 4 slot1 phase; 5 S64 balanced.
// Default 3 enables the accepted E1 split path and E2 cross-half layout under the existing TND guards.
#ifndef SMLA_L1_LAYOUT_MODE
#define SMLA_L1_LAYOUT_MODE 3
#endif

// Private diagnostic package only; a magic metadata marker is also required.
#ifndef SMLA_BANK_DIAGNOSTIC
#define SMLA_BANK_DIAGNOSTIC 0
#endif

// Actual-attention transfer ablation. Never enable in a production package.
#ifndef SMLA_ATTENTION_ABLATION
#define SMLA_ATTENTION_ABLATION 0
#endif

// Build counters/timestamps in a separate audit package, never formal timing.
#ifndef SMLA_ATTENTION_AUDIT
#define SMLA_ATTENTION_AUDIT 0
#endif

// Opposite D256 copy order for the two AIVs; whole-slot ownership is unchanged.
#ifndef SMLA_AIV_HALF_ORDER
#define SMLA_AIV_HALF_ORDER 0
#endif

// Private candidate: fuse the final non-split PV update with its output cast.
#ifndef SMLA_FINAL_UPDATE_CAST
#define SMLA_FINAL_UPDATE_CAST 0
#endif

// Collect per-row softmax max/sum in registers within the existing fused VF.
#ifndef SMLA_SOFTMAX_REGISTER_STATS
#define SMLA_SOFTMAX_REGISTER_STATS 0
#endif

// Resident P has its own lifetime event; do not wait for quarter-0 L0B credit.
#ifndef SMLA_P_EARLY_LOAD
#define SMLA_P_EARLY_LOAD 0
#endif

// 0 off; 1 layout/event control (only Qa resident); 2 all non-shared Q quarters resident.
// Both enabled modes keep two L0B slots and share one Q quarter with P.
#ifndef SMLA_L0A_QBP_MODE
#define SMLA_L0A_QBP_MODE 0
#endif

// 1: Qb/P at [16,32) KiB; 2: Qc/P at [32,48) KiB. Q quarter addresses stay fixed.
#ifndef SMLA_L0A_QBP_SHARED_QUARTER
#define SMLA_L0A_QBP_SHARED_QUARTER 1
#endif

// Independent candidate: load the preceding task's P after issuing all four QK quarters.
#ifndef SMLA_L0A_QBP_PREFETCH
#define SMLA_L0A_QBP_PREFETCH 0
#endif

// Direct cached-address pair issue for full sparse tiles; invalid rows still fall back.
#ifndef SMLA_KV_SCALAR_DIRECT
#define SMLA_KV_SCALAR_DIRECT 0
#endif

// Independent ND row loads precede both NZ stores; data layout is unchanged.
#ifndef SMLA_KV_NZ_LOAD_PAIR
#define SMLA_KV_NZ_LOAD_PAIR 0
#endif

#endif // SMLA_EXPERIMENT_CONFIG_H
