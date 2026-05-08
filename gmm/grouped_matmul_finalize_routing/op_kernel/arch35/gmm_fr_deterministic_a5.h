/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gmm_fr_deterministic_a5.h
 * \brief A5 platform deterministic deferred aggregation for GMMFR.
 *
 * A3 prototype: op_kernel/grouped_matmul_finalize_routing.h
 *   - FRDeterministic() line 653-687
 *   - VectorSync() line 622-649
 *
 * Core mechanism:
 *   1. SyncAll() - ensure all cores finished workspace writes
 *   2. Assign row ownership by outRow % coreNumVec
 *   3. Read workspace -> UB (queBind) -> SetAtomicAdd to yGm (single writer per row)
 *   4. SyncAll() - ensure all writes complete
 */

#ifndef GMM_FR_DETERMINISTIC_A5_H
#define GMM_FR_DETERMINISTIC_A5_H

#include "kernel_operator.h"

namespace GMMFRDeterministic {

using namespace AscendC;

// Consistent with A3 constants
constexpr uint32_t DETER_UB_SIZE = 12 * 1024;   // Deterministic UB buffer size (12KB)
constexpr uint32_t BUFFER_NUM = 2;

// Sliding window sync config (identical to A3 SyncConfig)
struct SyncConfig {
    uint64_t curM = 0;        // Current accumulated row offset
    uint64_t curGroup = 0;    // Current group index
    uint64_t curGroupM = 0;   // Current accumulated rows within group
    uint64_t lowBoundM = 0;   // Window lower bound
    uint64_t windowSize = 0;  // Window size = deterWorkspaceSize / (N * sizeof(float))
    uint64_t baseN = 0;       // N-dim block size, 128-aligned
};

/**
 * FRDeterministicA5 - A5 deterministic deferred aggregation core function
 *
 * A3 prototype: QuantGroupMatmul<P>::FRDeterministic()
 *               (grouped_matmul_finalize_routing.h line 653-687)
 *
 * Template params:
 *   DTYPE_OUT       - Output data type (float)
 *   ROW_INDEX_DTYPE - Row index data type (int64_t or int32_t)
 */
template <typename DTYPE_OUT, typename ROW_INDEX_DTYPE>
__aicore__ inline void FRDeterministicA5(
    SyncConfig& syncConfig,
    GlobalTensor<DTYPE_OUT>& deterBufferGm,
    GlobalTensor<DTYPE_OUT>& yGm,
    GlobalTensor<ROW_INDEX_DTYPE>& tokenRanksGm,
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>& queBind,
    uint32_t coreNum,
    uint32_t n)
{
    if (g_coreType == AIC) {
        // A5 SyncAll requires all cores to participate (strict 1:1 pairing)
        // AIC cores only sync, no actual data processing
        SyncAll();
        SyncAll();
        return;
    }

    SyncAll();  // Step 1: Full sync, ensure workspace writes complete

    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = coreNum * GetTaskRation();
    uint64_t baseOffset = syncConfig.lowBoundM - syncConfig.windowSize;

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(baseOffset + mOffset));

        // Row ownership: only the core with outRow % coreNumVec == GetBlockIdx() handles this row
        if (outRow % coreNumVec != GetBlockIdx()) {
            continue;
        }

        uint64_t curVecBaseN = syncConfig.baseN;
        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            if (nOffset + syncConfig.baseN >= n) {
                curVecBaseN = n - nOffset;  // Tail block
            }

            // Read from workspace to UB
            LocalTensor<DTYPE_OUT> bindLocal = queBind.AllocTensor<DTYPE_OUT>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
            DataCopyPad(bindLocal, deterBufferGm[mOffset * n + nOffset], copyParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_OUT>();

            // Atomic write to final output (single writer per row, deterministic order)
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
            DataCopyPad(yGm[outRow * n + nOffset], bindLocal, paramsOut);
            SetAtomicNone();

            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();  // Step 2: Full sync, ensure all writes complete
}

}  // namespace GMMFRDeterministic

#endif
