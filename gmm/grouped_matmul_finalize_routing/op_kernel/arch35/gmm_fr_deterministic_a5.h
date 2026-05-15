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
 * \brief A5 deterministic deferred aggregation for GMMFR.
 *
 * Ported from A3 FRDeterministic() (grouped_matmul_finalize_routing.h line 653-687).
 *
 * Data flow:
 *   1. Prologue -> yGm (zeros + residual)
 *   2. SequentialWrite epilogue -> workspace (no race)
 *   3. This function: workspace[mOffset] -> yGm[outRow] via AtomicAdd
 *
 * Determinism: each workspace address written by one core, each outRow owned by one core.
 *
 * Uses raw-offset LocalTensor (aggLocal) placed beyond the epilogue's UB usage
 * to avoid memory overlap with the epilogue's l0cOutUb_/outUb buffers.
 */

#ifndef GMM_FR_DETERMINISTIC_A5_H
#define GMM_FR_DETERMINISTIC_A5_H

#include "kernel_operator.h"

namespace GMMFRDeterministic {

using namespace AscendC;

// Constants consistent with A3
constexpr uint32_t DETER_UB_SIZE = 12 * 1024;
constexpr uint32_t BUFFER_NUM = 2;

// Sync config (matches A3 SyncConfig layout)
struct SyncConfig {
    uint64_t curM = 0;
    uint64_t curGroup = 0;
    uint64_t curGroupM = 0;
    uint64_t lowBoundM = 0;
    uint64_t windowSize = 0;
    uint64_t baseN = 0;
};

/**
 * FRDeterministicA5 - Deferred aggregation: workspace -> yGm via rowIndex + AtomicAdd.
 *
 * Uses raw-offset LocalTensor (aggLocal) at VECOUT position for DMA transfers.
 * MTE2 reads workspace -> aggLocal (VECOUT), then MTE3 writes aggLocal -> yGm (AtomicAdd).
 * No TPipe / TQueBind needed — avoids UB overlap with the epilogue's buffers.
 *
 * Template: DTYPE_OUT (float), ROW_INDEX_DTYPE (int32_t or int64_t)
 */
template <typename DTYPE_OUT, typename ROW_INDEX_DTYPE>
__aicore__ inline void FRDeterministicA5(
    SyncConfig& syncConfig,
    GlobalTensor<DTYPE_OUT>& deterBufferGm,
    GlobalTensor<DTYPE_OUT>& yGm,
    GlobalTensor<ROW_INDEX_DTYPE>& tokenRanksGm,
    LocalTensor<float>& aggLocal,
    uint32_t coreNum,
    uint32_t n,
    uint64_t globalRowOffset = 0)
{
    if (g_coreType == AIC) {
        SyncAll();
        SyncAll();
        return;
    }

    SyncAll();  // Ensure workspace writes complete

    uint64_t totalM = syncConfig.curM;
    uint64_t myVecId = static_cast<uint64_t>(GetBlockIdx());

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(globalRowOffset + mOffset));

        if (static_cast<uint64_t>(outRow % coreNum) != myVecId) {
            continue;
        }

        uint64_t curVecBaseN = syncConfig.baseN;
        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            if (nOffset + syncConfig.baseN >= n) {
                curVecBaseN = n - nOffset;
            }

            // MTE2: workspace -> aggLocal (VECOUT)
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_OUT> padParams{false, 0, 0, 0};
            DataCopyPad(aggLocal, deterBufferGm[mOffset * n + nOffset], copyParams, padParams);

            // MTE3: aggLocal (VECOUT) -> yGm (AtomicAdd)
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
            DataCopyPad(yGm[outRow * n + nOffset], aggLocal, paramsOut);
            SetAtomicNone();
        }
    }
    SyncAll();  // Ensure all writes complete
}

}  // namespace GMMFRDeterministic

#endif
