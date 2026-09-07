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
 * \file chunk_gated_delta_rule_arch35.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_ARCH35_H
#define CHUNK_GATED_DELTA_RULE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_fused_exact_arch35.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

struct CGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR seqlens;
    GM_ADDR gOptional;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename lowType, typename highType, typename stateType = lowType, bool gOptional = false>
class CGDR {
public:
    __aicore__ inline explicit CGDR(const ChunkGatedDeltaRuleTilingData *tilingData)
    {
        tiling_ = tilingData;
    };

    __aicore__ inline void Init(const CGDRInitParams &initParams)
    {
        uint64_t dataSize = tiling_->t * tiling_->nk * tiling_->dk;
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.query), dataSize);
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.key), dataSize);

        dataSize = tiling_->t * tiling_->nv * tiling_->dv;
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.value), dataSize);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.attnOut), dataSize);

        dataSize = tiling_->t * tiling_->nv;
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.beta), dataSize);
        if constexpr (gOptional) {
            g_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(initParams.gOptional), dataSize);
        }

        dataSize = tiling_->b * tiling_->nv * tiling_->dv * tiling_->dk;
        initState_.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(initParams.initState), dataSize);
        finalState_.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(initParams.finalState), dataSize);

        actualSeqLens_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(initParams.seqlens), tiling_->b);
    }

    __aicore__ inline void Process()
    {
        SyncAll<false>();
        if (!HasValidExactInputs()) {
            return;
        }
        RunExactBatches();
    }

private:
    __aicore__ inline bool HasValidExactInputs() const
    {
        if (tiling_->b <= 0 || tiling_->t <= 0 || tiling_->nk <= 0 || tiling_->nv <= 0 ||
            tiling_->nv % tiling_->nk != 0 || tiling_->dk <= 0 || tiling_->dk > 128 || tiling_->dv <= 0 ||
            tiling_->dv > 128 || tiling_->chunkSize != 64 || tiling_->aiCoreNum <= 0 || tiling_->stateStride0 < 0 ||
            tiling_->stateStride1 < 0) {
            return false;
        }
        const uint64_t targetLength = static_cast<uint64_t>(tiling_->t);
        uint64_t totalLength = 0;
        for (int64_t bid = 0; bid < tiling_->b; ++bid) {
            const int32_t length = actualSeqLens_.GetValue(bid);
            if (length < 0 || static_cast<uint64_t>(length) > targetLength - totalLength) {
                return false;
            }
            totalLength += static_cast<uint64_t>(length);
        }
        return totalLength == targetLength;
    }

    __aicore__ inline void RunExactBatches()
    {
        FusedStage123ExactParams<stateType> params{query_,
                                                   key_,
                                                   value_,
                                                   beta_,
                                                   g_,
                                                   initState_,
                                                   out_,
                                                   finalState_,
                                                   actualSeqLens_,
                                                   tiling_->scale,
                                                   static_cast<uint32_t>(tiling_->nk),
                                                   static_cast<uint32_t>(tiling_->nv),
                                                   static_cast<uint32_t>(tiling_->dk),
                                                   static_cast<uint32_t>(tiling_->dv),
                                                   static_cast<uint32_t>(tiling_->aiCoreNum),
                                                   static_cast<uint64_t>(tiling_->b),
                                                   static_cast<uint64_t>(tiling_->stateStride0),
                                                   static_cast<uint64_t>(tiling_->stateStride1)};
        FusedStage123Exact<stateType, gOptional> op;
        op.Init(params);
        op.Process();
    }

private:
    const ChunkGatedDeltaRuleTilingData *tiling_;
    GlobalTensor<lowType> query_;
    GlobalTensor<lowType> key_;
    GlobalTensor<lowType> value_;
    GlobalTensor<lowType> beta_;
    GlobalTensor<highType> g_;
    GlobalTensor<lowType> out_;
    GlobalTensor<stateType> finalState_;
    GlobalTensor<stateType> initState_;
    GlobalTensor<int32_t> actualSeqLens_;
};

} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_ARCH35_H
