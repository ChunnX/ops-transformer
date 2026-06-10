/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stem_oam_prep_varlen_q_tiling.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace optiling {

static constexpr int64_t DIM_QK_HOST = 128;
static constexpr uint32_t MAX_CORE_NUM_HOST = 64;
static constexpr uint32_t MAX_BATCH_SIZE_HOST = 256;

ge::graphStatus Tiling4StemOamPrepVarlenQ(gert::TilingContext* context) {
    auto compile_info = static_cast<const StemPrepQCompileInfo*>(context->GetCompileInfo());
    uint32_t N = static_cast<uint32_t>(compile_info->core_num);

    auto attrs = context->GetAttrs();
    const int64_t *stemBlockSizePtr = attrs->GetInt(0);
    const int64_t *stemStridePtr = attrs->GetInt(1);
    if (stemBlockSizePtr == nullptr || stemStridePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t stemBlockSize = *stemBlockSizePtr;
    int64_t stemStride = *stemStridePtr;

    uint32_t B = static_cast<uint32_t>(stemBlockSize);
    uint32_t S = static_cast<uint32_t>(stemStride);
    uint32_t R = B / S;
    uint32_t qflat_dim = S * DIM_QK_HOST;

    auto tiling = context->GetTilingData<StemPrepQTilingData>();
    tiling->stem_block_size = B;
    tiling->stem_stride = S;
    tiling->R_val = R;
    tiling->kflat_dim = qflat_dim;
    tiling->dim_qk = DIM_QK_HOST;

    tiling->batch_size = context->GetInputShape(1)->GetStorageShape().GetDim(0);
    tiling->num_q_heads = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    tiling->total_tokens = context->GetInputShape(0)->GetStorageShape().GetDim(0);

    uint32_t H_q = tiling->num_q_heads;
    uint32_t batch = tiling->batch_size;

    // Read q_seq_lens to compute per-batch num_Qb
    auto qSeqLensTensor = context->GetInputTensor(1);
    int32_t* qSeqLensData = nullptr;
    if (qSeqLensTensor != nullptr) {
        qSeqLensData = const_cast<int32_t*>(qSeqLensTensor->GetData<int32_t>());
    }

    uint32_t max_q_len = 0;
    uint32_t total_blocks = 0;
    memset(tiling->num_Qb_per_batch, 0, sizeof(tiling->num_Qb_per_batch));

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t q_len_b = 0;
        if (qSeqLensData != nullptr) {
            q_len_b = static_cast<uint32_t>(qSeqLensData[b]);
        }
        uint32_t q_padded_b = ((q_len_b + B - 1) / B) * B;
        uint32_t num_Qb_b = q_padded_b / B;
        tiling->num_Qb_per_batch[b] = num_Qb_b;
        total_blocks += H_q * num_Qb_b;
        if (q_len_b > max_q_len) max_q_len = q_len_b;
    }

    tiling->total_blocks = total_blocks;
    uint32_t max_q_padded = ((max_q_len + B - 1) / B) * B;
    uint32_t max_Qb = max_q_padded / B;
    tiling->max_q_padded = max_q_padded;
    tiling->max_Qb = max_Qb;

    uint32_t P = batch * H_q;

    if (P >= N) {
        tiling->split_mode = 0;
        tiling->total_tasks = P;
        tiling->used_core_num = std::min(P, N);
        tiling->cores_per_bh_base = 1;
        tiling->remainder_bh = 0;
    } else if (total_blocks > 0) {
        tiling->split_mode = 2;
        tiling->used_core_num = std::min(N, total_blocks);
        tiling->total_tasks = tiling->used_core_num;

        uint32_t bpc_base = total_blocks / tiling->used_core_num;
        uint32_t bpc_rem  = total_blocks % tiling->used_core_num;
        tiling->blocks_per_core_base = bpc_base;
        tiling->blocks_remainder = bpc_rem;

        // Build per-core assignment table via prefix sums over (b,h,qb) flat space
        // prefix[b] = Σ_{b'<b} H_q × num_Qb[b']
        uint32_t batch_prefix[MAX_BATCH_SIZE_HOST + 1];
        batch_prefix[0] = 0;
        for (uint32_t b = 0; b < batch; b++) {
            batch_prefix[b + 1] = batch_prefix[b] + H_q * tiling->num_Qb_per_batch[b];
        }

        memset(tiling->core_batch_idx, 0, sizeof(tiling->core_batch_idx));
        memset(tiling->core_head_idx, 0, sizeof(tiling->core_head_idx));
        memset(tiling->core_block_idx, 0, sizeof(tiling->core_block_idx));
        memset(tiling->core_block_num, 0, sizeof(tiling->core_block_num));

        for (uint32_t core = 0; core < tiling->used_core_num; core++) {
            uint32_t global_start;
            uint32_t block_num_core;
            if (core < bpc_rem) {
                global_start = core * (bpc_base + 1);
                block_num_core = bpc_base + 1;
            } else {
                global_start = bpc_rem * (bpc_base + 1) + (core - bpc_rem) * bpc_base;
                block_num_core = bpc_base;
            }

            // Map global_start to (b, h, qb) via prefix sums
            uint32_t b_found = 0;
            while (b_found < batch && global_start >= batch_prefix[b_found + 1]) {
                b_found++;
            }
            uint32_t local_in_batch = global_start - batch_prefix[b_found];
            uint32_t h_found = local_in_batch / tiling->num_Qb_per_batch[b_found];
            uint32_t qb_found = local_in_batch % tiling->num_Qb_per_batch[b_found];

            tiling->core_batch_idx[core] = b_found;
            tiling->core_head_idx[core] = h_found;
            tiling->core_block_idx[core] = qb_found;
            tiling->core_block_num[core] = block_num_core;
        }
    } else {
        tiling->split_mode = 0;
        tiling->total_tasks = 0;
        tiling->used_core_num = 0;
    }

    context->SetTilingKey(0);
    context->SetBlockDim(tiling->used_core_num);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4StemOamPrepVarlenQ(gert::TilingParseContext* context) {
    auto compileInfo = context->GetCompiledInfo<StemPrepQCompileInfo>();
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int32_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StemOamPrepVarlenQ)
    .Tiling(Tiling4StemOamPrepVarlenQ)
    .TilingParse<StemPrepQCompileInfo>(TilingPrepare4StemOamPrepVarlenQ);

}  // namespace optiling