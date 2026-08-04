/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_grad_apt.cpp
 * \brief 模板分发 kernel 入口（仅 Ascend 950 regbase）
 *
 * DxTilingKey: 203=BAB(BSND), 204=AB(SBND), 205=A(NO_BROADCAST)
 * Layout: 0=BSND, 1=SBND, 2=BNSD(参数预留), 3=NO_BROADCAST
 */
#include "atvoss/reduce/reduce_sch.h"
#include "arch35/apply_rotary_pos_emb_grad_dag.h"
#include "arch35/apply_rotary_pos_emb_grad_tiling_key.h"
#include "arch35/apply_rotary_pos_emb_grad_tiling_data.h"
#include "arch35/apply_rotary_pos_emb_grad_bab.h"
#include "arch35/apply_rotary_pos_emb_grad_ab.h"
#include "arch35/apply_rotary_pos_emb_grad_a_and_b.h"
#include "arch35/apply_rotary_pos_emb_grad_rotary_x.h"

enum class ApplyRopeGradDxTilingKey : uint32_t {
    TK_BAB = 203,   // BSND (含 TND)
    TK_AB  = 204,   // SBND
    TK_A   = 205,   // NO_BROADCAST (N=1, cos==grad)
};

using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM, uint32_t DxTilingKey, uint32_t DcosFlag, uint32_t Layout>
__global__ __aicore__ void apply_rotary_pos_emb_grad(
    GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR query, GM_ADDR key,
    GM_ADDR grad_query, GM_ADDR grad_key,
    GM_ADDR grad_cos, GM_ADDR grad_sin,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) return;
    if constexpr (DxTilingKey != static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TK_A)) {
        if (workspace == nullptr) return;
    }

    REGISTER_TILING_DEFAULT(ApplyRopeGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyRopeGradTilingData, tilingData, tiling);
    TPipe pipe;

    // ==================================================================
    // Phase 1: dx 计算 (Q + K 双路)
    // ==================================================================
    if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TK_BAB)) {
        // BSND (含 TND 映射) — BAB tiling
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradBAB<DTYPE_DY> op(
            &pipe, &tilingData.ropeGradParams);
        op.Init(grad_query_embed, grad_key_embed, cos, sin, grad_query, grad_key);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TK_AB)) {
        // SBND — AB tiling
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradAB<DTYPE_DY> op(
            &pipe, &tilingData.ropeGradABParams);
        op.Init(grad_query_embed, grad_key_embed, cos, sin, grad_query, grad_key);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TK_A)) {
        // NO_BROADCAST (N=1, cos==grad) — A tiling
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradA<DTYPE_DY> op(
            &pipe, &tilingData.ropeGradParams);
        op.Init(grad_query_embed, grad_key_embed, cos, sin, grad_query, grad_key);
        op.Process();
        pipe.Reset();
    }

    // ==================================================================
    // Phase 2: grad_cos/grad_sin (DcosFlag == 1)
    // ==================================================================
    if constexpr (DcosFlag) {
        // A template (NO_BROADCAST, N=1): 无广播轴, element-wise mul 直接出结果
        if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TK_A)) {
            ApplyRotaryPosEmbGrad::ApplyNoBroadcastDcosDsin<DTYPE_DY> opGrad(
                &pipe, &tilingData.rotaryXParams);
            opGrad.Init(grad_query_embed, grad_key_embed, query, key, grad_cos, grad_sin);
            opGrad.Process();
        } else {
            PipeBarrier<PIPE_ALL>();
            SetSysWorkspace(workspace);
            GM_ADDR usrWorkSpace = AscendC::GetUserWorkspace(workspace);

            int64_t wsPerQ = tilingData.rotaryXParams.b * tilingData.rotaryXParams.s *
                             tilingData.rotaryXParams.nQ * tilingData.rotaryXParams.d;

            // rotate_half query → workspace
            ApplyRotaryPosEmbGrad::ApplyRopeGradRotaryX<DTYPE_DY> rotQ(
                &pipe, &tilingData.rotaryXParams);
            rotQ.Init(query, usrWorkSpace);
            rotQ.Process(); pipe.Reset(); SyncAll();

            // rotate_half key → workspace[nQ*b*s*d]
            GM_ADDR keyRot = usrWorkSpace + wsPerQ * sizeof(DTYPE_DY);
            ApplyRotaryPosEmbGrad::ApplyRopeGradRotaryX<DTYPE_DY> rotK(
                &pipe, &tilingData.rotaryXParams);
            rotK.Init(key, keyRot);
            rotK.Process(); pipe.Reset(); SyncAll();

            // Reduce: grad_cos = sum(dy * x), grad_sin = sum(dy * x_rot)
            using ReduceOp = ReduceSch<REDUCE_TPL_VALUE,
                ApplyRotaryPosEmbGrad::ApplyRopeGradDag<DTYPE_DY, float>::OpDag>;
            // TODO: 双路 reduce — Q/K 分别 reduce 后累加到 grad_cos/grad_sin
        }
    }
}
