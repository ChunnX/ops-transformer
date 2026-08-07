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
 * \file apply_rotary_pos_emb_grad_apt.cpp
 * \brief
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/apply_rotary_pos_emb_grad_tiling_key.h"
#include "arch35/apply_rotary_pos_emb_grad_tiling_data.h"
#include "arch35/apply_rotary_pos_emb_grad_bab.h"
#include "arch35/apply_rotary_pos_emb_grad_ab.h"
#include "arch35/apply_rotary_pos_emb_grad_a.h"
#include "arch35/apply_rotary_pos_emb_grad_dag.h"
#include "arch35/apply_rotary_pos_emb_grad_rotary_x.h"

enum class ApplyRopeGradDxTilingKey : uint32_t {
    TILING_KEY_BAB = 203,
    TILING_KEY_AB  = 204,
    TILING_KEY_A   = 205
};

using namespace ApplyRotaryPosEmbGrad;
using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

constexpr int64_t INPUT_OUTPUT_NUM = 2; // grad_cos_partial + grad_sin_partial

template <REDUCE_TPL_PARAM, uint32_t DxTilingKey, uint32_t DcosFlag>
__global__ __aicore__ void apply_rotary_pos_emb_grad(
    GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR query, GM_ADDR key,
    GM_ADDR gradQueryOut, GM_ADDR gradKeyOut, GM_ADDR gradCosOut, GM_ADDR gradSinOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // AIC core 不执行 vector 计算
    if (g_coreType == AIC) {
        return;
    }
    if constexpr (DxTilingKey != static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
        if (workspace == nullptr) {
            return;
        }
    }

    REGISTER_TILING_DEFAULT(ApplyRopeGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyRopeGradTilingData, tilingData, tiling);
    TPipe pipe;

    // ================================================================
    // Phase 1: 计算 grad_query / grad_key
    //          当 DcosFlag=1 时，同步在 kernel 内累加 grad_cos / grad_sin 部分积
    // ================================================================
    if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_BAB)) {
        ApplyRotaryPosEmbGradBAB<DTYPE_DY> op(&pipe, &tilingData.ropeGradParams);
        op.Init(gradQueryEmbed, gradKeyEmbed, cos, sin,
                query, key,
                gradQueryOut, gradKeyOut, gradCosOut, gradSinOut);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_AB)) {
        ApplyRotaryPosEmbGradAB<DTYPE_DY> op(&pipe, &tilingData.ropeGradABParams);
        op.Init(gradQueryEmbed, gradKeyEmbed, cos, sin,
                query, key,
                gradQueryOut, gradKeyOut, gradCosOut, gradSinOut);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
        ApplyRotaryPosEmbGradA<DTYPE_DY> op(&pipe, &tilingData.ropeGradParams);
        op.Init(gradQueryEmbed, gradKeyEmbed, cos, sin,
                gradQueryOut, gradKeyOut);
        op.Process();
        pipe.Reset();
    }

    // ================================================================
    // Phase 2: DcosFlag=1 → Reduce 跨广播轴求和
    // ================================================================
    if constexpr (DcosFlag) {
        // A 模板 (205): 无广播轴，grad_cos/grad_sin 已在 kernel 内直接写回 GM，无需 reduce
        if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
            ApplyNoBroadcastDcosDsin<DTYPE_DY> opGrad(&pipe, &tilingData.rotaryXParams);
            opGrad.Init(gradQueryEmbed, gradKeyEmbed, query, key, gradCosOut, gradSinOut);
            opGrad.Process();
            return;
        }

        PipeBarrier<PIPE_ALL>();
        SetSysWorkspace(workspace);
        GM_ADDR usrWorkSpace = AscendC::GetUserWorkspace(workspace);

        // 单份中间结果大小 = B * S * max(nQ, nK) * D
        int64_t maxN = std::max(tilingData.ropeGradParams.nQ, tilingData.ropeGradParams.nK);
        int64_t partSize = tilingData.ropeGradParams.b * tilingData.ropeGradParams.s *
                           maxN * tilingData.ropeGradParams.d;

        using ReduceOp = ReduceSch<REDUCE_TPL_VALUE,
            ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradDag<DTYPE_DY, float>::OpDag>;
        ReduceOp reduceOp0(&tilingData.reduceTiling);
        ReduceOp reduceOp1(&tilingData.reduceTiling);

        // 第一份 workspace 存 dcos_partial，第二份存 dsin_partial
        // gradCosOut = reduce(dcos_partial), gradSinOut = reduce(dsin_partial)
        GM_ADDR reduceWorkSpace = usrWorkSpace + partSize * INPUT_OUTPUT_NUM * sizeof(DTYPE_DY);

        reduceOp0.Init(&pipe, gradQueryEmbed, usrWorkSpace, gradCosOut, reduceWorkSpace);
        reduceOp0.Process();
        pipe.Reset();
        PipeBarrier<PIPE_ALL>();

        GM_ADDR dSinWorkSpace = usrWorkSpace + partSize * sizeof(DTYPE_DY);
        reduceOp1.Init(&pipe, gradQueryEmbed, dSinWorkSpace, gradSinOut, reduceWorkSpace);
        reduceOp1.Process();
        pipe.Reset();
    }
}
