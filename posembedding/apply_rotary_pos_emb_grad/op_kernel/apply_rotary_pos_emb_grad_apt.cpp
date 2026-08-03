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
 * \brief apply_rotary_pos_emb_grad 模板分发 kernel 入口（仅 Ascend 950 regbase）
 * 四维 tiling key: REDUCE_TPL_KEY + DxTilingKey + DcosFlag + Layout
 * 参照 rotary_position_embedding_grad_apt.cpp
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/apply_rotary_pos_emb_grad_dag.h"
#include "arch35/apply_rotary_pos_emb_grad_tiling_key.h"
#include "arch35/apply_rotary_pos_emb_grad_tiling_data.h"
#include "arch35/apply_rotary_pos_emb_grad_bab.h"
#include "arch35/apply_rotary_pos_emb_grad_ab.h"
#include "arch35/apply_rotary_pos_emb_grad_aba_and_ba.h"
#include "arch35/apply_rotary_pos_emb_grad_a_and_b.h"
#include "arch35/apply_rotary_pos_emb_grad_rotary_x.h"

enum class ApplyRopeGradDxTilingKey : uint32_t
{
    TILING_KEY_ABA = 201,
    TILING_KEY_BA = 202,
    TILING_KEY_BAB = 203,
    TILING_KEY_AB = 204,
    TILING_KEY_A = 205,
    TILING_KEY_B = 206
};

using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

// ============================================================================
// kernel 模板函数签名
// 参数顺序:
//   gradQ, gradK, cos, sin,  xQ, xK,
//   dxQ, dxK, cosGrad, sinGrad,  workspace,  tiling
// 4-d tiling key: (REDUCE_TPL_KEY, DxTilingKey, DcosFlag, Layout)
// ============================================================================
template <REDUCE_TPL_PARAM, uint32_t DxTilingKey, uint32_t DcosFlag, uint32_t Layout>
__global__ __aicore__ void apply_rotary_pos_emb_grad(
    GM_ADDR gradQ, GM_ADDR gradK, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR xQ, GM_ADDR xK,
    GM_ADDR dxQ, GM_ADDR dxK,
    GM_ADDR cosGrad, GM_ADDR sinGrad,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(ApplyRopeGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyRopeGradTilingData, tilingData, tiling);
    TPipe pipe;

    // ==========================================================================
    // Phase 1: dx 计算 (Q + K 双路)
    // 根据 DxTilingKey 分发到对应的 kernel class
    // ==========================================================================
    if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_ABA)) {
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradABAAndBA<DTYPE_DY, false> op;
        op.Init(gradQ, gradK, cos, sin, xQ, xK, dxQ, dxK, cosGrad, sinGrad, workspace,
                &tilingData.ropeGradParams, &pipe, static_cast<bool>(DcosFlag));
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_BA)) {
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradABAAndBA<DTYPE_DY, true> op;
        op.Init(gradQ, gradK, cos, sin, xQ, xK, dxQ, dxK, cosGrad, sinGrad, workspace,
                &tilingData.ropeGradParams, &pipe, static_cast<bool>(DcosFlag));
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_BAB)) {
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradBAB<DTYPE_DY> op(&pipe, &tilingData.ropeGradParams);
        op.Init(gradQ, gradK, cos, sin, dxQ, dxK);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_AB)) {
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradAB<DTYPE_DY> op(&pipe, &tilingData.ropeGradABParams);
        op.Init(gradQ, gradK, cos, sin, dxQ, dxK);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradAAndB<DTYPE_DY, false> op(&pipe, &tilingData.ropeGradParams);
        op.Init(gradQ, gradK, cos, sin, dxQ, dxK);
        op.Process();
        pipe.Reset();
    } else if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_B)) {
        ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradAAndB<DTYPE_DY, true> op(&pipe, &tilingData.ropeGradParams);
        op.Init(gradQ, gradK, cos, sin, dxQ, dxK);
        op.Process();
        pipe.Reset();
    }

    // ==========================================================================
    // Phase 2: dcos / dsin 计算
    // 当 x_q 和 x_k 都非空时 (DcosFlag == 1)，需要计算 cos/sin 的梯度
    // Step 1: 对 x_q 和 x_k 分别做 rotate_half，存入 workspace
    // Step 2: Reduce 计算 dcos = sum(dy * x, reduce_dims)
    //         Reduce 计算 dsin = sum(dy * x_rot, reduce_dims)
    // ==========================================================================
    if constexpr (DcosFlag) {
        PipeBarrier<PIPE_ALL>();
        SetSysWorkspace(workspace);
        GM_ADDR usrWorkSpace = AscendC::GetUserWorkspace(workspace);

        int64_t workSpacePerTensor = tilingData.rotaryXParams.b * tilingData.rotaryXParams.s *
                                     tilingData.rotaryXParams.nQ * tilingData.rotaryXParams.d;

        // Step 1: rotate_half for x_q → workspace
        ApplyRotaryPosEmbGrad::ApplyRopeGradRotaryX<DTYPE_DY> opQ(&pipe, &tilingData.rotaryXParams);
        opQ.Init(xQ, usrWorkSpace);
        opQ.Process();
        pipe.Reset();
        SyncAll();

        // Step 1b: rotate_half for x_k → workspace (at offset nQ*b*s*d)
        GM_ADDR xKRotaryWorkspace = usrWorkSpace + workSpacePerTensor * sizeof(DTYPE_DY);
        ApplyRotaryPosEmbGrad::ApplyRopeGradRotaryX<DTYPE_DY> opK(&pipe, &tilingData.rotaryXParams);
        opK.Init(xK, xKRotaryWorkspace);
        opK.Process();
        pipe.Reset();
        SyncAll();

        // Step 2: Reduce for dcos and dsin
        // dcos = reduce_sum(gradQ * xQ + gradK * xK)
        // dsin = reduce_sum(gradQ * xQ_rot + gradK * xK_rot)
        // TODO: 实际的 Q+K 合并 reduce 需要分别对 Q 路和 K 路调用 Reduce × 2 次
        // 然后再次 reduce 将 Q 和 K 的结果累加

        // A 模板不需要 reduce: 直接 element-wise 累加
        if constexpr (DxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
            // TODO: ApplyComputeADcosDsin with both Q and K paths
        } else {
            // 使用 ReduceSch 模板做 reduce
            using ReduceOp = ReduceSch<REDUCE_TPL_VALUE,
                ApplyRotaryPosEmbGrad::ApplyRopeGradDag<DTYPE_DY, float>::OpDag>;
            // Q 路 reduce: dcosQ = sum(gradQ * xQ), dsinQ = sum(gradQ * xQ_rot)
            // K 路 reduce: dcosK = sum(gradK * xK), dsinK = sum(gradK * xK_rot)
            // 最终 dcos = dcosQ + dcosK, dsin = dsinQ + dsinK
            // TODO: 实现合并 reduce 逻辑
        }
    }
}
