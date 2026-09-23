/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_CUBE_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_CUBE_H_

#include "mhc_sandwich_norm_post_preonly_common.h"
#include "lib/matmul_intf.h"

constexpr MatmulConfig MHC_SANDWICH_CUBE_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

// Decode目标shape专用的FP32 Phi投影。AIV把R3后的低精度x2恢复为FP32并写入workspace，
// AIC通过成熟Matmul接口执行[1, 4D] x [4, 4D]^T，避免手写B2转置布局。
class MhcSandwichNormPostPreonlyCube {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

    __aicore__ inline void Init(GM_ADDR phi, GM_ADDR workspace, uint32_t totalTokens, uint32_t d,
                                const TCubeTiling *mmTiling)
    {
        totalTokens_ = totalTokens;
        d_ = d;
        k_ = N_MAX * d_;
        blockIdx_ = GetBlockIdx();

        const uint64_t partialElems = (uint64_t)totalTokens_ * 2U * PARTIAL_SLOT_ELEMS;
        const uint64_t x2Elems = (uint64_t)totalTokens_ * k_;
        GlobalTensor<float> workspaceFp32;
        workspaceFp32.SetGlobalBuffer((__gm__ float *)workspace,
                                      partialElems + x2Elems + (uint64_t)totalTokens_ * PARTIAL_SLOT_ELEMS);
        x2Gm_ = workspaceFp32[partialElems];
        dotGm_ = workspaceFp32[partialElems + x2Elems];
        phiGm_.SetGlobalBuffer((__gm__ float *)phi, (uint64_t)N_MAX * k_);

        mm_.Init(mmTiling, &pipe_);
        mm_.SetSubBlockIdx(0);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= totalTokens_) {
            return;
        }

        CrossCoreWaitFlag<0x4, PIPE_MTE2>(CUBE_INPUT_READY_FLAG);
        CrossCoreWaitFlag<0x4, PIPE_MTE2>(CUBE_INPUT_READY_FLAG + AIV1_FLAG_OFFSET);

        mm_.SetTensorA(x2Gm_[(uint64_t)blockIdx_ * k_]);
        mm_.SetTensorB(phiGm_, true);
        mm_.SetHF32(false, 1);
        mm_.SetOrgShape(1U, N_MAX, k_);
        mm_.SetSingleShape(1U, N_MAX, k_);
        mm_.template IterateAll<false>(dotGm_[(uint64_t)blockIdx_ * PARTIAL_SLOT_ELEMS]);
        mm_.End();

        CrossCoreSetFlag<0x4, PIPE_FIX>(CUBE_RESULT_READY_FLAG);
    }

    matmul::MatmulImpl<AType, BType, CType, CType, MHC_SANDWICH_CUBE_CFG> mm_;

private:
    TPipe pipe_;
    GlobalTensor<float> phiGm_, x2Gm_, dotGm_;
    uint32_t totalTokens_ = 0U;
    uint32_t d_ = 0U;
    uint32_t k_ = 0U;
    uint32_t blockIdx_ = 0U;
};

// MatmulImpl的局部状态远大于Vector路径对象。独立栈帧可避免统一入口为所有tiling key
// 预留Cube对象空间，保证非Cube路径的资源布局不受影响。
__aicore__ __attribute__((noinline)) void RunMhcSandwichNormPostPreonlyCube(
    GM_ADDR phi, GM_ADDR workspace, uint32_t totalTokens, uint32_t d, const TCubeTiling *mmTiling)
{
    MhcSandwichNormPostPreonlyCube op;
    op.Init(phi, workspace, totalTokens, d, mmTiling);
    op.Process();
}

#endif // MHC_SANDWICH_NORM_POST_PREONLY_CUBE_H_
