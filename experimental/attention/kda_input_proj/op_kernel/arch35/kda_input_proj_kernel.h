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
 * \file kda_input_proj_kernel.h
 * \brief
 */

#ifndef KDA_INPUT_PROJ_KERNEL_H
#define KDA_INPUT_PROJ_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "kda_input_proj_common.h"
#include "kda_input_proj_mm_bgg.h"
#include "kda_input_proj_mx_quant.h"
#include "kda_input_proj_qmm_qkv.h"
#include "kda_input_proj_sigmoid.h"
#include "../kda_input_proj_tiling_data.h"

namespace KdaInputProj {

template <typename TypePack>
class KdaInputProjKernel {
public:
    __aicore__ inline KdaInputProjKernel(__gm__ uint8_t *x, __gm__ uint8_t *weightQkv, __gm__ uint8_t *weightBeta,
                                      __gm__ uint8_t *weightGate, __gm__ uint8_t *weightG, __gm__ uint8_t *scaleQkv,
                                      __gm__ uint8_t *qkv, __gm__ uint8_t *beta, __gm__ uint8_t *gate,
                                      __gm__ uint8_t *g, __gm__ uint8_t *workspace,
                                      const optiling::KdaInputProjTilingData *__restrict tiling)
        : x_(x),
          weightQkv_(weightQkv),
          weightBeta_(weightBeta),
          weightGate_(weightGate),
          weightG_(weightG),
          scaleQkv_(scaleQkv),
          qkv_(qkv),
          beta_(beta),
          gate_(gate),
          g_(g),
          workspace_(workspace),
          tiling_(tiling)
    {
        mmBgg_.Init(x_, weightBeta_, weightGate_, weightG_, beta_, gate_, g_, workspace_, tiling_);
        mxQuant_.Init(x_, workspace_, tiling_);
        qmmQkv_.Init(weightQkv_, scaleQkv_, qkv_, workspace_, tiling_);
        sigmoid_.Init(beta_, gate_, g_, tiling_);
    }

    __aicore__ inline void Process();

protected:
    __aicore__ inline void ProcessStage1();
    __aicore__ inline void ProcessStage2();

    KdaInputProjMmBgg<TypePack> mmBgg_;
    KdaInputProjMxQuant<TypePack> mxQuant_;
    KdaInputProjQmmQkv<TypePack> qmmQkv_;
    KdaInputProjSigmoid<TypePack> sigmoid_;

    const optiling::KdaInputProjTilingData *tiling_{nullptr};
    __gm__ uint8_t *x_{nullptr};
    __gm__ uint8_t *weightQkv_{nullptr};
    __gm__ uint8_t *weightBeta_{nullptr};
    __gm__ uint8_t *weightGate_{nullptr};
    __gm__ uint8_t *weightG_{nullptr};
    __gm__ uint8_t *scaleQkv_{nullptr};
    __gm__ uint8_t *qkv_{nullptr};
    __gm__ uint8_t *beta_{nullptr};
    __gm__ uint8_t *gate_{nullptr};
    __gm__ uint8_t *g_{nullptr};
    __gm__ uint8_t *workspace_{nullptr};
};

template <typename TypePack>
__aicore__ inline void KdaInputProjKernel<TypePack>::ProcessStage1()
{
    // AIC: Matmul(beta/gate/g) || AIV: DynamicMxQuant
    if ASCEND_IS_AIC {
        mmBgg_.Process();
    }
    if ASCEND_IS_AIV {
        mxQuant_.Process();
    }
}

template <typename TypePack>
__aicore__ inline void KdaInputProjKernel<TypePack>::ProcessStage2()
{
    // AIC: QuantMatmul(qkv) || AIV: Sigmoid
    if ASCEND_IS_AIC {
        qmmQkv_.Process();
    }
    if ASCEND_IS_AIV {
        sigmoid_.Process();
    }
}

template <typename TypePack>
__aicore__ inline void KdaInputProjKernel<TypePack>::Process()
{
    ProcessStage1();
    AscendC::SyncAll();
    ProcessStage2();
}
} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_KERNEL_H
