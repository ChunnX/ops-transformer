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
 * \file kda_input_proj_qmm_qkv.h
 * \brief Stage2 AIC: QuantMatmul(qkv)
 */

#ifndef KDA_INPUT_PROJ_QMM_QKV_H
#define KDA_INPUT_PROJ_QMM_QKV_H

#include "kernel_operator.h"
#include "kda_input_proj_common.h"
#include "../kda_input_proj_tiling_data.h"

namespace KdaInputProj {

template <typename TypePack>
class KdaInputProjQmmQkv {
public:
    __aicore__ inline KdaInputProjQmmQkv()
    {
    }

    __aicore__ inline void Init(__gm__ uint8_t *weightQkv, __gm__ uint8_t *scaleQkv, __gm__ uint8_t *qkv,
                                __gm__ uint8_t *workspace, const optiling::KdaInputProjTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    using DtypeQkv = typename TypePack::DtypeQkv;

    const optiling::KdaInputProjTilingData *tiling_{nullptr};
    __gm__ uint8_t *weightQkv_{nullptr};
    __gm__ uint8_t *scaleQkv_{nullptr};
    __gm__ uint8_t *qkv_{nullptr};
    __gm__ uint8_t *workspace_{nullptr};
};

template <typename TypePack>
__aicore__ inline void KdaInputProjQmmQkv<TypePack>::Init(__gm__ uint8_t *weightQkv, __gm__ uint8_t *scaleQkv,
                                                       __gm__ uint8_t *qkv, __gm__ uint8_t *workspace,
                                                       const optiling::KdaInputProjTilingData *__restrict tiling)
{
    weightQkv_ = weightQkv;
    scaleQkv_ = scaleQkv;
    qkv_ = qkv;
    workspace_ = workspace;
    tiling_ = tiling;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjQmmQkv<TypePack>::Process()
{
}
} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_QMM_QKV_H
