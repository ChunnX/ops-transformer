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
 * \file kda_prolog_mx_quant.h
 * \brief Stage1 AIV: DynamicMxQuant on x
 */

#ifndef KDA_PROLOG_MX_QUANT_H
#define KDA_PROLOG_MX_QUANT_H

#include "kernel_operator.h"
#include "kda_prolog_common.h"
#include "../kda_prolog_tiling_data.h"

namespace KdaProlog {

template <typename TypePack>
class KdaPrologMxQuant {
public:
    __aicore__ inline KdaPrologMxQuant()
    {
    }

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *workspace,
                                const optiling::KdaPrologTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    using DtypeX = typename TypePack::DtypeX;

    const optiling::KdaPrologTilingData *tiling_{nullptr};
    __gm__ uint8_t *x_{nullptr};
    __gm__ uint8_t *workspace_{nullptr};
};

template <typename TypePack>
__aicore__ inline void KdaPrologMxQuant<TypePack>::Init(__gm__ uint8_t *x, __gm__ uint8_t *workspace,
                                                        const optiling::KdaPrologTilingData *__restrict tiling)
{
    x_ = x;
    workspace_ = workspace;
    tiling_ = tiling;
}

template <typename TypePack>
__aicore__ inline void KdaPrologMxQuant<TypePack>::Process()
{
}
} // namespace KdaProlog

#endif // KDA_PROLOG_MX_QUANT_H
