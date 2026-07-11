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
 * \file causal_topo_gcn.h
 * \brief
 */
#ifndef __CAUSAL_TOPO_GCN_H__
#define __CAUSAL_TOPO_GCN_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_topo_gcn_tiling_data.h"
#include "causal_topo_gcn_tiling_key.h"

namespace NsCausalTopoGcn {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class CausalTopoGcn {
public:
    __aicore__ inline CausalTopoGcn(){};

    __aicore__ inline void Init(/*参数列表*/);
    __aicore__ inline void Process(/*参数列表*/);

private:
    __aicore__ inline void CopyIn(/*参数列表*/);
    __aicore__ inline void CopyOut(/*参数列表*/);
    __aicore__ inline void Compute(/*参数列表*/);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> XXX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> YYY;
};

template <typename T>
__aicore__ inline void CausalTopoGcn<T>::Init(/*参数列表*/)
{
}

template <typename T>
__aicore__ inline void CausalTopoGcn<T>::CopyIn(/*参数列表*/)
{
}

template <typename T>
__aicore__ inline void CausalTopoGcn<T>::CopyOut(/*参数列表*/)
{
}

template <typename T>
__aicore__ inline void CausalTopoGcn<T>::Compute(/*参数列表*/)
{
}

template <typename T>
__aicore__ inline void CausalTopoGcn<T>::Process()
{
    for (int32_t i = 0; i < /*循环次数*/; i++) {
        CopyIn(/*参数列表*/);
        Compute(/*参数列表*/);
        CopyOut(/*参数列表*/);
    }
}

} // namespace NsCausalTopoGcn
#endif // CAUSAL_TOPO_GCN_H