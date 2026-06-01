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
 * \file norm_flash_attention.h
 * \brief 算子Kernel头文件
 */

#ifndef __NORM_FLASH_ATTENTION_KERNEL_H_
#define __NORM_FLASH_ATTENTION_KERNEL_H_

#include "kernel_operator.h"

namespace NormFlashAttentionNs {

using namespace AscendC;

struct InitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR attentionOut;
    GM_ADDR workspace;
    TPipe *tPipeIn;
    NormFlashAttentionTilingData *tilingData;
};

template <class T>
class NormFlashAttentionKernel {
public:
    __aicore__ inline NormFlashAttentionKernel() {}
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> queryGm_;
    GlobalTensor<T> keyGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<T> attentionOutGm_;
    TPipe *pipe_;
    NormFlashAttentionTilingData *tiling_;
};

template <class T>
__aicore__ inline void NormFlashAttentionKernel<T>::Init(InitParams initParams) {
    queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.query));
    keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.key));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.value));
    attentionOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.attentionOut));

    tiling_ = initParams.tilingData;
    pipe_ = initParams.tPipeIn;

    SyncAll();
}

template <class T>
__aicore__ inline void NormFlashAttentionKernel<T>::Process() {
    // TODO: 实现Flash Attention核心计算逻辑
}

} // namespace NormFlashAttentionNs

#endif // __NORM_FLASH_ATTENTION_KERNEL_H_
