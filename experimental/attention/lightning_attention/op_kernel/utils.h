/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

namespace LightningAttn::Kernel
{

template <typename T> __aicore__ inline T CeilDiv(const T a, const T b)
{
    return (a + b - 1) / b;
}

template <typename T> __aicore__ inline T Align(const T a, const T b)
{
    return CeilDiv(a, b) * b;
}

template <typename T> __aicore__ inline T AlignItemsByBytes(const T a, const T sizeOfItem, const T aligneInBytes)
{
    return (CeilDiv(a * sizeOfItem, aligneInBytes) * aligneInBytes) / sizeOfItem;
}

template <typename T>
__aicore__ inline GM_ADDR SetTensorBufferFromWorkspace(AscendC::GlobalTensor<T>& t,
                                                       const int64_t size,
                                                       GM_ADDR ws,
                                                       const int64_t alignSize = 512l)
{
    int64_t sourceBytes = size * sizeof(T);
    int64_t alignedBytes = Align(sourceBytes, alignSize);
    t.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ws), alignedBytes / sizeof(T));
    return ws + alignedBytes;
}

} // namespace LightningAttn::Kernel
