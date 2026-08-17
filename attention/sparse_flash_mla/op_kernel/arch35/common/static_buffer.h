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
 * \file static_buffer.h
 * \brief 静态 tensor + bufferId 管理的 buffer 原语。
 *        StaticBuffer 封装一块静态 LocalTensor 及其 bufferId；
 *        RingBuffer 对一组 StaticBuffer 做环形轮转。
 *        依赖：使用前需已包含 AscendC 内核头（LocalTensor 声明）。
 */
#ifndef STATIC_BUFFER_H
#define STATIC_BUFFER_H

template <typename ElemT>
struct StaticBuffer {
    AscendC::LocalTensor<ElemT> tensor;
    uint32_t idx;
};

template <typename ElemT>
struct RingBuffer {
    StaticBuffer<ElemT> *bufs;
    uint32_t             bufNum;
    uint32_t             curId;

    __aicore__ inline RingBuffer() : bufs(nullptr), bufNum(0), curId(0) {}
    __aicore__ inline RingBuffer(StaticBuffer<ElemT> *b, uint32_t n)
        : bufs(b), bufNum(n), curId(n - 1) {}

    __aicore__ inline StaticBuffer<ElemT> &GetNext()
    {
        curId = (curId + 1) % bufNum;
        return bufs[curId];
    }
};

#endif // STATIC_BUFFER_H
