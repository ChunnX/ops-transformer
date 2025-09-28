/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file buffer_manager.h
 * \brief buffer内存管理
 */
#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include "buffer.h"

// L1  TPosition::A1
// L0A TPosition::A2
// L0B TPosition::B2
// L0C TPosition::CO1
namespace fa_base_matmul {
template<BufferType Type>
class BufferManager {
public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t size) {
        TBuf<BufferInfo<Type>::Position> tbuf;
        bufferSize_ = size;
        pipe->InitBuffer(tbuf, size);
        mem_ = tbuf.template Get<uint8_t>();
    }

    __aicore__ inline Buffer<Type> AllocBuffer(uint32_t size) {
        LocalTensor<uint8_t> temp = mem_[offset_];
        offset_ += size;
        return Buffer<Type, true>(temp, size);
    }

    __aicore__ inline Buffer<Type, false> AllocBufferNoSync(uint32_t size) {
        LocalTensor<uint8_t> temp = mem_[offset_];
        offset_ += size;
        return Buffer<Type, false>(temp, size);
    }

    __aicore__ inline void FreeBuffer(Buffer<Type> &buffer){
    }
private:
    uint32_t offset_ = 0;
    uint32_t bufferSize_ = 0;
    LocalTensor<uint8_t> mem_;
};
}
#endif