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
 * \file mutex_buffers_policy.h
 * \brief 基于Mutex的buffer多缓冲策略（替代event同步）
 */
#ifndef MUTEX_BUFFERS_POLICY_H
#define MUTEX_BUFFERS_POLICY_H

#include "buffer_manager.h"
#include "mutex_buffer.h"
#ifndef NUM_2
#define NUM_2 2
#endif

namespace fa_base_matmul {

namespace detail {
template<BufferType bufferType>
__aicore__ inline MutexBuffer<bufferType> AllocMutexBuf(BufferManager<bufferType> &mgr, uint32_t size) {
    Buffer<bufferType, SyncType::NO_SYNC> tmp = mgr.template AllocBuffer<SyncType::NO_SYNC>(size);
    return MutexBuffer<bufferType>(tmp.template GetTensor<uint8_t>(), size);
}
}

template<BufferType bufferType>
class MutexBuffersPolicySingleBuffer {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        buffer_ = detail::AllocMutexBuf(bufferManager, size);
        buffer_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        buffer_.UnInit();
    }

    __aicore__ inline MutexBuffer<bufferType> &Get() {
        return buffer_;
    }

    __aicore__ inline MutexBuffer<bufferType> &GetPre() {
        return Get();
    }

    __aicore__ inline MutexBuffer<bufferType> &GetReused() {
        return Get();
    }
private:
    MutexBuffer<bufferType> buffer_;
};

template<BufferType bufferType>
class MutexBuffersPolicyDB {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        ping_ = detail::AllocMutexBuf(bufferManager, size);
        pong_ = detail::AllocMutexBuf(bufferManager, size);

        ping_.Init();
        pong_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        ping_.UnInit();
        pong_.UnInit();
    }

    __aicore__ inline MutexBuffer<bufferType> &Get() {
        if (flag1_) {
            flag1_ = 0;
            return ping_;
        } else {
            flag1_ = 1;
            return pong_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType> &GetPre() {
        if (flag1_) {
            return pong_;
        } else {
            return ping_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType> &GetReused() {
        if (flag2_ == 0) {
            flag2_ = 1;
            lastReusedFlag_ = 1;
            hasLastReused_ = true;
            return pong_;
        } else {
            flag2_ = 0;
            lastReusedFlag_ = 0;
            hasLastReused_ = true;
            return ping_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType> &GetReused(bool isNextS2IdxNoChange) {
        if (isNextS2IdxNoChange) {
            if (!hasLastReused_) {
                return GetReused();
            }
            if (lastReusedFlag_ == 1) {
                return pong_;
            } else {
                return ping_;
            }
        } else {
            return GetReused();
        }
    }

private:
    MutexBuffer<bufferType> ping_;
    MutexBuffer<bufferType> pong_;
    uint32_t flag1_ = 0;
    uint32_t flag2_ = 0;
    uint32_t lastReusedFlag_ = 0;
    bool hasLastReused_ = false;
};

template<BufferType bufferType>
class MutexBuffersPolicy3buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        a_ = detail::AllocMutexBuf(bufferManager, size);
        b_ = detail::AllocMutexBuf(bufferManager, size);
        c_ = detail::AllocMutexBuf(bufferManager, size);

        a_.Init();
        b_.Init();
        c_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();
    }

    __aicore__ inline MutexBuffer<bufferType> &Get() {
        if (flag1_ == 0) {
            flag1_ = 1;
            return a_;
        } else if (flag1_ == 1) {
            flag1_ = NUM_2;
            return b_;
        } else {
            flag1_ = 0;
            return c_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType> &GetPre() {
        if (flag1_ == 0) {
            return c_;
        } else if (flag1_ == 1) {
            return a_;
        } else {
            return b_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType> &GetReused() {
        if (flag2_ == 0) {
            flag2_ = 1;
            return a_;
        } else if (flag2_ == 1) {
            flag2_ = NUM_2;
            return b_;
        } else {
            flag2_ = 0;
            return c_;
        }
    }
private:
    MutexBuffer<bufferType> a_;
    MutexBuffer<bufferType> b_;
    MutexBuffer<bufferType> c_;
    uint32_t flag1_ = 0;
    uint32_t flag2_ = 0;
};
}
#endif
