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
 * \file buffer_policy.h
 * \brief 综合管理buffer的内存和同步
 */
#ifndef BUFFER_POLICY_H
#define BUFFER_POLICY_H

#include "./buffer_manager.h"
#define NUM_2 2
#define NUM_4 4
namespace fa_base_matmul {
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BufferPolicyDB {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size)
    {
        ping_ = bufferManager.template AllocBuffer<syncType>(size);
        pong_ = bufferManager.template AllocBuffer<syncType>(size);

        ping_.Init();
        pong_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager)
    {
        ping_.UnInit();
        pong_.UnInit();

        bufferManager.FreeBuffer(ping_);
        bufferManager.FreeBuffer(pong_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get()
    {
        if (flag1_) { // 1
            flag1_ = 0;
            return ping_;
        } else { // 0
            flag1_ = 1;
            return pong_;
        }
    }

    // 需要与Get联用， 首次调用Get，第二次调用GetPre(Q复用)
    __aicore__ inline Buffer<bufferType, syncType> &GetPre()
    {
        if (flag1_) { // 0->1
            return pong_;
        } else { // 1->0
            return ping_;
        }
    }

    // 需要与Get,GetPre联用， 首次调用Get，第二次调用GetPre,第三次复用时GetReused(KV复用)
    __aicore__ inline Buffer<bufferType, syncType> &GetReused()
    {
        if (flag2_ == 0) {
            flag2_ = 1;
            return pong_;
        } else {
            flag2_ = 0;
            return ping_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetReused(bool isNextS2IdxNoChange)
    {
        if (isNextS2IdxNoChange) {
            if (flag2_ == 0) {
                return pong_;
            } else {
                return ping_;
            }
        } else {
            return GetReused();
        }
    }

private:
    Buffer<bufferType, syncType> ping_;
    Buffer<bufferType, syncType> pong_;
    uint32_t flag1_ = 0;
    uint32_t flag2_ = 0;
};

// 申请3个buffer, 轮转
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BufferPolicy3buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size)
    {
        a_ = bufferManager.template AllocBuffer<syncType>(size);
        b_ = bufferManager.template AllocBuffer<syncType>(size);
        c_ = bufferManager.template AllocBuffer<syncType>(size);

        a_.Init();
        b_.Init();
        c_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager)
    {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();

        bufferManager.FreeBuffer(a_);
        bufferManager.FreeBuffer(b_);
        bufferManager.FreeBuffer(c_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get()
    {
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

    __aicore__ inline Buffer<bufferType, syncType> &GetVec()
    { // mixcore architecture
        if (flag1_vec1_ == 0) {
            flag1_vec1_ = 1;
            return a_;
        } else if (flag1_vec1_ == 1) {
            flag1_vec1_ = NUM_2;
            return b_;
        } else {
            flag1_vec1_ = 0;
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetCube()
    { // mixcore architecture
        if (flag1_bmm2_ == 0) {
            flag1_bmm2_ = 1;
            return a_;
        } else if (flag1_bmm2_ == 1) {
            flag1_bmm2_ = NUM_2;
            return b_;
        } else {
            flag1_bmm2_ = 0;
            return c_;
        }
    }

    // Q复用
    __aicore__ inline Buffer<bufferType, syncType> &GetPre()
    {
        if (flag1_ == 0) {
            return c_;
        } else if (flag1_ == 1) {
            return a_;
        } else {
            return b_;
        }
    }

    // KV复用
    __aicore__ inline Buffer<bufferType, syncType> &GetReused()
    {
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
    Buffer<bufferType, syncType> a_;
    Buffer<bufferType, syncType> b_;
    Buffer<bufferType, syncType> c_;
    uint32_t flag1_ = 0;
    uint32_t flag1_vec1_ = 0;
    uint32_t flag1_bmm2_ = 0;
    uint32_t flag2_ = 0;
};

// 申请4个buffer + kv复用
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BufferPolicy4buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size)
    {
        a_ = bufferManager.template AllocBuffer<syncType>(size);
        b_ = bufferManager.template AllocBuffer<syncType>(size);
        c_ = bufferManager.template AllocBuffer<syncType>(size);
        d_ = bufferManager.template AllocBuffer<syncType>(size);

        a_.Init();
        b_.Init();
        c_.Init();
        d_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager)
    {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();
        d_.UnInit();

        bufferManager.FreeBuffer(a_);
        bufferManager.FreeBuffer(b_);
        bufferManager.FreeBuffer(c_);
        bufferManager.FreeBuffer(d_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get(uint32_t id)
    {
        uint32_t flag = id % 4;
        if (flag == 0) {
            return a_;
        } else if (flag == 1) {
            return b_;
        } else if (flag == 2) { // 2:c_
            return c_;
        } else {
            return d_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get()
    {
        auto &buffer = Get(head_);
        head_++;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetReused()
    {
        auto &buffer = Get(used_);
        used_ = (used_ - tail_ + 1) % (head_ - tail_) + tail_;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetFree()
    {
        if (tail_ == used_) {
            used_++;
        }
        auto &buffer = Get(tail_);
        tail_++;
        return buffer;
    }

private:
    Buffer<bufferType, syncType> a_;
    Buffer<bufferType, syncType> b_;
    Buffer<bufferType, syncType> c_;
    Buffer<bufferType, syncType> d_;
    uint32_t tail_ = 0; // 表示当前正在使用的buffer队列队尾
    uint32_t head_ = 0; // 表示当前正在使用的buffer队列队首+1
    uint32_t used_ = 0; // 表示当前正在使用的buffer，于首尾间，左闭右开
};

} // namespace fa_base_matmul
#endif
