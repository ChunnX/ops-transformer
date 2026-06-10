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
 * \file buffer.h
 * \brief同步管理
 */
#ifndef TENSOR_MANAGER_H
#define TENSOR_MANAGER_H
#include <type_traits>
#include "lib/matmul_intf.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;
namespace fa_base_matmul {
__BLOCK_LOCAL__ __inline__ uint32_t idCounterNum;
#define MAKE_ID ((++idCounterNum) % 11)

// 核间同步中，AIC(flagId 0-10)对应AIV0(flagId 0-10)，对应AIV1(flagId 16-26)
#define AIV0_AIV1_OFFSET 16

enum class BufferType {
    L1 = 0,
    L0A = 1,
    L0B = 2,
    L0C = 3,
    UB = 4,
    GM = 5,
    C2 = 6,
};

enum class SyncType {
    NO_SYNC,
    INNER_CORE_SYNC,
    CROSS_CORE_SYNC_FORWARD,
    CROSS_CORE_SYNC_BOTH,
    CROSS_CORE_SYNC_BACKWARD,
};

constexpr uint32_t INVALID_CROSS_CORE_EVENT_ID = 16;
static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;

template<AscendC::Te::Location Type>
struct BufferInfo{
    // Cons 消费者，Prod 生产者
    __aicore__ const static constexpr HardEvent ConsWaitProdStatus() {
        if constexpr (Type == AscendC::Te::Location::L1) {
            return HardEvent::MTE2_MTE1;
        } else if constexpr (Type == AscendC::Te::Location::L0A) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == AscendC::Te::Location::L0B) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == AscendC::Te::Location::L0C) {
            return HardEvent::M_FIX;
        } else if constexpr (Type == AscendC::Te::Location::GM) {
            return HardEvent::MTE2_S;
        }
    }

    __aicore__ const static constexpr HardEvent ProdWaitConsStatus() {
        if constexpr (Type == AscendC::Te::Location::L1) {
            return HardEvent::MTE1_MTE2;
        } else if constexpr (Type == AscendC::Te::Location::L0A) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == AscendC::Te::Location::L0B) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == AscendC::Te::Location::L0C) {
            return HardEvent::FIX_M;
        } else if constexpr (Type == AscendC::Te::Location::GM) {
            return HardEvent::S_MTE2;
        }
    }

    __aicore__ const static constexpr TPosition GetTPosition() {
        return Type;
    }

    static constexpr HardEvent EventP2C = ConsWaitProdStatus(); // 生产者到消费者方向的HardEvent：消费者等生产者提供/生产者通知消费者已生成
    static constexpr HardEvent EventC2P = ProdWaitConsStatus(); // 消费者到生产者方向的HardEvent：生产者等消费者消耗/消费者通知生产者已消耗’
    static constexpr TPosition Position = GetTPosition();
};

// buffer绑定生产者、消费者关系
// L1 buffer的生产者为MTE2或者MTE3，消费者为MTE1
// L0A buffer的生产者为MTE1，消费者为M
// L0B buffer的生产者为MTE1，消费者为M
// L0C buffer的生产者为M，消费者为FIX
template<AscendC::Te::Location bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC, typename TensorType>
class Buffer {
public:
    __aicore__ inline Buffer() {}
    __aicore__ inline Buffer(TensorType tensor, uint32_t size) {
        tensor_ = tensor;
        size_ = size;
        if constexpr (syncType == SyncType::CROSS_CORE_SYNC_FORWARD) {
            id0_ = MAKE_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = MAKE_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
            id0_ = MAKE_ID;
            id1_ = MAKE_ID;
        } else {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        }
    }

    __aicore__ inline void Init() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // 确保只能被调用一次
                c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
                SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    __aicore__ inline void UnInit() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 确保只能被调用一次
                GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Wait() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                    WaitFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 消费者等待生产者完成生产
                } else {
                    WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_); // 生产者等待消费者完成消费
                }
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Set() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                    SetFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 生产者通知消费者已完成生产
                } else {
                    SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_); // 消费者通知生产者已完成消费
                }
            }
        }
    }

    __aicore__ inline void SetEventID() {
        if ASCEND_IS_AIC {
            p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // 确保只能被调用一次
            c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
        }
    }

    template<HardEvent EventType>
    __aicore__ inline TEventID GetEventID() {
        if ASCEND_IS_AIC {
            if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                return p2cEventId_; // 生产者通知消费者已完成生产
            } else {
                return c2pEventId_; // 消费者通知生产者已完成消费
            }
        }
    }

    __aicore__ inline void SetCrossCoreID(uint32_t id0, uint32_t id1)
    {
        id0_ = id0;
        id1_ = id1;
    }

    template<bool isReuse = false>
    __aicore__ inline void WaitCrossCore() {
        if constexpr (bufferType == AscendC::Te::Location::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id0_);
            }
        } else if constexpr (bufferType == AscendC::Te::Location::UB || bufferType == AscendC::Te::Location::GM) {
            // AIC属于生产者，AIV属于消费者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (isReuse) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id0_);
                } else {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id0_);
                }
            }
        } else if constexpr (bufferType == AscendC::Te::Location::L1) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
                }
            }
        }
    }

    template<bool isReuse = false>
    __aicore__ inline void SetCrossCore() {
        if constexpr (bufferType == AscendC::Te::Location::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
            }
        } else if constexpr (bufferType == AscendC::Te::Location::UB || bufferType == AscendC::Te::Location::GM) {
            // AIC属于生产者，AIV属于消费者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (isReuse) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
                } else {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id1_);
                }
            }
        } else if constexpr (bufferType == AscendC::Te::Location::L1) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_);
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_ + AIV0_AIV1_OFFSET);
                }
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id0_);
            }
        }
    }

    template<typename T>
    __aicore__ inline TargetTensorType<T> GetTensor() {
        return tensor_.template ReinterpretCast<T>();
    }

    template<typename T>
    __aicore__ inline TargetTensorType<T> GetTensor(uint64_t startindex) {
        TargetTensorType<T> tmpTensor = tensor_.template ReinterpretCast<T>();
        return tmpTensor[startindex];
    }

private:
    TensorType tensor_;
    uint32_t size_;
    TEventID p2cEventId_;
    TEventID c2pEventId_;
    uint32_t id0_;      // 用作正向同步：生产者通知消费者，或者消费者等待生产者；
    uint32_t id1_;      // 用作反向同步：消费者通知生产者，或者生产者等待消费者；
};
}

/*!
 * \file buffer_manager.h
 * \brief buffer内存管理
 */
// L1  TPosition::A1
// L0A TPosition::A2
// L0B TPosition::B2
// L0C TPosition::CO1
// UB  TPosition::VECIN
namespace fa_base_matmul {
template<AscendC::Te::Location bufferType>
class BufferManager {
    using TensorType = std::conditional_t<bufferType == AscendC::Te::Location::GM, GlobalTensor<uint8_t>, LocalTensor<uint8_t>>;
public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t size) {
        static_assert(bufferType != AscendC::Te::Location::GM, "GM should use workspace.");
        TBuf<BufferInfo<bufferType>::Position> tbuf;
        pipe->InitBuffer(tbuf, size);
        mem_ = tbuf.template Get<uint8_t>();
    }

    __aicore__ inline void Init(__gm__ uint8_t* workspace) {
        static_assert(bufferType == AscendC::Te::Location::GM, "AscendC::Te::Location should be GM.");
        mem_.SetGlobalBuffer((__gm__ uint8_t*)workspace);
    }

    template<SyncType syncType = SyncType::INNER_CORE_SYNC>
    __aicore__ inline Buffer<bufferType, syncType> AllocBuffer(uint32_t size) {
        // auto layoutQL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<Q_T>>(runInfo.actMSize, constInfo.dSize);
        // auto tensorQL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, Q_T>(offsetQL1), layoutQL1);
        TensorType temp = mem_[offset_];
        offset_ += size;
        return Buffer<bufferType, syncType>(temp, size);
    }

    template<SyncType syncType = SyncType::INNER_CORE_SYNC>
    __aicore__ inline void FreeBuffer(Buffer<bufferType, syncType> &buffer){
    }
private:
    uint32_t offset_ = 0;
    TensorType mem_;
};
}


/*!
 * \file buffers_policy.h
 * \brief 综合管理buffer的内存和同步
 */
#define NUM_2 2
#define NUM_3 3
#define NUM_4 4
// Q复用 KV复用
// 申请单块buffer
namespace fa_base_matmul {
template<AscendC::Te::Location bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicySingleBuffer {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size){
        buffer_ = bufferManager.template AllocBuffer<syncType>(size);
        buffer_.Init();
    }
 
    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager){
        buffer_.UnInit();
        bufferManager.FreeBuffer(buffer_);
    }
 
    __aicore__ inline Buffer<bufferType, syncType> &Get(){
        return buffer_;
    }
 
    __aicore__ inline Buffer<bufferType, syncType> &GetPre(){
        return Get();
    }
 
    __aicore__ inline Buffer<bufferType, syncType> &GetReused(){
        return Get();
    }
private:
    Buffer<bufferType, syncType> buffer_;
};

// 申请2个buffer，乒乓轮转
template<AscendC::Te::Location bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicyDB {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size){
        ping_ = bufferManager.template AllocBuffer<syncType>(size);
        pong_ = bufferManager.template AllocBuffer<syncType>(size);

        ping_.Init();
        pong_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager){
        ping_.UnInit();
        pong_.UnInit();

        bufferManager.FreeBuffer(ping_);
        bufferManager.FreeBuffer(pong_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag1_) { // 1
            flag1_ = 0;
            return ping_;
        } else { // 0
            flag1_ = 1;
            return pong_;
        }
    }

    // 需要与Get联用， 首次调用Get，第二次调用GetPre(Q复用)
    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag1_) { // 0->1
            return pong_;
        } else { // 1->0
            return ping_;
        }
    }

    // 需要与Get,GetPre联用， 首次调用Get，第二次调用GetPre,第三次复用时GetReused(KV复用)
    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        if (flag2_ == 0) {
            flag2_ = 1;
            return pong_;
        } else { 
            flag2_ = 0;
            return ping_;
        }
    }
 
    __aicore__ inline Buffer<bufferType, syncType> &GetReused(bool isNextS2IdxNoChange) {
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
template<AscendC::Te::Location bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicy3buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        a_ = bufferManager.template AllocBuffer<syncType>(size);
        b_ = bufferManager.template AllocBuffer<syncType>(size);
        c_ = bufferManager.template AllocBuffer<syncType>(size);

        a_.Init();
        b_.Init();
        c_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();

        bufferManager.FreeBuffer(a_);
        bufferManager.FreeBuffer(b_);
        bufferManager.FreeBuffer(c_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
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

    __aicore__ inline Buffer<bufferType, syncType> &GetVec() { // mixcore architecture
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

    __aicore__ inline Buffer<bufferType, syncType> &GetCube() { // mixcore architecture
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
    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag1_ == 0) {
            return c_;
        } else if (flag1_ == 1) { 
            return a_;
        } else {
            return b_;
        }
    }

    // KV复用
    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        if (flag2_ == 0) { 
            flag2_ = 1;
            return a_;
        } else if (flag2_ == 1){ 
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
template<AscendC::Te::Location bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicy4buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        a_ = bufferManager.template AllocBuffer<syncType>(size);
        b_ = bufferManager.template AllocBuffer<syncType>(size);
        c_ = bufferManager.template AllocBuffer<syncType>(size);
        d_ = bufferManager.template AllocBuffer<syncType>(size);

        a_.Init();
        b_.Init();
        c_.Init();
        d_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();
        d_.UnInit();

        bufferManager.FreeBuffer(a_);
        bufferManager.FreeBuffer(b_);
        bufferManager.FreeBuffer(c_);
        bufferManager.FreeBuffer(d_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get(uint32_t id) {
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

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        auto& buffer = Get(head_);
        head_++;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        auto& buffer = Get(used_);
        used_ = (used_ - tail_ + 1) % (head_ - tail_) + tail_;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetFree() {
        if (tail_ == used_) {
            used_++;
        }
        auto& buffer = Get(tail_);
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

template<AscendC::Te::Location bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class Matrix2x2BufferPolicy { // 4buffer
// 二维buffer管理，地址行优先，使用列优先
// MracBuffer:memory address with row first, alloc/use/free with column first
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        bufferM0k0_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM0k1_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM1k0_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM1k1_ = bufferManager.template AllocBuffer<syncType>(size);

        bufferM0k0_.Init();
        bufferM0k1_.Init();
        bufferM1k0_.Init();
        bufferM1k1_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        bufferM0k0_.UnInit();
        bufferM0k1_.UnInit();
        bufferM1k0_.UnInit();
        bufferM1k1_.UnInit();

        bufferManager.FreeBuffer(bufferM0k0_);
        bufferManager.FreeBuffer(bufferM0k1_);
        bufferManager.FreeBuffer(bufferM1k0_);
        bufferManager.FreeBuffer(bufferM1k1_);
    }

    __aicore__ inline void SetMExtent(int32_t mExtent) {
        aIdx_ = -1;
        amIdx_ = (amIdx_ + mSize_ - 1) % mSize_; // 翻转 0->1, 1->0
        akIdx_ = 0;

        uIdx_ = -1;
        umIdx_ = (umIdx_ + mSize_ - 1) % mSize_;
        ukIdx_ = 0;

        fIdx_ = -1;
        fmIdx_ = (fmIdx_ + mSize_ - 1) % mSize_;
        fkIdx_ = 0;

        mExtent_ = mExtent;
    }

    __aicore__ inline Buffer<bufferType, syncType> &AllocNext() {
        aIdx_++;
        return GetBuffer(aIdx_, amIdx_, akIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &ReuseNext() {
        uIdx_++;
        return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &FreeNext() {
        fIdx_++;
        return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &PeekNextK() { // 在Alloc阶段使用，k方向取下一个
        return PeekBuffer(amIdx_, (1 - akIdx_)); // k翻转
    }
private:
    __aicore__ inline Buffer<bufferType, syncType> &GetBuffer(int32_t xIdx, int32_t &mIdx, int32_t &kIdx) {
        // xIdx为入参，表示当前alloc/use/free的idx，mIdx和kIdx为下标出参，移动到下一个buffer并获取
        mIdx = (mIdx + mExtent_ - 1) % mExtent_;
        kIdx = (xIdx / mExtent_) % kSize_;
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if (mIdx == 1 && kIdx == 0) {
            return bufferM1k0_;
        } else { // 该分支条件为：mIdx == 1 && kIdx == 1
            return bufferM1k1_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &PeekBuffer(int32_t mIdx, int32_t kIdx) {
        // 只访问buffer，不进行下标移动
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if ((mIdx == 1) && (kIdx == 0)) {
            return bufferM1k0_;
        } else { // mIdx == 1 && kIdx == 1
            return bufferM1k1_;
        }
    }

    Buffer<bufferType, syncType> bufferM0k0_;
    Buffer<bufferType, syncType> bufferM0k1_;
    Buffer<bufferType, syncType> bufferM1k0_;
    Buffer<bufferType, syncType> bufferM1k1_;
    int32_t mSize_ = 2; // m的总buffer数
    int32_t kSize_ = 2; // k的总buffer数

    // Alloc
    int32_t aIdx_ = -1; // 当前第几次Alloc Buffer
    int32_t amIdx_ = 0; // 当前Alloc Buffer的m下标
    int32_t akIdx_ = 0; // 当前Alloc Buffer的k下标

    // Reuse
    int32_t uIdx_ = -1;
    int32_t umIdx_ = 0;
    int32_t ukIdx_ = 0;

    // Free
    int32_t fIdx_ = -1;
    int32_t fmIdx_ = 0;
    int32_t fkIdx_ = 0;

    int32_t mExtent_ = 0; // m实际使用的大小，可以为1或者2
};
}
#endif