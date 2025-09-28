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
 * \file buffer.h
 * \brief同步管理
 */
#ifndef BUFFER_H
#define BUFFER_H
#include<type_traits>
#include"lib/matmul_intf.h"
#include"kernel_event.h"
#include"kernel_common.h"
#include"kernel_tpipe.h"
namespace fa_base_matmul {
enum class BufferType {
    L1 = 0,
    L0A = 1,
    L0B = 2,
    L0C = 3,
};

template<BufferType Type>
struct BufferInfo{
    // Cons 消费者，Prod 生产者
    __aicore__ const static constexpr HardEvent ConsWaitProdStatus() {
        if constexpr (Type == BufferType::L1){
            return HardEvent::MTE2_MTE1;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::M_FIX;
        }
    }

    __aicore__ const static constexpr HardEvent ProdWaitConsStatus() {
        if constexpr (Type == BufferType::L1){
            return HardEvent::MTE1_MTE2;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::FIX_M;
        }
    }

    __aicore__ const static constexpr TPosition GetTPosition() {
        if constexpr (Type == BufferType::L1){
            return TPosition::A1;
        } else if constexpr (Type == BufferType::L0A) {
            return TPosition::A2;
        } else if constexpr (Type == BufferType::L0B) {
            return TPosition::B2;
        } else if constexpr (Type == BufferType::L0C) {
            return TPosition::CO1;
        }
    }

    static constexpr HardEvent EventP2C = ConsWaitProdStatus(); // 生产者到消费者方向的HardEvent：消费者等生产者提供/生产者通知消费者已生成
    static constexpr HardEvent EventC2P = ProdWaitConsStatus(); // 消费者到生产者方向的HardEvent：生产者等消费者消耗/消费者通知生产者已消耗’
    static constexpr TPosition Position = GetTPosition();
};

// buffer绑定生产者、消费者关系
// L1 buffer的生产者为MTE2，消费者为MTE1
// L0A buffer的生产者为MTE1，消费者为M
// L0B buffer的生产者为MTE1，消费者为M
// L0C buffer的生产者为M，消费者为FIX
template<BufferType Type, bool sync = true>
class Buffer {
public:
    __aicore__ inline Buffer() {}
    __aicore__ inline Buffer(LocalTensor<uint8_t> tensor, uint32_t size) {
        tensor_ = tensor;
        size_ = size;
    }

    __aicore__ inline void Init() {
        if ASCEND_IS_AIC {
            if constexpr (sync) {
                p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<Type>::EventP2C>(); // 确保只能被调用一次
                c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<Type>::EventC2P>();
                SetFlag<BufferInfo<Type>::EventC2P>(c2pEventId_);
            }
        }
    }

    __aicore__ inline void UnInit() {
        if ASCEND_IS_AIC {
            if constexpr (sync) {
                WaitFlag<BufferInfo<Type>::EventC2P>(c2pEventId_);
                GetTPipePtr()->ReleaseEventID<BufferInfo<Type>::EventP2C>(p2cEventId_); // 确保只能被调用一次
                GetTPipePtr()->ReleaseEventID<BufferInfo<Type>::EventC2P>(c2pEventId_);
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Wait() {
        if ASCEND_IS_AIC {
            if constexpr (sync) {
                if constexpr (EventType == BufferInfo<Type>::EventP2C) {
                    WaitFlag<BufferInfo<Type>::EventP2C>(p2cEventId_); // 消费者等待生产者完成生产
                } else {
                    WaitFlag<BufferInfo<Type>::EventC2P>(c2pEventId_); // 生产者等待消费者完成消费
                }
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Set() {
        if ASCEND_IS_AIC {
            if constexpr (sync) {
                if constexpr (EventType == BufferInfo<Type>::EventP2C) {
                    SetFlag<BufferInfo<Type>::EventP2C>(p2cEventId_); // 生产者通知消费者已完成生产
                } else {
                    SetFlag<BufferInfo<Type>::EventC2P>(c2pEventId_); // 消费者通知生产者已完成消费
                }
            }
        }
    }

    template<typename T>
    __aicore__ inline LocalTensor<T> GetTensor() {
        return tensor_.template ReinterpretCast<T>();
    }

    template<typename T>
    __aicore__ inline LocalTensor<T> GetTensor(uint64_t startindex) {
        LocalTensor<T> tmpTensor = tensor_.template ReinterpretCast<T>();
        return tmpTensor[startindex];
    }

private:
    LocalTensor<uint8_t> tensor_;
    uint32_t size_;
    TEventID p2cEventId_;
    TEventID c2pEventId_;
};
}
#endif