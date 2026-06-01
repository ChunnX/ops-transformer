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
 * \file fa_static_buffer.h
 * \brief 仅为 matmul.h 提供最小 L0A/L0B Buffer/BuffersPolicy 接口。
 *        L1/L0C/UB/GM 缓冲区和核间同步全部在 C/V 核代码中用裸 LocalTensor
 *        + SetFlag/WaitFlag/CrossCoreSetFlag/CrossCoreWaitFlag 直接实现，
 *        不依赖 common/op_kernel 下的 buffer.h/buffer_manager.h/buffers_policy.h。
 *
 *        通过预定义 BUFFER_H/BUFFER_MANAGER_H/BUFFERS_POLICY_H include-guard 宏，
 *        阻止 matmul.h 的 #include "buffers_policy.h" 拉入 common 版本。
 */
#ifndef FA_STATIC_BUFFER_H
#define FA_STATIC_BUFFER_H

// ===== 预定义common buffer文件的include守卫，阻止matmul.h拉入common版本 =====
#ifndef BUFFER_H
#define BUFFER_H
#endif
#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H
#endif
#ifndef BUFFERS_POLICY_H
#define BUFFERS_POLICY_H
#endif

// buffers_policy.h 原本定义的数值宏，随守卫一并提供
#ifndef NUM_2
#define NUM_2 2
#endif
#ifndef NUM_3
#define NUM_3 3
#endif
#ifndef NUM_4
#define NUM_4 4
#endif

#include "lib/matmul_intf.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

namespace fa_base_matmul {

// ============================================================
// 仅保留 matmul.h 内部所需的最小类型：
//   BufferType (L0A, L0B)
//   BufferInfo<L0A>, BufferInfo<L0B>
//   Buffer<L0A/L0B, INNER_CORE_SYNC>
//   BuffersPolicySingleBuffer<L0A/L0B>
//   BuffersPolicyDB<L0A/L0B>
// ============================================================

enum class BufferType {
    L0A = 1,
    L0B = 2,
};

enum class SyncType {
    INNER_CORE_SYNC = 1,
};

template<BufferType Type>
struct BufferInfo {
    __aicore__ const static constexpr HardEvent ConsWaitProdStatus() {
        // L0A/L0B: 生产者=MTE1, 消费者=M(matmul)
        return HardEvent::MTE1_M;
    }
    __aicore__ const static constexpr HardEvent ProdWaitConsStatus() {
        return HardEvent::M_MTE1;
    }
    __aicore__ const static constexpr TPosition GetTPosition() {
        if constexpr (Type == BufferType::L0A) {
            return TPosition::A2;
        } else {
            return TPosition::B2;
        }
    }
    static constexpr HardEvent EventP2C = ConsWaitProdStatus();
    static constexpr HardEvent EventC2P = ProdWaitConsStatus();
    static constexpr TPosition Position = GetTPosition();
};

// L0A/L0B Buffer —— 仅供 matmul.h 内部使用
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class Buffer {
public:
    __aicore__ inline Buffer() {}

    __aicore__ inline void InitStaticDirect(uint32_t offset, uint32_t size, TEventID p2cId, TEventID c2pId) {
        tensor_ = LocalTensor<uint8_t>(BufferInfo<bufferType>::Position, offset, size);
        if ASCEND_IS_AIC {
            p2cEventId_ = p2cId;
            c2pEventId_ = c2pId;
        }
    }

    __aicore__ inline void UnInitStatic() {
        if ASCEND_IS_AIC {
            WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Wait() {
        if ASCEND_IS_AIC {
            if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                WaitFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_);
            } else {
                WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Set() {
        if ASCEND_IS_AIC {
            if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                SetFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_);
            } else {
                SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    template<typename T>
    __aicore__ inline LocalTensor<T> GetTensor() {
        return tensor_.template ReinterpretCast<T>();
    }

private:
    LocalTensor<uint8_t> tensor_;
    TEventID p2cEventId_;
    TEventID c2pEventId_;
};

// 单buffer策略 —— 仅L0A/L0B供matmul使用
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicySingleBuffer {
public:
    __aicore__ inline void InitStaticDirect(uint32_t offset, uint32_t size, TEventID p2cId, TEventID c2pId) {
        buffer_.InitStaticDirect(offset, size, p2cId, c2pId);
    }
    __aicore__ inline void UninitStaticDirect() {
        buffer_.UnInitStatic();
    }
    __aicore__ inline Buffer<bufferType, syncType> &Get() { return buffer_; }
    __aicore__ inline Buffer<bufferType, syncType> &GetPre() { return buffer_; }
private:
    Buffer<bufferType, syncType> buffer_;
};

// 乒乓buffer策略 —— 仅L0A/L0B供matmul使用
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicyDB {
public:
    __aicore__ inline void InitStaticDirect(uint32_t pingOffset, uint32_t pongOffset, uint32_t size,
                                            TEventID pingP2cId, TEventID pingC2pId,
                                            TEventID pongP2cId, TEventID pongC2pId) {
        ping_.InitStaticDirect(pingOffset, size, pingP2cId, pingC2pId);
        pong_.InitStaticDirect(pongOffset, size, pongP2cId, pongC2pId);
    }
    __aicore__ inline void UninitStaticDirect() {
        ping_.UnInitStatic();
        pong_.UnInitStatic();
    }
    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag_) {
            flag_ = 0;
            return ping_;
        } else {
            flag_ = 1;
            return pong_;
        }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag_) {
            return pong_;
        } else {
            return ping_;
        }
    }
private:
    Buffer<bufferType, syncType> ping_;
    Buffer<bufferType, syncType> pong_;
    uint32_t flag_ = 0;
};

} // namespace fa_base_matmul

#endif // FA_STATIC_BUFFER_H
