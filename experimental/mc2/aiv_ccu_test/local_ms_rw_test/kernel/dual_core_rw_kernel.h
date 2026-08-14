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
 * \file dual_core_rw_kernel.h
 * \brief 双核混合读写 Kernel
 *        DualCoreRw:       同一 AI Core 内核0写核1读，共享 UB，数据量上限 120KB
 *        DualCoreCrossRw:  跨 AI Core 核0写核2读，独立 UB，数据量上限 240KB，核1空转
 */

#ifndef DUAL_CORE_RW_KERNEL_H
#define DUAL_CORE_RW_KERNEL_H

#include "kernel_operator.h"

namespace LocalMsRwImpl {

using namespace AscendC;

constexpr uint32_t DUAL_MS_ID = 1U;

// ========== 同一 AI Core 双核混合读写 ==========
// 核0写 MS1，核1读 MS1，共享 UB (248KB)，每核单 buffer 上限 120KB
template <typename T>
class DualCoreRw {
public:
    __aicore__ inline DualCoreRw(){};
    __aicore__ inline void Init(uint32_t enableWarmup, uint32_t msId,
                                uint32_t loopCount, uint64_t dataSize,
                                GM_ADDR msAddrs[4], GM_ADDR timeAddr, GM_ADDR timeAddr2,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    uint32_t enableWarmup_{0};
    bool sameAddr_{false};
    uint32_t loopCount_{1};
    uint64_t dataSize_{0};
    uint32_t aivId_{0};

    GlobalTensor<T> msGm_;
    GlobalTensor<uint64_t> timeGm_;
    GlobalTensor<uint64_t> timeGm2_;

    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readBuf_;
    LocalTensor<T> dataLocal_;
    LocalTensor<T> readLocal_;
    uint32_t totalElem_{0};
};

template <typename T>
__aicore__ inline void DualCoreRw<T>::Init(uint32_t enableWarmup, uint32_t msId,
                                           uint32_t loopCount, uint64_t dataSize,
                                           GM_ADDR msAddrs[4], GM_ADDR timeAddr, GM_ADDR timeAddr2,
                                           TPipe *pipe)
{
    enableWarmup_ = enableWarmup;
    sameAddr_ = (msId == 0);
    loopCount_ = loopCount == 0 ? 1 : loopCount;
    dataSize_ = dataSize;
    aivId_ = GetBlockIdx();

    msGm_.SetGlobalBuffer((__gm__ T *)msAddrs[DUAL_MS_ID]);
    msGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);
    timeGm2_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr2);

    if (aivId_ == 0) {
        pipe->InitBuffer(dataBuf_, dataSize);
        dataLocal_ = dataBuf_.Get<T>();
    } else {
        pipe->InitBuffer(readBuf_, dataSize);
        readLocal_ = readBuf_.Get<T>();
    }
    totalElem_ = static_cast<uint32_t>(dataSize / sizeof(T));
}

template <typename T>
__aicore__ inline void DualCoreRw<T>::Process()
{
    if (aivId_ == 0) {
        Duplicate<T>(dataLocal_, (T)2, totalElem_);
        PipeBarrier<PIPE_ALL>();

        uint64_t warmupCycles = 0;
        if (enableWarmup_ != 0) {
            uint64_t warmupStart = GetSystemCycle();
            for (uint32_t w = 0; w < enableWarmup_; w++) {
                DataCopy(msGm_, dataLocal_, totalElem_);
                PipeBarrier<PIPE_ALL>();
            }
            uint64_t warmupEnd = GetSystemCycle();
            warmupCycles = warmupEnd - warmupStart;
        }

        CrossCoreSetFlag<0x0, PIPE_ALL>(1);
        CrossCoreWaitFlag<0x0, PIPE_ALL>(2);

        uint64_t startCycle = GetSystemCycle();
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopy(msGm_, dataLocal_, totalElem_);
            PipeBarrier<PIPE_ALL>();
        }
        uint64_t endCycle = GetSystemCycle();

        timeGm_.SetValue(0, startCycle);
        timeGm_.SetValue(1, endCycle);
        timeGm_.SetValue(2, 0);
        timeGm_.SetValue(3, warmupCycles);
        DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(timeGm_);
    } else {
        uint64_t readOffset = sameAddr_ ? 0 : totalElem_;
        uint64_t warmupCycles = 0;
        if (enableWarmup_ != 0) {
            uint64_t warmupStart = GetSystemCycle();
            for (uint32_t w = 0; w < enableWarmup_; w++) {
                DataCopy(readLocal_, msGm_[readOffset], totalElem_);
                PipeBarrier<PIPE_ALL>();
            }
            uint64_t warmupEnd = GetSystemCycle();
            warmupCycles = warmupEnd - warmupStart;
        }

        CrossCoreWaitFlag<0x0, PIPE_ALL>(1);
        CrossCoreSetFlag<0x0, PIPE_ALL>(2);

        uint64_t startCycle = GetSystemCycle();
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopy(readLocal_, msGm_[readOffset], totalElem_);
            PipeBarrier<PIPE_ALL>();
        }
        uint64_t endCycle = GetSystemCycle();

        timeGm2_.SetValue(0, startCycle);
        timeGm2_.SetValue(1, endCycle);
        timeGm2_.SetValue(2, warmupCycles);
        timeGm2_.SetValue(3, 0);
        DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(timeGm2_);
    }
}

// ========== 跨 AI Core 双核混合读写 ==========
// 核0写 MS1，核2读 MS1，核1空转
// blockDim=3, 每个核独立 UB (248KB), 数据量可到 240KB
template <typename T>
class DualCoreCrossRw {
public:
    __aicore__ inline DualCoreCrossRw(){};
    __aicore__ inline void Init(uint32_t enableWarmup, uint32_t msId,
                                uint32_t loopCount, uint64_t dataSize,
                                GM_ADDR msAddrs[4], GM_ADDR timeAddr, GM_ADDR timeAddr2,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    uint32_t enableWarmup_{0};
    bool sameAddr_{false};
    uint32_t loopCount_{1};
    uint64_t dataSize_{0};
    uint32_t aivId_{0};

    GlobalTensor<T> msGm_;
    GlobalTensor<uint64_t> timeGm_;
    GlobalTensor<uint64_t> timeGm2_;

    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readBuf_;
    LocalTensor<T> dataLocal_;
    LocalTensor<T> readLocal_;
    uint32_t totalElem_{0};
};

template <typename T>
__aicore__ inline void DualCoreCrossRw<T>::Init(uint32_t enableWarmup, uint32_t msId,
                                                uint32_t loopCount, uint64_t dataSize,
                                                GM_ADDR msAddrs[4], GM_ADDR timeAddr, GM_ADDR timeAddr2,
                                                TPipe *pipe)
{
    enableWarmup_ = enableWarmup;
    sameAddr_ = (msId == 0);
    loopCount_ = loopCount == 0 ? 1 : loopCount;
    dataSize_ = dataSize;
    aivId_ = GetBlockIdx();

    msGm_.SetGlobalBuffer((__gm__ T *)msAddrs[DUAL_MS_ID]);
    msGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);
    timeGm2_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr2);

    if (aivId_ == 0) {
        pipe->InitBuffer(dataBuf_, dataSize);
        dataLocal_ = dataBuf_.Get<T>();
    } else if (aivId_ == 2) {
        pipe->InitBuffer(readBuf_, dataSize);
        readLocal_ = readBuf_.Get<T>();
    }
    totalElem_ = static_cast<uint32_t>(dataSize / sizeof(T));
}

template <typename T>
__aicore__ inline void DualCoreCrossRw<T>::Process()
{
    if (aivId_ == 1) return;

    if (aivId_ == 0) {
        Duplicate<T>(dataLocal_, (T)2, totalElem_);
        PipeBarrier<PIPE_ALL>();

        uint64_t warmupCycles = 0;
        if (enableWarmup_ != 0) {
            uint64_t warmupStart = GetSystemCycle();
            for (uint32_t w = 0; w < enableWarmup_; w++) {
                DataCopy(msGm_, dataLocal_, totalElem_);
                PipeBarrier<PIPE_ALL>();
            }
            uint64_t warmupEnd = GetSystemCycle();
            warmupCycles = warmupEnd - warmupStart;
        }

        CrossCoreSetFlag<0x0, PIPE_ALL>(1);
        CrossCoreWaitFlag<0x0, PIPE_ALL>(2);

        uint64_t startCycle = GetSystemCycle();
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopy(msGm_, dataLocal_, totalElem_);
            PipeBarrier<PIPE_ALL>();
        }
        uint64_t endCycle = GetSystemCycle();

        timeGm_.SetValue(0, startCycle);
        timeGm_.SetValue(1, endCycle);
        timeGm_.SetValue(2, 0);
        timeGm_.SetValue(3, warmupCycles);
        DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(timeGm_);
    } else {
        uint64_t readOffset = sameAddr_ ? 0 : totalElem_;
        uint64_t warmupCycles = 0;
        if (enableWarmup_ != 0) {
            uint64_t warmupStart = GetSystemCycle();
            for (uint32_t w = 0; w < enableWarmup_; w++) {
                DataCopy(readLocal_, msGm_[readOffset], totalElem_);
                PipeBarrier<PIPE_ALL>();
            }
            uint64_t warmupEnd = GetSystemCycle();
            warmupCycles = warmupEnd - warmupStart;
        }

        CrossCoreWaitFlag<0x0, PIPE_ALL>(1);
        CrossCoreSetFlag<0x0, PIPE_ALL>(2);

        uint64_t startCycle = GetSystemCycle();
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopy(readLocal_, msGm_[readOffset], totalElem_);
            PipeBarrier<PIPE_ALL>();
        }
        uint64_t endCycle = GetSystemCycle();

        timeGm2_.SetValue(0, startCycle);
        timeGm2_.SetValue(1, endCycle);
        timeGm2_.SetValue(2, warmupCycles);
        timeGm2_.SetValue(3, 0);
        DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(timeGm2_);
    }
}

} // namespace LocalMsRwImpl

#endif // DUAL_CORE_RW_KERNEL_H
