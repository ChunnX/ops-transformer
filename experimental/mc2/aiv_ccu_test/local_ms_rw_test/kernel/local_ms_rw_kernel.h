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
 * \file local_ms_rw_kernel.h
 * \brief 本地读写 MemorySlice (MS) Kernel 核心逻辑
 */

#ifndef LOCAL_MS_RW_KERNEL_H
#define LOCAL_MS_RW_KERNEL_H

#include "kernel_operator.h"

namespace LocalMsRwImpl {

using namespace AscendC;

constexpr uint32_t CCUA_NUM = 4U;           // 4 个 CCUA
constexpr uint32_t SYS_CYCLE_FREQ_MHZ = 1000U; // 1000MHz, 1 cycle = 1 ns
constexpr uint32_t DEFAULT_LOOP_COUNT = 1U;

enum RwMode : uint32_t {
    MODE_WRITE = 0,
    MODE_READ = 1,
    MODE_READ_WRITE = 2,
    MODE_WRITE_THEN_READ = 3,
    MODE_DUAL_CORE_RW = 4,
};

template <typename T>
class LocalMsRw {
public:
    __aicore__ inline LocalMsRw(){};
    __aicore__ inline void Init(uint32_t mode0, uint32_t enableWarmup, uint32_t msId,
                                uint32_t loopCount, uint64_t dataSize,
                                GM_ADDR msAddrs[4], GM_ADDR timeAddr, uint32_t blockDim, TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteMode();
    __aicore__ inline void WriteWarmup();
    __aicore__ inline void ReadWarmup();
    __aicore__ inline void ReadWriteWarmup();
    __aicore__ inline void WriteThenReadWarmup();
    __aicore__ inline void ReadMode();
    __aicore__ inline void ReadWriteMode();
    __aicore__ inline void WriteThenReadMode();
    __aicore__ inline void DualCoreWriteMode();
    __aicore__ inline void DualCoreReadMode();
    __aicore__ inline void FillData();
    __aicore__ inline void VerifyResult();
    __aicore__ inline void PrintSampleData();

    uint32_t mode0_;
    uint32_t enableWarmup_;
    uint32_t msId_;
    uint32_t loopCount_;
    uint64_t dataSize_;
    uint32_t blockDim_;
    uint32_t aivId_;

    GlobalTensor<T> msGm_[4];
    GlobalTensor<uint64_t> timeGm_;

    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readBuf_;
    LocalTensor<T> dataLocal_;
    LocalTensor<T> readLocal_;

    uint32_t totalElem_{0};
    uint32_t verifyResult_{0};
    bool sameAddr_{false};
};

template <typename T>
__aicore__ inline void LocalMsRw<T>::Init(uint32_t mode0, uint32_t enableWarmup, uint32_t msId,
                                          uint32_t loopCount, uint64_t dataSize,
                                          GM_ADDR msAddrs[4], GM_ADDR timeAddr, uint32_t blockDim, TPipe *pipe)
{
    mode0_ = mode0;
    enableWarmup_ = enableWarmup;
    msId_ = msId < CCUA_NUM ? msId : 0;
    loopCount_ = loopCount == 0 ? DEFAULT_LOOP_COUNT : loopCount;
    dataSize_ = dataSize;
    blockDim_ = blockDim;
    aivId_ = GetBlockIdx();

    for (uint32_t i = 0; i < CCUA_NUM; i++) {
        msGm_[i].SetGlobalBuffer((__gm__ T *)msAddrs[i]);
    }
    timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);

    if (mode0_ == MODE_DUAL_CORE_RW) {
        sameAddr_ = (msId_ == 0);
        constexpr uint32_t DUAL_MS_ID = 1;
        if (aivId_ == 0) {
            pipe->InitBuffer(dataBuf_, dataSize);
            dataLocal_ = dataBuf_.Get<T>();
        } else {
            pipe->InitBuffer(readBuf_, dataSize);
            readLocal_ = readBuf_.Get<T>();
        }
        totalElem_ = static_cast<uint32_t>(dataSize / sizeof(T));
    } else {
        if (aivId_ > 0) return;

        if (mode0_ == MODE_READ_WRITE || mode0_ == MODE_WRITE_THEN_READ) {
            pipe->InitBuffer(dataBuf_, dataSize);
            pipe->InitBuffer(readBuf_, dataSize);
        } else if (mode0_ == MODE_WRITE) {
            pipe->InitBuffer(dataBuf_, dataSize);
        } else if (mode0_ == MODE_READ) {
            pipe->InitBuffer(readBuf_, dataSize);
        }
        totalElem_ = static_cast<uint32_t>(dataSize / sizeof(T));
        dataLocal_ = dataBuf_.Get<T>();
        readLocal_ = readBuf_.Get<T>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::WriteWarmup()
{
    for (uint32_t i = 0; i < enableWarmup_; i++) {
        DataCopy(msGm_[msId_], dataLocal_, totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::WriteMode()
{
    for (uint32_t i = 0; i < loopCount_; i++) {
        DataCopy(msGm_[msId_], dataLocal_, totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::ReadWarmup()
{
    for (uint32_t i = 0; i < enableWarmup_; i++) {
        DataCopy(readLocal_, msGm_[msId_], totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::ReadMode()
{
    for (uint32_t i = 0; i < loopCount_; i++) {
        DataCopy(readLocal_, msGm_[msId_], totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::ReadWriteWarmup()
{
    for (uint32_t i = 0; i < enableWarmup_; i++) {
        DataCopy(readLocal_, msGm_[msId_], totalElem_);
        DataCopy(msGm_[msId_], dataLocal_, totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::ReadWriteMode()
{
    for (uint32_t i = 0; i < loopCount_; i++) {
        DataCopy(readLocal_, msGm_[msId_], totalElem_);
        DataCopy(msGm_[msId_], dataLocal_, totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::WriteThenReadWarmup()
{
    for (uint32_t i = 0; i < enableWarmup_; i++) {
        DataCopy(msGm_[msId_], dataLocal_, totalElem_);
        PipeBarrier<PIPE_ALL>();
        DataCopy(readLocal_, msGm_[msId_], totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::WriteThenReadMode()
{
    for (uint32_t i = 0; i < loopCount_; i++) {
        // // === 写阶段：一次写全部 ===
        DataCopy(msGm_[msId_], dataLocal_, totalElem_);  // UB → MS

        PipeBarrier<PIPE_ALL>();  // 确保写完

        // // === 读阶段：一次读全部 ===
        DataCopy(readLocal_, msGm_[msId_], totalElem_);  // MS → UB

        PipeBarrier<PIPE_ALL>();  // 确保读完
    }

/*
    uint64_t s0, e0, s1, e1;
    // // === 写阶段：一次写全部 ===
    s0 = GetSystemCycle();
    DataCopy(msGm_[1], dataLocal_, totalElem_);  // UB → MS

    PipeBarrier<PIPE_ALL>();  // 确保写完
    e0 = GetSystemCycle();

    // // === 读阶段：一次读全部 ===
    s1 = GetSystemCycle();
    DataCopy(readLocal_, msGm_[1], totalElem_);  // MS → UB

    PipeBarrier<PIPE_ALL>();  // 确保读完
    e1 = GetSystemCycle();

    printf("[lgx] startCycle0 = %llu\n", s0);
    printf("[lgx] endCycle0 = %llu\n", e0);
    printf("[lgx] startCycle1 = %llu\n", s1);
    printf("[lgx] endCycle1 = %llu\n", e1);
*/
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::DualCoreWriteMode()
{
    constexpr uint32_t DUAL_MS_ID = 1;
    for (uint32_t i = 0; i < loopCount_; i++) {
        DataCopy(msGm_[DUAL_MS_ID], dataLocal_, totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::DualCoreReadMode()
{
    constexpr uint32_t DUAL_MS_ID = 1;
    uint64_t readOffset = sameAddr_ ? 0 : totalElem_;
    for (uint32_t i = 0; i < loopCount_; i++) {
        DataCopy(readLocal_, msGm_[DUAL_MS_ID][readOffset], totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::FillData()
{
    for (uint32_t i = 0; i < totalElem_; i++) {
        dataLocal_.SetValue(i, static_cast<T>(i & 0xFF));
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::VerifyResult()
{
    verifyResult_ = 0;  // 0=通过, 1=失败

    for (uint32_t i = 0; i < totalElem_; i++) {
        T actual = readLocal_.GetValue(i);
        if (actual != static_cast<T>(i & 0xFF)) {
            printf("[lgx] current actual = %u, expected = %u\n", actual, static_cast<T>(i & 0xFF));
            verifyResult_ = 1;
            return;
        }
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::PrintSampleData()
{
    uint32_t printCount = totalElem_ < 8 ? totalElem_ : 8;
    printf("[INFO] MS数据预览:");
    for (uint32_t i = 0; i < printCount; i++) {
        printf(" [%u]=0x%02x", i, (uint32_t)readLocal_.GetValue(i));
    }
    printf("\n");
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::Process()
{
    if (mode0_ == MODE_DUAL_CORE_RW) {
        if (aivId_ == 0) {
            Duplicate<T>(dataLocal_, (T)2, totalElem_);
            PipeBarrier<PIPE_ALL>();

            uint64_t warmupCycles = 0;
            if (enableWarmup_ != 0) {
                uint64_t warmupStart = GetSystemCycle();
                for (uint32_t w = 0; w < enableWarmup_; w++) {
                    DualCoreWriteMode();
                }
                uint64_t warmupEnd = GetSystemCycle();
                warmupCycles = warmupEnd - warmupStart;
            }

            CrossCoreSetFlag<0x0, PIPE_ALL>(1);
            CrossCoreWaitFlag<0x0>(2);

            uint64_t startCycle = GetSystemCycle();
            DualCoreWriteMode();
            uint64_t endCycle = GetSystemCycle();

            CrossCoreWaitFlag<0x0>(3);

            timeGm_.SetValue(0, startCycle);
            timeGm_.SetValue(1, endCycle);
            timeGm_.SetValue(2, 0);
            timeGm_.SetValue(3, warmupCycles);
            DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(timeGm_);
        } else {
            uint64_t warmupCycles = 0;
            if (enableWarmup_ != 0) {
                uint64_t warmupStart = GetSystemCycle();
                for (uint32_t w = 0; w < enableWarmup_; w++) {
                    DualCoreReadMode();
                }
                uint64_t warmupEnd = GetSystemCycle();
                warmupCycles = warmupEnd - warmupStart;
            }

            CrossCoreWaitFlag<0x0>(1);
            CrossCoreSetFlag<0x0, PIPE_ALL>(2);

            uint64_t startCycle = GetSystemCycle();
            DualCoreReadMode();
            uint64_t endCycle = GetSystemCycle();

            CrossCoreSetFlag<0x0, PIPE_ALL>(3);

            timeGm_.SetValue(4, startCycle);
            timeGm_.SetValue(5, endCycle);
            timeGm_.SetValue(6, warmupCycles);
            timeGm_.SetValue(7, 0);
            DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(timeGm_);
        }
        return;
    }

    if (aivId_ > 0) return;

    // 打点外：填充数据
    if (mode0_ == MODE_WRITE_THEN_READ) {
        FillData();
    } else if (mode0_ == MODE_WRITE || mode0_ == MODE_READ_WRITE) {
        Duplicate<T>(dataLocal_, (T)2, totalElem_);
    }

    PipeBarrier<PIPE_ALL>();
    uint64_t warmupCycles = 0;
    if (enableWarmup_ != 0) {
        uint64_t warmupStart = GetSystemCycle();
        switch (mode0_) {
            case MODE_WRITE:
                WriteWarmup();
                break;
            case MODE_READ:
                ReadWarmup();
                break;
            case MODE_READ_WRITE:
                ReadWriteWarmup();
                break;
            case MODE_WRITE_THEN_READ:
                WriteThenReadWarmup();
                break;
            default:
                break;
        }
        uint64_t warmupEnd = GetSystemCycle();
        warmupCycles = warmupEnd - warmupStart;
        printf("[lgx] warmup %u times, total cycles = %llu\n", enableWarmup_, warmupCycles);
    }

    // 打点内
    uint64_t startCycle = GetSystemCycle();

    switch (mode0_) {
        case MODE_WRITE:
            WriteMode();
            break;
        case MODE_READ:
            ReadMode();
            break;
        case MODE_READ_WRITE:
            ReadWriteMode();
            break;
        case MODE_WRITE_THEN_READ:
            WriteThenReadMode();
            break;
        default:
            break;
    }

    uint64_t endCycle = GetSystemCycle();
    printf("[lgx] startCycle = %llu\n", startCycle);
    printf("[lgx] endCycle = %llu\n", endCycle);

    // 打点外：校验（仅新模式）
    if (mode0_ == MODE_WRITE_THEN_READ) {
        VerifyResult();
    } else if (mode0_ == MODE_READ) {
        PrintSampleData();
    }

    // 写结果
    timeGm_.SetValue(0, startCycle);
    timeGm_.SetValue(1, endCycle);
    timeGm_.SetValue(2, verifyResult_);
    timeGm_.SetValue(3, warmupCycles);
    DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(timeGm_);
}

} // namespace LocalMsRwImpl

#endif // LOCAL_MS_RW_KERNEL_H
