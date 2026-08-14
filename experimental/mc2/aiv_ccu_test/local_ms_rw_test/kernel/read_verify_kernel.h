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
 * \file read_verify_kernel.h
 * \brief 纯读校验 Kernel
 *        打点外：先把 1.5MB MS 填满递增 pattern
 *        打点内：只读，测量真实读带宽
 *        打点后：逐元素校验读出数据
 */

#ifndef READ_VERIFY_KERNEL_H
#define READ_VERIFY_KERNEL_H

#include "kernel_operator.h"

namespace LocalMsRwImpl {

using namespace AscendC;

constexpr uint64_t MS_FULL_SIZE = 1572864ULL;  // 1.5MB
constexpr uint32_t READ_VERIFY_CHUNK = 8 * 1024U;  // 8KB，填充用小块，留空间给 readBuf

template <typename T>
class ReadVerify {
public:
    __aicore__ inline ReadVerify(){};
    __aicore__ inline void Init(uint32_t enableWarmup, uint32_t msId,
                                uint32_t loopCount, uint64_t dataSize,
                                GM_ADDR msAddrs[4], GM_ADDR timeAddr,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FillMsFull();
    __aicore__ inline void ReadMode();
    __aicore__ inline void VerifyReadData();

    uint32_t enableWarmup_{0};
    uint32_t msId_{0};
    uint32_t loopCount_{1};
    uint64_t dataSize_{0};
    uint32_t aivId_{0};

    GlobalTensor<T> msGm_[4];
    GlobalTensor<uint64_t> timeGm_;

    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readBuf_;
    LocalTensor<T> dataLocal_;
    LocalTensor<T> readLocal_;

    uint32_t totalElem_{0};
    uint32_t verifyResult_{0};
};

template <typename T>
__aicore__ inline void ReadVerify<T>::Init(uint32_t enableWarmup, uint32_t msId,
                                           uint32_t loopCount, uint64_t dataSize,
                                           GM_ADDR msAddrs[4], GM_ADDR timeAddr,
                                           TPipe *pipe)
{
    enableWarmup_ = enableWarmup;
    msId_ = msId < CCUA_NUM ? msId : 0;
    loopCount_ = loopCount == 0 ? 1 : loopCount;
    dataSize_ = dataSize;
    aivId_ = GetBlockIdx();

    for (uint32_t i = 0; i < CCUA_NUM; i++) {
        msGm_[i].SetGlobalBuffer((__gm__ T *)msAddrs[i]);
        msGm_[i].SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);

    if (aivId_ > 0) return;

    pipe->InitBuffer(dataBuf_, READ_VERIFY_CHUNK);
    pipe->InitBuffer(readBuf_, dataSize);    dataLocal_ = dataBuf_.Get<T>();
    readLocal_ = readBuf_.Get<T>();
    totalElem_ = static_cast<uint32_t>(dataSize / sizeof(T));
}

template <typename T>
__aicore__ inline void ReadVerify<T>::FillMsFull()
{
    uint32_t chunkElem = READ_VERIFY_CHUNK / sizeof(T);
    uint32_t fullElem = static_cast<uint32_t>(MS_FULL_SIZE / sizeof(T));

    for (uint32_t offset = 0; offset < fullElem; offset += chunkElem) {
        uint32_t copyElem = (fullElem - offset) < chunkElem ? (fullElem - offset) : chunkElem;
        for (uint32_t i = 0; i < copyElem; i++) {
            dataLocal_.SetValue(i, static_cast<T>((offset + i) & 0xFF));
        }
        DataCopy(msGm_[msId_][offset], dataLocal_, copyElem);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void ReadVerify<T>::ReadMode()
{
    for (uint32_t i = 0; i < loopCount_; i++) {
        DataCopy(readLocal_, msGm_[msId_], totalElem_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void ReadVerify<T>::VerifyReadData()
{
    verifyResult_ = 0;

    DataCopy(readLocal_, msGm_[msId_], totalElem_);
    PipeBarrier<PIPE_ALL>();

    for (uint32_t i = 0; i < totalElem_; i++) {
        T actual = readLocal_.GetValue(i);
        if (actual != static_cast<T>(i & 0xFF)) {
            printf("[VERIFY] mismatch at [%u]: actual=0x%02x, expected=0x%02x\n",
                   i, (uint32_t)actual, (uint32_t)(i & 0xFF));
            verifyResult_ = 1;
            return;
        }
    }
}

template <typename T>
__aicore__ inline void ReadVerify<T>::Process()
{
    if (aivId_ > 0) return;

    // 打点外：填充 1.5MB MS
    FillMsFull();
    PipeBarrier<PIPE_ALL>();

    // 预热（只读）
    uint64_t warmupCycles = 0;
    if (enableWarmup_ != 0) {
        uint64_t warmupStart = GetSystemCycle();
        for (uint32_t w = 0; w < enableWarmup_; w++) {
            DataCopy(readLocal_, msGm_[msId_], totalElem_);
            PipeBarrier<PIPE_ALL>();
        }
        uint64_t warmupEnd = GetSystemCycle();
        warmupCycles = warmupEnd - warmupStart;
    }

    // 打点内：只读
    uint64_t startCycle = GetSystemCycle();
    ReadMode();
    uint64_t endCycle = GetSystemCycle();

    // 打点外：校验
    VerifyReadData();

    timeGm_.SetValue(0, startCycle);
    timeGm_.SetValue(1, endCycle);
    timeGm_.SetValue(2, verifyResult_);
    timeGm_.SetValue(3, warmupCycles);
    DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(timeGm_);
}

} // namespace LocalMsRwImpl

#endif // READ_VERIFY_KERNEL_H
