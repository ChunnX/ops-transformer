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
 * \file local_xncke_rw_kernel.h
 * \brief 本地读写 CCU XN/CKE 8B 资源的 Kernel 核心逻辑
 */

#ifndef LOCAL_XNCKE_RW_KERNEL_H
#define LOCAL_XNCKE_RW_KERNEL_H

#include "kernel_operator.h"

namespace LocalXnCkeRwImpl {

using namespace AscendC;

constexpr uint32_t RESULT_WORDS = 6U;

enum RwMode : uint32_t {
    MODE_WRITE = 0,
    MODE_READ = 1,
    MODE_WRITE_THEN_READ = 2,
    MODE_SET_VALUE = 3,
};

enum ResKind : uint32_t {
    RES_KIND_XN = 0,
    RES_KIND_CKE = 1,
    RES_KIND_BOTH = 2,
};

template <typename T>
class LocalXnCkeRw {
public:
    __aicore__ inline LocalXnCkeRw(){};
    __aicore__ inline void Init(uint32_t mode, uint32_t resKind, uint32_t enableWarmup, uint32_t loopCount,
                                T pattern, GM_ADDR xnAddr, GM_ADDR ckeAddr, GM_ADDR timeAddr, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline bool NeedAccessXn() const;
    __aicore__ inline bool NeedAccessCke() const;
    __aicore__ inline void PrepareWriteData();
    __aicore__ inline T DataCopyWrite(GlobalTensor<T> &dst, T value, uint32_t loops);
    __aicore__ inline void DataCopyRead(LocalTensor<T> &readLocal, GlobalTensor<T> &src, uint32_t loops);
    __aicore__ inline void DataCopyWriteThenRead(LocalTensor<T> &readLocal, GlobalTensor<T> &dst, uint32_t loops);
    __aicore__ inline void SetValueConst(GlobalTensor<T> &dst, T value, uint32_t loops);
    __aicore__ inline void RunSelectedOp(uint32_t loops);
    __aicore__ inline void CaptureReadBack();
    __aicore__ inline void CheckResult();

    uint32_t mode_{0};
    uint32_t resKind_{0};
    uint32_t enableWarmup_{0};
    uint32_t loopCount_{1};
    uint32_t aivId_{0};
    T pattern_{0};
    T xnReadBack_{0};
    T ckeReadBack_{0};
    uint32_t verifyResult_{0};

    GlobalTensor<T> xnGm_;
    GlobalTensor<T> ckeGm_;
    GlobalTensor<uint64_t> timeGm_;

    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readXnBuf_;
    TBuf<TPosition::VECCALC> readCkeBuf_;
    LocalTensor<T> dataLocal_;
    LocalTensor<T> readXnLocal_;
    LocalTensor<T> readCkeLocal_;
};

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::Init(uint32_t mode, uint32_t resKind, uint32_t enableWarmup,
                                             uint32_t loopCount, T pattern, GM_ADDR xnAddr, GM_ADDR ckeAddr,
                                             GM_ADDR timeAddr, TPipe *pipe)
{
    mode_ = mode;
    resKind_ = resKind;
    enableWarmup_ = enableWarmup;
    loopCount_ = loopCount == 0 ? 1 : loopCount;
    pattern_ = pattern;
    aivId_ = GetBlockIdx();
    if (aivId_ > 0) {
        return;
    }

    xnGm_.SetGlobalBuffer((__gm__ T *)xnAddr);
    ckeGm_.SetGlobalBuffer((__gm__ T *)ckeAddr);
    timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);
    pipe->InitBuffer(dataBuf_, sizeof(T));
    pipe->InitBuffer(readXnBuf_, sizeof(T));
    pipe->InitBuffer(readCkeBuf_, sizeof(T));
    dataLocal_ = dataBuf_.Get<T>();
    readXnLocal_ = readXnBuf_.Get<T>();
    readCkeLocal_ = readCkeBuf_.Get<T>();
}

template <typename T>
__aicore__ inline bool LocalXnCkeRw<T>::NeedAccessXn() const
{
    return resKind_ == RES_KIND_XN || resKind_ == RES_KIND_BOTH;
}

template <typename T>
__aicore__ inline bool LocalXnCkeRw<T>::NeedAccessCke() const
{
    return resKind_ == RES_KIND_CKE || resKind_ == RES_KIND_BOTH;
}

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::PrepareWriteData()
{
    dataLocal_.SetValue(0, pattern_);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline T LocalXnCkeRw<T>::DataCopyWrite(GlobalTensor<T> &dst, T value, uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        DataCopy(dst, dataLocal_, 1);
        PipeBarrier<PIPE_ALL>();
    }
    return value;
}

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::DataCopyRead(LocalTensor<T> &readLocal, GlobalTensor<T> &src,
                                                     uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        DataCopy(readLocal, src, 1);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::DataCopyWriteThenRead(LocalTensor<T> &readLocal, GlobalTensor<T> &dst,
                                                              uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        DataCopy(dst, dataLocal_, 1);
        PipeBarrier<PIPE_ALL>();
        DataCopy(readLocal, dst, 1);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::SetValueConst(GlobalTensor<T> &dst, T value, uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        dst.SetValue(0, value);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::RunSelectedOp(uint32_t loops)
{
    switch (mode_) {
        case MODE_WRITE:
            if (NeedAccessXn()) {
                xnReadBack_ = DataCopyWrite(xnGm_, pattern_, loops);
            }
            if (NeedAccessCke()) {
                ckeReadBack_ = DataCopyWrite(ckeGm_, pattern_, loops);
            }
            break;
        case MODE_READ:
            if (NeedAccessXn()) {
                DataCopyRead(readXnLocal_, xnGm_, loops);
            }
            if (NeedAccessCke()) {
                DataCopyRead(readCkeLocal_, ckeGm_, loops);
            }
            break;
        case MODE_WRITE_THEN_READ:
            if (NeedAccessXn()) {
                DataCopyWriteThenRead(readXnLocal_, xnGm_, loops);
            }
            if (NeedAccessCke()) {
                DataCopyWriteThenRead(readCkeLocal_, ckeGm_, loops);
            }
            break;
        case MODE_SET_VALUE:
            if (NeedAccessXn()) {
                SetValueConst(xnGm_, pattern_, loops);
            }
            if (NeedAccessCke()) {
                SetValueConst(ckeGm_, pattern_, loops);
            }
            break;
        default:
            verifyResult_ = 1;
            break;
    }
}

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::CaptureReadBack()
{
    if (mode_ == MODE_READ || mode_ == MODE_WRITE_THEN_READ) {
        if (NeedAccessXn()) {
            xnReadBack_ = readXnLocal_.GetValue(0);
        }
        if (NeedAccessCke()) {
            ckeReadBack_ = readCkeLocal_.GetValue(0);
        }
    }
}

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::CheckResult()
{
    if (mode_ == MODE_WRITE_THEN_READ) {
        T expected = pattern_;
        if (NeedAccessXn() && xnReadBack_ != expected) {
            verifyResult_ = 1;
        }
        if (NeedAccessCke() && ckeReadBack_ != expected) {
            verifyResult_ = 1;
        }
    }
}

template <typename T>
__aicore__ inline void LocalXnCkeRw<T>::Process()
{
    if (aivId_ > 0) {
        return;
    }

    verifyResult_ = 0;
    uint64_t warmupCycles = 0;
    PrepareWriteData();

    if (enableWarmup_ != 0) {
        uint64_t warmupStart = GetSystemCycle();
        RunSelectedOp(1);
        uint64_t warmupEnd = GetSystemCycle();
        warmupCycles = warmupEnd - warmupStart;
    }

    uint64_t startCycle = GetSystemCycle();
    RunSelectedOp(loopCount_);
    uint64_t endCycle = GetSystemCycle();
    CaptureReadBack();
    CheckResult();

    timeGm_.SetValue(0, startCycle);
    timeGm_.SetValue(1, endCycle);
    timeGm_.SetValue(2, static_cast<uint64_t>(verifyResult_));
    timeGm_.SetValue(3, static_cast<uint64_t>(xnReadBack_));
    timeGm_.SetValue(4, static_cast<uint64_t>(ckeReadBack_));
    timeGm_.SetValue(5, warmupCycles);
    DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        timeGm_);
}

} // namespace LocalXnCkeRwImpl

#endif // LOCAL_XNCKE_RW_KERNEL_H
