/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file local_xncke_rw_kernel.h
 * \brief 本地读写 CCU Variable(XN 64bit)/Event(CKE 16bit) 资源的 Kernel 核心逻辑
 *        XN 通过 DataCopy 做 8B 读写；CKE 因 16bit 宽度使用 SetValue/GetValue 做 2B 读写。
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

class LocalXnCkeRw {
public:
    __aicore__ inline LocalXnCkeRw(){};
    __aicore__ inline void Init(uint32_t mode, uint32_t resKind, uint32_t enableWarmup, uint32_t loopCount,
                                uint64_t pattern, GM_ADDR xnAddr, GM_ADDR ckeAddr, GM_ADDR timeAddr, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline bool NeedAccessXn() const;
    __aicore__ inline bool NeedAccessCke() const;
    __aicore__ inline void PrepareWriteData();
    __aicore__ inline void DataCopyWriteXn(uint32_t loops);
    __aicore__ inline void DataCopyReadXn(uint32_t loops);
    __aicore__ inline void DataCopyWriteThenReadXn(uint32_t loops);
    __aicore__ inline void SetValueXn(uint32_t loops);
    __aicore__ inline void SetValueCke(uint32_t loops);
    __aicore__ inline void GetValueCke(uint32_t loops);
    __aicore__ inline void SetValueThenGetValueCke(uint32_t loops);
    __aicore__ inline void RunSelectedOp(uint32_t loops);
    __aicore__ inline void CaptureReadBack();
    __aicore__ inline void CheckResult();

    uint32_t mode_{0};
    uint32_t resKind_{0};
    uint32_t enableWarmup_{0};
    uint32_t loopCount_{1};
    uint32_t aivId_{0};
    uint64_t pattern_{0};
    uint16_t ckePattern_{0};
    uint64_t xnReadBack_{0};
    uint16_t ckeReadBack_{0};
    uint32_t verifyResult_{0};

    GlobalTensor<uint64_t> xnGm_;
    GlobalTensor<uint16_t> ckeGm_;
    GlobalTensor<uint64_t> timeGm_;

    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readXnBuf_;
    LocalTensor<uint64_t> dataLocal_;
    LocalTensor<uint64_t> readXnLocal_;
};

__aicore__ inline void LocalXnCkeRw::Init(uint32_t mode, uint32_t resKind, uint32_t enableWarmup,
                                          uint32_t loopCount, uint64_t pattern, GM_ADDR xnAddr, GM_ADDR ckeAddr,
                                          GM_ADDR timeAddr, TPipe *pipe)
{
    mode_ = mode;
    resKind_ = resKind;
    enableWarmup_ = enableWarmup;
    loopCount_ = loopCount == 0 ? 1 : loopCount;
    pattern_ = pattern;
    ckePattern_ = static_cast<uint16_t>(pattern & 0xFFFFU);
    aivId_ = GetBlockIdx();
    if (aivId_ > 0) {
        return;
    }

    xnGm_.SetGlobalBuffer((__gm__ uint64_t *)xnAddr);
    ckeGm_.SetGlobalBuffer((__gm__ uint16_t *)ckeAddr);
    timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);
    pipe->InitBuffer(dataBuf_, sizeof(uint64_t));
    pipe->InitBuffer(readXnBuf_, sizeof(uint64_t));
    dataLocal_ = dataBuf_.Get<uint64_t>();
    readXnLocal_ = readXnBuf_.Get<uint64_t>();
}

__aicore__ inline bool LocalXnCkeRw::NeedAccessXn() const
{
    return resKind_ == RES_KIND_XN || resKind_ == RES_KIND_BOTH;
}

__aicore__ inline bool LocalXnCkeRw::NeedAccessCke() const
{
    return resKind_ == RES_KIND_CKE || resKind_ == RES_KIND_BOTH;
}

__aicore__ inline void LocalXnCkeRw::PrepareWriteData()
{
    dataLocal_.SetValue(0, pattern_);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void LocalXnCkeRw::DataCopyWriteXn(uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        DataCopy(xnGm_, dataLocal_, 1);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void LocalXnCkeRw::DataCopyReadXn(uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        DataCopy(readXnLocal_, xnGm_, 1);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void LocalXnCkeRw::DataCopyWriteThenReadXn(uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        DataCopy(xnGm_, dataLocal_, 1);
        PipeBarrier<PIPE_ALL>();
        DataCopy(readXnLocal_, xnGm_, 1);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void LocalXnCkeRw::SetValueXn(uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        xnGm_.SetValue(0, pattern_);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void LocalXnCkeRw::SetValueCke(uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        ckeGm_.SetValue(0, ckePattern_);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void LocalXnCkeRw::GetValueCke(uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        ckeReadBack_ = ckeGm_.GetValue(0);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void LocalXnCkeRw::SetValueThenGetValueCke(uint32_t loops)
{
    for (uint32_t i = 0; i < loops; i++) {
        ckeGm_.SetValue(0, ckePattern_);
        PipeBarrier<PIPE_ALL>();
        ckeReadBack_ = ckeGm_.GetValue(0);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void LocalXnCkeRw::RunSelectedOp(uint32_t loops)
{
    switch (mode_) {
        case MODE_WRITE:
            if (NeedAccessXn()) {
                DataCopyWriteXn(loops);
            }
            if (NeedAccessCke()) {
                SetValueCke(loops);
            }
            break;
        case MODE_READ:
            if (NeedAccessXn()) {
                DataCopyReadXn(loops);
            }
            if (NeedAccessCke()) {
                GetValueCke(loops);
            }
            break;
        case MODE_WRITE_THEN_READ:
            if (NeedAccessXn()) {
                DataCopyWriteThenReadXn(loops);
            }
            if (NeedAccessCke()) {
                SetValueThenGetValueCke(loops);
            }
            break;
        case MODE_SET_VALUE:
            if (NeedAccessXn()) {
                SetValueXn(loops);
            }
            if (NeedAccessCke()) {
                SetValueCke(loops);
            }
            break;
        default:
            verifyResult_ = 1;
            break;
    }
}

__aicore__ inline void LocalXnCkeRw::CaptureReadBack()
{
    if (mode_ == MODE_READ || mode_ == MODE_WRITE_THEN_READ) {
        if (NeedAccessXn()) {
            xnReadBack_ = readXnLocal_.GetValue(0);
        }
        if (NeedAccessCke()) {
            ckeReadBack_ = ckeGm_.GetValue(0);
        }
    }
}

__aicore__ inline void LocalXnCkeRw::CheckResult()
{
    if (mode_ == MODE_WRITE_THEN_READ) {
        if (NeedAccessXn() && xnReadBack_ != pattern_) {
            verifyResult_ = 1;
        }
        if (NeedAccessCke() && ckeReadBack_ != ckePattern_) {
            verifyResult_ = 1;
        }
    }
}

__aicore__ inline void LocalXnCkeRw::Process()
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
    timeGm_.SetValue(3, xnReadBack_);
    timeGm_.SetValue(4, static_cast<uint64_t>(ckeReadBack_));
    timeGm_.SetValue(5, warmupCycles);
    DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        timeGm_);
}

} // namespace LocalXnCkeRwImpl

#endif // LOCAL_XNCKE_RW_KERNEL_H
