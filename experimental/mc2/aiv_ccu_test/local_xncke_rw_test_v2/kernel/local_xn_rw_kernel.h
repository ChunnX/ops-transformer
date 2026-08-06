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
 * \file local_xn_rw_kernel.h
 * \brief 本地读写 CCU Variable(XN 64bit) Kernel 核心逻辑
 *        使用 WriteGmByPassDCache/ReadGmByPassDCache 绕过 DCache 直接读写寄存器。
 */

#ifndef LOCAL_XN_RW_KERNEL_H
#define LOCAL_XN_RW_KERNEL_H

#include "kernel_operator.h"

namespace LocalXnRwImpl {

using namespace AscendC;

enum RwMode : uint32_t {
    MODE_WRITE = 0,
    MODE_READ = 1,
    MODE_WRITE_THEN_READ = 2,
};

class LocalXnRw {
public:
    __aicore__ inline LocalXnRw(){};
    __aicore__ inline void Init(uint32_t mode, uint32_t enableWarmup, uint32_t loopCount,
                                uint64_t pattern, GM_ADDR xnAddr, GM_ADDR timeAddr)
    {
        mode_ = mode;
        warmupCount_ = enableWarmup;
        loopCount_ = loopCount == 0 ? 1 : loopCount;
        pattern_ = pattern;
        aivId_ = GetBlockIdx();
        if (aivId_ > 0) { return; }
        xnPtr_ = reinterpret_cast<__gm__ uint64_t *>(xnAddr);
        timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);
    }
    __aicore__ inline void Process()
    {
        if (aivId_ > 0) { return; }
        verifyResult_ = 0;
        uint64_t warmupCycles = 0;
        if (warmupCount_ != 0) {
            uint32_t savedLoop = loopCount_;
            loopCount_ = warmupCount_;
            uint64_t ws = GetSystemCycle();
            switch (mode_) {
                case MODE_WRITE:            WriteMode(); break;
                case MODE_READ:             ReadMode(); break;
                default:                    WriteThenReadMode(); break;
            }
            warmupCycles = GetSystemCycle() - ws;
            loopCount_ = savedLoop;
        }
        uint64_t startCycle = GetSystemCycle();
        switch (mode_) {
            case MODE_WRITE:            WriteMode(); break;
            case MODE_READ:             ReadMode(); break;
            default:                    WriteThenReadMode(); break;
        }
        uint64_t endCycle = GetSystemCycle();
        if (mode_ == MODE_WRITE_THEN_READ && readBack_ != pattern_) { verifyResult_ = 1; }
        timeGm_.SetValue(0, startCycle);
        timeGm_.SetValue(1, endCycle);
        timeGm_.SetValue(2, static_cast<uint64_t>(verifyResult_));
        timeGm_.SetValue(3, readBack_);
        timeGm_.SetValue(5, warmupCycles);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE,
            DcciDst::CACHELINE_OUT>(timeGm_);
    }
private:
    __aicore__ inline void WriteMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            WriteGmByPassDCache<uint64_t>(xnPtr_, pattern_);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void ReadMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            readBack_ = ReadGmByPassDCache<uint64_t>(xnPtr_);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void WriteThenReadMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            WriteGmByPassDCache<uint64_t>(xnPtr_, pattern_);
            PipeBarrier<PIPE_ALL>();
            readBack_ = ReadGmByPassDCache<uint64_t>(xnPtr_);
            PipeBarrier<PIPE_ALL>();
        }
    }
    uint32_t mode_{0};
    uint32_t warmupCount_{0};
    uint32_t loopCount_{1};
    uint32_t aivId_{0};
    uint32_t verifyResult_{0};
    uint64_t pattern_{0};
    uint64_t readBack_{0};
    __gm__ uint64_t *xnPtr_{nullptr};
    GlobalTensor<uint64_t> timeGm_;
};

} // namespace LocalXnRwImpl

#endif // LOCAL_XN_RW_KERNEL_H
