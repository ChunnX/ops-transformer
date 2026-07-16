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

constexpr uint32_t BLOCK_4K_BYTES = 4096U;
constexpr uint32_t GROUP_SIZE = 8U;         // 交织时每 8 个 4KB 块为一组
constexpr uint32_t CCUA_NUM = 4U;           // 4 个 CCUA
constexpr uint32_t BLOCKS_PER_CCUA = 384U;  // 1.5MB / 4KB = 384
constexpr uint32_t SYS_CYCLE_FREQ_MHZ = 1000U; // 1000MHz, 1 cycle = 1 ns

enum RwMode : uint32_t {
    MODE_WRITE = 0,
    MODE_READ = 1,
    MODE_READ_WRITE = 2,
};

enum InterleaveMode : uint32_t {
    MODE_SEQUENTIAL = 0,
    MODE_INTERLEAVED = 1,
};

template <typename T>
class LocalMsRw {
public:
    __aicore__ inline LocalMsRw(){};
    __aicore__ inline void Init(uint32_t mode0, uint32_t mode1, uint64_t dataSize,
                                GM_ADDR msAddrs[4], GM_ADDR timeAddr, uint32_t blockDim, TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteMode();
    __aicore__ inline void ReadMode();
    __aicore__ inline void ReadWriteMode();
    __aicore__ inline void GetBlockTargetAddr(uint64_t blockIdx, uint32_t &ccuaIdx, uint64_t &offsetBytes);

    uint32_t mode0_;
    uint32_t mode1_;
    uint64_t dataSize_;
    uint32_t blockDim_;
    uint32_t aivId_;

    GlobalTensor<T> msGm_[4];
    GlobalTensor<uint64_t> timeGm_;

    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readBuf_;

    uint32_t totalBlocks_;
    uint32_t blocksPerCore_;
    uint32_t myStartBlock_;
    uint32_t myBlocks_;
};

template <typename T>
__aicore__ inline void LocalMsRw<T>::Init(uint32_t mode0, uint32_t mode1, uint64_t dataSize,
                                          GM_ADDR msAddrs[4], GM_ADDR timeAddr, uint32_t blockDim, TPipe *pipe)
{
    mode0_ = mode0;
    mode1_ = mode1;
    dataSize_ = dataSize;
    blockDim_ = blockDim;
    aivId_ = GetBlockIdx();

    for (uint32_t i = 0; i < CCUA_NUM; i++) {
        msGm_[i].SetGlobalBuffer((__gm__ T *)msAddrs[i]);
    }
    timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);

    pipe->InitBuffer(dataBuf_, BLOCK_4K_BYTES);
    pipe->InitBuffer(readBuf_, BLOCK_4K_BYTES);

    totalBlocks_ = static_cast<uint32_t>(dataSize / BLOCK_4K_BYTES);
    blocksPerCore_ = totalBlocks_ / blockDim_;
    uint32_t tailBlocks = totalBlocks_ % blockDim_;
    // 顺序分核，序号小的核多搬一轮
    if (aivId_ < tailBlocks) {
        myStartBlock_ = aivId_ * (blocksPerCore_ + 1);
        myBlocks_ = blocksPerCore_ + 1;
    } else {
        myStartBlock_ = aivId_ * blocksPerCore_ + tailBlocks;
        myBlocks_ = blocksPerCore_;
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::GetBlockTargetAddr(uint64_t blockIdx, uint32_t &ccuaIdx,
                                                         uint64_t &offsetBytes)
{
    if (mode1_ == MODE_INTERLEAVED) {
        // 交织：每 8 个 4KB 块为一组，组间轮转 CCUA0→1→2→3
        uint32_t group = static_cast<uint32_t>(blockIdx / GROUP_SIZE);
        ccuaIdx = group % CCUA_NUM;
        uint32_t localGroup = group / CCUA_NUM;
        uint32_t posInGroup = static_cast<uint32_t>(blockIdx % GROUP_SIZE);
        offsetBytes = (localGroup * GROUP_SIZE + posInGroup) * BLOCK_4K_BYTES;
    } else {
        // 非交织：连续写满 CCUA0(1.5MB) 再写 CCUA1，以此类推
        ccuaIdx = static_cast<uint32_t>(blockIdx / BLOCKS_PER_CCUA);
        if (ccuaIdx >= CCUA_NUM) {
            ccuaIdx = CCUA_NUM - 1;
        }
        offsetBytes = (blockIdx % BLOCKS_PER_CCUA) * BLOCK_4K_BYTES;
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::WriteMode()
{
    LocalTensor<T> dataLocal = dataBuf_.Get<T>();
    uint32_t elemPerBlock = BLOCK_4K_BYTES / sizeof(T);
    Duplicate<T>(dataLocal, (T)1, elemPerBlock);

    for (uint32_t i = 0; i < myBlocks_; i++) {
        uint64_t blockIdx = myStartBlock_ + i;
        uint32_t ccuaIdx;
        uint64_t offsetBytes;
        GetBlockTargetAddr(blockIdx, ccuaIdx, offsetBytes);
        uint64_t offsetElem = offsetBytes / sizeof(T);
        DataCopy(msGm_[ccuaIdx][offsetElem], dataLocal, elemPerBlock);
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::ReadMode()
{
    LocalTensor<T> readLocal = readBuf_.Get<T>();
    uint32_t elemPerBlock = BLOCK_4K_BYTES / sizeof(T);

    for (uint32_t i = 0; i < myBlocks_; i++) {
        uint64_t blockIdx = myStartBlock_ + i;
        uint32_t ccuaIdx;
        uint64_t offsetBytes;
        GetBlockTargetAddr(blockIdx, ccuaIdx, offsetBytes);
        uint64_t offsetElem = offsetBytes / sizeof(T);
        DataCopy(readLocal, msGm_[ccuaIdx][offsetElem], elemPerBlock);
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::ReadWriteMode()
{
    LocalTensor<T> dataLocal = dataBuf_.Get<T>();
    LocalTensor<T> readLocal = readBuf_.Get<T>();
    uint32_t elemPerBlock = BLOCK_4K_BYTES / sizeof(T);

    for (uint32_t i = 0; i < myBlocks_; i++) {
        uint64_t blockIdx = myStartBlock_ + i;
        uint32_t ccuaIdx;
        uint64_t offsetBytes;
        GetBlockTargetAddr(blockIdx, ccuaIdx, offsetBytes);
        uint64_t offsetElem = offsetBytes / sizeof(T);
        // 先读 MS → UB
        DataCopy(readLocal, msGm_[ccuaIdx][offsetElem], elemPerBlock);
        // 再写 UB → MS（写回不同数据以区分）
        Duplicate<T>(dataLocal, (T)2, elemPerBlock);
        DataCopy(msGm_[ccuaIdx][offsetElem], dataLocal, elemPerBlock);
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::Process()
{
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
        default:
            break;
    }

    PipeBarrier<PIPE_ALL>();
    uint64_t endCycle = GetSystemCycle();

    // core 0 将计时结果写入 timeGm
    if (aivId_ == 0) {
        timeGm_.SetValue(0, startCycle);
        timeGm_.SetValue(1, endCycle);
    }
}

} // namespace LocalMsRwImpl

#endif // LOCAL_MS_RW_KERNEL_H
