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
    MODE_WRITE_THEN_READ = 3,
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
    __aicore__ inline void WriteThenReadMode();
    __aicore__ inline void FillData();
    __aicore__ inline void VerifyResult();
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
    LocalTensor<T> dataLocal_;
    LocalTensor<T> readLocal_;

    uint32_t totalBlocks_;
    uint32_t blocksPerCore_;
    uint32_t myStartBlock_;
    uint32_t myBlocks_;
    uint32_t totalElem_{0};
    uint32_t verifyResult_{0};
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
    if (aivId_ > 0) return;

    for (uint32_t i = 0; i < CCUA_NUM; i++) {
        msGm_[i].SetGlobalBuffer((__gm__ T *)msAddrs[i]);
    }
    timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);

    if (mode0_ == MODE_WRITE_THEN_READ) {
        // WriteThenRead 模式：按 dataSize 分配 buffer，一次搬运
        pipe->InitBuffer(dataBuf_, dataSize);
        pipe->InitBuffer(readBuf_, dataSize);
        totalElem_ = static_cast<uint32_t>(dataSize / sizeof(T));
        dataLocal_ = dataBuf_.Get<T>();
        readLocal_ = readBuf_.Get<T>();
    } else {
        // 其他模式：固定 4KB 分块搬运
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
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::GetBlockTargetAddr(uint64_t blockIdx, uint32_t &ccuaIdx,
                                                         uint64_t &offsetBytes)
{
    if (mode1_ == MODE_INTERLEAVED) {
        // 交织：每 8 个 4KB 块为一组，组间轮转 CCUA0→1→2→3
        uint32_t group = static_cast<uint32_t>(blockIdx / GROUP_SIZE); // group = 41 / 8 = 5
        ccuaIdx = group % CCUA_NUM; // ccuaidx = 5 % 4 = 1
        uint32_t localGroup = group / CCUA_NUM; // localGroup = 5 / 4 = 1
        uint32_t posInGroup = static_cast<uint32_t>(blockIdx % GROUP_SIZE); // posInGroup = 41 % 8 = 1
        offsetBytes = (localGroup * GROUP_SIZE + posInGroup) * BLOCK_4K_BYTES; //( 1 * 8 + 1 ) * 4KB = 36KB
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
    uint32_t elemPerBlock = BLOCK_4K_BYTES / sizeof(T);
    for (uint32_t i = 0; i < myBlocks_; i++) {
        uint64_t blockIdx = myStartBlock_ + i;
        uint32_t ccuaIdx;
        uint64_t offsetBytes;
        GetBlockTargetAddr(blockIdx, ccuaIdx, offsetBytes);
        uint64_t offsetElem = offsetBytes / sizeof(T);
        DataCopy(msGm_[ccuaIdx][offsetElem], dataLocal_, elemPerBlock);
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
    uint32_t elemPerBlock = BLOCK_4K_BYTES / sizeof(T);

    for (uint32_t i = 0; i < myBlocks_; i++) {
        uint64_t blockIdx = myStartBlock_ + i;
        uint32_t ccuaIdx;
        uint64_t offsetBytes;
        GetBlockTargetAddr(blockIdx, ccuaIdx, offsetBytes);
        uint64_t offsetElem = offsetBytes / sizeof(T);
        DataCopy(readLocal_, msGm_[ccuaIdx][offsetElem], elemPerBlock);
        Duplicate<T>(dataLocal_, (T)2, elemPerBlock);
        DataCopy(msGm_[ccuaIdx][offsetElem], dataLocal_, elemPerBlock);
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::WriteThenReadMode()
{
    // === 写阶段：一次写全部 ===
    DataCopy(msGm_[0], dataLocal_, totalElem_);  // UB → MS

    PipeBarrier<PIPE_ALL>();  // 确保写完

    // === 读阶段：一次读全部 ===
    DataCopy(readLocal_, msGm_[0], totalElem_);  // MS → UB

    PipeBarrier<PIPE_ALL>();  // 确保读完
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

    DataCopy(readLocal_, msGm_[0], totalElem_);  // 重读一次
    for (uint32_t i = 0; i < totalElem_; i++) {
        if (readLocal_.GetValue(i) != static_cast<T>(i & 0xFF)) {
            verifyResult_ = 1;
            return;
        }
    }
}

template <typename T>
__aicore__ inline void LocalMsRw<T>::Process()
{
    if (aivId_ > 0) return;

    // 打点外：填充数据
    if (mode0_ == MODE_WRITE_THEN_READ) {
        FillData();
    } else {
        uint32_t elemPerBlock = BLOCK_4K_BYTES / sizeof(T);
        Duplicate<T>(dataLocal_, (T)1, elemPerBlock);
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

    PipeBarrier<PIPE_ALL>();
    uint64_t endCycle = GetSystemCycle();

    // 打点外：校验（仅新模式）
    if (mode0_ == MODE_WRITE_THEN_READ) {
        VerifyResult();
    }

    // 写结果
    timeGm_.SetValue(0, startCycle);
    timeGm_.SetValue(1, endCycle);
    timeGm_.SetValue(2, verifyResult_);
}

} // namespace LocalMsRwImpl

#endif // LOCAL_MS_RW_KERNEL_H
