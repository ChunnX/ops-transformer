/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

// ============================================================================
// MTE AllToAll 算子 - Kernel 实现
// ============================================================================
//
// AllToAll 语义：每张卡将输入按 rank 切分成 rankDim 块，第 j 块发给第 j 张卡；
// 同时从所有卡接收属于自己的数据块（第 myRankId 块）。
//
// 核心区别（vs AllGather）：
//   AllGather: 从远程 Win 读取同偏移位置的数据，按 rank 拼接到 output
//   AllToAll:  从远程 Win 读取 myRankId 块的偏移位置数据，写入 output 的对应 rank 位

#ifndef MTE_ALLTOALL_KERNEL_H
#define MTE_ALLTOALL_KERNEL_H

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mte_alltoall_tiling_data.h"
#include "utils.h"
#include "mc2_comm.h"

namespace MteAllToAllImpl {

using namespace MC2CommImpl;
using namespace AscendC;

// 每次搬运/处理的数据量（元素数）
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;

// ============================================================================
// AllToAll Kernel 类
// ============================================================================
template<typename XType>
class MteAllToAll {
public:
    __aicore__ inline MteAllToAll() {};

    __aicore__ inline void InitWithMc2Context(GM_ADDR x, GM_ADDR output,
                                GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *pipe);

    // Kernel 主流程：CopyDataToWin → WriteStatusToWin → ReadStatus → Execute
    __aicore__ inline void Process();

private:
    // AllToAll 业务逻辑：从远程 Win 读取属于本卡的数据块
    __aicore__ inline void Execute();

    // ---- CopyDataToWin 阶段参数 ----
    uint64_t totalElements_{0};     // rankDim * sendCount
    uint64_t totalBytes_{0};        // totalElements * sizeof(int8_t)
    uint64_t tailXNums_{0};         // CopyDataToWin 尾块元素数

    // ---- Execute 阶段参数（双阶段核分配） ----
    uint64_t sendCount_{0};         // 每个 rank 发送/接收的块大小
    uint32_t execRound_{0};         // Execute 阶段每核基础块数
    uint32_t execTailBlockNums_{0}; // Execute 阶段多余块数
    uint32_t execAssignedBlockNums_{0};  // 本核 Execute 分配的块数
    uint64_t execXOffset_{0};       // 本核 Execute 起始偏移
    uint32_t execLastAivId_{0};     // Execute 阶段最后一个有任务的核 ID
    uint64_t execTailXNums_{0};     // Execute 阶段尾块元素数

    __gm__ MteAllToAllTilingData* tilingDataOut_;
    MTECommunication<XType> mteComm_;

    GlobalTensor<XType> remoteWinXTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xInQueue_;
};

// ============================================================================
// Init 实现
// ============================================================================
template <typename XType>
__aicore__ inline void MteAllToAll<XType>::InitWithMc2Context(
    GM_ADDR x, GM_ADDR output, GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *tPipe)
{
    // 1. 初始化 HCCL 通信上下文
    mteComm_.InitHcclContextByAddr(mc2Context);

    // 2. 读取 Tiling 数据
    tilingDataOut_ = (__gm__ MteAllToAllTilingData*)tilingGM;
    auto&& tilingInfo = tilingDataOut_->mteAllToAllTilingInfo;

    // 3. CopyDataToWin 阶段参数（按 totalElements = rankDim * sendCount 切分）
    sendCount_ = tilingInfo.sendCount;
    totalElements_ = tilingInfo.rankDim * sendCount_;
    totalBytes_ = totalElements_ * sizeof(XType);
    tailXNums_ = BlockAlignMod(totalElements_, X_PRE_BLOCK_NUM);
    uint32_t copyTotalBlocks = CeilDiv(totalBytes_, X_BLOCK_BYTES);

    // 4. CopyDataToWin 核分配（由 mteComm_ 管理）
    mteComm_.round_ = copyTotalBlocks / tilingInfo.aivNum;
    mteComm_.tailBlockNums_ = copyTotalBlocks % tilingInfo.aivNum;
    mteComm_.ComputeTailAivId(tilingInfo.aivNum);

    // 5. 初始化 Pipe 和 UB Buffer（包括 xInQueue_ 用于 Execute 阶段）
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES);

    // 6. 设置 CopyDataToWin 通信参数
    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, tilingInfo.aivNum, tailXNums_);
    mteComm_.InitParams();
    mteComm_.InitBuffer(tPipe);

    // 7. 初始化 GM 地址和 Win 窗口映射
    mteComm_.InitGMTensor(x, output, totalElements_, totalBytes_);

    // 8. Execute 阶段独立核分配（按 sendCount 切分）
    uint64_t sendBytes = sendCount_ * sizeof(XType);
    uint32_t execTotalBlocks = CeilDiv(sendBytes, X_BLOCK_BYTES);
    execRound_ = execTotalBlocks / tilingInfo.aivNum;
    execTailBlockNums_ = execTotalBlocks % tilingInfo.aivNum;

    uint32_t aivId = mteComm_.aivId_;
    execAssignedBlockNums_ = (aivId < execTailBlockNums_) ? execRound_ + 1 : execRound_;
    uint64_t execBlockIdx = aivId * execRound_ + (aivId < execTailBlockNums_ ? aivId : execTailBlockNums_);
    execXOffset_ = execBlockIdx * X_PRE_BLOCK_NUM;
    execTailXNums_ = BlockAlignMod(sendCount_, X_PRE_BLOCK_NUM);

    // Execute 阶段尾块核 ID
    if (execRound_ == 0) {
        execLastAivId_ = execTailBlockNums_ - 1;
    } else {
        execLastAivId_ = tilingInfo.aivNum - 1;
    }
}

// ============================================================================
// Execute 实现 — AllToAll 核心业务逻辑
// ============================================================================
//
// 对每个远程 rank，从其 Win 窗口读取第 myRankId 块数据（偏移 = myRankId * sendCount + execOffset），
// 写入本卡 output 的第 remoteRankId 位置（偏移 = remoteRankId * sendCount + execOffset）。
template <typename XType>
__aicore__ inline void MteAllToAll<XType>::Execute()
{
    for (uint64_t curBlock = 0; curBlock < execAssignedBlockNums_; ++curBlock) {
        uint64_t execOffset = execXOffset_ + curBlock * X_PRE_BLOCK_NUM;

        // 尾块处理：最后一个核的最后一块可能是非完整块
        uint32_t copyBlockNum = X_PRE_BLOCK_NUM;
        if ((mteComm_.aivId_ == execLastAivId_) && (curBlock == execAssignedBlockNums_ - 1)) {
            copyBlockNum = execTailXNums_;
        }

        // 遍历所有远程 rank
        for (uint32_t remoteRankId = 0; remoteRankId < mteComm_.hcclContext_->rankDim;
             ++remoteRankId) {
            // 获取远程 rank 的 Win 数据区地址
            GM_ADDR remoteDataGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);
            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataGm);

            // AllToAll 偏移计算（核心区别）：
            // 源：远程 rank 的 Win 中第 myRankId 块 + 本核的 Execute 偏移
            uint64_t srcOffset = mteComm_.hcclContext_->rankId * sendCount_ + execOffset;
            // 目的：输出的第 remoteRankId 块 + 本核的 Execute 偏移
            uint64_t dstOffset = remoteRankId * sendCount_ + execOffset;

            // Win → UB → Output
            LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
            DataCopy(xTmpTensor, remoteWinXTensor_[srcOffset], copyBlockNum);
            xInQueue_.EnQue(xTmpTensor);
            xTmpTensor = xInQueue_.DeQue<XType>();
            DataCopy(mteComm_.outputTensor_[dstOffset], xTmpTensor, copyBlockNum);
            xInQueue_.FreeTensor(xTmpTensor);
        }
    }
}

// ============================================================================
// Process 实现 — Kernel 主流程
// ============================================================================
template <typename XType>
__aicore__ inline void MteAllToAll<XType>::Process()
{
    // AIC 核不参与计算，直接返回
    if ASCEND_IS_AIC {
        return;
    }

    // 标准三步通信流程：
    // 1. 将本卡全部输入数据写入 Win 窗口（按 totalElements 切分）
    mteComm_.CopyDataToWin();
    // 2. 通知所有卡数据已就绪
    mteComm_.WriteStatusToWin();
    // 3. 等待所有卡数据就绪
    mteComm_.ReadStatus();
    // 4. AllToAll 业务逻辑：按 sendCount 切分，从远程 Win 读取属于本卡的数据块
    Execute();
}

}  // namespace MteAllToAllImpl
#endif
