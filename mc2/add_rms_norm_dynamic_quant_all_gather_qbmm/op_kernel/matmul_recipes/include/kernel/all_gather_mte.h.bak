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
 * \file all_gather_mte.h
 * \brief AllGather MTE 通信核心逻辑
 *
 * AIV 核上按 M×K 分块执行 AllGather 数据搬移（SyncMode=0 切K硬同步，>1 CV软同步），
 * 将多 rank 的输入数据汇聚到本端 win 区供 AIC 侧 Matmul 使用。
 */

#ifndef ALL_GATHER_MTE_H
#define ALL_GATHER_MTE_H

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/all_gather_matmul_tiling_data.h"
#include "all_gather_mte_comm.h"
#include "all_gather_mte_utils.h"

namespace AllGatherImpl {

using namespace QuantMTECommImpl;
using namespace AscendC;

// ── 常量定义 ──────────────────────────────────────────────────────────
constexpr static uint32_t NO_TILE_K = 1U;              // SyncMode 值：不切 K 路径
constexpr static uint32_t X_PER_BLOCK_NUM = 512U;      // 单次搬运 x 元素数（8bit dtype 时 512 个 = 512B）
constexpr static uint64_t ALLOC_UB_SPACE = 200UL * 1024UL; // UB 可用空间上限 200KB

/*!
 * \class AllGatherMte
 * \brief AllGather 通信核心模板类，管理 MTE win 区数据搬移和跨 rank 同步
 *
 * 模板参数 SyncMode: 0=切K硬同步, 1=不切K, >1=CV软同步
 */
template<AllGatherTemplateTypeClass>
class AllGatherMte {
public:
    __aicore__ inline AllGatherMte() {};

    // ── 初始化 ──
    __aicore__ inline void InitParams(const __gm__ allGatherTilingData *tilingData, GM_ADDR hcclContext);
    __aicore__ inline void InitBuffer(TPipe *tPipe);

    // ── 同步接口 ──
    __aicore__ inline void SetRemoteFlag();
    __aicore__ inline void WaitRemoteFlag();

    // ── 数据搬移 ──
    __aicore__ inline void ExecuteAllGather(GM_ADDR allGatherDataAddr, GM_ADDR allGatherScalesAddr);
    __aicore__ inline void CopyLocalData(GM_ADDR inputXAddr, GM_ADDR inputXScaleAddr);

    // ── 地址查询接口（供 Matmul 层使用）──
    __aicore__ inline GM_ADDR CalcCvFlagAddr(uint64_t mBlockIdx, uint64_t kBlockIdx);
    __aicore__ inline uint64_t GetWinDataOffset();
    __aicore__ inline uint64_t GetWinStatusOffset();
    __aicore__ inline GM_ADDR GetWinDataAddrGm();
    __aicore__ inline GM_ADDR GetSelfDataAddrGm();
    __aicore__ inline uint64_t GetSelfRankOffset();
    __aicore__ inline uint64_t GetSelfScaleOffset();
    __aicore__ inline GM_ADDR GetWinStatusAddrGm();

private:
    __aicore__ inline void ReadDataBlock(uint64_t curXOffset, uint32_t mCnt);
    __aicore__ inline void ReadScales();
    __aicore__ inline void SetCvAtomicFlag(uint32_t mBlockIdx, uint32_t kBlockIdx, uint32_t flagValue);

    // ── 基本维度参数 ──
    uint64_t rankSize_{0};              // 总 rank 数
    uint64_t xSize_{0};                 // 单 rank x 数据大小（字节）
    uint64_t xNums_{0};                 // 单 rank x 数据个数
    uint64_t scaleSize_{0};             // 单 rank scale 数据大小（字节）
    uint32_t totalBlockNums_{0};        // 按 X_BLOCK_BYTES 分块后的总块数
    uint64_t M_{0};                     // M 维度
    uint64_t K_{0};                     // K 维度

    // ── 核分工参数 ──
    uint32_t aicId_{0};                 // 当前 AIC 核编号（aivId / 2）
    uint32_t sendCoreNumPerRank_{0};    // 每个 remote rank 分配的核数
    uint32_t remoteRankId_{0};          // 当前核负责搬运的远端 rank ID
    uint64_t tileK_{0};                 // K 方向总 tile 数

    // ── UB 队列与缓冲区管理 ──
    uint64_t xInQueueSize_{0};          // xInQueue 单 buffer 大小

    // ── M×K 分块索引（SyncMode != NO_TILE_K 时使用）──
    int32_t mStartIndex_{0};            // 当前核负责的 M 起始行（含 remoteRank 偏移）
    int32_t kStartIndex_{0};            // 当前核负责的 K 起始偏移（按 512B 对齐）
    int32_t mEndIndex_{0};              // 当前核负责的 M 结束行（不含）
    uint32_t singleCoreM_{0};           // 每个 baseM 块的 M 行数（用于 CV 同步分块）
    uint32_t mLoop_{0};                 // M 方向循环次数（暂未使用）
    uint32_t mDim_{0};                  // M 方向分块数
    uint32_t kDim_{0};                  // K 方向分块数（通常为 2）
    uint32_t mBlockIdx_{0};             // 当前核的 M 方向块索引
    uint32_t kBlockIdx_{0};             // 当前核的 K 方向块索引
    uint32_t mMteCoreM_{0};             // 当前核负责的 M 行数（含尾部调整）
    uint32_t kMteCoreK_{0};             // K 方向每段长度
    uint32_t coreInnerMIndex_{0};       // 当前核在 M 方向的起始索引（不含 rank 偏移）

    // ── DataCopy 参数结构体 ──
    DataCopyExtParams dataCopyParamsIn_;            // 远端 → UB 的搬运参数
    DataCopyExtParams dataCopyParamsOut_;           // UB → 本端的搬运参数
    DataCopyExtParams scalesCopyParams_;            // scale 搬运参数
    DataCopyPadExtParams<int8_t> dataCopyPadParams_;    // x 数据 padding 参数
    DataCopyPadExtParams<ScalesType> scalesCopyPadParams_; // scale padding 参数

    // ── 通信子对象 ──
    MTECommunication<AllGatherTemplateType> mteComm_;   // MTE 底层通信（win 区地址、状态读写）

    // ── Tensor 资源 ──
    LocalTensor<int32_t> atomicAddTensor_;             // CV 同步用原子加本地缓冲
    GlobalTensor<int8_t> remoteWinXTensor_;             // 远端 win 区 x 数据 GM 映射
    GlobalTensor<ScalesType> remoteWinScaleTensor_;     // 远端 win 区 scale GM 映射
    GlobalTensor<int8_t> localWinXTensor_;              // 本端 win 区 x 数据 GM 映射
    GlobalTensor<int8_t> localXTensor_;                 // 原始输入 x GM 映射
    GlobalTensor<ScalesType> localXScaleTensor_;        // 原始输入 scale GM 映射
    GlobalTensor<ScalesType> localWinScaleTensor_;      // 本端 win 区 scale GM 映射
    GlobalTensor<int8_t> allGatherXOutTensor_;          // AllGather 调试输出 x GM 映射
    GlobalTensor<ScalesType> allGatherScaleOutTensor_;  // AllGather 调试输出 scale GM 映射
    GlobalTensor<int32_t> remoteCvFlagTensor_;          // CV 同步标志位 GM 映射

    // xInQueue: 双 buffer 队列，用于远端→UB→本端的数据中转
    // scaleInQue: 单 buffer 队列，用于 scale 的 UB 中转
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xInQueue_, scaleInQue;
    TBuf<> atomicAddBuf_;              // CV 同步原子加缓冲（64B 对齐）
};

/*!
 * \brief 初始化参数：解析 tiling，计算核分工，绑定 win 区地址
 *
 * 核分工计算（SyncMode != NO_TILE_K 时）：
 *   1. modCoreIndex = aicId_ % sendCoreNumPerRank_
 *   2. mBlockIdx_ = modCoreIndex % mDim_   （M 方向块索引）
 *   3. kBlockIdx_ = modCoreIndex / mDim_   （K 方向块索引）
 *   4. 大块分配：前 mTailNum 个 mBlock 各多分配 1 行（用于处理 M % mDim != 0 的余数）
 *   5. mStartIndex_ 包含 remoteRank 偏移：coreInnerMIndex_ + remoteRankId_ * M_
 *
 * @param tilingData   tiling 数据指针（__gm__ 地址空间）
 * @param hcclContext  HCCL 通信上下文 GM 地址
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::InitParams(
    const __gm__ allGatherTilingData *tilingData, GM_ADDR hcclContext)
{
    mteComm_.InitHcclContext(hcclContext);

    // 基本维度
    rankSize_ = tilingData->rankSize;
    M_ = tilingData->M;
    K_ = tilingData->Ka;
    xNums_ = tilingData->xNums;
    xSize_ = tilingData->xSize;
    scaleSize_ = tilingData->scaleSize;
    totalBlockNums_ = CeilDiv(xSize_, X_BLOCK_BYTES);
    sendCoreNumPerRank_ = tilingData->sendCoreNumPerRank;
    tileK_ = tilingData->mteTileK;

    // DataCopy 默认参数：无 padding
    dataCopyPadParams_ = {false, 0, 0, 0};
    scalesCopyParams_ = {1, scaleSize_, 0, 0, 0};
    scalesCopyPadParams_ = {false, 0, 0, 0};

    // 初始化底层通信参数（win 区地址、rankId 等）
    uint32_t aivNum = tilingData->aivNum;
    mteComm_.InitParams(xSize_, aivNum, sendCoreNumPerRank_);
    aicId_ = mteComm_.aivId_ / GetTaskRation();

    // 计算当前核在 sendCoreNumPerRank_ 中的索引
    uint32_t modCoreIndex = aicId_ % sendCoreNumPerRank_;

    // K 方向分为 kDim_ 段，M 方向分为 mDim_ 段
    kDim_ = tilingData->mteKSplitNum;
    mDim_ = tilingData->mteMSplitNum;

    // 当前核负责搬运的远端 rank
    remoteRankId_ = aicId_ / sendCoreNumPerRank_;

    // 从 modCoreIndex 分解出 (mBlockIdx_, kBlockIdx_)
    mBlockIdx_ = modCoreIndex % mDim_;
    kBlockIdx_ = modCoreIndex / mDim_;

    // 每段基础 M 行数和 K 长度
    mMteCoreM_ = tilingData->mteMSplitSize;
    kMteCoreK_ = tilingData->mteKSplitSize;
    singleCoreM_ = tilingData->singleCoreM;

    // 大块分配：前 mTailNum 个 mBlock 各多分配 1 行，吸收 M % mDim_ 的余数
    uint32_t mTailNum = tilingData->mteMSplitLargeBlockNum;
    if (mBlockIdx_ < mTailNum) {
        mMteCoreM_ += 1;
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_;
    } else {
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_ + mTailNum;
    }

    // 计算全局 M 起止索引（含 remoteRank 偏移，因为不同 rank 的数据在 win 区中按 rank 顺序排列）
    mStartIndex_ = coreInnerMIndex_ + remoteRankId_ * M_;
    kStartIndex_ = kBlockIdx_ * X_PER_BLOCK_NUM;
    mEndIndex_ = mStartIndex_ + mMteCoreM_;

    // 读取并翻转 0/1 分区标志位（用于区分当前是使用 win 区的 0 区还是 1 区）
    mteComm_.ReadAndFlipWinFlag();
}

/*!
 * \brief 分配 UB 缓冲区资源
 *
 * UB 空间分配优先级：
 *   1. scaleInQue: scaleSize_（scale 数据中转）
 *   2. atomicAddBuf_: 64B（CV 同步标志位）
 *   3. xInQueue_: 双 buffer，使用剩余空间（按 X_BLOCK_BYTES 对齐）
 *   4. mteComm_ 状态缓冲: writeStateBuf_ + readStateBuf_ + stateResetBuf_
 *
 * @param tPipe  管道管理器指针
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::InitBuffer(TPipe *tPipe)
{
    tPipe->Reset(); // UB 资源复位（规避 Ascend C 已知问题）

    // 计算除 xInQueue 外已分配的 UB 空间
    uint64_t usedSpace = scaleSize_ + CV_STATE_ALIGN * 2 + (rankSize_ * 2 + 1) * UB_ALIGN_BYTES;

    // xInQueue 双 buffer，每个 buffer 按 X_BLOCK_BYTES 对齐
    uint64_t availableSpaceForXInQueue = (ALLOC_UB_SPACE - usedSpace) / BUFFER_NUM / X_BLOCK_BYTES * X_BLOCK_BYTES;
    uint64_t demandSpaceForXInQueue = CeilAlignU64(xSize_, X_BLOCK_BYTES);
    xInQueueSize_ = availableSpaceForXInQueue < demandSpaceForXInQueue \
                    ? availableSpaceForXInQueue \
                    : demandSpaceForXInQueue;

    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xInQueueSize_);
    tPipe->InitBuffer(scaleInQue, 1, scaleSize_);
    tPipe->InitBuffer(atomicAddBuf_, CV_STATE_ALIGN); // 64B 对齐的原子加缓冲

    atomicAddTensor_ = atomicAddBuf_.Get<int32_t>();

    // 初始化底层通信模块的 UB 缓冲（状态位读写用）
    mteComm_.InitBuffer(tPipe);
}

/*!
 * \brief 向所有 rank 的 win 状态区写入本核数据搬运完成标志
 * 委托给 MTECommunication::WriteStatusToWin()
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::SetRemoteFlag()
{
    mteComm_.WriteStatusToWin();
}

/*!
 * \brief 轮询等待所有 rank 的状态就绪（软同步屏障）
 * 委托给 MTECommunication::ReadStatus()
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::WaitRemoteFlag()
{
    mteComm_.ReadStatus();
}

/*!
 * \brief 从远端 win 区搬运一个数据块到本端 win 区（经 UB 中转）
 *
 * 数据路径：远端 win[remoteRankId] → UB (xInQueue) → 本端 win[remoteRankId]
 *
 * NO_TILE_K 模式：搬运连续的完整 K 行
 * 切 K 模式：搬运 K 的一个分段，srcStride = K, dstStride = K（连续存储）
 *
 * @param curXOffset  在 win 区中的字节偏移（基于 win 区起始地址）
 * @param mCnt        本次搬运的 M 行数
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ReadDataBlock(uint64_t curXOffset, uint32_t mCnt)
{
    // self-rank 数据已由 CopyLocalData 写入 win 区，始终跳过
    if (remoteRankId_ == mteComm_.curRankId_) {
        return;
    }

    LocalTensor<int8_t> xTmpTensor = xInQueue_.AllocTensor<int8_t>();

    if constexpr (SyncMode == NO_TILE_K) {
        // 不切 K：连续搬运 mCnt 行完整 K 数据，无需 stride
        dataCopyParamsIn_ = {static_cast<uint16_t>(mCnt), static_cast<uint16_t>(K_ * sizeof(int8_t)), 0, 0, 0};
        dataCopyParamsOut_ = {static_cast<uint16_t>(mCnt), static_cast<uint16_t>(K_ * sizeof(int8_t)), 0, 0, 0};
    } else {
        // 切 K：srcStride = K（远端数据是连续完整 K 行），dstStride = K（本端也连续存储）
        dataCopyParamsIn_ = {static_cast<uint16_t>(mCnt), X_PER_BLOCK_NUM, K_ - X_PER_BLOCK_NUM, 0, 0};
        dataCopyParamsOut_ = {static_cast<uint16_t>(mCnt), X_PER_BLOCK_NUM, 0, K_ - X_PER_BLOCK_NUM, 0};
    }

    // 远端 win → UB → 本端 win
    DataCopyPad(xTmpTensor, remoteWinXTensor_[curXOffset], dataCopyParamsIn_, dataCopyPadParams_);
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<int8_t>();
    DataCopyPad(localWinXTensor_[curXOffset], xTmpTensor, dataCopyParamsOut_);

    // 调试输出：同时拷贝到 AllGather 独立输出缓冲区
    if constexpr (isOptionalOutput) {
        DataCopyPad(allGatherXOutTensor_[curXOffset], xTmpTensor, dataCopyParamsOut_);
    }
    xInQueue_.FreeTensor(xTmpTensor);
}

/*!
 * \brief 将本 rank 的输入数据（x + scale）拷贝到本端 win 区
 *
 * 核分工（以 28 个 AIC 核为例，GetTaskRation()=2）：
 *   - copyXBlockNum = 28/2 - 1 = 13 个核搬运 x 数据
 *   - 最后 1 个核（aicId == 13）搬运 scale 数据
 *
 * x 搬运的 M 分配：
 *   - 每核基础 mPerCore = M / copyXBlockNum + 1（向上取整近似）
 *   - 余数处理：前 remainderNum 个核各减 1 行，使总量 = M
 *
 * @param inputXAddr      输入 x 数据 GM 地址
 * @param inputXScaleAddr 输入 scale GM 地址
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::CopyLocalData(GM_ADDR inputXAddr, GM_ADDR inputXScaleAddr)
{
    // 获取本端 win 区数据起始地址
    GM_ADDR localWinGm = mteComm_.GetWinDataAddrGm(mteComm_.curRankId_, mteComm_.winFlag_);

    // 绑定 GM Tensor
    localXTensor_.SetGlobalBuffer((__gm__ int8_t*)inputXAddr);
    localXScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)inputXScaleAddr);
    localWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)localWinGm + mteComm_.curRankId_ * xSize_);
    localWinScaleTensor_.SetGlobalBuffer(
        (__gm__ ScalesType*)(localWinGm + mteComm_.winDataSize_ + mteComm_.curRankId_ * scaleSize_));

    // 搬运 x 的核数 = 总 AIC 核数 - 1（最后一个核留给 scale）
    uint32_t copyXBlockNum = GetBlockNum() / GetTaskRation() - 1;

    // 最后一个核搬运 scale
    if (aicId_ == copyXBlockNum) {
        LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
        DataCopyPad(scaleTmpTensor, localXScaleTensor_, scalesCopyParams_, scalesCopyPadParams_);
        scaleInQue.EnQue(scaleTmpTensor);
        scaleTmpTensor = scaleInQue.DeQue<ScalesType>();
        DataCopyPad(localWinScaleTensor_, scaleTmpTensor, scalesCopyParams_);
        scaleInQue.FreeTensor(scaleTmpTensor);
        return;
    }

    // 超出分配范围的核心不参与拷贝，防止越界写入覆盖 scale 区域
    if (aicId_ > copyXBlockNum) {
        return;
    }

    // M 分配：向上取整 + 余数调整
    uint32_t mPerCore = M_ / copyXBlockNum + 1;
    uint32_t remainderNum = M_ % copyXBlockNum;
    uint32_t mStartId = mPerCore * aicId_;
    if (aicId_ >= remainderNum) {
        mPerCore--;
        mStartId -= (aicId_ - remainderNum);
    }

    // 分批搬运：每批 mPerLoop 行，受 xInQueue 大小限制
    uint32_t mPerLoop = xInQueueSize_ / K_;
    uint32_t mLoopCnt = mPerCore / mPerLoop;
    uint32_t mTail = mPerCore % mPerLoop;
    uint64_t innerCurXOffset = mStartId * K_;

    for (uint32_t i = 0; i < mLoopCnt; ++i) {
        dataCopyParamsIn_ = {static_cast<uint16_t>(mPerLoop), static_cast<uint16_t>(K_ * sizeof(int8_t)), 0, 0, 0};
        dataCopyParamsOut_ = {static_cast<uint16_t>(mPerLoop), static_cast<uint16_t>(K_ * sizeof(int8_t)), 0, 0, 0};
        LocalTensor<int8_t> xTmpTensor = xInQueue_.AllocTensor<int8_t>();
        DataCopyPad(xTmpTensor, localXTensor_[innerCurXOffset], dataCopyParamsIn_, dataCopyPadParams_);
        xInQueue_.EnQue(xTmpTensor);
        xTmpTensor = xInQueue_.DeQue<int8_t>();
        DataCopyPad(localWinXTensor_[innerCurXOffset], xTmpTensor, dataCopyParamsOut_);
        xInQueue_.FreeTensor(xTmpTensor);
        innerCurXOffset += mPerLoop * K_;
    }
    // 搬运尾块
    if (mTail != 0) {
        dataCopyParamsIn_ = {static_cast<uint16_t>(mTail), static_cast<uint16_t>(K_ * sizeof(int8_t)), 0, 0, 0};
        dataCopyParamsOut_ = {static_cast<uint16_t>(mTail), static_cast<uint16_t>(K_ * sizeof(int8_t)), 0, 0, 0};
        LocalTensor<int8_t> xTmpTensor = xInQueue_.AllocTensor<int8_t>();
        DataCopyPad(xTmpTensor, localXTensor_[innerCurXOffset], dataCopyParamsIn_, dataCopyPadParams_);
        xInQueue_.EnQue(xTmpTensor);
        xTmpTensor = xInQueue_.DeQue<int8_t>();
        DataCopyPad(localWinXTensor_[innerCurXOffset], xTmpTensor, dataCopyParamsOut_);
        xInQueue_.FreeTensor(xTmpTensor);
    }
}

/*!
 * \brief 从远端 win 区搬运 scale 数据到本端 win 区
 *
 * 仅当 aicId_ % sendCoreNumPerRank_ == 0 时执行（每组核中只有第一个核搬运 scale）
 * 路径：远端 win scale → UB → 本端 win scale
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ReadScales()
{
    // self-rank 的 scale 已由 CopyLocalData 写入 win 区，始终跳过
    if (remoteRankId_ == mteComm_.curRankId_) {
        return;
    }

    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopyPad(scaleTmpTensor, remoteWinScaleTensor_, scalesCopyParams_, scalesCopyPadParams_);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

    DataCopyPad(localWinScaleTensor_, scaleTmpTensor, scalesCopyParams_);

    // 调试输出
    if constexpr (isOptionalOutput) {
        DataCopyPad(allGatherScaleOutTensor_, scaleTmpTensor, scalesCopyParams_);
    }
    scaleInQue.FreeTensor(scaleTmpTensor);
}

/*!
 * \brief 计算 CV 同步标志位在 win 状态区的地址
 *
 * CV 状态区布局（在通信状态区之后偏移 CV_SYNC_START_OFFSET=10KB）：
 *   cvFlagAddr = statusBase + CV_SYNC_START_OFFSET + (mBlockIdx * tileK + kBlockIdx) * 64B
 *
 * @param mBlockIdx  M 方向块索引
 * @param kBlockIdx  K 方向块索引
 * @return 标志位 GM 地址
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR AllGatherMte<AllGatherTemplateType>::CalcCvFlagAddr(
    uint64_t mBlockIdx, uint64_t kBlockIdx)
{
    GM_ADDR cvFlagBaseAddr = \
        mteComm_.GetWinStatusAddrGm(mteComm_.curRankId_, mteComm_.winFlag_) + CV_SYNC_START_OFFSET;
    GM_ADDR cvFlagAddr = cvFlagBaseAddr + (mBlockIdx * tileK_ + kBlockIdx) * CV_STATE_ALIGN;
    return cvFlagAddr;
}

/*!
 * \brief 获取 win 区数据区的偏移量（基于 0/1 分区标志）
 * 0 区偏移为 0（实际数据从 DATA_START_OFFSET 开始），1 区偏移为 WIN_DATA_OFFSET
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline uint64_t AllGatherMte<AllGatherTemplateType>::GetWinDataOffset()
{
    return mteComm_.winFlag_ * WIN_DATA_OFFSET;
}

/*!
 * \brief 获取远端 rank 的 win 数据区 GM 地址
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR AllGatherMte<AllGatherTemplateType>::GetWinDataAddrGm()
{
    return mteComm_.GetWinDataAddrGm(remoteRankId_, mteComm_.winFlag_);
}

/*!
 * \brief 获取 win 状态区偏移量（基于 0/1 分区标志）
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline uint64_t AllGatherMte<AllGatherTemplateType>::GetWinStatusOffset()
{
    return mteComm_.winFlag_ * WIN_STATUS_OFFSET;
}

/*!
 * \brief 获取本 rank 的 win 数据区 GM 地址（供 Matmul 读取 A 数据用）
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR AllGatherMte<AllGatherTemplateType>::GetSelfDataAddrGm()
{
    return mteComm_.GetWinDataAddrGm(mteComm_.curRankId_, mteComm_.winFlag_);
}

/*!
 * \brief 获取本 rank 数据在 win 区的偏移（字节数）
 * offset = curRankId * xSize
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline uint64_t AllGatherMte<AllGatherTemplateType>::GetSelfRankOffset()
{
    return mteComm_.curRankId_ * xSize_;
}

/*!
 * \brief 获取本 rank scale 在 win 区 scale 段的偏移（字节数）
 * offset = curRankId * scaleSize
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline uint64_t AllGatherMte<AllGatherTemplateType>::GetSelfScaleOffset()
{
    return mteComm_.curRankId_ * scaleSize_;
}

/*!
 * \brief 获取远端 rank 的 win 状态区 GM 地址
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR AllGatherMte<AllGatherTemplateType>::GetWinStatusAddrGm()
{
    return mteComm_.GetWinStatusAddrGm(remoteRankId_, mteComm_.winFlag_);
}

/*!
 * \brief 设置 CV 同步标志位（通过 atomicAdd 累加已完成行数）
 *
 * CV 软同步机制：
 *   - 每个 (mBlock, kBlock) 对应一个 64B 对齐的标志位
 *   - 多个核可能向同一个标志位累加（同 mBlock 不同核各自搬运部分行）
 *   - AIC 侧轮询标志位，当累加值 == singleCoreM 时表示该块数据就绪
 *
 * @param mBlockIdx  M 方向块索引
 * @param kBlockIdx  K 方向块索引
 * @param flagValue  本次累加的值（搬运的行数）
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::SetCvAtomicFlag(uint32_t mBlockIdx, uint32_t kBlockIdx, uint32_t flagValue)
{
    GM_ADDR cvFlagAddr = CalcCvFlagAddr(mBlockIdx, kBlockIdx);
    PipeBarrier<PIPE_ALL>();

    // 将 flagValue 写入本地缓冲，然后通过 atomicAdd 累加到远端标志位
    atomicAddTensor_.SetValue(0, flagValue);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    remoteCvFlagTensor_.SetGlobalBuffer((__gm__ int32_t*)cvFlagAddr);
    SetAtomicAdd<int32_t>();
    DataCopy(remoteCvFlagTensor_, atomicAddTensor_, CV_STATE_ALIGN / sizeof(int32_t));
    SetAtomicNone();
    PipeBarrier<PIPE_ALL>();
}

/*!
 * \brief 执行 AllGather：从对端 win 区读取数据写入本端 win 区
 *
 * Win 区数据布局：
 *   +------------+------------+-----+----------+-------------+-------------+-----+
 *   | Rank0 data | Rank1 data | ... | 512B pad | Rank0 scale | Rank1 scale | ... |
 *   +------------+------------+-----+----------+-------------+-------------+-----+
 *
 * 三种同步模式的数据搬运策略：
 *
 * 1. NO_TILE_K（SyncMode==0）：
 *    不切 K，每个核按连续 M 行搬运完整 K 数据。
 *    M 分配：与 CopyLocalData 类似的向上取整 + 余数调整。
 *
 * 2. 切 K 硬同步（SyncMode==1）：
 *    K 方向分 kDim_ 段，每段 kMteCoreK_ 长。
 *    每个 K 块之间 PipeBarrier 硬同步，确保所有核完成当前 K 块后再继续。
 *    K 交错：curXOffset + curKBlock * X_PER_BLOCK_NUM * 2（stride=2 因为 kDim_=2）
 *
 * 3. CV 软同步（SyncMode>1）：
 *    与切 K 硬同步类似，但用 atomicAdd 写 CV 标志位替代 PipeBarrier。
 *    AIC 侧可细粒度轮询每个 (mBlock, kBlock) 的就绪状态，实现通信-计算重叠。
 *    如果当前核的 M 范围跨越 baseM 块边界，则分两轮搬运。
 *
 * @param allGatherDataAddr   AllGather 调试输出 x 缓冲区地址
 * @param allGatherScalesAddr AllGather 调试输出 scale 缓冲区地址
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::ExecuteAllGather(
    GM_ADDR allGatherDataAddr, GM_ADDR allGatherScalesAddr)
{
    // ── 绑定远端和本端 win 区 GM Tensor ──

    // 远端 win 区数据地址：remoteRankId_ 对应的数据 slot
    GM_ADDR remoteWinGm = mteComm_.GetWinDataAddrGm(remoteRankId_, mteComm_.winFlag_);
    GM_ADDR remoteDataGm = remoteWinGm + remoteRankId_ * xSize_;
    remoteWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)remoteDataGm);

    // 本端 win 区数据地址：写入 remoteRankId_ 对应的 slot
    GM_ADDR localWinGm = mteComm_.GetWinDataAddrGm(mteComm_.curRankId_, mteComm_.winFlag_);
    GM_ADDR localDataGm = localWinGm + remoteRankId_ * xSize_;

    // 调试输出地址
    GM_ADDR allGatherOutDataGm = allGatherDataAddr + remoteRankId_ * xSize_;
    allGatherXOutTensor_.SetGlobalBuffer((__gm__ int8_t*)allGatherOutDataGm);
    localWinXTensor_.SetGlobalBuffer((__gm__ int8_t*)localDataGm);

    // ── Scale 搬运（每组第一个核执行）──
    if (aicId_ % sendCoreNumPerRank_ == 0) {
        GM_ADDR remoteScaleGm = remoteWinGm + mteComm_.winDataSize_ + remoteRankId_ * scaleSize_;
        remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)remoteScaleGm);
        GM_ADDR localScaleGm = localWinGm + mteComm_.winDataSize_ + remoteRankId_ * scaleSize_;
        localWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)localScaleGm);
        GM_ADDR allGatherOutScaleGm = allGatherScalesAddr + remoteRankId_ * scaleSize_;
        allGatherScaleOutTensor_.SetGlobalBuffer((__gm__ ScalesType*)allGatherOutScaleGm);
        ReadScales();
    }

    // ── X 数据搬运：按同步模式分发 ──
    if constexpr (SyncMode == NO_TILE_K) {
        // ── 模式 1：不切 K ──
        // 每个核搬运连续 M 行的完整 K 数据
        uint32_t mPerCore = M_ / sendCoreNumPerRank_;
        uint32_t remainderNum = M_ % sendCoreNumPerRank_;
        uint32_t mStartId = mPerCore * (aicId_ % sendCoreNumPerRank_);
        if (aicId_ % sendCoreNumPerRank_ >= sendCoreNumPerRank_ - remainderNum) {
            mPerCore++;
            mStartId += aicId_ % sendCoreNumPerRank_ - (sendCoreNumPerRank_ - remainderNum);
        }

        uint32_t mPerLoop = xInQueueSize_ / K_;
        uint32_t mLoopCnt = mPerCore / mPerLoop;
        uint32_t mTail = mPerCore % mPerLoop;
        uint64_t innerCurXOffset = mStartId * K_;

        for (uint32_t i = 0; i < mLoopCnt; ++i) {
            ReadDataBlock(innerCurXOffset, mPerLoop);
            innerCurXOffset += mPerLoop * K_;
        }
        if (mTail) {
            ReadDataBlock(innerCurXOffset, mTail);
        }
        SyncAll<true>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(9);
    } else {
        // 当前 K 段的循环次数
        uint32_t kLoop = kMteCoreK_ / X_PER_BLOCK_NUM;
        if (kBlockIdx_ < tileK_ % kDim_) {
            kLoop++;
        }

        // 当前核数据在 win 区中的起始偏移
        uint32_t curXOffset = coreInnerMIndex_ * K_ + kStartIndex_;
        if constexpr (SyncMode > NO_TILE_K) {
            // ── 模式 3：CV 软同步 ──
            // 检查 M 范围是否跨越 baseM 块边界
            uint32_t mCurOffset = M_ / singleCoreM_;
            uint32_t mTile = 1;
            uint32_t mFlagCount = mMteCoreM_;
            uint32_t baseMIndex = mStartIndex_ / singleCoreM_;

            // 若跨越 baseM 边界，分两轮：第一轮到边界，第二轮从边界到 mEndIndex_
            if (((mEndIndex_ - 1) / singleCoreM_ - mStartIndex_ / singleCoreM_) > 0) {
                mTile = 2;
                mFlagCount = Ceil(mStartIndex_, singleCoreM_) * singleCoreM_ - mStartIndex_;
            }

            // 第一轮：搬运并设置 CV 标志位
            for (uint64_t curKBlock = 0; curKBlock < kLoop; ++curKBlock) {
                // K 交错：步长 * 2（因为 kDim_=2，两个 K 块交错存储）
                uint64_t innerCurXOffset = curXOffset + curKBlock * X_PER_BLOCK_NUM * 2;
                ReadDataBlock(innerCurXOffset, mFlagCount);
                SetCvAtomicFlag(baseMIndex, kStartIndex_ / X_PER_BLOCK_NUM + curKBlock * 2, mFlagCount);
            }

            // 第二轮：如果 M 范围跨越了 baseM 边界
            if (mTile > 1) {
                curXOffset += mFlagCount * K_;
                mFlagCount = mEndIndex_ - Ceil(mStartIndex_, singleCoreM_) * singleCoreM_;
                baseMIndex++;
                for (uint64_t curKBlock = 0; curKBlock < kLoop; ++curKBlock) {
                    uint64_t innerCurXOffset = curXOffset + curKBlock * X_PER_BLOCK_NUM * 2;
                    ReadDataBlock(innerCurXOffset, mFlagCount);
                    SetCvAtomicFlag(baseMIndex, kStartIndex_ / X_PER_BLOCK_NUM + curKBlock * 2, mFlagCount);
                }
            }
            PipeBarrier<PIPE_MTE3>();
        } else {
            // ── 模式 2：切 K 硬同步 ──
            // 每个 K 块后 PipeBarrier 确保所有核同步
            for (uint64_t curKBlock = 0; curKBlock < kLoop; ++curKBlock) {
                uint64_t innerCurXOffset = curXOffset + curKBlock * X_PER_BLOCK_NUM * 2;
                ReadDataBlock(innerCurXOffset, mMteCoreM_);
                PipeBarrier<PIPE_MTE3>();
                SyncAll<true>();
                CrossCoreSetFlag<0x2, PIPE_MTE3>(9);
            }
            if (kLoop < ((tileK_ + kDim_ - 1) / kDim_)) {
                SyncAll<true>();
                CrossCoreSetFlag<0x2, PIPE_MTE3>(9);
            }
        }
    }
}
} // AllGatherImpl
#endif  // ALL_GATHER_MTE_H
