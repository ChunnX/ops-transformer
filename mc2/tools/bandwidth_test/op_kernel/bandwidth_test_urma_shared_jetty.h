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
 * \file bandwidth_test_urma_shared_jetty.h
 * \brief
 */

#ifndef BANDWIDTH_TEST_URMA_SHARED_JETTY_H
#define BANDWIDTH_TEST_URMA_SHARED_JETTY_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "bandwidth_test_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"

#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"

namespace BandwidthTestImpl {

using namespace AscendC;
using namespace Mc2Kernel;
using namespace Mc2Aclnn;

#define TemplateBandwidthTestTypeClass typename XType, typename YType
#define TemplateBandwidthTestTypeFunc XType, YType

// Wire layout matches the operator-local Torch context builder.
struct SharedJettyContext {
    uint32_t epRankId;
    uint32_t sharedJettyNum;
    uint64_t bufferSize;
    uint64_t epHcclBuffer_[1024];
    uint64_t sharedJettyHandle_[1024];
};

template <TemplateBandwidthTestTypeClass>
class BandwidthTestUrmaSharedJetty {
public:
    __aicore__ inline BandwidthTestUrmaSharedJetty(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR dstRankId, GM_ADDR y, GM_ADDR receiveCnt,
                                GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void StreamFromWindow();
    __aicore__ inline void SendWindowToRank(uint32_t rankId, GM_ADDR epochAddr);
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void BeginBatch(uint32_t rankId);
    __aicore__ inline void PrepareWrite(GM_ADDR dst, GM_ADDR src, uint32_t len, bool lastInBatch);
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BufferInit();
    __aicore__ inline void ProcessToken(GlobalTensor<XType> &outTokenGT, uint32_t tokenIndex);
    __aicore__ inline void SplitToCore(uint32_t itemCount, uint32_t coreCount, uint32_t &startIndex,
                                       uint32_t &endIndex);
    __aicore__ inline uint32_t RemoteRankForJetty(uint32_t jettyId)
    {
        return jettyId < static_cast<uint32_t>(epRankId_) ? jettyId : jettyId + 1;
    }

    // 获取指定 rank 的通信窗口数据区地址
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffset_;
    }

    // 获取指定 rank 的通信窗口状态区地址
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)context_->epHcclBuffer_[rankId] + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;                // 输入数据 x
    GlobalTensor<int32_t> dstRankIdGMTensor_;      // 专家ID (用于确定目标rank)
    GlobalTensor<float> windowInStatusFp32Tensor_; // 窗口状态区(用于同步)
    LocalTensor<XType> xTmpTensor_;                // 临时存储token数据
    LocalTensor<int32_t> dstRankIdsTensor_;        // 本地缓存的dst_rank_id数组
    LocalTensor<int32_t> statusTensor_;            // 状态数据
    LocalTensor<float> statusFp32Tensor_;          // 状态数据(FP32视图)
    TBuf<> dstRankIdsBuf_;    // dst_rank_id 缓冲区
    TBuf<> statusBuf_;        // 状态缓冲区
    TBuf<> waitStatusBuf_;    // 等待状态缓冲区
    TBuf<> gatherMaskOutBuf_; // 中间计算缓冲区
    TBuf<> scalarBuf_;        // 标量缓冲区
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列

    // ==================== 全局内存地址 ====================
    GM_ADDR yGM_;           // 输出y的GM地址
    GM_ADDR receiveCntGM_;  // 接收计数的GM地址
    GM_ADDR statusSpaceGm_; // 状态空间的GM地址
    GM_ADDR windowGM_;      // 本地窗口的GM地址

    // ==================== HCCL通信上下文 ====================
    __gm__ SharedJettyContext *context_{nullptr};

    uint32_t axisBS_{0};              // batch size (token数量)
    uint32_t axisH_{0};               // hidden size (特征维度)
    uint32_t aivNum_{0};              // AIV核总数
    uint32_t sharedJettyNum_{0};      // 共享jetty数，也是参与发送的核数
    uint32_t epWorldSize_{0};         // rank总数
    int32_t epRankId_{0};             // 当前rank ID
    int32_t mode_{0};                // 模式: 0/1=业务路径, 2/3/4=Win打流
    uint32_t aivId_{0};               // 当前AIV核ID
    uint32_t globalBS_{0};            // 全局最大batch size
    uint32_t hOutSize_{0};            // 输出hidden size (字节)
    uint32_t hAlignWinSize_{0};       // 窗口对齐后的hidden size
    uint32_t hOutAlignUbSize_{0};     // UB对齐后的hidden size
    uint32_t hAlignSize_{0};          // UB对齐后的hidden size
    uint32_t startRankId_{0};         // 当前核处理的起始expert ID
    uint32_t endRankId_{0};           // 当前核处理的结束expert ID
    uint32_t dataState_{0};           // 数据状态 (用于双缓冲)
    uint32_t totalUsedUB_{0};         // 已使用的UB大小
    uint64_t winDataSizeOffset_{0};   // 窗口数据区偏移
    uint64_t expertPerSizeOnWin_{0};  // 每个expert在窗口中占用的空间
    uint32_t stateOffset_{0};         // 状态偏移 (32字节)
    uint32_t recStatusNumPerCore_{0}; // 每个核负责等待的rank数量
    uint32_t rscvStatusNum_{0};       // 需要等待的rank总数
    uint32_t remainderRankNum_{0};    // 分配后剩余的rank数量
    uint32_t startStatusIndex_{0};    // 起始状态索引
    uint32_t totalUbSize_{0};         // 总UB大小
    uint64_t totalWinSizeEp_{0};      // 总窗口大小
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::MultiChannelHandle>;
    TBuf<> batchBuf_;
    LocalTensor<uint8_t> batchTensor_;
    HcommBatchHandle batchHandle_{};
    uint32_t batchRankId_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t countStride_{0};
    GM_ADDR countsGM_{nullptr};
    TBuf<> countsBuf_;
    LocalTensor<uint32_t> countsTensor_;
    __aicore__ inline void CountTokens();
    __aicore__ inline uint32_t GetCoreCount(uint32_t core, uint32_t rankId)
    {
        return ReadGmByPassDCache(reinterpret_cast<__gm__ uint32_t *>(countsGM_) + core * countStride_ + rankId);
    }
    uint32_t batchCapacity_{0};
    uint32_t batchBufferBytes_{0};
    uint32_t preparedWrites_{0};
    float sumTarget_;                 // 状态同步目标值 (1.0)

    // Win streaming modes use separate slots inside the 1 MiB state area, outside the business fields.
    static constexpr uint32_t STREAM_FLAG_STRIDE = 32;
    static constexpr uint32_t STREAM_FLAGS_OFFSET = 128 * 1024;
    static constexpr uint32_t STREAM_EPOCH_OFFSET =
        STREAM_FLAGS_OFFSET + BandwidthTestTiling::URMA_MAX_SHARED_JETTYS * HCCL_MAX_RANK_SIZE * sizeof(uint32_t);
};

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR dstRankId, GM_ADDR y, GM_ADDR receiveCnt, GM_ADDR workspaceGM, TPipe *pipe,
    const BandwidthTestTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx(); // 获取当前核ID
    mode_ = tilingData->mode;
    epWorldSize_ = tilingData->worldSize;

    context_ = (__gm__ SharedJettyContext *)context;

    // ============ 从tiling获取配置参数 ============
    axisBS_ = tilingData->maxBs;   // token数量
    axisH_ = tilingData->dataSize; // hidden size
    aivNum_ = tilingData->aivNum;
    sharedJettyNum_ = tilingData->sharedJettyNum;
    ASCENDC_ASSERT(context_->sharedJettyNum == sharedJettyNum_, {
        KERNEL_LOG(KERNEL_ERROR, "shared Jetty count mismatch, context=%u, tiling=%u", context_->sharedJettyNum,
                   sharedJettyNum_);
    });
    epRankId_ = context_->epRankId;
    hOutSize_ = axisH_ * sizeof(XType);
    batchBufferBytes_ = tilingData->batchBufferBytes;
    batchCapacity_ = tilingData->batchTokens;
    // Direct Win streaming only needs communication parameters; its WQE buffer is allocated in Process.
    if (mode_ >= 2 && mode_ <= 4) {
        return;
    }

    totalWinSizeEp_ = tilingData->totalWinSize;
    totalUbSize_ = static_cast<uint32_t>(tilingData->totalUbSize);
    globalBS_ = tilingData->maxBs * epWorldSize_;

    // ============ 初始化窗口状态 ============
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    statusSpaceGm_ = (GM_ADDR)(context_->epHcclBuffer_[epRankId_]);
    GlobalTensor<uint32_t> sharedDataStatusGMTensor;
    sharedDataStatusGMTensor.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusSpaceGm_ + DISPATCH_STATE_WIN_OFFSET));
    sharedDataStatusGMTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    // A shared context may be called with a different tiling-selected AIV count. Core 0 advances one common
    // ping-pong bit so every launched core observes the same window half, including cores absent from a prior call.
    LocalTensor<uint32_t> dataStateTensor = dataStateBuf.Get<uint32_t>();
    if (aivId_ == 0) {
        DataCopy(dataStateTensor, sharedDataStatusGMTensor, UB_ALIGN / sizeof(uint32_t));
        SyncFunc<HardEvent::MTE2_S>();
        uint32_t currentState = dataStateTensor.GetValue(ZERONE_STATE_POS);
        dataStateTensor.SetValue(ZERONE_STATE_POS, currentState == 0 ? 1 : 0);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy(sharedDataStatusGMTensor, dataStateTensor, UB_ALIGN / sizeof(uint32_t));
        SyncFunc<HardEvent::MTE3_S>();
    }
    SyncAll<true>();
    DataCopy(dataStateTensor, sharedDataStatusGMTensor, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<HardEvent::MTE2_S>();
    dataState_ = dataStateTensor.GetValue(ZERONE_STATE_POS) == 0 ? 1 : 0;

    // ============ 设置全局张量 ============
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    dstRankIdGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstRankId);

    yGM_ = y;
    receiveCntGM_ = receiveCnt;

    // ============ 计算对齐参数 ============
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;     // UB 32字节对齐
    hAlignWinSize_ = Ceil(hAlignSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // 窗口512字节对齐
    hOutAlignUbSize_ = hAlignSize_;

    // 每个rank在窗口中的空间大小 = global_bs/world_size * 对齐后的hidden size
    uint32_t axisMaxBS = globalBS_ / epWorldSize_;
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS) * hAlignWinSize_;

    // ============ 计算状态等待的任务分发 ============
    // 将 world_size 个 rank 的状态等待任务分发到各 AIV 核
    stateOffset_ = 32UL; // 32字节
    rscvStatusNum_ = epWorldSize_;
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_;
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_;
    if (aivId_ < remainderRankNum_) {
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }

    // ============ 初始化状态缓冲区 ============
    uint32_t statusBufCntAlign = Ceil(Ceil(epWorldSize_, sharedJettyNum_), 8) * 8;
    uint32_t statusBufSize = statusBufCntAlign * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    statusTensor_ = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign * 8);
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankId_);

    // 初始化状态为 1.0f (0x3F800000)，用于状态同步
    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = {0x101010101010101, 0};
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign / 8, 1, 8);

    // ============ 计算窗口数据区偏移 ============
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffset_ = dataState_ * (totalWinSizeEp_ / 2) + axisMaxBS * hSizeAlignCombine;
    windowGM_ = GetWindAddrByRankId(epRankId_);
    windowInStatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
    windowInStatusFp32Tensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

    // ============ 初始化其他缓冲区 ============
    uint32_t rankIdBufferBytes = batchBufferBytes_ / BandwidthTestTiling::URMA_WQE_BYTES * sizeof(int32_t);
    rankIdBufferBytes = Ceil(rankIdBufferBytes, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(dstRankIdsBuf_, rankIdBufferBytes);
    dstRankIdsTensor_ = dstRankIdsBuf_.Get<int32_t>();
    // ProcessToken waits for MTE3 before freeing the row, so the next token can reuse the same UB.
    tpipe_->InitBuffer(xQueue_, BUFFER_SINGLE, hOutAlignUbSize_);
    tpipe_->InitBuffer(batchBuf_, batchBufferBytes_);
    batchTensor_ = batchBuf_.Get<uint8_t>();
    // The first sharedJettyNum cores each own one jetty and one complete-token BS slice.
    SplitToCore(axisBS_, sharedJettyNum_, startTokenId_, endTokenId_);
    SplitToCore(epWorldSize_, sharedJettyNum_, startRankId_, endRankId_);
    countStride_ = Ceil(epWorldSize_, 8) * 8;
    countsGM_ = GetUserWorkspace(workspaceGM);
    tpipe_->InitBuffer(countsBuf_, countStride_ * sizeof(uint32_t));
    countsTensor_ = countsBuf_.Get<uint32_t>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::SplitToCore(
    uint32_t itemCount, uint32_t coreCount, uint32_t &startIndex, uint32_t &endIndex)
{
    uint32_t itemsPerCore = itemCount / coreCount;
    uint32_t remainder = itemCount % coreCount;
    startIndex = itemsPerCore * aivId_ + (aivId_ < remainder ? aivId_ : remainder);
    endIndex = startIndex + itemsPerCore + (aivId_ < remainder ? 1 : 0);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::ProcessToken(
    GlobalTensor<XType> &outTokenGT, uint32_t tokenIndex)
{
    // 处理单个token：
    // 1. 从输入GM读取token数据到UB
    // 2. 将token数据写入目标rank的窗口

    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams hCommuCopyOutParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};

    // 从GM读取token到UB
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopyPad(xTmpTensor_, xGMTensor_[static_cast<uint64_t>(tokenIndex) * axisH_], xCopyParams, padParams);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 写入目标rank的窗口
    DataCopyPad(outTokenGT, xTmpTensor_, hCommuCopyOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    xQueue_.FreeTensor<XType>(xTmpTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::BeginBatch(uint32_t rankId)
{
    batchRankId_ = rankId;
    preparedWrites_ = 0;
    // The multi-channel handle keeps one queue cursor for this core-owned Jetty; GetHandleRef selects the peer.
    batchHandle_ =
        hcomm_.MakeBatchHandle(static_cast<AscendC::MultiChannelHandle>(context_->sharedJettyHandle_[aivId_]),
                               batchTensor_, batchBufferBytes_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::PrepareWrite(
    GM_ADDR dst, GM_ADDR src, uint32_t len, bool lastInBatch)
{
    uint32_t channelIndex = batchRankId_ < static_cast<uint32_t>(epRankId_) ? batchRankId_ : batchRankId_ - 1;
    auto &peerHandle = hcomm_.GetHandleRef(batchHandle_, channelIndex, dst);
    // A failed batch must abort the launch before any success flag can be published.
    if (hcomm_.WriteNbi(peerHandle, dst, src, len) != HCOMM_SUCCESS) {
        Trap();
    }
    ++preparedWrites_;
    if (lastInBatch || preparedWrites_ == batchCapacity_) {
        if (hcomm_.BatchCommit(batchHandle_) != HCOMM_SUCCESS) {
            Trap();
        }
        if (hcomm_.Drain(batchHandle_) != HCOMM_SUCCESS) {
            Trap();
        }
        preparedWrites_ = 0;
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::CountTokens()
{
    Duplicate<uint32_t>(countsTensor_, 0, countStride_);
    SyncFunc<HardEvent::V_S>();
    if (aivId_ < sharedJettyNum_) {
        for (uint32_t start = startTokenId_; start < endTokenId_;) {
            uint32_t count = endTokenId_ - start;
            count = count < batchCapacity_ ? count : batchCapacity_;
            DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0U};
            SyncFunc<HardEvent::S_MTE2>();
            DataCopyPad(dstRankIdsTensor_, dstRankIdGMTensor_[start], copy, pad);
            SyncFunc<HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < count; ++i) {
                int32_t rankId = dstRankIdsTensor_.GetValue(i);
                // Match ordinary URMA: destination IDs outside the group do not contribute a route.
                if (rankId >= 0 && rankId < static_cast<int32_t>(epWorldSize_)) {
                    countsTensor_.SetValue(rankId, countsTensor_.GetValue(rankId) + 1);
                }
            }
            start += count;
        }
        GlobalTensor<uint32_t> counts;
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(countsGM_) + aivId_ * countStride_);
        counts.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy(counts, countsTensor_, countStride_);
        SyncFunc<HardEvent::MTE3_S>();
    }
    SyncAll<true>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::SendToMoeExpert()
{
    CountTokens();
    if (aivId_ >= sharedJettyNum_) {
        return;
    }
    GM_ADDR stagingGM = windowGM_ - expertPerSizeOnWin_;
    GlobalTensor<XType> dstWinGMTensor;
    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        uint32_t count = countsTensor_.GetValue(rankId);
        if (count == 0) {
            continue;
        }
        uint32_t prefix = 0;
        for (uint32_t core = 0; core < aivId_; ++core) {
            prefix += GetCoreCount(core, rankId);
        }
        bool isRemote = rankId != static_cast<uint32_t>(epRankId_);
        if (isRemote) {
            BeginBatch(rankId);
        }
        uint32_t sent = 0;
        for (uint32_t start = startTokenId_; start < endTokenId_;) {
            uint32_t chunk = endTokenId_ - start;
            chunk = chunk < batchCapacity_ ? chunk : batchCapacity_;
            DataCopyExtParams copy{1U, static_cast<uint32_t>(chunk * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0U};
            SyncFunc<HardEvent::S_MTE2>();
            DataCopyPad(dstRankIdsTensor_, dstRankIdGMTensor_[start], copy, pad);
            SyncFunc<HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < chunk; ++i) {
                if (dstRankIdsTensor_.GetValue(i) != static_cast<int32_t>(rankId)) {
                    continue;
                }
                uint32_t token = start + i;
                GM_ADDR remote = GetWindAddrByRankId(rankId) + epRankId_ * expertPerSizeOnWin_ +
                    static_cast<uint64_t>(prefix + sent) * hAlignWinSize_;
                GM_ADDR local = stagingGM + static_cast<uint64_t>(token) * hAlignWinSize_;
                dstWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(isRemote ? local : remote));
                dstWinGMTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                ProcessToken(dstWinGMTensor, token);
                ++sent;
                if (isRemote) {
                    PrepareWrite(remote, local, hOutSize_, sent == count);
                }
            }
            start += chunk;
        }
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::SetStatus()
{
    // Every data batch must complete before any core publishes the rank completion flag.
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>(); // 等待所有核完成统计
    if (startRankId_ >= epWorldSize_) {
        return;
    }

    // ============ 将状态写入目标rank的状态区 ============
    // 写入位置: 目标rank状态区 + 当前rank的偏移
    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t rankId = startRankId_; rankId < endRankId_; ++rankId) {
        uint32_t count = 0;
        for (uint32_t core = 0; core < sharedJettyNum_; ++core) {
            count += GetCoreCount(core, rankId);
        }
        statusTensor_.SetValue((rankId - startRankId_) * 8 + 1, count);
        GM_ADDR remoteAddr = GetWindStateAddrByRankId(rankId) + epRankId_ * stateOffset_;
        GM_ADDR localAddr = statusSpaceGm_ + (epWorldSize_ + rankId) * stateOffset_;
        bool isRemote = rankId != static_cast<uint32_t>(epRankId_);
        rankGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(isRemote ? localAddr : remoteAddr));
        rankGMTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy(rankGMTensor, statusTensor_[(rankId - startRankId_) * 8], 8U);
        SyncFunc<HardEvent::MTE3_S>();
        if (isRemote) {
            // Rebind the core-owned shared jetty for the status phase.
            BeginBatch(rankId);
            PrepareWrite(remoteAddr + sizeof(int32_t), localAddr + sizeof(int32_t), sizeof(int32_t), true);
            PrepareWrite(remoteAddr, localAddr, sizeof(int32_t), true);
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::BufferInit()
{
    // 重新初始化缓冲区，用于WaitDispatch阶段
    tpipe_->Reset();
    totalUsedUB_ = 0U;
    // WaitDispatch clears flags in groups of eight 32-byte status blocks.
    uint32_t waitStatusBufSize = Ceil(recStatusNumPerCore_, 8) * 8 * UB_ALIGN;
    waitStatusBufSize = waitStatusBufSize > 256 ? waitStatusBufSize : 256;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    totalUsedUB_ += waitStatusBufSize;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * sizeof(float);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t gatherMaskOutSize =
        (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    gatherMaskOutSize = Ceil(gatherMaskOutSize, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);
    totalUsedUB_ += gatherMaskOutSize;

    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2);
    totalUsedUB_ += UB_ALIGN * 2;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::WaitDispatch()
{
    // 等待所有rank完成数据发送
    // 通过轮询状态区，等待所有rank的状态flag变为1.0f

    BufferInit();
    startRankId_ = startStatusIndex_;
    endRankId_ = startRankId_ + recStatusNumPerCore_;

    // 如果当前核没有需要等待的rank，直接返回
    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        SyncAll<true>();
        return;
    }

    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    float compareTarget = sumTarget_ * recStatusNumPerCore_; // 目标值 = 1.0 * 等待的rank数量
    float sumOfFlag = static_cast<float>(-1.0);

    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};

    // ============ 轮询等待状态 ============
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        // 读取状态区的flag值
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(statusFp32Tensor_, windowInStatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        // 计算所有flag的和
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutBuf_.Get<float>(), 1, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    // ============ 清零状态区，为下一次通信做准备 ============
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInStatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    SyncAll<true>(); // 等待所有核完成
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::LocalWindowCopy()
{
    // 从本地窗口读取接收到的数据，拷贝到输出tensor
    // 输出按源 rank 顺序紧凑拼接，每段长度为该源 rank 发来的 token 数。

    if (startRankId_ >= rscvStatusNum_) {
        return;
    }

    // ============ 读取状态数据 ============
    TBuf<> localStatusBuf;
    // 状态区包含 world_size 个状态块，每个块 8 个 int32 (32B)
    const uint32_t statusCopySize = epWorldSize_ * UB_ALIGN;
    tpipe_->InitBuffer(localStatusBuf, statusCopySize);
    totalUsedUB_ += statusCopySize;
    // 获取状态张量（int32 视图）
    LocalTensor<int32_t> localStatusTensor = localStatusBuf.Get<int32_t>();

    // 从窗口状态区读取所有状态
    GlobalTensor<int32_t> srcStatus = windowInStatusFp32Tensor_.ReinterpretCast<int32_t>();
    DataCopyParams statusCopyParams{static_cast<uint16_t>(epWorldSize_), 1, 0, 0};
    DataCopy(localStatusTensor, srcStatus, statusCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 当前核负责的首个源 rank 之前，所有源 rank 的接收 token 数构成输出偏移。
    uint32_t preCnt = 0;
    for (uint32_t i = 0; i < startRankId_; i++) {
        uint32_t value = localStatusTensor(i * 8 + 1);
        preCnt += value; // token数量存储在位置1
    }

    for (uint32_t idx = startRankId_; idx < endRankId_; idx++) {
        // 只取 token 计数（索引 1）
        uint32_t receiveCount = localStatusTensor(idx * 8 + 1);
        WriteGmByPassDCache(reinterpret_cast<__gm__ uint32_t *>(receiveCntGM_) + idx, receiveCount);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    // ============ 初始化数据拷贝队列 ============
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tripleQueue;
    uint32_t leftUBSize = ((totalUbSize_ - totalUsedUB_) / UB_ALIGN) * UB_ALIGN;
    uint32_t dataSize = (leftUBSize / BUFFER_NUM / UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tripleQueue, BUFFER_NUM, dataSize);

    uint32_t countPerDataSize = dataSize / hOutAlignUbSize_; // 每次拷贝的token数量
    if (countPerDataSize == 0) {
        countPerDataSize = 1;
    }
    uint32_t beginIdx = preCnt;

    // ============ 遍历每个rank，从窗口拷贝数据到输出 ============
    for (uint32_t index = startRankId_; index < endRankId_; index++) {
        uint32_t count = localStatusTensor(index * 8 + 1); // 从该rank收到的token数量

        // 计算窗口中该rank数据的起始地址
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + index * expertPerSizeOnWin_;
        GlobalTensor<YType> tokGlobal, yOutGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        // 分批拷贝数据
        uint32_t loopTimes = Ceil(count, countPerDataSize);
        uint32_t lastTimeCount = count - countPerDataSize * (loopTimes - 1);
        for (uint32_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
            uint32_t tokenCopiedNumThisExpert = countPerDataSize * loopIdx;
            tokGlobal.SetGlobalBuffer(
                (__gm__ YType *)(wAddr + static_cast<uint64_t>(tokenCopiedNumThisExpert) * hAlignWinSize_));

            uint16_t blockCount = static_cast<uint16_t>(((loopIdx < loopTimes - 1) ? countPerDataSize : lastTimeCount));
            uint16_t copyBlockLen = static_cast<uint16_t>(hOutAlignUbSize_ / UB_ALIGN);
            uint16_t srcStrideTokGlobal = static_cast<uint16_t>((hAlignWinSize_ - hOutAlignUbSize_) / UB_ALIGN);
            DataCopyParams dataCopyParamsTokGlobal{blockCount, copyBlockLen, srcStrideTokGlobal, 0U};

            // 从窗口读取到UB
            LocalTensor<YType> tripleLocal = tripleQueue.AllocTensor<YType>();
            DataCopy(tripleLocal, tokGlobal, dataCopyParamsTokGlobal);
            tripleQueue.EnQue(tripleLocal);
            tripleLocal = tripleQueue.DeQue<YType>();

            // 从UB写入输出tensor
            yOutGlobal.SetGlobalBuffer(
                (__gm__ YType *)(yGM_ + (static_cast<uint64_t>(beginIdx) + tokenCopiedNumThisExpert) * hOutSize_));
            uint32_t srcStrideY = (hOutAlignUbSize_ - hOutSize_) / UB_ALIGN;
            DataCopyExtParams dataCopyParamsY{blockCount, hOutSize_, srcStrideY, 0U, 0U};
            DataCopyPad(yOutGlobal, tripleLocal, dataCopyParamsY);
            tripleQueue.FreeTensor(tripleLocal);
        }
        beginIdx += count;
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::SendWindowToRank(
    uint32_t rankId, GM_ADDR epochAddr)
{
    BeginBatch(rankId);
    GM_ADDR remoteWin = reinterpret_cast<GM_ADDR>(context_->epHcclBuffer_[rankId]);
    uint32_t start = 0;
    uint32_t remaining = axisBS_;
    uint64_t sourceBytes = static_cast<uint64_t>(axisBS_) * hOutSize_;
    GM_ADDR src = windowGM_ + STATE_SIZE;
    GM_ADDR dst = remoteWin + STATE_SIZE + sourceBytes + static_cast<uint64_t>(epRankId_) * sourceBytes;
    if (mode_ == 2) {
        uint32_t peerTokens = axisBS_ / epWorldSize_;
        uint64_t peerBytes = static_cast<uint64_t>(peerTokens) * hOutSize_;
        uint32_t end = 0;
        SplitToCore(peerTokens, sharedJettyNum_, start, end);
        remaining = end - start;
        dst = remoteWin + STATE_SIZE + sourceBytes + static_cast<uint64_t>(epRankId_) * peerBytes +
            static_cast<uint64_t>(start) * hOutSize_;
        src += static_cast<uint64_t>(rankId) * peerBytes + static_cast<uint64_t>(start) * hOutSize_;
    } else if (mode_ == 3) {
        // Each Jetty reads the full source block and writes its own receive slot on every remote rank.
        dst = remoteWin + STATE_SIZE + sourceBytes +
            (static_cast<uint64_t>(epRankId_) * sharedJettyNum_ + aivId_) * sourceBytes;
    }
    while (remaining != 0) {
        --remaining;
        PrepareWrite(dst, src, hOutSize_, remaining == 0);
        src += hOutSize_;
        dst += hOutSize_;
    }
    // Each (source rank, core) has a separate epoch flag, including cores with an empty BS range.
    GM_ADDR flag = remoteWin + STREAM_FLAGS_OFFSET +
        (static_cast<uint64_t>(epRankId_) * BandwidthTestTiling::URMA_MAX_SHARED_JETTYS + aivId_) * sizeof(uint32_t);
    PrepareWrite(flag, epochAddr, sizeof(uint32_t), true);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::StreamFromWindow()
{
    if (aivId_ >= sharedJettyNum_) {
        return;
    }
    tpipe_->InitBuffer(batchBuf_, batchBufferBytes_);
    batchTensor_ = batchBuf_.Get<uint8_t>();
    windowGM_ = reinterpret_cast<GM_ADDR>(context_->epHcclBuffer_[epRankId_]);
    auto epochAddr = reinterpret_cast<__gm__ uint32_t *>(
        windowGM_ + STREAM_EPOCH_OFFSET + aivId_ * STREAM_FLAG_STRIDE);
    uint32_t epoch = ReadGmByPassDCache(epochAddr) + 1U;
    WriteGmByPassDCache(epochAddr, epoch);
    if (mode_ == 4) {
        uint32_t remoteRank = RemoteRankForJetty(aivId_);
        SendWindowToRank(remoteRank, reinterpret_cast<GM_ADDR>(epochAddr));
        uint32_t senderJetty = static_cast<uint32_t>(epRankId_) < remoteRank
            ? static_cast<uint32_t>(epRankId_)
            : static_cast<uint32_t>(epRankId_ - 1);
        auto flag = reinterpret_cast<__gm__ uint32_t *>(
            windowGM_ + STREAM_FLAGS_OFFSET +
            (static_cast<uint64_t>(remoteRank) * BandwidthTestTiling::URMA_MAX_SHARED_JETTYS + senderJetty) *
                sizeof(uint32_t));
        while (static_cast<int32_t>(ReadGmByPassDCache(flag) - epoch) < 0) {}
        return;
    }
    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        if (rankId != static_cast<uint32_t>(epRankId_)) {
            SendWindowToRank(rankId, reinterpret_cast<GM_ADDR>(epochAddr));
        }
    }
    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        if (rankId != static_cast<uint32_t>(epRankId_)) {
            auto flag = reinterpret_cast<__gm__ uint32_t *>(
                windowGM_ + STREAM_FLAGS_OFFSET +
                (static_cast<uint64_t>(rankId) * BandwidthTestTiling::URMA_MAX_SHARED_JETTYS + aivId_) *
                    sizeof(uint32_t));
            while (static_cast<int32_t>(ReadGmByPassDCache(flag) - epoch) < 0) {}
        }
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTestUrmaSharedJetty<TemplateBandwidthTestTypeFunc>::Process()
{
    // 主处理函数
    // mode=0: 完整流程 (发送 + 同步 + 接收)
    // mode=1: 仅发送模式 (用于单向带宽测试)
    // mode=2/3/4: 按各自流量模型从Win发送，等待通信完成，不搬运或校验业务数据
    if ASCEND_IS_AIV {
        if (mode_ == 0) {
            SendToMoeExpert(); // 步骤1: 发送数据到各rank的窗口
            SetStatus();       // 步骤2: 设置状态通知对端
            WaitDispatch();    // 步骤3: 等待所有rank完成发送
            LocalWindowCopy(); // 步骤4: 从窗口读取数据到输出
        } else if (mode_ == 1) {
            SendToMoeExpert(); // 步骤1: 发送数据到各rank的窗口
            SetStatus();
            WaitDispatch();
        } else if (mode_ >= 2 && mode_ <= 4) {
            StreamFromWindow();
        }
    }
}

} // namespace BandwidthTestImpl
#endif // BANDWIDTH_TEST_URMA_SHARED_JETTY_H
