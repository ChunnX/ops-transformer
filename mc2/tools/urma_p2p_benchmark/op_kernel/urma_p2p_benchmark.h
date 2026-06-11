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
 * \file urma_p2p_benchmark.h
 * \brief URMA P2P基准测试Kernel类（SQE级通信）
 *
 * 使用P2pUrmaSqeComm进行SQE级URMA通信，通过HcclCombinOpParam上下文
 * （GetHcclContext内部获取）实现WriteWithNotify P2P通信。
 * 实现三种测试模式:
 * - mode=0: OverheadTest - 仅做本地DataCopy，测量算子调用固定开销
 * - mode=1: LatencyTest - P2P发送小数据包，每个token发送后立即Drain，测量单token时延
 * - mode=2: BandwidthTest - P2P发送完整数据块，所有WriteNbi完成后统一Drain，最大化带宽利用率
 */

#ifndef URMA_P2P_BENCHMARK_H
#define URMA_P2P_BENCHMARK_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "urma_p2p_benchmark_tiling.h"
#include "p2p_urma_comm.h"

namespace UrmaP2pBenchmarkImpl {

using namespace AscendC;

// ALIGNED_LEN_256, BUFFER_NUM, BUFFER_SINGLE 已在 p2p_urma_comm.h 中定义，此处不再重复

// UB大小上限，用于决定使用双缓冲还是单缓冲
#define MAX_UB_SIZE_P2P 240 * 1024UL

#define TemplateUrmaP2pBenchmarkTypeClass \
    typename XType, \
    typename YType
#define TemplateUrmaP2pBenchmarkTypeFunc XType, YType

template <TemplateUrmaP2pBenchmarkTypeClass>
class UrmaP2pBenchmark {
public:
    __aicore__ inline UrmaP2pBenchmark() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR receiveCnt,
        GM_ADDR workspaceGM, TPipe *pipe, const UrmaP2pBenchmarkTilingData *tilingData);
    __aicore__ inline void Process();

private:
    // ==================== 测试模式方法 ====================
    __aicore__ inline void OverheadTest();    // mode=0: 仅本地DataCopy，测量固定开销
    __aicore__ inline void LatencyTest();    // mode=1: P2P小包发送，逐token Drain
    __aicore__ inline void BandwidthTest();  // mode=2: P2P批量发送，统一 Drain

    // ==================== 通信辅助方法 ====================
    __aicore__ inline void ProcessToken(GlobalTensor<XType>& outTokenGT, uint32_t tokenIndex);
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BufferInit();
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstRankId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum);

    // ==================== TPipe ====================
    TPipe *tpipe_{nullptr};

    // ==================== 全局张量 ====================
    GlobalTensor<XType> xGMTensor_;              // 输入数据 x
    GlobalTensor<YType> yGMTensor_;              // 输出数据 y
    GlobalTensor<int32_t> receiveCntGMTensor_;   // 接收到的token计数
    GlobalTensor<float> windowInStatusFp32Tensor_; // 窗口状态区(用于同步)

    // ==================== 本地张量 ====================
    LocalTensor<XType> xTmpTensor_;              // 临时存储token数据
    LocalTensor<YType> yTmpTensor_;              // 临时存储输出数据
    LocalTensor<int32_t> dstRankIdsTensor_;      // 本地缓存的dst_rank_id数组
    LocalTensor<int32_t> statusTensor_;          // 状态数据
    LocalTensor<float> statusFp32Tensor_;        // 状态数据(FP32视图)
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;

    // ==================== 缓冲区 ====================
    TBuf<> dstRankIdsBuf_;      // dst_rank_id缓冲区
    TBuf<> statusBuf_;          // 状态缓冲区
    TBuf<> waitStatusBuf_;      // 等待状态缓冲区
    TBuf<> gatherMaskOutBuf_;   // 中间计算缓冲区
    TBuf<> scalarBuf_;          // 标量缓冲区
    TBuf<> tokenNumBuf_;        // token数量缓冲区
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列

    // ==================== 全局内存地址 ====================
    GM_ADDR yGM_;              // 输出y的GM地址
    GM_ADDR receiveCntGM_;     // 接收计数的GM地址
    GM_ADDR statusSpaceGm_;    // 状态空间的GM地址
    GM_ADDR windowGM_;         // 本地窗口(接收数据区)的GM地址

    // ==================== SQE级URMA通信 ====================
    P2pUrmaSqeComm urmaComm_;  // SQE级URMA P2P通信类实例

    // ==================== 基础参数 ====================
    uint32_t axisBS_{0};       // batch size (token数量)
    uint32_t axisH_{0};        // hidden size (特征维度)
    uint32_t aivNum_{0};       // AIV核总数
    uint32_t epWorldSize_{0};  // rank总数
    int32_t epRankId_{0};      // 当前rank ID
    int32_t mode_{0};          // 测试模式: 0=overhead, 1=latency, 2=bandwidth
    int32_t dstRankIdAttr_{0}; // P2P目标rank ID属性(-1表示使用输入tensor)
    uint32_t aivId_{0};        // 当前AIV核ID

    // ==================== 窗口参数 ====================
    uint64_t totalWinSizeEp_{0};      // 总窗口大小(每个rank)
    uint64_t winDataSizeOffset_{0};   // 窗口数据区偏移(双缓冲选择)
    uint64_t expertPerSizeOnWin_{0};  // 每个rank在窗口中占用的空间
    uint32_t hAlignWinSize_{0};       // 窗口对齐后的hidden size(512B对齐)
    uint32_t hAlignWinCnt_{0};        // 窗口对齐后的元素个数
    uint32_t hOutAlignUbSize_{0};     // UB对齐后的hidden size(32B对齐)
    uint32_t hAlignSize_{0};          // UB对齐后的hidden size(32B对齐)
    uint32_t hOutSize_{0};            // 原始hidden size(字节)

    // ==================== 状态参数 ====================
    uint32_t dataState_{0};           // 数据状态(用于双缓冲,0或1)
    uint32_t stateOffset_{0};         // 状态偏移(32字节)
    uint32_t recStatusNumPerCore_{0}; // 每个核负责等待的rank数量
    uint32_t rscvStatusNum_{0};       // 需要等待的rank总数
    uint32_t remainderRankNum_{0};    // 分配后剩余的rank数量
    uint32_t startStatusIndex_{0};    // 起始状态索引
    uint32_t bufferNum_{0};           // 缓冲区数量(1或2)
    uint32_t totalUbSize_{0};         // 总UB大小
    uint32_t globalBS_{0};            // 全局最大batch size
    uint32_t totalUsedUB_{0};         // 已使用的UB大小
    float sumTarget_;                 // 状态同步目标值(1.0)
};

// ============================================================================
// Init - 初始化Kernel类（两阶段UB分配: Phase1分配计算缓冲区）
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR receiveCnt,
    GM_ADDR workspaceGM, TPipe *pipe, const UrmaP2pBenchmarkTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx(); // 获取当前核ID

    // ============ 初始化SQE级URMA P2P通信类 ============
    urmaComm_.Init(tpipe_, workspaceGM);

    // ============ 从urmaComm_获取rank信息 ============
    epRankId_ = urmaComm_.GetEpRankId();

    // ============ 从tiling获取配置参数 ============
    totalWinSizeEp_ = tilingData->total_win_size;
    totalUbSize_ = static_cast<uint32_t>(tilingData->total_ub_size);
    axisBS_ = tilingData->max_bs;            // token数量
    axisH_ = tilingData->data_size;          // hidden size
    epWorldSize_ = tilingData->world_size;   // rank总数
    globalBS_ = tilingData->max_bs * epWorldSize_;
    aivNum_ = tilingData->aiv_num;           // AIV核总数
    mode_ = tilingData->mode;                // 测试模式
    dstRankIdAttr_ = tilingData->dst_rank_id; // P2P目标rank ID

    // ============ 初始化窗口状态(dataState_) ============
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    uint32_t epRankIdHccl = epRankId_;
    uint32_t epWorldSizeHccl = epWorldSize_;
    GM_ADDR statusDataSpaceGm_ = (GM_ADDR)(urmaComm_.GetHcclContext()->windowsOut[epRankIdHccl]);
    GlobalTensor<uint32_t> selfDataStatusGMTensor;
    selfDataStatusGMTensor.SetGlobalBuffer((__gm__ uint32_t*)(statusDataSpaceGm_ + DISPATCH_STATE_WIN_OFFSET
                                                    + aivId_ * WIN_ADDR_ALIGN));
    dataState_ = InitWinState(selfDataStatusGMTensor, epRankIdHccl, epWorldSizeHccl,
                                                    epRankId_, epWorldSize_, epWorldSize_, globalBS_, dataStateBuf);

    // 将双缓冲状态传递给SQE通信类，用于4参数P2pSend自动计算notifyAddr
    urmaComm_.SetNotifyDataState(dataState_);

    // ============ 设置全局张量 ============
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    yGMTensor_.SetGlobalBuffer((__gm__ YType*)y);
    receiveCntGMTensor_.SetGlobalBuffer((__gm__ int32_t*)receiveCnt);

    yGM_ = y;
    receiveCntGM_ = receiveCnt;

    // ============ 计算对齐参数 ============
    hOutSize_ = axisH_ * sizeof(XType);                                      // 原始hidden size
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;         // UB 32字节对齐
    hAlignWinSize_ = Ceil(hAlignSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;     // 窗口512字节对齐
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(XType);                          // 窗口对齐元素数
    hOutAlignUbSize_ = hAlignSize_;                                          // UB对齐hidden size

    // 每个rank在窗口中的空间大小 = global_bs/world_size * 对齐后的hidden size
    uint32_t axisMaxBS = globalBS_ / epWorldSize_;
    expertPerSizeOnWin_ = axisMaxBS * hAlignWinSize_;

    // ============ 计算窗口数据区偏移(双缓冲选择) ============
    winDataSizeOffset_ = dataState_ * (totalWinSizeEp_ / 2);
    // 本地窗口(接收数据区): 本rank的HCCL buffer数据区地址，其他rank通过URMA写入此处
    windowGM_ = urmaComm_.GetRemoteWinAddr(epRankId_, winDataSizeOffset_);

    // ============ 设置状态区地址 ============
    // 状态区: 本rank的HCCL buffer状态区地址，用于WaitDispatch轮询
    statusSpaceGm_ = urmaComm_.GetRemoteWinStateAddr(epRankId_, dataState_);
    windowInStatusFp32Tensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGm_));

    // ============ 计算状态等待的任务分发到各AIV核 ============
    // 将 world_size 个rank的状态等待任务分发到各AIV核
    stateOffset_ = 32UL;
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
    uint32_t statusBufCntAlign = Ceil(Ceil(epWorldSize_, aivNum_), 8) * 8;
    uint32_t statusBufSize = statusBufCntAlign * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
    statusTensor_ = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign * 8);

    // 初始化状态为 1.0f (0x3F800000)，用于状态同步标志
    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = { 0x01010101010101, 0 };
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, Ceil(statusBufCntAlign, 8), 1, 8);

    // ============ 初始化dstRankIds缓冲区 ============
    uint32_t dstRankIdsSize = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t dstRankIdsBufSize = dstRankIdsSize > bsAlign256 ? dstRankIdsSize : bsAlign256;
    tpipe_->InitBuffer(dstRankIdsBuf_, dstRankIdsBufSize);
    totalUsedUB_ += dstRankIdsBufSize;
    dstRankIdsTensor_ = dstRankIdsBuf_.Get<int32_t>();

    // ============ 初始化中间计算缓冲区 ============
    uint32_t hFp32Size = axisH_ * sizeof(float);
    uint32_t maxSize = dstRankIdsSize;
    maxSize = hFp32Size > dstRankIdsSize ? hFp32Size : dstRankIdsSize;
    tpipe_->InitBuffer(gatherMaskOutBuf_, maxSize);
    totalUsedUB_ += maxSize;
    workLocalTensor_ = gatherMaskOutBuf_.Get<float>();

    tpipe_->InitBuffer(dstExpBuf_, maxSize);
    totalUsedUB_ += maxSize;
    tpipe_->InitBuffer(subExpBuf_, maxSize);
    totalUsedUB_ += maxSize;

    // ============ 初始化数据队列 ============
    // 根据UB使用量决定双缓冲或单缓冲
    uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
    bufferNum_ = tmpTotalUB > MAX_UB_SIZE_P2P ? BUFFER_SINGLE : BUFFER_NUM;
    tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_);

    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();
}

// ============================================================================
// SplitToCore - 将任务分配到各AIV核
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    // 将curSendCnt个任务均匀分配到curUseAivNum个AIV核
    sendTokenNum = curSendCnt / curUseAivNum;                 // 每个aiv处理的任务数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;   // 余数
    uint32_t newAivId = aivId_;
    startTokenId = sendTokenNum * newAivId;
    if (newAivId < remainderTokenNum) {       // 前remainderRankNum个aiv多发1个任务
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

// ============================================================================
// ProcessToken - 处理单个token: 从GM读取到UB，写入目标窗口
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::ProcessToken(
    GlobalTensor<XType>& outTokenGT, uint32_t tokenIndex)
{
    // 步骤1: 从输入GM读取token数据到UB
    // 步骤2: 将token数据从UB写入目标窗口(本地workspace或HCCL窗口)
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams hCommuCopyOutParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};

    // 从GM读取token到UB
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams, padParams);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 从UB写入目标窗口
    DataCopyPad(outTokenGT, xTmpTensor_, hCommuCopyOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    xQueue_.FreeTensor<XType>(xTmpTensor_);
}

// ============================================================================
// CalTokenSendExpertCnt - 计算当前token之前发送到同一rank的token数量
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::CalTokenSendExpertCnt(
    uint32_t dstRankId, int32_t calCnt, int32_t &curExpertCnt)
{
    // 计算在calCnt个token中，有多少token的dstRankId等于给定的dstRankId
    // curExpertCnt = calCnt - 等于dstRankId的数量(即不同rank的数量)
    // 用于确定当前token在目标rank窗口中的偏移
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(dstExpIdTensor_, dstRankId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor_, dstRankIdsTensor_, dstExpIdTensor_, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor_, dstExpIdTensor_, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor_(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

// ============================================================================
// OverheadTest - mode=0: 仅本地DataCopy，测量算子调用固定开销
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::OverheadTest()
{
    // 仅做本地DataCopy从x到y，不做URMA通信
    // 目的: 测量算子调用的固定开销(数据搬运+调度开销)
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(axisBS_, aivNum_, startTokenId, endTokenId, sendTokenNum);

    if (startTokenId >= axisBS_) { return; }

    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams hCommuCopyOutParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};

    // 遍历分配的token，将x[tokenIndex]复制到y[tokenIndex]
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        // 从GM读取token到UB
        xTmpTensor_ = xQueue_.AllocTensor<XType>();
        DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams, padParams);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 从UB写入输出y(本地DataCopy，测量搬运开销)
        GlobalTensor<XType> yOutGT;
        yOutGT.SetGlobalBuffer((__gm__ XType*)(yGM_ + tokenIndex * hOutSize_));
        DataCopyPad(yOutGT, xTmpTensor_, hCommuCopyOutParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        xQueue_.FreeTensor<XType>(xTmpTensor_);
    }
}

// ============================================================================
// LatencyTest - mode=1: SQE级P2P发送小数据包，逐token Drain测量时延
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::LatencyTest()
{
    // SQE级P2P发送小数据包(1个token)到dst_rank_id
    // 每个token发送后立即Drain，精确测量单个token时延
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(axisBS_, aivNum_, startTokenId, endTokenId, sendTokenNum);

    if (startTokenId >= axisBS_) { return; }

    // 遍历token，逐个SQE级P2P发送
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        int32_t todDstRankId = dstRankIdsTensor_(tokenIndex);

        if (todDstRankId != epRankId_) {
            // 初始化与目标rank的SQE通信状态
            urmaComm_.InitRankComm(todDstRankId);

            // 计算curExpertCnt: 当前token之前发送到同一rank的token数量
            int32_t curExpertCnt = 0;
            if (tokenIndex > 0) {
                CalTokenSendExpertCnt(todDstRankId, tokenIndex, curExpertCnt);
            }

            // 计算本地窗口地址(localAddr)和远端窗口地址(remoteAddr)
            // 本地窗口: workspace中dstRankId对应的数据区 + 本rank偏移 + token偏移
            GM_ADDR localAddr = urmaComm_.GetLocalWinAddr(todDstRankId, totalWinSizeEp_, winDataSizeOffset_)
                              + epRankId_ * expertPerSizeOnWin_ + curExpertCnt * hAlignWinSize_;
            // 远端窗口: dstRankId的HCCL buffer数据区 + 本rank偏移 + token偏移
            GM_ADDR remoteAddr = urmaComm_.GetRemoteWinAddr(todDstRankId, winDataSizeOffset_)
                              + epRankId_ * expertPerSizeOnWin_ + curExpertCnt * hAlignWinSize_;

            // 将token数据写入本地窗口(ProcessToken方式)
            GlobalTensor<XType> outTokenGT;
            outTokenGT.SetGlobalBuffer((__gm__ XType*)localAddr);
            ProcessToken(outTokenGT, tokenIndex);

            // SQE级P2P发送到远端(WriteWithNotify SQE)
            urmaComm_.P2pSend(todDstRankId, reinterpret_cast<uint64_t>(localAddr), reinterpret_cast<uint64_t>(remoteAddr), hAlignWinSize_);
            // 每个token发送后立即Drain(精确测量单个token时延)
            urmaComm_.P2pSendDrain(todDstRankId);
        }
    }
}

// ============================================================================
// BandwidthTest - mode=2: SQE级P2P批量发送，统一Drain最大化带宽
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::BandwidthTest()
{
    // SQE级P2P发送完整数据块到dst_rank_id
    // 所有token的WriteWithNotify SQE完成后统一Drain，最大化带宽利用率
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(axisBS_, aivNum_, startTokenId, endTokenId, sendTokenNum);

    if (startTokenId >= axisBS_) { return; }

    // 遍历token，逐个SQE级P2P发送(仅WriteWithNotify SQE，不Drain)
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        int32_t todDstRankId = dstRankIdsTensor_(tokenIndex);

        if (todDstRankId != epRankId_) {
            // 初始化与目标rank的SQE通信状态
            urmaComm_.InitRankComm(todDstRankId);

            // 计算curExpertCnt: 当前token之前发送到同一rank的token数量
            int32_t curExpertCnt = 0;
            if (tokenIndex > 0) {
                CalTokenSendExpertCnt(todDstRankId, tokenIndex, curExpertCnt);
            }

            // 计算本地窗口地址和远端窗口地址
            GM_ADDR localAddr = urmaComm_.GetLocalWinAddr(todDstRankId, totalWinSizeEp_, winDataSizeOffset_)
                              + epRankId_ * expertPerSizeOnWin_ + curExpertCnt * hAlignWinSize_;
            GM_ADDR remoteAddr = urmaComm_.GetRemoteWinAddr(todDstRankId, winDataSizeOffset_)
                              + epRankId_ * expertPerSizeOnWin_ + curExpertCnt * hAlignWinSize_;

            // 将token数据写入本地窗口
            GlobalTensor<XType> outTokenGT;
            outTokenGT.SetGlobalBuffer((__gm__ XType*)localAddr);
            ProcessToken(outTokenGT, tokenIndex);

            // SQE级P2P发送到远端(仅WriteWithNotify SQE，不Drain)
            urmaComm_.P2pSend(todDstRankId, reinterpret_cast<uint64_t>(localAddr), reinterpret_cast<uint64_t>(remoteAddr), hAlignWinSize_);
        }
    }

    // 所有token发送完毕后，统一Drain等待全部完成(最大化带宽利用率)
    // Drain应在所有WriteWithNotify SQE之后统一调用
    for (int32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        if (rankId != epRankId_) {
            urmaComm_.InitRankComm(rankId);
            urmaComm_.P2pSendDrain(rankId);
        }
    }
}

// ============================================================================
// SetStatus - 设置状态: 统计发送到各rank的token数，通过SQE级P2pSend通知对端
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::SetStatus()
{
    // 状态区结构: [flag(1.0f), count, rank_id, world_size, ...] (每块8个int32 = 32B)
    // ============ 计算当前核负责的rank范围 ============
    uint32_t startRankId, endRankId, sendRankNum;
    SplitToCore(epWorldSize_, aivNum_, startRankId, endRankId, sendRankNum);

    // ============ 统计发送到各rank的token数量 ============
    for (uint32_t curRankId = startRankId; curRankId < endRankId; ++curRankId) {
        int32_t curRankCnt = 0;
        int32_t cntPosIndex = (curRankId - startRankId) * 8 + 1; // token数量存储在位置1
        CalTokenSendExpertCnt(curRankId, axisBS_, curRankCnt);
        statusTensor_(cntPosIndex) = curRankCnt;
    }

    PipeBarrier<PIPE_ALL>();
    SyncAll<true>(); // 等待所有核完成统计
    if (startRankId >= epWorldSize_) { return; }

    // ============ 写入状态到本地workspace和本rank HCCL buffer ============
    // 先将状态数据写入本地workspace状态区(作为SQE级P2pSend的源地址)
    // 对本rank则直接写入HCCL buffer状态区(无需SQE发送)
    GlobalTensor<int32_t> rankGMTensor;
    uint32_t offset = stateOffset_ * epRankId_; // 当前rank在目标状态区的偏移

    for (uint32_t rankIndex = startRankId; rankIndex < endRankId; ++rankIndex) {
        uint32_t toDstRankId = rankIndex;

        if (toDstRankId == epRankId_) {
            // 本rank: 直接写入本rank的HCCL buffer状态区
            GM_ADDR selfStatusGM = urmaComm_.GetRemoteWinStateAddr(toDstRankId, dataState_) + offset;
            rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)selfStatusGM);
            DataCopy<int32_t>(rankGMTensor, statusTensor_[(rankIndex - startRankId) * 8], 8UL);
        } else {
            // 其他rank: 写入本地workspace状态区(作为SQE级源地址)
            GM_ADDR localStatusGM = urmaComm_.GetLocalWinStateAddr(toDstRankId, totalWinSizeEp_, dataState_) + offset;
            rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)localStatusGM);
            DataCopy<int32_t>(rankGMTensor, statusTensor_[(rankIndex - startRankId) * 8], 8UL);
        }
    }
    PipeBarrier<PIPE_ALL>();

    // ============ SQE级P2pSend发送状态到远端rank ============
    // 将本地workspace中的状态数据SQE级P2pSend到各远端rank的HCCL buffer状态区
    for (uint32_t rankIndex = startRankId; rankIndex < endRankId; ++rankIndex) {
        uint32_t toDstRankId = rankIndex;
        if (toDstRankId == epRankId_) { continue; } // 本rank已直接写入，无需SQE发送

        // 初始化与目标rank的SQE通信状态
        urmaComm_.InitRankComm(toDstRankId);

        GM_ADDR localStatusGM = urmaComm_.GetLocalWinStateAddr(toDstRankId, totalWinSizeEp_, dataState_) + offset;
        GM_ADDR remoteStatusGM = urmaComm_.GetRemoteWinStateAddr(toDstRankId, dataState_) + offset;
        urmaComm_.P2pSend(toDstRankId, reinterpret_cast<uint64_t>(localStatusGM), reinterpret_cast<uint64_t>(remoteStatusGM), UB_ALIGN);
    }

    // ============ 等待所有SQE级P2pSend完成 ============
    for (uint32_t rankIndex = startRankId; rankIndex < endRankId; ++rankIndex) {
        uint32_t toDstRankId = rankIndex;
        if (toDstRankId != epRankId_) {
            urmaComm_.P2pSendDrain(toDstRankId);
        }
    }
}

// ============================================================================
// BufferInit - 重新初始化缓冲区(用于WaitDispatch阶段)
// 两阶段UB分配: Phase2 - tpipe_->Reset()后分配SQE通信缓冲区和WaitDispatch缓冲区
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::BufferInit()
{
    // WaitDispatch阶段需要重新分配UB缓冲区
    tpipe_->Reset();
    totalUsedUB_ = 0U;

    // 分配SQE级通信所需的UB缓冲区
    urmaComm_.InitCommBuffer();

    // 等待状态缓冲区大小
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256)
                                   ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    totalUsedUB_ += waitStatusBufSize;

    // 中间计算缓冲区大小
    uint64_t recvWinBlockNumSpace = epWorldSize_ * sizeof(float);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t gatherMaskOutSize = (recStatusNumPerCoreSpace > recvWinBlockNumSpace)
                                  ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);
    totalUsedUB_ += gatherMaskOutSize;

    // 标量缓冲区
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2);
    totalUsedUB_ += UB_ALIGN * 2;
}

// ============================================================================
// WaitDispatch - 等待所有rank完成数据发送，轮询状态区flag
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::WaitDispatch()
{
    // 等待所有rank完成数据发送
    // 通过轮询本rank的HCCL buffer状态区，等待所有rank的状态flag变为1.0f
    BufferInit();
    uint32_t startRankId = startStatusIndex_;
    uint32_t endRankId = startRankId + recStatusNumPerCore_;

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
        DataCopy(statusFp32Tensor_,
                 windowInStatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)], intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        // 计算所有flag的和
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutBuf_.Get<float>(),
                  1, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    // ============ 清零状态区，为下一次通信做准备 ============
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = { 0x01010101010101, 0 };
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInStatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    SyncAll<true>(); // 等待所有核完成
}

// ============================================================================
// LocalWindowCopy - 从本地窗口读取接收数据到输出y
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::LocalWindowCopy()
{
    // 从本rank的HCCL buffer数据区读取其他rank通过URMA写入的接收数据，拷贝到输出y
    uint32_t startRankId = startStatusIndex_;
    uint32_t endRankId = startRankId + recStatusNumPerCore_;

    if (startRankId >= rscvStatusNum_) { return; }

    // ============ 读取状态数据 ============
    TBuf<> localStatusBuf;
    // 状态区包含 world_size 个状态块，每个块 8 个 int32 (32B)
    const uint32_t statusCopySize = Ceil(epWorldSize_ * 8 * sizeof(int32_t), UB_ALIGN);
    tpipe_->InitBuffer(localStatusBuf, statusCopySize);
    LocalTensor<int32_t> localStatusTensor = localStatusBuf.Get<int32_t>();

    // 从窗口状态区读取所有状态
    GlobalTensor<int32_t> srcStatus = windowInStatusFp32Tensor_.ReinterpretCast<int32_t>();
    DataCopyParams statusCopyParams{static_cast<uint16_t>(epWorldSize_), 8, 0, 0};
    DataCopy(localStatusTensor, srcStatus, statusCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 计算前缀和: 当前核之前所有核收到的token总数
    uint32_t preCnt = 0;
    for (uint32_t i = 0; i < startRankId; i++) {
        uint32_t value = localStatusTensor(i * 8 + 1);
        preCnt += value; // token数量存储在位置1
    }

    // 设置各rank的receiveCnt
    for (uint32_t idx = 0; idx < epWorldSize_; idx++) {
        uint32_t receiveCount = localStatusTensor(idx * 8 + 1);
        receiveCntGMTensor_.SetValue(idx, receiveCount);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    // ============ 初始化数据拷贝队列 ============
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tripleQueue;
    uint32_t leftUBSize = ((totalUbSize_ - totalUsedUB_) / UB_ALIGN) * UB_ALIGN;
    uint32_t dataSize = leftUBSize / BUFFER_NUM;
    tpipe_->InitBuffer(tripleQueue, BUFFER_NUM, dataSize);

    uint32_t countPerDataSize = dataSize / hOutAlignUbSize_; // 每次拷贝的token数量
    uint32_t beginIdx = preCnt;

    // ============ 遍历每个rank，从窗口拷贝数据到输出 ============
    for (uint32_t index = startRankId; index < endRankId; index++) {
        uint32_t count = localStatusTensor(index * 8 + 1); // 从该rank收到的token数量

        // 计算窗口中该rank数据的起始地址
        GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + index * expertPerSizeOnWin_;
        GlobalTensor<YType> tokGlobal, yOutGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        // 分批拷贝数据
        uint32_t loopTimes = Ceil(count, countPerDataSize);
        uint32_t lastTimeCount = count - countPerDataSize * (loopTimes - 1);
        for (uint32_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
            uint32_t tokenCopiedNumThisExpert = countPerDataSize * loopIdx;
            tokGlobal.SetGlobalBuffer((__gm__ YType *)(wAddr + tokenCopiedNumThisExpert * hAlignWinSize_));

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
            yOutGlobal.SetGlobalBuffer((__gm__ YType *)(yGM_ + (beginIdx + tokenCopiedNumThisExpert) * hOutSize_));
            uint32_t srcStrideY = (hOutAlignUbSize_ - hOutSize_) / UB_ALIGN;
            DataCopyExtParams dataCopyParamsY{blockCount, hOutSize_, srcStrideY, 0U, 0U};
            DataCopyPad(yOutGlobal, tripleLocal, dataCopyParamsY);
            tripleQueue.FreeTensor(tripleLocal);
        }
        beginIdx += count;
    }
}

// ============================================================================
// Process - 主处理函数，根据mode_执行不同测试逻辑
// 两阶段UB分配:
//   Phase1 (Init): 分配计算缓冲区(statusBuf, dstRankIdsBuf, gatherMaskOutBuf等)
//   Phase2 (通信前): tpipe_->Reset() + urmaComm_.InitCommBuffer() + 通信缓冲区
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::Process()
{
    // 主处理函数
    // mode=0: OverheadTest + WaitDispatch + LocalWindowCopy (测量固定开销)
    // mode=1: LatencyTest + SetStatus + WaitDispatch + LocalWindowCopy (测量P2P时延)
    // mode=2: BandwidthTest + SetStatus + WaitDispatch + LocalWindowCopy (测量P2P带宽)
    if ASCEND_IS_AIV {
        // 所有AIV核填充dstRankIds数据： 使用属性dstRankIdAttr_填充整个数组
        // 如果dstRankIdAttr_==-1，表示自动模式(发送到(rank+1)%world_size)
        int32_t effectiveDstRankId = dstRankIdAttr_;
        if (effectiveDstRankId == -1) {
            effectiveDstRankId = (epRankId_ + 1) % static_cast<int32_t>(epWorldSize_);
        }
        Duplicate<int32_t>(dstRankIdsTensor_, effectiveDstRankId, axisBS_);
        PipeBarrier<PIPE_V>();

        if (mode_ == 0) {
            OverheadTest();     // 仅本地DataCopy，测量固定开销
            SetStatus();        // 发送状态通知对端（overhead模式不做URMA数据通信，但需要SetStatus让WaitDispatch能完成）
            // Phase2: Reset + InitCommBuffer准备SQE缓冲区
            tpipe_->Reset();
            urmaComm_.InitCommBuffer();
            WaitDispatch();     // 等待所有rank同步
            LocalWindowCopy();  // 从窗口读取数据到输出(overhead模式下receiveCnt为0)
        } else if (mode_ == 1) {
            LatencyTest();      // SQE级P2P小包发送，逐token Drain测量时延
            SetStatus();        // SQE级P2pSend状态通知对端
            // Phase2: Reset + InitCommBuffer准备SQE缓冲区(用于WaitDispatch阶段)
            tpipe_->Reset();
            urmaComm_.InitCommBuffer();
            WaitDispatch();     // 等待所有rank完成发送
            LocalWindowCopy();  // 从窗口读取接收数据到输出y
        } else if (mode_ == 2) {
            BandwidthTest();    // SQE级P2P批量发送，统一Drain最大化带宽
            SetStatus();        // SQE级P2pSend状态通知对端
            // Phase2: Reset + InitCommBuffer准备SQE缓冲区(用于WaitDispatch阶段)
            tpipe_->Reset();
            urmaComm_.InitCommBuffer();
            WaitDispatch();     // 等待所有rank完成发送
            LocalWindowCopy();  // 从窗口读取接收数据到输出y
        }
    }
}

} // namespace UrmaP2pBenchmarkImpl
#endif // URMA_P2P_BENCHMARK_H