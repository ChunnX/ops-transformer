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

// ==================== 性能计时常量 ====================
// 9个时间戳slot，记录Process()中每个SQE级操作的cycle计数
// [0]  PROCESS_START       - Process()开始
// [1]  DUPLICATE_END       - Duplicate(dstRankIds)+PipeBarrier完成
// [2]  INIT_SQ_CQ_LOAD_END - InitRankComm步骤1-6完成（SQ/CQ加载+PICI+CQE初始化+CQ轮询）
// [3]  SQE_TEMPLATE_END    - InitRankComm步骤7-8完成（SQE模板生成+元数据更新）
// [4]  PROCESS_TOKEN_END   - ProcessToken完成（数据写入本地窗口）
// [5]  P2P_SEND_END         - P2pSend完成（SQE填写+写入环形缓冲区）
// [6]  DOORBELL_END        - TriggerDoorBell完成（SendJFSDoorBell完成）
// [7]  POLL_CQ_END         - PollCommCQUpdateSQCI完成（CQ轮询获得完成通知+sqCi更新）
// [8]  UPDATE_PICI_END     - UpdatePICI完成（PICI保存回HcclCombinOpParam）= TOTAL_END
// [9-11] PADDING           - DataCopy对齐填充
constexpr uint32_t TIMING_SLOT_PROCESS_START = 0;
constexpr uint32_t TIMING_SLOT_DUPLICATE_END = 1;
constexpr uint32_t TIMING_SLOT_INIT_SQ_CQ_LOAD_END = 2;
constexpr uint32_t TIMING_SLOT_SQE_TEMPLATE_END = 3;
constexpr uint32_t TIMING_SLOT_PROCESS_TOKEN_END = 4;
constexpr uint32_t TIMING_SLOT_P2P_SEND_END = 5;
constexpr uint32_t TIMING_SLOT_DOORBELL_END = 6;
constexpr uint32_t TIMING_SLOT_POLL_CQ_END = 7;
constexpr uint32_t TIMING_SLOT_UPDATE_PICI_END = 8;
constexpr uint32_t TIMING_SLOTS = 9;
// DataCopy<int64_t>要求count对齐到32字节边界: 32/sizeof(int64_t)=4，9不是4的倍数，补齐到12
constexpr uint32_t PERFORMANCE_DATA_COPY_COUNT = 12;
constexpr uint64_t CYCLES_PER_US = 50UL;  // 昇腾AIV系统时钟50MHz, 1us=50cycles, 1cycle=20ns

#define TemplateUrmaP2pBenchmarkTypeClass \
    typename XType, \
    typename YType
#define TemplateUrmaP2pBenchmarkTypeFunc XType, YType

template <TemplateUrmaP2pBenchmarkTypeClass>
class UrmaP2pBenchmark {
public:
    __aicore__ inline UrmaP2pBenchmark() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR receiveCnt,
        GM_ADDR performanceInfo, GM_ADDR workspaceGM, TPipe *pipe, const UrmaP2pBenchmarkTilingData *tilingData);
    __aicore__ inline void Process();

private:
    // ==================== 测试模式方法（双模式设计） ====================
    // 每个测试函数接受cycle引用参数，在内部正确位置打点:
    //   cycleInitRankCommEndRef:  InitRankComm完成后打点（SLOT[2]）
    //   cycleProcessTokenEndRef:  ProcessToken完成后打点（SLOT[3]）
    //   cycleP2pSendEndRef:       P2pSend完成后打点（SLOT[4]）
    //   cycleDoorBellEndRef:      TriggerDoorBell完成后打点（SLOT[5]）
    //   cycleDrainWaitEndRef:     DrainWaitCompletion完成后打点（SLOT[6]）
    __aicore__ inline void OverheadLatencyTest(int64_t cycleDuplicateEndFallback,
                                                int64_t& cycleSqCqLoadEndRef, int64_t& cycleSqeTemplateEndRef,
                                                int64_t& cycleProcessTokenEndRef, int64_t& cycleP2pSendEndRef,
                                                int64_t& cycleDoorBellEndRef, int64_t& cyclePollCqEndRef,
                                                int64_t& cycleUpdatePiciEndRef);
    __aicore__ inline void BandwidthTest(int64_t cycleDuplicateEndFallback,
                                          int64_t& cycleSqeTemplateEndRef, int64_t& cycleProcessTokenEndRef,
                                          int64_t& cycleP2pSendEndRef, int64_t& cycleDoorBellEndRef,
                                          int64_t& cyclePollCqEndRef, int64_t& cycleUpdatePiciEndRef);

    // ==================== 通信辅助方法 ====================
    __aicore__ inline void ProcessToken(GlobalTensor<XType>& outTokenGT, uint32_t tokenIndex);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstRankId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum);

    // ==================== TPipe ====================
    TPipe *tpipe_{nullptr};

    // ==================== 全局张量 ====================
    GlobalTensor<XType> xGMTensor_;              // 输入数据 x
    GlobalTensor<int32_t> receiveCntGMTensor_;   // 接收到的token计数
    GlobalTensor<int64_t> performanceInfoGMTensor_; // 性能计时信息输出tensor

    // ==================== 本地张量 ====================
    LocalTensor<XType> xTmpTensor_;              // 临时存储token数据
    LocalTensor<int32_t> dstRankIdsTensor_;      // 本地缓存的dst_rank_id数组
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;

    // ==================== 缓冲区 ====================
    TBuf<> dstRankIdsBuf_;      // dst_rank_id缓冲区
    TBuf<> gatherMaskOutBuf_;   // 中间计算缓冲区
    TBuf<> scalarBuf_;          // 标量缓冲区
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> performanceBuf_;     // 性能计时数据缓冲区（必须在Init中分配，避免Process中Reset导致cycle变量spill丢失）
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列

    // ==================== 全局内存地址 ====================
    GM_ADDR receiveCntGM_;     // 接收计数的GM地址
    GM_ADDR performanceInfoGM_; // 性能计时信息的GM地址

    // ==================== SQE级URMA通信 ====================
    P2pUrmaSqeComm urmaComm_;  // SQE级URMA P2P通信类实例

    // ==================== 基础参数 ====================
    uint32_t axisBS_{0};       // batch size (token数量)
    uint32_t axisH_{0};        // hidden size (特征维度)
    uint32_t aivNum_{0};       // AIV核总数
    uint32_t epWorldSize_{0};  // rank总数
    int32_t epRankId_{0};      // 当前rank ID
    int32_t mode_{0};          // 测试模式: 0=overhead+latency, 1=bandwidth
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
    uint32_t totalUbSize_{0};         // 总UB大小
    uint32_t totalUsedUB_{0};         // 已使用的UB大小
};

// ============================================================================
// Init - 初始化Kernel类（纯P2P流程，无SetStatus/WaitDispatch/LocalWindowCopy）
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR receiveCnt,
    GM_ADDR performanceInfo, GM_ADDR workspaceGM, TPipe *pipe, const UrmaP2pBenchmarkTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();

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
    aivNum_ = tilingData->aiv_num;           // AIV核总数
    mode_ = tilingData->mode;                // 测试模式
    dstRankIdAttr_ = tilingData->dst_rank_id; // P2P目标rank ID

    // ============ 设置全局张量 ============
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    receiveCntGMTensor_.SetGlobalBuffer((__gm__ int32_t*)receiveCnt);
    performanceInfoGMTensor_.SetGlobalBuffer((__gm__ int64_t*)performanceInfo);

    receiveCntGM_ = receiveCnt;
    performanceInfoGM_ = performanceInfo;

    // ============ 计算对齐参数 ============
    hOutSize_ = axisH_ * sizeof(XType);
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    hAlignWinSize_ = Ceil(hAlignSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(XType);
    hOutAlignUbSize_ = hAlignSize_;

    // 每个rank在窗口中的空间大小
    uint32_t axisMaxBS = axisBS_;
    expertPerSizeOnWin_ = axisMaxBS * hAlignWinSize_;

    // ============ 计算窗口数据区偏移 ============
    winDataSizeOffset_ = 0; // 纯P2P流程不做双缓冲状态切换，固定使用0号缓冲区

    // ============ 初始化dstRankIds缓冲区 ============
    uint32_t dstRankIdsSize = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t dstRankIdsBufSize = dstRankIdsSize > bsAlign256 ? dstRankIdsSize : bsAlign256;
    tpipe_->InitBuffer(dstRankIdsBuf_, dstRankIdsBufSize);
    totalUsedUB_ += dstRankIdsBufSize;
    dstRankIdsTensor_ = dstRankIdsBuf_.Get<int32_t>();

    // ============ 初始化中间计算缓冲区 ============
    uint32_t hFp32Size = axisH_ * sizeof(float);
    uint32_t maxSize;
    if (mode_ == 0) {
        maxSize = dstRankIdsSize;
    } else {
        maxSize = hFp32Size > dstRankIdsSize ? hFp32Size : dstRankIdsSize;
    }
    tpipe_->InitBuffer(gatherMaskOutBuf_, maxSize);
    totalUsedUB_ += maxSize;
    workLocalTensor_ = gatherMaskOutBuf_.Get<float>();

    tpipe_->InitBuffer(dstExpBuf_, maxSize);
    totalUsedUB_ += maxSize;
    tpipe_->InitBuffer(subExpBuf_, maxSize);
    totalUsedUB_ += maxSize;

    // ============ 分配性能计时数据缓冲区（必须在xQueue之前分配，确保remainingUB计算正确） ============
    // 使用PERFORMANCE_DATA_COPY_COUNT(12)而非TIMING_SLOTS(9)，确保DataCopy<int64_t>对齐到32字节
    uint32_t performanceBufSize = Ceil(PERFORMANCE_DATA_COPY_COUNT * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(performanceBuf_, performanceBufSize);
    totalUsedUB_ += performanceBufSize;

    // ============ 分配SQE级通信缓冲区（必须在xQueue之前分配，确保remainingUB计算正确） ============
    urmaComm_.InitCommBuffer();

    // ============ 初始化数据队列 ============
    uint32_t remainingUB = MAX_UB_SIZE_P2P - totalUsedUB_;
    uint32_t maxQueueBufSize = remainingUB / 2;
    uint32_t bufferNum;
    if (hOutAlignUbSize_ > maxQueueBufSize) {
        bufferNum = BUFFER_SINGLE;
        uint32_t queueBufSize = remainingUB > hOutAlignUbSize_ ? hOutAlignUbSize_ : remainingUB;
        tpipe_->InitBuffer(xQueue_, bufferNum, queueBufSize);
    } else {
        uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
        bufferNum = tmpTotalUB > MAX_UB_SIZE_P2P ? BUFFER_SINGLE : BUFFER_NUM;
        tpipe_->InitBuffer(xQueue_, bufferNum, hOutAlignUbSize_);
    }

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
// OverheadLatencyTest - mode=0: SQE级P2P发送1个token，精确测量每个操作耗时
// 只处理第1个token（避免多token顺序Drain导致AIV核超时）
// 数据大小通过h(每token宽度)控制，bs固定为1
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::OverheadLatencyTest(
    int64_t cycleDuplicateEndFallback,
    int64_t& cycleSqCqLoadEndRef, int64_t& cycleSqeTemplateEndRef,
    int64_t& cycleProcessTokenEndRef, int64_t& cycleP2pSendEndRef,
    int64_t& cycleDoorBellEndRef, int64_t& cyclePollCqEndRef,
    int64_t& cycleUpdatePiciEndRef)
{
    uint32_t tokenIndex = 0;
    int32_t todDstRankId = dstRankIdsTensor_(tokenIndex);

    if (todDstRankId != epRankId_) {
        urmaComm_.InitRankComm(todDstRankId, &cycleSqCqLoadEndRef, &cycleSqeTemplateEndRef);

        int32_t curExpertCnt = 0;
        GM_ADDR localAddr = urmaComm_.GetLocalWinAddr(todDstRankId, totalWinSizeEp_, winDataSizeOffset_)
                          + epRankId_ * expertPerSizeOnWin_ + curExpertCnt * hAlignWinSize_;
        GM_ADDR remoteAddr = urmaComm_.GetRemoteWinAddr(todDstRankId, winDataSizeOffset_)
                          + epRankId_ * expertPerSizeOnWin_ + curExpertCnt * hAlignWinSize_;

        GlobalTensor<XType> outTokenGT;
        outTokenGT.SetGlobalBuffer((__gm__ XType*)localAddr);
        ProcessToken(outTokenGT, tokenIndex);
        cycleProcessTokenEndRef = GetSystemCycle(); // SLOT[4]

        urmaComm_.P2pSend(todDstRankId, reinterpret_cast<uint64_t>(localAddr),
                          reinterpret_cast<uint64_t>(remoteAddr), hAlignWinSize_);
        cycleP2pSendEndRef = GetSystemCycle(); // SLOT[5]

        urmaComm_.TriggerDoorBell();
        cycleDoorBellEndRef = GetSystemCycle(); // SLOT[6]

        urmaComm_.DrainWaitCompletion(todDstRankId, &cyclePollCqEndRef, &cycleUpdatePiciEndRef);
    } else {
        // todDstRankId == epRankId_: 不做P2P通信，用DUPLICATE_END的时间作为fallback
        int64_t fallbackCycle = cycleDuplicateEndFallback;
        cycleSqCqLoadEndRef = fallbackCycle;
        cycleSqeTemplateEndRef = fallbackCycle;
        cycleProcessTokenEndRef = fallbackCycle;
        cycleP2pSendEndRef = fallbackCycle;
        cycleDoorBellEndRef = fallbackCycle;
        cyclePollCqEndRef = fallbackCycle;
        cycleUpdatePiciEndRef = fallbackCycle;
    }
}

// ============================================================================
// BandwidthTest - mode=1: SQE级P2P批量发送，统一Drain最大化带宽
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::BandwidthTest(
    int64_t cycleDuplicateEndFallback,
    int64_t& cycleSqeTemplateEndRef, int64_t& cycleProcessTokenEndRef,
    int64_t& cycleP2pSendEndRef, int64_t& cycleDoorBellEndRef,
    int64_t& cyclePollCqEndRef, int64_t& cycleUpdatePiciEndRef)
{
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(axisBS_, aivNum_, startTokenId, endTokenId, sendTokenNum);

    if (startTokenId >= axisBS_) { return; }

    bool firstSqeTemplateTimingRecorded = false;
    bool firstP2pSendTimingRecorded = false;
    bool firstProcessTokenTimingRecorded = false;
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        int32_t todDstRankId = dstRankIdsTensor_(tokenIndex);

        if (todDstRankId != epRankId_) {
            urmaComm_.InitRankComm(todDstRankId);

            if (!firstSqeTemplateTimingRecorded) {
                cycleSqeTemplateEndRef = GetSystemCycle(); // SLOT[3]
                firstSqeTemplateTimingRecorded = true;
            }

            int32_t curExpertCnt = 0;
            if (tokenIndex > 0) {
                CalTokenSendExpertCnt(todDstRankId, tokenIndex, curExpertCnt);
            }

            GM_ADDR localAddr = urmaComm_.GetLocalWinAddr(todDstRankId, totalWinSizeEp_, winDataSizeOffset_)
                              + epRankId_ * expertPerSizeOnWin_ + curExpertCnt * hAlignWinSize_;
            GM_ADDR remoteAddr = urmaComm_.GetRemoteWinAddr(todDstRankId, winDataSizeOffset_)
                              + epRankId_ * expertPerSizeOnWin_ + curExpertCnt * hAlignWinSize_;

            GlobalTensor<XType> outTokenGT;
            outTokenGT.SetGlobalBuffer((__gm__ XType*)localAddr);
            ProcessToken(outTokenGT, tokenIndex);

            if (!firstProcessTokenTimingRecorded) {
                cycleProcessTokenEndRef = GetSystemCycle(); // SLOT[4]
                firstProcessTokenTimingRecorded = true;
            }

            urmaComm_.P2pSend(todDstRankId, reinterpret_cast<uint64_t>(localAddr), reinterpret_cast<uint64_t>(remoteAddr), hAlignWinSize_);

            if (!firstP2pSendTimingRecorded) {
                cycleP2pSendEndRef = GetSystemCycle(); // SLOT[5]
                firstP2pSendTimingRecorded = true;
            }
        }
    }

    // SLOT[6] DoorBell, SLOT[7] PollCQ, SLOT[8] UpdatePICI: BandwidthTest中统一P2pSendDrain，不单独打点

    for (int32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        if (rankId != epRankId_) {
            urmaComm_.InitRankComm(rankId);
            urmaComm_.P2pSendDrain(rankId);
        }
    }

    // SLOT[8]: 统一Drain完成后打点
    if (firstSqeTemplateTimingRecorded) {
        cycleUpdatePiciEndRef = GetSystemCycle();
    } else {
        // 该AIV核没有执行P2P（todDstRankId == epRankId_），用fallback值
        int64_t fallbackCycle = cycleDuplicateEndFallback;
        cycleSqeTemplateEndRef = fallbackCycle;
        cycleProcessTokenEndRef = fallbackCycle;
        cycleP2pSendEndRef = fallbackCycle;
        cycleDoorBellEndRef = fallbackCycle;
        cyclePollCqEndRef = fallbackCycle;
        cycleUpdatePiciEndRef = fallbackCycle;
    }
}

// ============================================================================
// Process - 纯P2P发送流程
// ============================================================================
template <TemplateUrmaP2pBenchmarkTypeClass>
__aicore__ inline void UrmaP2pBenchmark<TemplateUrmaP2pBenchmarkTypeFunc>::Process()
{
    // 纯P2P发送流程：Duplicate → OverheadLatencyTest/BandwidthTest → 结束
    // 不包含SetStatus/WaitDispatch/LocalWindowCopy（纯P2P测量）
    if ASCEND_IS_AIV {
        int64_t cycleProcessStart = GetSystemCycle();
        int64_t cycleDuplicateEnd = 0;
        int64_t cycleSqCqLoadEnd = 0;
        int64_t cycleSqeTemplateEnd = 0;
        int64_t cycleProcessTokenEnd = 0;
        int64_t cycleP2pSendEnd = 0;
        int64_t cycleDoorBellEnd = 0;
        int64_t cyclePollCqEnd = 0;
        int64_t cycleUpdatePiciEnd = 0;

        int32_t effectiveDstRankId = dstRankIdAttr_;
        if (effectiveDstRankId == -1) {
            effectiveDstRankId = (epRankId_ + 1) % static_cast<int32_t>(epWorldSize_);
        }
        Duplicate<int32_t>(dstRankIdsTensor_, effectiveDstRankId, axisBS_);
        PipeBarrier<PIPE_V>();
        cycleDuplicateEnd = GetSystemCycle(); // SLOT[1]

        if (mode_ == 0) {
            OverheadLatencyTest(cycleDuplicateEnd, cycleSqCqLoadEnd, cycleSqeTemplateEnd,
                                cycleProcessTokenEnd, cycleP2pSendEnd, cycleDoorBellEnd,
                                cyclePollCqEnd, cycleUpdatePiciEnd);
        } else if (mode_ == 1) {
            BandwidthTest(cycleDuplicateEnd, cycleSqeTemplateEnd, cycleProcessTokenEnd,
                          cycleP2pSendEnd, cycleDoorBellEnd, cyclePollCqEnd, cycleUpdatePiciEnd);
        }

        // AIV core 0 写入性能计时数据到performanceInfo输出tensor
        if (aivId_ == 0) {
            LocalTensor<int64_t> performanceLocal = performanceBuf_.Get<int64_t>();
            performanceLocal.SetValue(TIMING_SLOT_PROCESS_START, cycleProcessStart);
            performanceLocal.SetValue(TIMING_SLOT_DUPLICATE_END, cycleDuplicateEnd);
            performanceLocal.SetValue(TIMING_SLOT_INIT_SQ_CQ_LOAD_END, cycleSqCqLoadEnd);
            performanceLocal.SetValue(TIMING_SLOT_SQE_TEMPLATE_END, cycleSqeTemplateEnd);
            performanceLocal.SetValue(TIMING_SLOT_PROCESS_TOKEN_END, cycleProcessTokenEnd);
            performanceLocal.SetValue(TIMING_SLOT_P2P_SEND_END, cycleP2pSendEnd);
            performanceLocal.SetValue(TIMING_SLOT_DOORBELL_END, cycleDoorBellEnd);
            performanceLocal.SetValue(TIMING_SLOT_POLL_CQ_END, cyclePollCqEnd);
            performanceLocal.SetValue(TIMING_SLOT_UPDATE_PICI_END, cycleUpdatePiciEnd);
            PipeBarrier<PIPE_ALL>();
            DataCopy<int64_t>(performanceInfoGMTensor_, performanceLocal, PERFORMANCE_DATA_COPY_COUNT);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
    }
}

} // namespace UrmaP2pBenchmarkImpl
#endif // URMA_P2P_BENCHMARK_H