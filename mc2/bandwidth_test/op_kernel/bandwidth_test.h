/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bandwidth_test.h
 * \brief Bandwidth test kernel - MOE AlltoAll Dispatch 通信带宽测试算子
 * 
 * 算子功能说明：
 * 本算子实现了 MOE (Mixture of Experts) 场景下的 AlltoAll Dispatch 通信，
 * 用于测试 NPU 设备间的通信带宽。每个 rank 根据 expert_id 将 token 发送到
 * 对应的目标 rank，实现数据的分布式分发。
 * 
 * 执行流程 (mode=0 完整流程):
 * 1. SendToMoeExpert: 根据 expert_id 将 token 写入目标 rank 的通信窗口
 * 2. SetStatus: 统计发送到各 rank 的 token 数量，写入状态区通知对端
 * 3. WaitDispatch: 等待所有 rank 完成数据发送并写入状态
 * 4. LocalWindowCopy: 从本地窗口读取接收到的数据，拷贝到输出 tensor
 * 
 * 执行流程 (mode=1 仅发送):
 * 仅执行 SendToMoeExpert，用于单向带宽测试
 */

#ifndef BANDWIDTH_TEST_H
#define BANDWIDTH_TEST_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "bandwidth_test_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"

#if __has_include("../moe_distribute_dispatch_v2/moe_distribute_v2_base.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"
#endif

namespace BandwidthTestImpl {

using namespace AscendC;
using namespace Mc2Kernel;
using namespace Mc2Aclnn;

#define TemplateBandwidthTestTypeClass \
    typename XType, \
    typename YType
#define TemplateBandwidthTestTypeFunc XType, YType

template <TemplateBandwidthTestTypeClass>
class BandwidthTest {
public:
    __aicore__ inline BandwidthTest() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR dstrankId, GM_ADDR y, GM_ADDR receiveCnt,
        GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BufferInit();
    __aicore__ inline void ProcessToken(GlobalTensor<XType>& outTokenGT, uint32_t tokenIndex);
    
    // 获取指定 rank 的通信窗口数据区地址
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffset_;
    }

    // 获取指定 rank 的通信窗口状态区地址
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> contextGMTensor_;        // 输入context
    GlobalTensor<XType> xGMTensor_;               // 输入数据 x
    GlobalTensor<int32_t> dstrankIdGMTensor_;     // 目标rank ID (用于确定目标rank)
    GlobalTensor<YType> yGMTensor_;               // 输出数据 y
    GlobalTensor<int32_t> receiveCntGMTensor_;    // 接收到的token计数
    GlobalTensor<float> windowInStatusFp32Tensor_; // 窗口状态区(用于同步)
    LocalTensor<XType> xTmpTensor_;               // 临时存储token数据
    LocalTensor<YType> yTmpTensor_;               // 临时存储输出数据
    LocalTensor<int32_t> dstrankIdsTensor_;       // 本地缓存的dstrank_id数组
    LocalTensor<int32_t> statusTensor_;           // 状态数据
    LocalTensor<float> statusFp32Tensor_;         // 状态数据(FP32视图)
    TBuf<> dstrankIdsBuf_;     // dstrank_id 缓冲区
    TBuf<> statusBuf_;         // 状态缓冲区
    TBuf<> waitStatusBuf_;     // 等待状态缓冲区
    TBuf<> gatherMaskOutBuf_;  // 中间计算缓冲区
    TBuf<> scalarBuf_;         // 标量缓冲区
    TBuf<> tokenNumBuf_;       // token数量缓冲区
    TBuf<> sendCntBuf_;        // 发送计数缓冲区 (每个rank发送了多少token)
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列

    // ==================== 全局内存地址 ====================
    GM_ADDR yGM_;              // 输出y的GM地址
    GM_ADDR receiveCntGM_;     // 接收计数的GM地址
    GM_ADDR statusSpaceGm_;    // 状态空间的GM地址
    GM_ADDR windowGM_;         // 本地窗口的GM地址

    // ==================== HCCL通信上下文 ====================
    __gm__ Mc2MoeContext* mc2Context_{nullptr};

    uint32_t axisBS_{0};       // batch size (token数量)
    uint32_t axisH_{0};        // hidden size (特征维度)
    uint32_t aivNum_{0};       // AIV核总数
    uint32_t epWorldSize_{0};  // rank总数
    int32_t epRankId_{0};      // 当前rank ID (从HCCL context获取)
    int32_t cclBufferSize_{0}; // CCL缓冲区大小
    int32_t mode_{0};          // 模式: 0=完整流程, 1=仅发送
    int32_t commAlg_{0};       // 通信算法类型
    uint32_t aivId_{0};        // 当前AIV核ID
    uint32_t globalBS_{0};         // 全局最大batch size
    uint32_t hOutSize_{0};         // 输出hidden size (字节)
    uint32_t hAlignWinSize_{0};    // 窗口对齐后的hidden size
    uint32_t hAlignWinCnt_{0};     // 窗口对齐后的元素个数
    uint32_t hOutAlignUbSize_{0};  // UB对齐后的hidden size
    uint32_t hAlignSize_{0};       // UB对齐后的hidden size
    uint32_t startExpertId_{0};    // 当前核处理的起始expert ID
    uint32_t endExpertId_{0};      // 当前核处理的结束expert ID
    uint32_t sendExpertNum_{0};    // 当前核处理的expert数量
    uint32_t totalCnt_{0};         // 总计处理的token数
    uint32_t dataState_{0};        // 数据状态 (用于双缓冲)
    uint32_t totalUsedUB_{0};      // 已使用的UB大小
    uint64_t winDataSizeOffset_{0}; // 窗口数据区偏移
    uint64_t expertPerSizeOnWin_{0}; // 每个expert在窗口中占用的空间
    uint32_t stateOffset_{0};      // 状态偏移 (32字节)
    uint32_t recStatusNumPerCore_{0}; // 每个核负责等待的rank数量
    uint32_t rscvStatusNum_{0};    // 需要等待的rank总数
    uint32_t remainderRankNum_{0}; // 分配后剩余的rank数量
    uint32_t startStatusIndex_{0}; // 起始状态索引
    uint32_t bufferNum_{0};        // 缓冲区数量
    uint32_t totalUbSize_{0};      // 总UB大小
    uint64_t totalWinSizeEp_{0};   // 总窗口大小
    float sumTarget_;              // 状态同步目标值 (1.0)
};

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR dstrankId, GM_ADDR y, GM_ADDR receiveCnt,
    GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();

    totalWinSizeEp_ = tilingData->total_win_size;
    mc2Context_ = (__gm__ Mc2MoeContext *)context;

    totalUbSize_ = static_cast<uint32_t>(tilingData->total_ub_size);
    axisBS_ = tilingData->max_bs;
    axisH_ = tilingData->data_size;
    epWorldSize_ = tilingData->world_size;
    cclBufferSize_ = tilingData->ccl_buffer_size;
    globalBS_ = tilingData->max_bs * epWorldSize_;
    aivNum_ = tilingData->aiv_num;
    mode_ = tilingData->mode;
    commAlg_ = tilingData->comm_alg;

    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    uint32_t epRankIdHccl{0};
    uint32_t epWorldSizeHccl{0};
    epRankIdHccl = mc2Context_->epRankId;
    statusSpaceGm_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankIdHccl]);
    epWorldSizeHccl = epWorldSize_;
    epRankId_ = static_cast<int32_t>(epRankIdHccl);
    GlobalTensor<uint32_t> selfDataStatusGMTensor;
    selfDataStatusGMTensor.SetGlobalBuffer((__gm__ uint32_t*)(statusSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    dataState_ = MoeDistributeV2Base::InitWinState(selfDataStatusGMTensor, epRankIdHccl, epWorldSizeHccl,
                                                    epRankId_, epWorldSize_, epWorldSize_, globalBS_, dataStateBuf);

    contextGMTensor_.SetGlobalBuffer((__gm__ int32_t*)context);
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    dstrankIdGMTensor_.SetGlobalBuffer((__gm__ int32_t*)dstrankId);
    yGMTensor_.SetGlobalBuffer((__gm__ YType*)y);
    receiveCntGMTensor_.SetGlobalBuffer((__gm__ int32_t*)receiveCnt);

    yGM_ = y;
    receiveCntGM_ = receiveCnt;

    // ============ 计算对齐参数 ============
    hOutSize_ = axisH_ * sizeof(XType);                             // 原始hidden size
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; // UB 32字节对齐
    hAlignWinSize_ = Ceil(hAlignSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // 窗口512字节对齐
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(XType);
    hOutAlignUbSize_ = hAlignSize_;

    // 每个rank在窗口中的空间大小 = global_bs/world_size * 对齐后的hidden size
    uint32_t axisMaxBS = globalBS_ / epWorldSize_;
    expertPerSizeOnWin_ = axisMaxBS * hAlignWinSize_;

    // ============ 计算状态等待的任务分发 ============
    // 将 world_size 个 rank 的状态等待任务分发到各 AIV 核
    stateOffset_ = 32UL;  // 32字节
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
    //DumpTensor(statusTensor_,4,8);
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankId_);

    // 初始化状态为 1.0f (0x3F800000)，用于状态同步
    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = { 0x101010101010101, 0 };
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign / 8, 1, 8);
    //DumpTensor(statusTensor_,5,8);

    // ============ 计算窗口数据区偏移 ============
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffset_ = dataState_ * (totalWinSizeEp_ / 2) + axisMaxBS * hSizeAlignCombine;
    windowGM_ = GetWindAddrByRankId(epRankId_);
    windowInStatusFp32Tensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGm_));

    // ============ 初始化其他缓冲区 ============
    uint32_t dstrankIdsSize = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t dstrankIdsBufSize = dstrankIdsSize > bsAlign256 ? dstrankIdsSize : bsAlign256;
    tpipe_->InitBuffer(dstrankIdsBuf_, dstrankIdsBufSize);
    totalUsedUB_ += dstrankIdsSize;
    dstrankIdsTensor_ = dstrankIdsBuf_.Get<int32_t>();

    uint32_t hFp32Size = axisH_ * sizeof(float);
    uint32_t maxSize = hFp32Size > dstrankIdsSize ? hFp32Size : dstrankIdsSize;
    maxSize = maxSize > bsAlign256 ? maxSize : bsAlign256;
    tpipe_->InitBuffer(gatherMaskOutBuf_, maxSize);
    totalUsedUB_ += maxSize;

    // 发送计数缓冲区，用于记录发送到每个rank的token数量
    tpipe_->InitBuffer(sendCntBuf_, Ceil(epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor<uint32_t> sendCntTensor = sendCntBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(sendCntTensor, 0, Ceil(epWorldSize_, UB_ALIGN / sizeof(uint32_t)) * 8);
    totalUsedUB_ += Ceil(epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;

    // ============ 初始化数据队列 ============
    uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
    bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
    tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_);

}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::ProcessToken(
    GlobalTensor<XType>& outTokenGT, uint32_t tokenIndex)
{
    // 处理单个token：
    // 1. 从输入GM读取token数据到UB
    // 2. 将token数据写入目标rank的窗口
    
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams hCommuCopyOutParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};

    // 从GM读取token到UB
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams, padParams);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();

    SyncFunc<AscendC::HardEvent::MTE3_S>();

    // 写入目标rank的窗口
    DataCopyPad(outTokenGT, xTmpTensor_, hCommuCopyOutParams);
    xQueue_.FreeTensor<XType>(xTmpTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SendToMoeExpert()
{
    // 发送token到目标rank的窗口
    // 根据 expert_id 确定目标 rank，将 token 写入该 rank 的通信窗口
    
    // ============ 计算当前核处理的token范围 ============
    uint32_t sendTokenNumPerCore = axisBS_ / aivNum_;
    uint32_t remainder = axisBS_ % aivNum_;
    uint32_t startTokenId = sendTokenNumPerCore * aivId_;
    if (aivId_ < remainder) {
        sendTokenNumPerCore += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainder;
    }
    uint32_t endTokenId = startTokenId + sendTokenNumPerCore;

    if (startTokenId >= axisBS_) { return; }

    // ============ 读取expert_id数组到UB ============
    GlobalTensor<XType> dstWinGMTensor;
    DataCopyExtParams dstrankIdsParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> dstrankIdsCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(dstrankIdsTensor_, dstrankIdGMTensor_, dstrankIdsParams, dstrankIdsCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // LocalTensor<uint32_t> sendCntTensor = sendCntBuf_.Get<uint32_t>();
    TBuf<> dstExpBuf_;
    tpipe_->InitBuffer(dstExpBuf_, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    TBuf<> subExpBuf_;
    tpipe_->InitBuffer(subExpBuf_, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> subExpIdTensor_ = subExpBuf_.Get<int32_t>();
    LocalTensor<float> workLocalTensor_ = gatherMaskOutBuf_.Get<float>();

    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    // ============ 遍历token并发送到目标rank ============
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t dstRankId = dstrankIdsTensor_(tokenIndex);
        uint32_t dstExpertId = dstRankId;
        int32_t curExpertCnt = 0;

        if (tokenIndex > 0) {
            SyncFunc<AscendC::HardEvent::S_V>();
            Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, tokenIndex);
            PipeBarrier<PIPE_V>();
            Sub(subExpIdTensor_, dstrankIdsTensor_, dstExpIdTensor_, tokenIndex);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
            LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
            Abs(tmpoutFp32, tmpFp32, tokenIndex);
            PipeBarrier<PIPE_V>();
            Mins(subExpIdTensor_, dstExpIdTensor_, 1, tokenIndex);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(tmpoutFp32, tmpoutFp32, workLocalTensor_, tokenIndex);
            SyncFunc<AscendC::HardEvent::V_S>();
            int32_t curOtherExpertCnt = dstExpIdTensor_(0);
            if (tokenIndex >= curOtherExpertCnt) {
                curExpertCnt = tokenIndex - curOtherExpertCnt;
            } else {
                curExpertCnt = 0;
            }
        }

        SyncFunc<AscendC::HardEvent::S_V>();

        // 计算目标窗口地址:
        // 目标rank窗口基地址 + 当前rank的偏移(每个rank发来的数据独立存储) + 当前token偏移
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindAddrByRankId(dstRankId) +
                                           epRankId_ * expertPerSizeOnWin_ +
                                           curExpertCnt * hAlignWinSize_);
        dstWinGMTensor.SetGlobalBuffer((__gm__ XType*)rankGM);
        GlobalTensor<XType> tempTensor = dstWinGMTensor[0];
        ProcessToken(tempTensor, tokenIndex);

        // 更新发送计数
        //sendCntTensor.SetValue(dstRankId, curSendCnt + 1);
        SyncFunc<AscendC::HardEvent::V_S>();
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SetStatus()
{
    // 设置状态：统计发送到各rank的token数量，并写入状态区通知对端
    // 状态区结构: [flag(1.0f), count, rank_id, world_size, ...]
    
    // ============ 计算当前核负责的rank范围 ============
    uint32_t sendExpertNum = epWorldSize_ / aivNum_;
    uint32_t remainder = epWorldSize_ % aivNum_;
    uint32_t startExpertId = sendExpertNum * aivId_;
    if (aivId_ < remainder) {
        sendExpertNum += 1;
        startExpertId += aivId_;
    } else {
        startExpertId += remainder;
    }
    uint32_t endExpertId = startExpertId + sendExpertNum;
    startExpertId_ = startExpertId;
    endExpertId_ = endExpertId;

    // ============ 分配临时缓冲区 ============
    TBuf<> dstExpBuf;
    tpipe_->InitBuffer(dstExpBuf, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf.Get<int32_t>();
    TBuf<> subExpBuf;
    tpipe_->InitBuffer(subExpBuf, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> subExpIdTensor = subExpBuf.Get<int32_t>();
    LocalTensor<float> workLocalTensor = gatherMaskOutBuf_.Get<float>();

    // ============ 读取dstrank_id数组 ============
    DataCopyExtParams dstrankIdsParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> dstrankIdsCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(dstrankIdsTensor_, dstrankIdGMTensor_, dstrankIdsParams, dstrankIdsCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    // ============ 统计发送到各rank的token数量 ============
    for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
        int32_t curExpertCnt = 0;
        SyncFunc<AscendC::HardEvent::S_V>();
        
        Duplicate<int32_t>(dstExpIdTensor, static_cast<int32_t>(curExpertId), axisBS_);
        PipeBarrier<PIPE_V>();
        Sub(subExpIdTensor, dstrankIdsTensor_, dstExpIdTensor, axisBS_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
        LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
        Abs(tmpoutFp32, tmpFp32, axisBS_);
        PipeBarrier<PIPE_V>();
        Mins(subExpIdTensor, dstExpIdTensor, 1, axisBS_);  // 结果为0或1
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor, axisBS_);
        SyncFunc<AscendC::HardEvent::V_S>();

        curExpertCnt = dstExpIdTensor(0);
        statusTensor_((curExpertId - startExpertId) * 8 + 1) = curExpertCnt; // 存储到位置1
        //DumpTensor(statusTensor_,2,8);
    }

    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();  // 等待所有核完成统计
    if (startExpertId >= epWorldSize_) { return; }

    // ============ 将状态写入目标rank的状态区 ============
    // 写入位置: 目标rank状态区 + 当前rank的偏移
    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t expertIndex = startExpertId; expertIndex < endExpertId; ++expertIndex) {
        uint32_t dstRankId = expertIndex;
        uint32_t offset = stateOffset_ * epRankId_;  // 当前rank在目标状态区的偏移
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(expertIndex - startExpertId) * 8], 8UL);
        //DumpTensor(statusTensor_[(expertIndex - startExpertId) * 8],2,8);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::BufferInit()
{
    // 重新初始化缓冲区，用于WaitDispatch阶段
    tpipe_->Reset();
    totalUsedUB_ = 0U;
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256) ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    totalUsedUB_ += waitStatusBufSize;

    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, recStatusNumPerCoreSpace);
    totalUsedUB_ += recStatusNumPerCoreSpace;

    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2);
    totalUsedUB_ += UB_ALIGN * 2;

    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_);
    totalUsedUB_ += BUFFER_NUM * hOutAlignUbSize_;

    tpipe_->InitBuffer(tokenNumBuf_, UB_ALIGN);
    totalUsedUB_ += UB_ALIGN;

    uint32_t statusBufSize = epWorldSize_ * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::WaitDispatch()
{
    BufferInit();
    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        SyncAll<true>();
        return;
    }

    uint32_t pollBufSize = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(waitStatusBuf_, pollBufSize);
    LocalTensor<float> pollFlags = waitStatusBuf_.Get<float>();
    LocalTensor<float> sumOut = scalarBuf_.Get<float>();
    float targetSum = sumTarget_ * recStatusNumPerCore_;
    float currentSum = -1.0f;

    // 起始 flag 的 GlobalTensor（从该偏移开始）
    uint32_t srcBaseOffset = startStatusIndex_ * stateOffset_ / sizeof(float);
    GlobalTensor<float> srcFlags = windowInStatusFp32Tensor_[srcBaseOffset];

    DataCopyParams flagParams;
    flagParams.blockCount = static_cast<uint16_t>(recStatusNumPerCore_);
    flagParams.blockLen = 1;
    flagParams.srcStride = (stateOffset_ / sizeof(float)) - 1;  // 相邻 flag 间隔 stride-1 个元素
    flagParams.dstStride = 0;

    while (currentSum != targetSum) {
        DataCopy(pollFlags, srcFlags, flagParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(sumOut, pollFlags, gatherMaskOutBuf_.Get<float>(), 1, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        currentSum = sumOut.GetValue(0);
    }

    // 清零 flag（复用 pollFlags 缓冲区存放零值）
    LocalTensor<int32_t> zeroBuf = waitStatusBuf_.Get<int32_t>();
    Duplicate<int32_t>(zeroBuf, 0, recStatusNumPerCore_);
    GlobalTensor<int32_t> dstFlagsZero = srcFlags.ReinterpretCast<int32_t>();
    DataCopyParams zeroParams = flagParams;
    DataCopy(dstFlagsZero, zeroBuf, zeroParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    SyncAll<true>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::LocalWindowCopy()
{
    if (startExpertId_ >= rscvStatusNum_) { return; }

    // 批量读取状态区（每个 rank 32 字节）到 statusTensor_
    uint32_t totalStates = epWorldSize_ * 8;  // 每个 rank 8 个 int32
    uint32_t statusCopyCount = Ceil(totalStates, UB_ALIGN / sizeof(int32_t)) * (UB_ALIGN / sizeof(int32_t));
    tpipe_->InitBuffer(statusBuf_, statusCopyCount * sizeof(int32_t));
    statusTensor_ = statusBuf_.Get<int32_t>();
    DataCopyParams statusParams{static_cast<uint16_t>(epWorldSize_), 8, 0, 0};
    GlobalTensor<int32_t> srcStatus = windowInStatusFp32Tensor_.ReinterpretCast<int32_t>();
    DataCopy(statusTensor_, srcStatus, statusParams);
    DumpTensor(statusTensor_, 6, 16);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 计算前缀和
    uint32_t preCnt = 0;
    for (uint32_t i = 0; i < startExpertId_; ++i) {
        preCnt += statusTensor_(i * 8 + 1);
    }

    // 数据拷贝队列
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tripleQueue;
    uint32_t leftUBSize = ((totalUbSize_ - totalUsedUB_) / UB_ALIGN) * UB_ALIGN;
    uint32_t dataSize = leftUBSize / BUFFER_NUM;
    tpipe_->InitBuffer(tripleQueue, BUFFER_NUM, dataSize);
    uint32_t countPerDataSize = dataSize / hOutAlignUbSize_;
    uint32_t beginIdx = preCnt;

    for (uint32_t idx = startExpertId_; idx < endExpertId_; ++idx) {
        uint32_t count = statusTensor_(idx * 8 + 1);
        if (count == 0) continue;

        GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + idx * expertPerSizeOnWin_;
        GlobalTensor<YType> srcGt, dstGt;
        srcGt.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        uint32_t loopTimes = Ceil(count, countPerDataSize);
        uint32_t lastCount = count - countPerDataSize * (loopTimes - 1);
        for (uint32_t loop = 0; loop < loopTimes; ++loop) {
            uint32_t copied = countPerDataSize * loop;
            srcGt.SetGlobalBuffer((__gm__ YType*)(wAddr + copied * hAlignWinSize_));

            uint16_t blockCnt = static_cast<uint16_t>((loop < loopTimes - 1) ? countPerDataSize : lastCount);
            uint16_t copyBlockLen = static_cast<uint16_t>(hOutAlignUbSize_ / UB_ALIGN);
            uint16_t srcStride = static_cast<uint16_t>((hAlignWinSize_ - hOutAlignUbSize_) / UB_ALIGN);
            DataCopyParams srcParams{blockCnt, copyBlockLen, srcStride, 0U};

            LocalTensor<YType> localBuf = tripleQueue.AllocTensor<YType>();
            DataCopy(localBuf, srcGt, srcParams);
            tripleQueue.EnQue(localBuf);
            localBuf = tripleQueue.DeQue<YType>();

            dstGt.SetGlobalBuffer((__gm__ YType*)(yGM_ + (beginIdx + copied) * hOutSize_));
            uint32_t dstStride = (hOutAlignUbSize_ - hOutSize_) / UB_ALIGN;
            DataCopyExtParams dstParams{blockCnt, static_cast<uint16_t>(hOutSize_), static_cast<uint16_t>(dstStride), 0U, 0U};
            DataCopyPad(dstGt, localBuf, dstParams);

            tripleQueue.FreeTensor(localBuf);
        }
        beginIdx += count;
    }
    totalCnt_ = beginIdx;

    // 写回接收计数
    DataCopyExtParams recvCntParams{static_cast<uint16_t>(sendExpertNum_), sizeof(int32_t), 0U, 0U, 0U};
    DataCopyPad(receiveCntGMTensor_[startExpertId_], statusTensor_[startExpertId_ * 8 + 1], recvCntParams);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Process()
{
    // 主处理函数
    // mode=0: 完整流程 (发送 + 同步 + 接收)
    // mode=1: 仅发送模式 (用于单向带宽测试)
    if ASCEND_IS_AIV {
        if (mode_ == 0) {
            SendToMoeExpert();  // 步骤1: 发送数据到各rank的窗口
            SetStatus(); // 步骤2: 设置状态通知对端
            WaitDispatch();      // 步骤3: 等待所有rank完成发送
            LocalWindowCopy(); // 步骤4: 从窗口读取数据到输出
        } else if (mode_ == 1) {
            SendToMoeExpert();  // 步骤1: 发送数据到各rank的窗口
            SetStatus(); // 步骤2: 设置状态通知对端
        }
    }
}

} // namespace BandwidthTestImpl
#endif // BANDWIDTH_TEST_H