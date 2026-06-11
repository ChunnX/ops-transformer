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
 * \file p2p_urma_comm.h
 * \brief SQE级URMA P2P通信实现
 *
 * 使用HcclCombinOpParam（通过GetHcclContext获取）作为通信上下文，
 * 基于moe_distribute_base.h中的SQE级接口实现P2P通信。
 * 通信流程：
 *   InitRankComm -> 加载SQ/CQ信息、获取PICI、准备SQE模板
 *   P2pSend -> 提交单个WriteWithNotify SQE（不敲DoorBell、不轮询CQ）
 *   P2pSendDrain -> 敲DoorBell + 轮询CQ + 更新PICI，冲刷所有待发送SQE
 */

#ifndef P2P_URMA_COMM_H
#define P2P_URMA_COMM_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#if __has_include("../moe_distribute_dispatch_setup/op_kernel/moe_distribute_base.h")
#include "../moe_distribute_dispatch_setup/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../moe_distribute_dispatch_setup/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace UrmaP2pBenchmarkImpl {

// 状态区大小常量，参考dispatch_setup中WIN_STATE_OFFSET和STATE_WIN_OFFSET的定义
constexpr uint64_t P2P_STATE_SIZE = 350UL * 1024UL;
constexpr uint64_t P2P_WIN_STATE_OFFSET = 350UL * 1024UL;
constexpr uint64_t P2P_STATE_WIN_OFFSET = 950UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t P2P_CQ_DEPTH_256 = 256U;
constexpr uint32_t P2P_UB_ALIGN = 32U;
constexpr uint32_t P2P_WRITE_WITH_NOTIFY_SQE_SIZE = 96U;
constexpr uint32_t P2P_STATE_OFFSET = 32U;             // 状态块大小(8个int32 = 32字节)
constexpr uint64_t P2P_NOTIFY_DATA_FLAG = 0x3F800000UL; // 1.0f标志值的整数表示

// ============ 状态区常量(内联自moe_distribute_v2_constant.h) ============
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint8_t BUFFER_NUM = 2;        // 双缓冲
constexpr uint8_t BUFFER_SINGLE = 1;     // 单缓冲
constexpr uint64_t OP_CNT_POSUL = 3UL;
constexpr uint32_t ZERONE_STATE_POS = 0U;
constexpr uint32_t OPOSITION_POS = 1U;
constexpr uint32_t TILING_EPRANKID_POS = 2U;
constexpr uint32_t MOE_NUM_POS = 3U;
constexpr uint32_t TILING_WORLDSIZE_POS = 4U;
constexpr uint32_t GLOBALBS_POS = 5U;
constexpr uint32_t HCCL_DFX_POS = 8U;
constexpr uint32_t HCCL_DFX_NUM = 2U;
constexpr uint32_t HCCL_EPRANKId_POS = 0U;
constexpr uint32_t HCCL_WORLDSIZE_POS = 1U;
constexpr uint64_t DISPATCH_STATE_WIN_OFFSET = 256UL * 1024UL;

using namespace AscendC;

// ============================================================================
// InitWinState - 内联自moe_distribute_v2_base.h
// 初始化窗口双缓冲状态，返回dataState(0或1)
// ============================================================================
__aicore__ inline uint32_t InitWinState(GlobalTensor<uint32_t> selfDataStatusGMTensor,
    uint32_t epRankIdHccl, uint32_t epWorldSizeHccl, uint32_t epRankIdOriginal,
    uint32_t moeExpertNum, uint32_t epWorldSizeOriginal, uint32_t globalBS, TBuf<> dataStateBuf)
{
    LocalTensor<uint64_t> dataStateLocalTensor64 = dataStateBuf.Get<uint64_t>();
    LocalTensor<uint32_t> dataStateLocalTensor = dataStateBuf.Get<uint32_t>();
    DataCopy(dataStateLocalTensor, selfDataStatusGMTensor, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t dataState = dataStateLocalTensor.GetValue(ZERONE_STATE_POS);
    dataStateLocalTensor.SetValue(ZERONE_STATE_POS, dataState == 0 ? 1 : 0);
    dataStateLocalTensor.SetValue(OPOSITION_POS, 1);
    dataStateLocalTensor.SetValue(TILING_EPRANKID_POS, epRankIdOriginal);
    dataStateLocalTensor.SetValue(MOE_NUM_POS, moeExpertNum);
    dataStateLocalTensor.SetValue(TILING_WORLDSIZE_POS, epWorldSizeOriginal);
    dataStateLocalTensor.SetValue(GLOBALBS_POS, globalBS);
    uint32_t opCnt = dataStateLocalTensor64.GetValue(OP_CNT_POSUL);
    opCnt = opCnt + 1;
    dataStateLocalTensor64.SetValue(OP_CNT_POSUL, opCnt);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfDataStatusGMTensor, dataStateLocalTensor, UB_ALIGN / sizeof(uint32_t));

    if ((epRankIdOriginal != epRankIdHccl) || (epWorldSizeOriginal != epWorldSizeHccl)) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        DataCopyParams hcclDatacopyParams{1U, HCCL_DFX_NUM * sizeof(uint32_t), 0U, 0U};
        dataStateLocalTensor.SetValue(HCCL_EPRANKId_POS, epRankIdHccl);
        dataStateLocalTensor.SetValue(HCCL_WORLDSIZE_POS, epWorldSizeHccl);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor[HCCL_DFX_POS], dataStateLocalTensor, hcclDatacopyParams);
    }
    return dataState;
}

// ============================================================================
// P2pUrmaSqeComm - SQE级URMA P2P通信实现
// 使用HcclCombinOpParam上下文，通过SQE级接口完成WriteWithNotify P2P通信
// ============================================================================
class P2pUrmaSqeComm {
public:
    __aicore__ inline P2pUrmaSqeComm() {};

    /*!
     * \brief 初始化通信上下文
     * \param tpipe TPipe指针，用于InitBuffer分配UB缓冲区
     * \param workspace 工作空间地址（用于本地URMA窗口地址计算）
     */
    __aicore__ inline void Init(TPipe *tpipe, GM_ADDR workspace)
    {
        tpipe_ = tpipe;
        localUrmaWorkspace_ = workspace;

        // 通过GetHcclContext获取HcclCombinOpParam上下文
        hcclContext_ = (__gm__ HcclCombinOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        epRankId_ = hcclContext_->rankId;
        epWorldSize_ = hcclContext_->rankDim;
    }

    /*!
     * \brief 分配SQE级通信所需的UB缓冲区
     * 在通信阶段tpipe->Reset()之后调用
     */
    __aicore__ inline void InitCommBuffer()
    {
        // 分配SQ信息缓冲区，大小按UB_ALIGN对齐
        uint32_t sqInfoSize = Ceil(sizeof(HcclAiRMAWQ), P2P_UB_ALIGN) * P2P_UB_ALIGN;
        tpipe_->InitBuffer(sqInfoBuf_, sqInfoSize);
        sqInfoTensor_ = sqInfoBuf_.Get<uint8_t>();

        // 分配CQ信息缓冲区
        uint32_t cqInfoSize = Ceil(sizeof(HcclAiRMACQ), P2P_UB_ALIGN) * P2P_UB_ALIGN;
        tpipe_->InitBuffer(cqInfoBuf_, cqInfoSize);
        cqInfoTensor_ = cqInfoBuf_.Get<uint8_t>();

        // 分配SQE缓冲区（WriteWithNotify SQE为96字节）
        tpipe_->InitBuffer(sqeBuf_, P2P_WRITE_WITH_NOTIFY_SQE_SIZE);
        sqeTensor_ = sqeBuf_.Get<uint8_t>();

        // 分配CQE缓冲区（UB_ALIGN * CQ_DEPTH_256）
        tpipe_->InitBuffer(cqeBuf_, P2P_UB_ALIGN * P2P_CQ_DEPTH_256);
        cqeTensor_ = cqeBuf_.Get<uint8_t>();

        // 分配JFS DoorBell缓冲区
        tpipe_->InitBuffer(jfsDoorBellBuf_, P2P_UB_ALIGN);
        jfsDoorBellTensor_ = jfsDoorBellBuf_.Get<uint8_t>();

        // 分配JFC DoorBell缓冲区
        tpipe_->InitBuffer(jfcDoorBellBuf_, P2P_UB_ALIGN);
        jfcDoorBellTensor_ = jfcDoorBellBuf_.Get<uint8_t>();
    }

    /*!
     * \brief 初始化与指定目标rank的SQE通信状态
     * 加载SQ/CQ信息，获取PICI，检查首次通信标志，
     * 首次时InvalidateCqeStatus，轮询CQ清理残余，准备WriteWithNotify SQE模板
     * \param dstRankId 目标rank ID
     * \param cycleSqCqLoadEndPtr 可选：步骤1-6完成后的cycle打点（nullptr=不记录）
     * \param cycleSqeTemplateEndPtr 可选：步骤7-8完成后的cycle打点（nullptr=不记录）
     */
    __aicore__ inline void InitRankComm(int32_t dstRankId,
                                          int64_t* cycleSqCqLoadEndPtr = nullptr,
                                          int64_t* cycleSqeTemplateEndPtr = nullptr)
    {
        // 步骤1: 从HcclCombinOpParam中获取SQ信息到UB
        GetURMASqInfoTensor(sqInfoTensor_, (GM_ADDR)hcclContext_, dstRankId);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 步骤2: 从HcclCombinOpParam中获取CQ信息到UB
        GetURMACqInfoTensor(cqInfoTensor_, (GM_ADDR)hcclContext_, dstRankId);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 步骤3: 检查是否为首次与该rank通信
        bool isFirstComm = false;
        GetIsFirstInComm((GM_ADDR)hcclContext_, epRankId_, dstRankId, isFirstComm);
        isFirstComm_ = !isFirstComm; // GetIsFirstInComm返回true表示已经初始化过

        // 步骤4: 获取PICI（Producer-Consumer Index）
        GetPICI((GM_ADDR)hcclContext_, epRankId_, dstRankId,
                outSqPi_, outSqCi_, outCqPi_, outCqCi_, outSqPiLinear_, outCqCiLinear_);

        // 步骤5: 首次通信时，将CQ中所有CQE的status初始化为无效值（0xff）
        if (isFirstComm_) {
            InvalidateCqeStatus(cqInfoTensor_, cqeTensor_);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            UpdateIsFirstInComm((GM_ADDR)hcclContext_, epRankId_, dstRankId, true);
        }

        // 步骤6: 轮询CQ清理残余完成事件，更新sqCi
        PollCommCQUpdateSQCI(sqInfoTensor_, cqInfoTensor_, cqeTensor_, jfsDoorBellTensor_,
                             outSqCi_, outCqCi_, outCqCiLinear_);

        // 打点：步骤1-6完成（SQ/CQ加载+PICI+CQE初始化+CQ轮询）
        if (cycleSqCqLoadEndPtr != nullptr) {
            *cycleSqCqLoadEndPtr = GetSystemCycle();
        }

        // 步骤7: 生成WriteWithNotify SQE模板
        GenerateCommWriteWithNotifySQE(sqeTensor_, 1);

        // 步骤8: 更新SQE模板中的通信元数据（tp_id, jetty_id, rmt_eid等）
        UpdateCommWriteWithNotifySQE(sqeTensor_, sqInfoTensor_);

        // 打点：步骤7-8完成（SQE模板生成+元数据更新）
        if (cycleSqeTemplateEndPtr != nullptr) {
            *cycleSqeTemplateEndPtr = GetSystemCycle();
        }

        // 记录当前目标rank
        curDstRankId_ = dstRankId;
    }

    /*!
     * \brief 提交一个WriteWithNotify SQE（4参数简化版本）
     * 自动从远端窗口状态区计算notifyAddr，notifyData使用1.0f标志值
     * \param dstRankId 目标rank ID
     * \param localAddr 本地数据地址
     * \param remoteAddr 远端目标地址
     * \param length 发送数据长度（字节）
     */
    __aicore__ inline void P2pSend(int32_t dstRankId, uint64_t localAddr, uint64_t remoteAddr,
                                   uint32_t length)
    {
        // 从远端窗口状态区计算通知地址：dstRankId的状态区 + 本rank的状态偏移
        uint64_t notifyAddr = reinterpret_cast<uint64_t>(GetRemoteWinStateAddr(dstRankId, notifyDataState_))
                              + P2P_STATE_OFFSET * (uint64_t)epRankId_;
        uint64_t notifyData = P2P_NOTIFY_DATA_FLAG;  // 1.0f标志值(0x3F800000)
        SetCommWriteWithNotifySQE(sqeTensor_, localAddr, remoteAddr, length, notifyAddr, notifyData, 0);
        PutCommNotifySQE(sqInfoTensor_, cqInfoTensor_, sqeTensor_, cqeTensor_, jfcDoorBellTensor_,
                         1, outSqPi_, outSqPiLinear_, outSqCi_, outCqCi_, outCqCiLinear_);
    }

    /*!
     * \brief 提交一个WriteWithNotify SQE（6参数完整版本）
     * 仅设置SQE参数并通过PutCommNotifySQE写入SQ，sqPi递增
     * \param dstRankId 目标rank ID
     * \param localAddr 本地数据地址
     * \param remoteAddr 远端目标地址
     * \param length 发送数据长度（字节）
     * \param notifyAddr 远端通知地址
     * \param notifyData 通知数据值
     */
    __aicore__ inline void P2pSend(int32_t dstRankId, uint64_t localAddr, uint64_t remoteAddr,
                                   uint32_t length, uint64_t notifyAddr, uint64_t notifyData)
    {
        // 设置SQE中的数据传输参数
        SetCommWriteWithNotifySQE(sqeTensor_, localAddr, remoteAddr, length, notifyAddr, notifyData, 0);

        // 将SQE写入SQ，sqPi和sqPiLinear由PutCommNotifySQE在引用参数中更新
        PutCommNotifySQE(sqInfoTensor_, cqInfoTensor_, sqeTensor_, cqeTensor_, jfcDoorBellTensor_,
                         1, outSqPi_, outSqPiLinear_, outSqCi_, outCqCi_, outCqCiLinear_);
    }

    /*!
     * \brief 敲JFS DoorBell通知硬件有新的SQE（P2pSendDrain的第一步）
     * 用于精确测量头开销：从Process开始到DoorBell触发完成的时间间隔
     */
    __aicore__ inline void TriggerDoorBell()
    {
        SendJFSDoorBell(jfsDoorBellTensor_, sqInfoTensor_, outSqPiLinear_);
    }

    /*!
     * \brief 轮询CQ等待完成 + 更新PICI（P2pSendDrain的第二、三步）
     * 用于精确测量延迟：从DoorBell触发完成到Drain执行完毕的时间间隔
     * \param dstRankId 目标rank ID
     * \param cyclePollCqEndPtr 可选：PollCommCQUpdateSQCI完成后的cycle打点（nullptr=不记录）
     * \param cycleUpdatePiciEndPtr 可选：UpdatePICI完成后的cycle打点（nullptr=不记录）
     */
    __aicore__ inline void DrainWaitCompletion(int32_t dstRankId,
                                                int64_t* cyclePollCqEndPtr = nullptr,
                                                int64_t* cycleUpdatePiciEndPtr = nullptr)
    {
        // 步骤2: 轮询CQ等待完成通知，更新sqCi
        PollCommCQUpdateSQCI(sqInfoTensor_, cqInfoTensor_, cqeTensor_, jfsDoorBellTensor_,
                             outSqCi_, outCqCi_, outCqCiLinear_);

        // 打点：CQ轮询完成
        if (cyclePollCqEndPtr != nullptr) {
            *cyclePollCqEndPtr = GetSystemCycle();
        }

        // 步骤3: 更新PICI，保存到HcclCombinOpParam的windowsOut中
        UpdatePICI((GM_ADDR)hcclContext_, epRankId_, dstRankId,
                   outSqPi_, outSqCi_, outCqPi_, outCqCi_, outSqPiLinear_, outCqCiLinear_);

        // 打点：PICI更新完成
        if (cycleUpdatePiciEndPtr != nullptr) {
            *cycleUpdatePiciEndPtr = GetSystemCycle();
        }
    }

    /*!
     * \brief 冲刷所有待发送SQE：敲DoorBell + 轮询CQ + 更新PICI
     * P2pSendDrain = TriggerDoorBell + DrainWaitCompletion
     * \param dstRankId 目标rank ID
     */
    __aicore__ inline void P2pSendDrain(int32_t dstRankId)
    {
        TriggerDoorBell();
        DrainWaitCompletion(dstRankId);
    }

    /*!
     * \brief 获取指定rank的远端窗口数据区地址
     * 使用HcclCombinOpParam.windowsIn[rankId]获取远端窗口地址
     * \param rankId 目标rank ID
     * \param winDataSizeOffset 窗口数据区偏移
     * \return 远端窗口数据区起始地址
     */
    __aicore__ inline GM_ADDR GetRemoteWinAddr(int32_t rankId, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]) + P2P_STATE_SIZE + winDataSizeOffset;
    }

    /*!
     * \brief 获取指定rank的远端窗口状态区地址
     * 使用HcclCombinOpParam.windowsIn[rankId]获取远端窗口地址
     * \param rankId 目标rank ID
     * \param dataState 数据状态（用于双缓冲选择半区）
     * \return 远端窗口状态区起始地址
     */
    __aicore__ inline GM_ADDR GetRemoteWinStateAddr(int32_t rankId, uint32_t dataState)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]) + dataState * P2P_WIN_STATE_OFFSET;
    }

    /*!
     * \brief 获取指定rank的本地窗口数据区地址（在workspace中）
     * \param rankId 目标rank ID
     * \param totalWinSizeEp 每个rank的窗口大小
     * \param winDataSizeOffset 窗口数据区偏移
     * \return 本地窗口数据区起始地址
     */
    __aicore__ inline GM_ADDR GetLocalWinAddr(int32_t rankId, uint64_t totalWinSizeEp, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId + P2P_STATE_SIZE + winDataSizeOffset;
    }

    /*!
     * \brief 获取指定rank的本地窗口状态区地址（在workspace中）
     * \param rankId 目标rank ID
     * \param totalWinSizeEp 每个rank的窗口大小
     * \param dataState 数据状态（用于双缓冲选择半区）
     * \return 本地窗口状态区起始地址
     */
    __aicore__ inline GM_ADDR GetLocalWinStateAddr(int32_t rankId, uint64_t totalWinSizeEp, uint32_t dataState)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId + dataState * P2P_WIN_STATE_OFFSET;
    }

    // 获取当前rank ID
    __aicore__ inline int32_t GetEpRankId() { return epRankId_; }

    // 获取HcclCombinOpParam上下文地址
    __aicore__ inline __gm__ HcclCombinOpParam *GetHcclContext() { return hcclContext_; }

    // 获取工作空间地址
    __aicore__ inline GM_ADDR GetWorkspace() { return localUrmaWorkspace_; }

    /*!
     * \brief 设置双缓冲状态值，用于4参数P2pSend自动计算notifyAddr
     * \param dataState 双缓冲状态值(0或1)
     */
    __aicore__ inline void SetNotifyDataState(uint32_t dataState) { notifyDataState_ = dataState; }

private:
    // 通信上下文
    __gm__ HcclCombinOpParam *hcclContext_{nullptr};
    TPipe *tpipe_{nullptr};
    int32_t epRankId_{0};
    int32_t epWorldSize_{0};
    GM_ADDR localUrmaWorkspace_{nullptr};
    uint32_t notifyDataState_{0};  // 双缓冲状态值，用于4参数P2pSend自动计算notifyAddr

    // SQE级UB缓冲区
    LocalTensor<uint8_t> sqInfoTensor_;
    LocalTensor<uint8_t> cqInfoTensor_;
    LocalTensor<uint8_t> sqeTensor_;
    LocalTensor<uint8_t> cqeTensor_;
    LocalTensor<uint8_t> jfsDoorBellTensor_;
    LocalTensor<uint8_t> jfcDoorBellTensor_;
    TBuf<> sqInfoBuf_;
    TBuf<> cqInfoBuf_;
    TBuf<> sqeBuf_;
    TBuf<> cqeBuf_;
    TBuf<> jfsDoorBellBuf_;
    TBuf<> jfcDoorBellBuf_;

    // 每个rank的PICI状态（当前目标rank的索引值）
    uint32_t outSqPi_{0};
    uint32_t outSqCi_{0};
    uint32_t outCqPi_{0};
    uint32_t outCqCi_{0};
    uint32_t outSqPiLinear_{0};
    uint32_t outCqCiLinear_{0};
    bool isFirstComm_{true};
    int32_t curDstRankId_{-1};
};

} // namespace UrmaP2pBenchmarkImpl
#endif // P2P_URMA_COMM_H