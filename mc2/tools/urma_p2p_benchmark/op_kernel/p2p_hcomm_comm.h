/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file p2p_hcomm_comm.h
 * \brief Hcomm WriteNbi+Drain模式P2P通信实现
 *
 * 使用Hcomm<COMM_PROTOCOL_UBC_CTP>封装层实现P2P通信，
 * 核心API: WriteNbi(非阻塞写入) + Drain(等待完成)。
 * 与SQE级(P2pUrmaSqeComm)的区别:
 *   - 无需手动组装SQE、敲DoorBell、轮询CQ、管理PICI
 *   - Hcomm内部封装了所有URMA底层操作
 *   - 通信流程更简洁: WriteNbi → Drain
 *   - 地址模型不同: 需要Mc2MoeContext(context)提供epHcclBuffer和hcommHandle
 */

#ifndef P2P_HCOMM_COMM_H
#define P2P_HCOMM_COMM_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"

// Mc2MoeContext定义（从bandwidth_test项目中提取）
struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epWorldSize;
    uint64_t epHcclBuffer[64];   // 各rank的URMA Buffer地址
    uint64_t hcommHandle[64];    // 通信句柄（不含自身rank, 实际长度=worldSize-1）
};

namespace UrmaP2pBenchmarkImpl {

using namespace AscendC;

constexpr uint64_t HCOMM_INIT_SIZE = 512UL;
constexpr uint64_t HCOMM_STATE_SIZE = 350UL * 1024UL;
constexpr uint64_t HCOMM_WIN_STATE_OFFSET = 350UL * 1024UL;
constexpr uint64_t HCOMM_WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t HCOMM_UB_ALIGN = 32U;
constexpr uint64_t HCOMM_NOTIFY_DATA_FLAG = 0x3F800000UL;

// ============================================================================
// P2pHcommComm - Hcomm WriteNbi+Drain模式P2P通信实现
// ============================================================================
class P2pHcommComm {
public:
    __aicore__ inline P2pHcommComm() {}

    /*!
     * \brief 初始化通信上下文
     * \param tpipe TPipe指针
     * \param workspace 工作空间地址（本地URMA窗口）
     * \param context Mc2MoeContext GM地址（由Host端通过Mc2Context创建）
     */
    __aicore__ inline void Init(TPipe *tpipe, GM_ADDR workspace, GM_ADDR context)
    {
        tpipe_ = tpipe;
        localUrmaWorkspace_ = workspace;
        context_ = (__gm__ Mc2MoeContext *)context;
        epRankId_ = context_->epRankId;
        epWorldSize_ = context_->epWorldSize;

        // 初始化Hcomm对象
        tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
        hcommTensor_ = hcommBuf_.Get<uint8_t>();
        hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    }

    /*!
     * \brief 获取目标rank的通信句柄
     * hcommHandle数组不含自身rank，索引需要映射
     */
    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId)
    {
        uint32_t index = rankId > epRankId_ ? rankId - 1 : rankId;
        return context_->hcommHandle[index];
    }

    /*!
     * \brief P2P发送数据到目标rank（WriteNbi模式）
     * \param dstRankId 目标rank ID
     * \param remoteAddr 远端URMA Buffer目标地址
     * \param localAddr 本地数据源地址
     * \param length 发送数据长度（字节）
     */
    __aicore__ inline void P2pSend(int32_t dstRankId, uint64_t remoteAddr,
                                    uint64_t localAddr, uint32_t length)
    {
        hcomm_.WriteNbi(GetCommHandle(dstRankId), remoteAddr, localAddr, length);
    }

    /*!
     * \brief P2P发送状态通知到目标rank（WriteNbi模式，32字节状态块）
     * \param dstRankId 目标rank ID
     * \param remoteStateAddr 远端URMA Buffer状态区地址
     * \param localStateAddr 本地状态数据源地址
     */
    __aicore__ inline void P2pSendNotify(int32_t dstRankId, uint64_t remoteStateAddr,
                                          uint64_t localStateAddr)
    {
        // 先Drain确保之前的WriteNbi数据发送完成
        hcomm_.Drain(GetCommHandle(dstRankId));
        // 再发送状态通知
        hcomm_.WriteNbi(GetCommHandle(dstRankId), remoteStateAddr, localStateAddr, HCOMM_UB_ALIGN);
    }

    /*!
     * \brief 等待所有WriteNbi操作完成
     * \param dstRankId 目标rank ID
     */
    __aicore__ inline void Drain(int32_t dstRankId)
    {
        hcomm_.Drain(GetCommHandle(dstRankId));
    }

    /*!
     * \brief 获取远端rank的URMA Buffer数据区地址
     * \param rankId 目标rank ID
     * \param winDataSizeOffset 数据区偏移
     */
    __aicore__ inline GM_ADDR GetRemoteUrmaAddr(int32_t rankId, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)(context_->epHcclBuffer[rankId]) + HCOMM_STATE_SIZE + winDataSizeOffset;
    }

    /*!
     * \brief 获取远端rank的URMA Buffer状态区地址
     * \param rankId 目标rank ID
     */
    __aicore__ inline GM_ADDR GetRemoteUrmaStateAddr(int32_t rankId)
    {
        return (GM_ADDR)(context_->epHcclBuffer[rankId]);
    }

    /*!
     * \brief 获取本地workspace窗口数据区地址
     * \param rankId 目标rank ID
     * \param totalWinSizeEp 每个rank窗口大小
     * \param winDataSizeOffset 数据区偏移
     */
    __aicore__ inline GM_ADDR GetLocalWinAddr(int32_t rankId, uint64_t totalWinSizeEp,
                                                uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId
               + HCOMM_STATE_SIZE + winDataSizeOffset;
    }

    /*!
     * \brief 获取本地workspace窗口状态区地址
     * \param rankId 目标rank ID
     * \param totalWinSizeEp 每个rank窗口大小
     */
    __aicore__ inline GM_ADDR GetLocalWinStateAddr(int32_t rankId, uint64_t totalWinSizeEp)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId;
    }

    __aicore__ inline int32_t GetEpRankId() { return epRankId_; }
    __aicore__ inline int32_t GetEpWorldSize() { return epWorldSize_; }
    __aicore__ inline GM_ADDR GetWorkspace() { return localUrmaWorkspace_; }
    __aicore__ inline __gm__ Mc2MoeContext *GetContext() { return context_; }

private:
    TPipe *tpipe_{nullptr};
    __gm__ Mc2MoeContext *context_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    LocalTensor<uint8_t> hcommTensor_;
    TBuf<> hcommBuf_;

    int32_t epRankId_{0};
    int32_t epWorldSize_{0};
    GM_ADDR localUrmaWorkspace_{nullptr};
};

} // namespace UrmaP2pBenchmarkImpl
#endif // P2P_HCOMM_COMM_H
