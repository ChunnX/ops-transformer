/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
 *
 * 与SQE级(P2pUrmaSqeComm)的区别:
 *   - 无需手动组装SQE、敲DoorBell、轮询CQ、管理PICI
 *   - Hcomm内部封装了所有URMA底层操作
 *   - 通信流程更简洁: WriteNbi -> Drain
 *   - 地址模型不同: 需要P2pHcommContext(context)提供epHcclBuffer_和hcommHandle_
 *
 * 注意: P2pHcommContext定义了包含hcommHandle_的扩展结构体，
 *       因ops-trans-coke common的Mc2Aclnn::Mc2MoeContext不含hcommHandle_字段，
 *       且不能修改common文件。
 */

#ifndef P2P_HCOMM_COMM_H
#define P2P_HCOMM_COMM_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace UrmaP2pBenchmarkImpl {

// ============ Hcomm通信常量 ============
// 地址布局参考bandwidth_test的moe_distribute_v2_constant.h:
//   epHcclBuffer_[rankId]布局:
//     [0: 384KB]  - Buffer slot 0状态区(dataState_=0时使用)
//     [384KB: 768KB] - Buffer slot 1状态区(dataState_=1时使用，双缓冲)
//     [768KB: 1MB]   - Dispatch状态区(InitWinState读取dataState_)
//     [1MB: ...]     - 数据区(P2P WriteNbi目标)
constexpr uint64_t HCOMM_INIT_SIZE = 512UL;                     // Hcomm Init缓冲区大小(参考bandwidth_test)
constexpr uint64_t HCOMM_STATE_SIZE = 1024UL * 1024UL;          // 数据区偏移: 1MB(参考bandwidth_test STATE_SIZE)
constexpr uint64_t HCOMM_WIN_STATE_OFFSET = 384UL * 1024UL;     // 状态区偏移: 384KB(参考bandwidth_test WIN_STATE_OFFSET)
constexpr uint64_t HCOMM_STATE_WIN_OFFSET = 768UL * 1024UL;     // Dispatch状态区偏移: 768KB(参考bandwidth_test DISPATCH_STATE_WIN_OFFSET)
constexpr uint32_t HCOMM_UB_ALIGN = 32U;                        // UB对齐(32字节)
constexpr uint64_t HCOMM_NOTIFY_DATA_FLAG = 0x3F800000UL;       // 1.0f标志值的整数表示

// ============ P2pHcommContext扩展结构体 ============
// ops-trans-coke common的Mc2Aclnn::Mc2MoeContext不含hcommHandle_字段，
// 此处定义P2pHcommContext包含完整通信上下文(含hcommHandle_)，
// 与bandwidth_test的Mc2MoeContext布局一致。
constexpr uint32_t P2P_HCCL_MAX_RANK_SIZE = 1024;

struct P2pHcommContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer_[P2P_HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[P2P_HCCL_MAX_RANK_SIZE];  // 通信句柄(支持ROCE或URMA)
};

using namespace AscendC;

// ============================================================================
// P2pHcommComm - Hcomm WriteNbi+Drain模式P2P通信类
// 使用P2pHcommContext上下文，通过Hcomm<COMM_PROTOCOL_UBC_CTP>完成P2P通信
// ============================================================================
class P2pHcommComm {
public:
    __aicore__ inline P2pHcommComm() {};

    /*!
     * \brief 初始化通信上下文
     * \param tpipe TPipe指针，用于InitBuffer分配UB缓冲区
     * \param workspace 工作空间地址(用于本地URMA窗口地址计算)
     * \param contextGM Mc2Context GM地址(Host端GetMc2ContextTensor创建)
     */
    __aicore__ inline void Init(TPipe *tpipe, GM_ADDR workspace, GM_ADDR contextGM)
    {
        tpipe_ = tpipe;
        localUrmaWorkspace_ = workspace;
        context_ = (__gm__ P2pHcommContext *)contextGM;

        // 初始化Hcomm实例(参考bandwidth_test: hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE))
        tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
        hcommTensor_ = hcommBuf_.Get<uint8_t>();
        hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

        // 从context获取rank信息
        epRankId_ = context_->epRankId;
        epWorldSize_ = context_->rankSizePerServer;
    }

    // 获取当前rank ID
    __aicore__ inline int32_t GetEpRankId() { return epRankId_; }

    // 获取指定rank的Hcomm通信句柄(参考bandwidth_test: GetCommHandle)
    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId)
    {
        uint32_t index = rankId > epRankId_ ? rankId - 1 : rankId;
        return context_->hcommHandle_[index];
    }

    /*!
     * \brief 获取指定rank的远端URMA数据区地址
     * 使用context->epHcclBuffer_[rankId]获取远端URMA Buffer地址
     * \param rankId 目标rank ID
     * \param winDataSizeOffset 窗口数据区偏移
     * \return 远端URMA数据区起始地址
     */
    __aicore__ inline GM_ADDR GetRemoteUrmaAddr(int32_t rankId, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)(context_->epHcclBuffer_[rankId]) + HCOMM_STATE_SIZE + winDataSizeOffset;
    }

    /*!
     * \brief 获取指定rank的远端URMA状态区地址
     * 使用context->epHcclBuffer_[rankId]获取远端URMA Buffer地址
     * dataState_=0(无双缓冲): 状态区从buffer基地址开始，偏移=0*WIN_STATE_OFFSET=0
     * \param rankId 目标rank ID
     * \return 远端URMA状态区起始地址
     */
    __aicore__ inline GM_ADDR GetRemoteUrmaStateAddr(int32_t rankId)
    {
        // dataState_=0: 状态区偏移 = 0 * HCOMM_WIN_STATE_OFFSET(384KB) = 0
        return (GM_ADDR)(context_->epHcclBuffer_[rankId]);
    }

    /*!
     * \brief 获取指定rank的本地窗口数据区地址(在workspace中)
     * \param rankId 目标rank ID
     * \param totalWinSizeEp 每个rank的窗口大小
     * \param winDataSizeOffset 窗口数据区偏移
     * \return 本地窗口数据区起始地址
     */
    __aicore__ inline GM_ADDR GetLocalWinAddr(int32_t rankId, uint64_t totalWinSizeEp, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId + HCOMM_STATE_SIZE + winDataSizeOffset;
    }

    /*!
     * \brief 获取指定rank的本地窗口状态区地址(在workspace中)
     * dataState_=0(无双缓冲): 状态区从每个rank的窗口section开头开始
     * \param rankId 目标rank ID
     * \param totalWinSizeEp 每个rank的窗口大小
     * \return 本地窗口状态区起始地址
     */
    __aicore__ inline GM_ADDR GetLocalWinStateAddr(int32_t rankId, uint64_t totalWinSizeEp)
    {
        // dataState_=0: 状态区偏移 = 0 * HCOMM_WIN_STATE_OFFSET(384KB) = 0
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId;
    }

    /*!
     * \brief Write: 使用WriteNbi发送数据到远端
     * \param dstRankId 目标rank ID
     * \param remoteAddr 远端目标地址(GM_ADDR类型，匹配WriteNbi签名)
     * \param localAddr 本地数据地址(GM_ADDR类型，匹配WriteNbi签名)
     * \param length 发送数据长度(字节)
     */
    __aicore__ inline void Write(int32_t dstRankId, GM_ADDR remoteAddr, GM_ADDR localAddr, uint32_t length)
    {
        hcomm_.WriteNbi(GetCommHandle(dstRankId), remoteAddr, localAddr, length);
    }

    /*!
     * \brief Drain: 等待指定rank的WriteNbi完成
     * \param dstRankId 目标rank ID
     */
    __aicore__ inline void Drain(int32_t dstRankId)
    {
        hcomm_.Drain(GetCommHandle(dstRankId));
    }

    /*!
     * \brief P2pSendNotify: WriteNbi推送32字节状态通知到远端
     * 参考bandwidth_test SetStatus: WriteNbi(status)（不含冗余Drain）
     * 注意: 数据WriteNbi的Drain必须在调用P2pSendNotify之前完成
     * \param dstRankId 目标rank ID
     * \param remoteAddr 远端状态目标地址(GM_ADDR类型，匹配WriteNbi签名)
     * \param localAddr 本地状态数据地址(GM_ADDR类型，匹配WriteNbi签名)
     */
    __aicore__ inline void P2pSendNotify(int32_t dstRankId, GM_ADDR remoteAddr, GM_ADDR localAddr)
    {
        hcomm_.WriteNbi(GetCommHandle(dstRankId), remoteAddr, localAddr, HCOMM_UB_ALIGN);
    }

private:
    TPipe *tpipe_{nullptr};
    __gm__ P2pHcommContext *context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    LocalTensor<uint8_t> hcommTensor_;
    TBuf<> hcommBuf_;
    GM_ADDR localUrmaWorkspace_{nullptr};
    int32_t epRankId_{0};
    uint32_t epWorldSize_{0};
};

} // namespace UrmaP2pBenchmarkImpl
#endif // P2P_HCOMM_COMM_H
