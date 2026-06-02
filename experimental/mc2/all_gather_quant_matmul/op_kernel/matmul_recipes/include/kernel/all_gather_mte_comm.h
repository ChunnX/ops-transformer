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
 * \file all_gather_mte_comm.h
 * \brief MTE 通信底层：win 区地址管理、跨 rank 状态同步、0/1 分区双缓冲
 *
 * Win 区布局（状态区 1024KB + 数据区 200MB，各有 0/1 两个分区交替使用）
 *   状态区: [算子1-0区 | 算子2-0区 | 算子1-1区 | 算子2-1区 | V核标志 | C核标志 | ...]
 *   数据区: [算子1-0区 | 算子2-0区 | 算子1-1区 | 算子2-1区 | ...]
 */

#ifndef ALL_GATHER_MTE_COMM_H
#define ALL_GATHER_MTE_COMM_H

#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "all_gather_mte_utils.h"
#include "../../common/op_kernel/moe_distribute_base.h"


namespace AscendC {

/*!
 * \brief 硬件事件同步辅助函数：SetFlag + WaitFlag 配对使用
 * 用于确保前后两个 pipeline stage 之间的执行顺序
 * @tparam event  硬件事件类型（如 HardEvent::S_MTE3, MTE2_S 等）
 */
template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
}

namespace QuantMTECommImpl {

using namespace AscendC;

// ── 常量定义 ──────────────────────────────────────────────────────────
constexpr static uint32_t UB_ALIGN_BYTES = 32U;         // UB 32B 对齐
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;             // float 类型 32B 对齐需要 8 个元素
constexpr uint32_t BUFFER_NUM = 2U;                     // Double buffer
constexpr static uint32_t X_BLOCK_BYTES = 512U;         // 单个 x 数据块 512B
constexpr static uint32_t SCALES_BLOCK_BYTES = 256U;    // Scale 块大小（63*4B 按 32B 对齐）
constexpr static uint64_t WIN_ADDR_ALIGN = 512UL;       // Win 区数据 512B 对齐
constexpr static uint64_t CV_SYNC_START_OFFSET = 10UL * 1024UL; // CV 状态区相对于通信状态区偏移 10KB
constexpr static uint64_t CV_STATE_ALIGN = 64UL;        // CV 标志位间 64B 对齐（DataCache 刷新最小单位）
constexpr static uint64_t WIN_DATA_OFFSET = 100UL * 1024UL * 1024UL;   // 1 区数据起始偏移 100MB
constexpr static uint64_t WIN_STATUS_OFFSET = 400UL * 1024UL;          // 1 区状态起始偏移 400KB
constexpr static uint64_t WIN_AIV_ZERONE_FLAG_OFFSET = 800UL * 1024UL; // V 核 0/1 标志位区偏移 800KB
constexpr static uint64_t WIN_AIC_ZERONE_FLAG_OFFSET = 850UL * 1024UL; // C 核 0/1 标志位区偏移 850KB
constexpr static uint64_t ZERONE_STATUS_ALIGN = 64UL;   // 0/1 标志位 64B 对齐
constexpr static uint32_t ZERONE_STATUS_POS = 0U;       // 0/1 标志位在 64B 块中的 index
constexpr static uint32_t DATA_START_OFFSET = 1024UL * 1024UL; // 0 区数据起始偏移 1MB

// 状态区划分说明（见文件头注释中的 ASCII 图）

#define AllGatherTemplateTypeClass typename XType, typename ScalesType, typename OutputType, uint32_t SyncMode, bool isOptionalOutput
#define AllGatherTemplateType XType, ScalesType, OutputType, SyncMode, isOptionalOutput

/*!
 * \class MTECommunication
 * \brief MTE 通信底层封装，管理 win 区地址和跨 rank 软同步
 */
template<AllGatherTemplateTypeClass>
class MTECommunication {
public:
    __aicore__ inline MTECommunication() {};
    __aicore__ inline void InitHcclContext(GM_ADDR contextGM);
    __aicore__ inline void InitParams(uint64_t xSize, uint64_t aivNum, uint32_t sendCoreNumPerRank);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void ReadAndFlipWinFlag();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline GM_ADDR GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag);
    __aicore__ inline GM_ADDR GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag);
    __aicore__ inline GM_ADDR GetWinZeroneFlagAddrGm(uint32_t rankId);

    // ── 核心状态 ──
    uint32_t aivId_{0};                     // 当前 AIV 核编号
    uint32_t aicId_{0};                     // 当前 AIC 核编号（aivId / 2）
    uint64_t aivNum_{0};                    // 总 AIV 核数
    uint64_t winDataSize_{0};               // Win 区数据部分大小（所有 rank 数据 + 对齐填充）
    uint32_t curDstId_{0};                  // 当前核目标 rank（用于数据搬运目标）
    uint32_t sendCoreNumPerRank_{0};        // 每个 remote rank 分配的核数
    uint32_t curRankId_{0};                 // 当前 rank ID
    uint32_t rankSize_{0};                  // 总 rank 数
    uint32_t winFlag_{0};                   // 当前使用的 win 区分区（0 或 1）
    __gm__ Mc2Kernel::HcclOpParam* hcclContext_; // HCCL 通信上下文（含 win 区地址数组）

private:
    GlobalTensor<uint32_t> selfWinFlagGMTensor_;  // 本核 0/1 标志位的 GM 映射
    LocalTensor<float> stateResetTensor_;         // 状态清零用的 UB 缓冲（预填充 0.0）

    TBuf<> writeStateBuf_;    // 写状态用的 UB 缓冲
    TBuf<> readStateBuf_;     // 读状态用的 UB 缓冲（容纳 rankSize 个状态）
    TBuf<> stateResetBuf_;    // 状态清零缓冲（容纳 rankSize 个状态位）
};

/*!
 * \brief 初始化 HCCL 通信上下文
 * 保存 HCCL 参数指针，后续通过 hcclContext_->windowsIn/Out 访问各 rank 的 win 区
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitHcclContext(GM_ADDR contextGM)
{
    hcclContext_ = (__gm__ Mc2Kernel::HcclOpParam*)contextGM;
}

/*!
 * \brief 初始化通信参数
 *
 * 计算：
 *   - rankSize_:    总 rank 数（从 hcclContext 获取）
 *   - winDataSize_: win 区数据部分大小 = CeilAlign(rankSize * xSize, 512)
 *   - curDstId_:    当前核目标 rank = aicId / sendCoreNumPerRank
 *
 * @param xSize               单 rank x 数据大小（字节）
 * @param aivNum              总 AIV 核数
 * @param sendCoreNumPerRank  每个 remote rank 分配的核数
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitParams(
    uint64_t xSize, uint64_t aivNum, uint32_t sendCoreNumPerRank)
{
    aivNum_ = aivNum;
    aivId_ = GetBlockIdx();
    aicId_ = aivId_ / GetTaskRation();    // CV 比 1:2
    rankSize_ = Mc2Kernel::GetRankDim(hcclContext_);
    winDataSize_ = CeilAlignU64(rankSize_ * xSize, WIN_ADDR_ALIGN);
    curRankId_ = Mc2Kernel::GetRankId(hcclContext_);
    sendCoreNumPerRank_ = sendCoreNumPerRank;
    curDstId_ = aicId_ / sendCoreNumPerRank_;
}

/*!
 * \brief 读取并翻转本核的 0/1 分区标志位
 *
 * 机制：
 *   1. 从 win 区的 V 核标志位区读取当前值（0 或 1）
 *   2. 将其翻转（0→1 或 1→0）并写回
 *   3. winFlag_ 保存翻转前的值，标识本次通信使用哪个分区
 *
 * 双缓冲效果：
 *   - 第 N 次通信使用 0 区（winFlag_=0）
 *   - 第 N+1 次通信使用 1 区（winFlag_=1）
 *   - AIC 读取上一轮数据时，AIV 可以同时写入下一轮数据
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::ReadAndFlipWinFlag()
{
    // 每个核有独立的 64B 对齐标志位
    uint64_t curCoreFlagAddr = (uint64_t)GetWinZeroneFlagAddrGm(curRankId_) + aivId_ * ZERONE_STATUS_ALIGN;
    selfWinFlagGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)curCoreFlagAddr);

    // 刷新 DataCache 确保读取最新值
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfWinFlagGMTensor_);
    winFlag_ = selfWinFlagGMTensor_.GetValue(ZERONE_STATUS_POS);

    // if (GetBlockIdx() == 0) {
    //     PRINTF("[DBG-AG] ReadAndFlipWinFlag rank=%u aivId=%u oldFlag=%u newFlag=%u\n",
    //            curRankId_, aivId_, winFlag_, 1 - winFlag_);
    // }

    // 翻转标志位并写回
    selfWinFlagGMTensor_.SetValue(ZERONE_STATUS_POS, 1 - winFlag_);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfWinFlagGMTensor_);
}

/*!
 * \brief 分配状态同步所需的 UB 缓冲
 *
 * 分配：
 *   - writeStateBuf_:  32B（一个状态位）
 *   - readStateBuf_:   rankSize * 32B（容纳 rankSize 个状态位用于 Sum 归约）
 *   - stateResetBuf_:  rankSize * 32B（预填充 0.0，用于清零状态区）
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::InitBuffer(TPipe *tPipe)
{
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES);
    tPipe->InitBuffer(readStateBuf_, rankSize_ * UB_ALIGN_BYTES);
    tPipe->InitBuffer(stateResetBuf_, rankSize_ * UB_ALIGN_BYTES);

    // 预填充状态清零缓冲
    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(rankSize_ * FLOAT_UB_ALIGN_NUM));
}

/*!
 * \brief 向所有 rank 的 win 状态区写入本核数据搬运完成标志
 *
 * 写入位置：win 状态区中 (aivId * rankSize + curRankId) 偏移处
 * 每个核在所有 rank 的状态区中各写一个标志，表示本核数据已就绪
 *
 * 写入流程（对每个 rank 循环）：
 *   1. 从 writeStateBuf_ 获取 32B 空间
 *   2. 先清零（从 stateResetTensor_ 复制），然后设置 [0] = 1.0
 *   3. DataCopy 到目标 rank 的 win 状态区
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * rankSize_;
    // if (GetBlockIdx() == 0) {
    //     // GM_ADDR dbgAddr = GetWinStatusAddrGm(curRankId_, winFlag_);
    //     // GM_ADDR dbgWinBase = GetWinDataAddrGm(curRankId_, winFlag_);
    //     // PRINTF("[DBG-AG] WriteStatusToWin rank=%u aivId=%u winFlag=%u statusGM=%p winDataGM=%p\n",
    //     //        curRankId_, aivId_, winFlag_, dbgAddr, dbgWinBase);
    //     PRINTF("[DBG-AG] WriteStatusToWin rank=%u aivId=%u winFlag=%u\n",
    //             curRankId_, aivId_, winFlag_);
    // }
    // 循环外一次性准备 statusTensor = [1.0, 0, 0, 0, 0, 0, 0, 0]
    // 用 V→S→MTE3 串联同步替代 PipeBarrier<PIPE_ALL>，闭合 V→MTE3 顺序
    LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
    DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_UB_ALIGN_NUM);
    SyncFunc<AscendC::HardEvent::V_S>();
    statusTensor.SetValue(0, (float)1.0);
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    // 循环内只发 MTE3，向所有 rank 写同一份 statusTensor
    for (uint32_t curRank = 0; curRank < rankSize_; ++curRank) {
        GM_ADDR remoteWinStateGM = GetWinStatusAddrGm(curRank, winFlag_);
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
        uint64_t curOffset = (coreOffset + curRankId_) * FLOAT_UB_ALIGN_NUM;
        DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

/*!
 * \brief 读取本 rank win 状态区并忙等待所有 rank 就绪
 *
 * 软同步屏障机制：
 *   1. 从本 rank 的 win 状态区读取 rankSize 个状态位
 *   2. 使用 Sum 归约计算总完成数
 *   3. 当 Sum == rankSize 时表示所有 rank 的当前核数据都已就绪
 *   4. 就绪后清零状态区，为下一轮通信做准备
 *
 * 忙等待循环中的管道同步：
 *   S→MTE2（发起读取）→ MTE2→V（等待读取完成）→ V→S（等待 Sum 完成）
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline void MTECommunication<AllGatherTemplateType>::ReadStatus()
{
    GM_ADDR stateGM = GetWinStatusAddrGm(curRankId_, winFlag_);
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * rankSize_ * FLOAT_UB_ALIGN_NUM;
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0;
    uint32_t statusCnt = rankSize_ * FLOAT_UB_ALIGN_NUM;
    SumParams sumParams{1, statusCnt, statusCnt};

    // if (GetBlockIdx() == 0) {
    //     PRINTF("[DBG-AG] ReadStatus START rank=%u aivId=%u winFlag=%u stateGM=%p\n",
    //            curRankId_, aivId_, winFlag_, stateGM);
    // }

    // 容差范围：[rankSize - 0.5, rankSize + 0.5)，避免浮点精度问题
    float minTarget = rankSize_ - (float)0.5;
    float maxTarget = rankSize_ + (float)0.5;
    uint32_t loopCnt = 0;
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(statusTensor, selfStatusWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusTensor, statusTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        flag = statusTensor.GetValue(0);
        // if (GetBlockIdx() == 0 && (loopCnt < 5 || (loopCnt % 10000 == 0))) {
        //     PRINTF("[DBG-AG] ReadStatus rank=%u loop=%u flag=%f target=[%f,%f]\n",
        //            curRankId_, loopCnt, flag, minTarget, maxTarget);
        // }
        ++loopCnt;
    }

    // if (GetBlockIdx() == 0) {
    //     PRINTF("[DBG-AG] ReadStatus DONE rank=%u loop=%u flag=%f\n",
    //            curRankId_, loopCnt, flag);
    // }

    // 所有 rank 就绪，清零状态区
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(selfStatusWinTensor[offset], stateResetTensor_, statusCnt);
}

/*!
 * \brief 获取指定 rank 的 win 数据区 GM 地址
 *
 * 0 区数据：win 起始 + DATA_START_OFFSET (1MB)
 * 1 区数据：win 起始 + 0（直接从 win 起始开始）
 *
 * 注：windowsIn/Out 数组中存储的是 HCCL 分配的 win 区基地址
 *
 * @param rankId  目标 rank ID
 * @param winFlag 0=使用 0 区，1=使用 1 区
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<AllGatherTemplateType>::GetWinDataAddrGm(
    uint32_t rankId, uint32_t winFlag)
{
    if (winFlag == 0) {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId] + DATA_START_OFFSET);
    } else {
        return (GM_ADDR)(hcclContext_->windowsOut[rankId]);
    }
}

/*!
 * \brief 获取指定 rank 的 win 状态区 GM 地址
 * 0 区状态：win 起始 + 0
 * 1 区状态：win 起始 + WIN_STATUS_OFFSET (400KB)
 *
 * @param rankId  目标 rank ID
 * @param winFlag 0=使用 0 区状态，1=使用 1 区状态
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<AllGatherTemplateType>::GetWinStatusAddrGm(
    uint32_t rankId, uint32_t winFlag)
{
    return (GM_ADDR)(hcclContext_->windowsIn[rankId] + winFlag * WIN_STATUS_OFFSET);
}

/*!
 * \brief 获取指定 rank 的 0/1 分区标志位区 GM 地址
 *
 * V 核和 C 核使用不同的标志位区域：
 *   - AIV: 状态区 0 区 + WIN_AIV_ZERONE_FLAG_OFFSET (800KB)
 *   - AIC: 状态区 0 区 + WIN_AIC_ZERONE_FLAG_OFFSET (850KB)
 *
 * @param rankId  目标 rank ID
 */
template <AllGatherTemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<AllGatherTemplateType>::GetWinZeroneFlagAddrGm(uint32_t rankId)
{
    if ASCEND_IS_AIV {
        return (GM_ADDR)(GetWinStatusAddrGm(rankId, 0) + WIN_AIV_ZERONE_FLAG_OFFSET);
    } else {
        return (GM_ADDR)(GetWinStatusAddrGm(rankId, 0) + WIN_AIC_ZERONE_FLAG_OFFSET);
    }
}
} // QuantMTECommImpl
#endif  // ALL_GATHER_MTE_COMM_H
