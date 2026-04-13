/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_HCCL_IMPL_H
#define MC2_HCCL_IMPL_H

#include "../a2av_gmm_utils.h"
#include "kernel_operator.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/hccl/hccl.h"
#include "../common/a2av_common_tiling.h"

using namespace AscendC;

namespace MC2KernelTemplate {

template <typename T1, typename T2>
__aicore__ inline T1 Mc2Min(T1 a, T2 b)
{
    return (a < b) ? a : b;
}

template <typename hcclDataType, bool commBeforeComputeFlag>
class HcclA2avOp {
public:
    __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset,
                                const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer, GM_ADDR recvBuffer,
                                GM_ADDR preCommBuffer, GM_ADDR workspaceGM)
    {
        taskTilingInfo_ = taskTilingInfo;
        workspaceGM_ = workspaceGM;
        e_ = taskTilingInfo_->e;
        epWorldSize_ = taskTilingInfo_->epWorldSize;
        H1_ = taskTilingInfo_->H1;
        N1_ = taskTilingInfo_->N1;
        sendGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)sendBuffer);
        recvGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)recvBuffer);
        preCommSendGlobalBuffer_.SetGlobalBuffer((__gm__ uint64_t *)preCommBuffer);
        preCommRecvGlobalBuffer_.SetGlobalBuffer(
            (__gm__ uint64_t *)(preCommBuffer + epWorldSize_ * sizeof(uint64_t)));

        GM_ADDR hcclContextGm = GetHcclContext<HCCL_GROUP_ID_0>();
        hccl_.InitV2(hcclContextGm, hcclInitTiling);
        hccl_.SetCcTilingV2(hcclCcTilingOffset);
        rankId_ = hccl_.GetRankId();
        rankDim_ = hccl_.GetRankDim();

        if constexpr (AscendC::IsSameType<hcclDataType, bfloat16_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_BFP16;
        } else if constexpr (AscendC::IsSameType<hcclDataType, hifloat8_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_HIF8;
        } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e5m2_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_FP8E5M2;
        } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e4m3fn_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_FP8E4M3;
        } else {
            hcclDataType_ = HCCL_DATA_TYPE_FP16;
        }

        // 打印初始化信息
        AscendC::printf("=== HCCL Init ===\n");
        AscendC::printf("rankId=%u, rankDim=%u, e=%u, H1=%lu, N1=%lu\n",
               rankId_, rankDim_, e_, H1_, N1_);
        AscendC::printf("sendBuffer=%p, recvBuffer=%p, workspace=%p\n", sendBuffer, recvBuffer, workspaceGM_);
    }

    __aicore__ inline void InitPreComm()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        AscendC::printf("=== InitPreComm rank=%u ===\n", rankId_);
        const auto *recvCnt = &taskTilingInfo_->recvCnt[0];
        
        // 计算本卡总接收量
        uint64_t myRecvSum = 0;
        for (uint32_t i = 0U; i < e_ * epWorldSize_; i++) {
            myRecvSum += static_cast<uint64_t>(recvCnt[i]);
        }
        AscendC::printf("myRecvSum=%lu\n", myRecvSum);
        
        // 发送本卡的 recvSum 给所有卡
        for (uint32_t i = 0U; i < epWorldSize_; i++) {
            preCommSendGlobalBuffer_.SetValue(i, myRecvSum);
        }
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(preCommSendGlobalBuffer_);

        // 注释掉实际通信，仅打印
        AscendC::printf("InitPreComm: preRecvCountHandle simulated\n");
        // preRecvCountHandle_ = hccl_.AlltoAllV<true>(...);
    }

    __aicore__ inline void WaitPreComm()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        AscendC::printf("WaitPreComm: simulated wait\n");
        // hccl_.Wait(preRecvCountHandle_);
        
        // 从预通信结果中读取所有卡的 recvSum
        // 计算最大接收卡和 totalRound
        uint64_t maxRecvSum = 0;
        uint32_t maxRecvRank = 0;
        for (uint32_t i = 0U; i < epWorldSize_; i++) {
            uint64_t recvSum = preCommRecvGlobalBuffer_.GetValue(i);
            AscendC::printf("rank%u recvSum=%lu\n", i, recvSum);
            if (recvSum > maxRecvSum) {
                maxRecvSum = recvSum;
                maxRecvRank = i;
            }
        }
        
        // 最大卡 baseA 固定 2048
        baseA_ = 2048U;
        
        // 计算总轮次
        totalRound_ = static_cast<uint32_t>((maxRecvSum + baseA_ - 1) / baseA_);
        
        AscendC::printf("maxRecvRank=%u, maxRecvSum=%lu, baseA=%u, totalRound=%u\n",
               maxRecvRank, maxRecvSum, baseA_, totalRound_);
    }
    
    __aicore__ inline uint32_t GetBaseA() const { return baseA_; }
    __aicore__ inline uint32_t GetTotalRound() const { return totalRound_; }

    __aicore__ inline void LaunchMultiRound(uint32_t roundIdx, uint32_t slotIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        LaunchCommMultiRound(roundIdx, slotIdx);
    }

    __aicore__ inline void WaitMultiRound(uint32_t slotIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        AscendC::printf("WaitMultiRound slotIdx=%u: simulated wait\n", slotIdx);
        // hccl_.Wait(alltoAllvHandleQueue_[slotIdx]);
    }

    __aicore__ inline void Launch(uint32_t startExpertIdx, uint32_t expertNum)
    {
        LaunchMultiRound(0, 0);
    }

    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        WaitMultiRound(0);
    }

    __aicore__ inline void InitScaleBuffer(GM_ADDR sendBuffer, GM_ADDR recvBuffer)
    {
        sendScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)sendBuffer);
        recvScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)recvBuffer);
        AscendC::printf("InitScaleBuffer: sendScale=%p, recvScale=%p\n", sendBuffer, recvBuffer);
    }

    __aicore__ inline void LaunchScaleMultiRound(uint32_t roundIdx, uint32_t slotIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        AscendC::printf("=== LaunchScaleMultiRound round=%u rank=%u ===\n", roundIdx, rankId_);

        for (uint32_t i = 0U; i < rankDim_; i++) {
            alltoAllvScaleSendCnt[i] = 0UL;
            alltoAllvScaleRecvCnt[i] = 0UL;
        }

        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        const auto *recvCnt = &taskTilingInfo_->recvCnt[0];
        uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * SCALE_MULTIPLIER;

        AscendC::printf("Scale Send/Recv per target rank (axis=%lu):\n", axis);
        for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
            for (uint32_t rankIdx = 0U; rankIdx < rankDim_; rankIdx++) {
                int32_t sendToRank = sendCnt[expertIdx + rankIdx * e_];
                uint32_t startSend = roundIdx * baseA_;
                uint32_t endSend = Mc2Min((roundIdx + 1) * baseA_, static_cast<uint32_t>(sendToRank));
                uint64_t sendM = (endSend > startSend) ? (endSend - startSend) : 0;

                int32_t recvFromRank = recvCnt[expertIdx + rankIdx * e_];
                uint32_t startRecv = roundIdx * baseA_;
                uint32_t endRecv = Mc2Min((roundIdx + 1) * baseA_, static_cast<uint32_t>(recvFromRank));
                uint64_t recvM = (endRecv > startRecv) ? (endRecv - startRecv) : 0;

                alltoAllvScaleSendCnt[rankIdx] += sendM * axis;
                alltoAllvScaleRecvCnt[rankIdx] += recvM * axis;

                if (sendM > 0 || recvM > 0) {
                    AscendC::printf("  expert%u: sendToRank%u=%lu(recvNeed), recvFromRank%u=%lu\n",
                           expertIdx, rankIdx, sendM, rankIdx, recvM);
                }
            }
        }

        AscendC::printf("Scale SendCnt: ");
        for (uint32_t i = 0U; i < rankDim_; i++) {
            AscendC::printf("%lu ", alltoAllvScaleSendCnt[i]);
        }
        AscendC::printf("\n");
        AscendC::printf("Scale RecvCnt: ");
        for (uint32_t i = 0U; i < rankDim_; i++) {
            AscendC::printf("%lu ", alltoAllvScaleRecvCnt[i]);
        }
        AscendC::printf("\n");

        // 注释掉实际通信
        AscendC::printf("LaunchScaleMultiRound: simulated (AlltoAllV not called)\n");
        // alltoAllvScaleHandleQueue_[slotIdx] = hccl_.AlltoAllV<true>(...);
    }

    __aicore__ inline void WaitScaleMultiRound(uint32_t slotIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        AscendC::printf("WaitScaleMultiRound slotIdx=%u: simulated wait\n", slotIdx);
        // hccl_.Wait(alltoAllvScaleHandleQueue_[slotIdx]);
    }

    __aicore__ inline void WaitAll(uint32_t allNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        AscendC::printf("WaitAll allNum=%u: simulated\n", allNum);
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::printf("=== HCCL End rank=%u ===\n", rankId_);
        SyncAll<true>();
        hccl_.Finalize();
    }

private:
    __aicore__ inline void LaunchCommMultiRound(uint32_t roundIdx, uint32_t slotIdx)
    {
        AscendC::printf("\n=== LaunchCommMultiRound round=%u rank=%u ===\n", roundIdx, rankId_);

        for (uint32_t i = 0U; i < rankDim_; i++) {
            alltoAllvSendCnt[i] = 0UL;
            alltoAllvRecvCnt[i] = 0UL;
        }

        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        const auto *recvCnt = &taskTilingInfo_->recvCnt[0];
        uint64_t axis = CeilDiv(H1_, PACK_FACTOR);

        AscendC::printf("PACK_FACTOR=%u, axis=%lu\n", PACK_FACTOR, axis);
        AscendC::printf("sendCnt array: ");
        for (uint32_t i = 0U; i < e_ * epWorldSize_; i++) {
            AscendC::printf("%d ", sendCnt[i]);
        }
        AscendC::printf("\n");
        AscendC::printf("recvCnt array: ");
        for (uint32_t i = 0U; i < e_ * epWorldSize_; i++) {
            AscendC::printf("%d ", recvCnt[i]);
        }
        AscendC::printf("\n");

        AscendC::printf("\n--- Send/Recv per target rank (本轮发送/接收量) ---\n");
        for (uint32_t rankIdx = 0U; rankIdx < rankDim_; rankIdx++) {
            uint64_t totalSendToRank = 0;
            uint64_t totalRecvFromRank = 0;
            AscendC::printf("目标卡%u: ", rankIdx);
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                int32_t sendToRank = sendCnt[expertIdx + rankIdx * e_];
                uint32_t startSend = roundIdx * baseA_;
                uint32_t endSend = Mc2Min((roundIdx + 1) * baseA_, static_cast<uint32_t>(sendToRank));
                uint64_t sendM = (endSend > startSend) ? (endSend - startSend) : 0;

                int32_t recvFromRank = recvCnt[expertIdx + rankIdx * e_];
                uint32_t startRecv = roundIdx * baseA_;
                uint32_t endRecv = Mc2Min((roundIdx + 1) * baseA_, static_cast<uint32_t>(recvFromRank));
                uint64_t recvM = (endRecv > startRecv) ? (endRecv - startRecv) : 0;

                if (sendM > 0) {
                    AscendC::printf("send(e%u)=%lu ", expertIdx, sendM);
                }
                if (recvM > 0) {
                    AscendC::printf("recv(e%u)=%lu ", expertIdx, recvM);
                }
                totalSendToRank += sendM;
                totalRecvFromRank += recvM;
            }
            AscendC::printf(" | totalSend=%lu, totalRecv=%lu\n", totalSendToRank, totalRecvFromRank);
            alltoAllvSendCnt[rankIdx] = totalSendToRank * axis;
            alltoAllvRecvCnt[rankIdx] = totalRecvFromRank * axis;
        }

        AscendC::printf("\n--- AlltoAllV SendCnt/RecvCnt (实际通信量) ---\n");
        AscendC::printf("SendCnt: ");
        for (uint32_t i = 0U; i < rankDim_; i++) {
            AscendC::printf("%lu ", alltoAllvSendCnt[i]);
        }
        AscendC::printf("\n");
        AscendC::printf("RecvCnt: ");
        for (uint32_t i = 0U; i < rankDim_; i++) {
            AscendC::printf("%lu ", alltoAllvRecvCnt[i]);
        }
        AscendC::printf("\n");

        alltoAllvSendOffset[0] = 0UL;
        for (uint32_t round = 0U; round < roundIdx; round++) {
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                for (uint32_t rank = 0U; rank < rankDim_; rank++) {
                    int32_t cnt = sendCnt[expertIdx + rank * e_];
                    alltoAllvSendOffset[0] += static_cast<uint64_t>(Mc2Min(cnt, static_cast<int32_t>(baseA_))) * axis;
                }
            }
        }

        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
            for (uint32_t round = 0U; round < roundIdx; round++) {
                for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                    int32_t cnt = sendCnt[expertIdx + (rankDim_ - i) * e_];
                    alltoAllvSendOffset[i] += static_cast<uint64_t>(Mc2Min(cnt, static_cast<int32_t>(baseA_))) * axis;
                }
            }
        }

        alltoAllvRecvOffset[0] = 0UL;
        for (uint32_t round = 0U; round < roundIdx; round++) {
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                for (uint32_t rank = 0U; rank < rankDim_; rank++) {
                    int32_t cnt = recvCnt[expertIdx + rank * e_];
                    alltoAllvRecvOffset[0] += static_cast<uint64_t>(Mc2Min(cnt, static_cast<int32_t>(baseA_))) * axis;
                }
            }
        }

        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvRecvOffset[i] = alltoAllvRecvOffset[i - 1U];
            for (uint32_t round = 0U; round < roundIdx; round++) {
                for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                    int32_t cnt = recvCnt[expertIdx + (rankDim_ - i) * e_];
                    alltoAllvRecvOffset[i] += static_cast<uint64_t>(Mc2Min(cnt, static_cast<int32_t>(baseA_))) * axis;
                }
            }
        }

        AscendC::printf("\nSendOffset: ");
        for (uint32_t i = 0U; i < rankDim_; i++) {
            AscendC::printf("%lu ", alltoAllvSendOffset[i]);
        }
        AscendC::printf("\n");
        AscendC::printf("RecvOffset: ");
        for (uint32_t i = 0U; i < rankDim_; i++) {
            AscendC::printf("%lu ", alltoAllvRecvOffset[i]);
        }
        AscendC::printf("\n");

        AscendC::printf("\n--- GM Address Info ---\n");
        AscendC::printf("sendBuffer PhyAddr=%p\n", sendGlobalBuffer_.GetPhyAddr());
        AscendC::printf("recvBuffer PhyAddr=%p\n", recvGlobalBuffer_.GetPhyAddr());
        AscendC::printf("workspace=%p\n", workspaceGM_);

        // 注释掉实际通信
        AscendC::printf("\nLaunchCommMultiRound: AlltoAllV simulated (not actually called)\n");
        // alltoAllvHandleQueue_[slotIdx] = hccl_.AlltoAllV<true>(...);
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;
#endif

    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
    static constexpr uint64_t HANDLE_QUEUE_SIZE = 64U;
    static constexpr uint64_t SCALE_ALIGNMENT_BLOCK_SIZE = 64U;
    static constexpr uint64_t SCALE_MULTIPLIER = 2U;

    const TaskTilingInfo *taskTilingInfo_;
    GM_ADDR workspaceGM_ = nullptr;

    uint32_t rankId_ = 0U;
    uint32_t rankDim_ = 0U;
    uint32_t e_ = 0U;
    uint32_t epWorldSize_ = 0U;
    uint64_t H1_ = 0UL;
    uint64_t N1_ = 0UL;
    uint32_t baseA_ = 0U;
    uint32_t totalRound_ = 0U;

    GlobalTensor<hcclDataType> sendGlobalBuffer_;
    GlobalTensor<hcclDataType> recvGlobalBuffer_;

    GlobalTensor<fp8_e8m0_t> sendScaleGlobalBuffer_;
    GlobalTensor<fp8_e8m0_t> recvScaleGlobalBuffer_;
    GlobalTensor<uint64_t> preCommSendGlobalBuffer_;
    GlobalTensor<uint64_t> preCommRecvGlobalBuffer_;

    HcclHandle alltoAllvHandleQueue_[HANDLE_QUEUE_SIZE] = {INVALID_HANDLE_ID};
    HcclHandle alltoAllvScaleHandleQueue_[HANDLE_QUEUE_SIZE] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;

    HcclHandle preRecvCountHandle_ = INVALID_HANDLE_ID;

    uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};

    uint64_t alltoAllvScaleSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvOffset[MAX_EP_RANK_SIZE] = {0UL};
};
}; // namespace MC2KernelTemplate

#endif