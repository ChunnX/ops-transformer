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

#ifndef TILINGKEY_TPL_CCU
#define TILINGKEY_TPL_CCU 0
#endif
#ifndef TILINGKEY_TPL_AICPU
#define TILINGKEY_TPL_AICPU 1
#endif

namespace MC2KernelTemplate {

template<int commMode = TILINGKEY_TPL_CCU>
struct HcclTypeSelector {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;
};

template<>
struct HcclTypeSelector<TILINGKEY_TPL_AICPU> {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};

template <typename hcclDataType, bool commBeforeComputeFlag, int commMode = TILINGKEY_TPL_CCU>
class HcclA2avOp {
public:
    __aicore__ inline void Init(const void *hcclInitTiling, uint64_t hcclCcTilingOffset,
                                const TaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer, GM_ADDR recvBuffer)
    {
        taskTilingInfo_ = taskTilingInfo;
        GM_ADDR hcclContextGm = GetHcclContext<HCCL_GROUP_ID_0>();
        hccl_.InitV2(hcclContextGm, hcclInitTiling);
        hccl_.SetCcTilingV2(hcclCcTilingOffset);
        rankId_ = hccl_.GetRankId();
        rankDim_ = hccl_.GetRankDim();
        e_ = taskTilingInfo_->e;
        H1_ = taskTilingInfo_->H1;
        N1_ = taskTilingInfo_->N1;
        sendGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)sendBuffer);
        recvGlobalBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)recvBuffer);
        for (int i = 0; i < e_ * rankDim_; i++) {
            recvCounts_[i] = taskTilingInfo_->recvCnt[i];
        }
        if constexpr (AscendC::IsSameType<hcclDataType, bfloat16_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_BFP16;
        } else if constexpr (AscendC::IsSameType<hcclDataType, half>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_FP16;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        } else if constexpr (AscendC::IsSameType<hcclDataType, hifloat8_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_HIF8;
        } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e5m2_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_FP8E5M2;
        } else if constexpr (AscendC::IsSameType<hcclDataType, fp8_e4m3fn_t>::value) {
            hcclDataType_ = HCCL_DATA_TYPE_FP8E4M3;
#endif
        } else {
            hcclDataType_ = HCCL_DATA_TYPE_INT8;
        }
    }

    __aicore__ inline void Launch(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        // comm and compute order
        if constexpr (commBeforeComputeFlag) {
            LaunchCommBeforeCompute(startExpertIdx, expertNum);
        } else {
            LaunchCommAfterCompute(startExpertIdx, expertNum);
        }
    }

    __aicore__ inline void InitScaleBuffer(GM_ADDR sendBuffer, GM_ADDR commOutBuffer, GM_ADDR permuteOutBuffer)
    {
        sendScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)sendBuffer);
        recvScaleGlobalBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)commOutBuffer);
        scalePermuteOutBuffer_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)permuteOutBuffer);
        TPipe *pipe = GetTPipePtr();
        pipe->InitBuffer(scaleTBuf_, taskTilingInfo_->ubSize);
    }

    __aicore__ inline void PermuteScale()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }

        uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;
        uint64_t scaleUbSize = taskTilingInfo_->ubSize;
        uint64_t recvSumCntBefore[MAX_EXPERT_SIZE] = {0UL};
        uint64_t recvSumCntAfter[MAX_EXPERT_SIZE] = {0UL};

        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                uint32_t posBefore = rankIndex * e_ + expertIdx;
                recvSumCntBefore[posBefore] = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + expertIdx]);
                if (posBefore != 0U) {
                    recvSumCntBefore[posBefore] += recvSumCntBefore[posBefore - 1U];
                }
            }
        }

        for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
            for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
                uint32_t posAfter = expertIdx * rankDim_ + rankIndex;
                recvSumCntAfter[posAfter] = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + expertIdx]);
                if (posAfter != 0U) {
                    recvSumCntAfter[posAfter] += recvSumCntAfter[posAfter - 1U];
                }
            }
        }

        TPipe *permutePipe = GetTPipePtr();
        LocalTensor<fp8_e8m0_t> scaleUbTensor = scaleTBuf_.Get<fp8_e8m0_t>();
        constexpr uint64_t dataCopyMaxBlockLen = 65535UL;
        uint64_t actualTileSize = (scaleUbSize < dataCopyMaxBlockLen) ? scaleUbSize : dataCopyMaxBlockLen;
        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
                uint32_t posBefore = rankIndex * e_ + expertIdx;
                uint32_t posAfter = expertIdx * rankDim_ + rankIndex;
                uint64_t srcOffset = (posBefore == 0U) ? 0UL : recvSumCntBefore[posBefore - 1U] * axis;
                uint64_t dstOffset = (posAfter == 0U) ? 0UL : recvSumCntAfter[posAfter - 1U] * axis;
                uint64_t totalCnt = static_cast<uint64_t>(recvCounts_[rankIndex * e_ + expertIdx]) * axis;
                uint64_t tileNum = CeilDiv(totalCnt, actualTileSize);
                for (uint64_t tile = 0UL; tile < tileNum; tile++) {
                    uint64_t realLength = actualTileSize;
                    if (tile == tileNum - 1UL) {
                        realLength = totalCnt - tile * actualTileSize;
                    }
                    DataCopyParams dataCopyParams{1U, static_cast<uint16_t>(realLength), 0U, 0U};
                    DataCopyPadParams dataCopyPadParams{false, 0U, 0U, 0U};
                    // 从commOutBuffer读取（按rank排列）GM->UB
                    DataCopyPad(scaleUbTensor,
                                recvScaleGlobalBuffer_[srcOffset + tile * actualTileSize], dataCopyParams,
                                dataCopyPadParams);
                    eventID_ = static_cast<int32_t>(permutePipe->FetchEventID<HardEvent::MTE2_MTE3>());
                    SetFlag<HardEvent::MTE2_MTE3>(eventID_);
                    WaitFlag<HardEvent::MTE2_MTE3>(eventID_);
                    // 写入permuteOutBuffer（按专家排列）UB->GM
                    DataCopyPad(scalePermuteOutBuffer_[dstOffset + tile * actualTileSize],
                                scaleUbTensor, dataCopyParams);
                    eventID_ = static_cast<int32_t>(permutePipe->FetchEventID<HardEvent::MTE3_MTE2>());
                    SetFlag<HardEvent::MTE3_MTE2>(eventID_);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventID_);
                }
            }
        }
    }

    __aicore__ inline void LaunchScaleBeforeCompute(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }

        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        uint64_t axis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;

        for (uint64_t i = 0UL; i < rankDim_; i++) {
            alltoAllvScaleSendCnt[i] = 0UL;
            alltoAllvScaleRecvCnt[i] = 0UL;
            for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                alltoAllvScaleSendCnt[i] += static_cast<uint64_t>(sendCnt[expertIdx + i * e_]) * axis;
                alltoAllvScaleRecvCnt[i] += static_cast<uint64_t>(recvCounts_[expertIdx + i * e_]) * axis;
            }
        }

        alltoAllvScaleSendOffset[0] = 0UL;
        for (uint32_t j = 0U; j < startExpertIdx; j++) {
            alltoAllvScaleSendOffset[0] += static_cast<uint64_t>(sendCnt[j]) * axis;
        }
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvScaleSendOffset[i] = alltoAllvScaleSendOffset[i - 1U];
            for (uint32_t j = 0U; j < e_; j++) {
                alltoAllvScaleSendOffset[i] +=
                    static_cast<uint64_t>(sendCnt[startExpertIdx + (i - 1U) * e_ + j]) * axis;
            }
        }

        for (uint32_t i = 0U; i < rankDim_; i++) {
            if ((startExpertIdx == 0U) && (i == 0U)) {
                alltoAllvScaleRecvOffset[i] = 0UL;
                alltoAllvScaleRecvOffsetLastSum += alltoAllvScaleRecvCnt[0];
            } else {
                alltoAllvScaleRecvOffset[i] = alltoAllvScaleRecvOffsetLastSum;
                alltoAllvScaleRecvOffsetLastSum += alltoAllvScaleRecvCnt[i];
            }
        }

        alltoAllvScaleHandleId_[startExpertIdx] = hccl_.template AlltoAllV<true>(
            (__gm__ uint8_t *)sendScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleSendCnt, alltoAllvScaleSendOffset,
            HCCL_DATA_TYPE_FP8E8M0, (__gm__ uint8_t *)recvScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleRecvCnt,
            alltoAllvScaleRecvOffset, HCCL_DATA_TYPE_FP8E8M0);
    }

    __aicore__ inline void WaitScale(uint32_t startExpertIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(alltoAllvScaleHandleId_[startExpertIdx]);
    }

    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(alltoAllvHandleId_[startExpertIdx]);
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
        for (uint32_t i = 0U; i < allNum; i++) {
            hccl_.Wait(alltoAllvHandleId_[i]);
        }
    }

    __aicore__ inline void LaunchCombinedAic(uint32_t turnIdx, uint32_t realCombineSize)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if ASCEND_IS_AIC {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        uint32_t startExpertIdx = turnIdx * taskTilingInfo_->combineExpertSize;
        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        uint64_t axis = CeilDiv(H1_, PACK_FACTOR);
        for (uint64_t i = 0UL; i < rankDim_; i++) {
            alltoAllvSendCnt[i] = 0UL;
            alltoAllvRecvCnt[i] = 0UL;
            for (uint64_t j = 0UL; j < realCombineSize; j++) {
                alltoAllvSendCnt[i] += static_cast<uint64_t>(sendCnt[startExpertIdx + j + i * e_]) * axis;
                alltoAllvRecvCnt[i] += static_cast<uint64_t>(recvCounts_[startExpertIdx + j + i * e_]) * axis;
            }
        }
        alltoAllvSendOffset[0] = 0UL;
        for (uint32_t j = 0U; j < startExpertIdx; j++) {
            alltoAllvSendOffset[0] += static_cast<uint64_t>(sendCnt[j]) * axis;
        }
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
            for (uint32_t j = 0U; j < e_; j++) {
                alltoAllvSendOffset[i] += static_cast<uint64_t>(sendCnt[startExpertIdx + (i - 1U) * e_ + j]) * axis;
            }
        }
        for (uint32_t i = 0U; i < rankDim_; i++) {
            alltoAllvRecvOffset[i] = alltoAllvRecvOffsetLastSum;
            alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[i];
        }
        // AscendC::printf(
        //     "alltoAllvHandleId_[%d], sendGlobalBuffer_.GetPhyAddr() = [%d], recvGlobalBuffer_.GetPhyAddr() = [%d]\n",
        //     turnIdx, sendGlobalBuffer_.GetPhyAddr(), recvGlobalBuffer_.GetPhyAddr());
        // AscendC::printf("alltoallv sendGlobalBuffer_ turnIdx = %d\n", turnIdx);
        AscendC::DumpTensor(sendGlobalBuffer_, 0, 16);
        alltoAllvHandleId_[turnIdx] = hccl_.template AlltoAllV<true>(
            (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
            (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
    }

    __aicore__ inline void WaitCombinedAic(uint32_t turnIdx)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if ASCEND_IS_AIC {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(alltoAllvHandleId_[turnIdx]);
    }

    __aicore__ inline void LaunchScaleCombinedAic(uint32_t turnIdx, uint32_t realCombineSize)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if ASCEND_IS_AIC {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        uint32_t startExpertIdx = turnIdx * taskTilingInfo_->combineExpertSize;
        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        uint64_t scaleAxis = CeilDiv(H1_, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;
        for (uint64_t i = 0UL; i < rankDim_; i++) {
            alltoAllvScaleSendCnt[i] = 0UL;
            alltoAllvScaleRecvCnt[i] = 0UL;
            for (uint64_t j = 0UL; j < realCombineSize; j++) {
                alltoAllvScaleSendCnt[i] += static_cast<uint64_t>(sendCnt[startExpertIdx + j + i * e_]) * scaleAxis;
                alltoAllvScaleRecvCnt[i] += static_cast<uint64_t>(recvCounts_[startExpertIdx + j + i * e_]) * scaleAxis;
            }
        }
        alltoAllvScaleSendOffset[0] = 0UL;
        for (uint32_t j = 0U; j < startExpertIdx; j++) {
            alltoAllvScaleSendOffset[0] += static_cast<uint64_t>(sendCnt[j]) * scaleAxis;
        }
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvScaleSendOffset[i] = alltoAllvScaleSendOffset[i - 1U];
            for (uint32_t j = 0U; j < e_; j++) {
                alltoAllvScaleSendOffset[i] += static_cast<uint64_t>(sendCnt[startExpertIdx + (i - 1U) * e_ + j]) * scaleAxis;
            }
        }
        for (uint32_t i = 0U; i < rankDim_; i++) {
            alltoAllvScaleRecvOffset[i] = alltoAllvScaleRecvOffsetLastSum;
            alltoAllvScaleRecvOffsetLastSum += alltoAllvScaleRecvCnt[i];
        }
        alltoAllvScaleHandleId_[turnIdx] = hccl_.template AlltoAllV<true>(
            (__gm__ uint8_t *)sendScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleSendCnt, alltoAllvScaleSendOffset,
            HCCL_DATA_TYPE_FP8E8M0,
            (__gm__ uint8_t *)recvScaleGlobalBuffer_.GetPhyAddr(), alltoAllvScaleRecvCnt, alltoAllvScaleRecvOffset,
            HCCL_DATA_TYPE_FP8E8M0);
    }

    __aicore__ inline void WaitScaleCombinedAic(uint32_t turnIdx)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if ASCEND_IS_AIC {
            if (GetBlockIdx() != 0) {
                return;
            }
        }
        hccl_.Wait(alltoAllvScaleHandleId_[turnIdx]);
    }

    __aicore__ inline void InitOutputPermuteBuffer(GM_ADDR srcBuffer, GM_ADDR dstBuffer)
    {
        if ASCEND_IS_AIC {
            return;
        }
        outputPermuteSrcBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)srcBuffer);
        outputPermuteDstBuffer_.SetGlobalBuffer((__gm__ hcclDataType *)dstBuffer);
        TPipe *pipe = GetTPipePtr();
        uint64_t halfUbLen = taskTilingInfo_->ubSize / 2;
        pipe->InitBuffer(outputPermuteBufPing_, halfUbLen);
        pipe->InitBuffer(outputPermuteBufPong_, halfUbLen);
    }

    __aicore__ inline void PermuteOutput(uint32_t turnIdx, uint32_t realCombineSize)
    {
        if ASCEND_IS_AIC {
            return;
        }
        // AscendC::printf("outputPermuteSrcBuffer_.GetPhyAddr() = %d\n", outputPermuteSrcBuffer_.GetPhyAddr());
        // AscendC::printf("outputPermuteDstBuffer_.GetPhyAddr() = %d\n", outputPermuteDstBuffer_.GetPhyAddr());

        uint32_t startExpertIdx = turnIdx * taskTilingInfo_->combineExpertSize;
        uint64_t axis = CeilDiv(H1_, PACK_FACTOR);
        static constexpr uint64_t DATACOPY_MAX_BLOCK_LEN = 65535UL;

        uint32_t totalPairs     = realCombineSize * rankDim_; // 总块数
        uint32_t pairsPerBlock  = totalPairs / GetBlockNum(); // 每个核处理的块数
        uint32_t pairsRemainder = totalPairs % GetBlockNum(); // 剩余块数
        uint32_t pairStart      = GetBlockIdx() * pairsPerBlock +
                                (GetBlockIdx() < pairsRemainder ? GetBlockIdx() : pairsRemainder);
        uint32_t pairEnd        = pairStart + pairsPerBlock +
                                (GetBlockIdx() < pairsRemainder ? 1U : 0U);
        AscendC::printf("PermuteOutput pairStart = %d\n, pairEnd = %d\n", pairStart, pairEnd);

        uint64_t recvSumCntSrc[MAX_EXPERT_SIZE] = {0UL};
        uint64_t recvSumCntDst[MAX_EXPERT_SIZE] = {0UL};
        for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
            for (uint32_t localIdx = 0U; localIdx < realCombineSize; localIdx++) {
                uint32_t posBefore = rankIndex * realCombineSize + localIdx;
                uint32_t globalExpertIdx = startExpertIdx + localIdx;
                uint64_t cnt = static_cast<uint64_t>(recvCounts_[globalExpertIdx + rankIndex * e_]);
                recvSumCntSrc[posBefore] = cnt * axis;
                if (posBefore != 0U) {
                    recvSumCntSrc[posBefore] += recvSumCntSrc[posBefore - 1U];
                }
            }
        }
        for (uint32_t localIdx = 0U; localIdx < realCombineSize; localIdx++) {
            for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
                uint32_t posAfter = localIdx * rankDim_ + rankIndex;
                uint32_t globalExpertIdx = startExpertIdx + localIdx;
                recvSumCntDst[posAfter] = static_cast<uint64_t>(recvCounts_[globalExpertIdx + rankIndex * e_]) * axis;
                if (posAfter != 0U) {
                    recvSumCntDst[posAfter] += recvSumCntDst[posAfter - 1U];
                }
            }
        }
        TPipe *permutePipe = GetTPipePtr();
        LocalTensor<hcclDataType> ubBuf = outputPermuteBufPing_.Get<hcclDataType>();
        for (uint32_t pairIdx = pairStart; pairIdx < pairEnd; pairIdx++) {
            uint32_t localIdx  = pairIdx % realCombineSize;
            uint32_t rankIndex = pairIdx / realCombineSize;
            uint32_t posBefore = rankIndex * realCombineSize + localIdx;
            uint32_t posAfter  = localIdx * rankDim_ + rankIndex;
            uint32_t globalExpertIdx = startExpertIdx + localIdx;
            uint64_t srcOffset = outputPermuteSrcBaseOffset_;
            if (posBefore != 0U) {
                srcOffset += recvSumCntSrc[posBefore - 1U];
            }
            uint64_t dstOffset = outputPermuteDstBaseOffset_;
            if (posAfter != 0U) {
                dstOffset += recvSumCntDst[posAfter - 1U];
            }
            uint64_t totalCnt = static_cast<uint64_t>(recvCounts_[globalExpertIdx + rankIndex * e_]) * axis;

            if (totalCnt == 0UL) {
                continue;
            }

            uint64_t subBaseLen  = totalCnt / GetSubBlockNum();
            uint64_t subRemainder = totalCnt % GetSubBlockNum();
            uint64_t subStart = GetSubBlockIdx() * subBaseLen +
                                (GetSubBlockIdx() < subRemainder ? GetSubBlockIdx() : subRemainder);
            uint64_t subLen   = subBaseLen + ((GetSubBlockIdx() < subRemainder) ? 1UL : 0UL);

            for (uint64_t tile = 0UL; tile * DATACOPY_MAX_BLOCK_LEN < subLen; tile++) {
                uint64_t tileStart = subStart + tile * DATACOPY_MAX_BLOCK_LEN;
                uint64_t remaining = subLen - tile * DATACOPY_MAX_BLOCK_LEN;
                uint64_t tileLen = (remaining >= DATACOPY_MAX_BLOCK_LEN) ? DATACOPY_MAX_BLOCK_LEN : remaining;

                DataCopyParams dataCopyParams{static_cast<uint16_t>(tileLen), 0U, 0U, 0U};
                DataCopyPadParams dataCopyPadParams{false, 0U, 0U, 0U};
                AscendC::printf("DataCopyPad outputPermuteSrcBuffer_ srcOffset + tileStart = %d\n", srcOffset + tileStart);
                DataCopyPad(ubBuf, outputPermuteSrcBuffer_[srcOffset + tileStart],
                            dataCopyParams, dataCopyPadParams);
                AscendC::DumpTensor(outputPermuteSrcBuffer_[srcOffset + tileStart], 111, 16);
                for (int i = 0; i < 16; i++) {
                    AscendC::printf("DataCopyPad ubBuf[%d] = %d\n", i, ubBuf.GetValue(i));
                }
                int32_t readEvent = static_cast<int32_t>(permutePipe->FetchEventID<HardEvent::MTE2_MTE3>());
                SetFlag<HardEvent::MTE2_MTE3>(readEvent);
                WaitFlag<HardEvent::MTE2_MTE3>(readEvent);
                AscendC::printf("DataCopyPad outputPermuteDstBuffer_ dstOffset + tileStart = %d\n", dstOffset + tileStart);
                DataCopyPad(outputPermuteDstBuffer_[dstOffset + tileStart], ubBuf, dataCopyParams);
                int32_t writeEvent = static_cast<int32_t>(permutePipe->FetchEventID<HardEvent::MTE3_MTE2>());
                SetFlag<HardEvent::MTE3_MTE2>(writeEvent);
                WaitFlag<HardEvent::MTE3_MTE2>(writeEvent);
            }
        }

        uint64_t turnTotalData = 0UL;
        for (uint32_t pairIdx = 0U; pairIdx < totalPairs; pairIdx++) {
            uint32_t localIdx  = pairIdx % realCombineSize;
            uint32_t rankIndex = pairIdx / realCombineSize;
            turnTotalData += static_cast<uint64_t>(recvCounts_[startExpertIdx + localIdx + rankIndex * e_]) * axis;
        }
        outputPermuteSrcBaseOffset_ += turnTotalData;
        outputPermuteDstBaseOffset_ += turnTotalData;

        SyncAll<true>();
    }

    __aicore__ inline void End()
    {
        SyncAll<false>();
        hccl_.Finalize();
    }

private:
    __aicore__ inline void LaunchCommBeforeCompute(uint32_t startExpertIdx, uint32_t expertNum)
    {
        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        // mxfp4 半字节需要除以2向上取整 (安全编码)
        uint64_t axis = CeilDiv(H1_, PACK_FACTOR);
        for (uint64_t i = 0UL; i < rankDim_; i++) {
            alltoAllvSendCnt[i] = 0UL;
            alltoAllvRecvCnt[i] = 0UL;
            for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                alltoAllvSendCnt[i] += static_cast<uint64_t>(sendCnt[expertIdx + i * e_]) * axis;
                alltoAllvRecvCnt[i] += static_cast<uint64_t>(recvCounts_[expertIdx + i * e_]) * axis;
            }
        }
        alltoAllvSendOffset[0] = 0UL;
        for (uint32_t j = 0U; j < startExpertIdx; j++) {
            alltoAllvSendOffset[0] += static_cast<uint64_t>(sendCnt[j]) * axis;
        }
        for (uint32_t i = 1U; i < rankDim_; i++) {
            alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
            for (uint32_t j = 0U; j < e_; j++) {
                alltoAllvSendOffset[i] += static_cast<uint64_t>(sendCnt[startExpertIdx + (i - 1U) * e_ + j]) * axis;
            }
        }

        for (uint32_t i = 0U; i < rankDim_; i++) {
            if ((startExpertIdx == 0U) && (i == 0U)) {
                alltoAllvRecvOffset[i] = 0UL;
                alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[0];
            } else {
                alltoAllvRecvOffset[i] = alltoAllvRecvOffsetLastSum;
                alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[i];
            }
        }
        alltoAllvHandleId_[startExpertIdx] = hccl_.template AlltoAllV<true>(
            (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
            (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
    }

    __aicore__ inline void LaunchCommAfterCompute(uint32_t startExpertIdx, uint32_t expertNum)
    {
        const auto *sendCnt = &taskTilingInfo_->sendCnt[0];
        uint64_t axis = N1_;
        for (uint64_t i = 0UL; i < rankDim_; i++) {
            alltoAllvSendCnt[i] = 0UL;
            alltoAllvRecvCnt[i] = 0UL;
            for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                alltoAllvSendCnt[i] += static_cast<uint64_t>(sendCnt[expertIdx + i * e_]) * axis;
                alltoAllvRecvCnt[i] += static_cast<uint64_t>(recvCounts_[expertIdx + i * e_]) * axis;
            }
        }
        uint64_t expertOffset = 0UL;
        for (uint64_t i = 0UL; i < startExpertIdx; i++) {
            for (uint64_t j = 0UL; j < rankDim_; j++) {
                expertOffset += static_cast<uint64_t>(sendCnt[i + j * e_]);
            }
        }
        alltoAllvSendOffset[0] = expertOffset * N1_;
        for (uint64_t i = 1UL; i < rankDim_; i++) {
            alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1];
            for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                alltoAllvSendOffset[i] += static_cast<uint64_t>(sendCnt[expertIdx + (i - 1) * e_]) * N1_;
            }
        }
        alltoAllvRecvOffset[0] = 0UL;
        for (uint64_t i = 0UL; i < startExpertIdx; i++) {
            alltoAllvRecvOffset[0] += static_cast<uint64_t>(recvCounts_[i]) * N1_;
        }
        for (uint64_t i = 1UL; i < rankDim_; i++) {
            alltoAllvRecvOffset[i] = alltoAllvRecvOffset[i - 1];
            for (uint64_t j = 0UL; j < e_; j++) {
                alltoAllvRecvOffset[i] += static_cast<uint64_t>(recvCounts_[startExpertIdx + (i - 1) * e_ + j]) * N1_;
            }
        }
        alltoAllvHandleId_[startExpertIdx] = hccl_.template AlltoAllV<true>(
            (__gm__ uint8_t *)sendGlobalBuffer_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
            (__gm__ uint8_t *)recvGlobalBuffer_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
    }

// AICPU only: 只支持AICPU通信模式
    typename HcclTypeSelector<commMode>::type hccl_;

    const TaskTilingInfo *taskTilingInfo_;

    uint32_t rankId_ = 0U;
    uint32_t rankDim_ = 0U;
    uint32_t e_ = 0U;
    uint64_t H1_ = 0UL;
    uint64_t N1_ = 0UL;

    GlobalTensor<hcclDataType> sendGlobalBuffer_;
    GlobalTensor<hcclDataType> recvGlobalBuffer_;

    GlobalTensor<fp8_e8m0_t> sendScaleGlobalBuffer_;
    GlobalTensor<fp8_e8m0_t> recvScaleGlobalBuffer_;
    GlobalTensor<fp8_e8m0_t> scalePermuteOutBuffer_;

    TBuf<QuePosition::VECIN> scaleTBuf_;

    int32_t eventID_ = 0;

    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclHandle alltoAllvScaleHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;

    uint64_t alltoAllvRecvOffsetLastSum = 0UL;
    uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};

    // scale相关的alltoall通信变量
    uint64_t alltoAllvScaleSendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleSendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t alltoAllvScaleRecvOffsetLastSum = 0UL;

    // recvCounts
    uint64_t recvCounts_[MAX_EXPERT_SIZE] = {0UL};

    GlobalTensor<hcclDataType> outputPermuteSrcBuffer_;
    GlobalTensor<hcclDataType> outputPermuteDstBuffer_;
    TBuf<QuePosition::VECIN> outputPermuteBufPing_;
    TBuf<QuePosition::VECIN> outputPermuteBufPong_;
    uint64_t outputPermuteSrcBaseOffset_ = 0UL;
    uint64_t outputPermuteDstBaseOffset_ = 0UL;
};
}; // namespace MC2KernelTemplate

#endif
