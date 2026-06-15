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
 * \file moe_mrgsort_out.h
 * \brief
 */
#ifndef MOE_MRGSORT_OUT_H
#define MOE_MRGSORT_OUT_H

#include "moe_mrgsort.h"
#include "kernel_operator.h"

namespace MoeTokenPermute {
using namespace AscendC;

template <typename T, typename T2>
class MoeMrgsortOut
{
public:
    __aicore__ inline MoeMrgsortOut(){};
    __aicore__ inline void Init(MoeMrgsortParam* param, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void Process(int64_t capacity);
    __aicore__ inline void SetInput(GlobalTensor<float>& gmInput, LocalTensor<float>& ubInput);
    __aicore__ inline void SetOutput(
        GlobalTensor<T>& gmOutput1, GlobalTensor<T2>& gmOutput2, LocalTensor<float>& ubOutput1,
        LocalTensor<float>& ubOutput2);
    __aicore__ inline void SetBuffer(LocalTensor<float>& tempBuffer);

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void UpdateMrgParam();
    __aicore__ inline void MrgsortCompute();
    __aicore__ inline void UpdateSortInfo();
    __aicore__ inline void Extract();
    __aicore__ inline void PadOutput();
    __aicore__ inline void PadEnsureBufSpace(int64_t& outPos, int64_t needSlots, int64_t bufCapacity);
    __aicore__ inline void PadEmitMinusOnes(
        int64_t& outPos, int64_t count, LocalTensor<int32_t>& outBuf, int64_t bufCapacity);
    __aicore__ inline int32_t PadGetTokenId(const LocalTensor<float>& tempFloat, int64_t pairIdx);
    __aicore__ inline bool PadResumeOngoing(
        LocalTensor<float>& tempFloat, LocalTensor<int32_t>& tempInt, LocalTensor<int32_t>& outBuf,
        int64_t topK, int64_t pairCount, int64_t bufCapacity, bool mergeFinished, int64_t& outPos,
        int64_t& pairIdx);
    __aicore__ inline bool PadProcessPairs(
        LocalTensor<float>& tempFloat, LocalTensor<int32_t>& tempInt, LocalTensor<int32_t>& outBuf,
        int64_t topK, int64_t pairCount, int64_t bufCapacity, bool mergeFinished, int64_t& outPos,
        int64_t& pairIdx);
    __aicore__ inline void PadFinalizeChunk(
        LocalTensor<int32_t>& outBuf, int64_t topK, int64_t numTokens, int64_t bufCapacity,
        bool mergeFinished, int64_t& outPos);
    __aicore__ inline void FlushPadBuf(int64_t count);
    __aicore__ inline void CopyOut();
    __aicore__ inline void CopyOutCapacity(int64_t capacity);
    __aicore__ inline void ClearCache();

private:
    MoeMrgsortParam* param = nullptr;

    GlobalTensor<float> gmInputs[4];
    GlobalTensor<T> gmOutput1;
    GlobalTensor<T2> gmOutput2;

    LocalTensor<float> ubInputs[4];
    LocalTensor<float> tempBuffer;
    LocalTensor<T2> ubOutputCast2;
    LocalTensor<float> tmpUbInputs[4];

    // for extract
    LocalTensor<float> ubOutput1;
    LocalTensor<uint32_t> ubOutput2;

    // for copy out
    LocalTensor<T> ubOutputCast1;

    int64_t listNum{0};
    int64_t remainListNum{0};
    int64_t outOffset{0};
    int64_t offsets[4];
    int64_t listRemainElements[4];
    int64_t lengths[4];
    int64_t allRemainElements{0};
    int64_t curLoopSortedNum{0};

    // for MrgSort
    uint16_t validBitTail;
    uint16_t elementCountListTail[4];
    uint32_t listSortedNums[4];

    event_t eventIdMte3ToMte2;
    event_t eventIdVToMte3;
    event_t eventIdMte2ToV;

    // needPad 时跨多轮归并的补齐状态（勿在 PadOutput 末尾清零 allRemainElements）
    int32_t padNextExpectedTokenId{0};
    int32_t padOngoingTokenId{-1};
    int64_t padOngoingEmittedCount{0};
};

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::ClearCache()
{
    this->listNum = 0;
    this->allRemainElements = 0;
    this->outOffset = 0;
    this->padNextExpectedTokenId = 0;
    this->padOngoingTokenId = -1;
    this->padOngoingEmittedCount = 0;
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::SetInput(GlobalTensor<float>& gmInput, LocalTensor<float>& ubInput)
{
    this->gmInputs[listNum] = gmInput;
    this->ubInputs[listNum] = ubInput;
    this->listNum += 1;
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::SetOutput(
    GlobalTensor<T>& gmOutput1, GlobalTensor<T2>& gmOutput2, LocalTensor<float>& ubOutput1,
    LocalTensor<float>& ubOutput2)
{
    this->gmOutput1 = gmOutput1;
    this->ubOutput1 = ubOutput1;
    this->ubOutputCast1 = ubOutput1.ReinterpretCast<T>();

    this->gmOutput2 = gmOutput2;
    this->ubOutput2 = ubOutput2.ReinterpretCast<uint32_t>();
    this->ubOutputCast2 = ubOutput2.ReinterpretCast<int32_t>();
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::SetBuffer(LocalTensor<float>& tempBuffer)
{
    this->tempBuffer = tempBuffer;
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::UpdateMrgParam()
{
    if (this->remainListNum == MERGE_LIST_TWO) {
        elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0011;
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0111;
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        validBitTail = 0b1111;
    } else {
        validBitTail = 0b0001;
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::CopyIn()
{
    this->remainListNum = 0;
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        lengths[i] = Min(param->oneLoopMaxElements, listRemainElements[i]);
        if (lengths[i] > 0) {
            DataCopy(
                this->ubInputs[i], this->gmInputs[i][offsets[i]], Align(GetSortLen<float>(lengths[i]), sizeof(float)));
            tmpUbInputs[j] = this->ubInputs[i];
            elementCountListTail[j] = lengths[i];
            this->remainListNum += 1;
            j++;
        }
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::MrgsortCompute()
{
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (this->remainListNum == MERGE_LIST_TWO) {
        MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[0], tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        MrgSortSrcList sortListTail =
            MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        MrgSortSrcList sortListTail = MrgSortSrcList(
            tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[MERGE_LIST_IDX_THREE]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else {
        DataCopy(
            this->tempBuffer, this->tmpUbInputs[0], Align(GetSortLen<float>(elementCountListTail[0]), sizeof(float)));
        listSortedNums[0] = elementCountListTail[0];
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::UpdateSortInfo()
{
    curLoopSortedNum = 0;
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        if (lengths[i] > 0) {
            // update remain size
            listRemainElements[i] -= listSortedNums[j];
            allRemainElements -= listSortedNums[j];
            // update offset
            offsets[i] += GetSortOffset<float>(listSortedNums[j]);
            // update current loop sorted nums
            curLoopSortedNum += listSortedNums[j];
            j += 1;
        }
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::Extract()
{
    AscendC::Extract(this->ubOutput1, this->ubOutput2, this->tempBuffer, Ceil(curLoopSortedNum, ONE_REPEAT_SORT_NUM));
    // for sort: Muls(this->ubOutput1, this->ubOutput1, (float)-1, Align(curLoopSortedNum, sizeof(float)));
    // for sort: Cast(this->ubOutputCast1, this->ubOutput1, RoundMode::CAST_ROUND, Align(curLoopSortedNum,
    // sizeof(float)));
}

// 将 UB 中已补齐的 count 个 int32 写到 gmOutput1[outOffset]，并推进 outOffset
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::FlushPadBuf(int64_t count)
{
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = count * sizeof(int32_t);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPadCustom(this->gmOutput1[outOffset], this->ubOutput1.template ReinterpretCast<int32_t>(), params);
    outOffset += count;
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}

// UB 输出缓冲不足 needSlots 时先 flush，再将 outPos 归零
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::PadEnsureBufSpace(
    int64_t& outPos, int64_t needSlots, int64_t bufCapacity)
{
    if (outPos + needSlots > bufCapacity) {
        FlushPadBuf(outPos);
        outPos = 0;
    }
}

// 向 outBuf 连续写入 count 个 -1，表示该 topK 槽无有效 position
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::PadEmitMinusOnes(
    int64_t& outPos, int64_t count, LocalTensor<int32_t>& outBuf, int64_t bufCapacity)
{
    for (int64_t j = 0; j < count; j++) {
        PadEnsureBufSpace(outPos, 1, bufCapacity);
        outBuf.SetValue(outPos++, -1);
    }
}

// 从 sort pair 的 key（负 float）解析 token id
template <typename T, typename T2>
__aicore__ inline int32_t MoeMrgsortOut<T, T2>::PadGetTokenId(
    const LocalTensor<float>& tempFloat, int64_t pairIdx)
{
    return -static_cast<int32_t>(tempFloat.GetValue(pairIdx * 2));
}

// 续写上一轮 chunk 末尾未凑满 topK 的 token；返回 true 表示本轮 PadOutput 应提前结束
template <typename T, typename T2>
__aicore__ inline bool MoeMrgsortOut<T, T2>::PadResumeOngoing(
    LocalTensor<float>& tempFloat, LocalTensor<int32_t>& tempInt, LocalTensor<int32_t>& outBuf,
    int64_t topK, int64_t pairCount, int64_t bufCapacity, bool mergeFinished, int64_t& outPos,
    int64_t& pairIdx)
{
    if (padOngoingTokenId < 0) {
        return false;
    }
    if (pairCount > 0 && PadGetTokenId(tempFloat, 0) == padOngoingTokenId) {
        int64_t countForToken = padOngoingEmittedCount;
        while (pairIdx < pairCount && PadGetTokenId(tempFloat, pairIdx) == padOngoingTokenId) {
            PadEnsureBufSpace(outPos, 1, bufCapacity);
            outBuf.SetValue(outPos++, tempInt.GetValue(pairIdx * 2 + 1));
            pairIdx++;
            countForToken++;
        }
        bool tokenComplete = (pairIdx < pairCount) || mergeFinished;
        if (tokenComplete) {
            PadEmitMinusOnes(outPos, topK - countForToken, outBuf, bufCapacity);
            padNextExpectedTokenId = padOngoingTokenId + 1;
            padOngoingTokenId = -1;
            padOngoingEmittedCount = 0;
            return false;
        }
        padOngoingEmittedCount = countForToken;
        if (outPos > 0) {
            FlushPadBuf(outPos);
        }
        return true;
    }
    PadEmitMinusOnes(outPos, topK - padOngoingEmittedCount, outBuf, bufCapacity);
    padNextExpectedTokenId = padOngoingTokenId + 1;
    padOngoingTokenId = -1;
    padOngoingEmittedCount = 0;
    return false;
}

// 按 token 顺序处理当前 chunk 的 (token, position) 对并补齐 topK；跨 chunk 未完成时返回 true
template <typename T, typename T2>
__aicore__ inline bool MoeMrgsortOut<T, T2>::PadProcessPairs(
    LocalTensor<float>& tempFloat, LocalTensor<int32_t>& tempInt, LocalTensor<int32_t>& outBuf,
    int64_t topK, int64_t pairCount, int64_t bufCapacity, bool mergeFinished, int64_t& outPos,
    int64_t& pairIdx)
{
    while (pairIdx < pairCount) {
        int32_t curTokenId = PadGetTokenId(tempFloat, pairIdx);
        while (padNextExpectedTokenId < curTokenId) {
            PadEnsureBufSpace(outPos, topK, bufCapacity);
            PadEmitMinusOnes(outPos, topK, outBuf, bufCapacity);
            padNextExpectedTokenId++;
        }
        PadEnsureBufSpace(outPos, topK, bufCapacity);
        int64_t startIdx = pairIdx;
        while (pairIdx < pairCount && PadGetTokenId(tempFloat, pairIdx) == curTokenId) {
            PadEnsureBufSpace(outPos, 1, bufCapacity);
            outBuf.SetValue(outPos++, tempInt.GetValue(pairIdx * 2 + 1));
            pairIdx++;
        }
        int64_t countForToken = pairIdx - startIdx;
        bool tokenComplete = (pairIdx < pairCount) || mergeFinished;
        if (tokenComplete) {
            PadEmitMinusOnes(outPos, topK - countForToken, outBuf, bufCapacity);
            padNextExpectedTokenId = curTokenId + 1;
        } else {
            padOngoingTokenId = curTokenId;
            padOngoingEmittedCount = countForToken;
            if (outPos > 0) {
                FlushPadBuf(outPos);
            }
            return true;
        }
    }
    return false;
}

// 整次归并结束时补齐尾部无 routing 的 token（每 token topK 个 -1），并 flush 剩余 UB 数据
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::PadFinalizeChunk(
    LocalTensor<int32_t>& outBuf, int64_t topK, int64_t numTokens, int64_t bufCapacity,
    bool mergeFinished, int64_t& outPos)
{
    if (mergeFinished) {
        while (padNextExpectedTokenId < static_cast<int32_t>(numTokens)) {
            PadEnsureBufSpace(outPos, topK, bufCapacity);
            PadEmitMinusOnes(outPos, topK, outBuf, bufCapacity);
            padNextExpectedTokenId++;
        }
        padOngoingTokenId = -1;
        padOngoingEmittedCount = 0;
    }
    if (outPos > 0) {
        FlushPadBuf(outPos);
    }
}

// needPad 路径：将本轮归并结果按 token 分组并 pad 到 topK×numTokens 布局写入 sortedIndices
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::PadOutput()
{
    if (!this->param->needPad) {
        return;
    }
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    int64_t topK = this->param->topK;
    int64_t numTokens = this->param->numTokens;
    int64_t pairCount = curLoopSortedNum;
    int64_t bufCapacity = this->param->oneLoopMaxElements * MERGE_LIST_FOUR;
    bool mergeFinished = (this->allRemainElements == 0);

    LocalTensor<float> tempFloat = this->tempBuffer.template ReinterpretCast<float>();
    LocalTensor<int32_t> tempInt = this->tempBuffer.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outBuf = this->ubOutput1.template ReinterpretCast<int32_t>();

    int64_t outPos = 0;
    int64_t pairIdx = 0;

    if (PadResumeOngoing(
            tempFloat, tempInt, outBuf, topK, pairCount, bufCapacity, mergeFinished, outPos, pairIdx)) {
        return;
    }
    if (PadProcessPairs(
            tempFloat, tempInt, outBuf, topK, pairCount, bufCapacity, mergeFinished, outPos, pairIdx)) {
        return;
    }
    PadFinalizeChunk(outBuf, topK, numTokens, bufCapacity, mergeFinished, outPos);
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::CopyOut()
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curLoopSortedNum * sizeof(int32_t);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    LocalTensor<int32_t> ubOutputCast;
    if (this->param->needPad) {
        ubOutputCast = this->ubOutput1.template ReinterpretCast<int32_t>();
    } else {
        ubOutputCast = this->ubOutput2.template ReinterpretCast<int32_t>();
    }
    DataCopyPadCustom(this->gmOutput1[outOffset], ubOutputCast, intriParams);
    outOffset += curLoopSortedNum;
}
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::CopyOutCapacity(int64_t capacity)
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curLoopSortedNum * sizeof(int32_t);
    if (outOffset + curLoopSortedNum > capacity) {
        intriParams.blockLen = (capacity - outOffset) * sizeof(int32_t);
        allRemainElements = 0;
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    LocalTensor<int32_t> ubOutputCast = this->ubOutput2.template ReinterpretCast<int32_t>();
    DataCopyPadCustom(this->gmOutput1[outOffset], ubOutputCast, intriParams);
    outOffset += curLoopSortedNum;
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::Init(MoeMrgsortParam* param, TPipe* tPipe)
{
    this->param = param;
    this->allRemainElements = 0;
    eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    for (int64_t i = 0; i < listNum; i++) {
        offsets[i] = GetSortOffset<float>(param->perListElements * i);
        if (i == listNum - 1) {
            listRemainElements[i] = param->lastListElements;
        } else {
            listRemainElements[i] = param->perListElements;
        }
        allRemainElements += listRemainElements[i];
    }
    if (param->needPad) {
        padNextExpectedTokenId = 0;
        padOngoingTokenId = -1;
        padOngoingEmittedCount = 0;
    }
}
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::Process(int64_t capacity)
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        Extract();
        CopyOutCapacity(capacity);
    }
    ClearCache();
}

// 归并主循环：needPad 走 PadOutput，否则 Extract + CopyOut
template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        if (this->param->needPad) {
            PadOutput();
        } else {
            Extract();
            CopyOut();
        }
    }
    ClearCache();
}
} // namespace MoeTokenPermute
#endif // MOE_MRGSORT_OUT_H