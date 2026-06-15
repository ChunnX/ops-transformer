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
 * \file moe_sort_multi_core.h
 * \brief
 */
#ifndef MOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_HMOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_H
#define MOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_HMOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_H

#include "moe_sort_base.h"
#include "moe_mrgsort.h"
#include "moe_mrgsort_out.h"

namespace MoeTokenPermute {
using namespace AscendC;

__aicore__ inline int64_t KernelCeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t KernelFloorDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return a / b;
}

__aicore__ inline int64_t KernelCeilLog4(int64_t n)
{
    int64_t i = 0;
    int64_t p = 1;
    while (p < n) {
        i++;
        p = p << 2;
    }
    return i;
}

__aicore__ inline int64_t KernelPow4(int64_t n)
{
    return 1LL << (2 * n);
}

__aicore__ inline int64_t KernelAbsDiff(int64_t a, int64_t b)
{
    return a > b ? a - b : b - a;
}

__aicore__ inline int64_t KernelVmsLoops(int64_t listNum)
{
    constexpr int64_t ONE_LOOP_VMS = 24;
    if (listNum <= ONE_LOOP_VMS) {
        return 1;
    }
    return KernelCeilDiv(listNum, ONE_LOOP_VMS);
}

template <typename T>
class MoeSortMultiCore : public MoeSortBase
{
public:
    __aicore__ inline MoeSortMultiCore(){};
    __aicore__ inline void Init(
        GM_ADDR expertForSourceRow, GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
        const MoeTokenPermuteWithRoutingMapTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void SetNeedPad(bool needPad, int64_t topK, int64_t numTokens);
    __aicore__ inline void RecalcVBSMultiCoreParams(
        GM_ADDR expertForSourceRow, int64_t actualTotalLength, int64_t blockNum,
        const MoeTokenPermuteWithRoutingMapTilingData* tilingData);
    __aicore__ inline void InitAfterRecalc(
        GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
        const MoeTokenPermuteWithRoutingMapTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t VbsPerCoreElements() const
    {
        return vbsRecalced ? blockFactor : vbsTilingData->perCoreElements;
    }
    __aicore__ inline int64_t VbsLastCoreElements() const
    {
        return vbsRecalced ? rLastCoreElements : vbsTilingData->lastCoreElements;
    }
    __aicore__ inline int64_t VbsNeedCoreNum() const { return vbsRecalced ? rNeedCoreNum : vbsTilingData->needCoreNum; }
    __aicore__ inline int64_t VmsNeedCoreNum() const
    {
        return vbsRecalced ? rVmsNeedCoreNum : vmsTilingData->needCoreNum;
    }
    __aicore__ inline int64_t SortOutOneLoopMaxElements() const { return sortOutTilingData->oneLoopMaxElements; }
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void InitMoeMrgSort(MoeMrgsort* sorter, int64_t listNum, int64_t coreOffset, int64_t loopOffset);
    __aicore__ inline void InitMoeMrgSortOut(
        MoeMrgsortOut<int32_t, int32_t>* sorter, int64_t listNum, int64_t coreOffset);

private:
    GlobalTensor<float> workspaceGms[2];

    const PermuteVBSComputeRMTilingData* vbsTilingData;
    const PermuteVMSMiddleComputeRMTilingData* vmsTilingData;
    const PermuteSortOutComputeRMTilingData* sortOutTilingData;

    // for MoeMrgsort
    MoeMrgsort mrgsorter;
    MoeMrgsortParam mrgsortParam;
    TBuf<TPosition::VECCALC> indexBuffer;
    LocalTensor<int32_t> indexLocal;
    LocalTensor<float> concatLocal;

    TBuf<TPosition::VECCALC> concatBuffer;
    TBuf<TPosition::VECCALC> indexConcatBuffer;

    GlobalTensor<T> expertForSourceRowGm;
    GlobalTensor<int32_t> expertForSourceRowGmB32;

    GlobalTensor<int32_t> sortedExpertForSourceRowGm;
    GlobalTensor<float> debugGm;
    GlobalTensor<float> debugGm1;
    GlobalTensor<float> debugGm2;
    LocalTensor<float> expertForSourceRowLocalFp32;
    GM_ADDR expertForSourceRow1;
    int64_t coreNum;
    int64_t blockIdx;
    int64_t srcWsIndex = 0;
    int64_t blockFactor;
    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;

    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;
    static constexpr int32_t BLOCK_DATA_NUM = ONE_BLK_SIZE / sizeof(T);
    static constexpr int64_t MAX_MRGSORT_LIST = 4;

    bool vbsRecalced = false;
    int64_t rNeedCoreNum;
    int64_t rLastCoreElements;
    int64_t rVmsNeedCoreNum;
};

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<T> inLocal = sortDataCopyInQueue.AllocTensor<T>();
    int64_t inOffset = progress * sortCoreLoopElements;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams DataCopyPadCustomParams{false, 0, 0, (int)0};

    if constexpr (IsSameType<T, int64_t>::value) {
        DataCopyB64(inLocal, expertForSourceRowGm[inOffset], dataCopyParams, DataCopyPadCustomParams);
    } else {
        DataCopyPadCustom(inLocal, expertForSourceRowGm[inOffset], dataCopyParams, DataCopyPadCustomParams);
    }

    PipeBarrier<PIPE_V>();

    sortDataCopyInQueue.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<T> sortDataLocal = sortDataCopyInQueue.DeQue<T>();
    expertForSourceRowLocalFp32 = concatLocal;
    LocalTensor<int32_t> expertForSourceRowLocalInt32 = expertForSourceRowLocalFp32.template ReinterpretCast<int32_t>();

    Cast(expertForSourceRowLocalFp32, sortDataLocal, RoundMode::CAST_ROUND, size);

    sortDataCopyInQueue.FreeTensor(sortDataLocal);
    PipeBarrier<PIPE_V>();

    Muls(expertForSourceRowLocalFp32, expertForSourceRowLocalFp32, (float)-1, sortNum);
    PipeBarrier<PIPE_V>();

    int64_t duplicateNum = size % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = size - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<float> concatLocal;

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<uint32_t> sourceRowLocal = indexLocal.ReinterpretCast<uint32_t>();
    PipeBarrier<PIPE_V>();

    Sort<float, true>(
        outLocal, expertForSourceRowLocalFp32, sourceRowLocal, sortedLocal, sortNum / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue.EnQue<float>(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();
    DataCopy(
        workspaceGms[srcWsIndex]
                    [this->blockIdx * GetSortLen<float>(VbsPerCoreElements()) +
                     GetSortLen<float>(progress * sortCoreLoopElements)],
        outLocal, Align(GetSortLen<float>(size), sizeof(float)));
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::InitMoeMrgSort(
    MoeMrgsort* sorter, int64_t listNum, int64_t coreOffset, int64_t loopOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx * coreOffset + loopOffset];
    LocalTensor<float> inLocal = sortedBuffer.Get<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(SortOutOneLoopMaxElements()) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }
    GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx * coreOffset + loopOffset];

    sorter->SetOutput(dstWsGm, outLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::InitMoeMrgSortOut(
    MoeMrgsortOut<int32_t, int32_t>* sorter, int64_t listNum, int64_t coreOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex];
    LocalTensor<float> inLocal = indexConcatBuffer.Get<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(SortOutOneLoopMaxElements()) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }

    LocalTensor<float> outLocalV = outLocal[SortOutOneLoopMaxElements() * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedExpertForSourceRowGm, this->sortedExpertForSourceRowGm, outLocal, outLocalV);

    LocalTensor<float> tempBuffer =
        sortedBuffer.Get<float>(GetSortLen<float>(SortOutOneLoopMaxElements()) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(tempBuffer);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::OneCoreVMSProcess(
    int64_t listNum, int64_t perListElements, int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(VbsPerCoreElements());
    mrgsortParam.oneLoopMaxElements = SortOutOneLoopMaxElements();

    for (int64_t i = 0; listNum >= 1; i++) {
        int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = perListElements;

        int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, loop * loopOffset);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, (loops - 1) * loopOffset);
        mrgsorter.Init(&mrgsortParam);
        mrgsorter.Process();

        listNum = loops;
        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        if (loops == 1) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
{
    VBSCopyIn(progress, size, sortNum);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::VBSProcess()
{
    if (this->blockIdx < VbsNeedCoreNum()) {
        int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        ArithProgressionSupportInt32<int32_t>(
            indexLocal, static_cast<int32_t>(this->blockIdx * blockFactor), static_cast<int32_t>(1),
            sortCoreLoopElements);
        for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements, sortNum);
            PipeBarrier<PIPE_V>();

            Adds(indexLocal, indexLocal, (int32_t)sortCoreLoopElements, sortCoreLoopElements);
        }

        sortNum = Ceil(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sortNum);

        OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::VMSProcess()
{
    int64_t currentStageNeedCoreNum = VmsNeedCoreNum();
    perListElements = VbsPerCoreElements();
    lastListElements = VbsLastCoreElements();
    listNum = VbsNeedCoreNum();

    for (; listNum > MAX_MRGSORT_LIST;) {
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
        int64_t remainListNum = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

        if (this->blockIdx < currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = perListElements;
            mrgsortParam.oneLoopMaxElements = SortOutOneLoopMaxElements();
            InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        } else if (this->blockIdx == currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            mrgsortParam.oneLoopMaxElements = SortOutOneLoopMaxElements();
            InitMoeMrgSort(&mrgsorter, remainListNum, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }
        listNum = currentStageNeedCoreNum;
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        lastListElements = perListElements * (remainListNum - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
#ifndef __CCE_KT_TEST__
        AscendC::SyncAll();
#endif
    }
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::SortOutProcess()
{
    if (this->blockIdx < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        mrgsortParam.oneLoopMaxElements = SortOutOneLoopMaxElements();

        MoeMrgsortOut<int32_t, int32_t> sorter;
        InitMoeMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::SetNeedPad(bool needPad, int64_t topK, int64_t numTokens)
{
    this->mrgsortParam.needPad = needPad;
    this->mrgsortParam.topK = topK;
    this->mrgsortParam.numTokens = numTokens;
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::RecalcVBSMultiCoreParams(
    GM_ADDR expertForSourceRow, int64_t actualTotalLength, int64_t blockNum,
    const MoeTokenPermuteWithRoutingMapTilingData* tilingData)
{
    expertForSourceRow1 = expertForSourceRow;
    this->blockIdx = GetBlockIdx();
    this->vbsTilingData = &(tilingData->vbsComputeParamsOp);
    this->vmsTilingData = &(tilingData->vmsMiddleComputeParamsOp);
    this->sortOutTilingData = &(tilingData->sortOutComputeParamsOp);

    // 与 tiling 侧 Tinlig4VBSMultiCoreCompute 一致，使用 VBS 的 oneLoopMaxElements（非 sortOut 的 1024）
    constexpr int64_t SORT32_ALIGN_ELEMENT = 32;
    int64_t sortLoopMaxElement = this->vbsTilingData->oneLoopMaxElements;
    int64_t needCoreNum = KernelCeilDiv(actualTotalLength, sortLoopMaxElement);
    needCoreNum = KernelPow4(KernelCeilLog4(needCoreNum));
    if (needCoreNum > blockNum) {
        needCoreNum = blockNum;
    }

    int64_t perCoreElements = KernelFloorDiv(actualTotalLength, needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = actualTotalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements =
        perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = KernelCeilDiv(actualTotalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    int64_t perCoreLoops = KernelCeilDiv(perCoreElements, sortLoopMaxElement);
    int64_t perCorePerLoopElements =
        perCoreElements > sortLoopMaxElement ? sortLoopMaxElement : perCoreElements;
    int64_t perCoreLastLoopElements =
        perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    int64_t lastCoreElements = actualTotalLength - (needCoreNum - 1) * perCoreElements;
    int64_t lastCoreLoops = KernelCeilDiv(lastCoreElements, sortLoopMaxElement);
    int64_t lastCorePerLoopElements =
        lastCoreElements > sortLoopMaxElement ? sortLoopMaxElement : lastCoreElements;
    int64_t lastCoreLastLoopElements =
        lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    int64_t lastCoreWSindex =
        KernelAbsDiff(KernelVmsLoops(lastCoreLoops), KernelVmsLoops(perCoreLoops));

    rNeedCoreNum = needCoreNum;
    rLastCoreElements = lastCoreElements;
    vbsRecalced = true;

    if (blockIdx == needCoreNum - 1) {
        sortTotalLength = lastCoreElements;
        sortCoreLoops = lastCoreLoops;
        sortCoreLoopElements = lastCorePerLoopElements;
        sortCoreLastLoopElements = lastCoreLastLoopElements;
        srcWsIndex = lastCoreWSindex;
        tileLength = lastCorePerLoopElements;
    } else {
        sortTotalLength = perCoreElements;
        sortCoreLoops = perCoreLoops;
        sortCoreLoopElements = perCorePerLoopElements;
        sortCoreLastLoopElements = perCoreLastLoopElements;
        tileLength = perCorePerLoopElements;
    }

    blockFactor = perCoreElements;

    if (rNeedCoreNum <= MAX_MRGSORT_LIST) {
        rVmsNeedCoreNum = 0;
    } else {
        rVmsNeedCoreNum = KernelCeilDiv(rNeedCoreNum, MAX_MRGSORT_LIST);
    }

    expertForSourceRowGm.SetGlobalBuffer(
        (__gm__ T*)expertForSourceRow1 + blockIdx * perCoreElements,
        sortTotalLength);
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::Init(
    GM_ADDR expertForSourceRow, GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
    const MoeTokenPermuteWithRoutingMapTilingData* tilingData, TPipe* tPipe)
{
    expertForSourceRow1 = expertForSourceRow;
    this->totalLength = tilingData->n * tilingData->topK;
    this->coreNum = tilingData->coreNum;
    this->vbsTilingData = &(tilingData->vbsComputeParamsOp);
    this->vmsTilingData = &(tilingData->vmsMiddleComputeParamsOp);
    this->sortOutTilingData = &(tilingData->sortOutComputeParamsOp);

    this->blockIdx = GetBlockIdx();
    this->tileLength = this->vbsTilingData->perCorePerLoopElements;
    this->sortTotalLength = this->vbsTilingData->perCoreElements;
    if (this->blockIdx == tilingData->vbsComputeParamsOp.needCoreNum - 1) {
        this->tileLength = this->vbsTilingData->lastCorePerLoopElements;
        this->sortTotalLength = this->vbsTilingData->lastCoreElements;
        sortCoreLoops = this->vbsTilingData->lastCoreLoops;
        sortCoreLoopElements = this->vbsTilingData->lastCorePerLoopElements;
        sortCoreLastLoopElements = this->vbsTilingData->lastCoreLastLoopElements;
        srcWsIndex = this->vbsTilingData->lastCoreWSindex;
    } else {
        sortCoreLoops = this->vbsTilingData->perCoreLoops;
        sortCoreLoopElements = this->vbsTilingData->perCorePerLoopElements;
        sortCoreLastLoopElements = this->vbsTilingData->perCoreLastLoopElements;
    }

    this->pipe = tPipe;
    blockFactor = tilingData->vbsComputeParamsOp.perCoreElements;
    expertForSourceRowGm.SetGlobalBuffer(
        (__gm__ T*)expertForSourceRow + this->blockIdx * tilingData->vbsComputeParamsOp.perCoreElements,
        this->sortTotalLength);
    sortedExpertForSourceRowGm.SetGlobalBuffer((__gm__ int32_t*)sortedExpertForSourceRow, this->totalLength);
    // for sort: expandDstToSrcRowGm.SetGlobalBuffer((__gm__ int32_t*)workspace, Align(this->totalLength,
    // sizeof(int32_t)));

    int64_t kvFactor = 2;

    workspaceGms[0].SetGlobalBuffer((__gm__ float*)workspace, Align(this->totalLength, sizeof(int32_t)) * kvFactor);
    workspaceGms[1].SetGlobalBuffer(
        (__gm__ float*)workspace + Align(this->totalLength, sizeof(int32_t)) * kvFactor,
        Align(this->totalLength, sizeof(int32_t)) * kvFactor);

    int64_t indexNum = Ceil(
                           Max(this->sortOutTilingData->oneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements),
                           ONE_REPEAT_SORT_NUM) *
                       ONE_REPEAT_SORT_NUM;
    int64_t indexBufferSize = indexNum * sizeof(int32_t);
    int64_t sortDataBufferSize = indexNum * sizeof(int64_t);
    int64_t bufferSize = indexBufferSize * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, sortDataBufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(indexConcatBuffer, bufferSize);
    pipe->InitBuffer(sortedBuffer, bufferSize);
    auto indexConcatLocal = indexConcatBuffer.Get<int32_t>();
    indexLocal = indexConcatLocal;
    concatLocal = indexConcatLocal.template ReinterpretCast<float>()[indexNum];
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::InitAfterRecalc(
    GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
    const MoeTokenPermuteWithRoutingMapTilingData* tilingData, TPipe* tPipe)
{
    this->totalLength = tilingData->n * tilingData->topK;
    this->coreNum = tilingData->coreNum;
    this->pipe = tPipe;
    sortedExpertForSourceRowGm.SetGlobalBuffer((__gm__ int32_t*)sortedExpertForSourceRow, this->totalLength);

    int64_t kvFactor = 2;

    workspaceGms[0].SetGlobalBuffer((__gm__ float*)workspace, Align(this->totalLength, sizeof(int32_t)) * kvFactor);
    workspaceGms[1].SetGlobalBuffer(
        (__gm__ float*)workspace + Align(this->totalLength, sizeof(int32_t)) * kvFactor,
        Align(this->totalLength, sizeof(int32_t)) * kvFactor);

    int64_t indexNum = Ceil(
                           Max(SortOutOneLoopMaxElements() * MAX_MRGSORT_LIST, sortCoreLoopElements),
                           ONE_REPEAT_SORT_NUM) *
                       ONE_REPEAT_SORT_NUM;
    int64_t indexBufferSize = indexNum * sizeof(int32_t);
    int64_t sortDataBufferSize = indexNum * sizeof(int64_t);
    int64_t bufferSize = indexBufferSize * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, sortDataBufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(indexConcatBuffer, bufferSize);
    pipe->InitBuffer(sortedBuffer, bufferSize);
    auto indexConcatLocal = indexConcatBuffer.Get<int32_t>();
    indexLocal = indexConcatLocal;
    concatLocal = indexConcatLocal.template ReinterpretCast<float>()[indexNum];
}

template <typename T>
__aicore__ inline void MoeSortMultiCore<T>::Process()
{
    VBSProcess();
    VMSProcess();
    SortOutProcess();
}
} // namespace MoeTokenPermute
#endif // MOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_HMOE_TOKEN_PREMUTE_MOE_SORT_MULTI_CORE_H