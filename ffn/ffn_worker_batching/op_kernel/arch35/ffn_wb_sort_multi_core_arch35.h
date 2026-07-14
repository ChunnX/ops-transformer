/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_sort_multi_core_arch35.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SORT_MULTI_CORE_H
#define OP_KERNEL_FFN_WB_SORT_MULTI_CORE_H
#include "ffn_wb_arch35_reuse.h"


namespace FfnWbBatchingArch35 {
using namespace AscendC;

class FfnWbSortMultiCoreArch35 : public SortMaskBase {
public:
    __aicore__ inline FfnWbSortMultiCoreArch35(){};
    __aicore__ inline void Init(GM_ADDR expert_ids, GM_ADDR workspace, SortCustomTilingDataKernel *tilingData,
                                const ScheduleContextInfo *contextInfo, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElements, int64_t lastListElements);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum);
    __aicore__ inline void InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum, int64_t coreOffset,
                                               int64_t sortNumCoreOffset, int64_t loopOffset, int64_t loopIdxOffset);
    __aicore__ inline void InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum, int64_t coreOffset);
    __aicore__ inline void CopyOutValidCount();

private:
    GlobalTensor<float> workspaceGms[2];
    GlobalTensor<int32_t> workspaceSortNumGm_;

    SortCustomTilingDataKernel *tilingData_ = nullptr;
    const ScheduleContextInfo *contextInfo_ = nullptr;

    int32_t totalValidCnt_ = 0;
    int32_t curValidCnt_ = 0;

    SortCustomMrgsort mrgsorter;
    SortCustomMrgsortParam mrgsortParam;

    int64_t blockIdx_ = 0;
    int64_t srcWsIndex = 0;

    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;
    int64_t vmsSortNumStride_ = 0;

    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;

    int64_t perCoreExpert;
    int64_t needInitExpertCore;
    int64_t currentCoreExpert;

    static constexpr int64_t MAX_MRGSORT_LIST = 4;
};

__aicore__ inline void FfnWbSortMultiCoreArch35::VBSCopyIn(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    // 根因修复(脏数据 bug)：AllocTensor 返回的 UB 未初始化，aclnn 长 batch 连跑复用会残留上一次的脏值。
    // 这里整块预置确定 sentinel 后再搬入真值：expert 段(按 fp32 视图)置 MIN_FP32(< -expertStart_，
    // CompareScalar 恒判为无效，永不入选)，rowIds 段置 0；DataCopyPad 只覆盖 [0,size) 有效数据。
    // 这样 Cast/CompareScalar/GatherMask 在尾块 padding 及对齐 roundup 读区永远读到确定值，
    // 杜绝脏尾把 GatherMask 的 rsvdCnt 带偏→totalValidCnt_ 溢出/越界→workspace[0](actual_token_num)变负/减半。
    LocalTensor<float> inLocalFp32 = inLocal.ReinterpretCast<float>();
    Duplicate<float>(inLocalFp32, MIN_FP32, sortNum);
    Duplicate<int32_t>(inLocal[sortNum], 0, sortNum);
    // memset(V) 必须先于 DMA(MTE2) 落地，否则可能覆盖刚搬入的真实 expert 数据。
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);

    int64_t inOffset = progress * sortCoreLoopElements;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertIdsGm[inOffset], dataCopyParams, dataCopyPadParams);

    LocalTensor<int32_t> rowIdsLocal = inLocal[sortNum];
    int64_t startValue = blockIdx_ * tilingData_->perCoreElements + inOffset;
    ArithProgression<int32_t>(rowIdsLocal, startValue, 1, size);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void FfnWbSortMultiCoreArch35::UBSortCompute(int64_t progress, int64_t size, int64_t sortNum)
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdsLocal = inLocal[0];
    LocalTensor<float> expertIdsLocalFp32;

    expertIdsLocalFp32 = expertIdsLocal.ReinterpretCast<float>();
    Cast(expertIdsLocalFp32, expertIdsLocal, RoundMode::CAST_ROUND, size);

    LocalTensor<uint32_t> maskLocalTensor = sortedBuffer.Get<uint32_t>();
    uint64_t rsvdCnt = 0;

    // 根因修复(脏数据 bug)：mask 缓冲复用自 sortedBuffer(TBuf)，跨 loop/跨算子调用不清零。
    // CompareScalar 只写 [0,roundup64(size)) 的 mask 位，若 GatherMask 读到其后残留的脏 mask 位，
    // 会把无效槽当有效槽计入 → rsvdCnt 偏大甚至越界 → totalValidCnt_ 垃圾/OOB。用前整块清零，
    // 之后 CompareScalar 覆盖有效区、其余恒为 0，GatherMask 计数只由真实有效槽决定。
    Duplicate<uint32_t>(maskLocalTensor, static_cast<uint32_t>(0),
                        Ceil(sortNum, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM / ONE_REPEAT_SORT_NUM);

    Muls(expertIdsLocalFp32, expertIdsLocalFp32, (float)-1, size);

    LocalTensor<uint8_t> maskLocalTensorUInt8 = maskLocalTensor.ReinterpretCast<uint8_t>();
    AscendC::CompareScalar(maskLocalTensorUInt8, expertIdsLocalFp32, static_cast<float>(-expertStart_),
                           AscendC::CMPMODE::GT,
                           (size + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM);

    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8; // 8 blocks
    gatherMaskParams.src1RepeatStride = 0;
    GatherMask(expertIdsLocalFp32, expertIdsLocalFp32, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);
    curValidCnt_ = rsvdCnt;
    if (rsvdCnt == 0) {
        sortDataCopyInQueue.FreeTensor(inLocal);
        return;
    }
    this->totalValidCnt_ += rsvdCnt;
    int64_t duplicateNum = rsvdCnt % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = rsvdCnt - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdsLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }
    int32_t selectedCnt = (rsvdCnt + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;

    LocalTensor<uint32_t> rowIdsLocal = inLocal[sortNum].ReinterpretCast<uint32_t>();
    GatherMask(rowIdsLocal, rowIdsLocal, maskLocalTensor, true, size, gatherMaskParams, rsvdCnt);
    LocalTensor<float> concatLocal = expertIdsLocalFp32;
    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(sortNum));
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    Sort<float, true>(outLocal, concatLocal, rowIdsLocal, sortedLocal, selectedCnt / ONE_REPEAT_SORT_NUM);

    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void FfnWbSortMultiCoreArch35::VBSCopyOut(int64_t progress, int64_t size, int64_t sortNum)
{
    if (curValidCnt_ > 0) {
        LocalTensor<float> outLocal = sortDataCopyOutQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(GetSortLen<float>(curValidCnt_) * sizeof(int32_t)), 0, 0,
                                     0};
        int64_t wkOffset = blockIdx_ * GetSortLen<float>(tilingData_->perCoreElements) +
                           GetSortLen<float>(progress * sortCoreLoopElements);
        DataCopyPad(workspaceGms[0][wkOffset], outLocal, copyParams);
        sortDataCopyOutQueue.FreeTensor(outLocal);
    }

    LocalTensor<int32_t> tempTensor = tempBuffer.Get<int32_t>(BLOCK_BYTES / sizeof(int32_t));
    tempTensor.SetValue(0, curValidCnt_);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(workspaceSortNumGm_[blockIdx_ * tilingData_->sortNumWorkSpacePerCore + progress], tempTensor,
                copyParams1);
}

__aicore__ inline void FfnWbSortMultiCoreArch35::InitSortMaskMrgSort(SortCustomMrgsort *sorter, int64_t listNum,
                                                                     int64_t coreOffset, int64_t sortNumCoreOffset,
                                                                     int64_t loopOffset, int64_t loopIdxOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx_ * coreOffset + loopOffset];
    GlobalTensor<int32_t> srcSortNumGm = workspaceSortNumGm_[blockIdx_ * sortNumCoreOffset + loopIdxOffset];
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(tilingData_->oneLoopMaxElementsMrg) * i];
        sorter->SetInput(srcWsGm, srcSortNumGm, inLocalT);
    }
    GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx_ * coreOffset + loopOffset];
    LocalTensor<int32_t> outSortNumLocal = tempBuffer.Get<int32_t>(BLOCK_BYTES / sizeof(int32_t));
    sorter->SetOutput(dstWsGm, outLocal, outSortNumLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
    tempBuffer.FreeTensor(outSortNumLocal);
}

__aicore__ inline void FfnWbSortMultiCoreArch35::InitSortMaskMrgSortOut(SortCustomMrgsortOut *sorter, int64_t listNum,
                                                                        int64_t coreOffset)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex];
    GlobalTensor<int32_t> srcSortNumGm = workspaceSortNumGm_;
    LocalTensor<float> inLocal = sortDataCopyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

    for (int64_t i = 0; i < listNum; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(tilingData_->oneLoopMaxElementsMrg) * i];
        sorter->SetInput(srcWsGm, srcSortNumGm, inLocalT);
    }

    LocalTensor<float> outLocalV = outLocal[tilingData_->oneLoopMaxElementsMrg * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedexpertIdsGm, this->sortedRowIdsGm, outLocal, outLocalV);

    LocalTensor<float> tempBuffer =
        sortedBuffer.Get<float>(GetSortLen<float>(tilingData_->oneLoopMaxElementsMrg) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(tempBuffer);
    sortDataCopyInQueue.FreeTensor(inLocal);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void FfnWbSortMultiCoreArch35::OneCoreVMSProcess(int64_t listNum, int64_t perListElements,
                                                                   int64_t lastListElements)
{
    int64_t coreOffset = GetSortLen<float>(tilingData_->perCoreElements);
    int64_t sortNumCoreOffset = tilingData_->sortNumWorkSpacePerCore;
    mrgsortParam.oneLoopMaxElements = tilingData_->oneLoopMaxElementsMrg;

    int64_t curSortNumStride = 1;
    for (int64_t i = 0; listNum >= 1; i++) {
        int64_t loops = (listNum + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (loops - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.sortNumStride = curSortNumStride;

        int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
        int64_t loopIdxOffset = mrgsortParam.sortNumStride * MAX_MRGSORT_LIST;
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            InitSortMaskMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, sortNumCoreOffset, loop * loopOffset,
                                loop * loopIdxOffset);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }

        InitSortMaskMrgSort(&mrgsorter, remainListNum, coreOffset, sortNumCoreOffset, (loops - 1) * loopOffset,
                            (loops - 1) * loopIdxOffset);
        mrgsorter.Init(&mrgsortParam);
        mrgsorter.Process();

        listNum = loops;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        curSortNumStride = curSortNumStride * MAX_MRGSORT_LIST;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;
        if (loops == 1) {
            break;
        }
    }
}

__aicore__ inline void FfnWbSortMultiCoreArch35::UBSortProcess(int64_t progress, int64_t size, int64_t sortNum)
{
    VBSCopyIn(progress, size, sortNum);
    UBSortCompute(progress, size, sortNum);
    VBSCopyOut(progress, size, sortNum);
}

__aicore__ inline void FfnWbSortMultiCoreArch35::VBSProcess()
{
    if (blockIdx_ < tilingData_->needCoreNum) {
        int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements, sortNum);
        }

        sortNum = Ceil(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sortNum);

        CopyOutValidCount();

        if (sortCoreLoops > 1) {
            OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
        }
    }
    SyncAll();
}

__aicore__ inline void FfnWbSortMultiCoreArch35::VMSProcess()
{
    int64_t currentStageNeedCoreNum = tilingData_->needCoreNumMrg;
    perListElements = tilingData_->perCoreElements;
    listNum = tilingData_->needCoreNum;
    vmsSortNumStride_ = tilingData_->sortNumWorkSpacePerCore;

    for (; listNum > MAX_MRGSORT_LIST;) {
        currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);
        int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
        int64_t sortNumCoreOffset = vmsSortNumStride_ * MAX_MRGSORT_LIST;
        int64_t remainListNum = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElements;
        mrgsortParam.sortNumStride = vmsSortNumStride_;
        mrgsortParam.oneLoopMaxElements = tilingData_->oneLoopMaxElementsMrg;

        if (blockIdx_ < currentStageNeedCoreNum - 1) {
            InitSortMaskMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, sortNumCoreOffset, 0, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        } else if (blockIdx_ == currentStageNeedCoreNum - 1) {
            InitSortMaskMrgSort(&mrgsorter, remainListNum, coreOffset, sortNumCoreOffset, 0, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }
        listNum = currentStageNeedCoreNum;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        perListElements = perListElements * MAX_MRGSORT_LIST;
        vmsSortNumStride_ = vmsSortNumStride_ * MAX_MRGSORT_LIST;

        SyncAll();
    }
}

__aicore__ inline void FfnWbSortMultiCoreArch35::SortOutProcess()
{
    if (blockIdx_ < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.sortNumStride = vmsSortNumStride_;
        mrgsortParam.oneLoopMaxElements = tilingData_->oneLoopMaxElementsMrg;

        SortCustomMrgsortOut sorter;
        InitSortMaskMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
    }
    SyncAll();
}

__aicore__ inline void FfnWbSortMultiCoreArch35::CopyOutValidCount()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
    // 根因修复(脏数据 bug)：outLocal 由 AllocTensor 得到、UB 未初始化。先整块(1*32B=8*int32)清零，
    // 仅 [0] 写有效计数，再以 32B 对齐做 atomic-add：workspace[0]+=计数、workspace[1..7]+=0
    // (Init 中 InitGlobalMemory 已置 0)保持不变。避免连跑复用下 32B 粒度 atomic 把脏字节累加进
    // 计数区。V(memset)->S(SetValue)->MTE3(搬出) 逐级同步，确保每一步落地后再进入下一步。
    constexpr int32_t CNT_BLK_ELEM = static_cast<int32_t>(BLOCK_BYTES / sizeof(int32_t)); // 8
    Duplicate<int32_t>(outLocal, static_cast<int32_t>(0), CNT_BLK_ELEM);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    outLocal.SetValue(0, this->totalValidCnt_);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    SetAtomicAdd<int32_t>();
    DataCopy(rsvdCntGm[0], outLocal, CNT_BLK_ELEM);
    SetAtomicNone();
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void FfnWbSortMultiCoreArch35::Init(GM_ADDR expert_ids, GM_ADDR workspace,
                                                      SortCustomTilingDataKernel *tilingData,
                                                      const ScheduleContextInfo *contextInfo, TPipe *tPipe)
{
    this->pipe = tPipe;
    contextInfo_ = contextInfo;
    tilingData_ = tilingData;

    this->totalLength = tilingData->totalLength;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ == tilingData_->needCoreNum - 1) {
        sortCoreLoops = tilingData_->lastCoreLoops;
        sortCoreLoopElements = tilingData_->lastCorePerLoopElements;
        sortCoreLastLoopElements = tilingData_->lastCoreLastLoopElements;
        this->sortTotalLength = tilingData_->lastCoreElements;
    } else {
        sortCoreLoops = tilingData_->perCoreLoops;
        sortCoreLoopElements = tilingData_->perCorePerLoopElements;
        sortCoreLastLoopElements = tilingData_->perCoreLastLoopElements;
        this->sortTotalLength = tilingData_->perCoreElements;
    }

    expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)expert_ids + blockIdx_ * tilingData_->perCoreElements,
                                this->sortTotalLength);
    rsvdCntGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), OFFSET_SORTED_EXPERT_IDS);
    workspaceSortNumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS,
                                        contextInfo_->sortNumWorkSpace);

    if (blockIdx_ == 0) {
        InitGlobalMemory(rsvdCntGm, OFFSET_SORTED_EXPERT_IDS, 0);
        GM_ADDR targetAddr = workspace + OFFSET_SORTED_EXPERT_IDS * sizeof(int32_t) +
                             contextInfo_->sortNumWorkSpace * sizeof(int32_t) * (NUM_TWO * NUM_FOUR + 1);
        groupListTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(targetAddr), contextInfo_->expertNum);
        InitGlobalMemory(groupListTmpGm, contextInfo_->expertNum, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        // core0 清零 rsvdCnt(workspace[0]) 后必须 flush 到 GM 全局可见，再经 SyncAll 放行其他核：
        // 否则清零可能滞留 core0 cache，其他核 atomic-add 读到未清零的复用脏值(workspace 长跑复用
        // 脏页)→ 累加到脏基 → actual_token_num 偶发变负/减半(高频非确定)。单核路径(ffn_wb_sort_one_core.h)
        // 写 rsvdCnt 后同样用 DataCacheCleanAndInvalid 刷，多核清零此前漏了这步。
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
            rsvdCntGm);
    }
    SyncAll();

    sortedexpertIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                          contextInfo_->sortNumWorkSpace,
                                      this->totalLength);
    sortedRowIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                       contextInfo_->sortNumWorkSpace + this->totalLength,
                                   this->totalLength);

    int64_t kvFactor = 2;
    workspaceGms[0].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                        contextInfo_->sortNumWorkSpace + this->totalLength * kvFactor,
                                    this->totalLength * kvFactor);
    workspaceGms[1].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + OFFSET_SORTED_EXPERT_IDS +
                                        contextInfo_->sortNumWorkSpace + this->totalLength * (kvFactor + kvFactor),
                                    this->totalLength * kvFactor);

    int64_t bufferSize =
        Ceil(Max(tilingData_->oneLoopMaxElementsMrg * MAX_MRGSORT_LIST, sortCoreLoopElements), ONE_REPEAT_SORT_NUM) *
        ONE_REPEAT_SORT_NUM * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, bufferSize);
    pipe->InitBuffer(sortedBuffer, bufferSize);
    pipe->InitBuffer(tempBuffer, bufferSize);
}

__aicore__ inline void FfnWbSortMultiCoreArch35::Process()
{
    VBSProcess();
    VMSProcess();
    SortOutProcess();
}
} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_FFN_WB_SORT_MULTI_CORE_H
