/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_block_vec_flashdecode.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_FLASHDECODE_H
#define FIA_BLOCK_VEC_FLASHDECODE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../vector_common.h"

using namespace AttentionCommon;
struct TaskInfo {
    uint32_t bIdx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

template <typename FIAT> 
class FiaBlockVecFlashDecode {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using OUT_T = typename FIAT::outputType;  
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout; 

    __aicore__ inline void InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm, GlobalTensor<T> accumOutGm, 
            GlobalTensor<OUT_T> attentionOutGm, GlobalTensor<uint64_t> actualSeqLengthsGmQ);
    __aicore__ inline void InitParams(const AttentionCommon::ConstInfo &constInfo);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();   
    __aicore__ inline void FlashDecode(FDparams &fd);
protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);                            
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t cntM);
     __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal, 
                                          uint32_t cntKV, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount, uint64_t attenOutOffset);
private:
// =================================常量区=================================
    static constexpr uint64_t SYNC_LSE_SUM_BUF1_FLAG = 6;
    static constexpr uint64_t SYNC_LSE_SUM_BUF2_FLAG = 7;
    static constexpr uint64_t SYNC_LSE_MAX_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = fa_base_vector::BYTE_BLOCK / sizeof(T); // 32/4=8
    
protected:
    GlobalTensor<T> lseSumFdGm;       
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t blockIdx = 0U;
    AttentionCommon::ConstInfo constInfo{};
    TaskInfo taskInfo{};
private:    
 // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1;    // 1.5k: 16*24*4
    TBuf<> fdSumBuf2;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf1;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf2;    // 1.5k: 16*24*4
    TBuf<> fdLseExpBuf;  // 1.5k: 16*24*4
    TBuf<> fdMm2ResBuf1; // 32k: 16*512*4
    TBuf<> fdMm2ResBuf2; // 32k: 16*512*4
    TBuf<> fdReduceBuf;  // 32k: 16*512*4
    TBuf<> fdOutputBuf;  // 32k: 16*512*4

    TBuf<> fdLseMaxUbBuf; // 64B: 16*4
    TBuf<> fdLseSumUbBuf; // 64B: 16*4
};

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, 
                                                        GlobalTensor<T> lseSumFdGm, 
                                                        GlobalTensor<T> accumOutGm,
                                                        GlobalTensor<OUT_T> attentionOutGm,
                                                        GlobalTensor<uint64_t> actualSeqLengthsGmQ)
{
   this->lseMaxFdGm = lseMaxFdGm;
   this->lseSumFdGm = lseSumFdGm;
   this->accumOutGm = accumOutGm;
   this->attentionOutGm = attentionOutGm;
   this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitParams(const AttentionCommon::ConstInfo &constInfo)
{
   this->constInfo = constInfo;
}


template <typename FIAT>__aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitDecodeParams()
{
    this->blockIdx = GetBlockIdx();
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitBuffers(TPipe *pipe)
{
    if ASCEND_IS_AIV {
        pipe->Reset();
        pipe->InitBuffer(fdSumBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdSumBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdLseExpBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMm2ResBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdMm2ResBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdReduceBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdOutputBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdLseMaxUbBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseSumUbBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
    }
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex,
    uint32_t startRow, uint32_t dealRowCount)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (constInfo.headDimAlign - constInfo.headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (constInfo.headDimAlign - constInfo.headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;
    uint64_t combineAccumOutOffset = startRow * constInfo.headDim +                          // taskoffset + g轴offset
                                      splitKVIndex * constInfo.mBaseSize * constInfo.headDim; // 份数offset

    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::CopyLseIn(uint32_t startRow,
    uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
{
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);

    uint64_t combineLseOffset = (baseOffset + startRow) * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    uint64_t combineLoopOffset = constInfo.mBaseSize * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    uint64_t dealRowCountAlign = dealRowCount * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign); // 份数offset
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign);
    }

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
}

template <typename FIAT> __aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::ComputeScaleValue(LocalTensor<T> &lseExp, 
                                                    uint32_t startRow,
                                                    uint32_t dealRowCount, 
                                                    uint32_t cntM)
{
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    LocalTensor<T> lseMaxUb = fdLseMaxUbBuf.Get<T>();
    LocalTensor<T> lseSumUb = fdLseSumUbBuf.Get<T>();
    uint64_t dealRowCountAlign = dealRowCount * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    Duplicate(lseMaxUb, -AttentionCommon::ConstInfo::FLOAT_MAX, dealRowCountAlign);
    Duplicate(lseSumUb, AttentionCommon::ConstInfo::FLOAT_ZERO, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::ColMax(lseMaxUb, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::RowSub(lseExp, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    Exp(lseExp, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    Mul(lseExp, lseSum, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::ColAdd(lseSumUb, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::MatDivsVec(lseExp, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
}


template <typename FIAT>__aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                               uint32_t startRow, uint32_t dealRowCount,
                                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (fa_base_vector::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb,
                dataCopyParams);
}

template <typename FIAT>__aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::ReduceFinalRes(LocalTensor<T> &reduceOut, 
                                                      LocalTensor<T> &mm2Res, 
                                                      LocalTensor<T> &lseLocal, 
                                                      uint32_t cntKV, 
                                                      uint32_t dealRowCount)
{
    uint32_t dealRowCountAlign = dealRowCount * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<T> tmpRst =
        cntKV == 0 ? reduceOut : mm2Res; // 第一次mul结果直接写入reduceOut，否则在mm2Res原地进行mul，再加到reduceOut

    fa_base_vector::RowMuls(tmpRst, mm2Res, lseLocal[cntKV * dealRowCountAlign], dealRowCount, constInfo.headDimAlign, constInfo.headDim);

    if (cntKV != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        Add(reduceOut, reduceOut, tmpRst, dealRowCount * constInfo.headDimAlign);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal, 
                                                       uint32_t startRow,
                                                       uint32_t dealRowCount, 
                                                       uint64_t attenOutOffset)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.headDim};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * constInfo.headDimAlign);
    } else {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * constInfo.headDimAlign);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    if (constInfo.outputLayout == FIA_LAYOUT::NBSD || constInfo.outputLayout == FIA_LAYOUT::NTD) {
        FusedTransposeInfo transInfo;
        transInfo.n2Idx = 0; 
        transInfo.bIdx = taskInfo.bIdx;
        auto gS1StartIdx = taskInfo.gS1Idx + startRow;
        auto gS1EndIdx = gS1StartIdx + dealRowCount - 1;
        GetGS1Idx<LAYOUT_T>(gS1StartIdx, transInfo.gStartIdx, transInfo.s1StartIdx, constInfo);
        GetGS1Idx<LAYOUT_T>(gS1EndIdx, transInfo.gEndIdx, transInfo.s1EndIdx, constInfo);
        if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
            transInfo.gCount = transInfo.gEndIdx - transInfo.gStartIdx + 1;
            fa_base_vector::Bmm2DataCopyOutNBSDGTiling(tmpBmm2ResCastTensor, transInfo, constInfo, attentionOutGm);
        } else {
            transInfo.s1Count = transInfo.s1EndIdx - transInfo.s1StartIdx + 1;
            transInfo.gCount = dealRowCount;
            fa_base_vector::Bmm2DataCopyOutNBSDMTiling<LAYOUT_T, OUT_T>(tmpBmm2ResCastTensor, transInfo, constInfo,
                                                                        actualSeqLengthsGmQ, attentionOutGm);
        }
    } else {
        Bmm2DataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, constInfo.headDimAlign, constInfo.headDim);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}


template <typename FIAT> __aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::FlashDecode(FDparams &fd)
{
    if (blockIdx >= fd.usedVecNumOfFd) {
        return;
    }
    uint32_t fdTaskPrevEnd = (blockIdx > 0) ? fd.gS1IdxEndOfFdHead[blockIdx - 1] : 0; // 上一个核末尾是第几个规约
    uint32_t fdS1gOuterMPrevEnd =
        (blockIdx > 0) ? fd.gS1IdxEndOfFdHeadSplit[blockIdx - 1] : 0; //上一个核末尾是该规约的第几个base行
    uint32_t fdTaskEnd = fd.gS1IdxEndOfFdHead[blockIdx];                 // 当前核的末尾是第几个规约任务
    uint32_t fdS1gOuterMEnd = fd.gS1IdxEndOfFdHeadSplit[blockIdx]; // 当前核的末尾是该规约的第几个base行
    uint32_t tmpFdS1gOuterMStart = (blockIdx > 0) ? fdS1gOuterMPrevEnd + 1 : 0; // 当前核从第几个base行开始
    uint32_t tmpFdS1gOuterMEnd = 0;
    uint32_t reduceGlobaLoop = 0;
    uint32_t reduceMLoop = 0;

    for (uint32_t fdTaskId = fdTaskPrevEnd; fdTaskId <= fdTaskEnd; fdTaskId++) {
        tmpFdS1gOuterMEnd = (fdTaskId == fdTaskEnd) ? fdS1gOuterMEnd : (fd.gS1SplitNumOfFdHead[fdTaskId] - 1);
        taskInfo.bIdx = fd.bN2IdxOfFdHead[fdTaskId];
        taskInfo.gS1Idx = fd.gS1IdxOfFdHead[fdTaskId] * constInfo.mBaseSize;
        taskInfo.actualCombineLoopSize = fd.s2SplitNumOfFdHead[fdTaskId]; // 当前规约任务kv方向有几份

        uint64_t combineTaskPrefixSum = 0;
        for (int i = 0; i < fdTaskId; i++) {
            // 计算此前规约数据的累计份数，每一份的数据大小为 kvHeadNum * constInfo.tndSgBasicSize
            // |Task0-0|Task0-1|Task0-3|Task1-0|Task1-2|...|
            combineTaskPrefixSum += fd.s2SplitNumOfFdHead[i];
        }

        uint64_t curAttenOutOffset = 0;

        if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
            curAttenOutOffset =
                taskInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim + 
                    taskInfo.gS1Idx * constInfo.headDim;
        } else {
            uint64_t actualSeqQPrefixSum;
            if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
                actualSeqQPrefixSum =
                    (taskInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1);
            } else { // BSND
                actualSeqQPrefixSum = (taskInfo.bIdx <= 0) ? 0 : taskInfo.bIdx * constInfo.qSeqSize;
            }
            uint64_t batchOutOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
            curAttenOutOffset = batchOutOffset + taskInfo.gS1Idx * constInfo.headDim;
        }

        uint64_t taskOffset = combineTaskPrefixSum * constInfo.kvHeadNum * constInfo.mBaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭

            uint32_t actualGSplitSize = fd.gS1BaseSizeOfFd;
            if (fdS1gOuterMIdx == fd.gS1SplitNumOfFdHead[fdTaskId] - 1) {
                actualGSplitSize = fd.gS1LastPartSizeOfFdHead[fdTaskId];
            }
            uint32_t startRow = fdS1gOuterMIdx * fd.gS1BaseSizeOfFd;

            LocalTensor<T> lseExp = fdLseExpBuf.Get<T>();
            LocalTensor<T> reduceOut = fdReduceBuf.Get<T>();
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);

            LocalTensor<T> mm2Res;
            for (uint32_t preLoadIdx = 0; preLoadIdx < constInfo.preLoadNum; preLoadIdx++) {
                mm2Res = (reduceGlobaLoop + preLoadIdx) % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
            }

            ComputeScaleValue(lseExp, startRow, actualGSplitSize, reduceMLoop);

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                mm2Res = reduceGlobaLoop % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                if (i >= constInfo.preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                }

                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, curAttenOutOffset);
            reduceMLoop += 1;
        }
        tmpFdS1gOuterMStart = 0;
    }
}
#endif
