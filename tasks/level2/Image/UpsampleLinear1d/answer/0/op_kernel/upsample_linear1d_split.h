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
 * \file upsample_linear1d.h
 * \brief
 */
#ifndef UPSAMPLE_LINEAR1D
#define UPSAMPLE_LINEAR1D

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "upsample_linear_common.h"

namespace UpsampleLinear1d {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

template <typename T>
class UpsampleLinear1dND {
public:
    TPipe* pipe = nullptr;
    matmul::MatmulImpl<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, 
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, MDL_CFG>
        matmulW;

    __aicore__ inline UpsampleLinear1dND(TPipe* pipeIn){
        pipe = pipeIn;
    };
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleLinear1dTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingCommon(const UpsampleLinear1dTilingData *tilingData);
    __aicore__ inline void ParseTilingAIV();
    __aicore__ inline void ParseTilingAIC(const UpsampleLinear1dTilingData *tilingData);
    __aicore__ inline void WDirectionExpansion(int64_t startNum, int64_t endNum, bool isRemainder);
    __aicore__ inline void RowLoopFunc(int64_t index, int16_t length, int64_t mmLoopTimes, int64_t mmBlockTail, int64_t mmLoopTailTimes, int64_t mmLoopTailNum);
    __aicore__ inline void RowLoopFuncRemainder(int64_t index, int16_t length, int64_t mmLoopTimes, int64_t mmBlockTail, int64_t mmLoopTailTimes, int64_t mmLoopTailNum);
    __aicore__ inline void calculateWidthExtension(int64_t rowNum);
    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline void getSlideRange(const UpsampleLinear1dTilingData *tilingData);
    __aicore__ inline void PreLoad(int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow);
    __aicore__ inline void AfterMatMul(int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow);
    __aicore__ inline void part1(uint64_t inputWorkStartOffset, int64_t rowOffset, int64_t m_i, int64_t loopT, int64_t tailNum);
    __aicore__ inline void part3(uint64_t outputWorkStartOffset, int64_t rowOffset, int64_t m_i, int64_t index, int16_t length, int64_t loopT, int64_t tailNum);
    __aicore__ inline void doAicMM();
    __aicore__ inline void calculateRadio(int64_t loopIndex, int64_t length, int64_t& xMin, int64_t& singleCoreK, float scale_w, bool align_corners, int64_t wIn, int64_t slide_size_w);
    __aicore__ inline void aicLoop(int64_t start, int64_t end, int64_t loopNum, int64_t tailNum);

private:
    // 系数矩阵下标队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue;
    TBuf<TPosition::VECCALC> inputBuf;
    TBuf<TPosition::VECCALC> outputBuf;

    TCubeTiling matmulTiling_w;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<float> intermediateTensorGm;

    bool align_corners = false;
    bool isAicAvilable = false;
    int64_t subIdx = 0;
    int64_t blockIdx = 0;
    int64_t aicIdx = 0;
    int64_t slide_size_w = 0;
    float scale_w;

    int64_t need_core_num_w;
    int64_t need_core_num_aic;

    uint32_t radio_matrix_size_w;

    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    int64_t slideNumW;
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;

    int64_t slideStart_w = 0;
    int64_t slideEnd_w = 0;
    int64_t tailSlideStart_w = 0;
    int64_t tailSlideEnd_w = 0;
    int64_t tailRowStart_w = 0;
    int64_t tailRowEnd_w = 0;

    int64_t inputW = 0;
    int64_t outputW = 0;

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    int64_t xMin = 0;

    int64_t mPerTime;
    int64_t loopTimes;
    int64_t loopTail;
    int64_t inputUbSize;
    int64_t outputUbSize;
    int64_t numPerBlock;
    int64_t matmulLoopTimes;
    int64_t matmulBlockTail;
    int64_t matmulBlockPerTime;

    int64_t loopTailTimes;
    int64_t loopTailTail;
    int64_t remainderMatmulLoopTimes;
    int64_t remainderMatmulBlockTail;
    int64_t remainderLoopTailTimes;
    int64_t remainderLoopTailTail;

    int64_t inputWorkStartOffset_0 = 0;
    int64_t outputWorkStartOffset_0 = 0;
    int64_t inputH = 0;
    int64_t totalPerCore = 0;
    int64_t singleCoreN = 0;
    int64_t inputWorkStartOffsetAic = 0;
    int64_t outputWorkStartOffsetAic = 0;
    int64_t mmInputNum = 0;
    int64_t remainder_aiv_1_calc_num = 0;
    int64_t remainder_mm = 0;
    int64_t matmul_block_0_num = 0;
    int64_t matmul_block_1_num = 0;
    int64_t remainder_matmul_block_0_num = 0;
    int64_t remainderLoopTailTail_0 = 0;
    int64_t mm_tail_0 = 0;
    int64_t remainder_matmul_tail_0 = 0;
};

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleLinear1dTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    subIdx = blockIdx % 2;
    ParseTilingCommon(tilingData);
    ParseTilingAIC(tilingData);
    ParseTilingAIV();
    getSlideRange(tilingData);
    if (!FloatEqual(scale_w, 1.0)) {
        pipe->InitBuffer(inputBuf, inputUbSize);
        pipe->InitBuffer(outputBuf, outputUbSize);
        pipe->InitBuffer(radioQueue, BUFFER_NUM, radio_matrix_size_w * sizeof(float));
    }
    intermediateTensorGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    inTensorsGM.SetGlobalBuffer((__gm__ T *)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)output);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::PreLoad(
    int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow) 
{
    if (numRow <= 0) {
        return ;
    }
    uint32_t numColAlign = (numCol + numPerBlock - 1) / numPerBlock * numPerBlock;
    uint32_t calCount = numRow * numColAlign;

    LocalTensor<float> inputLocalFp32 = inputBuf.Get<float>();
    DataCopyExtParams copyInParams{static_cast<uint16_t>(numRow), static_cast<uint32_t>(numCol * sizeof(T)), static_cast<uint32_t>((inputW - numCol) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(numColAlign - numCol), static_cast<T>(0.0)};
    
    event_t event_v_mte2_0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    LocalTensor<T> inputLocal = inputLocalFp32.ReinterpretCast<T>()[calCount];
    DataCopyPad(inputLocal, inTensorsGM[inOffset], copyInParams, padParams);
    SetFlag<HardEvent::V_MTE2>(event_v_mte2_0);
    WaitFlag<HardEvent::V_MTE2>(event_v_mte2_0);

    event_t event_mte2_v_0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event_mte2_v_0);
    WaitFlag<HardEvent::MTE2_V>(event_mte2_v_0);

    Cast(inputLocalFp32, inputLocal, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    
    event_t event_mte2_mte3_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(event_mte2_mte3_1);
    WaitFlag<HardEvent::MTE2_MTE3>(event_mte2_mte3_1);

    event_t event_v_mte3_0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event_v_mte3_0);
    WaitFlag<HardEvent::V_MTE3>(event_v_mte3_0);

    DataCopyExtParams copyOutParams{static_cast<uint16_t>(numRow), static_cast<uint32_t>(numCol * sizeof(float)), static_cast<uint32_t>(((numColAlign - numCol) * sizeof(float)) / 32), 0, 0};
    DataCopyPad(intermediateTensorGm[outOffset], inputLocalFp32, copyOutParams);
    
    event_t event_mte3_mte2_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2_1);
    WaitFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2_1);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::AfterMatMul(
    int64_t inOffset, int64_t outOffset, int64_t numCol, uint16_t numRow) 
{
    if (numRow <= 0) {
        return ;
    }
    uint32_t calCount = numRow * slide_size_w;
    event_t event_v_mte2_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(event_v_mte2_1);
    WaitFlag<HardEvent::V_MTE2>(event_v_mte2_1);

    LocalTensor<float> outputLocalFp32 = outputBuf.Get<float>();
    LocalTensor<T> outputLocal = outputLocalFp32.ReinterpretCast<T>();
    DataCopy(outputLocalFp32, intermediateTensorGm[inOffset], calCount);
    event_t event_mte2_v_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event_mte2_v_1);
    WaitFlag<HardEvent::MTE2_V>(event_mte2_v_1);
    if constexpr (std::is_same<T, half>::value) {
        Cast(outputLocal, outputLocalFp32, RoundMode::CAST_NONE, calCount);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(outputLocal, outputLocalFp32, RoundMode::CAST_RINT, calCount);
    } 
    PipeBarrier<PIPE_V>();
    event_t event_mte2_mte3_3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(event_mte2_mte3_3);
    WaitFlag<HardEvent::MTE2_MTE3>(event_mte2_mte3_3);

    event_t event_v_mte3_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event_v_mte3_1);
    WaitFlag<HardEvent::V_MTE3>(event_v_mte3_1);

    DataCopyExtParams copyOutParams{
        static_cast<uint16_t>(numRow),
        static_cast<uint32_t>(numCol * sizeof(T)),
        static_cast<uint32_t>(((slide_size_w - numCol) * sizeof(T)) / 32),
        static_cast<uint32_t>((outputW - numCol) * sizeof(T)),
        0
    };
    DataCopyPad(outTensorsGM[outOffset], outputLocal, copyOutParams);
    event_t event_mte3_mte2_3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2_3);
    WaitFlag<HardEvent::MTE3_MTE2>(event_mte3_mte2_3);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::calculateRadio(
    int64_t loopIndex, int64_t length, int64_t& xMin, int64_t& singleCoreK, 
    float scale_w, bool align_corners, int64_t wIn, int64_t slide_size_w)
{
    calculateSingleCoreK(loopIndex, length, xMin, singleCoreK, scale_w, align_corners, wIn);
    if(subIdx == 1) {
        return ;
    }
    LocalTensor<float> radioTensor = radioQueue.AllocTensor<float>();
    Duplicate(radioTensor, (float)0.0, radioTensor.GetSize());
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    calculateRadioTensorW(loopIndex, length, radioTensor, xMin, singleCoreK, scale_w, align_corners, wIn, slide_size_w);
    radioQueue.EnQue(radioTensor);
    copyRadioTensorToGm(0);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::Process()
{
    if ASCEND_IS_AIV {
        if (FloatEqual(scale_w, 1.0) || blockIdx >= need_core_num_w) {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(VEC_FLAG_ID_1);
            return ;
        }
        WDirectionExpansion(slideStart_w, slideEnd_w, false);
        WDirectionExpansion(tailSlideStart_w, tailSlideEnd_w, true);
        
    }
    if ASCEND_IS_AIC {
        if (!isAicAvilable) {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(VEC_FLAG_ID_1);
            return ;
        }
        doAicMM();
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::WDirectionExpansion(int64_t startNum, int64_t endNum, bool isRemainder)
{   
    if (startNum < endNum) {
        for (int64_t index = startNum; index < endNum; index += slide_size_w) {
            int16_t length = Min(slide_size_w, endNum - index);
            calculateRadio(index, length, xMin, singleCoreK, scale_w, align_corners, inputW, slide_size_w);
            if (isRemainder) {
                RowLoopFuncRemainder(index, length, remainderMatmulLoopTimes, remainderMatmulBlockTail, remainderLoopTailTimes, remainderLoopTailTail);
            } else {
                RowLoopFunc(index, length, matmulLoopTimes, matmulBlockTail, loopTailTimes, loopTailTail);
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::part1(uint64_t inputWorkStartOffset, int64_t rowOffset, int64_t m_i, int64_t loopT, int64_t tailNum)
{
    int64_t rowNum = 0;
    int64_t preOutOffset = inputWorkStartOffset;
    for (int64_t ii = 0; ii < loopT; ii ++) {
        int64_t rowStart = Min(rowOffset + ii * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + mPerTime, inputH);
        int64_t preInOffset = rowStart * inputW + xMin;
        preOutOffset = preOutOffset + rowNum * singleCoreK;
        rowNum = rowEnd - rowStart;
        PreLoad(preInOffset, preOutOffset, singleCoreK, rowNum);
    }
    if(tailNum > 0) {
        int64_t rowStart = Min(rowOffset + loopT * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + tailNum, inputH);
        int64_t preInOffset = rowStart * inputW + xMin;
        preOutOffset = preOutOffset + rowNum * singleCoreK;
        rowNum = rowEnd - rowStart;
        PreLoad(preInOffset, preOutOffset, singleCoreK, rowNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::part3(
    uint64_t outputWorkStartOffset, int64_t rowOffset, int64_t m_i, 
    int64_t index, int16_t length, int64_t loopT, 
    int64_t tailNum)
{
    int64_t afterInOffset = outputWorkStartOffset;
    int64_t rowNum = 0;
    for (int64_t ii = 0; ii < loopT; ii ++) {
        int64_t rowStart = Min(rowOffset + ii * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + mPerTime, inputH);
        int64_t afterOutOffset = rowStart * outputW + index;
        afterInOffset = afterInOffset + rowNum * slide_size_w;
        rowNum = rowEnd - rowStart;
        AfterMatMul(afterInOffset, afterOutOffset, length, rowNum);
    }
    if(tailNum > 0) {
        int64_t rowStart = Min(rowOffset + loopT * mPerTime, inputH);
        int64_t rowEnd = Min(rowStart + tailNum, inputH);
        int64_t afterOutOffset = rowStart * outputW + index;
        afterInOffset = afterInOffset + rowNum * slide_size_w;
        rowNum = rowEnd - rowStart;
        AfterMatMul(afterInOffset, afterOutOffset, length, rowNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::RowLoopFuncRemainder(
    int64_t index, int16_t length, 
    int64_t mmLoopTimes, int64_t mmBlockTail, int64_t mmLoopTailTimes, 
    int64_t mmLoopTailNum)
{
    int64_t inputWorkStartOffset = inputWorkStartOffset_0 + subIdx * remainder_matmul_block_0_num * singleCoreK;
    int64_t outputWorkStartOffset = outputWorkStartOffset_0 + subIdx * remainder_matmul_block_0_num * slide_size_w;
    int64_t rowOffset = tailRowStart_w + subIdx * remainder_matmul_block_0_num;
    for (int64_t m_i = 0; m_i < mmLoopTimes; m_i ++) {
        rowOffset = tailRowStart_w + m_i * matmulBlockPerTime + subIdx * remainder_matmul_block_0_num;
        part1(inputWorkStartOffset, rowOffset, m_i, loopTimes, loopTail);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(VEC_FLAG_ID_1);
        part3(outputWorkStartOffset, rowOffset, m_i, index, length, loopTimes, loopTail);
    }

    if (mmBlockTail > 0) {
        rowOffset = tailRowStart_w + mmLoopTimes * matmulBlockPerTime + subIdx * remainder_matmul_tail_0;
        inputWorkStartOffset = inputWorkStartOffset_0 + subIdx * remainder_matmul_tail_0 * singleCoreK;
        outputWorkStartOffset = outputWorkStartOffset_0 + subIdx * remainder_matmul_tail_0 * slide_size_w;
        part1(inputWorkStartOffset, rowOffset, mmLoopTimes, mmLoopTailTimes, mmLoopTailNum);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_2);
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(VEC_FLAG_ID_3);
        part3(outputWorkStartOffset, rowOffset, mmLoopTimes, index, length, mmLoopTailTimes, mmLoopTailNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::RowLoopFunc(
    int64_t index, int16_t length, 
    int64_t mmLoopTimes, int64_t mmBlockTail, int64_t mmLoopTailTimes, 
    int64_t mmLoopTailNum)
{
    int64_t inputWorkStartOffset = inputWorkStartOffset_0 + subIdx * matmul_block_0_num * singleCoreK;
    int64_t outputWorkStartOffset = outputWorkStartOffset_0 + subIdx * matmul_block_0_num * slide_size_w;
    int64_t rowOffset = subIdx * matmul_block_0_num;
    for (int64_t m_i = 0; m_i < mmLoopTimes; m_i ++) {
        rowOffset = m_i * matmulBlockPerTime + subIdx * matmul_block_0_num;
        part1(inputWorkStartOffset, rowOffset, m_i, loopTimes, loopTail);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(VEC_FLAG_ID_1);
        part3(outputWorkStartOffset, rowOffset, m_i, index, length, loopTimes, loopTail);
    }

    if (mmBlockTail > 0) {
        rowOffset = mmLoopTimes * matmulBlockPerTime + subIdx * mm_tail_0;
        inputWorkStartOffset = inputWorkStartOffset_0 + subIdx * mm_tail_0 * singleCoreK;
        outputWorkStartOffset = outputWorkStartOffset_0 + subIdx * mm_tail_0 * slide_size_w;
        part1(inputWorkStartOffset, rowOffset, mmLoopTimes, mmLoopTailTimes, mmLoopTailNum);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_2);
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(VEC_FLAG_ID_3);
        part3(outputWorkStartOffset, rowOffset, mmLoopTimes, index, length, mmLoopTailTimes, mmLoopTailNum);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::copyRadioTensorToGm(int8_t direction)
{
    // 系数矩阵从ub拷贝到GM
    LocalTensor<float> radioTensor = radioQueue.DeQue<float>();
    DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensor, radioTensor.GetSize());
    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    radioQueue.FreeTensor(radioTensor);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::doAicMM()
{
    if (slideStart_w < slideEnd_w) {
        aicLoop(slideStart_w, slideEnd_w, matmulLoopTimes, matmulBlockTail);
    }
    if (tailSlideStart_w < tailSlideEnd_w) {
        aicLoop(tailSlideStart_w, tailSlideEnd_w, remainderMatmulLoopTimes, remainderMatmulBlockTail);
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::aicLoop(int64_t start, int64_t end, int64_t loopNum, int64_t tailNum)
{
    for (int64_t index = start; index < end; index += slide_size_w) {
        int16_t length = Min(slide_size_w, end - index);
        calculateSingleCoreK(index, length, xMin, singleCoreK, scale_w, align_corners, inputW);
        for (int64_t m_i = 0; m_i < loopNum; m_i ++) {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_0);
            calculateWidthExtension(matmulBlockPerTime);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(VEC_FLAG_ID_1);
        }
        if (tailNum > 0) {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE3>(VEC_FLAG_ID_2);
            calculateWidthExtension(tailNum);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(VEC_FLAG_ID_3);
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::calculateWidthExtension(int64_t rowNum)
{
    if (singleCoreK > 0) {
        // 尾块batch分批处理
        matmulW.SetOrgShape(rowNum, singleCoreN, singleCoreK, singleCoreK, singleCoreN);
        matmulW.SetSingleShape(rowNum, singleCoreN, singleCoreK);
        matmulW.SetTensorA(intermediateTensorGm[inputWorkStartOffsetAic], false);
        matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
        matmulW.IterateAll(intermediateTensorGm[outputWorkStartOffsetAic], false);
        matmulW.End();
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::ParseTilingAIC(const UpsampleLinear1dTilingData *tilingData)
{
    if ASCEND_IS_AIV {
        return ;
    }
    aicIdx = blockIdx;
    inputWorkStartOffsetAic = totalPerCore * blockIdx + radio_matrix_size_w;
    outputWorkStartOffsetAic = totalPerCore * blockIdx + radio_matrix_size_w + mmInputNum;
    workSpaceRadioOffset = totalPerCore * blockIdx;
    matmulTiling_w = tilingData->matmulTiling_w;
    matmulW.Init(&matmulTiling_w);
    singleCoreN = matmulTiling_w.singleCoreN;
    isAicAvilable = blockIdx < need_core_num_aic;
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::ParseTilingAIV()
{
    if ASCEND_IS_AIC {
        return ;
    }
    aicIdx = (blockIdx / 2);
    inputWorkStartOffset_0 = totalPerCore * aicIdx + radio_matrix_size_w;
    outputWorkStartOffset_0 = totalPerCore * aicIdx + radio_matrix_size_w + mmInputNum;
    workSpaceRadioOffset = totalPerCore * aicIdx;
    isAicAvilable = false;
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::ParseTilingCommon(const UpsampleLinear1dTilingData *tilingData)
{
    align_corners = tilingData->align_corners;
    slide_size_w = tilingData->slide_size_w;
    scale_w = tilingData->scale_w;
    need_core_num_aic = tilingData->need_core_num_w;
    need_core_num_w = need_core_num_aic * 2;
    inputW = tilingData->input_shapes[2];
    outputW = tilingData->output_shapes[2];
    radio_matrix_size_w = tilingData->radio_matrix_size_w;
    numPerBlock = tilingData->blockSizeNum;
    eachCoreSlideNumW = tilingData->eachCoreSlideNumW;
    tailStartSlideNumW = tilingData->tailStartSlideNumW;
    slideNumW = tilingData->slideNumW;
    groupCoreNumW = tilingData->groupCoreNumW;
    tailAvergingRowsW = tilingData->tailAvergingRowsW;
    remainderW = tilingData->remainderW;
    inputUbSize = tilingData->inputUbSize;
    outputUbSize = tilingData->outputUbSize;
    totalPerCore = tilingData->mmtotalPerCoreNum;
    mmInputNum = tilingData->mmInputNum;
    inputH = tilingData->inputH;
    matmulBlockPerTime = tilingData->matmulBlockPerTime;
    matmulLoopTimes = tilingData->matmulLoopTimes;
    matmulBlockTail = tilingData->matmulBlockTail;
    mm_tail_0 = tilingData->matmulBlockTail0;

    mPerTime = tilingData->mPerTime;
    loopTimes = subIdx == 0 ? tilingData->loopTimes0 : tilingData->loopTimes1;
    loopTail = subIdx == 0 ? tilingData->loopTail0 : tilingData->loopTail1;
    loopTailTimes = subIdx == 0 ? tilingData->loopTailTimes0 : tilingData->loopTailTimes1;
    loopTailTail = subIdx == 0 ? tilingData->loopTailTail0 : tilingData->loopTailTail1;
    matmul_block_0_num = tilingData->matmulBlockPerTime0;
}

template <typename T>
__aicore__ inline void UpsampleLinear1dND<T>::getSlideRange(const UpsampleLinear1dTilingData *tilingData)
{
    slideStart_w = aicIdx * eachCoreSlideNumW * slide_size_w;
    slideEnd_w = Min((Min((aicIdx + 1) * eachCoreSlideNumW, slideNumW)) * slide_size_w, outputW);
    int64_t groupIndex = groupCoreNumW > 0 ? aicIdx / groupCoreNumW : 0;
    if (groupIndex < remainderW) {
        tailSlideStart_w = (tailStartSlideNumW + groupIndex) * slide_size_w;
        tailSlideEnd_w = Min(tailSlideStart_w + slide_size_w, outputW);
        int64_t blockIdxInGroup = groupCoreNumW > 0 ? aicIdx % groupCoreNumW : 0;
        tailRowStart_w = blockIdxInGroup * tailAvergingRowsW;
        tailRowEnd_w = Min(tailRowStart_w + tailAvergingRowsW, inputH);
        int64_t remainder_num = Min(tailRowEnd_w - tailRowStart_w, matmulBlockPerTime);
        remainderMatmulLoopTimes = tilingData->remainderMatmulLoopTimes;
        remainderMatmulBlockTail = tilingData->remainderMatmulBlockTail;
        remainder_matmul_tail_0 = tilingData->remainderMatmulBlockTail0;
        remainderLoopTailTimes = subIdx == 0 ? tilingData->remainderLoopTailTimes0 : tilingData->remainderLoopTailTimes1;
        remainderLoopTailTail_0 = tilingData->remainderLoopTailTail0;
        remainderLoopTailTail = subIdx == 0 ? remainderLoopTailTail_0 : tilingData->remainderLoopTailTail1;
        remainder_matmul_block_0_num = (remainder_num / 2) > 0 ? (remainder_num / 2) : remainder_num;
    }
}
}  // namespace UpsampleLinear1d

#endif  // UPSAMPLE_LINEAR1D