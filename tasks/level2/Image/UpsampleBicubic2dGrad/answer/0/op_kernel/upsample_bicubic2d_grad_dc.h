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
 * \file upsample_bicubic2d_grad_dc.h
 * \brief
 */

#ifndef UPSAMPLE_BICUBIC2D_GRAD_DC
#define UPSAMPLE_BICUBIC2D_GRAD_DC

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace UpsampleBicubic2dGrad {
using namespace AscendC;

constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;

constexpr int32_t NUMBER_TWO = 2;
constexpr int32_t NUMBER_THREE = 3;
constexpr int32_t NUMBER_FOUR = 4;
constexpr int32_t NUMBER_SIX = 6;

constexpr int32_t DATA_BLOCK_BYTES = 32;
constexpr int32_t ONE_K_BYTES = 1024;
constexpr MatmulConfig MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

template <typename T>
class UpsampleBicubic2dGradDCND {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,MDL_CFG>
        matmulW;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,MDL_CFG>
        matmulH;
    __aicore__ inline UpsampleBicubic2dGradDCND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, UpsampleBicubic2dGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcWeights(float (&weights)[4], float tValue)
    {
        float x1 = tValue;  // tValue 为当前中心点偏移值，x1为左侧点偏移值
        weights[0] = CalcWeight1(x1 + 1);
        weights[1] = CalcWeight2(x1);
        float x2 = 1 - tValue;  // tValue 为当前中心点偏移值，x2为右侧点偏移值
        weights[NUMBER_TWO] = CalcWeight2(x2);
        weights[NUMBER_THREE] = CalcWeight1(x2 + 1);  // x2为右侧点偏移值，计算第二个点偏移值
    };
    // 计算weight,可将a替换为固定值
    __aicore__ inline float CalcWeight1(float x)
    {
        constexpr float COEFFICIENT_1 = -0.75f;
        constexpr float COEFFICIENT_2 = 3.75f;
        return ((x * COEFFICIENT_1 + COEFFICIENT_2) * x - static_cast<float>(NUMBER_SIX)) * x +
               static_cast<float>(NUMBER_THREE);
    };
    __aicore__ inline float CalcWeight2(float x)
    {
        constexpr float COEFFICIENT_1 = 1.25f;
        constexpr float COEFFICIENT_2 = 2.25f;
        return (x * COEFFICIENT_1 - COEFFICIENT_2) * x * x + 1.0f;
    };
    template <typename T1, typename T2>
    __aicore__ inline auto AlignUp(T1 a, T2 b) -> decltype(a + b)
    {
        if (b <= 0) {
            return a;
        }
        auto ca = static_cast<decltype(a + b)>(a);
        auto cb = static_cast<decltype(a + b)>(b);
        if (ca % cb == 0) {
            return ca;
        }
        return (ca + cb - 1) / cb * cb;
    }
    template <typename T1, typename T2>
    __aicore__ inline auto AlignDown(T1 a, T2 b) -> decltype(a + b)
    {
        if (b <= 0) {
            return a;
        }
        auto ca = static_cast<decltype(a + b)>(a);
        auto cb = static_cast<decltype(a + b)>(b);
        if (ca % cb == 0) {
            return ca;
        }
        return ca / cb * cb;
    }
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        return b == 0 ? a : ((a + b - 1) / b);
    };
    template <typename T1>
    __aicore__ inline int64_t Ceil(T1 x)
    {
        if (x < 0) {
            x = x - 1;
        }
        int64_t floor_v = int64_t(x);
        return x == floor_v ? floor_v : (floor_v + 1);
    };
    template <typename T1, typename T2>
    __aicore__ inline auto GetMin(T1 a, T2 b)
    {
        return a < b ? a : b;
    };
    template <typename T1, typename T2>
    __aicore__ inline auto GetMax(T1 a, T2 b)
    {
        return a >= b ? a : b;
    };
    __aicore__ inline void InitGlobalTensors(GM_ADDR input, GM_ADDR output, GM_ADDR workspace);
    __aicore__ inline void InitScalars();
    __aicore__ inline void InitLocalTensors();
    __aicore__ inline void WDirectionExpansion();
    __aicore__ inline void HDirectionExpansion();
    __aicore__ inline void CalculateIntermediateTensor(
        int64_t xMinStart, int64_t maxIdx, float scale, int64_t length);
    __aicore__ inline int64_t CalculateInstartIdx(int64_t startIdx, float scale);
    __aicore__ inline void ParseTilingData(UpsampleBicubic2dGradTilingData *tilingData);
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount);
    __aicore__ inline __gm__ T *GetTensorAddr(int64_t index, GM_ADDR tensorPtr);
    __aicore__ inline void CalculateRadioTensor(int64_t index, int64_t length, int64_t direction, int64_t slideKNum);
    __aicore__ inline void calculateWidthExtension(int64_t xMin, int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void CopyRadioTensorToGm(int64_t length, int64_t kStartIdx, int64_t slideKNum);
    __aicore__ inline void CopyRadioTensorToGmY(int64_t length, int64_t singleCoreK, int64_t kStartIdx, int64_t slideKNum);
    __aicore__ inline void calculateHeightExtension(int64_t xMin, int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void InitEventId();

private:
    TBuf<QuePosition::VECCALC> ubBuf;

    const TCubeTiling *__restrict matmulTilingW;
    const TCubeTiling *__restrict matmulTilingH;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<T> intermediateTensorGm;

    LocalTensor<float> centerTensor;
    LocalTensor<float> xTensor;
    LocalTensor<float> tTensor;

    LocalTensor<float> radioTensor;
    LocalTensor<T> radioCopyOutTensor;

    event_t eventIDVToS;
    event_t eventIDSToV;
    event_t eventIdMTE3ToMTE2;
    event_t eventIdSToMTE3;
    event_t eventIdVToMTE3;
    event_t eventIdMTE3ToV;
    event_t eventIdMTE3ToS;

    int64_t ubMaxBytes = 0;

    int64_t xMin = 0;
    int64_t aiCoreIdx = 0;
    int64_t blockIdx = 0;
    int64_t slideSize = 0;

    int64_t alignCorners = 0;
    float scaleW;
    float scaleH;

    uint64_t intermediateMatrixSize = 16;
    int64_t radioMatrixSize;
    // 切分块在原系数矩阵中的位置
    int64_t slideStartW;
    int64_t slideEndW;
    int64_t tailSlideStartW;
    int64_t tailSlideEndW;
    int64_t tailRowStartW;
    int64_t tailRowEndW;

    // 系数矩阵切块的宽度
    int64_t slidelenW;
    int64_t slidelenH;

    int64_t slideStartH;
    int64_t slideEndH;
    int64_t tailSlideStartH;
    int64_t tailSlideEndH;
    int64_t tailRowStartH;
    int64_t tailRowEndH;

    float realScaleW = 0;
    float realScaleH = 0;
    int64_t inputShapes[4] = {0, 0, 0, 0};
    int64_t outputShapes[4] = {0, 0, 0, 0};

    uint32_t needCoreNumW;
    uint32_t needCoreNumH;

    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreMaxKW = 0;
    int64_t singleCoreMaxKH = 0;
    int64_t singleCoreKW = 0;
    int64_t singleCoreKH = 0;

    bool needExpandW = false;
    bool needExpandH = false;

    bool isZeroVecCore = false;  // 当前是否为aicore的第0个vector核

    int64_t splitSingleCoreKMax = 0; // 单次切K的最大值
    int64_t perDataBlockNum = 1; // 每个dataBlock内的T类型数据量
    int64_t maxRadioBytes = 0; // 系数矩阵的最大内存
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, UpsampleBicubic2dGradTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    aiCoreIdx = blockIdx / 2;
    ParseTilingData(tilingData);

    InitScalars();
    InitGlobalTensors(input, output, workspace);

    InitEventId();
    InitLocalTensors();
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::InitScalars()
{
    isZeroVecCore = blockIdx % 2 == 0;

    realScaleW = (scaleW > 0 || inputShapes[3] > 0)? scaleW :1;
    realScaleH = (scaleH > 0 || inputShapes[2] > 0)? scaleH :1;

    ubMaxBytes = ubMaxBytes - ONE_K_BYTES;
    perDataBlockNum = DATA_BLOCK_BYTES / sizeof(T);
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::InitGlobalTensors(GM_ADDR input, GM_ADDR output, GM_ADDR workspace)
{
    intermediateTensorGm.SetGlobalBuffer((__gm__ T *)workspace);
    inTensorsGM.SetGlobalBuffer((__gm__ T *)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)output);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::InitEventId()
{
    eventIDVToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_S>());
    eventIDSToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_V>());
    eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE3>());
    eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    eventIdMTE3ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>());
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::InitLocalTensors()
{
    pipe.InitBuffer(ubBuf, ubMaxBytes);
    LocalTensor<uint8_t> tmp = ubBuf.Get<uint8_t>();
    // 均分ub,且32字节对齐
    int64_t imetermediateSize = AlignDown(ubMaxBytes / (slideSize + NUMBER_TWO), DATA_BLOCK_BYTES);
    int64_t offset = 0;
    // centerTensor与tTensor可共用空间
    centerTensor = tmp.ReinterpretCast<float>();
    tTensor = centerTensor;
    offset += imetermediateSize;
    xTensor = tmp[offset].ReinterpretCast<float>();
    offset += imetermediateSize;
    radioTensor= tmp[offset].ReinterpretCast<float>();
    radioCopyOutTensor = radioTensor.template ReinterpretCast<T>();

    maxRadioBytes = imetermediateSize * slideSize;
    splitSingleCoreKMax = imetermediateSize / sizeof(float);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::Process()
{
    // 先横向扩展
    if (needExpandW) {
        WDirectionExpansion();
    }

    SyncAll();

    // 再纵向扩展
    if (needExpandH || !needExpandW) {
        HDirectionExpansion();
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::WDirectionExpansion()
{
    int64_t count = 0;
    if (aiCoreIdx >= needCoreNumW) {
        return;
    }
    if (slideStartW < slideEndW) {
        for (int64_t index = slideStartW; index < slideEndW; index += slideSize) {
            if (isZeroVecCore == (++count % 2 == 0)) continue;
            xMin = CalculateInstartIdx(index, realScaleW);
            xMin = GetMin(xMin,inputShapes[3] -1);
            singleCoreKW = GetMin(singleCoreMaxKW, inputShapes[3] - xMin);
            int64_t slideLength = GetMin(slideSize, slideEndW - index);
            for (int64_t kStart = 0;kStart< GetMax(singleCoreKW,1);kStart+=splitSingleCoreKMax) {
                int64_t slideKNum = GetMin((singleCoreKW - kStart),splitSingleCoreKMax);
                CalculateIntermediateTensor(kStart + xMin, outputShapes[3] - 1, realScaleW, slideKNum);
                CalculateRadioTensor(index, slideLength, 0, slideKNum);
                CopyRadioTensorToGm(slideLength,kStart,slideKNum);
            }
            calculateWidthExtension(xMin, index, 0, 0);
        }
    }
    if (tailRowStartW < tailRowEndW) {
        for (int64_t index = tailSlideStartW; index < tailSlideEndW; index += slideSize) {
            if (isZeroVecCore == (++count % 2 == 0)) continue;
            xMin = CalculateInstartIdx(index, realScaleW);
            xMin = GetMin(xMin,inputShapes[3] -1);
            singleCoreKW = GetMin(singleCoreMaxKW, inputShapes[3] - xMin);
            int64_t slideLength = GetMin(slideSize, tailSlideEndW - index);
            for (int64_t kStart = 0;kStart< GetMax(singleCoreKW,1);kStart+=splitSingleCoreKMax) {
                int64_t slideKNum = GetMin((singleCoreKW - kStart),splitSingleCoreKMax);
                CalculateIntermediateTensor(kStart + xMin, outputShapes[3] - 1, realScaleW, slideKNum);
                CalculateRadioTensor(index, slideLength, 0, slideKNum);
                CopyRadioTensorToGm(slideLength,kStart,slideKNum);
            }
            calculateWidthExtension(xMin, index, tailRowStartW, tailRowEndW);
        }
    }
};

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::HDirectionExpansion()
{
    if (aiCoreIdx >= needCoreNumH) {
        return;
    }
    int64_t count = 0;
    if (slideStartH < slideEndH) {
        for (int64_t index = slideStartH; index < slideEndH; index += slideSize) {
            if (isZeroVecCore == (++count % 2 == 0)) continue;
            xMin = CalculateInstartIdx(index, realScaleH);
            xMin = GetMin(xMin,inputShapes[2] -1);
            singleCoreKH = GetMin(singleCoreMaxKH, inputShapes[2] - xMin);
            int64_t slideLength = GetMin(slideSize, slideEndH - index);
            for (int64_t kStart = 0;kStart< GetMax(singleCoreKH,1);kStart+=splitSingleCoreKMax) {
                int64_t slideKNum = GetMin((singleCoreKH - kStart),splitSingleCoreKMax);
                CalculateIntermediateTensor(kStart + xMin, outputShapes[2] - 1, realScaleH, slideKNum);
                CalculateRadioTensor(index, slideLength, 1, slideKNum);
                CopyRadioTensorToGmY(slideLength,singleCoreKH,kStart,slideKNum);
            }
            calculateHeightExtension(xMin, index, 0, 0);
        }
    }
    if (tailRowStartH < tailRowEndH) {
        for (int64_t index = tailSlideStartH; index < tailSlideEndH; index += slideSize) {
            if (isZeroVecCore == (++count % 2 == 0)) continue;
            xMin = CalculateInstartIdx(index, realScaleH);
            xMin = GetMin(xMin,inputShapes[2] -1);
            singleCoreKH = GetMin(singleCoreMaxKH, inputShapes[2] - xMin);
            int64_t slideLength = GetMin(slideSize, tailSlideEndH - index);
            for (int64_t kStart = 0;kStart< GetMax(singleCoreKH,1);kStart+=splitSingleCoreKMax) {
                int64_t slideKNum = GetMin((singleCoreKH - kStart),splitSingleCoreKMax);
                CalculateIntermediateTensor(kStart + xMin, outputShapes[2] - 1, realScaleH, slideKNum);
                CalculateRadioTensor(index, slideLength, 1, slideKNum);
                CopyRadioTensorToGmY(slideLength,singleCoreKH,kStart,slideKNum);
            }
            calculateHeightExtension(xMin, index, tailRowStartH, tailRowEndH);
        }
    }
};

template <typename T>
__aicore__ inline int64_t UpsampleBicubic2dGradDCND<T>::CalculateInstartIdx(int64_t startIdx, float scale)
{
    if (alignCorners) {
        return GetMax(Ceil((startIdx - NUMBER_TWO) / scale), (int64_t)0);
    } else {
        return GetMax(Ceil((startIdx - float(1.5)) / scale - (float)0.5), (int64_t)0);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::CalculateIntermediateTensor(
    int64_t xMinStart, int64_t maxIdx, float scale, int64_t length)
{
    // 使用标量计算中心点坐标，和cpu保持一致
    for (int64_t i = xMinStart; i < length + xMinStart; i++) {
        float value;
        if (alignCorners) {
            value = float(i) * scale;
        } else {
            value = (float(i) + float(0.5)) * scale - float(0.5);
        }
        centerTensor.SetValue(i - xMinStart, value);
    }
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);

    Floor(xTensor, centerTensor, length);
    PipeBarrier<PIPE_V>();

    Mins(xTensor, xTensor, (float)(maxIdx), length);
    PipeBarrier<PIPE_V>();

    Sub(tTensor, centerTensor, xTensor, length);
    PipeBarrier<PIPE_V>();

    Mins(tTensor, tTensor, (float)1, length);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::CalculateRadioTensor(int64_t index, int64_t length, int64_t direction, int64_t slideKNum)
{
    // 当为H方向时，采用向上取整方式计算radio在K维度的num
    int64_t alignSlideKNum = direction == 0 ? slideKNum : AlignUp(GetMax(slideKNum,1), perDataBlockNum);
    // 初始化为0
    Duplicate(radioTensor, float(0.0), GetMax(alignSlideKNum,perDataBlockNum) * length);
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    float weights[4] = {0};
    bool isInputLeftBorder = index == 0;
    bool isInputRightBorder = index + length == (direction == 0 ? outputShapes[3] : outputShapes[2]);
    for (int64_t i = 0; i < slideKNum; i++) {
        int64_t xValue = xTensor.GetValue(i);
        if (xValue + NUMBER_TWO < index || xValue > index + length) continue;

        int64_t xIdx = xValue - index;
        CalcWeights(weights, tTensor.GetValue(i));
        int64_t idxFirst = i;
        for(int64_t j = 0;j< NUMBER_FOUR;j++) {
            if(weights[j] == 0) continue;
            int64_t idxSecond = xIdx - 1 + j;
            if ((idxSecond < 0 && !isInputLeftBorder)|| (idxSecond >= length && !isInputRightBorder)) {
                continue;
            }
            idxSecond = GetMin(GetMax(idxSecond, 0), length - 1);
            int64_t realIdx = direction == 0 ? (idxFirst * length + idxSecond) : (idxSecond * alignSlideKNum + idxFirst);
            if (isInputLeftBorder || isInputRightBorder) {
                radioTensor.SetValue(realIdx, radioTensor.GetValue(realIdx) + weights[j]);
            } else {
                radioTensor.SetValue(realIdx, weights[j]);
            }
        }
    }
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);


    if constexpr (!IsSameType<T, float>::value) {
        Cast(radioCopyOutTensor, radioTensor, RoundMode::CAST_RINT, alignSlideKNum * length);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    } else {
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::CopyRadioTensorToGm(int64_t length, int64_t kStartIdx, int64_t slideKNum)
{
    workSpaceRadioOffset = intermediateMatrixSize + radioMatrixSize * blockIdx;
    int64_t wsOffset = workSpaceRadioOffset + kStartIdx * length;
    int64_t copyNum = AlignUp(GetMax(slideKNum, 1) * length, perDataBlockNum);

    DataCopy(intermediateTensorGm[wsOffset], radioCopyOutTensor, copyNum);
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::CopyRadioTensorToGmY(int64_t length, int64_t singleCoreK, int64_t kStartIdx, int64_t slideKNum)
{
    workSpaceRadioOffset = intermediateMatrixSize + radioMatrixSize * blockIdx;
    int64_t wsOffset = workSpaceRadioOffset + kStartIdx;

    int64_t copyNum = GetMax(slideKNum,1);
    int64_t dstStrideNum = singleCoreK - slideKNum;
    if (copyNum % perDataBlockNum == 0 && dstStrideNum % perDataBlockNum == 0) {
        DataCopyParams copyParams{static_cast<uint16_t>(length),static_cast<uint16_t>(copyNum / perDataBlockNum), 0, static_cast<uint16_t>(dstStrideNum / perDataBlockNum)};
        DataCopy(intermediateTensorGm[wsOffset], radioCopyOutTensor, copyParams);
    } else {
        DataCopyExtParams copyParams{static_cast<uint16_t>(length), (uint32_t)(copyNum * sizeof(T)), 0, static_cast<uint32_t>(dstStrideNum * sizeof(T)), 0};
        DataCopyPad(intermediateTensorGm[wsOffset], radioCopyOutTensor, copyParams);
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::calculateWidthExtension(
    int64_t xMin, int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTilingW->singleCoreM;
    int64_t singleCoreN = matmulTilingW->singleCoreN;

    if (singleCoreKW == 0) {
        singleCoreKW++;
    }

    if (tensorCIndex + slideSize > outputShapes[3]) {
        singleCoreN = outputShapes[3] - tensorCIndex;
    }

    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }

    matmulW.SetOrgShape(singleCoreM, singleCoreN, inputShapes[3], singleCoreKW, outputShapes[3]);

    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreKW);

    if (tensorCIndex + slideSize > outputShapes[3] - 1) {
        matmulW.SetTail(singleCoreM, outputShapes[3] - tensorCIndex, singleCoreKW);
    }
    int64_t xIndex = xMin + rowStart * inputShapes[3];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * outputShapes[3];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);

    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);

    if (!needExpandH) {
        matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    } else {
        matmulW.IterateAll(intermediateTensorGm[tensorCIndexWithOffset], false);
    }
    matmulW.End();
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::calculateHeightExtension(
    int64_t xMin, int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    int64_t singleCoreM = matmulTilingH->singleCoreM;
    int64_t singleCoreN = matmulTilingH->singleCoreN;

    if (singleCoreKH == 0) {
        singleCoreKH++;
    }
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreN = rowEnd - rowStart;
    }

    if (tensorCIndex + slideSize > outputShapes[2]) {
        singleCoreM = outputShapes[2] - tensorCIndex;
    }
    matmulH.SetOrgShape(singleCoreM, outputShapes[3], singleCoreKH, inputShapes[2], outputShapes[3]);

    matmulH.SetSingleShape(singleCoreM, singleCoreN, singleCoreKH);

    if (tensorCIndex + slideSize > outputShapes[2] - 1) {
        matmulH.SetTail(outputShapes[2] - tensorCIndex, singleCoreN, singleCoreKH);
    }

    int64_t xIndex = xMin * outputShapes[3] + rowStart;
    int64_t tensorCIndexWithOffset = tensorCIndex * outputShapes[3] + rowStart;

    for (int i = 0; i < outputShapes[0] * outputShapes[1]; i++) {
        // 系数矩阵起始位置
        matmulH.SetTensorA(intermediateTensorGm[workSpaceRadioOffset], false);

        if (!needExpandW) {
            matmulH.SetTensorB(inTensorsGM[xIndex + i * inputShapes[2] * outputShapes[3]], false);
        } else {
            matmulH.SetTensorB(intermediateTensorGm[xIndex + i * inputShapes[2] * outputShapes[3]], false);
        }

        matmulH.IterateAll(outTensorsGM[tensorCIndexWithOffset + i * outputShapes[2] * outputShapes[3]], false);
        matmulH.End();
    }
}

template <typename T>
__aicore__ inline void UpsampleBicubic2dGradDCND<T>::ParseTilingData(UpsampleBicubic2dGradTilingData *tilingData)
{
    slideSize = tilingData->slideSize;
    scaleW = tilingData->scalesW;
    scaleH = tilingData->scalesH;
    alignCorners = tilingData->alignCorners;

    ubMaxBytes = tilingData->ubSize;

    needCoreNumW = tilingData->CoreNumW;
    needCoreNumH = tilingData->CoreNumH;

    needExpandW = tilingData->needExpandW == 1;
    needExpandH = tilingData->needExpandH == 1;

    outputShapes[0] = tilingData->inputN;
    outputShapes[1] = tilingData->inputC;
    outputShapes[2] = tilingData->outputH;
    outputShapes[3] = tilingData->outputW;
    inputShapes[0] = tilingData->inputN;
    inputShapes[1] = tilingData->inputC;
    inputShapes[2] = tilingData->inputH;
    inputShapes[3] = tilingData->inputW;

    singleCoreMaxKW = tilingData->singleCoreKW;
    singleCoreMaxKH = tilingData->singleCoreKH;

    intermediateMatrixSize = tilingData->intermediateMatrixSize;
    radioMatrixSize = tilingData->radioMatrixSize;

    slideStartW = tilingData->perCoreSlideNumW * aiCoreIdx;
    slideEndW = slideStartW + tilingData->perCoreSlideNumW;

    tailSlideStartW = tilingData->perCoreSlideNumW * tilingData->CoreNum;
    tailSlideEndW = tilingData->outputW;

    if (aiCoreIdx >= tilingData->extraTailSlideCoreNumW) {
        tailRowStartW = tilingData->perCoreTailSlideNumW * aiCoreIdx + tilingData->extraTailSlideCoreNumW;
        tailRowEndW = tailRowStartW + tilingData->perCoreTailSlideNumW;
    } else {
        tailRowStartW = (tilingData->perCoreTailSlideNumW + 1) * aiCoreIdx;
        tailRowEndW = tailRowStartW +tilingData->perCoreTailSlideNumW + 1;
    }

    slideStartH = tilingData->perCoreSlideNumH * aiCoreIdx;
    slideEndH = slideStartH + tilingData->perCoreSlideNumH;

    tailSlideStartH = tilingData->perCoreSlideNumH * tilingData->CoreNum;
    tailSlideEndH = tilingData->outputH;

    if (aiCoreIdx >= tilingData->extraTailSlideCoreNumH) {
        tailRowStartH = tilingData->perCoreTailSlideNumH * aiCoreIdx + tilingData->extraTailSlideCoreNumH;
        tailRowEndH = tailRowStartH + tilingData->perCoreTailSlideNumH;
    } else {
        tailRowStartH = (tilingData->perCoreTailSlideNumH + 1) * aiCoreIdx;
        tailRowEndH = tailRowStartH +tilingData->perCoreTailSlideNumH + 1;
    }
    matmulTilingW = &tilingData->MMParamW;
    matmulTilingH = &tilingData->MMParamH;
}

}  // namespace UpsampleBicubic2dGrad
#endif