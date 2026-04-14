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
 * \file grid_sampler3_d_grad.h
 * \brief
 */
#ifndef GRID_SAMPLER3D_GRAD
#define GRID_SAMPLER3D_GRAD

#include <type_traits>
#include "kernel_operator.h"

namespace GridSampler3DGrad {

using namespace AscendC;

constexpr int32_t INT_MAX = 2147483647;
constexpr int32_t INT_MIN = -2147483648;
constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUFFER_APPLY_NUM = 3;
constexpr int32_t INPUT_NUM = 3;
constexpr int32_t OUTPUT_NUM = 2;
constexpr int32_t GRAD_INPUT_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 1;
constexpr int32_t GRID_INPUT_INDEX = 2;
constexpr int32_t DX_INPUT_INDEX = 3;
constexpr int32_t DGRID_INPUT_INDEX = 4;
constexpr int32_t GM_PARAMS_SIZE = 5;
constexpr int32_t DX_OUTPUT_INDEX = 0;
constexpr int32_t DGRID_OUTPUT_INDEX = 1;
constexpr int32_t BLOCK_BYTES = 32;
constexpr int32_t UINT8_BITS = 8;
constexpr int32_t ELE_NUM_PER_REPEAT = 64;
constexpr int32_t GATHER_MASK_NUM = 192;
constexpr int32_t REPEAT_STRIDE_0 = 24;
constexpr int32_t REPEAT_STRIDE_1 = 0;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int32_t ALIGN_256_BYTES = 256;
constexpr int32_t CHANNEL_1024 = 1024;
constexpr int32_t BORDER = 1;
constexpr int32_t REFLECTION = 2;

template <typename T>
class GridSampler3DGradNS {
public:
    __aicore__ inline GridSampler3DGradNS(){};
    __aicore__ inline void Init(
        const GridSampler3DGradTilingData* tilingData, GM_ADDR inputTensors[GM_PARAMS_SIZE + 1]);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampler3DGradTilingData* tilingData);
    __aicore__ inline void InitBilinearBuffer();
    __aicore__ inline void InitBilinearBufferExtend();
    __aicore__ inline void InitNearestBuffer();
    __aicore__ inline void InitLocalTensor();
    __aicore__ inline void InitBilinearLocalTensor();
    __aicore__ inline void InitBilinearLocalTensorExtend();
    __aicore__ inline void InitNearestLocalTensor();
    __aicore__ inline void ComputeNearestXGrad(
        LocalTensor<int32_t> srcIndex, LocalTensor<T> weight, const int32_t coorIndex, const int32_t cycle,
        const int64_t ncOffset, LocalTensor<T> gOutLocalTensor, LocalTensor<T> zIndex, LocalTensor<T> yIndex,
        LocalTensor<T> xIndex);
    __aicore__ inline void ComputeWeight(
        LocalTensor<T> dst, LocalTensor<T> xCoorTensor1, LocalTensor<T> xCoorTensor2, LocalTensor<T> yCoorTensor1,
        LocalTensor<T> yCoorTensor2, LocalTensor<T> zCoorTensor1, LocalTensor<T> zCoorTensor2, const int32_t calCount);
    __aicore__ inline void DupValue();
    __aicore__ inline void ComputeSourceIndexSetGrad(
        LocalTensor<T> dataTensor, LocalTensor<T> gradTensor, const T size, const int32_t calCount);
    __aicore__ inline void ReflectCoordinatesCommon(
        LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, int32_t newCalCount);
    __aicore__ inline void ClipCoordinatesSetGrad(
        LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, int32_t newCalCount);
    __aicore__ inline void ReflectCoordinatesSetGrad(
        LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, LocalTensor<T> tmpDataTensor, LocalTensor<T> tmpDupTensor,
        LocalTensor<int32_t> tmpIntTensor, LocalTensor<T> extraTensor, LocalTensor<T> flipTensor, int64_t twiceLow,
        int64_t twiceHigh, int32_t calCount);
    __aicore__ inline void WithinBounds3d(
        LocalTensor<T> dst, LocalTensor<T> izT, LocalTensor<T> iyT, LocalTensor<T> ixT, LocalTensor<T> weight,
        const int32_t calCount);
    __aicore__ inline void ComputeIndex(
        LocalTensor<int32_t> dstIndex, LocalTensor<int32_t> dstIndex2, LocalTensor<int32_t> zCoor,
        LocalTensor<int32_t> yCoor, LocalTensor<int32_t> xCoor, const int32_t calCount);
    __aicore__ inline void ComputeAfterTransposeGridGrad(
        LocalTensor<int32_t> srcIndex, LocalTensor<T> zCoor1, LocalTensor<T> zCoor2, LocalTensor<T> yCoor1,
        LocalTensor<T> yCoor2, LocalTensor<T> xCoor1, LocalTensor<T> xCoor2, LocalTensor<T> gOutLocalTensor,
        LocalTensor<T> zIndex, LocalTensor<T> yIndex, LocalTensor<T> xIndex, const int32_t coorIndex,
        const int32_t batchIdx, int32_t xTag, int32_t yTag, int32_t zTag);
    __aicore__ inline void ComputeAfterTransposeXGrad(
        LocalTensor<int32_t> srcIndex, LocalTensor<T> weight, const int32_t coorIndex, const int64_t ncOffset,
        LocalTensor<T> gOutLocalTensor, LocalTensor<T> zIndex, LocalTensor<T> yIndex, LocalTensor<T> xIndex);

    __aicore__ inline void InitComputeTensor();
    __aicore__ inline void ComputeBilinear(int32_t singleComputeCount, const int64_t curGridPointIndex);
    __aicore__ inline void InitComputeBilinearTensor(int32_t singleComputeCount);
    __aicore__ inline void ComputeBilinearCommon(int32_t singleComputeCount, const int64_t curGridPointIndex);
    __aicore__ inline void ComputeBilinearOutput(int32_t i, int64_t ncBaseOffset, LocalTensor<T> gOutLocalTensor);
    __aicore__ inline void ComputeNearest(int32_t singleComputeCount, const int64_t curGridPointIndex);
    __aicore__ inline void InitComputeNearestTensor(int32_t singleComputeCount);
    __aicore__ inline void ComputeGridPointIndex(int32_t gridPointIndex);

    __aicore__ inline void CopyIn(const int64_t offset, const int32_t calCount, const int32_t inputIndex);
    __aicore__ inline void CopyOut(const int64_t offset, const int32_t calCount);
    __aicore__ inline void Compute(const int32_t computeCount, const int64_t curGridPointIndex);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    };

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, NO_BUFFER_NUM> dataInQueue[INPUT_NUM];
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> dataOutQueue[OUTPUT_NUM];

    TBuf<TPosition::VECCALC> xCoordinateBuf;
    TBuf<TPosition::VECCALC> yCoordinateBuf;
    TBuf<TPosition::VECCALC> zCoordinateBuf;

    TBuf<TPosition::VECCALC> xGradInBuf;
    TBuf<TPosition::VECCALC> yGradInBuf;
    TBuf<TPosition::VECCALC> zGradInBuf;

    TBuf<TPosition::VECCALC> txNwBuf;

    TBuf<TPosition::VECCALC> txNwIntBuf;

    TBuf<TPosition::VECCALC> tnwBuf;

    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    TBuf<TPosition::VECCALC> tmp3Buf;
    TBuf<TPosition::VECCALC> tmp4Buf;

    TBuf<TPosition::VECCALC> tmp5Buf;
    TBuf<TPosition::VECCALC> tmp6Buf;
    TBuf<TPosition::VECCALC> tmp7Buf;
    TBuf<TPosition::VECCALC> tmp8Buf;
    TBuf<TPosition::VECCALC> tmp9Buf;

    TBuf<TPosition::VECCALC> bufferMaskXBuf;
    TBuf<TPosition::VECCALC> bufferMaskYBuf;
    TBuf<TPosition::VECCALC> bufferMaskZBuf;
    TBuf<TPosition::VECCALC> mask1Buf;
    TBuf<TPosition::VECCALC> mask2Buf;
    TBuf<TPosition::VECCALC> mask3Buf;
    TBuf<TPosition::VECCALC> mask4Buf;
    TBuf<TPosition::VECCALC> mask5Buf;
    TBuf<TPosition::VECCALC> mask6Buf;
    TBuf<TPosition::VECCALC> mask7Buf;

    TBuf<TPosition::VECCALC> computeIndexBuf;
    TBuf<TPosition::VECCALC> computeIndexBuf1;
    TBuf<TPosition::VECCALC> computeIndexBuf2;
    TBuf<TPosition::VECCALC> computeIndexBuf3;

    TBuf<TPosition::VECCALC> gixBuf;
    TBuf<TPosition::VECCALC> giyBuf;
    TBuf<TPosition::VECCALC> gizBuf;
    TBuf<TPosition::VECCALC> sumXBuf;
    TBuf<TPosition::VECCALC> sumYBuf;
    TBuf<TPosition::VECCALC> sumZBuf;
    TBuf<TPosition::VECCALC> clipLimitBuf;

    TBuf<TPosition::VECCALC> ixNearIntBuf;
    TBuf<TPosition::VECCALC> iyNearIntBuf;
    TBuf<TPosition::VECCALC> izNearIntBuf;
    TBuf<TPosition::VECCALC> ixFloatBuf;
    TBuf<TPosition::VECCALC> iyFloatBuf;
    TBuf<TPosition::VECCALC> izFloatBuf;
    TBuf<TPosition::VECCALC> dupOneBuf;
    TBuf<TPosition::VECCALC> selBuf1;
    TBuf<TPosition::VECCALC> selBuf2;
    TBuf<TPosition::VECCALC> selBuf3;
    TBuf<TPosition::VECCALC> selBuf4;
    TBuf<TPosition::VECCALC> selBuf5;
    TBuf<TPosition::VECCALC> selBuf6;
    TBuf<TPosition::VECCALC> selBuf7;
    TBuf<TPosition::VECCALC> selBuf8;

    GlobalTensor<T> inputGm[GM_PARAMS_SIZE];

    LocalTensor<uint8_t> mask1Tensor;
    LocalTensor<uint8_t> mask2Tensor;
    LocalTensor<uint8_t> mask3Tensor;
    LocalTensor<uint8_t> mask4Tensor;
    LocalTensor<uint8_t> mask5Tensor;
    LocalTensor<uint8_t> mask6Tensor;
    LocalTensor<uint8_t> mask7Tensor;
    LocalTensor<uint16_t> int8ToInt16Mask1;
    LocalTensor<uint16_t> int8ToInt16Mask2;
    LocalTensor<uint16_t> int8ToInt16Mask3;
    LocalTensor<uint16_t> int8ToInt16Mask4;
    LocalTensor<uint16_t> int8ToInt16Mask5;
    LocalTensor<uint16_t> int8ToInt16Mask6;
    LocalTensor<int32_t> tmpIndex1;
    LocalTensor<int32_t> tmpIndex2;
    LocalTensor<T> dupOneTensor;
    LocalTensor<T> selTensor1;
    LocalTensor<T> selTensor2;
    LocalTensor<T> selTensor3;
    LocalTensor<T> selTensor4;
    LocalTensor<T> selTensor5;
    LocalTensor<T> selTensor6;
    LocalTensor<T> selTensor7;
    LocalTensor<T> selTensor8;
    LocalTensor<T> tmp1Tensor;
    LocalTensor<T> tmp2Tensor;
    LocalTensor<T> tmp3Tensor;
    LocalTensor<T> tmp4Tensor;
    LocalTensor<T> gixLocalTensor;
    LocalTensor<T> giyLocalTensor;
    LocalTensor<T> gizLocalTensor;
    LocalTensor<T> sumX;
    LocalTensor<T> sumY;
    LocalTensor<T> sumZ;
    LocalTensor<T> clipLimit;

    LocalTensor<T> tmpDataTensor;
    LocalTensor<T> extraTensor;
    LocalTensor<int32_t> extraIntTensor;
    LocalTensor<T> flipsTensor;
    LocalTensor<int32_t> flipsIntTensor;
    LocalTensor<T> flipsTensor2;
    LocalTensor<T> dataTensor2;
    LocalTensor<T> dupTensor2;

    LocalTensor<T> xTensor;
    LocalTensor<T> yTensor;
    LocalTensor<T> zTensor;

    LocalTensor<T> xGradIn;
    LocalTensor<T> yGradIn;
    LocalTensor<T> zGradIn;

    LocalTensor<T> inputCoordinate;
    LocalTensor<T> dstLocal;

    LocalTensor<uint32_t> bufXPattern;

    LocalTensor<uint32_t> bufYPattern;

    LocalTensor<uint32_t> bufZPattern;

    // bilinear
    LocalTensor<T> txNw;
    LocalTensor<T> tyNw;
    LocalTensor<T> tzNw;
    LocalTensor<T> txNe;
    LocalTensor<T> tyNe;
    LocalTensor<T> tzNe;
    LocalTensor<T> txSw;
    LocalTensor<T> tySw;
    LocalTensor<T> tzSw;
    LocalTensor<T> txSe;
    LocalTensor<T> tySe;
    LocalTensor<T> tzSe;
    LocalTensor<T> bxNw;
    LocalTensor<T> byNw;
    LocalTensor<T> bzNw;
    LocalTensor<T> bxNe;
    LocalTensor<T> byNe;
    LocalTensor<T> bzNe;
    LocalTensor<T> bxSw;
    LocalTensor<T> bySw;
    LocalTensor<T> bzSw;
    LocalTensor<T> bxSe;
    LocalTensor<T> bySe;
    LocalTensor<T> bzSe;

    LocalTensor<int32_t> txNwInt;
    LocalTensor<int32_t> tyNwInt;
    LocalTensor<int32_t> tzNwInt;
    LocalTensor<int32_t> txNeInt;
    LocalTensor<int32_t> tyNeInt;
    LocalTensor<int32_t> tzNeInt;
    LocalTensor<int32_t> txSwInt;
    LocalTensor<int32_t> tySwInt;
    LocalTensor<int32_t> tzSwInt;
    LocalTensor<int32_t> txSeInt;
    LocalTensor<int32_t> tySeInt;
    LocalTensor<int32_t> tzSeInt;
    LocalTensor<int32_t> bxNwInt;
    LocalTensor<int32_t> byNwInt;
    LocalTensor<int32_t> bzNwInt;
    LocalTensor<int32_t> bxNeInt;
    LocalTensor<int32_t> byNeInt;
    LocalTensor<int32_t> bzNeInt;
    LocalTensor<int32_t> bxSwInt;
    LocalTensor<int32_t> bySwInt;
    LocalTensor<int32_t> bzSwInt;
    LocalTensor<int32_t> bxSeInt;
    LocalTensor<int32_t> bySeInt;
    LocalTensor<int32_t> bzSeInt;

    LocalTensor<T> tnw;
    LocalTensor<T> tne;
    LocalTensor<T> tsw;
    LocalTensor<T> tse;
    LocalTensor<T> bnw;
    LocalTensor<T> bne;
    LocalTensor<T> bsw;
    LocalTensor<T> bse;

    LocalTensor<int32_t> tNwIndex;
    LocalTensor<int32_t> tNeIndex;
    LocalTensor<int32_t> tSwIndex;
    LocalTensor<int32_t> tSeIndex;
    LocalTensor<int32_t> bNwIndex;
    LocalTensor<int32_t> bNeIndex;
    LocalTensor<int32_t> bSwIndex;
    LocalTensor<int32_t> bSeIndex;
    LocalTensor<int32_t> tnwIndex2;
    LocalTensor<int32_t> tneIndex2;
    LocalTensor<int32_t> tswIndex2;
    LocalTensor<int32_t> tseIndex2;
    LocalTensor<int32_t> bnwIndex2;
    LocalTensor<int32_t> bneIndex2;
    LocalTensor<int32_t> bswIndex2;
    LocalTensor<int32_t> bseIndex2;

    // nearest
    LocalTensor<int32_t> ixNearInt;
    LocalTensor<int32_t> iyNearInt;
    LocalTensor<int32_t> izNearInt;

    LocalTensor<T> ixFloat;
    LocalTensor<T> iyFloat;
    LocalTensor<T> izFloat;

    LocalTensor<int32_t> xIndex;

    const int64_t BLOCK_SIZE = 32;

    uint32_t batch = 0;
    int32_t channel = 0;
    int32_t xD = 0;
    int32_t xH = 0;
    int32_t xW = 0;
    uint32_t gridD = 0;
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    uint32_t alignCorners = 0;
    uint32_t blockNum = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tailPNum = 0;
    uint32_t perBlockCount = 0;
    uint32_t outD = 0;
    uint32_t outH = 0;
    uint32_t outW = 0;
    uint32_t ubFactorElement = 0;
    uint32_t maskSize = 0;
    uint32_t maskNum = 0;
    uint32_t blockIdx = 0;
    uint32_t alignChannel = 0;
    uint32_t group = 0;
    uint32_t isDeterministic = 0;
    uint32_t tailBNum = 0;

    int32_t inputStrideN = 0;
    int32_t inputStrideD = 0;
    int32_t inputStrideH = 0;
    int32_t inputStrideW = 0;
    int32_t gradStrideN = 0;
    int32_t gradStrideC = 0;
    int32_t gradStrideD = 0;
    int32_t gradStrideH = 0;
    int32_t gradStrideW = 0;
    int32_t xStrideC = 0;
    int32_t dxStrideN = 0;
    int32_t dxStrideC = 0;
    int32_t dxStrideD = 0;
    int32_t dxStrideH = 0;
    int32_t dxStrideW = 0;
    int64_t pointIndex = 0;
    int64_t gradGmOffset = 0;
    int64_t xGmOffset = 0;
    int32_t ncOffset = 0;

    float fwidth = 0;
    float fheight = 0;
    float fdepth = 0;

    int64_t n = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;

    int64_t gridPointIndex = 0;
    int64_t ncBaseOffset = 0;

    T gix = static_cast<T>(0);
    T giy = static_cast<T>(0);
    T giz = static_cast<T>(0);
};

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::Init(
    const GridSampler3DGradTilingData* tilingData, GM_ADDR inputTensors[GM_PARAMS_SIZE + 1])
{
    blockIdx = GetBlockIdx();
    // parse tilingData
    ParseTilingData(tilingData);
    // init params
    outD = gridD;
    outH = gridH;
    outW = gridW;
    inputStrideN = xW * xH * xD * channel;
    inputStrideD = xW * xH;
    inputStrideH = xW;
    inputStrideW = 1;
    gradStrideN = outD * outH * outW * channel;
    gradStrideC = outD * outH * outW;
    gradStrideD = outH * outW;
    gradStrideH = outW;
    gradStrideW = 1;
    xStrideC = xD * xH * xW;
    dxStrideN = xD * xH * xW * channel;
    dxStrideC = xD * xH * xW;
    dxStrideD = xH * xW;
    dxStrideH = xW;
    dxStrideW = 1;

    fdepth = static_cast<T>(xD);
    fheight = static_cast<T>(xH);
    fwidth = static_cast<T>(xW);
    maskSize = CeilAlign(CeilDiv(ubFactorElement, UINT8_BITS), BLOCK_BYTES);
    maskNum = maskSize / sizeof(uint8_t);
    perBlockCount = BLOCK_BYTES / sizeof(T);
    alignChannel = CeilAlign(channel, perBlockCount);

    // init inputTensor
    inputGm[GRAD_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[GRAD_INPUT_INDEX]));
    inputGm[X_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[X_INPUT_INDEX]));
    inputGm[GRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[GRID_INPUT_INDEX]));
    inputGm[DX_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[DX_INPUT_INDEX]));
    inputGm[DGRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[DGRID_INPUT_INDEX]));

    // initBuffer
    if (interpolation == 0) {
        InitBilinearBuffer();
    } else {
        InitNearestBuffer();
    }

    // init localTensor
    InitLocalTensor();
    if (interpolation == 0) {
        InitBilinearLocalTensor();
    } else {
        InitNearestLocalTensor();
    }
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ParseTilingData(const GridSampler3DGradTilingData* tilingData)
{
    batch = tilingData->batch;
    channel = tilingData->channel;
    xD = tilingData->xD;
    xH = tilingData->xH;
    xW = tilingData->xW;
    gridD = tilingData->gridD;
    gridH = tilingData->gridH;
    gridW = tilingData->gridW;
    interpolation = tilingData->interpolation;
    padding = tilingData->padding;
    alignCorners = tilingData->alignCorners;
    blockNum = tilingData->blockNum;
    pNumPerCore = tilingData->pNumPerCore;
    tailPNum = tilingData->tailPNum;
    ubFactorElement = tilingData->ubFactorElement;
    group = tilingData->group;
    isDeterministic = tilingData->isDeterministic;
    tailBNum = tilingData->tailBNum;
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitBilinearBuffer()
{
    pipe.InitBuffer(dataInQueue[GRAD_INPUT_INDEX], NO_BUFFER_NUM, alignChannel * sizeof(T));
    pipe.InitBuffer(dataInQueue[X_INPUT_INDEX], NO_BUFFER_NUM, alignChannel * sizeof(T));
    pipe.InitBuffer(dataInQueue[GRID_INPUT_INDEX], NO_BUFFER_NUM, BUFFER_APPLY_NUM * ubFactorElement * sizeof(T));
    pipe.InitBuffer(dataOutQueue[DX_OUTPUT_INDEX], NO_BUFFER_NUM, alignChannel * sizeof(T));
    pipe.InitBuffer(dataOutQueue[DGRID_OUTPUT_INDEX], NO_BUFFER_NUM, BUFFER_APPLY_NUM * ubFactorElement * sizeof(T));

    pipe.InitBuffer(xCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
    pipe.InitBuffer(yCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
    pipe.InitBuffer(zCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));

    pipe.InitBuffer(xGradInBuf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(yGradInBuf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(zGradInBuf, ubFactorElement * sizeof(T));

    pipe.InitBuffer(bufferMaskXBuf, BLOCK_SIZE * 6);
    pipe.InitBuffer(bufferMaskYBuf, BLOCK_SIZE * 6);
    pipe.InitBuffer(bufferMaskZBuf, BLOCK_SIZE * 6);

    pipe.InitBuffer(mask1Buf, maskSize);
    pipe.InitBuffer(mask2Buf, maskSize);
    pipe.InitBuffer(mask3Buf, maskSize);
    pipe.InitBuffer(mask4Buf, maskSize);
    pipe.InitBuffer(mask5Buf, maskSize);
    pipe.InitBuffer(mask6Buf, maskSize);
    pipe.InitBuffer(mask7Buf, maskSize);

    // LocBuf
    pipe.InitBuffer(txNwBuf, ubFactorElement * sizeof(T) * 24);

    // IntBuf
    pipe.InitBuffer(txNwIntBuf, ubFactorElement * sizeof(T) * 24);

    // WeightBuf
    pipe.InitBuffer(tnwBuf, ubFactorElement * sizeof(T) * 8);

    // tmpBuf
    pipe.InitBuffer(tmp1Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp2Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp3Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp4Buf, ubFactorElement * sizeof(T));

    pipe.InitBuffer(tmp5Buf, ubFactorElement * sizeof(int32_t));
    pipe.InitBuffer(tmp6Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp7Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp8Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp9Buf, ubFactorElement * sizeof(T));

    InitBilinearBufferExtend();
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitBilinearBufferExtend()
{
    pipe.InitBuffer(dupOneBuf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(selBuf1, ubFactorElement * sizeof(T));
    pipe.InitBuffer(selBuf2, ubFactorElement * sizeof(T));
    pipe.InitBuffer(selBuf3, ubFactorElement * sizeof(T));
    pipe.InitBuffer(selBuf4, ubFactorElement * sizeof(T));
    pipe.InitBuffer(selBuf5, ubFactorElement * sizeof(T));
    pipe.InitBuffer(selBuf6, ubFactorElement * sizeof(T));
    pipe.InitBuffer(selBuf7, ubFactorElement * sizeof(T));
    pipe.InitBuffer(selBuf8, ubFactorElement * sizeof(T));

    pipe.InitBuffer(computeIndexBuf1, ubFactorElement * sizeof(int32_t));
    pipe.InitBuffer(computeIndexBuf2, ubFactorElement * sizeof(int32_t));
    pipe.InitBuffer(computeIndexBuf3, ubFactorElement * sizeof(int32_t) * 16);

    pipe.InitBuffer(gixBuf, alignChannel * sizeof(T));
    pipe.InitBuffer(giyBuf, alignChannel * sizeof(T));
    pipe.InitBuffer(gizBuf, alignChannel * sizeof(T));
    pipe.InitBuffer(sumXBuf, alignChannel * sizeof(T));
    pipe.InitBuffer(sumYBuf, alignChannel * sizeof(T));
    pipe.InitBuffer(sumZBuf, alignChannel * sizeof(T));
    pipe.InitBuffer(clipLimitBuf, ubFactorElement * sizeof(T));
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitNearestBuffer()
{
    pipe.InitBuffer(dataInQueue[GRAD_INPUT_INDEX], NO_BUFFER_NUM, alignChannel * sizeof(T) * group);
    pipe.InitBuffer(dataInQueue[GRID_INPUT_INDEX], NO_BUFFER_NUM, BUFFER_APPLY_NUM * ubFactorElement * sizeof(T));

    pipe.InitBuffer(dataOutQueue[DX_OUTPUT_INDEX], NO_BUFFER_NUM, alignChannel * sizeof(T));
    pipe.InitBuffer(dataOutQueue[DGRID_OUTPUT_INDEX], NO_BUFFER_NUM, BUFFER_APPLY_NUM * ubFactorElement * sizeof(T));

    pipe.InitBuffer(bufferMaskXBuf, BLOCK_SIZE * 6);
    pipe.InitBuffer(bufferMaskYBuf, BLOCK_SIZE * 6);
    pipe.InitBuffer(bufferMaskZBuf, BLOCK_SIZE * 6);

    pipe.InitBuffer(xCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
    pipe.InitBuffer(yCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
    pipe.InitBuffer(zCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));

    pipe.InitBuffer(tmp5Buf, ubFactorElement * sizeof(int32_t));
    pipe.InitBuffer(tmp6Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp7Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp8Buf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(tmp9Buf, ubFactorElement * sizeof(T));

    pipe.InitBuffer(xGradInBuf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(yGradInBuf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(zGradInBuf, ubFactorElement * sizeof(T));

    pipe.InitBuffer(ixFloatBuf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(iyFloatBuf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(izFloatBuf, ubFactorElement * sizeof(T));

    pipe.InitBuffer(computeIndexBuf, ubFactorElement * sizeof(int32_t));
    pipe.InitBuffer(computeIndexBuf1, ubFactorElement * sizeof(int32_t));
    pipe.InitBuffer(computeIndexBuf2, ubFactorElement * sizeof(int32_t));

    pipe.InitBuffer(ixNearIntBuf, ubFactorElement * sizeof(int32_t));
    pipe.InitBuffer(iyNearIntBuf, ubFactorElement * sizeof(int32_t));
    pipe.InitBuffer(izNearIntBuf, ubFactorElement * sizeof(int32_t));

    pipe.InitBuffer(selBuf1, ubFactorElement * sizeof(T));
    pipe.InitBuffer(mask1Buf, maskSize);
    pipe.InitBuffer(mask2Buf, maskSize);
    pipe.InitBuffer(mask3Buf, maskSize);
    pipe.InitBuffer(mask4Buf, maskSize);
    pipe.InitBuffer(mask5Buf, maskSize);
    pipe.InitBuffer(mask6Buf, maskSize);
    pipe.InitBuffer(mask7Buf, maskSize);

    pipe.InitBuffer(dupOneBuf, ubFactorElement * sizeof(T));
    pipe.InitBuffer(clipLimitBuf, ubFactorElement * sizeof(T));
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitLocalTensor()
{
    mask1Tensor = mask1Buf.Get<uint8_t>(maskNum);
    mask2Tensor = mask2Buf.Get<uint8_t>(maskNum);
    mask3Tensor = mask3Buf.Get<uint8_t>(maskNum);
    mask4Tensor = mask4Buf.Get<uint8_t>(maskNum);
    mask5Tensor = mask5Buf.Get<uint8_t>(maskNum);
    mask6Tensor = mask6Buf.Get<uint8_t>(maskNum);
    mask7Tensor = mask7Buf.Get<uint8_t>(maskNum);
    dupOneTensor = dupOneBuf.Get<T>(ubFactorElement);
    tmpIndex1 = computeIndexBuf1.Get<int32_t>(ubFactorElement);
    tmpIndex2 = computeIndexBuf2.Get<int32_t>(ubFactorElement);
    clipLimit = clipLimitBuf.Get<T>(ubFactorElement);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitBilinearLocalTensor()
{
    selTensor1 = selBuf1.Get<T>(ubFactorElement);
    selTensor2 = selBuf2.Get<T>(ubFactorElement);
    selTensor3 = selBuf3.Get<T>(ubFactorElement);
    selTensor4 = selBuf4.Get<T>(ubFactorElement);
    selTensor5 = selBuf5.Get<T>(ubFactorElement);
    selTensor6 = selBuf6.Get<T>(ubFactorElement);
    selTensor7 = selBuf7.Get<T>(ubFactorElement);
    selTensor8 = selBuf8.Get<T>(ubFactorElement);
    tmp1Tensor = tmp1Buf.Get<T>(ubFactorElement);
    tmp2Tensor = tmp2Buf.Get<T>(ubFactorElement);
    tmp3Tensor = tmp3Buf.Get<T>(ubFactorElement);
    tmp4Tensor = tmp4Buf.Get<T>(ubFactorElement);
    sumX = sumXBuf.Get<T>(alignChannel);
    sumY = sumYBuf.Get<T>(alignChannel);
    sumZ = sumZBuf.Get<T>(alignChannel);

    tnw = tnwBuf.Get<T>();
    tne = tnw[ubFactorElement * 1];
    tsw = tnw[ubFactorElement * 2];
    tse = tnw[ubFactorElement * 3];
    bnw = tnw[ubFactorElement * 4];
    bne = tnw[ubFactorElement * 5];
    bsw = tnw[ubFactorElement * 6];
    bse = tnw[ubFactorElement * 7];

    tNwIndex = computeIndexBuf3.Get<int32_t>();
    tNeIndex = tNwIndex[ubFactorElement * 1];
    tSwIndex = tNwIndex[ubFactorElement * 2];
    tSeIndex = tNwIndex[ubFactorElement * 3];
    bNwIndex = tNwIndex[ubFactorElement * 4];
    bNeIndex = tNwIndex[ubFactorElement * 5];
    bSwIndex = tNwIndex[ubFactorElement * 6];
    bSeIndex = tNwIndex[ubFactorElement * 7];
    tnwIndex2 = tNwIndex[ubFactorElement * 8];
    tneIndex2 = tNwIndex[ubFactorElement * 9];
    tswIndex2 = tNwIndex[ubFactorElement * 10];
    tseIndex2 = tNwIndex[ubFactorElement * 11];
    bnwIndex2 = tNwIndex[ubFactorElement * 12];
    bneIndex2 = tNwIndex[ubFactorElement * 13];
    bswIndex2 = tNwIndex[ubFactorElement * 14];
    bseIndex2 = tNwIndex[ubFactorElement * 15];

    gixLocalTensor = gixBuf.Get<T>(alignChannel);
    giyLocalTensor = giyBuf.Get<T>(alignChannel);
    gizLocalTensor = gizBuf.Get<T>(alignChannel);

    InitBilinearLocalTensorExtend();
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitBilinearLocalTensorExtend()
{
    txNw = txNwBuf.Get<T>();
    tyNw = txNw[ubFactorElement * 1];
    tzNw = txNw[ubFactorElement * 2];
    txNe = txNw[ubFactorElement * 3];
    tyNe = txNw[ubFactorElement * 4];
    tzNe = txNw[ubFactorElement * 5];
    txSw = txNw[ubFactorElement * 6];
    tySw = txNw[ubFactorElement * 7];
    tzSw = txNw[ubFactorElement * 8];
    txSe = txNw[ubFactorElement * 9];
    tySe = txNw[ubFactorElement * 10];
    tzSe = txNw[ubFactorElement * 11];
    bxNw = txNw[ubFactorElement * 12];
    byNw = txNw[ubFactorElement * 13];
    bzNw = txNw[ubFactorElement * 14];
    bxNe = txNw[ubFactorElement * 15];
    byNe = txNw[ubFactorElement * 16];
    bzNe = txNw[ubFactorElement * 17];
    bxSw = txNw[ubFactorElement * 18];
    bySw = txNw[ubFactorElement * 19];
    bzSw = txNw[ubFactorElement * 20];
    bxSe = txNw[ubFactorElement * 21];
    bySe = txNw[ubFactorElement * 22];
    bzSe = txNw[ubFactorElement * 23];

    txNwInt = txNwIntBuf.Get<int32_t>();
    tyNwInt = txNwInt[ubFactorElement * 1];
    tzNwInt = txNwInt[ubFactorElement * 2];
    txNeInt = txNwInt[ubFactorElement * 3];
    tyNeInt = txNwInt[ubFactorElement * 4];
    tzNeInt = txNwInt[ubFactorElement * 5];
    txSwInt = txNwInt[ubFactorElement * 6];
    tySwInt = txNwInt[ubFactorElement * 7];
    tzSwInt = txNwInt[ubFactorElement * 8];
    txSeInt = txNwInt[ubFactorElement * 9];
    tySeInt = txNwInt[ubFactorElement * 10];
    tzSeInt = txNwInt[ubFactorElement * 11];
    bxNwInt = txNwInt[ubFactorElement * 12];
    byNwInt = txNwInt[ubFactorElement * 13];
    bzNwInt = txNwInt[ubFactorElement * 14];
    bxNeInt = txNwInt[ubFactorElement * 15];
    byNeInt = txNwInt[ubFactorElement * 16];
    bzNeInt = txNwInt[ubFactorElement * 17];
    bxSwInt = txNwInt[ubFactorElement * 18];
    bySwInt = txNwInt[ubFactorElement * 19];
    bzSwInt = txNwInt[ubFactorElement * 20];
    bxSeInt = txNwInt[ubFactorElement * 21];
    bySeInt = txNwInt[ubFactorElement * 22];
    bzSeInt = txNwInt[ubFactorElement * 23];
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitNearestLocalTensor()
{
    ixNearInt = ixNearIntBuf.Get<int32_t>(ubFactorElement);
    iyNearInt = iyNearIntBuf.Get<int32_t>(ubFactorElement);
    izNearInt = izNearIntBuf.Get<int32_t>(ubFactorElement);
    ixFloat = ixFloatBuf.Get<T>(ubFactorElement);
    iyFloat = iyFloatBuf.Get<T>(ubFactorElement);
    izFloat = izFloatBuf.Get<T>(ubFactorElement);
    xIndex = computeIndexBuf.Get<int32_t>(ubFactorElement);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::DupValue()
{
    Duplicate<T>(dupOneTensor, 1, ubFactorElement);
    PipeBarrier<PIPE_V>();
    if (interpolation == 0) {
        Duplicate<T>(sumX, 0, alignChannel);
        Duplicate<T>(sumY, 0, alignChannel);
        Duplicate<T>(sumZ, 0, alignChannel);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeWeight(
    LocalTensor<T> dst, LocalTensor<T> xCoorTensor1, LocalTensor<T> xCoorTensor2, LocalTensor<T> yCoorTensor1,
    LocalTensor<T> yCoorTensor2, LocalTensor<T> zCoorTensor1, LocalTensor<T> zCoorTensor2, const int32_t calCount)
{
    Muls(tmp1Tensor, xCoorTensor1, static_cast<T>(-1), calCount);
    Add(tmp1Tensor, xCoorTensor2, tmp1Tensor, calCount);

    Muls(tmp2Tensor, yCoorTensor1, static_cast<T>(-1), calCount);
    Add(tmp2Tensor, yCoorTensor2, tmp2Tensor, calCount);

    Muls(tmp3Tensor, zCoorTensor1, static_cast<T>(-1), calCount);
    Add(tmp3Tensor, zCoorTensor2, tmp3Tensor, calCount);

    Mul(tmp4Tensor, tmp1Tensor, tmp2Tensor, calCount);
    Mul(dst, tmp4Tensor, tmp3Tensor, calCount);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeSourceIndexSetGrad(
    LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, const int32_t calCount)
{
    // 标准化坐标计算
    if (alignCorners == 1) {
        T gradValue = static_cast<T>(size - 1) / 2;
        Duplicate<T>(dupTensor, gradValue, calCount);
        for (int32_t i = 0; i < calCount; i++) {
            T tmpData = dataTensor.GetValue(i);
            T tmp = ((tmpData + 1) / 2) * (size - 1);
            dataTensor.SetValue(i, tmp);
        }
    } else {
        T gradValue = static_cast<T>(size) / 2;
        Duplicate<T>(dupTensor, gradValue, calCount);
        for (int32_t i = 0; i < calCount; i++) {
            T tmpData = dataTensor.GetValue(i);
            T tmp = ((tmpData + 1) * size - 1) / 2;
            dataTensor.SetValue(i, tmp);
        }
    }
    int32_t newCalCount =
        ((calCount * FLOAT_BYTES - 1 + ALIGN_256_BYTES) / ALIGN_256_BYTES * ALIGN_256_BYTES) / FLOAT_BYTES;
    // 坐标边界值处理
    if (padding == BORDER) {
        Compare(mask7Tensor, dataTensor, dataTensor, CMPMODE::EQ, newCalCount);
        PipeBarrier<PIPE_V>();
        ClipCoordinatesSetGrad(dataTensor, dupTensor, size, newCalCount);
    } else if (padding == REFLECTION) {
        LocalTensor<T> tmpDataTensor = tmp6Buf.Get<T>();
        Muls(tmpDataTensor, dataTensor, (float)(0.0), newCalCount);
        PipeBarrier<PIPE_V>();
        Compare(mask7Tensor, tmpDataTensor, tmpDataTensor, CMPMODE::EQ, newCalCount);
        PipeBarrier<PIPE_V>();
        ReflectCoordinatesCommon(dataTensor, dupTensor, size, newCalCount);
    }

    // If the data is inf/-inf/nan, convert the data to -100.
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(INT_MAX - 1), CMPMODE::LE, newCalCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(INT_MIN), CMPMODE::GE, newCalCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    PipeBarrier<PIPE_V>();
    Compare(mask1Tensor, dataTensor, dataTensor, CMPMODE::EQ, newCalCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    PipeBarrier<PIPE_V>();
    if (padding == BORDER || padding == REFLECTION) {
        Select(
            dataTensor, mask7Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ReflectCoordinatesCommon(
    LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, int32_t newCalCount)
{
    // init Reflect localTensor
    LocalTensor<int32_t> tmpIntTensor = tmp5Buf.Get<int32_t>();
    LocalTensor<T> tmpDataTensor = tmp6Buf.Get<T>();
    LocalTensor<T> extraTensor = tmp7Buf.Get<T>();
    LocalTensor<T> flipTensor = tmp8Buf.Get<T>();
    LocalTensor<T> tmpDupTensor = tmp9Buf.Get<T>();

    if (alignCorners == 1) {
        int64_t twiceLow = 0;
        int64_t twiceHigh = 2 * (size - 1);
        ReflectCoordinatesSetGrad(
            dataTensor, dupTensor, tmpDataTensor, tmpDupTensor, tmpIntTensor, extraTensor, flipTensor, twiceLow,
            twiceHigh, newCalCount);
    } else {
        int64_t twiceLow = -1;
        int64_t twiceHigh = 2 * size - 1;
        ReflectCoordinatesSetGrad(
            dataTensor, dupTensor, tmpDataTensor, tmpDupTensor, tmpIntTensor, extraTensor, flipTensor, twiceLow,
            twiceHigh, newCalCount);
    }
    ClipCoordinatesSetGrad(dataTensor, dupTensor, size, newCalCount);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ClipCoordinatesSetGrad(LocalTensor<T> dataTensor,
    LocalTensor<T> dupTensor, const T size, int32_t newCalCount)
{
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(0), CMPMODE::GT, newCalCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    Select(dupTensor, mask1Tensor, dupTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);

    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(size - 1), CMPMODE::LT, newCalCount);
    Select(
        dataTensor, mask1Tensor, dataTensor, static_cast<T>(size - 1), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    Select(dupTensor, mask1Tensor, dupTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ReflectCoordinatesSetGrad(
    LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, LocalTensor<T> tmpDataTensor, LocalTensor<T> tmpDupTensor,
    LocalTensor<int32_t> tmpIntTensor, LocalTensor<T> extraTensor, LocalTensor<T> flipTensor, int64_t twiceLow,
    int64_t twiceHigh, int32_t newCalCount)
{
    if (twiceLow == twiceHigh) {
        Duplicate(dataTensor, (float)0.0, newCalCount);
        Duplicate(dupTensor, (float)0.0, newCalCount);
        return;
    }

    T min = static_cast<T>(twiceLow) / 2;
    T span = static_cast<T>(twiceHigh - twiceLow) / 2;

    Duplicate(tmpDupTensor, (float)(1.0), newCalCount);
    Adds(tmpDataTensor, dataTensor, static_cast<T>(-1) * min, newCalCount);
    Abs(tmpDataTensor, tmpDataTensor, newCalCount);
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(min), CMPMODE::GE, newCalCount);
    Select(tmpDupTensor, mask1Tensor, tmpDupTensor, static_cast<T>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);

    // extra = x - Cast(x / span) * span
    Muls(extraTensor, tmpDataTensor, static_cast<T>(static_cast<T>(1.0) / span), newCalCount);
    Cast(tmpIntTensor, extraTensor, RoundMode::CAST_FLOOR, newCalCount);
    Cast(extraTensor, tmpIntTensor, RoundMode::CAST_NONE, newCalCount);
    Muls(extraTensor, extraTensor, span, newCalCount);
    Sub(extraTensor, tmpDataTensor, extraTensor, newCalCount);

    // flips = floor(x / span)
    Muls(tmpDataTensor, tmpDataTensor, (static_cast<T>(1) / span), newCalCount);
    Cast(tmpIntTensor, tmpDataTensor, RoundMode::CAST_FLOOR, newCalCount);
    Cast(tmpDataTensor, tmpIntTensor, RoundMode::CAST_NONE, newCalCount);

    // flips % 2
    Muls(flipTensor, tmpDataTensor, static_cast<T>(0.5), newCalCount);
    Cast(tmpIntTensor, flipTensor, RoundMode::CAST_FLOOR, newCalCount);
    Cast(flipTensor, tmpIntTensor, RoundMode::CAST_NONE, newCalCount);
    Muls(flipTensor, flipTensor, static_cast<T>(2), newCalCount);
    Sub(flipTensor, tmpDataTensor, flipTensor, newCalCount);
    CompareScalar(mask2Tensor, flipTensor, static_cast<float>(0.0), CMPMODE::EQ, newCalCount);

    // x = extra + min
    Adds(tmpDataTensor, extraTensor, min, newCalCount);

    // x = (span - extra) + min
    Muls(extraTensor, extraTensor, static_cast<T>(-1), newCalCount);
    Adds(extraTensor, extraTensor, span, newCalCount);
    Adds(extraTensor, extraTensor, min, newCalCount);

    // flips % 2 == 0 ?
    Select(dataTensor, mask2Tensor, tmpDataTensor, extraTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, newCalCount);
    Muls(flipTensor, tmpDupTensor, static_cast<T>(-1), newCalCount);
    Select(tmpDupTensor, mask2Tensor, tmpDupTensor, flipTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, newCalCount);
    Mul(dupTensor, dupTensor, tmpDupTensor, newCalCount);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::WithinBounds3d(
    LocalTensor<T> dst, LocalTensor<T> izT, LocalTensor<T> iyT, LocalTensor<T> ixT, LocalTensor<T> weightDupTensor,
    const int32_t calCount)
{
    int32_t newCalCount = ((calCount * FLOAT_BYTES - 1 + ALIGN_256_BYTES) / ALIGN_256_BYTES * ALIGN_256_BYTES) /
                          FLOAT_BYTES; // align 256 bytes
    CompareScalar(mask1Tensor, izT, static_cast<T>(0), CMPMODE::GE, newCalCount);
    CompareScalar(mask2Tensor, izT, static_cast<T>(fdepth), CMPMODE::LT, newCalCount);
    int8ToInt16Mask1 = mask1Tensor.ReinterpretCast<uint16_t>();
    int8ToInt16Mask2 = mask2Tensor.ReinterpretCast<uint16_t>();
    And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);

    CompareScalar(mask3Tensor, iyT, static_cast<T>(0), CMPMODE::GE, newCalCount);
    CompareScalar(mask4Tensor, iyT, static_cast<T>(fheight), CMPMODE::LT, newCalCount);
    int8ToInt16Mask3 = mask3Tensor.ReinterpretCast<uint16_t>();
    int8ToInt16Mask4 = mask4Tensor.ReinterpretCast<uint16_t>();
    And(int8ToInt16Mask4, int8ToInt16Mask3, int8ToInt16Mask4, maskNum / 2);

    CompareScalar(mask5Tensor, ixT, static_cast<T>(0), CMPMODE::GE, newCalCount);
    CompareScalar(mask6Tensor, ixT, static_cast<T>(fwidth), CMPMODE::LT, newCalCount);
    int8ToInt16Mask5 = mask5Tensor.ReinterpretCast<uint16_t>();
    int8ToInt16Mask6 = mask6Tensor.ReinterpretCast<uint16_t>();
    And(int8ToInt16Mask6, int8ToInt16Mask5, int8ToInt16Mask6, maskNum / 2);

    And(int8ToInt16Mask4, int8ToInt16Mask2, int8ToInt16Mask4, maskNum / 2);
    And(int8ToInt16Mask6, int8ToInt16Mask4, int8ToInt16Mask6, maskNum / 2);

    if (interpolation == 0) {
        Select(dst, int8ToInt16Mask6, dupOneTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeIndex(
    LocalTensor<int32_t> dstIndex, LocalTensor<int32_t> dstIndex2, LocalTensor<int32_t> zCoor,
    LocalTensor<int32_t> yCoor, LocalTensor<int32_t> xCoor, const int32_t calCount)
{
    Mins(zCoor, zCoor, xD - 1, calCount);
    Maxs(zCoor, zCoor, 0, calCount);

    Mins(yCoor, yCoor, xH - 1, calCount);
    Maxs(yCoor, yCoor, 0, calCount);

    Mins(xCoor, xCoor, xW - 1, calCount);
    Maxs(xCoor, xCoor, 0, calCount);

    Muls(tmpIndex1, zCoor, inputStrideD, calCount);
    Muls(tmpIndex2, yCoor, inputStrideH, calCount);
    Add(dstIndex, tmpIndex1, tmpIndex2, calCount);
    Add(dstIndex, dstIndex, xCoor, calCount);
    Muls(dstIndex, dstIndex, channel, calCount);

    if (interpolation == 0) {
        Muls(tmpIndex1, zCoor, inputStrideD, calCount);
        Muls(tmpIndex2, yCoor, inputStrideH, calCount);
        Add(dstIndex2, tmpIndex1, tmpIndex2, calCount);
        Add(dstIndex2, dstIndex2, xCoor, calCount);
        Muls(dstIndex2, dstIndex2, channel, calCount);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeAfterTransposeGridGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> zCoor1, LocalTensor<T> zCoor2, LocalTensor<T> yCoor1,
    LocalTensor<T> yCoor2, LocalTensor<T> xCoor1, LocalTensor<T> xCoor2, LocalTensor<T> gOutLocalTensor,
    LocalTensor<T> zIndex, LocalTensor<T> yIndex, LocalTensor<T> xIndex, const int32_t coorIndex,
    const int32_t batchIdx, int32_t xTag, int32_t yTag, int32_t zTag)
{
    int32_t z = zIndex.GetValue(coorIndex);
    int32_t y = yIndex.GetValue(coorIndex);
    int32_t x = xIndex.GetValue(coorIndex);

    bool inRangeZ = z >= 0 && z < xD;
    bool inRangeY = y >= 0 && y < xH;
    bool inRangeX = x >= 0 && x < xW;

    if (!(inRangeZ && inRangeY && inRangeX)) {
        return;
    }

    pointIndex = srcIndex.GetValue(coorIndex);
    xGmOffset = batchIdx * inputStrideN + pointIndex;
    T xVal = xCoor1.GetValue(coorIndex) - xCoor2.GetValue(coorIndex);
    T yVal = yCoor1.GetValue(coorIndex) - yCoor2.GetValue(coorIndex);
    T zVal = zCoor1.GetValue(coorIndex) - zCoor2.GetValue(coorIndex);

    LocalTensor<T> inputXLocalTensor = dataInQueue[X_INPUT_INDEX].AllocTensor<T>();
    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    DataCopyPadExtParams padParams = {true, 0, 0, static_cast<T>(0)};
    copyParams.blockLen = channel * sizeof(T);
    padParams.rightPadding = alignChannel - channel;
    // X input GM -> Local
    DataCopyPad(inputXLocalTensor, inputGm[X_INPUT_INDEX][xGmOffset], copyParams, padParams);

    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    Muls(gixLocalTensor, inputXLocalTensor, yVal, channel);
    Muls(gixLocalTensor, gixLocalTensor, zVal, channel);
    Mul(gixLocalTensor, gOutLocalTensor, gixLocalTensor, channel);
    Muls(gixLocalTensor, gixLocalTensor, static_cast<T>(xTag), channel);
    PipeBarrier<PIPE_V>();

    Muls(giyLocalTensor, inputXLocalTensor, xVal, channel);
    Muls(giyLocalTensor, giyLocalTensor, zVal, channel);
    Mul(giyLocalTensor, gOutLocalTensor, giyLocalTensor, channel);
    Muls(giyLocalTensor, giyLocalTensor, static_cast<T>(yTag), channel);
    PipeBarrier<PIPE_V>();

    Muls(gizLocalTensor, inputXLocalTensor, xVal, channel);
    Muls(gizLocalTensor, gizLocalTensor, yVal, channel);
    Mul(gizLocalTensor, gOutLocalTensor, gizLocalTensor, channel);
    Muls(gizLocalTensor, gizLocalTensor, static_cast<T>(zTag), channel);
    PipeBarrier<PIPE_V>();

    Add(sumZ, gizLocalTensor, sumZ, channel);
    Add(sumY, giyLocalTensor, sumY, channel);
    Add(sumX, gixLocalTensor, sumX, channel);

    dataInQueue[X_INPUT_INDEX].FreeTensor(inputXLocalTensor);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeAfterTransposeXGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> weight, const int32_t coorIndex, const int64_t ncOffset,
    LocalTensor<T> gOutLocalTensor, LocalTensor<T> zIndex, LocalTensor<T> yIndex, LocalTensor<T> xIndex)
{
    int32_t y = yIndex.GetValue(coorIndex);
    int32_t x = xIndex.GetValue(coorIndex);
    int32_t z = zIndex.GetValue(coorIndex);

    bool inRangeY = y >= 0 && y < xH;
    bool inRangeX = x >= 0 && x < xW;
    bool inRangeZ = z >= 0 && z < xD;

    if (!(inRangeY && inRangeX && inRangeZ)) {
        return;
    }

    T weightVal = weight.GetValue(coorIndex);
    int64_t offset = ncOffset + srcIndex.GetValue(coorIndex);
    LocalTensor<T> localTensor = dataOutQueue[DX_OUTPUT_INDEX].AllocTensor<T>();
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);

    Muls(localTensor, gOutLocalTensor, weightVal, channel);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    copyParams.blockLen = channel * sizeof(T);
    SetAtomicAdd<T>();
    DataCopyPad(inputGm[DX_INPUT_INDEX][offset], localTensor, copyParams);
    SetAtomicNone();
    dataOutQueue[DX_OUTPUT_INDEX].FreeTensor(localTensor);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeNearestXGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> weight, const int32_t coorIndex, const int32_t cycle,
    const int64_t ncOffset, LocalTensor<T> gOutLocalTensor, LocalTensor<T> zIndex, LocalTensor<T> yIndex,
    LocalTensor<T> xIndex)
{
    int32_t x = xIndex.GetValue(coorIndex);
    int32_t y = yIndex.GetValue(coorIndex);
    int32_t z = zIndex.GetValue(coorIndex);

    bool inRangeX = x >= 0 && x < xW;
    bool inRangeY = y >= 0 && y < xH;
    bool inRangeZ = z >= 0 && z < xD;

    if (!(inRangeX && inRangeY && inRangeZ)) {
        return;
    }

    T weightVal = weight.GetValue(coorIndex);
    int64_t offset = ncOffset + srcIndex.GetValue(coorIndex);

    LocalTensor<T> localTensor = dataOutQueue[DX_OUTPUT_INDEX].AllocTensor<T>();
    Muls(localTensor, gOutLocalTensor[cycle * alignChannel], weightVal, channel);

    event_t eventIDVToMTE3Nearest = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3Nearest);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3Nearest);

    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    copyParams.blockLen = channel * sizeof(T);

    SetAtomicAdd<T>();
    DataCopyPad(inputGm[DX_INPUT_INDEX][offset], localTensor, copyParams);
    SetAtomicNone();
    dataOutQueue[DX_OUTPUT_INDEX].FreeTensor(localTensor);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::CopyIn(
    const int64_t offset, const int32_t calCount, const int32_t inputIndex)
{
    LocalTensor<T> dataLocal = dataInQueue[inputIndex].AllocTensor<T>();
    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    DataCopyPadExtParams padParams = {true, 0, 0, static_cast<T>(0)};

    int32_t alignCalCount = CeilAlign(calCount, perBlockCount);
    copyParams.blockLen = calCount * sizeof(T);
    padParams.rightPadding = alignCalCount - calCount;

    DataCopyPad(dataLocal, inputGm[inputIndex][offset], copyParams, padParams);
    dataInQueue[inputIndex].EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::CopyOut(const int64_t offset, const int32_t calCount)
{
    LocalTensor<T> dstLocal = dataOutQueue[DGRID_OUTPUT_INDEX].DeQue<T>();
    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    copyParams.blockLen = calCount * sizeof(T);
    DataCopyPad(inputGm[DGRID_INPUT_INDEX][offset], dstLocal, copyParams);
    dataOutQueue[DGRID_OUTPUT_INDEX].FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::Compute(const int32_t computeCount, const int64_t curGridPointIndex)
{
    int32_t singleComputeCount = computeCount / 3;
    uint32_t mask = ELE_NUM_PER_REPEAT;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;
    uint8_t zPattern = 3;
    bool reduceMode = true;
    uint8_t src0BlockStride = 1;
    uint16_t repeatTimes = CeilDiv(computeCount, ELE_NUM_PER_REPEAT);
    uint8_t src0RepeatStride = REPEAT_STRIDE_0;
    uint8_t src1RepeatStride = REPEAT_STRIDE_1;

    InitComputeTensor();

    DupValue();

    // 获取grid坐标值
    GatherMask(
        xTensor, inputCoordinate, bufXPattern, reduceMode, GATHER_MASK_NUM,
        {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(
        yTensor, inputCoordinate, bufYPattern, reduceMode, GATHER_MASK_NUM,
        {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(
        zTensor, inputCoordinate, bufZPattern, reduceMode, GATHER_MASK_NUM,
        {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // gather ix
    ComputeSourceIndexSetGrad(xTensor, xGradIn, fwidth, singleComputeCount);

    // gather iy
    ComputeSourceIndexSetGrad(yTensor, yGradIn, fheight, singleComputeCount);

    // gather iz
    ComputeSourceIndexSetGrad(zTensor, zGradIn, fdepth, singleComputeCount);

    // bilinear
    if (interpolation == 0) {
        ComputeBilinear(singleComputeCount, curGridPointIndex);
    }

    // nearest
    if (interpolation == 1) {
        ComputeNearest(singleComputeCount, curGridPointIndex);
    }
    dataOutQueue[DGRID_OUTPUT_INDEX].EnQue(dstLocal);
    dataInQueue[GRID_INPUT_INDEX].FreeTensor(inputCoordinate);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitComputeTensor()
{
    xTensor = xCoordinateBuf.Get<T>(ubFactorElement + ELE_NUM_PER_REPEAT);
    yTensor = yCoordinateBuf.Get<T>(ubFactorElement + ELE_NUM_PER_REPEAT);
    zTensor = zCoordinateBuf.Get<T>(ubFactorElement + ELE_NUM_PER_REPEAT);

    xGradIn = xGradInBuf.Get<T>(ubFactorElement);
    yGradIn = yGradInBuf.Get<T>(ubFactorElement);
    zGradIn = zGradInBuf.Get<T>(ubFactorElement);

    inputCoordinate = dataInQueue[GRID_INPUT_INDEX].DeQue<T>();
    dstLocal = dataOutQueue[DGRID_OUTPUT_INDEX].AllocTensor<T>();

    bufXPattern = bufferMaskXBuf.Get<uint32_t>();
    bufXPattern.SetValue(0, 0b01001001001001001001001001001001);
    bufXPattern.SetValue(1, 0b10010010010010010010010010010010);
    bufXPattern.SetValue(2, 0b00100100100100100100100100100100);
    bufXPattern.SetValue(3, 0b01001001001001001001001001001001);
    bufXPattern.SetValue(4, 0b10010010010010010010010010010010);
    bufXPattern.SetValue(5, 0b00100100100100100100100100100100);

    bufYPattern = bufferMaskYBuf.Get<uint32_t>();
    bufYPattern.SetValue(0, 0b10010010010010010010010010010010);
    bufYPattern.SetValue(1, 0b00100100100100100100100100100100);
    bufYPattern.SetValue(2, 0b01001001001001001001001001001001);
    bufYPattern.SetValue(3, 0b10010010010010010010010010010010);
    bufYPattern.SetValue(4, 0b00100100100100100100100100100100);
    bufYPattern.SetValue(5, 0b01001001001001001001001001001001);

    bufZPattern = bufferMaskZBuf.Get<uint32_t>();
    bufZPattern.SetValue(0, 0b00100100100100100100100100100100);
    bufZPattern.SetValue(1, 0b01001001001001001001001001001001);
    bufZPattern.SetValue(2, 0b10010010010010010010010010010010);
    bufZPattern.SetValue(3, 0b00100100100100100100100100100100);
    bufZPattern.SetValue(4, 0b01001001001001001001001001001001);
    bufZPattern.SetValue(5, 0b10010010010010010010010010010010);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeBilinear(
    int32_t singleComputeCount, const int64_t curGridPointIndex)
{
    InitComputeBilinearTensor(singleComputeCount);

    // 计算权重
    // compute tnw
    ComputeWeight(tnw, xTensor, bxSe, yTensor, bySe, zTensor, bzSe, singleComputeCount);
    // compute tne
    ComputeWeight(tne, bxSw, xTensor, yTensor, bySw, zTensor, bzSw, singleComputeCount);
    // compute tsw
    ComputeWeight(tsw, xTensor, bxNe, byNe, yTensor, zTensor, bzNe, singleComputeCount);
    // compute tse
    ComputeWeight(tse, bxNw, xTensor, byNw, yTensor, zTensor, bzNw, singleComputeCount);
    // compute bnw
    ComputeWeight(bnw, xTensor, txSe, yTensor, tySe, tzSe, zTensor, singleComputeCount);
    // compute bne
    ComputeWeight(bne, txSw, xTensor, yTensor, tySw, tzSw, zTensor, singleComputeCount);
    // compute bsw
    ComputeWeight(bsw, xTensor, txNe, tyNe, yTensor, tzNe, zTensor, singleComputeCount);
    // compute bse
    ComputeWeight(bse, txNw, xTensor, tyNw, yTensor, tzNw, zTensor, singleComputeCount);

    // 判断点是否在边界内
    WithinBounds3d(selTensor1, tzNw, tyNw, txNw, tnw, singleComputeCount);
    WithinBounds3d(selTensor2, tzNe, tyNe, txNe, tne, singleComputeCount);
    WithinBounds3d(selTensor3, tzSw, tySw, txSw, tsw, singleComputeCount);
    WithinBounds3d(selTensor4, tzSe, tySe, txSe, tse, singleComputeCount);
    WithinBounds3d(selTensor5, bzNw, byNw, bxNw, bnw, singleComputeCount);
    WithinBounds3d(selTensor6, bzNe, byNe, bxNe, bne, singleComputeCount);
    WithinBounds3d(selTensor7, bzSw, bySw, bxSw, bsw, singleComputeCount);
    WithinBounds3d(selTensor8, bzSe, bySe, bxSe, bse, singleComputeCount);

    ComputeIndex(tNwIndex, tnwIndex2, tzNwInt, tyNwInt, txNwInt, singleComputeCount);
    ComputeIndex(tNeIndex, tneIndex2, tzNeInt, tyNeInt, txNeInt, singleComputeCount);
    ComputeIndex(tSwIndex, tswIndex2, tzSwInt, tySwInt, txSwInt, singleComputeCount);
    ComputeIndex(tSeIndex, tseIndex2, tzSeInt, tySeInt, txSeInt, singleComputeCount);
    ComputeIndex(bNwIndex, bnwIndex2, bzNwInt, byNwInt, bxNwInt, singleComputeCount);
    ComputeIndex(bNeIndex, bneIndex2, bzNeInt, byNeInt, bxNeInt, singleComputeCount);
    ComputeIndex(bSwIndex, bswIndex2, bzSwInt, bySwInt, bxSwInt, singleComputeCount);
    ComputeIndex(bSeIndex, bseIndex2, bzSeInt, bySeInt, bxSeInt, singleComputeCount);

    ComputeBilinearCommon(singleComputeCount, curGridPointIndex);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitComputeBilinearTensor(int32_t singleComputeCount)
{
    // 向下取整
    Cast(txNwInt, xTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Cast(txSwInt, xTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Adds(txNeInt, txNwInt, static_cast<int32_t>(1), singleComputeCount);
    Adds(txSeInt, txNwInt, static_cast<int32_t>(1), singleComputeCount);
    Cast(tyNwInt, yTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Cast(tyNeInt, yTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Adds(tySwInt, tyNwInt, static_cast<int32_t>(1), singleComputeCount);
    Adds(tySeInt, tyNwInt, static_cast<int32_t>(1), singleComputeCount);
    Cast(tzNwInt, zTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Cast(tzNeInt, zTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Cast(tzSwInt, zTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Cast(tzSeInt, zTensor, RoundMode::CAST_FLOOR, singleComputeCount);

    Cast(bxNwInt, xTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Cast(bxSwInt, xTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Adds(bxNeInt, bxNwInt, static_cast<int32_t>(1), singleComputeCount);
    Adds(bxSeInt, bxNwInt, static_cast<int32_t>(1), singleComputeCount);
    Cast(byNwInt, yTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Cast(byNeInt, yTensor, RoundMode::CAST_FLOOR, singleComputeCount);
    Adds(bySwInt, byNwInt, static_cast<int32_t>(1), singleComputeCount);
    Adds(bySeInt, byNwInt, static_cast<int32_t>(1), singleComputeCount);
    Adds(bzNwInt, tzNwInt, static_cast<int32_t>(1), singleComputeCount);
    Adds(bzNeInt, tzNwInt, static_cast<int32_t>(1), singleComputeCount);
    Adds(bzSwInt, tzNwInt, static_cast<int32_t>(1), singleComputeCount);
    Adds(bzSeInt, tzNwInt, static_cast<int32_t>(1), singleComputeCount);

    // convert to float32
    Cast(txNw, txNwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(tyNw, tyNwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(tzNw, tzNwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(txNe, txNeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(tyNe, tyNeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(tzNe, tzNeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(txSw, txSwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(tySw, tySwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(tzSw, tzSwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(txSe, txSeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(tySe, tySeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(tzSe, tzSeInt, RoundMode::CAST_NONE, singleComputeCount);

    Cast(bxNw, bxNwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(byNw, byNwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bzNw, bzNwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bxNe, bxNeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(byNe, byNeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bzNe, bzNeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bxSw, bxSwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bySw, bySwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bzSw, bzSwInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bxSe, bxSeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bySe, bySeInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(bzSe, bzSeInt, RoundMode::CAST_NONE, singleComputeCount);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeBilinearCommon(
    int32_t singleComputeCount, const int64_t curGridPointIndex)
{
    for (int32_t i = 0; i < singleComputeCount; i++) {
        gridPointIndex = curGridPointIndex + i;
        ComputeGridPointIndex(gridPointIndex);

        LocalTensor<T> gOutLocalTensor = dataInQueue[GRAD_INPUT_INDEX].AllocTensor<T>();
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        DataCopyPadExtParams padParams = {true, 0, 0, static_cast<T>(0)};

        int32_t alignCalCount = CeilAlign(channel, perBlockCount);
        copyParams.blockLen = channel * sizeof(T);
        padParams.rightPadding = alignCalCount - channel;
        DataCopyPad(gOutLocalTensor, inputGm[GRAD_INPUT_INDEX][gradGmOffset], copyParams, padParams);

        ComputeBilinearOutput(i, ncBaseOffset, gOutLocalTensor);

        ReduceSum<T>(sumZ, sumZ, sumZ, channel);
        ReduceSum<T>(sumY, sumY, sumY, channel);
        ReduceSum<T>(sumX, sumX, sumX, channel);

        gix += sumX.GetValue(0);
        giy += sumY.GetValue(0);
        giz += sumZ.GetValue(0);

        dstLocal.SetValue(3 * i, gix * xGradIn.GetValue(i));
        dstLocal.SetValue(3 * i + 1, giy * yGradIn.GetValue(i));
        dstLocal.SetValue(3 * i + 2, giz * zGradIn.GetValue(i));

        Duplicate<T>(sumX, 0, alignChannel);
        Duplicate<T>(sumY, 0, alignChannel);
        Duplicate<T>(sumZ, 0, alignChannel);

        gix = static_cast<T>(0);
        giy = static_cast<T>(0);
        giz = static_cast<T>(0);
        dataInQueue[GRAD_INPUT_INDEX].FreeTensor(gOutLocalTensor);
    }
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeBilinearOutput(
    int32_t i, int64_t ncBaseOffset, LocalTensor<T> gOutLocalTensor)
{
    ComputeAfterTransposeGridGrad(
        tNwIndex, bzSe, zTensor, bySe, yTensor, bxSe, xTensor, gOutLocalTensor, tzNw, tyNw, txNw, i, n, -1, -1, -1);
    ComputeAfterTransposeXGrad(tnwIndex2, tnw, i, ncBaseOffset, gOutLocalTensor, tzNw, tyNw, txNw);
    ComputeAfterTransposeGridGrad(
        tNeIndex, bzSw, zTensor, bySw, yTensor, xTensor, bxSw, gOutLocalTensor, tzNe, tyNe, txNe, i, n, 1, -1, -1);
    ComputeAfterTransposeXGrad(tneIndex2, tne, i, ncBaseOffset, gOutLocalTensor, tzNe, tyNe, txNe);
    ComputeAfterTransposeGridGrad(
        tSwIndex, bzNe, zTensor, yTensor, byNe, bxNe, xTensor, gOutLocalTensor, tzSw, tySw, txSw, i, n, -1, 1, -1);
    ComputeAfterTransposeXGrad(tswIndex2, tsw, i, ncBaseOffset, gOutLocalTensor, tzSw, tySw, txSw);
    ComputeAfterTransposeGridGrad(
        tSeIndex, bzNw, zTensor, yTensor, byNw, xTensor, bxNw, gOutLocalTensor, tzSe, tySe, txSe, i, n, 1, 1, -1);
    ComputeAfterTransposeXGrad(tseIndex2, tse, i, ncBaseOffset, gOutLocalTensor, tzSe, tySe, txSe);
    ComputeAfterTransposeGridGrad(
        bNwIndex, zTensor, tzSe, tySe, yTensor, txSe, xTensor, gOutLocalTensor, bzNw, byNw, bxNw, i, n, -1, -1, 1);
    ComputeAfterTransposeXGrad(bnwIndex2, bnw, i, ncBaseOffset, gOutLocalTensor, bzNw, byNw, bxNw);
    ComputeAfterTransposeGridGrad(
        bNeIndex, zTensor, tzSw, tySw, yTensor, xTensor, txSw, gOutLocalTensor, bzNe, byNe, bxNe, i, n, 1, -1, 1);
    ComputeAfterTransposeXGrad(bneIndex2, bne, i, ncBaseOffset, gOutLocalTensor, bzNe, byNe, bxNe);
    ComputeAfterTransposeGridGrad(
        bSwIndex, zTensor, tzNe, yTensor, tyNe, txNe, xTensor, gOutLocalTensor, bzSw, bySw, bxSw, i, n, -1, 1, 1);
    ComputeAfterTransposeXGrad(bswIndex2, bsw, i, ncBaseOffset, gOutLocalTensor, bzSw, bySw, bxSw);
    ComputeAfterTransposeGridGrad(
        bSeIndex, zTensor, tzNw, yTensor, tyNw, xTensor, txNw, gOutLocalTensor, bzSe, bySe, bxSe, i, n, 1, 1, 1);
    ComputeAfterTransposeXGrad(bseIndex2, bse, i, ncBaseOffset, gOutLocalTensor, bzSe, bySe, bxSe);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeNearest(
    int32_t singleComputeCount, const int64_t curGridPointIndex)
{
    InitComputeNearestTensor(singleComputeCount);

    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    DataCopyPadExtParams padParams = {true, 0, 0, static_cast<T>(0)};
    copyParams.blockLen = channel * sizeof(T);
    padParams.rightPadding = alignChannel - channel;

    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));

    if (channel <= CHANNEL_1024) {
        for (int32_t i = 0; i < (singleComputeCount / group); i++) {
            LocalTensor<T> gOutLocalTensor = dataInQueue[GRAD_INPUT_INDEX].AllocTensor<T>();
            for (int32_t j = 0; j < group; j++) {
                gridPointIndex = curGridPointIndex + i * group + j;
                ComputeGridPointIndex(gridPointIndex);
                DataCopyPad(
                    gOutLocalTensor[j * alignChannel], inputGm[GRAD_INPUT_INDEX][gradGmOffset], copyParams, padParams);
            }
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            for (int32_t k = 0; k < group; k++) {
                gridPointIndex = curGridPointIndex + i * group + k;
                n = gridPointIndex / (outD * outH * outW);
                ncBaseOffset = n * dxStrideN;
                ComputeNearestXGrad(
                    xIndex, dupOneTensor, i * group + k, k, ncBaseOffset, gOutLocalTensor, izFloat, iyFloat, ixFloat);
            }
            dataInQueue[GRAD_INPUT_INDEX].FreeTensor(gOutLocalTensor);
        }

        for (int32_t i = 0; i < (singleComputeCount % group); i++) {
            gridPointIndex = curGridPointIndex + (singleComputeCount / group) * group + i;
            ComputeGridPointIndex(gridPointIndex);
            LocalTensor<T> gOutLocalTensor = dataInQueue[GRAD_INPUT_INDEX].AllocTensor<T>();
            DataCopyPad(gOutLocalTensor, inputGm[GRAD_INPUT_INDEX][gradGmOffset], copyParams, padParams);
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            ComputeNearestXGrad(
                xIndex, dupOneTensor, (singleComputeCount / group) * group + i, 0, ncBaseOffset, gOutLocalTensor,
                izFloat, iyFloat, ixFloat);
            dataInQueue[GRAD_INPUT_INDEX].FreeTensor(gOutLocalTensor);
        }
    } else {
        for (int32_t i = 0; i < singleComputeCount; i++) {
            gridPointIndex = curGridPointIndex + i;
            ComputeGridPointIndex(gridPointIndex);
            LocalTensor<T> gOutLocalTensor = dataInQueue[GRAD_INPUT_INDEX].AllocTensor<T>();
            DataCopyPad(gOutLocalTensor, inputGm[GRAD_INPUT_INDEX][gradGmOffset], copyParams, padParams);
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            ComputeNearestXGrad(xIndex, dupOneTensor, i, 0, ncBaseOffset, gOutLocalTensor, izFloat, iyFloat, ixFloat);
            dataInQueue[GRAD_INPUT_INDEX].FreeTensor(gOutLocalTensor);
        }
    }
    Duplicate<T>(dstLocal, 0, 3 * ubFactorElement);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::InitComputeNearestTensor(int32_t singleComputeCount)
{
    // 获取距离最近点坐标
    Cast(ixNearInt, xTensor, RoundMode::CAST_RINT, singleComputeCount);
    Cast(iyNearInt, yTensor, RoundMode::CAST_RINT, singleComputeCount);
    Cast(izNearInt, zTensor, RoundMode::CAST_RINT, singleComputeCount);

    Cast(ixFloat, ixNearInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(iyFloat, iyNearInt, RoundMode::CAST_NONE, singleComputeCount);
    Cast(izFloat, izNearInt, RoundMode::CAST_NONE, singleComputeCount);

    WithinBounds3d(selTensor1, izFloat, iyFloat, ixFloat, dupOneTensor, singleComputeCount);
    ComputeIndex(xIndex, xIndex, izNearInt, iyNearInt, ixNearInt, singleComputeCount);
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::ComputeGridPointIndex(int32_t gridPointIndex)
{
    w = gridPointIndex % outW;
    h = (gridPointIndex / outW) % outH;
    d = (gridPointIndex / outW / outH) % outD;
    n = gridPointIndex / (outH * outW * outD);
    ncBaseOffset = n * dxStrideN;
    gradGmOffset = n * gradStrideN + (d * gradStrideD + h * gradStrideH + w * gradStrideW) * channel;
}

template <typename T>
__aicore__ inline void GridSampler3DGradNS<T>::Process()
{
    int64_t computePNum = 0;
    int64_t gridGmOffset = 0;
    int64_t gridOffset = 0;
    int64_t cycleOffset = 0;
    int64_t curGridPointIndex = 0;

    if (blockIdx < tailPNum) {
        computePNum = pNumPerCore + 1;
        gridOffset = blockIdx * computePNum;
    } else {
        computePNum = pNumPerCore;
        gridOffset = blockIdx * pNumPerCore + tailPNum;
        // 确定性计算处理N的尾块
        int64_t gridStride = gridD * gridH * gridW;
        int64_t tailCorePNum = pNumPerCore + gridStride;
        if (isDeterministic == 1 && blockIdx < tailBNum) {
            computePNum = tailCorePNum;
            gridOffset = blockIdx * computePNum;
        } else if (isDeterministic == 1 && blockIdx >= tailBNum) {
            computePNum = pNumPerCore;
            gridOffset = (blockIdx - tailBNum) * pNumPerCore + tailBNum * tailCorePNum;
        }
    }

    int32_t copyCountPerTime = 3 * ubFactorElement;
    int32_t copyTimes = CeilDiv(computePNum * 3, copyCountPerTime);
    int32_t actualComputeNum = copyCountPerTime;

    for (int32_t i = 0; i < copyTimes; i++) {
        if (i == copyTimes - 1) {
            actualComputeNum = computePNum * 3 - (copyTimes - 1) * copyCountPerTime;
        }
        cycleOffset = i * copyCountPerTime;
        gridGmOffset = cycleOffset + gridOffset * 3;
        curGridPointIndex = gridOffset + i * copyCountPerTime / 3;

        CopyIn(gridGmOffset, actualComputeNum, GRID_INPUT_INDEX);
        Compute(actualComputeNum, curGridPointIndex);
        CopyOut(gridGmOffset, actualComputeNum);
    }
}
} // namespace GridSampler3DGrad

#endif // UPSAMPLE_GRIDSAMPLER3D_GRAD