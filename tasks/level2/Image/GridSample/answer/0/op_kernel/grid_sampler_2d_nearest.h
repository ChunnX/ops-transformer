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
 * \file grid_sampler_2d_nearest.h
 * \brief
 */
#ifndef GIRD_SAMPLER_2D_NEAREST
#define GIRD_SAMPLER_2D_NEAREST

#include "kernel_vec_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

template <typename T>
class GridSampler2DNearest {
public:
    __aicore__ inline GridSampler2DNearest(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData, TPipe pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GridSampleTilingData *tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
        LocalTensor<uint8_t> weightMaskUb);
    __aicore__ inline void CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound);
    __aicore__ inline void CoordinatesSelectScalar(LocalTensor<float> iFpUb, LocalTensor<float> oFpUb,
        LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum);
    __aicore__ inline void CoordinatesSelectTensor(
        LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb);
    __aicore__ inline void Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
        LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb, LocalTensor<uint8_t> maskTmpXUb,
        LocalTensor<uint8_t> maskTmpYUb);
    __aicore__ inline void ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void ZeroClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
        LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb, LocalTensor<uint8_t> maskUb,
        LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow, const int64_t twiceHigh);
    __aicore__ inline void MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal);
    __aicore__ inline void OutTransposeFp16(
        int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb);
    __aicore__ inline void OutTransposeFp32(
        int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb);

    __aicore__ inline void MTE3ForNCHWFp16(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset, LocalTensor<float> weightUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void MTE3ForNCHWFp32(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
        int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset, LocalTensor<float> weightUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd);

    __aicore__ inline void PointNearest(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
        LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
        LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void CopyOutFp16(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xBuf_;

    TBuf<QuePosition::VECCALC> gridFp32Buf_;
    TBuf<QuePosition::VECCALC> inputXYFPBuf_;
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> coorTmpBuf_;
    TBuf<QuePosition::VECCALC> intTmpBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;

    TBuf<QuePosition::VECCALC> gridFp16Buf_;
    TBuf<QuePosition::VECCALC> yFp16Buf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<float> gmWorkspace_;
    GlobalTensor<T> gmY_;

    const int64_t TRANSE_REP_STRIDE = 128;
    const int64_t B32_MASK = 64;
    const int64_t CHANNEL_BLOCK = 64;
    const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;

    const int64_t X_UB_SIZE_4_GENERAL = 32768;    // 32KB
    const int64_t X_UB_SIZE_4_FP16 = 16384;       // 16KB
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;  //  4KB
    const int64_t GRID_UB_SIZE_4_FP16 = 2048;     //  2KB
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;     //  2KB
    const int64_t CAL_H_W_BLOCK = 512;
    const int64_t MASK_UB_SIZE = CAL_H_W_BLOCK / 8;

    int64_t blockIDX = 0;

    // tiling params
    int64_t interpolationMode_ = 0;
    int64_t coreNum_ = 0;
    int64_t inputN_ = 0;
    int64_t inputC_ = 0;
    int64_t inputH_ = 0;
    int64_t inputW_ = 0;
    int64_t outputH_ = 0;
    int64_t outputW_ = 0;
    int64_t paddingMode_ = 0;
    int64_t channelLast_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t alignCorners_ = 0;

    int64_t preNUbLoop_ = 0;
    int64_t lastLoopHW_ = 0;
    int64_t totalUbLoop_ = 0;
    int64_t preCoreLoop_ = 0;
    int64_t channelLoop_ = 0;
    int64_t lastCoreLoop_ = 0;
    int64_t perLoopChannel_ = 0;
    int64_t lastLoopChannel_ = 0;
    int64_t gridHW_ = 0;

    // const define
    constexpr static int64_t PADDING_MODE_BORDER = 1;
    constexpr static int64_t PADDING_MODE_ZEROS = 0;
    constexpr static int64_t PADDING_MODE_REFLECTION = 2;
    constexpr static int64_t REFLECT_RATIO = 2;
    constexpr static int64_t LAYOUT_NHWC = 1;

    constexpr static uint64_t B32_VECTOR_MASK = 64;
    constexpr static uint64_t B32_BLOCK_STRIDE = 1;
    constexpr static uint64_t B32_REPEAT_STRIDE = 8;
    constexpr static int64_t B32_ALIGN_FACTOR = 8;
    constexpr static int64_t B16_ALIGN_FACTOR = 16;
};

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::ParseTilingData(const GridSampleTilingData *tilingData)
{
    coreNum_ = tilingData->coreNumVar;
    inputN_ = tilingData->inN;
    inputC_ = tilingData->inC;
    inputH_ = tilingData->inH;
    inputW_ = tilingData->inW;
    outputH_ = tilingData->outH;
    outputW_ = tilingData->outW;
    interpolationMode_ = tilingData->interpolationMode;
    paddingMode_ = tilingData->paddingMode;
    alignCorners_ = tilingData->alignCorners;
    channelLast_ = tilingData->channelLast;
    needCoreNum_ = tilingData->needCoreNum;
    gridHW_ = outputH_ * outputW_;
    preNUbLoop_ = (gridHW_ + CAL_H_W_BLOCK - 1) / CAL_H_W_BLOCK;
    lastLoopHW_ = gridHW_ - CAL_H_W_BLOCK * (preNUbLoop_ - 1);
    totalUbLoop_ = preNUbLoop_ * inputN_;
    preCoreLoop_ = (totalUbLoop_ + needCoreNum_ - 1) / needCoreNum_;
    needCoreNum_ = (totalUbLoop_ + preCoreLoop_ - 1) / preCoreLoop_;
    lastCoreLoop_ = totalUbLoop_ - preCoreLoop_ * (needCoreNum_ - 1);

    channelLoop_ = (inputC_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    perLoopChannel_ = CHANNEL_BLOCK;
    lastLoopChannel_ = inputC_ - perLoopChannel_ * (channelLoop_ - 1);
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::Init(
    GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace, const GridSampleTilingData *tilingData, TPipe pipeIn)
{
    pipe = pipeIn;
    blockIDX = GetBlockIdx();
    // 初始化tiling
    ParseTilingData(tilingData);

    gmX_.SetGlobalBuffer((__gm__ T *)x);
    gmY_.SetGlobalBuffer((__gm__ T *)y);
    gmGrid_.SetGlobalBuffer((__gm__ T *)gird);
    gmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);

    // buffer initialize
    pipe.InitBuffer(xBuf_, X_UB_SIZE_4_GENERAL);                 // 32KB
    pipe.InitBuffer(gridFp32Buf_, GRID_UB_SIZE_4_GENERAL);       //  4KB
    pipe.InitBuffer(inputXYFPBuf_, GRID_UB_SIZE_4_GENERAL);      //  4KB
    pipe.InitBuffer(inputXIntBuf_, GRID_UB_SIZE_4_GENERAL * 2);  //  8KB
    pipe.InitBuffer(inputYIntBuf_, GRID_UB_SIZE_4_GENERAL);      //  4KB
    pipe.InitBuffer(weightBuf_, Y_UB_SIZE_4_GENERAL * 4);        //  8KB
    pipe.InitBuffer(intTmpBuf_, Y_UB_SIZE_4_GENERAL);            //  2KB
    pipe.InitBuffer(coorBuf_, Y_UB_SIZE_4_GENERAL);              //  2KB
    pipe.InitBuffer(coorTmpBuf_, Y_UB_SIZE_4_GENERAL);           //  2KB
    pipe.InitBuffer(outValueBuf_, X_UB_SIZE_4_GENERAL);          // 32KB
    pipe.InitBuffer(maskBuf_, 960);                              // 960B
    pipe.InitBuffer(weightMaskBuf_, 320);                        // 320B
    pipe.InitBuffer(modBuf_, Y_UB_SIZE_4_GENERAL);               //  2KB
    pipe.InitBuffer(extraBuf_, Y_UB_SIZE_4_GENERAL);             //  2KB
    pipe.InitBuffer(outTmpBuf_, GRID_UB_SIZE_4_GENERAL);         //  4KB
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        pipe.InitBuffer(gridFp16Buf_, GRID_UB_SIZE_4_FP16);  //  2KB
        pipe.InitBuffer(yFp16Buf_, X_UB_SIZE_4_FP16);        // 16KB
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
    LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb, LocalTensor<int32_t> coorUb,
    LocalTensor<uint8_t> wMaskUb)
{
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> xIntTmpUb = coorUb;
    LocalTensor<int32_t> yIntTmpUb = tmpIntUb;
    PipeBarrier<PIPE_V>();
    Adds(xIntTmpUb, iXIntUb, 0, CAL_H_W_BLOCK);
    Adds(yIntTmpUb, iYIntUb, 0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Cast(iXFpUb, iXIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(iYFpUb, iYIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<uint8_t> maskXUbLocal = wMaskUb;
    LocalTensor<uint8_t> maskYUb = maskUb;
    LocalTensor<uint8_t> maskTmpXUb = maskUb[MASK_UB_SIZE];
    LocalTensor<uint8_t> maskTmpYUb = maskUb[MASK_UB_SIZE * 2];  // 2: iY temp mask
    CoordinatesGetMaskWithRange(iXFpUb, iYFpUb, maskXUbLocal, maskYUb, maskTmpXUb, maskTmpYUb);
    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskXUbTmp = maskXUbLocal.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskYUbTmp, maskXUbTmp, maskNum);
    wMaskUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_V>();

    CoordinatesFrameRange(xIntTmpUb, (int32_t)(inputW_ - 1));
    CoordinatesFrameRange(yIntTmpUb, (int32_t)(inputH_ - 1));

    PipeBarrier<PIPE_V>();

    Muls(yIntTmpUb, yIntTmpUb, (int32_t)inputW_, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Add(coorUb, coorUb, yIntTmpUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    if (paddingMode_ == PADDING_MODE_BORDER) {
        BorderClip(iXFpUb, iYFpUb);
    } else if (paddingMode_ == PADDING_MODE_REFLECTION) {
        ReflectClip(iXFpUb, iYFpUb);
    } else if (paddingMode_ == PADDING_MODE_ZEROS) {
        ZeroClip(iXFpUb, iYFpUb);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound)
{
    Mins(iIntUb, iIntUb, upBound, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iIntUb, iIntUb, 0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb,
    LocalTensor<float> iYFpUb, LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
    LocalTensor<uint8_t> maskTmpXUb, LocalTensor<uint8_t> maskTmpYUb)
{
    CompareScalar(maskTmpXUb, iXFpUb, 0.0f, CMPMODE::GE, CAL_H_W_BLOCK);
    CompareScalar(maskXUb, iXFpUb, static_cast<float>(inputW_ - 1), CMPMODE::LE, CAL_H_W_BLOCK);
    CompareScalar(maskTmpYUb, iYFpUb, 0.0f, CMPMODE::GE, CAL_H_W_BLOCK);
    CompareScalar(maskYUb, iYFpUb, static_cast<float>(inputH_ - 1), CMPMODE::LE, CAL_H_W_BLOCK);

    PipeBarrier<PIPE_V>();

    int32_t maskNum = (MASK_UB_SIZE + 1) / 2;  // 除2数据量按照uint16类型折半
    auto maskTmpXUbTmp = maskTmpXUb.ReinterpretCast<uint16_t>();
    auto maskXUbTmp = maskXUb.ReinterpretCast<uint16_t>();
    auto maskTmpYUbTmp = maskTmpYUb.ReinterpretCast<uint16_t>();
    auto maskYUbTmp = maskYUb.ReinterpretCast<uint16_t>();
    And(maskXUbTmp, maskTmpXUbTmp, maskXUbTmp, maskNum);
    And(maskYUbTmp, maskTmpYUbTmp, maskYUbTmp, maskNum);
    PipeBarrier<PIPE_V>();
    maskXUb = maskXUbTmp.ReinterpretCast<uint8_t>();
    maskYUb = maskYUbTmp.ReinterpretCast<uint8_t>();
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::CoordinatesSelectScalar(LocalTensor<float> iFpUb,
    LocalTensor<float> oFpUb, LocalTensor<uint8_t> maskUb, const float scalarVal, const uint32_t calNum)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = 0;
    repParams.src1RepStride = 0;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (calNum + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(oFpUb, maskUb, iFpUb, scalarVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::CoordinatesSelectTensor(
    LocalTensor<float> src0, LocalTensor<float> src1, LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb)
{
    BinaryRepeatParams repParams;
    repParams.src0BlkStride = B32_BLOCK_STRIDE;
    repParams.src0RepStride = B32_REPEAT_STRIDE;
    repParams.src1BlkStride = B32_BLOCK_STRIDE;
    repParams.src1RepStride = B32_REPEAT_STRIDE;
    repParams.dstBlkStride = B32_BLOCK_STRIDE;
    repParams.dstRepStride = B32_REPEAT_STRIDE;
    uint8_t repeat = (CAL_H_W_BLOCK + B32_VECTOR_MASK - 1) / B32_VECTOR_MASK;
    Select(coorUb, maskUb, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, B32_VECTOR_MASK, repeat, repParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iYFpUb, iYFpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iYFpUb, iYFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> maskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<float> tmpZeroUb = inputXYFPBuf_.Get<float>();
    Muls(tmpZeroUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpZeroUb, tmpZeroUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(tmpZeroUb, iYFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpZeroUb, tmpZeroUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpTmpUb)
{
    LocalTensor<float> coorSubUb = coorTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> extraFpUb = extraBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<float> fmodFpUb = modBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE * 3);
    LocalTensor<float> tmpFpUb = outTmpBuf_.Get<float>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> tmpIntUb = intTmpBuf_.Get<int32_t>(CAL_H_W_BLOCK);

    // coorUb = Y * inputW_ + X
    int64_t twiceLow = (alignCorners_ == 1) ? 0 : -1;
    int64_t twiceLowY = REFLECT_RATIO * (inputH_ - 1);
    int64_t twiceLowX = REFLECT_RATIO * (inputW_ - 1);
    if (alignCorners_ == 0) {
        twiceLow = -1;
        twiceLowY = REFLECT_RATIO * inputH_ - 1;
        twiceLowX = REFLECT_RATIO * inputW_ - 1;
    }
    ReflectCoordinatesGeneral(iYFpTmpUb, iYFpTmpUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowY);
    PipeBarrier<PIPE_V>();
    ReflectCoordinatesGeneral(iXFpUb, iXFpUb, extraFpUb, fmodFpUb, maskUb, tmpFpUb, tmpIntUb, twiceLow, twiceLowX);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> tmpUseUb = inputXYFPBuf_.Get<float>();
    Muls(tmpUseUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUseUb, tmpUseUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(tmpUseUb, iYFpTmpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUseUb, tmpUseUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iYFpTmpUb, iYFpTmpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iXFpUb, iXFpUb, (float)(inputW_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iXFpUb, iXFpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Mins(iYFpTmpUb, iYFpTmpUb, (float)(inputH_ - 1), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Maxs(iYFpTmpUb, iYFpTmpUb, (float)0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::ZeroClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb)
{
    LocalTensor<uint8_t> maskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    LocalTensor<float> tmpUb = inputXYFPBuf_.Get<float>();
    Muls(tmpUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, -100.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(tmpUb, iYFpUb, (float)(0.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    CoordinatesSelectScalar(iYFpUb, iYFpUb, maskUb, -100.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::ReflectCoordinatesGeneral(LocalTensor<float> iFpUb,
    LocalTensor<float> coorSubUb, LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb,
    LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpFpUb, LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow,
    const int64_t twiceHighVal)
{
    if (twiceLow == twiceHighVal) {
        Duplicate(coorSubUb, (float)0.0, CAL_H_W_BLOCK);
        return;
    }

    float minS = static_cast<float>(twiceLow) / 2;
    float negMinS = static_cast<float>(-1.0) * minS;
    float spanSFp = static_cast<float>(twiceHighVal - twiceLow) / 2;

    // new relative position
    Adds(coorSubUb, iFpUb, negMinS, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Abs(coorSubUb, coorSubUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // extra
    Muls(extraFpUb, coorSubUb, static_cast<float>(1.0f / spanSFp), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, extraFpUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(extraFpUb, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(extraFpUb, extraFpUb, spanSFp, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Sub(extraFpUb, coorSubUb, extraFpUb, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // flip
    Muls(coorSubUb, coorSubUb, static_cast<float>(1.0f / spanSFp), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, coorSubUb, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(coorSubUb, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    // coordinate
    /*
     S1: get two results for both possibilities, out1: extra + min, out2: muls(extra, -1.0) + span + min
     S2: get mod val, mods: flips % 2
     S3: get mask tensor, masks: CompareScalar(mods, 0.0)
     S4: select val from out1 and out2 by mask tensor, out: Select(out1, out2, mask)
    */
    LocalTensor<float> out1 = tmpFpUb;
    LocalTensor<float> out2LocalTensor = extraFpUb;
    LocalTensor<float> modsLocalTensor = fmodFpUb;

    Adds(out1, extraFpUb, minS, CAL_H_W_BLOCK);
    Muls(out2LocalTensor, extraFpUb, -1.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(out2LocalTensor, out2LocalTensor, spanSFp, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Adds(out2LocalTensor, out2LocalTensor, minS, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    Muls(modsLocalTensor, coorSubUb, static_cast<float>(1 / 2.0), CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(tmpIntUb, modsLocalTensor, RoundMode::CAST_FLOOR, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(modsLocalTensor, tmpIntUb, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Muls(modsLocalTensor, modsLocalTensor, 2.0f, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Sub(modsLocalTensor, coorSubUb, modsLocalTensor, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CompareScalar(maskUb, modsLocalTensor, static_cast<float>(0.0), CMPMODE::EQ, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    CoordinatesSelectTensor(out1, out2LocalTensor, coorSubUb, maskUb);
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    for (int32_t i = 0; i < loopElems; i++) {
        int64_t coordVal = coorUb.GetValue(loopOffset + i);
        int64_t baseLocation = nIdx * inputC_ * inputH_ * inputW_ + coordVal + cIdx * CHANNEL_BLOCK * inputH_ * inputW_;
        for (int cIter = 0; cIter < channelAlign; cIter++) {
            int32_t xLocalOffset = i * channelAlign + cIter;
            if (cIter >= calCElems) {
                if constexpr (IsSameType<T, bfloat16_t>::value) {
                    xLocal.SetValue(xLocalOffset, ToBfloat16(0.0));
                } else {
                    xLocal.SetValue(xLocalOffset, static_cast<T>(0.0));
                }
                continue;
            }

            int64_t coordinate = baseLocation + cIter * inputH_ * inputW_;
            xLocal.SetValue(xLocalOffset, gmX_.GetValue(coordinate));
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb, LocalTensor<T> xLocal)
{
    int64_t base = nIdx * inputH_ * inputW_ * inputC_ + cIdx * CHANNEL_BLOCK;
    auto timeStep = loopElems / 8;

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = calCElems * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padTmpParams{false, 0, 0, 0};
    for (int32_t i = 0; i < timeStep; i++) {
        int64_t coordTmpVal_0 = coorUb.GetValue(loopOffset + i * 8) * inputC_;
        int64_t coordTmpVal_1 = coorUb.GetValue(loopOffset + i * 8 + 1) * inputC_;
        int64_t coordTmpVal_2 = coorUb.GetValue(loopOffset + i * 8 + 2) * inputC_;
        int64_t coordTmpVal_3 = coorUb.GetValue(loopOffset + i * 8 + 3) * inputC_;
        int64_t coordTmpVal_4 = coorUb.GetValue(loopOffset + i * 8 + 4) * inputC_;
        int64_t coordTmpVal_5 = coorUb.GetValue(loopOffset + i * 8 + 5) * inputC_;
        int64_t coordTmpVal_6 = coorUb.GetValue(loopOffset + i * 8 + 6) * inputC_;
        int64_t coordTmpVal_7 = coorUb.GetValue(loopOffset + i * 8 + 7) * inputC_;
        int64_t location_0 = base + coordTmpVal_0;
        int64_t location_1 = base + coordTmpVal_1;
        int64_t location_2 = base + coordTmpVal_2;
        int64_t location_3 = base + coordTmpVal_3;
        int64_t location_4 = base + coordTmpVal_4;
        int64_t location_5 = base + coordTmpVal_5;
        int64_t location_6 = base + coordTmpVal_6;
        int64_t location_7 = base + coordTmpVal_7;

        DataCopyPad(xLocal[(i * 8) * channelAlign], gmX_[location_0], params, padTmpParams);
        DataCopyPad(xLocal[(i * 8 + 1) * channelAlign], gmX_[location_1], params, padTmpParams);
        DataCopyPad(xLocal[(i * 8 + 2) * channelAlign], gmX_[location_2], params, padTmpParams);
        DataCopyPad(xLocal[(i * 8 + 3) * channelAlign], gmX_[location_3], params, padTmpParams);
        DataCopyPad(xLocal[(i * 8 + 4) * channelAlign], gmX_[location_4], params, padTmpParams);
        DataCopyPad(xLocal[(i * 8 + 5) * channelAlign], gmX_[location_5], params, padTmpParams);
        DataCopyPad(xLocal[(i * 8 + 6) * channelAlign], gmX_[location_6], params, padTmpParams);
        DataCopyPad(xLocal[(i * 8 + 7) * channelAlign], gmX_[location_7], params, padTmpParams);
    }
    for (auto i = loopElems / 8 * 8; i < loopElems; i++) {
        int64_t coordVal_0 = coorUb.GetValue(loopOffset + i) * inputC_;
        int64_t location_0 = base + coordVal_0;
        DataCopyPad(xLocal[i * channelAlign], gmX_[location_0], params, padTmpParams);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::OutTransposeFp16(
    int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb)
{
    LocalTensor<T> srcList[16];
    LocalTensor<T> dstList[16];

    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (channelAlign == B16_ALIGN_FACTOR) {
        transDataParams.repeatTimes = 8;
        transDataParams.dstRepStride = 1;
        transDataParams.srcRepStride = 16;

        for (int32_t iValue = 0; iValue < 16; iValue++) {
            srcList[iValue] = xLocal[iValue * 16];
        }

        for (int32_t iValue = 0; iValue < 16; iValue++) {
            dstList[iValue] = outValueUb[iValue * TRANSE_REP_STRIDE];
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<T>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= 64) {
        transDataParams.repeatTimes = channelAlign / 16;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < 8; j++) {
            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = xLocal[i * channelAlign + j * 16 * channelAlign];
            }

            for (int32_t i = 0; i < 16; i++) {
                dstList[i] = outValueUb[i * TRANSE_REP_STRIDE + j * 16];
            }

            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<T>(dstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::OutTransposeFp32(
    int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb)
{
    LocalTensor<float> dstList[16];
    LocalTensor<float> srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (channelAlign == 8) {
        transDataParams.repeatTimes = 8;
        transDataParams.dstRepStride = 2;
        transDataParams.srcRepStride = 16;

        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = xLocal[i * 8];
        }

        for (int32_t i = 0; i < 8; i++) {
            dstList[i * 2] = outValueUb[i * TRANSE_REP_STRIDE];
            dstList[i * 2 + 1] = outValueUb[i * TRANSE_REP_STRIDE + 8];
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<float>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else if (channelAlign <= 64) {
        transDataParams.repeatTimes = channelAlign / 8;
        transDataParams.dstRepStride = TRANSE_REP_STRIDE;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < 8; j++) {
            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = xLocal[i * channelAlign + j * 16 * channelAlign];
            }

            for (int32_t i = 0; i < 8; i++) {
                dstList[i * 2] = outValueUb[i * TRANSE_REP_STRIDE + j * 16];
                dstList[i * 2 + 1] = outValueUb[i * TRANSE_REP_STRIDE + 8 + j * 16];
            }

            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<float>(dstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::MTE3ForNCHWFp32(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset,
    LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    // 512 * inputC_ * blockIDX 每个核的地址
    // loopOffset 偏移的是几个128
    int64_t gmYBaseOffsetTmp = outBaseOffset + loopOffset + cIdx * CHANNEL_BLOCK * gridHW_;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    if (calCElems == 1) {
        Mul(outValueUb, outValueUb, weightUb[loopOffset], TRANSE_REP_STRIDE);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            DataCopyPad(gmY_[gmYBaseOffsetTmp], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            SetAtomicNone();
        } else {
            DataCopyPad(gmY_[gmYBaseOffsetTmp], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
        }
    } else {
        for (int32_t loopIdx = 0; loopIdx < TRANSE_MUL_WEGHT_LOOPS; loopIdx++) {
            int32_t outOffsetVal = loopIdx * B32_MASK;
            int32_t weightOffset = loopOffset + loopIdx * B32_MASK;
            Mul(outValueUb[outOffsetVal],
                outValueUb[outOffsetVal],
                weightUb[weightOffset],
                B32_MASK,
                calCElems,
                {1, 1, 1, 16, 16, 0});
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            for (int32_t cLocalIdx = 0; cLocalIdx < calCElems; cLocalIdx++) {
                int64_t gmYOffset = gmYBaseOffsetTmp + cLocalIdx * gridHW_;
                event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
                WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);
                DataCopyPad(gmY_[gmYOffset],
                    outValueUb[cLocalIdx * TRANSE_REP_STRIDE],
                    {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
            SetAtomicNone();
        } else {
            for (int32_t cLocalIdx = 0; cLocalIdx < calCElems; cLocalIdx++) {
                int64_t gmYOffset = gmYBaseOffsetTmp + cLocalIdx * gridHW_;
                event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
                WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);
                DataCopyPad(gmY_[gmYOffset],
                    outValueUb[cLocalIdx * TRANSE_REP_STRIDE],
                    {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::MTE3ForNCHWFp16(int32_t nIdx, int32_t cIdx, int32_t calCElems,
    int32_t channelAlign, int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset,
    LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    // 512 * inputC_ * blockIDX 每个核的地址
    // loopOffset 偏移的是几个128
    int64_t gmYBaseOffset = CAL_H_W_BLOCK * inputC_ * blockIDX + loopOffset + cIdx * CHANNEL_BLOCK * CAL_H_W_BLOCK;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    if (calCElems == 1) {
        Mul(outValueUb, outValueUb, weightUb[loopOffset], TRANSE_REP_STRIDE);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            DataCopyPad(gmWorkspace_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            SetAtomicNone();
        } else {
            DataCopyPad(gmWorkspace_[gmYBaseOffset], outValueUb, {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
        }
    } else {
        for (int32_t i = 0; i < TRANSE_MUL_WEGHT_LOOPS; i++) {
            int32_t outOffset = i * B32_MASK;
            int32_t weightOffsetVal = loopOffset + i * B32_MASK;
            Mul(outValueUb[outOffset],
                outValueUb[outOffset],
                weightUb[weightOffsetVal],
                B32_MASK,
                calCElems,
                {1, 1, 1, 16, 16, 0});
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        if (isAutomicAdd) {
            SetAtomicAdd<float>();
            for (int32_t i = 0; i < calCElems; i++) {
                int64_t gmYOffset = gmYBaseOffset + i * CAL_H_W_BLOCK;
                event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
                WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);
                DataCopyPad(gmWorkspace_[gmYOffset],
                    outValueUb[i * TRANSE_REP_STRIDE],
                    {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
            SetAtomicNone();
        } else {
            for (int32_t i = 0; i < calCElems; i++) {
                int64_t gmYOffset = gmYBaseOffset + i * CAL_H_W_BLOCK;
                event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
                WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);
                DataCopyPad(gmWorkspace_[gmYOffset],
                    outValueUb[i * TRANSE_REP_STRIDE],
                    {1, (uint16_t)(loopElems * sizeof(float)), 0, 0});
            }
        }
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::PointNearest(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
    LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb, LocalTensor<uint8_t> weightMaskUb,
    LocalTensor<float> outValueUb, bool isAutomicAdd)
{
    if (paddingMode_ == PADDING_MODE_ZEROS) {
        CoordinatesSelectScalar(weightUb, weightUb, weightMaskUb, 0.0f, CAL_H_W_BLOCK);
    }

    LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>(MASK_UB_SIZE);
    auto maskUbTmp = maskUb.ReinterpretCast<uint64_t>();
    auto weightMaskUbTmp = weightMaskUb.ReinterpretCast<uint64_t>();

    int32_t trans_loop = (calHWElems + TRANSE_REP_STRIDE - 1) / TRANSE_REP_STRIDE;
    int32_t loop_offset = 0;
    int32_t loop_elems = TRANSE_REP_STRIDE;
    int64_t outBaseOffset = nIdx * gridHW_ * inputC_ + hwIdx * CAL_H_W_BLOCK;
    int32_t maskOffset = 0;
    int32_t ubOffset = 0;
    for (int32_t transLoopIdx = 0; transLoopIdx < trans_loop; transLoopIdx++) {
        if (transLoopIdx == trans_loop - 1) {
            loop_elems = calHWElems - TRANSE_REP_STRIDE * (trans_loop - 1);
        }
        loop_offset = transLoopIdx * TRANSE_REP_STRIDE;
        maskOffset = transLoopIdx * 2;
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        maskUbTmp.SetValue(0, weightMaskUbTmp.GetValue(maskOffset));
        maskUbTmp.SetValue(1, weightMaskUbTmp.GetValue(maskOffset + 1));

        LocalTensor<T> xLocal = xBuf_.AllocTensor<T>();
        for (int32_t cIdx = 0; cIdx < channelLoop_; cIdx++) {
            int32_t calCElems = perLoopChannel_;
            if (cIdx == channelLoop_ - 1) {
                calCElems = lastLoopChannel_;
            }
            int32_t channelAlign = Ceil(calCElems, B32_ALIGN_FACTOR) * B32_ALIGN_FACTOR;
            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                channelAlign = Ceil(calCElems, B16_ALIGN_FACTOR) * B16_ALIGN_FACTOR;
            }
            if (channelLast_ == LAYOUT_NHWC) {
                MTE2ForNHWC(nIdx, cIdx, calCElems, channelAlign, loop_offset, loop_elems, coordinatesUb, xLocal);
            } else {
                MTE2ForNCHW(nIdx, cIdx, calCElems, channelAlign, loop_offset, loop_elems, coordinatesUb, xLocal);
            }
            event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventMte2V);

            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {  // T: fp16
                LocalTensor<T> yFp16Ub = yFp16Buf_.Get<T>();
                OutTransposeFp16(channelAlign, xLocal, yFp16Ub);
                PipeBarrier<PIPE_V>();
                Cast(outValueUb, yFp16Ub, RoundMode::CAST_NONE, calCElems * TRANSE_REP_STRIDE);
            } else {  // T: fp32
                OutTransposeFp32(channelAlign, xLocal, outValueUb);
            }
            PipeBarrier<PIPE_V>();

            for (size_t elemIdx = 0; elemIdx < calCElems; elemIdx++) {
                ubOffset = elemIdx * TRANSE_REP_STRIDE;
                Select(outValueUb[ubOffset],
                    maskUbTmp,
                    outValueUb[ubOffset],
                    0.0f,
                    SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    TRANSE_REP_STRIDE);
            }
            PipeBarrier<PIPE_V>();

            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                MTE3ForNCHWFp16(nIdx,
                    cIdx,
                    calCElems,
                    channelAlign,
                    hwIdx,
                    loop_offset,
                    loop_elems,
                    outBaseOffset,
                    weightUb,
                    outValueUb,
                    isAutomicAdd);
            } else {
                MTE3ForNCHWFp32(nIdx,
                    cIdx,
                    calCElems,
                    channelAlign,
                    hwIdx,
                    loop_offset,
                    loop_elems,
                    outBaseOffset,
                    weightUb,
                    outValueUb,
                    isAutomicAdd);
            }
            event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMte3V);
            WaitFlag<HardEvent::MTE3_V>(eventMte3V);
        }
    }
}
template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::CopyOutFp16(int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    LocalTensor<float> outLocal = xBuf_.AllocTensor<float>();
    LocalTensor<T> outLocalFP16 = yFp16Buf_.AllocTensor<T>();
    // 每次处理16*512个数据
    int64_t loopTime = Ceil(inputC_, 16);
    int64_t lastC = inputC_ - 16 * (loopTime - 1);
    int64_t dataCount = CAL_H_W_BLOCK * 16;
    int64_t basegmWorkSpaceAddr = blockIDX * CAL_H_W_BLOCK * inputC_;

    event_t eventIdMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdV_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t eventIdMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    for (size_t i = 0; i < loopTime - 1; i++) {
        DataCopy(outLocal, gmWorkspace_[basegmWorkSpaceAddr + dataCount * i], dataCount);

        SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
        Cast(outLocalFP16, outLocal, RoundMode::CAST_RINT, dataCount);

        SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
        DataCopyExtParams params;
        params.blockCount = 16;
        params.blockLen = calHWElems * sizeof(T);
        params.srcStride = CAL_H_W_BLOCK / 16 - Ceil(calHWElems, 16);
        params.dstStride = (outputH_ * outputW_ - calHWElems) * sizeof(T);
        int64_t gmYOffset = (int64_t)nIdx * outputH_ * outputW_ * inputC_ + (int64_t)hwIdx * CAL_H_W_BLOCK +
                            i * 16 * outputH_ * outputW_;
        DataCopyPad(gmY_[gmYOffset], outLocalFP16, params);

        SetFlag<HardEvent::V_MTE2>(eventIdV_MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdV_MTE2);

        SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    }

    dataCount = CAL_H_W_BLOCK * lastC;
    DataCopy(outLocal, gmWorkspace_[basegmWorkSpaceAddr + CAL_H_W_BLOCK * 16 * (loopTime - 1)], dataCount);

    SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    Cast(outLocalFP16, outLocal, RoundMode::CAST_RINT, dataCount);

    SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    DataCopyExtParams params;
    params.blockCount = lastC;
    params.blockLen = calHWElems * sizeof(T);
    params.srcStride = CAL_H_W_BLOCK / 16 - Ceil(calHWElems, 16);
    params.dstStride = (outputH_ * outputW_ - calHWElems) * sizeof(T);
    int64_t gmYOffset = (int64_t)nIdx * outputH_ * outputW_ * inputC_ + (int64_t)hwIdx * CAL_H_W_BLOCK +
                        (loopTime - 1) * 16 * outputH_ * outputW_;
    DataCopyPad(gmY_[gmYOffset], outLocalFP16, params);
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems)
{
    int64_t gridGmOffset = nIdx * gridHW_ * 2 + hwIdx * CAL_H_W_BLOCK * 2;
    LocalTensor<float> gridFp32Local = gridFp32Buf_.Get<float>();
    DataCopyExtParams paramsGridObj;
    paramsGridObj.blockCount = 1;
    paramsGridObj.blockLen = calHWElems * 2 * sizeof(T);
    paramsGridObj.srcStride = 0;
    paramsGridObj.dstStride = 0;
    DataCopyPadExtParams<T> padParamsGrid{false, 0, 0, 0};
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {  // T: fp16
        LocalTensor<T> gridFp16Local = gridFp16Buf_.Get<T>();
        DataCopyPad(gridFp16Local, gmGrid_[gridGmOffset], paramsGridObj, padParamsGrid);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Cast(gridFp32Local, gridFp16Local, RoundMode::CAST_NONE, CAL_H_W_BLOCK * 2);
        PipeBarrier<PIPE_V>();
    } else {  // T: fp32
        DataCopyPad(gridFp32Local, gmGrid_[gridGmOffset], paramsGridObj, padParamsGrid);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

    LocalTensor<float> inputXYUb = inputXYFPBuf_.Get<float>();
    Adds(inputXYUb, gridFp32Local, (float)1.0, CAL_H_W_BLOCK * 2);

    uint32_t mask = CAL_H_W_BLOCK * 2;
    uint64_t rsvdCnt = 0;
    uint8_t xPattern = 1;
    uint8_t yPattern = 2;

    uint8_t src1RepeatStride = 8;
    uint8_t src0RepeatStrideVal = 8;
    PipeBarrier<PIPE_V>();
    LocalTensor<float> inputXFpLocal = gridFp32Local;
    LocalTensor<float> inputYFpLocalTmp = gridFp32Local[CAL_H_W_BLOCK];
    // 分别取x和y(inputXFpLocal, inputXYUb, xPattern, true, mask,
    GatherMask(inputXFpLocal, inputXYUb, xPattern, true, mask, {1, 1, src0RepeatStrideVal, src1RepeatStride}, rsvdCnt);
    GatherMask(inputYFpLocalTmp, inputXYUb, yPattern, true, mask, {1, 1, src0RepeatStrideVal, src1RepeatStride}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    if (alignCorners_ == 1) {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * (inputW_ - (float)1.0)), CAL_H_W_BLOCK);
        Muls(inputYFpLocalTmp, inputYFpLocalTmp, (float)((float)0.5 * (inputH_ - (float)1.0)), CAL_H_W_BLOCK);
    } else {
        Muls(inputXFpLocal, inputXFpLocal, (float)((float)0.5 * inputW_), CAL_H_W_BLOCK);
        Muls(inputYFpLocalTmp, inputYFpLocalTmp, (float)((float)0.5 * inputH_), CAL_H_W_BLOCK);
        PipeBarrier<PIPE_V>();
        Adds(inputXFpLocal, inputXFpLocal, (float)(-0.5), CAL_H_W_BLOCK);
        Adds(inputYFpLocalTmp, inputYFpLocalTmp, (float)(-0.5), CAL_H_W_BLOCK);
    }
    PipeBarrier<PIPE_V>();

    Clip(inputXFpLocal, inputYFpLocalTmp);

    LocalTensor<int32_t> inputXIntLocal = inputXIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<int32_t> inputYIntLocal = inputYIntBuf_.Get<int32_t>(CAL_H_W_BLOCK);

    Cast(inputXIntLocal, inputXFpLocal, RoundMode::CAST_RINT, CAL_H_W_BLOCK);
    Cast(inputYIntLocal, inputYFpLocalTmp, RoundMode::CAST_RINT, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();
    Cast(inputXFpLocal, inputXIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    Cast(inputYFpLocalTmp, inputYIntLocal, RoundMode::CAST_NONE, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> weightLocal = weightBuf_.Get<float>(CAL_H_W_BLOCK);
    Duplicate(weightLocal, (float)1.0, CAL_H_W_BLOCK);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> coordinatesLocal = coorBuf_.Get<int32_t>(CAL_H_W_BLOCK);
    LocalTensor<float> outValueLocal = outValueBuf_.Get<float>();
    LocalTensor<uint8_t> weightMaskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);

    ClipCoordinates(inputXFpLocal, inputYFpLocalTmp, inputXIntLocal, inputYIntLocal, coordinatesLocal, weightMaskUb);
    PointNearest(nIdx, hwIdx, calHWElems, coordinatesLocal, weightLocal, weightMaskUb, outValueLocal, false);

    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        event_t eventMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
        CopyOutFp16(nIdx, hwIdx, calHWElems);
        event_t eventMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3ToV);
    }
}

template <typename T>
__aicore__ inline void GridSampler2DNearest<T>::Process()
{
    if (blockIDX >= needCoreNum_) {
        return;
    }

    int32_t nIdx = 0;
    int32_t hwIdx = 0;
    int32_t preLoopNum = blockIDX * preCoreLoop_;
    int32_t calHWElems = 0;

    int64_t loopSize = preCoreLoop_;
    if (blockIDX == needCoreNum_ - 1) {
        loopSize = lastCoreLoop_;
    }

    for (int32_t loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        nIdx = (preLoopNum + loopIdx) / preNUbLoop_;
        hwIdx = (preLoopNum + loopIdx) % preNUbLoop_;
        calHWElems = CAL_H_W_BLOCK;
        if (hwIdx == preNUbLoop_ - 1) {
            calHWElems = lastLoopHW_;
        }
        PerLoopCompute(nIdx, hwIdx, calHWElems);
    }
}

}  // namespace GridSample
#endif  // GIRD_SAMPLER_2D_FP16_NEAREST