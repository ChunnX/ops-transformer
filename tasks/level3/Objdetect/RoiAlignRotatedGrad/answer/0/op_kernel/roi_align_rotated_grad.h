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
 * \file roi_align_rotated_grad.h
 * \brief
 */
#ifndef _ROI_ALIGN_ROTATED_GRAD_H_
#define _ROI_ALIGN_ROTATED_GRAD_H_
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
constexpr float half_value = -0.5f;
constexpr float half_plus_value = 0.5f;
constexpr uint32_t DTYPE_NUM = 32;
constexpr uint32_t BUFFER_NUM = 4;
constexpr uint32_t INDEX_ZERO = 0;
constexpr uint32_t INDEX_ONE = 1;
constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
constexpr uint32_t INDEX_FOUR = 4;
constexpr uint32_t INDEX_FIVE = 5;

class KernelRoiAlignRotatedGrad
{
public:
    __aicore__ inline KernelRoiAlignRotatedGrad() {}

    __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR rois, GM_ADDR grad_input,
                                const RoiAlignRotatedGradTilingData *__restrict tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        coreRoisNums = tiling_data->coreRoisNums;
        coreRoisTail = tiling_data->coreRoisTail;
        boxSize = tiling_data->boxSize;
        pooledWidth = tiling_data->pooledWidth;
        pooledHeight = tiling_data->pooledHeight;
        batchSize = tiling_data->batchSize;
        channelNum = tiling_data->channelNum;
        width = tiling_data->width;
        height = tiling_data->height;
        aligned = tiling_data->aligned;
        clockwise = tiling_data->clockwise;
        samplingRatio = tiling_data->samplingRatio;
        spatialScale = tiling_data->spatialScale;

        dataSize = DTYPE_NUM / sizeof(float);
        alignChannelNum = (channelNum + dataSize - 1) / dataSize;
        alignChannelNum = alignChannelNum * dataSize;

        uint32_t coreId = GetBlockIdx();
        if (coreId < coreRoisTail)
        {
            coreRoisNums += 1;
            startOffset = coreRoisNums * coreId;
        }
        else
        {
            startOffset = coreRoisNums * coreId + coreRoisTail;
        }

        eventIdMte2ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_MTE2>());

        copyParams = {2, static_cast<uint16_t>(channelNum * 2 / dataSize), 0, static_cast<uint16_t>((width - 2) * channelNum / dataSize)};

        roisGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rois), boxLength * boxSize);
        gradOutputsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(grad_output), boxLength * channelNum * pooledHeight * pooledWidth);
        gradInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(grad_input), batchSize * channelNum * height * width);
        InitBuffer();
    }

    __aicore__ inline void Process()
    {
        GetLocalTensor();
        uint32_t computeBatchSize = constComputeBatchSize;
        uint32_t computeBatchNum = (coreRoisNums + constComputeBatchSize - 1) / constComputeBatchSize;
        for (uint32_t taskBatchIdx = 0; taskBatchIdx < computeBatchNum; taskBatchIdx++)
        {
            uint32_t offset = startOffset + taskBatchIdx * constComputeBatchSize;
            if (taskBatchIdx == computeBatchNum - 1)
            {
                computeBatchSize = coreRoisNums - taskBatchIdx * computeBatchSize;
            }
            uint32_t alignComputeBatchNum = (computeBatchSize + dataSize - 1) / dataSize;
            alignComputeBatchNum = alignComputeBatchNum * dataSize;
            CopyIn(offset, alignComputeBatchNum);
            for (uint32_t taskIdx = 0; taskIdx < computeBatchSize; taskIdx++)
            {
                Compute(taskIdx, offset + taskIdx);
            }
        }
    }

    __aicore__ inline void InitBuffer()
    {
        pipe.InitBuffer(idxUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(xUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(yUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(hUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(wUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(angleUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(cosUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(sinUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(binSizeHUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(binSizeWUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(binGridWUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(binGridHUb, constComputeBatchSize * sizeof(int32_t));

        pipe.InitBuffer(binGridSizeWUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(binGridSizeHUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(deltaStartWUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(deltaStartHUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(countTmpUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(countUb, constComputeBatchSize * sizeof(float));
        pipe.InitBuffer(tmpUb, constComputeBatchSize * sizeof(float));

        pipe.InitBuffer(gradBinUb, alignChannelNum * sizeof(float));
        pipe.InitBuffer(gradW1Ub, alignChannelNum * sizeof(float));
        pipe.InitBuffer(gradW2Ub, alignChannelNum * sizeof(float));
        pipe.InitBuffer(gradW3Ub, alignChannelNum * sizeof(float));
        pipe.InitBuffer(gradW4Ub, alignChannelNum * sizeof(float));

        pipe.InitBuffer(gradOutUb, BUFFER_NUM * alignChannelNum * sizeof(float));

        pipe.InitBuffer(tmpChannelUb, alignChannelNum * sizeof(float));
    }

    __aicore__ inline void GetLocalTensor()
    {
        idxLocal = idxUb.Get<int32_t>();
        xLocal = xUb.Get<float>();
        yLocal = yUb.Get<float>();
        hLocal = hUb.Get<float>();
        wLocal = wUb.Get<float>();
        angleLocal = angleUb.Get<float>();

        cosLocal = cosUb.Get<float>();
        sinLocal = sinUb.Get<float>();

        binSizeHLocal = binSizeHUb.Get<float>();
        binSizeWLocal = binSizeWUb.Get<float>();

        binGridWLocal = binGridWUb.Get<int32_t>();
        binGridHLocal = binGridHUb.Get<int32_t>();

        binGridSizeWLocal = binGridSizeWUb.Get<float>();
        binGridSizeHLocal = binGridSizeHUb.Get<float>();

        deltaStartWLocal = deltaStartWUb.Get<float>();
        deltaStartHLocal = deltaStartHUb.Get<float>();

        countTmpLocal = countTmpUb.Get<int32_t>();
        countLocal = countUb.Get<float>();
        tmpLocal = tmpUb.Get<float>();

        gradBinLocal = gradBinUb.Get<float>();
        gradW1Local = gradW1Ub.Get<float>();
        gradW2Local = gradW2Ub.Get<float>();
        gradW3Local = gradW3Ub.Get<float>();
        gradW4Local = gradW4Ub.Get<float>();

        gradOutLocal = gradOutUb.Get<float>();

        tmpChannelLocal = tmpChannelUb.Get<float>();
        Duplicate(tmpChannelLocal, (float)0.0, alignChannelNum);
        Duplicate(tmpChannelLocal, (float)1.0, channelNum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t computeBatchSize)
    {
        DataCopy(tmpLocal, roisGM[offset + boxSize * INDEX_ZERO], computeBatchSize);
        DataCopy(xLocal, roisGM[offset + boxSize * INDEX_ONE], computeBatchSize);
        DataCopy(yLocal, roisGM[offset + boxSize * INDEX_TWO], computeBatchSize);
        DataCopy(wLocal, roisGM[offset + boxSize * INDEX_THREE], computeBatchSize);
        DataCopy(hLocal, roisGM[offset + boxSize * INDEX_FOUR], computeBatchSize);
        DataCopy(angleLocal, roisGM[offset + boxSize * INDEX_FIVE], computeBatchSize);

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if (clockwise)
        {
            Muls(angleLocal, angleLocal, (float)-1.0, computeBatchSize);
        }

        Cast(idxLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, computeBatchSize);
        Muls(xLocal, xLocal, (float)spatialScale, computeBatchSize);
        Muls(yLocal, yLocal, (float)spatialScale, computeBatchSize);
        Muls(hLocal, hLocal, (float)spatialScale, computeBatchSize);
        Muls(wLocal, wLocal, (float)spatialScale, computeBatchSize);

        if (aligned)
        {
            Adds(xLocal, xLocal, half_value, computeBatchSize);
            Adds(yLocal, yLocal, half_value, computeBatchSize);
        }
        else
        {
            Maxs(hLocal, hLocal, (float)1.0, computeBatchSize);
            Maxs(wLocal, wLocal, (float)1.0, computeBatchSize);
        }

        Cos(cosLocal, angleLocal, computeBatchSize);
        Sin(sinLocal, angleLocal, computeBatchSize);

        Duplicate(tmpLocal, (float)pooledHeight, computeBatchSize);
        Div(binSizeHLocal, hLocal, tmpLocal, computeBatchSize);
        Duplicate(tmpLocal, (float)pooledWidth, computeBatchSize);
        Div(binSizeWLocal, wLocal, tmpLocal, computeBatchSize);

        if (samplingRatio > 0)
        {
            Duplicate(binGridHLocal, samplingRatio, computeBatchSize);
            Duplicate(binGridWLocal, samplingRatio, computeBatchSize);
        }
        else
        {
            Cast(binGridHLocal, binSizeHLocal, AscendC::RoundMode::CAST_CEIL, computeBatchSize);
            Cast(binGridWLocal, binSizeWLocal, AscendC::RoundMode::CAST_CEIL, computeBatchSize);
        }

        Cast(tmpLocal, binGridHLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Div(binGridSizeHLocal, binSizeHLocal, tmpLocal, computeBatchSize);

        Cast(tmpLocal, binGridWLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Div(binGridSizeWLocal, binSizeWLocal, tmpLocal, computeBatchSize);

        Muls(deltaStartWLocal, wLocal, half_value, computeBatchSize);
        Muls(deltaStartHLocal, hLocal, half_value, computeBatchSize);

        Mul(countTmpLocal, binGridWLocal, binGridHLocal, computeBatchSize);
        Cast(countLocal, countTmpLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Maxs(countLocal, countLocal, (float)1.0, computeBatchSize);
    }

    __aicore__ inline void Compute(uint32_t taskIdx, uint32_t offset)
    {
        pIdx = idxLocal.GetValue(taskIdx);
        pX = xLocal.GetValue(taskIdx);
        pY = yLocal.GetValue(taskIdx);
        pCos = cosLocal.GetValue(taskIdx);
        pSin = sinLocal.GetValue(taskIdx);

        pBinSizeW = binSizeWLocal.GetValue(taskIdx);
        pBinSizeH = binSizeHLocal.GetValue(taskIdx);
        pBinGridW = binGridWLocal.GetValue(taskIdx);
        pBinGridH = binGridHLocal.GetValue(taskIdx);

        pDeltaStartW = deltaStartWLocal.GetValue(taskIdx);
        pDeltaStartH = deltaStartHLocal.GetValue(taskIdx);

        pBinGridSizeH = binGridSizeHLocal.GetValue(taskIdx);
        pBinGridSizeW = binGridSizeWLocal.GetValue(taskIdx);

        pCount = countLocal.GetValue(taskIdx);

        for (index = 0; index < pooledHeight * pooledWidth; index++)
        {
            pH = index / pooledWidth;
            pW = index - pH * pooledWidth;
            baseOffset = ((offset * pooledHeight + pH) * pooledWidth + pW) * channelNum;

            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            DataCopy(gradBinLocal, gradOutputsGm[baseOffset], alignChannelNum);

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            if (alignChannelNum != channelNum)
            {
                Mul(gradBinLocal, gradBinLocal, tmpChannelLocal, alignChannelNum);
            }
            Muls(gradBinLocal, gradBinLocal, (float)1.0 / (float)pCount, alignChannelNum);

            for (iy = 0; iy < pBinGridH; iy++)
            {
                yy = pDeltaStartH + pH * pBinSizeH + (iy + half_plus_value) * pBinGridSizeH;
                for (ix = 0; ix < pBinGridW; ix++)
                {
                    xx = pDeltaStartW + pW * pBinSizeW + (ix + half_plus_value) * pBinGridSizeW;

                    x = yy * pSin + xx * pCos + pX;
                    y = yy * pCos - xx * pSin + pY;

                    bilinearInterpolate();

                    if (xl >= 0 && xh >= 0 && yl >= 0 && yh >= 0)
                    {
                        if (channelNum == alignChannelNum && xh > xl && yh > yl)
                        {
                            CopyOutTogether();
                        }
                        else
                        {
                            CopyOut();
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void bilinearInterpolate()
    {
        if (y < -1 || y > height || x < -1 || x > width)
        {
            xl = -1;
            return;
        }
        if (y <= 0)
            y = 0;
        if (x <= 0)
            x = 0;

        yl = static_cast<int32_t>(y);
        xl = static_cast<int32_t>(x);

        if (yl >= height - 1)
        {
            yl = yh = height - 1;
            y = float(yl);
        }
        else
        {
            yh = yl + 1;
        }

        if (xl >= width - 1)
        {
            xl = xh = width - 1;
            x = float(xl);
        }
        else
        {
            xh = xl + 1;
        }

        ly = y - yl;
        lx = x - xl;

        w4 = ly * lx;
        w1 = w4 + 1 - ly - lx;
        w2 = lx - w4;
        w3 = ly - w4;
    }

    __aicore__ inline void CopyOut()
    {
        w1Offset = ((pIdx * height + yl) * width + xl) * channelNum;
        w2Offset = ((pIdx * height + yl) * width + xh) * channelNum;
        w3Offset = ((pIdx * height + yh) * width + xl) * channelNum;
        w4Offset = ((pIdx * height + yh) * width + xh) * channelNum;

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Muls(gradW1Local, gradBinLocal, w1, alignChannelNum);
        Muls(gradW2Local, gradBinLocal, w2, alignChannelNum);
        Muls(gradW3Local, gradBinLocal, w3, alignChannelNum);
        Muls(gradW4Local, gradBinLocal, w4, alignChannelNum);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<float>();

        DataCopy(gradInputGm[w1Offset], gradW1Local, alignChannelNum);
        DataCopy(gradInputGm[w2Offset], gradW2Local, alignChannelNum);
        DataCopy(gradInputGm[w3Offset], gradW3Local, alignChannelNum);
        DataCopy(gradInputGm[w4Offset], gradW4Local, alignChannelNum);

        SetAtomicNone();
    }

    __aicore__ inline void CopyOutTogether()
    {
        w1Offset = ((pIdx * height + yl) * width + xl) * channelNum;
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Muls(gradOutLocal, gradBinLocal, w1, channelNum);
        Muls(gradOutLocal[channelNum], gradBinLocal, w2, channelNum);
        Muls(gradOutLocal[channelNum * INDEX_TWO], gradBinLocal, w3, channelNum);
        Muls(gradOutLocal[channelNum * INDEX_THREE], gradBinLocal, w4, channelNum);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<float>();

        DataCopy(gradInputGm[w1Offset], gradOutLocal, copyParams);

        SetAtomicNone();
    }

private:
    TPipe pipe;
    GlobalTensor<float> roisGM, gradOutputsGm, gradInputGm;

    TBuf<TPosition::VECCALC> idxUb, xUb, yUb, hUb, wUb, angleUb;
    TBuf<TPosition::VECCALC> cosUb, sinUb;
    TBuf<TPosition::VECCALC> binSizeHUb, binSizeWUb;
    TBuf<TPosition::VECCALC> binGridWUb, binGridHUb;
    TBuf<TPosition::VECCALC> binGridSizeWUb, binGridSizeHUb;
    TBuf<TPosition::VECCALC> deltaStartWUb, deltaStartHUb;
    TBuf<TPosition::VECCALC> countTmpUb, countUb;
    TBuf<TPosition::VECCALC> tmpUb;

    TBuf<TPosition::VECCALC> gradBinUb;
    TBuf<TPosition::VECCALC> gradW1Ub, gradW2Ub, gradW3Ub, gradW4Ub;
    TBuf<TPosition::VECCALC> tmpChannelUb;
    TBuf<TPosition::VECCALC> gradOutUb;

    LocalTensor<int32_t> idxLocal;
    LocalTensor<float> xLocal, yLocal, hLocal, wLocal, angleLocal;
    LocalTensor<float> cosLocal, sinLocal;
    LocalTensor<float> binSizeHLocal, binSizeWLocal;
    LocalTensor<int32_t> binGridWLocal, binGridHLocal;
    LocalTensor<float> binGridSizeWLocal, binGridSizeHLocal;
    LocalTensor<float> deltaStartWLocal, deltaStartHLocal;
    LocalTensor<int32_t> countTmpLocal;
    LocalTensor<float> countLocal;
    LocalTensor<float> tmpLocal;

    LocalTensor<float> gradBinLocal;
    LocalTensor<float> gradW1Local, gradW2Local, gradW3Local, gradW4Local;
    LocalTensor<float> tmpChannelLocal;
    LocalTensor<float> gradOutLocal;

    uint32_t coreRoisNums;
    uint32_t coreRoisTail;
    uint32_t boxSize;
    uint32_t boxLength = 6;
    uint32_t batchSize;
    uint32_t channelNum;
    uint32_t width, height;
    int32_t pooledWidth;
    int32_t pooledHeight;
    bool aligned;
    bool clockwise;
    int32_t samplingRatio;
    float spatialScale;

    uint32_t dataSize;
    uint32_t alignChannelNum;

    uint32_t startOffset;
    int64_t baseOffset;
    uint32_t w1Offset, w2Offset, w3Offset, w4Offset;
    uint32_t constComputeBatchSize = 256;

    int32_t pIdx;
    int32_t index;
    float pX, pY;
    int32_t pH, pW;
    float pCos, pSin;

    float pBinSizeW, pBinSizeH;
    int32_t pBinGridW, pBinGridH;
    float pDeltaStartW, pDeltaStartH;
    float pBinGridSizeH, pBinGridSizeW;

    float pCount;

    float tmpH, tmpW;
    float tmpPH, tmpPW;

    int32_t ix, iy;
    float xx, yy;
    float x, y;
    int32_t xl, xh;
    int32_t yl, yh;

    float lx, hx;
    float ly, hy;

    float w1, w2, w3, w4;

    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV, eventIdMte3ToMte2;
    AscendC::DataCopyParams copyParams;
};
#endif // ROI_ALIGN_ROTATED_GRAD_H