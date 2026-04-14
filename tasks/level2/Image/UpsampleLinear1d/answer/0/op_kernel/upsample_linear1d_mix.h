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
#ifndef UPSAMPLE_LINEAR1D_MIX
#define UPSAMPLE_LINEAR1D_MIX

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "upsample_linear_common.h"

namespace UpsampleLinear1d {
using namespace AscendC;

constexpr MatmulConfig MDL_CFG_MIX = GetMDLConfig(true, false, 0, false, false, false, true);

template <typename T>
class UpsampleLinear1dMixND {
public:
    TPipe pipe;
    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, 
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, MDL_CFG_MIX>
        matmulW;

    __aicore__ inline UpsampleLinear1dMixND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleLinear1dTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const UpsampleLinear1dTilingData *tilingData);
    __aicore__ inline void calculateWidthExtensionFloat(int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd);
    __aicore__ inline void copyRadioTensorToGm(int8_t direction);
    __aicore__ inline void getSlideRange();
    __aicore__ inline void calculateRadio(int64_t loopIndex, int64_t length, int64_t& xMin, int64_t& singleCoreK, float scale_w, bool align_corners, int64_t wIn, int64_t slide_size_w);

private:
    // 系数矩阵下标队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> radioQueue;

    const TCubeTiling *__restrict matmulTiling_w;
    GlobalTensor<float> intermediateTensorGm;
    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    bool align_corners = false;
    int64_t blockIdxMix = 0;
    int64_t need_core_num_w;
    int64_t slide_size_w = 0;
    float scale_w;

    uint32_t radio_matrix_size_w;
    int64_t slideNumW;
    int64_t eachCoreSlideNumW;
    int64_t tailStartSlideNumW;
    
    int64_t groupCoreNumW;
    int64_t tailAvergingRowsW;
    int64_t remainderW;
    int64_t slideEnd_w = 0;   
    int64_t slideStart_w = 0;
    int64_t tailSlideStart_w = 0;
    int64_t tailRowEnd_w = 0;
    int64_t tailSlideEnd_w = 0;
    int64_t tailRowStart_w = 0;
    int64_t output_shapes[3] = {0, 0, 0};
    int64_t input_shapes[3] = {0, 0, 0};
    
    int64_t singleCoreKTiling = 0;
    int64_t workSpaceRadioOffset = 0;
    int64_t singleCoreK = 0;
    
    int64_t xMin = 0;
    int64_t inputH = 0;
};

template <typename T>
__aicore__ inline void UpsampleLinear1dMixND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const UpsampleLinear1dTilingData *tilingData)
{
    blockIdxMix = GetBlockIdx();
    ParseTilingData(tilingData);
    getSlideRange();
    if (!FloatEqual(scale_w, 1.0)) {
        pipe.InitBuffer(radioQueue, BUFFER_NUM, radio_matrix_size_w * sizeof(float));
    }
    intermediateTensorGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    inTensorsGM.SetGlobalBuffer((__gm__ T *)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T *)output);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dMixND<T>::Process()
{
    if (FloatEqual(scale_w, 1.0) || blockIdxMix >= need_core_num_w) {
        return ;
    }
    if constexpr (std::is_same<T, float>::value) {
        if (slideStart_w < slideEnd_w) {
            for (int64_t index = slideStart_w; index < slideEnd_w; index += slide_size_w) {
                int16_t length = Min(slide_size_w, slideEnd_w - index);
                // 计算系数矩阵
                calculateRadio(index, length, xMin, singleCoreK, scale_w, align_corners, input_shapes[2], slide_size_w);
                calculateWidthExtensionFloat(index, 0, 0);
            }
        }

        // 处理尾块部分数据
        if (tailSlideStart_w < tailSlideEnd_w) {
            for (int64_t index = tailSlideStart_w; index < tailSlideEnd_w; index += slide_size_w) {
                int16_t length = Min(slide_size_w, tailSlideEnd_w - index);
                calculateRadio(index, length, xMin, singleCoreK, scale_w, align_corners, input_shapes[2], slide_size_w);
                calculateWidthExtensionFloat(index, tailRowStart_w, tailRowEnd_w);
            }
        }
    }
}

template <typename T>
__aicore__ inline void UpsampleLinear1dMixND<T>::calculateRadio(
    int64_t loopIndex, int64_t length, int64_t& xMin, int64_t& singleCoreK, 
    float scale_w, bool align_corners, int64_t wIn, int64_t slide_size_w)
{
    calculateSingleCoreK(loopIndex, length, xMin, singleCoreK, scale_w, align_corners, wIn);
    LocalTensor<float> radioTensorMix = radioQueue.AllocTensor<float>();
    Duplicate(radioTensorMix, (float)0.0, radioTensorMix.GetSize());
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    calculateRadioTensorW(loopIndex, length, radioTensorMix, xMin, singleCoreK, scale_w, align_corners, wIn, slide_size_w);
    radioQueue.EnQue(radioTensorMix);
    copyRadioTensorToGm(0);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dMixND<T>::calculateWidthExtensionFloat(
    int64_t tensorCIndex, int64_t rowStart, int64_t rowEnd)
{
    if (singleCoreK <= 0) {
        return ;
    }
    int64_t singleCoreM = matmulTiling_w->singleCoreM;
    int64_t singleCoreN = matmulTiling_w->singleCoreN;
    // 尾块batch分批处理
    if (rowEnd != 0) {
        singleCoreM = rowEnd - rowStart;
    }
    matmulW.SetOrgShape(singleCoreM, singleCoreN, input_shapes[2], singleCoreK, output_shapes[2]);
    matmulW.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);

    if (tensorCIndex + slide_size_w > output_shapes[2]) {
        matmulW.SetTail(singleCoreM, output_shapes[2] - tensorCIndex, singleCoreK);
    }
    int64_t xIndex = xMin + rowStart * input_shapes[2];
    int64_t tensorCIndexWithOffset = tensorCIndex + rowStart * output_shapes[2];

    matmulW.SetTensorA(inTensorsGM[xIndex], false);
    matmulW.SetTensorB(intermediateTensorGm[workSpaceRadioOffset], false);
    matmulW.IterateAll(outTensorsGM[tensorCIndexWithOffset], false);
    matmulW.End();

    event_t eventID3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID3);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID3);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dMixND<T>::copyRadioTensorToGm(int8_t direction)
{
    // 系数矩阵从ub拷贝到GM
    workSpaceRadioOffset = radio_matrix_size_w * blockIdxMix;
    LocalTensor<float> radioTensorMix = radioQueue.DeQue<float>();
    DataCopy(intermediateTensorGm[workSpaceRadioOffset], radioTensorMix, radioTensorMix.GetSize());
    event_t eventID2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
    radioQueue.FreeTensor(radioTensorMix);
}

template <typename T>
__aicore__ inline void UpsampleLinear1dMixND<T>::ParseTilingData(const UpsampleLinear1dTilingData *tilingData)
{
    align_corners = tilingData->align_corners;
    slide_size_w = tilingData->slide_size_w;
    scale_w = tilingData->scale_w;
    need_core_num_w = tilingData->need_core_num_w;

    for (int8_t i = 0; i < 3; i++) {
        output_shapes[i] = tilingData->output_shapes[i];
        input_shapes[i] = tilingData->input_shapes[i];
    }

    radio_matrix_size_w = (tilingData->radio_matrix_size_w + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    eachCoreSlideNumW = tilingData->eachCoreSlideNumW;
    groupCoreNumW = tilingData->groupCoreNumW;
    tailStartSlideNumW = tilingData->tailStartSlideNumW;
    slideNumW = tilingData->slideNumW;
    remainderW = tilingData->remainderW;
    tailAvergingRowsW = tilingData->tailAvergingRowsW;
    matmulTiling_w = &tilingData->matmulTiling_w;
}

template <typename T>
__aicore__ inline void UpsampleLinear1dMixND<T>::getSlideRange()
{
    inputH = input_shapes[0] * input_shapes[1];
    slideStart_w = blockIdxMix * eachCoreSlideNumW * slide_size_w;
    slideEnd_w = Min((Min((blockIdxMix + 1) * eachCoreSlideNumW, slideNumW)) * slide_size_w, output_shapes[2]);
    int64_t groupIndex = groupCoreNumW > 0 ? blockIdxMix / groupCoreNumW : 0;
    if (groupIndex < remainderW) {
        tailSlideStart_w = (tailStartSlideNumW + groupIndex) * slide_size_w;
        tailSlideEnd_w = Min(tailSlideStart_w + slide_size_w, output_shapes[2]);
        int64_t blockIdxInGroup = groupCoreNumW > 0 ? blockIdxMix % groupCoreNumW : 0;
        tailRowStart_w = blockIdxInGroup * tailAvergingRowsW;
        tailRowEnd_w = Min(tailRowStart_w + tailAvergingRowsW, inputH);
    }
}

}  // namespace UpsampleLinear1d

#endif  // UPSAMPLE_LINEAR1D