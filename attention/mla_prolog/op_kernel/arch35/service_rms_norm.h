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
 * \file service_rms_norm.h
 * \brief
 */

#ifndef SERVICE_RMS_NORM_H
#define SERVICE_RMS_NORM_H

#include "mla_prolog_comm.h"
#include "mla_prolog_vector_comm.h"


namespace MlaProlog {

/**
 * @brief RmsNormNormal rmsNorm流程
 * @param outputLocal 输出tensor，[B, S1, H]
 * @param inputGm 输入tensor，[B, S1, H]，dtype支持bf16，fp32，int32
 * @param gammaLocal 系数gamma [H]，dtype只支持bf16
 * @param dequantScaleWDqLocal 权重w反量化系数
 * @param dequantScaleXLocal x反量化系数
 * @param shareTmpUb 临时buffer 所需空间[2 * cnt * sizeof(float) + ALIGN_BLOCK_SIZE]
          ALIGN_BLOCK_SIZE = 32Bytes, cnt = row * col
 * @param rmsNormParams rms所需系数，包括
          reciprocal rmsnorm系数reciprocal
          epsilon rmsnorm系数epsilon
          row 处理的行数；预留参数，当前仅支持单个batch的处理，row为1，对应S1
          col 列数，对应H
 */
template <typename T, typename GammaType, typename C, typename O, typename U, typename I = T>
__aicore__ inline void RmsNormNormal(const LocalTensor<O>& outputLocal, const GlobalTensor<T>& inputGm, const LocalTensor<GammaType>& gammaLocal,
                               const LocalTensor<float> &dequantScaleWDqLocal,
                               const LocalTensor<float> &dequantScaleXLocal, const LocalTensor<uint8_t>& shareTmpUb,
                               RmsNormParam& rmsNormParams, const SplitKnAccumParams& splitKnParams = DefaultSplitKnAccumParams()) {
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();
    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>();

    // load input [1, col]
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(rmsNormParams.col * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (splitKnParams.NeedAccumulate()) {
        AccumulateKnPartials(
                        inputLocal,
                        inputGm,
                        0, 
                        rmsNormParams.row,
                        rmsNormParams.col,
                        splitKnParams.kGroups,
                        splitKnParams.partialStride);
    } else {
        if constexpr (std::is_same<T, float>::value) {
            DataCopyPad(xFp32Local, inputGm, copyParams, padParams);
        } else {
            DataCopyPad(inputLocal, inputGm, copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    if constexpr (std::is_same<I, float>::value && std::is_same<U, FP8E8M0>::value) {
        xFp32Local = inputLocal;
    } else if constexpr (std::is_same<I, float>::value && std::is_same<U, float>::value) {
        xFp32Local = inputLocal;
        Rectangle rectangleParams {
            (uint32_t)rmsNormParams.row,
            (uint32_t)rmsNormParams.col,
            (uint32_t)rmsNormParams.col // columnStride
        };
        Dequant(xFp32Local, xFp32Local, dequantScaleWDqLocal, dequantScaleXLocal, rectangleParams);
        AscendC::PipeBarrier<PIPE_V>();
    } else if constexpr (std::is_same<I, int32_t>::value) {
        Rectangle rectangleParams {
            (uint32_t)rmsNormParams.row,
            (uint32_t)rmsNormParams.col,
            (uint32_t)rmsNormParams.col // columnStride
        };
        Dequant(xFp32Local, inputLocal, dequantScaleWDqLocal, dequantScaleXLocal, rectangleParams);
        AscendC::PipeBarrier<PIPE_V>();
    } else if constexpr (std::is_same<I, T>::value) {
        DataCopy(inputLocal[rmsNormParams.col], inputLocal, cnt);
        inputLocal = inputLocal[rmsNormParams.col];
        AscendC::PipeBarrier<PIPE_V>();
        // Cast input to fp32 [1, col]
        Cast(xFp32Local, inputLocal, RoundMode::CAST_NONE, cnt);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        xFp32Local = inputLocal;
    }
    LocalTensor<C> rmsnormShareUB = xFp32Local[rmsNormParams.col];
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);

    if constexpr (std::is_same<C, O>::value) {
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        // Cast xFp32 to outputLocal
        AscendC::PipeBarrier<PIPE_V>();
        Cast(outputLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
 * @brief RmsNormDynamicQuant rmsNorm + dynamicQuant融合流程
          目前C均为float
 * @param outputLocal 输出tensor，[B, S1, H], 量化完后为int8
 * @param outputScales dynamicQuant计算系数输出
 * @param inputGm 输入tensor，[B, S1, H]，dtype支持bf16，fp32，int32
 * @param gammaLocal 输入tensor，[H, ], dtype只支持bf16
 * @param smoothLocal dynamicQuant平滑参数
 * @param dequantScaleWDqLocal 权重w反量化系数
 * @param dequantScaleXLocal x反量化系数
 * @param shareTmpUb 临时buffer 所需空间[(4 * cnt + 8) * sizeof(float)] cnt = row * col
 * @param rmsNormParams rms所需系数，包括
          reciprocal rmsnorm系数reciprocal
          epsilon rmsnorm系数epsilon
          row 处理的行数；预留参数，当前仅支持单个batch的处理，row为1，对应S1
          col 列数，对应H
 * @param enableSmoothScalesCq 表示是否有smoothGm的需求
 */
template <typename T, typename GammaType, typename SmoothType, typename C, typename O, typename U, typename I = T>
__aicore__ inline void RmsNormDynamicQuant(const LocalTensor<O>& outputLocal, const LocalTensor<U> &outputScales, const GlobalTensor<T>& inputGm, const LocalTensor<GammaType>& gammaLocal,
                                           const LocalTensor<SmoothType>& smoothLocal, const LocalTensor<float> &dequantScaleWDqLocal,
                                           const LocalTensor<float>& dequantScaleXLocal, 
                                           const LocalTensor<uint8_t>& shareTmpUb,
                                           RmsNormParam& rmsNormParams, bool enableSmoothScalesCq = true, const SplitKnAccumParams& splitKnParams = DefaultSplitKnAccumParams()) {
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();
    RmsNormNormal<T, GammaType, C, C, U, I>(xFp32Local, inputGm, gammaLocal, dequantScaleWDqLocal, dequantScaleXLocal, shareTmpUb[rmsNormParams.col * sizeof(C)], rmsNormParams, splitKnParams);
    AscendC::PipeBarrier<PIPE_V>();
}

} // namespace MlaProlog

#endif