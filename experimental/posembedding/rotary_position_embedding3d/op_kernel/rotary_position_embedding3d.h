/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file rotary_position_embedding3d.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING3D_H
#define ROTARY_POSITION_EMBEDDING3D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "rotary_position_embedding3d_tiling_data.h"
#include "rotary_position_embedding3d_tiling_key.h"

namespace NsRotaryPositionEmbedding3d {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class RotaryPositionEmbedding3d {
public:
    __aicore__ inline RotaryPositionEmbedding3d(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const RotaryPositionEmbedding3dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(const int32_t dataLength);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    int64_t blockLength_ = 0;
    int64_t tileNum_ = 0;
    uint32_t tileLength_ = 0;
    int32_t lastDim_ = 0;
    int32_t halfD_ = 0;         // D/2
    int32_t numPositions_ = 0;  // 每 tile 的位置数
    TBuf<QuePosition::VECCALC> tmpCalc_; // FP32 临时缓冲区
};

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const RotaryPositionEmbedding3dTilingData* tilingData)
{
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    lastDim_ = static_cast<int32_t>(tilingData->lastDim);
    halfD_ = lastDim_ / 2;
    // 每个核处理全部数据作为一个 tile（无需双缓冲分片）
    tileLength_ = static_cast<uint32_t>(blockLength_);
    numPositions_ = static_cast<int32_t>(blockLength_ / lastDim_);
    tileNum_ = (numPositions_ > 0) ? 1 : 0;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * AscendC::GetBlockIdx(), blockLength_);
    inputGMY.SetGlobalBuffer((__gm__ T*)y + blockLength_ * AscendC::GetBlockIdx(), blockLength_);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + blockLength_ * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, tileLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, tileLength_ * sizeof(T));
    // FP32 临时缓冲：处理 half/bf16 时用于类型提升计算
    if constexpr (!std::is_same_v<T, float>) {
        pipe.InitBuffer(tmpCalc_, static_cast<uint32_t>(lastDim_) * sizeof(float) * 3);
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * tileLength_], tileLength_);
    AscendC::DataCopy(yLocal, inputGMY[progress * tileLength_], tileLength_);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();

    for (int32_t p = 0; p < numPositions_; p++) {
        int32_t base = p * lastDim_;
        // 取该位置的 x 左半/右半 和 y 左半(cos)/右半(sin)
        auto xL = xLocal[base];
        auto xR = xLocal[base + halfD_];
        auto cos = yLocal[base];
        auto sin = yLocal[base + halfD_];

        if constexpr (std::is_same_v<T, float>) {
            // RoPE half: zL = xL*cos - xR*sin,  zR = xL*sin + xR*cos
            // 步骤1: 用 cos/sin 算 xL 部分（cos/sin 读取后不再需要）
            AscendC::Mul(zLocal[base], xL, cos, halfD_);          // zL_tmp = xL*cos
            AscendC::Mul(zLocal[base + halfD_], xL, sin, halfD_); // zR_tmp = xL*sin
            // 步骤2: 用 xR 算剩余项，存 xL 区（xR 读取后不再需要）
            AscendC::Mul(xLocal[base], xR, sin, halfD_);          // tmp = xR*sin  → xL 区
            AscendC::Sub(zLocal[base], zLocal[base], xLocal[base], halfD_); // zL = xL*cos - xR*sin
            AscendC::Mul(xLocal[base], xR, cos, halfD_);          // tmp = xR*cos  → xL 区（覆盖）
            AscendC::Add(zLocal[base + halfD_], zLocal[base + halfD_], xLocal[base], halfD_); // zR = xL*sin + xR*cos
        } else {
            // half/bf16: Cast → FP32 向量计算 → Cast 回
            constexpr auto RN = RoundMode::CAST_NONE;
            uint32_t hd = static_cast<uint32_t>(halfD_);
            uint32_t hdBytes = hd * sizeof(float);
            // tmpCalc_ 划为 6 块: [xL_f32 | xR_f32 | cos_f32 | sin_f32 | zL_f32 | zR_f32], 每块 halfD 个 float
            auto xLF = tmpCalc_.Get<float>();
            auto xRF = tmpCalc_.Get<float>(hdBytes);
            auto cosF = tmpCalc_.Get<float>(hdBytes * 2);
            auto sinF = tmpCalc_.Get<float>(hdBytes * 3);
            auto zLf = tmpCalc_.Get<float>(hdBytes * 4);
            auto zRf = tmpCalc_.Get<float>(hdBytes * 5);

            Cast(xLF, xL, RN, hd);
            Cast(xRF, xR, RN, hd);
            Cast(cosF, cos, RN, hd);
            Cast(sinF, sin, RN, hd);
            AscendC::PipeBarrier<PIPE_V>();

            // zL = xL * cos - xR * sin
            AscendC::Mul(zLf, xLF, cosF, hd);
            AscendC::Mul(zRf, xRF, sinF, hd);
            AscendC::Sub(zLf, zLf, zRf, hd);

            // zR = xL * sin + xR * cos
            AscendC::Mul(zRf, xLF, sinF, hd);
            AscendC::Mul(xRF, xRF, cosF, hd); // 复用 xRF
            AscendC::Add(zRf, zRf, xRF, hd);

            AscendC::PipeBarrier<PIPE_V>();
            Cast(zLocal[base], zLf, RN, hd);
            Cast(zLocal[base + halfD_], zRf, RN, hd);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::Process()
{
    if (tileNum_ == 0) return;
    for (int32_t i = 0; i < tileNum_; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsRotaryPositionEmbedding3d
#endif