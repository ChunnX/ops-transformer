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
 * \file service_dequant_arch35.h
 * \brief
 */

#ifndef SERVICE_DEQUANT_ARCH35_H
#define SERVICE_DEQUANT_ARCH35_H

#include "mla_prolog_comm_arch35.h"
#include "mla_prolog_vector_comm_arch35.h"

namespace MlaProlog {

/**
 * @brief DequantPerTokenQc 用于对Qc做反量化；按行做dequant流程，给oriRow * col的数据做反量化
 * @param outputGm 输出tensor
 * @param inputGm 输入tensor
 * @param deqScaleQcQrWGm deqScaleQcQr反量化系数，原shape[1,ND]
 * @param deQuantScaleQcQrLocal deQuantScaleQcQr反量化系数，原shape[BS,1]
 * @param shareTmpUb 临时buffer
 * @param dequantRowColStrideParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param oriRow 一共有多少行
*/
template <typename T, typename C, typename O>
__aicore__ inline void DequantPerTokenQc(const GlobalTensor<O> &outputGm, const GlobalTensor<T> &inputGm,
                                         const GlobalTensor<C> &deqScaleQcQrWGm,
                                         const LocalTensor<float> deQuantScaleQcQrLocal,
                                         const LocalTensor<uint8_t> &shareTmpUb, Rectangle dequantRowColStrideParams,
                                         uint32_t oriRow)
{
    int64_t perTokenCount = dequantRowColStrideParams.row * dequantRowColStrideParams.col;

    LocalTensor<T> perTokenInputLocal = shareTmpUb.ReinterpretCast<T>(); // count * sizeof(T)
    LocalTensor<C> perTokenScaleLocal =
        perTokenInputLocal[perTokenCount + 16].template ReinterpretCast<C>(); // count * sizeof(C)
    LocalTensor<C> perTokenComputeLocal =
        perTokenScaleLocal[perTokenCount + 16].template ReinterpretCast<C>(); // count * sizeof(C)
    LocalTensor<O> perTokenOutputLocal =
        perTokenComputeLocal[perTokenCount + 16].template ReinterpretCast<O>(); // count * sizeof(O)

    DataCopyParams perTokenCopyParams{
        static_cast<uint16_t>(dequantRowColStrideParams.row),
        static_cast<uint16_t>(dequantRowColStrideParams.col * sizeof(T) / 32U),
        static_cast<uint16_t>((dequantRowColStrideParams.stride - dequantRowColStrideParams.col) * sizeof(T) / 32U), 0};

    Rectangle perTokenRectangleParams{
        static_cast<uint32_t>(1),             // row
        static_cast<uint32_t>(perTokenCount), // col
        static_cast<uint32_t>(perTokenCount)  // columnStride
    };

    for (int64_t perTokenRowOffset = 0; perTokenRowOffset < oriRow;
         perTokenRowOffset += dequantRowColStrideParams.row) {
        int64_t perTokenInputOffset = perTokenRowOffset * dequantRowColStrideParams.stride;
        int64_t perTokenOutputOffset = perTokenRowOffset * dequantRowColStrideParams.col;
        // copy in
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(perTokenInputLocal, inputGm[perTokenInputOffset], perTokenCopyParams);
        DataCopy(perTokenScaleLocal, deqScaleQcQrWGm[perTokenInputOffset], perTokenCopyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // compute
        Dequant(perTokenComputeLocal, perTokenInputLocal, perTokenScaleLocal, deQuantScaleQcQrLocal,
                perTokenRectangleParams);
        AscendC::PipeBarrier<PIPE_V>();
        // cast
        Cast(perTokenOutputLocal, perTokenComputeLocal, RoundMode::CAST_RINT, perTokenCount);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // copy out
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[perTokenOutputOffset], perTokenOutputLocal, perTokenCount);
    }
}

/**
 * @brief CastPerTokenQc 用于对Qc做类型转换；按行做cast流程，给oriRow * col的数据做类型转换
 * @param outputGm 输出tensor
 * @param inputGm 输入tensor
 * @param shareTmpUb 临时buffer
 * @param castRowColStrideParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param oriRow 一共有多少行
*/
template <typename T, typename O>
__aicore__ inline void CastPerTokenQc(const GlobalTensor<O> &outputGm, const GlobalTensor<T> &inputGm,
                                      const LocalTensor<uint8_t> &shareTmpUb, Rectangle castRowColStrideParams,
                                      uint32_t oriRow)
{
    int64_t castQcCount = castRowColStrideParams.row * castRowColStrideParams.col;

    LocalTensor<T> castQcInputLocal = shareTmpUb.ReinterpretCast<T>(); // count * sizeof(T)
    LocalTensor<O> castQcOutputLocal =
        castQcInputLocal[castQcCount + 16].template ReinterpretCast<O>(); // count * sizeof(O)

    DataCopyParams castQcCopyParams{
        static_cast<uint16_t>(castRowColStrideParams.row),
        static_cast<uint16_t>(castRowColStrideParams.col * sizeof(T) / 32U),
        static_cast<uint16_t>((castRowColStrideParams.stride - castRowColStrideParams.col) * sizeof(T) / 32U), 0};

    for (int64_t castQcRowOffset = 0; castQcRowOffset < oriRow; castQcRowOffset += castRowColStrideParams.row) {
        int64_t castQcInputOffset = castRowColStrideParams.stride * castQcRowOffset;
        int64_t castQcOutputOffset = castRowColStrideParams.col * castQcRowOffset;
        // copy in
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(castQcInputLocal, inputGm[castQcInputOffset], castQcCopyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // cast
        Cast(castQcOutputLocal, castQcInputLocal, RoundMode::CAST_RINT, castQcCount);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // copy out
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[castQcOutputOffset], castQcOutputLocal, castQcCount);
    }
}

/**
   DequantSplitNQc 用于enableGroupComputeOpt场景
   场景特征：半量化，BS = 1, headsize = 8
 * @brief 用于对Qc做反量化；按N方向切分的dequant,理解为做一个head的
 * @param outputGm 输出位置
 * @param inputGm 输入tensor
 * @param deqScaleQcQrWGm deqScaleQcQr反量化系数，原shape[1,ND]
 * @param deQuantScaleQcQrLocal deQuantScaleQcQr反量化系数，原shape[BS,1]
 * @param shareTmpUb 临时buffer
 * @param dequantRowColStrideParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param oriCol 一列真实的长度
 * @param dstStride 目的数据行间的偏移
 * @param subBlockIdx_ 这是奇数核还是偶数核
 */
// 用于enableGroupComputeOpt场景
template <typename T, typename C, typename O>
__aicore__ inline void DequantSplitNQc(const GlobalTensor<O> &outputGm, const GlobalTensor<T> &inputGm,
                                       const GlobalTensor<C> &deqScaleQcQrWGm,
                                       const LocalTensor<C> &deQuantScaleQcQrLocal,
                                       const LocalTensor<uint8_t> &shareTmpUb, Rectangle dequantRowColStrideParams,
                                       uint32_t oriCol, uint32_t dstStride, uint32_t subBlockIdx_)
{
    int64_t splitQcCount = dequantRowColStrideParams.col * 1;
    LocalTensor<T> splitQcInputLocal = shareTmpUb.ReinterpretCast<T>(); // count * sizeof(T)
    LocalTensor<C> splitQcScaleLocal =
        splitQcInputLocal[splitQcCount + 16].template ReinterpretCast<C>(); // count * sizeof(C)
    LocalTensor<C> splitQcComputeLocal =
        splitQcScaleLocal[splitQcCount + 16].template ReinterpretCast<C>(); // count * sizeof(C)
    LocalTensor<O> splitQcOutputLocal =
        splitQcComputeLocal[splitQcCount + 16].template ReinterpretCast<O>(); // count * sizeof(O)

    DataCopyParams splitQcInputCopyParams{
        static_cast<uint16_t>(1), static_cast<uint16_t>(splitQcCount * sizeof(T) / 32U),
        static_cast<uint16_t>((dequantRowColStrideParams.stride - dequantRowColStrideParams.col) * sizeof(T) / 32U), 0};

    DataCopyParams splitQcOutputCopyParams{static_cast<uint16_t>(1),
                                           static_cast<uint16_t>(splitQcCount * sizeof(O) / 32U), 0,
                                           static_cast<uint16_t>(dstStride * sizeof(O) / 32U)};

    Rectangle splitQcRectangleParams{
        static_cast<uint32_t>(1),            // row
        static_cast<uint32_t>(splitQcCount), // col
        static_cast<uint32_t>(splitQcCount)  // columnStride
    };

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    int64_t splitQcScaleOffset = subBlockIdx_ * splitQcCount;
    DataCopy(splitQcScaleLocal, deqScaleQcQrWGm[splitQcScaleOffset], dequantRowColStrideParams.col);
    for (int64_t splitQcRowOffset = 0; splitQcRowOffset < dequantRowColStrideParams.row; splitQcRowOffset++) {
        int64_t splitQcInputOffset = splitQcRowOffset * oriCol + subBlockIdx_ * splitQcCount;
        int64_t splitQcOutputOffset = splitQcRowOffset * oriCol + subBlockIdx_ * splitQcCount;
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(splitQcInputLocal, inputGm[splitQcInputOffset], splitQcInputCopyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // compute
        Dequant(splitQcComputeLocal, splitQcInputLocal, splitQcScaleLocal, deQuantScaleQcQrLocal,
                splitQcRectangleParams);
        AscendC::PipeBarrier<PIPE_V>();
        // cast
        Cast(splitQcOutputLocal, splitQcComputeLocal, RoundMode::CAST_RINT, splitQcCount);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // copy out
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[splitQcOutputOffset], splitQcOutputLocal, splitQcCount);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

} // namespace MlaProlog
#endif
