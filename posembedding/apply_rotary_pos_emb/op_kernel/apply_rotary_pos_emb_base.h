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
 * \file apply_rotary_pos_emb_base.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_BASE_H
#define APPLY_ROTARY_POS_EMB_BASE_H

#include "kernel_operator.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbBase {
public:
    __aicore__ inline ApplyRotaryPosEmbBase(){};

protected:
    template <typename T1, typename T2>
    __aicore__ inline void LocalTensor2NewTensor(LocalTensor<T1>& tensor_new, const LocalTensor<T2>& tensor_old)
    {
        tensor_new = tensor_old.template ReinterpretCast<T1>();
    };

    // [泛化SBND需求] 未对齐D时，用DataCopyPad把GM一个head的左/右半搬到UB内两个halfDimAlign位置
    __aicore__ inline void CopyInHalfByPad(const LocalTensor<T>& ubTensor, const GlobalTensor<T>& gmTensor,
                                           const uint64_t gmHeadOffset, const int64_t halfNumBytes,
                                           const int64_t halfPadNumBytes, const int64_t dtypeSize)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        // 左半：GM[0..halfNumBytes) -> UB[0..halfPadNumBytes)
        DataCopyExtParams leftParams{1, static_cast<uint16_t>(halfNumBytes), 0, 0, 0};
        DataCopyPad(ubTensor, gmTensor[gmHeadOffset], leftParams, padParams);
        // 右半：GM[halfNumBytes..lastDimBytes) -> UB[halfPadNumBytes..lastDimPadBytes)
        DataCopyExtParams rightParams{1, static_cast<uint16_t>(halfNumBytes), 0, 0, 0};
        DataCopyPad(ubTensor[halfPadNumBytes / dtypeSize], gmTensor[gmHeadOffset + halfNumBytes / dtypeSize],
                    rightParams, padParams);
    }

    // [泛化SBND需求] 未对齐D时，批量搬运多个head：每个head分左/右半
    __aicore__ inline void CopyInHeadsByPad(const LocalTensor<T>& ubTensor, const GlobalTensor<T>& gmTensor,
                                            const uint64_t gmStartOffset, const int64_t headCount,
                                            const int64_t lastDim, const int64_t lastDimAlign,
                                            const int64_t halfDimBytes, const int64_t halfDimPadBytes,
                                            const int64_t dtypeSize)
    {
        for (int64_t n = 0; n < headCount; n++) {
            CopyInHalfByPad(ubTensor[n * lastDimAlign], gmTensor, gmStartOffset + n * lastDim,
                            halfDimBytes, halfDimPadBytes, dtypeSize);
        }
    }

    // [泛化SBND需求] 未对齐D时，用DataCopyPad把UB内一个head的左/右半写回GM紧凑位置
    __aicore__ inline void CopyOutHalfByPad(const GlobalTensor<T>& gmTensor, const LocalTensor<T>& ubTensor,
                                            const uint64_t gmHeadOffset, const int64_t halfNumBytes,
                                            const int64_t halfPadNumBytes, const int64_t dtypeSize)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams leftParams{1, static_cast<uint16_t>(halfNumBytes), 0, 0, 0};
        DataCopyPad(gmTensor[gmHeadOffset], ubTensor, leftParams, padParams);
        DataCopyExtParams rightParams{1, static_cast<uint16_t>(halfNumBytes), 0, 0, 0};
        DataCopyPad(gmTensor[gmHeadOffset + halfNumBytes / dtypeSize],
                    ubTensor[halfPadNumBytes / dtypeSize], rightParams, padParams);
    }

    // [泛化SBND需求] 未对齐D时，批量写回多个head：每个head分左/右半
    __aicore__ inline void CopyOutHeadsByPad(const GlobalTensor<T>& gmTensor, const LocalTensor<T>& ubTensor,
                                             const uint64_t gmStartOffset, const int64_t headCount,
                                             const int64_t lastDim, const int64_t lastDimAlign,
                                             const int64_t halfDimBytes, const int64_t halfDimPadBytes,
                                             const int64_t dtypeSize)
    {
        for (int64_t n = 0; n < headCount; n++) {
            CopyOutHalfByPad(gmTensor, ubTensor[n * lastDimAlign], gmStartOffset + n * lastDim,
                             halfDimBytes, halfDimPadBytes, dtypeSize);
        }
    }
};
} // namespace ApplyRotaryPosEmb

#endif // APPLY_ROTARY_POS_EMB_BASE_H
