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
 * \file attenmask.h
 * \brief arch22 attention mask copy-in with compile-time layoutQ and isReconstructTemp parameters.
 *        Eliminates the runtime GS/SG branch in vector_common.h AttentionmaskCopyIn.
 *        S1_EQUAL1 falls back to the original runtime path.
 */
#ifndef ARCH22_ATTENMASK_H
#define ARCH22_ATTENMASK_H

#include "../vector_common.h"

namespace AttentionCommon {

// layoutQ and isReconstructTemp are compile-time parameters.
// isReconstructTemp=true means the caller has already performed the skip check externally,
// so no redundant check is needed inside this function.
template <typename T, typename U, fa_base_vector::LAYOUT_Q layoutQ,
          bool isReconstructTemp, bool ENABLE_TREE = false>
__aicore__ inline void AttentionmaskCopyIn(LocalTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr,
                                           LocalTensor<U> &tmpBuf, fa_base_vector::MaskInfo &info,
                                           bool isPre = false)
{
    if constexpr (layoutQ == fa_base_vector::GS) {
        fa_base_vector::AttentionmaskCopyInForGsLayout<T, U, ENABLE_TREE>(
            attenMaskUb, srcGmAddr, tmpBuf, info, isPre);
    } else if constexpr (layoutQ == fa_base_vector::SG) {
        fa_base_vector::AttentionmaskCopyInForSgLayout<T, U, ENABLE_TREE>(
            attenMaskUb, srcGmAddr, tmpBuf, info, isPre);
    } else {
        // S1_EQUAL1: fall back to the original runtime-dispatch path
        fa_base_vector::AttentionmaskCopyIn<T, U, ENABLE_TREE>(
            attenMaskUb, srcGmAddr, tmpBuf, info, isPre);
    }
}

} // namespace AttentionCommon

#endif // ARCH22_ATTENMASK_H
