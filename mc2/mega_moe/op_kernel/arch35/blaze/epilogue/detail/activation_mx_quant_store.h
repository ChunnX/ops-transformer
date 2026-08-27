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
 * \file activation_mx_quant_store.h
 * \brief MX quantization output store helpers.
 */

#ifndef ACTIVATION_MX_QUANT_STORE_H
#define ACTIVATION_MX_QUANT_STORE_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../common/mega_moe_constants.h"
#include "../../../common/mega_moe_utils.h"

namespace MegaMoeImpl {
namespace Detail {

template <typename OutputType>
__aicore__ inline void StoreQuantOutput(AscendC::GlobalTensor<int8_t> &dst, AscendC::LocalTensor<int8_t> &src,
                                        uint64_t blockCount, uint64_t offset, uint64_t n, uint64_t singleN)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount; // 128
    if constexpr (AscendC::IsSameType<OutputType, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<OutputType, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = singleN >> 1;
        ub2GmParams.dstStride = (n - singleN) >> 1;
        offset = offset >> 1;
    } else {
        uint64_t nDstUbAligned =
            Ops::Base::CeilAlign(static_cast<uint64_t>(singleN), static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
        ub2GmParams.blockLen = singleN; // 256
        ub2GmParams.srcStride = (nDstUbAligned - singleN) / AscendC::ONE_BLK_SIZE;
        ub2GmParams.dstStride = n - singleN;
    }
    AscendC::DataCopyPad(dst[offset], src, ub2GmParams);
}

__aicore__ inline void StoreQuantScaleCompact(AscendC::GlobalTensor<int8_t> &dst,
                                               AscendC::LocalTensor<int8_t> &src, uint64_t blockCount,
                                               uint64_t offset, uint64_t scaleN, uint64_t singleN)
{
    AscendC::DataCopyExtParams ub2GmParams{0, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN),
                                          static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                       MXFP_MULTI_BASE_SIZE;
    // scale layout in UB is already compact: (mSize, blockScaleN). Compact copy avoids (mSize*8)->(mSize,32).
    ub2GmParams.blockCount = blockCount; // 128
    ub2GmParams.blockLen = blockScaleN;  // 8
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = scaleN - blockScaleN;
    AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(dst[offset], src, ub2GmParams);
}

} // namespace Detail
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // ACTIVATION_MX_QUANT_STORE_H
