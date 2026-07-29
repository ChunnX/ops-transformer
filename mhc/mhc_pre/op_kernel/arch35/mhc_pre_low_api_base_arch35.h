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
 * \file mhc_pre_low_api_base_arch35.h
 * \brief
 */

#ifndef MHC_PRE_LOW_API_BASE_ARCH35_H
#define MHC_PRE_LOW_API_BASE_ARCH35_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace MhcPre {

using namespace AscendC;

constexpr uint32_t MHC_PRE_LOW_API_BLOCK_SIZE = 32;
constexpr uint32_t MHC_PRE_LOW_API_C0_SIZE = 8;
constexpr uint32_t MHC_PRE_LOW_API_VL_FP32 = 64;

constexpr AscendC::MicroAPI::CastTrait MHC_PRE_LOW_API_CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__aicore__ inline uint32_t LowApiCeilDiv(uint32_t value, uint32_t div)
{
    return (value + div - 1) / div;
}

__aicore__ inline uint32_t LowApiAlign(uint32_t value, uint32_t align)
{
    return LowApiCeilDiv(value, align) * align;
}

template <typename T>
__aicore__ inline uint32_t LowApiRoundUp(uint32_t value)
{
    return LowApiAlign(value, MHC_PRE_LOW_API_BLOCK_SIZE / sizeof(T));
}

__aicore__ inline void MhcPreLowApiLoadBroadcast(AscendC::MicroAPI::RegTensor<float>& dst,
    __local_mem__ float* src, AscendC::MicroAPI::MaskReg mask, uint32_t offset)
{
    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + offset);
}

template <typename T>
__aicore__ inline void LowApiCopyIn(const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputLocal,
    uint16_t blockCount, uint32_t copyLen, uint32_t srcStride = 0)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = copyLen * sizeof(T);
    copyParams.srcStride = srcStride * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, inputGm, copyParams, padParams);
}

template <typename T>
__aicore__ inline void LowApiCopyToL1(const LocalTensor<T>& srcLocal, const LocalTensor<T>& dstLocal,
    const DataCopyParams& copyParams)
{
    DataCopy(dstLocal, srcLocal, copyParams);
}
template <typename T>
__aicore__ inline void LowApiCopyOut(const LocalTensor<T>& outputLocal, const GlobalTensor<T>& outputGm,
    uint16_t blockCount, uint32_t copyLen, uint32_t dstStride = 0, uint32_t srcStride = 0)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = copyLen * sizeof(T);
    copyParams.srcStride = srcStride * sizeof(T);
    copyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(outputGm, outputLocal, copyParams);
}

template <typename T>
__aicore__ inline void MhcPreVFCompactRows(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    uint16_t rowCount, uint32_t srcRowStride, uint32_t rowWidth)
{
    __ubuf__ T* dst = (__ubuf__ T*)dstLocal.GetPhyAddr();
    __ubuf__ T* src = (__ubuf__ T*)srcLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> data;
        for (uint16_t row = 0; row < rowCount; ++row) {
            MicroAPI::Load<T>(data, src + static_cast<uint32_t>(row) * srcRowStride);
            MicroAPI::Store<T>(dst + static_cast<uint32_t>(row) * rowWidth, data, rowWidth);
        }
    }
}


} // namespace MhcPre

#endif // MHC_PRE_LOW_API_BASE_ARCH35_H
