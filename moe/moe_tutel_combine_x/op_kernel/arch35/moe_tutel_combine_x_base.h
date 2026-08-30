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
 * \file moe_tutel_combine_x_base.h
 * \brief MoeTutelCombineX arch35 公共件：搬运封装 + 跨 k 加权累加的 VF 主体。
 *        VF 惯用法取自同仓 moe/moe_finalize_routing_v2/op_kernel/arch35/moe_finalize_routing_v2_base.h。
 */

#ifndef MOE_TUTEL_COMBINE_X_BASE_H_
#define MOE_TUTEL_COMBINE_X_BASE_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace MoeTutelCombineXRegbase {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// 把 UB 上的一段 T 数据装进 fp32 RegTensor；16bit 类型先 unpack 再升 fp32。
template <typename T>
__aicore__ inline void LoadAsFloat(RegTensor<float> &dst, __ubuf__ T *src, MaskReg preg, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src + srcOffset);
    } else {
        RegTensor<T> tmp;
        LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, CAST_TRAIT_B16_TO_B32>(dst, tmp, preg);
    }
}

// 把 UB 上的一个 fp32 标量广播成整个 RegTensor。
__aicore__ inline void LoadScalarBrc(RegTensor<float> &dst, __ubuf__ float *src, uint32_t srcOffset)
{
    LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + srcOffset);
}

/**
 * acc[0:hLen] += sum_{k<validK} gate[k] * rows[k * rowStride : k * rowStride + hLen]
 *
 * acc  : fp32 累加器，调用方已按 0 初始化，每行至少留出一个完整 fp32 VL。
 * rows : 压紧后的有效行（无效路由不占位），行跨距 rowStride 按 64 元素对齐。
 * gates: 压紧后的有效门控，fp32。
 * 累加顺序恒为 0 -> validK-1，与分块无关（INV-5）。
 */
template <typename T>
__aicore__ inline void AccumulateRows(const LocalTensor<float> &acc, const LocalTensor<T> &rows,
                                      const LocalTensor<float> &gates, uint16_t validK, uint32_t hLen,
                                      uint32_t rowStride)
{
    __ubuf__ float *accAddr = (__ubuf__ float *)acc.GetPhyAddr();
    __ubuf__ T *rowAddr = (__ubuf__ T *)rows.GetPhyAddr();
    __ubuf__ float *gateAddr = (__ubuf__ float *)gates.GetPhyAddr();
    uint16_t loopCount = Ops::Base::CeilDiv<uint32_t>(hLen, VL_FP32);

    __VEC_SCOPE__
    {
        RegTensor<float> accReg;
        RegTensor<float> rowReg;
        RegTensor<float> gateReg;
        MaskReg preg;
        for (uint16_t k = 0; k < validK; k++) {
            LoadScalarBrc(gateReg, gateAddr, k);
            uint32_t sreg = hLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                preg = UpdateMask<float>(sreg);
                LoadAlign(accReg, accAddr + i * VL_FP32);
                LoadAsFloat<T>(rowReg, rowAddr, preg, k * rowStride + i * VL_FP32);
                Mul(rowReg, rowReg, gateReg, preg);
                Add(accReg, accReg, rowReg, preg);
                StoreAlign(accAddr + i * VL_FP32, accReg, preg);
            }
            // 下一个 k 会重新读回同一段 acc，必须挡一次写后读。
            LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
    }
}

template <typename T>
__aicore__ inline void CopyInPad(const GlobalTensor<T> &src, const LocalTensor<T> &dst, uint16_t blockCount,
                                 uint32_t blockLenElem, int64_t srcStrideByte, int64_t dstStrideBlock)
{
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLenElem * sizeof(T);
    copyParams.srcStride = srcStrideByte;
    copyParams.dstStride = dstStrideBlock;
    DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOutPad(const LocalTensor<T> &src, const GlobalTensor<T> &dst, uint16_t blockCount,
                                  uint32_t blockLenElem, int64_t srcStrideBlock, int64_t dstStrideByte)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLenElem * sizeof(T);
    copyParams.srcStride = srcStrideBlock;
    copyParams.dstStride = dstStrideByte;
    DataCopyPad(dst, src, copyParams);
}
} // namespace MoeTutelCombineXRegbase

#endif // MOE_TUTEL_COMBINE_X_BASE_H_
