/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_REGBASE_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_REGBASE_H_

namespace MhcSandwichRegBase {

using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t VL_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);

constexpr AscendC::Reg::CastTrait CAST_B16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait CAST_FP32_TO_B16_NONE = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr AscendC::Reg::CastTrait CAST_FP32_TO_B16_RINT = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// 低精度输入按FP32寄存器宽度展开，避免先把数据搬入额外FP32临时区。
template <typename T>
__aicore__ inline void CastB16ToFp32(const LocalTensor<float> &dst, const LocalTensor<T> &src, uint32_t count)
{
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<float> dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Cast<float, T, CAST_B16_TO_FP32>(dstReg, srcReg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 保持原有BF16使用RINT、FP16使用NONE的量化边界。
template <typename T>
__aicore__ inline void CastFp32ToB16(const LocalTensor<T> &dst, const LocalTensor<float> &src, uint32_t count)
{
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ T *dstAddr = (__local_mem__ T *)dst.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg;
        RegTensor<T> dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Cast<T, float, CAST_FP32_TO_B16_RINT>(dstReg, srcReg, mask);
            } else {
                Cast<T, float, CAST_FP32_TO_B16_NONE>(dstReg, srcReg, mask);
            }
            DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

__aicore__ inline void Mul(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                           const LocalTensor<float> &src1, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *src0Addr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *src1Addr = (__local_mem__ float *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> src0Reg, src1Reg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(src0Reg, src0Addr + i * VL_FP32);
            DataCopy<float, LoadDist::DIST_NORM>(src1Reg, src1Addr + i * VL_FP32);
            AscendC::Reg::Mul(dstReg, src0Reg, src1Reg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

__aicore__ inline void Add(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                           const LocalTensor<float> &src1, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *src0Addr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *src1Addr = (__local_mem__ float *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> src0Reg, src1Reg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(src0Reg, src0Addr + i * VL_FP32);
            DataCopy<float, LoadDist::DIST_NORM>(src1Reg, src1Addr + i * VL_FP32);
            AscendC::Reg::Add(dstReg, src0Reg, src1Reg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

__aicore__ inline void Muls(const LocalTensor<float> &dst, const LocalTensor<float> &src, float scalar, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            AscendC::Reg::Muls(dstReg, srcReg, scalar, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

__aicore__ inline void Axpy(const LocalTensor<float> &dst, const LocalTensor<float> &src, float scalar, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg, dstReg, scalarReg;
        MaskReg mask;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        Duplicate(scalarReg, scalar, maskAll);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            DataCopy<float, LoadDist::DIST_NORM>(dstReg, dstAddr + i * VL_FP32);
            MulAddDst(dstReg, srcReg, scalarReg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// clang-format off
}
// clang-format on

#endif // MHC_SANDWICH_NORM_POST_PREONLY_REGBASE_H_
