/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_KERNEL_REGBASE_H_
#define MHC_NORM_POST_PREONLY_KERNEL_REGBASE_H_

#include "mhc_norm_post_preonly_common.h"

namespace MhcNormRegBase {

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

constexpr AscendC::Reg::DivSpecificMode DIV_ZEROING_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
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

// 在一个VF内恢复相邻两路低精度数据，减少相同分块上的VF启动次数。
template <typename T>
__aicore__ inline void CastB16PairToFp32(const LocalTensor<float> &dst0, const LocalTensor<float> &dst1,
                                         const LocalTensor<T> &src0, const LocalTensor<T> &src1, uint32_t count)
{
    __local_mem__ T *src0Addr = (__local_mem__ T *)src0.GetPhyAddr();
    __local_mem__ T *src1Addr = (__local_mem__ T *)src1.GetPhyAddr();
    __local_mem__ float *dst0Addr = (__local_mem__ float *)dst0.GetPhyAddr();
    __local_mem__ float *dst1Addr = (__local_mem__ float *)dst1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> src0Reg, src1Reg;
        RegTensor<float> dst0Reg, dst1Reg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(src0Reg, src0Addr + i * VL_FP32);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(src1Reg, src1Addr + i * VL_FP32);
            Cast<float, T, CAST_B16_TO_FP32>(dst0Reg, src0Reg, mask);
            Cast<float, T, CAST_B16_TO_FP32>(dst1Reg, src1Reg, mask);
            DataCopy(dst0Addr + i * VL_FP32, dst0Reg, mask);
            DataCopy(dst1Addr + i * VL_FP32, dst1Reg, mask);
        }
    }
}

// 在一个VF内完成低精度恢复和平方，直接生成平方和归约输入。
template <typename T>
__aicore__ inline void CastB16ToFp32Square(const LocalTensor<float> &dst, const LocalTensor<T> &src, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<float> srcFp32Reg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            AscendC::Reg::Mul(dstReg, srcFp32Reg, srcFp32Reg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内恢复两路低精度数据并计算平方和，避免中间FP32结果写回UB。
template <typename T>
__aicore__ inline void PairSquareB16ToFp32(const LocalTensor<float> &dst, const LocalTensor<T> &src0,
                                           const LocalTensor<T> &src1, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *src0Addr = (__local_mem__ T *)src0.GetPhyAddr();
    __local_mem__ T *src1Addr = (__local_mem__ T *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> src0Reg, src1Reg;
        RegTensor<float> src0Fp32Reg, src1Fp32Reg, src0SqReg, src1SqReg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(src0Reg, src0Addr + i * VL_FP32);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(src1Reg, src1Addr + i * VL_FP32);
            Cast<float, T, CAST_B16_TO_FP32>(src0Fp32Reg, src0Reg, mask);
            Cast<float, T, CAST_B16_TO_FP32>(src1Fp32Reg, src1Reg, mask);
            AscendC::Reg::Mul(src0SqReg, src0Fp32Reg, src0Fp32Reg, mask);
            AscendC::Reg::Mul(src1SqReg, src1Fp32Reg, src1Fp32Reg, mask);
            AscendC::Reg::Add(dstReg, src0SqReg, src1SqReg, mask);
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

// 在一个VF内量化相邻两路FP32数据，保持BF16和FP16各自原有舍入模式。
template <typename T>
__aicore__ inline void CastFp32PairToB16(const LocalTensor<T> &dst0, const LocalTensor<T> &dst1,
                                         const LocalTensor<float> &src0, const LocalTensor<float> &src1, uint32_t count)
{
    __local_mem__ float *src0Addr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *src1Addr = (__local_mem__ float *)src1.GetPhyAddr();
    __local_mem__ T *dst0Addr = (__local_mem__ T *)dst0.GetPhyAddr();
    __local_mem__ T *dst1Addr = (__local_mem__ T *)dst1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> src0Reg, src1Reg;
        RegTensor<T> dst0Reg, dst1Reg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(src0Reg, src0Addr + i * VL_FP32);
            DataCopy<float, LoadDist::DIST_NORM>(src1Reg, src1Addr + i * VL_FP32);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Cast<T, float, CAST_FP32_TO_B16_RINT>(dst0Reg, src0Reg, mask);
                Cast<T, float, CAST_FP32_TO_B16_RINT>(dst1Reg, src1Reg, mask);
            } else {
                Cast<T, float, CAST_FP32_TO_B16_NONE>(dst0Reg, src0Reg, mask);
                Cast<T, float, CAST_FP32_TO_B16_NONE>(dst1Reg, src1Reg, mask);
            }
            DataCopy<T, StoreDist::DIST_PACK_B32>(dst0Addr + i * VL_FP32, dst0Reg, mask);
            DataCopy<T, StoreDist::DIST_PACK_B32>(dst1Addr + i * VL_FP32, dst1Reg, mask);
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

// 从UB广播标量权重，单独验证R6拆分为多个VF后的性能。
__aicore__ inline void MulsWithUbScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                        const LocalTensor<float> &scalars, uint32_t scalarIndex, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalars.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg, dstReg, scalarReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr + scalarIndex);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            AscendC::Reg::Mul(dstReg, srcReg, scalarReg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 单个head在一个VF内完成低精度读取、FP32转换和门控乘法，避免生成中间FP32张量。
template <typename T>
__aicore__ inline void MulsB16WithUbScalar(const LocalTensor<float> &dst, const LocalTensor<T> &src,
                                           const LocalTensor<float> &scalars, uint32_t scalarIndex, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalars.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<float> srcFp32Reg, dstReg, scalarReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr + scalarIndex);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            AscendC::Reg::Mul(dstReg, srcFp32Reg, scalarReg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在寄存器内依次完成标量缩放和gamma缩放，只把最终结果写回UB。
__aicore__ inline void ScaleAndMulGamma(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                        const LocalTensor<float> &gamma, float scale, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *gammaAddr = (__local_mem__ float *)gamma.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg, gammaReg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            DataCopy<float, LoadDist::DIST_NORM>(gammaReg, gammaAddr + i * VL_FP32);
            AscendC::Reg::Muls(dstReg, srcReg, scale, mask);
            AscendC::Reg::Mul(dstReg, dstReg, gammaReg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内完成低精度恢复、标量缩放和gamma缩放，只写回最终FP32结果。
template <typename T>
__aicore__ inline void CastB16ScaleAndMulGamma(const LocalTensor<float> &dst, const LocalTensor<T> &src,
                                               const LocalTensor<float> &gamma, float scale, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *gammaAddr = (__local_mem__ float *)gamma.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<float> srcFp32Reg, gammaReg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            DataCopy<float, LoadDist::DIST_NORM>(gammaReg, gammaAddr + i * VL_FP32);
            AscendC::Reg::Muls(dstReg, srcFp32Reg, scale, mask);
            AscendC::Reg::Mul(dstReg, dstReg, gammaReg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// R3在一个VF内完成恢复与缩放后直接写回低精度缓存，保留第二次量化边界。
template <typename T>
__aicore__ inline void CastB16ScaleAndMulGammaToB16(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                    const LocalTensor<float> &gamma, float scale, uint32_t count)
{
    __local_mem__ T *dstAddr = (__local_mem__ T *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *gammaAddr = (__local_mem__ float *)gamma.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg, dstReg;
        RegTensor<float> srcFp32Reg, gammaReg, scaledReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            DataCopy<float, LoadDist::DIST_NORM>(gammaReg, gammaAddr + i * VL_FP32);
            AscendC::Reg::Muls(scaledReg, srcFp32Reg, scale, mask);
            AscendC::Reg::Mul(scaledReg, scaledReg, gammaReg, mask);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Cast<T, float, CAST_FP32_TO_B16_RINT>(dstReg, scaledReg, mask);
            } else {
                Cast<T, float, CAST_FP32_TO_B16_NONE>(dstReg, scaledReg, mask);
            }
            DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 将sigmoid和epsilon补偿合并到同一个VF，输入保存-z，输出保存sigmoid(z)+epsilon。
__aicore__ inline void SigmoidAddEps(const LocalTensor<float> &dst, const LocalTensor<float> &src, float epsilon,
                                     uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg, expReg, denomReg, oneReg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            AscendC::Reg::Exp(expReg, srcReg, mask);
            AscendC::Reg::Adds(denomReg, expReg, 1.0f, mask);
            AscendC::Reg::Duplicate(oneReg, 1.0f, mask);
            AscendC::Reg::Div<float, &DIV_ZEROING_MODE>(dstReg, oneReg, denomReg, mask);
            AscendC::Reg::Adds(dstReg, dstReg, epsilon, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 一个VF内完成单个head的主分支、残差转换和四路累加，保持残差索引与累加顺序不变。
template <typename T>
__aicore__ inline void ComputePostHead(const LocalTensor<float> &dst, const LocalTensor<float> &hOut,
                                       const LocalTensor<T> &residual, const LocalTensor<float> &combWeights,
                                       uint32_t rowStride, uint32_t headIndex, float postWeight, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *hOutAddr = (__local_mem__ float *)hOut.GetPhyAddr();
    __local_mem__ T *residualAddr = (__local_mem__ T *)residual.GetPhyAddr();
    __local_mem__ float *combAddr = (__local_mem__ float *)combWeights.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> residualB16Reg;
        RegTensor<float> hOutReg, residualReg, weightReg, dstReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(hOutReg, hOutAddr + i * VL_FP32);
            AscendC::Reg::Muls(dstReg, hOutReg, postWeight, mask);
            for (uint16_t j = 0; j < static_cast<uint16_t>(N_MAX); ++j) {
                const uint32_t row = static_cast<uint32_t>(j);
                DataCopy<T, LoadDist::DIST_UNPACK_B16>(residualB16Reg, residualAddr + row * rowStride + i * VL_FP32);
                Cast<float, T, CAST_B16_TO_FP32>(residualReg, residualB16Reg, mask);
                DataCopy<float, LoadDist::DIST_BRC_B32>(weightReg, combAddr + row * N_MAX + headIndex);
                AscendC::Reg::MulAddDst(dstReg, residualReg, weightReg, mask);
            }
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 一个VF同时计算相邻两个head，共用h_out与residual的寄存器加载，并保持各自四路累加顺序不变。
template <typename T>
__aicore__ inline void ComputePostHeadPair(const LocalTensor<T> &dst0, const LocalTensor<T> &dst1,
                                           const LocalTensor<float> &hOut, const LocalTensor<T> &residual,
                                           const LocalTensor<float> &combWeights, uint32_t rowStride,
                                           uint32_t headIndex, float postWeight0, float postWeight1, uint32_t count)
{
    __local_mem__ T *dst0Addr = (__local_mem__ T *)dst0.GetPhyAddr();
    __local_mem__ T *dst1Addr = (__local_mem__ T *)dst1.GetPhyAddr();
    __local_mem__ float *hOutAddr = (__local_mem__ float *)hOut.GetPhyAddr();
    __local_mem__ T *residualAddr = (__local_mem__ T *)residual.GetPhyAddr();
    __local_mem__ float *combAddr = (__local_mem__ float *)combWeights.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> residualB16Reg, dst0B16Reg, dst1B16Reg;
        RegTensor<float> hOutReg, residualReg, weight0Reg, weight1Reg, dst0Reg, dst1Reg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(hOutReg, hOutAddr + i * VL_FP32);
            AscendC::Reg::Muls(dst0Reg, hOutReg, postWeight0, mask);
            AscendC::Reg::Muls(dst1Reg, hOutReg, postWeight1, mask);
            for (uint16_t j = 0; j < static_cast<uint16_t>(N_MAX); ++j) {
                const uint32_t row = static_cast<uint32_t>(j);
                DataCopy<T, LoadDist::DIST_UNPACK_B16>(residualB16Reg, residualAddr + row * rowStride + i * VL_FP32);
                Cast<float, T, CAST_B16_TO_FP32>(residualReg, residualB16Reg, mask);
                DataCopy<float, LoadDist::DIST_BRC_B32>(weight0Reg, combAddr + row * N_MAX + headIndex);
                DataCopy<float, LoadDist::DIST_BRC_B32>(weight1Reg, combAddr + row * N_MAX + headIndex + 1U);
                AscendC::Reg::MulAddDst(dst0Reg, residualReg, weight0Reg, mask);
                AscendC::Reg::MulAddDst(dst1Reg, residualReg, weight1Reg, mask);
            }
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Cast<T, float, CAST_FP32_TO_B16_RINT>(dst0B16Reg, dst0Reg, mask);
                Cast<T, float, CAST_FP32_TO_B16_RINT>(dst1B16Reg, dst1Reg, mask);
            } else {
                Cast<T, float, CAST_FP32_TO_B16_NONE>(dst0B16Reg, dst0Reg, mask);
                Cast<T, float, CAST_FP32_TO_B16_NONE>(dst1B16Reg, dst1Reg, mask);
            }
            DataCopy<T, StoreDist::DIST_PACK_B32>(dst0Addr + i * VL_FP32, dst0B16Reg, mask);
            DataCopy<T, StoreDist::DIST_PACK_B32>(dst1Addr + i * VL_FP32, dst1B16Reg, mask);
        }
    }
}

// R6在一个VF内完成四路低精度读取、FP32转换和门控累加，不生成四行FP32中间张量。
// 四个gate权重对整个D循环不变，外提到循环外广播一次，循环内显式展开四路累加，
// 避免每轮重复广播（参考mhc_pre::VFDoV1ProcessHInRows）。
template <typename T>
__aicore__ inline void GateWeightedSum(const LocalTensor<float> &dst, const LocalTensor<T> &src, uint32_t headStride,
                                       const LocalTensor<float> &gate, uint32_t count)
{
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *gateAddr = (__local_mem__ float *)gate.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<float> srcFp32Reg, accReg;
        RegTensor<float> g0Reg, g1Reg, g2Reg, g3Reg;
        MaskReg mask;
        // 四个gate权重只依赖head，对整个D循环不变，广播一次。
        DataCopy<float, LoadDist::DIST_BRC_B32>(g0Reg, gateAddr + 0U);
        DataCopy<float, LoadDist::DIST_BRC_B32>(g1Reg, gateAddr + 1U);
        DataCopy<float, LoadDist::DIST_BRC_B32>(g2Reg, gateAddr + 2U);
        DataCopy<float, LoadDist::DIST_BRC_B32>(g3Reg, gateAddr + 3U);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 0U * headStride + offset);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            AscendC::Reg::Mul(accReg, srcFp32Reg, g0Reg, mask);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 1U * headStride + offset);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            AscendC::Reg::MulAddDst(accReg, srcFp32Reg, g1Reg, mask);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 2U * headStride + offset);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            AscendC::Reg::MulAddDst(accReg, srcFp32Reg, g2Reg, mask);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 3U * headStride + offset);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            AscendC::Reg::MulAddDst(accReg, srcFp32Reg, g3Reg, mask);
            DataCopy(dstAddr + offset, accReg, mask);
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

// 从UB广播标量权重，并在寄存器内累加到目标张量。
__aicore__ inline void AxpyWithUbScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                        const LocalTensor<float> &scalars, uint32_t scalarIndex, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalars.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg, dstReg, scalarReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr + scalarIndex);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            DataCopy<float, LoadDist::DIST_NORM>(dstReg, dstAddr + i * VL_FP32);
            AscendC::Reg::MulAddDst(dstReg, srcReg, scalarReg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 单个head在一个VF内完成低精度读取、FP32转换和门控累加。
template <typename T>
__aicore__ inline void AxpyB16WithUbScalar(const LocalTensor<float> &dst, const LocalTensor<T> &src,
                                           const LocalTensor<float> &scalars, uint32_t scalarIndex, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalars.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<float> srcFp32Reg, dstReg, scalarReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr + scalarIndex);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            DataCopy<float, LoadDist::DIST_NORM>(dstReg, dstAddr + i * VL_FP32);
            AscendC::Reg::MulAddDst(dstReg, srcFp32Reg, scalarReg, mask);
            DataCopy(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// clang-format off
}

#endif // MHC_NORM_POST_PREONLY_KERNEL_REGBASE_H_
