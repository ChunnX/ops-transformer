/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#ifndef MHC_NORM_POST_PREONLY_KERNEL_REGBASE_H_
#define MHC_NORM_POST_PREONLY_KERNEL_REGBASE_H_

namespace MhcNormRegBase {

using namespace AscendC;

constexpr uint32_t VL_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);

constexpr Reg::CastTrait CAST_B16_TO_FP32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr Reg::CastTrait CAST_FP32_TO_B16_NONE = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr Reg::CastTrait CAST_FP32_TO_B16_RINT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr Reg::DivSpecificMode DIV_ZEROING_MODE = {
    Reg::MaskMergeMode::ZEROING,
    true,
};

// 低精度输入按FP32寄存器宽度展开，避免先把数据搬入额外FP32临时区。
template <typename T>
__aicore__ inline void VFCastB16ToFp32(const LocalTensor<float> &dst, const LocalTensor<T> &src, uint32_t count)
{
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg;
        Reg::RegTensor<float> dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(dstReg, srcReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内恢复相邻两路低精度数据，减少相同分块上的VF启动次数。
template <typename T>
__aicore__ inline void VFCastB16PairToFp32(const LocalTensor<float> &dst0, const LocalTensor<float> &dst1,
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
        Reg::RegTensor<T> src0Reg, src1Reg;
        Reg::RegTensor<float> dst0Reg, dst1Reg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(src0Reg, src0Addr + i * VL_FP32);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(src1Reg, src1Addr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(dst0Reg, src0Reg, mask);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(dst1Reg, src1Reg, mask);
            Reg::StoreAlign(dst0Addr + i * VL_FP32, dst0Reg, mask);
            Reg::StoreAlign(dst1Addr + i * VL_FP32, dst1Reg, mask);
        }
    }
}

// 在一个VF内完成低精度恢复和平方，直接生成平方和归约输入。
template <typename T>
__aicore__ inline void VFCastB16ToFp32Square(const LocalTensor<float> &dst, const LocalTensor<T> &src, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg;
        Reg::RegTensor<float> srcFp32Reg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::Mul(dstReg, srcFp32Reg, srcFp32Reg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内恢复两路低精度数据并计算平方和，避免中间FP32结果写回UB。
template <typename T>
__aicore__ inline void VFPairSquareB16ToFp32(const LocalTensor<float> &dst, const LocalTensor<T> &src0,
                                             const LocalTensor<T> &src1, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *src0Addr = (__local_mem__ T *)src0.GetPhyAddr();
    __local_mem__ T *src1Addr = (__local_mem__ T *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> src0Reg, src1Reg;
        Reg::RegTensor<float> src0Fp32Reg, src1Fp32Reg, src0SqReg, src1SqReg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(src0Reg, src0Addr + i * VL_FP32);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(src1Reg, src1Addr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(src0Fp32Reg, src0Reg, mask);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(src1Fp32Reg, src1Reg, mask);
            Reg::Mul(src0SqReg, src0Fp32Reg, src0Fp32Reg, mask);
            Reg::Mul(src1SqReg, src1Fp32Reg, src1Fp32Reg, mask);
            Reg::Add(dstReg, src0SqReg, src1SqReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 保持原有BF16使用RINT、FP16使用NONE的量化边界。
template <typename T>
__aicore__ inline void VFCastFp32ToB16(const LocalTensor<T> &dst, const LocalTensor<float> &src, uint32_t count)
{
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ T *dstAddr = (__local_mem__ T *)dst.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcReg;
        Reg::RegTensor<T> dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(dstReg, srcReg, mask);
            } else {
                Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(dstReg, srcReg, mask);
            }
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内量化相邻两路FP32数据，保持BF16和FP16各自原有舍入模式。
template <typename T>
__aicore__ inline void VFCastFp32PairToB16(const LocalTensor<T> &dst0, const LocalTensor<T> &dst1,
                                           const LocalTensor<float> &src0, const LocalTensor<float> &src1,
                                           uint32_t count)
{
    __local_mem__ float *src0Addr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *src1Addr = (__local_mem__ float *)src1.GetPhyAddr();
    __local_mem__ T *dst0Addr = (__local_mem__ T *)dst0.GetPhyAddr();
    __local_mem__ T *dst1Addr = (__local_mem__ T *)dst1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> src0Reg, src1Reg;
        Reg::RegTensor<T> dst0Reg, dst1Reg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src0Reg, src0Addr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src1Reg, src1Addr + i * VL_FP32);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(dst0Reg, src0Reg, mask);
                Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(dst1Reg, src1Reg, mask);
            } else {
                Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(dst0Reg, src0Reg, mask);
                Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(dst1Reg, src1Reg, mask);
            }
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst0Addr + i * VL_FP32, dst0Reg, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst1Addr + i * VL_FP32, dst1Reg, mask);
        }
    }
}

__aicore__ inline void VFMul(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                             const LocalTensor<float> &src1, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *src0Addr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *src1Addr = (__local_mem__ float *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> src0Reg, src1Reg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src0Reg, src0Addr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src1Reg, src1Addr + i * VL_FP32);
            Reg::Mul(dstReg, src0Reg, src1Reg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

__aicore__ inline void VFAdd(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                             const LocalTensor<float> &src1, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *src0Addr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *src1Addr = (__local_mem__ float *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> src0Reg, src1Reg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src0Reg, src0Addr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src1Reg, src1Addr + i * VL_FP32);
            Reg::Add(dstReg, src0Reg, src1Reg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内完成两路平方并求和，避免两个中间结果写回UB后再读出相加。
template <bool ACCUMULATE = false>
__aicore__ inline void VFPairSquareAdd(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                       const LocalTensor<float> &src1, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *src0Addr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *src1Addr = (__local_mem__ float *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> src0Reg, src1Reg, src0SqReg, src1SqReg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src0Reg, src0Addr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src1Reg, src1Addr + i * VL_FP32);
            Reg::Mul(src0SqReg, src0Reg, src0Reg, mask);
            Reg::Mul(src1SqReg, src1Reg, src1Reg, mask);
            Reg::Add(dstReg, src0SqReg, src1SqReg, mask);
            if constexpr (ACCUMULATE) {
                Reg::RegTensor<float> previous;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(previous, dstAddr + i * VL_FP32);
                Reg::Add(dstReg, previous, dstReg, mask);
            }
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内完成两路乘法并求和，用于相邻head对同一输出行的点积部分和。
template <bool ACCUMULATE = false>
__aicore__ inline void VFPairMulAdd(const LocalTensor<float> &dst, const LocalTensor<float> &src00,
                                    const LocalTensor<float> &src01, const LocalTensor<float> &src10,
                                    const LocalTensor<float> &src11, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *src00Addr = (__local_mem__ float *)src00.GetPhyAddr();
    __local_mem__ float *src01Addr = (__local_mem__ float *)src01.GetPhyAddr();
    __local_mem__ float *src10Addr = (__local_mem__ float *)src10.GetPhyAddr();
    __local_mem__ float *src11Addr = (__local_mem__ float *)src11.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> src00Reg, src01Reg, src10Reg, src11Reg, product0Reg, product1Reg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src00Reg, src00Addr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src01Reg, src01Addr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src10Reg, src10Addr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src11Reg, src11Addr + i * VL_FP32);
            Reg::Mul(product0Reg, src00Reg, src01Reg, mask);
            Reg::Mul(product1Reg, src10Reg, src11Reg, mask);
            Reg::Add(dstReg, product0Reg, product1Reg, mask);
            if constexpr (ACCUMULATE) {
                Reg::RegTensor<float> previous;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(previous, dstAddr + i * VL_FP32);
                Reg::Add(dstReg, previous, dstReg, mask);
            }
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 完整分块在寄存器内跨向量累加，每行仅向UB写回一个向量供最终归约。
template <bool ACCUMULATE = false>
__aicore__ inline void VFPairMulAddCompact(const LocalTensor<float> &dst, const LocalTensor<float> &src00,
                                            const LocalTensor<float> &src01, const LocalTensor<float> &src10,
                                            const LocalTensor<float> &src11, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *a = (__local_mem__ float *)src00.GetPhyAddr();
    __local_mem__ float *b = (__local_mem__ float *)src01.GetPhyAddr();
    __local_mem__ float *c = (__local_mem__ float *)src10.GetPhyAddr();
    __local_mem__ float *d = (__local_mem__ float *)src11.GetPhyAddr();
    const uint16_t loops = static_cast<uint16_t>(count / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> ar, br, cr, dr, p0, p1, pair, acc;
        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Duplicate(acc, 0.0f, mask);
        for (uint16_t i = 0; i < loops; ++i) {
            const uint32_t off = static_cast<uint32_t>(i) * VL_FP32;
            Reg::LoadAlign(ar, a + off);
            Reg::LoadAlign(br, b + off);
            Reg::LoadAlign(cr, c + off);
            Reg::LoadAlign(dr, d + off);
            Reg::Mul(p0, ar, br, mask);
            Reg::Mul(p1, cr, dr, mask);
            Reg::Add(pair, p0, p1, mask);
            Reg::Add(acc, acc, pair, mask);
        }
        if constexpr (ACCUMULATE) {
            Reg::RegTensor<float> previous;
            Reg::LoadAlign(previous, dstAddr);
            Reg::Add(acc, previous, acc, mask);
        }
        Reg::StoreAlign(dstAddr, acc, mask);
    }
}

__aicore__ inline void VFMuls(const LocalTensor<float> &dst, const LocalTensor<float> &src, float scalar,
                              uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcReg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            Reg::Muls(dstReg, srcReg, scalar, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 从UB广播标量权重，单独验证门控融合拆分为多个VF后的性能。
__aicore__ inline void VFMulsWithUbScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                          const LocalTensor<float> &scalars, uint32_t scalarIndex, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalars.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcReg, dstReg, scalarReg;
        Reg::MaskReg mask;
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr + scalarIndex);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            Reg::Mul(dstReg, srcReg, scalarReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 单个head在一个VF内完成低精度读取、FP32转换和门控乘法，避免生成中间FP32张量。
template <typename T>
__aicore__ inline void VFMulsB16WithUbScalar(const LocalTensor<float> &dst, const LocalTensor<T> &src,
                                             const LocalTensor<float> &scalars, uint32_t scalarIndex, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalars.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg;
        Reg::RegTensor<float> srcFp32Reg, dstReg, scalarReg;
        Reg::MaskReg mask;
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr + scalarIndex);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::Mul(dstReg, srcFp32Reg, scalarReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在寄存器内依次完成标量缩放和gamma缩放，只把最终结果写回UB。
__aicore__ inline void VFScaleAndMulGamma(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                          const LocalTensor<float> &gamma, float scale, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *gammaAddr = (__local_mem__ float *)gamma.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcReg, gammaReg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(gammaReg, gammaAddr + i * VL_FP32);
            Reg::Muls(dstReg, srcReg, scale, mask);
            Reg::Mul(dstReg, dstReg, gammaReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内完成低精度恢复、标量缩放和gamma缩放，只写回最终FP32结果。
template <typename T>
__aicore__ inline void VFCastB16ScaleAndMulGamma(const LocalTensor<float> &dst, const LocalTensor<T> &src,
                                                 const LocalTensor<float> &gamma, float scale, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *gammaAddr = (__local_mem__ float *)gamma.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg;
        Reg::RegTensor<float> srcFp32Reg, gammaReg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(gammaReg, gammaAddr + i * VL_FP32);
            Reg::Muls(dstReg, srcFp32Reg, scale, mask);
            Reg::Mul(dstReg, dstReg, gammaReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 在一个VF内完成低精度恢复、输入归一化缩放、gamma缩放以及规定的低精度往返边界。
template <typename T>
__aicore__ inline void VFCastB16ScaleGammaQuantizeToFp32(const LocalTensor<float> &dst, const LocalTensor<T> &src,
                                                         const LocalTensor<float> &gamma, float scale, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *gammaAddr = (__local_mem__ float *)gamma.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg, quantizedReg;
        Reg::RegTensor<float> srcFp32Reg, gammaReg, scaledReg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(gammaReg, gammaAddr + i * VL_FP32);
            Reg::Muls(scaledReg, srcFp32Reg, scale, mask);
            Reg::Mul(scaledReg, scaledReg, gammaReg, mask);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(quantizedReg, scaledReg, mask);
            } else {
                Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(quantizedReg, scaledReg, mask);
            }
            Reg::Cast<float, T, CAST_B16_TO_FP32>(dstReg, quantizedReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 中间归一化在一个VF内完成恢复与缩放后直接写回低精度缓存，保留第二次量化边界。
template <typename T>
__aicore__ inline void VFCastB16ScaleAndMulGammaToB16(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                      const LocalTensor<float> &gamma, float scale, uint32_t count)
{
    __local_mem__ T *dstAddr = (__local_mem__ T *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *gammaAddr = (__local_mem__ float *)gamma.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg, dstReg;
        Reg::RegTensor<float> srcFp32Reg, gammaReg, scaledReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(gammaReg, gammaAddr + i * VL_FP32);
            Reg::Muls(scaledReg, srcFp32Reg, scale, mask);
            Reg::Mul(scaledReg, scaledReg, gammaReg, mask);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(dstReg, scaledReg, mask);
            } else {
                Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(dstReg, scaledReg, mask);
            }
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 将sigmoid和epsilon补偿合并到同一个VF，输入保存-z，输出保存sigmoid(z)+epsilon。
__aicore__ inline void VFSigmoidAddEps(const LocalTensor<float> &dst, const LocalTensor<float> &src, float epsilon,
                                       uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcReg, expReg, denomReg, oneReg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            Reg::Exp(expReg, srcReg, mask);
            Reg::Adds(denomReg, expReg, 1.0f, mask);
            Reg::Duplicate(oneReg, 1.0f, mask);
            Reg::Div<float, &DIV_ZEROING_MODE>(dstReg, oneReg, denomReg, mask);
            Reg::Adds(dstReg, dstReg, epsilon, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 一个VF内完成单个head的主分支、残差转换和四路累加，保持残差索引与累加顺序不变。
template <typename T>
__aicore__ inline void VFComputePostHead(const LocalTensor<float> &dst, const LocalTensor<float> &hOut,
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
        Reg::RegTensor<T> residualB16Reg;
        Reg::RegTensor<float> hOutReg, residualReg, weightReg, dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(hOutReg, hOutAddr + i * VL_FP32);
            Reg::Muls(dstReg, hOutReg, postWeight, mask);
            for (uint16_t j = 0; j < static_cast<uint16_t>(N); ++j) {
                const uint32_t row = static_cast<uint32_t>(j);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(residualB16Reg,
                                                                  residualAddr + row * rowStride + i * VL_FP32);
                Reg::Cast<float, T, CAST_B16_TO_FP32>(residualReg, residualB16Reg, mask);
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weightReg, combAddr + row * N + headIndex);
                Reg::MulAddDst(dstReg, residualReg, weightReg, mask);
            }
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 一个VF同时计算相邻两个head，共用h_out与residual的寄存器加载，并保持各自四路累加顺序不变。
template <typename T, bool FUSE_NORM_SQUARE = false>
__aicore__ inline void VFComputePostHeadPair(const LocalTensor<T> &dst0, const LocalTensor<T> &dst1,
                                             const LocalTensor<float> &hOut, const LocalTensor<T> &residual,
                                             const LocalTensor<float> &combWeights, uint32_t rowStride,
                                             uint32_t headIndex, float postWeight0, float postWeight1, uint32_t count,
                                             const LocalTensor<T> &rawHOut = LocalTensor<T>(),
                                             const LocalTensor<float> &gamma = LocalTensor<float>(), float invRms = 1.0f)
{
    __local_mem__ T *dst0Addr = (__local_mem__ T *)dst0.GetPhyAddr();
    __local_mem__ T *dst1Addr = (__local_mem__ T *)dst1.GetPhyAddr();
    __local_mem__ float *hOutAddr = (__local_mem__ float *)hOut.GetPhyAddr();
    __local_mem__ T *residualAddr = (__local_mem__ T *)residual.GetPhyAddr();
    __local_mem__ float *combAddr = (__local_mem__ float *)combWeights.GetPhyAddr();
    __local_mem__ T *rawAddr = nullptr;
    __local_mem__ float *gammaAddr = nullptr;
    if constexpr (FUSE_NORM_SQUARE) {
        rawAddr = (__local_mem__ T *)rawHOut.GetPhyAddr();
        gammaAddr = (__local_mem__ float *)gamma.GetPhyAddr();
    }
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> residualB16Reg, dst0B16Reg, dst1B16Reg, normB16Reg;
        Reg::RegTensor<float> hOutReg, residualReg, dst0Reg, dst1Reg;
        Reg::RegTensor<float> weight00Reg, weight01Reg, weight10Reg, weight11Reg;
        Reg::RegTensor<float> weight20Reg, weight21Reg, weight30Reg, weight31Reg;
        Reg::MaskReg mask;
        // 8个组合权重与D无关，在循环外一次广播，避免每个vector block重复从UB取标量。
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight00Reg, combAddr + headIndex);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight01Reg, combAddr + headIndex + 1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight10Reg, combAddr + N + headIndex);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight11Reg, combAddr + N + headIndex + 1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight20Reg, combAddr + 2U * N + headIndex);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight21Reg, combAddr + 2U * N + headIndex + 1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight30Reg, combAddr + 3U * N + headIndex);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight31Reg, combAddr + 3U * N + headIndex + 1U);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            if constexpr (FUSE_NORM_SQUARE) {
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(normB16Reg, rawAddr + i * VL_FP32);
                Reg::Cast<float, T, CAST_B16_TO_FP32>(hOutReg, normB16Reg, mask);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(residualReg, gammaAddr + i * VL_FP32);
                Reg::Muls(hOutReg, hOutReg, invRms, mask);
                Reg::Mul(hOutReg, hOutReg, residualReg, mask);
                if constexpr (IsSameT<T, bfloat16_t>::value) {
                    Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(normB16Reg, hOutReg, mask);
                } else {
                    Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(normB16Reg, hOutReg, mask);
                }
                Reg::Cast<float, T, CAST_B16_TO_FP32>(hOutReg, normB16Reg, mask);
            } else {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(hOutReg, hOutAddr + i * VL_FP32);
            }
            Reg::Muls(dst0Reg, hOutReg, postWeight0, mask);
            Reg::Muls(dst1Reg, hOutReg, postWeight1, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(residualB16Reg, residualAddr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(residualReg, residualB16Reg, mask);
            Reg::MulAddDst(dst0Reg, residualReg, weight00Reg, mask);
            Reg::MulAddDst(dst1Reg, residualReg, weight01Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(residualB16Reg,
                                                              residualAddr + rowStride + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(residualReg, residualB16Reg, mask);
            Reg::MulAddDst(dst0Reg, residualReg, weight10Reg, mask);
            Reg::MulAddDst(dst1Reg, residualReg, weight11Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(residualB16Reg,
                                                              residualAddr + 2U * rowStride + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(residualReg, residualB16Reg, mask);
            Reg::MulAddDst(dst0Reg, residualReg, weight20Reg, mask);
            Reg::MulAddDst(dst1Reg, residualReg, weight21Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(residualB16Reg,
                                                              residualAddr + 3U * rowStride + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(residualReg, residualB16Reg, mask);
            Reg::MulAddDst(dst0Reg, residualReg, weight30Reg, mask);
            Reg::MulAddDst(dst1Reg, residualReg, weight31Reg, mask);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(dst0B16Reg, dst0Reg, mask);
                Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(dst1B16Reg, dst1Reg, mask);
            } else {
                Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(dst0B16Reg, dst0Reg, mask);
                Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(dst1B16Reg, dst1Reg, mask);
            }
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst0Addr + i * VL_FP32, dst0B16Reg, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst1Addr + i * VL_FP32, dst1B16Reg, mask);
            if constexpr (FUSE_NORM_SQUARE) {
                // Mid-RMS must square the quantized post outputs, not the
                // pre-quantization FP32 registers. Keep both rounding points.
                Reg::Cast<float, T, CAST_B16_TO_FP32>(dst0Reg, dst0B16Reg, mask);
                Reg::Cast<float, T, CAST_B16_TO_FP32>(dst1Reg, dst1B16Reg, mask);
                Reg::Mul(dst0Reg, dst0Reg, dst0Reg, mask);
                Reg::Mul(dst1Reg, dst1Reg, dst1Reg, mask);
                Reg::Add(dst0Reg, dst0Reg, dst1Reg, mask);
                Reg::StoreAlign(hOutAddr + i * VL_FP32, dst0Reg, mask);
            }
        }
    }
}

// 门控融合在一个VF内完成四路低精度读取、FP32转换和门控累加，不生成四行FP32中间张量。
// 四个gate权重对整个D循环不变，外提到循环外广播一次，循环内显式展开四路累加，
// 避免每轮重复广播（参考mhc_pre::VFDoV1ProcessHInRows）。
template <typename T>
__aicore__ inline void VFGateWeightedSum(const LocalTensor<float> &dst, const LocalTensor<T> &src, uint32_t headStride,
                                         const LocalTensor<float> &gate, uint32_t count)
{
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *gateAddr = (__local_mem__ float *)gate.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg;
        Reg::RegTensor<float> srcFp32Reg, accReg;
        Reg::RegTensor<float> g0Reg, g1Reg, g2Reg, g3Reg;
        Reg::MaskReg mask;
        // 四个gate权重只依赖head，对整个D循环不变，广播一次。
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(g0Reg, gateAddr + 0U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(g1Reg, gateAddr + 1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(g2Reg, gateAddr + 2U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(g3Reg, gateAddr + 3U);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 0U * headStride + offset);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::Mul(accReg, srcFp32Reg, g0Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 1U * headStride + offset);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::MulAddDst(accReg, srcFp32Reg, g1Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 2U * headStride + offset);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::MulAddDst(accReg, srcFp32Reg, g2Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 3U * headStride + offset);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::MulAddDst(accReg, srcFp32Reg, g3Reg, mask);
            Reg::StoreAlign(dstAddr + offset, accReg, mask);
        }
    }
}

// 门控融合一次VF完成四路门控和、低精度缓存以及量化后平方，保留低精度往返语义并消除中间FP32写回。
template <typename T>
__aicore__ inline void VFGateWeightedSumQuantizeSquare(const LocalTensor<T> &dstCache, const LocalTensor<float> &square,
                                                       const LocalTensor<T> &src, uint32_t headStride,
                                                       const LocalTensor<float> &gate, uint32_t count)
{
    __local_mem__ T *dstCacheAddr = (__local_mem__ T *)dstCache.GetPhyAddr();
    __local_mem__ float *squareAddr = (__local_mem__ float *)square.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *gateAddr = (__local_mem__ float *)gate.GetPhyAddr();
    static_assert(2560U % VL_FP32 == 0U && 1024U % VL_FP32 == 0U && 2048U % VL_FP32 == 0U);
    const uint16_t loops = static_cast<uint16_t>(count / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg, quantizedReg;
        Reg::RegTensor<float> srcFp32Reg, accReg, quantizedFp32Reg, squareReg;
        Reg::RegTensor<float> g0Reg, g1Reg, g2Reg, g3Reg;
        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(g0Reg, gateAddr + 0U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(g1Reg, gateAddr + 1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(g2Reg, gateAddr + 2U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(g3Reg, gateAddr + 3U);
        for (uint16_t i = 0; i < loops; ++i) {
            const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 0U * headStride + offset);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::Mul(accReg, srcFp32Reg, g0Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 1U * headStride + offset);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::MulAddDst(accReg, srcFp32Reg, g1Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 2U * headStride + offset);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::MulAddDst(accReg, srcFp32Reg, g2Reg, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + 3U * headStride + offset);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::MulAddDst(accReg, srcFp32Reg, g3Reg, mask);
            if constexpr (IsSameT<T, bfloat16_t>::value) {
                Reg::Cast<T, float, CAST_FP32_TO_B16_RINT>(quantizedReg, accReg, mask);
            } else {
                Reg::Cast<T, float, CAST_FP32_TO_B16_NONE>(quantizedReg, accReg, mask);
            }
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dstCacheAddr + offset, quantizedReg, mask);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(quantizedFp32Reg, quantizedReg, mask);
            Reg::Mul(squareReg, quantizedFp32Reg, quantizedFp32Reg, mask);
            Reg::StoreAlign(squareAddr + offset, squareReg, mask);
        }
    }
}

__aicore__ inline void VFAxpy(const LocalTensor<float> &dst, const LocalTensor<float> &src, float scalar,
                              uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcReg, dstReg, scalarReg;
        Reg::MaskReg mask;
        Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Duplicate(scalarReg, scalar, maskAll);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dstReg, dstAddr + i * VL_FP32);
            Reg::MulAddDst(dstReg, srcReg, scalarReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 从UB广播标量权重，并在寄存器内累加到目标张量。
__aicore__ inline void VFAxpyWithUbScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                          const LocalTensor<float> &scalars, uint32_t scalarIndex, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalars.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcReg, dstReg, scalarReg;
        Reg::MaskReg mask;
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr + scalarIndex);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + i * VL_FP32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dstReg, dstAddr + i * VL_FP32);
            Reg::MulAddDst(dstReg, srcReg, scalarReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// 单个head在一个VF内完成低精度读取、FP32转换和门控累加。
template <typename T>
__aicore__ inline void VFAxpyB16WithUbScalar(const LocalTensor<float> &dst, const LocalTensor<T> &src,
                                             const LocalTensor<float> &scalars, uint32_t scalarIndex, uint32_t count)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ T *srcAddr = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalars.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg;
        Reg::RegTensor<float> srcFp32Reg, dstReg, scalarReg;
        Reg::MaskReg mask;
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr + scalarIndex);
        for (uint16_t i = 0; i < loops; ++i) {
            mask = Reg::UpdateMask<float>(remain);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + i * VL_FP32);
            Reg::Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dstReg, dstAddr + i * VL_FP32);
            Reg::MulAddDst(dstReg, srcFp32Reg, scalarReg, mask);
            Reg::StoreAlign(dstAddr + i * VL_FP32, dstReg, mask);
        }
    }
}

// clang-format off
}
// clang-format on

#endif // MHC_NORM_POST_PREONLY_KERNEL_REGBASE_H_
