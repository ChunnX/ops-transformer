/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_KERNEL_SUPPORT_H_
#define MHC_NORM_POST_PREONLY_KERNEL_SUPPORT_H_

// ========================================================================
// mhc_norm_post_preonly_kernel.h — 类定义和共享计算模块
//
// 文件结构:
//   1. 类声明 + Init / Process
//   2. 共享计算模块
//   3. 成员变量
//   4. 引入各执行路径的实现头文件
// ========================================================================



// ========================================================================
// mhc_norm_post_preonly_common.h — 常量、类型萃取、workspace layout 定义
// ========================================================================

#include "kernel_operator.h"
using namespace AscendC;

// ---------------------------------------------------------------------------
// 全局常量
// ---------------------------------------------------------------------------
constexpr uint32_t N_MAX  = 4;
constexpr uint32_t ALIGN8 = 8;
constexpr uint32_t BUFFER_NUM_TWO = 2;
constexpr uint32_t BUFFER_NUM_ONE = 1;
constexpr uint32_t BUFFER_NUM_FOUR = 4; // phi ping-pong：4槽让相邻o的搬运与计算重叠
constexpr uint32_t GAMMA1_CACHE_MAX_D = 5120; // 目标模型维度以内缓存完整gamma1
constexpr uint32_t SCALAR_BUF_SIZE   = 256;    // 标量计算 UB buffer 大小 (bytes)
constexpr uint32_t EXP_DIV_ELEM      = 8;      // sigmoid ExpDiv 计算元素数
constexpr uint32_t POST_COMB_SIZE    = 8 + 16; // h_post[8] + h_res[16] buffer 大小
constexpr uint32_t PARTIAL_SLOT_ELEMS = 8;     // 每个lane的partial固定占32B
constexpr uint16_t MID_READY_FLAG = 0;         // R2平方和交换完成标志
constexpr uint16_t PAIR_READY_FLAG = 1;        // R4/R5 partial交换完成标志

// ---------------------------------------------------------------------------
// Type trait: fp32->half uses CAST_NONE, fp32->bfloat16_t uses CAST_RINT
// (per CANN RmsNorm reference)
// ---------------------------------------------------------------------------
template<typename A, typename B> struct IsSameT { static constexpr bool value = false; };
template<typename A> struct IsSameT<A, A> { static constexpr bool value = true; };






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
        RegTensor<float> srcFp32Reg, weightReg, accReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            mask = UpdateMask<float>(remain);
            const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            // TODO：组合优化稳定后，单独评估固定四路循环展开，避免混入本轮性能结论。
            for (uint16_t n = 0; n < static_cast<uint16_t>(N_MAX); ++n) {
                const uint32_t head = static_cast<uint32_t>(n);
                DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + head * headStride + offset);
                Cast<float, T, CAST_B16_TO_FP32>(srcFp32Reg, srcReg, mask);
                DataCopy<float, LoadDist::DIST_BRC_B32>(weightReg, gateAddr + head);
                if (n == 0U) {
                    AscendC::Reg::Mul(accReg, srcFp32Reg, weightReg, mask);
                } else {
                    AscendC::Reg::MulAddDst(accReg, srcFp32Reg, weightReg, mask);
                }
            }
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
// clang-format on







// 引入各执行路径的方法实现



// 构造一个低精度x2_post分块。R1使用normEps_，R2保持h_res的[j,n]索引顺序，
// 并显式保留设计规定的两处低精度往返边界。


// 双AIV仍需读取全部残差行，但每个lane只计算自己负责的两个输出head。


// 将完整x2以输出dtype常驻UB，同时完成设计要求的低精度往返；写GM仅用于保留x2_out输出语义。


// 双sibling路径仍在每个lane的私有UB中保留完整x2，便于两个lane按同一顺序执行R1/R2/R3；
// 写回x2_out时每个lane只写自己负责的两个head，避免任何输出竞争。


// R6读取完当前x2分块后，复用首个head缓存保留低精度h_in。
// 恢复FP32与平方在同一VF内完成，保留量化往返边界并减少UB中间写读。




// 固定的R4/R5归约树将与双sibling AIV路径共用。half0固定覆盖展平区间[0,2D)，
// half1固定覆盖[2D,4D)，调用方只执行一次顶层合并partial0+partial1。









// mode1只在同一物理AI Core的两个sibling AIV之间握手。
__aicore__ inline void MhcPairMidReadySync()
{
    CrossCoreSetFlag<0x1, PIPE_MTE3>(MID_READY_FLAG);
    CrossCoreWaitFlag<0x1, PIPE_MTE2>(MID_READY_FLAG);
}

// R4/R5使用独立flag，避免同一token内两次握手相互干扰。
__aicore__ inline void MhcPairReadySync()
{
    CrossCoreSetFlag<0x1, PIPE_MTE3>(PAIR_READY_FLAG);
    CrossCoreWaitFlag<0x1, PIPE_MTE2>(PAIR_READY_FLAG);
}

#endif // MHC_NORM_POST_PREONLY_KERNEL_SUPPORT_H_
