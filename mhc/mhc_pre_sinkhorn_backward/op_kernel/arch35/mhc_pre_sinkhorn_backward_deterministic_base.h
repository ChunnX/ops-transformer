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
 * \file mhc_pre_sinkhorn_backward_deterministic_base.h
 * \brief 公共基础设施: CopyIn/CopyOut 封装 + SIMD 纯计算函数
 */
#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_BASE_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

using namespace AscendC;

constexpr uint64_t DOUBLE_BUFFER = 2;

struct Process1Info {
    int64_t bsLen = 0;
    int64_t cLen = 0;
    int64_t cLenXTAlign = 0;
    int64_t cLenUAlign = 0;
    int64_t cLenGRADHINTAlign = 0;
};

constexpr static AscendC::MicroAPI::CastTrait castTrait16ToFloat = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTraitFloatTo16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

/*!
 * \brief 向上对齐计算，将 value 向上取整到 align 的整数倍
 * \param value 待对齐的值
 * \param align 对齐基准
 * \return 向上对齐后的值
 */
template <typename T>
constexpr T AlignUp(T value, T align)
{
    return (value + align - 1) / align * align;
}

/*!
 * \brief 将数据从 GlobalMemory 搬运到 UnifiedBuffer (GM → UB)
 * \param dstTensor 目标 LocalTensor (UB 侧)
 * \param srcTensor 源 GlobalTensor (GM 侧)
 * \param repTime 重复拷贝次数
 * \param dataLen 每次拷贝的数据元素个数
 * \param ubStride UB 侧步长
 * \param gmStride GM 侧步长
 */
template <typename PARAM_T>
__aicore__ inline void CopyIn(const LocalTensor<PARAM_T> &dstTensor, const GlobalTensor<PARAM_T> &srcTensor,
                              int64_t repTime, int64_t dataLen, uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                    static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubStride),
                                    static_cast<uint32_t>(0)};
    DataCopyPadExtParams<PARAM_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                               static_cast<PARAM_T>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
}

/*!
 * \brief 将数据从 UnifiedBuffer 搬运到 GlobalMemory (UB → GM)
 * \param dstTensor 目标 GlobalTensor (GM 侧)
 * \param srcTensor 源 LocalTensor (UB 侧)
 * \param repTime 重复拷贝次数
 * \param dataLen 每次拷贝的数据元素个数
 * \param ubStride UB 侧步长
 * \param gmStride GM 侧步长
 */
template <typename PARAM_T>
__aicore__ inline void CopyOut(const GlobalTensor<PARAM_T> &dstTensor, const LocalTensor<PARAM_T> &srcTensor,
                               int64_t repTime, int64_t dataLen, uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                    static_cast<uint32_t>(ubStride), static_cast<uint32_t>(gmStride),
                                    static_cast<uint32_t>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams);
}

/*!
 * \brief 计算 Sigmoid 的梯度 (grad_z)
 * \param gradHAddr 上游梯度 dy 地址
 * \param zAddr 前向输出 z 地址
 * \param gradZAddr 梯度输出 grad_z 地址
 * \param bsLen batch 序列长度
 * \param n 列数
 * \param vRegSize 向量寄存器大小 (字节)
 * \param hcEps 预留偏置 (仅 ISPRE 为真时使用)
 * \note 数学公式: s = sigmoid(z); grad = dy * s * (1 - s); 当 ISPRE 为真时 s = sigmoid(z) - hc_eps
 */
template <typename U, bool ISPRE>
__aicore__ inline void SigmoidGrad(__local_mem__ U *gradHAddr, __local_mem__ U *zAddr, __local_mem__ U *gradZAddr,
                                   uint64_t bsLen, uint64_t n, uint64_t vRegSize, float hcEps)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (bsLen * n + vfLen - 1) / vfLen;
    float hcEpsNeg = hcEps * (-1);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> onesReg;
        AscendC::MicroAPI::RegTensor<U> zReg;
        AscendC::MicroAPI::RegTensor<U> gradHReg;
        AscendC::MicroAPI::RegTensor<U> sigmaReg;
        AscendC::MicroAPI::RegTensor<U> dysigmaReg;
        AscendC::MicroAPI::RegTensor<U> dysigma2Reg;
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::MaskReg maskReg;
        uint32_t maskLen = static_cast<uint32_t>(bsLen * n);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            MicroAPI::Duplicate(onesReg, U(1), maskReg);
            AscendC::MicroAPI::AddrReg zOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradHOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradZOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::DataCopy(zReg, zAddr, zOfst);
            AscendC::MicroAPI::DataCopy(gradHReg, gradHAddr, gradHOfst);
            AscendC::MicroAPI::Neg(zReg, zReg, maskReg);
            AscendC::MicroAPI::Exp(zReg, zReg, maskReg);
            AscendC::MicroAPI::Add(zReg, zReg, onesReg, maskReg);
            AscendC::MicroAPI::Div(sigmaReg, onesReg, zReg, maskReg);
            if constexpr (ISPRE) {
                AscendC::MicroAPI::Adds(sigmaReg, sigmaReg, hcEpsNeg, maskReg);
            }
            AscendC::MicroAPI::Mul(dysigmaReg, sigmaReg, gradHReg, maskReg);
            AscendC::MicroAPI::Mul(dysigma2Reg, sigmaReg, dysigmaReg, maskReg);
            AscendC::MicroAPI::Sub(gradZReg, dysigmaReg, dysigma2Reg, maskReg);
            if constexpr (!ISPRE) {
                AscendC::MicroAPI::Muls(gradZReg, gradZReg, U(2), maskReg);
            }
            AscendC::MicroAPI::DataCopy(gradZAddr, gradZReg, gradZOfst, maskReg);
        }
    }
}

/*!
 * \brief 计算偏置 bias 的梯度 (grad_bias)，沿 bs 维度归约
 * \param gradBiasAddr 梯度输出 grad_bias 地址
 * \param gradZAddr 输入梯度 grad_z 地址
 * \param bsLen batch 序列长度
 * \param colLen 列长度
 * \param vRegSize 向量寄存器大小 (字节)
 * \note 数学公式: grad_bias = sum(grad_z) over bs dimension
 */
template <typename U>
__aicore__ inline void ComputeGradBias(__local_mem__ U *gradBiasAddr, __local_mem__ U *gradZAddr, uint64_t bsLen,
                                       uint64_t colLen, uint64_t vRegSize)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::RegTensor<U> gradBiasReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::MaskReg allMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            auto gradBiasStart = gradBiasAddr + i * vfLen;
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            MicroAPI::Duplicate(gradBiasReg, U(0), allMaskReg);
            for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
                auto gradZAddrStart = gradZAddr + j * colLen + i * vfLen;
                AscendC::MicroAPI::LoadUnAlignPre(u0, gradZAddrStart);
                AscendC::MicroAPI::LoadUnAlign(gradZReg, u0, gradZAddrStart);
                AscendC::MicroAPI::Add(gradBiasReg, gradBiasReg, gradZReg, maskReg);
            }
            AscendC::MicroAPI::DataCopy(gradBiasStart, gradBiasReg, maskReg);
        }
    }
}

/*!
 * \brief 计算缩放系数 alpha 的梯度 (grad_alpha)
 * \param normOutForwardPartAddr 前向归一化输出地址
 * \param gradZAddr 梯度 grad_z 地址
 * \param gradAlphaAddr 梯度输出 grad_alpha 地址
 * \param bsLen batch 序列长度
 * \param colLen 列长度
 * \param vRegSize 向量寄存器大小 (字节)
 * \note 数学公式: grad_alpha = sum(norm_out_forward * grad_z)
 */
template <typename U>
__aicore__ inline void ComputeGradAlpha(__local_mem__ U *normOutForwardPartAddr, __local_mem__ U *gradZAddr,
                                        __local_mem__ U *gradAlphaAddr, uint64_t bsLen, uint64_t colLen,
                                        uint64_t vRegSize)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (bsLen * colLen + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> normOutForwardPartReg;
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaLoopSumReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaSumReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(gradAlphaSumReg, U(0), OneMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(bsLen * colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            AscendC::MicroAPI::AddrReg normOutForwardPartOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradZOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::DataCopy(normOutForwardPartReg, normOutForwardPartAddr, normOutForwardPartOfst);
            AscendC::MicroAPI::DataCopy(gradZReg, gradZAddr, gradZOfst);
            AscendC::MicroAPI::Mul(gradAlphaReg, normOutForwardPartReg, gradZReg, maskReg);
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradAlphaLoopSumReg, gradAlphaReg,
                                                                       maskReg); // reduce_sum
            AscendC::MicroAPI::Add(gradAlphaSumReg, gradAlphaSumReg, gradAlphaLoopSumReg, OneMaskReg);
        }
        AscendC::MicroAPI::DataCopy(gradAlphaAddr, gradAlphaSumReg, OneMaskReg);
    }
}

/*!
 * \brief 计算 z 和前向归一化输出 (norm_out_forward)
 * \param zAddr 输出 z 地址
 * \param normOutForwardAddr 前向归一化输出地址
 * \param biasAddr 偏置 bias 地址
 * \param bsLen batch 序列长度
 * \param colLen 列长度
 * \param cLenAlign 对齐后的列长度
 * \param vRegSize 向量寄存器大小 (字节)
 * \param alpha 缩放系数
 * \note 数学公式: z = norm_out * alpha + bias
 */
template <typename U>
__aicore__ inline void ComputeZAndForwardPart(__local_mem__ U *zAddr, __local_mem__ U *normOutForwardAddr,
                                              __local_mem__ U *biasAddr, uint64_t bsLen, uint64_t colLen,
                                              uint64_t cLenAlign, uint64_t vRegSize, const U &alpha)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> zReg;
        AscendC::MicroAPI::RegTensor<U> normOutForwardReg;
        AscendC::MicroAPI::RegTensor<U> biasReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::UnalignRegForStore unStoreZReg;

        for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
            AscendC::MicroAPI::UnalignReg unAlignNormOutForwardReg;
            AscendC::MicroAPI::UnalignReg unAlignBiasReg;
            auto biasAddrStart = biasAddr;
            auto zAddrStart = zAddr + j * colLen;
            auto normOutForwardAddrStart = normOutForwardAddr + j * colLen;
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::LoadUnAlignPre(unAlignNormOutForwardReg, normOutForwardAddrStart + i * vfLen);
                AscendC::MicroAPI::LoadUnAlign(normOutForwardReg, unAlignNormOutForwardReg,
                                               normOutForwardAddrStart + i * vfLen);
                AscendC::MicroAPI::LoadUnAlignPre(unAlignBiasReg, biasAddrStart + i * vfLen);
                AscendC::MicroAPI::LoadUnAlign(biasReg, unAlignBiasReg, biasAddrStart + i * vfLen);
                AscendC::MicroAPI::Muls(zReg, normOutForwardReg, alpha, valueMaskReg);
                AscendC::MicroAPI::Add(zReg, zReg, biasReg, valueMaskReg);
                MicroAPI::StoreUnAlign<U>(zAddrStart, zReg, unStoreZReg, vfLen);
            }
            MicroAPI::StoreUnAlignPost<U>(zAddrStart, unStoreZReg, 0);
        }
    }
}

/*!
 * \brief 计算前向归一化输出 (norm_out_forward)
 * \param hcBeformNormAddr 归一化前输入地址
 * \param invRmsAddr 逆均方根地址
 * \param normOutForwardAddr 输出地址
 * \param bsLen batch 序列长度
 * \param colLen 列长度
 * \param cLenAlign 对齐后的列长度
 * \param vRegSize 向量寄存器大小 (字节)
 * \param alpha 缩放系数
 * \note 数学公式: norm_out = hc_before_norm * inv_rms * alpha
 */
template <typename U>
__aicore__ inline void ComputeForwardPart(__local_mem__ U *hcBeformNormAddr, __local_mem__ U *invRmsAddr,
                                          __local_mem__ U *normOutForwardAddr, uint64_t bsLen, uint64_t colLen,
                                          uint64_t cLenAlign, uint64_t vRegSize, const U &alpha)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> hcBeformNormReg;
        AscendC::MicroAPI::RegTensor<U> normOutForwardReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::UnalignRegForStore unStoreNormOutForwardReg;

        for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
            auto normOutForwardAddrStart = normOutForwardAddr + j * colLen;
            auto hcBeformNormAddrStart = hcBeformNormAddr + j * cLenAlign;
            U invRms = *(invRmsAddr + j);
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::AddrReg hcBeformNormOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
                AscendC::MicroAPI::DataCopy(hcBeformNormReg, hcBeformNormAddrStart, hcBeformNormOfst);
                AscendC::MicroAPI::Muls(normOutForwardReg, hcBeformNormReg, invRms, valueMaskReg);
                MicroAPI::StoreUnAlign<U>(normOutForwardAddrStart, normOutForwardReg, unStoreNormOutForwardReg, vfLen);
            }
            MicroAPI::StoreUnAlignPost<U>(normOutForwardAddrStart, unStoreNormOutForwardReg, 0);
        }
    }
}
/*!
 * \brief 计算张量乘以标量，使用 VF 循环实现
 * \param dstLocal 目标 LocalTensor
 * \param srcLocal 源 LocalTensor
 * \param scalar 标量乘数
 * \param count 参与计算的元素个数
 * \param offset 源张量偏移量
 * \note 数学公式: dst = src * scalar (VF循环)
 *       注意事项：需将该函数插入到 GradNormPre/GradNormRes 每核每次 BS 计算完成后；
 *       计算 GradHcBeforeNorm 后需添加 syncAll 以保证 matmul 的确定性。
 */
template <typename T>
__aicore__ inline void ComputeMulScalar(const LocalTensor<T>& dstLocal , const LocalTensor<T>& srcLocal, T scalar,
                                        uint32_t count, uint32_t offset)
{
    auto srcAddr = (__local_mem__ T*)srcLocal.GetPhyAddr() + offset;
    auto dstAddr = (__local_mem__ T*)dstLocal.GetPhyAddr();
    uint32_t repeatCount = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t maskCount = count;
    uint16_t repeatTimes = Ops::Base::CeilDiv(maskCount, repeatCount);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> reg0;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(maskCount);
            MicroAPI::UnalignReg u0;
            MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
            MicroAPI::DataCopyUnAlign(reg0, u0, srcAddr, repeatCount);
            MicroAPI::Muls(reg0, reg0, scalar, preg);
            MicroAPI::DataCopy(dstAddr + i * repeatCount, reg0, preg);
        }
    }
}

/*!
 * \brief 计算逆均方根的梯度 (grad_inv_rms)
 * \param gradNormOutLocal 归一化输出的梯度 LocalTensor
 * \param hcBeforeNormLocal 归一化前的输入 LocalTensor (兼作中间结果缓冲)
 * \param count 元素个数
 * \param gradInvRms 输出的梯度值
 * \note 数学公式: grad_inv_rms = sum(hc_before_norm * grad_norm_out)
 */
template <typename T>
__aicore__ inline void ComputeGradInvRms(const LocalTensor<T> &gradNormOutLocal,
                                         const LocalTensor<T> &hcBeforeNormLocal, int64_t count, T &gradInvRms)
{
    Mul(hcBeforeNormLocal, gradNormOutLocal, hcBeforeNormLocal, count);
    ReduceSum<T>(hcBeforeNormLocal, hcBeforeNormLocal, hcBeforeNormLocal, count);
    gradInvRms = hcBeforeNormLocal(0);
}

/*!
 * \brief 基于 RMS 计算输入 x 的梯度 (VF 循环实现)
 * \param xFp32Local 输入 x 的 FP32 LocalTensor
 * \param xFp32LocalOut 输出梯度 LocalTensor
 * \param gradInvRms 逆均方根的梯度
 * \param invRms 逆均方根值
 * \param count 元素个数
 * \param ncCount N*C 总数 (归约维度)
 * \note 数学公式: grad_x_rms = -(inv_rms^3 * x * grad_inv_rms) / (N*C)
 */
template <typename T>
__aicore__ inline void ComputeGradXFromRmsVF(const LocalTensor<T>& xFp32Local, const LocalTensor<T>& xFp32LocalOut,
                                                T gradInvRms, T invRms, int64_t count, int64_t ncCount)
{
    __local_mem__ T *xFp32Addr = (__local_mem__ T *)xFp32Local.GetPhyAddr();
    __local_mem__ T *xFp32AddrOut = (__local_mem__ T *)xFp32LocalOut.GetPhyAddr();
    uint32_t repeatCount = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t maskCount = count;
    uint16_t repeatTimes = Ops::Base::CeilDiv(maskCount, repeatCount);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> xFp32Reg;
        MicroAPI::RegTensor<T> ncCountReg;
        MicroAPI::Duplicate(ncCountReg, static_cast<T>(ncCount));

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(maskCount);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatCount);
            MicroAPI::DataCopy(xFp32Reg, (__local_mem__ T *)xFp32Addr, offset);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, static_cast<T>(-1), preg);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, gradInvRms, preg);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, invRms, preg);
            MicroAPI::Div(xFp32Reg, xFp32Reg, ncCountReg, preg);
            MicroAPI::DataCopy(xFp32AddrOut + i * repeatCount, xFp32Reg, preg);
        }
    }
}

// ============================================================
// SinkhornGrad 子步骤: 独立模板函数, for 循环在 __VEC_SCOPE__ 内
// ============================================================

/*!
 * \brief SinkhornGrad Step1: tmpMul = rowNorm ⊙ skGrad
 * \param rowNormPtr norm_out 指针 [bsLen, n, n]
 * \param skGradPtr skGrad 指针 [bsLen, n, n]
 * \param tmpPtr 输出 tmpMul 指针 [bsLen, n, n]
 * \param bsLenLoop BS 循环次数
 * \param n N 维度
 * \param nn N² 维度
 */
__aicore__ inline void SinkhornGradStep1_MulNorm(__local_mem__ float *rowNormPtr,
                                                   __local_mem__ float *skGradPtr,
                                                   __local_mem__ float *tmpPtr,
                                                   uint16_t bsLenLoop, uint32_t n, uint32_t nn)
{
    __VEC_SCOPE__
    {
        Reg::UnalignRegForLoad ureg0, ureg1;
        Reg::RegTensor<float> rowNormReg;
        Reg::RegTensor<float> skGradReg;
        Reg::RegTensor<float> tmpMulReg;
        Reg::MaskReg maskNN = Reg::UpdateMask<float>(nn);

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            int64_t bsOffset = j * nn;
            Reg::LoadUnAlignPre(ureg0, rowNormPtr + bsOffset);
            Reg::LoadUnAlign(rowNormReg, ureg0, rowNormPtr + bsOffset);
            Reg::LoadUnAlignPre(ureg1, skGradPtr + bsOffset);
            Reg::LoadUnAlign(skGradReg, ureg1, skGradPtr + bsOffset);
            Reg::Mul(tmpMulReg, rowNormReg, skGradReg, maskNN);
            Reg::StoreAlign(tmpPtr + bsOffset, tmpMulReg, maskNN);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

/*!
 * \brief SinkhornGrad Step2: tmpDiv = skGrad / colSum, tmpDiv2 = tmpMul / colSum²
 * \param tmpPtr tmpMul 指针 (Step1 输出)
 * \param skGradPtr skGrad 指针
 * \param colSumPtr 列求和指针 [bsLen, n]
 * \param tmpDivLocalPtr 输出 tmpDiv
 * \param tmpDiv2LocalPtr 输出 tmpDiv2
 * \param bsLenLoop BS 循环次数
 * \param n N 维度
 * \param nn N² 维度
 */
__aicore__ inline void SinkhornGradStep2_DivColSum(__local_mem__ float *tmpPtr,
                                                     __local_mem__ float *skGradPtr,
                                                     __local_mem__ float *colSumPtr,
                                                     __local_mem__ float *tmpDivLocalPtr,
                                                     __local_mem__ float *tmpDiv2LocalPtr,
                                                     uint16_t bsLenLoop, uint32_t n, uint32_t nn)
{
    __VEC_SCOPE__
    {
        Reg::UnalignRegForLoad ureg2, ureg3, ureg4;
        Reg::UnalignRegForStore uregStore0, uregStore1;
        Reg::RegTensor<float> colSumReg;
        Reg::RegTensor<float> colSumSquareReg;
        Reg::RegTensor<float> tmpDivReg;
        Reg::RegTensor<float> tmpDiv2Reg;
        Reg::MaskReg maskNN = Reg::UpdateMask<float>(nn);
        Reg::MaskReg maskN = Reg::UpdateMask<float>(n);

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            int64_t bsOffset = j * nn;
            auto tmpDivLocalPtr1 = tmpDivLocalPtr + bsOffset;
            auto tmpDiv2LocalPtr1 = tmpDiv2LocalPtr + bsOffset;
            auto tmpPtr1 = tmpPtr + bsOffset;
            auto skGradPtr1 = skGradPtr + bsOffset;

            Reg::LoadUnAlignPre(ureg2, colSumPtr + j * n);
            Reg::LoadUnAlign(colSumReg, ureg2, colSumPtr + j * n);
            Reg::Mul(colSumSquareReg, colSumReg, colSumReg, maskN);

            for (uint16_t k = 0; k < n; k++) {
                Reg::RegTensor<float> tmpReg, tmpReg1;
                Reg::LoadUnAlignPre(ureg3, tmpPtr1 + k * n);
                Reg::LoadUnAlign(tmpReg, ureg3, tmpPtr1 + k * n);
                Reg::LoadUnAlignPre(ureg4, skGradPtr1 + k * n);
                Reg::LoadUnAlign(tmpReg1, ureg4, skGradPtr1 + k * n);
                Reg::Div(tmpDivReg, tmpReg1, colSumReg, maskNN);
                Reg::Div(tmpDiv2Reg, tmpReg, colSumSquareReg, maskNN);
                Reg::StoreUnAlign(tmpDivLocalPtr1, tmpDivReg, uregStore0, n);
                Reg::StoreUnAlign(tmpDiv2LocalPtr1, tmpDiv2Reg, uregStore1, n);
            }
            Reg::StoreUnAlignPost(tmpDivLocalPtr1, uregStore0, 0);
            Reg::StoreUnAlignPost(tmpDiv2LocalPtr1, uregStore1, 0);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

/*!
 * \brief SinkhornGrad Step3: tmpDiv2 reduce_sum → gradXRowNormed
 * \param tmpDiv2LocalPtr tmpDiv2 指针 (Step2 输出)
 * \param tmpPtr 输出 gradXRowNormed 指针
 * \param bsLenLoop BS 循环次数
 * \param n N 维度
 * \param nn N² 维度
 * \param nAlignFp32 N 对齐后的元素数
 */
__aicore__ inline void SinkhornGradStep3_ReduceRow(__local_mem__ float *tmpDiv2LocalPtr,
                                                     __local_mem__ float *tmpPtr,
                                                     uint16_t bsLenLoop, uint32_t n, uint32_t nn,
                                                     int64_t nAlignFp32)
{
    __VEC_SCOPE__
    {
        Reg::UnalignRegForLoad ureg5;
        Reg::MaskReg maskN = Reg::UpdateMask<float>(n);

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            int64_t bsOffset = j * nn;
            auto tmpDiv2LocalPtr1 = tmpDiv2LocalPtr + bsOffset;
            auto tmpPtr1 = tmpPtr + j * nAlignFp32;

            Reg::RegTensor<float> reduceSumReg;
            Reg::Duplicate(reduceSumReg, (float)0.0, maskN);
            for (uint16_t k = 0; k < n; k++) {
                Reg::RegTensor<float> tmpReg;
                Reg::LoadUnAlignPre(ureg5, tmpDiv2LocalPtr1 + k * n);
                Reg::LoadUnAlign(tmpReg, ureg5, tmpDiv2LocalPtr1 + k * n);
                Reg::Add(reduceSumReg, reduceSumReg, tmpReg, maskN);
            }
            Reg::StoreAlign(tmpPtr1, reduceSumReg, maskN);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

// ============================================================
// ExpGrad 子步骤: 独立模板函数
// ============================================================

/*!
 * \brief ExpGrad Step1: 逐行 reduce_max → zResMax/zResMaxIdx/isMax
 * \note 对 [n, n] 矩阵逐行求最大值、最大值索引、isMax 标记
 */
__aicore__ inline void ExpGradStep1_ComputeMax(__local_mem__ float *zResPtr,
                                                 __local_mem__ float *zResMaxPtr,
                                                 __local_mem__ float *zResMaxIdxPtr,
                                                 __local_mem__ float *isMaxIdxPtr,
                                                 uint32_t n)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> zResReg;
        Reg::RegTensor<float> zResMaxReg;
        Reg::RegTensor<float> zResMaxIdxReg;
        Reg::RegTensor<float> zResMaxBcastReg;
        Reg::RegTensor<float> zResMaxIdxBcastReg;
        Reg::RegTensor<float> isMaxReg;
        Reg::RegTensor<float> idxReg, idxSelectReg;
        Reg::RegTensor<float> zeroReg, oneReg, negInfReg;
        Reg::UnalignRegForLoad ureg;
        Reg::UnalignRegForStore uregStore0, uregStore1, uregStore2;
        Reg::MaskReg maxMask;
        Reg::MaskReg maskN = Reg::UpdateMask<float>(n);
        Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        Reg::Duplicate(zeroReg, (float)0.0, allMask);
        Reg::Duplicate(oneReg, (float)1.0, allMask);
        Reg::Duplicate(negInfReg, (float)-FLT_MAX, allMask);
        Reg::Arange(idxReg, 0);

        for (uint16_t j = 0; j < n; j++) {
            int64_t nOffset = j * n;
            auto zResPtr1 = zResPtr + nOffset;
            auto zResMaxPtr2 = zResMaxPtr + nOffset;
            auto zResMaxIdxPtr2 = zResMaxIdxPtr + nOffset;

            Reg::LoadUnAlignPre(ureg, zResPtr1);
            Reg::LoadUnAlign(zResReg, ureg, zResPtr1);
            Reg::Reduce<Reg::ReduceType::MAX>(zResMaxReg, zResReg, maskN);
            Reg::Duplicate(zResMaxBcastReg, zResMaxReg, maskN);
            Reg::Compare<float>(maxMask, zResMaxBcastReg, zResReg, maskN);
            Reg::Select(idxSelectReg, idxReg, negInfReg, maxMask);
            Reg::Reduce<Reg::ReduceType::MAX>(zResMaxIdxReg, idxSelectReg, maskN);
            Reg::Select(isMaxReg, oneReg, zeroReg, maxMask);
            Reg::Duplicate(zResMaxIdxBcastReg, zResMaxIdxReg, maskN);

            for (uint16_t k = 0; k < n; k++) {
                Reg::StoreUnAlign<float>(zResMaxPtr2, zResMaxBcastReg, uregStore0, 1);
                Reg::StoreUnAlign<float>(zResMaxIdxPtr2, zResMaxIdxBcastReg, uregStore1, 1);
            }
            Reg::StoreUnAlignPost(zResMaxPtr2, uregStore0, 0);
            Reg::StoreUnAlignPost(zResMaxIdxPtr2, uregStore1, 0);
            Reg::StoreUnAlign<float>(isMaxIdxPtr, isMaxReg, uregStore2, n);
        }
        Reg::StoreUnAlignPost(isMaxIdxPtr, uregStore2, 0);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

/*!
 * \brief ExpGrad Step2: exp(zRes - zResMax) * skGrad → tmpMul
 */
__aicore__ inline void ExpGradStep2_ComputeExp(__local_mem__ float *zResPtr,
                                                 __local_mem__ float *zResMaxPtr,
                                                 __local_mem__ float *skGradPtr,
                                                 __local_mem__ float *tmpMulPtr,
                                                 uint32_t nn)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> zResReg;
        Reg::RegTensor<float> zResMaxReg;
        Reg::RegTensor<float> skGradReg;
        Reg::RegTensor<float> tmpSubReg;
        Reg::RegTensor<float> tmpExpReg;
        Reg::RegTensor<float> tmpMulReg;
        Reg::UnalignRegForLoad ureg0, ureg1, ureg2;
        Reg::MaskReg maskNN = Reg::UpdateMask<float>(nn);

        Reg::LoadUnAlignPre(ureg0, zResPtr);
        Reg::LoadUnAlign(zResReg, ureg0, zResPtr);
        Reg::LoadUnAlignPre(ureg1, zResMaxPtr);
        Reg::LoadUnAlign(zResMaxReg, ureg1, zResMaxPtr);
        Reg::LoadUnAlignPre(ureg2, skGradPtr);
        Reg::LoadUnAlign(skGradReg, ureg2, skGradPtr);

        Reg::Sub<float>(tmpSubReg, zResReg, zResMaxReg, maskNN);
        Reg::Exp<float>(tmpExpReg, tmpSubReg, maskNN);
        Reg::Mul<float>(tmpMulReg, tmpExpReg, skGradReg, maskNN);
        Reg::StoreAlign<float>(tmpMulPtr, tmpMulReg, maskNN);

        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

/*!
 * \brief ExpGrad Step3: tmpMul reduce_sum → expGrad = tmpMul - isMax * sumAll
 */
__aicore__ inline void ExpGradStep3_ComputeGrad(__local_mem__ float *tmpMulPtr,
                                                  __local_mem__ float *isMaxIdxPtr,
                                                  __local_mem__ float *expGradPtr1,
                                                  uint32_t n)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> tmpMulReg;
        Reg::RegTensor<float> isMaxReg;
        Reg::RegTensor<float> sumAllReg;
        Reg::RegTensor<float> sumAllBcastReg;
        Reg::RegTensor<float> tmpMul2Reg;
        Reg::RegTensor<float> expGradReg;
        Reg::UnalignRegForLoad ureg;
        Reg::UnalignRegForStore uregStore3;
        Reg::MaskReg maskN = Reg::UpdateMask<float>(n);

        for (uint16_t j = 0; j < n; j++) {
            Reg::LoadUnAlignPre(ureg, tmpMulPtr + j * n);
            Reg::LoadUnAlign(tmpMulReg, ureg, tmpMulPtr + j * n);
            Reg::LoadUnAlignPre(ureg, isMaxIdxPtr + j * n);
            Reg::LoadUnAlign(isMaxReg, ureg, isMaxIdxPtr + j * n);
            Reg::Reduce<Reg::ReduceType::SUM>(sumAllReg, tmpMulReg, maskN);
            Reg::Duplicate(sumAllBcastReg, sumAllReg, maskN);
            Reg::Mul(tmpMul2Reg, isMaxReg, sumAllBcastReg, maskN);
            Reg::Sub(expGradReg, tmpMulReg, tmpMul2Reg, maskN);
            Reg::StoreUnAlign<float>(expGradPtr1, expGradReg, uregStore3, n);
        }
        Reg::StoreUnAlignPost(expGradPtr1, uregStore3, 0);
    }
}

#endif // MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_BASE_H