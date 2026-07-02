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
 * \brief 公共基础设施: CopyInQue/CopyOutQue 封装 + SIMD 纯计算函数 + ComputeBlockInfo
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_BASE_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

using namespace AscendC;

constexpr uint64_t DOUBLE_BUFFER = 2;

constexpr static AscendC::MicroAPI::CastTrait castTrait16ToFloat = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTraitFloatTo16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename T>
constexpr T AlignUp(T value, T align)
{
    return (value + align - 1) / align * align;
}

struct ComputeBlockInfo {
    int64_t bsCount = 0;
    int64_t cCount = 0;
    int64_t cAlignedX = 0;
    int64_t cAlignedU = 0;
    int64_t cAlignedGradHin = 0;
};

// ============================================================
// CopyInQue: GM→UB 并自动 EnQue
//
//   rowCount:   搬运行数
//   elemPerRow: 每行有效元素数（从 GM 搬多少）
//   srcStride:  GM 行间字节跨步 (0 = 紧凑连续)
//   ubAlignLen: UB 行对齐元素数 (0 = 紧凑=elemPerRow, >0 时自动 padding 对齐)
//
// 用法:
//   CopyInQue(que, src, 1, totalElems);                     // 连续
//   CopyInQue(que, src, rowCnt, n_);                        // 多行紧凑
//   CopyInQue(que, src, rowCnt, n_, gmStride);               // GM跨步
//   CopyInQue(que, src, rowCnt, n_, 0, ubAlignLen);          // UB对齐+pad
//   CopyInQue(que, src, rowCnt, n_, gmStride, ubAlignLen);   // GM跨步+UB对齐
// ============================================================
template <typename T, typename QueT>
__aicore__ inline void CopyInQue(
    QueT &que,
    const GlobalTensor<T> &srcGm,
    int64_t rowCount,
    int64_t elemPerRow,
    int64_t srcStride = 0,
    int64_t ubAlignLen = 0)
{
    auto local = que.template AllocTensor<T>();
    int64_t alignLen = (ubAlignLen > 0) ? ubAlignLen : elemPerRow;
    int64_t ubStride = (alignLen - elemPerRow) * static_cast<int64_t>(sizeof(T));
    int64_t rightPad = alignLen - elemPerRow;

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerRow * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(srcStride);
    copyParams.dstStride  = static_cast<uint32_t>(ubStride);

    DataCopyPadExtParams<T> padParams{rightPad > 0, 0, static_cast<uint8_t>(rightPad), static_cast<T>(0)};
    DataCopyPad(local, srcGm, copyParams, padParams);
    que.EnQue(local);
}

// ============================================================
// CopyOutQue: UB→GM 并自动 DeQue + FreeTensor
//
//   ubAlignLen: UB 行对齐元素数 (0 = 紧凑=elemPerRow, >0 时跳过 padding)
//   dstStride:  GM 目标端行间字节跨步 (0 = 连续)
//
// 用法:
//   CopyOutQue(que, dst, 1, totalElems);                      // 连续
//   CopyOutQue(que, dst, rowCnt, n_);                         // 多行紧凑
//   CopyOutQue(que, dst, rowCnt, n_, ubAlignLen);              // UB对齐(跳过padding)
//   CopyOutQue(que, dst, rowCnt, n_, ubAlignLen, gmStride);    // UB对齐+GM跨步
// ============================================================
template <typename T, typename QueT>
__aicore__ inline void CopyOutQue(
    QueT &que,
    const GlobalTensor<T> &dstGm,
    int64_t rowCount,
    int64_t elemPerRow,
    int64_t ubAlignLen = 0,
    int64_t dstStride = 0)
{
    auto local = que.template DeQue<T>();
    int64_t alignLen = (ubAlignLen > 0) ? ubAlignLen : elemPerRow;
    int64_t ubStride = (alignLen - elemPerRow) * static_cast<int64_t>(sizeof(T));

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerRow * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(ubStride);
    copyParams.dstStride  = static_cast<uint32_t>(dstStride);
    DataCopyPad(dstGm, local, copyParams);
    que.FreeTensor(local);
}

// ============================================================
// CopyInBuf: GM→Buf (不涉及 EnQue/DeQue)
//
//   用法同 CopyInQue, 传入 TBuf 替代 TQue
// ============================================================
template <typename T>
__aicore__ inline void CopyInBuf(
    TBuf<TPosition::VECCALC> &buf,
    const GlobalTensor<T> &srcGm,
    int64_t rowCount,
    int64_t elemPerRow,
    int64_t srcStride = 0,
    int64_t ubAlignLen = 0)
{
    auto local = buf.template Get<T>();
    int64_t alignLen = (ubAlignLen > 0) ? ubAlignLen : elemPerRow;
    int64_t ubStride = (alignLen - elemPerRow) * static_cast<int64_t>(sizeof(T));
    int64_t rightPad = alignLen - elemPerRow;

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerRow * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(srcStride);
    copyParams.dstStride  = static_cast<uint32_t>(ubStride);

    DataCopyPadExtParams<T> padParams{rightPad > 0, 0, static_cast<uint8_t>(rightPad), static_cast<T>(0)};
    DataCopyPad(local, srcGm, copyParams, padParams);
}

// ============================================================
// CopyOutBuf: Buf→GM
//
//   用法同 CopyOutQue, 传入 TBuf 替代 TQue
// ============================================================
template <typename T>
__aicore__ inline void CopyOutBuf(
    TBuf<TPosition::VECCALC> &buf,
    const GlobalTensor<T> &dstGm,
    int64_t rowCount,
    int64_t elemPerRow,
    int64_t ubAlignLen = 0,
    int64_t dstStride = 0)
{
    auto local = buf.template Get<T>();
    int64_t alignLen = (ubAlignLen > 0) ? ubAlignLen : elemPerRow;
    int64_t ubStride = (alignLen - elemPerRow) * static_cast<int64_t>(sizeof(T));

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerRow * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(ubStride);
    copyParams.dstStride  = static_cast<uint32_t>(dstStride);
    DataCopyPad(dstGm, local, copyParams);
}

// ============================================================
// SigmoidGrad: σ'(z) = dy * σ(z) * (1 - σ(z))
//   ISPRE=true:  h_pre 分支，正向加了 hc_eps
//   ISPRE=false: h_post 分支，正向 ×2
// ============================================================
template <typename U, bool ISPRE>
__aicore__ inline void SigmoidGrad(__local_mem__ U *gradHAddr, __local_mem__ U *zAddr, __local_mem__ U *gradZAddr,
                                   uint64_t bsLen, uint64_t n, uint64_t vRegSize, float hcEps)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (bsLen * n + vfLen - 1) / vfLen;
    float hcEpsNeg = hcEps * (-1);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> onesReg, zReg, gradHReg, sigmaReg, dysigmaReg, dysigma2Reg, gradZReg;
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

// ============================================================
// ComputeGradBias: gradBias[0:colLen] = sum_i gradZ[i][0:colLen]
// ============================================================
template <typename U>
__aicore__ inline void ComputeGradBias(__local_mem__ U *gradBiasAddr, __local_mem__ U *gradZAddr, uint64_t bsLen,
                                       uint64_t colLen, uint64_t vRegSize)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> gradZReg, gradBiasReg;
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

// ============================================================
// ComputeGradAlpha: gradAlpha = sum(normOut_forward * gradZ)
// ============================================================
template <typename U>
__aicore__ inline void ComputeGradAlpha(__local_mem__ U *normOutForwardPartAddr, __local_mem__ U *gradZAddr,
                                        __local_mem__ U *gradAlphaAddr, uint64_t bsLen, uint64_t colLen,
                                        uint64_t vRegSize)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (bsLen * colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> normOutForwardPartReg, gradZReg, gradAlphaReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaLoopSumReg, gradAlphaSumReg;
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
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradAlphaLoopSumReg, gradAlphaReg, maskReg);
            AscendC::MicroAPI::Add(gradAlphaSumReg, gradAlphaSumReg, gradAlphaLoopSumReg, OneMaskReg);
        }
        AscendC::MicroAPI::DataCopy(gradAlphaAddr, gradAlphaSumReg, OneMaskReg);
    }
}

// ============================================================
// ComputeZAndForwardPart: z = hcBeforeNorm * invRms + bias, normOut = sigmoid(z) * alpha
// ============================================================
template <typename U>
__aicore__ inline void ComputeZAndForwardPart(__local_mem__ U *zAddr, __local_mem__ U *normOutForwardAddr,
                                              __local_mem__ U *biasAddr, uint64_t bsLen, uint64_t colLen,
                                              uint64_t cLenAlign, uint64_t vRegSize, const U &alpha)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> zReg, normOutForwardReg;
        AscendC::MicroAPI::RegTensor<U> biasReg, invRmsReg, alphaReg, onesReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg allMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(onesReg, U(1), allMaskReg);
        MicroAPI::Duplicate(alphaReg, alpha, allMaskReg);
        MicroAPI::Duplicate(onesReg, U(1), allMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            AscendC::MicroAPI::AddrReg zOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg normOutForwardOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg biasOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::DataCopy(zReg, zAddr, zOfst);
            AscendC::MicroAPI::DataCopy(biasReg, biasAddr, biasOfst);
            for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::SCALAR_LOAD>();
                U invRms = *(zAddr + j * colLen);
                AscendC::MicroAPI::Duplicate(invRmsReg, invRms, allMaskReg);
                AscendC::MicroAPI::Mul(zReg, zReg, invRmsReg, maskReg);
                AscendC::MicroAPI::Add(zReg, zReg, biasReg, maskReg);
                AscendC::MicroAPI::Neg(zReg, zReg, maskReg);
                AscendC::MicroAPI::Exp(zReg, zReg, maskReg);
                AscendC::MicroAPI::Add(zReg, zReg, onesReg, maskReg);
                AscendC::MicroAPI::Div(zReg, onesReg, zReg, maskReg);
                AscendC::MicroAPI::Mul(normOutForwardReg, zReg, alphaReg, maskReg);
                AscendC::MicroAPI::DataCopy(zAddr + j * colLen, zReg, maskReg);
                AscendC::MicroAPI::DataCopy(normOutForwardAddr + j * colLen, normOutForwardReg, maskReg);
            }
        }
    }
}

// ============================================================
// ComputeForwardPart: z = hcBeforeNorm * invRms + bias（仅 z，不算 normOut）
// ============================================================
template <typename U>
__aicore__ inline void ComputeForwardPart(__local_mem__ U *hcBeformNormAddr, __local_mem__ U *invRmsAddr,
                                          __local_mem__ U *biasAddr, uint64_t bsLen, uint64_t colLen,
                                          uint64_t cLenAlign, uint64_t vRegSize)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> zReg, biasReg, invRmsReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg allMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            AscendC::MicroAPI::AddrReg zOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg biasOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::DataCopy(zReg, hcBeformNormAddr, zOfst);
            AscendC::MicroAPI::DataCopy(biasReg, biasAddr, biasOfst);
            for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::SCALAR_LOAD>();
                U invRms = *(invRmsAddr + j);
                AscendC::MicroAPI::Duplicate(invRmsReg, invRms, allMaskReg);
                AscendC::MicroAPI::Mul(zReg, zReg, invRmsReg, maskReg);
                AscendC::MicroAPI::Add(zReg, zReg, biasReg, maskReg);
                AscendC::MicroAPI::DataCopy(hcBeformNormAddr + j * colLen, zReg, maskReg);
            }
        }
    }
}

// ============================================================
// ComputeGradNormOutOrGradHcBeforeNorm: gradZ = scalar * gradZ
// ============================================================
template <typename T>
__aicore__ inline void ComputeGradNormOutOrGradHcBeforeNorm(const LocalTensor<T> &gradZLocal, T scalar, uint32_t count,
                                                            uint32_t offset)
{
    __local_mem__ T *addr = (__local_mem__ T *)gradZLocal.GetPhyAddr() + offset;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> gradZReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::UnalignReg u0;
        MicroAPI::DataCopyUnAlignPre(u0, addr);
        MicroAPI::DataCopyUnAlign(gradZReg, u0, addr, 0);
        MicroAPI::Muls(gradZReg, gradZReg, scalar, preg);
        MicroAPI::DataCopyUnAlign(addr, gradZReg, u0, count);
        MicroAPI::DataCopyUnAlignPost(addr, u0, 0);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

// ============================================================
// ComputeGradInvRms: gradInvRms = sum(gradNormOut * hcBeforeNorm)
// ============================================================
template <typename T>
__aicore__ inline void ComputeGradInvRms(const LocalTensor<T> &gradNormOutLocal,
                                         const LocalTensor<T> &hcBeforeNormLocal, int64_t count, T &gradInvRms)
{
    Mul(hcBeforeNormLocal, gradNormOutLocal, hcBeforeNormLocal, count);
    ReduceSum<T>(hcBeforeNormLocal, hcBeforeNormLocal, hcBeforeNormLocal, count);
    gradInvRms = hcBeforeNormLocal(0);
}

// ============================================================
// ComputeGradXFromRms: gradX = -invRms³ * x * gradInvRms / NC
// ============================================================
template <typename T>
__aicore__ inline void ComputeGradXFromRms(const LocalTensor<T> &xFp32Local, const LocalTensor<T> &xFp32LocalOut,
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

#endif // MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_BASE_H
