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
    int64_t rightPad = alignLen - elemPerRow;
    // UB 端 stride 单位是 dataBlock(32B)，GM 端 stride 单位是字节
    int64_t ubStrideBlocks = (rightPad * static_cast<int64_t>(sizeof(T))) / ONE_BLK_SIZE;

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerRow * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(srcStride);  // GM 端，字节
    copyParams.dstStride  = static_cast<uint32_t>(ubStrideBlocks);  // UB 端，dataBlock

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
    int64_t rightPad = alignLen - elemPerRow;
    int64_t ubStrideBlocks = (rightPad * static_cast<int64_t>(sizeof(T))) / ONE_BLK_SIZE;

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerRow * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(ubStrideBlocks);  // UB 端，dataBlock
    copyParams.dstStride  = static_cast<uint32_t>(dstStride);       // GM 端，字节
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
    int64_t rightPad = alignLen - elemPerRow;
    int64_t ubStrideBlocks = (rightPad * static_cast<int64_t>(sizeof(T))) / ONE_BLK_SIZE;

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerRow * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(srcStride);          // GM 端，字节
    copyParams.dstStride  = static_cast<uint32_t>(ubStrideBlocks);     // UB 端，dataBlock

    DataCopyPadExtParams<T> padParams{rightPad > 0, 0, static_cast<uint8_t>(rightPad), static_cast<T>(0)};
    DataCopyPad(local, srcGm, copyParams, padParams);
}

// ============================================================
// CopyOutBuf: Buf→GM
//
//   用法同 CopyOutQue, 传入 TBuf 替代 TQue
//   srcOffset: UB 端起始偏移(元素数), 用于逐行搬非32B对齐padding的场景
// ============================================================
template <typename T>
__aicore__ inline void CopyOutBuf(
    TBuf<TPosition::VECCALC> &buf,
    const GlobalTensor<T> &dstGm,
    int64_t rowCount,
    int64_t elemPerRow,
    int64_t ubAlignLen = 0,
    int64_t dstStride = 0,
    int64_t srcOffset = 0)
{
    auto local = buf.template Get<T>();
    int64_t alignLen = (ubAlignLen > 0) ? ubAlignLen : elemPerRow;
    int64_t rightPad = alignLen - elemPerRow;
    int64_t ubStrideBlocks = (rightPad * static_cast<int64_t>(sizeof(T))) / ONE_BLK_SIZE;

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowCount);
    copyParams.blockLen   = static_cast<uint32_t>(elemPerRow * sizeof(T));
    copyParams.srcStride  = static_cast<uint32_t>(ubStrideBlocks);
    copyParams.dstStride  = static_cast<uint32_t>(dstStride);
    DataCopyPad(dstGm, local[srcOffset], copyParams);
}

// ============================================================
// SigmoidGrad: σ'(z) = dy * σ(z) * (1 - σ(z))
//   ISPRE=true:  h_pre 分支，正向加了 hc_eps
//   ISPRE=false: h_post 分支，正向 ×2
// ============================================================
template <typename U, bool ISPRE>
__aicore__ inline void SigmoidGrad(__local_mem__ U *gradHAddr, __local_mem__ U *zAddr, __local_mem__ U *gradZAddr,
                                   uint64_t bsLen, uint64_t n, uint64_t rowStride, uint64_t vRegSize, float hcEps)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t vfLoopCnt = (n + vfLen - 1) / vfLen;
    float hcEpsNeg = hcEps * (-1);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> onesReg, zReg, gradHReg, sigmaReg, dysigmaReg, dysigma2Reg, gradZReg;
        AscendC::MicroAPI::MaskReg maskReg;
        uint32_t maskLen = static_cast<uint32_t>(n);
        for (uint16_t bs = 0; bs < static_cast<uint16_t>(bsLen); bs++) {
            __local_mem__ U *zRow = zAddr + bs * rowStride;
            __local_mem__ U *gradHRow = gradHAddr + bs * n;
            __local_mem__ U *gradZRow = gradZAddr + bs * rowStride;
            for (uint16_t i = 0; i < vfLoopCnt; i++) {
                maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                MicroAPI::Duplicate(onesReg, U(1), maskReg);
                AscendC::MicroAPI::AddrReg zOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
                AscendC::MicroAPI::AddrReg gradHOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
                AscendC::MicroAPI::AddrReg gradZOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
                AscendC::MicroAPI::DataCopy(zReg, zRow, zOfst);
                AscendC::MicroAPI::DataCopy(gradHReg, gradHRow, gradHOfst);
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
                AscendC::MicroAPI::DataCopy(gradZRow, gradZReg, gradZOfst, maskReg);
            }
        }
    }
}

template <typename U>
__aicore__ inline void ComputeGradAlphaAll(__local_mem__ U *normOutAddr, __local_mem__ U *gradZAddr,
                                           __local_mem__ U *gradAlphaAddr, uint64_t bsLen, uint64_t n,
                                           uint64_t kAlign, uint64_t vRegSize)
{
    uint64_t nAlign = (n * sizeof(U) + 31) / 32 * 32 / sizeof(U);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> noReg, gzReg, mulReg;
        AscendC::MicroAPI::RegTensor<U> sumPre, sumPost, sumRes, reduceReg;
        AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg oneMask = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t maskLenN = static_cast<uint32_t>(n);

        MicroAPI::Duplicate(sumPre, U(0), oneMask);
        MicroAPI::Duplicate(sumPost, U(0), oneMask);
        MicroAPI::Duplicate(sumRes, U(0), oneMask);
        maskN = AscendC::MicroAPI::UpdateMask<U>(maskLenN);

        for (uint16_t bs = 0; bs < static_cast<uint16_t>(bsLen); bs++) {
            __local_mem__ U *noBs = normOutAddr + bs * kAlign;
            __local_mem__ U *gzBs = gradZAddr + bs * kAlign;

            AscendC::MicroAPI::LoadAlign(noReg, noBs);
            AscendC::MicroAPI::LoadAlign(gzReg, gzBs);
            AscendC::MicroAPI::Mul(mulReg, noReg, gzReg, maskN);
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(reduceReg, mulReg, maskN);
            AscendC::MicroAPI::Add(sumPre, sumPre, reduceReg, oneMask);

            AscendC::MicroAPI::LoadAlign(noReg, noBs + nAlign);
            AscendC::MicroAPI::LoadAlign(gzReg, gzBs + nAlign);
            AscendC::MicroAPI::Mul(mulReg, noReg, gzReg, maskN);
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(reduceReg, mulReg, maskN);
            AscendC::MicroAPI::Add(sumPost, sumPost, reduceReg, oneMask);

            for (uint16_t row = 0; row < static_cast<uint16_t>(n); row++) {
                __local_mem__ U *noRow = noBs + (2 + row) * nAlign;
                __local_mem__ U *gzRow = gzBs + (2 + row) * nAlign;
                AscendC::MicroAPI::LoadAlign(noReg, noRow);
                AscendC::MicroAPI::LoadAlign(gzReg, gzRow);
                AscendC::MicroAPI::Mul(mulReg, noReg, gzReg, maskN);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(reduceReg, mulReg, maskN);
                AscendC::MicroAPI::Add(sumRes, sumRes, reduceReg, oneMask);
            }
        }
        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            gradAlphaAddr, sumPre, oneMask);
        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            gradAlphaAddr + 1, sumPost, oneMask);
        AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            gradAlphaAddr + 2, sumRes, oneMask);
    }
}

// ============================================================
// ComputeGradBiasAll: 逐行遍历 (2+n) 行, BS 累加, 输出连续 (2+n)*nAlign
// ============================================================
template <typename U>
__aicore__ inline void ComputeGradBiasAll(__local_mem__ U *gradZAddr, uint64_t bsLen, uint64_t n,
                                          uint64_t kAlign, uint64_t vRegSize,
                                          __local_mem__ U *gradBiasOut)
{
    uint64_t nAlign = (n * sizeof(U) + 31) / 32 * 32 / sizeof(U);
    uint64_t kRows = 2 + n;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> gzReg, biasReg;
        AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t maskLenN = static_cast<uint32_t>(n);
        maskN = AscendC::MicroAPI::UpdateMask<U>(maskLenN);

        for (uint16_t row = 0; row < static_cast<uint16_t>(kRows); row++) {
            __local_mem__ U *biasRow = gradBiasOut + row * nAlign;
            MicroAPI::Duplicate(biasReg, U(0), maskN);
            for (uint16_t bs = 0; bs < static_cast<uint16_t>(bsLen); bs++) {
                __local_mem__ U *gzRow = gradZAddr + bs * kAlign + row * nAlign;
                AscendC::MicroAPI::LoadAlign(gzReg, gzRow);
                AscendC::MicroAPI::Add(biasReg, biasReg, gzReg, maskN);
            }
            AscendC::MicroAPI::StoreAlign(biasRow, biasReg, maskN);
        }
    }
}

// ============================================================
// ComputeZAndForwardPart
//   normOut = hc * invRms
//   z       = normOut * alpha + bias
//
//   UB 布局: [bsLen, rowStride], 尾轴 colLen 对齐到 nAlign
//   hc/bias: 输入 (bias 每 BS 共用同一份)
//   z/normOut: 输出
// ============================================================
template <typename U>
__aicore__ inline void ComputeZAndForwardPart(__local_mem__ U *zAddr, __local_mem__ U *normOutAddr,
                                              __local_mem__ U *hcAddr, __local_mem__ U *biasAddr,
                                              __local_mem__ U *invRmsAddr,
                                              uint64_t bsLen, uint64_t colLen, uint64_t rowStride,
                                              uint64_t vRegSize, const U &alpha)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t vfLoopCnt = static_cast<uint16_t>((colLen + vfLen - 1) / vfLen);
    uint32_t maskLen = static_cast<uint32_t>(colLen);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> hcReg, zReg, biasReg, normOutReg;
        AscendC::MicroAPI::RegTensor<U> invRmsReg, alphaReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(alphaReg, alpha, allMask);

        for (uint16_t bs = 0; bs < static_cast<uint16_t>(bsLen); bs++) {
            __local_mem__ U *hcBase = hcAddr + bs * rowStride;
            __local_mem__ U *zBase  = zAddr + bs * rowStride;
            __local_mem__ U *noBase = normOutAddr + bs * rowStride;

            MicroAPI::Duplicate(invRmsReg, invRmsAddr[bs], allMask);

            for (uint16_t vf = 0; vf < vfLoopCnt; vf++) {
                maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::AddrReg ofs = AscendC::MicroAPI::CreateAddrReg<U>(vf, vfLen);

                AscendC::MicroAPI::DataCopy(hcReg, hcBase, ofs);
                AscendC::MicroAPI::DataCopy(biasReg, biasAddr, ofs);

                // normOut = hc * invRms
                AscendC::MicroAPI::Mul(normOutReg, hcReg, invRmsReg, maskReg);
                // z = normOut * alpha + bias
                AscendC::MicroAPI::Mul(zReg, normOutReg, alphaReg, maskReg);
                AscendC::MicroAPI::Add(zReg, zReg, biasReg, maskReg);

                AscendC::MicroAPI::DataCopy(zBase + vf * vfLen, zReg, maskReg);
                AscendC::MicroAPI::DataCopy(noBase + vf * vfLen, normOutReg, maskReg);
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
