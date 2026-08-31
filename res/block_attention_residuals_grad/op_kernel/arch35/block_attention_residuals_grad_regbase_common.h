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
 * \file block_attention_residuals_grad_regbase_common.h
 * \brief arch35 RegBase VF helpers for block_attention_residuals_grad
 */
#ifndef BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_COMMON_H
#define BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_COMMON_H

#include "kernel_operator.h"

namespace NsBlockAttentionResidualsGrad {
namespace RegBase {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t GRAD_ELEM_PER_BLK_FP32 = 8;
constexpr uint32_t GRAD_SCALAR_LOCAL_ELEMS = GRAD_ELEM_PER_BLK_FP32;

constexpr uint32_t kVlFp32 = AscendC::VECTOR_REG_WIDTH / static_cast<uint32_t>(sizeof(float));

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

__aicore__ inline uint32_t RoundUpFp32(uint32_t num)
{
    return CeilDivU32(num, GRAD_ELEM_PER_BLK_FP32) * GRAD_ELEM_PER_BLK_FP32;
}

/*!
 * Kahan compensated accumulation: sum += src * broadcast(scalarSrc[0]).
 * sum keeps the running sum, cmp keeps the compensation (fixed roles).
 */
__aicore__ inline void KahanFusedMulAddDual(const LocalTensor<float> &sum, const LocalTensor<float> &cmp,
                                            const LocalTensor<float> &src, const LocalTensor<float> &scalarSrc,
                                            uint32_t hiddenSize)
{
    __local_mem__ float *sAddr = (__local_mem__ float *)sum.GetPhyAddr();
    __local_mem__ float *cAddr = (__local_mem__ float *)cmp.GetPhyAddr();
    __local_mem__ float *xAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *brcAddr = (__local_mem__ float *)scalarSrc.GetPhyAddr();

    // The dual-half path is only enabled when both halves are whole vectors
    // (H is a multiple of 2*kVlFp32). Every other shape, including odd H and
    // partial vector tails, stays on the single masked path below.
    if ((hiddenSize & (2U * kVlFp32 - 1U)) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        const uint16_t repeatTimes = static_cast<uint16_t>(halfCount / kVlFp32);
        __local_mem__ float *sAddr2 = sAddr + halfCount;
        __local_mem__ float *cAddr2 = cAddr + halfCount;
        __local_mem__ float *xAddr2 = xAddr + halfCount;
        __VEC_SCOPE__
        {
            RegTensor<float> scReg;
            RegTensor<float> x0, x1, s0, s1, c0, c1, y0, y1, t0, t1;
            MaskReg mask = CreateMask<float, MaskPattern::ALL>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(scReg, brcAddr);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                DataCopy<float, LoadDist::DIST_NORM>(x0, xAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(x1, xAddr2 + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(s0, sAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(s1, sAddr2 + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(c0, cAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(c1, cAddr2 + i * kVlFp32);
                Mul(x0, x0, scReg, mask);
                Mul(x1, x1, scReg, mask);
                Sub(y0, x0, c0, mask);
                Sub(y1, x1, c1, mask);
                Add(t0, y0, s0, mask);
                Add(t1, y1, s1, mask);
                Sub(c0, t0, s0, mask);
                Sub(c1, t1, s1, mask);
                Sub(c0, c0, y0, mask);
                Sub(c1, c1, y1, mask);
                DataCopy(sAddr + i * kVlFp32, t0, mask);
                DataCopy(sAddr2 + i * kVlFp32, t1, mask);
                DataCopy(cAddr + i * kVlFp32, c0, mask);
                DataCopy(cAddr2 + i * kVlFp32, c1, mask);
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> scReg;
            RegTensor<float> x0, s0, c0, y0, t0;
            MaskReg mask;
            DataCopy<float, LoadDist::DIST_BRC_B32>(scReg, brcAddr);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(x0, xAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(s0, sAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(c0, cAddr + i * kVlFp32);
                Mul(x0, x0, scReg, mask);
                Sub(y0, x0, c0, mask);
                Add(t0, y0, s0, mask);
                Sub(c0, t0, s0, mask);
                Sub(c0, c0, y0, mask);
                DataCopy(sAddr + i * kVlFp32, t0, mask);
                DataCopy(cAddr + i * kVlFp32, c0, mask);
            }
        }
    }
}

/*!
 * Number of independent fp32 accumulators used by ReduceMulSum.
 * Chunk partial sums are spread across these accumulators and combined with a
 * pairwise tree at the end, so rounding error grows with ceil(repeatTimes/8)
 * sequential adds plus a 3-level tree instead of repeatTimes sequential adds.
 */
constexpr uint32_t REDUCE_TREE_WIDTH = 8;

/*! 8-way blocked tree accumulation for dot products. */
__aicore__ inline void ReduceMulSumTree(const LocalTensor<float> &dstScalar, const LocalTensor<float> &src0,
                                    const LocalTensor<float> &src1, uint32_t elemCount)
{
    __local_mem__ float *aAddr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *bAddr = (__local_mem__ float *)src1.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dstScalar.GetPhyAddr();
    uint32_t sreg = elemCount;
    const uint16_t repeatTimes = static_cast<uint16_t>((elemCount + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__
    {
        RegTensor<float> a, b, prod, part;
        RegTensor<float> acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
        MaskReg mask;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        Duplicate(acc0, 0.0f, maskAll);
        Duplicate(acc1, 0.0f, maskAll);
        Duplicate(acc2, 0.0f, maskAll);
        Duplicate(acc3, 0.0f, maskAll);
        Duplicate(acc4, 0.0f, maskAll);
        Duplicate(acc5, 0.0f, maskAll);
        Duplicate(acc6, 0.0f, maskAll);
        Duplicate(acc7, 0.0f, maskAll);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = UpdateMask<float>(sreg);
            DataCopy<float, LoadDist::DIST_NORM>(a, aAddr + i * kVlFp32);
            DataCopy<float, LoadDist::DIST_NORM>(b, bAddr + i * kVlFp32);
            Mul(prod, a, b, mask);
            ReduceSum(part, prod, mask);
            switch (i % REDUCE_TREE_WIDTH) {
                case 0: Add(acc0, acc0, part, maskAll); break;
                case 1: Add(acc1, acc1, part, maskAll); break;
                case 2: Add(acc2, acc2, part, maskAll); break;
                case 3: Add(acc3, acc3, part, maskAll); break;
                case 4: Add(acc4, acc4, part, maskAll); break;
                case 5: Add(acc5, acc5, part, maskAll); break;
                case 6: Add(acc6, acc6, part, maskAll); break;
                default: Add(acc7, acc7, part, maskAll); break;
            }
        }
        Add(acc0, acc0, acc4, maskAll);
        Add(acc1, acc1, acc5, maskAll);
        Add(acc2, acc2, acc6, maskAll);
        Add(acc3, acc3, acc7, maskAll);
        Add(acc0, acc0, acc2, maskAll);
        Add(acc1, acc1, acc3, maskAll);
        Add(acc0, acc0, acc1, maskAll);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, acc0, maskAll);
    }
}

/*! dst[0] = sum(src0 * src1), src buffers are not modified */
__aicore__ inline void ReduceMulSum(const LocalTensor<float> &dstScalar, const LocalTensor<float> &src0,
                                    const LocalTensor<float> &src1, uint32_t elemCount)
{
    ReduceMulSumTree(dstScalar, src0, src1, elemCount);
}

/*!
 * Kahan-compensated accumulation of a newly loaded src into a running sum.
 * sumTensorList[outPos] holds the running sum and the other slot the
 * compensation; roles swap after each update (same scheme as arch22).
 */
__aicore__ inline void KahanSumUpdate(LocalTensor<float> &inputTensor,
                                      LocalTensor<float> sumTensorList[2],
                                      int32_t len, int32_t &outPos)
{
    LocalTensor<float> sumTensor = sumTensorList[outPos];
    LocalTensor<float> eTensor = sumTensorList[1 - outPos];
    AscendC::Sub(inputTensor, inputTensor, eTensor, len);   // y = x - e
    AscendC::Add(eTensor, inputTensor, sumTensor, len);     // t = y + s
    AscendC::Sub(sumTensor, eTensor, sumTensor, len);       // e = t - s
    AscendC::Sub(sumTensor, sumTensor, inputTensor, len);   // e = (t - s) - y
    outPos = 1 - outPos;
}

/*! sum += src * scalarSrc[0]; plain accumulation to keep the A5 UB footprint small. */
__aicore__ inline void FusedMulAddAccumulate(const LocalTensor<float> &sum, const LocalTensor<float> &src,
                                             const LocalTensor<float> &scalarSrc, uint32_t elemCount)
{
    __local_mem__ float *sumAddr = (__local_mem__ float *)sum.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *scAddr = (__local_mem__ float *)scalarSrc.GetPhyAddr();
    uint32_t sreg = elemCount;
    const uint16_t repeatTimes = static_cast<uint16_t>((elemCount + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__
    {
        RegTensor<float> scReg, x0, s0, acc;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scReg, scAddr);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = UpdateMask<float>(sreg);
            DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
            DataCopy<float, LoadDist::DIST_NORM>(s0, sumAddr + i * kVlFp32);
            Mul(acc, x0, scReg, mask);
            Add(acc, acc, s0, mask);
            DataCopy(sumAddr + i * kVlFp32, acc, mask);
        }
    }
}

/*! Copy one float between compact meta slots without requiring 32B alignment. */
__aicore__ inline void CopyScalarToDense(const LocalTensor<float> &dst, const LocalTensor<float> &src)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> scalar;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalar, srcAddr);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, scalar, maskAll);
    }
}

/*!
 * gradV = gradHiddenState * prob + scoreWeight * sc + vRow * varScale
 * scalarBase[0] = prob, scalarBase[24] = sc, scalarBase[32] = varScale
 */
__aicore__ inline void FusedGradVF(const LocalTensor<float> &dst, const LocalTensor<float> &gradHiddenState,
                                   const LocalTensor<float> &vRow, const LocalTensor<float> &scoreWeight,
                                   const LocalTensor<float> &scalarBase, uint32_t hiddenSize)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *gradHiddenStateAddr = (__local_mem__ float *)gradHiddenState.GetPhyAddr();
    __local_mem__ float *vAddr = (__local_mem__ float *)vRow.GetPhyAddr();
    __local_mem__ float *swAddr = (__local_mem__ float *)scoreWeight.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalarBase.GetPhyAddr();

    // The dual-half path is only enabled when both halves are whole vectors
    // (H is a multiple of 2*kVlFp32). Every other shape, including odd H and
    // partial vector tails, stays on the single masked path below.
    if ((hiddenSize & (2U * kVlFp32 - 1U)) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        const uint16_t repeatTimes = static_cast<uint16_t>(halfCount / kVlFp32);
        __local_mem__ float *dstAddr2 = dstAddr + halfCount;
        __local_mem__ float *gradHiddenStateAddr2 = gradHiddenStateAddr + halfCount;
        __local_mem__ float *vAddr2 = vAddr + halfCount;
        __local_mem__ float *swAddr2 = swAddr + halfCount;
        __VEC_SCOPE__
        {
            RegTensor<float> probReg, scReg, varReg;
            RegTensor<float> gradHiddenState0, gradHiddenState1, v0, v1, sw0, sw1, acc0, acc1;
            MaskReg mask = CreateMask<float, MaskPattern::ALL>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(probReg, scalarAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(scReg, scalarAddr + 3U * GRAD_ELEM_PER_BLK_FP32);
            DataCopy<float, LoadDist::DIST_BRC_B32>(varReg, scalarAddr + 4U * GRAD_ELEM_PER_BLK_FP32);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                DataCopy<float, LoadDist::DIST_NORM>(gradHiddenState0, gradHiddenStateAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(gradHiddenState1, gradHiddenStateAddr2 + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(v0, vAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(v1, vAddr2 + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(sw0, swAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(sw1, swAddr2 + i * kVlFp32);
                Mul(acc0, gradHiddenState0, probReg, mask);
                MulAddDst(acc0, sw0, scReg, mask);
                MulAddDst(acc0, v0, varReg, mask);
                Mul(acc1, gradHiddenState1, probReg, mask);
                MulAddDst(acc1, sw1, scReg, mask);
                MulAddDst(acc1, v1, varReg, mask);
                DataCopy(dstAddr + i * kVlFp32, acc0, mask);
                DataCopy(dstAddr2 + i * kVlFp32, acc1, mask);
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> probReg, scReg, varReg;
            RegTensor<float> gradHiddenState0, v0, sw0, acc0;
            MaskReg mask;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(probReg, scalarAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(scReg, scalarAddr + 3U * GRAD_ELEM_PER_BLK_FP32);
            DataCopy<float, LoadDist::DIST_BRC_B32>(varReg, scalarAddr + 4U * GRAD_ELEM_PER_BLK_FP32);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(gradHiddenState0, gradHiddenStateAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(v0, vAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(sw0, swAddr + i * kVlFp32);
                Mul(acc0, gradHiddenState0, probReg, mask);
                MulAddDst(acc0, sw0, scReg, mask);
                MulAddDst(acc0, v0, varReg, mask);
                DataCopy(dstAddr + i * kVlFp32, acc0, mask);
            }
        }
    }
}

/*! gradScore = probs*gradProb - probs*scalar, matching _softmax_backward_data. */
__aicore__ inline void SoftmaxBackwardVf(const LocalTensor<float> &gradScore, const LocalTensor<float> &probs,
                                         const LocalTensor<float> &gradProb, const LocalTensor<float> &scalar,
                                         uint32_t blockCount)
{
    __local_mem__ float *gsAddr = (__local_mem__ float *)gradScore.GetPhyAddr();
    __local_mem__ float *pAddr = (__local_mem__ float *)probs.GetPhyAddr();
    __local_mem__ float *gpAddr = (__local_mem__ float *)gradProb.GetPhyAddr();
    __local_mem__ float *scalarAddr = (__local_mem__ float *)scalar.GetPhyAddr();
    uint32_t sreg = blockCount;
    const uint16_t repeatTimes = static_cast<uint16_t>((blockCount + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__
    {
        RegTensor<float> scalarReg, pReg, gReg, tmp, tmp2;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, scalarAddr);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = UpdateMask<float>(sreg);
            DataCopy<float, LoadDist::DIST_NORM>(pReg, pAddr + i * kVlFp32);
            DataCopy<float, LoadDist::DIST_NORM>(gReg, gpAddr + i * kVlFp32);
            Mul(tmp, pReg, gReg, mask);
            Mul(tmp2, pReg, scalarReg, mask);
            Sub(tmp, tmp, tmp2, mask);
            DataCopy(gsAddr + i * kVlFp32, tmp, mask);
        }
    }
}

} // namespace RegBase
} // namespace NsBlockAttentionResidualsGrad

#endif // BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_COMMON_H
