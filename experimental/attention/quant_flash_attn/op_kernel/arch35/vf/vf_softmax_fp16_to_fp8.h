/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, and you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should have received a copy of the License along with this program.
 * If not, see <https://www.hiascend.com/legal/cannlicense>.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 * PURPOSE.
 */

#ifndef VF_SOFTMAX_FP16_TO_FP8_H_
#define VF_SOFTMAX_FP16_TO_FP8_H_

#include "kernel_tensor.h"
#include "vf_common_def.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief V1 softmax 主体（__simd_vf__ 内部函数，由 __aicore__ wrapper 调用）
 *
 * @param [out] pDst        NZ 源布局 P（vec1PUb[subLoopIdx]）
 * @param [in,out] accMax   accMax 槽（softmaxMaxUB[stateSlot]，half log2 整数）
 * @param [out] usedMaxOut  本 subLoop k_used（subLoopUsedMaxUB[subLoopIdx]）
 * @param [in] nzIdx        Gather 索引表（pNzIdxUB）
 * @param [in] x            mm1Res 输入（[validRows, colStride] half）
 * @param [in] pScale       外部 headroom 因子（half 标量，默认 1.0）
 * @param [in] validRows    有效 S2 行数（1..128）
 * @param [in] colStride    UB 行跨步 = 128（FixpipeMm1 固定 pitch）
 */
template <typename T = half, typename T2 = fp8_e4m3fn_t>
__simd_vf__ inline void VfSoftmaxFp16ToFp8VF(__ubuf__ T2 *pDst, __ubuf__ half *accMax,
                                               __ubuf__ half *usedMaxOut, __ubuf__ uint8_t *nzIdx,
                                               __ubuf__ T *x, half pScale, uint32_t validRows,
                                               uint32_t colStride)
{
    // TODO(批次3.2): 实现 §7.1 内部流程（归约/量化/exp/打包）
    (void)pDst; (void)accMax; (void)usedMaxOut; (void)nzIdx; (void)x;
    (void)pScale; (void)validRows; (void)colStride;
}

/*
 * @ingroup VfaVectorApi
 * @brief V1 softmax 主体（__aicore__ wrapper，接收 LocalTensor，内部转指针调 vf）
 *
 * 各参数的槽位偏移由本 wrapper 内部计算，编排层（ComputeVec1）只传 tensor + 索引。
 */
template <typename T = half, typename T2 = fp8_e4m3fn_t>
__aicore__ inline void VfSoftmaxFp16ToFp8(const LocalTensor<T2> &pDst, const LocalTensor<half> &accMax,
                                           const LocalTensor<half> &usedMaxOut,
                                           const LocalTensor<uint8_t> &nzIdx, const LocalTensor<T> &x,
                                           half pScale, uint32_t validRows, uint32_t subLoopIdx,
                                           uint32_t mmSlot, uint32_t stateSlot)
{
    constexpr uint32_t s2SubLoopSize = 128;
    constexpr uint32_t s1HalfSize = 128;
    constexpr uint32_t smaxSlot = 128;
    constexpr uint32_t mm1ResSlot = 128 * 128;
    constexpr uint32_t vec1PSlot = s2SubLoopSize * s1HalfSize;

    __ubuf__ T2 *pDstPtr = reinterpret_cast<__ubuf__ T2 *>(pDst.GetPhyAddr()) + subLoopIdx * vec1PSlot;
    __ubuf__ half *accMaxPtr = reinterpret_cast<__ubuf__ half *>(accMax.GetPhyAddr()) + stateSlot * smaxSlot;
    __ubuf__ half *usedMaxOutPtr = reinterpret_cast<__ubuf__ half *>(usedMaxOut.GetPhyAddr())
                                     + subLoopIdx * smaxSlot;
    __ubuf__ uint8_t *nzIdxPtr = reinterpret_cast<__ubuf__ uint8_t *>(nzIdx.GetPhyAddr());
    __ubuf__ T *xPtr = reinterpret_cast<__ubuf__ T *>(x.GetPhyAddr()) + mmSlot * mm1ResSlot;

    VfSoftmaxFp16ToFp8VF<T, T2>(pDstPtr, accMaxPtr, usedMaxOutPtr, nzIdxPtr, xPtr, pScale,
                                 validRows, s1HalfSize);
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_SOFTMAX_FP16_TO_FP8_H_
