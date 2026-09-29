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
 * \file mhc_pre_sinkhorn_premix_backward_simt_vf.h
 * \brief mhc_pre_sinkhorn_premix_backward
 */

#include "kernel_operator.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"

using namespace AscendC;
namespace {
constexpr int32_t THREAD_NUM = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t ELEMENTS_PER_32B = 32 / sizeof(float);
}; // namespace

// skSumPreLocal/skNormPreLocal：调用方已把 skIterCount 轮迭代的 sk 数据批量预载到 UB，
// UB 内块 j 对应 GM 中第 j 个迭代块（即 SIMT 第 skIterCount-1-j 轮所用的数据），
// 每块布局为 [taskCount][n*n]（norm）与 [taskCount][n]（sum，row 段之后接 col 段）。
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SinkhornKnoppSimt(
    __ubuf__ T *gradHRes2Local, __ubuf__ T *gradInvRmsLocal, __gm__ T *gradAlphaGlobal, __gm__ T *gradBiasGlobal,
    __ubuf__ T *skSumPreLocal, __ubuf__ T *skNormPreLocal, __ubuf__ T *gradResLocal, __ubuf__ T *hRes2Local,
    __ubuf__ T *invRmsLocal, __ubuf__ T *biasLocal, __ubuf__ T *tmpLocal, const T hcScaleRes,
    const int32_t taskCount, const int32_t skIterCount, const int32_t n)
{
    int32_t xTid = threadIdx.x;   // 16
    int32_t hinTid = threadIdx.y; // THREAD_NUM // 16
    int32_t yBlockDim = blockDim.y;
    int32_t xBlockDim = blockDim.x;
    int32_t tid = hinTid * xBlockDim + xTid;
    int32_t taskWidth = n * n + 2 * n;

    // sinkhorn
    // DataCopyPad 多块搬运的 UB 侧块间距按 32B 对齐，sum 每块实际占 AlignUp(taskCount*n, 8) 个元素
    int32_t sumBlockPitch = (taskCount * n + ELEMENTS_PER_32B - 1) / ELEMENTS_PER_32B * ELEMENTS_PER_32B;
    int32_t colSumSegOffset = skIterCount * sumBlockPitch;
    for (int32_t iter = 0; iter < skIterCount; ++iter) {
        int32_t preBlock = skIterCount - 1 - iter;
        int32_t normBase = preBlock * taskCount * n * n;
        int32_t sumBase = preBlock * sumBlockPitch;
        for (int32_t i = hinTid; i < taskCount; i += yBlockDim) {
            int32_t hOffset = i * xBlockDim + xTid;

            T gradResVal = gradResLocal[hOffset];

            T skRowNormVal = skNormPreLocal[normBase + hOffset];
            T skRowSumVal = skSumPreLocal[sumBase + i * n + xTid / n];
            T skColSumVal = skSumPreLocal[colSumSegOffset + sumBase + i * n + xTid % n];

            T grad1 = (gradResVal * skRowNormVal / powf(skColSumVal, 2.0f));

            // reduce dim -2
            T colReduceSumVal = grad1 + asc_shfl_xor(grad1, 4, 16);
            colReduceSumVal = colReduceSumVal + asc_shfl_xor(colReduceSumVal, 8, 16);
            T gradRowNormed = gradResVal / skColSumVal - colReduceSumVal;

            // reduce dim -1
            T grad2 = (gradRowNormed * skRowNormVal / skRowSumVal);
            T rowReduceSumVal = grad2 + asc_shfl_xor(grad2, 1, 16);
            rowReduceSumVal = rowReduceSumVal + asc_shfl_xor(rowReduceSumVal, 2, 16);

            T gradSinkhorn = gradRowNormed / skRowSumVal - rowReduceSumVal;

            gradResLocal[hOffset] = gradSinkhorn;
        }
    }

    T alphaVal = hcScaleRes;
    T biasVal = biasLocal[xTid];

    for (int32_t i = hinTid; i < taskCount; i += yBlockDim) {
        int32_t hOffset = i * xBlockDim + xTid;
        T invRmsVal = invRmsLocal[i];
        T h2Val = hRes2Local[hOffset];
        T h2ValNormed = h2Val * invRmsVal;
        T gradSinkhorn = gradResLocal[hOffset];

        T hRes = h2ValNormed * alphaVal + biasVal;

        T hRes0 = asc_shfl(hRes, 0, 4);
        T hRes1 = asc_shfl(hRes, 1, 4);
        T hRes2 = asc_shfl(hRes, 2, 4);
        T hRes3 = asc_shfl(hRes, 3, 4);

        int32_t maxIdx01 = hRes0 >= hRes1 ? 0 : 1;
        T hResMaxVal01 = fmaxf(hRes0, hRes1);

        int32_t maxIdx012 = hResMaxVal01 >= hRes2 ? maxIdx01 : 2;
        T hResMaxVal012 = fmaxf(hResMaxVal01, hRes2);

        int32_t maxIdx0123 = hResMaxVal012 >= hRes3 ? maxIdx012 : 3;
        T hResMaxVal0123 = fmaxf(hResMaxVal012, hRes3);

        T mask = maxIdx0123 != (xTid % n) ? 1 : 0;
        T safeHRes = expf(hRes - hResMaxVal0123);
        T gradMax = safeHRes * mask * gradSinkhorn;

        T gradMaxReduceSum = gradMax + asc_shfl_xor(gradMax, 1, 16);
        gradMaxReduceSum = -(gradMaxReduceSum + asc_shfl_xor(gradMaxReduceSum, 2, 16));

        gradMaxReduceSum = mask == 1 ? safeHRes * gradSinkhorn : gradMaxReduceSum;

        // compute grad_invRms
        T gradInvRms = gradMaxReduceSum * alphaVal * h2Val;
        T gradInvRmsReduceSum = gradInvRms + asc_shfl_xor(gradInvRms, 1, 16);
        gradInvRmsReduceSum = gradInvRmsReduceSum + asc_shfl_xor(gradInvRmsReduceSum, 2, 16);
        gradInvRmsReduceSum = gradInvRmsReduceSum + asc_shfl_xor(gradInvRmsReduceSum, 4, 16);
        gradInvRmsReduceSum = gradInvRmsReduceSum + asc_shfl_xor(gradInvRmsReduceSum, 8, 16);

        // 暂存
        if (i == hinTid) {
            gradResLocal[tid] = gradMaxReduceSum;
            tmpLocal[tid] = gradMaxReduceSum * h2ValNormed;
        } else {
            Simt::AtomicAdd(gradResLocal + tid, gradMaxReduceSum);
            Simt::AtomicAdd(tmpLocal + tid, gradMaxReduceSum * h2ValNormed);
        }

        gradHRes2Local[i * taskWidth + n * 2 + xTid] = gradMaxReduceSum * alphaVal * invRmsVal;

        if (xTid == 0) {
            Simt::AtomicAdd(gradInvRmsLocal + hinTid, gradInvRmsReduceSum);
        }
    }

    __syncthreads();

    // ReduceSum
    int32_t reduceTaskCount = taskCount < yBlockDim ? taskCount : yBlockDim;
    int32_t s = (reduceTaskCount + 1) / 2;
    int32_t validTask = reduceTaskCount;
    for (; validTask > 1; validTask = (validTask + 1) / 2) {
        if (hinTid + s < validTask) {
            int32_t otherOffset = (hinTid + s) * xBlockDim + xTid;

            gradResLocal[tid] += gradResLocal[otherOffset];
            tmpLocal[tid] += tmpLocal[otherOffset];
            s = (s + 1) / 2;
        }
        __syncthreads();
    }

    T gradAlphaRes = 0;
    if (hinTid == 0) {
        // copyout gradHcBase
        Simt::AtomicAdd(gradBiasGlobal + xTid, gradResLocal[xTid]);
        gradAlphaRes = tmpLocal[xTid];
    }

    gradAlphaRes = asc_reduce_add(gradAlphaRes);
    if (tid == 0) {
        Simt::AtomicAdd(gradAlphaGlobal, gradAlphaRes);
    }
}
