/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_init_only.h
 * \brief
 *
 * init-only 头开销测量算子的 kernel Init 实现（对照 Add 的"重" Init）。
 * 有意用普适原语构造一套显著重于 Add 的初始化：
 *   ① 4 路 GM SetGlobalBuffer(x / weight / group_list / y)；
 *   ② group_list 经 DataCopyPad 拷入 UB + 标量循环 G 次解析 per-group m 偏移；
 *   ③ 多块 L1 / UB InitBuffer（模拟 cube L1A / L1B / L0C 的资源初始化，size>0 且在容量内）；
 *   ④ 访问 host 传入的大 tilingData 结构体（模拟 TCubeTiling），做标量累加并落 UB。
 * 全程不调用任何 Iterate / mma / matmul，Init 完成即返回。
 */
#ifndef GROUPED_MATMUL_INIT_ONLY_H
#define GROUPED_MATMUL_INIT_ONLY_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "grouped_matmul_init_only_tiling_data.h"
#include "grouped_matmul_init_only_tiling_key.h"

namespace NsGmmInitOnly {

using namespace AscendC;

// 各资源块字节数（size>0 且在 L1/UB 容量内）
constexpr uint32_t GMM_L1A_BYTES = 64 * 1024;   // 模拟 cube L1A
constexpr uint32_t GMM_L1B_BYTES = 64 * 1024;   // 模拟 cube L1B
constexpr uint32_t GMM_UB_SCRATCH_BYTES = 16 * 1024;

template <typename T>
class GroupedMatmulInitOnly
{
public:
    __aicore__ inline GroupedMatmulInitOnly(){};

    // init-only：仅实现 Init，不实现 Process/Iterate
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR y,
        const GroupedMatmulInitOnlyTilingData* tilingData);

private:
    TPipe pipe;
    // group_list 拷入 UB 使用的队列
    TQue<QuePosition::VECIN, 1> groupListQueue;
    // 模拟 cube 的 L1A / L1B 资源
    TBuf<TPosition::A1> l1aBuf;
    TBuf<TPosition::B1> l1bBuf;
    // UB 上的 scratch（模拟 UB 侧初始化 + 落标量结果，防止被优化掉）
    TBuf<TPosition::VECCALC> ubScratch0;
    TBuf<TPosition::VECCALC> ubScratch1;

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> yGm;
    GlobalTensor<int64_t> groupListGm;
};

template <typename T>
__aicore__ inline void GroupedMatmulInitOnly<T>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR y,
    const GroupedMatmulInitOnlyTilingData* tilingData)
{
    // ---- 从 tilingData 取基本 shape ----
    int64_t mDim = tilingData->M;
    int64_t kDim = tilingData->K;
    int64_t nDim = tilingData->N;
    int64_t gDim = tilingData->G;
    if (gDim <= 0) {
        gDim = 1;
    }
    if (gDim > GMM_INIT_ONLY_MAX_G) {
        gDim = GMM_INIT_ONLY_MAX_G;
    }

    // ---- ① 4 路 GM SetGlobalBuffer ----
    int64_t xCount = mDim * kDim;
    int64_t weightCount = gDim * kDim * nDim;
    int64_t yCount = mDim * nDim;
    if (xCount <= 0) {
        xCount = 1;
    }
    if (weightCount <= 0) {
        weightCount = 1;
    }
    if (yCount <= 0) {
        yCount = 1;
    }
    xGm.SetGlobalBuffer((__gm__ T*)x, xCount);
    weightGm.SetGlobalBuffer((__gm__ T*)weight, weightCount);
    yGm.SetGlobalBuffer((__gm__ T*)y, yCount);
    groupListGm.SetGlobalBuffer((__gm__ int64_t*)group_list, gDim);

    // ---- ③ 多块 L1 / UB InitBuffer（资源初始化） ----
    pipe.InitBuffer(l1aBuf, GMM_L1A_BYTES);
    pipe.InitBuffer(l1bBuf, GMM_L1B_BYTES);
    pipe.InitBuffer(ubScratch0, GMM_UB_SCRATCH_BYTES);
    pipe.InitBuffer(ubScratch1, GMM_UB_SCRATCH_BYTES);
    pipe.InitBuffer(groupListQueue, 1, GMM_INIT_ONLY_MAX_G * sizeof(int64_t));

    // ---- ② group_list 拷入 UB + 标量循环解析 per-group m 偏移 ----
    LocalTensor<int64_t> glLocal = groupListQueue.AllocTensor<int64_t>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(gDim * sizeof(int64_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
    DataCopyPad(glLocal, groupListGm, copyParams, padParams);
    groupListQueue.EnQue(glLocal);

    LocalTensor<int64_t> glIn = groupListQueue.DeQue<int64_t>();
    int64_t cumOffset = 0;
    for (int64_t i = 0; i < gDim; i++) {
        int64_t gm = glIn.GetValue(i);
        cumOffset += gm;
    }
    groupListQueue.FreeTensor(glIn);

    // ---- ④ 访问大 tilingData 结构体，标量累加并落 UB（防优化） ----
    int64_t acc = mDim + kDim + nDim + gDim + cumOffset;
    acc += tilingData->baseM + tilingData->baseN + tilingData->baseK;
    acc += tilingData->singleCoreM + tilingData->singleCoreN + tilingData->singleCoreK;
    acc += tilingData->stepM + tilingData->stepN + tilingData->stepKa + tilingData->stepKb;
    acc += tilingData->depthA1 + tilingData->depthB1 + tilingData->iterateOrder;
    acc += tilingData->usedCoreNum + tilingData->coreNum;
    acc += tilingData->ubSize + tilingData->l1Size;
    acc += tilingData->l0cSize + tilingData->l0aSize + tilingData->l0bSize;
    acc += tilingData->mTiles + tilingData->nTiles + tilingData->kTiles;
    for (int64_t i = 0; i < gDim; i++) {
        acc += tilingData->groupList[i] + tilingData->groupMOffset[i];
    }

    LocalTensor<int64_t> scratch = ubScratch0.Get<int64_t>();
    scratch.SetValue(0, acc);
    // init-only：初始化完成即返回，不进入任何 Iterate / mma
}

} // namespace NsGmmInitOnly
#endif // GROUPED_MATMUL_INIT_ONLY_H
