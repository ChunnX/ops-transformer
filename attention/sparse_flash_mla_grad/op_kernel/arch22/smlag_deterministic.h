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
 * \file smlag_deterministic.h
 * \brief SparseFlashMlaGrad 确定性计算通路（Deterministic process path）
 *
 * 本文件集中承载“确定性 process 通路”的保序累加逻辑，避免散落在主编排器中：
 *   - ReduceDsinks : d_sinks 多核保序归约（替代跨核 atomicAdd）
 *   - ReduceDkv    : d_ori_kv / d_cmp_kv 多核保序归约（替代跨核 atomicAdd）
 *
 * dKV 思路：cube4/cube5（及 SCFA 的 ScatterAdd）在确定性模式下不再跨核 atomicAdd 写入共享
 * dkWorkspace，而是各核写入自己的私有累加区 dkvDetWs[c]（片内 atomic，顺序确定）。主循环 +
 * SyncAll 之后，ReduceDkv 按固定核序 c=0..usedCoreNum-1 把各核私有区合并写回 dkWorkspace
 * （main + additional 连续布局，与 Post 读取位置完全一致），Post 逻辑无需改动。
 *
 * cmp_softmax_l1_norm 的保序为“同一 block 内 aiv0/aiv1”的局部合并，强耦合于
 * 单 task 计算上下文，故其保序在 VecOp::Process 中就地完成（见 vec_op.h），此处不再重复。
 *
 * 设计约定见 docs/smlag_determinism_design.md。
 */

#pragma once
#include "kernel_operator.h"
#include "basic_modules/sparse_flash_mla_grad_common_header.h"

using namespace AscendC;

namespace SMLAG_DET {
using namespace SMLAG_BASIC;

/*
 * 确定性收尾处理器：仅在 AIV 上、主循环 + SyncAll 之后执行。
 * 负责把各核写入私有槽位的部分梯度，按固定核序保序合并为最终结果，消除浮点累加顺序不确定。
 */
template <typename TILING_CLASS>
class SmlagDetPost {
public:
    __aicore__ inline SmlagDetPost(){};

    __aicore__ inline void Init(GM_ADDR dsinks, GM_ADDR workspace,
                                const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);

    // d_sinks：把 [aivNum, G] 槽位按 c=0..aivNum-1 固定核序累加为 [G]，单写者写回。
    __aicore__ inline void ReduceDsinks();

    // dKV：把 [usedCoreNum, stride] 私有累加区按 c=0..usedCoreNum-1 固定核序合并写回 dkWorkspace。
    __aicore__ inline void ReduceDkv();

private:
    const TILING_CLASS *__restrict tilingData;
    TPipe *pipe;

    GlobalTensor<float> dsinksDetWsGm;   // [aivNum, G]，各核独占槽位
    GlobalTensor<float> dSinksGm;        // [N2 * G]

    GlobalTensor<float> dkvDetWsGm;      // [usedCoreNum, dkvStrideF]，各核私有累加区
    GlobalTensor<float> dkWorkspaceGm;   // dKV 归约目标（与 Post 读取基址一致）

    TBuf<> ubBuf;

    int64_t dimG;
    int64_t dimN2;
    int64_t aivNum;
    int64_t usedCoreNum;
    int64_t dkvStrideF;                  // 每核步长(float) = main + additional
    int32_t vecBlockIdx;

    constexpr static int32_t BLOCK_FP32 = 32 / sizeof(float);
    constexpr static uint32_t DET_UB_BYTES = 16 * 1024;
    constexpr static int64_t DKV_CHUNK_FP32 = 2048;   // 8KB/槽，acc + tmp 共 16KB

    event_t vWaitMte2;
    event_t mte3WaitV;
};

template <typename TILING_CLASS>
__aicore__ inline void SmlagDetPost<TILING_CLASS>::Init(GM_ADDR dsinks, GM_ADDR workspace,
                                                        const TILING_CLASS *__restrict ordTilingData, TPipe *pipeIn)
{
    tilingData = ordTilingData;
    pipe = pipeIn;
    vecBlockIdx = GetBlockIdx();

    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    usedCoreNum = static_cast<int64_t>(tilingData->opInfo.usedCoreNum);
    aivNum = usedCoreNum * 2;
    dkvStrideF = static_cast<int64_t>(tilingData->opInfo.dkvDetStride) / static_cast<int64_t>(sizeof(float));

    dSinksGm.SetGlobalBuffer((__gm__ float *)dsinks);
    dsinksDetWsGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->opInfo.dsinksDetWsOffset / sizeof(float));
    dkvDetWsGm.SetGlobalBuffer((__gm__ float *)workspace +
                               tilingData->opInfo.dkvDetWsOffset / sizeof(float));
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));

    pipe->InitBuffer(ubBuf, DET_UB_BYTES);
    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
}

template <typename TILING_CLASS>
__aicore__ inline void SmlagDetPost<TILING_CLASS>::ReduceDsinks()
{
    int64_t totalG = dimN2 * dimG;
    int64_t reduceCoreNum = Min(aivNum, totalG);   // 参与归约的核数（按 G 切片）
    if (vecBlockIdx >= reduceCoreNum) {
        return;
    }

    int64_t perCore = (totalG + reduceCoreNum - 1) / reduceCoreNum;
    int64_t gBegin = static_cast<int64_t>(vecBlockIdx) * perCore;
    int64_t gEnd = Min(gBegin + perCore, totalG);
    if (gBegin >= gEnd) {
        return;
    }
    int64_t len = gEnd - gBegin;
    int64_t lenAlign = AlignTo<int64_t>(len, static_cast<int64_t>(BLOCK_FP32));

    LocalTensor<float> acc = ubBuf.GetWithOffset<float>(lenAlign, 0);
    LocalTensor<float> tmp = ubBuf.GetWithOffset<float>(lenAlign, lenAlign * sizeof(float));

    DataCopyExtParams cpParam{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParam{false, 0, 0, 0};

    // c=0 初始化
    DataCopyPad(acc, dsinksDetWsGm[gBegin], cpParam, padParam);
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    // c=1..aivNum-1 按固定核序累加 -> 顺序确定
    for (int64_t c = 1; c < aivNum; ++c) {
        DataCopyPad(tmp, dsinksDetWsGm[c * totalG + gBegin], cpParam, padParam);
        SET_FLAG(MTE2, V, vWaitMte2);
        WAIT_FLAG(MTE2, V, vWaitMte2);
        Add(acc, acc, tmp, len);
        PIPE_BARRIER(PIPE_V);
    }

    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);
    // 单写者写回（owner 独占该 G 段），无原子
    DataCopyPad(dSinksGm[gBegin], acc, {1, static_cast<uint32_t>(len * sizeof(float)), 0, 0});
}

template <typename TILING_CLASS>
__aicore__ inline void SmlagDetPost<TILING_CLASS>::ReduceDkv()
{
    int64_t total = dkvStrideF;                      // 每核私有区总长(main + additional)
    if (total <= 0 || usedCoreNum <= 0) {
        return;
    }
    // 按元素切片到各 AIV，owner 独占连续区间 -> 写回无原子且核序固定
    int64_t reduceCoreNum = Min(aivNum, total);
    if (vecBlockIdx >= reduceCoreNum) {
        return;
    }
    int64_t perCore = (total + reduceCoreNum - 1) / reduceCoreNum;
    int64_t begin = static_cast<int64_t>(vecBlockIdx) * perCore;
    int64_t end = Min(begin + perCore, total);
    if (begin >= end) {
        return;
    }

    LocalTensor<float> acc = ubBuf.GetWithOffset<float>(DKV_CHUNK_FP32, 0);
    LocalTensor<float> tmp = ubBuf.GetWithOffset<float>(DKV_CHUNK_FP32, DKV_CHUNK_FP32 * sizeof(float));
    DataCopyPadExtParams<float> padParam{false, 0, 0, 0};

    for (int64_t off = begin; off < end; off += DKV_CHUNK_FP32) {
        int64_t len = Min(DKV_CHUNK_FP32, end - off);
        DataCopyExtParams cpParam{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};

        // c=0 初始化
        DataCopyPad(acc, dkvDetWsGm[off], cpParam, padParam);
        SET_FLAG(MTE2, V, vWaitMte2);
        WAIT_FLAG(MTE2, V, vWaitMte2);

        // c=1..usedCoreNum-1 按固定核序累加 -> 顺序确定
        for (int64_t c = 1; c < usedCoreNum; ++c) {
            DataCopyPad(tmp, dkvDetWsGm[c * total + off], cpParam, padParam);
            SET_FLAG(MTE2, V, vWaitMte2);
            WAIT_FLAG(MTE2, V, vWaitMte2);
            Add(acc, acc, tmp, len);
            PIPE_BARRIER(PIPE_V);
        }

        SET_FLAG(V, MTE3, mte3WaitV);
        WAIT_FLAG(V, MTE3, mte3WaitV);
        // 单写者写回（owner 独占该区间），无原子；与 Post 读取布局一致
        DataCopyPad(dkWorkspaceGm[off], acc, cpParam);
        SET_FLAG(MTE3, MTE2, vWaitMte2);
        WAIT_FLAG(MTE3, MTE2, vWaitMte2);
    }
}

} // namespace SMLAG_DET
