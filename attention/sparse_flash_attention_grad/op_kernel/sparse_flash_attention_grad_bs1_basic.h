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
 * \file sparse_flash_attention_grad_basic.h
 * \brief
 */

#pragma once
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "../basic_modules/cube_op.h"
#include "../basic_modules/vec_op.h"
#include "../basic_modules/common_header.h"
#include "sparse_flash_attention_grad_post.h"

namespace SFAG_BASIC {

template <typename SFAGT>
class SelectedAttentionGradBasic {
    using TILING_CLASS = typename SFAGT::tiling_class;
    using T1 = typename SFAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = SFAGT::atten_enable;
    static constexpr bool HAS_ROPE = SFAGT::has_rope;
    static constexpr bool IS_BSND = SFAGT::is_bsnd;

public:
    __aicore__ inline SelectedAttentionGradBasic(){};
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                   GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                   GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                   GM_ADDR query_rope, GM_ADDR key_rope,
                                   GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, 
                                   GM_ADDR workspace, const TILING_CLASS *__restrict tilingData);

private:
    __aicore__ inline void Init(const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void CubeCompute(CubeOp<SFAGT> &cubeOp);
    __aicore__ inline void VecCompute(VecOp<SFAGT> &vecOp);
    __aicore__ inline void UpdateGmOffset();
    __aicore__ inline void SaveLastInfo();
    __aicore__ inline void GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr, const GM_ADDR actual_seq_kvlen_addr,
                                        const int64_t t1Idx, int64_t &bIdx);
    __aicore__ inline void GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int32_t &actSelBlkCount);

    uint32_t cubeBlockIdx;
    uint32_t subBlockIdx;
    uint32_t formerCoreNum;
    uint32_t processBS1ByCore;
    uint32_t usedCoreNum;
    // shape info
    int64_t dimG;
    int64_t dimN1;
    int64_t dimN2;
    int64_t dimDqk;
    int64_t dimDv;
    int64_t dimRope;
    int64_t t1Offset;
    int64_t t2Offset{0};
    int64_t curS1;
    int64_t curS2;
    int64_t curMaxS2;
    int64_t dimS1;
    // attr
    uint32_t selectedBlockCount;
    // gmoffset
    uint64_t queryGmOffset;
    uint64_t queryRopeGmOffset;
    uint64_t dyGmOffset;
    uint64_t indicesGmOffset;
    uint64_t sumGmOffset;
    uint64_t keyGmOffset;
    uint64_t keyRopeGmOffset;
    uint64_t valueGmOffset;
    uint64_t mm12GmOffset;
    uint64_t mm345GmOffset;
    uint64_t selectedKGmOffset;
    uint64_t selectedVGmOffset;
    int32_t blkCntOffset;

    uint64_t lastQueryGmOffset;
    uint64_t lastQueryRopeGmOffset;
    uint64_t lastKeyGmOffset;
    uint64_t lastValueGmOffset;
    uint64_t lastDyGmOffset;
    uint64_t lastIndicesGmOffset;
    uint64_t lastmm12GmOffset;
    uint64_t lastmm345GmOffset;
    int32_t lastblkCntOffset;
    // workspace
    uint32_t selectedKWorkspaceLen;
    uint32_t selectedVWorkspaceLen;
    uint32_t mm12WorkspaceLen;
    uint32_t mm345WorkspaceLen;
    // Index
    int64_t bIndex{0};
    int64_t s1Index{0};
    int64_t n2Index{0};
    int64_t loopCnt{0};
    // 地址相关
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};
    uint32_t mmPingPongIdx{0};
    uint32_t selectdKPPPidx{0};
    constexpr static const int32_t BLOCK_FP32 = 32 / sizeof(float);
    // selectBlock相关
    int32_t selectedCountOffset{0};
    int32_t actualSelectedBlockCount{0};
    int32_t selectedBlockSize{0};
    // flag
    constexpr static uint32_t CUBE_WAIT_VEC = 0;
    constexpr static uint32_t VEC_WAIT_CUBE = 1;
    constexpr static uint32_t POST_WAIT_CUBE = 2;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER = 3;
    constexpr static uint32_t eventIdPing = 4;
    constexpr static uint32_t eventIdPong = 5;

    RunInfo runInfo[2];
    // gm
    GlobalTensor<int32_t> topkIndicesGm;
};

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::Init(const TILING_CLASS *__restrict tilingData)
{
    dimS1 = tilingData->opInfo.S1;
    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    dimN1 = dimG * dimN2;
    dimDqk = tilingData->opInfo.D;
    dimDv = tilingData->opInfo.D2;
    dimRope = tilingData->opInfo.ropeD;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(float);
    mm345WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(T1);
    if constexpr (IS_BSND == true) {
        curS1 = tilingData->opInfo.S1;
        curS2 = tilingData->opInfo.S2;
    }

    if ASCEND_IS_AIC {
        cubeBlockIdx = GetBlockIdx();
    }
    if ASCEND_IS_AIV {
        cubeBlockIdx = GetBlockIdx() / 2;
        subBlockIdx = GetBlockIdx() % 2;
    }

    formerCoreNum = tilingData->opInfo.formerCoreNum;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    if (cubeBlockIdx < formerCoreNum) {
        processBS1ByCore = tilingData->opInfo.formerCoreProcessNNum;
    } else {
        processBS1ByCore = tilingData->opInfo.remainCoreProcessNNum;
    }

    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    selectedVWorkspaceLen = tilingData->opInfo.selectedVWorkspaceLen;
    selectedVWorkspaceLen = tilingData->opInfo.selectedVWorkspaceLen;
    selectedKWspOffset = selectedKWorkspaceLen / sizeof(T1) / 3;
    selectedVWspOffset = selectedVWorkspaceLen / sizeof(T1) / 2;
    selectedCountOffset = 512 / tilingData->opInfo.selectedBlockSize;
    if (tilingData->opInfo.selectedBlockSize * tilingData->opInfo.selectedBlockCount <= 512) {
        selectedCountOffset = tilingData->opInfo.selectedBlockCount;
    }
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::Process(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
    GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, 
    GM_ADDR workspace, const TILING_CLASS *__restrict tilingData)
{
    Init(tilingData);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);

    // AIC Process
    if ASCEND_IS_AIC {
        TPipe pipeCube;
        CubeOp<SFAGT> cubeOp;
        cubeOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                    actual_seq_qlen, actual_seq_kvlen, query_rope, dq, dk, dv, workspace, tilingData, &pipeCube);
        AllocEventID();
        int64_t task = 0;
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            int32_t t1Index = cubeBlockIdx + usedCoreNum * i;
            GetTndSeqLen(actual_seq_qlen, actual_seq_kvlen, t1Index, bIndex);
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                GetActualSelCount(t1Index, n2Index, actualSelectedBlockCount);
                for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount; blkCntOffset += selectedCountOffset) {
                    UpdateGmOffset();
                    CubeCompute(cubeOp);
                    task++;
                }
            }
        }

        if (cubeBlockIdx < usedCoreNum && task > 0) {
            CrossCoreWaitFlag<2, PIPE_FIX>(CUBE_WAIT_VEC);
            cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
            CrossCoreSetFlag<2, PIPE_FIX>(POST_WAIT_CUBE);
        }
        FreeEventID();
    }

    // AIV Process
    if ASCEND_IS_AIV {
        TPipe pipeVec;
        VecOp<SFAGT> vecOp;
        vecOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                   actual_seq_qlen, actual_seq_kvlen, key_rope, dq, dk, dv, workspace, tilingData, &pipeVec);
        int64_t task = 0;
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            int32_t t1Index = cubeBlockIdx + usedCoreNum * i;
            GetTndSeqLen(actual_seq_qlen, actual_seq_kvlen, t1Index, bIndex);
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                GetActualSelCount(t1Index, n2Index, actualSelectedBlockCount);
                for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount; blkCntOffset += selectedCountOffset) {
                    UpdateGmOffset();
                    VecCompute(vecOp);
                    task++;
                }
            }
        }
        if (cubeBlockIdx < usedCoreNum && task > 0) {
            CrossCoreWaitFlag<2, PIPE_FIX>(POST_WAIT_CUBE);
        }
        SyncAll();
        pipeVec.Destroy();

        TPipe pipeCast;
        SparseFlashAttentionGradPost<T1, TILING_CLASS, true, 3, 0, HAS_ROPE> opCast;
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, dq_rope, dk_rope, workspace, tilingData, &pipeCast);
        opCast.Process();
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::CubeCompute(CubeOp<SFAGT> &cubeOp)
{
    // WaitVec for select & gather
    CrossCoreWaitFlag<2, PIPE_MTE3>(CUBE_WAIT_VEC_GATHER);
    if (unlikely(loopCnt == 0)) {
        cubeOp.cube12Process(runInfo[mmPingPongIdx], blkCntOffset, mmPingPongIdx);
        CrossCoreSetFlag<2, PIPE_FIX>(VEC_WAIT_CUBE);
        SaveLastInfo();
        return;
    }
    cubeOp.cube12Process(runInfo[mmPingPongIdx], blkCntOffset, mmPingPongIdx);
    CrossCoreSetFlag<2, PIPE_FIX>(VEC_WAIT_CUBE);
    CrossCoreWaitFlag<2, PIPE_FIX>(CUBE_WAIT_VEC);
    cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
    SaveLastInfo();
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::VecCompute(VecOp<SFAGT> &vecOp)
{
    if (subBlockIdx == 0) {
        vecOp.GatherKV(blkCntOffset, s1Index, n2Index, t1Offset, selectedKGmOffset, selectedVGmOffset, keyGmOffset, valueGmOffset, keyRopeGmOffset, curS1, curS2);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(CUBE_WAIT_VEC_GATHER);

    CrossCoreWaitFlag(VEC_WAIT_CUBE);
    if (subBlockIdx == 0) {
        vecOp.Process(dyGmOffset, sumGmOffset, indicesGmOffset, s1Index, blkCntOffset, mm12GmOffset, mm345GmOffset, runInfo[mmPingPongIdx].lastBlockSize, runInfo[mmPingPongIdx].isLastBasicBlock);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(CUBE_WAIT_VEC);
    mmPingPongIdx = 1 - mmPingPongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 3;
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::UpdateGmOffset()
{
    /*
     *  query:    B S1 N2 G D
     *  dy/out:   B S1 N2 G D2
     *  indices:  B S1 N2 SELCTED_BLOCK_COUNT
     *  sum/max   B S1 N2 G 8 --> N2 T1 G or B N2 S1 G
     *  key:      B S2 N2 D
     *  value:    B S2 N2 D2
     */
    if constexpr (IS_BSND) {
        queryGmOffset = t1Offset * (dimN1 * dimDqk) + n2Index * (dimG * dimDqk);
        queryRopeGmOffset = t1Offset * (dimN1 * dimRope) + n2Index * (dimG * dimRope);
        dyGmOffset = t1Offset * (dimN1 * dimDv) + n2Index * (dimG * dimDv);
        indicesGmOffset = t1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount;
        sumGmOffset = bIndex * dimN2 * dimS1 * dimG + n2Index * dimS1 * dimG + s1Index * dimG;
    } else {
        queryGmOffset = t1Offset * (dimN1 * dimDqk) + s1Index * (dimN1 * dimDqk) + n2Index * (dimG * dimDqk);
        queryRopeGmOffset = t1Offset * (dimN1 * dimRope) + s1Index * (dimN1 * dimRope) + n2Index * (dimG * dimRope);
        dyGmOffset = t1Offset * (dimN1 * dimDv) + s1Index * (dimN1 * dimDv) + n2Index * (dimG * dimDv);
        indicesGmOffset =
            t1Offset * (dimN2 * selectedBlockCount) + s1Index * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount;
        sumGmOffset = n2Index * dimS1 * dimG + (t1Offset + s1Index) * dimG;
    }
    keyGmOffset = t2Offset * (dimN2 * dimDqk) + n2Index * dimDqk;
    keyRopeGmOffset = t2Offset * (dimN2 * dimRope) + n2Index * dimRope;
    valueGmOffset = t2Offset * (dimN2 * dimDv) + n2Index * dimDv;

    curMaxS2 = ATTEN_ENABLE ? (s1Index + curS2 - curS1 + 1) : curS2;

    // worksapce
    mm12GmOffset = mmPingPongIdx * mm12WorkspaceLen;
    mm345GmOffset = mmPingPongIdx * mm345WorkspaceLen;
    selectedKGmOffset = selectdKPPPidx * selectedKWspOffset;
    selectedVGmOffset = mmPingPongIdx * selectedVWspOffset;
    runInfo[mmPingPongIdx].queryGmOffset = queryGmOffset;
    runInfo[mmPingPongIdx].queryRopeGmOffset = queryRopeGmOffset;
    runInfo[mmPingPongIdx].keyGmOffset = selectedKGmOffset;
    runInfo[mmPingPongIdx].dyGmOffset = dyGmOffset;
    runInfo[mmPingPongIdx].valueGmOffset = selectedVGmOffset;
    runInfo[mmPingPongIdx].indicesGmOffset = indicesGmOffset;
    runInfo[mmPingPongIdx].mm12GmOffset = mm12GmOffset;
    runInfo[mmPingPongIdx].mm345GmOffset = mm345GmOffset;
    runInfo[mmPingPongIdx].mm3OutGmOffset = queryGmOffset + queryRopeGmOffset;
    runInfo[mmPingPongIdx].mm4OutGmOffset = keyGmOffset + keyRopeGmOffset;
    runInfo[mmPingPongIdx].mm5OutGmOffset = valueGmOffset;
    runInfo[mmPingPongIdx].actualSelCntOffset = blkCntOffset + selectedCountOffset <= actualSelectedBlockCount ? selectedCountOffset : actualSelectedBlockCount - blkCntOffset;
    runInfo[mmPingPongIdx].lastBlockSize = curMaxS2 % selectedBlockSize != 0 ? curMaxS2 % selectedBlockSize : selectedBlockSize;
    runInfo[mmPingPongIdx].isLastBasicBlock = (blkCntOffset + selectedCountOffset >= actualSelectedBlockCount);
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::SaveLastInfo()
{
    lastQueryGmOffset = queryGmOffset;
    lastQueryRopeGmOffset = queryRopeGmOffset;
    lastKeyGmOffset = keyGmOffset;
    lastValueGmOffset = valueGmOffset;
    lastDyGmOffset = dyGmOffset;
    lastIndicesGmOffset = indicesGmOffset;
    lastmm12GmOffset = mm12GmOffset;
    lastmm345GmOffset = mm345GmOffset;
    lastblkCntOffset = blkCntOffset;
    mmPingPongIdx = 1 - mmPingPongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 3;
    loopCnt++;
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr,
                                                                       const GM_ADDR actual_seq_kvlen_addr,
                                                                       const int64_t t1Idx, int64_t &bIdx)
{
    if constexpr (IS_BSND == false) {
        int64_t curT1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex];
        while (t1Idx >= curT1) {
            curT1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[++bIndex];
        }

        if (unlikely(bIndex == 0)) {
            t1Offset = 0;
            t2Offset = 0;
            curS1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex];
            curS2 = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndex];
        } else {
            t1Offset = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex - 1];
            t2Offset = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndex - 1];
            curS1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex] - ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex - 1];
            curS2 = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndex] - ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndex - 1];
        }

        s1Index = t1Idx - t1Offset;
    } else {
        t1Offset = t1Idx;
        bIdx = t1Idx / curS1;
        s1Index = t1Idx % dimS1;
        t2Offset = bIdx * curS2;
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int32_t &actSelBlkCount)
{
    actualSelectedBlockCount = 0;
    int64_t topkGmOffset = t1Idx * (dimN2 * selectedBlockCount) + n2Idx * selectedBlockCount;
    for (int i = 0; i < selectedBlockCount; ++i) {
        if (topkIndicesGm[topkGmOffset].GetValue(i) == -1) {
            break;
        }
        actualSelectedBlockCount++;
    }
}

} // namespace SFAG_BASIC
