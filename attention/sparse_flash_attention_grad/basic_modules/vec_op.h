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
 * \file vec_op.h
 * \brief
 */

#pragma once
#include "kernel_operator.h"
#include "common_header.h"

namespace SFAG_BASIC {
struct StaticParams {
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t sftBaseM = 0;
    uint32_t sftBaseN = 0;
};

template <typename SFAGT>
class VecOp {
public:
    using TILING_CLASS = typename SFAGT::tiling_class;
    using T1 = typename SFAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = SFAGT::atten_enable;
    static constexpr bool HAS_ROPE = SFAGT::has_rope;
    static constexpr bool IS_BSND = SFAGT::is_bsnd;

    __aicore__ inline VecOp(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                GM_ADDR key_rope, 
                                GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);
    __aicore__ inline void Process(int64_t dyGmOffset, int64_t sumGmOffset, const uint64_t indicesGmOffset,
                                   const int64_t s1Index, const int32_t blkCntOffset, int64_t mm12Addr,
                                   int64_t mm345Addr, const int64_t lastBlockSize, const bool isLastBasicBlock);
    __aicore__ inline void GatherKV(const int32_t blkCntOffset, const int64_t s1Index, const int64_t n2Index, 
                                   uint64_t currentS1Offset, uint64_t kSelectedWsAddr, uint64_t vSelectedWsAddr, 
                                   uint64_t keyGmOffset, uint64_t valueGmOffset, uint64_t keyRopeGmOffset, 
                                   const int64_t curS1, const int64_t curS2);

protected:
    __aicore__ inline void InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                      GM_ADDR actual_seq_kvlen);
    __aicore__ inline void InitGMBuffer(GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
                                        GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR key_rope, GM_ADDR workspace);
    __aicore__ inline void InitUB(TPipe *pipe);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void DumpGmZero(GlobalTensor<float> &gm, int64_t num);
    __aicore__ inline void CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t sumGmOffset, const int32_t processM);
    __aicore__ inline void CalAttenMsk(const uint64_t indicesGmOffset, const int64_t s1Index, const int32_t processM,
                                       const int32_t blkCntOffset, const int32_t actualSelS2Align);
    __aicore__ inline void CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                      const int64_t mm345Addr, const uint64_t indicesGmOffset, const int64_t s1Index,
                                      const int32_t blkCntOffset);
    __aicore__ inline void CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                          const int64_t mm345Addr);
    __aicore__ inline void CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                  int32_t baseN);
    __aicore__ inline void Gather(GlobalTensor<T1> &inGmTensor, GlobalTensor<T1> &outGmTensor, uint64_t workSpaceOffset,
                                  int64_t headDim, uint64_t inGmOffset, int64_t dstStride = 0);

protected:
    // core info
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    uint32_t cubeBlockIdx;
    uint32_t vecBlockIdx;
    StaticParams params;
    GlobalTensor<T1> attentionGm;
    GlobalTensor<T1> attentionGradGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<T1> keyGm;
    GlobalTensor<T1> valueGm;
    GlobalTensor<T1> keyRopeGm;

    // workspace
    GlobalTensor<T1> selectedKWorkspaceGm;
    GlobalTensor<T1> selectedVWorkspaceGm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;

    TBuf<> vecQue;
    LocalTensor<uint8_t> helpTensor;
    LocalTensor<int32_t> topkIndicesTensor;
    LocalTensor<float> rowSumOutTensor;
    LocalTensor<float> maxTensor;
    LocalTensor<float> sumTensor;
    LocalTensor<float> maxTmp;
    LocalTensor<float> sumTmp;
    LocalTensor<T1> attentionT1Tensor;
    LocalTensor<float> attentionFP32Tensor;
    LocalTensor<T1> attentionGradT1Tensor;
    LocalTensor<float> attentionGradFP32Tensor;
    LocalTensor<float> pTensor;
    LocalTensor<float> dPTensor;
    LocalTensor<T1> sftOutT1Tensor;
    LocalTensor<T1> sftgOutT1Tensor;
    LocalTensor<T1> gatherTensor;

    // 地址相关
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};

    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t MSK_LEN = 64;
    constexpr static const uint32_t ATTEN_MASK_SCALE = 0xFF7FFFFF;

    const TILING_CLASS *__restrict tilingData;
    // Shape
    int64_t dimB;
    int64_t dimN2;
    int64_t dimS1;
    int64_t dimS2;
    int64_t dimG;
    int64_t dimD;
    int64_t dimD2;
    int64_t dimDqk; // Dqk = D
    int64_t dimDv;  // Dv = D2
    int64_t dimDAlign;
    int64_t dimD2Align;
    int64_t dimTkv{0};
    int64_t dimTq{0};
    int64_t selectedS2{0};
    int64_t curS1;
    int64_t curS2;
    int64_t lastBlock;
    int64_t actualSelS2;
    float scaleValue;
    uint32_t selectedBlockCount;
    uint32_t selectedBlockSize;
    uint32_t selectedCountOffset;
    uint32_t actualSelectedCount;
    int64_t dimRope;

    bool isLastBasicBlock;
    int64_t lastBlockSize;

    // workspace
    uint32_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t selectedKWorkspaceLen;
    int64_t selectedVWorkspaceLen;
    int64_t concatQWorkspaceLen;
    event_t vWaitMte2;
    event_t vWaitMte3;
    event_t mte3WaitMte2;
    event_t mte3WaitV;
    event_t sWaitMte2;
    event_t mte2WaitMte3;
};

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                          GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                          GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                          GM_ADDR key_rope, 
                                          GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                          const TILING_CLASS *__restrict ordTilingData, TPipe *pipe)
{
    InitParams(ordTilingData, actual_seq_qlen, actual_seq_kvlen);
    InitGMBuffer(key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, key_rope, workspace);
    InitUB(pipe);
    AtomicClean();
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                                GM_ADDR actual_seq_kvlen)
{
    cubeBlockIdx = GetBlockIdx() / 2;
    vecBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    formerCoreNum = tilingData->opInfo.formerCoreNum;

    dimB = tilingData->opInfo.B;
    dimN2 = tilingData->opInfo.N2;
    dimS1 = tilingData->opInfo.S1;
    dimS2 = tilingData->opInfo.S2;
    dimG = tilingData->opInfo.G;
    dimD = tilingData->opInfo.D;
    dimD2 = tilingData->opInfo.D2;
    dimDqk = tilingData->opInfo.D;
    dimDv = tilingData->opInfo.D2;
    dimRope = tilingData->opInfo.ropeD;
    dimDAlign = (dimD + dimRope + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    dimD2Align = (dimD2 + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;

    scaleValue = tilingData->opInfo.scaleValue;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    dvWorkspaceLen = tilingData->opInfo.dvWorkspaceLen;
    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    selectedVWorkspaceLen = tilingData->opInfo.selectedVWorkspaceLen;

    params.singleM = tilingData->splitCoreParams.singleM;
    params.singleN = tilingData->splitCoreParams.singleN;
    params.sftBaseM = tilingData->splitCoreParams.sftBaseM;
    params.sftBaseN = tilingData->splitCoreParams.sftBaseN;

    selectedS2 = selectedBlockCount * selectedBlockSize;
    selectedCountOffset = 512 / selectedBlockSize;
    if (tilingData->opInfo.selectedBlockSize * tilingData->opInfo.selectedBlockCount <= 512) {
        selectedCountOffset = tilingData->opInfo.selectedBlockCount;
    }

    if constexpr (IS_BSND == false) {
        for (int64_t i = 0; i < dimB; i++) {
            int64_t seqS1Len = 0;
            int64_t seqS2Len = 0;
            if (unlikely(i == 0)) {
                seqS1Len = ((__gm__ int32_t *)actual_seq_qlen)[i];
                seqS2Len = ((__gm__ int32_t *)actual_seq_kvlen)[i];
            } else {
                seqS1Len = ((__gm__ int32_t *)actual_seq_qlen)[i] - ((__gm__ int32_t *)actual_seq_qlen)[i - 1];
                seqS2Len = ((__gm__ int32_t *)actual_seq_kvlen)[i] - ((__gm__ int32_t *)actual_seq_kvlen)[i - 1];
            }
            dimTq += (int64_t)seqS1Len;
            dimTkv += (int64_t)seqS2Len;
        }
    } else {
        dimTq = dimB * dimS1;
        dimTkv = dimB * dimS2;
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitGMBuffer(GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad,
                                                  GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR key_rope, GM_ADDR workspace)
{
    /*
     * 初始化输入
     */
    attentionGm.SetGlobalBuffer((__gm__ T1 *)attention_out);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    keyRopeGm.SetGlobalBuffer((__gm__ T1 *)key_rope);

    /*
     * 初始化workspace
     */
    int64_t usedWorkspaceLen = 0;
    // select
    int64_t selectedKAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * selectedKWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedKWorkspaceLen * usedCoreNum;
    int64_t selectedVAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * selectedVWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedVWorkspaceLen * usedCoreNum;
    // mm1 与 p 复用workspace
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // mm2 与 ds 复用workspace
    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    int64_t dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    pWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + pAddr);
    dsWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + dsAddr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    dvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvAddr);
    selectedKWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedKAddr);
    selectedVWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedVAddr);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitUB(TPipe *pipe)
{
    uint32_t dataSize = params.sftBaseM * dimDv;
    uint32_t ubOffset = 0;
    uint32_t rowsumUbOffset = 0;
    uint32_t totalUbSpace = 191 * 1024;

    pipe->InitBuffer(vecQue, totalUbSpace);

    // topk
    int32_t topkNumber = AlignTo<int32_t>(selectedCountOffset, BLOCK);
    topkIndicesTensor = vecQue.GetWithOffset<int32_t>(topkNumber, ubOffset);
    ubOffset += topkNumber * sizeof(int32_t);

    // rowsum out
    rowSumOutTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);
    maxTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);
    sumTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);

    int32_t maxTmpSize = AlignUp(params.sftBaseM, BLOCK_FP32);
    maxTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    sumTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    rowsumUbOffset = ubOffset;

    // rowsum cal
    attentionGradT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(T1); //
    attentionGradFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(float);
    attentionT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(T1);
    attentionFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(float);

    // sftGard
    int32_t sftDataSize = params.sftBaseM * params.sftBaseN;
    pTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);
    dPTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    sftgOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);

    sftOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(T1);

    helpTensor = vecQue.GetWithOffset<uint8_t>((totalUbSpace - rowsumUbOffset) / sizeof(uint8_t), rowsumUbOffset);
    gatherTensor = vecQue.GetWithOffset<T1>((totalUbSpace - rowsumUbOffset) / sizeof(T1), rowsumUbOffset);

    uint32_t attentionShape[2] = {params.sftBaseM, static_cast<uint32_t>(dimDv)};
    uint32_t softmaxGradShape[2] = {params.sftBaseM, BLOCK_FP32};
    attentionGradFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    attentionFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    rowSumOutTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
    rowSumOutTensor.SetSize(params.sftBaseM * BLOCK_FP32);
    attentionGradT1Tensor.SetSize(dataSize);
    attentionGradFP32Tensor.SetSize(dataSize);
    attentionT1Tensor.SetSize(dataSize);
    attentionFP32Tensor.SetSize(dataSize);
    // params
    sumTensor.SetSize(params.sftBaseM * 8);
    maxTensor.SetSize(params.sftBaseM * 8);

    // set shape
    uint32_t softmaxShape[2] = {static_cast<uint32_t>(params.sftBaseM), 8};
    sumTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
    maxTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(params.sftBaseM), static_cast<uint32_t>(params.sftBaseN)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    sWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    vWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::AtomicClean()
{
    int64_t dqSize, dkSize, dvSize;
    dqSize = dimTq * dimN2 * dimG * dimDAlign;
    dkSize = dimTkv * dimN2 * dimDAlign;
    dvSize = dimTkv * dimN2 * dimD2Align;
    dqSize = (dqSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dkSize = (dkSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dvSize = (dvSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;

    DumpGmZero(dqWorkspaceGm, dqSize);
    DumpGmZero(dkWorkspaceGm, dkSize);
    DumpGmZero(dvWorkspaceGm, dvSize); // FP32 FUNC:T1 FP16
    if (vecBlockIdx % 2 == 0 && vecBlockIdx / 2 < usedCoreNum) {
        InitOutput<T1>(selectedKWorkspaceGm, selectedKWorkspaceLen / sizeof(T1), 0);
        InitOutput<T1>(selectedVWorkspaceGm, selectedVWorkspaceLen / sizeof(T1), 0);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::DumpGmZero(GlobalTensor<float> &gm, int64_t num)
{
    int64_t perSize = (num + tilingData->opInfo.castUsedCoreNum - 1) / tilingData->opInfo.castUsedCoreNum;
    int64_t coreNum = (num + perSize - 1) / perSize;
    int64_t tailSize = num - perSize * (coreNum - 1);
    int64_t initSize = perSize;

    if (vecBlockIdx == coreNum - 1) {
        initSize = tailSize;
    }

    if (vecBlockIdx < coreNum) {
        InitOutput<float>(gm[vecBlockIdx * perSize], initSize, 0);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                            int32_t baseN)
{
    constexpr static int32_t subMsk = 64;
    int8_t s2Repeat = baseN / subMsk;
    int8_t s2RepeatTail = baseN % subMsk;

    for (int32_t i = 0; i < baseM; i++) {
        Sub(dstTensor[i * baseN], dstTensor[i * baseN], srcTensor[i * BLOCK_FP32], subMsk, s2Repeat,
            {1, 1, 0, 8, 8, 0});
    }

    if (s2RepeatTail) {
        for (int32_t i = 0; i < baseM; i++) {
            Sub(dstTensor[s2Repeat * subMsk + i * baseN], dstTensor[s2Repeat * subMsk + i * baseN],
                srcTensor[i * BLOCK_FP32], s2RepeatTail, 1, {1, 1, 0, 8, 8, 0});
        }
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t sumGmOffset, const int32_t processM)
{
    uint32_t dataSize = processM * dimDv;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(processM * dimDv * sizeof(T1)), 0, 0, 0};
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(processM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParams2{false, 0, 0, 0};

    DataCopyPad(attentionGradT1Tensor, attentionGradGm[dyGmOffset], copyParams, padParams);
    DataCopyPad(attentionT1Tensor, attentionGm[dyGmOffset], copyParams, padParams);
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Cast(attentionGradFP32Tensor, attentionGradT1Tensor, RoundMode::CAST_NONE, dataSize);
    Cast(attentionFP32Tensor, attentionT1Tensor, RoundMode::CAST_NONE, dataSize);
    PIPE_BARRIER(PIPE_V);
    SoftmaxGradFront<float, false>(rowSumOutTensor, attentionGradFP32Tensor, attentionFP32Tensor, helpTensor,
                                   tilingData->softmaxGradTilingData);

    // [N2, T1, G] or [B, N2, S1, G] --> [sftBaseM, 1] --> [sftBaseM, 8]
    DataCopyPad(maxTmp, softmaxMaxGm[sumGmOffset], copyParams2, padParams2);
    DataCopyPad(sumTmp, softmaxSumGm[sumGmOffset], copyParams2, padParams2);
    SET_FLAG(MTE2, V, vWaitMte2);
    
    WAIT_FLAG(MTE2, V, vWaitMte2);
    Brcb(maxTensor, maxTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
    Brcb(sumTensor, sumTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
    PIPE_BARRIER(PIPE_V);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalAttenMsk(const uint64_t indicesGmOffset, const int64_t s1Index,
                                                 const int32_t processM, const int32_t blkCntOffset, const int32_t actualSelS2Align)
{ /*
   * 计算attenmask时对应blockSize需要保留的长度。
   * 例如：对于blockSize=64的块，需要保留0-9块，则attenMskRsvLen=10；
   */
    int32_t attenMskRsv = 0;
    int32_t attenMskEnd = 0;
    int32_t attenMskStartIdx = -1; // 计算attenmask时attenmask开始计算的下标
    float scalar = *((float *)&ATTEN_MASK_SCALE);
    uint64_t mask[1];
    LocalTensor<float> tmpTensor;

    int64_t valid_col_end = s1Index + curS2 - curS1;
    for (int32_t i = blkCntOffset; i < blkCntOffset + actualSelectedCount; i++) {
        int32_t topkIdx = topkIndicesGm[indicesGmOffset].GetValue(i);
        // 处于对角线上的block
        if (topkIdx * selectedBlockSize <= valid_col_end && (topkIdx + 1) * selectedBlockSize > valid_col_end) {
            attenMskRsv = valid_col_end - (topkIdx * selectedBlockSize) + 1;
            attenMskEnd = (i == blkCntOffset + actualSelectedCount - 1 && isLastBasicBlock) ? lastBlockSize : selectedBlockSize;
            attenMskStartIdx = i * selectedBlockSize + attenMskRsv - blkCntOffset * selectedBlockSize;
        }
    }

    if (attenMskStartIdx == -1) {
        return; // 默认mask掉的块为topk的无效值，不参与计算
    }

    int32_t selectIdx = (attenMskStartIdx - 1) / selectedBlockSize;
    int32_t selectBlkStartOffset = selectIdx * selectedBlockSize;
    int32_t selectBlkStartOffsetAlign = selectBlkStartOffset / 8 * 8;
    attenMskRsv += selectBlkStartOffset - selectBlkStartOffsetAlign;
    attenMskEnd += selectBlkStartOffset - selectBlkStartOffsetAlign;

    if (attenMskEnd - attenMskRsv > 0) {
        for (int32_t i = 0; i < processM; i++) { // selectedBlockSize need <= 64
            int32_t pOffset = i * actualSelS2Align;
            tmpTensor = pTensor[selectBlkStartOffsetAlign + pOffset];
            // attenMskRsv ~ attenMskEnd位置1
            uint64_t maskEnd = (attenMskEnd == 64) ? UINT64_MAX : (1ULL << attenMskEnd) - 1;
            uint64_t maskRsv = (attenMskRsv == 64) ? UINT64_MAX : (1ULL << attenMskRsv) - 1;
            mask[0] = maskEnd & ~maskRsv;
            Duplicate(tmpTensor, scalar, mask, 1, 0, 0);
        }
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                                const int64_t mm345Addr, const uint64_t indicesGmOffset,
                                                const int64_t s1Index, const int32_t blkCntOffset)
{
    int64_t actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;

    DataCopyPad(pTensor, mm1WorkspaceGm[mm12Addr], 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)), static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Muls(pTensor, pTensor, scaleValue, dataSize);
    PIPE_BARRIER(PIPE_V);

    if constexpr (ATTEN_ENABLE) {
        CalAttenMsk(indicesGmOffset, s1Index, processM, blkCntOffset, actualSelS2Align);
        PIPE_BARRIER(PIPE_V);
    }

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2Align)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2Align),
            static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2)};
    SimpleSoftMax<float, true, false>(pTensor, sumTensor, maxTensor, pTensor, helpTensor,
                                      tilingData->softmaxTilingData, softmaxShapeInfo);
    PIPE_BARRIER(PIPE_V);

    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2 - (i * 64) : 64;
        Cast(sftOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, processM, {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)), static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(pWorkspaceGm[mm345Addr], sftOutT1Tensor, 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0, static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM,
                                                    const int64_t mm12Addr, const int64_t mm345Addr)
{
    int64_t actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;
    DataCopyPad(dPTensor, mm2WorkspaceGm[mm12Addr], 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)), static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    CalSub(dPTensor, rowSumOutTensor, processM, actualSelS2Align);
    PIPE_BARRIER(PIPE_V);

    Mul(pTensor, pTensor, dPTensor, dataSize);
    PIPE_BARRIER(PIPE_V);

    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2 - (i * 64) : 64;
        Cast(sftgOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, params.sftBaseM, {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)), static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(dsWorkspaceGm[mm345Addr], sftgOutT1Tensor, 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0, static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::GatherKV(const int32_t blkCntOffset, const int64_t s1Index, const int64_t n2Index, 
                                              uint64_t currentS1Offset, uint64_t kSelectedWsAddr, uint64_t vSelectedWsAddr,
                                              uint64_t keyGmOffset, uint64_t valueGmOffset, uint64_t keyRopeGmOffset, 
                                              const int64_t curS1, const int64_t curS2)
{
    this->curS1 = curS1;
    this->curS2 = curS2;
    this->lastBlock = (this->curS2 + selectedBlockSize - 1) / selectedBlockSize;
    // copy tokIndices in
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = selectedCountOffset * sizeof(int32_t);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    // [T1,N2,topk] or [B,S1,N2,topk]
    uint64_t s1Offset = IS_BSND ? currentS1Offset : currentS1Offset + s1Index;
    uint64_t gmOffset = s1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount + blkCntOffset;
    DataCopyPad(topkIndicesTensor, topkIndicesGm[gmOffset], intriParams, {false, 0, 0, 0});
    SET_FLAG(MTE2, S, sWaitMte2);
    WAIT_FLAG(MTE2, S, sWaitMte2);

    Gather(keyGm, selectedKWorkspaceGm, kSelectedWsAddr, dimDqk, keyGmOffset, dimRope);
    Gather(valueGm, selectedVWorkspaceGm, vSelectedWsAddr, dimDv, valueGmOffset);
    if constexpr (HAS_ROPE) {
        Gather(keyRopeGm, selectedKWorkspaceGm, kSelectedWsAddr + dimDqk, dimRope, keyRopeGmOffset, dimDqk);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Gather(GlobalTensor<T1> &inGmTensor, GlobalTensor<T1> &outGmTensor,
                                                            uint64_t workSpaceOffset, int64_t headDim, uint64_t inGmOffset, int64_t dstStride)
{
    DataCopyParams intriParams;
    int32_t loop = selectedCountOffset;
    int32_t processLen = selectedBlockSize;
    int32_t startS2Idx = 0;
    int32_t dataSize = processLen * headDim;
    this->actualSelS2 = 0;
    this->actualSelectedCount = 0;

    for (uint32_t i = 0; i < loop; i++) {
        int32_t topkIdx = topkIndicesTensor.GetValue(i);
        startS2Idx = topkIdx * selectedBlockSize;
        if (topkIdx == -1) {
            break;
        }
        if (topkIdx == lastBlock - 1) {
            processLen = curS2 - topkIdx * selectedBlockSize;
            dataSize = processLen * headDim;
        }
        actualSelS2 += processLen;
        actualSelectedCount++;

        intriParams.blockCount = processLen;
        intriParams.blockLen = headDim * sizeof(T1);
        intriParams.srcStride = (dimN2 - 1) * headDim * sizeof(T1);
        intriParams.dstStride = 0;

        DataCopyPad(gatherTensor, inGmTensor[inGmOffset + startS2Idx * (dimN2 * headDim)], intriParams, {false, 0, 0, 0});
        SET_FLAG(MTE2, MTE3, mte3WaitMte2);
        WAIT_FLAG(MTE2, MTE3, mte3WaitMte2);

        intriParams.blockCount = processLen;
        intriParams.blockLen = headDim * sizeof(T1);
        intriParams.srcStride = 0;
        intriParams.dstStride = dstStride * sizeof(T1);
        
        DataCopyPad(outGmTensor[workSpaceOffset], gatherTensor, intriParams);
        SET_FLAG(MTE3, MTE2, mte2WaitMte3);
        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);

        workSpaceOffset += dataSize + processLen * dstStride;
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Process(int64_t dyGmOffset, int64_t sumGmOffset,
                                             const uint64_t indicesGmOffset, const int64_t s1Index,
                                             const int32_t blkCntOffset, int64_t mm12Addr, int64_t mm345Addr,
                                             const int64_t lastBlockSize, const bool isLastBasicBlock)
{
    this->lastBlockSize = lastBlockSize;
    this->isLastBasicBlock = isLastBasicBlock;
    int32_t loop = (params.singleM + params.sftBaseM - 1) / params.sftBaseM;
    int32_t processM = params.sftBaseM;
    int32_t tailM = params.singleM % params.sftBaseM;
    int64_t dataSize = processM * params.sftBaseN;

    for (int32_t i = 0; i < loop; i++) {
        if (i == loop - 1 && tailM != 0) {
            processM = tailM;
        }
        CalRowsumAndSftCopyIn(dyGmOffset, sumGmOffset, processM);
        dyGmOffset += (processM * dimDv);
        sumGmOffset += processM;

        CalSoftmax(i, processM, mm12Addr, mm345Addr, indicesGmOffset, s1Index, blkCntOffset);
        CalSoftmaxGrad(i, processM, mm12Addr, mm345Addr);
        mm12Addr += dataSize;
        mm345Addr += dataSize;
    }
}


} // namespace SFAG_BASIC
