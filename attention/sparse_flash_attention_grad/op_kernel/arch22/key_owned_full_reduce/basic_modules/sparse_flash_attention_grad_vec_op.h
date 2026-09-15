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
#include "sparse_flash_attention_grad_common_header.h"

namespace SFAG_KEY_OWNED_FULL_REDUCE {
struct StaticParams {
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t sftBaseM = 0;
    uint32_t sftBaseN = 0;
};

constexpr static int32_t repeatMaxBytes = 256;
constexpr static int32_t repeatMaxTimes = 255;
constexpr static int32_t repeatMaxSize = repeatMaxBytes / 4; // 4 means sizeof(T)

template <typename SFAGT>
class VecOp {
public:
    using TILING_CLASS = typename SFAGT::tiling_class;
    using T1 = typename SFAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = SFAGT::atten_enable;
    static constexpr bool HAS_ROPE = SFAGT::has_rope;
    static constexpr bool IS_BSND = SFAGT::is_bsnd;
    static constexpr bool IS_DETERMINISTIC = SFAGT::is_deterministic;
    static constexpr bool KV_MERGE = SFAGT::kv_merge;

    __aicore__ inline VecOp(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);
    __aicore__ inline void Process(const RunInfo &runInfo);
    __aicore__ inline void GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo);
    __aicore__ inline void MergeKeyOwnedEdge(const RunInfo &runInfo);
    __aicore__ inline void ScatterAddUnDeter(const RunInfo &runInfo);
    __aicore__ inline void ScatterAddDeter(const RunInfo &runInfo);
    __aicore__ inline void ScatterAdd(const RunInfo &runInfo);
    __aicore__ inline void CleanDqRow(const RunInfo &runInfo);

protected:
    __aicore__ inline void InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                      GM_ADDR actual_seq_kvlen);
    __aicore__ inline void InitGMBuffer(GM_ADDR key, GM_ADDR value, GM_ADDR dv, GM_ADDR attention_out,
                                        GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                        GM_ADDR topk_indices, GM_ADDR key_rope, GM_ADDR workspace);
    __aicore__ inline void InitUB(TPipe *pipe);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void GatherKVNonOptimized(const int64_t n2Index, uint64_t currentS1Offset,
                                                const RunInfo &runInfo);
    __aicore__ inline void GatherKVOptimized(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo);
    template <typename GM_TYPE>
    __aicore__ inline void ZeroWithScratch(GlobalTensor<GM_TYPE> gm, int64_t num);
    template <typename GM_TYPE>
    __aicore__ inline void DumpGmZero(GlobalTensor<GM_TYPE> &gm, int64_t num);
    __aicore__ inline void CalRowsumAndSftCopyIn(const int32_t loopIdx, const int64_t dyGmOffset,
                                                 const int64_t sumGmOffset, const int32_t processM,
                                                 const bool calcSoftmaxGradFront);
    __aicore__ inline void CalAttenMsk(const int32_t processM, const int32_t actualSelS2Align, const RunInfo &runInfo);
    __aicore__ inline void CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                      const int64_t mm345Addr, const RunInfo &runInfo);
    __aicore__ inline void CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                          const int64_t mm345Addr, const RunInfo &runInfo);
    __aicore__ inline void CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                  int32_t baseN);
    __aicore__ inline void NzToNd(Nz2NdInfo &nz2NdInfo, const GlobalTensor<float> &bmmResGm, LocalTensor<float> &tempUb,
                                  LocalTensor<float> &bmmResUb, event_t eventIdMte2ToV, event_t backEventTmp,
                                  event_t backEvent, bool isBackEvent = false, int64_t dstLastAxis = 0);

protected:
    // core info
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    uint32_t cubeBlockIdx;
    uint32_t vecBlockIdx;
    uint32_t subBlockIdx;
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
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;
    GlobalTensor<float> mm4ResWorkspaceGm; // 24 * SCATTER_BUFFER_NUM * K * Dk
    GlobalTensor<float> mm5ResWorkspaceGm; // 24 * SCATTER_BUFFER_NUM * K * Dv
    GlobalTensor<float> keyOwnedEdgeGm;

    TBuf<> vecQue;
    TBuf<> indexBuf;
    LocalTensor<int32_t> indexTensor;
    uint64_t invalidMasks[2][2] = {{0, 0}, {0, 0}};
    LocalTensor<uint8_t> helpTensor;
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
    LocalTensor<T1> gatherTensorPing;
    LocalTensor<T1> gatherTensorPong;
    LocalTensor<T1> gatherRopePing;
    LocalTensor<T1> gatherRopePong;
    LocalTensor<float> scatterAddTensorK;
    LocalTensor<float> scatterAddTensorV;
    LocalTensor<float> tempTensorK;
    LocalTensor<float> tempTensorV;

    // 地址相关
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};
    bool enableOptimizedScatter{false};

    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t MSK_LEN = 64;
    constexpr static const uint32_t ATTEN_MASK_SCALE = 0xFF7FFFFF;
    constexpr static const uint32_t MAX_GATHER_SIZE = 32;
    constexpr static const uint32_t UB_ROW_SIZE = 4;
    constexpr static const uint32_t NON_OPT_MAX_GATHER_SIZE = 64;
    constexpr static const uint32_t NON_OPT_UB_ROW_SIZE = 16;

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
    int64_t selectedBlockSizeDqk;
    int64_t selectedBlockSizeDrope;
    int64_t selectedBlockSizeDimDAlign;
    int64_t selectedBlockSizeDimD2Align;
    int64_t ubRowSizeDAlign;
    int64_t ubRowSizeD2Align;
    float scaleValue;
    uint32_t selectedBlockCount;
    uint32_t selectedBlockSize;
    int64_t scatterTokenCapacity;
    uint32_t selectedCountOffset;
    uint32_t actualSelectedCount;
    int32_t maxSelCnt;
    int64_t dimRope;
    // workspace
    uint64_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t selectedKWorkspaceLen;
    int64_t concatQWorkspaceLen;
    event_t vWaitMte2;
    event_t vWaitMte2Pong;
    event_t vWaitMte3;
    event_t mte3WaitMte2;
    event_t mte3WaitV;
    event_t sWaitMte2;
    event_t mte3WaitMte2Pong;
    event_t mte2WaitMte3Proc;
    event_t mte2WaitMte3;
    event_t mte2WaitMte3Pong;

    DataCopyPadExtParams<T1> padParams;
    DataCopyExtParams intriParamsKey;
    DataCopyExtParams intriParamsRope;
    DataCopyExtParams outParamK;
    DataCopyExtParams outParamRope;
    DataCopyExtParams scatterParams;
    uint32_t mergePingPong{0};
    int32_t pingPongIdx = 0;
};

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                          GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                          GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                          GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                          const TILING_CLASS *__restrict ordTilingData, TPipe *pipe)
{
    InitParams(ordTilingData, actual_seq_qlen, actual_seq_kvlen);
    InitGMBuffer(key, value, dv, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, key_rope,
                 workspace);
    InitUB(pipe);
    AtomicClean();
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                                GM_ADDR actual_seq_kvlen)
{
    cubeBlockIdx = GetBlockIdx() / 2;
    vecBlockIdx = GetBlockIdx();
    subBlockIdx = GetSubBlockIdx();
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
    selectedBlockSizeDqk = selectedBlockSize * dimDqk;
    selectedBlockSizeDrope = selectedBlockSize * dimRope;
    selectedBlockSizeDimDAlign = selectedBlockSize * dimDAlign;
    selectedBlockSizeDimD2Align = selectedBlockSize * dimD2Align;
    enableOptimizedScatter = tilingData->opInfo.enableOptimizedScatter;
    ubRowSizeDAlign = NON_OPT_UB_ROW_SIZE * dimDAlign;
    ubRowSizeD2Align = NON_OPT_UB_ROW_SIZE * dimD2Align;

    params.singleM = tilingData->splitCoreParams.singleM;
    params.singleN = tilingData->splitCoreParams.singleN;
    params.sftBaseM = tilingData->splitCoreParams.sftBaseM;
    params.sftBaseN = tilingData->splitCoreParams.sftBaseN;
    scatterTokenCapacity = enableOptimizedScatter ? params.singleN : selectedBlockCount * selectedBlockSize;

    selectedS2 = selectedBlockCount * selectedBlockSize;
    selectedCountOffset = PER_LOOP_BLOCK_SIZE / selectedBlockSize;
    if (tilingData->opInfo.selectedBlockSize * tilingData->opInfo.selectedBlockCount <= PER_LOOP_BLOCK_SIZE) {
        selectedCountOffset = tilingData->opInfo.selectedBlockCount;
    }

    if constexpr (IS_BSND == false) {
        dimTq = dimS1;
        dimTkv = dimS2;
    } else {
        dimTq = dimB * dimS1;
        dimTkv = dimB * dimS2;
    }

    intriParamsKey.blockLen = selectedBlockSize * dimDqk * sizeof(T1);
    intriParamsKey.dstStride = 0;
    intriParamsKey.blockCount = 2;

    intriParamsRope.blockLen = selectedBlockSize * dimRope * sizeof(T1);
    intriParamsRope.dstStride = 0;
    intriParamsRope.blockCount = 2;

    outParamK.blockCount = 2 * selectedBlockSize;
    outParamK.blockLen = dimDqk * sizeof(T1);
    outParamK.srcStride = 0;
    outParamK.dstStride = 0;

    outParamRope.blockCount = 2 * selectedBlockSize;
    outParamRope.blockLen = dimRope * sizeof(T1);
    outParamRope.srcStride = 0;
    outParamRope.dstStride = 0;

    scatterParams.blockCount = 2;
    scatterParams.srcStride = 0;
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitGMBuffer(GM_ADDR key, GM_ADDR value, GM_ADDR dv, GM_ADDR attention_out,
                                                  GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                                  GM_ADDR topk_indices, GM_ADDR key_rope, GM_ADDR workspace)
{
    (void)dv;
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
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;

    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    int64_t dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    // scatter add
    uint64_t scatterBufferNum = tilingData->opInfo.enableOptimizedScatter ? SCATTER_BUFFER_NUM : PING_PONG_BUFFER;
    int64_t mm4ResAddr = usedWorkspaceLen / sizeof(float);
    int64_t mm5ResAddr = mm4ResAddr + MAX_CORE_NUM * scatterTokenCapacity * dimDAlign * scatterBufferNum;
    usedWorkspaceLen +=
        MAX_CORE_NUM * scatterTokenCapacity * (dimDAlign + dimD2Align) * scatterBufferNum * sizeof(float);

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    pWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + pAddr);
    dsWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + dsAddr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    dvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvAddr);
    selectedKWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedKAddr);

    mm4ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm4ResAddr);
    mm5ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm5ResAddr);
    keyOwnedEdgeGm.SetGlobalBuffer((__gm__ float *)(workspace + KeyOwnedExtensionByteOffset(tilingData)));
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitUB(TPipe *pipe)
{
    uint32_t dataSize = params.sftBaseM * dimDv;
    uint32_t ubOffset = 0;
    uint32_t rowsumUbOffset = 0;
    uint32_t totalUbSpace = 191 * 1024;

    // The optimized path keeps rowSum/max/sum for the complete G row and
    // reuses them across K blocks. The non-optimized path recalculates them
    // for every K block, so one aligned softmax tile is sufficient.
    uint32_t softmaxParamM =
        enableOptimizedScatter ? AlignUp(params.singleM, BLOCK_FP32) : AlignUp(params.sftBaseM, BLOCK_FP32);

    pipe->InitBuffer(vecQue, totalUbSpace);
    pipe->InitBuffer(indexBuf, 1024);
    indexTensor = indexBuf.Get<int32_t>();


        uint32_t topkUbOffset = ubOffset;

        rowSumOutTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
        ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);
        maxTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
        ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);
        sumTensor = vecQue.GetWithOffset<float>(softmaxParamM * BLOCK_FP32, ubOffset);
        ubOffset += softmaxParamM * BLOCK_FP32 * sizeof(float);

        int32_t maxTmpSize = AlignUp(params.sftBaseM, BLOCK_FP32);
        maxTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
        ubOffset += maxTmpSize * sizeof(float);
        sumTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
        ubOffset += maxTmpSize * sizeof(float);
        rowsumUbOffset = ubOffset;

        attentionGradT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
        ubOffset += params.sftBaseM * dimDv * sizeof(T1);
        attentionGradFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
        ubOffset += params.sftBaseM * dimDv * sizeof(float);
        attentionT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
        ubOffset += params.sftBaseM * dimDv * sizeof(T1);
        attentionFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
        ubOffset += params.sftBaseM * dimDv * sizeof(float);

        int32_t sftDataSize = params.sftBaseM * params.sftBaseN;
        pTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
        ubOffset += sftDataSize * sizeof(float);
        dPTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
        sftgOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
        ubOffset += sftDataSize * sizeof(float);

        sftOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
        ubOffset += sftDataSize * sizeof(T1);

        helpTensor = vecQue.GetWithOffset<uint8_t>((totalUbSpace - rowsumUbOffset) / sizeof(uint8_t), rowsumUbOffset);
        scatterAddTensorK = vecQue.GetWithOffset<float>(NON_OPT_UB_ROW_SIZE * dimDAlign * 2, rowsumUbOffset);
        rowsumUbOffset += NON_OPT_UB_ROW_SIZE * dimDAlign * 2 * sizeof(float);
        scatterAddTensorV = vecQue.GetWithOffset<float>(NON_OPT_UB_ROW_SIZE * dimD2Align * 2, rowsumUbOffset);
        rowsumUbOffset += NON_OPT_UB_ROW_SIZE * dimD2Align * 2 * sizeof(float);

        maxSelCnt = selectedBlockSize <= 32 ? 2 : 1;
        gatherTensorPing = vecQue.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimDqk, topkUbOffset);
        topkUbOffset += NON_OPT_MAX_GATHER_SIZE * dimDqk * sizeof(T1);
        gatherTensorPong = vecQue.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimDqk, topkUbOffset);
        topkUbOffset += NON_OPT_MAX_GATHER_SIZE * dimDqk * sizeof(T1);
        gatherRopePing = vecQue.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimRope, topkUbOffset);
        topkUbOffset += NON_OPT_MAX_GATHER_SIZE * dimRope * sizeof(T1);
        gatherRopePong = vecQue.GetWithOffset<T1>(NON_OPT_MAX_GATHER_SIZE * dimRope, topkUbOffset);

        uint32_t attentionShape[2] = {params.sftBaseM, static_cast<uint32_t>(dimDv)};
        uint32_t softmaxGradShape[2] = {params.sftBaseM, BLOCK_FP32};
        attentionGradFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
        attentionFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
        rowSumOutTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
        rowSumOutTensor.SetSize(softmaxParamM * BLOCK_FP32);
        attentionGradT1Tensor.SetSize(dataSize);
        attentionGradFP32Tensor.SetSize(dataSize);
        attentionT1Tensor.SetSize(dataSize);
        attentionFP32Tensor.SetSize(dataSize);
        sumTensor.SetSize(softmaxParamM * BLOCK_FP32);
        maxTensor.SetSize(softmaxParamM * BLOCK_FP32);

        uint32_t softmaxShape[2] = {static_cast<uint32_t>(params.sftBaseM), BLOCK_FP32};
        sumTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
        maxTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

        uint32_t dstSoftShape[2] = {static_cast<uint32_t>(params.sftBaseM), static_cast<uint32_t>(params.sftBaseN)};
        pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
        mte2WaitMte3Proc = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        mte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        mte3WaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        sWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
        vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        vWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
}


template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::AtomicClean()
{
    int64_t dkSize, dvSize;
    dkSize = dimTkv * dimN2 * dimDAlign;
    dvSize = dimTkv * dimN2 * dimD2Align;
    dkSize = (dkSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dvSize = (dvSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;

    DumpGmZero(dkWorkspaceGm, dkSize);
    if constexpr (!KV_MERGE) {
        DumpGmZero(dvWorkspaceGm, dvSize);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CleanDqRow(const RunInfo &runInfo)
{
    // Non-empty rows are initialized by the first non-atomic MM3 FIXP. Only rows
    // without a K task still need an explicit zero result, and one AIV sub-core is enough.
    if (subBlockIdx == 0) {
        ZeroWithScratch<float>(dqWorkspaceGm[runInfo.mm3OutGmOffset], dimG * dimDAlign);
    }
}

template <typename SFAGT>
template <typename GM_TYPE>
__aicore__ inline void VecOp<SFAGT>::ZeroWithScratch(GlobalTensor<GM_TYPE> gm, int64_t num)
{
    // At initialization (or an empty-Q cleanup) vecQue is dead scratch; no PopStackBuffer.
    auto zero = vecQue.GetWithOffset<GM_TYPE>(2048, 0);
    Duplicate(zero, static_cast<GM_TYPE>(0), 2048);
    SET_FLAG(V, MTE3, mte3WaitV); WAIT_FLAG(V, MTE3, mte3WaitV);
    for (int64_t offset = 0; offset < num; offset += 2048) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<int64_t>(2048), num - offset));
        DataCopyPad(gm[offset], zero, {1, static_cast<uint32_t>(count * sizeof(GM_TYPE)), 0, 0, 0});
    }
    SET_FLAG(MTE3, V, vWaitMte3); WAIT_FLAG(MTE3, V, vWaitMte3);
}

template <typename SFAGT>
template <typename GM_TYPE>
__aicore__ inline void VecOp<SFAGT>::DumpGmZero(GlobalTensor<GM_TYPE> &gm, int64_t num)
{
    int64_t perSize = (num + tilingData->opInfo.castUsedCoreNum - 1) / tilingData->opInfo.castUsedCoreNum;
    int64_t coreNum = (num + perSize - 1) / perSize;
    int64_t tailSize = num - perSize * (coreNum - 1);
    int64_t initSize = perSize;

    if (vecBlockIdx == coreNum - 1) {
        initSize = tailSize;
    }

    if (vecBlockIdx < coreNum) {
        ZeroWithScratch<GM_TYPE>(gm[vecBlockIdx * perSize], initSize);
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
__aicore__ inline void VecOp<SFAGT>::CalRowsumAndSftCopyIn(const int32_t loopIdx, const int64_t dyGmOffset,
                                                           const int64_t sumGmOffset, const int32_t processM,
                                                           const bool calcSoftmaxGradFront)
{
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(processM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams2{false, 0, 0, 0};

    if (calcSoftmaxGradFront) {
        uint32_t dataSize = processM * dimDv;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataSize * sizeof(T1)), 0, 0, 0};
        DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
        uint32_t softmaxParamOffset = enableOptimizedScatter ? loopIdx * params.sftBaseM * BLOCK_FP32 : 0;
        LocalTensor<float> curRowSumTensor = rowSumOutTensor[softmaxParamOffset];
        uint32_t alignedProcessM = AlignUp(processM, BLOCK_FP32);
        uint32_t curSoftmaxGradShape[2] = {static_cast<uint32_t>(processM), BLOCK_FP32};
        curRowSumTensor.SetShapeInfo(ShapeInfo(2, curSoftmaxGradShape, DataFormat::ND));
        curRowSumTensor.SetSize(alignedProcessM * BLOCK_FP32);

        DataCopyPad(attentionGradT1Tensor, attentionGradGm[dyGmOffset], copyParams, padParams);
        DataCopyPad(attentionT1Tensor, attentionGm[dyGmOffset], copyParams, padParams);
        SET_FLAG(MTE2, V, vWaitMte2);
        WAIT_FLAG(MTE2, V, vWaitMte2);

        Cast(attentionGradFP32Tensor, attentionGradT1Tensor, RoundMode::CAST_NONE, dataSize);
        Cast(attentionFP32Tensor, attentionT1Tensor, RoundMode::CAST_NONE, dataSize);
        PIPE_BARRIER(PIPE_V);
        SoftmaxGradFront<float, false>(curRowSumTensor, attentionGradFP32Tensor, attentionFP32Tensor, helpTensor,
                                       tilingData->softmaxGradTilingData);

        // [N2, T1, G] or [B, N2, S1, G] --> current [sftBaseM, 1].
        // The optimized path stores all tiles; the non-optimized path reuses tile 0.
        DataCopyPad(maxTmp, softmaxMaxGm[sumGmOffset], copyParams2, padParams2);
        DataCopyPad(sumTmp, softmaxSumGm[sumGmOffset], copyParams2, padParams2);
        SET_FLAG(MTE2, V, vWaitMte2);

        WAIT_FLAG(MTE2, V, vWaitMte2);
        LocalTensor<float> curMaxTensor = maxTensor[softmaxParamOffset];
        LocalTensor<float> curSumTensor = sumTensor[softmaxParamOffset];
        Brcb(curMaxTensor, maxTmp, CeilDiv(processM, BLOCK_FP32), {1, BLOCK_FP32});
        PIPE_BARRIER(PIPE_V);
        Brcb(curSumTensor, sumTmp, CeilDiv(processM, BLOCK_FP32), {1, BLOCK_FP32});
        PIPE_BARRIER(PIPE_V);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalAttenMsk(const int32_t processM, const int32_t actualSelS2Align,
                                                 const RunInfo &runInfo)
{
    (void)actualSelS2Align;
    for (uint32_t word = 0; word < 2; ++word) {
        if (invalidMasks[runInfo.task & 1][word] != 0) {
            uint64_t mask[2] = {invalidMasks[runInfo.task & 1][word], 0};
            Duplicate(pTensor[word * 64], 0.0f, mask, processM, 1, 16);
        }
    }
    PIPE_BARRIER(PIPE_V);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                                const int64_t mm345Addr, const RunInfo &runInfo)
{
    int64_t actualSelS2 = runInfo.isLastBasicBlock ?
                              (runInfo.actualSelCntOffset - 1) * selectedBlockSize + runInfo.lastBlockSize :
                              runInfo.actualSelCntOffset * selectedBlockSize;
    int64_t actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;

    WAIT_FLAG(V, MTE2, runInfo.processMte2WaitV);
    DataCopyPad(pTensor, mm1WorkspaceGm[mm12Addr],
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)),
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Muls(pTensor, pTensor, scaleValue, dataSize);
    PIPE_BARRIER(PIPE_V);


    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2Align)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    uint32_t softmaxParamOffset = enableOptimizedScatter ? loopIdx * params.sftBaseM * BLOCK_FP32 : 0;
    LocalTensor<float> curSumTensor = sumTensor[softmaxParamOffset];
    LocalTensor<float> curMaxTensor = maxTensor[softmaxParamOffset];
    uint32_t alignedProcessM = AlignUp(processM, BLOCK_FP32);
    uint32_t curSoftmaxParamShape[2] = {static_cast<uint32_t>(processM), BLOCK_FP32};
    curSumTensor.SetShapeInfo(ShapeInfo(2, curSoftmaxParamShape, DataFormat::ND));
    curMaxTensor.SetShapeInfo(ShapeInfo(2, curSoftmaxParamShape, DataFormat::ND));
    curSumTensor.SetSize(alignedProcessM * BLOCK_FP32);
    curMaxTensor.SetSize(alignedProcessM * BLOCK_FP32);
    SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2Align),
                                      static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2)};
    SimpleSoftMax<float, true, false>(pTensor, curSumTensor, curMaxTensor, pTensor, helpTensor,
                                      tilingData->softmaxTilingData, softmaxShapeInfo);
    PIPE_BARRIER(PIPE_V);

    CalAttenMsk(processM, actualSelS2Align, runInfo);
    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2 - (i * 64) : 64;
        Cast(sftOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, processM,
             {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)),
              static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(pWorkspaceGm[mm345Addr], sftOutT1Tensor,
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0,
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM,
                                                    const int64_t mm12Addr, const int64_t mm345Addr,
                                                    const RunInfo &runInfo)
{
    int64_t actualSelS2 = runInfo.isLastBasicBlock ?
                              (runInfo.actualSelCntOffset - 1) * selectedBlockSize + runInfo.lastBlockSize :
                              runInfo.actualSelCntOffset * selectedBlockSize;
    int64_t actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;
    DataCopyPad(dPTensor, mm2WorkspaceGm[mm12Addr],
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)),
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    uint32_t softmaxParamOffset = enableOptimizedScatter ? loopIdx * params.sftBaseM * BLOCK_FP32 : 0;
    LocalTensor<float> curRowSumTensor = rowSumOutTensor[softmaxParamOffset];
    CalSub(dPTensor, curRowSumTensor, processM, actualSelS2Align);
    PIPE_BARRIER(PIPE_V);

    Mul(pTensor, pTensor, dPTensor, dataSize);
    PIPE_BARRIER(PIPE_V);

    // Match the original kernel: cast unscaled dS for the Cube matmuls.
    // Scale dQ in Post and each FP32 dK edge before merging its dV edge.

    CalAttenMsk(processM, actualSelS2Align, runInfo);
    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2 - (i * 64) : 64;
        Cast(sftgOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, processM,
             {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)),
              static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE2, runInfo.processMte2WaitV);
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(dsWorkspaceGm[mm345Addr], sftgOutT1Tensor,
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0,
                 static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo)
{
    if (runInfo.isSmallS2)
        return;
    if constexpr (IS_DETERMINISTIC) {
        GatherKVNonOptimized(n2Index, currentS1Offset, runInfo);
    } else {

            GatherKVNonOptimized(n2Index, currentS1Offset, runInfo);

    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::GatherKVNonOptimized(const int64_t n2Index, uint64_t currentS1Offset,
                                                          const RunInfo &runInfo)
{
    (void)n2Index; (void)currentS1Offset;
    // Scalar masks outlive this index buffer: Scatter may overwrite indices before Process(t-1).
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    const uint32_t selectedRows = static_cast<uint32_t>(runInfo.actualSelCntOffset * selectedBlockSize);
    const uint32_t subBlockBegin = subBlockIdx * 64;
    const uint32_t gatherRows = selectedRows > subBlockBegin ? Min<uint32_t>(64, selectedRows - subBlockBegin) : 0;
    DataCopyPad(indexTensor, topkIndicesGm[runInfo.indicesGmOffset + runInfo.blkCntOffset * selectedBlockSize],
                DataCopyExtParams{1, selectedRows * sizeof(int32_t), 0, 0, 0},
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    SET_FLAG(MTE2, S, sWaitMte2); WAIT_FLAG(MTE2, S, sWaitMte2);
    const uint32_t generation = runInfo.task & 1;
    invalidMasks[generation][0] = 0;
    invalidMasks[generation][1] = 0;
    const int64_t causalEnd = runInfo.s1Index + runInfo.curS2 - runInfo.curS1;
    for (uint32_t i = 0; i < selectedRows; ++i) {
        if (indexTensor.GetValue(i) > causalEnd) invalidMasks[generation][i / 64] |= 1ULL << (i % 64);
    }
    SET_FLAG(MTE3, MTE2, mte2WaitMte3); WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
    for (uint32_t i = 0; i < gatherRows; ++i) {
        const int64_t index = indexTensor.GetValue(subBlockBegin + i);
        DataCopy(gatherTensorPing[i * dimDqk], keyGm[runInfo.keyGmOffset + index * dimDqk], dimDqk);
        DataCopy(gatherRopePing[i * dimRope], keyRopeGm[runInfo.keyRopeGmOffset + index * dimRope], dimRope);
    }
    SET_FLAG(MTE2, MTE3, mte3WaitMte2); WAIT_FLAG(MTE2, MTE3, mte3WaitMte2);
    if (gatherRows > 0) {
        DataCopy(selectedKWorkspaceGm[runInfo.selectedKGmOffset + subBlockBegin * dimDqk], gatherTensorPing,
                gatherRows * dimDqk);
        DataCopy(selectedKWorkspaceGm[runInfo.selectedKGmOffset + 128 * dimDqk + subBlockBegin * dimRope],
                gatherRopePing, gatherRows * dimRope);
    }
    SET_FLAG(MTE3, MTE2, mte2WaitMte3); WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
    SET_FLAG(MTE3, V, vWaitMte3); WAIT_FLAG(MTE3, V, vWaitMte3);
}



template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Process(const RunInfo &runInfo)
{
    int32_t loopEnd = (params.singleM + params.sftBaseM - 1) / params.sftBaseM;
    int32_t firstLoopEnd = loopEnd / 2;
    int32_t substart = subBlockIdx == 0 ? 0 : firstLoopEnd;
    int32_t subLoopEnd = subBlockIdx == 0 ? firstLoopEnd : loopEnd;

    int32_t processM = params.sftBaseM;
    int32_t tailM = params.singleM % params.sftBaseM;
    int64_t dataSize = processM * params.sftBaseN;

    int64_t dyGmOffset = runInfo.dyGmOffset + subBlockIdx * (processM * dimDv * firstLoopEnd);
    int64_t sumGmOffset = runInfo.sumGmOffset + subBlockIdx * (processM * firstLoopEnd);
    int64_t mm12Addr = runInfo.mm12GmOffset + subBlockIdx * (dataSize * firstLoopEnd);
    int64_t mm345Addr = runInfo.mm345GmOffset + subBlockIdx * (dataSize * firstLoopEnd);

    for (int32_t i = substart; i < subLoopEnd; i++) {
        if (i == loopEnd - 1 && tailM != 0) {
            processM = tailM;
        }
        bool calcSoftmaxGradFront = !enableOptimizedScatter || runInfo.blkCntOffset == 0;
        CalRowsumAndSftCopyIn(i, dyGmOffset, sumGmOffset, processM, calcSoftmaxGradFront);
        dyGmOffset += (processM * dimDv);
        sumGmOffset += processM;

        CalSoftmax(i, processM, mm12Addr, mm345Addr, runInfo);

        SET_FLAG(MTE3, MTE2, mte2WaitMte3Proc);
        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3Proc);

        CalSoftmaxGrad(i, processM, mm12Addr, mm345Addr, runInfo);

        if (i < subLoopEnd - 1) {
            SET_FLAG(MTE3, MTE2, mte2WaitMte3Proc);
            WAIT_FLAG(MTE3, MTE2, mte2WaitMte3Proc);
        }
        mm12Addr += dataSize;
        mm345Addr += dataSize;
    }
}





template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::MergeKeyOwnedEdge(const RunInfo &runInfo)
{
    const uint32_t totalRows = runInfo.isLastBasicBlock ?
        (runInfo.actualSelCntOffset - 1) * selectedBlockSize + runInfo.lastBlockSize :
        runInfo.actualSelCntOffset * selectedBlockSize;
    const uint32_t firstRows = totalRows / 2;
    const uint32_t rowBegin = subBlockIdx == 0 ? 0 : firstRows;
    const uint32_t currentRows = subBlockIdx == 0 ? firstRows : totalRows - firstRows;
    if (currentRows == 0) {
        return;
    }

    const uint64_t queryIndex = static_cast<uint64_t>(runInfo.indicesGmOffset) /
                                (static_cast<uint64_t>(dimN2) * selectedBlockCount);
    const uint64_t edgeQuerySlot = queryIndex % KeyOwnedEdgeQueryCapacity(tilingData);
    const uint64_t dkSrcOffset = runInfo.scatterTaskId * MAX_CORE_NUM * scatterTokenCapacity * dimDAlign +
                                 cubeBlockIdx * scatterTokenCapacity * dimDAlign + rowBegin * dimDAlign;
    const uint64_t dvSrcOffset = runInfo.scatterTaskId * MAX_CORE_NUM * scatterTokenCapacity * dimD2Align +
                                 cubeBlockIdx * scatterTokenCapacity * dimD2Align + rowBegin * dimD2Align;
    const uint64_t edgeOffset =
        (edgeQuerySlot * selectedBlockCount + runInfo.blkCntOffset * selectedBlockSize + rowBegin) * dimDAlign;
    GlobalTensor<float> dkSrcGm = mm4ResWorkspaceGm[dkSrcOffset];
    GlobalTensor<float> dvSrcGm = mm5ResWorkspaceGm[dvSrcOffset];
    GlobalTensor<float> edgeOutGm = keyOwnedEdgeGm[edgeOffset];

    SetFlag<HardEvent::MTE3_MTE2>(mte2WaitMte3);
    SetFlag<HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    for (uint32_t row = 0; row < currentRows; row += NON_OPT_UB_ROW_SIZE) {
        const uint32_t rows = Min<uint32_t>(NON_OPT_UB_ROW_SIZE, currentRows - row);
        const uint32_t dkElements = rows * dimDAlign;
        const uint32_t dvElements = rows * dimD2Align;
        const event_t release = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
        const event_t ready = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
        WaitFlag<HardEvent::MTE3_MTE2>(release);
        LocalTensor<float> dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
        LocalTensor<float> dvInUb = scatterAddTensorV[pingPongIdx * ubRowSizeD2Align];
        DataCopy(dkInUb, dkSrcGm[row * dimDAlign], dkElements);
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);

        // Preserve the original kernel order: scale FP32 dK after its Cube
        // result is materialized, then add the unscaled FP32 dV contribution.
        Muls(dkInUb, dkInUb, static_cast<float>(tilingData->postTilingData.scaleValue), dkElements);
        DataCopy(dvInUb, dvSrcGm[row * dimD2Align], dvElements);
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams addRows;
        addRows.dstBlkStride = 1;
        addRows.src0BlkStride = 1;
        addRows.src1BlkStride = 1;
        addRows.dstRepStride = dimDAlign / BLOCK_FP32;
        addRows.src0RepStride = dimDAlign / BLOCK_FP32;
        addRows.src1RepStride = dimD2Align / BLOCK_FP32;
        for (uint32_t column = 0; column < dimD2Align; column += repeatMaxSize) {
            const uint32_t columns = Min<uint32_t>(repeatMaxSize, dimD2Align - column);
            Add(dkInUb[column], dkInUb[column], dvInUb[column], columns, rows, addRows);
        }
        SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
        DataCopy(edgeOutGm[row * dimDAlign], dkInUb, dkElements);
        SetFlag<HardEvent::MTE3_MTE2>(release);
        pingPongIdx = 1 - pingPongIdx;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(mte2WaitMte3);
    WaitFlag<HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    SetFlag<HardEvent::MTE3_V>(vWaitMte3);
    WaitFlag<HardEvent::MTE3_V>(vWaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::ScatterAddDeter(const RunInfo &runInfo)
{
    LocalTensor<float> dkInUb;

    GlobalTensor<float> dkOutGm = dkWorkspaceGm[runInfo.mm4OutGmOffset];
    int64_t s2RealSize = Min(selectedBlockCount, runInfo.actualSelectedBlockCount);

    int64_t totalVec = (runInfo.s1End - runInfo.s1Begin) * 2;

    int64_t firstCoreKSize = s2RealSize / totalVec;
    int64_t currentCoreKSize =
        (vecBlockIdx == totalVec - 1) ? (s2RealSize % totalVec + firstCoreKSize) : firstCoreKSize;

    if (currentCoreKSize == 0) {
        return;
    }

    int64_t cubeBlockIdxCalculate = runInfo.s1Index - runInfo.s1Begin;
    int64_t actTotalRows = (vecBlockIdx == totalVec - 1) ?
                               (currentCoreKSize - 1) * selectedBlockSize + runInfo.lastBlockSize :
                               currentCoreKSize * selectedBlockSize;
    int64_t maxLoops = CeilDiv(actTotalRows, NON_OPT_UB_ROW_SIZE);
    int64_t tailRows = actTotalRows - (maxLoops - 1) * NON_OPT_UB_ROW_SIZE;

    int64_t currentDkSrcOffset =
        runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimDAlign +
        cubeBlockIdxCalculate * selectedBlockCount * selectedBlockSizeDimDAlign +
        vecBlockIdx * firstCoreKSize * selectedBlockSizeDimDAlign;
    int64_t currentIndicesOffset = runInfo.indicesGmOffset + vecBlockIdx * firstCoreKSize;
    GlobalTensor<float> dkSrcGm = mm4ResWorkspaceGm[currentDkSrcOffset];
    const int64_t indexBase = currentIndicesOffset & ~7LL;
    const int64_t indexShift = currentIndicesOffset - indexBase;
    const int64_t indexCount = AlignUp(indexShift + currentCoreKSize, 8);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopy(indexTensor, topkIndicesGm[indexBase], indexCount);
    SET_FLAG(MTE2, S, sWaitMte2); WAIT_FLAG(MTE2, S, sWaitMte2);
    SetAtomicAdd<float>();

    int64_t curSelBlk = 0;
    int64_t curRow = 0;
    int32_t s2Idx = 0;
    if (!runInfo.isSmallS2) {
        s2Idx = indexTensor.GetValue(indexShift + curSelBlk);
    }
    int64_t curProcessRow = Min(NON_OPT_UB_ROW_SIZE, selectedBlockSize);

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
        DataCopy(dkInUb, dkSrcGm[loop * ubRowSizeDAlign], ubRowSizeDAlign);
        event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        if (runInfo.isSmallS2) {
            int64_t baseRow = loop * NON_OPT_UB_ROW_SIZE + vecBlockIdx * firstCoreKSize * selectedBlockSize;
            DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb, ubRowSizeDAlign);
        } else {
            for (int64_t row = 0; row < NON_OPT_UB_ROW_SIZE;) {
                if (curRow / selectedBlockSize > curSelBlk) {
                    curSelBlk += 1;
                    s2Idx = indexTensor.GetValue(indexShift + curSelBlk);
                }
                if (s2Idx >= 0 && s2Idx <= runInfo.s1Index + runInfo.curS2 - dimS1) {
                    DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign],
                             dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                }
                row += curProcessRow;
                curRow += curProcessRow;
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        pingPongIdx = 1 - pingPongIdx;
    }
    event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3 : mte2WaitMte3Pong;
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
    DataCopy(dkInUb, dkSrcGm[(maxLoops - 1) * ubRowSizeDAlign], tailRows * dimDAlign);
    event_t event = pingPongIdx == 0 ? vWaitMte2 : vWaitMte2Pong;
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
    WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

    int64_t totalRound = CeilDiv(tailRows, curProcessRow);
    int64_t row = 0;
    if (runInfo.isSmallS2) {
        int64_t baseRow = (maxLoops - 1) * NON_OPT_UB_ROW_SIZE + vecBlockIdx * firstCoreKSize * selectedBlockSize;
        DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb, tailRows * dimDAlign);
    } else {
        for (int64_t loop = 0; loop < totalRound; loop++) {
            if (curRow / selectedBlockSize > curSelBlk) {
                curSelBlk += 1;
                s2Idx = indexTensor.GetValue(indexShift + curSelBlk);
            }
            if (s2Idx >= 0 && s2Idx <= runInfo.s1Index + runInfo.curS2 - dimS1) {
                if (vecBlockIdx == totalVec - 1 && loop == totalRound - 1) {
                    curProcessRow =
                        (runInfo.lastBlockSize % curProcessRow) ? runInfo.lastBlockSize % curProcessRow : curProcessRow;
                }
                DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign],
                         dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
            }
            row += curProcessRow;
            curRow += curProcessRow;
        }
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    pingPongIdx = 1 - pingPongIdx;

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

    SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
    SetAtomicNone();
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::ScatterAdd(const RunInfo &runInfo)
{
    if constexpr (IS_DETERMINISTIC) {
        ScatterAddDeter(runInfo);
    } else {
        ScatterAddUnDeter(runInfo);
    }
}

} // namespace SFAG_KEY_OWNED_FULL_REDUCE
