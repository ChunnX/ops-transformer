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
 * \file sparse_flash_attention_grad_block_vec.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_GRAD_BLOCK_VEC_H
#define SPARSE_FLASH_ATTENTION_GRAD_BLOCK_VEC_H
 
#include "sparse_flash_attention_grad_common.h"
#include "vector_api/cast_softmax_grad.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax.h"
#include "vector_api/vf_broadcast_sub_mul.h"
#include "vector_api/vf_cast_transdata_deconflict.h"
namespace SfagBaseApi {
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t SYNC_V0_V1_DS_A_MAX_DONE_FLAG = 10;
constexpr uint32_t BIT_MASK_NUM = 8;
// 【新增】dSink slot 步长别名（optiling::sfag::MAX_N1 的 SfagBaseApi 内可见名）
constexpr int64_t MAX_N1 = optiling::sfag::MAX_N1;

TEMPLATES_DEF
class FAGBlockVec {
private:
    __aicore__ inline void GetBS1Index(int64_t bS1Index, int64_t &bIdx, int64_t &s1Idx, FagConstInfo &constInfo);
    __aicore__ inline void GetRunInfo(int64_t bIdx, int64_t s1Idx, FagRunInfo &runInfo, FagConstInfo &constInfo,
                                        int64_t accumS2Len, int32_t actualSeqLensQ, int32_t actualSeqLensK);
    __aicore__ inline int32_t GetActualSeqLens(int64_t bIdx, GlobalTensor<int32_t> &actualSeqLensGm,
                                                int64_t &accumLen, FagConstInfo constInfo);
public:
    __aicore__ inline FAGBlockVec(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo, AttenMaskInfo &attenMaskInfo,
                                                                     PseInfo &pseInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR dy, GM_ADDR y, GM_ADDR sparseIndices,
                                            GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR keyRope,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR actualSeqQlen,
                                            GM_ADDR actualSeqKvlen, GM_ADDR workspace, GM_ADDR sinks, GM_ADDR dSinks);
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void InitCubeVecSharedParams(FagCVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, float qScaleDs);
    __aicore__ inline void GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm, const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                      LocalTensor<CALC_TYPE> &dkInTensor, LocalTensor<CALC_TYPE> &dvInTensor,
                                      FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                           FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t taskId);
    // 【新增】dSink 第1步：S1 循环末尾写回 dSinkAcc（deter→dSinkWorkSpace 本核 slot halfG 半区；非 deter→AtomicAdd 直写 dSinksGm）
    __aicore__ inline void FinalizeDSinkAcc(FagConstInfo &constInfo);
    // 【新增】dSink 第2步：循环外单 writer 跨核 reduce（仅 deter；搬 slot→scratch + Add 归约→dSinksGm）
    __aicore__ inline void ReduceDSink(LocalTensor<CALC_TYPE> &scratchTensor, LocalTensor<CALC_TYPE> &reduceOutTensor,
                                       FagConstInfo &constInfo);
    template <const bool IS_DQ = false>
    __aicore__ inline void CopyUB2L1(FagConstInfo &constInfo, FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                     LocalTensor<INPUT_TYPE> &srcTensor);

    constexpr static bool IS_D_NO_EQUAL = true;
    constexpr static bool IS_FP8_INPUT =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static float FP8_MAX = IsSameType<INPUT_TYPE, fp8_e5m2_t>::value ? 57344 : 448;
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = 128;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t DETER_EXCEED_USE_SIZE = 2 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 32 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D512 = 64 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE =
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D512 : DETER_DQ_UB_SIZE_FP32_D256) :
                        DETER_DQ_UB_SIZE_FP16;
                        
    // vector gm addr
    GlobalTensor<INPUT_TYPE> keyGm, keyRopeGm, dyGm;
    GlobalTensor<OUTDTYPE> yGm, pseGm;
    GlobalTensor<uint8_t> dropMaskGm, attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, pseFloatGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm;
    GM_ADDR pseSlope;
    GlobalTensor<uint8_t> dropMaskWorkspaceGm;
    GlobalTensor<float> dsAmaxWorkSpaceGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm;
    GlobalTensor<int32_t> actualSeqLengthsKeyGm;
    // 【新增】sink 相关 GM：sinksGm(输入)、dSinksGm(输出)、dSinkWorkSpaceGm(跨核 reduce 中间，deter)
    GlobalTensor<float> sinksGm, dSinksGm, dSinkWorkSpaceGm;
 
    // ub buffer
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECIN, 1> maxSumQue[2];
    TBuf<> softmaxGradResBuf;
    TBuf<> dropMaskBuf;
    TBuf<> dropmaskIndexVecBuf;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterInOutQue;
    TBuf<> deterOffsetBuf;
    TBuf<> vselrIndexesBuf;
    TQue<QuePosition::VECOUT, 1> dsAmaxOutQue;
    // 【新增】sink 相关 UB buffer（均 halfG 半份；sinks 为空时 IS_SINKS=false 不分配，零开销）
    // sink 路径与主路径同用 firstHalfGRealSize/halfGRealSize（平衡切分，P_sink 与 D_i 在 V3 做 Mul 须同切分）。
    // sinks/max/sum 各一个 TQue，DataCopyPad {1,halfG*4,0,0} 单块连续搬运（非对齐 N1=24 也可连续）；
    // 不复用 maxSumQue——它是 {halfG,4,0,0} 摊开布局（BRC 读），sink 用微指令 default LoadAlign 连续读，布局不兼容。
    // P_sink 微指令算（LoadAlign/Sub/Exp/Div/StoreAlign），结果写 sinkTensor(TBuf) 给 V3 标准 Mul 用。
    TQue<QuePosition::VECIN, 1> sinksInQue;    // sinks GM→TQue（DataCopyPad 单块连续）
    TQue<QuePosition::VECIN, 1> maxForSinkQue; // max  GM→TQue（sink 专用连续，与 maxSumQue 独立）
    TQue<QuePosition::VECIN, 1> sumForSinkQue; // sum  GM→TQue
    TBuf<> sinkTensor;                         // P_sink=exp(sink-m)/ℓ 暂存（V2→V3 交接，微指令 StoreAlign 写、V3 标准 Mul 读）
    TBuf<> dSinkAcc;                           // dSink per-sub-block 累加器（跨 S1 token 累加，持久存活）
 
    TPipe *pipe;
    SFagTilingType tilingData;
 
    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;
 
    // optional info
    AttenMaskInfo *attenMaskInfoPtr;
    PseInfo *pseInfoPtr;

    DataCopyPadExtParams<INPUT_TYPE> padParams;
    DataCopyExtParams intriParamsKey;
    DataCopyExtParams intriParamsRope;
    DataCopyExtParams outParamK;
    DataCopyExtParams outParamRope;
};
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData,
                                                                     uint32_t vBlockIdx, uint32_t cBlockIdx,
                                                                     uint32_t vSubBlockIdx, FagConstInfo &constInfo, AttenMaskInfo &attenMaskInfo,
                                                                     PseInfo &pseInfo)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->vBlockIdx = vBlockIdx;
    this->cBlockIdx = cBlockIdx;
    this->vSubBlockIdx = vSubBlockIdx;
    attenMaskInfoPtr = &attenMaskInfo;
    pseInfoPtr = &pseInfo;

    intriParamsKey.blockLen = constInfo.selectedBlockSize * constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
    intriParamsKey.dstStride = 0;
    intriParamsKey.blockCount = 2;

    intriParamsRope.blockLen = constInfo.selectedBlockSize * constInfo.dRopeSize * sizeof(INPUT_TYPE);
    intriParamsRope.dstStride = 0;
    intriParamsRope.blockCount = 2;

    outParamK.blockCount = 2;
    outParamK.blockLen = constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
    outParamK.srcStride = 0;
    outParamK.dstStride = constInfo.dRopeSize * sizeof(INPUT_TYPE);

    outParamRope.blockCount = 2;
    outParamRope.blockLen = constInfo.dRopeSize * sizeof(INPUT_TYPE);
    outParamRope.srcStride = 0;
    outParamRope.dstStride = constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR key, GM_ADDR dy, GM_ADDR y, GM_ADDR sparseIndices,
                                                                    GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR keyRope,
                                                                    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                                                    GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
                                                                    GM_ADDR workspace, GM_ADDR sinks, GM_ADDR dSinks)
{
    keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
    keyRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)keyRope);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dy);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);

    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);

    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlen);
    actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqKvlen);

    // 【新增】sink GM 绑定：仅 IS_SINKS 时绑定（IS_SINKS=false 完全不触达，等同原版）
    if constexpr (IS_SINKS) {
        sinksGm.SetGlobalBuffer((__gm__ float *)sinks);
        dSinksGm.SetGlobalBuffer((__gm__ float *)dSinks);
        dSinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                         tilingData->postTilingData.dSinkWorkSpaceOffset / sizeof(float));
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitUbBuffer()
{
    /**
     * UB划分，buffer大小分配
     * attenMaskOrYInQue: for y and attenMask
     * pseOrDyInQue: for dy and pse
     * dSOutQue: for dq dk left ub matrix
     * pOutQue: for dv left ub matrix
     * softmaxGradResBuf: for softmax_grad result
     * dropMaskBuf: for dropMask
     * maxSumQue: for max sum double buffer
     **/
    pipe->InitBuffer(attenMaskOrYInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(pseOrDyInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(OUTDTYPE));
    pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
    pipe->InitBuffer(maxSumQue[0], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
    pipe->InitBuffer(maxSumQue[1], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);

    pipe->InitBuffer(dSOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE + DETER_EXCEED_USE_SIZE);
    pipe->InitBuffer(pOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE);

    // 【新增】sink 相关 UB（仅 IS_SINKS 分配；dSinkAcc 跨 S1 循环累加，初始化清零）
    if constexpr (IS_SINKS) {
        pipe->InitBuffer(sinksInQue, 1, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(maxForSinkQue, 1, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(sumForSinkQue, 1, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(sinkTensor, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(dSinkAcc, VECTOR_BASEM * sizeof(CALC_TYPE));
        LocalTensor<CALC_TYPE> dSinkAccTensor = dSinkAcc.Get<CALC_TYPE>();
        Duplicate(dSinkAccTensor, 0.0f, VECTOR_BASEM);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    outParamK.blockCount = 2;
    outParamRope.blockCount = 2;
    uint64_t gmOffset = runInfo.t1Index * (constInfo.n2Size * constInfo.selectedBlockCount) + runInfo.n2Index * constInfo.selectedBlockCount + runInfo.blkCntOffset;
    uint32_t mergePingPong = 0;
    AscendC::TEventID mte2WaitMte3EventId;
    AscendC::TEventID mte3WaitMte2EventId;
    AscendC::TEventID mte2WaitMte3Ping = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    AscendC::TEventID mte2WaitMte3Pong = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    AscendC::TEventID mte3WaitMte2Ping = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE3>();
    AscendC::TEventID mte3WaitMte2Pong = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE3>();
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Ping);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

    // ------------- MergeKv --------------
    uint32_t s2Pair = CeilDiv(runInfo.actualSelCntOffset, 2) * 2;
    uint32_t firstVecEnd = (s2Pair / 2);
    uint32_t curBlk = GetSubBlockIdx() == 0 ? 0 : firstVecEnd;
    uint32_t curActualSelCntEnd = GetSubBlockIdx() == 0 ? firstVecEnd : runInfo.actualSelCntOffset;
    uint32_t curActualSelCntOffset = curActualSelCntEnd - curBlk;
    uint64_t outWsOffset = GetSubBlockIdx() == 0 ? 0 : firstVecEnd * constInfo.selectedBlockSize * constInfo.dTotalSize;
    uint32_t i;

    LocalTensor<INPUT_TYPE> gatherTensorPing = dSOutQue.AllocTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> gatherTensorPong = pOutQue.AllocTensor<INPUT_TYPE>();

    LocalTensor<INPUT_TYPE> gatherRopeTensorPing = gatherTensorPing[2 * constInfo.commonConstInfo.dSize];
    LocalTensor<INPUT_TYPE> gatherRopeTensorPong = gatherTensorPong[2 * constInfo.commonConstInfo.dSize];

    for (i = curBlk; i < curBlk + curActualSelCntOffset / 2 * 2; i += 2) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i) * constInfo.selectedBlockSize;
        int64_t keyOffset2 = topkIndicesGm.GetValue(gmOffset + i + 1) * constInfo.selectedBlockSize;

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - constInfo.selectedBlockSize;
        intriParamsKey.blockCount = 2;
        intriParamsRope.blockCount = 2;

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Ping : mte2WaitMte3Pong;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Ping : mte3WaitMte2Pong;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        // CopyIn
        intriParamsKey.srcStride = s2OrgStride * constInfo.n2Size * constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
        LocalTensor<INPUT_TYPE> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;

        if (keyOffset2 <= keyOffset1) {
            intriParamsKey.blockCount = 1;
            DataCopyPad(gatherTensor, keyGm[runInfo.keyOffsetWithRopeForMm12 + keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize], intriParamsKey, padParams);
            DataCopyPad(gatherTensor[constInfo.selectedBlockSize * constInfo.commonConstInfo.dSize], keyGm[runInfo.keyOffsetWithRopeForMm12 + keyOffset2 * constInfo.n2Size * constInfo.commonConstInfo.dSize], intriParamsKey, padParams);
        } else {
            DataCopyPad(gatherTensor, keyGm[runInfo.keyOffsetWithRopeForMm12 + keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize], intriParamsKey, padParams);
        }

        intriParamsRope.srcStride = s2OrgStride * constInfo.n2Size * constInfo.dRopeSize * sizeof(INPUT_TYPE);
        LocalTensor<INPUT_TYPE> &gatherRopeTensor = mergePingPong ? gatherRopeTensorPing : gatherRopeTensorPong;

        if constexpr (IS_ROPE) {
            if (keyOffset2 <= keyOffset1) {
                intriParamsRope.blockCount = 1;
                DataCopyPad(
                    gatherRopeTensor,
                    keyRopeGm[runInfo.commonRunInfo.kRopeOffset + keyOffset1 * constInfo.n2Size * constInfo.dRopeSize],
                    intriParamsRope, padParams);
                DataCopyPad(
                    gatherRopeTensor[constInfo.selectedBlockSize * constInfo.dRopeSize],
                    keyRopeGm[runInfo.commonRunInfo.kRopeOffset + keyOffset2 * constInfo.n2Size * constInfo.dRopeSize],
                    intriParamsRope, padParams);
            } else {
                DataCopyPad(
                    gatherRopeTensor,
                    keyRopeGm[runInfo.commonRunInfo.kRopeOffset + keyOffset1 * constInfo.n2Size * constInfo.dRopeSize],
                    intriParamsRope, padParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        // CopyOut
        DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
        if constexpr (IS_ROPE) {
            DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + constInfo.commonConstInfo.dSize + outWsOffset],
                        gatherRopeTensor, outParamRope);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        outWsOffset += 2 * constInfo.dTotalSize;
        mergePingPong = 1 - mergePingPong;
    }
    if (i < curActualSelCntEnd) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i) * constInfo.selectedBlockSize;

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Ping : mte2WaitMte3Pong;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Ping : mte3WaitMte2Pong;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);

        // CopyIn
        intriParamsRope.blockCount = 1;
        intriParamsKey.blockCount = 1;
        LocalTensor<INPUT_TYPE> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<INPUT_TYPE> &gatherRopeTensor = mergePingPong ? gatherRopeTensorPing : gatherRopeTensorPong;

        DataCopyPad(gatherTensor, keyGm[runInfo.keyOffsetWithRopeForMm12 + keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize], intriParamsKey, padParams);
        if constexpr (IS_ROPE) {
            DataCopyPad(
                gatherRopeTensor,
                keyRopeGm[runInfo.commonRunInfo.kRopeOffset + keyOffset1 * constInfo.n2Size * constInfo.dRopeSize],
                intriParamsRope, padParams);
        }

        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);

        outParamK.blockCount = 1;
        outParamRope.blockCount = 1;
        // CopyOut
        DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
        if constexpr (IS_ROPE) {
            DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + constInfo.commonConstInfo.dSize + outWsOffset],
                        gatherRopeTensor, outParamRope);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        mergePingPong = 1 - mergePingPong;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Ping);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    dSOutQue.FreeTensor(gatherTensorPing);
    pOutQue.FreeTensor(gatherTensorPong);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: Cast + SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    uint32_t loopNum = Ceil<uint32_t>(runInfo.halfGRealSize, constInfo.sfmgMaxLoopSize);
    uint32_t loopSize = Ceil<uint32_t>(runInfo.halfGRealSize, loopNum);
    uint32_t tailLoopSize = runInfo.halfGRealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    for (int32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
        }
        CopyInSoftmaxGrad<INPUT_TYPE, CALC_TYPE, OUTDTYPE, VECTOR_BASEM, 512, IS_D_NO_EQUAL>(
            constInfo, runInfo, loopIdx, curLoopSize, loopSize, attenMaskOrYInQue, pseOrDyInQue, dyGm, yGm);
        CalculateCastSoftmaxGrad<INPUT_TYPE, CALC_TYPE, OUTDTYPE, VECTOR_BASEM, 512>(
            constInfo, curLoopSize, attenMaskOrYInQue, pseOrDyInQue, softmaxGradResTensor[loopSize * loopIdx],
            runInfo.quantScaleInfo.deqScaleDyValue);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                              int64_t taskId)
{
    CopyInMaxSum<float, VECTOR_BASEM>(constInfo, runInfo, maxSumQue[taskId & 1], softmaxMaxGm, softmaxSumGm);
}

// 【新增】dSink 第1步：S1 循环末尾写回 dSinkAcc（halfG 半份）
// deter：dSinkAcc→dSinkWorkSpace 本核 slot 对应 halfG 半区（2 sub-block 各写不相交半份沿 G 轴拼成完整 [N1] slot）
// 非 deter：SetAtomicAdd + DataCopy 半份直写 dSinksGm（[N1] 输出，顺序不定符合非确定语义）
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::FinalizeDSinkAcc(FagConstInfo &constInfo)
{
    if constexpr (IS_SINKS) {
        // V→MTE3 同步：dSinkAcc 由 V3 的 Mul/Sub 写入，DataCopy(UB→GM) 走 MTE3 须等 V 完成
        event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);

        LocalTensor<CALC_TYPE> dSinkAccTensor = dSinkAcc.Get<CALC_TYPE>();
        int64_t gSize = constInfo.commonConstInfo.gSize;
        int64_t firstHalfG = (gSize + 1) >> 1;
        int64_t halfGRealSize = (vSubBlockIdx == 0) ? firstHalfG : (gSize - firstHalfG);
        int64_t halfGOffset = firstHalfG * vSubBlockIdx;  // sub0: 0, sub1: firstHalfG
        if (halfGRealSize <= 0) {
            return;  // 边界：gSize=1 时 sub1 halfG=0，跳过（无数据可写）
        }

        if constexpr (IS_DETER) {
            // deter：写本核 slot 对应 halfG 半区（slot 起始 512B 对齐，DataCopyPad 处理 halfG 偏移）
            int64_t slotOffset = static_cast<int64_t>(cBlockIdx) * MAX_N1 + halfGOffset;
            DataCopyExtParams dSinkCopyParams;
            dSinkCopyParams.blockCount = 1;
            dSinkCopyParams.blockLen = static_cast<uint16_t>(halfGRealSize * sizeof(CALC_TYPE));
            dSinkCopyParams.srcStride = 0;
            dSinkCopyParams.dstStride = 0;
            DataCopyPad(dSinkWorkSpaceGm[slotOffset], dSinkAccTensor, dSinkCopyParams);
        } else {
            // 非 deter：AtomicAdd 半份到 dSinksGm（[N1] 输出，顺序不定符合非确定语义）
            // 用 DataCopyPad（字节级 blockLen）而非 DataCopy——halfGOffset/halfGRealSize 在小 N1 下
            // 非 32B 对齐，DataCopy 是对齐搬运接口语义未定义；DataCopyPad 专为非对齐设计，稳妥。
            DataCopyExtParams dSinkCopyParams;
            dSinkCopyParams.blockCount = 1;
            dSinkCopyParams.blockLen = static_cast<uint16_t>(halfGRealSize * sizeof(CALC_TYPE));
            dSinkCopyParams.srcStride = 0;
            dSinkCopyParams.dstStride = 0;
            SetAtomicAdd<CALC_TYPE>();
            DataCopyPad(dSinksGm[halfGOffset], dSinkAccTensor, dSinkCopyParams);
            SetAtomicNone();
        }
    }
}

// 【新增】dSink 第2步：循环外，仅 deter，单 writer（cBlockIdx==0 && vSubBlockIdx==0）跨核 reduce
// 全核 barrier 后，单 writer 搬 usedCoreNum 个 slot→scratch + Add 归约→dSinksGm
// scratch/reduceOut 复用 mm1ResBuf[0]/[1]（循环后空闲，TBuf 无 Position 约束，PipeBarrier 手动管同步）
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ReduceDSink(LocalTensor<CALC_TYPE> &scratchTensor,
                                                               LocalTensor<CALC_TYPE> &reduceOutTensor,
                                                               FagConstInfo &constInfo)
{
    if constexpr (IS_SINKS && IS_DETER) {
        // AIV 间全核 barrier：复用 SCATTER_VEC_SYNC_FLAG(12) + modeId=0 + PIPE_MTE3（set+wait 配对=1 轮）
        CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
        CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
        if (cBlockIdx == 0 && vSubBlockIdx == 0) {
            int64_t coreNum = tilingData->baseParams.usedCoreNum;
            // MTE2: 搬入 coreNum 个 slot（每 MAX_N1 floats，slots 连续，count 32B 对齐）→ scratch
            DataCopy(scratchTensor, dSinkWorkSpaceGm[0], coreNum * MAX_N1);
            PipeBarrier<PIPE_MTE2>();
            // V: Add 归约 over 核序 → reduceOut[0..MAX_N1)
            Duplicate(reduceOutTensor, 0.0f, MAX_N1);
            for (int64_t c = 0; c < coreNum; c++) {
                Add(reduceOutTensor, reduceOutTensor, scratchTensor[c * MAX_N1], MAX_N1);
            }
            PipeBarrier<PIPE_V>();
            // MTE3: 写 dSinksGm[0..N1)（N1 = gSize，n2=1；DataCopyPad 处理非对齐）
            int64_t n1 = constInfo.commonConstInfo.gSize;
            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = static_cast<uint16_t>(n1 * sizeof(CALC_TYPE));
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            DataCopyPad(dSinksGm[0], reduceOutTensor, outParams);
        }
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_DQ>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1(FagConstInfo &constInfo, FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                                             LocalTensor<INPUT_TYPE> &srcTensor)
{
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    uint32_t scmOffset = vSubBlockIdx == 0 ? 0 : runInfo.firstHalfGRealSize * FRACTAL_NZ_C0_SIZE;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE;
    dataCopyParams.blockLen = (uint16_t)(runInfo.halfGRealSize * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
    dataCopyParams.srcStride =
        (uint16_t)((VECTOR_BASEM + 1 - runInfo.halfGRealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
        uint32_t s1RealSizeAlignTo16 = AlignTo16(constInfo.commonConstInfo.gSize);
    dataCopyParams.dstStride =
        (s1RealSizeAlignTo16 - runInfo.halfGRealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
    DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_SINKS) {
        // 【新增】sink 路径：sinks/max/sum 各自 DataCopyPad 单块连续进 sink 专用 TQue；max/sum 不复用 maxSumQue（布局不兼容）
        if (runInfo.halfGRealSize == 0) {
            return;
        }
        LocalTensor<CALC_TYPE> maxSumTensor = maxSumQue[runInfo.commonRunInfo.taskIdMod2].DeQue<CALC_TYPE>();
        CopyInSinksAndMaxSum<CALC_TYPE, VECTOR_BASEM>(constInfo, runInfo, sinksGm, softmaxMaxGm, softmaxSumGm,
                                                      sinksInQue, maxForSinkQue, sumForSinkQue);  // MTE2→TQue + EnQue
        // P 路径（原功能不变）：maxSumTensor 来自 maxSumQue，Reuse 重载不 Free，留到下面统一 Free
        CalculatePseMulsSelSimpleSoftMaxReuse<OUTDTYPE, CALC_TYPE, false, false, false, VECTOR_BASEM, VECTOR_BASEN>(
            constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, maxSumTensor,
            attenMaskOrYInQue, pseOrDyInQue, mm2ResTensor, mm2ResTensor, pseSlope);
        LocalTensor<CALC_TYPE> sinkPkTensor = sinkTensor.Get<CALC_TYPE>();
        // P_sink：微指令直接读 sink 专用 TQue（TQue DeQue 自动 MTE2→V 同步，无需 PipeBarrier）
        CalculateSinkSimpleSoftMax<CALC_TYPE, VECTOR_BASEM>(constInfo, runInfo,
                                                            sinksInQue, maxForSinkQue, sumForSinkQue, sinkPkTensor);
        maxSumQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(maxSumTensor);
    } else {
        // 原版路径（IS_SINKS=false）：行为与改动前完全一致
        CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, false, false, false, VECTOR_BASEM, VECTOR_BASEN>(
            constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, maxSumQue[runInfo.commonRunInfo.taskIdMod2],
            attenMaskOrYInQue, pseOrDyInQue, mm2ResTensor, mm2ResTensor, pseSlope);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////
 
    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    LocalTensor<INPUT_TYPE> vecOutBuffer = dSOutQue.AllocTensor<INPUT_TYPE>();
    if (runInfo.commonRunInfo.s2RealSize > static_cast<uint32_t>(S2TemplateType::Aligned64)) {
        BroadcastSubMul<CALC_TYPE, static_cast<uint32_t>(S2TemplateType::Aligned128), 0>(mm1ResTensor, mm1ResTensor,
            softmaxGradResTensor, mm2ResTensor, runInfo.halfGRealSize, runInfo.commonRunInfo.s2RealSize);
    } else {
        BroadcastSubMul<CALC_TYPE, static_cast<uint32_t>(S2TemplateType::Aligned64), 0>(mm1ResTensor, mm1ResTensor,
            softmaxGradResTensor, mm2ResTensor, runInfo.halfGRealSize, runInfo.commonRunInfo.s2RealSize);
    }

    LocalTensor<uint8_t> selrIndexesTensor;
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer, mm1ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.DeQue<INPUT_TYPE>();
 
    // copy ds from ub to l1
    LocalTensor<INPUT_TYPE> dsL1Tensor = dstBuffer.GetTensor<INPUT_TYPE>();
    CopyUB2L1<true>(constInfo, runInfo, dsL1Tensor, vecOutBuffer);

    dSOutQue.FreeTensor(vecOutBuffer);

    // 【新增】dSink 累加：dSinkAcc += P_sink · (-D_i)（sinks 存在时）
    if constexpr (IS_SINKS) {
        LocalTensor<CALC_TYPE> sinkPkTensor = sinkTensor.Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> dSinkAccTensor = dSinkAcc.Get<CALC_TYPE>();
        // sinkPk = P_sink × D_i（D_i 主路径切分，softmaxGradResTensor[0:halfG]）
        Mul(sinkPkTensor, sinkPkTensor, softmaxGradResTensor, runInfo.halfGRealSize);
        // dSinkAcc = -sinkPk（= P_sink·(-D_i)）。Muls 只写 dSinkAcc 不读（读 dSinkAcc 16x，见下注）
        // 注：不能用 Sub(dSinkAcc,dSinkAcc,sinkPk) 或 Axpy——读 dSinkAcc 16x。Muls 覆盖语义，S1=1 正确；S1>1 需另设计。
        Muls(dSinkAccTensor, sinkPkTensor, (CALC_TYPE)-1.0, runInfo.halfGRealSize);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF5: cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<uint8_t> selrIndexesTensor;
    LocalTensor<INPUT_TYPE> vecOutBuffer1 = pOutQue.AllocTensor<INPUT_TYPE>();
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer1, mm2ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    pOutQue.EnQue(vecOutBuffer1);
    pOutQue.DeQue<INPUT_TYPE>();
 
    // copy p from ub to l1
    LocalTensor<INPUT_TYPE> pL1Tensor = dstBuffer.GetTensor<INPUT_TYPE>();
    CopyUB2L1(constInfo, runInfo, pL1Tensor, vecOutBuffer1);
 
    pOutQue.FreeTensor(vecOutBuffer1);
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    FagCVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, float qScaleDs)
{
    sharedParams.qScaleDs = qScaleDs;
    /* ssbuf send message */
    if ASCEND_IS_AIV {  
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);  // 
            #pragma unroll
            for (int i = 0; i < sizeof(FagCVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                                              LocalTensor<CALC_TYPE> &dkInTensor,
                                                              LocalTensor<CALC_TYPE> &dvInTensor,
                                                              FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    int64_t UB_ROW_SIZE = 8;
    int64_t s2RealSize = runInfo.commonRunInfo.s2RealSize;
    int64_t firstCoreKSize = s2RealSize / 2;
    int64_t currentCoreKSize = (vSubBlockIdx == 0) ? firstCoreKSize : (s2RealSize - firstCoreKSize);
    if (currentCoreKSize == 0) {
        return;
    }
    
    SetAtomicAdd<CALC_TYPE>();
    int64_t maxLoops = CeilDiv(currentCoreKSize, UB_ROW_SIZE);
    int64_t tailRows = currentCoreKSize - (maxLoops - 1) * UB_ROW_SIZE;

    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    GlobalTensor<float> dkOutGm = dkWorkSpaceGm[runInfo.keyOffsetWithRope];
    GlobalTensor<float> dvOutGm = dvWorkSpaceGm[runInfo.commonRunInfo.valueOffset];
    int64_t currentMm4SrcOffset = runInfo.mm4ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
    int64_t currentMm5SrcOffset = runInfo.mm5ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * 512;
    // 1 - main loop
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    uint64_t gmOffset = runInfo.t1Index * (constInfo.n2Size * constInfo.selectedBlockCount) + vSubBlockIdx * firstCoreKSize + runInfo.blkCntOffset;
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        DataCopy(dkInTensor,
                 mm4ResWorkSpaceGm[currentMm4SrcOffset + loop * UB_ROW_SIZE * HEAD_DIM_ALIGN], UB_ROW_SIZE * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, UB_ROW_SIZE * HEAD_DIM_ALIGN);
        DataCopy(dvInTensor,
                 mm5ResWorkSpaceGm[currentMm5SrcOffset + loop * UB_ROW_SIZE * 512], UB_ROW_SIZE * 512);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
            int32_t s2Idx = topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
            if (s2Idx >= 0) {
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                DataCopy(dvOutGm[s2Idx * constInfo.commonConstInfo.dSizeV],
                         dvInTensor[row * constInfo.commonConstInfo.dSizeV],
                         constInfo.commonConstInfo.dSizeV);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    // 2 - tail loop
    DataCopy(dkInTensor,
                 mm4ResWorkSpaceGm[currentMm4SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN], tailRows * HEAD_DIM_ALIGN);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, tailRows * HEAD_DIM_ALIGN);
    DataCopy(dvInTensor,
                 mm5ResWorkSpaceGm[currentMm5SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * 512], tailRows * 512);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    for (int64_t row = 0; row < tailRows; row++) {
        int32_t s2Idx = topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
        if (s2Idx >= 0) {
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
            DataCopy(dvOutGm[s2Idx * constInfo.commonConstInfo.dSizeV],
                     dvInTensor[row * constInfo.commonConstInfo.dSizeV],
                     constInfo.commonConstInfo.dSizeV);
        }
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    SetAtomicNone();
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                                                   FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    int64_t UB_ROW_SIZE = 8;
    int64_t usedCoreNum = tilingData->baseParams.usedCoreNum;
    int64_t coreNum = GetBlockNum();
    int64_t totalVec = coreNum * 2;
    int64_t bS1Index = -1;
    int64_t bIdx;
    int64_t s1Idx;
    int64_t preBIdx = -1;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int64_t actualSeqLensQ = 0;
    int64_t actualSeqLensK = 0;
    LocalTensor<CALC_TYPE> dkInTensor = dSOutQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> dvInTensor = pOutQue.AllocTensor<CALC_TYPE>();
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    for (int64_t idx = 0; idx < usedCoreNum; idx++) {
        bS1Index = runInfo.sTaskId * usedCoreNum + idx;
        if (bS1Index >= tilingData->baseParams.totalSize) {
            break;
        }

        GetBS1Index(bS1Index, bIdx, s1Idx, constInfo);
        if (bIdx != preBIdx) {
            actualSeqLensQ = GetActualSeqLens(bIdx, this->actualSeqLengthsQueryGm, accumS1Len, constInfo);
            actualSeqLensK = GetActualSeqLens(bIdx, this->actualSeqLengthsKeyGm, accumS2Len, constInfo);
            preBIdx = bIdx;
        }
        GetRunInfo(bIdx, s1Idx, runInfo, constInfo, accumS2Len, actualSeqLensQ, actualSeqLensK);

        // 对应s2real的平均和取模值，做均分操作
        int64_t remainder = runInfo.actualSelectedBlockCount % totalVec;
        int64_t avgSize = runInfo.actualSelectedBlockCount / totalVec;
        // 前remainder 个块分配 avgSize + 1， 其余分配avgSize
        int64_t currentCoreKSize = avgSize + (this->vBlockIdx < remainder ? 1 : 0);
        if (currentCoreKSize == 0) {
            // V核同步等待所有V核完成某一个C核上S2的计算
            CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
            continue;
        }
        int64_t s2SrcOffset = this->vBlockIdx < remainder ? (this->vBlockIdx * (avgSize + 1)) :
                                (remainder * (avgSize + 1) + (this->vBlockIdx - remainder) * avgSize);
        
        SetAtomicAdd<CALC_TYPE>();
        int64_t maxLoops = CeilDiv(currentCoreKSize, UB_ROW_SIZE);
        int64_t tailRows = currentCoreKSize - (maxLoops - 1) * UB_ROW_SIZE;

        int64_t currentMm4SrcOffset = runInfo.sTaskIdMod2 * constInfo.selectedBlockCount * HEAD_DIM_ALIGN * coreNum +
                                      idx * constInfo.selectedBlockCount * HEAD_DIM_ALIGN +
                                      s2SrcOffset * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
        int64_t currentMm5SrcOffset = runInfo.sTaskIdMod2 * constInfo.selectedBlockCount *
                                      constInfo.commonConstInfo.dSizeV * coreNum +
                                      idx * constInfo.selectedBlockCount * constInfo.commonConstInfo.dSizeV +
                                      s2SrcOffset * constInfo.selectedBlockSize * constInfo.commonConstInfo.dSizeV;
        // 1 - main loop
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        uint64_t gmOffset = bS1Index * (constInfo.n2Size * constInfo.selectedBlockCount) + s2SrcOffset;
        for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            DataCopy(dkInTensor,
                     mm4ResWorkSpaceGm[currentMm4SrcOffset + loop * UB_ROW_SIZE * HEAD_DIM_ALIGN],
                     UB_ROW_SIZE * HEAD_DIM_ALIGN);
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, UB_ROW_SIZE * HEAD_DIM_ALIGN);
            DataCopy(dvInTensor,
                     mm5ResWorkSpaceGm[currentMm5SrcOffset + loop * UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV],
                     UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV);
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                int32_t s2Idx = topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
                if (s2Idx >= 0) {
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    DataCopy(dkWorkSpaceGm[runInfo.keyOffsetWithRope + s2Idx * HEAD_DIM_ALIGN],
                             dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                    int64_t dvOffset = runInfo.commonRunInfo.valueOffset + s2Idx * constInfo.commonConstInfo.dSizeV;
                    DataCopy(dvWorkSpaceGm[dvOffset],
                             dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
                }
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        // 2 - tail loop
        DataCopy(dkInTensor,
                 mm4ResWorkSpaceGm[currentMm4SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN],
                 tailRows * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, tailRows * HEAD_DIM_ALIGN);
        DataCopy(dvInTensor,
                 mm5ResWorkSpaceGm[currentMm5SrcOffset +
                                  (maxLoops - 1) * UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV],
                 tailRows * constInfo.commonConstInfo.dSizeV);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        for (int64_t row = 0; row < tailRows; row++) {
            int32_t s2Idx = topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
            if (s2Idx >= 0) {
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopy(dkWorkSpaceGm[runInfo.keyOffsetWithRope + s2Idx * HEAD_DIM_ALIGN],
                         dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                DataCopy(dvWorkSpaceGm[runInfo.commonRunInfo.valueOffset + s2Idx * constInfo.commonConstInfo.dSizeV],
                         dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        // V核同步等待所有V核完成某一个C核上S2的计算
        CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
        CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
    }
    SetAtomicNone();
    dSOutQue.FreeTensor(dkInTensor);
    pOutQue.FreeTensor(dvInTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::GetBS1Index(int64_t bS1Index, int64_t &bIdx,
                                                               int64_t &s1Idx, FagConstInfo &constInfo)
{
    if constexpr (IS_TND) {
        int64_t actualSum = 0;
        for (int index = 0; index < constInfo.bSize; index++) {
            int64_t actualLen = this->actualSeqLengthsQueryGm.GetValue(index);
            if (bS1Index < actualLen) {
                bIdx = index;
                break;
            }
            actualSum = actualLen;
        }
        s1Idx = bS1Index - actualSum;
    } else {
        bIdx = bS1Index / constInfo.commonConstInfo.s1Size;
        s1Idx = bS1Index - bIdx * constInfo.commonConstInfo.s1Size;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::GetRunInfo(int64_t bIdx, int64_t s1Idx,
                                                              FagRunInfo &runInfo, FagConstInfo &constInfo,
                                                              int64_t accumS2Len, int32_t actualSeqLensQ,
                                                              int32_t actualSeqLensK)
{
    int64_t accumS2Idx;
    if constexpr (IS_TND) {
        runInfo.commonRunInfo.actualS1Size = actualSeqLensQ;
        runInfo.commonRunInfo.actualS2Size = actualSeqLensK;
        accumS2Idx = accumS2Len;
    } else {
        accumS2Idx = bIdx * constInfo.commonConstInfo.s2Size;
    }
    runInfo.keyOffsetWithRope = accumS2Idx * constInfo.n2Size * constInfo.dTotalSize;
    runInfo.commonRunInfo.valueOffset = accumS2Idx * constInfo.n2Size * constInfo.commonConstInfo.dSizeV;
    if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
        runInfo.actualSelectedBlockCount = Min(Max(runInfo.commonRunInfo.actualS2Size -
                                                   runInfo.commonRunInfo.actualS1Size + s1Idx + 1, 0),
                                               constInfo.selectedBlockCount);
    } else {
        runInfo.actualSelectedBlockCount = Min(constInfo.selectedBlockCount, runInfo.commonRunInfo.actualS2Size);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t FAGBlockVec<TEMPLATE_ARGS>::GetActualSeqLens(int64_t bIdx,
                                                                       GlobalTensor<int32_t> &actualSeqLensGm,
                                                                       int64_t &accumLen, FagConstInfo constInfo)
{
    if constexpr (IS_TND) {
        if (bIdx == 0) {
            accumLen = 0;
            return actualSeqLensGm.GetValue(0);
        } else {
            accumLen = actualSeqLensGm.GetValue(bIdx - 1);
            return (actualSeqLensGm.GetValue(bIdx) - accumLen);
        }
    } else {
        return 0;
    }
}

TEMPLATES_DEF
class FAGBlockVecDummy {
public:
    __aicore__ inline void InitUbBuffer(){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR dy, GM_ADDR y, GM_ADDR sparseIndices,
                                            GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR keyRope,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR actualSeqQlen,
                                            GM_ADDR actualSeqKvlen, GM_ADDR workspace, GM_ADDR sinks, GM_ADDR dSinks){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo, AttenMaskInfo &attenMaskInfo,
                                                                     PseInfo &pseInfo){};
    __aicore__ inline void GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm, const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                      LocalTensor<CALC_TYPE> &dkInTensor, LocalTensor<CALC_TYPE> &dvInTensor,
                                      FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                           FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t taskId){};
    __aicore__ inline void FinalizeDSinkAcc(FagConstInfo &constInfo){};
    __aicore__ inline void ReduceDSink(LocalTensor<CALC_TYPE> &scratchTensor, LocalTensor<CALC_TYPE> &reduceOutTensor,
                                       FagConstInfo &constInfo){};
    __aicore__ inline void InitCubeVecSharedParams(FagCVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, float qScaleDs){};
};
 
} // namespace SfagBaseApi
 
#endif