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
#include "vector_api/vf_cal_dsink.h"
#include "vector_api/vf_cal_softmax_sink.h"
namespace SfagBaseApi {
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t SYNC_V0_V1_DS_A_MAX_DONE_FLAG = 10;
constexpr uint32_t BIT_MASK_NUM = 8;
constexpr uint16_t SRCM = 128;

TEMPLATES_DEF
class FAGBlockVec {
private:
    __aicore__ inline bool GetBS1Index(int64_t usedT1Index, int64_t &bIdx, int64_t &s1Idx, int64_t &bS1Index,
                                        int64_t &accumS1Len, int64_t &accumS2Len, int64_t &accumS3Len, 
                                        int64_t &actualSeqLensQ, int64_t &actualSeqOriLensK, int64_t &actualSeqCmpLensK,
                                        int64_t &actualOriSelectedBlockCount, int64_t &actualCmpSelectedBlockCount, int64_t &winStart, int64_t taskId, FagConstInfo &constInfo);
    __aicore__ inline void GetRunInfo(int64_t bIdx, int64_t s1Idx, FagRunInfo &runInfo, FagConstInfo &constInfo,
                                        int64_t accumS2Len, int32_t actualSeqLensQ, int32_t actualSeqLensOriK, int32_t actualSeqLensCmpK);
    __aicore__ inline int32_t GetActualSeqLens(int64_t bIdx, GlobalTensor<int32_t> &actualSeqLensGm,
                                                int64_t &accumLen, FagConstInfo constInfo);
    __aicore__ inline void ComputeSoftmaxL1(FagConstInfo &constInfo, FagRunInfo &runInfo);
public:
    __aicore__ inline FAGBlockVec(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo, AttenMaskInfo &attenMaskInfo,
                                                                     PseInfo &pseInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out, GM_ADDR out, GM_ADDR ori_sparse_indices, GM_ADDR cmp_sparse_indices,
                                            GM_ADDR softmax_lse, GM_ADDR sink, GM_ADDR dsink, GM_ADDR ori_softmax_l1, GM_ADDR cmp_softmax_l1, GM_ADDR dq, 
                                            GM_ADDR dori_kv, GM_ADDR dcmp_kv, GM_ADDR actualSeqQlen, GM_ADDR actualSeqOriKvlen, GM_ADDR actualSeqCmpKvlen,
                                            GM_ADDR usedSeqQlenAddr, GM_ADDR usedOriSeqKvlenAddr, GM_ADDR usedCmpSeqKvlenAddr, GM_ADDR cmpResidualKAddr,
                                            GM_ADDR oriTopkLengthAddr, GM_ADDR cmpTopkLengthAddr, GM_ADDR workspace, FagConstInfo &constInfo);
    __aicore__ inline void InitUbBuffer(FagConstInfo &constInfo);
    __aicore__ inline void InitCubeVecSharedParams(FagRunInfo &runInfo, int32_t s2RealSize);
    __aicore__ inline int32_t GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <const bool IS_SPARSE = true>
    __aicore__ inline void ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm, const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm, LocalTensor<CALC_TYPE> &dkInTensor, LocalTensor<CALC_TYPE> &dvInTensor,
                                      FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dOriKWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dCmpKWorkSpaceGm,
                                      FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void CopyLse(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void CopySink(FagConstInfo &constInfo);
    __aicore__ inline void ProcessSoftmaxSink(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo);
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
    GlobalTensor<INPUT_TYPE> oriKVGm, cmpKVGm, dyGm;
    GlobalTensor<OUTDTYPE> yGm, pseGm;
    GlobalTensor<uint8_t> dropMaskGm, attenMaskU8Gm;
    GlobalTensor<float> pseFloatGm, softmaxLseGm, sinkGm, dsinkGm, oriSoftmaxL1Gm, cmpSoftmaxL1Gm;
    GM_ADDR pseSlope;
    GlobalTensor<float> softmaxL1WorkSpaceGm, dsinkWorkSpaceGm;

    // GlobalTensor<int32_t> topkIndicesGm; // 待删除
    GlobalTensor<int32_t> oritopkIndicesGm;
    GlobalTensor<int32_t> cmptopkIndicesGm;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm;
    GlobalTensor<int32_t> actualSeqLengthsOriKeyGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKeyGm;
 
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqOriKvlenAddr;
    GM_ADDR actualSeqCmpKvlenAddr;
    GM_ADDR usedSeqQlenAddr;
    GM_ADDR usedOriSeqKvlenAddr;
    GM_ADDR usedCmpSeqKvlenAddr;
    GM_ADDR cmpResidualKAddr;
    GM_ADDR oriTopkLengthAddr;
    GM_ADDR cmpTopkLengthAddr;

    // ub buffer
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECIN, 1> sotmaxL1Que[2];
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECOUT, 2> softmaxL1OutQue;
    TQue<QuePosition::VECIN, 1> LseQue[2];
    TQue<QuePosition::VECIN, 1> sinkQue;
    TQue<QuePosition::VECOUT, 1> dSinkQue;
    TBuf<> softmaxSinkResBuf[2];
    TBuf<> softmaxL1SumBuf[2];
    TBuf<> softmaxGradResBuf;
    TBuf<> dropMaskBuf;
    TBuf<> dropmaskIndexVecBuf;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterInOutQue;
    TBuf<> deterOffsetBuf;
    TBuf<> vselrIndexesBuf;
    TQue<QuePosition::VECOUT, 1> dsAmaxOutQue;
 
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
    DataCopyExtParams outParamK;
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

    outParamK.blockCount = 2;
    outParamK.blockLen = constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
    outParamK.srcStride = 0;
    outParamK.dstStride = 0;
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out, GM_ADDR out, GM_ADDR ori_sparse_indices, GM_ADDR cmp_sparse_indices,
                                                                    GM_ADDR softmax_lse, GM_ADDR sink, GM_ADDR dsink, GM_ADDR ori_softmax_l1, GM_ADDR cmp_softmax_l1, GM_ADDR dq, 
                                                                    GM_ADDR dori_kv, GM_ADDR dcmp_kv, GM_ADDR actualSeqQlen, GM_ADDR actualSeqOriKvlen, GM_ADDR actualSeqCmpKvlen,
                                                                    GM_ADDR usedSeqQlenAddr, GM_ADDR usedOriSeqKvlenAddr, GM_ADDR usedCmpSeqKvlenAddr, GM_ADDR cmpResidualKAddr,
                                                                    GM_ADDR oriTopkLengthAddr, GM_ADDR cmpTopkLengthAddr, GM_ADDR workspace, FagConstInfo &constInfo)
{
    oriKVGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)ori_kv);
    cmpKVGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)cmp_kv);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)d_out);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)out);
    oriSoftmaxL1Gm.SetGlobalBuffer((__gm__ float *)ori_softmax_l1);
    cmpSoftmaxL1Gm.SetGlobalBuffer((__gm__ float *)cmp_softmax_l1);
    actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlen);
    actualSeqLengthsOriKeyGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqOriKvlen);
    actualSeqLengthsCmpKeyGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqCmpKvlen);

    this->actualSeqQlenAddr = actualSeqQlen;
    this->actualSeqOriKvlenAddr = actualSeqOriKvlen;
    this->actualSeqCmpKvlenAddr = actualSeqCmpKvlen;
    this->usedSeqQlenAddr = usedSeqQlenAddr;
    this->usedOriSeqKvlenAddr = usedOriSeqKvlenAddr;
    this->usedCmpSeqKvlenAddr = usedCmpSeqKvlenAddr;
    this->cmpResidualKAddr = cmpResidualKAddr;
    this->oriTopkLengthAddr = oriTopkLengthAddr;
    this->cmpTopkLengthAddr = cmpTopkLengthAddr;

    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmax_lse);
    oritopkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)ori_sparse_indices);
    cmptopkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmp_sparse_indices);
    if (IsSink) { //todo is_sink
        sinkGm.SetGlobalBuffer((__gm__ float *)sink);
        dsinkGm.SetGlobalBuffer((__gm__ float *)dsink);
        dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->baseParams.dSinkWorkSpaceOffset / sizeof(float)); //todo offset
    }
    int64_t firstHalfGSize = (constInfo.commonConstInfo.gSize + 1) >> 1;
    int64_t selectedBlockCount = Max(constInfo.selectedBlockCount, VECTOR_BASEN);
    int64_t softmaxL1WSOffset = tilingData->baseParams.softmaxL1WorkspaceOffset / sizeof(float) + this->cBlockIdx * selectedBlockCount * constInfo.commonConstInfo.gSize + this->vSubBlockIdx * firstHalfGSize * selectedBlockCount;
    softmaxL1WorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + softmaxL1WSOffset);
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitUbBuffer(FagConstInfo &constInfo)
{
    /**
     * UB划分，buffer大小分配
     * attenMaskOrYInQue: for y and attenMask
     * pseOrDyInQue: for dy and pse
     * dSOutQue: for dq dk left ub matrix
     * pOutQue: for dv left ub matrix
     * softmaxGradResBuf: for softmax_grad result
     * dropMaskBuf: for dropMask
     * LseQue: forlse double buffer
     **/
    pipe->InitBuffer(attenMaskOrYInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(pseOrDyInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    sotmaxL1Que[0] = attenMaskOrYInQue;
    sotmaxL1Que[1] = pseOrDyInQue;

    pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
    pipe->InitBuffer(softmaxSinkResBuf[0], 64 * sizeof(float));
    pipe->InitBuffer(softmaxSinkResBuf[1], 64 * sizeof(float));
    pipe->InitBuffer(LseQue[0], 1, 64 * sizeof(float));
    pipe->InitBuffer(LseQue[1], 1, 64 * sizeof(float));
    pipe->InitBuffer(sinkQue, 1, 64 * sizeof(float));
    pipe->InitBuffer(dSinkQue, 1, 64 * sizeof(float));

    pipe->InitBuffer(dSOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE + DETER_EXCEED_USE_SIZE);
    pipe->InitBuffer(pOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE);
    pipe->InitBuffer(softmaxL1OutQue, 2, 128 * sizeof(float));
    pipe->InitBuffer(softmaxL1SumBuf[0], 128 * sizeof(float));
    pipe->InitBuffer(softmaxL1SumBuf[1], 128 * sizeof(float));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                                const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                                const GlobalTensor<CALC_TYPE> &dOriKWorkSpaceGm,
                                                                const GlobalTensor<CALC_TYPE> &dCmpKWorkSpaceGm,
                                                                FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    int64_t UB_ROW_SIZE = 8;
    int64_t usedCoreNum = tilingData->baseParams.usedCoreNum;
    int64_t coreNum = GetBlockNum();
    int64_t totalVec = coreNum * 2;
    int64_t usedBS1Index = -1;
    int64_t bS1Index = -1;
    int64_t bIdx;
    int64_t s1Idx;
    int64_t preBIdx = -1;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int64_t accumS3Len = 0;
    int64_t actualSeqLensQ = 0;
    int64_t actualSeqLensOriK = 0;
    int64_t actualSeqLensCmpK = 0;
    int64_t actualOriSelectedBlockCount = 0;
    int64_t actualCmpSelectedBlockCount = 0;
    int64_t winStart = 0;
    GlobalTensor<CALC_TYPE> dkWorkSpaceGm;
    GlobalTensor<int32_t> topkIndicesGm;
    int64_t currentSelectedBlockCount;
    if (runInfo.isOriKV) {
        dkWorkSpaceGm = dOriKWorkSpaceGm;
        topkIndicesGm = oritopkIndicesGm;
        currentSelectedBlockCount = constInfo.oriSelectedBlockCount;
    } else {
        dkWorkSpaceGm = dCmpKWorkSpaceGm;
        topkIndicesGm = cmptopkIndicesGm;
        currentSelectedBlockCount = constInfo.cmpSelectedBlockCount;
    }
    LocalTensor<CALC_TYPE> dkInTensor = dSOutQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> dvInTensor = pOutQue.AllocTensor<CALC_TYPE>();
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    // printf("ScatterAddDeter>>>> coreid: %u\n", this->vBlockIdx);

    for (int64_t idx = 0; idx < usedCoreNum; idx++) {
        usedBS1Index = runInfo.sTaskId * usedCoreNum + idx;
        if (!GetBS1Index(usedBS1Index, bIdx, s1Idx, bS1Index, accumS1Len, accumS2Len, accumS3Len, actualSeqLensQ,
            actualSeqLensOriK, actualSeqLensCmpK, actualOriSelectedBlockCount, actualCmpSelectedBlockCount, winStart, runInfo.sTaskId, constInfo)) 
        {
            continue;
        }        
        // GetRunInfo(bIdx, s1Idx, runInfo, constInfo, accumS1Len, accumS2Len, accumS3Len, actualSeqLensQ, actualSeqLensOriK, actualSeqLensCmpK);

        // 对应s2real的平均和取模值，做均分操作
        int64_t actualSelectedBlockCount = runInfo.isOriKV ? actualOriSelectedBlockCount : actualCmpSelectedBlockCount;
        int64_t remainder = actualSelectedBlockCount % totalVec;
        int64_t avgSize = actualSelectedBlockCount / totalVec;
        // 前remainder 个块分配 avgSize + 1， 其余分配avgSize
        int64_t currentCoreKSize = avgSize + (this->vBlockIdx < remainder ? 1 : 0);
        // printf("this->vBlockIdx : %u actualSelectedBlockCount : %d currentCoreKSize : %d, remainder: %d, avgSize: %d\n",this->vBlockIdx, actualSelectedBlockCount, currentCoreKSize, remainder, avgSize);
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
        // printf("runInfo.deterTaskIdMod2: %d, coreNum: %d, idx: %d\n", runInfo.deterTaskIdMod2, coreNum, idx);
        int64_t currentMm4SrcOffset = runInfo.deterTaskIdMod2 * 4096 * HEAD_DIM_ALIGN * coreNum +
                                    idx * 4096 * HEAD_DIM_ALIGN +
                                    s2SrcOffset * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
        int64_t currentMm5SrcOffset = runInfo.deterTaskIdMod2 * 4096 * constInfo.commonConstInfo.dSizeV * coreNum +
                                    idx * 4096 * constInfo.commonConstInfo.dSizeV +
                                    s2SrcOffset * constInfo.selectedBlockSize * constInfo.commonConstInfo.dSizeV;
        runInfo.keyOffsetWithRope = accumS2Len * constInfo.n2Size * constInfo.dTotalSize;
        if (!runInfo.isOriKV) {
            runInfo.keyOffsetWithRope = accumS3Len * constInfo.n2Size * constInfo.dTotalSize;
        }
        // if (!runInfo.isSparse){
            // todo currentMm4SrcOffset currentMm5SrcOffset 待适配非sparse场景
        // }

        int64_t kvOffset = runInfo.isSparse ? runInfo.keyOffsetWithRope : runInfo.keyOffsetWithRope + s2SrcOffset * constInfo.dTotalSize;
        GlobalTensor<float> dkOutGm = dkWorkSpaceGm[kvOffset];
        // printf("this->vBlockIdx: %u, runInfo.isSparse: %d, kvOffset: %d, runInfo.keyOffsetWithRope: %d, s2SrcOffset: %d, currentMm4SrcOffset: %d, currentMm5SrcOffset: %d, maxLoops: %d\n", this->vBlockIdx, runInfo.isSparse, kvOffset, runInfo.keyOffsetWithRope, s2SrcOffset, currentMm4SrcOffset, currentMm5SrcOffset, maxLoops);

        // 1 - main loop
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        uint64_t gmOffset = bS1Index * (constInfo.n2Size * currentSelectedBlockCount) + s2SrcOffset;
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
            if (runInfo.isSparse) {
                for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                    int s2Idx = topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
                    if (s2Idx >= 0) {
                        Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                            dvInTensor[row * constInfo.commonConstInfo.dSizeV],
                            constInfo.commonConstInfo.dSizeV);
                        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                        DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN],
                                dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                    }
                }
            } else {
                Add(dkInTensor, dkInTensor, dvInTensor, UB_ROW_SIZE * HEAD_DIM_ALIGN);
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopy(dkOutGm[loop * UB_ROW_SIZE * HEAD_DIM_ALIGN], dkInTensor, UB_ROW_SIZE * HEAD_DIM_ALIGN);
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
        if (runInfo.isSparse) {
            for (int64_t row = 0; row < tailRows; row++) {
                int32_t s2Idx = topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
                if (s2Idx >= 0) {
                    Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                        dvInTensor[row * constInfo.commonConstInfo.dSizeV],
                        constInfo.commonConstInfo.dSizeV);
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN],
                            dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                }
            }
        } else {
            Add(dkInTensor, dkInTensor, dvInTensor, tailRows * HEAD_DIM_ALIGN);
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopy(dkOutGm[(maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN], dkInTensor, tailRows * HEAD_DIM_ALIGN);
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
__aicore__ inline bool FAGBlockVec<TEMPLATE_ARGS>::GetBS1Index(int64_t usedT1Index, int64_t &bIdx, int64_t &s1Idx, int64_t &bS1Index,
                                                         int64_t &accumS1Len, int64_t &accumS2Len, int64_t &accumS3Len, 
                                                         int64_t &actualSeqLensQ, int64_t &actualSeqOriLensK, int64_t &actualSeqCmpLensK,
                                                         int64_t &actualOriSelectedBlockCount, int64_t &actualCmpSelectedBlockCount, 
                                                         int64_t &winStart, int64_t taskId, FagConstInfo &constInfo)
{
    int64_t t1Offset = 0;
    int64_t t2Offset = 0;
    int64_t t3Offset = 0;
    int64_t bIndex = HasUsedSeqQ ? -1 : 0;
    int64_t s1Index = 0;
    int64_t nextUsedT1Offset = 0;
    int64_t usedT1Offset = 0;
    int64_t curT1 = 0;
    int64_t curT2 = 0;
    int64_t curS1 = constInfo.commonConstInfo.s1Size;
    int64_t curS2 = constInfo.commonConstInfo.s2Size;
    int64_t curS3 = constInfo.s3Size;
    int64_t usedS1 = 0;
    int64_t usedS2 = 0;
    int64_t usedS3 = 0;
    int64_t oriWinStart = 0;
    int64_t oriWinEnd = 0;
    int64_t oriTopkLength = 0;
    int64_t cmpTopkLength = 0;

    if (usedT1Index >= constInfo.totalNum){
        return false;
    }
    if constexpr (!HasUsedSeqQ) {
        if constexpr (IS_TND) {
            curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            while (usedT1Index >= curT1) {
                curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[++bIndex];
            }
            bIndex = bIndex - 1;
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            s1Index = usedT1Index - t1Offset;
        } else {
            bIndex = usedT1Index / constInfo.commonConstInfo.s1Size;
            s1Index = usedT1Index % constInfo.commonConstInfo.s1Size;
            t1Offset = usedT1Index - s1Index;
        }
        bS1Index = usedT1Index;
    } else {
        while (usedT1Index >= nextUsedT1Offset) {
            usedT1Offset = nextUsedT1Offset;
            nextUsedT1Offset = ((__gm__ int32_t *)usedSeqQlenAddr)[++bIndex] + usedT1Offset;
        }
        s1Index = usedT1Index - usedT1Offset;
        if constexpr (IS_TND) {
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
        } else {
            t1Offset = bIndex * constInfo.commonConstInfo.s1Size;
        }
        bS1Index = t1Offset + s1Index;
    }
    if constexpr (IS_TND) {
        if (unlikely(bIndex == 0)) {
            t2Offset = 0;
            t3Offset = 0;
            curS1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex + 1];
            if constexpr(IsOriKVExist) {
                curS2 = ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex + 1];
            }
            if constexpr(IsCmpKVExist) {
                curS3 = ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex + 1];
            }
        } else {
            curS1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex + 1] - ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            if constexpr(IsOriKVExist) {
                curS2 = ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex + 1] - ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex];
                t2Offset = ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex];
            }
            if constexpr(IsCmpKVExist) {
                curS3 = ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex + 1] - ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex];
                t3Offset = ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex];
            }            
        }
    } else {
        t2Offset = bIndex * curS2;
        t3Offset = bIndex * curS3;
    }

    if constexpr(HasUsedSeqQ) {
        usedS1 = ((__gm__ int32_t *)usedSeqQlenAddr)[bIndex];
    } else {
        usedS1 = curS1;
    }
    if constexpr(HasUsedSeqOriKV) {
        usedS2 = ((__gm__ int32_t *)usedOriSeqKvlenAddr)[bIndex];
    } else {
        usedS2 = curS2;
    }
    if constexpr(HasUsedSeqCmpKV) {
        usedS3 = ((__gm__ int32_t *)usedCmpSeqKvlenAddr)[bIndex];
    } else {
        usedS3 = curS3;
    }

    if (tilingData->baseParams.oriSparseMode == 4) {
        oriWinStart = Max(s1Index + usedS2 - usedS1 - tilingData->baseParams.oriWinLeft, 0);
        oriWinEnd = Min(s1Index + usedS2 - usedS1 + tilingData->baseParams.oriWinRight + 1, usedS2);
        usedS2 = Max(oriWinEnd - oriWinStart, 0);
    } else if (tilingData->baseParams.oriSparseMode == 3) {
        usedS2 = Max(usedS2 - usedS1 + s1Index + 1, 0);
    }

    if (tilingData->baseParams.cmpSparseMode == 3 && constInfo.cmpRatio == 1) {
        usedS3 = Max(usedS3 - usedS1 + s1Index + 1, 0);
    } else if (tilingData->baseParams.cmpSparseMode == 3 && constInfo.cmpRatio != 1) {
        int64_t oriUsedS3 = usedS3 * constInfo.cmpRatio + cmpResidualKAddr[bIndex];
        usedS3 = Max((oriUsedS3 - usedS1 + s1Index + 1) / constInfo.cmpRatio, 0);
    }

    if constexpr(HasOriTopK) {
        oriTopkLength = ((__gm__ int32_t *)oriTopkLengthAddr)[bS1Index];
    } else if constexpr(IsOriKVSparse) {
        oriTopkLength = constInfo.oriSelectedBlockCount;
    } else {
        oriTopkLength = curS2;
    }

    if constexpr(HasCmpTopK) {
        cmpTopkLength = ((__gm__ int32_t *)cmpTopkLengthAddr)[bS1Index];
    } else if constexpr(IsCmpKVSparse) {
        cmpTopkLength = constInfo.cmpSelectedBlockCount;
    } else {
        cmpTopkLength = curS3;
    }

    actualOriSelectedBlockCount = Min(Min(oriTopkLength, curS2), usedS2);
    actualCmpSelectedBlockCount = Min(Min(cmpTopkLength, curS3), usedS3);

    bIdx = bIndex;
    s1Idx = s1Index;
    actualSeqLensQ = usedS1;
    actualSeqOriLensK = usedS2;
    actualSeqCmpLensK = usedS3;
    accumS1Len = t1Offset;
    accumS2Len = t2Offset;
    accumS3Len = t3Offset;
    winStart = oriWinStart;
    return true;
}
 	 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::GetRunInfo(int64_t bIdx, int64_t s1Idx,
                                                            FagRunInfo &runInfo, FagConstInfo &constInfo,
                                                            int64_t accumS2Len, int32_t actualSeqLensQ,
                                                            int32_t actualSeqLensOriK, int32_t actualSeqLensCmpK)
{
    // int64_t accumS2Idx;
    // // runInfo.bIdx = bIdx;
    // runInfo.s1Idx = s1Idx;
    // runInfo.actS1Size = actualSeqLensQ;
    // runInfo.actS2Size = actualSeqLensOriK;
    // runInfo.actS3Size = actualSeqLensCmpK;
    // runInfo.accumS1Idx = accumS1Len + s1Idx;
    // runInfo.accumS2Idx = accumS2Len;
    // runInfo.accumS3Idx = accumS3Len;
    // if constexpr (IS_TND) {
    //     runInfo.commonRunInfo.actualS1Size = actualSeqLensQ;
    //     runInfo.commonRunInfo.actualS2Size = actualSeqLensOriK + actualSeqLensCmpK;
    //     accumS2Idx = accumS2Len;
    // } else {
    //     accumS2Idx = bIdx * constInfo.commonConstInfo.s2Size;
    // }
    // runInfo.keyOffsetWithRope = accumS2Idx * constInfo.n2Size * constInfo.dTotalSize;
    // if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
    //     runInfo.actualSelectedBlockCount = Min(Max(runInfo.commonRunInfo.actualS2Size -
    //                                             runInfo.commonRunInfo.actualS1Size + s1Idx + 1, 0),
    //                                         constInfo.selectedBlockCount);
    // } else {
    //     runInfo.actualSelectedBlockCount = 0;
    // }
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

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t FAGBlockVec<TEMPLATE_ARGS>::GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (!runInfo.isSparse) {
        return 0;
    }

    int32_t s2RealSize = 0;
    bool multiCopy = true;
    outParamK.blockCount = 2;
    uint64_t indicesGmOffset = runInfo.sparseIndicesOffset;
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

    // LocalTensor<INPUT_TYPE> gatherRopeTensorPing = gatherTensorPing[2 * constInfo.commonConstInfo.dSize];
    // LocalTensor<INPUT_TYPE> gatherRopeTensorPong = gatherTensorPong[2 * constInfo.commonConstInfo.dSize];

    GlobalTensor<int32_t> curTopkIndicesGm = runInfo.isOriKV ? oritopkIndicesGm : cmptopkIndicesGm;
    GlobalTensor<INPUT_TYPE> keyGm = runInfo.isOriKV ? oriKVGm : cmpKVGm;
    for (i = curBlk; i < curBlk + curActualSelCntOffset / 2 * 2; i += 2) {
        int32_t indices1 = curTopkIndicesGm.GetValue(indicesGmOffset + i);
        int32_t indices2 = curTopkIndicesGm.GetValue(indicesGmOffset + i + 1);

        if (unlikely(indices1 < 0)) {
            multiCopy = false;
            break;
        } else if (unlikely(indices2 < 0)) {
            s2RealSize += 1;
            multiCopy = false;
        } else {
            s2RealSize += 2;
        }

        int64_t keyOffset1 = indices1 * constInfo.selectedBlockSize;
        int64_t keyOffset2 = indices2 * constInfo.selectedBlockSize;

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - constInfo.selectedBlockSize;
        intriParamsKey.blockCount = 2;

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Ping : mte2WaitMte3Pong;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Ping : mte3WaitMte2Pong;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        // CopyIn
        intriParamsKey.srcStride = s2OrgStride * constInfo.n2Size * constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
        LocalTensor<INPUT_TYPE> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;

        if (keyOffset2 <= keyOffset1) {
            intriParamsKey.blockCount = 1;
            DataCopyPad(gatherTensor, keyGm[runInfo.commonRunInfo.keyOffset + keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize], intriParamsKey, padParams);
            if (multiCopy) {
                DataCopyPad(gatherTensor[constInfo.selectedBlockSize * constInfo.commonConstInfo.dSize], keyGm[runInfo.commonRunInfo.keyOffset + keyOffset2 * constInfo.n2Size * constInfo.commonConstInfo.dSize], intriParamsKey, padParams);
            }
        } else {
            DataCopyPad(gatherTensor, keyGm[runInfo.commonRunInfo.keyOffset + keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize], intriParamsKey, padParams);
        }

        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        // CopyOut
        DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        outWsOffset += 2 * constInfo.dTotalSize;
        mergePingPong = 1 - mergePingPong;
    }
    if (i < curActualSelCntEnd && multiCopy) {
        int64_t indices1 = curTopkIndicesGm.GetValue(indicesGmOffset + i);
        int64_t keyOffset1 = indices1 * constInfo.selectedBlockSize;
        if (indices1 >= 0) {
            s2RealSize += 1;
        }

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Ping : mte2WaitMte3Pong;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Ping : mte3WaitMte2Pong;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        // CopyIn
        intriParamsKey.blockCount = 1;
        LocalTensor<INPUT_TYPE> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        if (indices1 >= 0) {
            DataCopyPad(gatherTensor, keyGm[runInfo.commonRunInfo.keyOffset + keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize], intriParamsKey, padParams);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
            outParamK.blockCount = 1;
            // CopyOut
            DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        mergePingPong = 1 - mergePingPong;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Ping);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    dSOutQue.FreeTensor(gatherTensorPing);
    pOutQue.FreeTensor(gatherTensorPong);

    return s2RealSize;
    //     pipe_barrier(PIPE_ALL);
    // uint32_t array[] = {static_cast<uint32_t>(2), static_cast<uint32_t>(512)};
    // AscendC::ShapeInfo shapeInfo(2, array);
    // AscendC::DumpTensor(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr], 110, 2 * 512, shapeInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: Cast + SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    if (runInfo.halfGRealSize == 0 || runInfo.commonRunInfo.s2RealSize == 0) {
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
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyLse(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    CopyInLse<float, VECTOR_BASEM>(constInfo, runInfo, LseQue[runInfo.sTaskIdMod2], softmaxLseGm);
    LocalTensor<float> lseTensor = LseQue[runInfo.sTaskIdMod2].AllocTensor<float>();
    LseQue[runInfo.sTaskIdMod2].FreeTensor(lseTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopySink(FagConstInfo &constInfo)
{
    if (constInfo.halfGRealSize == 0) {
        return;
    }    
    CopyInSink<float, VECTOR_BASEM>(constInfo, sinkQue, sinkGm);
}
 
TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_DQ>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1(FagConstInfo &constInfo, FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                                             LocalTensor<INPUT_TYPE> &srcTensor)
{
    if (runInfo.halfGRealSize == 0 || runInfo.commonRunInfo.s2RealSize == 0) {
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
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    // LocalTensor<CALC_TYPE> softmaxL1ResTensor = softmaxL1OutQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> softmaxL1SumTensor = softmaxL1SumBuf[0].Get<CALC_TYPE>();
    float pScalar = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.commonConstInfo.gSize));
    if (runInfo.isSparse) {
        CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, false, false, false, VECTOR_BASEM, VECTOR_BASEN, true>(
            constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, LseQue[runInfo.sTaskIdMod2],
            attenMaskOrYInQue, pseOrDyInQue, mm2ResTensor, mm2ResTensor, softmaxL1SumTensor, pseSlope, pScalar, runInfo.needCleanSoftmaxL1);
        AscendC::TEventID mte3WaitV = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(mte3WaitV);
            // TODO copy p out
        DataCopyExtParams dataCopyGmParams;
        dataCopyGmParams.blockCount = 1;
        dataCopyGmParams.blockLen = runInfo.halfGRealSize * sizeof(CALC_TYPE) * VECTOR_BASEN; // 0.5 * G * 128
        dataCopyGmParams.srcStride = 0;
        dataCopyGmParams.dstStride = 0;
        DataCopyPad(softmaxL1WorkSpaceGm[runInfo.halfGRealSize * runInfo.blkCntOffset], mm2ResTensor, dataCopyGmParams);
        
        // printf("softmaxl1 offset1: %d, runInfo.needCleanSoftmaxL1: %d\n", runInfo.halfGRealSize * runInfo.blkCntOffset, runInfo.needCleanSoftmaxL1);
        // pipe_barrier(PIPE_ALL);
        // uint32_t array[] = {static_cast<uint32_t>(32), static_cast<uint32_t>(128)};
        // AscendC::ShapeInfo shapeInfo(2, array);
        // AscendC::DumpTensor(softmaxL1WorkSpaceGm[runInfo.halfGRealSize * runInfo.blkCntOffset], 310, 32 * 128, shapeInfo);
        // AscendC::DumpTensor(softmaxL1SumTensor, 313, 1 * 32);
    } else {
        CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, false, false, false, VECTOR_BASEM, VECTOR_BASEN, false>(
            constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, LseQue[runInfo.sTaskIdMod2],
            attenMaskOrYInQue, pseOrDyInQue, mm2ResTensor, mm2ResTensor, softmaxL1SumTensor, pseSlope, pScalar, false);
    }

    if (runInfo.isSparse && runInfo.needComputeSoftmaxL1) {
        ComputeSoftmaxL1(constInfo, runInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ComputeSoftmaxL1(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

    DataCopyPadExtParams<CALC_TYPE> softmaxPadParams;
    DataCopyExtParams dataCopyGmParams;
    dataCopyGmParams.blockCount = 1;
    dataCopyGmParams.srcStride = 0;
    dataCopyGmParams.dstStride = VECTOR_BASEN * sizeof(CALC_TYPE) / 32;
    DataCopyExtParams dataCopyGmOutParams;
    dataCopyGmOutParams.blockCount = 1;

    int64_t actualSelectedBlockCount = runInfo.isOriKV ? runInfo.actualSelectedBlockCount : runInfo.actualCmpSelectedBlockCount;
    // printf("actualSelectedBlockCount: %d, runInfo.isOriKV: %d, runInfo.actualSelectedBlockCount: %d, runInfo.actualCmpSelectedBlockCount: %d\n", actualSelectedBlockCount, runInfo.isOriKV, runInfo.actualSelectedBlockCount, runInfo.actualCmpSelectedBlockCount);
    int64_t loopNum = CeilDiv(actualSelectedBlockCount, VECTOR_BASEN);
    int64_t loopSize = VECTOR_BASEN;
    int64_t loopTailSize = actualSelectedBlockCount % VECTOR_BASEN == 0 ? VECTOR_BASEN : actualSelectedBlockCount % VECTOR_BASEN;
    LocalTensor<CALC_TYPE> softmaxL1SumTensor = softmaxL1SumBuf[0].Get<CALC_TYPE>();
    float pScalar = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.commonConstInfo.gSize));
    GlobalTensor<float> outputGm = runInfo.isOriKV ? oriSoftmaxL1Gm : cmpSoftmaxL1Gm;
    
    SetAtomicAdd<CALC_TYPE>();
    // printf("loopNum: %d, runInfo.isOriKV: %d\n", loopNum, runInfo.isOriKV);
    int64_t outOffset = runInfo.t1Index * runInfo.selectedBlockCount;
    for (int64_t i = 0; i < loopNum; i++) {
        if (i == loopNum - 1) {
            loopSize = loopTailSize;
        }
        LocalTensor<CALC_TYPE> softmaxL1SrcTensor = sotmaxL1Que[i % 2].AllocTensor<CALC_TYPE>();
        LocalTensor<CALC_TYPE> softmaxL1DstTensor = softmaxL1OutQue.AllocTensor<CALC_TYPE>();

        dataCopyGmParams.blockLen = runInfo.halfGRealSize * VECTOR_BASEN * sizeof(CALC_TYPE); // 0.5 * G * 128
        DataCopyPad(softmaxL1SrcTensor, softmaxL1WorkSpaceGm[i * runInfo.halfGRealSize * VECTOR_BASEN], dataCopyGmParams, softmaxPadParams);
        sotmaxL1Que[i % 2].EnQue(softmaxL1SrcTensor);
        sotmaxL1Que[i % 2].DeQue<CALC_TYPE>();
        // pipe_barrier(PIPE_ALL);
        // printf("softmaxl1 offset2: %d\n", i * runInfo.halfGRealSize * VECTOR_BASEN);
        // uint32_t array[] = {static_cast<uint32_t>(32), static_cast<uint32_t>(128)};
        // AscendC::ShapeInfo shapeInfo(2, array);
        // AscendC::DumpTensor(softmaxL1SumTensor, 311, 1 * 32);
        // AscendC::DumpTensor(softmaxL1SrcTensor, 312, 32 * 128, shapeInfo);

        SoftmaxL1Reduce<CALC_TYPE>(softmaxL1DstTensor, softmaxL1SrcTensor, softmaxL1SumTensor, runInfo.halfGRealSize, loopSize, pScalar);
        softmaxL1OutQue.EnQue(softmaxL1DstTensor);
        softmaxL1OutQue.DeQue<CALC_TYPE>();
        // pipe_barrier(PIPE_ALL);
        // AscendC::DumpTensor(softmaxL1DstTensor, 313, 1 * 128, shapeInfo);

        dataCopyGmOutParams.blockLen = loopSize * sizeof(CALC_TYPE);
        DataCopyPad(outputGm[outOffset + i * VECTOR_BASEN], softmaxL1DstTensor, dataCopyGmOutParams);
        sotmaxL1Que[i % 2].FreeTensor(softmaxL1SrcTensor);
        softmaxL1OutQue.FreeTensor(softmaxL1DstTensor);
    }
    SetAtomicNone();
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
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
    // if ASCEND_IS_AIV {
    //     pipe_barrier(PIPE_ALL);
    //     uint32_t array[] = {static_cast<uint32_t>(32), static_cast<uint32_t>(128)};
    //     AscendC::ShapeInfo shapeInfo(2, array);
    //     // ds
    //     AscendC::DumpTensor(mm1ResTensor, 114, 32 * 128, shapeInfo);
    // }

    LocalTensor<uint8_t> selrIndexesTensor;
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer, mm1ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.DeQue<INPUT_TYPE>();
 
    // copy ds from ub to l1
    LocalTensor<INPUT_TYPE> dsL1Tensor = dstBuffer.GetTensor<INPUT_TYPE>();
    CopyUB2L1<true>(constInfo, runInfo, dsL1Tensor, vecOutBuffer);

    // if ASCEND_IS_AIV {
    //     pipe_barrier(PIPE_ALL);
    //     uint32_t array[] = {static_cast<uint32_t>(32), static_cast<uint32_t>(128)};
    //     AscendC::ShapeInfo shapeInfo(2, array);
    //     // ds trans
    //     AscendC::DumpTensor(vecOutBuffer, 115, 32 * 128, shapeInfo);
    // }
 
    dSOutQue.FreeTensor(vecOutBuffer);
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
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
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessSoftmaxSink(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF: softmaxSink
    ///////////////////////////////////////////////////////////////
    if (runInfo.halfGRealSize == 0) {
        return;
    }    
    LocalTensor<CALC_TYPE> lseTensor = LseQue[runInfo.sTaskIdMod2].AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> sinkTensor = sinkQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> softmaxSinkTensor = softmaxSinkResBuf[runInfo.sTaskIdMod2].Get<CALC_TYPE>();
    LseQue[runInfo.sTaskIdMod2].EnQue(lseTensor);
    LseQue[runInfo.sTaskIdMod2].DeQue<CALC_TYPE>();
    sinkQue.EnQue(sinkTensor);
    sinkQue.DeQue<CALC_TYPE>();
    CalculateSoftMaxSink<CALC_TYPE>(softmaxSinkTensor, lseTensor, sinkTensor, runInfo.halfGRealSize);
    LseQue[runInfo.sTaskIdMod2].FreeTensor(lseTensor);
    sinkQue.FreeTensor(sinkTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VFSink: Sink
    ///////////////////////////////////////////////////////////////
    if (runInfo.halfGRealSize == 0 || runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }        
    LocalTensor<CALC_TYPE> dsinkOutTensor = dSinkQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> softmaxSinkTensor = softmaxSinkResBuf[runInfo.sTaskIdMod2].Get<CALC_TYPE>();

    CalculateDsink<CALC_TYPE>(
        dsinkOutTensor, mm2ResTensor, mm1ResTensor, softmaxSinkTensor, runInfo.actualSelCntOffset, runInfo.halfGRealSize);

    dSinkQue.EnQue(dsinkOutTensor);
    dSinkQue.DeQue<CALC_TYPE>();
    SetAtomicAdd<CALC_TYPE>();
    uint64_t dsinkWorkSpaceOffset = cBlockIdx * constInfo.commonConstInfo.gSize + vSubBlockIdx * runInfo.firstHalfGRealSize;
    DataCopyPad(dsinkWorkSpaceGm[dsinkWorkSpaceOffset], dsinkOutTensor,
                {1, static_cast<uint32_t>(runInfo.halfGRealSize * sizeof(CALC_TYPE)), 0, 0, 0});
    SetAtomicNone();
    dSinkQue.FreeTensor(dsinkOutTensor);

}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    FagRunInfo &runInfo, int32_t s2RealSize)
{
    if (!runInfo.isSparse) {
        return;
    }
    SfagCVSharedParams sharedParams;
    sharedParams.s2RealSize = s2RealSize;
    // runInfo.commonRunInfo.s2RealSize = s2RealSize;
    /* ssbuf send message */
    if ASCEND_IS_AIV {
        int32_t ssbufAddress = vSubBlockIdx * sizeof(SfagCVSharedParams);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ int32_t*>(ssbufAddress);
        auto tempTiling = reinterpret_cast<int32_t *>(&sharedParams);
        *tempTilingSSbuf = *tempTiling;
        CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);

        int32_t ssbufAddress3 = 2 * sizeof(SfagCVSharedParams);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf3 = reinterpret_cast<__ssbuf__ int32_t*>(ssbufAddress3);
        auto tempTiling3 = reinterpret_cast<int32_t *>(&sharedParams);
        *tempTiling3 = *tempTilingSSbuf3;
        runInfo.commonRunInfo.s2RealSize = sharedParams.s2RealSize;
        runInfo.actualSelCntOffset = runInfo.commonRunInfo.s2RealSize;

        if (runInfo.commonRunInfo.s2RealSize < 128) {
            if (!runInfo.isOriKV) {
                runInfo.isNextS1IdxNoChange = false;
            } else {
                runInfo.isNextS1IdxNoChange = runInfo.actualCmpSelectedBlockCount > 0;
                
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_SPARSE>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
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

    int64_t kvOffset = IS_SPARSE ? runInfo.commonRunInfo.keyOffset : runInfo.kSelectedWsAddr + vSubBlockIdx * firstCoreKSize * HEAD_DIM_ALIGN;
    GlobalTensor<float> dkOutGm = dkWorkSpaceGm[kvOffset];
    int64_t currentMm4SrcOffset = runInfo.mm4ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
    int64_t currentMm5SrcOffset = runInfo.mm5ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
    // 1 - main loop
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    uint64_t gmOffset = runInfo.sparseIndicesOffset + vSubBlockIdx * firstCoreKSize;
    GlobalTensor<int32_t> topkIndicesGm = runInfo.isOriKV ? oritopkIndicesGm : cmptopkIndicesGm;
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        DataCopy(dkInTensor,
                 mm4ResWorkSpaceGm[currentMm4SrcOffset + loop * UB_ROW_SIZE * HEAD_DIM_ALIGN], UB_ROW_SIZE * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, UB_ROW_SIZE * HEAD_DIM_ALIGN);
        DataCopy(dvInTensor,
                 mm5ResWorkSpaceGm[currentMm5SrcOffset + loop * UB_ROW_SIZE * HEAD_DIM_ALIGN], UB_ROW_SIZE * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        if constexpr(IS_SPARSE) {
            for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                int32_t s2Idx = topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
                if (s2Idx >= 0) {
                    Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], dvInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                }
            }
        } else {
            Add(dkInTensor, dkInTensor, dvInTensor, UB_ROW_SIZE * HEAD_DIM_ALIGN);
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopy(dkOutGm[loop * UB_ROW_SIZE * HEAD_DIM_ALIGN], dkInTensor, UB_ROW_SIZE * HEAD_DIM_ALIGN);
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
                 mm5ResWorkSpaceGm[currentMm5SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN], tailRows * HEAD_DIM_ALIGN);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    if constexpr(IS_SPARSE) {
        for (int64_t row = 0; row < tailRows; row++) {
            int32_t s2Idx = topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
            if (s2Idx >= 0) {
                Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], dvInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
            }
        }
    } else {
        Add(dkInTensor, dkInTensor, dvInTensor, tailRows * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(dkOutGm[(maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN], dkInTensor, tailRows * HEAD_DIM_ALIGN);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    SetAtomicNone();
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}
 
TEMPLATES_DEF
class FAGBlockVecDummy {
public:
    __aicore__ inline void InitUbBuffer(FagConstInfo &constInfo){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out, GM_ADDR out, GM_ADDR ori_sparse_indices, GM_ADDR cmp_sparse_indices,
                                            GM_ADDR softmax_lse, GM_ADDR sink, GM_ADDR dsink, GM_ADDR ori_softmax_l1, GM_ADDR cmp_softmax_l1, GM_ADDR dq, 
                                            GM_ADDR dori_kv, GM_ADDR dcmp_kv, GM_ADDR actualSeqQlen, GM_ADDR actualSeqOriKvlen, GM_ADDR actualSeqCmpKvlen,
                                            GM_ADDR usedSeqQlenAddr, GM_ADDR usedOriSeqKvlenAddr, GM_ADDR usedCmpSeqKvlenAddr, GM_ADDR cmpResidualKAddr,
                                            GM_ADDR oriTopkLengthAddr, GM_ADDR cmpTopkLengthAddr, GM_ADDR workspace, FagConstInfo &constInfo){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo, AttenMaskInfo &attenMaskInfo,
                                                                     PseInfo &pseInfo){};
    __aicore__ inline int32_t GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo, FagRunInfo &runInfo){return 0;};
    template <const bool IS_SPARSE = true>
    __aicore__ inline void ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm, const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm, LocalTensor<CALC_TYPE> &dkInTensor, LocalTensor<CALC_TYPE> &dvInTensor,
                                      FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorOriKSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorCmpKSpaceGm,
                                      FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void CopyLse(FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void CopySink(FagConstInfo &constInfo){};
    __aicore__ inline void ProcessSoftmaxSink(FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void InitCubeVecSharedParams(FagRunInfo &runInfo, int32_t s2RealSize){};
};
 
} // namespace SfagBaseApi
 
#endif