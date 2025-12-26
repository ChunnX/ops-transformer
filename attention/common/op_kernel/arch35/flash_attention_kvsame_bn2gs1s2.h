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
 * \file flash_attention_kvsame_bn2gs1s2.h
 * \brief
 */

#ifndef FLASH_ATTENTION_KVSAME_BN2GS1S2_H_
#define FLASH_ATTENTION_KVSAME_BN2GS1S2_H_
#include "./flash_attention_score_common_regbase.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "./vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "./vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "./vf/vf_flashupdate_new.h"
#include "./vf/vf_div_cast.h"
#include "./vf/vf_flash_decode.h"
#include "./vf/vf_post_quant.h"
#include "./attenmask.h"

#include "../matmul.h"
#include "../FixpipeOut.h"
#include "../CopyInL1.h"

#include "pse.h"
#include "./infer_flash_attention_comm.h"
#include "./infer_flash_attention_kvcache.h"
#include "./infer_flash_attention_sparse.h"
#include "kernel_operator_list_tensor_intf.h"
#include "util_regbase.h"
#include "flash_attention_score_tiling_regbase.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

static constexpr uint32_t BYTE_BLOCK_32B = 32;

#define CV_RATIO 2

#define PRELOAD_N 1

#define SOFTMAXSUMBUF_SIZE 256
#define SUMBRDCST_SIZE 2048
#define SOFTMAXMAXBUF_SIZE 256
#define SOFTMAXEXPBUF_SIZE 256
#define NUM_2 2
#define NUM_4 4
#define NUM_8 8

__aicore__ constexpr uint16_t Align64Func_(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

template<typename INPUT_T, typename T = INPUT_T, ImplModeEnum implMode = ImplModeEnum::AA_HIGH_PRECISION, 
    LayOutTypeEnum layout = LayOutTypeEnum::None,
    S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
    DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128,
    PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasAtten = false, bool hasDrop = false, bool hasRope = false,
    typename OUTPUT_T = INPUT_T, bool isInfer = false, bool isPa = false, bool isFd = false>
class FlashAttentionKvsameBN2GS1S2 {
public:
    __aicore__ inline FlashAttentionKvsameBN2GS1S2() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    static constexpr bool isW8In = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value ||
                                  IsSameType<INPUT_T, int8_t>::value;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = typename AscendC::Conditional<isW8In, half, INPUT_T>::type;

protected:
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    __aicore__ inline void MlaInitOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                                    __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                                    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline int64_t GetQueryRopeOffset(RunInfo<isInfer> &runInfo);
    __aicore__ inline int64_t GetKeyRopeOffset(RunInfo<isInfer> &runInfo);
    __aicore__ inline void InitPostQuant(__gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset);

    __aicore__ inline bool IsLastBN(uint32_t bnStartIdx, uint32_t bnEndIdx);
    __aicore__ inline void IterateBmm1(RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam, bool isLast);
    __aicore__ inline void WaitBmm1Result(RunInfo<isInfer> &runInfo);
    __aicore__ inline void IterateBmm2(RunInfo<isInfer> &runInfo);
    __aicore__ inline void WaitBmm2Result(RunInfo<isInfer> &runInfo);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp, 
        RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
        RunInfo<isInfer> &runInfo);
    __aicore__ inline void SetRunInfo(RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndx, int64_t gS1Index, int64_t &multiCoreInnerIdx, RunParamStr<isInfer>& runParam);
    __aicore__ inline void ComputeBmm1Tail(RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam);

    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar, 
                                                  SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void InvalidLineProcess(RunInfo<isInfer> &runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb);
    __aicore__ inline void ProcessVec1(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void ProcessVec1Nd(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline int64_t ComputeOffsetForSoftmax(RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx);
    __aicore__ inline void MlaAttenMaskCopyIn(TQue<QuePosition::VECIN, 1> &attenMaskInQue, TQue<QuePosition::VECIN, 1> &attenMaskInQuePre, 
        GlobalTensor<uint8_t> &srcTensor, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, 
        AttenMaskInfo &attenMaskInfo, RunParamStr<isInfer>& runParam);
    __aicore__ inline void MlaBoolCopyInRegbase(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
        int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t s2BaseSize, 
        ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam);
    __aicore__ inline void ProcessVec2S2Split(RunInfo<isInfer> &runInfo);
    __aicore__ inline void ProcessVec2(RunInfo<isInfer> &runInfo);  

    /*VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = INPUT_T 那么不需要做Cast。另外，无效场景当前默认需要做Cast*/
    template<typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, 
                                           int64_t vec2CalcSize = 0);

    __aicore__ inline void Bmm2FDOut(RunInfo<isInfer> &runInfo, LocalTensor<T> &vec2ResUb, int64_t vec2CalcSize = 0);
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo);
    template<typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, RunInfo<isInfer> &runInfo);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);
    __aicore__ inline void InitLseOutputSingleCore();
    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline void InitFDBuffers();
    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void CombineSplitKVRes(uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void CopyLseIn(uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, 
                                            uint32_t splitSize, uint64_t lseOffset);
    __aicore__ inline void CopyFinalResOut(uint64_t attenOutOffset, LocalTensor<T>& accumOutLocal, uint32_t startRow,
                                           uint32_t dealRowCount, uint64_t perChannelQuantOffset);
    __aicore__ inline void CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);                                                                              
    __aicore__ inline void ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx, LocalTensor<T>& dst, LocalTensor<T>& lseLocal, 
                                          uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void ReduceFDDataCopyOut(uint64_t attenOutOffset, LocalTensor<OUTPUT_T>& attenOutUb, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx);

    __aicore__ inline void FDPostQuant(LocalTensor<OUTPUT_T> &attenOut, LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount);

    template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
    __aicore__ inline void PostQuantPerChnl(LocalTensor<OUTPUT_T> &attenOut,
    LocalTensor<VEC2_RES_T> &vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm);
    
    TPipe *pipe;

    // pageAttention需要
    __gm__ uint8_t* currentKey;
    __gm__ uint8_t* currentValue;
    __gm__ uint8_t* blocktablePtr;
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    KVLAYOUT kvLayout;
    // block_table
    GlobalTensor<int32_t> blockTableGm;

    static constexpr uint64_t kvHeadNum =1ULL;
    static constexpr uint64_t headDim =512ULL;
    static constexpr uint64_t headDimRope =64ULL;

    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    /*================GM变量================*/
    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<pseShiftType> pseGm;
    __gm__ uint8_t *pseSlope;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    GlobalTensor<T> bmm2ResGm[3];
    GlobalTensor<T> vec2ResGm[3];
    GlobalTensor<float> deScaleQGm;
    GlobalTensor<float> deScaleKGm;
    GlobalTensor<float> deScaleVGm;
    GlobalTensor<INPUT_T> queryRopeGm;
    GlobalTensor<INPUT_T> keyRopeGm;
    GM_ADDR prefixNAddr;

    GlobalTensor<int64_t> queryPaddingSizeGm;
    GlobalTensor<int64_t> kvPaddingSizeGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> softmaxFDMaxGm;
    GlobalTensor<T> softmaxFDSumGm;

    GlobalTensor<float> postQuantScaleGm;
    GlobalTensor<float> postQuantOffsetGm;
    GlobalTensor<bfloat16_t> postQuantScaleBf16Gm;
    GlobalTensor<bfloat16_t> postQuantOffsetBf16Gm;

    /*===========UB变量==============*/
    TBuf<> commonTBuf; // common的复用空间

    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue;
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TQue<QuePosition::VECIN, 1> pseInQue;
    TBuf<TPosition::VECIN> bmm1ResBuf[2];
    TBuf<TPosition::VECIN> bmm2ResBuf[2];
    TQue<QuePosition::VECOUT, 1> stage2OutQue[2];
    TQue<QuePosition::VECIN, 1> postQuantScaleQue;; // postQuant
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue;; // postQuant

    TBuf<> softmaxMaxBuf[3];
    TBuf<> softmaxSumBuf[3];
    TBuf<> softmaxExpBuf[3];
    TBuf<> vselrIndexesBuf[4];
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;

    /*用来做Broadcast[S1, 1]->[S2, 8]的临时UB区域*/
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    /*vector1结果存放在L1上，使用Tscm变量存储*/
    TSCM<QuePosition::VECIN, 1, 0x4> scm[2];
    /*用于减少在没有对应可选输入时AllocTensor上的Scalar开销*/
    LocalTensor<pseShiftType> dummyPseTensor;
    LocalTensor<uint8_t> dummyAttenMaskTensor;

    /*========核Index信息========*/
    int32_t blockIdx;
    int32_t aicIdx;
    int32_t aivIdx;

    /*========编译期常量的基本信息块信息========*/
    static constexpr uint32_t dTemplateAlign64 = Align64Func_((uint16_t)dVTemplateType);
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyCountDn = s1BaseSize / 32;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize / 2;
    static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize / 2;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = s1BaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = s1BaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = s1BaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = s1BaseSize / 2 + 1;

    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    static constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8;

    static constexpr bool useDn = false;
    static constexpr bool enableKVPrefix = false;
    /*========常量信息，只和输入shape相关的信息========*/
    ConstInfo<isInfer, hasRope> constInfo;
    PseInfo pseInfo;
    AttenMaskInfo attenMaskInfo;

    // keyOffset记录，value总是可以使用上一次key的offset
    int64_t keyRopeOffset[3];
    // 用来判断s2是否可以复用上一次的B和N2的index
    int64_t lastBIdx = -1;
    int64_t lastN2Idx = -1;
    // bmm2阶段subblock在Gm上的偏移
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;
    static constexpr bool hasPse = pseMode != PseTypeEnum::PSE_NONE_TYPE;
    static constexpr bool hasPseOuter = (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) ||
                                        (pseMode == PseTypeEnum::PSE_OUTER_MUL_ADD_TYPE);
    static constexpr bool containAllOptionalInput = hasPse && hasAtten;
    bool softMaxCheckRes = true;
    T negativeFloatScalar;
    T positiveFloatScalar;
    
    static constexpr bool splitD = (uint16_t) dVTemplateType > (uint16_t)DTemplateType::Aligned256;
    static constexpr bool mm2LeftFromUB = true;
    static constexpr bool mm2RightStillInL1 = true;

    // Unpack参数
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t b1SSOffset;
    uint64_t b1SSOffsetAlign16;
    int64_t dBasicBlock;
    uint32_t splitKVNum;

    static constexpr uint64_t SYNC_C1_V1_FLAG[4] = {0, 1, 6, 9};
    static constexpr uint64_t SYNC_V1_C2_FLAG[4] = {2, 3, 7, 10};
    static constexpr uint64_t SYNC_C2_V2_FLAG[4] = {4, 5, 8, 11};
    static constexpr uint64_t mm2ResIntraEvent[2] = {12, 13};
    static constexpr uint64_t mm1ResIntraEvent[2] = {14, 15};

    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L1> l1BufferManager;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0A> l0aBufferManager;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0B> l0bBufferManager;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0C> l0cBufferManager;

    // mm1左矩阵，GS1循环内左矩阵复用，GS1循环间不开pingpong
    fa_base_matmul::BuffersPolicySingleBuffer<fa_base_matmul::BufferType::L1> mm1AL1Buffers;
    // mm1和mm2右矩阵，在L1上复用，其中K_rope内存空间与bmm2的左矩阵p复用
    fa_base_matmul::BuffersPolicy3buff<fa_base_matmul::BufferType::L1> mm12Bmm2AL1Buffers;
    // L0A
    fa_base_matmul::BuffersPolicyDB<fa_base_matmul::BufferType::L0A> mmL0ABuffers;
    // L0B
    fa_base_matmul::BuffersPolicyDB<fa_base_matmul::BufferType::L0B> mmL0BBuffers;
    // L0C
    fa_base_matmul::BuffersPolicyDB<fa_base_matmul::BufferType::L0C> mmL0CBuffers;

    event_t UbToL1Event;

    TEventID mte2ToV[1];    // 存放 MTE2_V 的 eventID
};

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    this->tilingData = tiling;
    this->blockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        this->aicIdx = this->blockIdx >> 1;
        this->aivIdx = this->blockIdx;
    }
    if ASCEND_IS_AIC {
        this->aicIdx = this->blockIdx;
    }
    
    this->MlaInitOutput(attentionOut, softmaxLse);
    this->InitInput(query, key, value, pse, attenMask, actualSeqLengths, actualSeqLengthsKv, blockTable, postQuantScale, postQuantOffset, queryRope, keyRope, softmaxLse, attentionOut, workspace, tiling, tPipe);
    this->ComputeConstexpr();
    this->InitBuffer();

    mte2ToV[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::MlaInitOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse)
{
    this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
    if constexpr (POST_QUANT) {
        this->attentionOutInitGm.SetGlobalBuffer((__gm__ half *)attentionOut);
    }
    if (this->tilingData->inputParamsRegbase.isSoftMaxLseEnable) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
    }

    if ASCEND_IS_AIV {
        if (this->tilingData->initOutputParams.needInit == 1) {
            InitOutputSingleCore();
            if (this->tilingData->inputParamsRegbase.isSoftMaxLseEnable) {
                InitLseOutputSingleCore();
            }
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::InitOutputSingleCore()
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - this->aivIdx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    if constexpr (POST_QUANT) {
        InitOutput<half>(attentionOutInitGm[this->aivIdx * initParams.singleCoreSize / 2], singleInitOutputSize / 2, 0.0);
    } else {
        InitOutput<OUTPUT_T>(attentionOutGm[this->aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
    }

    SyncAll();
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::InitLseOutputSingleCore()
{
    int64_t tmpBlockIdx = this->aivIdx;
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParams = this->tilingData->initOutputParams;
    if (coreNum != 0 && tmpBlockIdx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        uint32_t tailSize = initParams.totalSoftMaxLseOutputSize - this->aivIdx * singleCoreLseSize;
        uint32_t singleInitLseSize = tailSize < singleCoreLseSize ? tailSize : singleCoreLseSize;
        InitOutput<float>(softmaxLseGm[tmpBlockIdx * singleCoreLseSize], singleInitLseSize, 3e+99); // 3e+99: set the value of invaild batch to inf
        SyncAll();
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar, T &positiveScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
            positiveScalar = *((half *)&tmp2);
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::InitInput(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    constInfo.subBlockIdx = GetSubBlockIdx();
    this->pipe = tPipe;
    this->tilingData = tiling;

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    ListTensorDesc keyListTensorDescInit((__gm__ void*)key);
    ListTensorDesc valueListTensorDescInit((__gm__ void*)value);
    currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    if (this->tilingData->inputParamsRegbase.isKvContinuous == 1) {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)currentKey);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)currentValue);
    } else {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    }

    if constexpr (isPa) {
        blocktablePtr = blockTable;
        this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        this->kvCacheBlockSize = this->tilingData->inputParamsRegbase.blockSize;
        this->maxBlockNumPerBatch = this->tilingData->inputParamsRegbase.blockTableDim2;
        if (this->tilingData->inputParamsRegbase.paLayoutType == 2) { // NZ下paLayoutType == 2
            kvLayout = KVLAYOUT::NZ;
        } else {
            kvLayout = this->tilingData->inputParamsRegbase.paLayoutType == 1 ? KVLAYOUT::BBH : KVLAYOUT::BNBD;
        }
    }
    this->pseGm.SetGlobalBuffer((__gm__ pseShiftType *)pse);
    this->pseSlope = pse;

    if constexpr (hasAtten) {
        this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    }

    if (this->tilingData->inputParamsRegbase.isActualSeqLengthsNull != 1) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
    }
    if (this->tilingData->inputParamsRegbase.isActualSeqLengthsKVNull != 1) {
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    }
    if constexpr (hasRope) {
        this->queryRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)queryRope);
        this->keyRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)keyRope);
    }

    this->dBasicBlock = Align64Func_((uint16_t)this->tilingData->inputParamsRegbase.dSizeV);
    if constexpr (isFd) {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        this->splitKVNum = this->tilingData->inputParamsRegbase.kvSplitPart;
        uint64_t accumOutSize = this->tilingData->inputParamsRegbase.accumOutSize;
        uint64_t logSumExpSize = this->tilingData->inputParamsRegbase.logSumExpSize;

        accumOutGm.SetGlobalBuffer((__gm__ T *)(workspace));
        workspace += accumOutSize * sizeof(float);
        softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
        softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
    }

    if constexpr (POST_QUANT) {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        this->constInfo.isPostQuantPerChnl = inputParamsRegbase.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = inputParamsRegbase.isPostQuantBF16;
        this->InitPostQuant(postQuantScale, postQuantOffset);
    }

    this->GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::InitPostQuant(__gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset)
{
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantOffsetExist = false;
        if (!this->constInfo.isPostQuantPerChnl && !this->constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
                this->constInfo.postQuantScaleValue = postQuantScaleGm.GetValue(0);
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
                this->constInfo.postQuantOffsetValue = postQuantOffsetGm.GetValue(0);
            } else {
                this->constInfo.postQuantOffsetValue = 0.0;
            }
        }

        if (!this->constInfo.isPostQuantPerChnl && this->constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
                this->constInfo.postQuantScaleValue = ToFloat(postQuantScaleBf16Gm.GetValue(0));
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
                this->constInfo.postQuantOffsetValue = ToFloat(postQuantOffsetBf16Gm.GetValue(0));
            } else {
                this->constInfo.postQuantOffsetValue = 0.0;
            }
        }

        if (this->constInfo.isPostQuantPerChnl && !this->constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                this->postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                this->constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
            }
        }

        if (this->constInfo.isPostQuantPerChnl && this->constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                this->constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
            }
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    this->pipe->InitBuffer(this->softmaxSumBuf[0], SOFTMAXSUMBUF_SIZE); // [64, 1] SOFTMAXSUMBUF_SIZE:256
    this->pipe->InitBuffer(this->softmaxSumBuf[1], SOFTMAXSUMBUF_SIZE); // [64, 1]
    this->pipe->InitBuffer(this->softmaxSumBuf[NUM_2], SOFTMAXSUMBUF_SIZE); // [64, 1]
    this->pipe->InitBuffer(this->sumBrdcst, 1, SUMBRDCST_SIZE); // [64, 8] SUMBRDCST_SIZE:2048
    this->pipe->InitBuffer(this->softmaxMaxBuf[0], SOFTMAXMAXBUF_SIZE); // [64, 1] SOFTMAXMAXBUF_SIZE:256
    this->pipe->InitBuffer(this->softmaxMaxBuf[1], SOFTMAXMAXBUF_SIZE); // [64, 1]
    this->pipe->InitBuffer(this->softmaxMaxBuf[NUM_2], SOFTMAXMAXBUF_SIZE); // [64, 1]
    this->pipe->InitBuffer(this->softmaxExpBuf[0], SOFTMAXEXPBUF_SIZE); // [64, 1] SOFTMAXEXPBUF_SIZE:256
    this->pipe->InitBuffer(this->softmaxExpBuf[1], SOFTMAXEXPBUF_SIZE); // [64, 1]
    this->pipe->InitBuffer(this->softmaxExpBuf[NUM_2], SOFTMAXEXPBUF_SIZE); // [64, 1]
    if (constInfo.isSoftmaxLseEnable) {
        pipe->InitBuffer(softmaxLseQueue, 1, (s1BaseSize >> 1U) * sizeof(float) * 8); // 8:适配TND， 每行的结果存为8个重复lse元素（32B对齐）
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::InitBuffer()
{
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    uint32_t mm12RightSize = max((uint32_t)dTemplateType, (uint32_t)dVTemplateType) * s2BaseSize * sizeof(INPUT_T);

    if ASCEND_IS_AIC {
        // L0A B C 当前写死，能否通过基础api获取
        l1BufferManager.Init(pipe, 524288);
        l0aBufferManager.Init(pipe, 65536);
        l0bBufferManager.Init(pipe, 65536);
        l0cBufferManager.Init(pipe, 262144);

        // s1=64, s2=128分核方案，mm1和mm2结果全部在ub上
        if constexpr(s1BaseSize == 64 && s2BaseSize == 128) {
            // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
            mm12Bmm2AL1Buffers.Init(l1BufferManager, mm12RightSize); // L1P与L1K_rope复用
            mm1AL1Buffers.Init(l1BufferManager, (uint32_t)dTemplateType * s1BaseSize * 2);
            // L0A B C当前写死，要改成通过计算获取
            mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
            mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
            mmL0CBuffers.Init(l0cBufferManager, 128 * 1024);
            this->pipe->InitBuffer(this->bmm1ResBuf[0], mm1ResultSize);
            this->pipe->InitBuffer(this->bmm1ResBuf[1], mm1ResultSize);
            this->pipe->InitBuffer(this->bmm2ResBuf[0], mm2ResultSize);
            this->pipe->InitBuffer(this->bmm2ResBuf[1], mm2ResultSize);
        }
    }
    if ASCEND_IS_AIV {
        if constexpr(s1BaseSize == 64 && s2BaseSize == 128) {
            //申请保存P结果的L1内存
            l1BufferManager.Init(pipe, 524288);
            mm12Bmm2AL1Buffers.Init(l1BufferManager, mm12RightSize);
            // 保存MM1和MM2结果的UB内存必须放在前两个 policy上，且必须位于其它ub申请之前，以保证和CUBE申请的地址相同
            this->pipe->InitBuffer(this->bmm1ResBuf[0], mm1ResultSize);
            this->pipe->InitBuffer(this->bmm1ResBuf[1], mm1ResultSize);
            this->pipe->InitBuffer(this->bmm2ResBuf[0], mm2ResultSize);
            this->pipe->InitBuffer(this->bmm2ResBuf[1], mm2ResultSize);

            // vec侧需要先发mm1和mm2的两个核间同步
            CrossCoreSetFlag<4, PIPE_V>(mm2ResIntraEvent[0]);
            CrossCoreSetFlag<4, PIPE_V>(mm2ResIntraEvent[1]);
            CrossCoreSetFlag<4, PIPE_V>(mm1ResIntraEvent[0]);
            CrossCoreSetFlag<4, PIPE_V>(mm1ResIntraEvent[1]);

            SoftmaxInitBuffer();
            this->pipe->InitBuffer(this->commonTBuf, 512); // 实际只需512Bytes
            if constexpr (hasPseOuter) {
                this->pipe->InitBuffer(this->pseInQue, 1, s1BaseSize / CV_RATIO * s2BaseSize * sizeof(INPUT_T));
            }
            if constexpr (hasAtten) {
                this->pipe->InitBuffer(this->attenMaskInQue[0], 1, 4096); // GS1方向需要循环处理
                this->pipe->InitBuffer(this->attenMaskInQue[1], 1, 4096); // 在当前分核逻辑下，一个vector计算softmax的数据量最大为32*128，对应mask(bool/int8/uint8)的数据量为4096Byte
            }
            this->pipe->InitBuffer(this->stage1OutQue[0], 1, (s1BaseSize / CV_RATIO + 1) * s2BaseSize * sizeof(INPUT_T));
            this->pipe->InitBuffer(this->stage2OutQue[0], 1, s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T));

            if constexpr (POST_QUANT) {
                this->pipe->InitBuffer(postQuantScaleQue, 1, 2048);
                if (this->constInfo.isPostQuantOffsetExist) {
                    this->pipe->InitBuffer(postQuantOffsetQue, 1, 2048);
                }
            }
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;
    // 计算轴的乘积
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;

    constInfo.n2Size = inputParamsRegbase.n2Size;
    constInfo.s1Size = inputParamsRegbase.s1Size;
    constInfo.s2Size = inputParamsRegbase.s2Size;
    if constexpr (isFd) {
        constInfo.sInnerLoopSize = CeilDivision(inputParamsRegbase.s2Size, splitKVNum);
        constInfo.actualCombineLoopSize = CeilDivision(constInfo.s2Size, constInfo.sInnerLoopSize);
    }
    constInfo.dSize = inputParamsRegbase.dSize;
    constInfo.dSizeV = inputParamsRegbase.dSizeV;
    if constexpr (hasRope) {
        constInfo.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        constInfo.dSizeRope = 0;
    }
    constInfo.gSize = inputParamsRegbase.gSize;
    constInfo.s1OuterSize = this->tilingData->multiCoreParamsRegbase.s1OuterSize;

    constInfo.s1D = constInfo.s1Size * constInfo.dSize;
    constInfo.s2D = constInfo.s2Size * constInfo.dSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2D = constInfo.n2Size * constInfo.dSize;
    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    int64_t bSize = inputParamsRegbase.bSize;
    constInfo.bN2D = bSize * constInfo.n2D;
    constInfo.bN2Dv = bSize * constInfo.n2Dv;
    constInfo.gS1D = constInfo.gSize * constInfo.s1D;
    constInfo.n2S2D = constInfo.n2Size * constInfo.s2D;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;
    constInfo.bN2GD = bSize * constInfo.n2GD;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.bN2GDv = bSize * constInfo.n2GDv;

    constInfo.n2GS1D = constInfo.n2Size * constInfo.gS1D;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;
    // 计算切分轴的乘积

    constInfo.s2BaseN2D = s2BaseSize * constInfo.n2D;
    constInfo.s2BaseN2Dv = s2BaseSize * constInfo.n2Dv;
    if constexpr (isInfer) {
        constInfo.n2S2D /= inputParamsRegbase.headNumRatio;
        constInfo.n2S2Dv /= inputParamsRegbase.headNumRatio;
        constInfo.s2BaseN2D /= inputParamsRegbase.headNumRatio;
        constInfo.s2BaseN2Dv /= inputParamsRegbase.headNumRatio;
    }
    if constexpr (hasRope) {
        constInfo.s1DR = constInfo.s1Size * constInfo.dSizeRope;
        constInfo.s2DR = constInfo.s2Size * constInfo.dSizeRope;
        constInfo.gDR = constInfo.gSize * constInfo.dSizeRope;
        constInfo.n2DR = constInfo.n2Size * constInfo.dSizeRope;
        constInfo.bN2DR = bSize * constInfo.n2DR;
        constInfo.gS1DR = constInfo.gSize * constInfo.s1DR;
        constInfo.n2S2DR = constInfo.n2Size * constInfo.s2DR;
        constInfo.n2GDR = constInfo.n2Size * constInfo.gDR;
        constInfo.bN2GDR = bSize * constInfo.n2GDR;
        constInfo.n2GS1DR = constInfo.n2Size * constInfo.gS1DR;
        constInfo.s2BaseN2DR = s2BaseSize * constInfo.n2DR;
    }
    constInfo.layoutType = inputParamsRegbase.layoutType;
    constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        if constexpr (hasRope) {
            constInfo.s1BaseDR = s1BaseSize * constInfo.dSizeRope;
            constInfo.mm1RopeKa = constInfo.dSizeRope;
            constInfo.mm1RopeKb = constInfo.n2DR;
        }
        constInfo.mm1Ka = constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2D;
        constInfo.mm2Kb = constInfo.n2Dv;
        if constexpr (isInfer) {
            constInfo.mm1Kb /= inputParamsRegbase.headNumRatio;
            constInfo.mm2Kb /= inputParamsRegbase.headNumRatio;
        }
        constInfo.attentionOutStride = 0;
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
            constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
            if constexpr (hasRope) {
                constInfo.s1BaseDR = s1BaseSize * constInfo.dSizeRope;
                constInfo.mm1RopeKa = constInfo.dSizeRope;
                constInfo.mm1RopeKb = constInfo.n2DR;
            }
            constInfo.mm1Ka = constInfo.dSize;
            constInfo.mm1Kb = constInfo.n2D;
            constInfo.mm2Kb = constInfo.n2Dv;
            if constexpr (isInfer) {
                constInfo.mm1Kb /= inputParamsRegbase.headNumRatio;
                constInfo.mm2Kb /= inputParamsRegbase.headNumRatio;
            }
            constInfo.attentionOutStride = 0;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            constInfo.s1BaseBN2GD = s1BaseSize * constInfo.bN2GD;
            constInfo.s2BaseBN2D = bSize * constInfo.s2BaseN2D;
            constInfo.s1BaseBN2GDv = s1BaseSize * constInfo.bN2GDv;
            constInfo.s2BaseBN2Dv = bSize * constInfo.s2BaseN2Dv;
            if constexpr (hasRope) {
                constInfo.s1BaseBN2GDR = s1BaseSize * constInfo.bN2GDR;
                constInfo.s2BaseBN2DR = bSize * constInfo.s2BaseN2DR;
                constInfo.mm1RopeKa = constInfo.bN2GDR;
                constInfo.mm1RopeKb = constInfo.bN2DR;
            }
            constInfo.mm1Ka = constInfo.bN2GD;
            constInfo.mm1Kb = constInfo.bN2D;
            constInfo.mm2Kb = constInfo.bN2Dv;
            constInfo.attentionOutStride = 
                (bSize * constInfo.n2Size * constInfo.gSize - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // BNSD
            constInfo.s1BaseD = s1BaseSize * constInfo.dSize;
            constInfo.s2BaseD = s2BaseSize * constInfo.dSize;
            constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
            constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
            if constexpr (hasRope) {
                constInfo.s1BaseDR = s1BaseSize * constInfo.dSizeRope;
                constInfo.s2BaseDR = s2BaseSize * constInfo.dSizeRope;
                constInfo.mm1RopeKa = constInfo.dSizeRope;
                constInfo.mm1RopeKb = constInfo.dSizeRope;
            }
            constInfo.mm1Ka = constInfo.dSize;
            constInfo.mm1Kb = constInfo.dSize;
            constInfo.mm2Kb = constInfo.dSizeV;
            constInfo.attentionOutStride = 0;
        }
    }

    if constexpr (hasPse == true) {
        this->pseInfo.pseLayoutType = inputParamsRegbase.pseShapeType;
        this->pseInfo.pseType = inputParamsRegbase.pseType;
        this->pseInfo.pseBSize = inputParamsRegbase.pseBSize;
        this->pseInfo.pseS1Size = inputParamsRegbase.pseS1Size;
        this->pseInfo.pseS2Size = inputParamsRegbase.pseS2Size;
        this->pseInfo.pseEncodeType = (uint32_t)inputParamsRegbase.pseEncodeType;
        this->pseInfo.pseStride = pseInfo.pseLayoutType == pse1S2 ? 0 : s2BaseSize;
        this->pseInfo.qStartIdx = inputParamsRegbase.qStartIdx;
        this->pseInfo.kvStartIdx = inputParamsRegbase.kvStartIdx;
        if (inputParamsRegbase.pseShapeType == pse1S2) {
            constInfo.gS2 = constInfo.gSize * constInfo.s2Size;
        }
    }
    if constexpr (hasAtten == true) {
        this->attenMaskInfo.preTokens = inputParamsRegbase.preTokens;
        this->attenMaskInfo.nextTokens = inputParamsRegbase.nextTokens;
        this->attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
        this->attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
        this->attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
        this->attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
    }

    constInfo.isRowInvalid = inputParamsRegbase.isRowInvalid;
    constInfo.headNumRatio = inputParamsRegbase.headNumRatio;
    constInfo.dSizeRope = inputParamsRegbase.ropeHeadSize;
    constInfo.isGqa = inputParamsRegbase.isGqa;
    constInfo.n2GDR = constInfo.gSize * constInfo.n2Size * constInfo.dSizeRope;
    constInfo.n2DR = constInfo.n2Size * constInfo.dSizeRope;
    constInfo.isKvContinuous = inputParamsRegbase.isKvContinuous;
    constInfo.actualSeqLenSize = inputParamsRegbase.actualSeqLengthsSize;
    constInfo.actualSeqLenKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    constInfo.isActualLenDimsNull = (inputParamsRegbase.isActualSeqLengthsNull == 1) ? true : false;
    constInfo.isActualLenDimsKVNull = (inputParamsRegbase.isActualSeqLengthsKVNull == 1) ? true : false;
    constInfo.isQHasLeftPadding = (inputParamsRegbase.isQHasLeftPadding == 1) ? true : false;
    constInfo.isKVHasLeftPadding = (inputParamsRegbase.isKVHasLeftPadding == 1) ? true : false;
    // pageAttention
    constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
    constInfo.blockSize = inputParamsRegbase.blockSize;
    constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
    constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;

    // service vector2
    constInfo.isBSNDOut = inputParamsRegbase.isBSNDOut;
    if (constInfo.isBSNDOut == 1) {
        constInfo.attentionOutStride = 
            (constInfo.n2Size * constInfo.gSize - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
    }

    // lse output
    constInfo.isSoftmaxLseEnable = inputParamsRegbase.isSoftMaxLseEnable;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline int64_t FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::GetQueryRopeOffset(
    RunInfo<isInfer> &runInfo) 
{
    // 计算gm上的offset
    int64_t bOffsetRope = 0;
    // s1需要考虑inner轴的影响
    int64_t s1OffsetRope = 0;
    int64_t n2OffsetRope = 0;
    int64_t gOffsetRope = 0;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        bOffsetRope = runInfo.s1SizeAcc * constInfo.n2GDR;
        s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseDR;
        n2OffsetRope = runInfo.n2oIdx * constInfo.gDR;
        gOffsetRope = runInfo.goIdx * constInfo.dSizeRope;
    }  else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            bOffsetRope = runInfo.boIdx * constInfo.n2GS1DR;
            s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseDR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.gDR;
            gOffsetRope = runInfo.goIdx * constInfo.dSizeRope;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH / SBNGD
            s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseBN2GDR;
            bOffsetRope = runInfo.boIdx * constInfo.n2GDR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.gDR;
            gOffsetRope = runInfo.goIdx * constInfo.dSizeRope;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            bOffsetRope = runInfo.boIdx * constInfo.n2GS1DR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.gS1DR;
            gOffsetRope = runInfo.goIdx * constInfo.s1DR;
            s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseDR;
        }
    } 
    int64_t ret = bOffsetRope + n2OffsetRope + gOffsetRope + s1OffsetRope;
    if ((layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_BSH) && runInfo.nextTokensPerBatch < 0) {
        ret += (-runInfo.nextTokensPerBatch) * constInfo.dSizeRope;
    }
    return ret;
}


CHILD_SPEC_TEMPLATE
__aicore__ inline int64_t FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::GetKeyRopeOffset(
    RunInfo<isInfer> &runInfo) 
{
    // 计算gm上的offset
    int64_t bOffsetRope = 0;
    int64_t n2OffsetRope = 0;
    int64_t s2OffsetRope = 0;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        bOffsetRope = runInfo.s2SizeAcc * constInfo.n2DR;
        s2OffsetRope = runInfo.s2StartIdx * constInfo.n2DR + (runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) * constInfo.s2BaseN2DR;
        n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSND
            bOffsetRope = runInfo.boIdx * constInfo.n2S2DR;
            s2OffsetRope = runInfo.s2StartIdx * constInfo.n2DR + (runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) * constInfo.s2BaseN2DR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH / SBND
            s2OffsetRope = runInfo.s2StartIdx * constInfo.bN2DR + (runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) * constInfo.s2BaseBN2DR;
            bOffsetRope = runInfo.boIdx * constInfo.n2DR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // BNSD
            bOffsetRope = runInfo.boIdx * constInfo.n2S2DR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.s2DR;
            s2OffsetRope = runInfo.s2StartIdx * constInfo.dSizeRope + (runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) * constInfo.s2BaseDR;
        }
    }
    return bOffsetRope + n2OffsetRope + s2OffsetRope;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline bool FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::IsLastBN(uint32_t bnStartIdx, uint32_t bnEndIdx)
{
    if constexpr(layout == LayOutTypeEnum::LAYOUT_TND) {
        if (bnStartIdx != bnEndIdx - 1) {
            for (uint32_t bnIdx = bnStartIdx + 1; bnIdx < bnEndIdx; bnIdx++) {
                uint32_t boIdx = bnIdx / constInfo.n2Size;
                uint32_t boStart = bnStartIdx / constInfo.n2Size;
                if (actualSeqQlenAddr[boIdx] != actualSeqQlenAddr[boStart]) {
                    return false;
                }
            }
        }
        return true;
    } else {
        return bnStartIdx == bnEndIdx - 1;
    }
    return false;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::Process()
{
    int32_t actualCoreNums = this->tilingData->multiCoreParamsRegbase.coreNum;
    if constexpr (isFd) {
        actualCoreNums = this->tilingData->inputParamsRegbase.bSize * constInfo.n2Size * splitKVNum; // b * n2 * splitKv
    }

    if (aicIdx >= actualCoreNums) {
        return;
    }
    // 确定核内切分起点
    int64_t gS1StartIdx = 0;
    int64_t gS1EndIdx = 1;
    uint32_t bnStartIdx = 0;
    uint32_t bnEndIdx = 1;
    int64_t s2LoopStart = 0;
    int64_t s2LoopLimit = 0;

    if constexpr (!isFd) {
        bnStartIdx = this->tilingData->multiCoreParamsRegbase.bnStartIdx[aicIdx];
        gS1StartIdx = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx];
        if (likely((this->tilingData->multiCoreParamsRegbase.coreNum - 1) > aicIdx)) {
            bnEndIdx = this->tilingData->multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
            // 下一个核从0开始gs1循环，当前核bn不需要多计算一个，否则需要多计算一个bn
            if (this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1] != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = this->tilingData->inputParamsRegbase.bSize * constInfo.n2Size;
        }
    } 
    int64_t taskId = 0;
    bool isLastBmm1 = false;
    RunInfo<isInfer> runInfo[NUM_4];
    RunParamStr<isInfer> runParam;

    if constexpr (isFd) {
        runParam.boIdx = (aicIdx) / (constInfo.n2Size * splitKVNum);
        runParam.n2oIdx = ((aicIdx) / splitKVNum) % constInfo.n2Size;
        bnStartIdx = runParam.boIdx * constInfo.n2Size + runParam.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }
    int64_t multiCoreInnerIdx = 0;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; bnIdx++) {
        bool lastBN = IsLastBN(bnIdx, bnEndIdx);
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / constInfo.n2Size;
            runParam.n2oIdx = bnIdx % constInfo.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, this->attenMaskInfo, keyGm, 
            actualSeqQlenAddr, actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, lastBN, this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1]);
        if constexpr (isFd) {
            if (constInfo.sInnerLoopSize * (aicIdx % splitKVNum) > runParam.actualSeqLengthKVPerBatch) {
                runParam.actualSInnerLoopSize = 0;
            } else {
                int64_t tailSInnerLoopSize =
                    runParam.actualSeqLengthKVPerBatch - constInfo.sInnerLoopSize * (aicIdx % splitKVNum);
                runParam.actualSInnerLoopSize =
                    tailSInnerLoopSize > constInfo.sInnerLoopSize ? constInfo.sInnerLoopSize : tailSInnerLoopSize;
            }
            runParam.s1LoopTimes = 1; // GQA支持后解决
        }

        gS1EndIdx = runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index <runParam.s1LoopTimes; gS1Index++) {
            s2LoopLimit = 0;
            this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, multiCoreInnerIdx, runParam);
            bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, gS1Index, actualSeqQlenAddr, this->pseInfo);
            bool s2NoNeedCalc = ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo);
            bool lastLoopThisCore = lastBN && (gS1Index == runParam.s1LoopTimes - 1);
            bool lastBnNoNeedCalc = ComputeLastBN<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, actualSeqQlenAddr);
            if (((s1NoNeedCalc || s2NoNeedCalc) && !lastLoopThisCore) || lastBnNoNeedCalc) {
                continue;
            }
            // s2轴循环计数，支持sparse和非sparse场景
            s2LoopLimit = runParam.s2LoopEndIdx - 1;
            if (lastLoopThisCore) {
                isLastBmm1 = true;
                s2LoopLimit += PRELOAD_N;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
                if (s2LoopCount < runParam.s2LoopEndIdx) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, runParam.s2LoopEndIdx - 1, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->IterateBmm1(runInfo1, runParam, isLastBmm1 && (s2LoopCount == (runParam.s2LoopEndIdx - 1)));
                        CrossCoreSetFlag<4, PIPE_FIX>(SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
                        CrossCoreSetFlag<4, PIPE_FIX>(16 + SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
                    }
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag<4, PIPE_V>(SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // 等待bmm1完成/等待SYNC_C1_V1_FLAG置位
                        this->ProcessVec1(runInfo1, runParam);
                        CrossCoreSetFlag<4, PIPE_MTE3>(SYNC_V1_C2_FLAG[runInfo1.taskIdMod2]); // mte3将结果搬运到L1后，设置SYNC_V1_C2_FLAG
                    }
                }
                if (taskId >= PRELOAD_N) {
                    RunInfo<isInfer> &runInfo2 = runInfo[(taskId - PRELOAD_N) & 3];
                    if ASCEND_IS_AIC {
                        CrossCoreWaitFlag<4, PIPE_MTE1>(SYNC_V1_C2_FLAG[runInfo2.taskIdMod2]); 
                        CrossCoreWaitFlag<4, PIPE_MTE1>(16 + SYNC_V1_C2_FLAG[runInfo2.taskIdMod2]); 
                        this->IterateBmm2(runInfo2);
                        CrossCoreSetFlag<4, PIPE_FIX>(SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C2_V2_FLAG
                        CrossCoreSetFlag<4, PIPE_FIX>(16 + SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C2_V2_FLAG
                    }
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag<4, PIPE_V>(SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // 等待bmm2完成/等待SYNC_C2_V2_FLAG置位
                        this->ProcessVec2(runInfo2);
                    }
                }
                taskId++;
            }
        }
        gS1StartIdx = 0;
    }

    if constexpr (isFd) {
        SyncAll();
        InitFDBuffers();
        FlashDecodeCompute();
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::InitFDBuffers()
{
    pipe->Reset();
    pipe->InitBuffer(lseTmpBuff, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(softmaxMaxInputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(softmaxSumInputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(FDResOutputQue, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(accumOutInputQue, 1, BUFFER_SIZE_BYTE_32K);
    if constexpr (POST_QUANT) {
        this->pipe->InitBuffer(postQuantScaleQue, 1, BUFFER_SIZE_BYTE_32K);
        if (this->constInfo.isPostQuantOffsetExist) {
            this->pipe->InitBuffer(postQuantOffsetQue, 1, BUFFER_SIZE_BYTE_32K);
        }
    }
    if (constInfo.isSoftmaxLseEnable) {
        pipe->InitBuffer(softmaxLseQueue, 1, (s1BaseSize >> 1U) * sizeof(float) * 8); // 8：适配TND，每行的结果存为8个重复lse元素(32B对齐)
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::FlashDecodeCompute()
{
    uint32_t bIdx = aivIdx / constInfo.n2Size;
    uint32_t n2Idx = aivIdx % constInfo.n2Size;
    uint32_t batchSize = tilingData->inputParamsRegbase.bSize;
    if (aivIdx >= batchSize * constInfo.n2Size) {
        return;
    }

    uint64_t attenOutOffset = (uint64_t)bIdx * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV + n2Idx * constInfo.gSize * constInfo.dSizeV;

    CombineSplitKVRes(attenOutOffset, bIdx, n2Idx);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::CombineSplitKVRes(uint64_t attenOutOffset,
                                                                                        uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_32K / (BYTE_BLOCK_32B * splitKVNum);  // 16k / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = BUFFER_SIZE_BYTE_32K / sizeof(float) / constInfo.dSizeV; // aline
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;

    gSplitSize = (gSplitSize > constInfo.gSize) ? constInfo.gSize : gSplitSize;

    uint32_t loopCount = CeilDivision(constInfo.gSize, gSplitSize);
    uint32_t tailSplitSize = constInfo.gSize - (loopCount - 1) * gSplitSize;
    uint64_t lseOffset = 0;

    // 尾块与非尾块都使用这些ub，减少处理次数
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>();  // 复用内存
    uint32_t shapeArray[] = {(uint32_t)gSplitSize, FP32_ONE_BLOCK_SIZE};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

    uint64_t perChannelQuantOffset = n2Idx * this->constInfo.dSizeV * this->constInfo.gSize;

    // 非尾块处理
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        uint32_t startRow = i * gSplitSize;
        CopyLseIn(bIdx, n2Idx, startRow, gSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出scale值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + i * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, gSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, gSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(attenOutOffset, tmp1, startRow, gSplitSize, perChannelQuantOffset);
    }
    // 尾块处理
    if (tailSplitSize > 0) {
        uint32_t startRow = (loopCount - 1) * gSplitSize;
        CopyLseIn(bIdx, n2Idx, startRow, tailSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + (loopCount - 1) * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, tailSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, tailSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(attenOutOffset, tmp1, startRow, tailSplitSize, perChannelQuantOffset);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::CopyFinalResOut(
    uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount, uint64_t perChannelQuantOffset)
{
    LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = FDResOutputQue.AllocTensor<OUTPUT_T>(); // 复用内存
    uint32_t dSizeAligned64 = (uint32_t)dVTemplateType;
    uint32_t shapeArray[] = {(uint32_t)dealRowCount, dSizeAligned64};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape
    if constexpr (!POST_QUANT) {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * dSizeAligned64);
    } else {
        FDPostQuant(tmpBmm2ResCastTensor, accumOutLocal, perChannelQuantOffset + startRow * this->constInfo.dSizeV, dealRowCount);
    }
    FDResOutputQue.EnQue(tmpBmm2ResCastTensor);
    FDResOutputQue.DeQue<OUTPUT_T>();
    ReduceFDDataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, dSizeAligned64, this->constInfo.dSizeV);
    FDResOutputQue.FreeTensor(tmpBmm2ResCastTensor);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ReduceFDDataCopyOut(
    uint64_t attenOutOffset, LocalTensor<OUTPUT_T>& attenOutUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK_32B / sizeof(OUTPUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(this->attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::CopyLseIn(uint32_t bIdx, uint32_t n2Idx,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.AllocTensor<T>();
    LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = splitKVNum;
    copyInParams.blockLen = dealRowCount * FP32_ONE_BLOCK_SIZE * sizeof(T);
    copyInParams.srcStride = (constInfo.gSize - dealRowCount) * FP32_ONE_BLOCK_SIZE * sizeof(T);
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    uint64_t combineLseOffset = ((uint64_t) bIdx * constInfo.n2Size * splitKVNum + n2Idx * splitKVNum) * constInfo.gSize * FP32_ONE_BLOCK_SIZE + 
                                startRow * FP32_ONE_BLOCK_SIZE;
    
    DataCopyPad(softmaxMaxLocal, softmaxFDMaxGm[combineLseOffset], copyInParams, copyInPadParams);
    DataCopyPad(softmaxSumLocal, softmaxFDSumGm[combineLseOffset], copyInParams, copyInPadParams);
    softmaxMaxInputQue.EnQue(softmaxMaxLocal);
    softmaxSumInputQue.EnQue(softmaxSumLocal);
}         

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ComputeScaleValue(LocalTensor<T> lseMaxUb,
                                                                                        LocalTensor<T> lseSumUb,
                                                                                        uint32_t splitSize,
                                                                                        uint64_t lseOffset)
{
    LocalTensor<T> lseOutputUb;
    if (constInfo.isSoftmaxLseEnable) {
        lseOutputUb = softmaxLseQueue.template AllocTensor<T>();
    }
    ComputeScaleValue_VF(lseMaxUb, lseSumUb, lseOutputUb, splitSize, constInfo.actualCombineLoopSize, constInfo.isSoftmaxLseEnable);
    if (constInfo.isSoftmaxLseEnable) {
        softmaxLseQueue.template EnQue<T>(lseOutputUb);
        softmaxLseQueue.DeQue<T>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = splitSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[lseOffset], lseOutputUb, intriParams1);
        softmaxLseQueue.FreeTensor(lseOutputUb);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx, 
                                                                                    LocalTensor<T>& dst,
                                                                                    LocalTensor<T>& lseLocal,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    for (uint32_t j = 0; j < constInfo.actualCombineLoopSize; ++j) {
        // 第一次，mul结果直接放到dst里
        CopyAccumOutIn(bIdx, n2Idx, j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();
        ReduceFinalRes_const_VF<T, (uint32_t)dVTemplateType>(dst, lseLocal, accumOutLocal, dealRowCount, j);
        accumOutInputQue.FreeTensor(accumOutLocal);
    }
}        

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx,
                                                                                    uint32_t splitKVIndex,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = ((int64_t)dVTemplateType - constInfo.dSizeV) / 8;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = ((int64_t)dVTemplateType - constInfo.dSizeV) % 8;
    copyInPadParams.paddingValue = 0;

    uint64_t combineAccumOutOffset =
        ((uint64_t)bIdx * constInfo.n2Size * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * constInfo.gSize * constInfo.dSizeV +
        startRow * constInfo.dSizeV;
    DataCopyPad(accumOutLocal, this->accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}                                        

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::GetSeqQlenKvlenByBoidx(int64_t boIdx, 
    int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvlen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    if constexpr (isPa) {
        actualSeqKvlen = actualSeqKvlenAddr[boIdx];
    } else {
        actualSeqKvlen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
    }
}


CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, int64_t &multiCoreInnerIdx, RunParamStr<isInfer> &runParam)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if (runParam.boIdx == 0) {
            this->s1SizeAcc = 0;
            this->s2SizeAcc = 0;
        } else {
            this->s1SizeAcc = actualSeqQlenAddr[runParam.boIdx - 1];
            if constexpr (isPa) {
                this->s2SizeAcc = 0;
                for (uint32_t boIdx = 0; boIdx < runParam.boIdx; boIdx++) {
                    this->s2SizeAcc += actualSeqKvlenAddr[boIdx];
                }
            } else {
                this->s2SizeAcc = actualSeqKvlenAddr[runParam.boIdx - 1];
            }
        }
    }
    runParam.goIdx = gS1Index / constInfo.s1OuterSize;
    runParam.s1oIdx = gS1Index % constInfo.s1OuterSize;
    multiCoreInnerIdx++;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::WaitBmm1Result(RunInfo<isInfer> &runInfo) 
{
    CrossCoreWaitFlag<4, PIPE_V>(SYNC_C1_V1_FLAG[runInfo.taskIdMod2]); // 等待bmm1完成/等待SYNC_C1_V1_FLAG置位
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::SetRunInfo(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * constInfo.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, constInfo.s1Size, constInfo.s2Size);
        runInfo.b1SSOffset = this->b1SSOffset;
        runInfo.b1SSOffsetAlign = this->b1SSOffsetAlign16;
    } else {
        runInfo.b1SSOffset = runInfo.boIdx * constInfo.s1S2;
        runInfo.b1SSOffsetAlign = runInfo.boIdx * constInfo.s1Size * Align(constInfo.s2Size);
    }

    if constexpr (isFd) {
        runInfo.flashDecodeS2Idx = (this->aicIdx) % splitKVNum;
    }
    runInfo.actualS1Size = constInfo.s1Size;
    runInfo.actualS2Size = constInfo.s2Size;

    this->ComputeBmm1Tail(runInfo, runParam);
    if constexpr (isInfer) {
        runInfo.qRopeOffset = runParam.qRopeNBGOffset;
        InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, runInfo);
        ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, s2LoopCount + runInfo.s2StartIdx / s2BaseSize, runInfo);
    }
}   

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ComputeBmm1Tail(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam)
{
    // -----------S1 Base Related----------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;  // D>128 这里需要适配
    runInfo.vecCoreOffset = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;

    // -----------S2 Base Related-----------------
    runInfo.s2RealSize = s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::IterateBmm1(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam, bool isLast)
{
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L1> mm1A;
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L1> mm1B;
    // 左矩阵复用 ,s2的第一次循环加载左矩阵
    // 加载左矩阵到L1 当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本快：搬运0
        mm1A = mm1AL1Buffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用
        LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
        Nd2NzParams Gm2L1Nd2NzParams;
        Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
        Gm2L1Nd2NzParams.nValue = runInfo.s1RealSize; // 单个ND矩阵的实际行数，单位为元素个数
        Gm2L1Nd2NzParams.dValue = constInfo.dSize; // 单个ND矩阵的实际列数，单位为元素个数
        Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.srcDValue = constInfo.mm1Ka; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
        DataCopy(mm1ATensor, this->queryGm[runParam.tensorQOffset], Gm2L1Nd2NzParams);

        // 拷贝 Qrope
        if constexpr (hasRope) {
            int64_t queryRopeOffset = GetQueryRopeOffset(runInfo);
            Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
            Gm2L1Nd2NzParams.nValue = runInfo.s1RealSize; // 单个ND矩阵的实际行数，单位为元素个数
            Gm2L1Nd2NzParams.dValue = constInfo.dSizeRope; // 单个ND矩阵的实际列数，单位为元素个数
            Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
            Gm2L1Nd2NzParams.srcDValue = constInfo.mm1RopeKa; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
            Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
            Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
            Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
            DataCopy(mm1ATensor[Gm2L1Nd2NzParams.dstNzC0Stride * constInfo.dSize], this->queryRopeGm[queryRopeOffset], Gm2L1Nd2NzParams);
        }

        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非s2的第一次循环直接复用Q
        mm1A = mm1AL1Buffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        // mm1A.Wait<HardEvent::MTE1_MTE2>();
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知 // 是否可以省略
    }
    // 加载当前轮的右矩阵到L1
    mm1B = mm12Bmm2AL1Buffers.Get();
    mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用
    LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
    if constexpr (isPa) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;
        startPos.dIdx = 0; 
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = kvHeadNum;
        shape.headDim = headDim;
        shape.actHeadDim = headDim;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        PAShape ropeShape = shape;
        ropeShape.headDim = headDimRope;
        ropeShape.actHeadDim = headDimRope;
        uint32_t dstNzC0Stride = (runInfo.s2RealSize + 15) >> 4 << 4;
        LocalTensor<INPUT_T> mm1BRopeTensor = mm1BTensor[dstNzC0Stride * constInfo.dSize];
        GlobalTensor<INPUT_T> mm1BNopeGmTensor = this->keyGm;
        GlobalTensor<INPUT_T> mm1BRopeGmTensor = this->keyRopeGm;
        GmCopyInToL1HasRopePA<INPUT_T>(mm1BTensor, mm1BRopeTensor, mm1BNopeGmTensor, mm1BRopeGmTensor, blockTableGm, kvLayout, shape, ropeShape, startPos);
    } else {
        Nd2NzParams Gm2L1Nd2NzParams;
        Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
        Gm2L1Nd2NzParams.nValue = runInfo.s2RealSize; // 单个ND矩阵的实际行数，单位为元素个数
        Gm2L1Nd2NzParams.dValue = constInfo.dSize; // 单个ND矩阵的实际列数，单位为元素个数
        Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.srcDValue = constInfo.mm1Kb; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
        DataCopy(mm1BTensor, this->keyGm[runInfo.keyOffset], Gm2L1Nd2NzParams);

            // 拷贝 Krope
        if constexpr (hasRope) {
            keyRopeOffset[runInfo.taskIdMod3] = GetKeyRopeOffset(runInfo);
            Nd2NzParams Gm2L1Nd2NzParams;
            Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
            Gm2L1Nd2NzParams.nValue = runInfo.s2RealSize; // 单个ND矩阵的实际行数，单位为元素个数
            Gm2L1Nd2NzParams.dValue = constInfo.dSizeRope; // 单个ND矩阵的实际列数，单位为元素个数
            Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
            Gm2L1Nd2NzParams.srcDValue = constInfo.mm1RopeKb; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
            Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
            Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
            Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
            DataCopy(mm1BTensor[Gm2L1Nd2NzParams.dstNzC0Stride * constInfo.dSize], this->keyRopeGm[keyRopeOffset[runInfo.taskIdMod3]], Gm2L1Nd2NzParams);
        }
    }

    mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知
    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    fa_base_matmul::MMParam param = {(uint32_t)runInfo.s1RealSize,  // singleM 64
                    (uint32_t)runInfo.s2RealSize,  // singleN 128
                    (uint32_t)(constInfo.dSize + constInfo.dSizeRope), // singleK 576
                    0,    // isLeftTranspose
                    1     // isRightTranspose 
                    };
    
    // 这里base M N K不要写死
    fa_base_matmul::MatmulK<INPUT_T, INPUT_T, T, 64, 128, 128, fa_base_matmul::ABLayout::MK, fa_base_matmul::ABLayout::KN>(
        mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runParam.s2LoopEndIdx - 1)) {
        mm1A.Set<HardEvent::MTE1_MTE2>();
    }
    // mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放

    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
    
    CrossCoreWaitFlag<4, PIPE_FIX>(mm1ResIntraEvent[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<4, PIPE_FIX>(16 + mm1ResIntraEvent[runInfo.taskIdMod2]);

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) //源NZ矩阵中相邻Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    LocalTensor<T> mm1Tensor = this->bmm1ResBuf[runInfo.taskIdMod2].template Get<T>();
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(mm1Tensor, mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB

    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放

    // 下面核减同步放在主循环里
    // CrossCoreSetFlag<4, PIPE_FIX>(SYNC_C1_V1_FLAG[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
    // CrossCoreSetFlag<4, PIPE_FIX>(16 + SYNC_C1_V1_FLAG[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ProcessVec1(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam)
{
    if (runInfo.actualS2Size == 0) {
        return;
    }
    ProcessVec1Nd(runInfo, runParam);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline bool FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::SoftmaxInvalidLineCheck(
    LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isUpdateNeedCheck = false;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
    for (uint32_t i = 0; i < softmaxShapeInfo.srcM; i++) {
        T maxValue = maxUb.GetValue(i);
        uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
        if (checkValue == negativeIntScalar) {
            isUpdateNeedCheck = true;
            break;
        }
    }
    SetMaskNorm();
    ResetMask();
    return isUpdateNeedCheck;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::InvalidLineProcess(
    RunInfo<isInfer> &runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb)
{
    if (this->softMaxCheckRes) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1),
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1)};
        bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VAULE_FP32, softmaxShapeInfo);
        if (!res) {
            this->softMaxCheckRes = false;
        } else {
            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
                SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.halfS1RealSize, this->negativeFloatScalar,
                    this->positiveFloatScalar);
            }
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ProcessVec1Nd(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam)
{
    PseCopyIn<T, pseShiftType, hasPseOuter>(this->pseInQue, this->pseGm, runInfo, constInfo, pseInfo);
    LocalTensor<pseShiftType> pseUb;
    if constexpr (hasPseOuter == true) {
        pseUb = this->pseInQue.template DeQue<pseShiftType>();
    } else {
        pseUb = dummyPseTensor;
    }
    float slopes = 0.0f;
    float posShift = 0.0f;
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE || 
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            if (this->tilingData->inputParamsRegbase.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL) && 
                runInfo.boIdx != 0) {
                this->pseInfo.qStartIdx = 0;
                this->pseInfo.kvStartIdx = 0;
            }
        }
        ComputeInnerPseOffset<T, INPUT_T, hasPse>(slopes, posShift, runInfo, constInfo, pseInfo, this->pseSlope);
    }
    this->MlaAttenMaskCopyIn(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
                             this->attenMaskGmInt, runInfo, constInfo, attenMaskInfo, runParam);
    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (hasAtten == true) {
        attenMaskUb = this->attenMaskInQue[runInfo.taskIdMod2].template DeQue<uint8_t>();
    } else {
        attenMaskUb = dummyAttenMaskTensor;
    }

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
    LocalTensor<uint8_t> apiTmpBuffer;
    apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

    LocalTensor<uint8_t> dropMaskUb;
    LocalTensor<T> stage1PongTensor = this->bmm1ResBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = 0;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    if (unlikely(runInfo.s2LoopCount == 0)) {
        if (runInfo.s2RealSize == 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, false, hasRope && (dTemplateType == DTemplateType::Aligned576) && layout != LayOutTypeEnum::LAYOUT_BNSD>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                this->constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, false, hasRope && (dTemplateType == DTemplateType::Aligned576) && layout != LayOutTypeEnum::LAYOUT_BNSD>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                this->constInfo.keepProb);
        } else {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, false, hasRope && (dTemplateType == DTemplateType::Aligned576) && layout != LayOutTypeEnum::LAYOUT_BNSD>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                this->constInfo.keepProb);
        }
    } else {
         if (runInfo.s2RealSize == 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, false, hasRope && (dTemplateType == DTemplateType::Aligned576) && layout != LayOutTypeEnum::LAYOUT_BNSD>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                this->constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, false, hasRope && (dTemplateType == DTemplateType::Aligned576) && layout != LayOutTypeEnum::LAYOUT_BNSD>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                this->constInfo.keepProb);
        } else {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, false, hasRope && (dTemplateType == DTemplateType::Aligned576) && layout != LayOutTypeEnum::LAYOUT_BNSD>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                this->constInfo.keepProb);
        }
    }

    CrossCoreSetFlag<4, PIPE_V>(mm1ResIntraEvent[runInfo.taskIdMod2]);

    if constexpr (hasAtten) {
        this->attenMaskInQue[runInfo.taskIdMod2].template FreeTensor(attenMaskUb);
    }
    if constexpr (hasPseOuter) {
        this->pseInQue.template FreeTensor(pseUb);
    }

    // ===============DataCopy to L1===============
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    // 注意：这里后保存p的L1空间暂时不需要增加同步，因为保存p的结果buffer有两个，当前perload2次，L1空间v1和v2间时序已经保证同步
    uint64_t VDsize = (uint32_t)dVTemplateType;
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L1> mm2AL1Buffer = mm12Bmm2AL1Buffers.Get();
    LocalTensor<INPUT_T> mm2AL1Tensor = mm2AL1Buffer.GetTensor<INPUT_T>(s2BaseSize * VDsize); // L1P与L1K_rope复用

    SetFlag<HardEvent::V_MTE3>(this->UbToL1Event);
    WaitFlag<HardEvent::V_MTE3>(this->UbToL1Event);

    uint32_t vec1ScmBlockTrue = s1BaseSize * (16 / sizeof(INPUT_T));

    if (likely(runInfo.halfS1RealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1ScmBlockTrue], stage1CastTensor, 
                {s2BaseSize / 16, (uint16_t)runInfo.halfS1RealSize, 
                (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
    }
    // mm2AL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 释放
    // 核间同步放在主循环里
    // CrossCoreSetFlag<4, PIPE_MTE3>(SYNC_V1_C2_FLAG[runInfo.taskIdMod2]); // mte3将结果搬运到L1，设置SYNC_V1_C2_FLAG
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    // =======================================================
    if (runInfo.s2LoopCount != 0) {
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            this->InvalidLineProcess(runInfo, sumUb, maxUb);
        }
    }
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        SoftmaxDataCopyOut(runInfo);
        if constexpr (isFd) {
            ComputeLogSumExpAndCopyToGm(sumUb, maxUb, runInfo);
            return;
        }
        SoftmaxLseCopyOut(sumUb, maxUb, runInfo, runParam);
    }
    return;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::MlaAttenMaskCopyIn(
    TQue<QuePosition::VECIN, 1> &attenMaskInQue, TQue<QuePosition::VECIN, 1> &attenMaskInQuePre, GlobalTensor<uint8_t> &srcTensor,
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, AttenMaskInfo &attenMaskInfo, RunParamStr<isInfer>& runParam)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
        int64_t maskOffset = ComputeAttenMaskOffset<hasAtten, enableKVPrefix, isFd, hasRope, isInfer, dTemplateType>(runInfo, constInfo, attenMaskInfo);
        this->MlaBoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
            attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo, runInfo, runParam);
        attenMaskInQue.template EnQue(attenMaskUb);
        return;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::MlaBoolCopyInRegbase(LocalTensor<uint8_t> &dstTensor, 
    GlobalTensor<uint8_t> &srcTensor, int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t s2BaseSize, 
    ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam)
{
    if (s1Size == 0 || s2Size == 0) {
        return;
    }

    if (totalS2Size % blockBytes != 0) {
        return;
    }

    if constexpr (isInfer == false) {
        return;
    }

    uint32_t neededSouterSize = s1Size;
    uint32_t s2BlockLenAlign = (s2Size + blockBytesU8 - 1) / blockBytesU8;
    DataCopyExtParams intriParams;
    intriParams.blockCount = s1Size;
    intriParams.blockLen = s2Size;
    intriParams.srcStride = totalS2Size - s2Size;
    intriParams.dstStride = s2BaseSize / blockBytesU8 - s2BlockLenAlign;
    DataCopyPadExtParams<uint8_t> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.paddingValue = 1;
    padParams.rightPadding = 0;
    if ((hasRope && (dTemplateType == DTemplateType::Aligned576)) && 
        (constInfo.layoutType != (uint32_t)LayOutTypeEnum::LAYOUT_BNSD))
    {
        intriParams.blockCount = 1;
        DataCopyPad(dstTensor, srcTensor[srcOffset], intriParams, padParams);
        SetFlag<HardEvent::MTE2_V>(mte2ToV[0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV[0]);
        return;
    } else if ((hasRope && (dTemplateType == DTemplateType::Aligned576)) && 
               (constInfo.layoutType == (uint32_t)LayOutTypeEnum::LAYOUT_BNSD)) {
        int64_t s1OfMla = runParam.actualS1Size / constInfo.gSize;
        int64_t firstS1Start = runParam.sOuterOffset % s1OfMla;
        intriParams.blockCount = (s1OfMla - firstS1Start) < neededSouterSize ? 
            (s1OfMla - firstS1Start) : neededSouterSize;
        DataCopyPad(dstTensor, srcTensor[srcOffset + firstS1Start * totalS2Size], intriParams, padParams);
        if (firstS1Start != 0 && (s1OfMla - firstS1Start) < neededSouterSize) {
            intriParams.blockCount = firstS1Start < (neededSouterSize - s1OfMla + firstS1Start) ?
                firstS1Start : (neededSouterSize - s1OfMla + firstS1Start);
            DataCopyPad(dstTensor[(s1OfMla - firstS1Start) * s2BaseSize], 
                srcTensor[srcOffset], intriParams, padParams);
        }

        SetFlag<HardEvent::MTE2_V>(mte2ToV[0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV[0]);

        for (int64_t i = 1; (i + 1) * s1OfMla <= neededSouterSize; i++) {
            DataCopy(dstTensor[s1OfMla * i * s2BaseSize], dstTensor, s1OfMla * s2BaseSize);
        }
        if (neededSouterSize > s1OfMla && neededSouterSize % s1OfMla != 0) {
            DataCopy(dstTensor[s1OfMla * (neededSouterSize / s1OfMla) * s2BaseSize], dstTensor,
                (neededSouterSize % s1OfMla) * s2BaseSize);
        }
        return;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::WaitBmm2Result(RunInfo<isInfer> &runInfo)
{
    // 放在主循环里
    // CrossCoreWaitFlag<4, PIPE_V>(SYNC_C2_V2_FLAG[runInfo.taskIdMod2]); // 等待bmm2完成/等待SYNC_C2_V2_FLAG置位
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::IterateBmm2(
    RunInfo<isInfer> &runInfo)
{
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L1> mm2AB;
    if constexpr (mm2LeftFromUB) {
        // 核间同步放在主循环里
        // CrossCoreWaitFlag<4, PIPE_MTE1>(SYNC_V1_C2_FLAG[runInfo.taskIdMod2]);
        // CrossCoreWaiFlag<4, PIPE_MTE1>(16 + SYNC_V1_C2_FLAG[runInfo.taskIdMod2]);
    } else {
        // 从gm输入，待实现
    }

    // 加载当前轮的右矩阵到L1
    if (mm2RightStillInL1) {
        mm2AB = mm12Bmm2AL1Buffers.GetReused();
        // mm2AB.Wait<HardEvent::MTE1_MTE2>(); // 占用?
    } else {
        // 从gm输入，待实现
    }

    uint64_t VDsize = (uint32_t)dVTemplateType;
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    fa_base_matmul::MMParam param = {(uint32_t)s1BaseSize,  // singleM 64
                    (uint32_t)constInfo.dSizeV,  // singleN 512
                    (uint32_t)runInfo.s2RealSize,  // singleK 128
                    false,    // isLeftTranspose
                    false     // isRightTranspose 
                    };
    // 这里base M N K不要写死
    fa_base_matmul::MatmulN<INPUT_T, INPUT_T, T, 64, 128, 128, fa_base_matmul::ABLayout::MK, fa_base_matmul::ABLayout::KN>(
                                mm2AB.GetTensor<INPUT_T>(s2BaseSize * VDsize), 
                                mm2AB.GetTensor<INPUT_T>(),
                                mmL0ABuffers,
                                mmL0BBuffers,
                                mm2ResL0C.GetTensor<T>(),
                                param);
    mm2AB.Set<HardEvent::MTE1_MTE2>(); // 释放

    mm2ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
    
    CrossCoreWaitFlag<4, PIPE_FIX>(mm2ResIntraEvent[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<4, PIPE_FIX>(16 + mm2ResIntraEvent[runInfo.taskIdMod2]);

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = constInfo.dSizeV; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    fixpipeParams.mSize = s1BaseSize; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) //源NZ矩阵中相邻Z排布的起始地址偏移
    fixpipeParams.dstStride = (fixpipeParams.nSize + 15) >> 4 << 4; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    LocalTensor<T> mm2Tensor = this->bmm2ResBuf[runInfo.taskIdMod2].template Get<T>();
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(mm2Tensor, mm2ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB

    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
}

CHILD_SPEC_TEMPLATE
__aicore__ inline int64_t FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ComputeOffsetForSoftmax(
    RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx)
{
    return vec2S1Idx * runInfo.vec2S1BaseSize;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ProcessVec2S2Split(RunInfo<isInfer> &runInfo) {
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    if (unlikely(runInfo.vec2S1RealSize == 0)) {
        return;
    }
    LocalTensor<T> bmm2Ub = this->bmm2ResBuf[runInfo.taskIdMod2].template Get<T>();
    LocalTensor<T> vec2ResUb = this->stage2OutQue[0].template AllocTensor<T>();
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;

    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, bmm2Ub, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        float deSCalePreVValue = 1.0f;
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == 1)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                    vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, deSCalePreVValue);
            } else {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, deSCalePreVValue);
            }
        } else {
            if (unlikely(runInfo.s2LoopCount == 1)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                    vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, deSCalePreVValue);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, deSCalePreVValue);
            }
        }
    }
    CrossCoreSetFlag<4, PIPE_V>(mm2ResIntraEvent[runInfo.taskIdMod2]);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, (uint16_t)dTemplateAlign64, 1.0);
        }

        this->stage2OutQue[0].template EnQue(vec2ResUb);
        this->stage2OutQue[0].template DeQue<OUTPUT_T>();
        if constexpr (isFd) {
            Bmm2FDOut(runInfo, vec2ResUb, vec2CalcSize);
        } else {
            Bmm2DataCopyOut(runInfo, vec2ResUb, 0, vec2CalcSize);
        }
    }
    this->stage2OutQue[0].template FreeTensor(vec2ResUb);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ProcessVec2(RunInfo<isInfer> &runInfo) 
{
    if (runInfo.actualS2Size == 0) {
        return;
    }

    ProcessVec2S2Split(runInfo);
}


CHILD_SPEC_TEMPLATE
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb,
    int64_t vec2S1Idx, RunInfo<isInfer> &runInfo)
{
    if constexpr (isInfer && hasAtten) {
        if (this->attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return;
        }
        int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
        LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        bool isRowInvalidNeedUpdate = false;
        for (uint32_t i = 0; i < runInfo.vec2S1RealSize; i++) {
            float maxValue = maxTensor.GetValue(i);
            uint32_t checkValue = *(uint32_t*)&maxValue;
            if (checkValue == NEGATIVE_MIN_VAULE_FP32) {
                isRowInvalidNeedUpdate = true;
                break;
            }
        }
        if (isRowInvalidNeedUpdate) {
            if constexpr (!POST_QUANT) {
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor,  runInfo.vec2S1RealSize, constInfo.dSizeV, static_cast<uint32_t>(dVTemplateType));
            } else {
                uint32_t dStride = CeilDivision(static_cast<uint32_t>(static_cast<uint32_t>(dVTemplateType)), sizeof(float));
                uint16_t dSize = CeilDivision(constInfo.dSizeV, sizeof(float)); // w8后量化后的处理长度
                RowInvalidUpdateVF<float>(*((LocalTensor<float>*)&vec2ResUb), maxTensor, runInfo.vec2S1RealSize, dSize, dStride);
            }
        }
    }
}

CHILD_SPEC_TEMPLATE
template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::PostQuantPerChnl(LocalTensor<OUTPUT_T>& attenOut,
    LocalTensor<VEC2_RES_T>& vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<POSTQUANT_PARAMS_T> copyInPadParams;
    copyInParams.blockCount = gSplitSize;
    copyInParams.blockLen = this->constInfo.dSizeV * sizeof(POSTQUANT_PARAMS_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = ((int64_t)dVTemplateType - this->constInfo.dSizeV) / 8; // 8 for align factor 

    LocalTensor<POSTQUANT_PARAMS_T> postQuantScaleUb = this->postQuantScaleQue.template AllocTensor<POSTQUANT_PARAMS_T>();
    DataCopyPad(postQuantScaleUb, postQuantScaleGm[perChannelQuantOffset], copyInParams, copyInPadParams);

    this->postQuantScaleQue.template EnQue(postQuantScaleUb);
    this->postQuantScaleQue.template DeQue<POSTQUANT_PARAMS_T>();
    if (this->constInfo.isPostQuantOffsetExist) {
        LocalTensor<POSTQUANT_PARAMS_T> postQuantOffsetUb = this->postQuantOffsetQue.template AllocTensor<POSTQUANT_PARAMS_T>();
        DataCopyPad(postQuantOffsetUb, postQuantOffsetGm[perChannelQuantOffset], copyInParams, copyInPadParams);
        this->postQuantOffsetQue.template EnQue(postQuantOffsetUb);
        this->postQuantOffsetQue.template DeQue<POSTQUANT_PARAMS_T>();
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, postQuantOffsetUb, gSplitSize, s1RowCount, this->constInfo.dSizeV, (uint16_t)dVTemplateType);
        this->postQuantOffsetQue.FreeTensor(postQuantOffsetUb);
    } else {
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, gSplitSize, s1RowCount, this->constInfo.dSizeV, (uint16_t)dVTemplateType);
    }
    this->postQuantScaleQue.FreeTensor(postQuantScaleUb);

}

CHILD_SPEC_TEMPLATE
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::PostQuant(RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx)
{
    if (this->constInfo.isPostQuantPerChnl) {
        uint64_t perChannelQuantOffset = runInfo.n2oIdx * this->constInfo.gDv  + vec2S1Idx * runInfo.vec2S1BaseSize * this->constInfo.dSizeV;
        uint32_t quantSplitOffset;
        for (uint32_t startRow = 0; startRow < runInfo.vec2S1RealSize; startRow++) {
            uint32_t splitOffset = startRow * this->constInfo.dSizeV;     
            if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
                quantSplitOffset = ((startRow + runInfo.sOuterOffset) / this->constInfo.s1Size) * this->constInfo.dSizeV;
            } else {
                quantSplitOffset = ((startRow + runInfo.sOuterOffset) % this->constInfo.gSize) * this->constInfo.dSizeV;
            }
            if (this->constInfo.isPostQuantBF16) {
                PostQuantPerChnl(attenOut, vec2ResUb, perChannelQuantOffset + quantSplitOffset, 1U, 1U, splitOffset, postQuantScaleBf16Gm, postQuantOffsetBf16Gm); // 逐行量化
            } else {
                PostQuantPerChnl(attenOut, vec2ResUb, perChannelQuantOffset + quantSplitOffset, 1U, 1U, splitOffset, postQuantScaleGm, postQuantOffsetGm);  // 逐行量化
            }
        }
    } else {     
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(attenOut, vec2ResUb, this->constInfo.postQuantScaleValue, this->constInfo.postQuantOffsetValue, runInfo.vec2S1RealSize, this->constInfo.dSizeV, (uint16_t)dVTemplateType);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::FDPostQuant(LocalTensor<OUTPUT_T> &attenOut, LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount)
{
    if (this->constInfo.isPostQuantPerChnl) {
        if (this->constInfo.isPostQuantBF16) {
            PostQuantPerChnl(attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount, 1U, 0U, postQuantScaleBf16Gm, postQuantOffsetBf16Gm); // q_s = 1
        } else {
            PostQuantPerChnl(attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount, 1U, 0U, postQuantScaleGm, postQuantOffsetGm); // q_s = 1
        }
    } else {
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(attenOut, accumOutLocal, this->constInfo.postQuantScaleValue, this->constInfo.postQuantOffsetValue, dealRowCount, 1U, this->constInfo.dSizeV, (uint16_t)dVTemplateType);
    }
}

CHILD_SPEC_TEMPLATE
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::Bmm2DataCopyOut(
    RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize) 
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (splitD) {
        dSizeAligned64 = this->dBasicBlock;
    }
    if constexpr (!IsSameType<INPUT_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
                InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, runInfo.vec2S1RealSize,
                    dSizeAligned64, this->negativeFloatScalar, 0.0);
            }
        }
        RowInvalid(vec2ResUb, vec2S1Idx, runInfo);
        if constexpr (!POST_QUANT) {
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        } else {
            PostQuant(runInfo, attenOut, vec2ResUb, vec2S1Idx);
        }
        stage2OutQue[0].EnQue(attenOut);
        stage2OutQue[0].DeQue<OUTPUT_T>();
    } else {
        stage2OutQue[runInfo.taskIdMod2].EnQue(vec2ResUb);
        stage2OutQue[runInfo.taskIdMod2].template DeQue<OUTPUT_T>();
        attenOut = vec2ResUb;
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    if constexpr (IsSameType<INPUT_T, float>::value) {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 3;
    } else {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    }
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    int64_t attenOutOffset = constInfo.dSizeV;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        attenOutOffset = constInfo.n2GDv;
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            attenOutOffset = constInfo.n2GDv;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            attenOutOffset = constInfo.bN2GDv;
        }
        if constexpr (isInfer) {
            if (constInfo.isBSNDOut == 1) {
                attenOutOffset = constInfo.n2GDv;
            }
        }
    }

    dataCopyParams.blockCount = runInfo.vec2S1RealSize;
    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset], 
            attenOut, dataCopyParams);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::Bmm2FDOut(
    RunInfo<isInfer> &runInfo, LocalTensor<T> &vec2ResUb, int64_t vec2CalcSize) 
{
    LocalTensor<T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;

    stage2OutQue[runInfo.taskIdMod2].EnQue(vec2ResUb);
    stage2OutQue[runInfo.taskIdMod2].template DeQue<OUTPUT_T>();
    attenOut = vec2ResUb;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = runInfo.firstHalfS1RealSize;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (BYTE_BLOCK_32B / sizeof(T));
    dataCopyParams.dstStride = 0;

    uint32_t mStart = constInfo.subBlockIdx * runInfo.halfS1RealSize;
    size_t base = (runInfo.boIdx * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV +
                  runInfo.n2oIdx * constInfo.gSize * constInfo.dSizeV) * splitKVNum + mStart * constInfo.dSizeV;
    // {B n2 split g d}
    // base = (bIdx * qHeadNum * headDim + n2Idx * gSize * headDim) * splitKVNum
    // offset = base + s2Idx * gSize * actualColumnCount + startRow * actualColumnCount
    DataCopyPad(this->accumOutGm[base + runInfo.flashDecodeS2Idx * constInfo.gSize * constInfo.dSizeV],
                attenOut, dataCopyParams);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    RunInfo<isInfer> &runInfo) 
{
    if constexpr (isInfer) {
        return;
    }
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }
    int64_t bOffset;
    int64_t n2Offset;
    int64_t gOffset;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = constInfo.n2G * runInfo.s1SizeAcc;
        n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size;
        gOffset = runInfo.goIdx * runInfo.actualS1Size;
    } else {
        bOffset = runInfo.boIdx * constInfo.n2Size * constInfo.gS1;
        n2Offset = runInfo.n2oIdx * constInfo.gS1;
        gOffset = runInfo.goIdx * constInfo.s1Size;
    }
    int64_t s1Offset = (runInfo.s1oIdx * s1BaseSize +
        constInfo.subBlockIdx * runInfo.firstHalfS1RealSize);
    int64_t gmOffset = (bOffset + n2Offset + gOffset + s1Offset) * fp32BaseSize;
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;

    // Copy sum to gm
    LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> sumOutTensor = sumBrdcst.AllocTensor<float>();
    fa::BroadcastMaxSum(sumOutTensor, sumTensor, runInfo.halfS1RealSize);
    sumBrdcst.EnQue(sumOutTensor);
    sumBrdcst.DeQue<float>();
    DataCopy(this->softmaxSumGm[gmOffset], sumOutTensor, calculateSize);
    this->sumBrdcst.template FreeTensor(sumOutTensor);

    // Copy max to gm
    LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    if constexpr (!IsSameType<INPUT_T, float>::value || !containAllOptionalInput) {
        LocalTensor<float> maxOutTensor = maxBrdcst.AllocTensor<float>();
        fa::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        maxBrdcst.EnQue(maxOutTensor);
        maxBrdcst.DeQue<float>();
        DataCopy(this->softmaxMaxGm[gmOffset], maxOutTensor, calculateSize);
        this->maxBrdcst.template FreeTensor(maxOutTensor);
    } else {
        LocalTensor<float> maxOutTensor = sumBrdcst.AllocTensor<float>();
        fa::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        sumBrdcst.EnQue(maxOutTensor);
        sumBrdcst.DeQue<float>();
        DataCopy(this->softmaxMaxGm[gmOffset], maxOutTensor, calculateSize);
        this->sumBrdcst.template FreeTensor(maxOutTensor);
    }
}
CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::SoftmaxLseCopyOut(
    LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp, 
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam) 
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

    if constexpr (isInfer) {
        if (!constInfo.isSoftmaxLseEnable) {
            return;
        }
        LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.halfS1RealSize);
        softmaxLseQueue.template EnQue(lseUb);
        softmaxLseQueue.DeQue<float>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = runInfo.halfS1RealSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH
            && hasRope && (dTemplateType == DTemplateType::Aligned576)) {
                intriParams1.dstStride = sizeof(float) * (constInfo.s1Size - 1);
        }
        DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParams1);

        softmaxLseQueue.FreeTensor(lseUb);
    } else {
        return;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionKvsameBN2GS1S2<CHILD_SPEC_TEMPLATE_ARGS>::ComputeLogSumExpAndCopyToGm(
    LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp, 
    RunInfo<isInfer> &runInfo) 
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

    int64_t bOffset;
    int64_t n2Offset;
    int64_t gOffset;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = constInfo.n2G * runInfo.s1SizeAcc;
        n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size;
        gOffset = runInfo.goIdx * runInfo.actualS1Size;
    } else {
        bOffset = runInfo.boIdx * this->constInfo.n2Size * this->constInfo.gS1;
        n2Offset = runInfo.n2oIdx * constInfo.gS1;
        gOffset = runInfo.goIdx * constInfo.s1Size;
    }
    int64_t s1Offset = (runInfo.s1oIdx * s1BaseSize +
        constInfo.subBlockIdx * runInfo.firstHalfS1RealSize);
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;
    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t gmOffset = runInfo.boIdx * constInfo.n2Size * splitKVNum * constInfo.gSize * FP32_ONE_BLOCK_SIZE + 
                        runInfo.n2oIdx * splitKVNum * constInfo.gSize * FP32_ONE_BLOCK_SIZE + runInfo.flashDecodeS2Idx * constInfo.gSize * FP32_ONE_BLOCK_SIZE + mStart * FP32_ONE_BLOCK_SIZE;
    // Copy sum to gm
    LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> sumOutTensor =sumBrdcst.AllocTensor<float>();
    fa::BroadcastMaxSum(sumOutTensor, sumTensor, runInfo.halfS1RealSize);
    sumBrdcst.EnQue(sumOutTensor);
    sumBrdcst.DeQue<float>();
    DataCopy(this->softmaxFDSumGm[gmOffset], sumOutTensor, calculateSize);
    this->sumBrdcst.template FreeTensor(sumOutTensor);

    // Copy max to gm
    LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    if constexpr (!IsSameType<INPUT_T, float>::value || !containAllOptionalInput) {
        LocalTensor<float> maxOutTensor = maxBrdcst.AllocTensor<float>();
        fa::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        maxBrdcst.EnQue(maxOutTensor);
        maxBrdcst.DeQue<float>();
        DataCopy(this->softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        this->maxBrdcst.template FreeTensor(maxOutTensor);
    } else {
        LocalTensor<float> maxOutTensor = sumBrdcst.AllocTensor<float>();
        fa::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        sumBrdcst.EnQue(maxOutTensor);
        sumBrdcst.DeQue<float>();
        DataCopy(this->softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        this->sumBrdcst.template FreeTensor(maxOutTensor);
    }
}
#endif