/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file flash_attention_score_antiquant_baseapi.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_ANTIQUANT_BASEAPI_H_
#define FLASH_ATTENTION_SCORE_ANTIQUANT_BASEAPI_H_

const int CV_L1_EVENT[2] = {0, 1};
const int CV_MM1RES_EVENT[2] = {2, 3};
const int CV_MM2RES_EVENT[2] = {4, 5};

const int VC_L1_EVENT[2] = {6, 7};
const int VC_MM1RES_EVENT[2] = {8, 9};
const int VC_MM2RES_EVENT[2] = {10, 15};

enum class AntiquantTypeEnum : uint8_t {
    PER_CHANNEL = 0, // enable per-channel antiquant mode，include per-tensor
    PER_TOKEN = 1,  // enable per-token antiquant mode
    K_PER_CHANNEL_V_PER_TOKEN = 2, // enable split antiquant mode, k per-channel and v per-token
    PER_TOKEN_HEAD = 3, // enable both per-token and per-head antiquant mode
    PER_TOKEN_PAGE_ATTENTION = 4, // enable per-token antiquant mode, and enable PA for memory management
    PER_TOKEN_HEAD_PAGE_ATTENTION = 5, // enable both per-token and per-head antiquant mode, and enable PA for memory management
};

#include "util_regbase.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/CopyInL1.h"
#include "flash_attention_score_common_regbase.h"
#include "infer_flash_attention_comm.h"
#include "kernel_operator_list_tensor_intf.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flash_decode.h"
#include "flash_attention_score_antiquant_processor.h"
#include "infer_flash_attention_kvcache.h"
#include "infer_flash_attention_sparse.h"
#include "kernel_operator.h"
#include "attenmask.h"
#include "pse.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using matmul::MatmulType;
using namespace optiling;

namespace BaseApi {
__aicore__ constexpr uint16_t Align64FuncAntiquant(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

template <typename T>
__aicore__ inline T ALIGNAntiquant(T num, T rnd) {
  return ((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd));
}
static constexpr uint32_t FA_BYTE_BLOCK_ANTIQUANT = 32;
constexpr uint64_t BYTE_BLOCK_ANTIQUANT = 32UL;
template<typename Q_T, typename KV_T, typename T, typename OUTPUT_T, ImplModeEnum implMode = ImplModeEnum::AA_HIGH_PRECISION, 
    LayOutTypeEnum layout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    AntiquantTypeEnum antiquantMode = AntiquantTypeEnum::PER_CHANNEL, bool hasAtten = false, bool hasDrop = false, bool hasRope = false,
    bool isInfer = false, bool isPa = false, bool isFd = false>
class FlashAttentionScoreAntiquantKernel {
public:
    using INPUT_T = KV_T;
    __aicore__ inline FlashAttentionScoreAntiquantKernel() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, 
        __gm__ uint8_t *pse, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
        __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *workspace, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitQuant(__gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
        __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
        __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset);
    __aicore__ inline void Process();

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool KVFP4 = (IsSameType<KV_T, fp4x2_e1m2_t>::value || IsSameType<KV_T, fp4x2_e2m1_t>::value);
    static constexpr bool ANTIQUANT_PER_GROUP = (IsSameType<KV_T, fp4x2_e1m2_t>::value
        || IsSameType<KV_T, fp4x2_e2m1_t>::value);
    static constexpr bool PAGE_ATTENTION_ANTIQUANT = (antiquantMode == AntiquantTypeEnum::PER_TOKEN_PAGE_ATTENTION ||
        antiquantMode == AntiquantTypeEnum::PER_TOKEN_HEAD_PAGE_ATTENTION);
    static constexpr bool KEY_ANTIQUANT_PER_TOKEN = (antiquantMode == AntiquantTypeEnum::PER_TOKEN
        || antiquantMode == AntiquantTypeEnum::PER_TOKEN_PAGE_ATTENTION 
        || antiquantMode == AntiquantTypeEnum::PER_TOKEN_HEAD_PAGE_ATTENTION);
    static constexpr bool VALUE_ANTIQUANT_PER_TOKEN = (antiquantMode == AntiquantTypeEnum::PER_TOKEN 
        || antiquantMode == AntiquantTypeEnum::PER_TOKEN_PAGE_ATTENTION
        || antiquantMode == AntiquantTypeEnum::PER_TOKEN_HEAD_PAGE_ATTENTION 
        || antiquantMode == AntiquantTypeEnum::K_PER_CHANNEL_V_PER_TOKEN);
    using KEY_ANTIQ_PARAMS_T = typename std::conditional<KEY_ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    using VALUE_ANTIQ_PARAMS_T = typename std::conditional<VALUE_ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;

protected:
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1ScmBlockTrue = s1BaseSize * (16 / 2);
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64FuncAntiquant((uint16_t)dVTemplateType);
    static constexpr bool hasPse = pseMode != PseTypeEnum::PSE_NONE_TYPE;
    static constexpr bool hasPseOuter = (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) ||
        (pseMode == PseTypeEnum::PSE_OUTER_MUL_ADD_TYPE);
    static constexpr bool containAllOptionalInput = hasPse && hasAtten && hasDrop;
    /*相关内存大小信息*/
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dBaseSizediv4 = dBaseSize / 4;
    static constexpr uint32_t s2BaseSizediv4 = s2BaseSize / 4;
    static constexpr uint32_t kvInputSize = dTemplateAlign64 * s2BaseSize * 1 *0.5 * 0.5;
    static constexpr uint32_t kvOutSize = dTemplateAlign64 * s2BaseSize * 2 *0.25 + 1024;
    static constexpr uint32_t pseInputSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t attenMaskSize = s1BaseSize / CV_RATIO * s2BaseSize * 1;
    static constexpr uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t kvAntiquantResSize = s2BaseSize * dTemplateAlign64 * sizeof(Q_T);
    static constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t stage1OutQueSize = (s1BaseSize / CV_RATIO + 1) * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t stage2OutQueSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    static constexpr bool useDn = false;

    uint32_t antiqSeqSize = 0;
    bool antiquantPerTensorFlag = true;
    bool antiquantPerHeadFlag = false;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K_ANTIQUANT = 4096;
    AntiquantTaskParamBaseAPI taskParam;

    TPipe *pipe;
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    __gm__ uint8_t *currentKey;
    __gm__ uint8_t *currentValue;
    __gm__ uint8_t *blocktablePtr;
    GlobalTensor<int32_t> blockTableGm;
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    KVLAYOUT kvLayout;
    uint32_t paKvShapeType = 0;
    bool isBeforeHalf;
    
    bool antiqOffsetExistFlag;
    int32_t aicIdx;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;

    //GM变量
    GlobalTensor<KV_T> value;
    GlobalTensor<KV_T> key;
    GlobalTensor<Q_T> query;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    __gm__ uint8_t *pseSlope;
    using pseGmType = typename std::conditional<hasPse, GlobalTensor<OUTPUT_T>, int8_t>::type;
    pseGmType pseGm;
    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    attenMaskGmType attenMaskGmInt;
    GlobalTensor<float> softmaxLseGm;
    using FDGmType = typename std::conditional<isFd, GlobalTensor<float>, int8_t>::type;
    FDGmType accumOutGm;
    FDGmType softmaxFDMaxGm;
    FDGmType softmaxFDSumGm;
    LocalTensor<OUTPUT_T> dummyPseTensor;
    LocalTensor<uint8_t> dummyAttenMaskTensor;
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    /*伪量化参数*/
    GlobalTensor<KEY_ANTIQ_PARAMS_T> keyAntiquantOffsetGm;
    GlobalTensor<KEY_ANTIQ_PARAMS_T> keyAntiqScaleGm;
    GlobalTensor<VALUE_ANTIQ_PARAMS_T> valueAntiquantOffsetGm;
    GlobalTensor<VALUE_ANTIQ_PARAMS_T> valueAntiqScaleGm;

    /*GM信息*/
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;

    /* =====================V侧UB变量==================== */
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TQue<QuePosition::VECIN, 1> pseInQue;
    TQue<QuePosition::VECOUT, 1> stage2OutQue[2];
    TBuf<> softmaxMaxBuf[3];
    TBuf<> softmaxSumBuf[3];
    TBuf<> softmaxExpBuf[3];
    TBuf<> vselrIndexesBuf[4];
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue;
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    TQue<QuePosition::VECIN, 1> kvInputQue;
    TQue<QuePosition::VECOUT, 1> kvOutputQue;
    TQue<QuePosition::VECIN, 1> keyAntiqScaleInputQue;
    TQue<QuePosition::VECIN, 1> keyAntiqOffsetInputQue;
    TQue<QuePosition::VECIN, 1> valueAntiqScaleInputQue;
    TQue<QuePosition::VECIN, 1> valueAntiqOffsetInputQue;
    TBuf<> kvAntiqMxScaleRes;  // for w4

    /* =====================核间通道==================== */
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::L1> mm2AL1Buffers;
    BuffersPolicyDB<BufferType::L1> kvAntiquantRes;
    TBuf<TPosition::VECIN> bmm1ResBuf[2];
    TBuf<TPosition::VECIN> bmm2ResBuf[2];
    /* =====================Cube相关内存==================== */
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    BuffersPolicyDB<BufferType::L1> mm1AL1Buffers;
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;

    /* =================初始化后不变的信息================= */
    ConstInfo<isInfer, hasRope> constInfo;
    PseInfo pseInfo;
    AttenMaskInfo attenMaskInfo;
    T negativeFloatScalar;
    T positiveFloatScalar;
    event_t UbToL1Event;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    AntiquantProcessorBaseAPI<ANTIQUANT_PROCESSOR_ARGS, KEY_ANTIQUANT_PER_TOKEN> keyAntiquantProcessor;
    AntiquantProcessorBaseAPI<ANTIQUANT_PROCESSOR_ARGS, VALUE_ANTIQUANT_PER_TOKEN> valueAntiquantProcessor;
    static constexpr uint32_t bufferSizeByte32K = 32768;
    static constexpr uint32_t gSplitMax = 16;
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key,
        __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
        __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace);
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void InitAntiquantBuffer();
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void setConstAntiTaskParam();
    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline void InitLseOutputSingleCore();
    __aicore__ inline bool IsLastBN(uint32_t bnStartIdx, uint32_t bnEndIdx);
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndx, int64_t gS1Index, RunParamStr<isInfer>& runParam);
    __aicore__ inline void SetRunInfo(RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam, int64_t taskId, int64_t s2LoopCount,
        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void IterateBmm1(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, const int64_t &subTaskId);
    __aicore__ inline void AntiquantKey(RunInfo<isInfer> &runInfo, int64_t &subTaskId, bool &first, RunParamStr<isInfer> &runParam);
    __aicore__ inline void AntiquantValue(RunInfo<isInfer> &runInfo, int64_t &subTaskId, bool &first, RunParamStr<isInfer> &runParam);
    __aicore__ inline void ProcessVec1(RunInfo<isInfer> &runInfo);
    __aicore__ inline void ProcessVec1Nd(RunInfo<isInfer> &runInfo);
    __aicore__ inline void ProcessVec2(RunInfo<isInfer> &runInfo);
    __aicore__ inline void ProcessVec2S2Split(RunInfo<isInfer> &runInfo);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp, RunInfo<isInfer> &runInfo);
    template<typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template<typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, RunInfo<isInfer> &runInfo);
    __aicore__ inline int64_t ComputeOffsetForSoftmax(RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx);
    __aicore__ inline void IterateBmm2(const int64_t &subTaskId, RunInfo<isInfer> &runInfo);
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);
    /*FD相关*/
    __aicore__ inline void Bmm2FDOut( RunInfo<isInfer> &runInfo, LocalTensor<T> &vec2ResUb, int64_t vec2CalcSize);
    __aicore__ inline void InitFDBuffers();
    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void GetActualSeqLenKV(int64_t boIdx, int64_t &actualSeqKvLen);
    __aicore__ inline void CombineSplitKVRes(uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, uint32_t splitSize, uint64_t lseOffset);
    __aicore__ inline void CopyLseIn(uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ReduceFDDataCopyOut(uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
        uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo);
    __aicore__ inline void InvalidLineProcess(RunInfo<isInfer> &runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb);
    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo);
};

CHILD_SPEC_TEMPLATE_ANTI 
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::Init(__gm__ uint8_t *query, 
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    this->tilingData = tiling;
    this->pipe = tPipe;
    this->ComputeConstexpr();
    this->InitInput(query, key, value, pse, attenMask, actualSeqLengths, actualSeqLengthsKv, blockTable, queryPaddingSize, kvPaddingSize,
        softmaxLse, attentionOut, workspace);
    this->InitBuffer();
    if ASCEND_IS_AIV {
        if (this->tilingData->initOutputParams.needInit == 1) {
            InitOutputSingleCore();
            if (constInfo.isSoftmaxLseEnable) {
                SyncAll();
                InitLseOutputSingleCore();
            }
        }
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    constInfo.n2Size = inputParamsRegbase.n2Size;
    constInfo.s1Size = inputParamsRegbase.s1Size;
    constInfo.s2Size = inputParamsRegbase.s2Size;
    constInfo.dSize = inputParamsRegbase.dSize;
    constInfo.dSizeV = inputParamsRegbase.dSizeV;
    if constexpr (hasRope) {
        constInfo.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        constInfo.dSizeRope = 0;
    }
    constInfo.gSize = inputParamsRegbase.gSize;

    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    constInfo.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    constInfo.s1D = constInfo.s1Size * constInfo.dSize;
    constInfo.s2D = constInfo.s2Size * constInfo.dSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2D = constInfo.n2Size * constInfo.dSize;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.bN2D = inputParamsRegbase.bSize * constInfo.n2D;
    constInfo.gS1D = constInfo.gSize * constInfo.s1D;
    constInfo.n2S2D = constInfo.n2Size * constInfo.s2D;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;
    constInfo.bN2GD = inputParamsRegbase.bSize * constInfo.n2GD;
    constInfo.n2GS1D = constInfo.n2Size * constInfo.gS1D;
    constInfo.s2BaseN2D = s2BaseSize * constInfo.n2D;
    constInfo.s1Dv = constInfo.s1D;
    constInfo.s2Dv = constInfo.s2D;
    constInfo.n2Dv = constInfo.n2D;
    constInfo.gDv = constInfo.gD;
    constInfo.gS1Dv = constInfo.gS1D;
    constInfo.n2S2Dv = constInfo.n2S2D;
    constInfo.n2GDv = constInfo.n2GD;
    constInfo.s2BaseN2Dv = constInfo.s2BaseN2D;
    constInfo.n2GS1Dv = constInfo.n2GS1D;
    constInfo.layoutType = inputParamsRegbase.layoutType;
    if ASCEND_IS_AIV {
        UbToL1Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    }
    if constexpr (hasRope) {
        constInfo.s1DR = constInfo.s1Size * constInfo.dSizeRope;
        constInfo.s2DR = constInfo.s2Size * constInfo.dSizeRope;
        constInfo.gDR = constInfo.gSize * constInfo.dSizeRope;
        constInfo.n2DR = constInfo.n2Size * constInfo.dSizeRope;
        constInfo.bN2DR = inputParamsRegbase.bSize * constInfo.n2DR;
        constInfo.gS1DR = constInfo.gSize * constInfo.s1DR;
        constInfo.n2S2DR = constInfo.n2Size * constInfo.s2DR;
        constInfo.n2GDR = constInfo.n2Size * constInfo.gDR;
        constInfo.bN2GDR = inputParamsRegbase.bSize * constInfo.n2GDR;
        constInfo.n2GS1DR = constInfo.n2Size * constInfo.gS1DR;
        constInfo.s2BaseN2DR = s2BaseSize * constInfo.n2DR;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        if constexpr (hasRope) {
            constInfo.s1BaseN2GDR = s1BaseSize * constInfo.n2GDR;
            constInfo.mm1RopeKa = constInfo.n2GDR;
            constInfo.mm1RopeKb = constInfo.n2DR;
        }

        constInfo.mm1Ka = constInfo.n2GD;
        constInfo.mm1Kb = constInfo.n2D;
        constInfo.mm2Kb = constInfo.n2Dv;
        if constexpr (isInfer) {
            if (inputParamsRegbase.isGqa) {
                constInfo.mm1Ka = constInfo.dSize;
            }
        }
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if constexpr (isInfer) {
                if (inputParamsRegbase.isGqa) {
                    constInfo.attentionOutStride = 0;
                }
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
            constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
            if constexpr (hasRope) {
                constInfo.s1BaseN2GDR = s1BaseSize * constInfo.n2GDR;
                constInfo.mm1RopeKa = constInfo.n2GDR;
                constInfo.mm1RopeKb = constInfo.n2DR;
            }
            constInfo.mm1Ka = constInfo.n2GD;
            constInfo.mm1Kb = constInfo.n2D;
            constInfo.mm2Kb = constInfo.n2Dv;
            if constexpr (isInfer) {
                if (inputParamsRegbase.isGqa) {
                    constInfo.mm1Ka = constInfo.dSize;
                }
            }
            if ASCEND_IS_AIV {
                constInfo.attentionOutStride =
                    (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
                if constexpr (isInfer) {
                    if (inputParamsRegbase.isGqa) {
                        constInfo.attentionOutStride = 0;
                    }
                }
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            constInfo.s1BaseBN2GD = s1BaseSize * constInfo.bN2GD;
            constInfo.s2BaseBN2D = inputParamsRegbase.bSize * constInfo.s2BaseN2D;
            constInfo.bN2GDv = inputParamsRegbase.bSize * constInfo.n2GDv;
            constInfo.s1BaseBN2GDv = s1BaseSize * constInfo.bN2GDv;
            constInfo.s2BaseBN2Dv = inputParamsRegbase.bSize * constInfo.s2BaseN2Dv;
            if constexpr (hasRope) {
                constInfo.s1BaseBN2GDR = s1BaseSize * constInfo.bN2GDR;
                constInfo.s2BaseBN2DR = inputParamsRegbase.bSize * constInfo.s2BaseN2DR;
                constInfo.mm1RopeKa = constInfo.bN2GDR;
                constInfo.mm1RopeKb = constInfo.bN2DR;
            }
            constInfo.mm1Ka = constInfo.bN2GD;
            constInfo.mm1Kb = constInfo.bN2D;
            constInfo.mm2Kb = inputParamsRegbase.bSize * constInfo.n2Dv;
            if ASCEND_IS_AIV {
                constInfo.attentionOutStride =
                    (inputParamsRegbase.bSize * constInfo.n2Size * constInfo.gSize - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
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
            if ASCEND_IS_AIV {
                constInfo.attentionOutStride = 0;
            }
        }
    }
    if ASCEND_IS_AIC {
        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = inputParamsRegbase.preTokens;
            attenMaskInfo.nextTokens = inputParamsRegbase.nextTokens;
            attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
            attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
        }
    }
    if ASCEND_IS_AIV {
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            pseInfo.pseLayoutType = inputParamsRegbase.pseShapeType;
            pseInfo.pseType = inputParamsRegbase.pseType;
            pseInfo.pseBSize = inputParamsRegbase.pseBSize;
            pseInfo.pseS1Size = inputParamsRegbase.pseS1Size;
            pseInfo.pseS2Size = inputParamsRegbase.pseS2Size;
            pseInfo.pseEncodeType = (uint32_t)inputParamsRegbase.pseEncodeType;
            pseInfo.pseStride = pseInfo.pseLayoutType == pse1S2 ? 0 : s2BaseSize;
            pseInfo.qStartIdx = inputParamsRegbase.qStartIdx;
            pseInfo.kvStartIdx = inputParamsRegbase.kvStartIdx;
            if (inputParamsRegbase.pseShapeType == pse1S2) {
                constInfo.gS2 = constInfo.gSize * constInfo.s2Size;
            }
        }

        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = inputParamsRegbase.preTokens;
            attenMaskInfo.nextTokens = inputParamsRegbase.nextTokens;
            attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
            attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
            attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
            attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
        }
        constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
        constInfo.isSoftmaxLseEnable = inputParamsRegbase.isSoftMaxLseEnable;
    }
    if constexpr (isFd) {
        this->constInfo.splitKVNum = inputParamsRegbase.kvSplitPart;
        this->constInfo.sInnerLoopSize = CeilDivision(this->constInfo.s2Size, this->constInfo.splitKVNum);
        this->constInfo.actualCombineLoopSize = CeilDivision(this->constInfo.s2Size, this->constInfo.sInnerLoopSize);
    }
    this->constInfo.isRowInvalid = inputParamsRegbase.isRowInvalid;
    this->constInfo.headNumRatio = inputParamsRegbase.headNumRatio;
    this->constInfo.isGqa = inputParamsRegbase.isGqa;
    this->constInfo.isKvContinuous = inputParamsRegbase.isKvContinuous;
    this->constInfo.actualSeqLenSize = inputParamsRegbase.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(inputParamsRegbase.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(inputParamsRegbase.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(inputParamsRegbase.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(inputParamsRegbase.isKVHasLeftPadding);
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        this->constInfo.blockSize = inputParamsRegbase.blockSize;
        this->constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
        this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }

    this->constInfo.isBSNDOut = inputParamsRegbase.isBSNDOut;
    if (this->constInfo.isBSNDOut == 1) {
        this->constInfo.attentionOutStride =
            (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    }
    if ((!this->constInfo.isActualLenDimsNull) || (!this->constInfo.isActualLenDimsKVNull) ||
        this->constInfo.isQHasLeftPadding || this->constInfo.isKVHasLeftPadding ||
        (this->constInfo.isKvContinuous == 0)) {
        this->constInfo.enableKVPrefetch = false;
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::InitInput(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
{
    constInfo.subBlockIdx = get_subblockid();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
    }
    if ASCEND_IS_AIC{
        this->queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    }
    ListTensorDesc keyListTensorDescInit((__gm__ void *)key);
    currentKey = (__gm__ uint8_t *)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    if (this->tilingData->inputParamsRegbase.isKvContinuous == 1) {
        this->keyGm.SetGlobalBuffer((__gm__ KV_T *)currentKey);
    } else {
        this->keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
    }
    if ASCEND_IS_AIV {
        ListTensorDesc valueListTensorDescInit((__gm__ void *)value);
        currentValue = (__gm__ uint8_t *)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        if (this->tilingData->inputParamsRegbase.isKvContinuous == 1) {
            this->valueGm.SetGlobalBuffer((__gm__ KV_T *)currentValue);
        } else {
            this->valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
        }
    }
    if constexpr (isPa) {
        blocktablePtr = blockTable;
        this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        this->kvCacheBlockSize = this->tilingData->inputParamsRegbase.blockSize;
        this->maxBlockNumPerBatch = this->tilingData->inputParamsRegbase.blockTableDim2;
        this->kvLayout = this->tilingData->inputParamsRegbase.paLayoutType == 3 ? KVLAYOUT::BBH : KVLAYOUT::BNBD;
    }
    if (constInfo.isQHasLeftPadding) {
        constInfo.queryRightPaddingSize = ((__gm__ int64_t *)queryPaddingSize)[0];
        if (constInfo.queryRightPaddingSize < 0) {
            constInfo.queryRightPaddingSize = 0;
        }
    }
    if (constInfo.isKVHasLeftPadding) {
        constInfo.kvRightPaddingSize = ((__gm__ int64_t *)kvPaddingSize)[0];
        if (constInfo.kvRightPaddingSize < 0) {
            constInfo.kvRightPaddingSize = 0;
        }
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    } else {
        if constexpr (isInfer) {
            if (!constInfo.isActualLenDimsNull) {
                actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
            }
            if (!constInfo.isActualLenDimsKVNull) {
                actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
            }
        }
    }
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        if constexpr (hasPse) {
            pseGm.SetGlobalBuffer((__gm__ OUTPUT_T *)pse);
            pseSlope = pse;
        }
        if constexpr (hasAtten) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }
        if constexpr (isFd) {
            auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
            uint64_t accumOutSize = this->tilingData->inputParamsRegbase.accumOutSize;
            uint64_t logSumExpSize = this->tilingData->inputParamsRegbase.logSumExpSize;
            accumOutGm.SetGlobalBuffer((__gm__ T *)(workspace));
            workspace += accumOutSize * sizeof(float);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)(workspace));
            workspace += logSumExpSize * sizeof(float);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)(workspace));
            workspace += logSumExpSize * sizeof(float);
        }
        this->GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::GetExtremeValue(
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
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::InitBuffer()
{
    l1BufferManager.Init(pipe, 524288);  // 524288 is 512 * 1024
    if ASCEND_IS_AIC {
        mm2AL1Buffers.Init(l1BufferManager, mm2LeftSize);
        kvAntiquantRes.Init(l1BufferManager, kvAntiquantResSize);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(CV_L1_EVENT[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + CV_L1_EVENT[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(CV_L1_EVENT[1]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + CV_L1_EVENT[1]);
        mm1AL1Buffers.Init((l1BufferManager), mm1LeftSize);

        this->pipe->InitBuffer(this->bmm1ResBuf[0], mm1ResultSize);
        this->pipe->InitBuffer(this->bmm1ResBuf[1], mm1ResultSize);
        this->pipe->InitBuffer(this->bmm2ResBuf[0], mm2ResultSize);
        this->pipe->InitBuffer(this->bmm2ResBuf[1], mm2ResultSize);

        l0aBufferManager.Init(pipe, 65536); // 64 * 1024
        l0bBufferManager.Init(pipe, 65536); // 64 * 1024
        l0cBufferManager.Init(pipe, 262144); // 256 * 1024
        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
        mmL0CBuffers.Init(l0cBufferManager, 128 * 1024);
    }
    if ASCEND_IS_AIV {
        mm2AL1Buffers.Init(l1BufferManager, mm2LeftSize);
        kvAntiquantRes.Init(l1BufferManager, kvAntiquantResSize);

        this->pipe->InitBuffer(this->bmm1ResBuf[0], mm1ResultSize);
        this->pipe->InitBuffer(this->bmm1ResBuf[1], mm1ResultSize);
        this->pipe->InitBuffer(this->bmm2ResBuf[0], mm2ResultSize);
        this->pipe->InitBuffer(this->bmm2ResBuf[1], mm2ResultSize);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM1RES_EVENT[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM1RES_EVENT[1]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM2RES_EVENT[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM2RES_EVENT[1]);

        this->InitAntiquantBuffer();
        this->SoftmaxInitBuffer();
        this->pipe->InitBuffer(commonTBuf, 512);
        if constexpr (hasPseOuter) {
            this->pipe->InitBuffer(pseInQue, 1, pseInputSize);
        }
        if constexpr (hasAtten) {
            this->pipe->InitBuffer(attenMaskInQue[0], 1, attenMaskSize);
            this->pipe->InitBuffer(attenMaskInQue[1], 1, attenMaskSize);
        }
        if (constInfo.isSoftmaxLseEnable) {
            this->pipe->InitBuffer(softmaxLseQueue, 1, (s1BaseSize >> 1U) * sizeof(float) * 8);
        }
        this->pipe->InitBuffer(this->stage1OutQue[0], 1, stage1OutQueSize);
        this->pipe->InitBuffer(this->stage2OutQue[0], 1, stage2OutQueSize);
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::InitAntiquantBuffer()
{
    this->pipe->InitBuffer(kvInputQue, 3, kvInputSize); // 3 buffer
    this->pipe->InitBuffer(kvOutputQue, 2, kvOutSize);  // 2 buffer
    this->pipe->InitBuffer(keyAntiqScaleInputQue, 1, 2048); // 2048 is 2 * 1024
    this->pipe->InitBuffer(keyAntiqOffsetInputQue, 1, 2048); // 2048 is 2 * 1024
    this->pipe->InitBuffer(valueAntiqScaleInputQue, 1, 2048); // 2048 is 2 * 1024
    this->pipe->InitBuffer(valueAntiqOffsetInputQue, 1, 2048); // 2048 is 2 * 1024
    if constexpr (KVFP4) {
        this->pipe->InitBuffer(kvAntiqMxScaleRes, BUFFER_SIZE_BYTE_4K_ANTIQUANT);
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::SoftmaxInitBuffer()
{
    this->pipe->InitBuffer(this->softmaxSumBuf[0], 256); // [64, 1] SOFTMAXSUMBUF_SIZE:256
    this->pipe->InitBuffer(this->softmaxSumBuf[1], 256); // [64, 1] 1 is second buffer,
    this->pipe->InitBuffer(this->softmaxSumBuf[2], 256); // [64, 1] 2 is third buffer
    this->pipe->InitBuffer(this->maxBrdcst, 1, 2048); // [64, 8] SUMBRDCST_SIZE:2048
    this->pipe->InitBuffer(this->sumBrdcst, 1, 2048); // [64, 8] SUMBRDCST_SIZE:2048
    this->pipe->InitBuffer(this->softmaxMaxBuf[0], 256); // [64, 1] SOFTMAXMAXBUF_SIZE:256
    this->pipe->InitBuffer(this->softmaxMaxBuf[1], 256); // [64, 1] 1 is second buffer
    this->pipe->InitBuffer(this->softmaxMaxBuf[2], 256); // [64, 1] 2 is third buffer
    this->pipe->InitBuffer(this->softmaxExpBuf[0], 256); // [64, 1] SOFTMAXEXPBUF_SIZE:256
    this->pipe->InitBuffer(this->softmaxExpBuf[1], 256); // [64, 1] 1 is second buffer
    this->pipe->InitBuffer(this->softmaxExpBuf[2], 256); // [64, 1] 2 is third buffer
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::InitOutputSingleCore()
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::InitLseOutputSingleCore()
{
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParams = this->tilingData->initOutputParams;
    if (coreNum != 0 && constInfo.aivIdx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        uint32_t tailSize = initParams.totalSoftMaxLseOutputSize - constInfo.aivIdx * singleCoreLseSize;
        uint32_t singleInitLseSize = tailSize < singleCoreLseSize ? tailSize : singleCoreLseSize;
        InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreLseSize], singleInitLseSize, 3e+99);
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::InitQuant(
    __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
    __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
    __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset)
{
    if constexpr (ANTIQUANT) {
        if (keyAntiquantScale == nullptr) {
            int64_t antiValueOffsetInitPos = constInfo.n2D;
            if constexpr (KVFP4) {
                antiValueOffsetInitPos = (uint64_t)(this->tilingData->inputParamsRegbase.bSize) * constInfo.n2S2D / 32 / 2;
            }
            if constexpr (KEY_ANTIQUANT_PER_TOKEN) {
                antiValueOffsetInitPos = (uint64_t)(this->tilingData->inputParamsRegbase.bSize) * antiqSeqSize;
            }
            if (antiquantPerTensorFlag) {
                antiValueOffsetInitPos = 1;
            }
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)antiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer(((__gm__ VALUE_ANTIQ_PARAMS_T*)antiquantScale) + antiValueOffsetInitPos);
            antiqOffsetExistFlag = (antiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiquantOffsetGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)antiquantOffset);
                valueAntiquantOffsetGm.SetGlobalBuffer(((__gm__ VALUE_ANTIQ_PARAMS_T*)antiquantOffset) + antiValueOffsetInitPos);
            }
        } else {
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)keyAntiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer((__gm__ VALUE_ANTIQ_PARAMS_T*)valueAntiquantScale);
            antiqOffsetExistFlag = (keyAntiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiquantOffsetGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)keyAntiquantOffset);
                valueAntiquantOffsetGm.SetGlobalBuffer((__gm__ VALUE_ANTIQ_PARAMS_T*)valueAntiquantOffset);
            }
        }
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::Process()
{
    if (this->tilingData->initOutputParams.needInit) {
        SyncAll<false>();
    }
    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    int32_t actualCoreNums = multiCoreParamsRegbase.coreNum;
    if constexpr (isFd) {
        actualCoreNums =  inputParamsRegbase.bSize * this->constInfo.n2Size
            * this->constInfo.splitKVNum;
    }
    if ((aicIdx) >= actualCoreNums) {
        if constexpr (isFd) {
            SyncAll();
        }
        return;
    }
    setConstAntiTaskParam();
    // 确定核内切分起点
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    if constexpr (!isFd) {
        bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
        gS1StartIdx = multiCoreParamsRegbase.sparseStartIdx[aicIdx];
        if (likely((multiCoreParamsRegbase.coreNum - 1) > (aicIdx))) {
            bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
            if (nextGs1Idx != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = inputParamsRegbase.bSize * this->constInfo.n2Size *
                this->constInfo.headNumRatio;
        }
    } else {
        gS1StartIdx = 0;
        bnStartIdx = 0;
        bnEndIdx = 1;
        s2LoopLimit = 0;
    }
    int64_t taskId = 0;
    int64_t subTaskId = 0;
    bool isFirstAntiquantKey = true;
    bool isFirstAntiquantValue = true;
    bool isLastAntiquantKey = false;
    bool isLastAntiquantValue = false;
    bool isLastBmm1 = false;
    RunInfo<isInfer> runInfo[NUM_4];
    RunParamStr<isInfer> runParam;
    if constexpr (isFd) {
        runParam.boIdx = (aicIdx) / (this->constInfo.n2Size * this->constInfo.splitKVNum);
        runParam.n2oIdx = ((aicIdx) / this->constInfo.splitKVNum) % this->constInfo.n2Size;
        bnStartIdx = runParam.boIdx * this->constInfo.n2Size + runParam.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }
    int64_t multiCoreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; bnIdx++) {
        bool lastBN;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            lastBN = IsLastBN(bnIdx, bnEndIdx);
        } else {
            lastBN = (bnIdx == bnEndIdx - 1);
        }
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / (this->constInfo.n2Size * this->constInfo.headNumRatio);
            runParam.n2oIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, useDn>(runParam, this->constInfo, this->attenMaskInfo, this->keyGm, 
            this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn>(runParam, this->constInfo, lastBN, nextGs1Idx);
        if constexpr (isFd) {
            if (constInfo.sInnerLoopSize * (aicIdx % constInfo.splitKVNum) > runParam.actualS2Size) {
                runParam.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize =
                    runParam.actualS2Size -
                    this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum);
                runParam.s2LineEndIdx = tailSInnerLoopSize > this->constInfo.sInnerLoopSize ?
                                        this->constInfo.sInnerLoopSize :
                                        tailSInnerLoopSize;
            }
            runParam.s1LoopTimes = 1;
        }
        for (int64_t gS1Index = gS1StartIdx; gS1Index <runParam.s1LoopTimes; gS1Index++) {
            s2LoopLimit = 0;
            runParam.s2LoopStartIdx = 0;
            this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
            bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, useDn>(runParam, this->constInfo,
                gS1Index, this->actualSeqQlenAddr, this->pseInfo);
            bool s2NoNeedCalc = ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn>(runParam, this->constInfo);
            bool lastLoopThisCore = lastBN && (gS1Index == runParam.s1LoopTimes - 1);
            if ((s1NoNeedCalc || s2NoNeedCalc) && !lastLoopThisCore) {
                continue;
            }
            if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                if (runParam.boIdx > 0 && actualSeqQlenAddr[runParam.boIdx] - actualSeqQlenAddr[runParam.boIdx - 1] == 0) {
                    continue;
                }
            }
            s2LoopLimit = runParam.s2LoopEndIdx - 1;
            if (lastLoopThisCore) {
                isLastBmm1 = true;
                s2LoopLimit += 2; // 2 is Preload Num
            }
            for (int64_t s2LoopCount = runParam.s2LoopStartIdx; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
                if (s2LoopCount < runParam.s2LoopEndIdx) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];  // 3 is mod 4
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, runParam.s2LoopEndIdx - 1, multiCoreInnerIdx);
                    if ASCEND_IS_AIV {
                        AntiquantKey(runInfo1, subTaskId, isFirstAntiquantKey, runParam);
                        isFirstAntiquantKey = false;
                        subTaskId++;
                    }
                    if ASCEND_IS_AIC {
                        IterateBmm1(runInfo1, runParam, subTaskId);
                        subTaskId++;
                    }
                }
                if (taskId >= 1 && (s2LoopCount < runParam.s2LoopEndIdx + 1)) {
                    RunInfo<isInfer> &runInfo2 = runInfo[(taskId - 1) & 3];  // 3 is mod 4
                    if ASCEND_IS_AIV {
                        ProcessVec1(runInfo2);
                        AntiquantValue(runInfo2, subTaskId, isFirstAntiquantValue, runParam);
                        isFirstAntiquantValue = false;
                        subTaskId++;
                    }
                    if ASCEND_IS_AIC {
                        IterateBmm2(subTaskId, runInfo2);
                        subTaskId++;
                    }
                }
                if (taskId >= 2 && (s2LoopCount < runParam.s2LoopEndIdx + 2)) {  // Later Than mm1 is 2 
                    if ASCEND_IS_AIV {
                        RunInfo<isInfer> &runInfo3 = runInfo[(taskId - 2) & 3]; // 3 is mod 4
                        ProcessVec2(runInfo3);
                    }
                }
                taskId++;
            }
            multiCoreInnerIdx++;
        }
        gS1StartIdx = 0;
    }
    if constexpr (isFd) {
        if ASCEND_IS_AIV {
            SyncAll();
            InitFDBuffers();
            FlashDecodeCompute();
        }
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline bool FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::IsLastBN(uint32_t bnStartIdx, uint32_t bnEndIdx)
{
    if (bnStartIdx != bnEndIdx - 1) {
        for (uint32_t bnIdx = bnStartIdx + 1; bnIdx < bnEndIdx; bnIdx++) {
            uint32_t boIdx = bnIdx / constInfo.n2Size;
            uint32_t boStart = bnStartIdx / constInfo.n2Size;
            if (actualSeqQlenAddr[boIdx] != actualSeqQlenAddr[boStart]) {
                if (!isPa && (actualSeqKvlenAddr[boIdx] == actualSeqKvlenAddr[boIdx - 1]))
                    continue;
                return false;
            }
        }
    }
    return true;
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
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
    if (this->constInfo.isGqa) {
        runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    } else {
        runParam.goIdx = bnIndex % this->constInfo.headNumRatio;
    }
    runParam.s1oIdx = gS1Index % constInfo.s1OuterSize;
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::SetRunInfo(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2LoopStartIdx = runParam.s2LoopStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;  // 3 is mod 3
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
    runInfo.taskIdMod3 = taskId % 3;  // 3 is mod num 
    runInfo.s2LoopLimit = s2LoopLimit;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, constInfo.s1Size, constInfo.s2Size);
    } else {
        runInfo.b1SSOffset = runInfo.boIdx * constInfo.s1S2;
        runInfo.b1SSOffsetAlign = runInfo.boIdx * constInfo.s1Size * Align(constInfo.s2Size);
    }

    if constexpr (isFd) {
        runInfo.flashDecodeS2Idx = this->aicIdx % constInfo.splitKVNum;
    }
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.queryOffset = runParam.tensorQOffset;
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    if constexpr (isInfer) {
        runInfo.qRopeOffset = runParam.qRopeNBGOffset;
        InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, useDn>(runParam, runInfo);
        ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, useDn>(runParam, constInfo, s2LoopCount, runInfo);
        if ASCEND_IS_AIV{
            ComputeOffsetForAntiquant<CHILD_SPEC_TEMPLATE_ARGS, useDn>(runParam, constInfo, s2LoopCount, runInfo);
        }
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::GetSeqQlenKvlenByBoidx(int64_t boIdx, 
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

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ComputeBmm1Tail(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    if constexpr (isInfer) {
        if ((runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    } else {
        if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::setConstAntiTaskParam()
{
    isBeforeHalf = ((constInfo.aivIdx & 1) == 0);
    taskParam.batchSize = this->tilingData->inputParamsRegbase.bSize;
    taskParam.seqSize = constInfo.s2Size;
    taskParam.kvHeadNum = constInfo.n2Size;
    taskParam.headDim = constInfo.dSize;
    taskParam.headDimAlignBlock = ALIGNAntiquant((uint64_t)taskParam.headDim, BYTE_BLOCK_ANTIQUANT / sizeof(Q_T));
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
        taskParam.kvStep = constInfo.dSize;
    } else {
        taskParam.kvStep = taskParam.kvHeadNum * constInfo.dSize;
    }
    taskParam.copySplitS = kvInputSize / sizeof(KV_T) / dTemplateAlign64;
    taskParam.isPertensor = antiquantPerTensorFlag;
    taskParam.isPerHead = antiquantPerHeadFlag;
    taskParam.kvCacheBlockSize = kvCacheBlockSize;
    taskParam.maxBlockNumPerSeq = maxBlockNumPerBatch;
    taskParam.paKvShapeType = paKvShapeType;
    taskParam.isExistOffset = antiqOffsetExistFlag;
    taskParam.singleSInnerSize = constInfo.s2BaseSize;
    taskParam.sInnerLoopSize = constInfo.sInnerLoopSize;
    taskParam.antiqSeqSize = antiqSeqSize;
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::AntiquantKey(RunInfo<isInfer> &runInfo, 
    int64_t &subTaskId, bool &first, RunParamStr<isInfer> &runParam)
{
    Buffer<BufferType::L1> outBufAntiKey = this->kvAntiquantRes.Get();
    if(isBeforeHalf) {
        taskParam.copyTotalS = runInfo.s2RealSize / 2;  // 2 is Vecnum 
    } else {
        taskParam.copyTotalS = runInfo.s2RealSize - (runInfo.s2RealSize / 2);  // 2 is Vecnum 
    }
    if (taskParam.copyTotalS == 0) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(VC_L1_EVENT[subTaskId % 2]);  // 2 is double buffer 
        return;
    }
    uint32_t curSequence = constInfo.s2BaseSize * runInfo.s2LoopCount + runInfo.kvLeftPaddingSize;

    taskParam.kvGmOffset = runInfo.keyOffset + constInfo.subBlockIdx * (runInfo.s2RealSize / 2) * taskParam.kvStep;  // 2 is Vec num

    taskParam.s2BatchOffset = curSequence;
    taskParam.kvPaddingBeginOffset = runInfo.kvLeftPaddingSize;

    if constexpr (KVFP4) {
        taskParam.isLoadAntiqParam = true;
        taskParam.isFreeAntiqParam = false;
    } else if constexpr (KEY_ANTIQUANT_PER_TOKEN) {
        taskParam.isLoadAntiqParam = true;
        taskParam.isFreeAntiqParam = true;
    } else {
        if (antiquantPerTensorFlag) {
            taskParam.isLoadAntiqParam = (first);
            taskParam.isFreeAntiqParam = (first);
        } else {
            taskParam.isLoadAntiqParam = unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx);
            taskParam.isFreeAntiqParam = unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit);
        }
    }

    taskParam.antiqParamOffset = runInfo.n2oIdx * taskParam.headDim;

    taskParam.bIdx = runInfo.boIdx;
    taskParam.n2Idx = runInfo.n2oIdx;
    taskParam.s2Idx = runInfo.s2LoopCount;
    taskParam.flashDecodeS2Idx = runInfo.flashDecodeS2Idx;
    keyAntiquantProcessor.ProcessBaseAPI(outBufAntiKey, this->keyGm, keyAntiqScaleGm,
                              keyAntiquantOffsetGm, blockTableGm, kvInputQue, kvOutputQue, keyAntiqScaleInputQue,
                              keyAntiqOffsetInputQue, kvAntiqMxScaleRes, taskParam, subTaskId, isBeforeHalf, runInfo.s2RealSize);
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::AntiquantValue(RunInfo<isInfer> &runInfo, 
    int64_t &subTaskId, bool &first, RunParamStr<isInfer> &runParam)
{
    Buffer<BufferType::L1> outBufAntiValue = this->kvAntiquantRes.Get();
    if(isBeforeHalf) {
        taskParam.copyTotalS = runInfo.s2RealSize / 2;  // 2 is Vec num
    } else {
        taskParam.copyTotalS = runInfo.s2RealSize - (runInfo.s2RealSize / 2);  // 2 is Vec num
    }
    if (taskParam.copyTotalS == 0) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(VC_L1_EVENT[subTaskId % 2]);  // 2 is double buffer
        return;
    }
    uint32_t curSequence = constInfo.s2BaseSize * runInfo.s2LoopCount + runInfo.kvLeftPaddingSize;

    taskParam.kvGmOffset = runInfo.valueOffset + constInfo.subBlockIdx * (runInfo.s2RealSize / 2) * taskParam.kvStep;  // 2 is Vec num
    
    taskParam.s2BatchOffset = curSequence;
    taskParam.kvPaddingBeginOffset = runInfo.kvLeftPaddingSize;

    if constexpr (VALUE_ANTIQUANT_PER_TOKEN || ANTIQUANT_PER_GROUP) {
        taskParam.isLoadAntiqParam = true;
        taskParam.isFreeAntiqParam = true;
    } else {
        if (antiquantPerTensorFlag) {
            taskParam.isLoadAntiqParam = (first);
            taskParam.isFreeAntiqParam = (first);
        } else {
            taskParam.isLoadAntiqParam = unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx);
            taskParam.isFreeAntiqParam = unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit);
        }
    }

    taskParam.antiqParamOffset = runInfo.n2oIdx * taskParam.headDim;

    taskParam.bIdx = runInfo.boIdx;
    taskParam.n2Idx = runInfo.n2oIdx;
    taskParam.s2Idx = runInfo.s2LoopCount;
    taskParam.flashDecodeS2Idx = runInfo.flashDecodeS2Idx;
    valueAntiquantProcessor.ProcessBaseAPI(outBufAntiValue, valueGm,
                                valueAntiqScaleGm, valueAntiquantOffsetGm, blockTableGm, kvInputQue, kvOutputQue, valueAntiqScaleInputQue,
                                valueAntiqOffsetInputQue, kvAntiqMxScaleRes, taskParam, subTaskId, isBeforeHalf, runInfo.s2RealSize);
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ProcessVec1(RunInfo<isInfer> &runInfo)
{
    if (runInfo.actualS2Size == 0) {
        return;
    }
    ProcessVec1Nd(runInfo);
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ProcessVec1Nd(RunInfo<isInfer> &runInfo)
{
    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(CV_MM1RES_EVENT[runInfo.taskIdMod2]);
    LocalTensor<OUTPUT_T> pseUb;
    if constexpr (hasPseOuter == true) {
        PseCopyIn<T, OUTPUT_T, hasPseOuter>(this->pseInQue, this->pseGm, runInfo, constInfo, pseInfo);
        pseUb = this->pseInQue.template DeQue<OUTPUT_T>();
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
                pseInfo.qStartIdx = 0;
                pseInfo.kvStartIdx = 0;
            }
        }
        ComputeInnerPseOffset<T, Q_T, hasPse>(slopes, posShift, runInfo, constInfo, pseInfo, this->pseSlope);
    }
    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (hasAtten == true) {
        AttenMaskCopyIn<hasAtten, isFd>(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
            this->attenMaskGmInt, runInfo, constInfo, attenMaskInfo);
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
    LocalTensor<T> inputTensorVec = this->bmm1ResBuf[runInfo.taskIdMod2].template Get<T>();
    auto stage1CastTensor = this->stage1OutQue[0].template AllocTensor<Q_T>();
    //这一坨代码段接近3k
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
        if (runInfo.s2RealSize == 128) {  // 128 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, false, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 64) { // 64 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 128 && runInfo.s2RealSize > 64) {  // 64 and 128 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, false, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 256 && runInfo.s2RealSize > 128) {  // 128 and 256 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, false, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 512 && runInfo.s2RealSize > 256) {  // 256 and 512 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, false, s1BaseSize, s2BaseSize, GT_256_AND_LTE_512, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 1024 && runInfo.s2RealSize > 512) {  // 512 and 1024 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, false, s1BaseSize, s2BaseSize, GT_512_AND_LTE_1024, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        }
    } else {
         if (runInfo.s2RealSize == 128) { // 128 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, true, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 64) { // 64 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 128 && runInfo.s2RealSize > 64) { // 64 and 128 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, true, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 256 && runInfo.s2RealSize > 128) { // 128 and 256 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, true, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 512 && runInfo.s2RealSize > 256) { // 256 and 512 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, true, s1BaseSize, s2BaseSize, GT_256_AND_LTE_512, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 1024 && runInfo.s2RealSize > 512) { // 512 and 1024 is s2RealSize
            ProcessVec1Vf<T, Q_T, OUTPUT_T, true, s1BaseSize, s2BaseSize, GT_512_AND_LTE_1024, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        }
    }
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM1RES_EVENT[runInfo.taskIdMod2]);
    if constexpr (hasAtten) {
        this->attenMaskInQue[runInfo.taskIdMod2].template FreeTensor(attenMaskUb);
    }
    if constexpr (hasPseOuter) {
        this->pseInQue.template FreeTensor(pseUb);
    }

    // ===============DataCopy to L1===============
    this->stage1OutQue[0].template EnQue(stage1CastTensor);
    this->stage1OutQue[0].template DeQue<Q_T>();
    Buffer<BufferType::L1> outBufVec1 = this->mm2AL1Buffers.Get();
    LocalTensor<Q_T> mm2AL1Tensor = outBufVec1.GetTensor<Q_T>();

    SetFlag<HardEvent::V_MTE3>(this->UbToL1Event);
    WaitFlag<HardEvent::V_MTE3>(this->UbToL1Event);

    uint32_t vec1ScmBlockTrue = s1BaseSize * (16 / sizeof(Q_T));

    if (likely(runInfo.halfS1RealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1ScmBlockTrue], stage1CastTensor, 
                {s2BaseSize / 16, (uint16_t)runInfo.halfS1RealSize, 
                (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
    }
    this->stage1OutQue[0].template FreeTensor(stage1CastTensor);
    // =======================================================
    if (runInfo.s2LoopCount != runInfo.s2LoopStartIdx) {
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<Q_T, float>::value) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            this->InvalidLineProcess(runInfo, sumUb, maxUb);
        }
    }
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        SoftmaxDataCopyOut(runInfo);
        if constexpr (isFd) {
            ComputeLogSumExpAndCopyToGm(runInfo);
            return;
        }
        SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
    }
    return;
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ProcessVec2(RunInfo<isInfer> &runInfo)
{
    if (runInfo.actualS2Size == 0) {
        return;
    }
    ProcessVec2S2Split(runInfo);
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ProcessVec2S2Split(RunInfo<isInfer> &runInfo)
{
    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(CV_MM2RES_EVENT[runInfo.taskIdMod2]);
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    if (unlikely(runInfo.vec2S1RealSize == 0)) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM2RES_EVENT[runInfo.taskIdMod2]);
        return;
    }
    LocalTensor<T> inputTensorVec = this->bmm2ResBuf[runInfo.taskIdMod2].template Get<T>();
    LocalTensor<T> vec2ResUb = this->stage2OutQue[0].template AllocTensor<T>();
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
        DataCopy(vec2ResUb, inputTensorVec, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        float deSCalePreVValue = 1.0f;
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, true>(
                    vec2ResUb, inputTensorVec, vec2ResUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, deSCalePreVValue);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, inputTensorVec, vec2ResUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, deSCalePreVValue);
            }
        } else {
            if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, true>(
                    vec2ResUb, inputTensorVec, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, deSCalePreVValue);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, inputTensorVec, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, deSCalePreVValue);
            }
        }
    }
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(
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
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM2RES_EVENT[runInfo.taskIdMod2]);
}

CHILD_SPEC_TEMPLATE_ANTI
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::Bmm2DataCopyOut(RunInfo<isInfer> &runInfo, 
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (!IsSameType<Q_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<Q_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
                InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, runInfo.vec2S1RealSize,
                    dSizeAligned64, this->negativeFloatScalar, 0.0);
            }
        }
        RowInvalid(vec2ResUb, vec2S1Idx, runInfo);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        stage2OutQue[0].EnQue(attenOut);
        stage2OutQue[0].DeQue<OUTPUT_T>();
    } else {
        stage2OutQue[runInfo.taskIdMod2].EnQue(vec2ResUb);
        stage2OutQue[runInfo.taskIdMod2].template DeQue<OUTPUT_T>();
        attenOut = vec2ResUb;
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    if constexpr (IsSameType<Q_T, float>::value) {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 3;
    } else {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    }
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    int64_t attenOutOffset = constInfo.dSizeV;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        attenOutOffset = constInfo.n2GDv;
        if constexpr (isInfer) {
            if (constInfo.isGqa == 1) {
                attenOutOffset = constInfo.dSizeV;
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            attenOutOffset = constInfo.n2GDv;
            if constexpr (isInfer) {
                if (constInfo.isGqa == 1) {
                    attenOutOffset = constInfo.dSizeV;
                }
            }
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

CHILD_SPEC_TEMPLATE_ANTI
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::RowInvalid(
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, RunInfo<isInfer> &runInfo)
{
    if constexpr (isInfer && hasAtten) {
        if (!constInfo.isRowInvalid || \
            attenMaskInfo.compressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
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
            RowInvalidUpdateVF<float, static_cast<uint32_t>(dVTemplateType)>(vec2ResUb, maxTensor, runInfo.vec2S1RealSize, constInfo.dSizeV);
        }
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline int64_t FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ComputeOffsetForSoftmax(
        RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx)
{
    return vec2S1Idx * runInfo.vec2S1BaseSize;
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::SoftmaxLseCopyOut(
    LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp, RunInfo<isInfer> &runInfo)
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
        if (layout == LayOutTypeEnum::LAYOUT_TND) {
            if (constInfo.isGqa) {
                intriParams1.dstStride = 0;
            } else {
                intriParams1.dstStride = sizeof(float) * (constInfo.n2G - 1);
            }
        } else {
            intriParams1.dstStride = 0;
        }
        DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParams1);

        softmaxLseQueue.FreeTensor(lseUb);
    } else {
        return;
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::Bmm2FDOut(RunInfo<isInfer> &runInfo,
    LocalTensor<T> &vec2ResUb, int64_t vec2CalcSize)
{
    LocalTensor<T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;

    stage2OutQue[runInfo.taskIdMod2].EnQue(vec2ResUb);
    stage2OutQue[runInfo.taskIdMod2].template DeQue<OUTPUT_T>();
    attenOut = vec2ResUb;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = runInfo.firstHalfS1RealSize;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (BYTE_BLOCK_ANTIQUANT / sizeof(T));
    dataCopyParams.dstStride = 0;

    uint32_t mStart = constInfo.subBlockIdx * runInfo.halfS1RealSize;
    size_t base = (runInfo.boIdx * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV +
                  runInfo.n2oIdx * constInfo.gSize * constInfo.dSizeV) * constInfo.splitKVNum + mStart * constInfo.dSizeV;
    DataCopyPad(this->accumOutGm[base + runInfo.flashDecodeS2Idx * constInfo.gSize * constInfo.dSizeV],
                attenOut, dataCopyParams);
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::IterateBmm1(
        RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, const int64_t &subTaskId)
{
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B = this->kvAntiquantRes.Get();
    LocalTensor<T> outputTensor = this->bmm1ResBuf[runInfo.taskIdMod2].template Get<T>();
    if (unlikely(runInfo.s2LoopCount == 0)) {
        mm1A = mm1AL1Buffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用
        LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
        Nd2NzParams Gm2L1Nd2NzParams;
        Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
        Gm2L1Nd2NzParams.nValue = runInfo.s1RealSize; // 单个ND矩阵的实际行数，单位为元素个数
        Gm2L1Nd2NzParams.dValue = constInfo.dSize; // 单个ND矩阵的实际列数，单位为元素个数
        Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.srcDValue = constInfo.mm1Ka; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数 15 >> 4 << 4 is Align 16
        Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移，单位为元素数量
        DataCopy(mm1ATensor, this->queryGm[runParam.tensorQOffset], Gm2L1Nd2NzParams);
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else {
        mm1A = mm1AL1Buffers.GetPre();
        mm1A.Set<HardEvent::MTE2_MTE1>();
    }
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(VC_L1_EVENT[subTaskId % 2]); // 2 is double buffer
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + VC_L1_EVENT[subTaskId % 2]); // 16 is Vec num, 2 is double buffer

    mm1A.Wait<HardEvent::MTE2_MTE1>();

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)runInfo.s1RealSize,
                    (uint32_t)runInfo.s2RealSize,
                    (uint32_t)(constInfo.dSize),
                    0,
                    1
                    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseSizediv4, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<Q_T>(), mm1B.GetTensor<Q_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runParam.s2LoopEndIdx - 1)) {
        mm1A.Set<HardEvent::MTE1_MTE2>();
    }

    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(CV_L1_EVENT[subTaskId % 2]); // 2 is double buffer
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + CV_L1_EVENT[subTaskId % 2]); // 16 is Vec num, 2 is double buffer
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(VC_MM1RES_EVENT[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + VC_MM1RES_EVENT[runInfo.taskIdMod2]); // 16 is Vec num

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐 7 >> 3 <<3 is Align
    fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) 15 is align
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputTensor, mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB

    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CV_MM1RES_EVENT[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + CV_MM1RES_EVENT[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG, 16 is Vec num

    mm1ResL0C.Set<HardEvent::FIX_M>();
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::IterateBmm2(
    const int64_t &subTaskId, RunInfo<isInfer> &runInfo)
{
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(VC_L1_EVENT[subTaskId % 2]); // 2 is double buffer
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + VC_L1_EVENT[subTaskId % 2]); // 16 is Vec num, 2 is double buffer

    Buffer<BufferType::L1> inputBufA = this->mm2AL1Buffers.Get();
    Buffer<BufferType::L1> inputBufB = this->kvAntiquantRes.Get();
    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)s1BaseSize,  // singleM 64
                    (uint32_t)constInfo.dSizeV,  // singleN 512
                    (uint32_t)runInfo.s2RealSize,  // singleK 128
                    false,    // isLeftTranspose
                    false     // isRightTranspose 
                    };
    MatmulN<Q_T, Q_T, T, s1BaseSize, dBaseSizediv4, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                                inputBufA.GetTensor<Q_T>(), 
                                inputBufB.GetTensor<Q_T>(),
                                mmL0ABuffers,
                                mmL0BBuffers,
                                mm2ResL0C.GetTensor<T>(),
                                param);
    // inputBufA.Set<HardEvent::MTE1_MTE2>(); // 释放
    // inputBufB.Set<HardEvent::MTE1_MTE2>(); // 释放

    mm2ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(CV_L1_EVENT[subTaskId % 2]); // 2 is double buffer
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + CV_L1_EVENT[subTaskId % 2]); // 16 is Vec num, 2 is double buffer
    
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(VC_MM2RES_EVENT[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + VC_MM2RES_EVENT[runInfo.taskIdMod2]); // 16 is Vec num

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = constInfo.dSizeV; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    fixpipeParams.mSize = s1BaseSize; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) 15 is align
    fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除 15 >> 4 << 4 is align
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    LocalTensor<T> outputTensor = this->bmm2ResBuf[runInfo.taskIdMod2].template Get<T>();
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputTensor, mm2ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CV_MM2RES_EVENT[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + CV_MM2RES_EVENT[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG, 16 is aiv num
}

//fd
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ComputeLogSumExpAndCopyToGm(
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
        bOffset = runInfo.boIdx * constInfo.n2Size * constInfo.gS1;
        n2Offset = runInfo.n2oIdx * constInfo.gS1;
        gOffset = runInfo.goIdx * constInfo.s1Size;
    }
    int64_t s1Offset = (runInfo.s1oIdx * s1BaseSize +
        constInfo.subBlockIdx * runInfo.firstHalfS1RealSize);
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;
    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t gmOffset = runInfo.boIdx * constInfo.n2Size * constInfo.splitKVNum * constInfo.gSize * FP32_ONE_BLOCK_SIZE + 
                        runInfo.n2oIdx * constInfo.splitKVNum * constInfo.gSize * FP32_ONE_BLOCK_SIZE +
                        runInfo.flashDecodeS2Idx * constInfo.gSize * FP32_ONE_BLOCK_SIZE + mStart * FP32_ONE_BLOCK_SIZE;
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
    if constexpr (!IsSameType<Q_T, float>::value || !containAllOptionalInput) {
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

/*FD*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::InitFDBuffers()
{
    this->pipe->Reset();
    this->pipe->InitBuffer(lseTmpBuff, bufferSizeByte32K);
    this->pipe->InitBuffer(softmaxMaxInputQue, 1, bufferSizeByte32K);
    this->pipe->InitBuffer(softmaxSumInputQue, 1, bufferSizeByte32K);
    this->pipe->InitBuffer(FDResOutputQue, 1, bufferSizeByte32K);
    this->pipe->InitBuffer(accumOutInputQue, 1, bufferSizeByte32K);
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND, 每行结果存为8个重复lse元素(32B对齐)
        this->pipe->InitBuffer(softmaxLseQueue, 1, (s1BaseSize >> 1U) * sizeof(float) * 8);
    }
}

/*FD*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::FlashDecodeCompute()
{
    int64_t bIdx = constInfo.aivIdx / constInfo.n2Size;
    int64_t n2Idx = constInfo.aivIdx % constInfo.n2Size;
    int64_t batchSize = this->tilingData->inputParamsRegbase.bSize;
    if (constInfo.aivIdx >= batchSize * constInfo.n2Size) {
        return;
    }
    int64_t actualSeqLen;
    GetActualSeqLenKV(bIdx, actualSeqLen);
    if (actualSeqLen == 0) {
        return;
    }
    uint64_t attenOutOffset = (uint64_t)bIdx * constInfo.n2GDv + n2Idx * constInfo.gDv;
    CombineSplitKVRes(attenOutOffset, bIdx, n2Idx);
}

/*FD*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::GetActualSeqLenKV(
    int64_t boIdx, int64_t &actualSeqLen)
{
    int64_t s2InCurrentBatch = constInfo.s2Size;
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, boIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            s2InCurrentBatch = kvTensorDesc.GetShape(2); // 2 is shape of kvTensorDesc 
        } else {
            s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
    if (constInfo.isActualLenDimsKVNull) {
        actualSeqLen = s2InCurrentBatch;
    } else {
        actualSeqLen = (constInfo.actualSeqLenKVSize == 1) ? actualSeqKvlenAddr[0] :
                                                             actualSeqKvlenAddr[boIdx];
    }
    if (constInfo.isKVHasLeftPadding) {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualSeqLen - constInfo.kvRightPaddingSize;
        if (kvLeftPaddingSize < 0) {
            actualSeqLen = 0;
        }
    }
}

/*FD*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::CombineSplitKVRes(
    uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t gSplitSizeLse =
        bufferSizeByte32K / (FA_BYTE_BLOCK_ANTIQUANT * constInfo.splitKVNum); // 32K / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = bufferSizeByte32K / sizeof(float) / (uint32_t)dVTemplateType;
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
    if (constInfo.gSize > gSplitMax) {
        gSplitSize = (gSplitSize > gSplitMax) ? gSplitMax : gSplitSize;
    } else {
        gSplitSize = (gSplitSize > constInfo.gSize) ? constInfo.gSize : gSplitSize;
    }
    uint32_t loopCount = CeilDivision(constInfo.gSize, gSplitSize);
    uint32_t tailSplitSize = constInfo.gSize - (loopCount - 1) * gSplitSize;
    uint64_t lseOffset = 0;

    // 尾块与非尾块都使用这些ub，减少处理次数
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>(); // 复用内存
    uint32_t shapeArray[] = {(uint32_t)gSplitSize, fp32BaseSize};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape

    // 非尾块处理
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        uint32_t startRow = i * gSplitSize;
        CopyLseIn(bIdx, n2Idx, startRow, gSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + i * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, gSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, gSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(attenOutOffset, tmp1, startRow, gSplitSize);
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
        CopyFinalResOut(attenOutOffset, tmp1, startRow, tailSplitSize);
    }
}

/*FD*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::CopyLseIn(
    uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.AllocTensor<T>();
    LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = constInfo.splitKVNum;
    copyInParams.blockLen = dealRowCount * fp32BaseSize * sizeof(T);
    copyInParams.srcStride = (constInfo.gSize - dealRowCount) * fp32BaseSize * sizeof(T);
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    uint64_t combineLseOffset =
        ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum + n2Idx * constInfo.splitKVNum) *
            constInfo.gSize * fp32BaseSize +
        startRow * fp32BaseSize;

    DataCopyPad(softmaxMaxLocal, softmaxFDMaxGm[combineLseOffset], copyInParams, copyInPadParams);
    DataCopyPad(softmaxSumLocal, softmaxFDSumGm[combineLseOffset], copyInParams, copyInPadParams);
    softmaxMaxInputQue.EnQue(softmaxMaxLocal);
    softmaxSumInputQue.EnQue(softmaxSumLocal);
}

/*FD*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ComputeScaleValue(LocalTensor<T> lseMaxUb,
    LocalTensor<T> lseSumUb, uint32_t splitSize, uint64_t lseOffset)
{
    LocalTensor<T> lseOutputUb;
    if (constInfo.isSoftmaxLseEnable) {
        lseOutputUb = softmaxLseQueue.template AllocTensor<T>();
    }
    ComputeScaleValue_VF(lseMaxUb, lseSumUb, lseOutputUb, splitSize, constInfo.actualCombineLoopSize,
                         constInfo.isSoftmaxLseEnable);
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

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ReduceFinalRes(
    uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow, uint32_t dealRowCount)
{
    for (uint32_t j = 0; j < constInfo.actualCombineLoopSize; ++j) {
        // 第一次，mul结果直接放到dst里
        CopyAccumOutIn(bIdx, n2Idx, j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();
        ReduceFinalRes_const_VF<T, (uint32_t)dVTemplateType>(dst, lseLocal, accumOutLocal, dealRowCount, j);
        accumOutInputQue.FreeTensor(accumOutLocal);
    }
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::CopyAccumOutIn(
    uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = ((int64_t)dVTemplateType - constInfo.dSizeV) / 8; // 8 for align factor

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = ((int64_t)dVTemplateType - constInfo.dSizeV) % 8; // 8 for align factor
    copyInPadParams.paddingValue = 0;

    uint64_t combineAccumOutOffset = ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum +
                                      n2Idx * constInfo.splitKVNum + splitKVIndex) *
                                         constInfo.gSize * constInfo.dSizeV +
                                     startRow * constInfo.dSizeV;
    DataCopyPad(accumOutLocal, this->accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::CopyFinalResOut(
    uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = FDResOutputQue.AllocTensor<OUTPUT_T>();
    uint32_t dSizeAligned64 = (uint32_t)dVTemplateType;
    uint32_t shapeArray[] = {(uint32_t)dealRowCount, dSizeAligned64};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape
    Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * dSizeAligned64);

    FDResOutputQue.EnQue(tmpBmm2ResCastTensor);
    FDResOutputQue.DeQue<OUTPUT_T>();
    ReduceFDDataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, dSizeAligned64,
                        constInfo.dSizeV);
    FDResOutputQue.FreeTensor(tmpBmm2ResCastTensor);
}

CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::ReduceFDDataCopyOut(
    uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (FA_BYTE_BLOCK_ANTIQUANT / sizeof(OUTPUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(this->attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

/*当前用不到*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::InvalidLineProcess(
    RunInfo<isInfer> &runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb)
{
    if (constInfo.softMaxCheckRes) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1),
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1)};
        bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VAULE_FP32, softmaxShapeInfo);
        if (!res) {
            constInfo.softMaxCheckRes = false;
        } else {
            if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.halfS1RealSize, this->negativeFloatScalar,
                    this->positiveFloatScalar);
            }
        }
    }
}

/*当前用不到*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline bool FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::SoftmaxInvalidLineCheck(
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

/*当前用不到*/
CHILD_SPEC_TEMPLATE_ANTI
__aicore__ inline void FlashAttentionScoreAntiquantKernel<CHILD_SPEC_TEMPLATE_ARGS_ANTI>::SoftmaxDataCopyOut(
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
    if constexpr (!IsSameType<Q_T, float>::value || !containAllOptionalInput) {
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
}
#endif