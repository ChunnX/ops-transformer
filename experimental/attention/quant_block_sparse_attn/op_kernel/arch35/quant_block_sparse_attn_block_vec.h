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
 * \file quant_block_sparse_attn_block_vec.h
 * \brief
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_BLOCK_VEC_H_
#define QUANT_BLOCK_SPARSE_ATTN_BLOCK_VEC_H_
#include "common/util_regbase.h"
#include "common/copy_gm_to_ub.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "quant_block_sparse_attn_attenmask.h"
#include "quant_block_sparse_attn_fp8_static.h"
#include "quant_block_sparse_attn_fp8_static_vec.h"
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/activation/softmax.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF
class QBSABlockVec {
public:
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = s1BaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = s1BaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = s1BaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = s1BaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool containAllOptionalInput = hasAtten;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;
    static constexpr TPosition bmm2OutPos = GetC2Position();
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr bool isW8In = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                   IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value;
    static constexpr bool POST_QUANT = false;
    static constexpr bool USE_DN = true;
    static constexpr int64_t FP8_QUANT_KV_BLOCK_SIZE = 256;
    static constexpr int64_t FP8_QUANT_K_BLOCK_SIZE = 256;
    static constexpr int64_t FP8_QUANT_V_BLOCK_SIZE = 512;
    static constexpr uint32_t bufferSizeByte32K = 32768;
    static constexpr uint32_t gSplitMax = 16;
    static constexpr uint32_t preloadTimes = 3;
    static constexpr uint64_t SYNC_MODE = 4;

    __aicore__ inline QBSABlockVec(){};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const QuantBlockSparseAttnTilingData *__restrict tiling,
                                        CVSharedParams<isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
                                        AttenMaskInfo &attenMaskInfo)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->negativeFloatScalar = this->GetNegativeValue();
            attenMaskInfoPtr = &attenMaskInfo;
        }
    }
    __aicore__ inline void InitCommonGlobalBuffer(__gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
                                                  __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
                                                  __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
                                                  __gm__ uint8_t *blockTable, __gm__ uint8_t *&workspace,
                                                  ConstInfo &constInfo);
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    __aicore__ inline void InitStaticLocalBuffer(ConstInfo &constInfo);
    __aicore__ inline void InitStaticEvents(ConstInfo &constInfo);
    __aicore__ inline void FinalizeStaticBuffers(ConstInfo &constInfo);

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo, int32_t subLoop = 0);
    __aicore__ inline void ProcessVec1Static(LocalTensor<uint8_t> &outputBuf, uint32_t outputEventId,
                                             LocalTensor<uint8_t> &bmm1ResBuf, uint32_t bmm1EventId,
                                             RunInfo &runInfo, ConstInfo &constInfo);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec2Static(LocalTensor<uint8_t> &bmm2ResBuf, uint32_t bmm2EventId,
                                             RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams<isPa> &sharedParams, int32_t aicIdx,
                                                   uint8_t subBlockIdx);
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
                                            __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale, __gm__ uint8_t *prefix,
                                            __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable,
                                            __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                                            __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                            __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
                                            ConstInfo &constInfo);
    __aicore__ inline void InitUniqueLocalBuffer(ConstInfo &constInfo);
    __aicore__ inline void GenerateDropoutMask(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<uint8_t> &dropMaskUb)
    {}
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb);
    __aicore__ inline void SoftmaxDataCopyOutFp8(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<half> &sumUb,
                                                 LocalTensor<half> &maxUb)
    {}
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                               LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);

    TPipe *tPipe;
    const QuantBlockSparseAttnTilingData *__restrict tilingData;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOut_initGm;

    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    attenMaskGmType attenMaskGmInt;
    using quantGmType = typename std::conditional<isFp8, GlobalTensor<float>, int8_t>::type;
    quantGmType deScaleQGm;
    quantGmType deScaleKGm;
    quantGmType deScaleVGm;
    quantGmType pScaleGm;
    GlobalTensor<int32_t> blockTableGm;
    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;
    vec2ResGmType vec2ResGm[3];

    TBuf<> commonTBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2];
    TEventID vToMte3Id[2];
    TBuf<> softmaxMaxBuf[3];
    TBuf<> softmaxSumBuf[3];
    TBuf<> softmaxExpBuf[3];
    TBuf<> pScaleBuf[3];
    TBuf<> preLoopMaxBuf;
    TBuf<> preLoopSumBuf;
    TBuf<> firstLoopSumBuf;
    TQue<QuePosition::VECIN, 1> qScaleInputQue;
    TQue<QuePosition::VECIN, 1> kScaleInputQue;
    TBuf<> vselrIndexesBuf[5];
    TBuf<> mm2InBuf;
    TBuf<> mm2OutBuf;
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    CopyQueryScaleGmToUb<float, layout> copyQueryScaleGmToUb;
    CopyAntiquantGmToUb<float, kvLayout> copyKeyScaleGmToUb;

    AttenMaskInfo *attenMaskInfoPtr;
    T negativeFloatScalar;
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;

    GlobalTensor<float> softmaxLseGm;
    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;

    LocalTensor<uint8_t> attenMaskBufStatic_;
    LocalTensor<INPUT_T> stage1OutBufStatic_[2];
    LocalTensor<T> stage2OutBufStatic_;
    LocalTensor<float> softmaxMaxBufStatic_[3];
    LocalTensor<float> softmaxSumBufStatic_[3];
    LocalTensor<T> softmaxExpBufStatic_[3];
    LocalTensor<float> preLoopMaxBufStatic_;
    LocalTensor<float> preLoopSumBufStatic_;
    LocalTensor<float> firstLoopSumBufStatic_;
    LocalTensor<float> qScaleBufStatic_;
    LocalTensor<float> kScaleBufStatic_;
    LocalTensor<float> softmaxLseBufStatic_;
    LocalTensor<uint8_t> vselrIndexesBufStatic_[5];

protected:
    __aicore__ inline void BroadCastAndCopyOut(RunInfo &runInfo, GlobalTensor<T> &sumGm, GlobalTensor<T> &maxGm,
                                               int64_t gmOffset, int64_t calculateSize);

    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                           int64_t vec2S1Idx, int64_t vec2CalcSize = 0);

private:
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar,
                                                   SoftMaxShapeInfo &softmaxShapeInfo);

    __aicore__ inline int64_t ComputeOffsetForSoftmax(RunInfo &runInfo, const int64_t vec2S1Idx);
    __aicore__ inline T GetNegativeValue();
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void InitLseOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo &runInfo, ConstInfo &constInfo);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitCommonGlobalBuffer(
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable, __gm__ uint8_t *&workspace,
    ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        deScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
        deScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
        deScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
        if (pScale != nullptr) {
            pScaleGm.SetGlobalBuffer((__gm__ float *)pScale);
            constInfo.pScale = this->pScaleGm.GetValue(0);
        } else {
            constInfo.pScale = 1.0f;
        }
        if constexpr (isPa) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        if constexpr (hasAtten) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }
        if constexpr (!bmm2Write2Ub) {
            int64_t bmm2ResBlock = tilingData->inputParamsRegbase.dSizeV;
            if constexpr (splitD) {
                bmm2ResBlock = (int64_t)dVTemplateType;
            }
            int64_t vec2ResultSize = (s1BaseSize)*constInfo.dBasicBlock;
            int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
            int64_t mm2ResultSize = (s1BaseSize)*bmm2ResBlock;
            if constexpr (splitD) {
                vec2SubBlockOffset = constInfo.subBlockIdx * vec2ResultSize >> 1;
                vec2ResGm[0].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
                vec2ResGm[1].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
                vec2ResGm[2].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
            }
            bmm2SubBlockOffset = constInfo.subBlockIdx * mm2ResultSize >> 1;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo, ConstInfo &constInfo,
    int32_t subLoop)
{
    ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool QBSABlockVec<TEMPLATE_ARGS>::SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb,
                                                                            uint32_t negativeIntScalar,
                                                                            SoftMaxShapeInfo &softmaxShapeInfo)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isUpdateNeedCheck = false;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
    for (uint32_t i = 0; i < softmaxShapeInfo.srcM; i++) {
        T maxValue = maxUb.GetValue(i);
        uint32_t checkValue = *reinterpret_cast<uint32_t *>(&maxValue);
        if (checkValue == negativeIntScalar) {
            isUpdateNeedCheck = true;
            break;
        }
    }
    SetMaskNorm();
    ResetMask();
    return isUpdateNeedCheck;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::BroadCastAndCopyOut(RunInfo &runInfo, GlobalTensor<T> &sumGm,
                                                                        GlobalTensor<T> &maxGm, int64_t gmOffset,
                                                                        int64_t calculateSize)
{
    LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
    FaVectorApi::BroadcastMaxSum(sumOutTensor, sumTensor, runInfo.halfS1RealSize);
    sumBrdcst.template EnQue(sumOutTensor);
    sumBrdcst.template DeQue<float>();
    DataCopy(sumGm[gmOffset], sumOutTensor, calculateSize);
    sumBrdcst.template FreeTensor(sumOutTensor);

    LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
    LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    FaVectorApi::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
    maxBrdcst.template EnQue(maxOutTensor);
    maxBrdcst.template DeQue<float>();
    DataCopy(maxGm[gmOffset], maxOutTensor, calculateSize);
    maxBrdcst.template FreeTensor(maxOutTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::ProcessVec1Dn(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    // bmm1ResBuf.WaitCrossCore();
    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (isFp8 && hasAtten) {
        AttenMaskCopyInDn<hasAtten>(this->attenMaskInQue[0], this->attenMaskGmInt, runInfo, constInfo,
                                    *attenMaskInfoPtr, runInfo.sparseBlk1PartialMask, runInfo.sparseBlk2PartialMask);
        attenMaskUb = this->attenMaskInQue[0].template DeQue<uint8_t>();
    }
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];

    auto expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[0];
    int64_t stage1Offset = runInfo.taskIdMod2;

    LocalTensor<float> qScaleUbTensor = qScaleInputQue.template AllocTensor<float>();
    LocalTensor<float> kScaleUbTensor = kScaleInputQue.template AllocTensor<float>();

    if (unlikely(runInfo.s2SparseBlk2RealSize < constInfo.kvSparseBlockSize)) {
        Duplicate<float>(kScaleUbTensor, 1, s2BaseSize);
        TEventID vDoneEvent = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(vDoneEvent);  // Vector 完成
        WaitFlag<HardEvent::V_MTE2>(vDoneEvent); // MTE2 等待
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vDoneEvent);
    }

    copyQueryScaleGmToUb(qScaleUbTensor, deScaleQGm, runInfo, constInfo);
    copyKeyScaleGmToUb(kScaleUbTensor, deScaleKGm, blockTableGm, runInfo, constInfo);
    qScaleInputQue.template EnQue(qScaleUbTensor);
    qScaleUbTensor = qScaleInputQue.DeQue<float>();
    kScaleInputQue.template EnQue(kScaleUbTensor);
    kScaleUbTensor = kScaleInputQue.DeQue<float>();

    bmm1ResBuf.WaitCrossCore();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    if constexpr (layout == QBSALayout::NTD) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, hasAtten, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor,
                kScaleUbTensor, Align64Func(static_cast<uint16_t>(runInfo.s1RealSizeAlign32 >> 1)),
                runInfo.s2AlignedSize, runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar,
                runInfo.sparseBlk1PartialMask || runInfo.sparseBlk2PartialMask, constInfo.pScale);
        } else {
            FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor,
                kScaleUbTensor, Align64Func(static_cast<uint16_t>(runInfo.s1RealSizeAlign32 >> 1)),
                runInfo.s2AlignedSize, runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar,
                runInfo.sparseBlk1PartialMask || runInfo.sparseBlk2PartialMask, constInfo.pScale);
        }
    } else {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, false, hasAtten, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor,
                kScaleUbTensor, Align64Func(static_cast<uint16_t>(runInfo.s1RealSizeAlign32 >> 1)),
                runInfo.s2AlignedSize, runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar,
                runInfo.sparseBlk1PartialMask || runInfo.sparseBlk2PartialMask, constInfo.pScale);
        } else {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, hasAtten, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor,
                kScaleUbTensor, Align64Func(static_cast<uint16_t>(runInfo.s1RealSizeAlign32 >> 1)),
                runInfo.s2AlignedSize, runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar,
                runInfo.sparseBlk1PartialMask || runInfo.sparseBlk2PartialMask, constInfo.pScale);
        }
    }
    qScaleInputQue.FreeTensor(qScaleUbTensor);
    kScaleInputQue.FreeTensor(kScaleUbTensor);
    bmm1ResBuf.SetCrossCore();
    if constexpr (isFp8 && hasAtten) {
        this->attenMaskInQue[0].template FreeTensor(attenMaskUb);
    }
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    //-------------------------Data copy to l1-------------------------
    LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

    // 按照64对齐搬运
    DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize *
                          Align64Func(static_cast<uint16_t>(runInfo.s2RealSize))],
             stage1CastTensor, {static_cast<uint16_t>((runInfo.s2RealSize + 63) >> 6), 64, 66, 0});
    DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize *
                              Align64Func(static_cast<uint16_t>(runInfo.s2RealSize)) +
                          Align64Func(static_cast<uint16_t>(runInfo.s2RealSize)) * 32],
             stage1CastTensor[65 << 5], {static_cast<uint16_t>((runInfo.s2RealSize + 63) >> 6), 64, 66, 0});

    outputBuf.SetCrossCore();
    //-----------------------------------------------------------------
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    if constexpr (hasLse) {
        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
            SoftmaxDataCopyOut(runInfo, constInfo, sumUb, maxUb);
        }
    }
    return;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::ProcessVec1Static(LocalTensor<uint8_t> &outputBuf,
                                                                      uint32_t outputEventId,
                                                                      LocalTensor<uint8_t> &bmm1ResBuf,
                                                                      uint32_t bmm1EventId, RunInfo &runInfo,
                                                                      ConstInfo &constInfo)
{
    if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                  bmm2Write2Ub) {
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (hasAtten) {
            WaitFlag<HardEvent::V_MTE2>(Fp8Static::MASK_EVENT_ID);
            attenMaskUb = attenMaskBufStatic_;
            AttenMaskCopyInDnFun<hasAtten>(attenMaskUb, attenMaskGmInt, runInfo, constInfo, *attenMaskInfoPtr,
                                           runInfo.sparseBlk1PartialMask, 0, runInfo.sparseBlkIdx1,
                                           runInfo.s2SparseBlk1RealSize);
            AttenMaskCopyInDnFun<hasAtten>(attenMaskUb, attenMaskGmInt, runInfo, constInfo, *attenMaskInfoPtr,
                                           runInfo.sparseBlk2PartialMask,
                                           (constInfo.s1BaseSize >> 1) * runInfo.s2SparseBlk1RealSize,
                                           runInfo.sparseBlkIdx2, runInfo.s2SparseBlk2RealSize);
            if ((runInfo.sparseBlk1PartialMask && runInfo.sparseBlkIdx1 != -1) ||
                (runInfo.sparseBlk2PartialMask && runInfo.sparseBlkIdx2 != -1)) {
                SetFlag<HardEvent::MTE2_V>(Fp8Static::MASK_EVENT_ID);
                WaitFlag<HardEvent::MTE2_V>(Fp8Static::MASK_EVENT_ID);
            }
        }

        LocalTensor<float> sumUb = softmaxSumBufStatic_[runInfo.multiCoreIdxMod3];
        LocalTensor<float> maxUb = softmaxMaxBufStatic_[runInfo.multiCoreIdxMod3];
        LocalTensor<T> expUb = softmaxExpBufStatic_[runInfo.taskIdMod3];
        const int64_t stage1Offset = runInfo.taskIdMod2;
        const event_t stage1EventId = static_cast<event_t>(Fp8Static::Stage1EventId(stage1Offset));
        WaitFlag<HardEvent::MTE3_V>(stage1EventId);

        WaitFlag<HardEvent::V_MTE2>(Fp8Static::SCALE_EVENT_ID);
        Duplicate<float>(kScaleBufStatic_, 1, s2BaseSize);
        SetFlag<HardEvent::V_MTE2>(Fp8Static::SCALE_EVENT_ID);
        WaitFlag<HardEvent::V_MTE2>(Fp8Static::SCALE_EVENT_ID);
        copyQueryScaleGmToUb(qScaleBufStatic_, deScaleQGm, runInfo, constInfo);
        copyKeyScaleGmToUb(kScaleBufStatic_, deScaleKGm, blockTableGm, runInfo, constInfo);
        SetFlag<HardEvent::MTE2_V>(Fp8Static::SCALE_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(Fp8Static::SCALE_EVENT_ID);

        Fp8Static::AivWaitUbReady(bmm1EventId);
        LocalTensor<T> mmRes = bmm1ResBuf.ReinterpretCast<T>();
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutBufStatic_[stage1Offset];
        if constexpr (layout == QBSALayout::NTD) {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                FaVectorApi::ProcessVec1VfDnPerTokenHeadStatic<T, INPUT_T, false, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBufStatic_, attenMaskUb,
                    qScaleBufStatic_, kScaleBufStatic_,
                    Align64Func(static_cast<uint16_t>(runInfo.s1RealSizeAlign32 >> 1)), runInfo.s2AlignedSize,
                    runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar,
                    runInfo.sparseBlk1PartialMask || runInfo.sparseBlk2PartialMask, constInfo.pScale);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHeadStatic<T, INPUT_T, true, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBufStatic_, attenMaskUb,
                    qScaleBufStatic_, kScaleBufStatic_,
                    Align64Func(static_cast<uint16_t>(runInfo.s1RealSizeAlign32 >> 1)), runInfo.s2AlignedSize,
                    runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar,
                    runInfo.sparseBlk1PartialMask || runInfo.sparseBlk2PartialMask, constInfo.pScale);
            }
        } else {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                FaVectorApi::ProcessVec1VfDnStatic<T, INPUT_T, false, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBufStatic_, attenMaskUb,
                    qScaleBufStatic_, kScaleBufStatic_,
                    Align64Func(static_cast<uint16_t>(runInfo.s1RealSizeAlign32 >> 1)), runInfo.s2AlignedSize,
                    runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar,
                    runInfo.sparseBlk1PartialMask || runInfo.sparseBlk2PartialMask, constInfo.pScale);
            } else {
                FaVectorApi::ProcessVec1VfDnStatic<T, INPUT_T, true, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBufStatic_, attenMaskUb,
                    qScaleBufStatic_, kScaleBufStatic_,
                    Align64Func(static_cast<uint16_t>(runInfo.s1RealSizeAlign32 >> 1)), runInfo.s2AlignedSize,
                    runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar,
                    runInfo.sparseBlk1PartialMask || runInfo.sparseBlk2PartialMask, constInfo.pScale);
            }
        }
        SetFlag<HardEvent::V_MTE2>(Fp8Static::SCALE_EVENT_ID);
        Fp8Static::AivSetUbFree(bmm1EventId);
        if constexpr (hasAtten) {
            SetFlag<HardEvent::V_MTE2>(Fp8Static::MASK_EVENT_ID);
        }

        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.ReinterpretCast<INPUT_T>();
        SetFlag<HardEvent::V_MTE3>(stage1EventId);
        WaitFlag<HardEvent::V_MTE3>(stage1EventId);
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize *
                              Align64Func(static_cast<uint16_t>(runInfo.s2RealSize))],
                 stage1CastTensor, {static_cast<uint16_t>((runInfo.s2RealSize + 63) >> 6), 64, 66, 0});
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize *
                                  Align64Func(static_cast<uint16_t>(runInfo.s2RealSize)) +
                              Align64Func(static_cast<uint16_t>(runInfo.s2RealSize)) * 32],
                 stage1CastTensor[65 << 5], {static_cast<uint16_t>((runInfo.s2RealSize + 63) >> 6), 64, 66, 0});
        SetFlag<HardEvent::MTE3_V>(stage1EventId);
        Fp8Static::AivSetL1PReady(outputEventId);
        if (constInfo.isSoftmaxLseEnable && unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
            SoftmaxLseCopyOut(sumUb, maxUb, runInfo, constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t QBSABlockVec<TEMPLATE_ARGS>::ComputeOffsetForSoftmax(RunInfo &runInfo,
                                                                               const int64_t vec2S1Idx)
{
    return vec2S1Idx * runInfo.vec2S1BaseSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::ProcessVec2OnUb(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    if (unlikely(runInfo.vec2S1BaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;

    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
    float deSCaleVValue = 1.0f;
    // QBSA: V 为 per-head 反量化，vScale shape [N2]，按 n2 head 取值。
    // 原 deScaleKvOffset 未被赋值（未初始化），GetValue 会越界读 GM，
    // 触发 AICore SVM page fault(507015)。
    deSCaleVValue = this->deScaleVGm.GetValue(runInfo.n2oIdx);
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        // QBSA: V per-head 反量化，前一块与当前块同为 vScale[n2oIdx]（非 FIA 的 per-block scale），
        // 不能用 deScaleKvOffset 的 per-block ping-pong（未初始化且 -1 会越界，触发 507015）。
        float deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.n2oIdx);
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == 1)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(vec2ResUb, mmRes, vec2ResUb, expUb,
                                                                             runInfo.vec2S1RealSize, dTemplateAlign64,
                                                                             deSCaleVValue, deSCalePreVValue);
            } else {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, mmRes, vec2ResUb, expUb,
                                                                              runInfo.vec2S1RealSize, dTemplateAlign64,
                                                                              deSCaleVValue, deSCalePreVValue);
            }
        } else {
            if (unlikely(runInfo.s2LoopCount == 1)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                    deSCalePreVValue);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                    deSCalePreVValue);
            }
        }
    }
    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize,
                                                               (uint16_t)dTemplateAlign64, deSCaleVValue);
        }
        CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
                                                                ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    ProcessVec2OnUb(bmm2ResBuf, runInfo, constInfo);
    return;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::ProcessVec2Static(LocalTensor<uint8_t> &bmm2ResBuf,
                                                                      uint32_t bmm2EventId, RunInfo &runInfo,
                                                                      ConstInfo &constInfo)
{
    if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                  bmm2Write2Ub) {
        Fp8Static::AivWaitUbReady(bmm2EventId);
        if (unlikely(runInfo.vec2S1BaseSize == 0)) {
            Fp8Static::AivSetUbFree(bmm2EventId);
            return;
        }
        runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
        int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
        float deSCaleVValue = this->deScaleVGm.GetValue(runInfo.n2oIdx);
        LocalTensor<T> vec2ResUb = stage2OutBufStatic_;
        LocalTensor<T> mmRes = bmm2ResBuf.ReinterpretCast<T>();
        WaitFlag<HardEvent::MTE3_V>(Fp8Static::STAGE2_EVENT_ID);
        if (unlikely(runInfo.s2LoopCount == 0)) {
            Adds<T>(vec2ResUb, mmRes, static_cast<T>(0.0F), vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBufStatic_[runInfo.taskIdMod3];
            float deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.n2oIdx);
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                if (unlikely(runInfo.s2LoopCount == 1)) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                        deSCalePreVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                        deSCalePreVValue);
                }
            } else {
                LocalTensor<float> sumUb = softmaxSumBufStatic_[runInfo.multiCoreIdxMod3];
                if (unlikely(runInfo.s2LoopCount == 1)) {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                        deSCaleVValue, deSCalePreVValue);
                }
            }
        }
        Fp8Static::AivSetUbFree(bmm2EventId);
        if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                LocalTensor<float> sumUb = softmaxSumBufStatic_[runInfo.multiCoreIdxMod3];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb,
                                                                   runInfo.vec2S1RealSize,
                                                                   static_cast<uint16_t>(dTemplateAlign64),
                                                                   deSCaleVValue);
            }
            CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
        }
        SetFlag<HardEvent::MTE3_V>(Fp8Static::STAGE2_EVENT_ID);
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                   LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                   int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (splitD) {
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    if constexpr (!IsSameType<INPUT_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                      bmm2Write2Ub) {
            SetFlag<HardEvent::V_MTE3>(Fp8Static::STAGE2_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(Fp8Static::STAGE2_EVENT_ID);
        } else {
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        }
    } else {
        if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                      bmm2Write2Ub) {
            SetFlag<HardEvent::V_MTE3>(Fp8Static::STAGE2_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(Fp8Static::STAGE2_EVENT_ID);
        } else {
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
        }
        attenOut = vec2ResUb;
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2S1RealSize;

    int64_t attenOutOffset = constInfo.dSizeV;
    if constexpr (layout == QBSALayout::TND) {
        attenOutOffset = constInfo.n2GDv;
        // QBSA: 每核独立计算一个 (n2,g) head，输出 TND [T, N1, Dv]，按 head 直写；
        // 不走 FIA 的 GS1 合轴输出，避免把单 head 结果按 G 合并散写到错误位置。
        if (constInfo.isGqa) {
            attenOutOffset = constInfo.dSizeV;
        }
    }

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
                attenOut, dataCopyParams);
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 256);
    tPipe->InitBuffer(softmaxSumBuf[1], 256);
    tPipe->InitBuffer(softmaxSumBuf[2], 256);
    tPipe->InitBuffer(maxBrdcst, 1, 2048);
    tPipe->InitBuffer(sumBrdcst, 1, 2048);
    tPipe->InitBuffer(softmaxMaxBuf[0], 256);
    tPipe->InitBuffer(softmaxMaxBuf[1], 256);
    tPipe->InitBuffer(softmaxMaxBuf[2], 256);
    tPipe->InitBuffer(softmaxExpBuf[0], 256);
    tPipe->InitBuffer(softmaxExpBuf[1], 256);
    tPipe->InitBuffer(softmaxExpBuf[2], 256);
    tPipe->InitBuffer(preLoopMaxBuf, 256);
    tPipe->InitBuffer(preLoopSumBuf, 256);
    tPipe->InitBuffer(firstLoopSumBuf, 256);
    tPipe->InitBuffer(qScaleInputQue, 1, 256 * 8);
    tPipe->InitBuffer(kScaleInputQue, 1, 1024);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                  bmm2Write2Ub) {
        InitStaticLocalBuffer(constInfo);
        return;
    }
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    if constexpr (s2BaseSize == 256) {
        if constexpr (s1BaseSize == 128) {
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            SoftmaxInitBuffer();
            // s1BaseSize/2 * s2BaseSize = 128/2 * 256 = 16384,
            // 为防止硬件copy的冲突，会多开辟512
            tPipe->InitBuffer(stage1OutQue[0], 1, 16896);
            tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
            if constexpr (hasAtten) {
                // 16384 = s1BaseSize / 2 * s2BaseSize = 64 * 256
                tPipe->InitBuffer(attenMaskInQue[0], 1, 16384);
            }
        } else {
            SoftmaxInitBuffer();
            tPipe->InitBuffer(commonTBuf, 512);
            tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T));
            tPipe->InitBuffer(stage1OutQue[0], 1, 16896);
            tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
            if constexpr (hasAtten) {
                // 8192 = s1BaseSize=128 * s2BaseSize=64 * 1B(mask), UB buffer 分配
                tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
                tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
            }
        }
    } else {
        SoftmaxInitBuffer();

        if constexpr (hasAtten) {
            tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
            tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
        }

        if constexpr (!IsSameType<INPUT_T, float>::value) {
            tPipe->InitBuffer(commonTBuf, 512);
        }
        if constexpr (bmm2Write2Ub) {
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        } else if constexpr (dTemplateAlign64 <= 256) {
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        } else {
            tPipe->InitBuffer(stage2OutBuf, 32768);
        }
        tPipe->InitBuffer(stage1OutQue[0], 1, 8320);
        tPipe->InitBuffer(stage1OutQue[1], 1, 8320);
    }
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)], 128);
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], 64);
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)], 256);
    if constexpr (layout != QBSALayout::NTD) {
        tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::QSCALE_GATHER_INDEX)],
                          64 * sizeof(uint32_t));
    }
    LocalTensor<uint8_t> vselrIndexesTensor =
        vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
    for (uint32_t i = 0; i < 4; i++) {
        for (uint32_t j = 0; j < (256 >> 2); j++) {
            vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
        }
    }
    vselrIndexesTensor =
        vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
    for (uint32_t i = 0; i < 128; i++) {
        vselrIndexesTensor.SetValue(i, i << 1);
    }
    vselrIndexesTensor =
        vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
    for (uint32_t i = 0; i < 64; i++) {
        vselrIndexesTensor.SetValue(i, i << 2);
    }
    if constexpr (layout != QBSALayout::NTD) {
        LocalTensor<uint32_t> qGatherTensor =
            vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::QSCALE_GATHER_INDEX)].template Get<uint32_t>();
        for (uint32_t i = 0; i < 64; ++i) {
            qGatherTensor.SetValue(i, i * 8);
        }
    }

    InitUniqueLocalBuffer(constInfo);

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitStaticLocalBuffer(ConstInfo &constInfo)
{
    if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                  bmm2Write2Ub) {
        constexpr uint32_t stage2Bytes = (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T);
        constexpr uint32_t stage1Bytes = 16896U;
        constexpr uint32_t softmaxBytes = 256U;
        constexpr uint32_t qScaleBytes = 256U * sizeof(float) * 2U;
        constexpr uint32_t kScaleBytes = 1024U;
        constexpr uint32_t maskBytes = hasAtten ? ((s1BaseSize / CV_RATIO) * s2BaseSize) : 0U;
        constexpr uint32_t lseBytes = (s1BaseSize >> 1U) * sizeof(float) * 8U;
        constexpr uint32_t gt128Bytes = 128U;
        constexpr uint32_t gt64Bytes = 64U;
        constexpr uint32_t dnIndexBytes = 256U;
        constexpr uint32_t qGatherBytes = 64U * sizeof(uint32_t);

        constexpr uint32_t stage2Addr = Fp8Static::VEC_PRIVATE_UB_BASE;
        constexpr uint32_t stage1Addr0 = stage2Addr + stage2Bytes;
        constexpr uint32_t stage1Addr1 = stage1Addr0 + stage1Bytes;
        constexpr uint32_t sumAddr = stage1Addr1 + stage1Bytes;
        constexpr uint32_t maxAddr = sumAddr + 3U * softmaxBytes;
        constexpr uint32_t expAddr = maxAddr + 3U * softmaxBytes;
        constexpr uint32_t preMaxAddr = expAddr + 3U * softmaxBytes;
        constexpr uint32_t preSumAddr = preMaxAddr + softmaxBytes;
        constexpr uint32_t firstSumAddr = preSumAddr + softmaxBytes;
        constexpr uint32_t qScaleAddr = firstSumAddr + softmaxBytes;
        constexpr uint32_t kScaleAddr = qScaleAddr + qScaleBytes;
        constexpr uint32_t maskAddr = kScaleAddr + kScaleBytes;
        constexpr uint32_t lseAddr = maskAddr + maskBytes;
        constexpr uint32_t gt128Addr = lseAddr + lseBytes;
        constexpr uint32_t gt64Addr = gt128Addr + gt128Bytes;
        constexpr uint32_t dnIndexAddr = gt64Addr + gt64Bytes;
        constexpr uint32_t qGatherAddr = dnIndexAddr + dnIndexBytes;
        constexpr uint32_t ubEnd = qGatherAddr + qGatherBytes;
        static_assert(ubEnd <= Fp8Static::UB_CAPACITY, "FP8 vector UB layout exceeds capacity");

        stage2OutBufStatic_ = LocalTensor<uint8_t>(TPosition::VECCALC, stage2Addr, stage2Bytes).ReinterpretCast<T>();
        stage1OutBufStatic_[0] =
            LocalTensor<uint8_t>(TPosition::VECOUT, stage1Addr0, stage1Bytes).ReinterpretCast<INPUT_T>();
        stage1OutBufStatic_[1] =
            LocalTensor<uint8_t>(TPosition::VECOUT, stage1Addr1, stage1Bytes).ReinterpretCast<INPUT_T>();
        for (uint32_t i = 0U; i < 3U; ++i) {
            softmaxSumBufStatic_[i] =
                LocalTensor<uint8_t>(TPosition::VECCALC, sumAddr + i * softmaxBytes, softmaxBytes)
                    .ReinterpretCast<float>();
            softmaxMaxBufStatic_[i] =
                LocalTensor<uint8_t>(TPosition::VECCALC, maxAddr + i * softmaxBytes, softmaxBytes)
                    .ReinterpretCast<float>();
            softmaxExpBufStatic_[i] =
                LocalTensor<uint8_t>(TPosition::VECCALC, expAddr + i * softmaxBytes, softmaxBytes)
                    .ReinterpretCast<T>();
        }
        preLoopMaxBufStatic_ =
            LocalTensor<uint8_t>(TPosition::VECCALC, preMaxAddr, softmaxBytes).ReinterpretCast<float>();
        preLoopSumBufStatic_ =
            LocalTensor<uint8_t>(TPosition::VECCALC, preSumAddr, softmaxBytes).ReinterpretCast<float>();
        firstLoopSumBufStatic_ =
            LocalTensor<uint8_t>(TPosition::VECCALC, firstSumAddr, softmaxBytes).ReinterpretCast<float>();
        qScaleBufStatic_ =
            LocalTensor<uint8_t>(TPosition::VECIN, qScaleAddr, qScaleBytes).ReinterpretCast<float>();
        kScaleBufStatic_ =
            LocalTensor<uint8_t>(TPosition::VECIN, kScaleAddr, kScaleBytes).ReinterpretCast<float>();
        if constexpr (hasAtten) {
            attenMaskBufStatic_ = LocalTensor<uint8_t>(TPosition::VECIN, maskAddr, maskBytes);
        }
        softmaxLseBufStatic_ =
            LocalTensor<uint8_t>(TPosition::VECOUT, lseAddr, lseBytes).ReinterpretCast<float>();
        vselrIndexesBufStatic_[static_cast<uint32_t>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)] =
            LocalTensor<uint8_t>(TPosition::VECCALC, gt128Addr, gt128Bytes);
        vselrIndexesBufStatic_[static_cast<uint32_t>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)] =
            LocalTensor<uint8_t>(TPosition::VECCALC, gt64Addr, gt64Bytes);
        vselrIndexesBufStatic_[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)] =
            LocalTensor<uint8_t>(TPosition::VECCALC, dnIndexAddr, dnIndexBytes);
        if constexpr (layout != QBSALayout::NTD) {
            vselrIndexesBufStatic_[static_cast<uint32_t>(VselrIndexEnum::QSCALE_GATHER_INDEX)] =
                LocalTensor<uint8_t>(TPosition::VECCALC, qGatherAddr, qGatherBytes);
        }

        LocalTensor<uint8_t> vselrIndexesTensor =
            vselrIndexesBufStatic_[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)];
        for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t j = 0; j < (256 >> 2); j++) {
                vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
            }
        }
        vselrIndexesTensor =
            vselrIndexesBufStatic_[static_cast<uint32_t>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)];
        for (uint32_t i = 0; i < 128; i++) {
            vselrIndexesTensor.SetValue(i, i << 1);
        }
        vselrIndexesTensor =
            vselrIndexesBufStatic_[static_cast<uint32_t>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)];
        for (uint32_t i = 0; i < 64; i++) {
            vselrIndexesTensor.SetValue(i, i << 2);
        }
        if constexpr (layout != QBSALayout::NTD) {
            LocalTensor<uint32_t> qGatherTensor =
                vselrIndexesBufStatic_[static_cast<uint32_t>(VselrIndexEnum::QSCALE_GATHER_INDEX)]
                    .ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < 64; ++i) {
                qGatherTensor.SetValue(i, i * 8);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitStaticEvents(ConstInfo &constInfo)
{
    if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                  bmm2Write2Ub) {
        if ASCEND_IS_AIV {
            SetFlag<HardEvent::MTE3_V>(Fp8Static::STAGE1_EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(Fp8Static::STAGE1_EVENT_ID1);
            SetFlag<HardEvent::V_MTE2>(Fp8Static::SCALE_EVENT_ID);
            SetFlag<HardEvent::MTE3_V>(Fp8Static::STAGE2_EVENT_ID);
            SetFlag<HardEvent::MTE3_V>(Fp8Static::LSE_EVENT_ID);
            if constexpr (hasAtten) {
                SetFlag<HardEvent::V_MTE2>(Fp8Static::MASK_EVENT_ID);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::FinalizeStaticBuffers(ConstInfo &constInfo)
{
    if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                  bmm2Write2Ub) {
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_V>(Fp8Static::STAGE1_EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(Fp8Static::STAGE1_EVENT_ID1);
            WaitFlag<HardEvent::V_MTE2>(Fp8Static::SCALE_EVENT_ID);
            WaitFlag<HardEvent::MTE3_V>(Fp8Static::STAGE2_EVENT_ID);
            WaitFlag<HardEvent::MTE3_V>(Fp8Static::LSE_EVENT_ID);
            if constexpr (hasAtten) {
                WaitFlag<HardEvent::V_MTE2>(Fp8Static::MASK_EVENT_ID);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline T QBSABlockVec<TEMPLATE_ARGS>::GetNegativeValue()
{
    uint32_t negativeValue = NEGATIVE_MIN_VALUE_FP32;
    return *((float *)&negativeValue);
}

// ================================= Child-specific functions =================================

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(CVSharedParams<isPa> &sharedParams,
                                                                            int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    sharedParams.bSize = inputParamsRegbase.bSize;
    sharedParams.t1Size = inputParamsRegbase.t1Size;
    sharedParams.n2Size = inputParamsRegbase.n2Size;
    sharedParams.gSize = inputParamsRegbase.gSize;
    sharedParams.dSize = inputParamsRegbase.dSize;
    sharedParams.dSizeV = inputParamsRegbase.dSizeV;
    sharedParams.preTokens = inputParamsRegbase.preTokens;
    sharedParams.nextTokens = inputParamsRegbase.nextTokens;
    sharedParams.bandIndex = inputParamsRegbase.bandIndex;
    sharedParams.layoutType = inputParamsRegbase.layoutType;
    sharedParams.compressMode = inputParamsRegbase.attenMaskCompressMode;
    sharedParams.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;

    sharedParams.fromFused = inputParamsRegbase.fromFused;
    sharedParams.isGqa = inputParamsRegbase.isGqa;
    sharedParams.isKvContinuous = inputParamsRegbase.isKvContinuous;
    sharedParams.seqUsedQlenSize = inputParamsRegbase.seqUsedQlenSize;
    sharedParams.seqUsedKvlenSize = inputParamsRegbase.seqUsedKvlenSize;
    if constexpr (isPa) {
        auto &paParams = this->tilingData->paParams;
        sharedParams.blockTableDim2 = paParams.blockTableDim2;
        sharedParams.kvSparseBlockSize = paParams.kvBlockSize;
        sharedParams.qSparseBlockSize = paParams.qBlockSize;
        sharedParams.paLayoutType = paParams.paLayoutType;
        sharedParams.paBlockNumSum = paParams.paBlockNumSum;
        sharedParams.paBlockStride = paParams.paBlockStride;
        sharedParams.isRowInvalid = paParams.isRowInvalid;
    }

    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    sharedParams.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    sharedParams.coreNum = multiCoreParamsRegbase.coreNum;
    sharedParams.bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
    sharedParams.bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
    sharedParams.needInit = this->tilingData->initOutputParams.needInit;

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t *>(0);
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
#pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams<isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *softmaxLse,
                                                                __gm__ uint8_t *attentionOut,
                                                                ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if constexpr (hasLse) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }
        if (this->tilingData->initOutputParams.needInit == 1) {
            InitOutputSingleCore(constInfo);
            if constexpr (hasLse) {
                SyncAll();
                InitLseOutputSingleCore(constInfo);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace,
    uint64_t singleCoreOffset, uint32_t aicIdx, ConstInfo &constInfo)
{
    InitCommonGlobalBuffer(deqScaleQ, deqScaleK, deqScaleV, pScale, prefix, attenMask, blockTable, workspace,
                           constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitUniqueLocalBuffer(ConstInfo &constInfo)
{
    if constexpr (hasLse) {
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (s1BaseSize >> 1U) * sizeof(float) * 8);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::SoftmaxDataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                       LocalTensor<float> &sumUb,
                                                                       LocalTensor<float> &maxUb)
{
    SoftmaxLseCopyOut(sumUb, maxUb, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                       LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                       int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp,
                                                                      LocalTensor<float> &softmaxMaxTmp,
                                                                      RunInfo &runInfo, ConstInfo &constInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

<<<<<<< HEAD
=======
    if (!constInfo.isSoftmaxLseEnable) {
        return;
    }
    if constexpr (useDn && isFp8 && s1BaseSize == 128U && s2BaseSize == 256U && dTemplateAlign64 == 128U &&
                  bmm2Write2Ub) {
        WaitFlag<HardEvent::MTE3_V>(Fp8Static::LSE_EVENT_ID);
        LocalTensor<float> lseUb = softmaxLseBufStatic_;
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.halfS1RealSize);
        SetFlag<HardEvent::V_MTE3>(Fp8Static::LSE_EVENT_ID);
        WaitFlag<HardEvent::V_MTE3>(Fp8Static::LSE_EVENT_ID);
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = runInfo.halfS1RealSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParams1);
        SetFlag<HardEvent::MTE3_V>(Fp8Static::LSE_EVENT_ID);
        return;
    }
>>>>>>> 静态Tensor穿刺 第一版本
    LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
    ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.halfS1RealSize);
    softmaxLseQueue.template EnQue(lseUb);
    softmaxLseQueue.DeQue<float>();
    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(float);
    intriParams1.blockCount = runInfo.halfS1RealSize;
    intriParams1.srcStride = 0;
    intriParams1.dstStride = 0;
    DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParams1);
    softmaxLseQueue.FreeTensor(lseUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) > 0 ?
                            (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) :
                            0;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QBSABlockVec<TEMPLATE_ARGS>::InitLseOutputSingleCore(ConstInfo &constInfo)
{
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParams = this->tilingData->initOutputParams;
    if (coreNum != 0 && constInfo.aivIdx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        if (constInfo.aivIdx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(softmaxLseGm[constInfo.aivIdx * (initParams.totalSoftMaxLseOutputSize / coreNum)],
                          singleCoreLseSize, QBSA_EMPTY_LSE_VALUE);
    }
}

TEMPLATES_DEF
class QBSABlockVecDummy {
public:
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr TPosition bmm2OutPos = GetC2Position();
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    __aicore__ inline QBSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, ConstInfo &constInfo)
    {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const QuantBlockSparseAttnTilingData *__restrict tiling,
                                        CVSharedParams<isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
                                        AttenMaskInfo &attenMaskInfo) {};
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
                                            __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale, __gm__ uint8_t *prefix,
                                            __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable,
                                            __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                                            __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                            __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
                                            ConstInfo &constInfo)
    {}

    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void InitStaticEvents(ConstInfo &constInfo) {}
    __aicore__ inline void FinalizeStaticBuffers(ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo)
    {}
    __aicore__ inline void ProcessVec1Static(LocalTensor<uint8_t> &outputBuf, uint32_t outputEventId,
                                             LocalTensor<uint8_t> &bmm1ResBuf, uint32_t bmm1EventId,
                                             RunInfo &runInfo, ConstInfo &constInfo)
    {}

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec2Static(LocalTensor<uint8_t> &bmm2ResBuf, uint32_t bmm2EventId,
                                             RunInfo &runInfo, ConstInfo &constInfo)
    {}
};
} // namespace BaseApi
#endif // QUANT_BLOCK_SPARSE_ATTN_BLOCK_VEC_H_
