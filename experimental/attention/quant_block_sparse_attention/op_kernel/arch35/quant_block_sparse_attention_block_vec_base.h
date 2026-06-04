/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BSA-COPY-OF: FIA fullquant GQA local migration.
// TODO: 待实现 - 当前已完成自包含迁移和命名/include 适配；sparseSeqLen/sparseIndices 流水、per-token qScale、vScale、空 sparse 输出和 mask sparse offset 仍需按 BSA 方案 2.4 实现。

/*!
 * ile quant_block_sparse_attention_block_vec_base.h
 * \brief
 */
#ifndef QUANT_BLOCK_SPARSE_ATTENTION_BLOCK_VEC_BASE_H_
#define QUANT_BLOCK_SPARSE_ATTENTION_BLOCK_VEC_BASE_H_
#include "common/util_regbase.h"
#include "common/infer_flash_attention_comm.h"
#include "common/flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/activation/softmax.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"
#include "common/flash_attention_score_tiling_regbase.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF_BASE
class BSABlockVecBase {
public:
    /* =================编译期常量的基本块信息================= */
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
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool containAllOptionalInput = hasAtten;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;
    static constexpr TPosition bmm2OutPos = GetC2Position();
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr bool isW8In = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value ||
                                  IsSameType<INPUT_T, int8_t>::value;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    static constexpr int64_t FP8_QUANT_KV_BLOCK_SIZE = 256;

    /*HIFLOAT8场景 K_BLOCK_SIZE = 256 V_BLOCK_SIZE = 512*/
    static constexpr int64_t FP8_QUANT_K_BLOCK_SIZE = 256;
    static constexpr int64_t FP8_QUANT_V_BLOCK_SIZE = 512;
    // ==================== Functions ======================
    __aicore__ inline BSABlockVecBase() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const QuantBlockSparseAttentionTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, AttenMaskInfo &attenMaskInfo) {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            GetDerived()->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
            attenMaskInfoPtr = &attenMaskInfo;
        }
    }
    __aicore__ inline void InitCommonGlobalBuffer(
        __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *&workspace,
        ConstInfo<isInfer> &constInfo);
    __aicore__ inline float GetPaKScaleValue(
        const RunInfo<isInfer> &runInfo, const ConstInfo<isInfer> &constInfo);
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer> &constInfo);

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer> &constInfo, int32_t subLoop = 0);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer> &constInfo);

    TPipe *tPipe;
    const QuantBlockSparseAttentionTilingData *__restrict tilingData;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;

    /* =====================可选GM变量==================== */
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
    GlobalTensor<INPUT_T> sinkGm;

    /* =====================V侧UB变量==================== */
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[3];
    TBuf<> softmaxSumBuf[3];
    TBuf<> softmaxExpBuf[3];
    TBuf<> pScaleBuf[3];
    TBuf<> preLoopMaxBuf;
    TBuf<> preLoopSumBuf;
    TBuf<> firstLoopSumBuf;
    TQue<QuePosition::VECIN, 1> queryScaleQue[2];
    TBuf<> vselrIndexesBuf[4];
    TBuf<> mm2InBuf;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    /* =================初始化后不变的信息================= */
    AttenMaskInfo *attenMaskInfoPtr;
    T negativeFloatScalar;
    T positiveFloatScalar;
    // Bmm2阶段subblock在Gm上的偏移
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;
    __aicore__ inline ChildClass* GetDerived() {
        return static_cast<ChildClass*>(this);
    }
protected:
    __aicore__ inline void BroadCastAndCopyOut(RunInfo<isInfer> &runInfo, GlobalTensor<T> &sumGm,
                                               GlobalTensor<T> &maxGm, int64_t gmOffset, int64_t calculateSize);

    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = INPUT_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
private:
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer> &constInfo);
    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo);

    __aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo);
    __aicore__ inline void ProcessVec2NoGlobalUpdate(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, int64_t vec2CalcSize);

    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void InvalidLineProcess(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb);

    __aicore__ inline int64_t ComputeOffsetForSoftmax(RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer> &constInfo, int64_t dSizeAligned64);
    __aicore__ inline void MlaTransposeDataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo,
 	    LocalTensor<OUTPUT_T> &attenOut);
    __aicore__ inline void MlaTranspose2DataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo,
        LocalTensor<OUTPUT_T> &attenOut);
};

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::InitCommonGlobalBuffer(
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *&workspace,
    ConstInfo<isInfer> &constInfo)
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
            int64_t vec2ResultSize = (s1BaseSize) * constInfo.dBasicBlock;
            int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
            int64_t mm2ResultSize = (s1BaseSize) * bmm2ResBlock; // 使用Cube计算的总大小， Gm上的数据按照实际的dSize存储
            if constexpr (splitD) {
                vec2SubBlockOffset = constInfo.subBlockIdx * vec2ResultSize >> 1;
                vec2ResGm[0].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
                vec2ResGm[1].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
                vec2ResGm[2].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
            }
            bmm2SubBlockOffset = constInfo.subBlockIdx * mm2ResultSize >> 1; // s1BaseSize一定可以被2整除
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline float BSABlockVecBase<TEMPLATE_BASE_ARGS>::GetPaKScaleValue(
    const RunInfo<isInfer> &runInfo, const ConstInfo<isInfer> &constInfo)
{
    const uint64_t logicalToken = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;
    const uint64_t logicalBlock = logicalToken / constInfo.blockSize;
    const uint64_t tokenInBlock = logicalToken % constInfo.blockSize;
    const uint64_t physicalBlock =
        blockTableGm.GetValue(runInfo.boIdx * constInfo.blockTableDim2 + logicalBlock);
    const uint64_t byteOffset = physicalBlock * constInfo.paBlockStride +
        runInfo.n2oIdx * constInfo.blockSize * constInfo.combineDim + tokenInBlock * constInfo.combineDim;
    return deScaleKGm.GetValue(byteOffset / sizeof(float));
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer> &constInfo, int32_t subLoop)
{
        ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo, constInfo);
}

// =================================Private Functions================================
TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline bool BSABlockVecBase<TEMPLATE_BASE_ARGS>::SoftmaxInvalidLineCheck(
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

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::InvalidLineProcess(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb)
{
    if (constInfo.softMaxCheckRes) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1),
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1)};
        bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VALUE_FP32, softmaxShapeInfo);
        if (!res) {
            constInfo.softMaxCheckRes = false;
        } else {
            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
                SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.halfS1RealSize, this->negativeFloatScalar,
                    this->positiveFloatScalar);
            }
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::BroadCastAndCopyOut(
    RunInfo<isInfer> &runInfo, GlobalTensor<T> &sumGm,
    GlobalTensor<T> &maxGm, int64_t gmOffset, int64_t calculateSize)
{
    // Copy sum to gm
    LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
    FaVectorApi::BroadcastMaxSum(sumOutTensor, sumTensor, runInfo.halfS1RealSize);
    sumBrdcst.template EnQue(sumOutTensor);
    sumBrdcst.template DeQue<float>();
    DataCopy(sumGm[gmOffset], sumOutTensor, calculateSize);
    sumBrdcst.template FreeTensor(sumOutTensor);

    // Copy max to gm
    LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
    LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    FaVectorApi::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
    maxBrdcst.template EnQue(maxOutTensor);
    maxBrdcst.template DeQue<float>();
    DataCopy(maxGm[gmOffset], maxOutTensor, calculateSize);
    maxBrdcst.template FreeTensor(maxOutTensor);
}


TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec1Nd(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer> &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (hasAtten == true) {
        AttenMaskCopyIn<hasAtten, false>(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
                this->attenMaskGmInt, runInfo, constInfo, *attenMaskInfoPtr);
        attenMaskUb = this->attenMaskInQue[runInfo.taskIdMod2].template DeQue<uint8_t>();
    }

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
    LocalTensor<T> pScaleUb;
    LocalTensor<T> queryScaleUb;
    LocalTensor<uint8_t> apiTmpBuffer;
    if constexpr (IsSameType<INPUT_T, float>::value) {
        apiTmpBuffer = this->sumBrdcst.template AllocTensor<uint8_t>();
    } else {
        apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    }

    int64_t stage1Offset = 0;
    if constexpr (!IsSameType<INPUT_T, float>::value) {
        stage1Offset = runInfo.taskIdMod2;
    }
    float descaleQK = 1.0;
    float deSCaleKValue = 1.0;
    int64_t deScaleQOffset = 0;
    int64_t deScaleKvOffset = 0;
    // 训练模板：128*128 推理模板：128*256
    int64_t s1BlockCnt = CeilDiv(constInfo.s1Size, FP8_QUANT_BLOCK_SIZE);
    int64_t s2BlockCnt = CeilDiv(constInfo.s2Size, FP8_QUANT_KV_BLOCK_SIZE);
    /* Q的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S1, 128), 1] */
    deScaleQOffset = runInfo.boIdx * constInfo.n2G * s1BlockCnt +
                            runInfo.n2oIdx * constInfo.gSize * s1BlockCnt +
                            runInfo.goIdx * s1BlockCnt + runInfo.s1oIdx;
    if constexpr (isInfer) {
        runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
                                runInfo.n2oIdx * s2BlockCnt +
                                (runInfo.s2StartIdx >> 8) + ((runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) >> 1); // 8 ：按照256分块计算deScaleKv偏移
    } else {
        runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
                            runInfo.n2oIdx * s2BlockCnt +
                            (runInfo.s2StartIdx >> 8) + (runInfo.s2LoopCount >> 1); // 8 ：按照256分块计算deScaleKv偏移
    }
    deScaleKvOffset = runInfo.deScaleKvOffset;

    float deSCaleQValue = this->deScaleQGm.GetValue(deScaleQOffset);
    if constexpr (isPa) {
        deSCaleKValue = GetPaKScaleValue(runInfo, constInfo);
    } else {
        deSCaleKValue = this->deScaleKGm.GetValue(deScaleKvOffset);
    }
    descaleQK = deSCaleQValue * deSCaleKValue;

    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    if (runInfo.s2LoopCount == 0) {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, INPUT_T, false, s1BaseSize, s2BaseSize, EQ_128, hasAtten, false>(
            stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
            attenMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
            constInfo.keepProb, queryScaleUb, deSCaleKValue);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, false>(
            stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
            attenMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
            constInfo.keepProb, queryScaleUb, deSCaleKValue);
        } else if (runInfo.s2RealSize < 128) {
            ProcessVec1Vf<T, INPUT_T, false, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, false>(
            stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
            attenMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
            constInfo.keepProb, queryScaleUb, deSCaleKValue);
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, INPUT_T, true, s1BaseSize, s2BaseSize, EQ_128, hasAtten, false>(
            stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
            attenMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
            constInfo.keepProb, queryScaleUb, deSCaleKValue);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, false>(
            stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
            attenMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
            constInfo.keepProb, queryScaleUb, deSCaleKValue);
        } else if (runInfo.s2RealSize < 128) {
            ProcessVec1Vf<T, INPUT_T, true, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, false>(
            stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
            attenMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
            constInfo.keepProb, queryScaleUb, deSCaleKValue);
        }
    }
    bmm1ResBuf.SetCrossCore();
    if constexpr (hasAtten) {
        this->attenMaskInQue[runInfo.taskIdMod2].template FreeTensor(attenMaskUb);
    }

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    LocalTensor<INPUT_T> mm2AL1Tensor;
    mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    if (likely(runInfo.halfS1RealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) * (runInfo.s1RealSize - runInfo.halfS1RealSize)], stage1CastTensor,
            {s2BaseSize / 32, (uint16_t)runInfo.halfS1RealSize, (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
            (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
    }
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0) {
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        GetDerived()->SoftmaxDataCopyOut(runInfo, constInfo, sumUb, maxUb);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline int64_t BSABlockVecBase<TEMPLATE_BASE_ARGS>::ComputeOffsetForSoftmax(
    RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx)
{
    return vec2S1Idx * runInfo.vec2S1BaseSize;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2OnUb(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer> &constInfo) {
    if (unlikely(runInfo.vec2S1BaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;

    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
    float deSCaleVValue = 1.0f;
    deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        LocalTensor<T> pScaleUb;
        float deSCalePreVValue = 1.0f;
        if constexpr (isInfer) {
            if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) & 1) {   // 7：KV基本块大小128，按照256分块计算deScaleKv偏移
                deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
            } else {
                deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
            }
        } else {
            if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount) & 1) {   // 7：KV基本块大小128，按照256分块计算deScaleKv偏移
                deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
            } else {
                deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
            }
        }
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == 1)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    deSCaleVValue, deSCalePreVValue);
            } else {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    deSCaleVValue, deSCalePreVValue);             
            }
        } else {
            if (unlikely(runInfo.s2LoopCount == 1)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    deSCaleVValue, deSCalePreVValue);            
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    deSCaleVValue, deSCalePreVValue);
            }
        }
    }
    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
        }
        GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2NoGlobalUpdate(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer> &constInfo, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
    int64_t vec2CalcSize) {
    LocalTensor<INPUT_T> vec2ResUb = this->stage2OutBuf.template Get<INPUT_T>()[runInfo.taskIdMod2 * vec2CalcSize];
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[runInfo.taskIdMod2]);
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    DivCast<T, INPUT_T, dTemplateAlign64>(vec2ResUb, bmm2ResBuf.template GetTensor<T>(), sumUb, runInfo.vec2S1RealSize);
    bmm2ResBuf.SetCrossCore();
    Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, 0);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[runInfo.taskIdMod2]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2DSplit(
    GlobalTensor<T> &mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo) {
    // bmm2 result is on GM and global update data on UB
    runInfo.vec2S1BaseSize = 8192 / constInfo.dBasicBlock;
    int64_t vec2LoopLimit = CeilDiv(runInfo.halfS1RealSize, runInfo.vec2S1BaseSize);
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    LocalTensor<T> bmm2Ub = mm2InBuf.template Get<T>();

    event_t mte2ToV = static_cast<event_t>(tPipe->FetchEventID(HardEvent::MTE2_V));
    event_t vToMte2 = static_cast<event_t>(tPipe->FetchEventID(HardEvent::V_MTE2));
    event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    float deSCaleVValue;
    deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
    for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; vec2S1Idx++) {
        runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
        if (vec2S1Idx == vec2LoopLimit - 1) {
            runInfo.vec2S1RealSize = runInfo.halfS1RealSize - vec2S1Idx * runInfo.vec2S1BaseSize;
        }

        int64_t vec2CalcSize = runInfo.vec2S1RealSize * constInfo.dBasicBlock;
        // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
        int64_t mm2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * dTemplateAlign64;
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        if (constInfo.dSizeV == dTemplateAlign64) {
            DataCopy(bmm2Ub, mmRes[constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * (int64_t)dVTemplateType + mm2ResInnerOffset], vec2CalcSize);
        } else {
            DataCopyParams dataCopyParams;
            DataCopyPadParams dataCopyPadParams;
            dataCopyParams.blockCount = runInfo.vec2S1RealSize;
            dataCopyParams.dstStride = (constInfo.dBasicBlock - constInfo.dSizeV) * sizeof(T) / blockBytes;
            dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T);
            dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
            DataCopyPad(bmm2Ub, mmRes[constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * (int64_t)dVTemplateType + mm2ResInnerOffset],
                    dataCopyParams, dataCopyPadParams);
        }

        // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
        int64_t vec2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dBasicBlock;
        if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            DataCopy(vec2ResUb, this->vec2ResGm[runInfo.multiCoreIdxMod3][constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * constInfo.dBasicBlock + vec2ResInnerOffset], vec2CalcSize);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        if (unlikely(runInfo.s2LoopCount == 0)) {
            DataCopy(vec2ResUb, bmm2Ub, vec2CalcSize);
        } else {
            int64_t vec2ExpBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
            float deSCalePreVValue = 1.0f;
            if constexpr (isInfer) {
                if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) & 1) {
                    deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                } else {
                    deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                }
            } else {
                if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount) & 1) {
                    deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                } else {
                    deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                }
            }
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[vec2ExpBufOffset];
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                if (unlikely(runInfo.s2LoopCount == 1)) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, true>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                }
            } else {
                int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                if (unlikely(runInfo.s2LoopCount == 1)) {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, true>(vec2ResUb, bmm2Ub,
                        vec2ResUb, expUb, expUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(vec2ResUb, bmm2Ub,
                        vec2ResUb, expUb, expUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                }
            }
        }

        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                LastDivNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock, deSCaleVValue);
            }
            GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
        } else if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            DataCopy(this->vec2ResGm[runInfo.multiCoreIdxMod3][constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * constInfo.dBasicBlock + vec2ResInnerOffset], vec2ResUb, vec2CalcSize);
        }
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2(
    mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo)
{
    bmm2ResBuf.WaitCrossCore();

    if constexpr (bmm2Write2Ub) {
        ProcessVec2OnUb(bmm2ResBuf, runInfo, constInfo);
    } else if constexpr (splitD) {
        GlobalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        ProcessVec2DSplit(mmRes, runInfo, constInfo);
    } else {
        // bmm2 result is on GM and global update data on UB
        runInfo.vec2S1BaseSize = 8192 / dTemplateAlign64;
        int64_t vec2LoopLimit = CeilDiv(runInfo.halfS1RealSize, runInfo.vec2S1BaseSize);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        LocalTensor<T> bmm2Ub = mm2InBuf.template Get<T>();
        GlobalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

        event_t mte2ToV = static_cast<event_t>(tPipe->FetchEventID(HardEvent::MTE2_V));
        event_t vToMte2 = static_cast<event_t>(tPipe->FetchEventID(HardEvent::V_MTE2));
        float deSCaleVValue;
        deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
        for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; vec2S1Idx++) {
            runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
            if (vec2S1Idx == vec2LoopLimit - 1) {
                runInfo.vec2S1RealSize = runInfo.halfS1RealSize - vec2S1Idx * runInfo.vec2S1BaseSize;
            }

            int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
            // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
            int64_t mm2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV;
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            if (constInfo.dSizeV == dTemplateAlign64) {
                DataCopy(bmm2Ub, mmRes[constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * constInfo.dSizeV + mm2ResInnerOffset], vec2CalcSize);

            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = runInfo.vec2S1RealSize;
                dataCopyParams.dstStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T) / blockBytes;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
                DataCopyPad(bmm2Ub, mmRes[constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * constInfo.dSizeV + mm2ResInnerOffset],
                            dataCopyParams, dataCopyPadParams);
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
            LocalTensor vec2ResInner = vec2ResUb[vec2S1Idx * runInfo.vec2S1BaseSize * dTemplateAlign64];

            if (unlikely(runInfo.s2LoopCount == 0)) {
                DataCopy(vec2ResInner, bmm2Ub, vec2CalcSize);
            } else {
                int64_t vec2ExpBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                float deSCalePreVValue = 1.0f;
                if constexpr (isInfer) {
                    if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) & 1) {
                        deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                    } else {
                        deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                    }
                } else {
                    if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount) & 1) {
                        deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                    } else {
                        deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                    }
                }
                LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[vec2ExpBufOffset];
                if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                    if (unlikely(runInfo.s2LoopCount == 1)) {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                            vec2ResInner, bmm2Ub, vec2ResInner, expUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                            deSCaleVValue, deSCalePreVValue);
                    } else {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                            vec2ResInner, bmm2Ub, vec2ResInner, expUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                            deSCaleVValue, deSCalePreVValue);
                    }
                } else {
                    int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                    if (unlikely(runInfo.s2LoopCount == 1)) {
                        FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(vec2ResInner, bmm2Ub,
                            vec2ResInner, expUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                            deSCalePreVValue);
                    } else {
                        FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResInner, bmm2Ub,
                            vec2ResInner, expUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                            deSCalePreVValue);
                    }   
                }
            }

            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
                if (unlikely(runInfo.s2LoopCount == 0)) {
                    int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                        vec2ResInner, vec2ResInner, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue);
                }
                GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResInner, vec2S1Idx, vec2CalcSize);
            }
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }
    return;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb,
    int64_t vec2S1Idx, RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, int64_t dSizeAligned64)
{
    if constexpr (isInfer && hasAtten) {
        if ((!constInfo.isRowInvalid ||
            attenMaskInfoPtr->compressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE))) {
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
            if (checkValue == NEGATIVE_MIN_VALUE_FP32) {
                isRowInvalidNeedUpdate = true;
                break;
            }
        }
        if (isRowInvalidNeedUpdate) {
            RowInvalidUpdateVF<float>(vec2ResUb, maxTensor,  runInfo.vec2S1RealSize, constInfo.dSizeV, static_cast<uint32_t>(dSizeAligned64));
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::MlaTransposeDataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<OUTPUT_T> &attenOut)
{
    int64_t s1DealSize = runInfo.vec2S1RealSize;
    int64_t headSize = 0;
    int64_t attenOutOffset = constInfo.dSizeV;
    int64_t headUbOffset = 0, headGmOffset = 0;
    int64_t curGIdx = runInfo.goIdx, curS1Idx = runInfo.s1oIdx;

    if (constInfo.gSize == 128) { // G为128时，基本块位于同一S1行
        curGIdx = (curS1Idx % 2 == 0) ? curGIdx : 64; // 64 s1Base基本块
        curS1Idx /= 2;
    } else if (constInfo.gSize <= 32) { // G<=32时，每64/G行为一个基本块
        curS1Idx *= (64 / constInfo.gSize); // 64 s1Base基本块
    }

    if (constInfo.subBlockIdx == 1) {
        int64_t firstCurGIdx = curGIdx;
        curGIdx = (firstCurGIdx + s1DealSize) % constInfo.gSize;
        curS1Idx += (firstCurGIdx + s1DealSize) / constInfo.gSize;
    }

    DataCopyExtParams dataCopyParams;
    if (curGIdx != 0 && constInfo.gSize != 1) { // 首块
        headSize = curGIdx + s1DealSize < constInfo.gSize ? s1DealSize : constInfo.gSize - curGIdx;
        headUbOffset = headSize * constInfo.dSizeV;
        headGmOffset = constInfo.dSizeV - curGIdx * constInfo.t1Size * constInfo.dSizeV;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.t1Size * constInfo.dSizeV - constInfo.dSizeV) * sizeof(OUTPUT_T);
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParams.blockCount = headSize;

        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
    }

    s1DealSize -= headSize; // 中间块 & 尾块
    int64_t blocks = CeilDiv(s1DealSize, constInfo.gSize);
    bool hasTail = s1DealSize % constInfo.gSize != 0;
    for (int64_t i = 0; i < blocks; i++) {
        attenOutOffset = i * constInfo.dSizeV;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.t1Size * constInfo.dSizeV - constInfo.dSizeV) * sizeof(OUTPUT_T);
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
        if (i == blocks - 1 && hasTail) {
            dataCopyParams.blockCount = s1DealSize % constInfo.gSize;
        } else {
            dataCopyParams.blockCount = constInfo.gSize;
        }

        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + headGmOffset + attenOutOffset],
            attenOut[headUbOffset + i * constInfo.gSize * constInfo.dSizeV], dataCopyParams);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::MlaTranspose2DataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<OUTPUT_T> &attenOut)
{
    int64_t s1DealSize = runInfo.vec2S1RealSize;
    int64_t curGIdx = runInfo.sOuterOffset / constInfo.s1Size;
    int64_t curS1Idx = runInfo.sOuterOffset % constInfo.s1Size;
    bool hasHeadBlock = curS1Idx != 0;
    int headBlock = hasHeadBlock ? constInfo.s1Size - curS1Idx : 0;
    int gCount = hasHeadBlock ? (runInfo.vec2S1BaseSize - headBlock) / constInfo.s1Size : runInfo.vec2S1BaseSize / constInfo.s1Size;
    bool hasTailBlock = (runInfo.vec2S1BaseSize - headBlock) % constInfo.s1Size;
    int tailBlock = hasTailBlock ? runInfo.vec2S1BaseSize - gCount * constInfo.s1Size - headBlock : 0;
    uint32_t attenOutUbOffset = 0;

    if (curS1Idx != 0) { // 首块
        DataCopyExtParams dataCopyParams;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        dataCopyParams.blockLen = (constInfo.s1Size - curS1Idx) * constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParams.blockCount = 1;
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
        attenOutUbOffset += (constInfo.s1Size - curS1Idx) * constInfo.dSizeV;
    }
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = gCount; // 处理多少个G
    dataCopyParams.blockLen = constInfo.s1Size * constInfo.dSizeV * sizeof(OUTPUT_T); // 一个S1*D的大小
    dataCopyParams.srcStride = 0;                                                                    // 连读
    dataCopyParams.dstStride = (constInfo.bSize - 1) * constInfo.s1Size * constInfo.dSizeV * sizeof(OUTPUT_T); // 跳写
    runInfo.attentionOutOffset += int(hasHeadBlock) * constInfo.bSize * constInfo.s1Size * constInfo.dSizeV - curS1Idx * constInfo.dSizeV;
    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut[attenOutUbOffset], dataCopyParams);
    attenOutUbOffset += dataCopyParams.blockCount * (constInfo.s1Size * constInfo.dSizeV);

    if (hasTailBlock) { // 尾块单独一条DataCopy指令
        DataCopyExtParams dataCopyParamsTail;
        dataCopyParamsTail.blockCount = 1;
        dataCopyParamsTail.blockLen = tailBlock * constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParamsTail.srcStride = 0;
        dataCopyParamsTail.dstStride = 0;
        runInfo.attentionOutOffset += gCount * constInfo.bSize * constInfo.s1Size * constInfo.dSizeV;
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut[attenOutUbOffset], dataCopyParamsTail);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::Bmm2DataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (splitD) {
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    if constexpr (!IsSameType<INPUT_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        RowInvalid(vec2ResUb, vec2S1Idx, runInfo, constInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    } else {
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
        attenOut = vec2ResUb; 
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2S1RealSize;

    int64_t attenOutOffset = constInfo.dSizeV;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        attenOutOffset = constInfo.n2GDv;
        if constexpr (isInfer) {
            if (constInfo.isPfaGS1Merge) {
                attenOutOffset = 0;
                dataCopyParams.blockLen *= constInfo.gSize;
                dataCopyParams.blockCount /= constInfo.gSize;
            } else if (constInfo.isGqa) {
                attenOutOffset = constInfo.dSizeV;
            }
        }
    } else {
        if (constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH)) {
            attenOutOffset = constInfo.n2GDv;
            if constexpr (isInfer) {
                if (constInfo.isPfaGS1Merge) {
                    attenOutOffset = 0;
                    dataCopyParams.blockLen *= constInfo.gSize;
                    dataCopyParams.blockCount /= constInfo.gSize;
                } else if (constInfo.isGqa) {
                    attenOutOffset = constInfo.dSizeV;
                }
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            attenOutOffset = constInfo.bN2GDv;
        }
        if constexpr (isInfer) {
            if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND) ||
 	            constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::NTD_TND)) {
                attenOutOffset = constInfo.n2GDv;
            }
        }
    }

    if constexpr (isInfer) {
        if (constInfo.isPfaGS1Merge && dSizeAligned64 - constInfo.dSizeV != 0 && (constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH) || constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_TND))) {
            for(int64_t i = 0; i < runInfo.vec2S1BaseSize / constInfo.gSize; i++){
                attenOutOffset = i * constInfo.dSizeV * constInfo.gSize * constInfo.n2Size;
                dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
                dataCopyParams.blockCount = constInfo.gSize;
                dataCopyParams.dstStride = 0;
                DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + attenOutOffset],
                    attenOut[i * constInfo.gSize * dSizeAligned64], dataCopyParams);
            }
        } else {
            DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
                attenOut, dataCopyParams);
        }         
    } else {
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams);
    }
}
TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxSumBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxSumBuf[2], 256); // [64, 1]
    tPipe->InitBuffer(maxBrdcst, 1, 2048); // [64, 8]
    tPipe->InitBuffer(sumBrdcst, 1, 2048); // [64, 8]
    tPipe->InitBuffer(softmaxMaxBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxMaxBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxMaxBuf[2], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[2], 256); // [64, 1]
    tPipe->InitBuffer(preLoopMaxBuf, 256); // [64, 1]
    tPipe->InitBuffer(preLoopSumBuf, 256); // [64, 1]
    tPipe->InitBuffer(firstLoopSumBuf, 256); // [64, 1]
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer> &constInfo)
{
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    if constexpr (!bmm2Write2Ub) {
        tPipe->InitBuffer(mm2InBuf, 32768); // bmm2结果在Gm，vector2开启多层循环，每次处理32KB
    }
    if constexpr (s2BaseSize == 256) { // s1BaseSize = 128
        if constexpr (s1BaseSize == 128) { // s1BaseSize = 128 s2BaseSize = 256
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            SoftmaxInitBuffer();
            tPipe->InitBuffer(stage1OutQue[0], 1, 16896);  // (32 + 1) * (256 / 32) * 64
            tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
            if constexpr (hasAtten) {
                tPipe->InitBuffer(attenMaskInQue[0], 1, 16384); // 256 * 64
            }
        } else { // s1BaseSize = 64 s2BaseSize = 256
            SoftmaxInitBuffer();
            tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
            tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T));
            tPipe->InitBuffer(stage1OutQue[0], 1, 16896);
            tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
            if constexpr (hasAtten) {
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
            tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
        }
        if constexpr (bmm2Write2Ub) {
            // 小于128Bmm2结果和Vec2结果都在UB
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        } else if constexpr (dTemplateAlign64 <= 256) {
            // bmm2结果在Gm，Vector2结果在UB，开启多层循环，每次处理32KB
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        } else {
            // bmm2结果在Gm，Vector2结果也在Gm，开启多层循环，每次处理32KB
            tPipe->InitBuffer(stage2OutBuf, 32768);
        }
        tPipe->InitBuffer(stage1OutQue[0], 1, 8320);
        tPipe->InitBuffer(stage1OutQue[1], 1, 8320);
    }
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)], 128);
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], 64);
    LocalTensor<uint8_t> vselrIndexesTensor =
        vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
    for (uint32_t i = 0; i < 128; i++) {
        vselrIndexesTensor.SetValue(i, i << 1);
    }
    vselrIndexesTensor =
        vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
    for (uint32_t i = 0; i < 64; i++) {
        vselrIndexesTensor.SetValue(i, i << 2);
    }

    GetDerived()->InitUniqueLocalBuffer(constInfo);

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}


TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void BSABlockVecBase<TEMPLATE_BASE_ARGS>::GetExtremeValue(
    T &negativeScalar, T &positiveScalar)
{
    uint16_t tmp1 = NEGATIVE_MIN_VALUE_FP16;
    negativeScalar = *((half *)&tmp1);
}

TEMPLATES_DEF
class BSABlockVecDummy {
public:
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr TPosition bmm2OutPos = GetC2Position();
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;

    __aicore__ inline BSABlockVecDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, 
        ConstInfo<isInfer> &constInfo) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const QuantBlockSparseAttentionTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
        AttenMaskInfo &attenMaskInfo) {};
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
        __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,__gm__ uint8_t *prefix,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
        __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink,
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset,
        uint32_t aicIdx, ConstInfo<isInfer> &constInfo) {}

    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer> &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer> &constInfo) {}

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer> &constInfo) {}
};
}
#endif // QUANT_BLOCK_SPARSE_ATTENTION_BLOCK_VEC_BASE_H_
