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
 * \file fia_block_vec_fullquant_mx.h
 * \brief
 */
#ifndef FLASH_ATTENTION_FULLQUANT_MX_BLOCK_VEC_H_
#define FLASH_ATTENTION_FULLQUANT_MX_BLOCK_VEC_H_

#include "kernel_operator.h"

#include "flash_attention_score_common_regbase.h"
#include "adv_api/activation/softmax.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flash_decode.h"
#include "fia_public_define_arch35.h"
#include "../vector_common.h"
#include "../memory_copy_arch35.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

/* ============确定bmm2ResBuffer的类型============= */
template <bool useDn, bool isFp8>
struct Bmm2ResBuffSel {
    using Type =
        std::conditional_t<(useDn && isFp8), BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                           BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>>;
};

namespace BaseApi {

template <typename INPUT_T, typename T, typename OUTPUT_T, ImplModeEnum implMode = ImplModeEnum::AA_HIGH_PRECISION,
          LayOutTypeEnum layout = LayOutTypeEnum::None, LayOutTypeEnum outLayout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasAtten = false, bool hasDrop = false, bool hasRope = false, bool isPa = false, bool isFd = false,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMxBlockVec {
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
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2;  // C1 C1 C2
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool USE_DN = useDn;

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using attenMaskGmType = typename std::conditional<HAS_MASK, GlobalTensor<uint8_t>, int8_t>::type;

    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;

    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;

    using flashdecodeGmType = typename std::conditional<isFd, GlobalTensor<float>, int8_t>::type;

    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;


    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // gm
    TPipe *tPipe;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;

    attenMaskGmType attenMaskGmInt;

    postQuantGmType postQuantScaleGm;
    postQuantGmType postQuantOffsetGm;
    postQuantBf16GmType postQuantScaleBf16Gm;
    postQuantBf16GmType postQuantOffsetBf16Gm;
    quantGmType pScaleGm;

    vec2ResGmType vec2ResGm[3];

    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    // ub
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf[PRELOAD_N + 1];
    TBuf<> preLoopMaxBuf;
    TBuf<> preLoopSumBuf;
    TBuf<> firstLoopSumBuf;
    TBuf<> vselrIndexesBuf[4];
    TBuf<> mm2InBuf;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;

    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECIN, 1> postQuantScaleQue;; // postQuant
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue;; // postQuant

    ConstInfoX &constInfo;
    T negativeFloatScalar;
    T positiveFloatScalar;
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;

    // ==================== Functions ======================
    __aicore__ inline FAFullQuantMxBlockVec(ConstInfoX &constInfo) : constInfo(constInfo) {};

    __aicore__ inline void InitVecBlock(TPipe *pipe)
    {
        tPipe = pipe;
        GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
        InitLocalBuffer();
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *pScale, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);

        if (constInfo.actualSeqLenSize != 0) {
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        }

        if constexpr (HAS_MASK) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }
        if constexpr (isFp8) {
            if (pScale != nullptr) {
                pScaleGm.SetGlobalBuffer((__gm__ float *)pScale);
                constInfo.pScale = this->pScaleGm.GetValue(0);
            } else {
                constInfo.pScale = 1.0f;
            }
        }

        if constexpr (splitD) {
            int64_t vec2ResultSize = s1BaseSize * constInfo.dBasicBlock;
            vec2SubBlockOffset = constInfo.subBlockIdx * vec2ResultSize >> 1;
            int64_t prevCoretotalOffset = constInfo.aicIdx * 3 * vec2ResultSize; // 3为preload次数
            vec2ResGm[0].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset);
            vec2ResGm[1].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset + vec2ResultSize);
            vec2ResGm[2].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset + 2 * vec2ResultSize);
            workspace = workspace + constInfo.coreNum * 3 * vec2ResultSize * sizeof(T);
        }

        if constexpr (isFd) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize + constInfo.logSumExpSize);
        }
    }

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfoX runInfo, uint32_t subLoop)
    {
        ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo, subLoop);
    }

    __aicore__ inline void ClearOutput()
    {
        // if (this->tilingData->initOutputParams.needInit == 1) {
        //     InitOutputSingleCore(constInfo);
        //     if (constInfo.isSoftmaxLseEnable) {
        //         SyncAll();
        //         InitLseOutputSingleCore(constInfo);
        //     }
        // }
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        // auto &initParams = this->tilingData->initOutputParams;
        // uint32_t tailSize = initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize;
        // uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
        // if constexpr (POST_QUANT) {
        //     InitOutput<half>(this->attentionOutInitGm[constInfo.aivIdx * initParams.singleCoreSize / 2],
        //                      singleInitOutputSize / 2, 0.0); // 将内容一分为二
        // } else {
        //     InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize],
        //                          singleInitOutputSize, 0.0);
        // }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        // int64_t coreNum = GetBlockNum() * GetTaskRation();
        // auto &initParams = this->tilingData->initOutputParams;
        // if (coreNum != 0 && constInfo.aivIdx < coreNum) {
        //     int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        //     if (constInfo.aivIdx == coreNum - 1) {
        //         singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        //     }
        //     InitOutput<float>(softmaxLseGm[constInfo.aivIdx * (initParams.totalSoftMaxLseOutputSize / coreNum)],
        //                       singleCoreLseSize, 3e+99); // 3e+99: set the value of invalid batch to inf
        // }
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo)
    {
        bmm1ResBuf.WaitCrossCore();
        LocalTensor<uint8_t> attenMaskUb;

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];

        auto expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>()[0];
        int64_t stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0;

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
        if (unlikely(runInfo.isFirstSInnerLoop)) {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, false, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actualSingleProcessSInnerSizeAlign, runInfo.actualSingleProcessSInnerSize,
                static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, false);
        } else {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actualSingleProcessSInnerSizeAlign, runInfo.actualSingleProcessSInnerSize,
                static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, false);
        }
        bmm1ResBuf.SetCrossCore();

        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        //-------------------------Data copy to l1-------------------------
        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

        if (runInfo.actualSingleProcessSInnerSize > vec1S2CopyLenDn) {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.actualSingleProcessSInnerSizeAlign], stage1CastTensor,
                     {vec1S2CopyCountDn, vec1S2CopyLenDn, 1,
                      static_cast<uint16_t>(runInfo.actualSingleProcessSInnerSizeAlign - vec1S2CopyLenDn)});
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.actualSingleProcessSInnerSizeAlign + vec1S2strideDn],
                     stage1CastTensor[vec1ResOffsetDn],
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actualSingleProcessSInnerSizeAlign - vec1S2CopyLenDn),
                      static_cast<uint16_t>(s2BaseSize - runInfo.actualSingleProcessSInnerSizeAlign + 1), vec1S2CopyLenDn});
        } else {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.actualSingleProcessSInnerSizeAlign], stage1CastTensor,
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actualSingleProcessSInnerSizeAlign),
                      static_cast<uint16_t>(vec1S2CopyLenDn - runInfo.actualSingleProcessSInnerSizeAlign + 1), 0});
        }

        outputBuf.SetCrossCore();
        //-----------------------------------------------------------------
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
        return;
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (runInfo.isS2SplitCore) {
            if constexpr (isFd) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        } else {
            SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
        }
    }

    // TODO 使用A3 的实现
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        if (!constInfo.isSoftmaxLseEnable) {
            return;
        }
        LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        softmaxLseQueue.template EnQue(lseUb);
        softmaxLseQueue.DeQue<float>();

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = runInfo.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
            uint64_t bN2Offset = prefixBS1 * constInfo.n2Size * constInfo.gSize + runInfo.n2Idx * constInfo.gSize;
            DataCopySoftmaxLseTND(softmaxLseGm, lseUb, bN2Offset, runInfo.vecMbaseIdx, runInfo.actVecMSize, constInfo);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
            uint32_t prefixBS1 = runInfo.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
            uint32_t s1Size = runInfo.bIdx == 0U ?
                    actualSeqLengthsGmQ.GetValue(0U) : actualSeqLengthsGmQ.GetValue(runInfo.bIdx) - actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1U);
            uint64_t bN2Offset = prefixBS1 * constInfo.n2Size * constInfo.gSize + runInfo.n2Idx * constInfo.gSize;
            DataCopySoftmaxLseNTD(softmaxLseGm, lseUb, bN2Offset, runInfo.vecMbaseIdx, runInfo.actVecMSize, constInfo, s1Size);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size + runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            DataCopySoftmaxLseBSND(softmaxLseGm, lseUb, bN2Offset, runInfo.vecMbaseIdx, runInfo.actVecMSize, constInfo, actualSeqLengthsGmQ.GetValue(runInfo.bIdx), runInfo.bIdx);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size + runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            DataCopySoftmaxLseBNSD(softmaxLseGm, lseUb, bN2Offset, runInfo.vecMbaseIdx, runInfo.actVecMSize, constInfo, actualSeqLengthsGmQ.GetValue(runInfo.bIdx), runInfo.bIdx);
        }

        softmaxLseQueue.FreeTensor(lseUb);
    }

    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar,
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

    __aicore__ inline void InvalidLineProcess(RunInfoX runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb)
    {
        if (constInfo.softMaxCheckRes) {
            SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(runInfo.actVecMSize), static_cast<uint32_t>(1),
                                              static_cast<uint32_t>(runInfo.actVecMSize), static_cast<uint32_t>(1)};
            bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VALUE_FP32, softmaxShapeInfo);
            if (!res) {
                // constInfo.softMaxCheckRes = false;
            } else {
                if (unlikely(runInfo.isLastS2Loop)) {
                    SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.actVecMSize, this->negativeFloatScalar,
                                        this->positiveFloatScalar);
                }
            }
        }
    }

    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo, uint32_t subLoop)
    {
        bmm1ResBuf.WaitCrossCore();
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;

        //TODO:mask修改
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (HAS_MASK) {
            attenMaskUb = this->attenMaskInQue[runInfo.loop % DB].template AllocTensor<uint8_t>();
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize,  runInfo); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();

        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf.template Get<float>();; // Loop 0拷贝一次全局, Loop 1用来计算update_mul
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf.template Get<float>();; // 两轮循环前的sum值
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf.template Get<float>(); // Loop 0的sum值传给loop 1用来计算更新的sum值

        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

        int64_t stage1Offset = 0;
        stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();

        static constexpr int32_t mxfp8s2BaseSize = s2BaseSize / 2;
        uint32_t mxfp8s2RealSize = runInfo.actualSingleProcessSInnerSize;
        if (runInfo.actualSingleProcessSInnerSize > 256) {
            mxfp8s2RealSize = subLoop == 0 ? mxfp8s2BaseSize : runInfo.actualSingleProcessSInnerSize - mxfp8s2BaseSize;
        }
        if (runInfo.isFirstSInnerLoop) {
            if (likely(runInfo.actualSingleProcessSInnerSize == 128)) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize / 2, EQ_128, HAS_MASK, pseMode,
                              hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actualSingleProcessSInnerSize,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                    negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue, constInfo.pScale);
            } else if (runInfo.actualSingleProcessSInnerSize <= 64) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize / 2, GT_0_AND_LTE_64, HAS_MASK,
                              pseMode, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actualSingleProcessSInnerSize,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                    negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue, constInfo.pScale);
            } else if (runInfo.actualSingleProcessSInnerSize < 128) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize / 2, GT_64_AND_LTE_128, HAS_MASK,
                              pseMode, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actualSingleProcessSInnerSize,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                    negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue, constInfo.pScale);
            } else {
                if constexpr (mxfp8s2BaseSize  == 256) {
                    ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize / 2, GT_128_AND_LTE_256, HAS_MASK,
                                  pseMode, hasDrop, false, false, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop,
                        runInfo.actVecMSize, runInfo.actualSingleProcessSInnerSize,
                        pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                        negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue, constInfo.pScale);
                }
            }
        } else {
            if (likely(runInfo.actualSingleProcessSInnerSize == 128)) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize / 2, EQ_128, HAS_MASK, pseMode,
                              hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actualSingleProcessSInnerSize,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                    negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue, constInfo.pScale);
            } else if (runInfo.actualSingleProcessSInnerSize <= 64) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize / 2, GT_0_AND_LTE_64, HAS_MASK,
                              pseMode, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actualSingleProcessSInnerSize,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                    negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue, constInfo.pScale);
            } else if (runInfo.actualSingleProcessSInnerSize < 128) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize / 2, GT_64_AND_LTE_128, HAS_MASK,
                              pseMode, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actualSingleProcessSInnerSize,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                    negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue, constInfo.pScale);
            } else {
                if constexpr (mxfp8s2BaseSize == 256) {
                    ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize / 2, GT_128_AND_LTE_256, HAS_MASK,
                                  pseMode, hasDrop, false, false, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop,
                        runInfo.actMSize, runInfo.actualSingleProcessSInnerSize,
                        pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                        negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue, constInfo.pScale);
                }
            }
        }
        bmm1ResBuf.SetCrossCore();
        if constexpr (HAS_MASK) {
            this->attenMaskInQue[runInfo.loop % DB].template FreeTensor(attenMaskUb);
        }

        // ===================DataCopy to L1 ====================
        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        LocalTensor<INPUT_T> mm2AL1Tensor;

        mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        //TODO:写到L1的地址需要修改
        if (likely(runInfo.actVecMSize != 0)) {
            int64_t dstOffset = constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) * (runInfo.actMSize - runInfo.actVecMSize) + mxfp8s2BaseSize * s1BaseSize * subLoop;
            DataCopy(mm2AL1Tensor[dstOffset],
                     stage1CastTensor,
                     {s2BaseSize / 2 / 32, (uint16_t)runInfo.actVecMSize,
                      (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(s1BaseSize - runInfo.actVecMSize)});
        }
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
        //TODO:只有第二轮结束才进行第一轮C2
        if (((runInfo.actualSingleProcessSInnerSize > 256) && (subLoop % 2 == 1) )|| (runInfo.actualSingleProcessSInnerSize <= 256)) {
            outputBuf.SetCrossCore();
        }
        // TODO: 修改这里
        // if (runInfo.isFirstSInnerLoop) {
        //     UpdateExpSumAndExpMaxMxFp8<T, false>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize, subLoop, preLoopMaxUb, preLoopSumUb, firstLoopSumUb);
        // } else {
        //     UpdateExpSumAndExpMaxMxFp8<T, true>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize, subLoop, preLoopMaxUb, preLoopSumUb, firstLoopSumUb);
        // }
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
            this->InvalidLineProcess(runInfo, sumUb, maxUb);
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
        float deSCaleVValue = 1.0f;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        if (unlikely(runInfo.isFirstSInnerLoop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;

            float deSCalePreVValue = 1.0f;
            if (!runInfo.isLastS2Loop) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64, 1.0,
                    1.0);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstSInnerLoop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2CastAndCopyOut(RunInfoX &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, uint32_t mStartVec,
                                              uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        if constexpr (splitD) {
            dSizeAligned64 = constInfo.dBasicBlock;
        }

        attenOut.SetAddr(vec2ResUb.address_);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
            int64_t vec2MaxBufOffset = mStartVec;
            LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[vec2MaxBufOffset];
            InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, mDealSize,
                                                    dSizeAligned64, this->negativeFloatScalar, 0.0);
        }
        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        } else {
            PostQuant(runInfo, attenOut, vec2ResUb, mStartVec, mDealSize, dSizeAligned64);
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfoX &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                     LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec,  int64_t mDealSize, int64_t dSizeAligned64)
    {
        return;
    }

    // mStartVec： [0, runInfo.vecMsize)
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,  uint32_t mStartVec, uint32_t mDealSize)
    {
        if constexpr (isFd) {
            if (runInfo.isS2SplitCore) {
                Bmm2FDDataCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2CastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2CastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            if (!constInfo.isRowInvalid ||
                constInfo.sparseMode != static_cast<uint8_t>(SparseMode::DEFAULT_MASK)) {
                return;
            }

            int64_t vec2MaxBufOffset = mStartVec;
            LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[vec2MaxBufOffset];
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
            bool isRowInvalidNeedUpdate = false;
            for (uint32_t i = 0; i < mDealSize; i++) {
                float maxValue = maxTensor.GetValue(i);
                uint32_t checkValue = *(uint32_t *)&maxValue;
                if (checkValue == NEGATIVE_MIN_VALUE_FP32) {
                    isRowInvalidNeedUpdate = true;
                    break;
                }
            }
            if (isRowInvalidNeedUpdate) {
                if constexpr (!POST_QUANT) {
                    RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
                                              static_cast<uint32_t>(dSizeAligned64));
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo.dSizeV, sizeof(float)); // w8后量化后的处理长度
                    RowInvalidUpdateVF<float>(*((LocalTensor<float> *)&vec2ResUb), maxTensor, mDealSize,
                                              dSize, dStride);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t vecMIdx,
                                                uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = (uint32_t)(splitD ? constInfo.dBasicBlock : dTemplateAlign64)
        };
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.n2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};

        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(
                constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSizeV,
                actualSeqLengthsGmQ, constInfo.actualSeqLenSize, 0, 0);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(
                constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSizeV,
                actualSeqLengthsGmQ, constInfo.actualSeqLenSize, 0, 0);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NBSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGBSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
                                              constInfo.s1Size, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV,
                                              actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV,
                                              actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
        LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        sumBrdcst.template EnQue(sumOutTensor);
        sumBrdcst.template DeQue<float>();
        DataCopy(softmaxFDSumGm[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst.template FreeTensor(sumOutTensor);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        maxBrdcst.template EnQue(maxOutTensor);
        maxBrdcst.template DeQue<float>();
        DataCopy(softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst.template FreeTensor(maxOutTensor);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        // 是否要改成halfMRealSize
        int64_t gmOffset = runInfo.faTmpOutWsPos * s1BaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // flashDecodeS2Idx?nBufferStartM?
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfoX &runInfo, LocalTensor<MM2_OUT_T> &vec2ResUb,
        uint32_t mStartVec, uint32_t mDealSize)
    {
        LocalTensor<T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        if constexpr (splitD) {
            dSizeAligned64 = constInfo.dBasicBlock;
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.loop % DB]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.loop % DB]);
        attenOut = vec2ResUb;
        uint64_t gmOffset = runInfo.faTmpOutWsPos * s1BaseSize * constInfo.dSizeV +
            (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm[gmOffset], attenOut, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        if constexpr (bmm2Write2Ub) {
            ProcessVec2OnUb(bmm2ResBuf, runInfo);
        }
        return;
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        tPipe->InitBuffer(softmaxSumBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxSumBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxSumBuf[2], 256); // [64, 1]

        tPipe->InitBuffer(softmaxMaxBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxMaxBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxMaxBuf[2], 256); // [64, 1]

        tPipe->InitBuffer(softmaxExpBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxExpBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxExpBuf[2], 256); // [64, 1]

        tPipe->InitBuffer(preLoopMaxBuf, 256); // [64, 1]
        tPipe->InitBuffer(preLoopSumBuf, 256); // [64, 1]
        tPipe->InitBuffer(firstLoopSumBuf, 256); // [64, 1]

        if constexpr (isFd) {
            tPipe->InitBuffer(maxBrdcst, 1, 2048);    // [64, 8]
            tPipe->InitBuffer(sumBrdcst, 1, 2048);    // [64, 8]
        }
    }

    __aicore__ inline void InitLocalBuffer()
    {
        uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
        if constexpr (!bmm2Write2Ub) {
            tPipe->InitBuffer(mm2InBuf, 32768); // bmm2结果在Gm，vector2开启多层循环，每次处理32KB
        }
        SoftmaxInitBuffer();
        tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        tPipe->InitBuffer(stage1OutQue[0], 1, 16640);  // (32 + 1) * (256 / 32) * 64
        tPipe->InitBuffer(stage1OutQue[1], 1, 16640);
        tPipe->InitBuffer(commonTBuf, 512);
        if constexpr (HAS_MASK) {
            tPipe->InitBuffer(attenMaskInQue[0], 1, 16384); // 256 * 64
        }

        if (constInfo.isSoftmaxLseEnable) {
            // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
            this->tPipe->InitBuffer(softmaxLseQueue, 1, (s1BaseSize >> 1U) * sizeof(float) * 8);
        }
        if constexpr (isFp8) {
            if constexpr (USE_DN) {
                tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)], 256);

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < (256 >> 2); j++) {
                        vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
                    }
                }
            } else {
                tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)], 128); // s2realsize (64, 128]
                tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], 64);  // s2realsize (0, 64]

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
                vselrIndexesTensor.SetValue(0, 0x7f);
                for (int i = 0; i < 128; i++) {
                    vselrIndexesTensor.SetValue(i, i * 2);
                }
                vselrIndexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
                for (int i = 0; i < 64; i++) {
                    vselrIndexesTensor.SetValue(i, i * 4);
                }
            }
        }

        mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

        vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    }

    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar)
    {
        uint16_t tmp1 = NEGATIVE_MIN_VALUE_FP16;
        negativeScalar = *((half *)&tmp1);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
                positiveScalar = *((half *)&tmp2);
            }
        }
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize, RunInfoX &runInfo)
    {
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actualSingleProcessSInnerSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - Align(maskInfo.s2dealNum, 32U)) / 32; //TODO， 这里需要传一个完成的stride，要重构， 新写一个attenmask_gs1.h
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        if (constInfo.s1Size == 1) {
            maskInfo.layout = S1_EQUAL1;
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND || outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            maskInfo.layout = SG;
        } else {
            maskInfo.layout = GS;
        }

        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        if (!IsSkipAttentionmask(maskInfo)) {
            AttentionmaskCopyIn(attenMaskUb, attenMaskGmInt, maskInfo);
        }
        if (!IsSkipAttentionmaskForPre(maskInfo)) {
            AttentionmaskCopyIn(attenMaskUb, attenMaskGmInt, maskInfo, true);
        }
    }

    __aicore__ inline void DealZeroActSeqLen(uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur)
    {
        uint32_t n2Idx = bN2Cur % constInfo.n2Size;
        uint32_t bIdx = bN2Cur / constInfo.n2Size;

        // 对整个batch的结果置0
        if constexpr (POST_QUANT) { // out int8
                // vectorService.DealZeroActSeqLenWithPostQuant(bIdx, n2Idx);
        } else {
            if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
                OffsetCalculator<GmFormat::BSNGD> offsetCalculator;
                offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                      constInfo.dSize, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
                DealActSeqLenIsZero<GmFormat::BSNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
            } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
                OffsetCalculator<GmFormat::BNGSD> offsetCalculator;
                offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                      constInfo.dSize, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
                DealActSeqLenIsZero<GmFormat::BNGSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
            } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NBSD) {
                OffsetCalculator<GmFormat::NGBSD> offsetCalculator;
                offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                      constInfo.dSize, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
                DealActSeqLenIsZero<GmFormat::NGBSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
            } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
                OffsetCalculator<GmFormat::TNGD> offsetCalculator;
                offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize, actualSeqLengthsGmQ,
                                      constInfo.actualSeqLenSize);
                DealActSeqLenIsZero<GmFormat::TNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
            } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
                OffsetCalculator<GmFormat::NGTD> offsetCalculator;
                offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize, actualSeqLengthsGmQ,
                                      constInfo.actualSeqLenSize);
                DealActSeqLenIsZero<GmFormat::NGTD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
            }
        }
    }
};
} // namespace BaseApi
#endif // FLASH_ATTENTION_FULLQUANT_MX_BLOCK_VEC_H_