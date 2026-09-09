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
 * \file quant_flash_attn_block_vec_mxfp8.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_H_

#include "kernel_operator.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "adv_api/activation/softmax.h"
#include "quant_flash_attn_common_def.h"
#if __has_include("../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h")
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#else
#include "../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../common/op_kernel/vector_common.h"
#include "../../common/op_kernel/arch35/util_regbase.h"
#endif

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

// bisheng后端不支持float->bf16标量static_cast, 按位做RNE舍入转换(同chunk_kda_fwd的FloatToType惯例)
__aicore__ inline uint16_t FloatToBf16Bits(float value)
{
    union FloatBits {
        __aicore__ FloatBits() {}
        float f;
        uint32_t u;
    } floatBits;
    floatBits.f = value;
    uint32_t bias = 0x7FFFu + ((floatBits.u >> 16) & 1u);
    return static_cast<uint16_t>((floatBits.u + bias) >> 16);
}

// bisheng同样不支持fp8_e4m3fn->float标量static_cast(实测静默返回0, 尾块P全0的根因), 位级展开:
// e4m3fn: 1s+4e+3m, bias=7; normal=(1.m)*2^(e-7); subnormal=m*2^-9; e=15&m=7为NaN, e=15&m<=6最大448
__aicore__ inline float Fp8E4m3ToFloat(uint8_t code)
{
    union F8Bits {
        __aicore__ F8Bits() {}
        uint32_t u;
        float f;
    } r;
    uint32_t s = (static_cast<uint32_t>(code) >> 7) & 1U;
    uint32_t e = (static_cast<uint32_t>(code) >> 3) & 0xFU;
    uint32_t m = static_cast<uint32_t>(code) & 0x7U;
    if (e == 0xFU && m == 0x7U) {
        r.u = 0x7FC00000U | (s << 31); // NaN
    } else if (e == 0U && m == 0U) {
        r.u = s << 31; // +-0
    } else if (e == 0U) { // subnormal: m*2^-9, 归一化到float
        uint32_t k = (m >= 4U) ? 2U : ((m >= 2U) ? 1U : 0U);
        r.u = (s << 31) | ((118U + k) << 23) | ((m - (1U << k)) << (23U - k));
    } else {
        r.u = (s << 31) | ((e - 7U + 127U) << 23) | (m << 20);
    }
    return r.f;
}

__aicore__ inline bfloat16_t FloatToBf16Rne(float value)
{
    union Bf16Bits {
        __aicore__ Bf16Bits() {}
        uint16_t u;
        bfloat16_t f;
    } bf16Bits;
    bf16Bits.u = FloatToBf16Bits(value);
    return bf16Bits.f;
}

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false, bool isDAligned = true, bool hasVTail = false>
class QuantFlashAttnBlockVecMxfp8 {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = mBaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = mBaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = mBaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr uint32_t s2SplitSize =
        (dTemplateAlign64 == static_cast<uint32_t>(DTemplateType::Aligned256)) ? 128U : 256U;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr bool HAS_V_TAIL = hasVTail;

    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需加同步

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr MaskFormat MASK_LAYOUT = MaskFormat::SG;

    static constexpr bool USE_DN = useDn;

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar_ = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using attenMaskGmType = typename std::conditional<HAS_MASK, GlobalTensor<uint8_t>, int8_t>::type;
    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;

    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;
    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // gm
    TPipe *tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<half> attentionOutInitGm_;
    GlobalTensor<float> softmaxLseGm_;

    GlobalTensor<int32_t> actualSeqLengthsGmQ_;
    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND), ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;
    QSeqParserType *qActSeqLensParser_ = nullptr;

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType &qParser)
    {
        this->qActSeqLensParser_ = &qParser;
    }

    // [VTAIL] AIV侧v_tail指针绑定(B尾块由AIV搬运: 尾片上下文的AIC MTE2 GM->L1实测静默失效)
    __aicore__ inline void SetVTailPtr(__gm__ uint8_t *vTail, __gm__ uint8_t *blockTableTail)
    {
        vTailGmVec_ = (__gm__ bfloat16_t *)vTail;
        vTailBttGm_ = (__gm__ int32_t *)blockTableTail;
    }

    attenMaskGmType attenMaskGmInt_;

    postQuantGmType postQuantScaleGm_;
    postQuantGmType postQuantOffsetGm_;
    postQuantBf16GmType postQuantScaleBf16Gm_;
    postQuantBf16GmType postQuantOffsetBf16Gm_;
    quantGmType pScaleGm_;

    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    // ub
    TBuf<> commonTBuf_; // common的复用空间
    TBuf<> pScaleBuf_;  // P的组scale暂存: subLoop0 [0,128) + subLoop1 [128,256), 显式所有权重叠越界写
    TBuf<> vTailStripUb_; // B尾块分形中转(双缓冲): 2 x (64行x16列bf16=2KB)
    __gm__ bfloat16_t *vTailGmVec_ = nullptr; // AIV侧v_tail裸指针(B尾块AIV搬运)
    __gm__ int32_t *vTailBttGm_ = nullptr;    // AIV侧block_table_tail裸指针
    TQue<QuePosition::VECOUT, 1> stage1OutQue_[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue_[2];
    TBuf<> stage2OutBuf_;
    TEventID mte3ToVId_[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id_[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf_[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf_[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf_[PRELOAD_N + 1];
    TBuf<> preLoopMaxBuf_;
    TBuf<> preLoopSumBuf_;
    TBuf<> firstLoopSumBuf_;
    TBuf<> vselrIndexesBuf_[4];
    TBuf<> mm2InBuf_;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst_;
    TQue<QuePosition::VECOUT, 1> sumBrdcst_;

    TBuf<> lseTmpBuff_;
    TBuf<> vTailPCastBuf_; // 尾列P的bf16 cast暂存区(mBase/2 x 64 x 2B)
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue_;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue_;
    TQue<QuePosition::VECIN, 1> accumOutInputQue_;
    TQue<QuePosition::VECIN, 1> postQuantScaleQue_;  // postQuant
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue_; // postQuant

    const ConstInfoX &constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    bool isSkipMask_{false};
    bool isFullMask_{false};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP32_LN2};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecMxfp8(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *pScale,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        tPipe_ = pipe;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32_LN2;
        this->negativeFloatScalar_ = *((T *)&tmp1);
        if constexpr (USE_DN) {
            UpdateMinCheckValue();
        }

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, attenMask, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *pScale, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        uint64_t actualLenQSize =
            (layout == LayOutTypeEnum::LAYOUT_TND) ? constInfo_.cuSeqLensQSize : constInfo_.seqUsedQSize;
        actualSeqLengthsGmQ_.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr, actualLenQSize);

        if constexpr (HAS_MASK) {
            attenMaskGmInt_.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }
        if constexpr (isFp8) {
            if (pScale != nullptr) {
                pScaleGm_.SetGlobalBuffer((__gm__ float *)pScale);
                pScaleValue_ = this->pScaleGm_.GetValue(0);
            }
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
    }

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfoX runInfo, uint32_t subLoop)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
                outputBuf.SetCrossCore();
            }
            return;
        }

        if constexpr (USE_DN) {
            ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo, subLoop);
        } else {
            ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo, subLoop);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore();
            if (constInfo_.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo_.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.dSizeV;
        if constexpr (POST_QUANT) {
            totalOutputSize /= 2;
        }
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo_.coreNum) - 1) / (2 * constInfo_.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if constexpr (IsSameType<OUT_T, int8_t>::value) {
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOutGm_.GetPhyAddr(0)));
                matmul::InitOutput<half>(attentionOutTmpGm[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
            } else {
                matmul::InitOutput<OUT_T>(attentionOutGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo_.coreNum) - 1) / (2 * constInfo_.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize,
                                      3e+99); // 3e+99: set the value of invalid batch to inf
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo, uint32_t subLoop)
    {
        int64_t maskLine = 0;
        if constexpr (HAS_MASK) {
            maskLine = ComputeMaskLineDN(runInfo, subLoop);
        }
        if (isFullMask_) {
            maskLine = -256;
        }

        uint32_t softmaxBufIdx = runInfo.mloop % (PRELOAD_N + 1);
        uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1);
        int64_t stage1Offset = subLoop % DB;

        LocalTensor<float> sumUb = this->softmaxSumBuf_[softmaxBufIdx].template Get<float>()[0];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[softmaxBufIdx].template Get<float>()[0];
        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf_.template Get<float>()[0];
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf_.template Get<float>()[0];
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf_.template Get<float>()[0];
        LocalTensor<float> expUb = this->softmaxExpBuf_[expBufIdx].template Get<T>()[0];

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();
        // 回滚验证: 恢复原stage1CastTensor[16640]越界视图(11:40的TBuf版本疑破坏A尾scale时序, 待排除)
        constexpr int32_t pScaleByteOffset = 16640; // 16640: softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        float descaleQK = 1.0;
        static constexpr int32_t s2BaseSizeCur = s2BaseSize >> 1;
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2CalcSize = subLoop == 0 ? s2BaseSizeCur : runInfo.actSingleLoopS2Size - s2BaseSizeCur;
        }

        if (unlikely(runInfo.isFirstS2Loop)) {
            if (unlikely(!isSkipMask_)) {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, false, hasAtten, s2BaseSizeCur>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, false, false, s2BaseSizeCur>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            }
        } else {
            if (unlikely(!isSkipMask_)) {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, true, hasAtten, s2BaseSizeCur>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, true, false, s2BaseSizeCur>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            }
        }
        bmm1ResBuf.SetCrossCore();

        this->stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        //-------------------------Data copy to l1-------------------------
        uint64_t pScaleL1Offset = mBaseSize * s2BaseSize; // PScale在L1P的偏移量（单位：元素）
        LocalTensor<fp8_e8m0_t> mm2AScaleL1Tensor = outputBuf.GetTensor<fp8_e8m0_t>(pScaleL1Offset);
        constexpr uint64_t pScaleDataLen = (mBaseSize >> 1) * s2BaseSizeCur / MXFP_GROUP_SIZE;
        constexpr uint16_t pScaleDstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        uint64_t vecOffset = constInfo_.subBlockIdx * pScaleDataLen;
        constexpr uint32_t copyTimes = s2BaseSizeCur / MXFP_GROUP_SIZE / 2;
        if ((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                // PScale s2方向block块大小为32，共8个，L1需满足16x2分形，重复拷贝4次
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor, {4, 1, 0, pScaleDstStride});
            }
        }

        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        int64_t subLoopOffset = s2BaseSizeCur * mBaseSize * subLoop;
        constexpr uint16_t elementSize = 32;
        constexpr uint32_t singleProcessSOuterSize = mBaseSize >> 1;
        constexpr uint32_t actCopyCount = (singleProcessSOuterSize + elementSize - 1) / elementSize;
        constexpr uint32_t actVec0Align32 = mBaseSize >> 1;
        uint32_t s2RealSizeAlign = (((runInfo.actSingleLoopS2Size + 63) >> 6) << 6);
        for (uint32_t i = 0; i < actCopyCount; i++) {
            uint32_t dstOffset = 32 * s2BaseSizeCur * i + subLoopOffset;
            if (constInfo_.subBlockIdx == 1) {
                dstOffset += s2BaseSizeCur * actVec0Align32;
            }
            uint32_t srcOffset = i * (((s2BaseSizeCur >> 2) + 1) << 5);
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor[srcOffset],
                     {4, s2BaseSizeCur / 4, s2BaseSizeCur / 4 + 2, 0});
        }
        //-----------------------------------------------------------------
        this->stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
            (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            outputBuf.SetCrossCore();
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
        return;
    }

    // 尾列P提升为bf16并写L1 P尾区。布局契约参照非量化模板 flash_attn(bf16) 的 P 写L1铁律:
    //   - L1 NZ 分形 = 16列(bf16 C0) x 行, 分形间距 = mBaseSize*32B(即dstStride单位块数 = mBaseSize-actVecM)
    //   - 源 pTailUb 采用分形紧邻布局(每分形 actVecM行 x 16列), srcStride=0
    __aicore__ inline void CastPTailToBf16AndCopyToL1(
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf, LocalTensor<INPUT_T> &pE4m3Ub,
        LocalTensor<fp8_e8m0_t> &pScaleUb, RunInfoX &runInfo, uint32_t kMainSize, uint32_t actVecMSizeAlign16,
        uint32_t subLoop)
    {
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        // 双subblock并发执行, 暂存区按subBlockIdx分半区避免竞态
        constexpr uint32_t pTailUbHalfElems = (mBaseSize >> 1U) * MXFP8_V_TAIL_BOUNDARY;
        LocalTensor<bfloat16_t> pTailUb =
            vTailPCastBuf_.template Get<bfloat16_t>()[constInfo_.subBlockIdx * pTailUbHalfElems];
        // 尾列相对本subLoop起点的列号
        uint32_t tailStartRel = kMainSize - subLoopColsBase(runInfo, kMainSize);
        uint32_t tailLen = runInfo.kvTailLen;
        LocalTensor<uint8_t> pScaleCodeUb = pScaleUb.ReinterpretCast<uint8_t>();
        // PScale布局: 每次ProcessVec1VfMxfp8调用将当前subLoop的组scale写入 subLoop*128 基址处
        // (主体搬运代码 subLoop1 时从 pScaleSubLoop0Tensor[128] 读取印证), 故需加subLoop基址
        constexpr uint32_t scaleSubLoopStride = s2BaseSizeCur / 2; // 128
        // 逐32列组(quant组): e4m3 -> float -> x组scale -> bf16。
        // 对齐契约: 编译器可能将逐字节标量访问合并为4B访问, 若地址非4B对齐将触发硬件misaligned异常
        // (AIV VEC_ERR)。故显式按4B粒度读stage1(4个fp8)/写pTailUb(2个bf16), 且布局保证4B对齐:
        //   stage1: 列块基址2080(4对齐)+row*32(4对齐)+colStart%32(=0, tailStartRel为64对齐)
        //   pTailUb: 分形距272(4对齐)+row*16(4对齐)+列内偶偏移
        LocalTensor<uint32_t> pE4m3U32 = pE4m3Ub.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> pTailU32 = pTailUb.template ReinterpretCast<uint32_t>();
        union {
            uint8_t u8;
            fp8_e4m3fn_t f8;
        } f8Cvt;
        for (uint32_t g = 0; g < (tailLen + MXFP_GROUP_SIZE - 1) / MXFP_GROUP_SIZE; g++) {
            uint32_t colStart = tailStartRel + g * MXFP_GROUP_SIZE;
            for (uint32_t row = 0; row < actVecMSizeAlign16; row++) {
                uint32_t scaleIdx =
                    subLoop * scaleSubLoopStride + (row / 16) * (s2BaseSizeCur / MXFP_GROUP_SIZE) +
                    (colStart / MXFP_GROUP_SIZE);
                uint32_t scaleCode = static_cast<uint32_t>(pScaleCodeUb.GetValue(scaleIdx));
                uint32_t scaleBits = scaleCode << 23; // e8m0 -> float 指数位
                float scaleF;
                *(reinterpret_cast<uint32_t *>(&scaleF)) = scaleBits;
                // stage1源: 块号*2080 + row*32 + 块内偏移, 按4元素(u32)步进
                uint32_t srcBase = (colStart / MXFP_GROUP_SIZE) * 2080U + row * MXFP_GROUP_SIZE +
                                   (colStart % MXFP_GROUP_SIZE);
                for (uint32_t cw = 0; cw < MXFP_GROUP_SIZE / 4; cw++) {
                    uint32_t colBase = g * MXFP_GROUP_SIZE + cw * 4; // 尾列内列号(4列一组)
                    // 无条件读(段内安全), pad列(col>=tailLen)贡献0; 运行时条件避免imm bitcast常量折叠
                    uint32_t word = pE4m3U32.GetValue(srcBase / 4 + cw);
                    uint32_t dstBase = (colBase >> 4) * (actVecMSizeAlign16 * 8) + row * 8 + (colBase & 15) / 2; // 分形=16行x8u32连续
                    uint32_t bits[4] = {0, 0, 0, 0};
                    if (colBase < tailLen) {
                        f8Cvt.u8 = word & 0xFF;
                        bits[0] = FloatToBf16Bits(Fp8E4m3ToFloat(f8Cvt.u8) * scaleF);
                    }
                    if (colBase + 1 < tailLen) {
                        f8Cvt.u8 = (word >> 8) & 0xFF;
                        bits[1] = FloatToBf16Bits(Fp8E4m3ToFloat(f8Cvt.u8) * scaleF);
                    }
                    if (colBase + 2 < tailLen) {
                        f8Cvt.u8 = (word >> 16) & 0xFF;
                        bits[2] = FloatToBf16Bits(Fp8E4m3ToFloat(f8Cvt.u8) * scaleF);
                    }
                    if (colBase + 3 < tailLen) {
                        f8Cvt.u8 = (word >> 24) & 0xFF;
                        bits[3] = FloatToBf16Bits(Fp8E4m3ToFloat(f8Cvt.u8) * scaleF);
                    }
                    pTailU32.SetValue(dstBase, bits[0] | (bits[1] << 16));
                    pTailU32.SetValue(dstBase + 1, bits[2] | (bits[3] << 16));
                }
            }
        }
        // 写L1尾区(与Cube侧pTailOffset约定一致), 搬运参数参照flash_attn bf16的P写L1:
        //   blockCount = ceil(tailLen/16)个16列分形, blockLen = actVecM块(每行32B),
        //   srcStride = 0(源分形紧邻), dstStride = mBaseSize-actVecM(L1分形行距mBaseSize块)
        constexpr uint64_t pTailOffset = (static_cast<uint64_t>(mBaseSize) * s2BaseSize +
                                          mBaseSize * s2BaseSize / MXFP_GROUP_SIZE) / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> mm2ATailL1Tensor = outputBuf.GetTensor<bfloat16_t>(pTailOffset);
        uint64_t dstOffset = constInfo_.subBlockIdx * (blockBytes / sizeof(bfloat16_t)) * actVecMSizeAlign16;
        AscendC::PipeBarrier<PIPE_V>();
        // 无效subblock半区清零: decode场景actVecMSize=1时sub=1整块无效(L1残留inf/nan会进Mmad k=64累加);
        // B侧pad行兜底仅覆盖B, A侧垃圾列×B有效行同样污染L0C, 故A尾区未写半区必须显式清0
        if (runInfo.actVecMSize == 0) {
            constexpr uint32_t aTailHalfElems = (mBaseSize >> 1) * MXFP8_V_TAIL_BOUNDARY;
            LocalTensor<uint32_t> zeroSrc = pTailUb.ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < aTailHalfElems / 2; i++) {
                zeroSrc.SetValue(i, 0);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // 4元组形式写零半区: blockCount=4分形, blockLen=mBase/2块, 全零dstStride
            DataCopy(mm2ATailL1Tensor[dstOffset], pTailUb,
                     {(uint16_t)(MXFP8_V_TAIL_BOUNDARY / 16), (uint16_t)(mBaseSize >> 1),
                      (uint16_t)(vec1Srcstride - (mBaseSize >> 1)), 0});
        }
        // 源分形距: 每分形16行x16列=512B=16块(源紧凑无gap) -> srcGap=0;
        // pTailUb 分形f起点=f*128u32(512B), blockLen=actVecM块(16行x1块) — 布局: 每分形内行距8u32?
        // 重定义紧凑分形: 分形f = f*(16行*8u32) + row*8 + col/2 (每行8u32=16列, 行间无gap)
        DataCopy(mm2ATailL1Tensor[dstOffset], pTailUb,
                 {(uint16_t)(MXFP8_V_TAIL_BOUNDARY / 16), (uint16_t)actVecMSizeAlign16,
                  0,
                  (uint16_t)(mBaseSize - actVecMSizeAlign16)});
    }

    // [VTAIL] B尾块装载(AIV路径, 纯标量GM直读版): GM标量读(printf已证100%可靠, 绕开MTE2事件竞态)
    // -> UB摆NZ(u32) -> UB->L1(A侧同款非零dstStride参数, 绕开dstGap=0坑)。
    // L1布局: 8个16列分形x 2KB(64行距), 每分形前32行=数据(tail有效+pad0), 后32行=0 —
    // 与Cube侧 MatmulFull k=64 的 LoadDataToL0B srcStride(4*512B)匹配。
    __aicore__ inline void CopyVTailToL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         RunInfoX &runInfo)
    {
        if (constInfo_.subBlockIdx != 0) {
            return; // 单subblock搬运, 避免双核写L1/UB中转竞争
        }
        constexpr uint64_t pTailOffset = (static_cast<uint64_t>(mBaseSize) * s2BaseSize +
                                          mBaseSize * s2BaseSize / MXFP_GROUP_SIZE) / sizeof(bfloat16_t);
        constexpr uint64_t bTailOffset = pTailOffset + mBaseSize * MXFP8_V_TAIL_BOUNDARY;
        constexpr uint32_t stripNum = dTemplateAlign64 / 16;
        constexpr uint32_t dataRows = MXFP8_V_TAIL_BOUNDARY / 2; // 32行数据(tailAlign16上限)
        // UB中转: 独立vTailStripUb_(4KB)逐分形轮换 — 不可用vTailPCastBuf_:
        // A尾cast的MTE3 DataCopy读pTailUb与B侧标量写同buffer竞态(实测inf/nan污染)
        LocalTensor<uint32_t> nzU32 = vTailStripUb_.template Get<uint32_t>();
        // v_tail GM基址(u32视角): 行主序(Bn,N2,Bs,D), btt[bIdx*tailMaxBlockNum]定位物理块
        int32_t blkId = vTailBttGm_[runInfo.bIdx * constInfo_.tailMaxBlockNum];
        uint64_t gmBase = static_cast<uint64_t>(blkId) * constInfo_.n2Size * constInfo_.blockSize * constInfo_.dSizeV +
                          static_cast<uint64_t>(runInfo.n2Idx) * constInfo_.blockSize * constInfo_.dSizeV;
        __gm__ uint32_t *gmU32 = (__gm__ uint32_t *)(vTailGmVec_ + gmBase);
        uint32_t rowsU32 = constInfo_.dSizeV / 2; // 每行D列=64 u32
        uint32_t tailLen = runInfo.kvTailLen;
        LocalTensor<bfloat16_t> bTailL1 = outputBuf.GetTensor<bfloat16_t>(bTailOffset);
        for (uint32_t c = 0; c < stripNum; c++) {
            // nzU32双分形轮换(c&1, 各2KB): 偶分形用[0,512)u32, 奇分形用[512,1024)u32 —
            // 防分形c+1标量写覆盖分形c未完成的MTE3读
            LocalTensor<uint32_t> nzCur = nzU32[(c & 1U) * (MXFP8_V_TAIL_BOUNDARY * 8)];
            // nzCur布局(每分形2KB=512u32): [0,256)=数据32行(行距8u32); [256,512)=零32行
            for (uint32_t r = 0; r < dataRows; r++) {
                for (uint32_t j = 0; j < 8; j++) { // 每行16列=8 u32
                    uint32_t v = 0;
                    if (r < tailLen) {
                        v = gmU32[r * rowsU32 + c * 8 + j]; // GM标量直读(已验证可靠)
                    }
                    nzCur.SetValue(r * 8 + j, v);
                }
            }
            for (uint32_t i = dataRows * 8; i < MXFP8_V_TAIL_BOUNDARY * 8; i++) { // 零段256u32
                nzCur.SetValue(i, 0);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // 每分形2次搬运(单块形式, dstStride无意义置0): L1分形c 2KB=[数据32行][零32行]
            DataCopy(bTailL1[c * MXFP8_V_TAIL_BOUNDARY * 16], nzCur.ReinterpretCast<bfloat16_t>(),
                     {1, (uint16_t)dataRows, 0, 0});
            DataCopy(bTailL1[c * MXFP8_V_TAIL_BOUNDARY * 16 + dataRows * 16],
                     nzCur.ReinterpretCast<bfloat16_t>()[dataRows * 16], {1, (uint16_t)dataRows, 0, 0});
        }
    }

    __aicore__ inline uint32_t subLoopColsBase(RunInfoX &runInfo, uint32_t kMainSize)
    {
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        (void)runInfo;
        return (kMainSize / s2BaseSizeCur) * s2BaseSizeCur;
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if constexpr (FLASH_DECODE) {
            if (!runInfo.isS2SplitCore && constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void UpdateMinCheckValue()
    {
        float min = *((float *)&minValue_);
        if constexpr (USE_DN) {
            min *= constInfo_.scaleValue;
        }
        float tmp = min * INV_LN2;
        // int32_t 范围 [-2147483648, 2147483647]，下限约 -2.1e9
        // 若 tmp 小于 int32 最小值，float 精度丢失，直接原样计算
        if (tmp > -2147483648.0f) {
            min = static_cast<int32_t>(tmp) * LN2;
        } else {
            min = tmp * LN2;
        }
        minValue_ = static_cast<uint32_t>(*reinterpret_cast<int32_t *>(&min));
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMSize = USE_DN ? runInfo.actVecMSize : (runInfo.actMSizeAlign32 / 2);
        uint32_t gmDealRowCount;
        if constexpr (USE_DN) {
            gmDealRowCount = runInfo.actVecMSize;
        } else {
            uint32_t groupsOf32 = (runInfo.actMSize + 31) / 32;
            if (constInfo_.subBlockIdx == 0) {
                gmDealRowCount = groupsOf32 * 16 > runInfo.actMSize ? runInfo.actMSize : groupsOf32 * 16;
            } else {
                int32_t vec1RemainRows = runInfo.actMSize - 16 * groupsOf32;
                gmDealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
            }
        }
        if (gmDealRowCount == 0) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = this->softmaxLseQueue_.template AllocTensor<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue_);
        softmaxLseQueue_.template EnQue(lseUb);
        softmaxLseQueue_.DeQue<float>();

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // LSE 输出改为 N-major 排布 [N2*G, T]: N 在外, T 在内
            // bN2Offset = n2Idx * G * T_total + prefixBS1, 内部按 gIdx * T_total + s1Idx 步进
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                     gmDealRowCount, constInfo_);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
            uint32_t s1Size = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * constInfo_.n2Size * constInfo_.gSize + runInfo.n2Idx * constInfo_.gSize;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                       constInfo_, s1Size);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, s1LeftPaddingSize);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, qActSeqLens, s1LeftPaddingSize);
        }

        softmaxLseQueue_.FreeTensor(lseUb);
    }

    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo, uint32_t subLoop)
    {
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;
        uint32_t actVecMSizeAlign16 = runInfo.actMSizeAlign32 >> 1;

        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (HAS_MASK) {
            attenMaskUb = this->attenMaskInQue_[0].template AllocTensor<uint8_t>();
            AttenMaskCopyIn(attenMaskUb, 0, actVecMSizeAlign16, runInfo, subLoop); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
        // Loop 0拷贝一次全局, Loop 1用来计算update_mul
        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf_.template Get<float>();
        // 两轮循环前的sum值
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf_.template Get<float>();
        // Loop 0的sum值传给loop 1用来计算更新的sum值
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf_.template Get<float>();
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonTBuf_.template Get<uint8_t>();

        int64_t stage1Offset = 0;
        stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();
        // 回滚验证: 恢复原stage1CastTensor[16640]越界视图(11:40的TBuf版本疑破坏A尾scale时序, 待排除)
        constexpr int32_t pScaleByteOffset = 16640; // 16640: softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        static constexpr int32_t s2BaseSizeCur = s2BaseSize >> 1;
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2CalcSize = subLoop == 0 ? s2BaseSizeCur : runInfo.actSingleLoopS2Size - s2BaseSizeCur;
        }
        // [VTAIL] 二期预留: 纯尾task可改调 ProcessVec1VfMxfp8<T, bfloat16_t, ...>(T2模板参数直出bf16,
        // 状态机仍为Mxfp8原版)使P尾列获得8位有效数字; 实验确认ProcessVec1Vf(非量化版)的update流与
        // Mxfp8状态机(ln2格点化max+preLoop三件套)不兼容会NaN, 勿直接混用, 详见交接文档。
        if (unlikely(runInfo.isFirstS2Loop)) {
            if (unlikely(s2CalcSize == 128)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSizeCur, EQ_128, HAS_MASK,
                                   PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize <= 64)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSizeCur, GT_0_AND_LTE_64, HAS_MASK,
                                   PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize < 128)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSizeCur, GT_64_AND_LTE_128,
                                   HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else {
                if constexpr (s2BaseSizeCur == 256) {
                    ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSizeCur, GT_128_AND_LTE_256,
                                       HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                        firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                        static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                        deSCaleKValue, pScaleValue_);
                }
            }
        } else {
            if (unlikely(s2CalcSize == 128)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSizeCur, EQ_128, HAS_MASK,
                                   PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize <= 64)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSizeCur, GT_0_AND_LTE_64, HAS_MASK,
                                   PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize < 128)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSizeCur, GT_64_AND_LTE_128,
                                   HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else {
                if constexpr (s2BaseSizeCur == 256) {
                    ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSizeCur, GT_128_AND_LTE_256,
                                       HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                        firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                        static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                        deSCaleKValue, pScaleValue_);
                }
            }
        }
        bmm1ResBuf.SetCrossCore();
        if constexpr (HAS_MASK) {
            this->attenMaskInQue_[0].template FreeTensor(attenMaskUb);
        }

        // ===================DataCopy to L1 ====================
        this->stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        uint64_t pScaleL1Offset = mBaseSize * s2BaseSize; // PScale在L1P的偏移量（单位：元素）
        LocalTensor<fp8_e8m0_t> mm2AScaleL1Tensor = outputBuf.GetTensor<fp8_e8m0_t>(pScaleL1Offset);
        uint64_t pScaleDataLen = actVecMSizeAlign16 * s2BaseSizeCur / MXFP_GROUP_SIZE;
        constexpr uint16_t pScaleDstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        uint64_t pScaleSubLoopOffset = pScaleDataLen * 2;
        uint16_t copyCount = actVecMSizeAlign16 / 16;
        uint64_t vecOffset = constInfo_.subBlockIdx * pScaleDataLen;
        uint16_t dstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        constexpr uint32_t copyTimes = s2BaseSizeCur / MXFP_GROUP_SIZE / 2;
        if ((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                // PScale s2方向block块大小为32，共256/32=8个，L1需满足16x2分形，重复拷贝4次
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor,
                         {copyCount, 1, 0, pScaleDstStride});
                DataCopy(mm2AScaleL1Tensor[pScaleSubLoopOffset + vecOffset + i * 32], pScaleSubLoop0Tensor[128],
                         {copyCount, 1, 0, pScaleDstStride});
            }
        } else if (unlikely(runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor,
                         {copyCount, 1, 0, pScaleDstStride});
            }
        }

        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        if (likely(runInfo.actVecMSize != 0)) {
            int64_t dstOffset = constInfo_.subBlockIdx * (blockBytes / sizeof(INPUT_T)) * actVecMSizeAlign16 +
                                s2BaseSizeCur * mBaseSize * subLoop;
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor,
                     {s2BaseSizeCur / 32, (uint16_t)actVecMSizeAlign16,
                      (uint16_t)(vec1Srcstride - actVecMSizeAlign16), (uint16_t)(mBaseSize - actVecMSizeAlign16)});
        }
        if constexpr (HAS_V_TAIL) {
            // 尾块列的P以bf16写入L1尾区。仅覆盖序列末尾的task(判定: s2Idx+actS2Block>=actS2Size,
            // 不能用isLastS2Loop, 其语义是本核区间最后一块, S2SplitCore多核切分时会误判)的最后一个有效subLoop
            uint32_t kMainSize = runInfo.actSingleLoopS2Size - runInfo.kvTailLen;
            uint32_t tailSubLoop = (kMainSize > 0 && (s2SplitSize > 0)) ? (kMainSize / s2BaseSizeCur) : 0;
            bool isSeqLastTask = runInfo.s2Idx + runInfo.actSingleLoopS2Size >= runInfo.actS2Size;
            if (isSeqLastTask && (runInfo.kvTailLen > 0) && (kMainSize < runInfo.actSingleLoopS2Size) &&
                (subLoop == tailSubLoop || runInfo.actSingleLoopS2Size <= s2BaseSizeCur)) {
                CastPTailToBf16AndCopyToL1(outputBuf, stage1CastTensor, pScaleSubLoop0Tensor, runInfo, kMainSize,
                                           actVecMSizeAlign16, subLoop);
                // B尾块装载(与A尾区同一outputBuf同步点, SetCrossCore前MTE3排空)
                CopyVTailToL1(outputBuf, runInfo);
            }
        }
        this->stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
            (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            outputBuf.SetCrossCore();
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

        uint32_t vecMSize = USE_DN ? runInfo.actVecMSize : (runInfo.actMSizeAlign32 / 2);
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;
        constexpr float deSCaleVValue = 1.0f;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf_.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;

            constexpr float deSCalePreVValue = 1.0f;
            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, vecMSize,
                                                                          (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            uint32_t DealRowCount;
            if constexpr (USE_DN) {
                DealRowCount = runInfo.actVecMSize;
            } else {
                uint32_t groupsOf32 = (runInfo.actMSize + 31) / 32;
                if (constInfo_.subBlockIdx == 0) {
                    DealRowCount = groupsOf32 * 16 > runInfo.actMSize ? runInfo.actMSize : groupsOf32 * 16;
                } else {
                    int32_t vec1RemainRows = runInfo.actMSize - 16 * groupsOf32;
                    DealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
                }
            }
            if (DealRowCount == 0) {
                SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
                return;
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize, DealRowCount);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);

        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        } else {
            PostQuant(runInfo, attenOut, vec2ResUb, mStartVec, mDealSize, dSizeAligned64);
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize, gmDealRowCount);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfoX &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                     LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                     int64_t dSizeAligned64)
    {
        return;
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize, gmDealRowCount);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize, gmDealRowCount);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo_.realGSize;
            s1EndTdx = vecMEndIdx / constInfo_.realGSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            // GS1 layout
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                // 只跨1个G
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
                // 跨多个G
                ret = (s1StartTdx < s1FirstValidToken);
                ret = ret || (s1EndTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = ((constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
                if constexpr (!POST_QUANT) {
                    RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo_.dSizeV,
                                              static_cast<uint32_t>(dSizeAligned64), minValue_);
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo_.dSizeV, sizeof(float)); // w8后量化后的处理长度
                    RowInvalidUpdateVF<float>(*((LocalTensor<float> *)&vec2ResUb), maxTensor, mDealSize, dSize, dStride,
                                              minValue_);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        // mxfp8 colCount 只能为64或者128，与dDealSize相等
        FaUbTensor<OUTPUT_T, !isDAligned> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.realN2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = gmDealRowCount,
                        .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, !isDAligned> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcst_.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        sumBrdcst_.template EnQue(sumOutTensor);
        sumBrdcst_.template DeQue<float>();
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst_.template FreeTensor(sumOutTensor);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcst_.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        maxBrdcst_.template EnQue(maxOutTensor);
        maxBrdcst_.template DeQue<float>();
        DataCopy(softmaxFDMaxGm_[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst_.template FreeTensor(maxOutTensor);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        // 是否要改成halfMRealSize
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // flashDecodeS2Idx?nBufferStartM?
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[runInfo.loop % DB]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[runInfo.loop % DB]);
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        ProcessVec2OnUb(bmm2ResBuf, runInfo);
        return;
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        tPipe_->InitBuffer(softmaxSumBuf_[0], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxSumBuf_[1], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxSumBuf_[2], 256); // [64, 1]

        tPipe_->InitBuffer(softmaxMaxBuf_[0], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxMaxBuf_[1], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxMaxBuf_[2], 256); // [64, 1]

        tPipe_->InitBuffer(softmaxExpBuf_[0], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxExpBuf_[1], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxExpBuf_[2], 256); // [64, 1]

        tPipe_->InitBuffer(preLoopMaxBuf_, 256);   // [64, 1]
        tPipe_->InitBuffer(preLoopSumBuf_, 256);   // [64, 1]
        tPipe_->InitBuffer(firstLoopSumBuf_, 256); // [64, 1]

        if constexpr (FLASH_DECODE) {
            tPipe_->InitBuffer(maxBrdcst_, 1, 2048); // [64, 8]
            tPipe_->InitBuffer(sumBrdcst_, 1, 2048); // [64, 8]
        }
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
        uint32_t attenMaskSize = mBaseSize / CV_RATIO * (s2BaseSize >> 1U);
        SoftmaxInitBuffer();
        tPipe_->InitBuffer(stage2OutBuf_, 64 * dTemplateAlign64 * sizeof(T));
        tPipe_->InitBuffer(stage1OutQue_[0], 1, 16640); // (32 + 1) * (256 / 32) * 64
        tPipe_->InitBuffer(stage1OutQue_[1], 1, 16640);
        tPipe_->InitBuffer(commonTBuf_, 512);
        // P组scale独立buffer: 原实现 pScaleSubLoop0Tensor=stage1CastTensor[16640] 落在队列buffer之外,
        // VF每subLoop向该地址写128~256B(踩TPipe池后续内存, 靠分配间隔兜底); 收编为显式TBuf,
        // VF公共代码零改动(只往传入指针写, [128]相对偏移语义不变)
        tPipe_->InitBuffer(pScaleBuf_, 512);
        if constexpr (HAS_V_TAIL) {
            // 尾列P bf16暂存: 双subblock并发各占半区, 2 x (mBaseSize/2)行 x 64列 x 2B
            tPipe_->InitBuffer(vTailPCastBuf_, 2 * (mBaseSize >> 1U) * MXFP8_V_TAIL_BOUNDARY * sizeof(bfloat16_t));
            // B尾块分形中转(双缓冲防MTE2覆盖未完成的MTE3读): 2 x 2KB
            tPipe_->InitBuffer(vTailStripUb_, 2 * MXFP8_V_TAIL_BOUNDARY * 16 * sizeof(bfloat16_t));
        }
        if constexpr (HAS_MASK) {
            tPipe_->InitBuffer(attenMaskInQue_[0], 1, attenMaskSize); // 256 * 64
        }

        if (constInfo_.isSoftmaxLseEnable) {
            // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
            this->tPipe_->InitBuffer(softmaxLseQueue_, 1, (mBaseSize >> 1U) * sizeof(float) * 8);
        }
        if constexpr (isFp8) {
            if constexpr (USE_DN) {
                tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], 256);

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < (256 >> 2); j++) {
                        vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
                    }
                }
            } else {
                tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)],
                                   128); // s2realsize (64, 128]
                tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)],
                                   64); // s2realsize (0, 64]

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
                vselrIndexesTensor.SetValue(0, 0x7f);
                for (int i = 0; i < 128; i++) {
                    vselrIndexesTensor.SetValue(i, i * 2);
                }
                vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
                for (int i = 0; i < 64; i++) {
                    vselrIndexesTensor.SetValue(i, i * 4);
                }
            }
        }
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[1]);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX &runInfo, uint32_t subLoop)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2RealSize = subLoop == 0 ? s2SplitSize : runInfo.actSingleLoopS2Size - s2SplitSize;
        }

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.realGSize;
        maskInfo.s2StartIdx = (subLoop == 0) ? runInfo.s2Idx : (runInfo.s2Idx + s2BaseSizeCur);
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSizeCur;
        maskInfo.preToken = constInfo_.preTokens;
        maskInfo.nextToken = constInfo_.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = (constInfo_.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSizeCur - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar_;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        if (unlikely(!IsSkipMask)) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSizeCur>(attenMaskUb, attenMaskGmInt_, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
        }
    }

    __aicore__ inline int64_t ComputeMaskLineDN(RunInfoX &runInfo, uint32_t subLoop)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2RealSize = subLoop == 0 ? s2SplitSize : runInfo.actSingleLoopS2Size - s2SplitSize;
        }
        int64_t nextToken = static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
        uint32_t s1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        uint32_t s2StartIdx = (subLoop == 0) ? runInfo.s2Idx : (runInfo.s2Idx + s2BaseSizeCur);
        if (static_cast<int64_t>(s2StartIdx + s2RealSize) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            isSkipMask_ = true;
        } else {
            isSkipMask_ = false;
        }
        int64_t maskLine = nextToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(s2StartIdx);
        if (maskLine < 0) {
            isFullMask_ = -maskLine >= (mBaseSize / 2);
        } else {
            isFullMask_ = maskLine >= s2BaseSizeCur;
        }
        return maskLine;
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecMxfp8Dummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecMxfp8Dummy(ConstInfoX &constInfo){};
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_H_
