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
#include "../../../common/op_kernel/init_output.h"
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
#include "../../common/op_kernel/init_output.h"
#endif
#include "attenmask_gs1_static_qfa.h"
#include "quant_flash_attn_vselr_buf.h"

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

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false, bool isDAligned = true>
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

    // ClearOutput 专用静态ID/地址（静态Tensor范式）。
    // matmul::InitOutput 内部依赖 GetTPipePtr()->FetchEventID（无TPipe会崩溃），改用 common 静态版
    // AttentionCommon::InitOutput（静态Mutex ID + 静态pop buffer地址，照 fa vec 模式）。
    // 时序上 ClearOutput 先于 FA 阶段完成并 SyncAll，Mutex ID 数值可与 FA 阶段复用；
    // pop buffer 与 CV 区重叠安全（FA 阶段启动前已完成）。
    static constexpr uint32_t INIT_OUTPUT_MUTEX_ID = 0;
    static constexpr uint32_t LSE_INIT_OUTPUT_MUTEX_ID = 1;
    static constexpr uint32_t INIT_OUTPUT_POP_BUF_BYTES = 32 * 1024;
    static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0;
    static constexpr uint32_t LSE_POP_BUF_START_ADDR = INIT_OUTPUT_POP_BUF_BYTES;
    static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = INIT_OUTPUT_POP_BUF_BYTES / sizeof(float);

    // ====================静态同步ID====================
    // 核内 Mutex ID（Lock/Unlock 专用空间，与 SetFlag/WaitFlag 事件ID空间相互独立）
    static constexpr uint32_t MASK_MTE2_MUTEX_ID = 0;   // attenMask 单buffer
    static constexpr uint32_t STAGE1_OUT_MUTEX_ID0 = 1; // stage1 双buffer ping
    static constexpr uint32_t STAGE1_OUT_MUTEX_ID1 = 2; // stage1 双buffer pong
    static constexpr uint32_t LSE_OUT_MUTEX_ID = 3;
    static constexpr uint32_t BRDCT_SUM_MUTEX_ID = 4; // FD only
    static constexpr uint32_t BRDCT_MAX_MUTEX_ID = 5; // FD only

    // SetFlag/WaitFlag 事件ID（MTE3_V 与 V_MTE3 各自独立编号空间）
    static constexpr uint32_t MTE3_TO_V_EVENT_ID0 = 0; // MTE3_V，stage2OutBuf 保护
    static constexpr uint32_t MTE3_TO_V_EVENT_ID1 = 1; // MTE3_V，仅 AllocEventID 预置/FreeEventID drain 配平
    static constexpr uint32_t V_TO_MTE3_EVENT_ID0 = 1; // V_MTE3，copyout 前序
    static constexpr uint32_t V_TO_MTE3_EVENT_ID1 = 2; // V_MTE3，FD ping-pong

    // ====================静态UB布局====================
    // vec 区基址 = kernel 侧 ubBufferManager_ 的总量（mm1ResultSize*2 + mm2ResultSize），
    // vec 侧按 kernel 公式 constexpr 重算，AIC/AIV 两侧一致
    static constexpr uint32_t VEC_UB_BASE = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T) / 2 * 2 +
                                            mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T);
    static constexpr uint32_t SOFTMAX_BUF_BYTES = 256; // [64, 1]
    static constexpr uint32_t BRDCT_BUF_BYTES = 2048;  // [64, 8]
    static constexpr uint32_t STAGE2_OUT_BUF_BYTES = 64 * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t STAGE1_OUT_BUF_BYTES = 16640; // (32 + 1) * (256 / 32) * 64
    static constexpr uint32_t COMMON_T_BUF_BYTES = 512;
    static constexpr uint32_t ATTEN_MASK_BUF_BYTES = mBaseSize / CV_RATIO * (s2BaseSize >> 1U);
    static constexpr uint32_t LSE_OUT_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float) * 8; // 8: 适配TND，每行8个重复lse
    static constexpr uint32_t VSELR_DN_BUF_BYTES = 256;
    static constexpr uint32_t VSELR_GT64_BUF_BYTES = 128; // s2realsize (64, 128]
    static constexpr uint32_t VSELR_GT0_BUF_BYTES = 64;   // s2realsize (0, 64]

    // 顺序累加偏移（所有尺寸天然32B对齐）
    static constexpr uint32_t OFF_SOFTMAX_SUM = VEC_UB_BASE;
    static constexpr uint32_t OFF_SOFTMAX_MAX = OFF_SOFTMAX_SUM + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_SOFTMAX_EXP = OFF_SOFTMAX_MAX + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_PRELOOP_MAX = OFF_SOFTMAX_EXP + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_PRELOOP_SUM = OFF_PRELOOP_MAX + SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_FIRSTLOOP_SUM = OFF_PRELOOP_SUM + SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_BRDCT = OFF_FIRSTLOOP_SUM + SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_STAGE2_OUT = OFF_BRDCT + (FLASH_DECODE ? 2 * BRDCT_BUF_BYTES : 0);
    static constexpr uint32_t OFF_STAGE1_OUT = OFF_STAGE2_OUT + STAGE2_OUT_BUF_BYTES;
    static constexpr uint32_t OFF_COMMON_T = OFF_STAGE1_OUT + 2 * STAGE1_OUT_BUF_BYTES;
    static constexpr uint32_t OFF_ATTEN_MASK = OFF_COMMON_T + COMMON_T_BUF_BYTES;
    static constexpr uint32_t OFF_LSE_OUT = OFF_ATTEN_MASK + (HAS_MASK ? ATTEN_MASK_BUF_BYTES : 0);
    static constexpr uint32_t OFF_VSELR = OFF_LSE_OUT + LSE_OUT_BUF_BYTES;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr MaskFormat MASK_LAYOUT = MaskFormat::SG;

    static constexpr bool USE_DN = useDn;
    static constexpr UbFormat OUT_UB_FORMAT = USE_DN ? UbFormat::S1_ONLY : GetOutUbFormat<layout>();
    using OutGmCoord = std::conditional_t<USE_DN, GmCoordS1Only, GmCoordGs1Merge>;

    static constexpr uint32_t VEC_UB_TOTAL =
        OFF_VSELR + (USE_DN ? VSELR_DN_BUF_BYTES : (VSELR_GT64_BUF_BYTES + VSELR_GT0_BUF_BYTES));
    static_assert(VEC_UB_TOTAL <= 248 * 1024, "vec UB buffer too large");

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
    // ClearOutput pop buffer 元素数（依赖 OUT_T，须在别名之后）
    static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = INIT_OUTPUT_POP_BUF_BYTES / sizeof(OUT_T);
    static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE_HALF = INIT_OUTPUT_POP_BUF_BYTES / sizeof(half);

    // gm
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

    attenMaskGmType attenMaskGmInt_;

    postQuantGmType postQuantScaleGm_;
    postQuantGmType postQuantOffsetGm_;
    postQuantBf16GmType postQuantScaleBf16Gm_;
    postQuantBf16GmType postQuantOffsetBf16Gm_;
    quantGmType pScaleGm_;

    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    // ub（静态Tensor范式：LocalTensor 直连 UB 绝对偏移，布局见 InitBuffers）
    LocalTensor<uint8_t> commonTBuf_;                 // common的复用空间
    LocalTensor<uint8_t> stage1OutBuf_[2];            // softmax结果拷至L1
    LocalTensor<uint8_t> attenMaskBuf_;               // HAS_MASK 输入BUF: MASK拷入
    LocalTensor<T> stage2OutBuf_;                     // 输出BUF: attn_out中间结果
    LocalTensor<float> softmaxMaxBuf_[PRELOAD_N + 1]; // [64, 1] 常驻BUF
    LocalTensor<float> softmaxSumBuf_[PRELOAD_N + 1]; // [64, 1] 常驻BUF
    LocalTensor<T> softmaxExpBuf_[PRELOAD_N + 1];     // [64, 1] 常驻BUF
    LocalTensor<float> preLoopMaxBuf_;
    LocalTensor<float> preLoopSumBuf_;
    LocalTensor<float> firstLoopSumBuf_;
    VselrIndexesBuffer vselrIndexesBuf_[4];
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域（FD only），[0]=sum [1]=max */
    LocalTensor<float> brdctBuf_[2];
    LocalTensor<float> lseOutBuf_; // 输出BUF: LSE结果拷出

    const ConstInfoX &constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    bool isSkipMask_{false};
    bool isFullMask_{false};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP32_LN2};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecMxfp8(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *pScale, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
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
            int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
            if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                          layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
                tSize = qActSeqLensParser_->GetTSize();
            }
            int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.dSizeV;
            if constexpr (POST_QUANT) {
                totalOutputSize /= 2;
            }
            uint32_t vecCoreNum = 2 * constInfo_.coreNum; // 2 means c:v = 1:2
            if constexpr (IsSameType<OUT_T, int8_t>::value) {
                // int8 输出按 half 元素粒度初始化（总字节数不变）
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOutGm_.GetPhyAddr(0)));
                AttentionCommon::InitOutput<half, INIT_OUTPUT_MUTEX_ID, ATTEN_OUT_POP_BUF_START_ADDR,
                                            ATTEN_OUT_POP_BUF_ELE_SIZE_HALF, true>(attentionOutTmpGm, totalOutputSize,
                                                                                   vecCoreNum, 0);
            } else {
                AttentionCommon::InitOutput<OUT_T, INIT_OUTPUT_MUTEX_ID, ATTEN_OUT_POP_BUF_START_ADDR,
                                            ATTEN_OUT_POP_BUF_ELE_SIZE, true>(attentionOutGm_, totalOutputSize,
                                                                              vecCoreNum, 0);
            }
            if (constInfo_.isSoftmaxLseEnable) {
                int64_t lseTotalSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;
                AttentionCommon::InitOutput<float, LSE_INIT_OUTPUT_MUTEX_ID, LSE_POP_BUF_START_ADDR,
                                            LSE_POP_BUF_ELE_SIZE, true>(
                    softmaxLseGm_, lseTotalSize, vecCoreNum, 3e+99); // 3e+99: set the value of invalid batch to inf
            }
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo_.needInitOutput;
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

        LocalTensor<float> sumUb = this->softmaxSumBuf_[softmaxBufIdx];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[softmaxBufIdx];
        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf_;
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf_;
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf_;
        LocalTensor<T> expUb = this->softmaxExpBuf_[expBufIdx];

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        // 静态同步协议（照 fa vec 模式）：V 侧 Lock→VF 写→Unlock→MTE3 Lock→DataCopy UB→L1→Unlock
        Mutex::Lock<PIPE_V>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutBuf_[stage1Offset].template ReinterpretCast<INPUT_T>();
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
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, false, hasAtten, s2BaseSizeCur, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, false, false, s2BaseSizeCur, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            }
        } else {
            if (unlikely(!isSkipMask_)) {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, true, hasAtten, s2BaseSizeCur, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, true, false, s2BaseSizeCur, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            }
        }
        bmm1ResBuf.SetCrossCore();

        Mutex::Unlock<PIPE_V>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        Mutex::Lock<PIPE_MTE3>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
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
        Mutex::Unlock<PIPE_MTE3>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));

        if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
            (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            outputBuf.SetCrossCore();
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
        return;
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if constexpr (FLASH_DECODE) {
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
        Mutex::Lock<PIPE_V>(LSE_OUT_MUTEX_ID);
        LocalTensor<float> lseUb = this->lseOutBuf_;
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue_);
        Mutex::Unlock<PIPE_V>(LSE_OUT_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(LSE_OUT_MUTEX_ID);

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

        Mutex::Unlock<PIPE_MTE3>(LSE_OUT_MUTEX_ID);
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
            attenMaskUb = this->attenMaskBuf_;
            AttenMaskCopyIn(attenMaskUb, 0, actVecMSizeAlign16, runInfo, subLoop); // 全量拷贝
            // 静态同步协议（照 fa vec_nd 模式）：VF 读前 Lock<V> 等 MTE2 拷入完成，读后 Unlock
            Mutex::Lock<PIPE_V>(MASK_MTE2_MUTEX_ID);
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
        // Loop 0拷贝一次全局, Loop 1用来计算update_mul
        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf_;
        // 两轮循环前的sum值
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf_;
        // Loop 0的sum值传给loop 1用来计算更新的sum值
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf_;
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonTBuf_;

        int64_t stage1Offset = 0;
        stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        // 静态同步协议（照 fa vec 模式）：V 侧 Lock→VF 写→Unlock→MTE3 Lock→DataCopy UB→L1→Unlock
        Mutex::Lock<PIPE_V>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutBuf_[stage1Offset].template ReinterpretCast<INPUT_T>();
        constexpr int32_t pScaleByteOffset = 16640; // 16640: softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        static constexpr int32_t s2BaseSizeCur = s2BaseSize >> 1;
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2CalcSize = subLoop == 0 ? s2BaseSizeCur : runInfo.actSingleLoopS2Size - s2BaseSizeCur;
        }
        if (unlikely(runInfo.isFirstS2Loop)) {
            if (unlikely(s2CalcSize == 128)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSizeCur, EQ_128, HAS_MASK,
                                   PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false, VselrIndexesBuffer>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize <= 64)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSizeCur, GT_0_AND_LTE_64, HAS_MASK,
                                   PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false, VselrIndexesBuffer>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize < 128)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSizeCur, GT_64_AND_LTE_128,
                                   HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false,
                                   VselrIndexesBuffer>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else {
                if constexpr (s2BaseSizeCur == 256) {
                    ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSizeCur, GT_128_AND_LTE_256,
                                       HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false,
                                       VselrIndexesBuffer>(
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
                                   PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false, VselrIndexesBuffer>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize <= 64)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSizeCur, GT_0_AND_LTE_64, HAS_MASK,
                                   PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false, VselrIndexesBuffer>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize < 128)) {
                ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSizeCur, GT_64_AND_LTE_128,
                                   HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false,
                                   VselrIndexesBuffer>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else {
                if constexpr (s2BaseSizeCur == 256) {
                    ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSizeCur, GT_128_AND_LTE_256,
                                       HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false, false, false, false,
                                       VselrIndexesBuffer>(
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
            Mutex::Unlock<PIPE_V>(MASK_MTE2_MUTEX_ID);
        }

        // ===================DataCopy to L1 ====================
        Mutex::Unlock<PIPE_V>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        Mutex::Lock<PIPE_MTE3>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
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
                     {s2BaseSizeCur / 32, (uint16_t)actVecMSizeAlign16, (uint16_t)(vec1Srcstride - actVecMSizeAlign16),
                      (uint16_t)(mBaseSize - actVecMSizeAlign16)});
        }
        Mutex::Unlock<PIPE_MTE3>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));

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

        LocalTensor<T> vec2ResUb = this->stage2OutBuf_;
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID0);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
            LocalTensor<T> pScaleUb;

            constexpr float deSCalePreVValue = 1.0f;
            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
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
                SetFlag<HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID0);
                return;
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize, DealRowCount);
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID0);
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
        SetFlag<HardEvent::V_MTE3>(V_TO_MTE3_EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(V_TO_MTE3_EVENT_ID0);

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
                LocalTensor<float> maxTensor = softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)][mStartVec];
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
        if constexpr (USE_DN) {
            // DN分支下realGSize=1，gS1Idx即s1Idx，gIdx恒为0
            OutGmCoord gmCoord{.bIdx = info.bIdx,
                               .n2Idx = info.realN2Idx,
                               .gIdx = 0,
                               .s1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                               .dIdx = 0,
                               .s1DealSize = gmDealRowCount,
                               .dDealSize = (uint32_t)constInfo_.dSizeV};
            CopyAttentionOut(ubTensor, gmCoord);
        } else {
            OutGmCoord gmCoord{.bIdx = info.bIdx,
                               .n2Idx = info.realN2Idx,
                               .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                               .dIdx = 0,
                               .gS1DealSize = gmDealRowCount,
                               .dDealSize = (uint32_t)constInfo_.dSizeV};
            CopyAttentionOut(ubTensor, gmCoord);
        }
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, !isDAligned> &ubTensor, OutGmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        // 静态同步协议（照 fa vec 模式）：Lock<V>→Broadcast→Unlock<V>→Lock<MTE3>→DataCopy→Unlock<MTE3>
        LocalTensor<float> sumOutTensor = brdctBuf_[0];
        Mutex::Lock<PIPE_V>(BRDCT_SUM_MUTEX_ID);
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(BRDCT_SUM_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(BRDCT_SUM_MUTEX_ID);
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(BRDCT_SUM_MUTEX_ID);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = brdctBuf_[1];
        Mutex::Lock<PIPE_V>(BRDCT_MAX_MUTEX_ID);
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(BRDCT_MAX_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(BRDCT_MAX_MUTEX_ID);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxOutTensor, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(BRDCT_MAX_MUTEX_ID);
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
        SetFlag<HardEvent::V_MTE3>(V_TO_MTE3_EVENT_ID0 + (runInfo.loop % DB));
        WaitFlag<HardEvent::V_MTE3>(V_TO_MTE3_EVENT_ID0 + (runInfo.loop % DB));
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
        for (uint32_t i = 0; i < PRELOAD_N + 1; i++) {
            softmaxSumBuf_[i] =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SOFTMAX_SUM + i * SOFTMAX_BUF_BYTES, SOFTMAX_BUF_BYTES)
                    .template ReinterpretCast<float>();
            softmaxMaxBuf_[i] =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SOFTMAX_MAX + i * SOFTMAX_BUF_BYTES, SOFTMAX_BUF_BYTES)
                    .template ReinterpretCast<float>();
            softmaxExpBuf_[i] =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SOFTMAX_EXP + i * SOFTMAX_BUF_BYTES, SOFTMAX_BUF_BYTES)
                    .template ReinterpretCast<T>();
        }

        preLoopMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_PRELOOP_MAX, SOFTMAX_BUF_BYTES)
                             .template ReinterpretCast<float>();
        preLoopSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_PRELOOP_SUM, SOFTMAX_BUF_BYTES)
                             .template ReinterpretCast<float>();
        firstLoopSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FIRSTLOOP_SUM, SOFTMAX_BUF_BYTES)
                               .template ReinterpretCast<float>();

        if constexpr (FLASH_DECODE) {
            brdctBuf_[0] =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BRDCT, BRDCT_BUF_BYTES).template ReinterpretCast<float>();
            brdctBuf_[1] = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BRDCT + BRDCT_BUF_BYTES, BRDCT_BUF_BYTES)
                               .template ReinterpretCast<float>();
        }
    }

    __aicore__ inline void InitBuffers()
    {
        SoftmaxInitBuffer();
        stage2OutBuf_ =
            LocalTensor<uint8_t>(TPosition::VECIN, OFF_STAGE2_OUT, STAGE2_OUT_BUF_BYTES).template ReinterpretCast<T>();
        stage1OutBuf_[0] = LocalTensor<uint8_t>(TPosition::VECIN, OFF_STAGE1_OUT, STAGE1_OUT_BUF_BYTES);
        stage1OutBuf_[1] =
            LocalTensor<uint8_t>(TPosition::VECIN, OFF_STAGE1_OUT + STAGE1_OUT_BUF_BYTES, STAGE1_OUT_BUF_BYTES);
        commonTBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_COMMON_T, COMMON_T_BUF_BYTES);
        if constexpr (HAS_MASK) {
            attenMaskBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_ATTEN_MASK, ATTEN_MASK_BUF_BYTES); // 256 * 64
        }
        lseOutBuf_ =
            LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSE_OUT, LSE_OUT_BUF_BYTES).template ReinterpretCast<float>();

        if constexpr (isFp8) {
            if constexpr (USE_DN) {
                vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)].tensor =
                    LocalTensor<uint8_t>(TPosition::VECIN, OFF_VSELR, VSELR_DN_BUF_BYTES);

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)].tensor;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < (256 >> 2); j++) {
                        vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
                    }
                }
            } else {
                vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].tensor =
                    LocalTensor<uint8_t>(TPosition::VECIN, OFF_VSELR, VSELR_GT64_BUF_BYTES); // s2realsize (64, 128]
                vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].tensor =
                    LocalTensor<uint8_t>(TPosition::VECIN, OFF_VSELR + VSELR_GT64_BUF_BYTES,
                                         VSELR_GT0_BUF_BYTES); // s2realsize (0, 64]

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].tensor;
                vselrIndexesTensor.SetValue(0, 0x7f);
                for (int i = 0; i < 128; i++) {
                    vselrIndexesTensor.SetValue(i, i * 2);
                }
                vselrIndexesTensor = vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].tensor;
                for (int i = 0; i < 64; i++) {
                    vselrIndexesTensor.SetValue(i, i * 4);
                }
            }
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // 静态Tensor范式：事件ID为静态常量，无需动态分配/释放；
        // 保留原预置语义（首轮无前置生产者，预置 [0]/[1] 两个并保持计数器平衡）
        SetFlag<HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID1);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID1);
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
            AttentionmaskCopyInStatic<uint8_t, MASK_LAYOUT, true, s2BaseSizeCur>(attenMaskUb, attenMaskGmInt_, maskInfo,
                                                                                 false, MASK_MTE2_MUTEX_ID);
        } else {
            Mutex::Lock<PIPE_V>(MASK_MTE2_MUTEX_ID);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
            Mutex::Unlock<PIPE_V>(MASK_MTE2_MUTEX_ID);
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
