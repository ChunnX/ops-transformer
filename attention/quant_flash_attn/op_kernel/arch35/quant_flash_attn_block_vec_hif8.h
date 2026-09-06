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
 * \file quant_flash_attn_block_vec_hif8.h
 * \brief HIF8 per-tensor quantization vec block. No pScale, v_scale applied in vec2.
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_HIF8_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_HIF8_H_

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
#include "../../../common/op_kernel/init_output.h"
#include "quant_flash_attn_vselr_buf.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

/* ============确定bmm2ResBuffer的类型============= */
template <bool useDn, bool isFp8>
struct Bmm2ResBuffSelHif8 {
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
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecHif8 {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize;
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
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

    static constexpr bool USE_DN = useDn;

    // ClearOutput 专用静态ID/地址（静态Tensor范式）。
    // matmul::InitOutput 内部依赖 GetTPipePtr()->FetchEventID（无TPipe会崩溃），改用 common 静态版
    // AttentionCommon::InitOutput（静态Mutex ID + 静态pop buffer地址，照 fa vec / mxfp8 / fp8 模式）。
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
    static constexpr uint32_t VEC_UB_BASE =
        mBaseSize / CV_RATIO * s2BaseSize * sizeof(T) * 2 + mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T);
    static constexpr uint32_t SOFTMAX_BUF_BYTES = 256; // [64, 1]
    static constexpr uint32_t BRDCT_BUF_BYTES = 2048;  // [64, 8]
    static constexpr uint32_t STAGE2_OUT_BUF_BYTES = 64 * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t STAGE1_OUT_BUF_BYTES = 16640;                              // (32 + 1) * (256 / 32) * 64
    static constexpr uint32_t LSE_OUT_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float) * 8; // 8: 适配TND，每行8个重复lse
    static constexpr uint32_t VSELR_DN_BUF_BYTES = 256;

    // 顺序累加偏移（所有尺寸天然32B对齐）
    static constexpr uint32_t OFF_SOFTMAX_SUM = VEC_UB_BASE;
    static constexpr uint32_t OFF_SOFTMAX_MAX = OFF_SOFTMAX_SUM + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_SOFTMAX_EXP = OFF_SOFTMAX_MAX + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_BRDCT = OFF_SOFTMAX_EXP + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_STAGE2_OUT = OFF_BRDCT + (FLASH_DECODE ? 2 * BRDCT_BUF_BYTES : 0);
    static constexpr uint32_t OFF_STAGE1_OUT = OFF_STAGE2_OUT + STAGE2_OUT_BUF_BYTES;
    static constexpr uint32_t OFF_LSE_OUT = OFF_STAGE1_OUT + 2 * STAGE1_OUT_BUF_BYTES;
    static constexpr uint32_t OFF_VSELR = OFF_LSE_OUT + LSE_OUT_BUF_BYTES;

    static constexpr uint32_t VEC_UB_TOTAL = OFF_VSELR + (isFp8 ? VSELR_DN_BUF_BYTES : 0);
    static_assert(VEC_UB_TOTAL <= 248 * 1024, "vec UB buffer too large");

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar_ = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;
    // ClearOutput pop buffer 元素数（依赖 OUT_T，须在别名之后）
    static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = INIT_OUTPUT_POP_BUF_BYTES / sizeof(OUT_T);
    static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE_HALF = INIT_OUTPUT_POP_BUF_BYTES / sizeof(half);

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm_;
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

    quantGmType pScaleGm_;
    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    // ub（静态Tensor范式：LocalTensor 直连 UB 绝对偏移，布局见 InitBuffers）
    LocalTensor<uint8_t> stage1OutBuf_[2];            // softmax结果拷至L1
    LocalTensor<T> stage2OutBuf_;                     // 输出BUF: attn_out中间结果
    LocalTensor<float> softmaxMaxBuf_[PRELOAD_N + 1]; // [64, 1] 常驻BUF
    LocalTensor<float> softmaxSumBuf_[PRELOAD_N + 1]; // [64, 1] 常驻BUF
    LocalTensor<T> softmaxExpBuf_[PRELOAD_N + 1];     // [64, 1] 常驻BUF
    VselrIndexesBuffer vselrIndexesBuf_[4];
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域（FD only），[0]=sum [1]=max */
    LocalTensor<float> brdctBuf_[2];
    LocalTensor<float> lseOutBuf_; // 输出BUF: LSE结果拷出

    const ConstInfoX &constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    float descaleQKValue_{1.0f};
    float vScaleValue_{1.0f};
    bool isSkipMask_{false};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP32};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecHif8(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void SetDescaleQK(float qScale, float kScale)
    {
        descaleQKValue_ = qScale * kScale;
    }

    __aicore__ inline void SetVScale(float vScale)
    {
        vScaleValue_ = vScale;
    }

    __aicore__ inline void InitVecBlock(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                        __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *pScale,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        float qScale = 1.0f;
        float kScale = 1.0f;
        float vScale = 1.0f;
        if (dequantScaleQuery != nullptr) {
            qScale = *((__gm__ float *)dequantScaleQuery);
        }
        if (dequantScaleKey != nullptr) {
            kScale = *((__gm__ float *)dequantScaleKey);
        }
        if (dequantScaleValue != nullptr) {
            vScale = *((__gm__ float *)dequantScaleValue);
        }
        SetDescaleQK(qScale, kScale);
        SetVScale(vScale);
        UpdateMinCheckValue();

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
                                       RunInfoX runInfo)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            outputBuf.SetCrossCore();
            return;
        }

        if constexpr (USE_DN) {
            ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo);
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
                totalOutputSize /= 2; // 2 : post-quant output is 1 byte, init as half (2 bytes)
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
                                         RunInfoX runInfo)
    {
        int64_t maskLine = 0;
        if constexpr (HAS_MASK) {
            maskLine = ComputeMaskLineDN(runInfo);
        }

        uint32_t softmaxBufIdx = runInfo.mloop % (PRELOAD_N + 1);
        uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1);
        int64_t stage1Offset = runInfo.loop % DB;

        LocalTensor<float> sumUb = this->softmaxSumBuf_[softmaxBufIdx];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[softmaxBufIdx];
        LocalTensor<float> expUb = this->softmaxExpBuf_[expBufIdx];

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        // 静态同步协议（照 fa vec 模式）：V 侧 Lock→VF 写→Unlock→MTE3 Lock→DataCopy UB→L1→Unlock
        Mutex::Lock<PIPE_V>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutBuf_[stage1Offset].template ReinterpretCast<INPUT_T>();
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;

        if (unlikely(runInfo.isFirstS2Loop)) {
            if (unlikely(!isSkipMask_)) {
                FaVectorApi::ProcessVec1VfDnHif8<T, INPUT_T, false, hasAtten, s2BaseSize, false, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQKValue_, pScaleValue_, negativeFloatScalar_, 0.0F,
                    maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnHif8<T, INPUT_T, false, false, s2BaseSize, false, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQKValue_, pScaleValue_, negativeFloatScalar_, 0.0F,
                    maskLine);
            }
        } else {
            if (unlikely(!isSkipMask_)) {
                FaVectorApi::ProcessVec1VfDnHif8<T, INPUT_T, true, hasAtten, s2BaseSize, false, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQKValue_, pScaleValue_, negativeFloatScalar_, 0.0F,
                    maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnHif8<T, INPUT_T, true, false, s2BaseSize, false, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQKValue_, pScaleValue_, negativeFloatScalar_, 0.0F,
                    maskLine);
            }
        }
        bmm1ResBuf.SetCrossCore();

        Mutex::Unlock<PIPE_V>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        Mutex::Lock<PIPE_MTE3>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));

        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        uint32_t s2RealSizeAlign = ((runInfo.actSingleLoopS2Size + 63) >> 6) << 6;
        constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
        DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * s2RealSizeAlign], stage1CastTensor,
                 {static_cast<uint16_t>(s2RealSizeAlign >> 6), 64, 66, 0});
        DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * s2RealSizeAlign + s2RealSizeAlign * 32],
                 stage1CastTensor[65 << 5], {static_cast<uint16_t>(s2RealSizeAlign >> 6), 64, 66, 0});
        Mutex::Unlock<PIPE_MTE3>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));

        outputBuf.SetCrossCore();
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
            min *= constInfo_.scaleValue * descaleQKValue_;
        }
        minValue_ = static_cast<uint32_t>(*reinterpret_cast<int32_t *>(&min));
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMSize = runInfo.actVecMSize;
        uint32_t gmDealRowCount = runInfo.actVecMSize;
        if (gmDealRowCount == 0) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        // 静态同步协议（照 fa vec 模式）：Lock<V>→VF→Unlock<V>→Lock<MTE3>→DataCopy→Unlock<MTE3>
        Mutex::Lock<PIPE_V>(LSE_OUT_MUTEX_ID);
        LocalTensor<float> lseUb = this->lseOutBuf_;
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue_);
        Mutex::Unlock<PIPE_V>(LSE_OUT_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(LSE_OUT_MUTEX_ID);

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                     gmDealRowCount, constInfo_);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.s1Size +
                                 runInfo.realN2Idx * constInfo_.realGSize * constInfo_.s1Size;
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, s1LeftPaddingSize);
        } else {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.s1Size +
                                 runInfo.realN2Idx * constInfo_.realGSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, qActSeqLens, s1LeftPaddingSize);
        }

        Mutex::Unlock<PIPE_MTE3>(LSE_OUT_MUTEX_ID);
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        uint32_t vecMSize = runInfo.actVecMSize;
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf_;
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID0);
        if (unlikely(runInfo.isFirstS2Loop)) {
            Muls(vec2ResUb, mmRes, vScaleValue_, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
            LocalTensor<T> pScaleUb;

            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, vScaleValue_, 1.0f);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, vScaleValue_,
                    1.0f);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, vecMSize,
                                                                          (uint16_t)dTemplateAlign64, 1.0f);
            }
            uint32_t DealRowCount = runInfo.actVecMSize;
            if (DealRowCount == 0) {
                SetFlag<HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID0);
                return;
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize, DealRowCount);
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_TO_V_EVENT_ID0);
    }

    __aicore__ inline void PostQuant(RunInfoX &runInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<T> &vec2ResUb,
                                     int64_t mStartVec, int64_t mDealSize, int64_t dSizeAligned64)
    {
        return;
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
            s1StartTdx = vecMStartIdx / constInfo_.realGSize;
            s1EndTdx = vecMEndIdx / constInfo_.realGSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
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
            bool batchNeedRowInvalid = ((constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) && hasValidRow);
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
                    uint16_t dSize = CeilDiv(constInfo_.dSizeV, sizeof(float));
                    RowInvalidUpdateVF<float>(*((LocalTensor<float> *)&vec2ResUb), maxTensor, mDealSize, dSize, dStride,
                                              minValue_);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        FaUbTensor<OUTPUT_T, false> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.realN2Idx,
                                .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                                .dIdx = 0,
                                .gS1DealSize = gmDealRowCount,
                                .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, false> &ubTensor, GmCoordGs1Merge &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
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
        // 静态同步协议（照 fa vec 模式）：Lock<V>→Broadcast→Unlock<V>→Lock<MTE3>→DataCopy→Unlock<MTE3>
        LocalTensor<float> sumOutTensor = brdctBuf_[0];
        Mutex::Lock<PIPE_V>(BRDCT_SUM_MUTEX_ID);
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(BRDCT_SUM_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(BRDCT_SUM_MUTEX_ID);
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(BRDCT_SUM_MUTEX_ID);

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
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
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
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
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
        lseOutBuf_ =
            LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSE_OUT, LSE_OUT_BUF_BYTES).template ReinterpretCast<float>();

        if constexpr (isFp8) {
            vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)].tensor =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_VSELR, VSELR_DN_BUF_BYTES);

            LocalTensor<uint8_t> vselrIndexesTensor =
                vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)].tensor;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < (256 >> 2); j++) {
                    vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
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

    __aicore__ inline int64_t ComputeMaskLineDN(RunInfoX &runInfo)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        int64_t nextToken = static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
        uint32_t s1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        uint32_t s2StartIdx = runInfo.s2Idx;
        if (static_cast<int64_t>(s2StartIdx + s2RealSize) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            isSkipMask_ = true;
        } else {
            isSkipMask_ = false;
        }
        int64_t maskLine = nextToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(s2StartIdx);
        return maskLine;
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecHif8Dummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecHif8Dummy(ConstInfoX &constInfo){};
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_VEC_HIF8_H_
