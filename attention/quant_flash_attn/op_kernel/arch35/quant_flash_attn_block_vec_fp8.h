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
 * \file quant_flash_attn_block_vec_fullquant_fp8.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_FP8_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_FP8_H_
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/init_output.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "quant_flash_attn_vselr_buf.h"
#include "quant_flash_attn_type.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <typename FA_T>
class QuantFlashAttnBlockVecGqaFp8 {
public:
    /* ============FA_T派生的基本块信息（对齐fa）============= */
    using INPUT_T = typename FA_T::inputType;
    using T = float;
    using OUTPUT_T = typename FA_T::outputType;
    using ConstInfoX = ConstInfo_t;
    // fp8 原分派为 TND||NTD→cuParser（与 mxfp8 仅 TND 不同），tool 侧 INCLUDE_NTD=true
    using QSeqLensToolType = QFASeqLensTool<FA_T::qLayout, false, true, int32_t, GetQActSeqMode<FA_T::qLayout>()>;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr DTemplateType dTemplateType = FA_T::dBaseSize;
    static constexpr DTemplateType dVTemplateType = FA_T::dVBaseSize;
    static constexpr LayOutTypeEnum layout = FA_T::qLayout;
    static constexpr LayOutTypeEnum outLayout = FA_T::attnOutLayout;
    static constexpr bool hasAtten = FA_T::hasMask;
    static constexpr bool isFd = FA_T::isFd;
    static constexpr bool useDn = FA_T::useDn;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;

    static constexpr bool USE_DN = useDn;
    static constexpr bool IS_PER_TOEKN_HEAD = true;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;

    // ClearOutput 专用静态ID/地址（静态Tensor范式）。
    // matmul::InitOutput 内部依赖 GetTPipePtr()->FetchEventID（无TPipe会崩溃），改用 common 静态版
    // AttentionCommon::InitOutput（静态Mutex ID + 静态pop buffer地址，照 fa vec / mxfp8 模式）。
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
    // qScale/kScale 拷入buf跨任务读写保护：单块任务(KV<=s2BaseSize)每任务 isFirstS2Loop 均为 true，
    // MTE2 重写同一 buf 与上一任务 V 侧 VF 读无天然屏障（动态版 TBuf 裸奔，静态版时序放大竞态窗口），
    // 照 fa mask 模式加 MTE2↔V 双向握手（MTE2 拿锁写 / V 拿锁读）
    static constexpr uint32_t QK_SCALE_MUTEX_ID = 6;
    static constexpr uint32_t UB_OUT_VEC2_RES_MUTEX_ID =
        7; // vec2Res(V计算)↔MTE3(copyout)所有权交接（fa UB_OUT_VEC2_RES）

    // ====================静态UB布局====================
    // fa式跨核共享区（cube/vec双侧各自持有同布局裸数组，跨核flag=slotID，AIC侧±16双AIV镜像）：
    // UB: [bmm2Res 1槽][bmm1Res 2槽] 从0起（vec专用区紧随其后）；L1: P区3槽从0起（cube侧Q/KV紧随其后）
    // fp8 无 s2 拆分，MM1_RES 公式无 /2（对照现 kernel_fp8 InitMMResBuf）
    static constexpr uint32_t MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T);
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t VEC_UB_BASE = MM2_RES_BUF_BYTES + MM1_RES_BUFCNT * MM1_RES_BUF_BYTES;
    // 跨核flag ID（mode 4 独立编号空间）
    static constexpr uint32_t CC_MM_BMM1_0 = 0U;
    static constexpr uint32_t CC_MM_BMM1_1 = 1U;
    static constexpr uint32_t CC_MM_BMM2_0 = 2U;
    static constexpr uint32_t CC_L1P_0 = 3U;
    static constexpr uint32_t SOFTMAX_BUF_BYTES = 256; // [64, 1]
    static constexpr uint32_t QSCALE_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float);
    static constexpr uint32_t KSCALE_BUF_BYTES =
        (s2BaseSize + (FA_BYTE_BLOCK / sizeof(T))) * sizeof(float) * 4; // 4: 4份解bank冲突
    static constexpr uint32_t BRDCT_BUF_BYTES = 2048;                   // [64, 8]
    static constexpr uint32_t STAGE2_OUT_BUF_BYTES = 64 * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t STAGE1_OUT_BUF_BYTES = 16640;
    static constexpr uint32_t LSE_OUT_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float) * 8; // 8: 适配TND，每行8个重复lse
    static constexpr uint32_t VSELR_DN_BUF_BYTES = 256;
    static constexpr uint32_t VSELR_GT64_BUF_BYTES = 128; // s2realsize (64, 128]
    static constexpr uint32_t VSELR_GT0_BUF_BYTES = 64;   // s2realsize (0, 64]

    // 顺序累加偏移（所有尺寸天然32B对齐）
    static constexpr uint32_t OFF_SOFTMAX_SUM = VEC_UB_BASE;
    static constexpr uint32_t OFF_SOFTMAX_MAX = OFF_SOFTMAX_SUM + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_SOFTMAX_EXP = OFF_SOFTMAX_MAX + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_QSCALE = OFF_SOFTMAX_EXP + (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
    static constexpr uint32_t OFF_KSCALE = OFF_QSCALE + QSCALE_BUF_BYTES;
    static constexpr uint32_t OFF_BRDCT = OFF_KSCALE + 2 * KSCALE_BUF_BYTES;
    static constexpr uint32_t OFF_STAGE2_OUT = OFF_BRDCT + (FLASH_DECODE ? 2 * BRDCT_BUF_BYTES : 0);
    static constexpr uint32_t OFF_STAGE1_OUT = OFF_STAGE2_OUT + STAGE2_OUT_BUF_BYTES;
    static constexpr uint32_t OFF_LSE_OUT = OFF_STAGE1_OUT + 2 * STAGE1_OUT_BUF_BYTES;
    static constexpr uint32_t OFF_VSELR = OFF_LSE_OUT + LSE_OUT_BUF_BYTES;

    static constexpr uint32_t VEC_UB_TOTAL =
        OFF_VSELR + (USE_DN ? VSELR_DN_BUF_BYTES : (VSELR_GT64_BUF_BYTES + VSELR_GT0_BUF_BYTES));
    static_assert(VEC_UB_TOTAL <= 248 * 1024, "vec UB buffer too large");

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr GmFormat Q_SCALE_FORMAT = GmFormat::NGT;
    static constexpr GmFormat K_SCALE_FORMAT = GmFormat::PA_BnNBs;
    using pseShiftType = half;

    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;
    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;

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
    QSeqLensToolType &qSeqLensTool_;

    GlobalTensor<int32_t> blockTableGm_;

    postQuantGmType postQuantScaleGm_;
    postQuantGmType postQuantOffsetGm_;
    postQuantBf16GmType postQuantScaleBf16Gm_;
    postQuantBf16GmType postQuantOffsetBf16Gm_;
    quantGmType pScaleGm_;
    static constexpr bool Q_SCALE_NEEDS_WZH = (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_NT);
    using FaGmTensorQScale = FaGmTensor<float, Q_SCALE_FORMAT, int32_t, Q_SCALE_NEEDS_WZH>;
    using FaGmTensorKScale = FaGmTensor<float, K_SCALE_FORMAT, int32_t>;
    FaGmTensorQScale queryScaleGm_;
    FaGmTensorKScale keyScaleGm_;
    quantGmType deScaleVGm_; // GQA: 1D [N2]的V descale

    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    CopyQueryScaleGmToUb<float, Q_SCALE_FORMAT> copyQueryScaleGmToUb_;
    // 跨核共享区（fa式：与cube侧同布局裸数组；AIV写P/读mmRes）
    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> ubBmm1ResBuffers_;
    LocalTensor<uint8_t> ubBmm2ResBuffers_;
    uint32_t mm1ResBufId_ = 0U;

    // ub（静态Tensor范式：LocalTensor 直连 UB 绝对偏移，布局见 InitBuffers）
    LocalTensor<uint8_t> stage1OutBuf_[2];            // softmax结果拷至L1
    LocalTensor<T> stage2OutBuf_;                     // 输出BUF: attn_out中间结果
    LocalTensor<float> softmaxMaxBuf_[PRELOAD_N + 1]; // [64, 1] 常驻BUF
    LocalTensor<float> softmaxSumBuf_[PRELOAD_N + 1]; // [64, 1] 常驻BUF
    LocalTensor<T> softmaxExpBuf_[PRELOAD_N + 1];     // [64, 1] 常驻BUF
    VselrIndexesBuffer vselrIndexesBuf_[4];
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域（FD only），[0]=sum [1]=max */
    LocalTensor<float> brdctBuf_[2];
    LocalTensor<float> lseOutBuf_;           // 输出BUF: LSE结果拷出
    LocalTensor<float> queryAntiqScaleBuf_;  // qScale MTE2 拷入
    LocalTensor<float> keyAntiqScaleBuf_[2]; // kScale DB ping-pong
    const ConstInfoX &constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    bool isSkipMask_{false};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecGqaFp8(ConstInfoX &constInfo, QSeqLensToolType &qSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *pScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *deqScaleQ,
                                        __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, blockTable, deqScaleQ, deqScaleK, deqScaleV,
                     softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *pScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *deqScaleQ,
                                        __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        uint64_t actualLenQSize = (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) ?
                                      constInfo_.cuSeqLensQSize :
                                      constInfo_.seqUsedQSize;
        actualSeqLengthsGmQ_.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr, actualLenQSize);

        if (pScale != nullptr) {
            pScaleGm_.SetGlobalBuffer((__gm__ float *)pScale);
            pScaleValue_ = this->pScaleGm_.GetValue(0);
        }
        if (deqScaleV != nullptr) {
            deScaleVGm_.SetGlobalBuffer((__gm__ float *)deqScaleV);
        }
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size,
                         queryScaleGm_, deqScaleQ);
        InitKScaleBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, keyScaleGm_,
                         deqScaleK, constInfo_.kDescaleStrides.bnStride, constInfo_.kDescaleStrides.n2Stride);
        if constexpr (FLASH_DECODE) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            FaGmTensorQScale &qScaleGmTensor, __gm__ uint8_t *gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float *)gm);
        if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_NT) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, FaGmTensorKScale &kScaleGmTensor,
                                            __gm__ uint8_t *gm, uint64_t bnStrides, uint64_t n2Strides)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float *)gm);
        kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, blockTableGm_, constInfo_.maxBlockNumPerBatch,
                                             bnStrides, n2Strides);
    }

    __aicore__ inline void ProcessVec1(RunInfoX runInfo)
    {
        uint32_t mm1ResBufId = mm1ResBufId_;
        mm1ResBufId_ = (mm1ResBufId_ + 1) % MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t mmSyncIdx = CC_MM_BMM1_0 + mm1ResBufId;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<uint8_t> pL1Buf = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        LocalTensor<T> mm1ResUbTensor =
            ubBmm1ResBuffers_[mm1ResBufId * MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 等AIC写入该槽（ready）
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // free
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
                v1c2CrossCoreSyncIdx); // P ready（内容未写，Bmm2输出被Vec2丢弃）
            return;
        }

        if constexpr (USE_DN) {
            ProcessVec1Dn(pL1Buf, mm1ResUbTensor, runInfo);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);               // 读完mmRes，free
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx); // P ready
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
            if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                          layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
                tSize = qSeqLensTool_.cuSeqLensParser.GetTSize();
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

    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<float> &dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX &runInfo)
    {
        FaUbTensor<float> ubTensor{
            .tensor = dstTensor,
            .rowCount = runInfo.actVecMSize,
            .colCount = dRealSize,
        };

        GmCoord gmCoord{
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.realN2Idx,
            .gS1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx,
            .dIdx = dOffset,
            .gS1DealSize = runInfo.actVecMSize,
        };
        copyQueryScaleGmToUb_(ubTensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<float> &dstTensor, RunInfoX &runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, 1, runInfo);
    }

    __aicore__ inline void CopyKeyScaleSlice(const LocalTensor<float> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                             RunInfoX &runInfo, bool isPreload)
    {
        FaUbTensor<float> ubTensor{.tensor = dstTensor, .rowCount = runInfo.actSingleLoopS2Size, .colCount = 1};

        uint32_t s2Idx = runInfo.s2Idx;
        uint32_t s2DealSize = runInfo.actSingleLoopS2Size;
        if (likely(isPreload)) {
            s2Idx = runInfo.s2Idx + s2BaseSize;
            if (s2Idx + s2BaseSize > runInfo.actS2Size) {
                s2DealSize = runInfo.actS2Size - s2Idx;
            }
        }

        AntiqGmCoord antiqGmCoord{
            .bIdx = runInfo.bIdx, .n2Idx = runInfo.n2Idx, .s2Idx = s2Idx, .s2DealSize = s2DealSize};
        ProcessAntiqPA(ubTensor, keyScaleGm_, antiqGmCoord);
    }

    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<float> &dstTensor, RunInfoX &runInfo, bool isPreload)
    {
        CopyKeyScaleSlice(dstTensor, 0, 1, runInfo, isPreload);
    }

    __aicore__ inline void ProcessAntiqPA(FaUbTensor<T> &dstTensor, FaGmTensorKScale &srcTensor,
                                          AntiqGmCoord &antiqGmCoord)
    {
        auto &offsetCal = srcTensor.offsetCalculator;

        uint64_t dstOffset = 0;
        uint32_t copyFinishElmeCnt = 0;
        uint32_t curS2Idx = antiqGmCoord.s2Idx;

        while (copyFinishElmeCnt < antiqGmCoord.s2DealSize) {
            uint32_t copyElemCnt =
                offsetCal.GetDimBlockSize() - curS2Idx % offsetCal.GetDimBlockSize(); // 一次只能处理一个block
            if (copyFinishElmeCnt + copyElemCnt > antiqGmCoord.s2DealSize) {
                copyElemCnt = antiqGmCoord.s2DealSize - copyFinishElmeCnt; // 一个block未拷满
            }

            uint64_t srcOffset = offsetCal.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, curS2Idx);
            CopyKeyScaleNDToND(dstTensor.tensor[dstOffset], srcTensor.gmTensor[srcOffset], copyElemCnt);

            dstOffset += copyElemCnt;
            copyFinishElmeCnt += copyElemCnt;
            curS2Idx += copyElemCnt;
        }
    }

    __aicore__ inline void CopyKeyScaleNDToND(LocalTensor<T> ubTensor, const GlobalTensor<T> gmTensor,
                                              uint32_t actDataLen)
    {
        uint32_t blockElemNum = FA_BYTE_BLOCK / sizeof(T);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(4); // 4: 4份，用于解bank冲突
        dataCopyParams.blockLen = actDataLen * sizeof(T);
        dataCopyParams.srcStride = -static_cast<int64_t>(dataCopyParams.blockLen);
        dataCopyParams.dstStride = (s2BaseSize + blockElemNum - actDataLen) * sizeof(T) / FA_BYTE_BLOCK;

        DataCopyPadExtParams<T> dataCopyPadParams;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, dataCopyPadParams);
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(LocalTensor<uint8_t> &pL1Buf, LocalTensor<T> &mm1ResUbTensor, RunInfoX runInfo)
    {
        int64_t maskLine = 0;
        if constexpr (HAS_MASK) {
            maskLine = ComputeMaskLineDN(runInfo, 0);
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)];

        auto expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
        int64_t stage1Offset = runInfo.loop % DB;
        int64_t kscaleOffset = (runInfo.s2Idx >> 8) & 1;

        float descaleQK = 1.0;

        // 加载qScale/kScale（MTE2 写侧拿锁，跨任务与 V 读串行化；
        // Lock<V> 在 Unlock<MTE2> 之后，已提供 MTE2→V 的等价顺序保证，无需 MTE2_V flush）
        LocalTensor<float> qScaleUbTensor;
        LocalTensor<float> kScaleUbTensor;
        if (unlikely(runInfo.isFirstS2Loop)) {
            qScaleUbTensor = queryAntiqScaleBuf_;
            kScaleUbTensor = keyAntiqScaleBuf_[kscaleOffset];
            Mutex::Lock<PIPE_MTE2>(QK_SCALE_MUTEX_ID);
            CopyQueryScaleTile(qScaleUbTensor, runInfo);
            CopyKeyScaleTile(kScaleUbTensor, runInfo, false);
            Mutex::Unlock<PIPE_MTE2>(QK_SCALE_MUTEX_ID);
        } else {
            kScaleUbTensor = keyAntiqScaleBuf_[kscaleOffset];
        }

        // V 读侧拿锁：等 MTE2 拷入完成（Unlock 生效），并阻止下一任务 MTE2 提前覆写
        Mutex::Lock<PIPE_V>(QK_SCALE_MUTEX_ID);

        LocalTensor<T> mmRes = mm1ResUbTensor;
        // 静态同步协议（照 fa vec 模式）：V 侧 Lock→VF 写→Unlock→MTE3 Lock→DataCopy UB→L1→Unlock
        Mutex::Lock<PIPE_V>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutBuf_[stage1Offset].template ReinterpretCast<INPUT_T>();
        if (unlikely(runInfo.isLastS2Loop && !runInfo.isFirstS2Loop)) {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize, true,
                                                         VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, false, s2BaseSize, true, VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        } else if (unlikely(runInfo.isFirstS2Loop)) {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, hasAtten, s2BaseSize, true,
                                                         VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, false, s2BaseSize, true,
                                                         VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        } else {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize, false,
                                                         VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, false, s2BaseSize, false,
                                                         VselrIndexesBuffer>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        }

        // V 侧读锁释放：VF 读 q/k 完成，允许下一任务 MTE2 覆写
        Mutex::Unlock<PIPE_V>(QK_SCALE_MUTEX_ID);

        if (likely(!runInfo.isLastS2Loop)) {
            LocalTensor<float> kScaleUbNextTensor = keyAntiqScaleBuf_[1 - kscaleOffset];
            CopyKeyScaleTile(kScaleUbNextTensor, runInfo, true);
        }

        Mutex::Unlock<PIPE_V>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        Mutex::Lock<PIPE_MTE3>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));
        //-------------------------Data copy to l1-------------------------
        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Buf.template ReinterpretCast<INPUT_T>();

        // 按照64对齐搬运
        uint32_t singleProcessSOuterSize = mBaseSize >> 1;
        uint32_t s2RealSizeAlign = (((runInfo.actSingleLoopS2Size + 63) >> 6) << 6);
        uint32_t mm2aBase = constInfo_.subBlockIdx * singleProcessSOuterSize * s2RealSizeAlign;
        DataCopy(mm2AL1Tensor[mm2aBase], stage1CastTensor,
                 {static_cast<uint16_t>((runInfo.actSingleLoopS2Size + 63) >> 6), 64, 66, 0});
        DataCopy(mm2AL1Tensor[mm2aBase + s2RealSizeAlign * 32], stage1CastTensor[65 << 5],
                 {static_cast<uint16_t>((runInfo.actSingleLoopS2Size + 63) >> 6), 64, 66, 0});

        //-----------------------------------------------------------------
        Mutex::Unlock<PIPE_MTE3>(STAGE1_OUT_MUTEX_ID0 + static_cast<uint32_t>(stage1Offset));

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

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        // 静态同步协议（照 fa vec 模式）：Lock<V>→VF→Unlock<V>→Lock<MTE3>→DataCopy→Unlock<MTE3>
        Mutex::Lock<PIPE_V>(LSE_OUT_MUTEX_ID);
        LocalTensor<float> lseUb = this->lseOutBuf_;

        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(LSE_OUT_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(LSE_OUT_MUTEX_ID);

        uint32_t prefixBS1 = qSeqLensTool_.cuSeqLensParser.GetTBase(runInfo.bIdx);
        uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
        DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                 runInfo.actVecMSize, constInfo_);

        Mutex::Unlock<PIPE_MTE3>(LSE_OUT_MUTEX_ID);
    }

    __aicore__ inline void ProcessVec2OnUb(LocalTensor<T> &mm2ResUbTensor, RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
        int32_t deScaleVOffset = runInfo.n2Idx;
        float deScaleVValue = this->deScaleVGm_.GetValue(deScaleVOffset); // GQA: per-head V descale

        LocalTensor<T> vec2ResUb = this->stage2OutBuf_;
        LocalTensor<T> mmRes = mm2ResUbTensor;
        // fa式锁交接（UB_OUT_VEC2_RES）：V持锁计算，释放后MTE3持锁copyout
        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_MUTEX_ID);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
            LocalTensor<T> pScaleUb;

            if (!runInfo.isLastS2Loop) {
                if (unlikely(runInfo.s2Idx == s2BaseSize)) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                }
            } else {
                if (unlikely(runInfo.s2Idx == s2BaseSize)) {
                    LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                } else {
                    LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                }
            }
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, deScaleVValue);
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_MUTEX_ID);
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
            return;
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_MUTEX_ID);
    }

    __aicore__ inline int64_t ComputeMaskLineDN(RunInfoX &runInfo, uint32_t vecMIdx)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;

        int64_t nextToken = static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
        uint32_t s1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        uint32_t s2StartIdx = runInfo.s2Idx;
        if (static_cast<int64_t>(s2StartIdx + s2RealSize) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            isSkipMask_ = true;
        } else {
            isSkipMask_ = false;
        }
        int64_t maskLine = nextToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(s2StartIdx);
        return maskLine;
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);

        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_MUTEX_ID);
        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        } else {
            PostQuant(runInfo, attenOut, vec2ResUb, mStartVec, mDealSize, dSizeAligned64);
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_MUTEX_ID);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_MUTEX_ID);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfoX &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                     LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                     int64_t dSizeAligned64)
    {
        return;
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
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
                                              static_cast<uint32_t>(dSizeAligned64));
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo_.dSizeV, sizeof(float));
                    RowInvalidUpdateVF<float>(*((LocalTensor<float> *)&vec2ResUb), maxTensor, mDealSize, dSize,
                                              dStride);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.realN2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              qSeqLensTool_.cuSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              qSeqLensTool_.cuSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, qSeqLensTool_.cuSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, qSeqLensTool_.cuSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // 静态同步协议（照 fa vec 模式）：Lock<V>→Broadcast→Unlock<V>→Lock<MTE3>→DataCopy→Unlock<MTE3>
        // Copy sum to gm
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
        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_MUTEX_ID);
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_MUTEX_ID);
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_MUTEX_ID);
    }

    __aicore__ inline void ProcessVec2(RunInfoX runInfo)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_BMM2_0); // 等AIC写入（ready）
        LocalTensor<T> mm2ResUbTensor = ubBmm2ResBuffers_.template ReinterpretCast<T>();
        ProcessVec2OnUb(mm2ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_BMM2_0); // 读完，free
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
        // fa式跨核共享区（与cube侧同布局：L1 P区3槽从A1偏移0起；UB bmm2Res单槽+bmm1Res双槽从VECIN偏移0起）
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_P_BUFCNT * L1_P_BUF_BYTES);
        ubBmm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, 0U, MM2_RES_BUF_BYTES);
        ubBmm1ResBuffers_ =
            LocalTensor<uint8_t>(TPosition::VECIN, MM2_RES_BUF_BYTES, MM1_RES_BUFCNT * MM1_RES_BUF_BYTES);
        SoftmaxInitBuffer();
        queryAntiqScaleBuf_ =
            LocalTensor<uint8_t>(TPosition::VECIN, OFF_QSCALE, QSCALE_BUF_BYTES).template ReinterpretCast<float>();
        keyAntiqScaleBuf_[0] =
            LocalTensor<uint8_t>(TPosition::VECIN, OFF_KSCALE, KSCALE_BUF_BYTES).template ReinterpretCast<float>();
        keyAntiqScaleBuf_[1] = LocalTensor<uint8_t>(TPosition::VECIN, OFF_KSCALE + KSCALE_BUF_BYTES, KSCALE_BUF_BYTES)
                                   .template ReinterpretCast<float>();
        stage2OutBuf_ =
            LocalTensor<uint8_t>(TPosition::VECIN, OFF_STAGE2_OUT, STAGE2_OUT_BUF_BYTES).template ReinterpretCast<T>();
        stage1OutBuf_[0] = LocalTensor<uint8_t>(TPosition::VECIN, OFF_STAGE1_OUT, STAGE1_OUT_BUF_BYTES);
        stage1OutBuf_[1] =
            LocalTensor<uint8_t>(TPosition::VECIN, OFF_STAGE1_OUT + STAGE1_OUT_BUF_BYTES, STAGE1_OUT_BUF_BYTES);
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

    __aicore__ inline void InitCrossCoreSync()
    {
        // 预置全部mmRes槽为free（AIC首写前无需等待）
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_BMM1_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_BMM1_1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_BMM2_0);
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}
};

template <typename FA_T>
class QuantFlashAttnBlockVecGqaFp8Dummy {
public:
    static constexpr bool HAS_MASK = FA_T::hasMask;
    static constexpr bool FLASH_DECODE = FA_T::isFd;
    using OUT_T = typename FA_T::outputType;
    using ConstInfoX = ConstInfo_t;
    using QSeqLensToolType = QFASeqLensTool<FA_T::qLayout, false, true, int32_t, GetQActSeqMode<FA_T::qLayout>()>;
    __aicore__ inline QuantFlashAttnBlockVecGqaFp8Dummy(ConstInfoX &constInfo, QSeqLensToolType &){};
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_FULLQUANT_FP8_H_
