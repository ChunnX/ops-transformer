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
 * \file flash_attn_block_vec_dn.h
 * \brief FANoQuantGqaBlockVecDnUnmerged —— Dn 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTN_BLOCK_VEC_DN_UNMERGED_H_
#define FLASH_ATTN_BLOCK_VEC_DN_UNMERGED_H_

#include <limits>
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "vf_flashupdate_dn.h"
#include "vf_softmax_dn_segmented.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/const_def.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/init_output.h"
#include "memory_copy_arch35.h"
#include "../utils/attn_sink_gs1.h"
#include "../../../common/op_kernel/arch_info.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;

namespace FlashAttnKernel {
using ArchInfo::CV_RATIO;

template <typename FA_T>
class FANoQuantGqaBlockVecDnUnmerged {
public:
    using ConstInfo = ConstInfo_t;
    static constexpr uint32_t PRELOAD_N =
        FA_T::causalOpt ? 3U : (static_cast<uint32_t>(FA_T::dBaseSize) == 64U ? 3U : 2U);
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;

    using T = float;
    static constexpr uint32_t dTemplateAlign64 = BaseApi::Align64Func((uint16_t)FA_T::dVBaseSize);

    static constexpr uint32_t DB = 2;
    // 索引使用 loop & (DB - 1) 代替 loop % DB，要求 DB 必须是2的幂，否则位掩码结果错误
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t CC_MM1_0 = 0U;
    static constexpr uint32_t CC_MM2_0 = 2U;
    static constexpr uint32_t CC_L1P_0 = 5U;
    static constexpr uint32_t CC_L1P_1 = 6U;
    static constexpr uint32_t CC_L1P_2 = 7U;

    // 核内同步ID
    // MTE3<->V, 输出buffer
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = 3;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = 5;

    // L1
    static constexpr uint32_t L1_P_BUFCNT = PRELOAD_N + 1U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> l1PBuffers_;

    // UB
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_MM2_RES_BUFCNT =
        FA_T::causalOpt ? (dBaseSize == 80U ? 2U : 1U) : (dVBaseSize <= 64U ? 2U : 1U);
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubMm2ResBuffers_;
    uint32_t mm1ResUbBufId_ = 0U;
    uint32_t mm2ResUbBufId_ = 0U;
    static constexpr uint32_t UB_MM_RES_TOTAL_BYTES =
        UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES + UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
    static __aicore__ inline constexpr uint32_t GetFdBaseOffset()
    {
        return UB_MM_RES_TOTAL_BYTES;
    }
    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES =
        mBaseSize / CV_RATIO * (FA_T::causalOpt ? dVBaseSize : dTemplateAlign64) * sizeof(T);
    LocalTensor<T> ubVec2Res_; // 存放vec2阶段VEC的中间处理结果, 并且作为attn_out的输出buffer, 需配对的MTE3和V的同步ID

    static constexpr uint32_t UB_VEC1_RES_BUFCNT = FA_T::causalOpt ? 2U : (dVBaseSize <= 64U ? 2U : 1U);
    // DN packs 64 query lanes; align each slot before the next UB region.
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES =
        AttentionCommon::Align((64U + 1U) * s2BaseSize * static_cast<uint32_t>(sizeof(INPUT_T)), 256U);
    LocalTensor<uint8_t> ubVec1ResBuffers_;
    uint32_t vec1ResUbBufId_ = 0;

    static constexpr uint32_t UB_SOFTMAX_MAX_BUFCNT = PRELOAD_N + 1U;
    static constexpr uint32_t UB_SOFTMAX_MAX_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxSumBuf_;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUFCNT = PRELOAD_N + 1U;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxMaxBuf_;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUFCNT = PRELOAD_N + 1U;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxExpBuf_;

    static constexpr uint32_t UB_LSE_OUT_BUFCNT = FA_T::causalOpt && dVBaseSize == 80U ? 1U : 2U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * 8 * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<uint8_t> ubLseOutBuffers_;
    uint32_t lseOutUbBufId_ = 0;

    static constexpr uint32_t UB_TRIANGLE_BYTES = 2U * 64U * 64U;
    LocalTensor<int8_t> ubTriangleMask_;

    const ConstInfo &constInfo_;

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool_;

    // GM
    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensor_;
    // 不合轴 DN: 输出按 (b, n1, s1) 寻址(quant_flash_attn USE_DN 先例用 UbFormat::S1_ONLY)
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, UbFormat::S1_ONLY> copyAttenOutUbToGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxFDSumGm_;
    GlobalTensor<float> softmaxFDMaxGm_;
    GlobalTensor<float> sinkGm_;

    T negativeFloatScalar_;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0;
    uint32_t negativeIntScalar_ = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecDnUnmerged(ConstInfo_t &constInfo,
                                                     SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                     SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                     __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        InitAttenOutBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSizeV,
                           outGmTensor_, attentionOut);
        if constexpr (FA_T::causalOpt && dBaseSize == 80U) {
            attentionOutGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            outGmTensor_.gmTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if (constInfo_.enableFlashDecode) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
        if (constInfo_.learnableSinkFlag) {
            sinkGm_.SetGlobalBuffer((__gm__ float *)learnableSink);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // P follows the PRELOAD_N + 1 slot ring (four slots for D64, otherwise three).
        uint32_t addrL1 = FA_T::causalOpt && dBaseSize == 80U ? 2U * mBaseSize * dBaseSize * sizeof(INPUT_T) : 0U;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);

        static_assert(UB_MM_RES_TOTAL_BYTES + UB_VEC2_RES_BUF_BYTES + UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES +
                              UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES +
                              UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES +
                              UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES +
                              UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES + (HAS_MASK ? UB_TRIANGLE_BYTES : 0U) <=
                          248U * 1024U,
                      "DN UB capacity exceeded");
        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);
        addrUb += UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES;
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES);
        addrUb += UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
        ubVec2Res_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_VEC2_RES_BUF_BYTES)
                         .template ReinterpretCast<T>(); // 输出BUF: attn_out拷出
        addrUb += UB_VEC2_RES_BUF_BYTES;
        ubVec1ResBuffers_ =
            LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                 UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES); // 按N定容的P槽, softmax结果拷贝至L1
        addrUb += UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES;

        // Sum/max/exp each use PRELOAD_N + 1 slots of 256 bytes.
        softmaxSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 常驻统计槽
        addrUb += UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES;
        softmaxMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 常驻统计槽
        addrUb += UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES;
        softmaxExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 常驻统计槽
        addrUb += UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES;

        ubLseOutBuffers_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_LSE_OUT_BUFCNT *
                UB_LSE_OUT_BUF_BYTES); // 输出BUF:
                                       // FD中间结果SUM和MAX拷出至GM，或者LSE结果拷出；按actVecMSize*8(fp32块对齐)定容
        addrUb += UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES;
        if constexpr (HAS_MASK) {
            ubTriangleMask_ = LocalTensor<int8_t>(TPosition::VECIN, addrUb, UB_TRIANGLE_BYTES);
        }
    }

    // FD may overwrite FA scratch between sections. Initialize once per
    // FA section, then keep the two templates resident across all its tiles.
    __aicore__ inline void InitTriangleMasks()
    {
        if constexpr (HAS_MASK) {
            FaVectorApi::InitDnTriangleMaskVF((__ubuf__ int8_t *)ubTriangleMask_.GetPhyAddr());
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ResetSoftmaxBuffer(uint32_t slotIdx, const RunInfo &runInfo)
    {
        if (runInfo.actVecMSize == 0) {
            return;
        }
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T);
        LocalTensor<T> sumUb = softmaxSumBuf_[slotIdx * softmaxBufElementCount];
        LocalTensor<T> maxUb = softmaxMaxBuf_[slotIdx * softmaxBufElementCount];
        if (constInfo_.learnableSinkFlag && runInfo.isFirstFdBlock) {
            Duplicate<T>(sumUb, static_cast<T>(1), softmaxBufElementCount);

            // sink 按 GS 线性坐标定位当前 G 头: g*S1 + s1
            uint32_t gs1Start = runInfo.gIdx * runInfo.actS1Size + runInfo.s1Idx + runInfo.vecMbaseIdx;

            Mutex::Lock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();
                AttentionCommon::SinkCopyInGS1(sinkTmpUb, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                               runInfo.n2Idx, constInfo_.gSize);
            }
            Mutex::Unlock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

            Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();

                AttentionCommon::SinkExpandMaxVf<T, float, true>(maxUb, sinkTmpUb, gs1Start, runInfo.actVecMSize,
                                                                 runInfo.actS1Size, constInfo_.gSize);
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        } else {
            Duplicate<T>(sumUb, static_cast<T>(0), softmaxBufElementCount);
            Duplicate<T>(maxUb, static_cast<T>(-std::numeric_limits<float>::infinity()), softmaxBufElementCount);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        for (uint32_t slot = 0; slot < UB_MM1_RES_BUFCNT; ++slot) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_0 + slot);
        }
        for (uint32_t slot = 0; slot < UB_MM2_RES_BUFCNT; ++slot) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM2_0 + slot);
        }
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline void ProcessVec1(RunInfo runInfo)
    {
        uint32_t mm1ResUbBufId = mm1ResUbBufId_;
        mm1ResUbBufId_ = (mm1ResUbBufId_ + 1) % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t mmSyncIdx = CC_MM1_0 + mm1ResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        if (unlikely(runInfo.isFirstS2Loop)) {
            AscendC::PipeBarrier<PIPE_V>();
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT, runInfo);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        ProcessVec1Dn(pL1Tensor, mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 通知BMM1: Vec1已读完MM1结果, 可覆写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx);
    }

    __aicore__ inline void ClearOutput()
    {
        if (constInfo_.needInitOutput) {
            uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum;
            uint64_t tSize = constInfo_.bSize * constInfo_.s1Size;
            if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                tSize = qSeqLensTool_.cuSeqLensParser.GetTSize();
            }
            uint64_t attenOutTotalSize = tSize * constInfo_.n2Size * constInfo_.gSize * constInfo_.dSizeV;

            static constexpr OUTPUT_T ATTEN_OUT_INIT_VAL = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(OUTPUT_T);
            AttentionCommon::InitOutput<OUTPUT_T, EVENT_ID0, ATTEN_OUT_POP_BUF_START_ADDR, ATTEN_OUT_POP_BUF_ELE_SIZE,
                                        true>(attentionOutGm_, attenOutTotalSize, vecCoreNum, ATTEN_OUT_INIT_VAL);

            if (constInfo_.isSoftmaxLseEnable) {
                uint64_t lseTotalSize = tSize * constInfo_.n2Size * constInfo_.gSize;

                static constexpr float LSE_INIT_VAL = 3e+99;
                static constexpr uint32_t LSE_POP_BUF_START_ADDR = BUFFER_SIZE_BYTE_32K;
                static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(float);
                AttentionCommon::InitOutput<float, EVENT_ID1, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, true>(
                    softmaxLseGm_, lseTotalSize, vecCoreNum, LSE_INIT_VAL);
            }

            SyncAll();
        }
    }

    __aicore__ inline void InitAttenOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                              uint32_t headDim, FaGmTensorOut &outGmTensor, __gm__ uint8_t *gm)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUTPUT_T *)gm);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool_.seqUsedParser);
        } else {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfo runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (constInfo_.enableFlashDecode) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if (constInfo_.enableFlashDecode) {
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
                                             RunInfo &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        LocalTensor<float> lseUb =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize,
                           HAS_MASK ? negativeIntScalar_ : NEGATIVE_MIN_VALUE_FP32);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopySoftmaxLseS1Only<LAYOUT_T>(softmaxLseGm_, lseUb, runInfo.bIdx, runInfo.n2Idx, runInfo.gIdx,
                                           runInfo.s1Idx + runInfo.vecMbaseIdx, runInfo.actVecMSize, constInfo_,
                                           qSeqLensTool_);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1) % UB_LSE_OUT_BUFCNT;
    }

    template <bool ApplyMask, bool FullColumns, int ScaleSign>
    __aicore__ inline void SoftmaxDnTile(LocalTensor<INPUT_T> &dst, LocalTensor<T> &sum, LocalTensor<T> &max,
                                         LocalTensor<T> &scores, LocalTensor<T> &exp, LocalTensor<uint8_t> &mask,
                                         const RunInfo &info)
    {
        const float scale = static_cast<float>(constInfo_.scaleValue);
        if constexpr (ApplyMask) {
            const int64_t row = static_cast<int64_t>(info.s1Idx) + info.vecMbaseIdx;
            const int64_t lower = row - info.preTokensLeftUp - static_cast<int64_t>(info.s2Idx);
            const int64_t upper = row + info.nextTokensLeftUp - static_cast<int64_t>(info.s2Idx);
            if (FullColumns && info.actVecMSize == 64U && (info.actMSizeAlign32 >> 1) == 64U) {
                FaVectorApi::DnSegmentedSoftmax<INPUT_T, s2BaseSize, ScaleSign, true, FullColumns, FA_T::causalOpt>(
                    dst, scores, sum, max, exp, mask, info.actMSizeAlign32 >> 1, info.actSingleLoopS2Size, scale,
                    negativeFloatScalar_, lower, upper);
            } else {
                FaVectorApi::DnSegmentedSoftmax<INPUT_T, s2BaseSize, ScaleSign, false, FullColumns, FA_T::causalOpt>(
                    dst, scores, sum, max, exp, mask, info.actMSizeAlign32 >> 1, info.actSingleLoopS2Size, scale,
                    negativeFloatScalar_, lower, upper);
            }
        } else {
            if (FullColumns && info.actVecMSize == 64U && (info.actMSizeAlign32 >> 1) == 64U) {
                FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize, false, true, true, true, true, 0,
                                             FullColumns, ScaleSign>(
                    dst, sum, max, scores, exp, nullptr, mask, info.actMSizeAlign32 >> 1, info.actSingleLoopS2SizeAlign,
                    info.actSingleLoopS2Size, scale, 1.0F, negativeFloatScalar_, 0.0F, false);
            } else {
                FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize, false, false, true, true, true, 0,
                                             FullColumns, ScaleSign>(
                    dst, sum, max, scores, exp, nullptr, mask, info.actMSizeAlign32 >> 1, info.actSingleLoopS2SizeAlign,
                    info.actSingleLoopS2Size, scale, 1.0F, negativeFloatScalar_, 0.0F, false);
            }
        }
    }

    template <bool ApplyMask, bool FullColumns>
    __aicore__ inline void SoftmaxDnScale(LocalTensor<INPUT_T> &dst, LocalTensor<T> &sum, LocalTensor<T> &max,
                                          LocalTensor<T> &scores, LocalTensor<T> &exp, LocalTensor<uint8_t> &mask,
                                          const RunInfo &info)
    {
        // Dispatch outside the VF hardware loops; both mask modes share this policy.
        if (constInfo_.scaleValue > 0.0F) {
            SoftmaxDnTile<ApplyMask, FullColumns, 1>(dst, sum, max, scores, exp, mask, info);
        } else if (constInfo_.scaleValue < 0.0F) {
            SoftmaxDnTile<ApplyMask, FullColumns, -1>(dst, sum, max, scores, exp, mask, info);
        } else {
            SoftmaxDnTile<ApplyMask, FullColumns, 0>(dst, sum, max, scores, exp, mask, info);
        }
    }

    template <bool ApplyMask>
    __aicore__ inline void SoftmaxDn(LocalTensor<INPUT_T> &dst, LocalTensor<T> &sum, LocalTensor<T> &max,
                                     LocalTensor<T> &scores, LocalTensor<T> &exp, LocalTensor<uint8_t> &mask,
                                     const RunInfo &info)
    {
        // A Q tail can still have a full KV tile. Its score lanes need no KV padding.
        if (info.actSingleLoopS2Size == s2BaseSize && info.actSingleLoopS2SizeAlign == s2BaseSize) {
            SoftmaxDnScale<ApplyMask, true>(dst, sum, max, scores, exp, mask, info);
        } else {
            SoftmaxDnScale<ApplyMask, false>(dst, sum, max, scores, exp, mask, info);
        }
    }

    __aicore__ inline void ProcessVec1Dn(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor,
                                         RunInfo runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize / CV_RATIO;
        static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
        static constexpr uint32_t vec1S2CopyCountDn = (mBaseSize >> 4) / CV_RATIO;
        static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
        static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;

        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (HAS_MASK) {
            attenMaskUb = ubTriangleMask_.template ReinterpretCast<uint8_t>();
        }
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        // Only boundary tiles generate causal/band predicates. Fully visible
        // tiles use the exact same VF specialization as the no-mask route.
        bool needMask = false;
        if constexpr (HAS_MASK) {
            const int64_t firstQ = static_cast<int64_t>(runInfo.s1Idx) + runInfo.vecMbaseIdx;
            const int64_t lastQ = firstQ + runInfo.actVecMSize - 1;
            const int64_t lastKv = static_cast<int64_t>(runInfo.s2Idx) + runInfo.actSingleLoopS2Size - 1;
            needMask = firstQ + runInfo.nextTokensLeftUp < lastKv ||
                       lastQ - runInfo.preTokensLeftUp > static_cast<int64_t>(runInfo.s2Idx);
        }

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

        LocalTensor<INPUT_T> stage1CastTensor =
            ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        // Runtime mask dispatch stays outside VF loops. Both modes transform raw
        // MM1 scores inside softmax; no intermediate score store or V barrier.
        if constexpr (HAS_MASK) {
            if (needMask) {
                SoftmaxDn<true>(stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, attenMaskUb, runInfo);
            } else {
                SoftmaxDn<false>(stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, attenMaskUb, runInfo);
            }
        } else {
            SoftmaxDn<false>(stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, attenMaskUb, runInfo);
        }

        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Tensor;

        // N208 has a 104-column half; a 97..104-column tail occupies 112
        // physical columns and must copy both packed banks.
        if (runInfo.actSingleLoopS2SizeAlign > vec1S2CopyLenDn) {
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, vec1S2CopyLenDn, 1,
                      static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn)});
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign +
                                  vec1S2strideDn],
                     stage1CastTensor[vec1ResOffsetDn],
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn),
                      static_cast<uint16_t>(s2BaseSize - runInfo.actSingleLoopS2SizeAlign + 1), vec1S2CopyLenDn});
        } else {
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                      static_cast<uint16_t>(vec1S2CopyLenDn - runInfo.actSingleLoopS2SizeAlign + 1), 0});
        }

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        vec1ResUbBufId_ = (vec1ResUbBufId_ + 1U) % UB_VEC1_RES_BUFCNT;
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoordS1Only gmCoord{.bIdx = info.bIdx,
                              .n2Idx = info.n2Idx,
                              .gIdx = info.gIdx,
                              .s1Idx = info.s1Idx + info.vecMbaseIdx + vecMIdx,
                              .dIdx = 0,
                              .s1DealSize = dealRowCount,
                              .dDealSize = (uint32_t)constInfo_.dSizeV};
        copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfo &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        LocalTensor<float> sumBrdcstBuf =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        FaVectorApi::BroadcastMaxSum(sumBrdcstBuf, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopy(softmaxFDSumGm_[gmOffset], sumBrdcstBuf, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1U) % UB_LSE_OUT_BUFCNT;

        LocalTensor<float> maxBrdcstBuf =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        FaVectorApi::BroadcastMaxSum(maxBrdcstBuf, maxUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxBrdcstBuf, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1U) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfo &runInfo, LocalTensor<T> &ubVec2Res, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        // FlashOutputDnWorkspaceVF writes rows with the 64-aligned workspace pitch.
        int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (AttentionCommon::BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm_[gmOffset], ubVec2Res, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(RunInfo runInfo)
    {
        uint32_t mm2ResUbBufId = mm2ResUbBufId_;
        mm2ResUbBufId_ = (mm2ResUbBufId_ + 1) % UB_MM2_RES_BUFCNT;
        uint32_t mm2SyncIdx = CC_MM2_0 + mm2ResUbBufId;
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mm2SyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mm2SyncIdx);
            return;
        }

        const uint32_t dSizeV = constInfo_.dSizeV;
        const uint32_t dSizeAligned16 = (dSizeV + 15U) / 16U * 16U;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mm2SyncIdx);
        {
            LocalTensor<T> mm2ResUbTensor =
                ubMm2ResBuffers_[mm2ResUbBufId * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<T>();
            Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            if (runInfo.isFirstS2Loop) {
                DataCopy(ubVec2Res_, mm2ResUbTensor, dSizeV * 64U);
            } else {
                LocalTensor<T> expUb =
                    softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
                if (runInfo.isLastS2Loop) {
                    FaVectorApi::FlashUpdateDnRows64VF<true>(ubVec2Res_, mm2ResUbTensor, expUb,
                                                             static_cast<uint16_t>(dSizeV));
                } else {
                    FaVectorApi::FlashUpdateDnRows64VF<false>(ubVec2Res_, mm2ResUbTensor, expUb,
                                                              static_cast<uint16_t>(dSizeV));
                }
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        }

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mm2SyncIdx);

        if (runInfo.isLastS2Loop) {
            // Borrow the next Vec1 UB slot without advancing its ring index.
            static_assert(UB_VEC1_RES_BUF_BYTES >= 64U * dTemplateAlign64 * sizeof(OUTPUT_T),
                          "Vec1 slot must hold the packed Vec2 output");
            uint32_t vec1ResSyncIdx = UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_;
            LocalTensor<T> vec2TmpUb =
                ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<T>();
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<float> sumUb =
                softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
            LocalTensor<float> maxUb =
                softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
            if constexpr (HAS_MASK) {
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                FaVectorApi::FinalizeDnMaskRowsVF(ubVec2Res_, sumUb, maxUb, runInfo.actVecMSize,
                                                  static_cast<uint16_t>(dSizeV));
                AscendC::PipeBarrier<PIPE_V>();
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            }
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
            if (constInfo_.enableFlashDecode && runInfo.isS2SplitCore) {
                constexpr uint32_t rowsPerCopy = UB_VEC1_RES_BUF_BYTES / (dTemplateAlign64 * sizeof(T));
                static_assert(rowsPerCopy > 0, "Vec1 slot must hold one workspace row");
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                for (uint32_t mStartVec = 0; mStartVec < runInfo.actVecMSize; mStartVec += rowsPerCopy) {
                    uint32_t mDealSize = AttentionCommon::Min(rowsPerCopy, runInfo.actVecMSize - mStartVec);
                    Mutex::Lock<PIPE_V>(vec1ResSyncIdx);
                    FaVectorApi::FlashOutputDnWorkspaceVF(
                        vec2TmpUb, ubVec2Res_[mStartVec], sumUb[mStartVec], static_cast<uint16_t>(mDealSize),
                        static_cast<uint16_t>(dSizeV), static_cast<uint16_t>(dTemplateAlign64));
                    Mutex::Unlock<PIPE_V>(vec1ResSyncIdx);
                    Mutex::Lock<PIPE_MTE3>(vec1ResSyncIdx);
                    Bmm2ResForFDCopyOut(runInfo, vec2TmpUb, mStartVec, mDealSize);
                    Mutex::Unlock<PIPE_MTE3>(vec1ResSyncIdx);
                }
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            } else {
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                Mutex::Lock<PIPE_V>(vec1ResSyncIdx);
                LocalTensor<OUTPUT_T> vec2ResUbTensor = vec2TmpUb.template ReinterpretCast<OUTPUT_T>();
                FaVectorApi::FlashPackOutputDnRows64VF(vec2ResUbTensor, ubVec2Res_, sumUb,
                                                       static_cast<uint16_t>(dSizeV));
                AscendC::PipeBarrier<PIPE_V>();
                LocalTensor<OUTPUT_T> attenOut = ubVec2Res_.template ReinterpretCast<OUTPUT_T>();
                uint64_t srcList[16], dstList[16];
                for (uint32_t dOffset = 0; dOffset < dSizeAligned16; dOffset += 16) {
                    for (uint32_t i = 0; i < 16; ++i) {
                        srcList[i] = reinterpret_cast<uint64_t>(vec2ResUbTensor.GetPhyAddr()) +
                                     ((dOffset + i) * 64U) * sizeof(OUTPUT_T);
                        dstList[i] = reinterpret_cast<uint64_t>(attenOut.GetPhyAddr()) +
                                     (i * dTemplateAlign64 + dOffset) * sizeof(OUTPUT_T);
                    }
                    TransDataTo5HD<OUTPUT_T>(dstList, srcList, {false, false, 4, dTemplateAlign64, 1});
                }
                // The next Vec1 can reuse this slot on PIPE_V without an intervening MTE3.
                AscendC::PipeBarrier<PIPE_V>();
                Mutex::Unlock<PIPE_V>(vec1ResSyncIdx);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2DataCopyOutTrans(runInfo, attenOut, 0, runInfo.actVecMSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            }
        }
    }
};

// AIC/AIV 分编译占位（Mix kernel 在 AIC 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockVecDummyDnUnmerged {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;

    __aicore__ inline FANoQuantGqaBlockVecDummyDnUnmerged(ConstInfo_t &constInfo,
                                                          SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                          SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace FlashAttnKernel
#endif
