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
 * \brief FANoQuantGqaBlockVecDn —— Dn 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTN_BLOCK_VEC_DN_H_
#define FLASH_ATTN_BLOCK_VEC_DN_H_

#include <limits>
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
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
class FANoQuantGqaBlockVecDn {
public:
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

    static constexpr bool D64_DEEP_PRELOAD =
        dBaseSize == 64 && dVBaseSize == 64 && mBaseSize == 128 && (s2BaseSize == 128 || s2BaseSize == 256);

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t CC_MM_0 = 0U;
    static constexpr uint32_t CC_MM_1 = 1U;
    static constexpr uint32_t CC_MM_2 = 2U;
    static constexpr uint32_t CC_MM_3 = 3U;
    static constexpr uint32_t CC_L1P_0 = (D64_DEEP_PRELOAD ? 4U : 5U) + 0U;
    static constexpr uint32_t CC_L1P_1 = (D64_DEEP_PRELOAD ? 4U : 5U) + 1U;
    static constexpr uint32_t CC_L1P_2 = (D64_DEEP_PRELOAD ? 4U : 5U) + 2U;

    // 核内同步ID
    // MTE3<->V, 输出buffer
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = 3;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = 5;

    // L1
    static constexpr uint32_t L1_P_BUFCNT = D64_DEEP_PRELOAD ? 4U : 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    __aicore__ inline uint32_t PSlotOffset(uint32_t id) const
    {
        return qPreScale_ ? (id & 1U) * 262144U + (id >> 1U) * 65536U : id * L1_P_BUF_BYTES;
    }
    LocalTensor<uint8_t> l1PBuffers_;

    // UB
    static constexpr bool SPLIT_MM_QUEUES = D64_DEEP_PRELOAD && s2BaseSize == 256;
    static constexpr uint32_t UB_MM_RES_BUFCNT =
        (D64_DEEP_PRELOAD && s2BaseSize == 256) ? 4U : ((dBaseSize > 128 || s2BaseSize > 128) ? 2U : 4U);
    static constexpr uint32_t UB_MM_RES_BUF_BYTES =
        mBaseSize / CV_RATIO * (s2BaseSize > dVBaseSize ? s2BaseSize : dVBaseSize) * sizeof(T) +
        ((dBaseSize == 64 && dVBaseSize == 64 && mBaseSize == 128 && s2BaseSize == 128) ? 128U : 0U);
    static constexpr uint32_t UB_MM1_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_MM2_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
    // bmm1/bmm2(mmRes) 区域总字节数：FD block 以此作为 FD 业务区的 UB 起始偏移
    static constexpr uint32_t UB_MM_RES_TOTAL_BYTES =
        SPLIT_MM_QUEUES ? 2U * (UB_MM1_BUF_BYTES + UB_MM2_BUF_BYTES + 256U) : UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES;
    // FD 业务区在 UB 中的起始字节偏移，供 kernel 传给 FD block 的 InitBuffers
    static __aicore__ inline constexpr uint32_t GetFdBaseOffset()
    {
        return UB_MM_RES_TOTAL_BYTES;
    }
    LocalTensor<uint8_t> ubMmResBuffers_;
    uint32_t mmResBufId_ = 0;
    uint32_t mm1ResBufId_ = 0U;
    uint32_t mm2ResBufId_ = 0U;

    __aicore__ inline uint32_t AcquireMmSlot(bool second)
    {
        if constexpr (SPLIT_MM_QUEUES) {
            if (second) {
                uint32_t id = mm2ResBufId_;
                mm2ResBufId_ = (mm2ResBufId_ + 1U) % 2U;
                return 2U + id;
            }
            uint32_t id = mm1ResBufId_;
            mm1ResBufId_ = (mm1ResBufId_ + 1U) % 2U;
            return id;
        } else {
            uint32_t id = mmResBufId_;
            mmResBufId_ = (mmResBufId_ + 1U) % UB_MM_RES_BUFCNT;
            return id;
        }
    }

    __aicore__ inline uint32_t MmSlotOffset(uint32_t id)
    {
        if constexpr (SPLIT_MM_QUEUES) {
            // Rotate the four slot bases through UB phases 0/128/256/384B.
            return id < 2U ? id * (UB_MM1_BUF_BYTES + 128U) :
                             2U * (UB_MM1_BUF_BYTES + 128U) + (id - 2U) * (UB_MM2_BUF_BYTES + 128U);
        } else {
            return id * UB_MM_RES_BUF_BYTES;
        }
    }

    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    LocalTensor<T> ubVec2Res_; // 存放vec2阶段VEC的中间处理结果, 并且作为attn_out的输出buffer, 需配对的MTE3和V的同步ID

    static constexpr uint32_t UB_VEC1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = (mBaseSize / CV_RATIO + 1U) * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> ubVec1ResBuffers_;
    uint32_t vec1ResUbBufId_ = 0;

    static constexpr uint32_t UB_SOFTMAX_MAX_BUFCNT = D64_DEEP_PRELOAD ? 4U : 3U;
    static constexpr uint32_t UB_SOFTMAX_MAX_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxSumBuf_;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUFCNT = D64_DEEP_PRELOAD ? 4U : 3U;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxMaxBuf_;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUFCNT = D64_DEEP_PRELOAD ? 4U : 3U;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxExpBuf_;

    static constexpr uint32_t UB_LSE_OUT_BUFCNT = SPLIT_MM_QUEUES ? 1U : 2U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * 8 * static_cast<uint32_t>(sizeof(T)), 256U);
    static_assert(UB_MM_RES_TOTAL_BYTES + UB_VEC2_RES_BUF_BYTES + UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES +
                          UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES +
                          UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES +
                          UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES + UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES <=
                      248U * 1024U,
                  "FA buffers exceed available UB");
    LocalTensor<uint8_t> ubLseOutBuffers_;
    uint32_t lseOutUbBufId_ = 0;

    const ConstInfo_t &constInfo_;

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool_;

    // GM
    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensor_;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxFDSumGm_;
    GlobalTensor<float> softmaxFDMaxGm_;
    GlobalTensor<float> sinkGm_;

    T negativeFloatScalar_;
    bool qPreScale_ = false;
    uint32_t qPrescaleSlot_ = 0;
    GlobalTensor<INPUT_T> queryScaleGm_;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecDn(ConstInfo_t &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                             SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                     __gm__ uint8_t *workspace, __gm__ uint8_t *query)
    {
        queryScaleGm_.SetGlobalBuffer((__gm__ INPUT_T *)query);
        if constexpr (SPLIT_MM_QUEUES && IsSameType<INPUT_T, bfloat16_t>::value && !PAGE_ATTENTION && !FA_T::hasMask &&
                      LAYOUT_T == FA_LAYOUT::BNSD && LAYOUT_KV == FA_LAYOUT::BNSD &&
                      FA_T::attnOutLayout == FA_LAYOUT::BNSD) {
            qPreScale_ = constInfo_.bSize == 1 && constInfo_.gSize == 1 && constInfo_.dSize == 64 &&
                         constInfo_.dSizeV == 64 && constInfo_.scaleValue == 0.125f && constInfo_.s1Size > 0 &&
                         constInfo_.s1Size % 128 == 0 && constInfo_.s2Size > 0 && constInfo_.s2Size % 256 == 0 &&
                         qSeqLensTool_.GetActualSeqLength(0) == constInfo_.s1Size &&
                         kvSeqLensTool_.GetActualSeqLength(0) == constInfo_.s2Size;
        }
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        InitAttenOutBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSizeV,
                           outGmTensor_, attentionOut);

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
        // l1P 三缓冲
        uint32_t addrL1 = 0;
        l1PBuffers_ =
            LocalTensor<uint8_t>(TPosition::A1, addrL1, qPreScale_ ? 512U * 1024U : L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMmResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                               UB_MM_RES_TOTAL_BYTES); // CV通信BUF
        addrUb = UB_MM_RES_TOTAL_BYTES;
        ubVec2Res_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_VEC2_RES_BUF_BYTES)
                         .template ReinterpretCast<T>(); // 输出BUF: attn_out拷出
        addrUb += UB_VEC2_RES_BUF_BYTES;
        ubVec1ResBuffers_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES); // 2 * 32.25K = 64.5K, 输出BUF: softmax结果拷贝至L1
        addrUb += UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES;

        // softmaxSum×3 + softmaxMax×3 + softmaxExp×3，各 256 bytes
        softmaxSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES;
        softmaxMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES;
        softmaxExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES;

        ubLseOutBuffers_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_LSE_OUT_BUFCNT *
                UB_LSE_OUT_BUF_BYTES); // 输出BUF:
                                       // FD中间结果SUM和MAX拷出至GM，或者LSE结果拷出；按actVecMSize*8(fp32块对齐)定容
        addrUb += UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES;
    }

    __aicore__ inline void ResetSoftmaxBuffer(uint32_t slotIdx, const RunInfo &runInfo)
    {
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T);
        LocalTensor<T> sumUb = softmaxSumBuf_[slotIdx * softmaxBufElementCount];
        LocalTensor<T> maxUb = softmaxMaxBuf_[slotIdx * softmaxBufElementCount];
        if (constInfo_.learnableSinkFlag && runInfo.isFirstFdBlock) {
            Duplicate<T>(sumUb, static_cast<T>(1), softmaxBufElementCount);

            uint32_t gs1Start = runInfo.gS1Idx + runInfo.vecMbaseIdx;

            Mutex::Lock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();
                if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
                    AttentionCommon::SinkCopyInS1G(sinkTmpUb, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                                   runInfo.n2Idx, constInfo_.gSize);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    AttentionCommon::SinkCopyInGS1(sinkTmpUb, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                                   runInfo.n2Idx, constInfo_.gSize);
                }
            }
            Mutex::Unlock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

            Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();

                if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    AttentionCommon::SinkExpandMaxVf<T, float, true>(maxUb, sinkTmpUb, gs1Start, runInfo.actVecMSize,
                                                                     runInfo.actS1Size, constInfo_.gSize);
                } else {
                    AttentionCommon::SinkExpandMaxVf<T, float, false>(maxUb, sinkTmpUb, gs1Start, runInfo.actVecMSize,
                                                                      runInfo.actS1Size, constInfo_.gSize);
                }
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        } else {
            Duplicate<T>(sumUb, static_cast<T>(0), softmaxBufElementCount);
            Duplicate<T>(maxUb, static_cast<T>(-std::numeric_limits<float>::infinity()), softmaxBufElementCount);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        for (uint32_t slot = 0; slot < UB_MM_RES_BUFCNT; ++slot) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_0 + slot);
        }
    }

    __aicore__ inline void UnInitCrossCoreSync()
    {
        if (qPreScale_ && constInfo_.subBlockIdx == 0) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(10);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(11);
        }
    }

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline void PrepareScaledQuery(RunInfo &runInfo)
    {
        if constexpr (SPLIT_MM_QUEUES && IsSameType<INPUT_T, bfloat16_t>::value) {
            if (!qPreScale_ || constInfo_.subBlockIdx != 0) {
                return;
            }
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(10 + qPrescaleSlot_);
            Mutex::Lock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            auto qUb = ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
            uint64_t offset = (static_cast<uint64_t>(runInfo.n2Idx) * constInfo_.s1Size + runInfo.s1Idx) * 64U;
            DataCopy(qUb, queryScaleGm_[offset], 8192U);
            Mutex::Unlock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            FaVectorApi::ScaleQueryD64VF((__ubuf__ bfloat16_t *)qUb.GetPhyAddr());
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            auto qL1 = LocalTensor<uint8_t>(TPosition::A1, qPrescaleSlot_ * 262144U + 131072U, 16384U)
                           .template ReinterpretCast<INPUT_T>();
            for (uint32_t c = 0; c < 4; ++c) {
                DataCopy(qL1[c * 2048U], qUb[c * 16U], {128, 1, 3, 0});
            }
            Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(8 + qPrescaleSlot_);
            qPrescaleSlot_ = (qPrescaleSlot_ + 1U) % 2U;
        }
    }

    __aicore__ inline void ProcessVec1(RunInfo runInfo)
    {
        uint32_t mmResUbBufId = AcquireMmSlot(false);
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t mmSyncIdx = CC_MM_0 + mmResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[PSlotOffset(pL1BufId)].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMmResBuffers_[MmSlotOffset(mmResUbBufId)].template ReinterpretCast<T>();

        if (unlikely(runInfo.isFirstS2Loop)) {
            PrepareScaledQuery(runInfo);
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT, runInfo);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        ProcessVec1Dn(pL1Tensor, mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 通知BMM2: Vec1已读完mmRes, 可覆写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx);
        Vec1PostProcess(runInfo);
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
        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
            uint32_t prefixBS1 = qSeqLensTool_.cuSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo_.gSize * constInfo_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfo_t>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                            runInfo.actVecMSize, constInfo_);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfo_t>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                         constInfo_);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfo_t>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                         constInfo_, qActSeqLens);
        }
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1) % UB_LSE_OUT_BUFCNT;
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
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

        float descaleQK = 1.0;

        LocalTensor<INPUT_T> stage1CastTensor =
            ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        // VEC1_COMPUTE_BEGIN
        if constexpr (SPLIT_MM_QUEUES && IsSameType<INPUT_T, bfloat16_t>::value) {
            if (qPreScale_ && runInfo.actVecMSize == 64 && (runInfo.actMSizeAlign32 >> 1) == 64 &&
                runInfo.actSingleLoopS2Size == 256 && runInfo.actSingleLoopS2SizeAlign == 256) {
                FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize, false, true, true>(
                    stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, nullptr, attenMaskUb,
                    runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, false);
            } else {
                FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, nullptr, attenMaskUb,
                    runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, false);
            }
        } else {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, nullptr, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, false);
        }
        // VEC1_COMPUTE_END

        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Tensor;

        if (runInfo.actSingleLoopS2Size > vec1S2CopyLenDn) {
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

    __aicore__ inline void Vec1PostProcess(RunInfo runInfo)
    {
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.n2Idx,
                                .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
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
        int64_t dSizeAligned64 = (int64_t)dVBaseSize;
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (AttentionCommon::BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm_[gmOffset], ubVec2Res, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2DnOutput(RunInfo &runInfo, uint32_t slot, uint32_t sync)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(sync);
        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        auto cur = ubMmResBuffers_[MmSlotOffset(slot)].template ReinterpretCast<T>();
        if (runInfo.isFirstS2Loop) {
            DataCopy(ubVec2Res_, cur, 4096U);
        } else {
            auto factor =
                softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
            if (runInfo.isLastS2Loop) {
                FaVectorApi::FlashUpdateD64DnVF<true>((__ubuf__ float *)ubVec2Res_.GetPhyAddr(),
                                                      (__ubuf__ float *)cur.GetPhyAddr(),
                                                      (__ubuf__ float *)factor.GetPhyAddr());
            } else {
                FaVectorApi::FlashUpdateD64DnVF<false>((__ubuf__ float *)ubVec2Res_.GetPhyAddr(),
                                                       (__ubuf__ float *)cur.GetPhyAddr(),
                                                       (__ubuf__ float *)factor.GetPhyAddr());
            }
        }
        if (runInfo.isLastS2Loop) {
            // Current PV is fully consumed; keep its slot owned until output MTE3 completes.
            AscendC::PipeBarrier<PIPE_V>();
            auto sum = softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
            auto output = cur.template ReinterpretCast<OUTPUT_T>();
            FaVectorApi::FlashPackOutputD64DnVF((__ubuf__ bfloat16_t *)output.GetPhyAddr(),
                                                (__ubuf__ float *)ubVec2Res_.GetPhyAddr(),
                                                (__ubuf__ float *)sum.GetPhyAddr());
            AscendC::PipeBarrier<PIPE_V>();
            auto finalOutput = ubVec2Res_.template ReinterpretCast<OUTPUT_T>();
            uint64_t srcList[16], dstList[16];
            for (uint32_t c = 0; c < 64; c += 16) {
                for (uint32_t i = 0; i < 16; ++i) {
                    srcList[i] = reinterpret_cast<uint64_t>(output.GetPhyAddr()) + ((c + i) * 64U) * sizeof(OUTPUT_T);
                    dstList[i] =
                        reinterpret_cast<uint64_t>(finalOutput.GetPhyAddr()) + (i * 64U + c) * sizeof(OUTPUT_T);
                }
                TransDataTo5HD<OUTPUT_T>(dstList, srcList, {false, false, 4, 64, 1});
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            Bmm2DataCopyOutTrans(runInfo, finalOutput, 0, 64);
            Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(sync);
        } else {
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(sync);
        }
    }

    __aicore__ inline void ProcessVec2(RunInfo runInfo)
    {
        uint32_t mmResUbBufId = AcquireMmSlot(true);
        uint32_t mmSyncIdx = CC_MM_0 + mmResUbBufId;
        if constexpr (SPLIT_MM_QUEUES && IsSameType<INPUT_T, bfloat16_t>::value &&
                      IsSameType<OUTPUT_T, bfloat16_t>::value) {
            if (qPreScale_ && !runInfo.isS2SplitCore) {
                ProcessVec2DnOutput(runInfo, mmResUbBufId, mmSyncIdx);
                return;
            }
        }
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            return;
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        {
            Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            LocalTensor<T> mm2ResUbTensor = ubMmResBuffers_[MmSlotOffset(mmResUbBufId)].template ReinterpretCast<T>();
            if (unlikely(runInfo.isFirstS2Loop)) {
                uint32_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
                DataCopy(ubVec2Res_, mm2ResUbTensor, vec2CalcSize);
            } else {
                LocalTensor<T> expUb =
                    softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
                LocalTensor<T> pScaleUb;

                float deSCalePreVValue = 1.0f;
                if (!runInfo.isLastS2Loop) {
                    if constexpr (SPLIT_MM_QUEUES && IsSameType<INPUT_T, bfloat16_t>::value && dTemplateAlign64 == 64) {
                        if (runInfo.actVecMSize == 64) {
                            FaVectorApi::FlashUpdateD64FullVF((__ubuf__ float *)ubVec2Res_.GetPhyAddr(),
                                                              (__ubuf__ float *)mm2ResUbTensor.GetPhyAddr(),
                                                              (__ubuf__ float *)ubVec2Res_.GetPhyAddr(),
                                                              (__ubuf__ float *)expUb.GetPhyAddr());
                        } else {
                            FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                                ubVec2Res_, mm2ResUbTensor, ubVec2Res_, expUb, pScaleUb, runInfo.actVecMSize,
                                dTemplateAlign64, 1.0, 1.0);
                        }
                    } else {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                            ubVec2Res_, mm2ResUbTensor, ubVec2Res_, expUb, pScaleUb, runInfo.actVecMSize,
                            dTemplateAlign64, 1.0, 1.0);
                    }
                } else {
                    LocalTensor<float> sumUb = softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) *
                                                              (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        ubVec2Res_, mm2ResUbTensor, ubVec2Res_, expUb, pScaleUb, sumUb, runInfo.actVecMSize,
                        dTemplateAlign64, 1.0, 1.0);
                }
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 通知下个BMM1: Vec2已读完mmRes, slot空闲

        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                LocalTensor<float> sumUb =
                    softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    ubVec2Res_, ubVec2Res_, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, 0.0F);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            }
            uint32_t mStartVec = 0;
            uint32_t mDealSize = runInfo.actVecMSize;
            if (constInfo_.enableFlashDecode && runInfo.isS2SplitCore) {
                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2ResForFDCopyOut(runInfo, ubVec2Res_, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            } else {
                LocalTensor<OUTPUT_T> attenOut;
                int64_t dSizeAligned64 = (int64_t)dVBaseSize;

                attenOut.SetAddr(ubVec2Res_.address_);
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                Cast(attenOut, ubVec2Res_, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);

                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            }
        }
    }
};

// AIC/AIV 分编译占位（Mix kernel 在 AIC 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockVecDummyDn {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;

    __aicore__ inline FANoQuantGqaBlockVecDummyDn(ConstInfo_t &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                  SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace FlashAttnKernel
#endif // FLASH_ATTN_BLOCK_VEC_DN_H_
