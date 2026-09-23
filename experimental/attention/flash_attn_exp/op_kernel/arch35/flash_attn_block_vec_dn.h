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

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast_arch35.h"
#include "vf/vf_flash_decode_arch35.h"

#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/init_output.h"

#include "memory_copy_arch35.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

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
    static constexpr bool HAS_REL = FA_T::hasRel;

    using T = float;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)FA_T::dVBaseSize);

    static constexpr uint32_t DB = 2;
    // 索引使用 loop & (DB - 1) 代替 loop % DB，要求 DB 必须是2的幂，否则位掩码结果错误
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t CC_BMM1_0 = 0U;
    static constexpr uint32_t CC_BMM1_1 = 1U;
    static constexpr uint32_t CC_BMM2_0 = 2U;
    static constexpr uint32_t CC_BMM2_1 = 3U;
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
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> l1PBuffers_;

    // UB
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
    LocalTensor<uint8_t> ubMm2ResBuffers_;

    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    LocalTensor<uint8_t> ubMm1ResBuffers_;

    LocalTensor<T> ubVec2Res_; // 存放vec2阶段VEC的中间处理结果, 并且作为attn_out的输出buffer, 需配对的MTE3和V的同步ID

    static constexpr uint32_t UB_VEC1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = 33024U;
    LocalTensor<uint8_t> ubVec1ResBuffers_;
    uint32_t vec1ResUbBufId_ = 0;

    static constexpr uint32_t UB_SOFTMAX_MAX_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_MAX_BUF_BYTES = 256U;
    LocalTensor<T> softmaxSumBuf_;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUF_BYTES = 256U;
    LocalTensor<T> softmaxMaxBuf_;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUF_BYTES = 256U;
    LocalTensor<T> softmaxExpBuf_;

    static constexpr uint32_t UB_LSE_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES = 2048U;
    LocalTensor<uint8_t> ubLseOutBuffers_;
    uint32_t lseOutUbBufId_ = 0;

    static constexpr uint32_t UB_REL_BUFCNT = 2U; // rel 转置结果双缓冲，临时拷贝区借用 ubVec1ResBuffers_ 槽位
    static constexpr uint32_t UB_REL_BUF_BYTES = 8192U; // TODO: 根据实际分配
    LocalTensor<INPUT_T> relHBuffers_;
    LocalTensor<INPUT_T> relWBuffers_;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;
    const ConstInfoX &constInfo_;

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
    GlobalTensor<INPUT_T> relHGm_;
    GlobalTensor<INPUT_T> relWGm_;

    T negativeFloatScalar_;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecDn(ConstInfoX &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                             SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink, __gm__ uint8_t *relH,
                                     __gm__ uint8_t *relW, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                     __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        InitAttenOutBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSizeV,
                           outGmTensor_, attentionOut);

        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if (HAS_REL) {
            relHGm_.SetGlobalBuffer((__gm__ INPUT_T *)relH);
            relWGm_.SetGlobalBuffer((__gm__ INPUT_T *)relW);
        }

        if (constInfo_.enableFlashDecode) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // l1P 三缓冲
        uint32_t addrL1 = 0;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                                UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES); // 2 * 32K = 64K, CV通信BUF
        addrUb = UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                                UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES); // 2 * 32K = 64K, CV通信BUF
        addrUb += UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES;
        ubVec2Res_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, 32768U)
                         .template ReinterpretCast<T>(); // 32K, 输出BUF: attn_out拷出
        addrUb += 32768U;
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

        relHBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_REL_BUFCNT * UB_REL_BUF_BYTES)
                           .template ReinterpretCast<INPUT_T>(); // REL buf, 2 * 8K = 16K
        addrUb += UB_REL_BUFCNT * UB_REL_BUF_BYTES;

        relWBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_REL_BUFCNT * UB_REL_BUF_BYTES)
                           .template ReinterpretCast<INPUT_T>(); // REL buf, 2 * 8K = 16K
        addrUb += UB_REL_BUFCNT * UB_REL_BUF_BYTES;

        // rel_h/rel_w 的 GM→UB 临时拷贝区借用 ubVec1ResBuffers_ 槽位(relTempBuf_ 已省去)

        ubLseOutBuffers_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_LSE_OUT_BUFCNT *
                UB_LSE_OUT_BUF_BYTES); // 2 * 2K = 4K, 输出BUF: FD中间结果SUM和MAX拷出至GM，或者LSE结果拷出
        addrUb += UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES;
    }

    __aicore__ inline void ResetSoftmaxBuffer(uint32_t slotIdx)
    {
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T);
        LocalTensor<T> sumUb = softmaxSumBuf_[slotIdx * softmaxBufElementCount];
        LocalTensor<T> maxUb = softmaxMaxBuf_[slotIdx * softmaxBufElementCount];
        Duplicate<T>(sumUb, static_cast<T>(0), softmaxBufElementCount);
        Duplicate<T>(maxUb, static_cast<T>(-std::numeric_limits<float>::infinity()), softmaxBufElementCount);
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_1);
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline void ProcessVec1(RunInfoX runInfo)
    {
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CC_BMM1_0 + mm1ResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        if (unlikely(runInfo.isFirstS2Loop)) {
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);
        ProcessVec1Dn(pL1Tensor, mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(
            c1v1CrossCoreSyncIdx); // C1与V1的反向同步, C1收到后可以启动FIXPIPE向UB的写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx);
        Vec1PostProcess(runInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        if (constInfo_.needInitOutput) {
            uint32_t vecCoreNum = 2 * constInfo_.coreNum;
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

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
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
                                             RunInfoX &runInfo)
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
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                           runInfo.actVecMSize, constInfo_);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo_);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo_, qActSeqLens);
        }
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ProcessVec1Dn(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor,
                                         RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
        static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
        static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
        static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
        static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;

        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<INPUT_T> relHUb;
        LocalTensor<INPUT_T> relWUb;
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        const uint32_t relBufId = runInfo.loop & (DB - 1);
        if constexpr (HAS_REL) {
            relHUb = relHBuffers_[relBufId * (UB_REL_BUF_BYTES / sizeof(INPUT_T))];
            relWUb = relWBuffers_[relBufId * (UB_REL_BUF_BYTES / sizeof(INPUT_T))];
            AttnRelCopyIn(runInfo, vec1ResUbBufId_);
        }

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

        if constexpr (HAS_REL) {
            AttnRelTranspose(relHUb, relWUb, runInfo, vec1ResUbBufId_);
        }

        float descaleQK = 1.0;

        LocalTensor<INPUT_T> stage1CastTensor =
            ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize, false, HAS_REL>(
            stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, nullptr, attenMaskUb, relHUb, relWUb,
            runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
            static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, false, 1.0, 0.0,
            constInfo_.relWSize, runInfo.s2Idx);

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

    __aicore__ inline void Vec1PostProcess(RunInfoX runInfo)
    {
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
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

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
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

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
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

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, LocalTensor<T> &ubVec2Res, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVBaseSize;
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm_[gmOffset], ubVec2Res, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(RunInfoX runInfo)
    {
        uint32_t mm2ResUbBufId = runInfo.loop % UB_MM2_RES_BUFCNT;
        uint32_t c2v2CrossCoreSyncIdx = CC_BMM2_0 + mm2ResUbBufId;
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c2v2CrossCoreSyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c2v2CrossCoreSyncIdx);
            return;
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c2v2CrossCoreSyncIdx);
        {
            Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            LocalTensor<T> mm2ResUbTensor =
                ubMm2ResBuffers_[mm2ResUbBufId * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<T>();
            if (unlikely(runInfo.isFirstS2Loop)) {
                uint32_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
                DataCopy(ubVec2Res_, mm2ResUbTensor, vec2CalcSize);
            } else {
                LocalTensor<T> expUb =
                    softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
                LocalTensor<T> pScaleUb;

                float deSCalePreVValue = 1.0f;
                if (!runInfo.isLastS2Loop) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        ubVec2Res_, mm2ResUbTensor, ubVec2Res_, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        1.0, 1.0);
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
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(
            c2v2CrossCoreSyncIdx); // mmRes在之后不能使用, 否则与C2的FIXPIPE读写数据冲突

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

    __aicore__ inline void AttnRelCopyIn(const RunInfoX &runInfo, uint32_t vec1ResUbBufId)
    {
        // 拷贝REL进入Ub, 借用 ubVec1ResBuffers_
        // MTE2 写 rel 临时数据 → V 转置读取+写 stage1 → MTE3 拷出 stage1 → (下下轮) MTE2 再写
        LocalTensor<INPUT_T> relHTempTensor =
            ubVec1ResBuffers_[vec1ResUbBufId * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        LocalTensor<INPUT_T> relWTempTensor = relHTempTensor[UB_REL_BUF_BYTES / sizeof(INPUT_T)];

        Mutex::Lock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId);
        RelCopyIn(relHTempTensor, relWTempTensor, relHGm_, relWGm_, runInfo);
        Mutex::Unlock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId);
    }

    // rel 分段拷贝: 长度(segRows * relSize * sizeof(INPUT_T))在 M 尾块/小 M 场景不保证 32B 对齐
    // (如 S1=1 单行 x relHSize=8 => 16B), 硬件按 32B 块搬运时会静默丢弃不足一个块的数据,
    // 导致尾行 bias 读到残留 UB 值。对齐时走 DataCopy 快路径, 非对齐走 DataCopyPad 补尾块。
    __aicore__ inline void RelDataCopy(LocalTensor<INPUT_T> dst, GlobalTensor<INPUT_T> src, int64_t offset,
                                       uint32_t elemCount)
    {
        constexpr uint32_t byteAlign = 32U;
        if ((elemCount * sizeof(INPUT_T)) % byteAlign == 0U) {
            DataCopy(dst, src[offset], elemCount);
            return;
        }
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elemCount * sizeof(INPUT_T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyPad(dst, src[offset], copyParams, padParams);
    }

    __aicore__ inline void AttnRelTranspose(LocalTensor<INPUT_T> &relHUb, LocalTensor<INPUT_T> &relWUb,
                                            const RunInfoX &runInfo, uint32_t vec1ResUbBufId)
    {
        LocalTensor<INPUT_T> relHTempTensor =
            ubVec1ResBuffers_[vec1ResUbBufId * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        LocalTensor<INPUT_T> relWTempTensor = relHTempTensor[UB_REL_BUF_BYTES / sizeof(INPUT_T)];
        // 进行转置，使用 kernel 访问的实际 stride (actMSizeAlign32 >> 1) 作为行间距，
        // 确保 LoadAlign(rel_h + row * m) 读到的行布局与转置输出一致
        uint32_t mStride = runInfo.actMSizeAlign32 >> 1;
        Transpose(relHUb, relHTempTensor, runInfo.actVecMSize, constInfo_.relHSize, mStride);
        Transpose(relWUb, relWTempTensor, runInfo.actVecMSize, constInfo_.relWSize, mStride);
    }

    __aicore__ inline void RelCopyIn(LocalTensor<INPUT_T> &relHUb, LocalTensor<INPUT_T> &relWUb,
                                     GlobalTensor<INPUT_T> &relHGm, GlobalTensor<INPUT_T> &relWGm,
                                     const RunInfoX &runInfo)
    {
        uint32_t n1Size = constInfo_.n2Size * constInfo_.gSize;
        uint32_t absGS1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx;

        uint32_t s1Idx = 0;
        uint32_t gIdx = 0;
        if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
            s1Idx = absGS1Idx / constInfo_.gSize;
            gIdx = absGS1Idx % constInfo_.gSize;
        } else {
            s1Idx = absGS1Idx % runInfo.actS1Size;
            gIdx = absGS1Idx / runInfo.actS1Size;
        }
        uint32_t n1Idx = runInfo.n2Idx * constInfo_.gSize + gIdx;

        if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
            uint32_t prefixS1 = qSeqLensTool_.cuSeqLensParser.GetTBase(runInfo.bIdx);
            int64_t relHOffset = prefixS1 * n1Size * constInfo_.relHSize + s1Idx * n1Size * constInfo_.relHSize +
                                 n1Idx * constInfo_.relHSize;
            int64_t relWOffset = prefixS1 * n1Size * constInfo_.relWSize + s1Idx * n1Size * constInfo_.relWSize +
                                 n1Idx * constInfo_.relWSize;
            uint32_t relHLen = runInfo.actVecMSize * constInfo_.relHSize;
            uint32_t relWLen = runInfo.actVecMSize * constInfo_.relWSize;
            RelDataCopy(relHUb, relHGm, relHOffset, relHLen);
            RelDataCopy(relWUb, relWGm, relWOffset, relWLen);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
            int64_t relHOffset = runInfo.bIdx * constInfo_.s1Size * n1Size * constInfo_.relHSize +
                                 s1Idx * n1Size * constInfo_.relHSize + n1Idx * constInfo_.relHSize;
            int64_t relWOffset = runInfo.bIdx * constInfo_.s1Size * n1Size * constInfo_.relWSize +
                                 s1Idx * n1Size * constInfo_.relWSize + n1Idx * constInfo_.relWSize;
            uint32_t relHLen = runInfo.actVecMSize * constInfo_.relHSize;
            uint32_t relWLen = runInfo.actVecMSize * constInfo_.relWSize;
            RelDataCopy(relHUb, relHGm, relHOffset, relHLen);
            RelDataCopy(relWUb, relWGm, relWOffset, relWLen);
        } else {
            // BNSD: M轴按GS1紧凑空间(actS1Size)切块, seqused_q < s1时物理布局按s1Size跨head存在padding间隙,
            // 子块跨g边界时连续拷贝会串到下一head的数据, 需按g边界分段拷贝(与ProcessGS1/DataCopySoftmaxLseBNSD一致)
            uint32_t remainRows = runInfo.actVecMSize;
            uint32_t curS1Idx = s1Idx;
            uint32_t curN1Idx = n1Idx;
            uint32_t dstRow = 0;
            while (remainRows > 0) {
                uint32_t segRows = runInfo.actS1Size - curS1Idx;
                if (segRows > remainRows) {
                    segRows = remainRows;
                }
                int64_t relHOffset = (runInfo.bIdx * n1Size + curN1Idx) * constInfo_.s1Size * constInfo_.relHSize +
                                     curS1Idx * constInfo_.relHSize;
                int64_t relWOffset = (runInfo.bIdx * n1Size + curN1Idx) * constInfo_.s1Size * constInfo_.relWSize +
                                     curS1Idx * constInfo_.relWSize;
                RelDataCopy(relHUb[dstRow * constInfo_.relHSize], relHGm, relHOffset, segRows * constInfo_.relHSize);
                RelDataCopy(relWUb[dstRow * constInfo_.relWSize], relWGm, relWOffset, segRows * constInfo_.relWSize);
                dstRow += segRows;
                remainRows -= segRows;
                curS1Idx = 0;
                curN1Idx += 1;
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
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline FANoQuantGqaBlockVecDummyDn(ConstInfoX &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                  SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace BaseApi
#endif // FLASH_ATTN_BLOCK_VEC_DN_H_
