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
 * \file flash_attn_block_vec_noquant_gqa_nd.h
 * \brief FANoQuantGqaBlockVecNd —— Nd 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_ND_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_ND_H_

#include "kernel_operator.h"

#include "../utils/attenmask_gs1.h"

#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "init_output.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <typename FA_T>
class FANoQuantGqaBlockVecNd {
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
    static constexpr uint32_t CC_L1P_0  = 5U;
    static constexpr uint32_t CC_L1P_1  = 6U;
    static constexpr uint32_t CC_L1P_2  = 7U;

    // 核内同步ID
    // MTE3<->V, 输出buffer
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = EVENT_ID0;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = EVENT_ID2;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = EVENT_ID3;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = EVENT_ID4;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = EVENT_ID5;
    // MTE2<->V, 输入buffer
    static constexpr uint32_t UB_IN_MASK_EVENT0 = EVENT_ID0;
    static constexpr uint32_t UB_IN_MASK_EVENT1 = EVENT_ID1;

    // L1
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> l1PBuffers;

    // UB
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
    LocalTensor<uint8_t> ubMm2ResBuffers;

    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    LocalTensor<uint8_t> ubMm1ResBuffers;

    static constexpr uint32_t UB_MASK_BUFCNT = 2U;
    static constexpr uint32_t UB_MASK_BUF_BYTES = 8192U;
    LocalTensor<uint8_t> ubMaskBuffers;

    LocalTensor<T> ubVec2Res; // 存放vec2阶段VEC的中间处理结果, 并且作为attn_out的输出buffer, 需配对的MTE3和V的同步ID

    static constexpr uint32_t UB_VEC1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = 33024U;
    LocalTensor<uint8_t> ubVec1ResBuffers;
    uint32_t vec1ResUbBufId = 0;

    static constexpr uint32_t UB_SOFTMAX_MAX_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_MAX_BUF_BYTES = 256U;
    LocalTensor<T> softmaxSumBuf;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUF_BYTES = 256U;
    LocalTensor<T> softmaxMaxBuf;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUF_BYTES = 256U;
    LocalTensor<T> softmaxExpBuf;

    static constexpr uint32_t UB_LSE_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES = 2048U;
    LocalTensor<uint8_t> ubLseOutBuffers;
    uint32_t lseOutUbBufId = 0;

    LocalTensor<uint8_t> vec1ApiTmpBuf;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;
    const ConstInfoX &constInfo;

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool;

    // GM
    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensor;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<float> softmaxFDSumGm;
    GlobalTensor<float> softmaxFDMaxGm;

    static constexpr MaskFormat MASK_LAYOUT = (LAYOUT_T == FA_LAYOUT::BSH || LAYOUT_T == FA_LAYOUT::TND) ? MaskFormat::SG : MaskFormat::GS;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    T negativeFloatScalar;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecNd(ConstInfoX &constInfo,
        SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool, SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool) :
        constInfo(constInfo), qSeqLensTool(qSeqLensTool), kvSeqLensTool(kvSeqLensTool) {};

    __aicore__ inline void InitBlock(__gm__ uint8_t *attenMask,
                                     __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar = *((T *)&tmp1);

        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        InitAttenOutBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSizeV, outGmTensor, attentionOut);

        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if constexpr (HAS_MASK) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if (constInfo.enableFlashDecode) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                           constInfo.logSumExpSize);
        }
    }

     __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // l1P 三缓冲
        uint32_t addrL1 = 0;
        l1PBuffers = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMm2ResBuffers = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES); // 2 * 32K = 64K, CV通信BUF
        addrUb = UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
        ubMm1ResBuffers = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES); // 2 * 32K = 64K, CV通信BUF
        addrUb += UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES;
        ubMaskBuffers = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MASK_BUFCNT * UB_MASK_BUF_BYTES); // 2 * 8K = 16K, 输入BUF: MASK拷入
        addrUb += UB_MASK_BUFCNT * UB_MASK_BUF_BYTES;
        ubVec2Res = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, 32768U).template ReinterpretCast<T>(); // 32K, 输出BUF: attn_out拷出
        addrUb += 32768U;
        ubVec1ResBuffers = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES); // 2 * 32.25K = 64.5K, 输出BUF: softmax结果拷贝至L1
        addrUb += UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES;

        // softmaxSum×3 + softmaxMax×3 + softmaxExp×3，各 256 bytes
        softmaxSumBuf = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES).template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES;
        softmaxMaxBuf = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES).template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES;
        softmaxExpBuf = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES).template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES;

        ubLseOutBuffers = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES); // 2 * 2K = 4K, 输出BUF: FD中间结果SUM和MAX拷出至GM，或者LSE结果拷出
        addrUb += UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES;

        vec1ApiTmpBuf = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, 512U); // 0.5K, 常驻BUF, 用于softmax计算的中间结果缓存
        addrUb += 512U;
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE3_V>(UB_OUT_VEC2_RES_EVENT0);
        SetFlag<HardEvent::MTE3_V>(UB_OUT_VEC1_RES_EVENT0);
        SetFlag<HardEvent::MTE3_V>(UB_OUT_VEC1_RES_EVENT1);
        SetFlag<HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT0);
        SetFlag<HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT1);
        if constexpr (HAS_MASK) {
            SetFlag<HardEvent::V_MTE2>(UB_IN_MASK_EVENT0);
            SetFlag<HardEvent::V_MTE2>(UB_IN_MASK_EVENT1);
        }

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_1);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(UB_OUT_VEC2_RES_EVENT0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(UB_OUT_VEC1_RES_EVENT0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(UB_OUT_VEC1_RES_EVENT1);
        WaitFlag<AscendC::HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT1);
        if constexpr (HAS_MASK) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(UB_IN_MASK_EVENT0);
            WaitFlag<AscendC::HardEvent::V_MTE2>(UB_IN_MASK_EVENT1);
        }
    }

    __aicore__ inline void ProcessVec1(RunInfoX runInfo)
    {
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId   = runInfo.loop % L1_P_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CC_BMM1_0 + mm1ResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx  = CC_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMm1ResBuffers[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);
        ProcessVec1Nd(pL1Tensor, mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx); // C1与V1的反向同步, C1收到后可以启动FIXPIPE向UB的写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx);
        Vec1PostProcess(runInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        if (constInfo.needInitOutput) {
            uint32_t vecCoreNum = 2 * constInfo.coreNum;
            uint64_t tSize = constInfo.bSize * constInfo.s1Size;
            if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                tSize = qSeqLensTool.cuSeqLensParser.GetTSize();
            }
            uint64_t attenOutTotalSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV;

            static constexpr OUTPUT_T ATTEN_OUT_INIT_VAL = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(OUTPUT_T);
            AttentionCommon::InitOutput<OUTPUT_T, EVENT_ID0, ATTEN_OUT_POP_BUF_START_ADDR, ATTEN_OUT_POP_BUF_ELE_SIZE>(
                attentionOutGm, attenOutTotalSize, vecCoreNum, ATTEN_OUT_INIT_VAL);

            if (constInfo.isSoftmaxLseEnable) {
                uint64_t lseTotalSize = tSize * constInfo.n2Size * constInfo.gSize;

                static constexpr float LSE_INIT_VAL = 3e+99;
                static constexpr uint32_t LSE_POP_BUF_START_ADDR = BUFFER_SIZE_BYTE_32K;
                static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(float);
                AttentionCommon::InitOutput<float, EVENT_ID1, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE>(
                    softmaxLseGm, lseTotalSize, vecCoreNum, LSE_INIT_VAL);
            }

            SyncAll();
        }
    }

    __aicore__ inline void InitAttenOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                              uint32_t headDim, FaGmTensorOut &outGmTensor, __gm__ uint8_t *gm)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUTPUT_T *)gm);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool.seqUsedParser);
        } else {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool.cuSeqLensParser);
        }
        }
        }
    }

    // =================================Private Functions=================================
    }
    }

    // =================================Private Functions=================================

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (constInfo.enableFlashDecode) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if (constInfo.enableFlashDecode) {
            if (!runInfo.isS2SplitCore && constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo.isSoftmaxLseEnable) {
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

        WaitFlag<HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = ubLseOutBuffers[lseOutUbBufId * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        WaitFlag<HardEvent::V_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
            uint32_t prefixBS1 = qSeqLensTool.cuSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo.gSize * constInfo.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                           constInfo);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qSeqLensTool.seqUsedParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qSeqLensTool.seqUsedParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo, qActSeqLens);
        }
        SetFlag<HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        lseOutUbBufId = (lseOutUbBufId + 1) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ProcessVec1Nd(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor, RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        static constexpr bool hasDrop = false;

        LocalTensor<uint8_t> vselrIndexesBuf[4];
        LocalTensor<uint8_t> dropMaskUb;
        LocalTensor<INPUT_T> nonePseUb; // PSE不支持，占位
        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> attenMaskUbPre;
        if constexpr (HAS_MASK == true) {
            const uint32_t maskBufId = runInfo.loop & (DB - 1);
            attenMaskUb = ubMaskBuffers[maskBufId * UB_MASK_BUF_BYTES];
            WaitFlag<HardEvent::V_MTE2>(UB_IN_MASK_EVENT0 + maskBufId);
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo);
        }

        LocalTensor<T> sumUb = softmaxSumBuf[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb = softmaxMaxBuf[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb = softmaxExpBuf[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;

        WaitFlag<HardEvent::MTE3_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId);

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<INPUT_T> stage1CastTensor = ubVec1ResBuffers[vec1ResUbBufId * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        if (runInfo.isFirstS2Loop) {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, vec1ApiTmpBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, vec1ApiTmpBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, vec1ApiTmpBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize,
                                  GT_128_AND_LTE_256, HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, vec1ApiTmpBuf, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                        descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        } else {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, vec1ApiTmpBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, vec1ApiTmpBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, vec1ApiTmpBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_128_AND_LTE_256,
                                  HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, vec1ApiTmpBuf, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                        descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        }
        if constexpr (HAS_MASK) {
            const uint32_t maskBufId = runInfo.loop & (DB - 1);
            SetFlag<HardEvent::V_MTE2>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        SetFlag<HardEvent::V_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId);
        WaitFlag<HardEvent::V_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId);
        LocalTensor<INPUT_T> mm2AL1Tensor= pL1Tensor;

        if (likely(runInfo.actVecMSize != 0)) {
            static constexpr uint32_t VEC1_SRC_STRIDE = (mBaseSize >> 1) + 1;
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) *
                                  (runInfo.actMSize - runInfo.actVecMSize)],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(VEC1_SRC_STRIDE - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }

        SetFlag<HardEvent::MTE3_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId);
        vec1ResUbBufId = (vec1ResUbBufId + 1U) % UB_VEC1_RES_BUFCNT;
    }

    __aicore__ inline void Vec1PostProcess(RunInfoX runInfo)
    {
        LocalTensor<T> sumUb = softmaxSumBuf[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb = softmaxMaxBuf[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb = softmaxExpBuf[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, vec1ApiTmpBuf, runInfo.actVecMSize);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo.gSize;
            s1EndTdx = vecMEndIdx / constInfo.gSize;
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
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &ubVec2Res, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken = AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T)) + mStartVec];
                RowInvalidUpdateVF<float>(ubVec2Res, maxTensor, mDealSize, constInfo.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
                                      .rowCount = dealRowCount,
                                      .colCount = dTemplateAlign64};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.n2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        LocalTensor<float> sumBrdcstBuf = ubLseOutBuffers[lseOutUbBufId * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        WaitFlag<HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        FaVectorApi::BroadcastMaxSum(sumBrdcstBuf, sumUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        WaitFlag<HardEvent::V_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        DataCopy(softmaxFDSumGm[gmOffset], sumBrdcstBuf, calculateSize);
        SetFlag<HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        lseOutUbBufId = (lseOutUbBufId + 1U) % UB_LSE_OUT_BUFCNT;

        LocalTensor<float> maxBrdcstBuf = ubLseOutBuffers[lseOutUbBufId * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        WaitFlag<HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        FaVectorApi::BroadcastMaxSum(maxBrdcstBuf, maxUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        WaitFlag<HardEvent::V_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        DataCopy(softmaxFDMaxGm[gmOffset], maxBrdcstBuf, calculateSize);
        SetFlag<HardEvent::MTE3_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId);
        lseOutUbBufId = (lseOutUbBufId + 1U) % UB_LSE_OUT_BUFCNT;
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
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfo.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm[gmOffset], ubVec2Res, dataCopyParams);
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

        WaitFlag<HardEvent::MTE3_V>(UB_OUT_VEC2_RES_EVENT0);
        {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c2v2CrossCoreSyncIdx);
            LocalTensor<T> mm2ResUbTensor = ubMm2ResBuffers[mm2ResUbBufId * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<T>();
            if (unlikely(runInfo.isFirstS2Loop)) {
                uint32_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
                DataCopy(ubVec2Res, mm2ResUbTensor, vec2CalcSize);
            } else {
                LocalTensor<T> expUb = softmaxExpBuf[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
                LocalTensor<T> pScaleUb;

                float deSCalePreVValue = 1.0f;
                if (!runInfo.isLastS2Loop) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        ubVec2Res, mm2ResUbTensor, ubVec2Res, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
                } else {
                    LocalTensor<float> sumUb = softmaxSumBuf[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        ubVec2Res, mm2ResUbTensor, ubVec2Res, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64, 1.0,
                        1.0);
                }
            }
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c2v2CrossCoreSyncIdx); // mmRes在之后不能使用, 否则与C2的FIXPIPE读写数据冲突
        }

        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = softmaxSumBuf[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    ubVec2Res, ubVec2Res, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, 0.0F);
            }
            uint32_t mStartVec = 0;
            uint32_t mDealSize = runInfo.actVecMSize;
            if (constInfo.enableFlashDecode && runInfo.isS2SplitCore) {
                SetFlag<HardEvent::V_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                WaitFlag<HardEvent::V_MTE3>(UB_OUT_VEC2_RES_EVENT0);

                Bmm2ResForFDCopyOut(runInfo, ubVec2Res, mStartVec, mDealSize);
            } else {
                LocalTensor<OUTPUT_T> attenOut;
                int64_t dSizeAligned64 = (int64_t)dVBaseSize;

                attenOut.SetAddr(ubVec2Res.address_);

                RowInvalid(ubVec2Res, mStartVec, mDealSize, runInfo, dSizeAligned64);
                Cast(attenOut, ubVec2Res, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);

                SetFlag<HardEvent::V_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                WaitFlag<HardEvent::V_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
            }
        }
        SetFlag<HardEvent::MTE3_V>(UB_OUT_VEC2_RES_EVENT0);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX &runInfo)
    {
        const uint32_t bufIdx = runInfo.loop & (DB - 1);
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = 0;
        maskInfo.s2LeftPaddingSize = 0;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);

        if (IsSkipMask && IsSkipMaskForPre) {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            return;
        }

        if (!IsSkipMask) {
            const uint32_t mte2ToVId = UB_IN_MASK_EVENT0 + bufIdx;
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(
                attenMaskUb, attenMaskGmInt, maskInfo, false, mte2ToVId);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
        }

        if (!IsSkipMaskForPre) {
            const uint32_t preBufId = bufIdx ^ 1U;
            const uint32_t preMte2ToVId = UB_IN_MASK_EVENT0 + preBufId;
            WaitFlag<HardEvent::V_MTE2>(preMte2ToVId);
            LocalTensor<uint8_t> attenMaskUbPre = ubMaskBuffers[preBufId * UB_MASK_BUF_BYTES];
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(
                attenMaskUbPre, attenMaskGmInt, maskInfo, true, preMte2ToVId);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
            SetFlag<HardEvent::V_MTE2>(preMte2ToVId);
        }
    }
};

// AIC/AIV 分编译占位（Mix kernel 在 AIC 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockVecDummyNd {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline FANoQuantGqaBlockVecDummyNd(ConstInfoX &constInfo,
        SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool, SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool) {};
};

} // namespace BaseApi
#endif // FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_ND_H_