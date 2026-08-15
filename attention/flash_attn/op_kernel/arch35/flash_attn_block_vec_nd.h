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
 * \file flash_attn_block_vec_nd.h
 * \brief FANoQuantGqaBlockVecNd —— Nd 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTN_BLOCK_VEC_ND_H_
#define FLASH_ATTN_BLOCK_VEC_ND_H_

#include <limits>

#include "../utils/attenmask_gs1.h"

#if __has_include("../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h")
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/init_output.h"
#else
#include "../../common/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../common/arch35/vf/vf_flashupdate_new.h"
#include "../../common/arch35/vf/vf_div_cast_arch35.h"
#include "../../common/arch35/vf/vf_flash_decode_arch35.h"
#include "../../common/vector_common.h"
#include "../../common/init_output.h"
#endif

#include "memory_copy_arch35.h"

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
    // MTE2<->V, 输入buffer
    static constexpr uint32_t UB_IN_MASK_EVENT0 = 6;
    static constexpr uint32_t UB_IN_MASK_EVENT1 = 7;

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

    static constexpr uint32_t UB_MASK_BUFCNT = 2U;
    static constexpr uint32_t UB_MASK_BUF_BYTES = 8192U;
    LocalTensor<uint8_t> ubMaskBuffers_;

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

    LocalTensor<uint8_t> vec1ApiTmpBuf_;

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
    GlobalTensor<uint8_t> attenMaskGmInt_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxFDSumGm_;
    GlobalTensor<float> softmaxFDMaxGm_;
    GlobalTensor<INPUT_T> sinkGm_;

    static constexpr MaskFormat MASK_LAYOUT =
        (LAYOUT_T == FA_LAYOUT::BSH || LAYOUT_T == FA_LAYOUT::TND) ? MaskFormat::SG : MaskFormat::GS;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar_ = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    T negativeFloatScalar_;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecNd(ConstInfoX &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
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

        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if constexpr (HAS_MASK) {
            attenMaskGmInt_.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if (constInfo_.enableFlashDecode) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
        if (constInfo_.learnableSinkFlag) {
            sinkGm_.SetGlobalBuffer((__gm__ INPUT_T *)learnableSink);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // l1P 三缓冲
        uint32_t addrL1 = 0;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        struct UbLayout {
            uint8_t mm2ResBuffers[UB_MM2_RES_BUFCNT][UB_MM2_RES_BUF_BYTES]; // 2 * 32K = 64K, CV通信BUF
            uint8_t mm1ResBuffers[UB_MM1_RES_BUFCNT][UB_MM1_RES_BUF_BYTES]; // 2 * 32K = 64K, CV通信BUF
            uint8_t maskBuffers[UB_MASK_BUFCNT][UB_MASK_BUF_BYTES];         // 2 * 8K = 16K, 输入BUF: MASK拷入
            uint8_t vec2Res[32768U];                                        // 32K, 输出BUF: attn_out拷出
            uint8_t vec1ResBuffers[UB_VEC1_RES_BUFCNT]
                                  [UB_VEC1_RES_BUF_BYTES];                           // 2 * 32.25K, softmax结果拷至L1
            uint8_t softmaxSumBuf_[UB_SOFTMAX_SUM_BUFCNT][UB_SOFTMAX_SUM_BUF_BYTES]; // 3 * 0.25K = 0.75K, sum常驻BUF
            uint8_t softmaxMaxBuf_[UB_SOFTMAX_MAX_BUFCNT][UB_SOFTMAX_MAX_BUF_BYTES]; // 3 * 0.25K = 0.75K, max常驻BUF
            uint8_t softmaxExpBuf_[UB_SOFTMAX_EXP_BUFCNT][UB_SOFTMAX_EXP_BUF_BYTES]; // 3 * 0.25K = 0.75K, exp常驻BUF
            uint8_t lseOutBuffers[UB_LSE_OUT_BUFCNT]
                                 [UB_LSE_OUT_BUF_BYTES]; // 2 * 2K = 4K, 输出BUF:
                                                         // FD中间结果SUM和MAX拷出至GM，或者LSE结果拷出
            uint8_t softmaxTmpBuf[512U];                 // 0.5K, 常驻BUF, 用于softmax计算的中间结果缓存
        };
        static_assert(sizeof(UbLayout) <= 248 * 1024, "UB buffer too large");
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, mm2ResBuffers),
                                                SIZE_OF_MEMBER(UbLayout, mm2ResBuffers));
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, mm1ResBuffers),
                                                SIZE_OF_MEMBER(UbLayout, mm1ResBuffers));
        ubMaskBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, maskBuffers),
                                              SIZE_OF_MEMBER(UbLayout, maskBuffers));
        ubVec2Res_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, vec2Res),
                                          SIZE_OF_MEMBER(UbLayout, vec2Res))
                         .template ReinterpretCast<T>();
        ubVec1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, vec1ResBuffers),
                                                 SIZE_OF_MEMBER(UbLayout, vec1ResBuffers));
        // softmaxSum×3 + softmaxMax×3 + softmaxExp×3，各 256 bytes
        softmaxSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxSumBuf_),
                                              SIZE_OF_MEMBER(UbLayout, softmaxSumBuf_))
                             .template ReinterpretCast<T>();
        softmaxMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxMaxBuf_),
                                              SIZE_OF_MEMBER(UbLayout, softmaxMaxBuf_))
                             .template ReinterpretCast<T>();
        softmaxExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxExpBuf_),
                                              SIZE_OF_MEMBER(UbLayout, softmaxExpBuf_))
                             .template ReinterpretCast<T>();
        ubLseOutBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, lseOutBuffers),
                                                SIZE_OF_MEMBER(UbLayout, lseOutBuffers));
        vec1ApiTmpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxTmpBuf),
                                              SIZE_OF_MEMBER(UbLayout, softmaxTmpBuf));
    }

    static constexpr uint32_t BRCB_NUM = 8U;

    __aicore__ inline void SinkCopyInGS1(LocalTensor<INPUT_T> &sinkTmpUb, GlobalTensor<INPUT_T> &sinkGm,
                                         uint32_t gs1Start, uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
    {
        uint32_t gStart = gs1Start / actS1Size;
        uint32_t gs1End = gs1Start + actVecMSize - 1;
        uint32_t gEnd = gs1End / actS1Size;

        uint32_t gCount = gEnd - gStart + 1;
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
    }

    __aicore__ inline void SinkExpandGS1(LocalTensor<T> &sinkExpandUb, LocalTensor<T> &sinkBrcbUb,
                                         uint32_t gs1Start, uint32_t actVecMSize, uint32_t actS1Size, uint32_t gSize)
    {
        uint32_t s1Start = gs1Start % actS1Size;
        uint32_t gStart = gs1Start / actS1Size;

        uint32_t gs1End = gs1Start + actVecMSize - 1;
        uint32_t s1End = gs1End % actS1Size;
        uint32_t gEnd = gs1End / actS1Size;

        // 头块: 指头部不完整的s1块, 并且起始位置不是0
        // 中间块: 指完整的s1块
        // 尾块: 指尾部不完整的s1块, 并且其实位置是0
        uint32_t headS1 = 0;
        if (s1Start != 0) {
            headS1 = (gStart == gEnd) ? (s1End - s1Start + 1) : (actS1Size - s1Start);
        } else {
            headS1 = 0; // 起始位置为0, 不能算入头块
        }

        uint32_t midCnt = (actVecMSize - headS1) / actS1Size;
        uint32_t tailS1 = (actVecMSize - headS1) % actS1Size;

        uint32_t ubOffset = 0;
        uint32_t gIdx = 0;
        if (headS1 > 0) {
            for (uint32_t i = 0; i < headS1; i++) {
                DataCopy(sinkExpandUb[ubOffset], sinkBrcbUb[gIdx * BRCB_NUM], BRCB_NUM);
                ubOffset += BRCB_NUM;
            }
            gIdx++;
        }
        for (uint32_t j = 0; j < midCnt; j++) {
            for (uint32_t i = 0; i < actS1Size; i++) {
                DataCopy(sinkExpandUb[ubOffset], sinkBrcbUb[gIdx * BRCB_NUM], BRCB_NUM);
                ubOffset += BRCB_NUM;
            }
            gIdx++;
        }
        if (tailS1 > 0) {
            for (uint32_t i = 0; i < tailS1; i++) {
                DataCopy(sinkExpandUb[ubOffset], sinkBrcbUb[gIdx * BRCB_NUM], BRCB_NUM);
                ubOffset += BRCB_NUM;
            }
            gIdx++;
        }
    }

    __aicore__ inline void SinkCopyInS1G(LocalTensor<INPUT_T> &sinkTmpUb, GlobalTensor<INPUT_T> &sinkGm,
                                         uint32_t gs1Start, uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
    {
        uint32_t gStart = gs1Start % gSize;
        uint32_t s1Start = gs1Start / gSize;

        uint32_t gs1End = gs1Start + actVecMSize - 1;
        uint32_t gEnd = gs1End % gSize;
        uint32_t s1End = gs1End / gSize;

        if (s1Start == s1End) {
            // 情况1: 只有一个头块, 此时[gStart, gEnd]之间的行就是实际处理的M轴
            uint32_t gCount = gEnd - gStart + 1;
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
        } else if (s1Start + 1 == s1End) {
            if (gStart > gEnd) {
                // 情况2.1: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
                // 并且gStart > gEnd, 此时它们不重叠.
                // 由于头尾块的长度都可能不对齐, 并且gSize有可能很大, 而UB要求32B对齐,
                // 约定先拷贝尾块的GM[0, gEnd]->UB[0, Align(gEnd + 1, 16) - 1],
                // 再拷贝头块的GM[gStart, gSize)->UB[Align(gEnd + 1, 16), Align(gEnd + 1, 16) + Align(gSize - gStart, 16))
                DataCopyExtParams sinkCopyParams;
                sinkCopyParams.blockCount = 1;
                sinkCopyParams.blockLen = (gEnd + 1) * sizeof(INPUT_T);
                sinkCopyParams.srcStride = 0;
                sinkCopyParams.dstStride = 0;
                DataCopyPadExtParams<INPUT_T> sinkPadParams;
                sinkPadParams.isPad = true;
                sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
                DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);

                uint32_t ubOffset = AttentionCommon::Align(gEnd + 1, 16U);
                sinkCopyParams.blockLen = (gSize - gStart) * sizeof(INPUT_T);
                DataCopyPad(sinkTmpUb[ubOffset], sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);

            } else {
                // 情况2.2: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
                // 并且gStart <= gEnd, 此时它们重叠, 合并后的覆盖范围为[0, gSize)
                DataCopyExtParams sinkCopyParams;
                sinkCopyParams.blockCount = 1;
                sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
                sinkCopyParams.srcStride = 0;
                sinkCopyParams.dstStride = 0;
                DataCopyPadExtParams<INPUT_T> sinkPadParams;
                sinkPadParams.isPad = true;
                sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
                DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
            }
        } else {
            // 情况3: 头块 + 中间块 + 尾块, 合并后的覆盖范围为[0, gSize)
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
        }
    }

    __aicore__ inline void SinkExpandS1G(LocalTensor<T> &sinkExpandUb, LocalTensor<T> &sinkBrcbUb,
                                         uint32_t gs1Start, uint32_t actVecMSize, uint32_t actS1Size, uint32_t gSize)
    {
        uint32_t gStart = gs1Start % gSize;
        uint32_t s1Start = gs1Start / gSize;

        uint32_t gs1End = gs1Start + actVecMSize - 1;
        uint32_t gEnd = gs1End % gSize;
        uint32_t s1End = gs1End / gSize;

        if (s1Start == s1End) {
            // 情况1: 只有一个头块, 此时[gStart, gEnd]之间的行就是实际处理的M轴
            DataCopy(sinkExpandUb, sinkBrcbUb, (gEnd - gStart + 1) * BRCB_NUM);
        } else if (s1Start + 1 == s1End) {
            if (gStart > gEnd) {
                // 情况2.1: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
                // 并且gStart > gEnd, 此时它们不重叠.
                // 由于头尾块的长度都可能不对齐, 并且gSize有可能很大, 而UB要求32B对齐,
                // 约定先拷贝尾块的GM[0, gEnd]->UB[0, Align(gEnd + 1, 16) - 1],
                // 再拷贝头块的GM[gStart, gSize)->UB[Align(gEnd + 1, 16), Align(gEnd + 1, 16) + Align(gSize - gStart, 16))
                uint32_t srcUbOffset = AttentionCommon::Align(gEnd + 1, 16U) * BRCB_NUM;
                uint32_t copySize = (gSize - gStart) * BRCB_NUM;
                DataCopy(sinkExpandUb, sinkBrcbUb[srcUbOffset], copySize);
                DataCopy(sinkExpandUb[copySize], sinkBrcbUb[0], (gEnd + 1) * BRCB_NUM);
            } else {
                // 情况2.2: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
                // 并且gStart <= gEnd, 此时它们重叠, 合并后的覆盖范围为[0, gSize)
                uint32_t copySize = (gSize - gStart) * BRCB_NUM;
                DataCopy(sinkExpandUb, sinkBrcbUb[gStart * BRCB_NUM], copySize);
                DataCopy(sinkExpandUb[copySize], sinkBrcbUb[0], (gEnd + 1) * BRCB_NUM);
            }
        } else {
            // 情况3: 头块 + 中间块 + 尾块, 合并后的覆盖范围为[0, gSize)

            // 头块: 指头部不完整的G块, 并且起始位置不是0
            // 中间块: 指完整的G块
            // 尾块: 指尾部不完整的G块, 并且其实位置是0
            uint32_t headG = 0;
            if (gStart != 0) {
                headG = gSize - gStart;
            } else {
                headG = 0; // 起始位置为0, 不能算入头块
            }

            uint32_t midCnt = (actVecMSize - headG) / gSize;
            uint32_t tailG = (actVecMSize - headG) % gSize;

            uint32_t ubOffset = 0;
            if (headG > 0) {
                DataCopy(sinkExpandUb, sinkBrcbUb[gStart * BRCB_NUM], headG * BRCB_NUM);
                ubOffset += headG * BRCB_NUM;
            }
            for (uint32_t i = 0; i < midCnt; i++) {
                DataCopy(sinkExpandUb[ubOffset], sinkBrcbUb[0], gSize * BRCB_NUM);
                ubOffset += gSize * BRCB_NUM;
            }
            if (tailG > 0) {
                DataCopy(sinkExpandUb[ubOffset], sinkBrcbUb[0], tailG * BRCB_NUM);
            }
        }
    }

    __aicore__ inline void ResetSoftmaxBuffer(uint32_t slotIdx, const RunInfoX &runInfo)
    {
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T);
        LocalTensor<T> sumUb = softmaxSumBuf_[slotIdx * softmaxBufElementCount];
        LocalTensor<T> maxUb = softmaxMaxBuf_[slotIdx * softmaxBufElementCount];
        if (constInfo_.learnableSinkFlag && !runInfo.headS2Split) {
            Duplicate<T>(sumUb, static_cast<T>(1), softmaxBufElementCount);

            uint32_t gs1Start = runInfo.gS1Idx + runInfo.vecMbaseIdx;

            Mutex::Lock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<INPUT_T> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
                if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
                    SinkCopyInS1G(sinkTmpUb, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                  runInfo.n2Idx, constInfo_.gSize);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    SinkCopyInGS1(sinkTmpUb, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                  runInfo.n2Idx, constInfo_.gSize);
                }
            }
            Mutex::Unlock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

            Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                // S1G场景,仅头尾两块且未覆盖完整G时,因为可能不32B对齐,每一段需预留16个数,故预留32元素, 预计brcb后要预留32行
                static constexpr uint32_t MAX_VEC_M_SIZE = ((mBaseSize / CV_RATIO + 15U) & ~15U) + 32U;
                LocalTensor<INPUT_T> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
                LocalTensor<T> sinkCastUb = sinkTmpUb[MAX_VEC_M_SIZE].template ReinterpretCast<T>();
                LocalTensor<T> sinkBrcbUb = sinkCastUb[MAX_VEC_M_SIZE];
                LocalTensor<T> sinkExpandUb = sinkBrcbUb[MAX_VEC_M_SIZE * BRCB_NUM];
                LocalTensor<T> sinkResUb = sinkExpandUb[MAX_VEC_M_SIZE * BRCB_NUM];

                Cast(sinkCastUb, sinkTmpUb, AscendC::RoundMode::CAST_NONE, MAX_VEC_M_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                Brcb(sinkBrcbUb, sinkCastUb, (MAX_VEC_M_SIZE + BRCB_NUM - 1) / BRCB_NUM, {1, BRCB_NUM});
                AscendC::PipeBarrier<PIPE_V>();

                if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
                    SinkExpandS1G(sinkExpandUb, sinkBrcbUb, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                  constInfo_.gSize);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    SinkExpandGS1(sinkExpandUb, sinkBrcbUb, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                  constInfo_.gSize);
                }
                AscendC::PipeBarrier<PIPE_V>();

                constexpr uint32_t REDUCE_MAX_REPEAT = 128;
                uint32_t remain = runInfo.actVecMSize;
                uint32_t resOffset = 0;
                uint32_t expandOffset = 0;
                while (remain > 0) {
                    uint32_t chunk = (remain > REDUCE_MAX_REPEAT) ? REDUCE_MAX_REPEAT : remain;
                    WholeReduceMax(sinkResUb[resOffset], sinkExpandUb[expandOffset * BRCB_NUM], BRCB_NUM, chunk,
                                   1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
                    resOffset += chunk;
                    expandOffset += chunk;
                    remain -= chunk;
                }
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(maxUb, sinkResUb, AttentionCommon::Align(runInfo.actVecMSize, 8U));
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        } else {
            Duplicate<T>(sumUb, static_cast<T>(0), softmaxBufElementCount);
            Duplicate<T>(maxUb, static_cast<T>(-std::numeric_limits<float>::infinity()), softmaxBufElementCount);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_1);
    }

    __aicore__ inline void UnInitCrossCoreSync()
    {
    }

    __aicore__ inline void AllocEventID()
    {
    }

    __aicore__ inline void FreeEventID()
    {
    }

    __aicore__ inline void ProcessVec1(RunInfoX runInfo)
    {
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CC_BMM1_0 + mm1ResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        if (unlikely(runInfo.isFirstS2Loop)) {
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT, runInfo);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);
        ProcessVec1Nd(pL1Tensor, mm1ResUbTensor, runInfo);
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

    __aicore__ inline void ProcessVec1Nd(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor,
                                         RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        LocalTensor<uint8_t> dropMaskUb;
        LocalTensor<INPUT_T> nonePseUb; // PSE不支持，占位
        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> attenMaskUbPre;
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        const uint32_t maskBufId = runInfo.loop & (DB - 1);
        if constexpr (HAS_MASK) {
            attenMaskUb = ubMaskBuffers_[maskBufId * UB_MASK_BUF_BYTES];
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo);
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

        LocalTensor<INPUT_T> stage1CastTensor =
            ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        if (likely(runInfo.actSingleLoopS2Size == 128)) {
            FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, EQ_128,
                                       HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                stage1CastTensor, nullptr, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                nonePseUb, dropMaskUb, vec1ApiTmpBuf_, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                0, 0.0f, 0.0f, static_cast<T>(constInfo_.scaleValue),
                descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb, deSCaleKValue);
        } else if (runInfo.actSingleLoopS2Size <= 64) {
            FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize,
                                       GT_0_AND_LTE_64, HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false,
                                       false>(
                stage1CastTensor, nullptr, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                nonePseUb, dropMaskUb, vec1ApiTmpBuf_, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                0 /* pseStride */, 0.0f /* slopes */, 0.0f /* posShift */, static_cast<T>(constInfo_.scaleValue),
                descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb, deSCaleKValue);
        } else if (runInfo.actSingleLoopS2Size < 128) {
            FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize,
                                       GT_64_AND_LTE_128, HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, false,
                                       false>(
                stage1CastTensor, nullptr, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                nonePseUb, dropMaskUb, vec1ApiTmpBuf_, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                0 /* pseStride */, 0.0f /* slopes */, 0.0f /* posShift */, static_cast<T>(constInfo_.scaleValue),
                descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb, deSCaleKValue);
        } else {
            if constexpr (s2BaseSize == 256) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize,
                                           GT_128_AND_LTE_256, HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mm1ResUbTensor, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, vec1ApiTmpBuf_, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /* pseStride */, 0.0f /* slopes */, 0.0f /* posShift */,
                    static_cast<T>(constInfo_.scaleValue),
                    descaleQK, negativeFloatScalar_, 0.0F);
            }
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

        if constexpr (HAS_MASK) {
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Tensor;
        if (likely(runInfo.actVecMSize != 0)) {
            static constexpr uint32_t VEC1_SRC_STRIDE = (mBaseSize >> 1) + 1;
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * (blockBytes / sizeof(INPUT_T)) *
                                  (runInfo.actMSize - runInfo.actVecMSize)],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(VEC1_SRC_STRIDE - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
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
        LocalTensor<T> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, vec1ApiTmpBuf_, runInfo.actVecMSize);

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
            s1StartTdx = vecMStartIdx / constInfo_.gSize;
            s1EndTdx = vecMEndIdx / constInfo_.gSize;
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
                // 跨多个G: 后续G的s1均从0开始, 存在左无效行(s1FirstValidToken>0)必命中;
                // 中间G的s1均到actS1Size-1结束, 存在右无效行(s1LastValidToken<actS1Size-1)必命中
                ret = (s1FirstValidToken > 0) ||
                      (s1LastValidToken < static_cast<int64_t>(runInfo.actS1Size) - 1);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &ubVec2Res, int64_t mStartVec, int64_t mDealSize,
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
                    softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T)) +
                                   mStartVec];
                RowInvalidUpdateVF<float>(ubVec2Res, maxTensor, mDealSize, constInfo_.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoord gmCoord{.bIdx = info.bIdx,
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
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
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
                    LocalTensor<float> sumUb =
                        softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) *
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
            bool isNeedFd = (constInfo_.enableFlashDecode && runInfo.isS2SplitCore);
            if (isNeedFd) {
                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2ResForFDCopyOut(runInfo, ubVec2Res_, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            } else {
                LocalTensor<OUTPUT_T> attenOut;
                int64_t dSizeAligned64 = (int64_t)dVBaseSize;

                attenOut.SetAddr(ubVec2Res_.address_);

                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                RowInvalid(ubVec2Res_, mStartVec, mDealSize, runInfo, dSizeAligned64);
                Cast(attenOut, ubVec2Res_, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);

                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            }
        }
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX &runInfo)
    {
        const uint32_t bufIdx = runInfo.loop & (DB - 1);
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo_.preTokens;
        maskInfo.nextToken = constInfo_.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = (constInfo_.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar_;
        maskInfo.s1LeftPaddingSize = 0;
        maskInfo.s2LeftPaddingSize = 0;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);
        if (IsSkipMask && IsSkipMaskForPre) {
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
            return;
        }

        if (!IsSkipMask) {
            const uint32_t mte2ToVId = UB_IN_MASK_EVENT0 + bufIdx;
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUb, attenMaskGmInt_, maskInfo, false,
                                                                        mte2ToVId);
        } else {
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
        }

        if (!IsSkipMaskForPre) {
            const uint32_t preBufId = bufIdx ^ 1U;
            const uint32_t preMte2ToVId = UB_IN_MASK_EVENT0 + preBufId;
            LocalTensor<uint8_t> attenMaskUbPre = ubMaskBuffers_[preBufId * UB_MASK_BUF_BYTES];
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUbPre, attenMaskGmInt_, maskInfo, true,
                                                                        preMte2ToVId);
            Mutex::Lock<PIPE_V>(preMte2ToVId);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
            Mutex::Unlock<PIPE_V>(preMte2ToVId);
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

    __aicore__ inline FANoQuantGqaBlockVecDummyNd(ConstInfoX &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                  SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace BaseApi
#endif // FLASH_ATTN_BLOCK_VEC_ND_H_
