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
 * \file fia_block_vec_fullquant_mx.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_FULLQUANT_MX_MX_MMQM_H_
#define FIA_BLOCK_VEC_FULLQUANT_MX_MX_MMQM_H_

// #include "kernel_operator.h"

// #include "flash_attention_score_common_regbase.h"
// #include "adv_api/activation/softmax.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_mx.h"
#include "vf/vf_flashupdate_new_mx.h"
// #include "vf/vf_div_cast.h"
// #include "vf/vf_flash_decode.h"
// #include "fia_public_define_arch35.h"
// #include "../vector_common.h"
// #include "../memory_copy_arch35.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {
template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0, bool isFd = false,
    bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMxBlockVec {
public:

    // using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    // const ConstInfoX &constInfo;

    static constexpr uint32_t actS1Size = (uint32_t)4;//修改
    static constexpr uint32_t actS2Size = (uint32_t)10240;//修改
    static constexpr uint32_t mBaseSize = (uint32_t)128;
    static constexpr uint32_t s2BaseSize = (uint32_t)512;
    static constexpr uint32_t dBaseSize = (uint32_t)128;
    static constexpr uint32_t dVBaseSize = (uint32_t)128;
    static constexpr uint32_t bSize = (uint32_t)48;//修改
    static constexpr uint32_t n2Size = (uint32_t)8; 
    static constexpr uint32_t gSize = (uint32_t)8;
    static constexpr uint32_t s1Size = (uint32_t)4;
    static constexpr uint32_t s2Size = (uint32_t)512;
    // static constexpr uint32_t n2Size = (uint32_t)8;
    static constexpr uint32_t realGSize = (uint32_t)8;
    static constexpr uint32_t maxBlockNumPerBatch = (uint32_t)20;//修改
    static constexpr uint32_t attenMaskS1Size = 2048;
    static constexpr uint32_t attenMaskS2Size = 2048;
    static constexpr float scaleValue = 0.08838834764831843;
    static constexpr int64_t preTokens = 2147483647;
    static constexpr int64_t nextTokensLeftUp = 10240 - 4;//修改 actSeqLenKV - actSeqLensQ
    static constexpr bool isRowInvalidOpen = 0;
    static constexpr uint32_t sparseMode = 3;
    static constexpr uint32_t actMSize = (uint32_t)32;
    static constexpr uint32_t actVecMSize = (uint32_t)16;
    static constexpr uint32_t actMSizeAlign32 = (uint32_t)32;
    static constexpr uint32_t actMSizeAlign16 = (uint32_t)16;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t gS1Idx = (uint32_t)0;
    static constexpr uint32_t FirstS2Loop = (uint32_t)0;//修改
    static constexpr uint32_t LastS2Loop = (uint32_t)20;//修改
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    uint8_t subBlockIdx = GetSubBlockIdx();
    float pScaleValue { 1.0f };
    
    __gm__ uint64_t *actualSeqLengthsGmQ;
    __gm__ float *pScaleGm;
    __ubuf__ void *stage1OutQue[2];
    __ubuf__ void *stage2OutBuf;
    __ubuf__ void *attenMaskUB[2];
    __ubuf__ void *softmaxSumUB[3];
    __ubuf__ void *softmaxMaxUB[3];
    __ubuf__ void *softmaxExpUB[3];
    __ubuf__ void *preLoopMaxUB;
    __ubuf__ void *preLoopSumUB;
    __ubuf__ void *firstLoopSumUB;
    __ubuf__ void *commonUB;
    __ubuf__ void *pScaleSubLoop0UB;
    __ubuf__ uint8_t *vselrIndexesUB[4];
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    __ubuf__ float * pScaleUb;
    __gm__ uint8_t* attenMaskGmInt;
    __gm__ OUTPUT_T* attentionOutGm;
    int ikv{-1};
    int iq{-1};
    int iquse = -1;
    int ipid{-1};
    // int bmm1_id{0};
    int vec1_id{-1};
    int mm1_id{-1};
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    float negativeFloatScalar = 0.0f;

    using OUT_T = OUTPUT_T;
    static constexpr bool HAS_MASK = hasAtten;

    __aicore__ inline FAFullQuantMxBlockVec(){};
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *pScale,
                                        __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *attentionOut)
    {
       uint32_t tmp1 = 0xFF7FFFFF;
        this->negativeFloatScalar = *((float *)&tmp1);

        InitVecInput(actualSeqQlenAddr, pScale, attenMask, attentionOut);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr,__gm__ uint8_t *pScale, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *attentionOut)
    {
        attentionOutGm = (__gm__ OUTPUT_T *) attentionOut;
        attenMaskGmInt = attenMask;
        // actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        actualSeqLengthsGmQ = (__gm__ uint64_t *)actualSeqQlenAddr;
        pScaleGm = (__gm__ float *)pScale;
        pScaleValue = pScaleGm[0];  
    __aicore__ inline void ub_l1_vec1(__cbuf__ void *p1, uint32_t subLoop, uint32_t createdTaskCount)
    {
        // block_count = 256 / 32;
        // block_len = (uint16_t)16;//(uint16_t)actMSizeAlign16
        // srcGap = (uint16_t)(65 - 16);//(uint16_t)(vec1Srcstride - actMSizeAlign16)
        // dstGap = (uint16_t)(128 - 16);//(uint16_t)(128 - actMSizeAlign16)
        int64_t dstOffset =
                subBlockIdx * 32 * actMSizeAlign16 +
                256 * 128 * subLoop;
        //set_mark(0x11223390);
        //set_mark(dstOffset);
        copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ float8_e4m3_t *)p1 + dstOffset), 
                          (__ubuf__ void*)((__ubuf__ float8_e4m3_t *)stage1OutQue[createdTaskCount]), 
                           0, 8, (uint16_t)actMSizeAlign16 , (uint16_t)(vec1Srcstride - actMSizeAlign16), (uint16_t)(128 - actMSizeAlign16));

    }

    }

    __aicore__ inline void process_vec1_noupdate(__ubuf__ void *mm1Res, int mm1_id, uint32_t subLoop, uint32_t loop, uint32_t mloop, float minValue, const float pScale = 1.0f)
    {
        const uint32_t blockStride = 65;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[loop % 2];
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[loop % 2] + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)commonUB;
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)commonUB;
        __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *maxUbStart = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[loop % 3];
        __ubuf__ float *inMaxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1Res;//这个看mm1_id需要改吗
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesUB[0];
        __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB;
        ////printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloop, loop);
        ////printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloop % 3, loop % 3);

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)attenMaskUB[0];
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64);
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 2);
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 3);
        // __ubuf__ uint32_t *dropMaskUb1 = (__ubuf__ uint32_t *)dropTensor.GetPhyAddr();
        // __ubuf__ uint32_t *dropMaskUb2 = (__ubuf__ uint32_t *)(dropTensor.GetPhyAddr() + 256 / 16);

        const uint32_t nPadding = (256 + 32 - 1) / 32 * 32;
        const uint32_t oriTailN1 = 256 - 64 * 2 < 64 ? 256 - 64 * 2 : 64;
        const uint32_t oriTailN2 = static_cast<int32_t>(256 - 64 * 3) <= 0 ? 0 : 256 - 64 * 3;
        const uint32_t tailN1 = 256 - 64 * 2;
        const uint32_t tailN2 = 256 - 64 * 3;
        uint32_t pltOriTailN1 = oriTailN1;
        uint32_t pltOriTailN2 = oriTailN2;
        uint32_t pltTailN1 = tailN1;
        uint32_t pltTailN2 = tailN2;
        // float divValue = 1.0f / keepProb;
        // uint32_t pltN = 256;

        if (subLoop == 0) {
            FaVectorApi::process_vec1_noupdate_subloop0<float, fp8_e4m3fn_t>(expUb1, expUb2, maxUb, maxUbStart, srcUb, expSumUb, indexesUb, 
                maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(scaleValue), minValue, pScaleValue);
        } else {
            FaVectorApi::process_vec1_noupdate_subloop1<float, fp8_e4m3fn_t>(expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb,
                inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, 
                maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(scaleValue), minValue, pScaleValue);
        }
    }

    __aicore__ inline void process_vec1_update(__ubuf__ void *mm1Res, int mm1_id, uint32_t subLoop, uint32_t loop, uint32_t mloop, float minValue, const float pScale = 1.0f)
    {
        const uint32_t blockStride = 65;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[loop % 2];
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[loop % 2] + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)commonUB;
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)commonUB;
        __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *maxUbStart = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[loop % 3];
        __ubuf__ float *inMaxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1Res;//这个看mm1_id需要改吗
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesUB[0];

        __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB;
        __ubuf__ float *preLoopMaxUb = (__ubuf__ float *)preLoopMaxUB;
        __ubuf__ float *preLoopSumUb = (__ubuf__ float *)preLoopSumUB;;
        __ubuf__ float *firstLoopSumUb = (__ubuf__ float *)firstLoopSumUB;

        ////printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloopCount, loop);
        ////printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloopCount % 3, loop % 3);

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)attenMaskUB[0];
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64);
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 2);
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 3);
        // __ubuf__ uint32_t *dropMaskUb1 = (__ubuf__ uint32_t *)dropTensor.GetPhyAddr();
        // __ubuf__ uint32_t *dropMaskUb2 = (__ubuf__ uint32_t *)(dropTensor.GetPhyAddr() + 256 / 16);

        const uint32_t nPadding = (256 + 32 - 1) / 32 * 32;
        const uint32_t oriTailN1 = 256 - 64 * 2 < 64 ? 256 - 64 * 2 : 64;
        const uint32_t oriTailN2 = static_cast<int32_t>(256 - 64 * 3) <= 0 ? 0 : 256 - 64 * 3;
        const uint32_t tailN1 = 256 - 64 * 2;
        const uint32_t tailN2 = 256 - 64 * 3;
        uint32_t pltOriTailN1 = oriTailN1;
        uint32_t pltOriTailN2 = oriTailN2;
        uint32_t pltTailN1 = tailN1;
        uint32_t pltTailN2 = tailN2;
        // float divValue = 1.0f / keepProb;
        // uint32_t pltN = 256;

        if (subLoop == 0) {
            FaVectorApi::process_vec1_update_subloop0<float, fp8_e4m3fn_t>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(scaleValue), minValue, pScaleValue);
        } else {
            FaVectorApi::process_vec1_update_subloop1<float, fp8_e4m3fn_t>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(scaleValue), minValue, pScaleValue);
        }
    }

    __aicore__ inline bool IsSkipAttentionmask_MX(uint32_t gs1StartIdx, uint32_t s2StartIdx)
    {
        // int32_t s1StartIdx = gs1StartIdx / 8;
        // int64_t nextToken = 
        //         static_cast<int64_t>(10240) - static_cast<int64_t>(4); // 统一以左上角为原点计算Token
        
        // if (static_cast<int64_t>(s2StartIdx + 256) <= static_cast<int64_t>(s1StartIdx) + 10236) {
        //     return true;
        // }
        int32_t s1StartIdx = gs1StartIdx / gSize;
        int64_t nextToken = 
            static_cast<int64_t>(actS2Size) - static_cast<int64_t>(actS1Size);
        if (static_cast<int64_t>(s2StartIdx + 256) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            return true;
        }
        return false;
    }
 
        __aicore__ inline uint64_t ComputeAttenMaskOffset_MX(int32_t s2StartIdx, uint32_t s1StartIdx)
    {
        int64_t nextToken = 0; // sparse2 本身原点就是左上角
        nextToken =
            static_cast<int64_t>(actS2Size) - static_cast<int64_t>(actS1Size); // 统一以左上角为原点计算token
        
        uint64_t offset = 0;
        int64_t delta = nextToken + s1StartIdx - s2StartIdx;
        uint32_t attenMaskSizeAlign = 256;
        if (delta < 0) {
            offset = (-delta) < static_cast<int64_t>(actVecMSize) ? (-delta) : actVecMSize; // min (-delta, s1Size)
        } else {
            offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                     attenMaskS2Size; // min(delta, s2inner)
        }
        set_mark(0x1234450);
        set_mark(actS2Size);//200
        set_mark(actS1Size);//4
        set_mark(s1StartIdx);//0
        set_mark(s2StartIdx);//100
        set_mark(attenMaskSizeAlign);//100
        set_mark(delta);//fc
        set_mark(actVecMSize);//10
        set_mark(attenMaskS2Size);//800
        set_mark(offset);//0x7e000
        return offset;
    }

    __simd_vf__ void MaskUbCopyS1G_MX(__ubuf__ uint8_t * maskUb, int32_t headGLoop, int32_t midGLoop,
                                        int32_t tailGLoop, int32_t midS1Count)
    {
        if (midGLoop <= 1) {
            return;
        }
        constexpr uint32_t repeatStride = 8;//256 >> 5

        RegTensor<uint8_t> vreg_g_head;
        RegTensor<uint8_t> vreg_g_mid;
        RegTensor<uint8_t> vreg_g_tail;

        MaskReg preg_all;
        preg_all = CreateMask<bool, MaskPattern::ALL>();

        for (uint16_t x = headGLoop; x > 0; x = 0) {     // if (headGLoop > 0) {}
            // LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //     vreg_g_head, maskUb, 1, repeatStride, preg_all);
            constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
            vsldb((RegTensor<uint8_t>&)vreg_g_head, (__ubuf__ uint8_t*&)maskUb,
                (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            for (uint16_t i = 1; i < headGLoop; ++i) {
                // StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                //     maskUb, vreg_g_head, 1, repeatStride, preg_all);
                constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                vsstb((RegTensor<uint8_t>&)vreg_g_head, (__ubuf__ uint8_t*&)maskUb,
                  (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            }
        }

        for (uint16_t x = midGLoop; x > 0; x = 0) {     // if (midGLoop > 0) {}
            for (uint16_t i = 0; i < midS1Count; ++i) {
                // LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                //     vreg_g_mid, maskUb, 1, repeatStride, preg_all);
                constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                vsldb((RegTensor<uint8_t>&)vreg_g_mid, (__ubuf__ uint8_t*&)maskUb,
                    (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
                for (uint16_t j = 1; j < midGLoop; ++j) {
                    // StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    //     maskUb, vreg_g_mid, 1, repeatStride, preg_all);
                    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                    vsstb((RegTensor<uint8_t>&)vreg_g_mid, (__ubuf__ uint8_t*&)maskUb,
                      (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
                    }
                // LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                mem_bar(VST_VLD);
            }
        }

        for (uint16_t x = tailGLoop; x > 1; x = 0) {     // if (tailGLoop > 1) {}
            // LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //     vreg_g_tail, maskUb, 1, repeatStride, preg_all);
            constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
            vsldb((RegTensor<uint8_t>&)vreg_g_tail, (__ubuf__ uint8_t*&)maskUb,
                (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            for (uint16_t i = 1; i < tailGLoop; ++i) {
                // StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                //     maskUb, vreg_g_tail, 1, repeatStride, preg_all);
                constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                    vsstb((RegTensor<uint8_t>&)vreg_g_tail, (__ubuf__ uint8_t*&)maskUb,
                      (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            }
        }


    }

    // __aicore__ inline void AttentionmaskDataCopy_MX(
    //                                             int32_t s2StartIdx, uint32_t s1StartIdx, uint32_t s1EndIdx, bool isPre = false)
    // {
    //     uint32_t attenMaskSizeAlign = 256;
    //     uint64_t maskOffset = ComputeAttenMaskOffset_MX(s2StartIdx, s1StartIdx);
    //     // DataCopyExtParams dataCopyParams;
    //     // dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    //     // dataCopyParams.blockLen = 256;
    //     // dataCopyParams.srcStride = attenMaskS2Size - 256;
    //     // dataCopyParams.dstStride = 0;
    //     // DataCopyPadExtParams<T> padParams{true, 0, 0, 1U};
    //     const uint8_t cacheMode = ExtractCacheMode((__gm__ uint8_t*)attenMaskGmInt + maskOffset);
    //     set_mov_pad_val(GetScalarBitcodeValue(1U));
    //     // DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
        
    //     copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)attenMaskUB[0],
    //                                  (__gm__ uint8_t*)attenMaskGmInt + maskOffset, 0, 
    //                                     s1EndIdx - s1StartIdx, 256,
    //                                     0, 0, true, cacheMode, attenMaskS2Size - 256  + 256,
    //                                     256);
    // }

    __aicore__ inline void AttentionmaskCopyInForSgLayout_MX(int32_t s2StartIdx, uint32_t gs1StartIdx)
    {
        uint32_t s1StartIdx = gs1StartIdx / gSize;
        uint32_t s1EndIdx = (gs1StartIdx + actVecMSize - 1) / gSize;
        uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
        uint32_t headGSize = actVecMSize;
        if (s1Count > 1) {
            headGSize = (gs1StartIdx % gSize == 0) ? 0 : (gSize - gs1StartIdx % gSize);
        }
        uint32_t remainRowCount = actVecMSize - headGSize;
        uint32_t midS1Count = remainRowCount / gSize;
        uint32_t tailGSize = remainRowCount % gSize;
        // uint32_t attenMaskSizeAlign = 256;
        
        // uint32_t attenMaskS2Stride = 256;
    
        // // ub-head
        // if (headGSize > 0) {
        //     AttentionmaskDataCopy_MX(s2StartIdx, actVecMSize, s1StartIdx, s1StartIdx + 1, isPre);
        //     s1StartIdx++;
        // }
    
        // ub-remain
        if (remainRowCount > 0) {
            uint64_t maskOffset = ComputeAttenMaskOffset_MX(s2StartIdx, s1StartIdx);
            // DataCopyExtParams dataCopyParams;
            // dataCopyParams.blockCount = midS1Count + (tailGSize > 0);
            // dataCopyParams.blockLen = 256;
            // dataCopyParams.srcStride = attenMaskS2Size - 256;
            // dataCopyParams.dstStride = (gSize - 1) * attenMaskS2Stride / 32 + attenMaskDstStride;
            uint16_t blockCount = midS1Count + (tailGSize > 0);
            int64_t srcStride = attenMaskS2Size - 256;
            int64_t dstStride = (gSize - 1) * 256 / 32;



            // DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 1U};//isPad,leftPadding,rightPadding,paddingValue
            const uint8_t cacheMode = ExtractCacheMode((__gm__ uint8_t*)attenMaskGmInt + maskOffset);
            set_mov_pad_val(GetScalarBitcodeValue(1U));
            // DataCopyPad(attenMaskUB[headGSize * attenMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);

            uint32_t unitOfBytes = 1;
            // uint32_t burstLength = blockLen;
            uint32_t padLength = 0 * 1 + 0 * 1;
            uint64_t srcStride310 = srcStride * unitOfBytes + 256; // GM  DataCopy:32Bytes, DataCopyPad:1Byte
            uint32_t  dstStride310 =
                AlignUp(static_cast<uint32_t>(dstStride) * 32 + 256 + 0, 32);
            uint8_t leftPaddingCnt = 0;
            uint8_t rigntPaddingCnt = 0;

            using PrimType = PrimT<uint8_t>;
            set_mark(0x776690);
        set_mark(maskOffset);//0x7e000,0x7f000
            copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)attenMaskUB[0] + headGSize * 256,
                                     (__gm__ uint8_t*)attenMaskGmInt + maskOffset, 0, 
                                        blockCount, 256,
                                        0, 0, true, cacheMode, srcStride310,
                                        dstStride310);
        }
    
        set_flag(PIPE_MTE2, PIPE_V, event_t(0));
        wait_flag(PIPE_MTE2, PIPE_V, event_t(0));
        MaskUbCopyS1G_MX((__ubuf__ uint8_t*)attenMaskUB[0], headGSize, gSize, tailGSize, midS1Count);
    
        // MaskUbCopyS1G<T, s2BaseSize>(attenMaskUb, headGSize, info.gSize, tailGSize, midS1Count);
    }

     template <typename T1, typename RegType>
    __simd_vf__ inline void DuplicateFromScalarImpl(__ubuf__ T1 * dstLocal, const T1 scalarValue, const int32_t calCount)
    {
        RegType dstReg;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        Reg::MaskReg mask;
        constexpr uint32_t repeatStride = static_cast<uint32_t>(GetVecLen() * RegType::trait.REG_NUM);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
        Reg::MaskReg maskFull = Reg::CreateMask<T1, Reg::MaskPattern::ALL, RegType::trait>();
        Reg::Duplicate(dstReg, scalarValue, maskFull);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = Reg::UpdateMask<T1, RegType::trait>(sreg);
            Reg::StoreAlign(dstLocal + i * repeatStride, dstReg, mask);
        }
    }



    __aicore__ inline void AttenMaskCopyIn_MX(uint32_t bIdx, uint32_t s2Idx, uint32_t vecMbaseIdx, uint32_t mDealSize, uint32_t subLoop)
    {
        uint32_t gs1StartIdx = gS1Idx + vecMbaseIdx + 0;
        uint32_t s2StartIdx = (subLoop == 0) ? s2Idx : (s2Idx + 256);
        bool IsSkipMask = IsSkipAttentionmask_MX(gs1StartIdx, s2StartIdx);
        // bool IsSkipMask = IsSkipAttentionmask_MX(maskInfo);
        if (IsSkipMask) {
            // printf("IsSkipMask\n");
            DuplicateFromScalarImpl<uint8_t, Reg::RegTensor<uint8_t, Reg::RegTraitNumOne>>(
                (__ubuf__ uint8_t*)attenMaskUB[0], static_cast<uint8_t>(0U), mDealSize * 256);
            return;
        } else {    
            AttentionmaskCopyInForSgLayout_MX(s2StartIdx, gs1StartIdx);
        }
    }

    __aicore__ inline void ub_l1_vec1_pscale(__cbuf__ void *pscale, uint32_t subLoop, uint32_t createdTaskCount)
    {
        
        uint16_t copyCount = actMSizeAlign16 / 16;
        uint64_t pScaleDataLen = actMSizeAlign16 * 256 / MXFP_GROUP_SIZE;
        uint64_t vecOffset = subBlockIdx * pScaleDataLen;
        uint64_t subLoopOffset = pScaleDataLen * 2;
        uint16_t dstStride = 256 / MXFP_GROUP_SIZE / 2 - 1;
        for (uint16_t i = 0; i < 4; i++) { // PScale在s2方向的block块大小为32，所以一共有256/32=8个，而L1上需要满足16x2的分形，所以重复拷贝4次
            // DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0UB, {copyCount, 1, 0, dstStride});
            // DataCopy(mm2AScaleL1Tensor[subLoopOffset + vecOffset + i * 32], (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB + 128,
            //          {copyCount, 1, 0, dstStride});
            copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ fp8_e8m0_t *)pscale + vecOffset + i * 32), (__ubuf__ void*)pScaleSubLoop0UB, 0, copyCount, (uint16_t)1, (uint16_t)0, (uint16_t)dstStride);
            copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ fp8_e8m0_t *)pscale + subLoopOffset + vecOffset + i * 32), (__ubuf__ void*)((__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB + 128), 0,  copyCount, (uint16_t)1, (uint16_t)0, (uint16_t)dstStride);
        }
        

    }


//subloop1   搬运mask  比 subloop0还快  ，导致数据踩踏
    __aicore__ inline void ProcessVec1(__cbuf__ void *p1, __ubuf__ void *mm1Res, int bmm1_id, int ipuse, uint32_t bIdx, uint32_t s2Idx,
                          uint32_t loop, uint32_t mloop, uint32_t vecMbaseIdx, uint32_t curS2Loop, uint32_t subLoop)
    {
        // bmm1ResBuf.WaitCrossCore();
        int bmm1_idused = 6 + bmm1_id * 2;
        wait_intra_block(PIPE_V, bmm1_idused + 0);

        wait_flag(PIPE_V, PIPE_MTE2, (event_t)4); 
        AttenMaskCopyIn_MX(bIdx, s2Idx, vecMbaseIdx, actMSizeAlign16, subLoop);//actMSizeAlign16 = runInfo.actMSizeAlign32 >> 1;

        wait_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)4); //pscale AllocTensor
        if(curS2Loop == FirstS2Loop) {
           process_vec1_noupdate(mm1Res, bmm1_id, subLoop, loop, mloop, negativeFloatScalar);
        } else{
           process_vec1_update(mm1Res, bmm1_id, subLoop, loop, mloop, negativeFloatScalar);
        }
        // bmm1ResBuf.SetCrossCore();
        set_intra_block(PIPE_V, bmm1_idused + 1);

        set_flag(PIPE_V, PIPE_MTE2, (event_t)4); // attenMaskUb FreeTensor

        set_flag(PIPE_V, PIPE_MTE3, (event_t)3);//pscale EnQue
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)3);//pscale DeQue
        if (subLoop % 2 == 1) {
            ub_l1_vec1_pscale((__cbuf__ fp8_e8m0_t*)p1 + mBaseSize * s2BaseSize, subLoop, loop % 2);
        }
        set_flag(PIPE_MTE3, PIPE_V, (event_t)4);// pscale  FreeTensor

        
        set_flag(PIPE_V, PIPE_MTE3, (event_t)2);//stage1OutQue EnQue
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)2);//stage1OutQue DeQue
        ub_l1_vec1(p1, subLoop, loop % 2);
        set_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);//stage1OutQue FreeTensor


        if (subLoop % 2 == 1) {
            // outputBuf.SetCrossCore();//L1
            set_intra_block(PIPE_MTE3, ipuse);
        }
    }


    __aicore__ inline bool CalcBlockNeedRowInvalid(uint32_t actVecMSize, uint32_t vecMbaseIdx,
                                                    int64_t s1FirstValidToken, int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = gS1Idx + vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        // S1G layout
        s1StartTdx = vecMStartIdx / gSize;
        s1EndTdx = vecMEndIdx / gSize;
        ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
       
        return ret;
    }

    // template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(__ubuf__ float *vec2ResUb, int64_t mStartVec,
                                        uint32_t actVecMSize, uint32_t vecMbaseIdx, uint32_t mDealSize, uint32_t mloop, int64_t dSizeAligned64)
    {
        // if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken = Min(Max(-nextTokensLeftUp, 0), actS1Size);
            int64_t s1LastValidToken = Min(Max(preTokens + actS2Size, 0), actS1Size);
            s1LastValidToken = Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < actS1Size);
            bool batchNeedRowInvalid = isRowInvalidOpen || // 手动开启行无效
                                       ((sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(actVecMSize, vecMbaseIdx, s1FirstValidToken, s1LastValidToken);
            blockNeedRowInvalid = blockNeedRowInvalid || isRowInvalidOpen;

            if (blockNeedRowInvalid) {
                uint32_t min = 0xFF7FFFFF; // min value of float
                __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3]+ mStartVec;
                RowInvalidUpdateVF_MX<float>(vec2ResUb, maxUb, mDealSize, dVBaseSize,
                                          static_cast<uint32_t>(dSizeAligned64), min);
            }
        // }
    }
    

    __aicore__ inline void Bmm2DataCopyOutTrans(__ubuf__ OUTPUT_T * attenOutUb,
                                                uint32_t vecMIdx, uint32_t bIdx, uint32_t n2Idx,
                                                uint32_t actVecMSize, uint32_t vecMbaseIdx, uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        // FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
        //                               .rowCount = dealRowCount,
        //                               .colCount = (uint32_t)(splitD ? constInfo.dBasicBlock : dTemplateAlign64)};
        // GmCoord gmCoord{.bIdx = info.bIdx,
        //                 .n2Idx = info.realN2Idx,
        //                 .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
        //                 .dIdx = 0,
        //                 .gS1DealSize = gmDealRowCount,
        //                 .dDealSize = (uint32_t)constInfo.dSizeV};

        // CopyAttentionOut(ubTensor, gmCoord);

        uint32_t blockCount = 8;
        uint32_t blockLen = 128 * sizeof(OUTPUT_T);
        uint32_t srcStride = 0;
        uint32_t dstStride = 0;
        
        uint64_t tIdx = 0;
        if (bIdx == 0) {
            tIdx = 0;
        } else {
            // tIdx = actualSeqLengthsGm.GetValue(bIdx - 1);
            tIdx = actualSeqLengthsGmQ[bIdx -1];
        }

        tIdx += (gS1Idx + vecMbaseIdx + vecMIdx) / gSize;
        set_mark(0x873423432);
                        set_mark(gS1Idx);//0
set_mark(vecMbaseIdx);//0,10
set_mark(vecMIdx);//0
        uint64_t attenOutGmbaseOffset = tIdx * 8 * 8 * 128 + n2Idx * 8 * 128;
        uint64_t gmOffset = attenOutGmbaseOffset;
        uint64_t ubOffset = 0;

        const uint8_t cacheMode = ExtractCacheMode(attentionOutGm);
        uint32_t unitOfBytes = 1;
        uint32_t burstLength = blockLen;
        uint32_t srcStride310 = 0;
        srcStride310 = static_cast<uint32_t>(srcStride) * 32
                       + burstLength; // UB   DataCopy:32Bytes, DataCopyPad:32Bytes
        // srcstride should be 32B aligned
        srcStride310 = DivCeil(srcStride310, 32) * 32;
    
        uint64_t dstStride310 = dstStride + burstLength; // GM   DataCopy:32Bytes, DataCopyPad:1Byte
        set_mark(0x9988770);
        set_mark(bIdx);
        set_mark(ubOffset);//0
        set_mark(gmOffset);//0x4000,0x4400
        set_mark(attenOutGmbaseOffset);//0x4000,0x4400
        set_mark(tIdx);//2
        set_mark(n2Idx);//0
        copy_ubuf_to_gm_align_v2((__gm__ void*)((__gm__ OUTPUT_T*)attentionOutGm + gmOffset), (__ubuf__ void*)((__ubuf__ OUTPUT_T *)attenOutUb + ubOffset), 0,      blockCount, blockLen, cacheMode, 
	blockLen, srcStride310);

        gmOffset = attenOutGmbaseOffset + 8 * 8 * 128;
        ubOffset = ((uint64_t)8) * ((uint64_t)128);
        copy_ubuf_to_gm_align_v2((__gm__ void*)(attentionOutGm + gmOffset), (__ubuf__ void*)(attenOutUb + ubOffset), 0, blockCount, blockLen, cacheMode,
                                     blockLen, srcStride310);
    }

    template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
    __simd_vf__ inline void CastIntrinsicsImplVF_MX(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint32_t calCount)
    {
        uint16_t oneRepSize = GetVecLen() / sizeof(SRC_TYPE);
        if constexpr (sizeof(SRC_TYPE) < sizeof(DST_TYPE)) {
            oneRepSize = GetVecLen() / sizeof(DST_TYPE);
        }
        uint16_t repeatTime = (calCount + oneRepSize - 1) / oneRepSize;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        static constexpr Reg::CastTrait castTrait = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, roundMode};
        Reg::MaskReg preg;
        Reg::RegTensor<SRC_TYPE> srcVreg;
        Reg::RegTensor<DST_TYPE> dstVreg;
        Reg::RegTensor<float> tmpVreg;
        for (uint16_t i = 0; i < repeatTime; ++i) {//有问题
            // preg = Reg::UpdateMask<SRC_TYPE>(sreg);
            preg = plt_b32(sreg, POST_UPDATE);
            // GenLoadL2<DST_TYPE, SRC_TYPE>(srcVreg, src + i * oneRepSize, preg);
            constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
            vlds(srcVreg, src + i * oneRepSize, 0, distValue);
            // Reg::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, preg);
            constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(Reg::RegLayout::ZERO)>();
            constexpr auto modeValue = std::integral_constant<::Mode, static_cast<::Mode>(Reg::MaskMergeMode::ZEROING)>();
            constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
            constexpr auto satModeValue = std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(Reg::SatMode::SAT)>();
            vcvt(dstVreg, srcVreg, preg, roundModeValue, satModeValue, partModeValue, modeValue);
            // CastOperator<DST_TYPE, SRC_TYPE, Reg::RegLayout::ZERO, Reg::MaskMergeMode::ZEROING>(dstVreg, srcVreg, preg);
            // GenStoreL2<DST_TYPE, SRC_TYPE>(dst + i * oneRepSize, dstVreg, preg);
            constexpr auto distValue2 = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<DST_TYPE, Reg::StoreDist::DIST_PACK_B32>())>();
            vsts(dstVreg, dst + i * oneRepSize, 0, distValue2, preg);
        }
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(__ubuf__ float *vec2ResUb, uint32_t mStartVec, uint32_t bIdx, uint32_t n2Idx, uint32_t vecMbaseIdx,
                                                uint32_t mDealSize, uint32_t gmDealRowCount, uint32_t mloop)
    {
        // LocalTensor<OUTPUT_T> attenOut;
        __ubuf__ OUTPUT_T *attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        // attenOut.SetAddr(vec2ResUb.address_);
        // attenOut.SetAddr(vec2ResUb);
        attenOut = (__ubuf__ OUTPUT_T *)vec2ResUb;

        RowInvalid(vec2ResUb, mStartVec, actVecMSize, vecMbaseIdx, mDealSize, mloop, dSizeAligned64);
        // Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, actVecMSize * dSizeAligned64);
        using DstPrimType = PrimT<OUTPUT_T>;
        using SrcPrimType = PrimT<float>;
        // CastImpl((__ubuf__ DstPrimType*)attenOut, (__ubuf__ SrcPrimType*)vec2ResUb, RoundMode::CAST_ROUND, actVecMSize * dSizeAligned64);

        using DST_TYPE = typename CastParam::CastTypeTrait<DstPrimType>::RealType;
        using SRC_TYPE = typename CastParam::CastTypeTrait<SrcPrimType>::RealType;
        __ubuf__ DST_TYPE* dst = reinterpret_cast<__ubuf__ DST_TYPE*>(attenOut);
        __ubuf__ SRC_TYPE* src = reinterpret_cast<__ubuf__ SRC_TYPE*>(vec2ResUb);
        CastIntrinsicsImplVF_MX<DST_TYPE, SRC_TYPE, RoundMode::CAST_ROUND>(dst, src, actVecMSize * dSizeAligned64);
           
        // SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        // WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
        Bmm2DataCopyOutTrans(attenOut, mStartVec, bIdx, n2Idx, actVecMSize, vecMbaseIdx, mDealSize, gmDealRowCount);
    }

    __aicore__ inline void ProcessVec2(__ubuf__ void *mm2Res, int bmm2_id, uint32_t loop, uint32_t mloop, uint32_t bIdx, uint32_t n2Idx,
                                        uint32_t vecMbaseIdx, uint32_t curS2Loop)
    {
        // bmm2ResBuf.WaitCrossCore();
        wait_intra_block(PIPE_V, bmm2_id);
        // if (unlikely(runInfo.actVecMSize == 0)) {
        //     bmm2ResBuf.SetCrossCore();
        //     return;
        // }

        uint32_t vecMSize = actMSizeAlign32 / 2;
        int64_t vec2CalcSize = actVecMSize * 128;
        float deSCaleVValue = 1.0f;

        // LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        // LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        // WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);

        __ubuf__ float * dstUb = (__ubuf__ T*)stage2OutBuf;
        __ubuf__ float * curUb = (__ubuf__ T*)mm2Res;
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0);

        // set_mark(0x89838990);
        // // set_mark(runInfo.isFirstS2Loop);
        // set_mark(inner);
        if (curS2Loop == FirstS2Loop) {
            // DataCopy(vec2ResUb, mmRes, vec2CalcSize);
            using PrimType = PrimT<T>;
            struct DataCopyParams repeatParams;
            repeatParams.blockLen = vec2CalcSize / AscendCUtils::GetC0Count(sizeof(PrimType));
            copy_ubuf_to_ubuf((__ubuf__ void*)stage2OutBuf, (__ubuf__ void*)mm2Res, 0, 1U, 256U, 0U, 0U);

            // printf("repeatParams.blockCount = %d, repeatParams.blockLen =%d,repeatParams.srcStride =%d,repeatParams.dstStride =%d\n",repeatParams.blockCount,repeatParams.blockLen,repeatParams.srcStride,repeatParams.dstStride);
            // copy_ubuf_to_ubuf((__ubuf__ void*)stage2OutBuf, (__ubuf__ void*)mm2Res, 0, repeatParams.blockCount, repeatParams.blockLen, repeatParams.srcStride, repeatParams.dstStride);
            // copy_ubuf_to_ubuf((__ubuf__ void*)stage2OutBuf, (__ubuf__ void*)mm2Res, 0, 1, vec2CalcSize, 0, 0);
        } else {
            // // LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            // // LocalTensor<T> pScaleUb;

            __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[loop % 3];

            float deSCalePreVValue = 1.0f;
            if ((curS2Loop + 1 )!= LastS2Loop) {//修改
                __ubuf__ float * preUb = (__ubuf__ T*)stage2OutBuf;
                // __ubuf__ float * rowMaxUb = (__ubuf__ T*)pScaleUb;
                FaVectorApi::FlashUpdateBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, vecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                __ubuf__ float * preUb = (__ubuf__ T*)stage2OutBuf;
                __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
                // __ubuf__ float * rowMaxUb = (__ubuf__ T*)pScaleUb;
                FaVectorApi::FlashUpdateLastBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, expSumUb, vecMSize, dTemplateAlign64, 1.0,
                    1.0);
            }
        }
        // bmm2ResBuf.SetCrossCore();
        // set_intra_block(PIPE_V, bmm2_id + 1);
        set_intra_block(PIPE_V, bmm2_id + 1);
        // set_mark(0x89838991);
        // // set_mark(runInfo.isLastS2Loop);
        // set_mark(inner);
        if ((curS2Loop + 1) == LastS2Loop) {// 修改
            if (curS2Loop == FirstS2Loop) {
                // LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                __ubuf__ float * srcUb = (__ubuf__ T*)stage2OutBuf;
                __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
                FaVectorApi::LastDivNewVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    dstUb, srcUb, expSumUb, actVecMSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            uint32_t gmDealRowCount;
            uint32_t groupsOf32 = (actMSize + 31) / 32;
            if (subBlockIdx == 0) {
                gmDealRowCount = groupsOf32 * 16 > actMSize ? actMSize : groupsOf32 * 16;
            } else {
                int32_t vec1RemainRows = actMSize - 16 * groupsOf32;
                gmDealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
            }
            
            if (gmDealRowCount == 0) {
                set_flag(PIPE_MTE3, PIPE_V, (event_t)0);
                return;
            }
            Bmm2ResCastAndCopyOut(dstUb, 0, bIdx, n2Idx, vecMbaseIdx, vecMSize, gmDealRowCount, mloop);
        }
        // SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)0);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t stage1OutSize = (64 + 1) * 256;
        constexpr uint32_t maskSize = 64  * 256;
        constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
        constexpr uint32_t mm2ResultSize = 128 / 2 * 128 * 4;
        stage1OutQue[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize);
        stage1OutQue[1] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize);

        attenMaskUB[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2);//16384


        softmaxSumUB[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize);
        softmaxSumUB[1] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256);
        softmaxSumUB[2] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 2);
        softmaxMaxUB[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 3);
        softmaxMaxUB[1] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 4);
        softmaxMaxUB[2] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 5);
        softmaxExpUB[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 6);
        softmaxExpUB[1] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 7);
        softmaxExpUB[2] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 8);
        preLoopMaxUB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 9);
        preLoopSumUB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 10);
        firstLoopSumUB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 11);
        commonUB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12);

        stage2OutBuf = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512);
        pScaleSubLoop0UB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4);
        vselrIndexesUB[0] = (__ubuf__ uint8_t *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4 + 512);
        vselrIndexesUB[1] = (__ubuf__ uint8_t *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4+ 512 + 128);
         
        // constexpr auto orderMode = std::integral_constant<::Order, static_cast<::Order>(order)>();
        // vci((__ubuf__ uint8_t *)vselrIndexesUB[0], 2, orderMode);
        // vselrIndexesUB[0] = 0x7f;
        // 先获取 UB 内存地址（uint8 指针）
        // __ubuf__ uint8_t *ub0 = (__ubuf__ uint8_t *)vselrIndexesUB[0];
        // __ubuf__ uint8_t *ub1 = (__ubuf__ uint8_t *)vselrIndexesUB[1];

        for (int i = 0; i < 128; i++) {
            *(vselrIndexesUB[0] + i) = i * 2;
        }
        for (int i = 0; i < 64; i++) {
            *(vselrIndexesUB[1] + i) = i * 4;
            // vselrIndexesUB[1][i] = i * 4;
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        // mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        // vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        // vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        // SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        // SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
        // SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        // //printf("AllocEventID mte3ToVId[0] = %d,mte3ToVId[1] = %d,vToMte3Id[0] = %d,vToMte3Id[1] = %d\n",mte3ToVId[0],mte3ToVId[1],vToMte3Id[0],vToMte3Id[1]);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)0);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)1);

        set_flag(PIPE_MTE3, PIPE_V, (event_t)2);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)3);

        set_flag(PIPE_MTE3, PIPE_V, (event_t)4);

        set_flag(PIPE_V, PIPE_MTE2, (event_t)4); 
        // set_flag(PIPE_MTE3, PIPE_V, (event_t)5); 
    }

    __aicore__ inline void FreeEventID()
    {
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)1);

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)2);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)3);

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)4);

        wait_flag(PIPE_V, PIPE_MTE2, (event_t)4); 
        // wait_flag(PIPE_MTE3, PIPE_V, (event_t)5); 
        // WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
        // WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
        // GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[0]);
        // GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[1]);
        // GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[0]);
        // GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[1]);
        // //printf("FreeEventID mte3ToVId[0] = %d,mte3ToVId[1] = %d,vToMte3Id[0] = %d,vToMte3Id[1] = %d\n",mte3ToVId[0],mte3ToVId[1],vToMte3Id[0],vToMte3Id[1]);
    }
};

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0, bool isFd = false,
    bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMxBlockVecDummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    __aicore__ inline FAFullQuantMxBlockVecDummy(ConstInfoX &constInfo) {};
};
} // namespace BaseApi
#endif // FIA_BLOCK_VEC_FULLQUANT_MX_H_
