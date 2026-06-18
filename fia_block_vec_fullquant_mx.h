/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
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
    static constexpr uint32_t actS2Size = (uint32_t)512;//修改
    static constexpr uint32_t mBaseSize = (uint32_t)128;
    static constexpr uint32_t s2BaseSize = (uint32_t)512;
    static constexpr uint32_t dBaseSize = (uint32_t)128;
    static constexpr uint32_t dVBaseSize = (uint32_t)128;
    static constexpr uint32_t bSize = (uint32_t)2;//修改
    static constexpr uint32_t n2Size = (uint32_t)8;
    static constexpr uint32_t gSize = (uint32_t)8;
    static constexpr uint32_t s1Size = (uint32_t)4;
    static constexpr uint32_t s2Size = (uint32_t)512;
    // static constexpr uint32_t n2Size = (uint32_t)8;
    static constexpr uint32_t realGSize = (uint32_t)8;
    static constexpr uint32_t maxBlockNumPerBatch = (uint32_t)1;//修改
    static constexpr uint32_t attenMaskS1Size = 2048;
    static constexpr uint32_t attenMaskS2Size = 2048;
    static constexpr float scaleValue = 0.08838834764831843;
    static constexpr int64_t preTokens = 2147483647;
    static constexpr int64_t nextTokensLeftUp = 512 - 4;//修改 actSeqLenKV - actSeqLensQ
    static constexpr bool isRowInvalidOpen = 0;
    static constexpr uint32_t sparseMode = 3;
    static constexpr uint32_t actVecMSize = (uint32_t)16;
    static constexpr uint32_t gS1Idx = (uint32_t)0;
    static constexpr uint32_t FirstS2Loop = (uint32_t)0;
    static constexpr uint32_t LastS2Loop = (uint32_t)1;
    uint8_t subBlockIdx = GetSubBlockIdx();

    __gm__ uint64_t *actualSeqLengthsGmQ;
    TBuf<TPosition::VECIN> UB_stage1OutQue0;
    TBuf<TPosition::VECIN> UB_stage1OutQue1;
    TBuf<TPosition::VECIN> UB_stage2OutBuf;
    TBuf<TPosition::VECIN> UB_attenMask;
    TBuf<TPosition::VECIN> UB_softmaxSum0;
    TBuf<TPosition::VECIN> UB_softmaxSum1;
    TBuf<TPosition::VECIN> UB_softmaxSum2;
    TBuf<TPosition::VECIN> UB_softmaxMax0;
    TBuf<TPosition::VECIN> UB_softmaxMax1;
    TBuf<TPosition::VECIN> UB_softmaxMax2;
    TBuf<TPosition::VECIN> UB_softmaxExp0;
    TBuf<TPosition::VECIN> UB_softmaxExp1;
    TBuf<TPosition::VECIN> UB_softmaxExp2;
    TBuf<TPosition::VECIN> UB_preLoopMax;
    TBuf<TPosition::VECIN> UB_preLoopSum;
    TBuf<TPosition::VECIN> UB_firstLoopSum;
    TBuf<TPosition::VECIN> UB_common;
    TBuf<TPosition::VECIN> UB_vselrIndexes0;
    TBuf<TPosition::VECIN> UB_vselrIndexes1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    TBuf<TPosition::VECIN> UB_pScale;
    TPipe *tPipePtr = nullptr;
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

    __aicore__ inline void SetTPipe(TPipe& pipe) { tPipePtr = &pipe; }

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
        // if constexpr (isFp8) {
        //     if (pScale != nullptr) {
        //         pScaleGm.SetGlobalBuffer((__gm__ float *)pScale);
        //         pScaleValue = this->pScaleGm.GetValue(0);
        //     }
        // }

    }
    __aicore__ inline void ub_l1_vec1(LocalTensor<float8_e4m3_t> p1Tensor, uint32_t subLoop, uint32_t createdTaskCount)
    {
        // block_count = 256 / 32;
        // block_len = (uint16_t)16;
        // srcGap = (uint16_t)(65 - 16);
        // dstGap = (uint16_t)(128 - 16);
        int64_t dstOffset =
                subBlockIdx * 32 * 16 +
                256 * 128 * subLoop;
        //set_mark(0x11223390);
        //set_mark(dstOffset);
        LocalTensor<float8_e4m3_t> dstTensor = p1Tensor[dstOffset];
        LocalTensor<float8_e4m3_t> srcTensor = (createdTaskCount == 0)
            ? UB_stage1OutQue0.Get<float8_e4m3_t>()
            : UB_stage1OutQue1.Get<float8_e4m3_t>();

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 8;
        dataCopyParams.blockLen = 16;
        dataCopyParams.srcStride = 49;
        dataCopyParams.dstStride = 112;

        DataCopy(dstTensor, srcTensor, dataCopyParams);

    }

    __aicore__ inline void process_vec1_noupdate(LocalTensor<float> mm1ResTensor, int mm1_id, uint32_t subLoop, uint32_t loop, uint32_t mloop, float minValue, const float pScale = 1.0f)
    {
        const uint32_t blockStride = 65;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (loop % 2 == 0)
            ? (__ubuf__ fp8_e4m3fn_t *)UB_stage1OutQue0.Get<fp8_e4m3fn_t>().GetPhyAddr()
            : (__ubuf__ fp8_e4m3fn_t *)UB_stage1OutQue1.Get<fp8_e4m3fn_t>().GetPhyAddr();
        __ubuf__ fp8_e4m3fn_t *expUb2 = expUb1 + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxSum0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxSum1.Get<float>().GetPhyAddr() : UB_softmaxSum2.Get<float>().GetPhyAddr()));
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxSum0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxSum1.Get<float>().GetPhyAddr() : UB_softmaxSum2.Get<float>().GetPhyAddr()));
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr();
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr();
        __ubuf__ float *maxUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxMax0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxMax1.Get<float>().GetPhyAddr() : UB_softmaxMax2.Get<float>().GetPhyAddr()));
        __ubuf__ float *maxUbStart = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxMax0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxMax1.Get<float>().GetPhyAddr() : UB_softmaxMax2.Get<float>().GetPhyAddr()));
        __ubuf__ float *expMaxUb = (__ubuf__ float *)(loop % 3 == 0 ? UB_softmaxExp0.Get<float>().GetPhyAddr() : (loop % 3 == 1 ? UB_softmaxExp1.Get<float>().GetPhyAddr() : UB_softmaxExp2.Get<float>().GetPhyAddr()));
        __ubuf__ float *inMaxUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxMax0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxMax1.Get<float>().GetPhyAddr() : UB_softmaxMax2.Get<float>().GetPhyAddr()));
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr() + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr() + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1ResTensor.GetPhyAddr();//这个看mm1_id需要改吗
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)UB_vselrIndexes0.Get<uint8_t>().GetPhyAddr();

        ////printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloop, loop);
        ////printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloop % 3, loop % 3);

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)UB_attenMask.Get<uint8_t>().GetPhyAddr();
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)UB_attenMask.Get<uint8_t>().GetPhyAddr() + 64);
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)UB_attenMask.Get<uint8_t>().GetPhyAddr() + 64 * 2);
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)UB_attenMask.Get<uint8_t>().GetPhyAddr() + 64 * 3);
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
                maskUb1, maskUb2, maskUb3, maskUb4, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(scaleValue), minValue, pScale);
        } else {
            FaVectorApi::process_vec1_noupdate_subloop1<float, fp8_e4m3fn_t>(expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb,
                inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb,
                maskUb1, maskUb2, maskUb3, maskUb4, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(scaleValue), minValue, pScale);
        }
    }

    __aicore__ inline void process_vec1_update(LocalTensor<float> mm1ResTensor, int mm1_id, uint32_t subLoop, uint32_t loop, uint32_t mloop, float minValue, const float pScale = 1.0f)
    {
        const uint32_t blockStride = 65;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (loop % 2 == 0)
            ? (__ubuf__ fp8_e4m3fn_t *)UB_stage1OutQue0.Get<fp8_e4m3fn_t>().GetPhyAddr()
            : (__ubuf__ fp8_e4m3fn_t *)UB_stage1OutQue1.Get<fp8_e4m3fn_t>().GetPhyAddr();
        __ubuf__ fp8_e4m3fn_t *expUb2 = expUb1 + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxSum0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxSum1.Get<float>().GetPhyAddr() : UB_softmaxSum2.Get<float>().GetPhyAddr()));
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxSum0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxSum1.Get<float>().GetPhyAddr() : UB_softmaxSum2.Get<float>().GetPhyAddr()));
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr();
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr();
        __ubuf__ float *maxUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxMax0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxMax1.Get<float>().GetPhyAddr() : UB_softmaxMax2.Get<float>().GetPhyAddr()));
        __ubuf__ float *maxUbStart = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxMax0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxMax1.Get<float>().GetPhyAddr() : UB_softmaxMax2.Get<float>().GetPhyAddr()));
        __ubuf__ float *expMaxUb = (__ubuf__ float *)(loop % 3 == 0 ? UB_softmaxExp0.Get<float>().GetPhyAddr() : (loop % 3 == 1 ? UB_softmaxExp1.Get<float>().GetPhyAddr() : UB_softmaxExp2.Get<float>().GetPhyAddr()));
        __ubuf__ float *inMaxUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxMax0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxMax1.Get<float>().GetPhyAddr() : UB_softmaxMax2.Get<float>().GetPhyAddr()));
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr() + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr() + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1ResTensor.GetPhyAddr();//这个看mm1_id需要改吗
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)UB_vselrIndexes0.Get<uint8_t>().GetPhyAddr();

        __ubuf__ float *preLoopMaxUb = (__ubuf__ float *)UB_preLoopMax.Get<float>().GetPhyAddr();
        __ubuf__ float *preLoopSumUb = (__ubuf__ float *)UB_preLoopSum.Get<float>().GetPhyAddr();;
        __ubuf__ float *firstLoopSumUb = (__ubuf__ float *)UB_firstLoopSum.Get<float>().GetPhyAddr();;

        ////printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloopCount, loop);
        ////printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloopCount % 3, loop % 3);

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)UB_attenMask.Get<uint8_t>().GetPhyAddr();
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)UB_attenMask.Get<uint8_t>().GetPhyAddr() + 64);
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)UB_attenMask.Get<uint8_t>().GetPhyAddr() + 64 * 2);
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)UB_attenMask.Get<uint8_t>().GetPhyAddr() + 64 * 3);
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
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(scaleValue), minValue, pScale);
        } else {
            FaVectorApi::process_vec1_update_subloop1<float, fp8_e4m3fn_t>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(scaleValue), minValue, pScale);
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
            offset = (delta < static_cast<int64_t>(256) ? delta : 256) *
                     attenMaskS2Size; // min(delta, s2inner)
        }
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
            LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_head, maskUb, 1, repeatStride, preg_all);
            for (uint16_t i = 1; i < headGLoop; ++i) {
                // StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                //     maskUb, vreg_g_head, 1, repeatStride, preg_all);
                StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_head, 1, repeatStride, preg_all);
            }
        }

        for (uint16_t x = midGLoop; x > 0; x = 0) {     // if (midGLoop > 0) {}
            for (uint16_t i = 0; i < midS1Count; ++i) {
                // LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                //     vreg_g_mid, maskUb, 1, repeatStride, preg_all);
                LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vreg_g_mid, maskUb, 1, repeatStride, preg_all);
                for (uint16_t j = 1; j < midGLoop; ++j) {
                    // StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    //     maskUb, vreg_g_mid, 1, repeatStride, preg_all);
                    StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maskUb, vreg_g_mid, 1, repeatStride, preg_all);
                    }
                // LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
        }

        for (uint16_t x = tailGLoop; x > 1; x = 0) {     // if (tailGLoop > 1) {}
            // LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //     vreg_g_tail, maskUb, 1, repeatStride, preg_all);
            LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_tail, maskUb, 1, repeatStride, preg_all);
            for (uint16_t i = 1; i < tailGLoop; ++i) {
                // StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                //     maskUb, vreg_g_tail, 1, repeatStride, preg_all);
                StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_tail, 1, repeatStride, preg_all);
            }
        }


    }

    __aicore__ inline void AttentionmaskDataCopy_MX(
                                                int32_t s2StartIdx, uint32_t s1StartIdx, uint32_t s1EndIdx, bool isPre = false)
    {
        uint32_t attenMaskSizeAlign = 256;
        uint64_t maskOffset = ComputeAttenMaskOffset_MX(s2StartIdx, s1StartIdx);
        // DataCopyExtParams dataCopyParams;
        // dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
        // dataCopyParams.blockLen = 256;
        // dataCopyParams.srcStride = attenMaskS2Size - 256;
        // dataCopyParams.dstStride = 0;
        // DataCopyPadExtParams<T> padParams{true, 0, 0, 1U};
        LocalTensor<uint8_t> attenMaskTensor = UB_attenMask.Get<uint8_t>();
        GlobalTensor<uint8_t> attenMaskGmTensor;
        attenMaskGmTensor.SetGlobalBuffer((__gm__ uint8_t*)(attenMaskGmInt + maskOffset));
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
        dataCopyParams.blockLen = 256;
        dataCopyParams.srcStride = attenMaskS2Size - 256;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 1U};
        DataCopyPad(attenMaskTensor, attenMaskGmTensor, dataCopyParams, padParams);
    }

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
            LocalTensor<uint8_t> attenMaskTensor = UB_attenMask.Get<uint8_t>()[headGSize * 256];
            GlobalTensor<uint8_t> attenMaskGmTensor;
            attenMaskGmTensor.SetGlobalBuffer((__gm__ uint8_t*)(attenMaskGmInt + maskOffset));

            uint32_t unitOfBytes = 1;
            // uint32_t burstLength = blockLen;
            uint32_t padLength = 0 * 1 + 0 * 1;
            uint64_t srcStride310 = srcStride * unitOfBytes + 256; // GM  DataCopy:32Bytes, DataCopyPad:1Byte
            uint32_t  dstStride310 =
                AlignUp(static_cast<uint32_t>(dstStride) * 32 + 256 + 0, 32);
            uint8_t leftPaddingCnt = 0;
            uint8_t rigntPaddingCnt = 0;
            // copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)attenMaskUB[0] + headGSize * attenMaskS2Stride, (__gm__ uint8_t*)attenMaskGmInt + maskOffset, 0, 2, 256,
            //                          0, 0, true, 0, 1792 + 256 , 2048);
            using PrimType = PrimT<uint8_t>;
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = blockCount;
            dataCopyParams.blockLen = 256;
            dataCopyParams.srcStride = srcStride;
            dataCopyParams.dstStride = dstStride;
            DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 1U};
            DataCopyPad(attenMaskTensor, attenMaskGmTensor, dataCopyParams, padParams);
        }

        set_flag(PIPE_MTE2, PIPE_V, event_t(0));
        wait_flag(PIPE_MTE2, PIPE_V, event_t(0));
        MaskUbCopyS1G_MX((__ubuf__ uint8_t*)UB_attenMask.Get<uint8_t>().GetPhyAddr(), headGSize, gSize, tailGSize, midS1Count);

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



    __aicore__ inline void AttenMaskCopyIn_MX(uint32_t bIdx, uint32_t s2Idx, uint32_t vecMbaseIdx, uint32_t subLoop)
    {
        uint32_t gs1StartIdx = gS1Idx + vecMbaseIdx;
        uint32_t s2StartIdx = (subLoop == 0) ? s2Idx : (s2Idx + 256);
        bool IsSkipMask = IsSkipAttentionmask_MX(gs1StartIdx, s2StartIdx);
        // bool IsSkipMask = IsSkipAttentionmask_MX(maskInfo);
        if (IsSkipMask) {
            // printf("IsSkipMask\n");
            DuplicateFromScalarImpl<uint8_t, Reg::RegTensor<uint8_t, Reg::RegTraitNumOne>>(
                (__ubuf__ uint8_t*)UB_attenMask.Get<uint8_t>().GetPhyAddr(), static_cast<uint8_t>(0U), 16 * 256);
            return;
        } else {
            AttentionmaskCopyInForSgLayout_MX(s2StartIdx, gs1StartIdx);
        }
    }

//subloop1   搬运mask  比 subloop0还快  ，导致数据踩踏
    __aicore__ inline void ProcessVec1(LocalTensor<float8_e4m3_t> p1Tensor, LocalTensor<float> mm1ResTensor, int bmm1_id, int ipuse, uint32_t bIdx, uint32_t s2Idx,
                          uint32_t loop, uint32_t mloop, uint32_t vecMbaseIdx, uint32_t curS2Loop, uint32_t subLoop)
    {
        // bmm1ResBuf.WaitCrossCore();
        int bmm1_idused = 6 + bmm1_id * 2;
        wait_intra_block(PIPE_V, bmm1_idused + 0);

        wait_flag(PIPE_V, PIPE_MTE2, (event_t)4);
        AttenMaskCopyIn_MX(bIdx, s2Idx, vecMbaseIdx, subLoop);

        wait_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);
        if(curS2Loop == FirstS2Loop) {
           process_vec1_noupdate(mm1ResTensor, bmm1_id, subLoop, loop, mloop, negativeFloatScalar);
        } else{
           process_vec1_update(mm1ResTensor, bmm1_id, subLoop, loop, mloop, negativeFloatScalar);
        }
        set_flag(PIPE_V, PIPE_MTE3, (event_t)2);

        set_flag(PIPE_V, PIPE_MTE2, (event_t)4);


        // bmm1ResBuf.SetCrossCore();
        set_intra_block(PIPE_V, bmm1_idused + 1);

        wait_flag(PIPE_V, PIPE_MTE3, (event_t)2);
        ub_l1_vec1(p1Tensor, subLoop, loop % 2);
        set_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);


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
                                        uint32_t actVecMSize, uint32_t vecMbaseIdx, uint32_t mloop, int64_t dSizeAligned64)
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
                __ubuf__ float *maxUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxMax0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxMax1.Get<float>().GetPhyAddr() : UB_softmaxMax2.Get<float>().GetPhyAddr()))+ mStartVec;
                RowInvalidUpdateVF_MX<float>(vec2ResUb, maxUb, actVecMSize, dVBaseSize,
                                          static_cast<uint32_t>(dSizeAligned64), min);
            }
        // }
    }


    __aicore__ inline void Bmm2DataCopyOutTrans(__ubuf__ OUTPUT_T * attenOutUb,
                                                uint32_t vecMIdx, uint32_t bIdx, uint32_t n2Idx,
                                                uint32_t actVecMSize, uint32_t vecMbaseIdx)
    {
        // FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
        //                               .rowCount = dealRowCount,
        //                               .colCount = (uint32_t)(splitD ? constInfo.dBasicBlock : dTemplateAlign64)};
        // GmCoord gmCoord{.bIdx = info.bIdx,
        //                 .n2Idx = info.realN2Idx,
        //                 .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
        //                 .dIdx = 0,
        //                 .gS1DealSize = dealRowCount,
        //                 .dDealSize = (uint32_t)constInfo.dSizeV};

        // CopyAttentionOut(ubTensor, gmCoord);

        uint32_t blockCount = 8;
        uint32_t blockLen = 128;
        uint32_t srcStride = 0;
        uint32_t dstStride = 0;

        uint64_t tIdx = 0;
        if (bIdx == 0) {
            tIdx = 0;
        } else {
            // tIdx = actualSeqLengthsGm.GetValue(bIdx - 1);
            tIdx = actualSeqLengthsGmQ[bIdx -1];
        }

        tIdx += (gS1Idx + vecMbaseIdx + vecMIdx) / 8;
        uint64_t attenOutGmbaseOffset = tIdx * 8 * 8 * 128 + n2Idx * 8 * 128;
        uint64_t gmOffset = attenOutGmbaseOffset;
        uint64_t ubOffset = 0;

        GlobalTensor<OUTPUT_T> outputGmTensor;
        outputGmTensor.SetGlobalBuffer((__gm__ OUTPUT_T*)(attentionOutGm + gmOffset));
        LocalTensor<OUTPUT_T> outputLocalTensor(attenOutUb + ubOffset);
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = srcStride;
        dataCopyParams.dstStride = dstStride;
        DataCopy(outputGmTensor, outputLocalTensor, dataCopyParams);

        gmOffset = attenOutGmbaseOffset + 8 * 8 * 128;
        ubOffset = ((uint64_t)8) * ((uint64_t)128);
        GlobalTensor<OUTPUT_T> outputGmTensor2;
        outputGmTensor2.SetGlobalBuffer((__gm__ OUTPUT_T*)(attentionOutGm + gmOffset));
        LocalTensor<OUTPUT_T> outputLocalTensor2(attenOutUb + ubOffset);
        DataCopy(outputGmTensor2, outputLocalTensor2, dataCopyParams);
    }

    template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
    __simd_vf__ inline void CastIntrinsicsImplVF_MX(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint32_t calCount)
    {
        CastImpl(dst, src, roundMode, calCount);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(__ubuf__ float *vec2ResUb, uint32_t mStartVec, uint32_t bIdx, uint32_t n2Idx,
                                                uint32_t vecMbaseIdx, uint32_t mloop)
    {
        // LocalTensor<OUTPUT_T> attenOut;
        __ubuf__ OUTPUT_T *attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        // attenOut.SetAddr(vec2ResUb.address_);
        // attenOut.SetAddr(vec2ResUb);
        attenOut = (__ubuf__ OUTPUT_T *)vec2ResUb;

        RowInvalid(vec2ResUb, mStartVec, actVecMSize, vecMbaseIdx, mloop, dSizeAligned64);
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
        Bmm2DataCopyOutTrans(attenOut, mStartVec, bIdx, n2Idx, actVecMSize, vecMbaseIdx);
    }

    __aicore__ inline void ProcessVec2(LocalTensor<float> mm2ResTensor, int bmm2_id, uint32_t loop, uint32_t mloop, uint32_t bIdx, uint32_t n2Idx,
                                        uint32_t vecMbaseIdx, uint32_t curS2Loop)
    {
        // bmm2ResBuf.WaitCrossCore();
        wait_intra_block(PIPE_V, bmm2_id);
        // if (unlikely(runInfo.actVecMSize == 0)) {
        //     bmm2ResBuf.SetCrossCore();
        //     return;
        // }

        int64_t vec2CalcSize = actVecMSize * 128;
        float deSCaleVValue = 1.0f;

        // LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        // LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        // WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);

        __ubuf__ float * dstUb = (__ubuf__ T*)UB_stage2OutBuf.Get<T>().GetPhyAddr();
        __ubuf__ float * curUb = (__ubuf__ T*)mm2ResTensor.GetPhyAddr();
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0);

        // set_mark(0x89838990);
        // // set_mark(runInfo.isFirstS2Loop);
        // set_mark(inner);
        if (curS2Loop == FirstS2Loop) {
            // DataCopy(vec2ResUb, mmRes, vec2CalcSize);
            using PrimType = PrimT<T>;
            struct DataCopyParams repeatParams;
            repeatParams.blockLen = vec2CalcSize / AscendCUtils::GetC0Count(sizeof(PrimType));
            LocalTensor<T> dstLocal = UB_stage2OutBuf.Get<T>();
            LocalTensor<T> srcLocal((__ubuf__ T*)mm2ResTensor.GetPhyAddr());
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = 256;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopy(dstLocal, srcLocal, dataCopyParams);

            // printf("repeatParams.blockCount = %d, repeatParams.blockLen =%d,repeatParams.srcStride =%d,repeatParams.dstStride =%d\n",repeatParams.blockCount,repeatParams.blockLen,repeatParams.srcStride,repeatParams.dstStride);
            // copy_ubuf_to_ubuf((__ubuf__ void*)stage2OutBuf, (__ubuf__ void*)mm2Res, 0, repeatParams.blockCount, repeatParams.blockLen, repeatParams.srcStride, repeatParams.dstStride);
            // copy_ubuf_to_ubuf((__ubuf__ void*)stage2OutBuf, (__ubuf__ void*)mm2Res, 0, 1, vec2CalcSize, 0, 0);
        } else {
            // // LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            // // LocalTensor<T> pScaleUb;

            __ubuf__ float *expMaxUb = (__ubuf__ float *)(loop % 3 == 0 ? UB_softmaxExp0.Get<float>().GetPhyAddr() : (loop % 3 == 1 ? UB_softmaxExp1.Get<float>().GetPhyAddr() : UB_softmaxExp2.Get<float>().GetPhyAddr()));

            float deSCalePreVValue = 1.0f;
            if ((curS2Loop + 1 )!= LastS2Loop) {//修改
                __ubuf__ float * preUb = (__ubuf__ T*)UB_stage2OutBuf.Get<T>().GetPhyAddr();
                __ubuf__ float * rowMaxUb = (__ubuf__ T*)UB_pScale.Get<T>().GetPhyAddr();
                FaVectorApi::FlashUpdateBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, rowMaxUb, actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                __ubuf__ float * preUb = (__ubuf__ T*)UB_stage2OutBuf.Get<T>().GetPhyAddr();
                __ubuf__ float *expSumUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxSum0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxSum1.Get<float>().GetPhyAddr() : UB_softmaxSum2.Get<float>().GetPhyAddr()));
                __ubuf__ float * rowMaxUb = (__ubuf__ T*)UB_pScale.Get<T>().GetPhyAddr();
                FaVectorApi::FlashUpdateLastBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, expSumUb, rowMaxUb, actVecMSize, dTemplateAlign64, 1.0,
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
                __ubuf__ float * srcUb = (__ubuf__ T*)UB_stage2OutBuf.Get<T>().GetPhyAddr();
                __ubuf__ float *expSumUb = (__ubuf__ float *)(mloop % 3 == 0 ? UB_softmaxSum0.Get<float>().GetPhyAddr() : (mloop % 3 == 1 ? UB_softmaxSum1.Get<float>().GetPhyAddr() : UB_softmaxSum2.Get<float>().GetPhyAddr()));
                FaVectorApi::LastDivNewVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    dstUb, srcUb, expSumUb, actVecMSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            Bmm2ResCastAndCopyOut(dstUb, 0, bIdx, n2Idx, vecMbaseIdx, mloop);
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

        tPipePtr->InitBuffer(UB_stage1OutQue0, (64+1)*256);
        tPipePtr->InitBuffer(UB_stage1OutQue1, (64+1)*256);

        tPipePtr->InitBuffer(UB_attenMask, 64*256);


        tPipePtr->InitBuffer(UB_softmaxSum0, 256);
        tPipePtr->InitBuffer(UB_softmaxSum1, 256);
        tPipePtr->InitBuffer(UB_softmaxSum2, 256);
        tPipePtr->InitBuffer(UB_softmaxMax0, 256);
        tPipePtr->InitBuffer(UB_softmaxMax1, 256);
        tPipePtr->InitBuffer(UB_softmaxMax2, 256);
        tPipePtr->InitBuffer(UB_softmaxExp0, 256);
        tPipePtr->InitBuffer(UB_softmaxExp1, 256);
        tPipePtr->InitBuffer(UB_softmaxExp2, 256);
        tPipePtr->InitBuffer(UB_preLoopMax, 256);
        tPipePtr->InitBuffer(UB_preLoopSum, 256);
        tPipePtr->InitBuffer(UB_firstLoopSum, 256);
        tPipePtr->InitBuffer(UB_common, 512);

        tPipePtr->InitBuffer(UB_stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));

        tPipePtr->InitBuffer(UB_vselrIndexes0, 128);
        tPipePtr->InitBuffer(UB_vselrIndexes1, 64);

        // constexpr auto orderMode = std::integral_constant<::Order, static_cast<::Order>(order)>();
        // vci((__ubuf__ uint8_t *)vselrIndexesUB[0], 2, orderMode);
        // vselrIndexesUB[0] = 0x7f;
        // 先获取 UB 内存地址（uint8 指针）
        // __ubuf__ uint8_t *ub0 = (__ubuf__ uint8_t *)vselrIndexesUB[0];
        // __ubuf__ uint8_t *ub1 = (__ubuf__ uint8_t *)vselrIndexesUB[1];

        LocalTensor<uint8_t> vselrIndexes0Tensor = UB_vselrIndexes0.Get<uint8_t>();
        LocalTensor<uint8_t> vselrIndexes1Tensor = UB_vselrIndexes1.Get<uint8_t>();
        for (int i = 0; i < 128; i++) {
            vselrIndexes0Tensor.SetValue(i, (uint8_t)(i * 2));
        }
        for (int i = 0; i < 64; i++) {
            vselrIndexes1Tensor.SetValue(i, (uint8_t)(i * 4));
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

        set_flag(PIPE_V, PIPE_MTE2, (event_t)4);
        // set_flag(PIPE_MTE3, PIPE_V, (event_t)5);
    }

    __aicore__ inline void FreeEventID()
    {
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)1);

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)2);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)3);

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
    __aicore__ inline void SetTPipe(TPipe&) {}
};
} // namespace BaseApi
#endif // FIA_BLOCK_VEC_FULLQUANT_MX_H_
