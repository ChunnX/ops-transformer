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
#ifndef FIA_BLOCK_VEC_FULLQUANT_MX_MX_H_
#define FIA_BLOCK_VEC_FULLQUANT_MX_MX_H_

#include "kernel_operator.h"

#include "flash_attention_score_common_regbase.h"
#include "adv_api/activation/softmax.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_mx.h"
#include "vf/vf_flashupdate_new_mx.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flash_decode.h"
#include "fia_public_define_arch35.h"
#include "../vector_common.h"
#include "../memory_copy_arch35.h"

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

    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    const ConstInfoX &constInfo;

    __gm__ uint64_t *actualSeqLengthsGmQ;
    TPipe *tPipePtr = nullptr;
    __aicore__ inline void SetTPipe(TPipe& pipe) { tPipePtr = &pipe; }
    TBuf<TPosition::A1> L1_P;
    TBuf<TPosition::VECIN> UB_MM2Res;
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
    TBuf<TPosition::VECIN> UB_pScale;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

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

    __aicore__ inline FAFullQuantMxBlockVec(ConstInfoX &constInfo) : constInfo(constInfo){};
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
    __aicore__ inline void ub_l1_vec1(uint32_t subLoop, uint32_t createdTaskCount)
    {
        // block_count = 256 / 32;
        // block_len = (uint16_t)16;
        // srcGap = (uint16_t)(65 - 16);
        // dstGap = (uint16_t)(128 - 16);
        int64_t dstOffset =
                constInfo.subBlockIdx * 32 * 16 +
                256 * 128 * subLoop;
        //set_mark(0x11223390);
        //set_mark(dstOffset);
        constexpr uint32_t stage1OutQueSize = 16640;
        LocalTensor<float8_e4m3_t> dstTensor = L1_P.Get<float8_e4m3_t>()[dstOffset];
        LocalTensor<float8_e4m3_t> srcTensor = UB_stage1OutQue0.Get<float8_e4m3_t>()[createdTaskCount * stage1OutQueSize];
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 8;
        dataCopyParams.blockLen = 16;
        dataCopyParams.srcStride = 49;
        dataCopyParams.dstStride = 112;
        DataCopy(dstTensor, srcTensor, dataCopyParams);

    }

    __aicore__ inline void process_vec1_noupdate(LocalTensor<float> mm1ResTensor, int mm1_id, uint32_t subLoop, uint32_t createdTaskCount, uint32_t mloopCount, float minValue, const float pScale = 1.0f)
    {
        const uint32_t blockStride = 65;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)UB_stage1OutQue0.Get<fp8_e4m3fn_t>().GetPhyAddr() + createdTaskCount % 2 * 16640;
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)UB_stage1OutQue0.Get<fp8_e4m3fn_t>().GetPhyAddr() + createdTaskCount % 2 * 16640 + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)UB_softmaxSum0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)UB_softmaxSum0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr();
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr();
        __ubuf__ float *maxUb = (__ubuf__ float *)UB_softmaxMax0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *maxUbStart = (__ubuf__ float *)UB_softmaxMax0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *expMaxUb = (__ubuf__ float *)UB_softmaxExp0.Get<float>().GetPhyAddr() + createdTaskCount % 3 * 256;
        __ubuf__ float *inMaxUb = (__ubuf__ float *)UB_softmaxMax0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr() + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr() + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1ResTensor.GetPhyAddr();
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)UB_vselrIndexes0.Get<uint8_t>().GetPhyAddr();

        ////printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloopCount, createdTaskCount);
        ////printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloopCount % 3, createdTaskCount % 3);

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
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(constInfo.scaleValue), minValue, pScale);
        } else {
            FaVectorApi::process_vec1_noupdate_subloop1<float, fp8_e4m3fn_t>(expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb,
                inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb,
                maskUb1, maskUb2, maskUb3, maskUb4, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(constInfo.scaleValue), minValue, pScale);
        }
    }

    __aicore__ inline void process_vec1_update(LocalTensor<float> mm1ResTensor, int mm1_id, uint32_t subLoop, uint32_t createdTaskCount, uint32_t mloopCount, float minValue, const float pScale = 1.0f)
    {
        const uint32_t blockStride = 65;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)UB_stage1OutQue0.Get<fp8_e4m3fn_t>().GetPhyAddr() + createdTaskCount % 2 * 16640;
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)UB_stage1OutQue0.Get<fp8_e4m3fn_t>().GetPhyAddr() + createdTaskCount % 2 * 16640 + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)UB_softmaxSum0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)UB_softmaxSum0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr();
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr();
        __ubuf__ float *maxUb = (__ubuf__ float *)UB_softmaxMax0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *maxUbStart = (__ubuf__ float *)UB_softmaxMax0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *expMaxUb = (__ubuf__ float *)UB_softmaxExp0.Get<float>().GetPhyAddr() + createdTaskCount % 3 * 256;
        __ubuf__ float *inMaxUb = (__ubuf__ float *)UB_softmaxMax0.Get<float>().GetPhyAddr() + mloopCount % 3 * 256;
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr() + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)UB_common.Get<float>().GetPhyAddr() + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1ResTensor.GetPhyAddr();
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)UB_vselrIndexes0.Get<uint8_t>().GetPhyAddr();

        __ubuf__ float *preLoopMaxUb = (__ubuf__ float *)UB_preLoopMax.Get<float>().GetPhyAddr();
        __ubuf__ float *preLoopSumUb = (__ubuf__ float *)UB_preLoopSum.Get<float>().GetPhyAddr();;
        __ubuf__ float *firstLoopSumUb = (__ubuf__ float *)UB_firstLoopSum.Get<float>().GetPhyAddr();;

        ////printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloopCount, createdTaskCount);
        ////printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloopCount % 3, createdTaskCount % 3);

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
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(constInfo.scaleValue), minValue, pScale);
        } else {
            FaVectorApi::process_vec1_update_subloop1<float, fp8_e4m3fn_t>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, 16, static_cast<float>(constInfo.scaleValue), minValue, pScale);
        }
    }

    __aicore__ inline bool IsSkipAttentionmask_MX(MaskInfo &info)
    {
        // int32_t s1StartIdx = gs1StartIdx / 8;
        // int64_t nextToken = 
        //         static_cast<int64_t>(10240) - static_cast<int64_t>(4); // 统一以左上角为原点计算Token
        
        // if (static_cast<int64_t>(s2StartIdx + 256) <= static_cast<int64_t>(s1StartIdx) + 10236) {
        //     return true;
        // }
        int32_t s1StartIdx = info.gs1StartIdx / info.gSize;
        int64_t nextToken = 
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
        if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            return true;
        }
        return false;
    }

    // __aicore__ inline uint64_t ComputeAttenMaskOffset(uint32_t s1StartIdx, uint32_t s2StartIdx)
    // {
    //     int64_t nextToken = 0; // sparse2 本身原点就是左上角
    //     // nextToken =
    //     //     static_cast<int64_t>(10240) - static_cast<int64_t>(4); // 统一以左上角为原点计算token
         
    //     uint64_t offset = 0;
    //     int64_t delta = 10236 + s1StartIdx - s2StartIdx;
    //     uint32_t attenMaskSizeAlign = 256;

    //     offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
    //                 2048; // min(delta, s2inner)
        
    //     return offset;
    // }
        __aicore__ inline uint64_t ComputeAttenMaskOffset_MX(MaskInfo &info, uint32_t s1StartIdx)
    {
        int64_t nextToken = 0; // sparse2 本身原点就是左上角
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
        
        uint64_t offset = 0;
        int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
        uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
        if (delta < 0) {
            offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
        } else {
            offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                     info.attenMaskS1Stride; // min(delta, s2inner)
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
            LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_head, maskUb, 1, repeatStride, preg_all);
            for (uint16_t i = 1; i < headGLoop; ++i) {
                StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_head, 1, repeatStride, preg_all);
            }
        }

        for (uint16_t x = midGLoop; x > 0; x = 0) {     // if (midGLoop > 0) {}
            for (uint16_t i = 0; i < midS1Count; ++i) {
                LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vreg_g_mid, maskUb, 1, repeatStride, preg_all);
                for (uint16_t j = 1; j < midGLoop; ++j) {
                    StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maskUb, vreg_g_mid, 1, repeatStride, preg_all);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
        }

        for (uint16_t x = tailGLoop; x > 1; x = 0) {     // if (tailGLoop > 1) {}
            LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_tail, maskUb, 1, repeatStride, preg_all);
            for (uint16_t i = 1; i < tailGLoop; ++i) {
                StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_tail, 1, repeatStride, preg_all);
            }
        }


    }


    // __aicore__ inline void AttentionmaskCopyInForSgLayout(uint32_t gs1StartIdx, uint32_t s2StartIdx)
    // {

    //     uint32_t s1StartIdx = gs1StartIdx / 8;
    //     uint32_t s1EndIdx = (gs1StartIdx + 16 - 1) / 8;
    //     uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    //     uint32_t headGSize = 16;
    //     if (s1Count > 1) {
    //         headGSize = (gs1StartIdx % 8 == 0) ? 0 : (8 - gs1StartIdx % 8);
    //     }
    //     uint32_t remainRowCount = 16 - headGSize;
    //     uint32_t midS1Count = remainRowCount / 8;
    //     uint32_t tailGSize = remainRowCount % 8;
    //     uint32_t attenMaskSizeAlign = 256;
    //     uint32_t attenMaskS2Stride = attenMaskSizeAlign;

    //     // // ub-head
    //     // if (headGSize > 0) {
    //     //     //printf("headGSize > 0 = %d\n",headGSize);
    //     //     AttentionmaskDataCopy(attenMaskUB, srcGmAddr, info, s1StartIdx, s1StartIdx + 1, isPre);
    //     //     s1StartIdx++;
    //     // }

    //     // // ub-remain
    //     if (remainRowCount > 0) {
    //         uint64_t maskOffset = ComputeAttenMaskOffset(s1StartIdx, s2StartIdx);
    //         DataCopyExtParams dataCopyParams;
    //         dataCopyParams.blockCount = 2;
    //         dataCopyParams.blockLen = 256;
    //         dataCopyParams.srcStride = 1792;
    //         dataCopyParams.dstStride = 56;
    //         // blockLen + padParams.len + dstStride = gSize * s2Size
    //         DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 1U};//isPad,leftPadding,rightPadding,paddingValue
    //         set_mov_pad_val(GetScalarBitcodeValue(1U));
    //         // DataCopyPad(attenMaskUB[headGSize * attenMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);
    //         copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)attenMaskUB[0] + headGSize * attenMaskS2Stride, (__gm__ uint8_t*)attenMaskGmInt + maskOffset, 0, 2, 256,
    //                                  0, 0, true, 0, 1792 + 256 , 2048);
    //     }
    //     set_flag(PIPE_MTE2, PIPE_V, event_t(0));
    //     wait_flag(PIPE_MTE2, PIPE_V, event_t(0));
    //     MaskUbCopyS1G((__ubuf__ uint8_t*)attenMaskUB[0], tailGSize, midS1Count);
    // }
    __aicore__ inline void AttentionmaskCopyInForSgLayout_MX(MaskInfo &info)
    {
        uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
        uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
        uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
        uint32_t headGSize = info.gs1dealNum;
        if (s1Count > 1) {
            headGSize = (info.gs1StartIdx % info.gSize == 0) ? 0 : (info.gSize - info.gs1StartIdx % info.gSize);
        }
        uint32_t remainRowCount = info.gs1dealNum - headGSize;
        uint32_t midS1Count = remainRowCount / info.gSize;
        uint32_t tailGSize = remainRowCount % info.gSize;
        uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
        uint32_t attenMaskS2Stride = attenMaskSizeAlign + 32 * info.attenMaskDstStride;
    
        // // ub-head
        // if (headGSize > 0) {
        //     AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1StartIdx + 1, isPre);
        //     s1StartIdx++;
        // }
    
        // ub-remain
        if (remainRowCount > 0) {
            uint64_t maskOffset = ComputeAttenMaskOffset_MX(info, s1StartIdx);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = midS1Count + (tailGSize > 0);
            dataCopyParams.blockLen = info.s2dealNum;
            dataCopyParams.srcStride = info.attenMaskS1Stride - info.s2dealNum;
            dataCopyParams.dstStride = (info.gSize - 1) * attenMaskS2Stride / 32 + info.attenMaskDstStride;

            // printf("midS1Count = %d, tailGSize = %d, dataCopyParams.blockCount = %d\n",midS1Count,tailGSize,dataCopyParams.blockCount);//2,0,2
            // printf("info.s2dealNum = %d, dataCopyParams.blockLen = %d\n",info.s2dealNum,dataCopyParams.blockLen);//256.256
            // printf("info.attenMaskS1Stride = %d, info.s2dealNum = %d, dataCopyParams.srcStride = %d\n",
            //         info.attenMaskS1Stride,info.s2dealNum,dataCopyParams.srcStride);//2048,256,1792
            // printf("(info.gSize - 1) * attenMaskS2Stride / 32 = %d, info.attenMaskDstStride = %d, dataCopyParams.dstStride = %d\n",
            //             (info.gSize - 1) * attenMaskS2Stride / 32,info.attenMaskDstStride, dataCopyParams.dstStride);//56,0,56
            // printf("headGSize = %d, attenMaskS2Stride = %d, maskOffset = %d\n",headGSize,attenMaskS2Stride,maskOffset);//0,256,516096/50192

            // // blockLen + padParams.len + dstStride = gSize * s2Size
            // DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};
            // DataCopyPad(attenMaskUb[headGSize * attenMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);

             // blockLen + padParams.len + dstStride = gSize * s2Size
            DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 1U};//isPad,leftPadding,rightPadding,paddingValue
            LocalTensor<uint8_t> attenMaskTensor = UB_attenMask.Get<uint8_t>()[headGSize * attenMaskS2Stride];
            GlobalTensor<uint8_t> attenMaskGmTensor;
            attenMaskGmTensor.SetGlobalBuffer((__gm__ uint8_t*)(attenMaskGmInt + maskOffset));
            DataCopyPad(attenMaskTensor, attenMaskGmTensor, dataCopyParams, padParams);
        }
    
        set_flag(PIPE_MTE2, PIPE_V, event_t(0));
        wait_flag(PIPE_MTE2, PIPE_V, event_t(0));
        MaskUbCopyS1G_MX((__ubuf__ uint8_t*)UB_attenMask.Get<uint8_t>().GetPhyAddr(), headGSize, info.gSize, tailGSize, midS1Count);
    
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


    // __aicore__ inline void AttenMaskCopyIn(uint64_t inner, uint32_t subLoop)
    // {
    //     uint64_t gs1StartIdx = 0;
    //     if (constInfo.subBlockIdx == 1) {
    //         gs1StartIdx = 16;
    //     }
    //     uint64_t s2StartIdx = (subLoop == 0) ? inner * 512 : (inner * 512 + 256);
    //     bool IsSkipMask = IsSkipAttentionmask(gs1StartIdx, s2StartIdx);
    //     if (IsSkipMask) {
    //         // Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
    //         DuplicateFromScalarImpl<uint8_t, Reg::RegTensor<uint8_t, Reg::RegTraitNumOne>>(
    //             (__ubuf__ uint8_t*)attenMaskUB[0], static_cast<uint8_t>(0U), 16 * 256);
    //         return;
    //     } else {
    //         AttentionmaskCopyInForSgLayout(gs1StartIdx, s2StartIdx);
    //     }
    // }

    __aicore__ inline void AttenMaskCopyIn_MX(RunInfoX &runInfo, uint32_t subLoop)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = 512 >> 1;
        if (runInfo.actSingleLoopS2Size > 256) {
            s2RealSize = subLoop == 0 ? 256 : runInfo.actSingleLoopS2Size - 256;
        }

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        maskInfo.gs1dealNum = runInfo.actVecMSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.realGSize;
        maskInfo.s2StartIdx = (subLoop == 0) ? runInfo.s2Idx : (runInfo.s2Idx + s2BaseSizeCur);
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSizeCur;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSizeCur - Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.layout = LAYOUT_Q::SG;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask_MX(maskInfo);
        // bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);
        if (IsSkipMask) {
            // printf("IsSkipMask\n");
            LocalTensor<uint8_t> attenMaskTensor = UB_attenMask.Get<uint8_t>();
            Duplicate(attenMaskTensor, static_cast<uint8_t>(0U), 16 * 256);
            return;
        } else {
            AttentionmaskCopyInForSgLayout_MX(maskInfo);
        }

        // if (!IsSkipMask) {
        //     AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSizeCur>(attenMaskUb, attenMaskGmInt, maskInfo);
        // } else {
        //     Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
        // }

        // if (!IsSkipMaskForPre) {
        //     LocalTensor<uint8_t> attenMaskUbPre = this->attenMaskInQue[0].template AllocTensor<uint8_t>();
        //     AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSizeCur>(attenMaskUbPre, attenMaskGmInt,
        //         maskInfo, true);
        //     MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSizeCur);
        //     this->attenMaskInQue[0].template FreeTensor(attenMaskUbPre);
        // }
    }

//subloop1   搬运mask  比 subloop0还快  ，导致数据踩踏
    __aicore__ inline void ProcessVec1(LocalTensor<float> mm1ResTensor, int bmm1_id, int ipuse,
                         uint32_t inner, uint32_t loop, uint32_t mloop, uint32_t subLoop)
    {
        // bmm1ResBuf.WaitCrossCore();
        int bmm1_idused = 6 + bmm1_id * 2;
        wait_intra_block(PIPE_V, bmm1_idused + 0);

        wait_flag(PIPE_V, PIPE_MTE2, (event_t)4);
        // AttenMaskCopyIn_MX(runInfo, subLoop);

        wait_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);
        if(inner == 0) {
           process_vec1_noupdate(mm1ResTensor, bmm1_id, subLoop, loop, mloop, negativeFloatScalar);
        } else{
           process_vec1_update(mm1ResTensor, bmm1_id, subLoop, loop, mloop, negativeFloatScalar);
        }
        set_flag(PIPE_V, PIPE_MTE3, (event_t)2);

        set_flag(PIPE_V, PIPE_MTE2, (event_t)4);


        // bmm1ResBuf.SetCrossCore();
        set_intra_block(PIPE_V, bmm1_idused + 1);

        wait_flag(PIPE_V, PIPE_MTE3, (event_t)2);
        ub_l1_vec1(subLoop, loop % 2);
        set_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);


        if (subLoop % 2 == 1) {
            // outputBuf.SetCrossCore();//L1
            set_intra_block(PIPE_MTE3, ipuse);
        }

        // if (unlikely(runInfo.isLastS2Loop)) {
        //     SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        // }
    }


    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo.realGSize;
            s1EndTdx = vecMEndIdx / constInfo.realGSize;
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

    // template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(__ubuf__ float *vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken = Min(Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = Min(Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = constInfo.isRowInvalidOpen || // 手动开启行无效
                                       ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
            blockNeedRowInvalid = blockNeedRowInvalid || constInfo.isRowInvalidOpen;

            if (blockNeedRowInvalid) {
                uint32_t min = 0xFF7FFFFF; // min value of float
                // if constexpr (USE_DN) {
                //     UpdateMinCheckValueForDn(min);
                // }
                // LocalTensor<float> maxTensor =
                //     softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
                __ubuf__ float *maxUb = (__ubuf__ float *)UB_softmaxMax0.Get<float>().GetPhyAddr() + runInfo.mloop % 3 * 256 + mStartVec;
                // if constexpr (!POST_QUANT) {
                    RowInvalidUpdateVF_MX<float>(vec2ResUb, maxUb, mDealSize, constInfo.dSizeV,
                                              static_cast<uint32_t>(dSizeAligned64), min);
                // } else {
                //     uint32_t dStride =
                //         CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                //     uint16_t dSize = CeilDiv(constInfo.dSizeV, sizeof(float)); // w8后量化后的处理长度
                //     RowInvalidUpdateVF_MX<float>(*((LocalTensor<float> *)&vec2ResUb), maxUb, mDealSize, dSize,
                //                               dStride, min);
                // }
            }
        }
    }
    
    // template <typename T>
    // __aicore__ inline uint8_t ExtractCacheMode(__gm__ T* addr) {
    //     return static_cast<uint8_t>(((uint64_t)addr) >> L2_CACHE_OFFSET);
    // }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info,__ubuf__ OUTPUT_T * attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
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
        if (info.bIdx == 0) {
            tIdx = 0;
        } else {
            // tIdx = actualSeqLengthsGm.GetValue(bIdx - 1);
            tIdx = actualSeqLengthsGmQ[info.bIdx -1];
        }

        tIdx += (info.gS1Idx + info.vecMbaseIdx + vecMIdx) / 8;
        uint64_t attenOutGmbaseOffset = tIdx * 8 * 8 * 128 + info.realN2Idx * 8 * 128;
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
        LocalTensor<OUTPUT_T> attenOutTensor = UB_stage2OutBuf.Get<OUTPUT_T>();
        GlobalTensor<OUTPUT_T> attenOutGmTensor;
        attenOutGmTensor.SetGlobalBuffer((__gm__ OUTPUT_T*)(attentionOutGm + gmOffset));
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = srcStride;
        dataCopyParams.dstStride = dstStride;
        DataCopy(attenOutTensor, attenOutGmTensor, dataCopyParams);

        gmOffset = attenOutGmbaseOffset + 8 * 8 * 128;
        ubOffset = ((uint64_t)8) * ((uint64_t)128);
        GlobalTensor<OUTPUT_T> attenOutGmTensor2;
        attenOutGmTensor2.SetGlobalBuffer((__gm__ OUTPUT_T*)(attentionOutGm + gmOffset));
        LocalTensor<OUTPUT_T> attenOutTensor2 = UB_stage2OutBuf.Get<OUTPUT_T>()[ubOffset];
        DataCopy(attenOutTensor2, attenOutGmTensor2, dataCopyParams);
    }


    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, __ubuf__ float *vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        // LocalTensor<OUTPUT_T> attenOut;
        __ubuf__ OUTPUT_T *attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        // attenOut.SetAddr(vec2ResUb.address_);
        // attenOut.SetAddr(vec2ResUb);
        attenOut = (__ubuf__ OUTPUT_T *)vec2ResUb;

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        // Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        using DstPrimType = PrimT<OUTPUT_T>;
        using SrcPrimType = PrimT<float>;
        CastImpl((__ubuf__ DstPrimType*)attenOut, (__ubuf__ SrcPrimType*)vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
           
        // SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        // WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    __aicore__ inline void ProcessVec2(LocalTensor<float> mm2ResTensor, int bmm2_id, RunInfoX runInfo)
    {
        // bmm2ResBuf.WaitCrossCore();
        wait_intra_block(PIPE_V, bmm2_id);
        // if (unlikely(runInfo.actVecMSize == 0)) {
        //     bmm2ResBuf.SetCrossCore();
        //     return;
        // }

        int64_t vec2CalcSize = runInfo.actVecMSize * 128;
        float deSCaleVValue = 1.0f;

        // LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        // LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        // WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);

        __ubuf__ float * dstUb = (__ubuf__ float *)UB_stage2OutBuf.Get<float>().GetPhyAddr();
        __ubuf__ float * curUb = (__ubuf__ float *)mm2ResTensor.GetPhyAddr();
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0);


        if (unlikely(runInfo.isFirstS2Loop)) {
            // DataCopy(vec2ResUb, mmRes, vec2CalcSize);
            LocalTensor<T> stage2OutTensor = UB_stage2OutBuf.Get<T>();
            DataCopy(stage2OutTensor, mm2ResTensor, vec2CalcSize);
        } else {
            // // LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            // // LocalTensor<T> pScaleUb;

            __ubuf__ float *expMaxUb = (__ubuf__ float *)UB_softmaxExp0.Get<float>().GetPhyAddr() + runInfo.loop % 3 * 256;

            float deSCalePreVValue = 1.0f;
            if (!runInfo.isLastS2Loop) {
                __ubuf__ float * preUb = (__ubuf__ float *)UB_stage2OutBuf.Get<float>().GetPhyAddr();
                __ubuf__ float * rowMaxUb = (__ubuf__ float *)UB_pScale.Get<float>().GetPhyAddr();
                FaVectorApi::FlashUpdateBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, rowMaxUb, runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                __ubuf__ float * preUb = (__ubuf__ float *)UB_stage2OutBuf.Get<float>().GetPhyAddr();
                __ubuf__ float *expSumUb = (__ubuf__ float *)UB_softmaxSum0.Get<float>().GetPhyAddr() + runInfo.mloop % 3 * 256;
                __ubuf__ float * rowMaxUb = (__ubuf__ float *)UB_pScale.Get<float>().GetPhyAddr();
                FaVectorApi::FlashUpdateLastBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, expSumUb, rowMaxUb, runInfo.actVecMSize, dTemplateAlign64, 1.0,
                    1.0);
            }
        }
        // bmm2ResBuf.SetCrossCore();
        // set_intra_block(PIPE_V, bmm2_id + 1);
        set_intra_block(PIPE_V, bmm2_id + 1);
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                // LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                __ubuf__ float * srcUb = (__ubuf__ float *)UB_stage2OutBuf.Get<float>().GetPhyAddr();
                __ubuf__ float *expSumUb = (__ubuf__ float *)UB_softmaxSum0.Get<float>().GetPhyAddr() + runInfo.mloop % 3 * 256;
                FaVectorApi::LastDivNewVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    dstUb, srcUb, expSumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            Bmm2ResCastAndCopyOut(runInfo, dstUb, 0, runInfo.actVecMSize);
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

        constexpr uint32_t mm2LeftSize = 128 * 512;
        tPipePtr->InitBuffer(L1_P, mm2LeftSize * 3);
        tPipePtr->InitBuffer(UB_MM2Res, mm2ResultSize);
        tPipePtr->InitBuffer(UB_stage1OutQue0, stage1OutSize);
        tPipePtr->InitBuffer(UB_stage1OutQue1, stage1OutSize);
        tPipePtr->InitBuffer(UB_attenMask, maskSize);
        tPipePtr->InitBuffer(UB_softmaxSum0, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_softmaxSum1, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_softmaxSum2, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_softmaxMax0, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_softmaxMax1, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_softmaxMax2, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_softmaxExp0, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_softmaxExp1, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_softmaxExp2, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_preLoopMax, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_preLoopSum, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_firstLoopSum, 256 * sizeof(float));
        tPipePtr->InitBuffer(UB_common, 512 * sizeof(float));
        tPipePtr->InitBuffer(UB_stage2OutBuf, 64 * 128 * sizeof(float));
        tPipePtr->InitBuffer(UB_vselrIndexes0, 128);
        tPipePtr->InitBuffer(UB_vselrIndexes1, 128);
        tPipePtr->InitBuffer(UB_pScale, 256 * sizeof(float));

        LocalTensor<uint8_t> vselrIndexes0Tensor = UB_vselrIndexes0.Get<uint8_t>();
        LocalTensor<uint8_t> vselrIndexes1Tensor = UB_vselrIndexes1.Get<uint8_t>();
        for (int i = 0; i < 128; i++) {
            vselrIndexes0Tensor[i] = i * 2;
        }
        for (int i = 0; i < 64; i++) {
            vselrIndexes1Tensor[i] = i * 4;
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
};
} // namespace BaseApi
#endif // FIA_BLOCK_VEC_FULLQUANT_MX_H_