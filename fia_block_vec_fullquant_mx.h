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
#ifndef FIA_BLOCK_VEC_FULLQUANT_MX_PA_WHILE_LOOP_P_L1_H_
#define FIA_BLOCK_VEC_FULLQUANT_MX_PA_WHILE_LOOP_P_L1_H_

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
    // static constexpr uint32_t gS1Idx = (uint32_t)0;
    static constexpr uint32_t FirstS2Loop = (uint32_t)0;//修改
    static constexpr uint32_t LastS2Loop = (uint32_t)20;//修改
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    uint8_t subBlockIdx = GetSubBlockIdx();
    float pScaleValue { 1.0f };
    
    __gm__ uint64_t *actualSeqLengthsGmQ;
    __gm__ float *pScaleGm;
    TBuf<TPosition::VECIN> UB_stage1OutQue0;
    TBuf<TPosition::VECIN> UB_stage1OutQue1;
    TBuf<TPosition::VECIN> UB_pScaleSubLoop0;

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
    __aicore__ inline void ub_l1_vec1_pscale(LocalTensor<float8_e4m3_t> p1Tensor, uint32_t subLoop, uint32_t createdTaskCount)
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
            LocalTensor<fp8_e8m0_t> dstTensor1 = p1Tensor.template ReinterpretCast<fp8_e8m0_t>()[vecOffset + i * 32]; // 感觉要cast
            LocalTensor<fp8_e8m0_t> srcTensor1 = UB_pScaleSubLoop0.Get<fp8_e8m0_t>();

            LocalTensor<fp8_e8m0_t> dstTensor2 = p1Tensor.template ReinterpretCast<fp8_e8m0_t>()[subLoopOffset + vecOffset + i * 32]; // 感觉要cast
            LocalTensor<fp8_e8m0_t> srcTensor2 = UB_pScaleSubLoop0.Get<fp8_e8m0_t>()[128];
            DataCopy(dstTensor1, srcTensor1, {copyCount, 1, 0, dstStride});
            DataCopy(dstTensor2, srcTensor2, {copyCount, 1, 0, dstStride});
        }
    }

    __aicore__ inline void ub_l1_vec1(LocalTensor<float8_e4m3_t> p1Tensor, uint32_t subLoop, uint32_t createdTaskCount)
    {
        // block_count = 256 / 32;
        // block_len = (uint16_t)16;//(uint16_t)actMSizeAlign16
        // srcGap = (uint16_t)(65 - 16);//(uint16_t)(vec1Srcstride - actMSizeAlign16)
        // dstGap = (uint16_t)(128 - 16);//(uint16_t)(128 - actMSizeAlign16)
        int64_t dstOffset =
                subBlockIdx * 32 * actMSizeAlign16 +
                256 * mBaseSize * subLoop;
        LocalTensor<float8_e4m3_t> dstTensor = p1Tensor[dstOffset];
        LocalTensor<float8_e4m3_t> srcTensor = (createdTaskCount == 0)
            ? UB_stage1OutQue0.Get<float8_e4m3_t>()
            : UB_stage1OutQue1.Get<float8_e4m3_t>();

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 8;
        dataCopyParams.blockLen = (uint16_t)16;
        dataCopyParams.srcStride =(uint16_t) 49;
        dataCopyParams.dstStride = (uint16_t)112;

        DataCopy(dstTensor, srcTensor, dataCopyParams);

    }

//subloop1   搬运mask  比 subloop0还快  ，导致数据踩踏
    __aicore__ inline void ProcessVec1(LocalTensor<float8_e4m3_t> p1Tensor, int bmm1_id, int ipuse, uint32_t loop, uint32_t subLoop)
    {
        // bmm1ResBuf.WaitCrossCore();
        int bmm1_idused = 6 + bmm1_id * 2;
        wait_intra_block(PIPE_V, bmm1_idused + 0);

        wait_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);//stage1OutQue AllocTensor

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)4); //pscale AllocTensor
        // bmm1ResBuf.SetCrossCore();
        set_intra_block(PIPE_V, bmm1_idused + 1);

        set_flag(PIPE_V, PIPE_MTE3, (event_t)3);//pscale EnQue
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)3);//pscale DeQue
        if (subLoop % 2 == 1) {
            ub_l1_vec1_pscale(p1Tensor[mBaseSize * s2BaseSize], subLoop, loop % 2);
        }
        set_flag(PIPE_MTE3, PIPE_V, (event_t)4);// pscale  FreeTensor

        set_flag(PIPE_V, PIPE_MTE3, (event_t)2);//stage1OutQue EnQue
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)2);//stage1OutQue DeQue
        ub_l1_vec1(p1Tensor, subLoop, loop % 2);
        set_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);//stage1OutQue FreeTensor

        if (subLoop % 2 == 1) {
            // outputBuf.SetCrossCore();//L1
            set_intra_block(PIPE_MTE3, ipuse);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t stage1OutSize = (64 + 1) * 256;
        constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
        constexpr uint32_t mm2ResultSize = 128 / 2 * 128 * 4;
        tPipePtr->InitBuffer(UB_stage1OutQue0, stage1OutSize);
        tPipePtr->InitBuffer(UB_stage1OutQue1, stage1OutSize);

        tPipePtr->InitBuffer(UB_pScaleSubLoop0, 512);

    }

    __aicore__ inline void AllocEventID()
    {
        set_flag(PIPE_MTE3, PIPE_V, (event_t)0);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)1);

        set_flag(PIPE_MTE3, PIPE_V, (event_t)2);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)3);

        set_flag(PIPE_MTE3, PIPE_V, (event_t)4);

    }

    __aicore__ inline void FreeEventID()
    {
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)1);

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)2);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)3);

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)4);

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
