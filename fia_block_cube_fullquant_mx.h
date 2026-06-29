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
 * \file fia_block_cube_fullquant_mx.h
 * \brief
 */
#ifndef FIA_BLOCK_CUBE_FULLQUANT_PA_WHILE_LOOP_P_L1_H_
#define FIA_BLOCK_CUBE_FULLQUANT_PA_WHILE_LOOP_P_L1_H_
// #include "../offset_calculator.h"
// #include "../matmul.h"
// #include "../FixpipeOut.h"
// #include "../memory_copy_arch35.h"

// #include "infer_flash_attention_comm.h"
// #include "flash_attention_score_common_regbase.h"
// #include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasRope = false, uint8_t KvLayoutType = 0,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMxBlockCube {
public:
    // using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    // const ConstInfoX &constInfo;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    __gm__ uint8_t* queryGm = nullptr;
    __gm__ uint8_t *keyGm = nullptr;
    __gm__ uint8_t *valueGm = nullptr;
    __gm__ int32_t *blockTableGm = nullptr;
    __gm__ uint64_t *actualSeqLengthsGmQ = nullptr;
    __gm__ uint8_t* deScaleQGm = nullptr;
    __gm__ uint8_t *deScaleKGm = nullptr;
    __gm__ uint8_t *deScaleVGm = nullptr;

    TPipe *tPipePtr = nullptr;
    __aicore__ inline void SetTPipe(TPipe& pipe) { tPipePtr = &pipe; }
    TBuf<TPosition::A1> L1_Q;
    TBuf<TPosition::A1> L1_KV;
    int ikv{-1};
    int iq{-1};
    int iquse = -1;
    TBuf<TPosition::A2> L0_A;
    TBuf<TPosition::B2> L0_B;
    TBuf<TPosition::CO1> L0_C;
    int l0a_id{-1};
    int l0c_id{-1};

    // using Q_T = INPUT_T;
    // using KV_T = INPUT_T;
    // using MM_T = T;
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
    // static constexpr uint64_t gS1Idx = (uint64_t)0;
    static constexpr uint32_t FirstS2Loop = (uint32_t)0;
    static constexpr uint32_t LastS2Loop = (uint32_t)20;//修改
    static constexpr uint32_t actMSize = (uint32_t)32;
    static constexpr uint32_t actMSizeAlign32 = (uint32_t)32;

    // static constexpr uint32_t l1BaseD = 128;
    // static constexpr uint32_t s2SplitSize = 256U;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;
    // static constexpr LayOutTypeEnum LAYOUT = layout;

    __aicore__ inline FAFullQuantMxBlockCube(){};
    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                         __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                         __gm__ uint8_t *dequantScaleValue)
    {
        InitCubeInput(query, key, value, blockTable, queryRope, keyRope, actualSeqQlenAddr, actualSeqKvlenAddr,
                      dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t mm1QSize = 128 * 128;
        constexpr uint32_t mm1QScaleSize = 128 * 128 / MXFP_GROUP_SIZE;
        constexpr uint32_t mm1KVSize = 128 * 256;
        constexpr uint32_t mm1KVScaleSize = 128 * 256 / MXFP_GROUP_SIZE;
        constexpr uint32_t mm2LeftSize = 128 * 512 + 128 * 512 / 32;

        tPipePtr->InitBuffer(L1_Q, (mm1QSize + mm1QScaleSize) * 2);
        tPipePtr->InitBuffer(L1_KV, (mm1KVSize + mm1KVScaleSize) * 4);
        tPipePtr->InitBuffer(L0_A, 32 * 1024 * 2);
        tPipePtr->InitBuffer(L0_B, 32 * 1024 * 2);
        tPipePtr->InitBuffer(L0_C, 128 * 256 * 4 * 2);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                         __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                         __gm__ uint8_t *dequantScaleValue)
    {   
        queryGm = query;
        keyPtr = key;
        valuePtr = value;
        actualSeqLengthsGmQ = (__gm__ uint64_t *)actualSeqQlenAddr;
        deScaleQGm = dequantScaleQuery;
        deScaleKGm = dequantScaleKey;
        deScaleVGm = dequantScaleValue;

        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        keyGm = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        valueGm = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        blockTableGm = (__gm__ int32_t *)blockTable;
    }

        __aicore__ inline uint64_t GetTBaseQ_MX(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        return actualSeqLengthsGmQ[bIdx - 1];
    }
 
    __aicore__ inline void gm_l1_q(uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx, uint32_t q1use)
    
    {
        uint32_t gIdxStart = gS1Idx % n2Size;
        uint64_t tIdx = GetTBaseQ_MX(bIdx) + gS1Idx / gSize;
        uint64_t tStride = n2Size * n2Size * dBaseSize;
        uint64_t gmQOffset = tIdx * n2Size * n2Size * dBaseSize +
                             n2Idx * n2Size * dBaseSize;
        constexpr uint32_t qSlotSize = 128 * 128 + 128 * 128 / 32;
        LocalTensor<float8_e4m3_t> q1Tensor = L1_Q.Get<float8_e4m3_t>()[(iquse % 2) * qSlotSize];
        Nd2NzParams nd2nzPara1;
        nd2nzPara1.ndNum = 1;
        nd2nzPara1.nValue = n2Size - gIdxStart;
        nd2nzPara1.dValue = dBaseSize;
        nd2nzPara1.srcDValue = dBaseSize;
        nd2nzPara1.dstNzC0Stride = (actMSize + 31) >> 5 << 5;
        nd2nzPara1.dstNzNStride = 1;
        nd2nzPara1.srcNdMatrixStride = 0;
        nd2nzPara1.dstNzMatrixStride = 0;
        GlobalTensor<float8_e4m3_t> gmQTensor1;
        gmQTensor1.SetGlobalBuffer((__gm__ float8_e4m3_t *)(queryGm + gmQOffset));
        DataCopy(q1Tensor, gmQTensor1, nd2nzPara1);
    }
 
    __aicore__ inline uint64_t GetTBaseQsacle_MX(uint32_t bIdx)
    {
        if (bIdx == 0) {
            return 0;
        }
        return (bIdx) * s1Size;
    }
 
    __aicore__ inline void gm_l1_qscale(uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx, uint32_t q1use)
    {
        uint32_t qScaleOffset = ((actMSize + 31) >> 5 << 5) * dBaseSize;//((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;runInfo.actMSize=gs1=8*4
        uint32_t s1IdxStart = gS1Idx / n2Size;
        uint32_t gIdxStart = gS1Idx % n2Size;
        uint64_t tIdx = GetTBaseQsacle_MX(bIdx) + s1IdxStart;
        uint64_t gmQScaleOffset = tIdx * dBaseSize / MXFP_GROUP_SIZE * n2Size +
                                  n2Idx * bSize * s1Size * dBaseSize / MXFP_GROUP_SIZE * n2Size
                                  + gIdxStart * dBaseSize / MXFP_GROUP_SIZE;
 
        constexpr uint32_t qSlotSize = 128 * 128 + 128 * 128 / 32;
        Dn2NzParams dn2nzPara;
        dn2nzPara.dnNum = 1;
        dn2nzPara.nValue = dBaseSize / MXFP_GROUP_SIZE / 2;
        dn2nzPara.dValue = actMSize;
        dn2nzPara.srcDnMatrixStride = 0;
        dn2nzPara.srcDValue = dBaseSize / MXFP_GROUP_SIZE / 2;
        dn2nzPara.dstNzC0Stride = dBaseSize / MXFP_GROUP_SIZE / 2;
        dn2nzPara.dstNzNStride = 1;
        dn2nzPara.dstNzMatrixStride = dBaseSize / MXFP_GROUP_SIZE / 2;

        LocalTensor<float8_e8m0_t> q1ScaleTensor = L1_Q.Get<float8_e8m0_t>()[q1use * qSlotSize + qScaleOffset];
        LocalTensor<bfloat16_t> q1ScaleTensorCast = q1ScaleTensor.template ReinterpretCast<bfloat16_t>();
        GlobalTensor<float8_e8m0_t> gmQScaleTensor;
        gmQScaleTensor.SetGlobalBuffer((__gm__ float8_e8m0_t *)deScaleQGm + gmQScaleOffset);
        GlobalTensor<bfloat16_t> gmQScaleTensorCast;
        gmQScaleTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(gmQScaleTensor.GetPhyAddr())));
        DataCopy(q1ScaleTensorCast, gmQScaleTensorCast, dn2nzPara);
    }

    __aicore__ inline void gm_l1_k(uint32_t s2Idx, uint32_t bIdx, uint32_t n2Idx, uint32_t subLoop, int ivkuse)
    {
        // uint32_t s2Offset = (subLoop % 2 == 0) ? inner * 512 : inner * 512 + 256;
        uint32_t s2Offset = (subLoop % 2 == 0) ? s2Idx : s2Idx + 256;
        uint64_t blockIdxInBatch = s2Offset / 512;  // 获取block table上的索引
        uint64_t bsIdx = s2Offset % 512;            // 获取在单个块上超出的行数
        uint64_t blockId = *(__gm__ int32_t *)(blockTableGm + bIdx * maxBlockNumPerBatch + blockIdxInBatch);
        uint64_t gmKOffset = blockId * n2Size * dBaseSize / 32 * 512 * 32 + n2Idx * dBaseSize / 32  * 512 * 32 + bsIdx * 32;
        constexpr uint32_t kvSlotSize = 128 * 256;
        LocalTensor<float8_e4m3_t> kv1Tensor = L1_KV.Get<float8_e4m3_t>()[ivkuse * kvSlotSize];
        DataCopyParams intriParamsK;
        intriParamsK.blockCount = dBaseSize / 32;
        intriParamsK.blockLen = 256;
        intriParamsK.dstStride = 0;
        intriParamsK.srcStride = 256;
        GlobalTensor<float8_e4m3_t> gmKTensor;
        gmKTensor.SetGlobalBuffer((__gm__ float8_e4m3_t *)(keyGm + gmKOffset));
        DataCopy(kv1Tensor, gmKTensor, intriParamsK);
    }

    __aicore__ inline void gm_l1_kscale(uint32_t s2Idx, uint32_t bIdx, uint32_t n2Idx, uint32_t subLoop, int ivkuse)
    {
        uint32_t s2Offset = (subLoop % 2 == 0) ? s2Idx : s2Idx + 256;
        uint64_t bsIdx = s2Offset % 512;            // 获取在单个块上超出的行数
        uint64_t blockKScaleId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * maxBlockNumPerBatch +
                                s2Offset / 512);
        uint64_t gmKScaleOffset = blockKScaleId * n2Size * 512 / 32 * MXFP_MULTI_BASE_SIZE * dBaseSize / MXFP_GROUP_SIZE / MXFP_MULTI_BASE_SIZE * 32 +
                                  n2Idx * 512 / 32 * MXFP_MULTI_BASE_SIZE * dBaseSize / MXFP_GROUP_SIZE / MXFP_MULTI_BASE_SIZE * 32 +
                                  bsIdx / 16 * dBaseSize / MXFP_GROUP_SIZE / MXFP_MULTI_BASE_SIZE * 32 + bsIdx % 16;

        constexpr uint32_t kvSlotSize = 128 * 256 + 128 * 256 / 32;
        LocalTensor<float8_e8m0_t> kv1ScaleTensor = L1_KV.Get<float8_e8m0_t>()[ivkuse * kvSlotSize + 256 * 128];
        DataCopyParams intriParamsKScale;
        intriParamsKScale.blockCount = 1;
        intriParamsKScale.blockLen = 32;
        intriParamsKScale.dstStride = 0;
        intriParamsKScale.srcStride = 0;
        GlobalTensor<float8_e8m0_t> gmKScaleTensor;
        gmKScaleTensor.SetGlobalBuffer((__gm__ float8_e8m0_t *)(deScaleKGm + gmKScaleOffset));
        DataCopy(kv1ScaleTensor, gmKScaleTensor, intriParamsKScale);
    }

    __aicore__ inline void mm1_l1_l0a_no_trans(int l0a_iduse)
    {
        constexpr uint32_t qSlotSize = 128 * 128 + 128 * 128 / 32;
        LocalTensor<mx_fp8_e4m3_t> aL0Tensor = L0_A.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];
        LocalTensor<float8_e4m3_t> aL1Tensor = L1_Q.Get<float8_e4m3_t>()[(iquse % 2) * qSlotSize];
        uint32_t qScaleOffset = 32 * 128;
        LocalTensor<float8_e8m0_t> aScaleL1Tensor = L1_Q.Get<float8_e8m0_t>()[(iquse % 2) * qSlotSize + qScaleOffset];

        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.ifTranspose = false;
        loadData2DParamsA.mStep = ((actMSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsA.kStep = 4;
        loadData2DParamsA.srcStride = ((actMSize + 31) >> 5 << 5) >> 4;
        loadData2DParamsA.dstStride = ((actMSize + 15) >> 4 << 4) >> 4;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = 0;
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = ((actMSize + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsA.yStep = 2;
        loadData2DMxParamsA.srcStride = 2;
        loadData2DMxParamsA.dstStride = 2;

        LoadData(aL0Tensor, aL1Tensor, aScaleL1Tensor, loadData2DParamsA, loadData2DMxParamsA);

    }
    __aicore__ inline void mm1_l1_l0b_no_trans(int ivkuse, int l0b_iduse)
    {
        constexpr uint32_t kvSlotSize = 128 * 256 + 128 * 256 / 32;
        LocalTensor<mx_fp8_e4m3_t> bL0Tensor = L0_B.Get<mx_fp8_e4m3_t>()[((l0b_iduse - 4) % 2) * 32 * 1024];
        LocalTensor<float8_e4m3_t> bL1Tensor = L1_KV.Get<float8_e4m3_t>()[ivkuse * kvSlotSize];
        uint32_t kScaleOffset = 256 * dBaseSize;
        LocalTensor<float8_e8m0_t> bScaleL1Tensor = L1_KV.Get<float8_e8m0_t>()[ivkuse * kvSlotSize + kScaleOffset];

        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.ifTranspose = false;
        loadData2DParamsB.mStep = 16;
        loadData2DParamsB.kStep = ((dBaseSize + 31) >> 5 << 5) >> 5;
        loadData2DParamsB.srcStride = 16;
        loadData2DParamsB.dstStride = 16;

        LoadData2DMxParams loadData2DMxParamsB;
        loadData2DMxParamsB.xStartPosition = 0;
        loadData2DMxParamsB.yStartPosition = 0;
        loadData2DMxParamsB.xStep = 16;
        loadData2DMxParamsB.yStep = (dBaseSize + 63) >> 5 >> 1;
        loadData2DMxParamsB.srcStride = (dBaseSize + 63) >> 5 >> 1;
        loadData2DMxParamsB.dstStride = (dBaseSize + 63) >> 5 >> 1;

        LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB, loadData2DMxParamsB);
    }

    __aicore__ inline void compute_mm1(int l0c_iduse, int l0a_iduse)
    {
        LocalTensor<float> cL0Tensor = L0_C.Get<float>()[l0c_iduse * 128 * 256];
        LocalTensor<mx_fp8_e4m3_t> aL0Tensor = L0_A.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];
        LocalTensor<mx_fp8_e4m3_t> bL0Tensor = L0_B.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];

        MmadParams mmadParams;
        mmadParams.m = actMSize;
        mmadParams.n = 256;
        mmadParams.k = dBaseSize;
        mmadParams.cmatrixInitVal = 1;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;

        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
    }

        __aicore__ inline void l0c_ub_mm1(LocalTensor<float> mm1Res, int l0c_iduse)
    {
        LocalTensor<float> srcTensor = L0_C.Get<float>()[l0c_iduse * 128 * 256];
        PipeBarrier<PIPE_FIX>();
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = 256;
        fixpipeParams.mSize = 32;  // (runInfo.actMSize + 31) >> 5 << 5
        fixpipeParams.srcStride = 32; // ((runInfo.actMSize + 15) / 16) * 16;
        fixpipeParams.dstStride = 256;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<float, float, PFA_CFG_ROW_MAJOR_UB>(mm1Res, srcTensor, fixpipeParams);
    }

    __aicore__ inline void IterateBmm1(LocalTensor<float> mm1Res, int bmm1_id, uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx,
                                        uint32_t s2Idx, uint32_t curS2Loop, uint32_t subLoop)
    {
        if (curS2Loop == FirstS2Loop && subLoop == 0){
            iq ++;
            uint32_t q1use = iq % 2;
            iquse =  4 + q1use;
            wait_flag(PIPE_MTE1, PIPE_MTE2, iquse);
            gm_l1_q(bIdx, n2Idx, gS1Idx, q1use);
            gm_l1_qscale(bIdx, n2Idx, gS1Idx, q1use);
            set_flag(PIPE_MTE2, PIPE_MTE1, iquse);
        } else {
            set_flag(PIPE_MTE2, PIPE_MTE1, iquse);
        }
        //k and kscale:gm->l1
        ikv++;
        int ivkuse = ikv % 4;
        wait_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
 
        gm_l1_k(s2Idx, bIdx, n2Idx, subLoop, ivkuse);
        gm_l1_kscale(s2Idx, bIdx, n2Idx, subLoop, ivkuse);
 
        set_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
        wait_flag(PIPE_MTE2, PIPE_MTE1, iquse);
        wait_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
 
        //mte1 start
        l0c_id ++ ;
        int l0c_iduse = l0c_id % 2;
        wait_flag(PIPE_FIX, PIPE_M, l0c_iduse);
 
        l0a_id ++ ;
        int l0a_iduse = l0a_id % 2 + 4;
        wait_flag(PIPE_M, PIPE_MTE1, l0a_iduse);// mte1 alloc l0ab buffer
        mm1_l1_l0a_no_trans(l0a_iduse);
        mm1_l1_l0b_no_trans(ivkuse, l0a_iduse);
        set_flag(PIPE_MTE1, PIPE_M, l0a_iduse);
        wait_flag(PIPE_MTE1, PIPE_M, l0a_iduse);
 
        compute_mm1(l0c_iduse, l0a_iduse);
        set_flag(PIPE_M, PIPE_MTE1, l0a_iduse);
 
        if (((curS2Loop + 1) == LastS2Loop) && (subLoop % 2 == 1)) {//修改
            set_flag(PIPE_MTE1, PIPE_MTE2, iquse);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
 
        set_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        wait_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        int bmm1_idused = 6 + bmm1_id * 2;
        wait_intra_block(PIPE_FIX, 0 + bmm1_idused + 1);//9
        wait_intra_block(PIPE_FIX, 16 + bmm1_idused + 1);//15
 
        l0c_ub_mm1(mm1Res, l0c_iduse);//fixp
        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);
 
        set_intra_block(PIPE_FIX, 0 + bmm1_idused + 0);//8
        set_intra_block(PIPE_FIX, 16 + bmm1_idused + 0);//14
    }
    __aicore__ inline void gm_l1_v(uint32_t inner, uint32_t bIdx, uint32_t n2Idx, uint32_t k, int ivkuse)
    {
        uint32_t s2Offset = inner + k * 256;
        uint64_t blockIdxInBatch = s2Offset / 512;  // 获取block table上的索引
        uint64_t bsIdx = s2Offset % 512;            // 获取在单个块上超出的行数
        uint64_t blockId = *(__gm__ int32_t *)(blockTableGm + bIdx * maxBlockNumPerBatch + blockIdxInBatch);
        uint64_t gmVOffset = blockId * n2Size * dVBaseSize / 32 * 512 * 32 + n2Idx * dVBaseSize / 32  * 512 * 32 + bsIdx * 32;
        
        constexpr uint32_t kvSlotSize = 128 * 256;
        LocalTensor<float8_e4m3_t> kv1Tensor = L1_KV.Get<float8_e4m3_t>()[ivkuse * kvSlotSize];
        DataCopyParams intriParamsV;
        intriParamsV.blockCount = dVBaseSize / 32;
        intriParamsV.blockLen = 256;
        intriParamsV.dstStride = 0;
        intriParamsV.srcStride = 256;
        GlobalTensor<float8_e4m3_t> gmVTensor;
        gmVTensor.SetGlobalBuffer((__gm__ float8_e4m3_t *)(valueGm + gmVOffset));
        DataCopy(kv1Tensor, gmVTensor, intriParamsV);
    }

    __aicore__ inline void gm_l1_vscale(uint32_t inner, uint32_t bIdx, uint32_t n2Idx, uint32_t k, int ivkuse)
    {
        uint32_t s2VscaleScaleOffset = (inner + k * 256) / MXFP_DIVISOR_SIZE;
        uint64_t blockVscaleIdxInBatch = s2VscaleScaleOffset / (dVBaseSize * MXFP_MULTI_BASE_SIZE / 32);  // 获取block table上的索引
        uint64_t bsVscaleIdx = s2VscaleScaleOffset % (dVBaseSize * MXFP_MULTI_BASE_SIZE / 32);            // 获取在单个块上超出的行数
        uint64_t blockVscaleId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * maxBlockNumPerBatch +
                                blockVscaleIdxInBatch);
        uint64_t gmVScaleOffset = blockVscaleId * n2Size * dVBaseSize * MXFP_MULTI_BASE_SIZE / 32 * 512 / MXFP_DIVISOR_SIZE * 32 +
                                  n2Idx * dVBaseSize * MXFP_MULTI_BASE_SIZE / 32 * 512 / MXFP_DIVISOR_SIZE * 32 + bsVscaleIdx * 32;
        constexpr uint32_t kvSlotSize = 128 * 256 + 128 * 256 / 32;
        LocalTensor<float8_e8m0_t> kv1VScaleTensor = L1_KV.Get<float8_e8m0_t>()[ivkuse * kvSlotSize + 256 * dVBaseSize];
        DataCopyParams intriParamsVScale;
        intriParamsVScale.blockCount = dVBaseSize * MXFP_MULTI_BASE_SIZE / 32;
        intriParamsVScale.blockLen = 256 / MXFP_DIVISOR_SIZE;
        intriParamsVScale.dstStride = 0;
        intriParamsVScale.srcStride = dVBaseSize * MXFP_MULTI_BASE_SIZE / 32 - 256 / MXFP_DIVISOR_SIZE;
        GlobalTensor<float8_e8m0_t> gmVScaleTensor;
        gmVScaleTensor.SetGlobalBuffer((__gm__ float8_e8m0_t *)(deScaleVGm + gmVScaleOffset));
        DataCopy(kv1VScaleTensor, gmVScaleTensor, intriParamsVScale);
    }


    __aicore__ inline void mm2_l1_l0a_no_trans(LocalTensor<float8_e4m3_t> p1Tensor, uint32_t k, int ipuse, int l0a_iduse)
    {
        uint32_t pScaleOffset = mBaseSize * s2BaseSize; // PScale在L1P的偏移量（单位：元素）; // VScale在mm1B的偏移量（单位：元素）
        uint64_t l1ScaleOffset = k * 256 / MXFP_GROUP_SIZE * actMSizeAlign32 ;
        LocalTensor<mx_fp8_e4m3_t> aL0Tensor = L0_A.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];
        LocalTensor<float8_e4m3_t> aL1Tensor = p1Tensor[k * 256 * dBaseSize];
        LocalTensor<float8_e8m0_t> aScaleL1Tensor = p1Tensor.template ReinterpretCast<float8_e8m0_t>()[pScaleOffset + l1ScaleOffset];

        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.ifTranspose = false;
        loadData2DParamsA.mStep = 2;
        loadData2DParamsA.kStep = 8;
        loadData2DParamsA.srcStride = 8;
        loadData2DParamsA.dstStride = 2;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = 0;
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = 2;
        loadData2DMxParamsA.yStep = 4;
        loadData2DMxParamsA.srcStride = 4;
        loadData2DMxParamsA.dstStride = 4;

        LoadData(aL0Tensor, aL1Tensor, aScaleL1Tensor, loadData2DParamsA, loadData2DMxParamsA);  
    }
    __aicore__ inline void mm2_l1_l0b_trans(int ivkuse, int l0b_iduse)
    {
        constexpr uint32_t kvSlotSize = 128 * 256 + 128 * 256 / 32;
        LocalTensor<mx_fp8_e4m3_t> bL0Tensor = L0_B.Get<mx_fp8_e4m3_t>()[((l0b_iduse - 4) % 2) * 32 * 1024];
        LocalTensor<float8_e4m3_t> bL1Tensor = L1_KV.Get<float8_e4m3_t>()[ivkuse * kvSlotSize];
        uint32_t vScaleOffset = 256 * dVBaseSize;
        LocalTensor<float8_e8m0_t> bScaleL1Tensor = L1_KV.Get<float8_e8m0_t>()[ivkuse * kvSlotSize + vScaleOffset];

        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.ifTranspose = true;
        loadData2DParamsB.mStep = 16;
        loadData2DParamsB.kStep = 4;
        loadData2DParamsB.srcStride = 16;
        loadData2DParamsB.dstStride = 8;

        LoadData2DMxParams loadData2DMxParamsB;
        loadData2DMxParamsB.xStartPosition = 0;
        loadData2DMxParamsB.yStartPosition = 0;
        loadData2DMxParamsB.xStep = 8;
        loadData2DMxParamsB.yStep = 4;
        loadData2DMxParamsB.srcStride = 4;
        loadData2DMxParamsB.dstStride = 4;

        LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB, loadData2DMxParamsB);
    }

    __aicore__ inline void compute_mm2(int l0c_iduse, int l0a_iduse, uint32_t k)
    {
        LocalTensor<float> cL0Tensor = L0_C.Get<float>()[l0c_iduse * 128 * 256];
        LocalTensor<mx_fp8_e4m3_t> aL0Tensor = L0_A.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];
        LocalTensor<mx_fp8_e4m3_t> bL0Tensor = L0_B.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];

        MmadParams mmadParams;
        mmadParams.m = actMSize;
        mmadParams.n = dVBaseSize;
        mmadParams.k = 256;
        mmadParams.cmatrixInitVal = k == 0 ? 1 : 0;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;

        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
    }

    __aicore__ inline void l0c_ub_mm2(LocalTensor<float> mm2Res, int l0c_iduse)
    {
        LocalTensor<float> srcTensor = L0_C.Get<float>()[l0c_iduse * 128 * 256];

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = 128;
        fixpipeParams.mSize = 32;
        fixpipeParams.srcStride = 32;
        fixpipeParams.dstStride = 128;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        PipeBarrier<PIPE_FIX>();
        Fixpipe<float, float, PFA_CFG_ROW_MAJOR_UB>(mm2Res, srcTensor, fixpipeParams);
    }


    __aicore__ inline void IterateBmm2(int ipuse, LocalTensor<float> mm2Res, LocalTensor<float8_e4m3_t> p1Tensor, uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx)
    {   
        wait_intra_block(PIPE_MTE1, ipuse);
        wait_intra_block(PIPE_MTE1, ipuse + 16);
        l0c_id ++ ;
        int l0c_iduse = l0c_id % 2;
        wait_flag(PIPE_FIX, PIPE_M, l0c_iduse);
        for (uint32_t k = 0; k < 2; k++) {
            ikv++;
            int ivkuse = ikv % 4;
            wait_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
            gm_l1_v(s2Idx, bIdx, n2Idx, k, ivkuse);
            gm_l1_vscale(s2Idx, bIdx, n2Idx, k, ivkuse);

            set_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
            wait_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
                        // mte1 start
            l0a_id ++ ;
            int l0a_iduse = l0a_id % 2 + 4;
            wait_flag(PIPE_M, PIPE_MTE1, l0a_iduse);// mte1 alloc l0ab buffer
            mm2_l1_l0a_no_trans(p1Tensor, k, ipuse, l0a_iduse);
            mm2_l1_l0b_trans(ivkuse, l0a_iduse);
            set_flag(PIPE_MTE1, PIPE_M, l0a_iduse);
            wait_flag(PIPE_MTE1, PIPE_M, l0a_iduse);
            compute_mm2(l0c_iduse, l0a_iduse, k);
            set_flag(PIPE_M, PIPE_MTE1, l0a_iduse);
            set_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
        }
        set_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        wait_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        l0c_ub_mm2(mm2Res, l0c_iduse);//fixp
        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);
    }
}; // FAFullQuantMxBlockCube

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasRope = false, uint8_t KvLayoutType = 0,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMxBlockCubeDummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool HAS_ROPE = hasRope;
    static constexpr bool HAS_PREFIX = enableKVPrefix;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;

    using ROPE_T = bfloat16_t;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    __aicore__ inline void SetTPipe(TPipe&) {}
    __aicore__ inline FAFullQuantMxBlockCubeDummy(ConstInfoX &constInfo) {};
};
} // namespace BaseApi

#endif // FIA_BLOCK_CUBE_FULLQUANT_MX_H_