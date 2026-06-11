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
 * \file fia_kernel_fullquant_mx.h
 * \brief
 */

#ifndef FIA_KERNEL_FULLQUANT_MX_MASK_H_
#define FIA_KERNEL_FULLQUANT_MX_MASK_H_

#include "fia_public_define_arch35.h"
#include "fia_block_cube_fullquant_mx.h"
#include "fia_block_vec_fullquant_mx.h"
#include "../memory_copy_arch35.h"
#include "fia_block_vec_flashdecode_fullquant.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "fia_tiling_data_fullquant_mx.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionFullQuantMxKernel {
public:
    using ConstInfoX = typename CubeBlockType::ConstInfoX;
    ConstInfoX constInfo;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    __gm__ uint8_t* queryGm;
    __gm__ uint8_t *keyGm;
    __gm__ uint8_t *valueGm;
    __gm__ uint8_t* deScaleQGm;
    __gm__ uint8_t *deScaleKGm;
    __gm__ uint8_t *deScaleVGm;
    __gm__ int32_t *blockTableGm;
    __gm__ uint8_t* attenMaskGmInt;
    __gm__ uint32_t *fiaMetaDataGm;
    TPipe tPipe;
    TBuf<TPosition::A1> L1_Q;
    TBuf<TPosition::A1> L1_KV;
    TBuf<TPosition::A1> L1_P;
    TBuf<TPosition::A1> L1_PScale;
    TBuf<TPosition::A2> L0_A;
    TBuf<TPosition::B2> L0_B;
    TBuf<TPosition::CO1> L0_C;
    TBuf<TPosition::VECIN> UB_PV;
    TBuf<TPosition::VECIN> UB_MM1Res;
    TBuf<TPosition::VECIN> UB_MM2Res;
    TBuf<TPosition::VECIN> UB_STAGE1OutQue;
    TBuf<TPosition::VECIN> UB_AttenMask;
    int ikv{-1};
    int iq{-1};
    int iquse = -1;
    int ipid{-1};
    int bmm1_id{1};
    int vec1_id{-1};
    int l0a_id{-1};
    int l0c_id{-1};
    int mm1_id{-1};

    uint64_t inner_loop_limit;
    uint64_t outer_loop_limit;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    static constexpr uint32_t s2BaseSize = 512;
    static constexpr uint32_t s2SplitSize = 256;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32;

    __aicore__ inline FlashAttentionFullQuantMxKernel(){};

    __aicore__ inline void InitCubeBuffer()
    {
        constexpr uint32_t mm1QSize = 128 * 128;
        constexpr uint32_t mm1QScaleSize = 128 * 128 / 32;
        constexpr uint32_t mm1KVSize = 128 * 256;
        constexpr uint32_t mm1KVScaleSize = 128 * 256 / 32;
        constexpr uint32_t mm2LeftSize = 128 * 512;
        constexpr uint32_t pScaleSize = 128 * 512 / 32;
        constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
        constexpr uint32_t mm2ResultSize = 128 / 2 * 128 * 4;

        tPipe.InitBuffer(L1_P, mm2LeftSize * 3);
        tPipe.InitBuffer(L1_PScale, pScaleSize);
        tPipe.InitBuffer(L1_Q, (mm1QSize + mm1QScaleSize) * 2);
        tPipe.InitBuffer(L1_KV, (mm1KVSize + mm1KVScaleSize) * 4);
        tPipe.InitBuffer(L0_A, 32 * 1024 * 2);
        tPipe.InitBuffer(L0_B, 32 * 1024 * 2);
        tPipe.InitBuffer(L0_C, 128 * 256 * 4 * 2);
        tPipe.InitBuffer(UB_MM1Res, mm1ResultSize * 2);
        tPipe.InitBuffer(UB_MM2Res, mm2ResultSize);
    }
    __aicore__ inline void InitVecBuffer()
    {
        tPipe.InitBuffer(UB_STAGE1OutQue, 16640 * 2);
        tPipe.InitBuffer(UB_AttenMask, 16384 * 2);
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        constInfo.bSize = 4;
        constInfo.t1Size = 192;
        constInfo.t2Size = 0;
        constInfo.n2Size = 8; // fiaBaseParams.n2Size;
        constInfo.gSize = 8;
        constInfo.s1Size = 4;
        constInfo.s2Size = actSeqLensKv;
        constInfo.dSize = 128;
        constInfo.dSizeV = 128;

        constInfo.dSizeRope = 0;
        constInfo.actualSeqLenSize = 48;
        constInfo.actualSeqLenKVSize = 48;
        constInfo.scaleValue = 0.088388;
        constInfo.isKvContinuous = 1;
        constInfo.coreNum = 32;
        constInfo.outputLayout = FIA_LAYOUT::TND;

        constInfo.sparseMode = 3;
        constInfo.preTokens = 2147483647;
        constInfo.nextTokens = 2147483647;
        constInfo.attenMaskBatch = 1;
        constInfo.attenMaskS1Size = 2048;
        constInfo.attenMaskS2Size = 2048;
        constInfo.isRowInvalidOpen = 0;
        constInfo.isExistRowInvalid = 0;

        constInfo.accumOutSize = 0;
        constInfo.logSumExpSize = 0;

        constInfo.maxBlockNumPerBatch = 20;
        constInfo.blockSize = 512;
        constInfo.paLayoutType = 2;

        constInfo.isSoftmaxLseEnable = 0;

        constInfo.bN2Start = 12 * constInfo.aicIdx;
        constInfo.gS1OStart = 0;
        constInfo.s2OStart = 0;
        constInfo.bN2End = 12 * (constInfo.aicIdx + 1);
        constInfo.gS1OEnd = 0;
        constInfo.s2OEnd = 0;
        constInfo.coreFirstTmpOutWsPos = 0;

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *pScale, __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *keyRope, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace)
    {
        actSeqLensKv = ((__gm__ uint64_t *)actualSeqLengthsKv)[0];
        actSeqLensQ = 4;
        inner_loop_limit = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        outer_loop_limit = (actSeqLensQ * 8 + 127) >> 7;
        
        InitConstInfo();
        queryGm = query;
        keyPtr = key;
        valuePtr = value;
        attenMaskGmInt = attenMask;
        deScaleQGm = dequantScaleQuery;
        deScaleKGm = dequantScaleKey;
        deScaleVGm = dequantScaleValue;
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        keyGm = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        valueGm = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        blockTableGm = (__gm__ int32_t *)blockTable;

        InitCubeBuffer();
        InitVecBuffer();
    }

    __aicore__ inline void prefree_kv_l1()
    {
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)0);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)1);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)2);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)3);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)4);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)5);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)6);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)7);
    }

    __aicore__ inline void postalloc_kv_l1()
    {
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)3);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)4);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)5);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)6);
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)7);
    }

    __aicore__ inline void prefree_l0c(){
        set_flag(PIPE_FIX, PIPE_M, (event_t)0);
        set_flag(PIPE_FIX, PIPE_M, (event_t)1);
    }

    __aicore__ inline void prefree_l0ab(){
        set_flag(PIPE_M, PIPE_MTE1, (event_t)4);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)5);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)6);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)7);
    }

    __aicore__ inline void postalloc_l0ab(){
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)4);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)5);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)6);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)7);
    }

    __aicore__ inline void postalloc_l0c(){
        wait_flag(PIPE_FIX, PIPE_M, (event_t)0);
        wait_flag(PIPE_FIX, PIPE_M, (event_t)1);
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            set_intra_block(PIPE_V, 0 + 0 + 5);
            // set_intra_block(PIPE_V, 16 + 0 + 1);
            // bmm2Buffers.Get().SetCrossCore();
        }
        if ASCEND_IS_AIV {
            set_intra_block(PIPE_V, 0 + 0 + 9);
            set_intra_block(PIPE_V, 0 + 0 + 7);
            // bmm1Buffers.Get().SetCrossCore();
            // bmm1Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        // set_mark(0x63739280);
        if ASCEND_IS_AIC {
            wait_intra_block(PIPE_FIX, 9);
            wait_intra_block(PIPE_FIX, 9 + 16);

            wait_intra_block(PIPE_FIX, 7 );
            wait_intra_block(PIPE_FIX, 7 + 16);
            // bmm1Buffers.Get().WaitCrossCore();
            // bmm1Buffers.Get().WaitCrossCore();
        }
        if ASCEND_IS_AIC {
            wait_intra_block(PIPE_FIX, 5);
            wait_intra_block(PIPE_FIX, 5 + 16);
            // bmm2Buffers.Get().WaitCrossCore();
        }        
    }
    __aicore__ inline void gm_l1_q(uint64_t outer, uint64_t bIdx, uint64_t n2Idx, uint64_t q1use)
    {
        constexpr uint32_t mm1QSize = 128 * 128;
        constexpr uint32_t mm1QScaleSize = 128 * 128 / 32;
        constexpr uint32_t singleQBufferOffset = mm1QSize + mm1QScaleSize;

        LocalTensor<float8_e4m3_t> qL1Tensor = L1_Q.Get<float8_e4m3_t>()[q1use * singleQBufferOffset];
        GlobalTensor<float8_e4m3_t> queryGmTensor;
        uint64_t gmQOffset = (bIdx * 4 + outer * 128 / 8) * 8 * 128 * 8 + n2Idx * 8 * 128;
        queryGmTensor.SetGlobalBuffer((__gm__ float8_e4m3_t *)(queryGm + gmQOffset));

        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 4;
        nd2nzParams.nValue = 8;
        nd2nzParams.dValue = 128;
        nd2nzParams.srcDValue = 128;
        nd2nzParams.dstNzC0Stride = 32;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.srcNdMatrixStride = 8 * 128 * 8;
        nd2nzParams.dstNzMatrixStride = 8 * 32;

        DataCopy(qL1Tensor, queryGmTensor, nd2nzParams);
    }

    __aicore__ inline void gm_l1_qscale(uint64_t outer, uint64_t bIdx, uint64_t n2Idx, uint64_t q1use)
    {   
        uint32_t qScaleOffset = 32 * 128;
        constexpr uint32_t mm1QSize = 128 * 128;
        constexpr uint32_t mm1QScaleSize = 128 * 128 / 32;
        constexpr uint32_t singleQBufferOffset = mm1QSize + mm1QScaleSize;

        LocalTensor<float8_e8m0_t> qScaleL1Tensor =
            L1_Q.Get<float8_e8m0_t>()[q1use * singleQBufferOffset + qScaleOffset];
        LocalTensor<bfloat16_t> qScaleL1TensorCast = qScaleL1Tensor.template ReinterpretCast<bfloat16_t>();

        uint64_t gmQScaleOffset = (bIdx * 4 + outer * 128 / 8) * 4 * 8 + n2Idx * 4 * 8 * (1 + 48) * 48 / 2 * 4;

        GlobalTensor<float8_e8m0_t> qScaleGmTensor;
        qScaleGmTensor.SetGlobalBuffer((__gm__ float8_e8m0_t *)(deScaleQGm + gmQScaleOffset));

        GlobalTensor<bfloat16_t> qScaleGmTensorCast;
        qScaleGmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(qScaleGmTensor.GetPhyAddr())));

        Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.nValue = 2;
        dn2nzParams.dValue = 32;
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = 2;
        dn2nzParams.dstNzC0Stride = 2;
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 2;

        DataCopy(qScaleL1TensorCast, qScaleGmTensorCast, dn2nzParams);
    }

    __aicore__ inline void gm_l1_k(uint64_t inner, uint64_t bIdx, uint64_t n2Idx, uint32_t subLoop, int ivkuse,
                                    uint32_t s2CalcSize)
    {
        constexpr uint32_t mm1KSize = 128 * 256;
        constexpr uint32_t mm1KScaleSize = 128 * 256 / 32;
        constexpr uint32_t singleKBufferOffset = mm1KSize + mm1KScaleSize;

        LocalTensor<float8_e4m3_t> kL1Tensor =
            L1_KV.Get<float8_e4m3_t>()[ivkuse * singleKBufferOffset];

        uint32_t s2Offset = (subLoop % 2 == 0) ? inner * s2BaseSize : inner * s2BaseSize + s2SplitSize;
        uint64_t blockId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * constInfo.maxBlockNumPerBatch +
                                s2Offset / s2BaseSize);
        uint64_t gmKOffset = blockId * s2BaseSize * 8 * 128 + n2Idx * s2BaseSize * 128 + (s2Offset % s2BaseSize) * 32;

        GlobalTensor<float8_e4m3_t> keyGmTensor;
        keyGmTensor.SetGlobalBuffer((__gm__ float8_e4m3_t *)(keyGm + gmKOffset));

        uint32_t s2CalcSizeAlign32 = (s2CalcSize + MXFP_GROUP_SIZE - 1) / MXFP_GROUP_SIZE * MXFP_GROUP_SIZE; // s2: L1行步长32对齐

        DataCopyParams intriParams;
        intriParams.blockCount = 4;
        intriParams.blockLen = s2CalcSize; // s2: 实际行数
        intriParams.dstStride = s2CalcSizeAlign32 - s2CalcSize; // s2: L1填充
        intriParams.srcStride = s2BaseSize - s2CalcSize; // s2: GM跳转步长

        DataCopy(kL1Tensor, keyGmTensor, intriParams);
    }

    __aicore__ inline void gm_l1_kscale(uint64_t inner, uint64_t bIdx, uint64_t n2Idx, uint32_t subLoop, int ivkuse,
                                         uint32_t s2CalcSize)
    {
        constexpr uint32_t mm1KSize = 128 * 256;
        constexpr uint32_t mm1KScaleSize = 128 * 256 / 32;
        constexpr uint32_t singleKBufferOffset = mm1KSize + mm1KScaleSize;

        LocalTensor<float8_e8m0_t> kScaleL1Tensor =
            L1_KV.Get<float8_e8m0_t>()[ivkuse * singleKBufferOffset + mm1KSize];

        uint32_t s2Offset = (subLoop % 2 == 0) ? inner * s2BaseSize : inner * s2BaseSize + s2SplitSize;
        uint64_t bsIdx = s2Offset % s2BaseSize;

        uint64_t blockKScaleId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * constInfo.maxBlockNumPerBatch +
                                s2Offset / s2BaseSize);
        uint64_t gmKScaleOffset = blockKScaleId * s2BaseSize * 64 * 8 + n2Idx * s2BaseSize * 64 + bsIdx / 16 * 64 + bsIdx % 16;

        GlobalTensor<float8_e8m0_t> kScaleGmTensor;
        kScaleGmTensor.SetGlobalBuffer((__gm__ float8_e8m0_t *)(deScaleKGm + gmKScaleOffset));

        DataCopyParams intriParams;
        intriParams.blockCount = (s2CalcSize + 15) / 16; // s2: PA_NZ_K_SCALE按16行分组
        intriParams.blockLen = constInfo.dSize / MXFP_GROUP_SIZE / 2;
        intriParams.dstStride = 0;
        intriParams.srcStride = 0;

        DataCopy(kScaleL1Tensor, kScaleGmTensor, intriParams);
    }

    __aicore__ inline void mm1_l1_l0a_no_trans(int l0a_iduse)
    {
        uint32_t qScaleOffset = 32 * 128;
        constexpr uint32_t mm1QSize = 128 * 128;
        constexpr uint32_t mm1QScaleSize = 128 * 128 / 32;
        constexpr uint32_t singleQBufferOffset = mm1QSize + mm1QScaleSize;

        LocalTensor<mx_fp8_e4m3_t> aL0Tensor = L0_A.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];
        LocalTensor<fp8_e4m3fn_t> aL1Tensor = L1_Q.Get<fp8_e4m3fn_t>()[(iquse % 2) * singleQBufferOffset];
        LocalTensor<float8_e8m0_t> aScaleL1Tensor = L1_Q.Get<float8_e8m0_t>()[(iquse % 2) * singleQBufferOffset + qScaleOffset];

        constexpr uint32_t kSplitSize = 128; // Q的K维度=d
        constexpr uint32_t mSplitSize = 32;  // Q的M维度=32行(ndNum=4,nValue=8)

        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.ifTranspose = false;
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsA.kStep = ((kSplitSize + 31) >> 5 << 5) >> 5;
        loadData2DParamsA.srcStride = ((mSplitSize + 31) >> 5 << 5) >> 4;
        loadData2DParamsA.dstStride = loadData2DParamsA.mStep;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = 0;
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsA.yStep = (kSplitSize + 63) >> 5 >> 1;
        loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
        loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;

        LoadData(aL0Tensor, aL1Tensor, aScaleL1Tensor, loadData2DParamsA, loadData2DMxParamsA);
    }
    __aicore__ inline void mm1_l1_l0b_no_trans(int ivkuse, int l0b_iduse, uint32_t s2CalcSize)
    {
        constexpr uint32_t mm1KSize = 128 * 256;
        constexpr uint32_t mm1KScaleSize = 128 * 256 / 32;
        constexpr uint32_t singleKBufferOffset = mm1KSize + mm1KScaleSize;

        LocalTensor<mx_fp8_e4m3_t> bL0Tensor = L0_B.Get<mx_fp8_e4m3_t>()[((l0b_iduse - 4) % 2) * 32 * 1024];
        LocalTensor<float8_e4m3_t> bL1Tensor = L1_KV.Get<float8_e4m3_t>()[ivkuse * singleKBufferOffset];
        LocalTensor<float8_e8m0_t> bScaleL1Tensor = L1_KV.Get<float8_e8m0_t>()[ivkuse * singleKBufferOffset + mm1KSize];

        uint32_t nSplitSize = s2CalcSize; // s2: K矩阵的M维度(行数)
        constexpr uint32_t kSplitSize = 128;

        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.ifTranspose = false;
        loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsB.kStep = ((kSplitSize + 31) >> 5 << 5) >> 5;
        loadData2DParamsB.srcStride = ((nSplitSize + 31) >> 5 << 5) >> 4; // s2: 32对齐匹配L1行步长
        loadData2DParamsB.dstStride = loadData2DParamsB.mStep;

        LoadData2DMxParams loadData2DMxParamsB;
        loadData2DMxParamsB.xStartPosition = 0;
        loadData2DMxParamsB.yStartPosition = 0;
        loadData2DMxParamsB.xStep = ((nSplitSize + 15) >> 4 << 4) >> 4; // s2: 与mStep一致
        loadData2DMxParamsB.yStep = (kSplitSize + 63) >> 5 >> 1;
        loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
        loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep;

        LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB, loadData2DMxParamsB);
    }

    __aicore__ inline void compute_mm1(int l0c_iduse, int l0a_iduse, uint32_t s2CalcSize)
    {
        LocalTensor<float> cL0Tensor = L0_C.Get<float>()[l0c_iduse * 128 * 256];
        LocalTensor<mx_fp8_e4m3_t> aL0Tensor = L0_A.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];
        LocalTensor<mx_fp8_e4m3_t> bL0Tensor = L0_B.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];

        MmadParams mmadParams;
        mmadParams.m = 32;
        mmadParams.n = s2CalcSize; // s2: N维度
        mmadParams.k = 128;
        mmadParams.cmatrixInitVal = 1;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;

        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
    }
    __aicore__ inline void l0c_ub_mm1(int mm1_id, int l0c_iduse, uint32_t s2CalcSize){
        LocalTensor<float> dstTensor = UB_MM1Res.Get<float>()[mm1_id * 64 * 256];
        LocalTensor<float> srcTensor = L0_C.Get<float>()[l0c_iduse * 128 * 256];

        uint32_t s2CalcSizeAlign8 = (s2CalcSize + 7) / 8 * 8;

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = s2CalcSizeAlign8;
        fixpipeParams.mSize = 32;
        fixpipeParams.srcStride = 32;
        fixpipeParams.dstStride = 256;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<float, float, PFA_CFG_ROW_MAJOR_UB>(dstTensor, srcTensor, fixpipeParams);
    }

    __aicore__ inline void mm1(uint64_t inner, uint64_t outer, uint64_t bIdx, uint64_t n2Idx, uint32_t subLoop,
                               uint32_t actSingleLoopS2Size, bool isFirstS2Loop, bool isLastS2Loop)
    {
        uint32_t s2CalcSize;
        if (actSingleLoopS2Size <= s2SplitSize) {
            s2CalcSize = actSingleLoopS2Size;
        } else {
            s2CalcSize = (subLoop == 0) ? s2SplitSize : (actSingleLoopS2Size - s2SplitSize);
        }
        uint32_t c1v1Loop = (actSingleLoopS2Size + s2SplitSize - 1) / s2SplitSize;
        bool isLastSubLoop = (subLoop + 1 == c1v1Loop);

        if (isFirstS2Loop && subLoop == 0){
            iq ++;
            uint64_t q1use = iq % 2;
            iquse =  4 + q1use;
            wait_flag(PIPE_MTE1, PIPE_MTE2, iquse);
            gm_l1_q(outer, bIdx, n2Idx, q1use);
            gm_l1_qscale(outer, bIdx, n2Idx, q1use);
            set_flag(PIPE_MTE2, PIPE_MTE1, iquse);
        } else {
            set_flag(PIPE_MTE2, PIPE_MTE1, iquse);
        }
        ikv++;
        int ivkuse = ikv % 4;
        wait_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
        gm_l1_k(inner, bIdx, n2Idx, subLoop, ivkuse, s2CalcSize);
        gm_l1_kscale(inner, bIdx, n2Idx, subLoop, ivkuse, s2CalcSize);
        set_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
        wait_flag(PIPE_MTE2, PIPE_MTE1, iquse);
        wait_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
        
        l0c_id ++ ;
        int l0c_iduse = l0c_id % 2;
        wait_flag(PIPE_FIX, PIPE_M, l0c_iduse);

        l0a_id ++ ;
        int l0a_iduse = l0a_id % 2 + 4;
        wait_flag(PIPE_M, PIPE_MTE1, l0a_iduse);
        mm1_l1_l0a_no_trans(l0a_iduse);
        mm1_l1_l0b_no_trans(ivkuse, l0a_iduse, s2CalcSize);
        set_flag(PIPE_MTE1, PIPE_M, l0a_iduse);
        wait_flag(PIPE_MTE1, PIPE_M, l0a_iduse);

        compute_mm1(l0c_iduse, l0a_iduse, s2CalcSize);
        set_flag(PIPE_M, PIPE_MTE1, l0a_iduse);

        if (isLastS2Loop && isLastSubLoop) {
            set_flag(PIPE_MTE1, PIPE_MTE2, iquse);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);

        set_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        wait_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        bmm1_id ++;
        int bmm1_idused = 6 + (bmm1_id % 2) * 2;
        wait_intra_block(PIPE_FIX, 0 + bmm1_idused + 1);
        wait_intra_block(PIPE_FIX, 16 + bmm1_idused + 1);
        mm1_id ++;
        l0c_ub_mm1(mm1_id % 2, l0c_iduse, s2CalcSize);
        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);
        set_intra_block(PIPE_FIX, 0 + bmm1_idused + 0);
        set_intra_block(PIPE_FIX, 16 + bmm1_idused + 0);

    }

    __aicore__ inline void gm_l1_v(uint64_t inner, uint64_t bIdx, uint64_t n2Idx, uint32_t k, int ivkuse,
                                    uint32_t s2RealSize)
    {
        constexpr uint32_t mm2VSize = 128 * 256;
        constexpr uint32_t mm2VScaleSize = 128 * 256 / 32;
        constexpr uint32_t singleVBufferOffset = mm2VSize + mm2VScaleSize;

        LocalTensor<float8_e4m3_t> vL1Tensor =
            L1_KV.Get<float8_e4m3_t>()[ivkuse * singleVBufferOffset];

        uint32_t s2Offset = inner * s2BaseSize + k * s2SplitSize;
        uint64_t blockId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * constInfo.maxBlockNumPerBatch +
                                s2Offset / s2BaseSize);
        uint64_t gmVOffset = blockId * s2BaseSize * 8 * 128 + n2Idx * s2BaseSize * 128 + (s2Offset % s2BaseSize) * 32;

        GlobalTensor<float8_e4m3_t> valueGmTensor;
        valueGmTensor.SetGlobalBuffer((__gm__ float8_e4m3_t *)(valueGm + gmVOffset));

        uint32_t s2RealSizeAlign64 = (s2RealSize + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE; // s2: L1行步长64对齐(匹配mm2 LoadData)

        DataCopyParams intriParams;
        intriParams.blockCount = 4;
        intriParams.blockLen = s2RealSize; // s2: 实际行数
        intriParams.dstStride = s2RealSizeAlign64 - s2RealSize; // s2: L1填充
        intriParams.srcStride = s2BaseSize - s2RealSize; // s2: GM跳转步长

        DataCopy(vL1Tensor, valueGmTensor, intriParams);
    }

     __aicore__ inline void gm_l1_vscale(uint64_t inner, uint64_t bIdx, uint64_t n2Idx, uint32_t k, int ivkuse,
                                          uint32_t s2RealSize)
    {
        constexpr uint32_t mm2VSize = 128 * 256;
        constexpr uint32_t mm2VScaleSize = 128 * 256 / 32;
        constexpr uint32_t singleVBufferOffset = mm2VSize + mm2VScaleSize;

        LocalTensor<float8_e8m0_t> vScaleL1Tensor =
            L1_KV.Get<float8_e8m0_t>()[ivkuse * singleVBufferOffset + mm2VSize];

        uint32_t s2VscaleScaleOffset = (inner * s2BaseSize + k * s2SplitSize) / MXFP_GROUP_SIZE;
        uint64_t blockVscaleIdxInBatch = s2VscaleScaleOffset / 8;
        uint64_t bsVscaleIdx = s2VscaleScaleOffset % 8;
        uint64_t blockVscaleId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * constInfo.maxBlockNumPerBatch +
                                blockVscaleIdxInBatch);
        uint64_t gmVScaleOffset = blockVscaleId * 32 * 8 * 8 * 8 + n2Idx * 32 * 8 * 8 + bsVscaleIdx * 32;

        GlobalTensor<float8_e8m0_t> vScaleGmTensor;
        vScaleGmTensor.SetGlobalBuffer((__gm__ float8_e8m0_t *)(deScaleVGm + gmVScaleOffset));

        // V Scale走PA_NZ格式(与K/V Data相同，非PA_NZ_K_SCALE)
        // V Scale的L1行步长不对齐，参考CopyValueScaleSlice: rowCount = s2RealSize
        DataCopyParams intriParams;
        intriParams.blockCount = 4;
        intriParams.blockLen = s2RealSize; // s2: 实际行数
        intriParams.dstStride = 0; // s2: 不对齐，V Scale的LoadData不依赖L1行步长
        intriParams.srcStride = s2BaseSize - s2RealSize; // s2: GM跳转步长

        DataCopy(vScaleL1Tensor, vScaleGmTensor, intriParams);
    }

    __aicore__ inline void mm2_l1_l0a_no_trans(uint32_t k, int ipuse, int l0a_iduse, uint32_t realK)
    {
        constexpr uint32_t mm2LeftSize = 128 * 512;

        LocalTensor<mx_fp8_e4m3_t> aL0Tensor = L0_A.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];
        LocalTensor<fp8_e4m3fn_t> aL1Tensor = L1_P.Get<fp8_e4m3fn_t>()[ipuse * mm2LeftSize + k * 256 * 128];
        LocalTensor<float8_e8m0_t> aScaleL1Tensor = L1_PScale.Get<float8_e8m0_t>();

        constexpr uint32_t mSplitSize = 32;  // P的M维度=s1
        uint32_t kSplitSize = (realK + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE; // s2: 64对齐匹配MmadParams.k

        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.ifTranspose = false;
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsA.kStep = ((kSplitSize + 31) >> 5 << 5) >> 5; // s2: K方向32B块数
        loadData2DParamsA.srcStride = ((mSplitSize + 31) >> 5 << 5) >> 4;
        loadData2DParamsA.dstStride = loadData2DParamsA.mStep;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = 0;
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsA.yStep = (kSplitSize + 63) >> 5 >> 1; // s2: Scale Y方向块数=kSplitSize/64
        loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
        loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;

        LoadData(aL0Tensor, aL1Tensor, aScaleL1Tensor, loadData2DParamsA, loadData2DMxParamsA);
    }
    __aicore__ inline void mm2_l1_l0b_trans(int ivkuse, int l0b_iduse, uint32_t s2RealSize)
    {
        constexpr uint32_t mm2VSize = 128 * 256;
        constexpr uint32_t mm2VScaleSize = 128 * 256 / 32;
        constexpr uint32_t singleVBufferOffset = mm2VSize + mm2VScaleSize;

        LocalTensor<mx_fp8_e4m3_t> bL0Tensor = L0_B.Get<mx_fp8_e4m3_t>()[(l0b_iduse % 2) * 32 * 1024];
        LocalTensor<fp8_e4m3fn_t> bL1Tensor = L1_KV.Get<fp8_e4m3fn_t>()[ivkuse * singleVBufferOffset];
        LocalTensor<float8_e8m0_t> bScaleL1Tensor = L1_KV.Get<float8_e8m0_t>()[ivkuse * singleVBufferOffset + mm2VSize];

        // V矩阵(B矩阵)转置: N×K = d×s2 → ifTranspose=true, M方向=s2, K方向=d
        uint32_t kSplitSize = (s2RealSize + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE; // s2: 64对齐匹配MmadParams.k
        constexpr uint32_t nSplitSize = 128;

        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.ifTranspose = true;
        loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // s2: M方向分形数
        loadData2DParamsB.kStep = ((nSplitSize + 31) >> 5 << 5) >> 5;
        loadData2DParamsB.srcStride = ((kSplitSize + 31) >> 5 << 5) >> 4; // s2: 32对齐匹配L1行步长
        loadData2DParamsB.dstStride = (nSplitSize + 15) >> 4;

        LoadData2DMxParams loadData2DMxParamsB;
        loadData2DMxParamsB.xStartPosition = 0;
        loadData2DMxParamsB.yStartPosition = 0;
        loadData2DMxParamsB.xStep = ((nSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsB.yStep = (kSplitSize + 63) >> 5 >> 1; // s2: Scale Y方向块数=kSplitSize/64
        loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
        loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep;

        LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB, loadData2DMxParamsB);
    }

    __aicore__ inline void compute_mm2(int l0c_iduse, int l0a_iduse, uint32_t k, uint32_t realK)
    {
        LocalTensor<float> cL0Tensor = L0_C.Get<float>()[l0c_iduse * 128 * 256];
        LocalTensor<mx_fp8_e4m3_t> aL0Tensor = L0_A.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];
        LocalTensor<mx_fp8_e4m3_t> bL0Tensor = L0_B.Get<mx_fp8_e4m3_t>()[(l0a_iduse - 4) * 32 * 1024];

        // mm2: P(A) × V(B), M×N×K = s1×d×s2
        uint32_t realKAlign64 = (realK + 63) / 64 * 64; // s2: K维度64对齐(MX量化要求)

        MmadParams mmadParams;
        mmadParams.m = 32;
        mmadParams.n = 128;
        mmadParams.k = realKAlign64; // s2: 64对齐
        mmadParams.cmatrixInitVal = (k == 0) ? 1 : 0;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;

        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
    }
    __aicore__ inline void l0c_ub_mm2(int l0c_iduse)
    {
        LocalTensor<float> dstTensor = UB_MM2Res.Get<float>();
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

        Fixpipe<float, float, PFA_CFG_ROW_MAJOR_UB>(dstTensor, srcTensor, fixpipeParams);
    }

    __aicore__ inline void mm2(int ipuse, uint64_t inner, uint64_t outer, uint64_t bIdx, uint64_t n2Idx,
                               uint32_t actSingleLoopS2Size)
    {
        wait_intra_block(PIPE_MTE1, ipuse);
        wait_intra_block(PIPE_MTE1, ipuse + 16);

        l0c_id ++ ;
        int l0c_iduse = l0c_id % 2;
        wait_flag(PIPE_FIX, PIPE_M, l0c_iduse);

        uint32_t kLoops = (actSingleLoopS2Size + s2SplitSize - 1) / s2SplitSize;
        for (uint32_t k = 0; k < kLoops; k++) {
            uint32_t realK = s2SplitSize;
            if (k == kLoops - 1) {
                realK = actSingleLoopS2Size - k * s2SplitSize;
            }

            ikv++;
            int ivkuse = ikv % 4;
            wait_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
            gm_l1_v(inner, bIdx, n2Idx, k, ivkuse, realK);
            gm_l1_vscale(inner, bIdx, n2Idx, k, ivkuse, realK);
            set_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);

            wait_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
            l0a_id ++ ;
            int l0a_iduse = l0a_id % 2 + 4;
            wait_flag(PIPE_M, PIPE_MTE1, l0a_iduse);
            mm2_l1_l0a_no_trans(k, ipuse, l0a_iduse, realK);
            mm2_l1_l0b_trans(ivkuse, l0a_iduse, realK);
            set_flag(PIPE_MTE1, PIPE_M, l0a_iduse);
            wait_flag(PIPE_MTE1, PIPE_M, l0a_iduse);

            compute_mm2(l0c_iduse, l0a_iduse, k, realK);

            set_flag(PIPE_M, PIPE_MTE1, l0a_iduse);

            set_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
        }

        set_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        wait_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        l0c_ub_mm2(l0c_iduse);
        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);

    }
    __aicore__ inline void ub_l1_vec1(uint32_t subLoop, int ipuse, int vec1_idused)
    {
        int64_t dstOffset =
                constInfo.subBlockIdx * 32 * 16 +
                256 * 128 * subLoop;
        constexpr uint32_t mm2LeftSize = 128 * 512;
        constexpr uint32_t stage1OutQueSize = 16640;
        
        LocalTensor<float8_e4m3_t> dstTensor = L1_P.Get<float8_e4m3_t>()[ipuse * mm2LeftSize + dstOffset];
        LocalTensor<float8_e4m3_t> srcTensor = UB_STAGE1OutQue.Get<float8_e4m3_t>()[vec1_idused * stage1OutQueSize];
        
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 8;
        dataCopyParams.blockLen = 16;
        dataCopyParams.srcStride = 49;
        dataCopyParams.dstStride = 112;
        
        DataCopy(dstTensor, srcTensor, dataCopyParams);
    }

    __aicore__ inline bool IsSkipAttentionmask(uint32_t gs1StartIdx, uint32_t s2StartIdx, uint32_t s2CalcSize)
    {
        int32_t s1StartIdx = gs1StartIdx / 8;

        if (static_cast<int64_t>(s2StartIdx + s2CalcSize) <= static_cast<int64_t>(s1StartIdx) + (actSeqLensKv - actSeqLensQ)) {
            return true;
        }
        return false;
    }
    __aicore__ inline uint64_t ComputeAttenMaskOffset(uint32_t s1StartIdx, uint32_t s2StartIdx)
    {
        int64_t nextToken = 0;

        uint64_t offset = 0;
        int64_t delta = (actSeqLensKv - actSeqLensQ) + s1StartIdx - s2StartIdx;
        uint32_t attenMaskSizeAlign = 256;

        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                    2048;

        return offset;
    }
    __simd_vf__ void MaskUbCopyS1G(__ubuf__ uint8_t * maskUb, uint16_t tailGLoop, uint16_t midS1Count)
    {
        constexpr uint32_t repeatStride = 8;//256 >> 5

        RegTensor<uint8_t> vreg_g_head;
        RegTensor<uint8_t> vreg_g_mid;
        RegTensor<uint8_t> vreg_g_tail;

        MaskReg preg_all;
        preg_all = CreateMask<bool, MaskPattern::ALL>();

        // for (uint16_t x = 0; x > 0; x = 0) {     // if (headGLoop > 0) {}
        //     LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        //         vreg_g_head, maskUb, 1, repeatStride, preg_all);
        //     for (uint16_t i = 1; i < 0; ++i) {
        //         StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        //             maskUb, vreg_g_head, 1, repeatStride, preg_all);
        //     }
        // }

        for (uint16_t x = 8; x > 0; x = 0) {     // if (midGLoop > 0) {}
            for (uint16_t i = 0; i < midS1Count; ++i) {
                LoadAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vreg_g_mid, maskUb, 1, repeatStride, preg_all);
                for (uint16_t j = 1; j < 8; ++j) {
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
    __aicore__ inline void AttentionmaskCopyInForSgLayout(uint32_t gs1StartIdx, uint32_t s2StartIdx)
    {
        uint32_t s1StartIdx = gs1StartIdx / 8;
        uint32_t s1EndIdx = (gs1StartIdx + 16 - 1) / 8;
        uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
        uint32_t headGSize = 16;
        if (s1Count > 1) {
            headGSize = (gs1StartIdx % 8 == 0) ? 0 : (8 - gs1StartIdx % 8);
        }
        uint32_t remainRowCount = 16 - headGSize;
        uint32_t midS1Count = remainRowCount / 8;
        uint32_t tailGSize = remainRowCount % 8;
        uint32_t attenMaskSizeAlign = 256;
        uint32_t attenMaskS2Stride = attenMaskSizeAlign;

        // // ub-head
        // if (headGSize > 0) {
        //     printf("headGSize > 0 = %d\n",headGSize);
        //     AttentionmaskDataCopy(attenMaskUB, srcGmAddr, info, s1StartIdx, s1StartIdx + 1, isPre);
        //     s1StartIdx++;
        // }
        LocalTensor<uint8_t> attenMaskTensor = UB_AttenMask.Get<uint8_t>()[headGSize * attenMaskS2Stride];
        // // ub-remain
        if (remainRowCount > 0) {
            uint64_t maskOffset = ComputeAttenMaskOffset(s1StartIdx, s2StartIdx);
            GlobalTensor<uint8_t> attenMaskGmTensor;
            attenMaskGmTensor.SetGlobalBuffer((__gm__ uint8_t*)(attenMaskGmInt + maskOffset));  
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 2;
            dataCopyParams.blockLen = 256;
            dataCopyParams.srcStride = 1792;
            dataCopyParams.dstStride = 56;
            // blockLen + padParams.len + dstStride = gSize * s2Size
            DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 1U};//isPad,leftPadding,rightPadding,paddingValue
            DataCopyPad(attenMaskTensor, attenMaskGmTensor, dataCopyParams, padParams);
        }
        set_flag(PIPE_MTE2, PIPE_V, event_t(0));
        wait_flag(PIPE_MTE2, PIPE_V, event_t(0));
        MaskUbCopyS1G((__ubuf__ uint8_t*)UB_AttenMask.Get<uint8_t>().GetPhyAddr(), tailGSize, midS1Count);
    }
    __aicore__ inline void AttenMaskCopyIn(uint64_t inner, uint32_t subLoop, uint32_t s2CalcSize)
    {
        LocalTensor<uint8_t> attenMaskTensor = UB_AttenMask.Get<uint8_t>();
        uint64_t gs1StartIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            gs1StartIdx = 16;
        }
        uint64_t s2StartIdx = (subLoop == 0) ? inner * s2BaseSize : (inner * s2BaseSize + s2SplitSize);
        bool IsSkipMask = IsSkipAttentionmask(gs1StartIdx, s2StartIdx, s2CalcSize);
        if (IsSkipMask) {
            Duplicate(attenMaskTensor, static_cast<uint8_t>(0U), 16 * 256);
            return;
        } else {
            AttentionmaskCopyInForSgLayout(gs1StartIdx, s2StartIdx);
        }
    }
    __aicore__ inline void vec1(int ipuse, int vec1_idused, uint64_t inner, uint64_t outer, uint32_t subLoop,
                                uint32_t actSingleLoopS2Size)
    {
        uint32_t s2CalcSize;
        if (actSingleLoopS2Size <= s2SplitSize) {
            s2CalcSize = actSingleLoopS2Size;
        } else {
            s2CalcSize = (subLoop == 0) ? s2SplitSize : (actSingleLoopS2Size - s2SplitSize);
        }
        uint32_t c1v1Loop = (actSingleLoopS2Size + s2SplitSize - 1) / s2SplitSize;
        bool isLastSubLoop = (subLoop + 1 == c1v1Loop);

        bmm1_id ++;
        int bmm1_idused = 6 + (bmm1_id % 2) * 2;
        wait_intra_block(PIPE_V, 0 + bmm1_idused + 0);
        AttenMaskCopyIn(inner, subLoop, s2CalcSize);

        set_intra_block(PIPE_V, 0 + bmm1_idused + 1);

        ub_l1_vec1(subLoop, ipuse, vec1_idused);
        if (isLastSubLoop) {
            set_intra_block(PIPE_MTE3, ipuse);
        }

    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != constInfo.bN2End) || (gS1Cur != constInfo.gS1OEnd) || (s2Cur != constInfo.s2OEnd));
    }

    __aicore__ inline void UpdateAxisInfo(uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur)
    {
        // 当前S2未处理完
        if (s2Cur + 1 < inner_loop_limit) {
            s2Cur++;
            return;
        }
        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < outer_loop_limit) {
            gS1Cur++;
            return;
        }
        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline bool ShouldExecuteTask(bool runInfo[3])
    {
        for (uint32_t i = 0; i < 3; i++) {
            if (runInfo[i] == true) {
                return true;
            }
        }
        return false;
    }
    __aicore__ inline void Process()
    {
        CrossCoreBufferInit();
        if ASCEND_IS_AIC {
            prefree_kv_l1();
            prefree_l0ab();
            prefree_l0c();
        }

        uint64_t id = 0;
        uint64_t inner = 0, outer = 0;
        uint64_t outerp[3], innerp[3];
        uint64_t bIdxp[3], n2Idxp[3];
        uint32_t actSingleLoopS2Sizep[3];
        bool isFirstS2Loopp[3];
        bool isLastS2Loopp[3];
        bool runInfo[3] = {true, true, true};
        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        uint32_t bN2Cur = constInfo.bN2Start;
        uint32_t gS1Cur = 0;
        uint32_t s2Cur = 0;
        uint32_t currentActSingleLoopS2Size = s2BaseSize;
        bool currentIsFirstS2Loop = true;
        bool currentIsLastS2Loop = false;
        while (shouldDispatchTask || shouldExecuteTask) {
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            int64_t bIdx = bN2Cur / 8;
            int64_t n2Idx = bN2Cur % 8;
            if (shouldDispatchTask){
                inner = s2Cur;
                outer = gS1Cur;
                currentActSingleLoopS2Size = s2BaseSize;
                if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
                    currentActSingleLoopS2Size = actSeqLensKv - s2Cur * s2BaseSize;
                }
                currentIsFirstS2Loop = (s2Cur == 0);
                currentIsLastS2Loop = (s2Cur + 1 == inner_loop_limit);
                UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                runInfo[id % 3] = true;
            }
            shouldExecuteTask = ShouldExecuteTask(runInfo);
            if (shouldExecuteTask) {
                bIdxp[id % 3] = bIdx;
                n2Idxp[id % 3] = n2Idx;
                outerp[id % 3] = outer;
                innerp[id % 3] = inner;
                actSingleLoopS2Sizep[id % 3] = currentActSingleLoopS2Size;
                isFirstS2Loopp[id % 3] = currentIsFirstS2Loop;
                isLastS2Loopp[id % 3] = currentIsLastS2Loop;
                int ipuse = id % 3;
                if (runInfo[id % 3] == true) {
                    uint32_t c1v1Loop = (actSingleLoopS2Sizep[id % 3] + s2SplitSize - 1) / s2SplitSize;
                    for (uint32_t subLoop = 0; subLoop < c1v1Loop; ++subLoop) {
                        if ASCEND_IS_AIC {
                            mm1(inner, outer, bIdx, n2Idx, subLoop,
                                actSingleLoopS2Sizep[id % 3], isFirstS2Loopp[id % 3], isLastS2Loopp[id % 3]);
                        } else {
                            vec1(ipuse, subLoop % 2, inner, outer, subLoop, actSingleLoopS2Sizep[id % 3]);
                        }
                    }
                }
                if (id >= 2) {
                    if (runInfo[(id - 2) % 3] == true) {
                        if ASCEND_IS_AIC {
                            mm2((id - 2) % 3, innerp[(id - 2) % 3], outerp[(id - 2) % 3],
                                bIdxp[(id - 2) % 3], n2Idxp[(id - 2) % 3], actSingleLoopS2Sizep[(id - 2) % 3]);
                        }
                        runInfo[(id - 2) % 3] = false;
                    }
                }
                id ++;
            }
        }
        if ASCEND_IS_AIC {
            postalloc_kv_l1();
            postalloc_l0ab();
            postalloc_l0c();
        }
        CrossCoreBufferUnInit();
    }
};  // FlashAttentionFullQuantMxKernel

}  // namespace BaseApi

#endif  // FIA_KERNEL_FULLQUANT_MX_H_