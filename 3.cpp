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

#ifndef FIA_KERNEL_FULLQUANT_MX_FIXP_H_
#define FIA_KERNEL_FULLQUANT_MX_FIXP_H_

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
    __gm__ uint32_t *fiaMetaDataGm;
    __cbuf__ void *q1[2], *kv1[4], *p1[3], *pscale[1];
    __ubuf__ void *pv[2];
    int ikv{-1};
    int iq{-1};
    int iquse = -1;
    int ipid{-1};
    __ca__ void * l0a[2];
    __cb__ void * l0b[2];
    __cc__ void * l0c[2];
    int l0a_id{-1};
    int l0c_id{-1};
    __ubuf__ void *mm1Res[2], *mm2Res[1];
    int mm1_id{-1};

    uint64_t inner_loop_limit;
    uint64_t outer_loop_limit;
    __aicore__ inline FlashAttentionFullQuantMxKernel(){};

    __aicore__ inline void InitBuffer()
    {
        constexpr uint32_t mm1QSize = 128 * 128;
        constexpr uint32_t mm1QScaleSize = 128 * 128 / 32;
        constexpr uint32_t mm1KVSize = 128 * 256;
        constexpr uint32_t mm1KVScaleSize = 128 * 256 / 32;
        constexpr uint32_t mm2LeftSize = 128 * 512;
        constexpr uint32_t pScaleSize = 128 * 512 / 32;

        p1[0] = (__cbuf__ void *)get_imm(0);
        p1[1] = (__cbuf__ void *)get_imm(mm2LeftSize);
        p1[2] = (__cbuf__ void *)get_imm(mm2LeftSize * 2);
        pscale[0] = (__cbuf__ void *)get_imm(mm2LeftSize * 3);

        q1[0] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + pScaleSize);
        q1[1] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + pScaleSize + mm1QSize + mm1QScaleSize);
        kv1[0] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + pScaleSize + (mm1QSize + mm1QScaleSize) * 2);
        kv1[1] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + pScaleSize + (mm1QSize + mm1QScaleSize) * 2  + mm1KVSize + mm1KVScaleSize);
        kv1[2] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + pScaleSize + (mm1QSize + mm1QScaleSize) * 2  + (mm1KVSize + mm1KVScaleSize) * 2);
        kv1[3] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + pScaleSize + (mm1QSize + mm1QScaleSize) * 2  + (mm1KVSize + mm1KVScaleSize) * 3);
        l0a[0] = (__ca__ void *)get_imm(0); 
        l0a[1] = (__ca__ void *)get_imm(32 * 1024);
        l0b[0] = (__cb__ void *)get_imm(0);
        l0b[1] = (__cb__ void *)get_imm(32 * 1024);
        l0c[0] = (__cc__ void *)get_imm(0);
        l0c[1] = (__cc__ void *)get_imm(128 * 256 * 4);//sizeof(fp32)

        constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
        constexpr uint32_t mm2ResultSize = 128 / 2 * 128 * 4;
        mm1Res[0] = (__ubuf__ void *)get_imm(0); 
        mm1Res[1] = (__ubuf__ void *)get_imm(mm1ResultSize);
        mm2Res[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2);

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
        constInfo.s2Size = 10240;
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
        // gs1,现在固定case是b=48,g=8,qs=4,kvs=10240  ,qn=64,kvn=8
        inner_loop_limit = (10240 >> 9); // constInfo.s2OEnd
        outer_loop_limit = (32 + 127) >> 7; // constInfo.gS1OEnd
        
        InitConstInfo();
        queryGm = query;
        keyPtr = key;
        valuePtr = value;
        deScaleQGm = dequantScaleQuery;
        deScaleKGm = dequantScaleKey;
        deScaleVGm = dequantScaleValue;
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        keyGm = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        valueGm = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        blockTableGm = (__gm__ int32_t *)blockTable;

        InitBuffer();
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
        set_flag(PIPE_M, PIPE_MTE1, (event_t)0);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)1);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)2);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)3);
    }

    __aicore__ inline void postalloc_l0ab(){
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)0);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)1);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)2);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)3);
    }

    __aicore__ inline void postalloc_l0c(){
        wait_flag(PIPE_FIX, PIPE_M, (event_t)0);
        wait_flag(PIPE_FIX, PIPE_M, (event_t)1);
    }

    __aicore__ inline void gm_l1_q(uint64_t outer, uint64_t bIdx, uint64_t n2Idx, uint64_t q1use)
    {
        uint64_t gmQOffset = (bIdx * 4 + outer *  128 / 8) * 8 * 128 * 8 + n2Idx * 8 * 128;
     
        int64_t mte2Q2NzPara = static_cast<uint64_t>(static_cast<uint16_t>(32)) << 48; // MTE2_NZ_PARA[63:48]
        mte2Q2NzPara |= static_cast<uint64_t>(32) << 32;         // MTE2_NZ_PARA[47:32]
        mte2Q2NzPara |= static_cast<uint64_t>(1) << 16;         // MTE2_NZ_PARA[31:16]
        mte2Q2NzPara |= static_cast<uint64_t>(4);            // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2Q2NzPara); // CCE: store parameters for ND2NZ DMA instructions
        copy_gm_to_cbuf_multi_nd2nz((__cbuf__ float8_e4m3_t *)q1[q1use], (__gm__ float8_e4m3_t *)(queryGm + gmQOffset), 0, 128, 0, 8, 128, 8 * 128 * 8, false);
    }

    __aicore__ inline void gm_l1_qscale(uint64_t outer, uint64_t bIdx, uint64_t n2Idx, uint64_t q1use)
    {
        uint32_t qScaleOffset = 32 * 128;//((runInfo.actMSize + 31) >> 5 << 5) * constInfo.dSize;runInfo.actMSize=gs1=8*4
        uint64_t gmQScaleOffset = (bIdx * 4 + outer * 128 / 8) * 4 * 8 + n2Idx * 4 * 8 * (1 + 48) * 48 / 2 * 4;
        // Dn2NzParams dn2nzPara;
        // dn2nzPara.dnNum = 1;                 // ND矩阵的个数
        // dn2nzPara.nValue = 2;       // 单个DN矩阵的实际列数，单位为元素个数
        // dn2nzPara.dValue = 32;           // 单个DN矩阵的实际行数，单位为元素个数
        // dn2nzPara.srcDnMatrixStride = 0;     // 相邻Dn矩阵起始地址之间的偏移， 单位为元素个数
        // dn2nzPara.srcDValue = 2; // 同一个Dn矩阵中相邻行起始地址之间的偏移， 单位为元素个数
        // dn2nzPara.dstNzC0Stride = 2;
        // dn2nzPara.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
        // dn2nzPara.dstNzMatrixStride = 2;

        uint64_t mte2QScaleNzPara = static_cast<uint64_t>(2 * 2 / 32) << 48; // MTE2_NZ_PARA[63:48]
        mte2QScaleNzPara |= static_cast<uint64_t>(2) << 32;         // MTE2_NZ_PARA[47:32]
        mte2QScaleNzPara |= static_cast<uint64_t>(1) << 16;         // MTE2_NZ_PARA[31:16]
        mte2QScaleNzPara |= static_cast<uint64_t>(1);            // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2QScaleNzPara); // CCE: store parameters for DN2NZ DMA instructions
        copy_gm_to_cbuf_multi_dn2nz((__cbuf__ bfloat16_t *)((__cbuf__ float8_e8m0_t *)q1[q1use] + qScaleOffset), 
                                    (__gm__ bfloat16_t *)((__gm__ float8_e8m0_t *)deScaleQGm + gmQScaleOffset), 0, 2 * 2, 0, 2, 32, 0, false);
    }

    __aicore__ inline void gm_l1_k(uint64_t inner, uint64_t bIdx, uint64_t n2Idx, uint32_t subLoop, int ivkuse)
    {
        uint32_t s2Offset = (subLoop % 2 == 0) ? inner * 512 : inner * 512 + 256;
        uint64_t blockIdxInBatch = s2Offset / 512;  // 获取block table上的索引
        uint64_t bsIdx = s2Offset % 512;            // 获取在单个块上超出的行数
        uint64_t blockId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * constInfo.maxBlockNumPerBatch +
                                blockIdxInBatch);
        uint64_t gmKOffset = blockId * 512 * 8 * 128 + n2Idx * 512 * 128 + bsIdx * 32;

        // DataCopyParams intriParams;
        // intriParams.blockCount = 4;
        // intriParams.blockLen = 256;
        // intriParams.dstStride = 0;
        // intriParams.srcStride = 256;
        /* copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t*)dst, (__gm__ uint8_t*)src, 0, blockCount, burstLength,
                                     leftPaddingCnt, rigntPaddingCnt, true, cacheMode, actSrcStride, actDstStride); */
        copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t *)kv1[ivkuse], (__gm__ uint8_t *)(keyGm + gmKOffset), 0, 4,
                                 256 * 32, 0, 0, true, 0, 256 * 32 * 2, 8192);
    }

    __aicore__ inline void gm_l1_kscale(uint64_t inner, uint64_t bIdx, uint64_t n2Idx, uint32_t subLoop, int ivkuse)
    {
        uint32_t s2Offset = (subLoop % 2 == 0) ? inner * 512 : inner * 512 + 256;
        uint64_t bsIdx = s2Offset % 512;            // 获取在单个块上超出的行数
        // uint32_t kScaleOffset = 256 * 128;
        
        // intriParams.blockCount = 16;修改成1
        // intriParams.blockLen = 2;修改成32
        // intriParams.dstStride = 0;
        // intriParams.srcStride = 0;
        uint64_t blockKScaleId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * constInfo.maxBlockNumPerBatch +
                                s2Offset / 512);
        uint64_t gmKScaleOffset = blockKScaleId * 512 * 64 * 8 + n2Idx * 512 * 64 + bsIdx / 16 * 64 + bsIdx % 16;
        copy_gm_to_cbuf_align_v2((__cbuf__ float8_e8m0_t *)kv1[ivkuse] + 256 * 128, (__gm__ float8_e8m0_t *)(deScaleKGm + gmKScaleOffset), 0, 1,
                                 32 * 32, 0, 0, true, 0, 0, 1024);
    }

    __aicore__ inline void mm1_l1_l0a_no_trans(int l0a_iduse)
    {
        // loadData2DParamsA.mStartPosition = 0
        // loadData2DParamsA.kStartPosition = 0
        // loadData2DParamsA.ifTranspose = 0
        // loadData2DParamsA.mStep = 2
        // loadData2DParamsA.kStep = 4
        // loadData2DParamsA.srcStride = 2
        // loadData2DParamsA.dstStride = 2

        // loadData2DMxParamsA.xStartPosition = 0
        // loadData2DMxParamsA.yStartPosition = 0
        // loadData2DMxParamsA.xStep = 2
        // loadData2DMxParamsA.yStep = 2
        // loadData2DMxParamsA.srcStride =2
        // loadData2DMxParamsA.dstStride = 2
        // LoadData(aL0Tensor, aL1Tensor[L1Aoffset], aScaleL1Tensor[L1Aoffset >> 5], loadData2DParamsA,
        //              loadData2DMxParamsA);
        load_cbuf_to_ca(reinterpret_cast<__ca__ float8_e4m3_t *>((__ca__ mx_fp8_e4m3_t *)l0a[l0a_iduse]),
                    (__cbuf__ float8_e4m3_t *)q1[iquse % 2],
                    0,
                    0,
                    2,
                    4,
                    2,
                    2,
                    0);
        uint32_t qScaleOffset = 32 * 128;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__ca__ mx_fp8_e4m3_t *)l0a[l0a_iduse])) / 16;
        load_cbuf_to_ca_mx(mxDstAddr,
            static_cast<__cbuf__ void *>((__cbuf__ float8_e8m0_t *)q1[iquse % 2] + qScaleOffset),
            0,
            0,
            2,
            2,
            2,
            2);
        
    }
    __aicore__ inline void mm1_l1_l0b_no_trans(int ivkuse, int l0b_iduse)
    {
         // loadData2DParamsB.mStartPosition = 0
        // loadData2DParamsB.kStartPosition = 0
        // loadData2DParamsB.ifTranspose = 0
        // loadData2DParamsB.mStep = 16
        // loadData2DParamsB.kStep = 4
        // loadData2DParamsB.srcStride = 16
        // loadData2DParamsB.dstStride = 16

        // loadData2DMxParamsB.xStartPosition = 0
        // loadData2DMxParamsB.yStartPosition = 0
        // loadData2DMxParamsB.xStep = 16
        // loadData2DMxParamsB.yStep = 2
        // loadData2DMxParamsB.srcStride =2
        // loadData2DMxParamsB.dstStride = 2
        load_cbuf_to_cb(reinterpret_cast<__cb__ float8_e4m3_t *>((__cb__ mx_fp8_e4m3_t *)l0b[l0b_iduse]),
                    (__cbuf__ float8_e4m3_t *)kv1[ivkuse],
                    0,
                    0,
                    16,
                    4,
                    16,
                    16,
                    0);

        uint32_t kScaleOffset = 256 * 128;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__cb__ mx_fp8_e4m3_t *)l0b[l0b_iduse])) / 16;
        load_cbuf_to_cb_mx(mxDstAddr,
            static_cast<__cbuf__ void *>((__cbuf__ float8_e8m0_t *)kv1[ivkuse] + kScaleOffset),
            0,
            0,
            16,
            2,
            2,
            2);
    }

    __aicore__ inline void compute_mm1(int l0c_iduse, int l0a_iduse)
    {
        // MmadParams mmadParams;
        // mmadParams.m = 32;
        // mmadParams.n = 256;
        // mmadParams.k = 128;
        // mmadParams.cmatrixInitVal =1;
        // mmadParams.cmatrixSource = 0;
        // mmadParams.unitFlag = 0;
        using AType = typename GetDstType<mx_fp8_e4m3_t>::Type;
        using BType = typename GetDstType<mx_fp8_e4m3_t>::Type;
        mad_mx((__cc__ float *)l0c[l0c_iduse], (__ca__ AType*)l0a[l0a_iduse], (__cb__ BType*)l0b[l0a_iduse],
                32, 128, 256, 0, false, false, true);
    }

    __aicore__ inline void l0c_ub_mm1(int mm1_id, int l0c_iduse){

        /*
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = 256;
        fixpipeParams.mSize = 32;
        fixpipeParams.srcStride = 32;
        fixpipeParams.dstStride = 256; 
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        */
        uint64_t loop3Para = static_cast<uint64_t>(0) << 32;  // LOOP3_PARA[63:32]
        loop3Para |= static_cast<uint64_t>(0) << 16;  // LOOP3_PARA[31:16]
        loop3Para |= static_cast<uint64_t>(1);              // LOOP3_PARA[15:0]
        set_loop3_para(loop3Para);

        PipeBarrier<PIPE_FIX>();

        return copy_matrix_cc_to_ub((__ubuf__ float *)mm1Res[mm1_id], (__cc__ float *)l0c[l0c_iduse], 0,
        256, 32, 256, 32, 1,
        false, 0, 0,
        static_cast<uint64_t>(QuantMode_t::NoQuant), static_cast<uint8_t>(false),
        false, true,
        static_cast<uint64_t>(QuantMode_post::NoConv),
        0, false, false, 0, false, false, false, false, false, false);
    }

    __aicore__ inline void mm1(uint64_t inner, uint64_t outer, uint64_t bIdx, uint64_t n2Idx, uint32_t subLoop)
    {
        //q and qscale:gm->l1
        if (inner == 0 && subLoop == 0){
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
        //k and kscale:gm->l1
        ikv++;
        int ivkuse = ikv % 4;
        wait_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
        gm_l1_k(inner, bIdx, n2Idx, subLoop, ivkuse);
        gm_l1_kscale(inner, bIdx, n2Idx, subLoop, ivkuse);
        set_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
        wait_flag(PIPE_MTE2, PIPE_MTE1, iquse);
        wait_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
        
        //mte1 start
        l0c_id ++ ;
        int l0c_iduse = l0c_id % 2;
        wait_flag(PIPE_FIX, PIPE_M, l0c_iduse);

        l0a_id ++ ;
        int l0a_iduse = l0a_id % 2;
        wait_flag(PIPE_M, PIPE_MTE1, l0a_iduse);// mte1 alloc l0ab buffer
        mm1_l1_l0a_no_trans(l0a_iduse);
        mm1_l1_l0b_no_trans(ivkuse, l0a_iduse);
        set_flag(PIPE_MTE1, PIPE_M, l0a_iduse);
        wait_flag(PIPE_MTE1, PIPE_M, l0a_iduse);

        compute_mm1(l0c_iduse, l0a_iduse);
        set_flag(PIPE_M, PIPE_MTE1, l0a_iduse);

        if (inner == 19 && (subLoop % 2 == 1)) {
            set_flag(PIPE_MTE1, PIPE_MTE2, iquse);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);

        set_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        wait_flag(PIPE_M, PIPE_FIX, l0c_iduse);

        mm1_id ++;
        l0c_ub_mm1(mm1_id % 2, l0c_iduse);//fixp

        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);

    }

    __aicore__ inline void gm_l1_v(uint64_t inner, uint64_t bIdx, uint64_t n2Idx, uint32_t k, int ivkuse)
    {
        uint32_t s2Offset = inner * 512 + k * 256;
        uint64_t blockIdxInBatch = s2Offset / 512;  // 获取block table上的索引
        uint64_t bsIdx = s2Offset % 512;            // 获取在单个块上超出的行数
        uint64_t blockId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * constInfo.maxBlockNumPerBatch +
                                blockIdxInBatch);
        uint64_t gmOffset = blockId * 512 * 8 * 128 + n2Idx * 512 * 128 + bsIdx * 32;
        // intriParams.blockCount = 4;
        // intriParams.blockLen = 256;
        // intriParams.dstStride = 0;
        // intriParams.srcStride = 256;
        /* copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t*)dst, (__gm__ uint8_t*)src, 0, blockCount, burstLength,
                                 leftPaddingCnt, rigntPaddingCnt, true, cacheMode, actSrcStride, actDstStride); */
        copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t *)kv1[ivkuse], (__gm__ uint8_t *)(valueGm + gmOffset),
                                     0, 4, 256 * 32, 0, 0, true, 0, 256 * 32 * 2, 8192);
    }

     __aicore__ inline void gm_l1_vscale(uint64_t inner, uint64_t bIdx, uint64_t n2Idx, uint32_t k, int ivkuse)
    {
        // intriParams.blockCount = 8;
        // intriParams.blockLen = 4;
        // intriParams.dstStride = 0;
        // intriParams.srcStride = 4;
        uint32_t s2VscaleScaleOffset = (inner * 512 + k * 256) / 32;
        uint64_t blockVscaleIdxInBatch = s2VscaleScaleOffset / 8;  // 获取block table上的索引
        uint64_t bsVscaleIdx = s2VscaleScaleOffset % 8;            // 获取在单个块上超出的行数
        uint64_t blockVscaleId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * constInfo.maxBlockNumPerBatch +
                                blockVscaleIdxInBatch);
        uint64_t gmVScaleOffset = blockVscaleId * 32 * 8 * 8 * 8 + n2Idx * 32 * 8 * 8 + bsVscaleIdx * 32;
        copy_gm_to_cbuf_align_v2((__cbuf__ float8_e8m0_t *)kv1[ivkuse] + 256 * 128, (__gm__ float8_e8m0_t *)(deScaleVGm + gmVScaleOffset), 0, 8,
                                 4 * 32, 0, 0, true, 0, 4 * 32 * 2, 128);
    }

    __aicore__ inline void mm2_l1_l0a_no_trans(uint32_t k, int ipuse, int l0a_iduse)
    {
        // loadData2DParamsA.mStartPosition = 0
        // loadData2DParamsA.kStartPosition = 0
        // loadData2DParamsA.ifTranspose = 0
        // loadData2DParamsA.mStep = 2
        // loadData2DParamsA.kStep = 8
        // loadData2DParamsA.srcStride = 8
        // loadData2DParamsA.dstStride = 2

        // loadData2DMxParamsA.xStartPosition = 0
        // loadData2DMxParamsA.yStartPosition = 0
        // loadData2DMxParamsA.xStep = 2
        // loadData2DMxParamsA.yStep = 4
        // loadData2DMxParamsA.srcStride = 4
        // loadData2DMxParamsA.dstStride = 4
        // LoadData(aL0Tensor, aL1Tensor[L1Aoffset], aScaleL1Tensor[L1Aoffset >> 5], loadData2DParamsA,
        //              loadData2DMxParamsA);
        load_cbuf_to_ca(reinterpret_cast<__ca__ float8_e4m3_t *>((__ca__ mx_fp8_e4m3_t *)l0a[l0a_iduse]),
                    (__cbuf__ float8_e4m3_t *)p1[ipuse] + k * 256 * 128,
                    0,
                    0,
                    2,
                    8,
                    8,
                    2,
                    0);
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__ca__ mx_fp8_e4m3_t *)l0a[l0a_iduse])) / 16;
        load_cbuf_to_ca_mx(mxDstAddr,
            static_cast<__cbuf__ void *>((__cbuf__ float8_e8m0_t *)pscale[0]),
            0,
            0,
            2,
            4,
            4,
            4);
        
    }
    __aicore__ inline void mm2_l1_l0b_trans(int ivkuse, int l0b_iduse)
    {
         // loadData2DParamsB.mStartPosition = 0
        // loadData2DParamsB.kStartPosition = 0
        // loadData2DParamsB.ifTranspose = 1
        // loadData2DParamsB.mStep = 16
        // loadData2DParamsB.kStep = 4
        // loadData2DParamsB.srcStride = 16
        // loadData2DParamsB.dstStride = 8

        // loadData2DMxParamsB.xStartPosition = 0
        // loadData2DMxParamsB.yStartPosition = 0
        // loadData2DMxParamsB.xStep = 8
        // loadData2DMxParamsB.yStep = 4
        // loadData2DMxParamsB.srcStride =4
        // loadData2DMxParamsB.dstStride = 4
        load_cbuf_to_cb(reinterpret_cast<__cb__ float8_e4m3_t *>((__cb__ mx_fp8_e4m3_t *)l0b[l0b_iduse]),
                    (__cbuf__ float8_e4m3_t *)kv1[ivkuse],
                    0,
                    0,
                    16,
                    4,
                    16,
                    8,
                    0);
        uint32_t vScaleOffset = 256 * 128;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__cb__ mx_fp8_e4m3_t *)l0b[l0b_iduse])) / 16;
        load_cbuf_to_cb_mx(mxDstAddr,
            static_cast<__cbuf__ void *>((__cbuf__ float8_e8m0_t *)kv1[ivkuse] + vScaleOffset),
            0,
            0,
            8,
            4,
            4,
            4);
    }

    __aicore__ inline void compute_mm2(int l0c_iduse, int l0a_iduse, uint32_t k)
    {
        // MmadParams mmadParams;
        // mmadParams.m = 32;
        // mmadParams.n = 128;
        // mmadParams.k = 256;
        // mmadParams.cmatrixInitVal =1,0;
        // mmadParams.cmatrixSource = 0;
        // mmadParams.unitFlag = 0;
        using AType = typename GetDstType<mx_fp8_e4m3_t>::Type;
        using BType = typename GetDstType<mx_fp8_e4m3_t>::Type;
        mad_mx((__cc__ float *)l0c[l0c_iduse], (__ca__ AType*)l0a[l0a_iduse], (__cb__ BType*)l0b[l0a_iduse],
                32, 256, 128, 0, false, false, k == 0);
    }

    __aicore__ inline void l0c_ub_mm2(int l0c_iduse){

        /*
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = 128;
        fixpipeParams.mSize = 32;
        fixpipeParams.srcStride = 32;
        fixpipeParams.dstStride = 128; 
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        */
        uint64_t loop3Para = static_cast<uint64_t>(0) << 32;  // LOOP3_PARA[63:32]
        loop3Para |= static_cast<uint64_t>(0) << 16;  // LOOP3_PARA[31:16]
        loop3Para |= static_cast<uint64_t>(1);              // LOOP3_PARA[15:0]
        set_loop3_para(loop3Para);

        PipeBarrier<PIPE_FIX>();

        return copy_matrix_cc_to_ub((__ubuf__ float *)mm2Res[0], (__cc__ float *)l0c[l0c_iduse], 0,
        128, 32, 128, 32, 1,
        false, 0, 0,
        static_cast<uint64_t>(QuantMode_t::NoQuant), static_cast<uint8_t>(false),
        false, true,
        static_cast<uint64_t>(QuantMode_post::NoConv),
        0, false, false, 0, false, false, false, false, false, false);
    }

    __aicore__ inline void mm2(uint64_t inner, uint64_t outer, uint64_t bIdx, uint64_t n2Idx)
    {
        ipid ++;
        int ipuse = ipid % 3;
        l0c_id ++ ;
        int l0c_iduse = l0c_id % 2;
        wait_flag(PIPE_FIX, PIPE_M, l0c_iduse);

        for (uint32_t k = 0; k < 2; k++) {
            ikv++;
            int ivkuse = ikv % 4;
            wait_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
            gm_l1_v(inner, bIdx, n2Idx, k, ivkuse);
            gm_l1_vscale(inner, bIdx, n2Idx, k, ivkuse);
            set_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);

            wait_flag(PIPE_MTE2, PIPE_MTE1, ivkuse);
            //mte1 start
            l0a_id ++ ;
            int l0a_iduse = l0a_id % 2;
            wait_flag(PIPE_M, PIPE_MTE1, l0a_iduse);// mte1 alloc l0ab buffer
            mm2_l1_l0a_no_trans(k, ipuse, l0a_iduse);
            mm2_l1_l0b_trans(ivkuse, l0a_iduse);
            set_flag(PIPE_MTE1, PIPE_M, l0a_iduse);
            wait_flag(PIPE_MTE1, PIPE_M, l0a_iduse);

            compute_mm2(l0c_iduse, l0a_iduse, k);
            set_flag(PIPE_M, PIPE_MTE1, l0a_iduse);

            set_flag(PIPE_MTE1, PIPE_MTE2, ivkuse);
        }

        set_flag(PIPE_M, PIPE_FIX, l0c_iduse);
        wait_flag(PIPE_M, PIPE_FIX, l0c_iduse);

        l0c_ub_mm2(l0c_iduse);//fixp

        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);
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
        if ASCEND_IS_AIC {
            prefree_kv_l1();  // 4buffer
            prefree_l0ab();
            prefree_l0c();
            
            uint64_t id = 0;
            uint64_t inner = 0, outer = 0;
            uint64_t outerp[3], innerp[3];
            uint64_t bIdxp[3], n2Idxp[3];
            bool runInfo[3] = {true, true, true};

            bool shouldDispatchTask = true;
            bool shouldExecuteTask = false;
            uint32_t bN2Cur = constInfo.bN2Start;
            uint32_t gS1Cur = 0;
            uint32_t s2Cur = 0;
            while (shouldDispatchTask || shouldExecuteTask) {
                shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
                int64_t bIdx = bN2Cur / 8; //bN2Cur / 8;
                int64_t n2Idx = bN2Cur % 8; //bN2Cur % 8 s
                if (shouldDispatchTask){
                    inner = s2Cur;
                    outer = gS1Cur;
                    UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                    runInfo[id % 3] = true;
                }
                // 执行任务
                shouldExecuteTask = ShouldExecuteTask(runInfo);
                if (shouldExecuteTask) {
                    bIdxp[id % 3] = bIdx;
                    n2Idxp[id % 3] = n2Idx;
                    outerp[id % 3] = outer;
                    innerp[id % 3] = inner;
                    if (runInfo[id % 3] == true) {
                        for (uint32_t subLoop = 0; subLoop < 2; ++subLoop) {
                            mm1(inner, outer, bIdx, n2Idx, subLoop);
                        }
                    }

                    if (id >= 2) {
                        if (runInfo[(id - 2) % 3] == true) {
                            mm2(innerp[(id - 2) % 3], outerp[(id - 2) % 3], bIdxp[(id - 2) % 3], n2Idxp[(id - 2) % 3]);
                            runInfo[(id - 2) % 3] = false;
                        }
                    }
                    id ++;
                }
            }
            postalloc_kv_l1();
            postalloc_l0ab();
            postalloc_l0c();
        }
    }
};  // FlashAttentionFullQuantMxKernel

}  // namespace BaseApi

#endif  // FIA_KERNEL_FULLQUANT_MX_H_