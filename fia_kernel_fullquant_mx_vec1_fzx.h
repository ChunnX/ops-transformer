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

#ifndef FIA_KERNEL_FULLQUANT_MX_VEC1_H_
#define FIA_KERNEL_FULLQUANT_MX_VEC1_H_

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
    __cbuf__ void *q1[2], *kv1[4], *p1[3], *pscale[1];
    __ubuf__ void *pv[2];
    int ikv{-1};
    int iq{-1};
    int iquse = -1;
    int ipid{-1};
    int bmm1_id{1};
    int vec1_id{-1};
    __ca__ void * l0a[2];
    __cb__ void * l0b[2];
    __cc__ void * l0c[2];
    int l0a_id{-1};
    int l0c_id{-1};
    __ubuf__ void *mm1Res[2], *mm2Res[1];
    int mm1_id{-1};

    __ubuf__ void *stage1OutQue[2];
    __ubuf__ void *stage2OutQue;
    __ubuf__ void *attenMaskUB[2];
    __ubuf__ void *softmaxSumUB[3];
    __ubuf__ void *softmaxMaxUB[3];
    __ubuf__ void *softmaxExpUB[3];
    __ubuf__ void *preLoopMaxUB;
    __ubuf__ void *preLoopSumUB;
    __ubuf__ void *firstLoopSumUB;
    __ubuf__ void *commonUB;
    __ubuf__ uint8_t *vselrIndexesUB[4];

    uint64_t inner_loop_limit = 0;
    uint64_t outer_loop_limit = 0;
    uint32_t mloop = 0;
    uint32_t createdTaskCount = 0;
    float negativeFloatScalar = 0.0f;
    __aicore__ inline FlashAttentionFullQuantMxKernel(){};

    __aicore__ inline void InitCubeBuffer()
    {
        constexpr uint32_t mm1QSize = 128 * 128;
        constexpr uint32_t mm1QScaleSize = 128 * 128 / 32 ;
        constexpr uint32_t mm1KVSize = 128 * 256;
        constexpr uint32_t mm1KVScaleSize = 128 * 256 / 32;
        constexpr uint32_t mm2LeftSize = 128 * 512 ;
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
    __aicore__ inline void InitVecBuffer()
    {
        constexpr uint32_t stage1OutSize = (64 + 1) * 256;
        constexpr uint32_t maskSize = 64  * 256;
        stage1OutQue[0] = (__ubuf__ void *)get_imm(0);
        stage1OutQue[1] = (__ubuf__ void *)get_imm(stage1OutSize);

        attenMaskUB[0] = (__ubuf__ void *)get_imm(stage1OutSize * 2);//16384


        softmaxSumUB[0] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize);
        softmaxSumUB[1] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256);
        softmaxSumUB[2] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 2);
        softmaxMaxUB[0] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 3);
        softmaxMaxUB[1] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 4);
        softmaxMaxUB[2] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 5);
        softmaxExpUB[0] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 6);
        softmaxExpUB[1] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 7);
        softmaxExpUB[2] = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 8);
        preLoopMaxUB = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 9);
        preLoopSumUB = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 10);
        firstLoopSumUB = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 11);
        commonUB = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 12);

        stage2OutQue = (__ubuf__ void *)get_imm(stage1OutSize * 2 + maskSize + 256 * 12 + 512);

        vselrIndexesUB[0] = (__ubuf__ uint8_t *)get_imm(stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4);
        vselrIndexesUB[1] = (__ubuf__ uint8_t *)get_imm(stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4 + 128);

        // vselrIndexesUB[0] = 0x7f;
        // // 先获取 UB 内存地址（uint8 指针）
        // __ubuf__ uint8_t *ub0 = (__ubuf__ uint8_t *)vselrIndexesUB[0];
        // __ubuf__ uint8_t *ub1 = (__ubuf__ uint8_t *)vselrIndexesUB[1];

        for (int i = 0; i < 128; i++) {
            *(vselrIndexesUB[0] + i) = i * 2;
        }
        // for (int i = 0; i < 64; i++) {
        //     vselrIndexesUB[1][i] = i * 4;
        // }
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
        constInfo.scaleValue = 0.08838834764831843;
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

    __aicore__ inline void InitVecBlock()
    {
        uint32_t tmp1 = 0xFF7FFFFF;
        this->negativeFloatScalar = *((float *)&tmp1);
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
        if ASCEND_IS_AIV {
            InitVecBlock();
        }
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
        uint64_t gmQOffset = (bIdx * 4 + outer *  128 / 8) * 8 * 128 * 8 + n2Idx * 8 * 128;
        // Nd2NzParams nd2nzPara;
        // nd2nzPara.ndNum = 1;
        // nd2nzPara.nValue = 8; // nd矩阵的行数
        // nd2nzPara.dValue = 128;       // nd矩阵的列数
        // nd2nzPara.srcDValue = 128; // 同一nd矩阵相邻行起始地址间的偏移
        // nd2nzPara.dstNzC0Stride = 32;
        // nd2nzPara.dstNzNStride = 1;
        // nd2nzPara.srcNdMatrixStride = 0;
        // nd2nzPara.dstNzMatrixStride = 0;
        uint64_t mte2NzPara = static_cast<uint64_t>(0) << 48; // MTE2_NZ_PARA[63:48]
        mte2NzPara |= static_cast<uint64_t>(32) << 32;         // MTE2_NZ_PARA[47:32]
        mte2NzPara |= static_cast<uint64_t>(1) << 16;         // MTE2_NZ_PARA[31:16]
        mte2NzPara |= static_cast<uint64_t>(1);            // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2NzPara); // CCE: store parameters for ND2NZ DMA instructions
        copy_gm_to_cbuf_multi_nd2nz((__cbuf__ float8_e4m3_t *)q1[iquse % 2], (__gm__ float8_e4m3_t *)(queryGm + gmQOffset), 0, 128, 0, 8, 128, 0, false);

        gmQOffset +=  8 * 128 * 8;
        // Nd2NzParams nd2nzPara;
        // nd2nzPara.ndNum = 3;
        // nd2nzPara.nValue = 8; // nd矩阵的行数
        // nd2nzPara.dValue = 128;       // nd矩阵的列数
        // nd2nzPara.srcDValue = 128; // 同一nd矩阵相邻行起始地址间的偏移
        // nd2nzPara.dstNzC0Stride = 32;
        // nd2nzPara.dstNzNStride = 1;
        // nd2nzPara.srcNdMatrixStride = 8 * 128 * 8;
        // nd2nzPara.dstNzMatrixStride = 8 * 32;

        uint64_t mte2Q2NzPara = static_cast<uint64_t>(static_cast<uint16_t>(8)) << 48; // MTE2_NZ_PARA[63:48]
        mte2Q2NzPara |= static_cast<uint64_t>(32) << 32;         // MTE2_NZ_PARA[47:32]
        mte2Q2NzPara |= static_cast<uint64_t>(1) << 16;         // MTE2_NZ_PARA[31:16]
        mte2Q2NzPara |= static_cast<uint64_t>(3);            // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2Q2NzPara); // CCE: store parameters for ND2NZ DMA instructions
        copy_gm_to_cbuf_multi_nd2nz((__cbuf__ float8_e4m3_t *)q1[iquse % 2] + 8 * 32, (__gm__ float8_e4m3_t *)(queryGm + gmQOffset), 0, 128, 0, 8, 128, 8 * 128 * 8, false);

        // int64_t mte2Q2NzPara = static_cast<uint64_t>(static_cast<uint16_t>(32)) << 48; // MTE2_NZ_PARA[63:48]
        // mte2Q2NzPara |= static_cast<uint64_t>(32) << 32;         // MTE2_NZ_PARA[47:32]
        // mte2Q2NzPara |= static_cast<uint64_t>(1) << 16;         // MTE2_NZ_PARA[31:16]
        // mte2Q2NzPara |= static_cast<uint64_t>(4);            // MTE2_NZ_PARA[15:0]
        // set_mte2_nz_para(mte2Q2NzPara); // CCE: store parameters for ND2NZ DMA instructions
        // copy_gm_to_cbuf_multi_nd2nz((__cbuf__ float8_e4m3_t *)q1[q1use], (__gm__ float8_e4m3_t *)(queryGm + gmQOffset), 0, 128, 0, 8, 128, 8 * 128 * 8, false);
    }

    __aicore__ inline void gm_l1_qscale(uint64_t outer, uint64_t bIdx, uint64_t n2Idx, uint64_t q1use)
    {
        uint32_t qScaleOffset = 32 * 128;//((runInfo.actMSize + 31) >> 5 << 5) * constInfo.dSize;runInfo.actMSize=gs1=8*4
        uint64_t gmQScaleOffset = (bIdx * 4 + outer *  128 / 8) * 4 * 8 + n2Idx * 4 * 8 * (1 + 48) * 48 / 2 * 4;
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

    __aicore__ inline void l0c_ub_mm1(int mm1_id, int l0c_iduse)
    {
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

        // outputBuf.WaitCrossCore();
        bmm1_id ++;
        int bmm1_idused = 6 + (bmm1_id % 2) * 2;
        wait_intra_block(PIPE_FIX, 0 + bmm1_idused + 1);//3
        wait_intra_block(PIPE_FIX, 16 + bmm1_idused + 1);//19

        mm1_id ++;
        l0c_ub_mm1(mm1_id % 2, l0c_iduse);//fixp
        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);

        // outputBuf.SetCrossCore();
        set_intra_block(PIPE_FIX, 0 + bmm1_idused + 0);//2
        set_intra_block(PIPE_FIX, 16 + bmm1_idused + 0);//18

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

    __aicore__ inline void l0c_ub_mm2(int l0c_iduse)
    {

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

        return copy_matrix_cc_to_ub((__ubuf__ float *)mm2Res, (__cc__ float *)l0c[l0c_iduse], 0,
                                    128, 32, 128, 32, 1,
                                    false, 0, 0,
                                    static_cast<uint64_t>(QuantMode_t::NoQuant), static_cast<uint8_t>(false),
                                    false, true,
                                    static_cast<uint64_t>(QuantMode_post::NoConv),
                                    0, false, false, 0, false, false, false, false, false, false);
    }

    __aicore__ inline void mm2(int ipuse, uint64_t inner, uint64_t outer, uint64_t bIdx, uint64_t n2Idx)
    {
        // mm2A.WaitCrossCore();
        wait_intra_block(PIPE_MTE1, ipuse);
        wait_intra_block(PIPE_MTE1, ipuse + 16);

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

        // outputBuf.WaitCrossCore();
        wait_intra_block(PIPE_MTE1);
        wait_intra_block(PIPE_MTE1);

        l0c_ub_mm2(l0c_iduse);//fixp

        set_flag(PIPE_FIX, PIPE_M, l0c_iduse);
        // outputBuf.SetCrossCore();
        set_intra_block(PIPE_MTE1);
        set_intra_block(PIPE_MTE1);
        
    }

    __aicore__ inline void ub_l1_vec1(uint32_t subLoop, int ipuse, uint32_t createdTaskCount)
    {
        // block_count = 256 / 32;
        // block_len = (uint16_t)16;
        // srcGap = (uint16_t)(65 - 16);
        // dstGap = (uint16_t)(128 - 16);
        int64_t dstOffset =
                constInfo.subBlockIdx * 32 * 16 +
                256 * 128 * subLoop;
        copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ float8_e4m3_t *)p1[ipuse] + dstOffset), 
                          (__ubuf__ void*)((__ubuf__ float8_e4m3_t *)stage1OutQue[createdTaskCount]), 
                           0, 8, (uint16_t)16, (uint16_t)49, (uint16_t)112);

    }

    template <typename T, typename T2>
    __simd_vf__ void process_vec1_noupdate_subloop0(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float *expSumUb, __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_sink_input;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        // MaskReg preg_tail_n1 = UpdateMask<float>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<float>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);
        MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;

        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();

        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        Duplicate(vreg_p_scale, static_cast<float>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

        Duplicate(vreg_min, minValue);
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * 256);
            LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * 256);
            LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * 256);
            LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * 256);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // } else {
            //     if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            //                   IsSameType<T2, hifloat8_t>::value) {
            //         Muls(vreg_input_x1, vreg_input_x1, dScaleQK, preg_all); // Muls(dScaleQK)
            //         Muls(vreg_input_x2, vreg_input_x2, dScaleQK, preg_all);
            //         Muls(vreg_input_x3, vreg_input_x3, dScaleQK, preg_all);
            //         Muls(vreg_input_x4, vreg_input_x4, dScaleQK, preg_all);
            //     }
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                // atten mask
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_sel1,
                                                                  preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                                                                  vreg_sel2, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                                                                  vreg_sel3_new, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                                                                  vreg_sel4_new, preg_tail_n2);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                                  vreg_max_tmp3, preg_all);
            // } 
            // else {
            //     Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
            //     Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
            //                                                       preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
            //                                                       vreg_input_x2, preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
            //                                                       vreg_input_x3_new, preg_tail_n1);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
            //                                                       vreg_input_x4_new, preg_tail_n2);

            //     Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
            //     Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
            //     Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            //     Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
            //                                                                                       vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)maxUb), ureg_max, 0);
        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
        // }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);
            Sub(vreg_max_brc, vreg_max_brc, vreg_ln_p_scale, preg_all);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * 256);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4,
                                                              srcUb + floatRepSize * 2 + i * 256);
            ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                              preg_all);
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)expSumUb), vreg_exp_sum3,
                                                                         ureg_exp_sum, 1);
            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8);
                LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
                Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            // }
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
    }

    template <typename T, typename T2>
    __simd_vf__ void process_vec1_noupdate_subloop1(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_sink_input;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        // MaskReg preg_tail_n1 = UpdateMask<float>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<float>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);
        MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;

        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();

        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        Duplicate(vreg_p_scale, static_cast<float>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

        Duplicate(vreg_min, minValue);
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * 256);
            LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * 256);
            LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * 256);
            LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * 256);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // } else {
            //     if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            //                   IsSameType<T2, hifloat8_t>::value) {
            //         Muls(vreg_input_x1, vreg_input_x1, dScaleQK, preg_all); // Muls(dScaleQK)
            //         Muls(vreg_input_x2, vreg_input_x2, dScaleQK, preg_all);
            //         Muls(vreg_input_x3, vreg_input_x3, dScaleQK, preg_all);
            //         Muls(vreg_input_x4, vreg_input_x4, dScaleQK, preg_all);
            //     }
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                // atten mask
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_sel1,
                                                                  preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                                                                  vreg_sel2, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                                                                  vreg_sel3_new, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                                                                  vreg_sel4_new, preg_tail_n2);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                                  vreg_max_tmp3, preg_all);
            // }
            // else {
            //     Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
            //     Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
            //                                                       preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
            //                                                       vreg_input_x2, preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
            //                                                       vreg_input_x3_new, preg_tail_n1);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
            //                                                       vreg_input_x4_new, preg_tail_n2);

            //     Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
            //     Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
            //     Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            //     Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
            //                                                                                       vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max,
                                                                     1);
        }
        StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
        RegTensor<float> vreg_in_max;
        LoadAlign(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_input_max, tmpMaxUb2);
        Max(vreg_input_max, vreg_in_max, vreg_input_max,
            preg_all); // loop1的max需要和loop0的max取一个max，保证loop1的max是当前的全局最大max，不要被loop1的max覆盖掉
        ExpSub(vreg_exp_max, vreg_in_max, vreg_input_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_input_max, preg_all);

        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
        // }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);
            Sub(vreg_max_brc, vreg_max_brc, vreg_ln_p_scale, preg_all);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * 256);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4,
                                                              srcUb + floatRepSize * 2 + i * 256);
            ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                              preg_all);
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3,
                                                                         ureg_exp_sum, 1);
            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8);
                LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
                Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            // }
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> first_loop_sum;
        LoadAlign(first_loop_sum, inExpSumUb);
        LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
        Mul(first_loop_sum, vreg_exp_max, first_loop_sum, preg_all);
        Add(vreg_exp_sum3, first_loop_sum, vreg_exp_sum_brc, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_sum3, preg_all);
    }

    __aicore__ inline void process_vec1_noupdate(int mm1_id, uint32_t subLoop, uint32_t createdTaskCount, uint32_t mloopCount, float minValue, const float pScale = 1.0f)
    {
        const uint32_t blockStride = 65;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[createdTaskCount % 2];
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[createdTaskCount % 2] + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloopCount % 3];
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)softmaxSumUB[mloopCount % 3];
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)commonUB;
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)commonUB;
        __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloopCount % 3];
        __ubuf__ float *maxUbStart = (__ubuf__ float *)softmaxMaxUB[mloopCount % 3];
        __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[createdTaskCount % 3];
        __ubuf__ float *inMaxUb = (__ubuf__ float *)softmaxMaxUB[mloopCount % 3];
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1Res[mm1_id];//这个看mm1_id需要改吗
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesUB[0];

        //printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloopCount, createdTaskCount);
        //printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloopCount % 3, createdTaskCount % 3);

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)attenMaskUB[0];
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)attenMaskUB[0] + 64;
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)attenMaskUB[0] + 64 * 2;
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)attenMaskUB[0] + 64 * 3;
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
            process_vec1_noupdate_subloop0<float, fp8_e4m3fn_t>(expUb1, expUb2, maxUb, maxUbStart, srcUb, expSumUb, indexesUb, 
                maskUb1, maskUb2, maskUb3, maskUb4, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN2, 16, static_cast<float>(constInfo.scaleValue), minValue, pScale);
        } else {
            process_vec1_noupdate_subloop1<float, fp8_e4m3fn_t>(expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb,
                inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, 
                maskUb1, maskUb2, maskUb3, maskUb4, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN2, 16, static_cast<float>(constInfo.scaleValue), minValue, pScale);
        }
    }

    template <typename T, typename T2>
    __simd_vf__ void process_vec1_update_subloop0(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4,  __ubuf__ float *preLoopMaxUb, __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb,
        const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_subloop_update;
        RegTensor<float> vreg_pre_loop_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_pre_sum;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_sink_input;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;

        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        // MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
        // MaskReg preg_tail_n1 = UpdateMask<T>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<T>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<T>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<T>(pltOriTailN2);
        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        Duplicate(vreg_p_scale, static_cast<float>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

        Duplicate(vreg_min, minValue);
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * 256);
            LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * 256);
            LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * 256);
            LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * 256);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                // atten mask
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_sel1,
                                                                  preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                                                                  vreg_sel2, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                                                                  vreg_sel3_new, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                                                                  vreg_sel4_new, preg_all);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                                  vreg_max_tmp3, preg_all);
            // } else {
            //     Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
            //     Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
            //                                                       preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
            //                                                       vreg_input_x2, preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
            //                                                       vreg_input_x3_new, preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
            //                                                       vreg_input_x4_new, preg_all);
            //     Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
            //     Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
            //     Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            //     Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
            //                                                                                       vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), vreg_input_max,
                                                                         ureg_max, 1);
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);

        LoadAlign(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_input_max, tmpMaxUb2);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)preLoopMaxUb, vreg_in_max,
                                                          preg_all);
        Max(vreg_max_new, vreg_in_max, vreg_input_max, preg_all);
        ExpSub(vreg_exp_max, vreg_in_max, vreg_max_new, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max_new, preg_all);


        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        // }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUbStart + i);
            Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * 256);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4,
                                                              srcUb + floatRepSize * 2 + i * 256);
            ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all);
            ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all);
            ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all);
            ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                              preg_all);
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3,
                                                                         ureg_exp_sum, 1);

            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8);
                LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
                Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            // } else if constexpr (IsSameType<T2, half>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_f16, (RegTensor<uint16_t> &)vreg_exp_even1_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_f16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_f16, (RegTensor<uint16_t> &)vreg_exp_even2_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_f16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
            // }
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);

        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_in_exp_sum, inExpSumUb);
        LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)preLoopSumUb, vreg_in_exp_sum,
                                                          preg_all);
        Mul(vreg_exp_max, vreg_exp_max, vreg_in_exp_sum, preg_all);
        Add(vreg_exp_max, vreg_exp_max, vreg_exp_sum_brc,
            preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)firstLoopSumUb, vreg_exp_sum_brc,
                                                          preg_all); // 传给loop 1用于更新rowSum
    }

    template <typename T, typename T2>
    __simd_vf__ void process_vec1_update_subloop1(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4,  __ubuf__ float *preLoopMaxUb, __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb,
        const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_subloop_update;
        RegTensor<float> vreg_pre_loop_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_pre_sum;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_sink_input;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;

        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        // MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
        // MaskReg preg_tail_n1 = UpdateMask<T>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<T>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<T>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<T>(pltOriTailN2);
        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        Duplicate(vreg_p_scale, static_cast<float>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

        Duplicate(vreg_min, minValue);
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * 256);
            LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * 256);
            LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * 256);
            LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * 256);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                // atten mask
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_sel1,
                                                                  preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                                                                  vreg_sel2, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                                                                  vreg_sel3_new, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                                                                  vreg_sel4_new, preg_all);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                                  vreg_max_tmp3, preg_all);
            // } else {
            //     Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
            //     Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
            //                                                       preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
            //                                                       vreg_input_x2, preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
            //                                                       vreg_input_x3_new, preg_all);
            //     StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
            //                                                       vreg_input_x4_new, preg_all);
            //     Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
            //     Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
            //     Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
            //     Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
            //                                                                                       vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), vreg_input_max,
                                                                         ureg_max, 1);
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);

        LoadAlign(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_input_max, tmpMaxUb2);
        Max(vreg_max_new, vreg_in_max, vreg_input_max, preg_all);
        ExpSub(vreg_subloop_update, vreg_in_max, vreg_max_new, preg_all);
        LoadAlign(vreg_pre_loop_max, preLoopMaxUb);
        ExpSub(vreg_exp_max, vreg_pre_loop_max, vreg_max_new, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max_new, preg_all);

        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        // }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUbStart + i);
            Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * 256);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4,
                                                              srcUb + floatRepSize * 2 + i * 256);
            ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all);
            ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all);
            ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all);
            ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                              preg_all);
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3,
                                                                         ureg_exp_sum, 1);

            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8);
                LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
                Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            // } else if constexpr (IsSameType<T2, half>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_f16, (RegTensor<uint16_t> &)vreg_exp_even1_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_f16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_f16, (RegTensor<uint16_t> &)vreg_exp_even2_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_f16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
            // }
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);

        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
        RegTensor<float> vreg_first_sum;
        LoadAlign(vreg_first_sum, firstLoopSumUb);
        LoadAlign(vreg_pre_sum, preLoopSumUb);
        Mul(vreg_first_sum, vreg_subloop_update, vreg_first_sum, preg_all);
        Add(vreg_first_sum, vreg_first_sum, vreg_exp_sum_brc, preg_all);
        Mul(vreg_pre_sum, vreg_exp_max, vreg_pre_sum, preg_all);
        Add(vreg_pre_sum, vreg_first_sum, vreg_pre_sum, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_pre_sum, preg_all);
    }

    __aicore__ inline void process_vec1_update(int mm1_id, uint32_t subLoop, uint32_t createdTaskCount, uint32_t mloopCount, float minValue, const float pScale = 1.0f)
    {
        const uint32_t blockStride = 65;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[createdTaskCount % 2];
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[createdTaskCount % 2] + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloopCount % 3];
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)softmaxSumUB[mloopCount % 3];
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)commonUB;
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)commonUB;
        __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloopCount % 3];
        __ubuf__ float *maxUbStart = (__ubuf__ float *)softmaxMaxUB[mloopCount % 3];
        __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[createdTaskCount % 3];
        __ubuf__ float *inMaxUb = (__ubuf__ float *)softmaxMaxUB[mloopCount % 3];
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1Res[mm1_id];//这个看mm1_id需要改吗
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesUB[0];

        __ubuf__ float *preLoopMaxUb = (__ubuf__ float *)preLoopMaxUB;
        __ubuf__ float *preLoopSumUb = (__ubuf__ float *)preLoopSumUB;;
        __ubuf__ float *firstLoopSumUb = (__ubuf__ float *)firstLoopSumUB;;

        //printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloopCount, createdTaskCount);
        //printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloopCount % 3, createdTaskCount % 3);

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)attenMaskUB[0];
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)attenMaskUB[0] + 64;
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)attenMaskUB[0] + 64 * 2;
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)attenMaskUB[0] + 64 * 3;
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
            process_vec1_update_subloop0<float, fp8_e4m3fn_t>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4, 
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN2, 16, static_cast<float>(constInfo.scaleValue), minValue, pScale);
        } else {
            process_vec1_update_subloop1<float, fp8_e4m3fn_t>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4, 
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN2, 16, static_cast<float>(constInfo.scaleValue), minValue, pScale);
        }
    }

    __aicore__ inline bool IsSkipAttentionmask(uint32_t gs1StartIdx, uint32_t s2StartIdx)
    {
        int32_t s1StartIdx = gs1StartIdx / 8;
        // int64_t nextToken = 
        //         static_cast<int64_t>(10240) - static_cast<int64_t>(4); // 统一以左上角为原点计算Token
        
        if (static_cast<int64_t>(s2StartIdx + 256) <= static_cast<int64_t>(s1StartIdx) + 10236) {
            return true;
        }
        return false;
    }

    __aicore__ inline uint64_t ComputeAttenMaskOffset(uint32_t s1StartIdx, uint32_t s2StartIdx)
    {
        int64_t nextToken = 0; // sparse2 本身原点就是左上角
        // nextToken =
        //     static_cast<int64_t>(10240) - static_cast<int64_t>(4); // 统一以左上角为原点计算token
         
        uint64_t offset = 0;
        int64_t delta = 10236 + s1StartIdx - s2StartIdx;
        uint32_t attenMaskSizeAlign = 256;

        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                    2048; // min(delta, s2inner)
        
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

        // // ub-remain
        if (remainRowCount > 0) {
            uint64_t maskOffset = ComputeAttenMaskOffset(s1StartIdx, s2StartIdx);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 2;
            dataCopyParams.blockLen = 256;
            dataCopyParams.srcStride = 1792;
            dataCopyParams.dstStride = 56;
            // blockLen + padParams.len + dstStride = gSize * s2Size
            DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 1U};//isPad,leftPadding,rightPadding,paddingValue
            set_mov_pad_val(GetScalarBitcodeValue(1U));
            // DataCopyPad(attenMaskUB[headGSize * attenMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);
            copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)attenMaskUB[0] + headGSize * attenMaskS2Stride, (__gm__ uint8_t*)attenMaskGmInt + maskOffset, 0, 2, 256,
                                     0, 0, true, 0, 1792 + 256 , 2048);
        }
        set_flag(PIPE_MTE2, PIPE_V, event_t(0));
        wait_flag(PIPE_MTE2, PIPE_V, event_t(0));
        MaskUbCopyS1G((__ubuf__ uint8_t*)attenMaskUB[0], tailGSize, midS1Count);
    }

    template <typename T, typename RegType>
    __simd_vf__ inline void DuplicateFromScalarImpl(__ubuf__ T * dstLocal, const T scalarValue, const int32_t calCount)
    {
        RegType dstReg;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        Reg::MaskReg mask;
        constexpr uint32_t repeatStride = static_cast<uint32_t>(GetVecLen() * RegType::trait.REG_NUM);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
        Reg::MaskReg maskFull = Reg::CreateMask<T, Reg::MaskPattern::ALL, RegType::trait>();
        Reg::Duplicate(dstReg, scalarValue, maskFull);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = Reg::UpdateMask<T, RegType::trait>(sreg);
            Reg::StoreAlign(dstLocal + i * repeatStride, dstReg, mask);
        }
    }

    __aicore__ inline void AttenMaskCopyIn(uint64_t inner, uint32_t subLoop)
    {
        uint64_t gs1StartIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            gs1StartIdx = 16;
        }
        uint64_t s2StartIdx = (subLoop == 0) ? inner * 512 : (inner * 512 + 256);
        bool IsSkipMask = IsSkipAttentionmask(gs1StartIdx, s2StartIdx);
        if (IsSkipMask) {
            // Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
            DuplicateFromScalarImpl<uint8_t, Reg::RegTensor<uint8_t, Reg::RegTraitNumOne>>(
                (__ubuf__ uint8_t*)attenMaskUB[0], static_cast<uint8_t>(0U), 16 * 256);
            return;
        } else {
            AttentionmaskCopyInForSgLayout(gs1StartIdx, s2StartIdx);
        }
    }

    __aicore__ inline void vec1(int ipuse, uint64_t inner, uint64_t outer, uint32_t createdTaskCount, uint32_t mloopCount, uint32_t subLoop)
    {
        // bmm1ResBuf.WaitCrossCore();
        bmm1_id ++;
        int bmm1_idused = 6 + (bmm1_id % 2) * 2;
        wait_intra_block(PIPE_V, 0 + bmm1_idused + 0);
        AttenMaskCopyIn(inner, subLoop);

        if(inner == 0) {
           //printf("inner=0,createdTaskCount =%d, mloopCount = %d\n",createdTaskCount,mloopCount);
           process_vec1_noupdate(bmm1_id % 2 ,subLoop, createdTaskCount, mloopCount, negativeFloatScalar);
       } else {
           process_vec1_update(bmm1_id % 2 ,subLoop, createdTaskCount, mloopCount, negativeFloatScalar);
       }
        // bmm1ResBuf.SetCrossCore();
        set_intra_block(PIPE_V, 0 + bmm1_idused + 1);
        
        ub_l1_vec1(subLoop, ipuse, createdTaskCount % 2);
        if (subLoop % 2 == 1) {
            // outputBuf.SetCrossCore();//L1
            set_intra_block(PIPE_MTE3, ipuse);
        }
    }

    __aicore__ inline void vec2()
    {
        // if (unlikely(runInfo.actVecMSize == 0)) {
        //     bmm2ResBuf.SetCrossCore();
        //     return;
        // }

        int64_t vec2CalcSize = runInfo.actVecMSize * 128;
        float deSCaleVValue = 1.0f;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        if (inner == 0) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            // LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            // LocalTensor<T> pScaleUb;

            __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[createdTaskCount % 3];

            float deSCalePreVValue = 1.0f;
            if (!runInfo.isLastS2Loop) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64, 1.0,
                    1.0);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
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

    __aicore__ inline void GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        //printf("s2Cur= %d\n",s2Cur);
        if (s2Cur == 0) {
            mloop++;
        }
    }

    __aicore__ inline void Process()
    {
        CrossCoreBufferInit();
        if ASCEND_IS_AIC {
            prefree_kv_l1();  // 4buffer
            prefree_l0ab();
            prefree_l0c();
        }

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
                GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                inner = s2Cur;
                outer = gS1Cur;
                runInfo[createdTaskCount % 3] = true;
                createdTaskCount ++;
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
                // ipid ++;
                int ipuse = id % 3;
                if (runInfo[id % 3] == true) {
                    for (uint32_t subLoop = 0; subLoop < 2; ++subLoop) {
                        if ASCEND_IS_AIC {
                            mm1(inner, outer, bIdx, n2Idx, subLoop);
                        } else {
                            if (subLoop % 2 == 0) {
                                vec1(ipuse, inner, outer, createdTaskCount, mloop, subLoop);
                            } else {
                                vec1(ipuse, inner, outer, createdTaskCount, mloop, subLoop);
                            }
                        }
                    }
                }
                if (id >= 2) {
                    if (runInfo[(id - 2) % 3] == true) {
                        if ASCEND_IS_AIC {
                            mm2((id - 2) % 3, innerp[(id - 2) % 3], outerp[(id - 2) % 3], bIdxp[(id - 2) % 3], n2Idxp[(id - 2) % 3]);
                        } else {
                            vec2((id - 2) % 3, innerp[(id - 2) % 3], outerp[(id - 2) % 3]);
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


