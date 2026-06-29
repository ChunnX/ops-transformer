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

    __cbuf__ void *q1[2], *kv1[4];
    int ikv{-1};
    int iq{-1};
    int iquse = -1;
    __ca__ void * l0a[2];
    __cb__ void * l0b[2];
    __cc__ void * l0c[2];
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
        constexpr uint32_t mm2LeftSize = 128 * 512  + 128 * 512 / 32;

        q1[0] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 );
        q1[1] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + mm1QSize + mm1QScaleSize);
        kv1[0] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + (mm1QSize + mm1QScaleSize) * 2);
        kv1[1] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + (mm1QSize + mm1QScaleSize) * 2 + mm1KVSize + mm1KVScaleSize);
        kv1[2] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + (mm1QSize + mm1QScaleSize) * 2 + (mm1KVSize + mm1KVScaleSize) * 2);
        kv1[3] = (__cbuf__ void *)get_imm(mm2LeftSize * 3 + (mm1QSize + mm1QScaleSize) * 2 + (mm1KVSize + mm1KVScaleSize) * 3);
        l0a[0] = (__ca__ void *)get_imm(0); 
        l0a[1] = (__ca__ void *)get_imm(32 * 1024);
        l0b[0] = (__cb__ void *)get_imm(0);
        l0b[1] = (__cb__ void *)get_imm(32 * 1024);
        l0c[0] = (__cc__ void *)get_imm(0);
        l0c[1] = (__cc__ void *)get_imm(128 * 256 * 4);//sizeof(fp32)
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
        uint64_t tIdx = GetTBaseQ_MX(bIdx) + gS1Idx / gSize;
        uint64_t tStride = n2Size * n2Size * dBaseSize;
        uint64_t gmQOffset = tIdx * n2Size * n2Size * dBaseSize +
                             n2Idx * n2Size * dBaseSize;
        int64_t mte2Q2NzPara = static_cast<uint64_t>(static_cast<uint16_t>(n2Size)) << 48; // MTE2_NZ_PARA[63:48]
        mte2Q2NzPara |= static_cast<uint64_t>((actMSize + 31) >> 5 << 5) << 32;         // MTE2_NZ_PARA[47:32]
        mte2Q2NzPara |= static_cast<uint64_t>(1) << 16;         // MTE2_NZ_PARA[31:16]
        mte2Q2NzPara |= static_cast<uint64_t>(s1Size);            // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2Q2NzPara); // CCE: store parameters for ND2NZ DMA instructions
        uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)(__gm__ float8_e4m3_t *)(queryGm + gmQOffset)) >> 60);
        copy_gm_to_cbuf_multi_nd2nz((__cbuf__ float8_e4m3_t *)q1[q1use], (__gm__ float8_e4m3_t *)(queryGm + gmQOffset), 0, dBaseSize, cacheMode, 
                                     n2Size, dBaseSize, tStride, false);
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

        uint64_t mte2QScaleNzPara = static_cast<uint64_t>(dBaseSize / MXFP_GROUP_SIZE / 2 * 2 / 32) << 48; // MTE2_NZ_PARA[63:48]
        mte2QScaleNzPara |= static_cast<uint64_t>(dBaseSize / MXFP_GROUP_SIZE / 2) << 32;         // MTE2_NZ_PARA[47:32]
        mte2QScaleNzPara |= static_cast<uint64_t>(1) << 16;         // MTE2_NZ_PARA[31:16]
        mte2QScaleNzPara |= static_cast<uint64_t>(1);            // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2QScaleNzPara); // CCE: store parameters for DN2NZ DMA instructions
        
        const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)((__gm__ float8_e8m0_t *)deScaleQGm + gmQScaleOffset)) >> 60);
        copy_gm_to_cbuf_multi_dn2nz((__cbuf__ bfloat16_t *)((__cbuf__ float8_e8m0_t *)q1[q1use] + qScaleOffset), 
                                    (__gm__ bfloat16_t *)((__gm__ float8_e8m0_t *)deScaleQGm + gmQScaleOffset),
                                    0, dBaseSize / MXFP_GROUP_SIZE / 2 * 2, cacheMode, dBaseSize / MXFP_GROUP_SIZE / 2, actMSize, 0, false);
    }

    __aicore__ inline void gm_l1_k(uint32_t s2Idx, uint32_t bIdx, uint32_t n2Idx, uint32_t subLoop, int ivkuse)
    {
        // uint32_t s2Offset = (subLoop % 2 == 0) ? inner * 512 : inner * 512 + 256;
        uint32_t s2Offset = (subLoop % 2 == 0) ? s2Idx : s2Idx + 256;
        uint64_t blockIdxInBatch = s2Offset / 512;  // 获取block table上的索引
        uint64_t bsIdx = s2Offset % 512;            // 获取在单个块上超出的行数
        uint64_t blockId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * maxBlockNumPerBatch +
                                blockIdxInBatch);
        uint64_t gmKOffset = blockId * n2Size * dBaseSize / 32 * 512 * 32 + n2Idx * dBaseSize / 32  * 512 * 32 + bsIdx * 32;
        const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)(__gm__ float8_e4m3_t *)(keyGm + gmKOffset)) >> 60);
        copy_gm_to_cbuf_align_v2((__cbuf__ float8_e4m3_t *)kv1[ivkuse], (__gm__ float8_e4m3_t *)(keyGm + gmKOffset), 0, dBaseSize / 32,
                                 256 * 32, 0, 0, true, cacheMode, 256 * 32 * 2, 256 * 32);
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
        const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)(__gm__ float8_e8m0_t *)(deScaleKGm + gmKScaleOffset)) >> 60);
        copy_gm_to_cbuf_align_v2((__cbuf__ float8_e8m0_t *)kv1[ivkuse] + 256 * 128, (__gm__ float8_e8m0_t *)(deScaleKGm + gmKScaleOffset), 0, 1,
                                 32 * 32, 0, 0, true, cacheMode, 0, 32 * 32);
    }
    __aicore__ inline void mm1_l1_l0a_no_trans(int l0a_iduse)
    {
        load_cbuf_to_ca(reinterpret_cast<__ca__ float8_e4m3_t *>((__ca__ mx_fp8_e4m3_t *)l0a[l0a_iduse]),
                    (__cbuf__ float8_e4m3_t *)q1[iquse % 2],
                    0,
                    0,
                    ((actMSize + 15) >> 4 << 4) >> 4,
                    4,
                    ((actMSize + 31) >> 5 << 5) >> 4,
                    ((actMSize + 15) >> 4 << 4) >> 4,
                    0);
        uint32_t qScaleOffset = 32 * 128;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__ca__ mx_fp8_e4m3_t *)l0a[l0a_iduse])) / 16;
        load_cbuf_to_ca_mx(mxDstAddr,
            static_cast<__cbuf__ void *>((__cbuf__ float8_e8m0_t *)q1[iquse % 2] + qScaleOffset),
            0,
            0,
            ((actMSize + 15) >> 4 << 4) >> 4,
            2,
            2,
            2);
        
    }
    __aicore__ inline void mm1_l1_l0b_no_trans(int ivkuse, int l0b_iduse)
    {
        load_cbuf_to_cb(reinterpret_cast<__cb__ float8_e4m3_t *>((__cb__ mx_fp8_e4m3_t *)l0b[l0b_iduse]),
                    (__cbuf__ float8_e4m3_t *)kv1[ivkuse],
                    0,
                    0,
                    16,
                    ((dBaseSize + 31) >> 5 << 5) >> 5,
                    16,
                    16,
                    0);

        uint32_t kScaleOffset = 256 * dBaseSize;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__cb__ mx_fp8_e4m3_t *)l0b[l0b_iduse])) / 16;
        load_cbuf_to_cb_mx(mxDstAddr,
            static_cast<__cbuf__ void *>((__cbuf__ float8_e8m0_t *)kv1[ivkuse] + kScaleOffset),
            0,
            0,
            16,
            (dBaseSize + 63)  >> 5 >> 1,
            (dBaseSize + 63)  >> 5 >> 1,
            (dBaseSize + 63)  >> 5 >> 1);
    }

    __aicore__ inline void compute_mm1(int l0c_iduse, int l0a_iduse)
    {
        using AType = typename GetDstType<mx_fp8_e4m3_t>::Type;
        using BType = typename GetDstType<mx_fp8_e4m3_t>::Type;
        mad_mx((__cc__ float *)l0c[l0c_iduse], (__ca__ AType*)l0a[l0a_iduse], (__cb__ BType*)l0b[l0a_iduse],
                actMSize, dBaseSize, 256, 0, false, false, true);
    }

    __aicore__ inline void l0c_ub_mm1(__ubuf__ void *mm1Res, int l0c_iduse)
    {
        uint64_t loop3Para = static_cast<uint64_t>(0) << 32;  // LOOP3_PARA[63:32]
        loop3Para |= static_cast<uint64_t>(0) << 16;  // LOOP3_PARA[31:16]
        loop3Para |= static_cast<uint64_t>(1);              // LOOP3_PARA[15:0]
        set_loop3_para(loop3Para);

        PipeBarrier<PIPE_FIX>();

        return copy_matrix_cc_to_ub((__ubuf__ float *)mm1Res, (__cc__ float *)l0c[l0c_iduse], 0,
        256, (actMSize + 31) >> 5 << 5, 256, ((actMSize + 15) / 16) * 16, 1,
        false, 0, 0,
        static_cast<uint64_t>(QuantMode_t::NoQuant), static_cast<uint8_t>(false),
        false, true,
        static_cast<uint64_t>(QuantMode_post::NoConv),
        0, false, false, 0, false, false, false, false, false, false);
    }
    __aicore__ inline void IterateBmm1(__ubuf__ void *mm1Res, int bmm1_id, uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx,
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
        int l0a_iduse = l0a_id % 2;
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
        uint64_t blockId =
            *(__gm__ int32_t *)(blockTableGm + bIdx * maxBlockNumPerBatch +
                                blockIdxInBatch);
        uint64_t gmVOffset = blockId * n2Size * dVBaseSize / 32 * 512 * 32 + n2Idx * dVBaseSize / 32  * 512 * 32 + bsIdx * 32;
        const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)(__gm__ float8_e4m3_t *)(valueGm + gmVOffset)) >> 60);
        copy_gm_to_cbuf_align_v2((__cbuf__ float8_e4m3_t *)kv1[ivkuse], (__gm__ float8_e4m3_t *)(valueGm + gmVOffset), 0, dVBaseSize / 32,
                                 256 * 32, 0, 0, true, cacheMode, 256 * 32 * 2, 256 * 32);
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
        const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)(__gm__ float8_e8m0_t *)(deScaleVGm + gmVScaleOffset)) >> L2_CACHE_OFFSET);
        copy_gm_to_cbuf_align_v2((__cbuf__ float8_e8m0_t *)kv1[ivkuse] + 256 * dVBaseSize, (__gm__ float8_e8m0_t *)(deScaleVGm + gmVScaleOffset), 0, dVBaseSize * MXFP_MULTI_BASE_SIZE / 32,
                                 256 / MXFP_DIVISOR_SIZE * 32, 0, 0, true, 0, 256 / MXFP_DIVISOR_SIZE * 32 * 2, 4 * 32);
    }

    __aicore__ inline void mm2_l1_l0a_no_trans(__cbuf__ void *p1, uint32_t k, int ipuse, int l0a_iduse)
    {

        uint32_t pScaleOffset = mBaseSize * s2BaseSize; // PScale在L1P的偏移量（单位：元素）; // VScale在mm1B的偏移量（单位：元素）

        load_cbuf_to_ca(reinterpret_cast<__ca__ float8_e4m3_t *>((__ca__ mx_fp8_e4m3_t *)l0a[l0a_iduse]),
                    (__cbuf__ float8_e4m3_t *)p1 + k * 256 * dBaseSize,
                    0,
                    0,
                    ((actMSize + 15) >> 4 << 4) >> 4,
                    8,
                    8,
                    ((actMSize + 15) >> 4 << 4) >> 4,
                    0);
        uint64_t l1ScaleOffset = k * 256 / MXFP_GROUP_SIZE * actMSizeAlign32 ;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__ca__ mx_fp8_e4m3_t *)l0a[l0a_iduse])) / 16;
        load_cbuf_to_ca_mx(mxDstAddr,
            static_cast<__cbuf__ void *>((__cbuf__ float8_e8m0_t *)p1 + pScaleOffset + l1ScaleOffset),
            0,
            0,
            ((actMSize + 15) >> 4 << 4) >> 4,
            4,
            4,
            4);
        
    }
    __aicore__ inline void mm2_l1_l0b_trans(int ivkuse, int l0b_iduse)
    {
        load_cbuf_to_cb(reinterpret_cast<__cb__ float8_e4m3_t *>((__cb__ mx_fp8_e4m3_t *)l0b[l0b_iduse]),
                    (__cbuf__ float8_e4m3_t *)kv1[ivkuse],
                    0,
                    0,
                    16,
                    4,
                    16,
                    8,
                    1);
        uint32_t vScaleOffset = 256 * dVBaseSize;
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
                actMSize, 256, dVBaseSize, 0, false, false, k == 0);
    }

    __aicore__ inline void l0c_ub_mm2(__ubuf__ void *mm2Res, int l0c_iduse)
    {
        uint64_t loop3Para = static_cast<uint64_t>(0) << 32;  // LOOP3_PARA[63:32]
        loop3Para |= static_cast<uint64_t>(0) << 16;  // LOOP3_PARA[31:16]
        loop3Para |= static_cast<uint64_t>(1);              // LOOP3_PARA[15:0]
        set_loop3_para(loop3Para);

        PipeBarrier<PIPE_FIX>();

        return copy_matrix_cc_to_ub((__ubuf__ float *)mm2Res, (__cc__ float *)l0c[l0c_iduse], 0,
                                    dVBaseSize, (actMSize + 31) >> 5 << 5, dVBaseSize, ((actMSize + 15) / 16) * 16, 1,
                                    false, 0, 0,
                                    static_cast<uint64_t>(QuantMode_t::NoQuant), static_cast<uint8_t>(false),
                                    false, true,
                                    static_cast<uint64_t>(QuantMode_post::NoConv),
                                    0, false, false, 0, false, false, false, false, false, false);
    }

    __aicore__ inline void IterateBmm2(int ipuse, __ubuf__ void *mm2Res, __cbuf__ void *p1, uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx)
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
            int l0a_iduse = l0a_id % 2;
            wait_flag(PIPE_M, PIPE_MTE1, l0a_iduse);// mte1 alloc l0ab buffer
            mm2_l1_l0a_no_trans(p1, k, ipuse, l0a_iduse);
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
    __aicore__ inline FAFullQuantMxBlockCubeDummy(ConstInfoX &constInfo) {};
};
} // namespace BaseApi

#endif // FIA_BLOCK_CUBE_FULLQUANT_MX_H_