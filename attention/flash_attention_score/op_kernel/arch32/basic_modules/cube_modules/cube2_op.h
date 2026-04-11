/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube2_op.h
 * \brief
 */


#ifndef _CUBE2_OP_H_
#define _CUBE2_OP_H_

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void
CubeOp1<TYPE, layOutType>::Cube2Compute(const AddrInfo &shapeInfo, __gm__ TYPE* left, __gm__ TYPE* right, 
                                        __gm__ float* out)
{
    uint64_t gm2L1SrcDValue = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        gm2L1SrcDValue = headDim * kvHeadNum;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        gm2L1SrcDValue = headDim * kvHeadNum;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        gm2L1SrcDValue = bNum * headDim * kvHeadNum;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        gm2L1SrcDValue = headDim;
    }

    uint64_t left_offset = left - (__gm__ TYPE*)0;
    uint64_t right_offset = right - (__gm__ TYPE*)0;
    uint64_t out_offset = out - (__gm__ float*)0;

    // left matrix is (ky, kx)
    // right matrix is (kx, headDim)
    int32_t kn = shapeInfo.kx;
    int32_t km = shapeInfo.ky;

    int32_t l1_m_size = km;
    int32_t l1_n_size = kn;

    int32_t l1_m_size_align = RoundUp(l1_m_size, C0_SIZE);
    int32_t l1_n_size_align = RoundUp(l1_n_size, C0_SIZE);
    int32_t l1_m_block_size_tail = (l1_m_size % BASE_M_128) == 0 ? BASE_M_128 : (l1_m_size % BASE_M_128);
    int32_t l1_n_block_size_tail = (l1_n_size % BASE_N_128) == 0 ? BASE_N_128 : (l1_n_size % BASE_N_128);
    int32_t l1_m_block_size_align_tail = (l1_m_size_align % BASE_M_128) == 0 ? BASE_M_128 : (l1_m_size_align % BASE_M_128);
    int32_t l1_n_block_size_align_tail = (l1_n_size_align % BASE_N_128) == 0 ? BASE_N_128 : (l1_n_size_align % BASE_N_128);

    int32_t m_loop = CeilDiv(km, BASE_M_128);
    int32_t n_loop = CeilDiv(kn, BASE_N_128);
    for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
        int32_t n_remain = (n_loop_index == n_loop - 1) ? l1_n_block_size_tail : BASE_N_128;
        int32_t n_remain_align = (n_loop_index == n_loop - 1) ? l1_n_block_size_align_tail : BASE_N_128;
        bool l0_c_init_flag = (n_loop_index == 0);

        // load right matrix gm (kx, headDim)-> L1B
        AscendC::LocalTensor<TYPE>* l1_b_buf_tensor = ping_pong_flag_l1_b_ ? &l1_b_pong_tensor : &l1_b_ping_tensor;
        Cube2LoadDataBToL1(*l1_b_buf_tensor, 
                            temp_tensor_bf16[right_offset + n_loop_index * BASE_N_128 * gm2L1SrcDValue], 
                            n_remain, gm2L1SrcDValue, n_remain_align);

        // Load Left_GM matrix A -> L1A
        ping_pong_flag_l1_a_ = 0;
        for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
            AscendC::LocalTensor<TYPE>* l1_a_buf_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;
            int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : BASE_M_128;
            int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : BASE_M_128;
            
            Cube2LoadDataAToL1(*l1_a_buf_tensor, 
                                temp_tensor_bf16[left_offset + m_loop_index * BASE_M_128 * C0_SIZE + n_loop_index * l1_m_size_align * BASE_N_128], 
                                n_remain_align, m_remain_align, l1_m_size_align);

            ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
        }

        // load L1B (n, headDim) -> L0B
        AscendC::LocalTensor<TYPE>* l0_b_buf_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;
        
        Cube2LoadDataBToL0((*l0_b_buf_tensor), (*l1_b_buf_tensor), n_remain_align);

        ping_pong_flag_l1_a_ = 0;
        ping_pong_flag_l0_a_ = 0;
        WAIT_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);
        // do m_loop times mad with l0B常驻
        for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
            AscendC::LocalTensor<TYPE>* l1_a_buf_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;
            AscendC::LocalTensor<TYPE>* l0_a_buf_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
            AscendC::LocalTensor<float>* l0_c_buf_tensor = m_loop_index ? &l0_c_pong_tensor : &l0_c_ping_tensor;

            int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : BASE_M_128;
            int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : BASE_M_128;
            
            Cube2LoadDataAToL0((*l0_a_buf_tensor), (*l1_a_buf_tensor), n_remain_align, m_remain_align);

            // mad (m_remain, n_remain) x (n_remain, headDim)
            bool last_k = false;
            last_k = n_loop_index == n_loop - 1;

            Cube2Mmad(*l0_c_buf_tensor, *l0_a_buf_tensor, *l0_b_buf_tensor, m_remain, n_remain, last_k, l0_c_init_flag);

            if (last_k) {
                Cube2CopyOut(temp_tensor_fp32[out_offset + m_loop_index * BASE_M_128 * C0_SIZE], 
                            *l0_c_buf_tensor, m_remain, m_remain_align, l1_m_size);
            }
            ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
            ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
        }
        SET_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2 + FLAG_SHIFT);

        ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
        ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
    }
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube2CopyOut(GlobalTensor<float> dstTensor, 
                                        LocalTensor<float> srcTensor, 
                                        int32_t m_remain, int32_t m_remain_align, int32_t l1_m_size)
{
    commonFixpipeParamsV220.mSize = m_remain;
    commonFixpipeParamsV220.nSize = headDim;
    commonFixpipeParamsV220.srcStride = m_remain_align;
    commonFixpipeParamsV220.dstStride = l1_m_size * 2;
    // NZ出设置
    AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(
        dstTensor, 
        srcTensor,
        commonFixpipeParamsV220);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube2Mmad(LocalTensor<float> dstCTensor,
                                    LocalTensor<TYPE> srcATensor,
                                    LocalTensor<TYPE> srcBTensor,
                                    int32_t m_remain, int32_t n_remain, bool last_k, bool l0_c_init_flag)
{
    WAIT_FLAG(MTE1, M, ping_pong_flag_l0_a_);
    uint16_t m_modify = (m_remain == 1) ? 2 : m_remain;
    AscendC::Mmad(
        dstCTensor, 
        srcATensor, 
        srcBTensor, 
        AscendC::MmadParams(
            m_modify,
            headDim,
            n_remain,
            last_k ? 3 : 2,
            false,
            l0_c_init_flag
        )
    );
    SET_FLAG(M, MTE1, ping_pong_flag_l0_a_ + FLAG_SHIFT);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube2LoadDataAToL0(LocalTensor<TYPE> dstTensor,
                                        LocalTensor<TYPE> srcTensor,
                                        int32_t n_remain_align, int32_t m_remain_align)
{
    WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
    WAIT_FLAG(M, MTE1, ping_pong_flag_l0_a_ + FLAG_SHIFT);
    commonLoadData2dParamsNoTranspose.repeatTimes = n_remain_align / SIZE_16;
    commonLoadData2dParamsNoTranspose.srcStride = m_remain_align / SIZE_16;
    for (int32_t i = 0; i < m_remain_align / SIZE_16; i++) {
        AscendC::LoadData(
            dstTensor[i * n_remain_align * C0_SIZE],
            srcTensor[i * C0_SIZE * C0_SIZE],
            commonLoadData2dParamsNoTranspose
        );
    }
    SET_FLAG(MTE1, M, ping_pong_flag_l0_a_);
    SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube2LoadDataBToL0(LocalTensor<TYPE> dstTensor,
                                        LocalTensor<TYPE> srcTensor,
                                        int32_t n_remain_align)
{
    commonLoadData2dParamsTranspose.repeatTimes = headDim / C0_SIZE;
    commonLoadData2dParamsTranspose.srcStride = n_remain_align / C0_SIZE;
    WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
    WAIT_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2 + FLAG_SHIFT);
    for (int32_t i = 0; i < n_remain_align / C0_SIZE; i++) {
        AscendC::LoadData(
            dstTensor[i * headDim * C0_SIZE],
            srcTensor[i * C0_SIZE * C0_SIZE],
            commonLoadData2dParamsTranspose
        );
    }
    SET_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);
    SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube2LoadDataAToL1(LocalTensor<TYPE> dstTensor,
                                        GlobalTensor<TYPE> srcTensor,
                                        int32_t n_remain_align, int32_t m_remain_align, int32_t l1_m_size_align)
{
    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
    DataCopyParams nzToNzParam;
    nzToNzParam.blockCount = n_remain_align / C0_SIZE;
    nzToNzParam.blockLen = m_remain_align;
    nzToNzParam.srcStride = l1_m_size_align - m_remain_align;
    nzToNzParam.dstStride = 0;
    AscendC::DataCopy(
        dstTensor,
        srcTensor,
        nzToNzParam
    );
    SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube2LoadDataBToL1(LocalTensor<TYPE> dstTensor,
                                        GlobalTensor<TYPE> srcTensor,
                                        int32_t n_remain, uint64_t gm2L1SrcDValue, int32_t n_remain_align)
{
    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
    commonNd2NzParams.nValue = n_remain;
    commonNd2NzParams.dValue = headDim;
    commonNd2NzParams.srcDValue = gm2L1SrcDValue;
    commonNd2NzParams.dstNzC0Stride = n_remain_align;
    AscendC::DataCopy(
        dstTensor,
        srcTensor,
        commonNd2NzParams
    );
    SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
}

#endif