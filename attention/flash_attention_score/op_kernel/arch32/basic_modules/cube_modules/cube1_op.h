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
 * \file cube1_op.h
 * \brief
 */


#ifndef _CUBE1_OP_H_
#define _CUBE1_OP_H_

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void
CubeOp1<TYPE, layOutType>::Cube1Compute(const AddrInfo &shapeInfo, __gm__ TYPE* left, __gm__ TYPE* right, __gm__ float* out, bool needNz2Nd)
{
    uint64_t gm2L1SrcDValueA = 0;
    uint64_t gm2L1SrcDValueB = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        gm2L1SrcDValueA = headDim * qHeadNum;
        gm2L1SrcDValueB = headDim * kvHeadNum;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        gm2L1SrcDValueA = headDim * qHeadNum;
        gm2L1SrcDValueB = headDim * kvHeadNum;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        gm2L1SrcDValueA = bNum * headDim * qHeadNum;
        gm2L1SrcDValueB = bNum * headDim * kvHeadNum;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        gm2L1SrcDValueA = headDim;
        gm2L1SrcDValueB = headDim;
    }
    int32_t km = shapeInfo.ky;
    int32_t kn = shapeInfo.kx;

    int32_t l1_m_size_ = km;

    int32_t n_loop = CeilDiv(kn, BASE_N_128);

    auto gm_a = left + shapeInfo.left;
    auto gm_b = right + shapeInfo.right;
    auto gm_c = out;

    uint64_t left_offset = gm_a - (__gm__ TYPE*)0;
    uint64_t right_offset = gm_b - (__gm__ TYPE*)0;
    uint64_t out_offset = gm_c - (__gm__ float*)0;

    LocalTensor<TYPE>* l1_a_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;

    Cube1LoadDataAToL1(*l1_a_tensor, temp_tensor_bf16[left_offset], l1_m_size_, gm2L1SrcDValueA);

    for (int n_index = 0; n_index < n_loop; n_index++) {
        int32_t l1_n_size_ = (n_index == n_loop - 1) ? (kn - n_index * BASE_N_128) : BASE_N_128;
        LocalTensor<TYPE>* l1_b_tensor = ping_pong_flag_l1_b_ ? &l1_b_pong_tensor : &l1_b_ping_tensor;

        Cube1LoadDataBToL1(*l1_b_tensor, temp_tensor_bf16[right_offset + n_index * BASE_N_128 * gm2L1SrcDValueB], 
                      l1_n_size_, gm2L1SrcDValueB);

        int32_t l1_m_size_align_ = RoundUp(l1_m_size_, C0_SIZE);
        int32_t l1_n_size_align_ = RoundUp(l1_n_size_, C0_SIZE);

        for (int n_offset = 0; n_offset < l1_n_size_; n_offset += BASE_BLOCK_LENGTH) {
            int32_t n_mad_ = Min((l1_n_size_ - n_offset), BASE_BLOCK_LENGTH);
            int32_t n0_ = RoundUp(n_mad_, C0_SIZE);

            LocalTensor<TYPE>* l0_b_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;

            Cube1LoadDataBToL0((*l0_b_tensor), (*l1_b_tensor), n0_, n_offset, l1_n_size_align_, l1_n_size_);

            for (int m_offset = 0; m_offset < l1_m_size_; m_offset += SIZE_128) {
                int32_t m_mad_ = Min((l1_m_size_ - m_offset), BASE_BLOCK_LENGTH);
                int32_t m0_ = RoundUp(m_mad_, C0_SIZE);

                LocalTensor<TYPE>* l0_a_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;

                Cube1LoadDataAToL0((*l0_a_tensor), (*l1_a_tensor), l1_m_size_align_, m0_, headDim, m_offset);
                
                LocalTensor<float>* l0_c_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;
                Cube1Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, m_mad_, n_mad_);

                SET_FLAG(M, MTE1, FLAG_SHIFT + ping_pong_flag_l0_a_);

                Cube1CopyOut(temp_tensor_fp32, *l0_c_tensor, out_offset, m_mad_, m0_, 
                            n0_, l1_m_size_, n_index, m_offset, n_mad_, kn, needNz2Nd);

                ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
                ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
            }
            SET_FLAG(M, MTE1, FLAG_SHIFT +  ping_pong_flag_l0_b_ + 2);
            ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
        }
        SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
        ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
    }
    SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
    ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void
CubeOp1<TYPE, layOutType>::Cube1LoadDataBToL1(LocalTensor<TYPE> dstTensor,
                                        GlobalTensor<TYPE> srcTensor,
                                        int32_t l1_n_size_, uint64_t gm2L1SrcDValueB)
{
    int32_t l1_n_size_align_ = RoundUp(l1_n_size_, C0_SIZE);
    commonNd2NzParams.nValue = l1_n_size_;
    commonNd2NzParams.dValue = headDim;
    commonNd2NzParams.srcDValue = gm2L1SrcDValueB;
    commonNd2NzParams.dstNzC0Stride = l1_n_size_align_;
    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
    AscendC::DataCopy(
        dstTensor,
        srcTensor,
        commonNd2NzParams
    );
    SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
    WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void
CubeOp1<TYPE, layOutType>::Cube1LoadDataAToL1(LocalTensor<TYPE> dstTensor,
                                        GlobalTensor<TYPE> srcTensor,
                                        int32_t l1_m_size_, uint64_t gm2L1SrcDValueA)
{
    int32_t l1_m_size_align_ = RoundUp(l1_m_size_, C0_SIZE);
    commonNd2NzParams.nValue = l1_m_size_;
    commonNd2NzParams.dValue = headDim;
    commonNd2NzParams.srcDValue = gm2L1SrcDValueA;
    commonNd2NzParams.dstNzC0Stride = l1_m_size_align_;
    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
    AscendC::DataCopy(
        dstTensor,
        srcTensor,
        commonNd2NzParams
    );
    SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
    WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube1CopyOut(GlobalTensor<float> dstTensor, 
                                        LocalTensor<float> srcTensor, uint64_t gm_out_offset, int32_t m_mad_, 
                                        int32_t m0_, int32_t n0_, int32_t l1_m_size_, 
                                        int32_t n_index, int m_offset, int32_t n_mad_, int32_t kn, 
                                        bool needNz2Nd)
{
    commonFixpipeParamsV220.mSize = m_mad_;
    commonFixpipeParamsV220.srcStride = m0_;

    if (unlikely(needNz2Nd)) {
        // NZ出
        commonFixpipeParamsV220.nSize = n0_;
        commonFixpipeParamsV220.dstStride = l1_m_size_ * 2;
        auto out_offset = n_index * l1_m_size_ * BASE_N_128 + m_offset * SIZE_16;
        AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(dstTensor[gm_out_offset + out_offset],
                                                            srcTensor, commonFixpipeParamsV220);
    } else {
        // ND出
        commonFixpipeParamsV220.nSize = n_mad_;
        commonFixpipeParamsV220.dstStride = kn;
        auto out_offset = kn * m_offset + n_index * BASE_N_128;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(dstTensor[gm_out_offset + out_offset],
                                                                srcTensor, commonFixpipeParamsV220);
    }
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube1Mmad(LocalTensor<float> dstCTensor,
                                    LocalTensor<TYPE> srcATensor,
                                    LocalTensor<TYPE> srcBTensor,
                                    int32_t m_mad_, int32_t n_mad_)
{
    uint16_t m_modify = (m_mad_ == 1) ? 2 : m_mad_;
    commonMadParams.m = m_modify;
    commonMadParams.n = n_mad_;
    commonMadParams.k = headDim;
    commonMadParams.unitFlag = 3;
    commonMadParams.cmatrixInitVal = true;
    AscendC::Mmad(
        dstCTensor,
        srcATensor,
        srcBTensor,
        commonMadParams
    );
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube1LoadDataAToL0(LocalTensor<TYPE> dstTensor,
                                        LocalTensor<TYPE> srcTensor,
                                        int32_t l1_m_size_align_, int32_t m0_, uint64_t headDim, int m_offset)
{
    WAIT_FLAG(M, MTE1, FLAG_SHIFT + ping_pong_flag_l0_a_);
    commonLoadData2dParamsNoTranspose.repeatTimes = headDim / SIZE_16;
    commonLoadData2dParamsNoTranspose.srcStride = l1_m_size_align_ / SIZE_16;
    for (int32_t i = 0; i < m0_ / SIZE_16; i++) {
        AscendC::LoadData(
            dstTensor[i * headDim * SIZE_16],
            srcTensor[m_offset * SIZE_16 + i * SIZE_256],
            commonLoadData2dParamsNoTranspose
        );
    }
    SET_FLAG(MTE1, M, ping_pong_flag_l0_a_);
    WAIT_FLAG(MTE1, M, ping_pong_flag_l0_a_);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::Cube1LoadDataBToL0(LocalTensor<TYPE> dstTensor,
                                        LocalTensor<TYPE> srcTensor,
                                        int32_t n0_, int n_offset, int32_t l1_n_size_align_, int32_t l1_n_size_)
{
    WAIT_FLAG(M, MTE1, FLAG_SHIFT + ping_pong_flag_l0_b_ + 2);
    if (l1_n_size_ == BASE_BLOCK_LENGTH) {
        commonLoadData2dParamsNoTranspose.repeatTimes = headDim * n0_ / SIZE_256;
        commonLoadData2dParamsNoTranspose.srcStride = 1;
        AscendC::LoadData(
            dstTensor,
            srcTensor,
            commonLoadData2dParamsNoTranspose
        );
    } else {
        commonLoadData2dParamsNoTranspose.repeatTimes = n0_ / SIZE_16;
        commonLoadData2dParamsNoTranspose.srcStride = 1;
        for (int i = 0; i < headDim / SIZE_16; i++) {
            AscendC::LoadData(
                dstTensor[i * n0_ * SIZE_16],
                srcTensor[i * l1_n_size_align_ * SIZE_16 + n_offset * SIZE_16],
                commonLoadData2dParamsNoTranspose
            );
        }
    }
    SET_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);
    WAIT_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::SetFlag()
{
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(MTE1, MTE2, EVENT_ID4);
    SET_FLAG(MTE1, MTE2, EVENT_ID5);

    SET_FLAG(M, MTE1, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID4);
    SET_FLAG(M, MTE1, EVENT_ID5);
    SET_FLAG(M, MTE1, EVENT_ID6);
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void CubeOp1<TYPE, layOutType>::WaitFlag()
{
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID5);

    WAIT_FLAG(M, MTE1, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID4);
    WAIT_FLAG(M, MTE1, EVENT_ID5);
    WAIT_FLAG(M, MTE1, EVENT_ID6);
}
#endif