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
 * \file cube_op1.h
 * \brief
 */


#ifndef _CUBE_OP1_H_
#define _CUBE_OP1_H_

#include "common_header.h"


namespace CUBE_OP1 {
template <typename TYPE, LayOutTypeEnum layOutType>
class CubeOp1 {
public:
    /* clang-format off */
    __aicore__ inline CubeOp1(){};
    __aicore__ inline void Init(uint64_t qHeadNumIn, 
                                uint64_t kvHeadNumIn, 
                                uint64_t headDimIn, 
                                uint64_t bNumIn);
    __aicore__ inline void Cube1Process(int64_t qCoreOffset, int64_t kCoreOffset, int32_t cubeS1RealSize, int32_t s2RealSize, 
                                       GlobalTensor<TYPE> queryGm, GlobalTensor<TYPE> keyGm, GlobalTensor<float> mm1Res, bool needNz2Nd);
    __aicore__ inline void Cube2Process(int64_t vCoreOffset, int32_t cubeS1RealSize, int32_t s2RealSize, 
                                        GlobalTensor<TYPE> stage1Res, GlobalTensor<TYPE> valueGm, GlobalTensor<float> mm2Res);

private:
    __aicore__ inline void Cube1Compute(const AddrInfo &shapeInfo, 
                                        __gm__ TYPE* left, 
                                        __gm__ TYPE* right, 
                                        __gm__ float* out, 
                                        bool needNz2Nd);
    __aicore__ inline void Cube2Compute(const AddrInfo &shapeInfo, 
                                           __gm__ TYPE* left, 
                                           __gm__ TYPE* right, 
                                           __gm__ float* out);
    __aicore__ inline void Cube1LoadDataAToL1(LocalTensor<TYPE> dstTensor,
                                        GlobalTensor<TYPE> srcTensor,
                                        int32_t l1_m_size_, uint64_t gm2L1SrcDValueA);
    __aicore__ inline void Cube1LoadDataBToL1(LocalTensor<TYPE> dstTensor,
                                        GlobalTensor<TYPE> srcTensor,
                                        int32_t l1_n_size_, uint64_t gm2L1SrcDValueB);
    __aicore__ inline void Cube1LoadDataBToL0(LocalTensor<TYPE> dstTensor,
                                        LocalTensor<TYPE> srcTensor,
                                        int32_t n0_, int n_offset, int32_t l1_n_size_align_, int32_t l1_n_size_);
    __aicore__ inline void Cube1LoadDataAToL0(LocalTensor<TYPE> dstTensor,
                                        LocalTensor<TYPE> srcTensor,
                                        int32_t l1_m_size_align_, int32_t m0_, uint64_t headDim, int m_offset);
    __aicore__ inline void Cube1Mmad(LocalTensor<float> dstCTensor,
                                    LocalTensor<TYPE> srcATensor,
                                    LocalTensor<TYPE> srcBTensor,
                                    int32_t m_mad_, int32_t n_mad_);
    __aicore__ inline void Cube1CopyOut(GlobalTensor<float> dstTensor, 
                                        LocalTensor<float> srcTensor, 
                                        uint64_t gm_out_offset, int32_t m_mad_, int32_t m0_, int32_t n0_, int32_t l1_m_size_, 
                                        int32_t n_index, int m_offset, int32_t n_mad_, int32_t kn, bool needNz2Nd);

    __aicore__ inline void Cube2LoadDataAToL1(LocalTensor<TYPE> dstTensor,
                                        GlobalTensor<TYPE> srcTensor,
                                        int32_t n_remain_align, int32_t m_remain_align, int32_t l1_m_size_align);
    __aicore__ inline void Cube2LoadDataBToL1(LocalTensor<TYPE> dstTensor,
                                        GlobalTensor<TYPE> srcTensor,
                                        int32_t n_remain, uint64_t gm2L1SrcDValue, int32_t n_remain_align);
    __aicore__ inline void Cube2LoadDataBToL0(LocalTensor<TYPE> dstTensor,
                                        LocalTensor<TYPE> srcTensor,
                                        int32_t n_remain_align);
    __aicore__ inline void Cube2LoadDataAToL0(LocalTensor<TYPE> dstTensor,
                                        LocalTensor<TYPE> srcTensor,
                                        int32_t n_remain_align, int32_t m_remain_align);
    __aicore__ inline void Cube2Mmad(LocalTensor<float> dstCTensor,
                                    LocalTensor<TYPE> srcATensor,
                                    LocalTensor<TYPE> srcBTensor,
                                    int32_t m_remain, int32_t n_remain, bool last_k, bool l0_c_init_flag);
    __aicore__ inline void Cube2CopyOut(GlobalTensor<float> dstTensor, 
                                        LocalTensor<float> srcTensor, 
                                        int32_t m_remain, int32_t m_remain_align, int32_t l1_m_size);
    __aicore__ inline void SetFlag();
    __aicore__ inline void WaitFlag();

    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;
    AscendC::Nd2NzParams commonNd2NzParams {
        1,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        1,
        0
    };

    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose {
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        0,
        false,
        0
    };

    AscendC::LoadData2dParams commonLoadData2dParamsTranspose {
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        0,
        true,
        0
    };

    AscendC::MmadParams commonMadParams {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        3,
        false,
        true
    };

    AscendC::FixpipeParamsV220 commonFixpipeParamsV220{
        BASE_BLOCK_LENGTH, 
        BASE_BLOCK_LENGTH, 
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH, 
        false
    };

    LocalTensor<TYPE> l1_a_ping_tensor;
    LocalTensor<TYPE> l1_a_pong_tensor;
    LocalTensor<TYPE> l1_b_ping_tensor;
    LocalTensor<TYPE> l1_b_pong_tensor;

    GlobalTensor<TYPE> temp_tensor_bf16;
    GlobalTensor<float> temp_tensor_fp32;

    uint32_t ping_pong_flag_l1_a_ = 0;
    uint32_t ping_pong_flag_l1_b_ = 0;

    // L0A L0B
    LocalTensor<TYPE> l0_a_ping_tensor;
    LocalTensor<TYPE> l0_a_pong_tensor;
    LocalTensor<TYPE> l0_b_ping_tensor;
    LocalTensor<TYPE> l0_b_pong_tensor;

    // L0C
    LocalTensor<float> l0_c_ping_tensor;
    LocalTensor<float> l0_c_pong_tensor;

    uint32_t ping_pong_flag_l0_a_ = 0;
    uint32_t ping_pong_flag_l0_b_ = 0;
    uint32_t ping_pong_flag_l0_c_ = 0;
    uint32_t FLAG_SHIFT = 3;

    uint64_t qHeadNum;
    uint64_t kvHeadNum;
    uint64_t headDim;
    uint64_t bNum;
};


template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void
CubeOp1<TYPE, layOutType>::Init(uint64_t qHeadNumIn, 
                                uint64_t kvHeadNumIn, 
                                uint64_t headDimIn, 
                                uint64_t bNumIn)
{
    qHeadNum = qHeadNumIn;
    kvHeadNum = kvHeadNumIn;
    headDim = headDimIn;
    bNum = bNumIn;
    temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(0));
    temp_tensor_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(0));
    l1_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(0);
    l1_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_128 * SIZE_ONE_K);
    l1_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K);

    // init L0A/L0B/L0C tensor
    l0_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(0);
    l0_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(0);
    l0_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(SIZE_32 * SIZE_ONE_K);

    l0_c_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_64 * SIZE_ONE_K);

    commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
    commonFixpipeParamsV220.unitFlag = 3;
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void
CubeOp1<TYPE, layOutType>::Cube1Process(int64_t qCoreOffset, int64_t kCoreOffset, int32_t cubeS1RealSize, int32_t s2RealSize, 
                                       GlobalTensor<TYPE> queryGm, GlobalTensor<TYPE> keyGm, GlobalTensor<float> mm1Res, bool needNz2Nd)
{
    AddrInfo shapeInfo;
    shapeInfo.left = qCoreOffset;
    shapeInfo.right = kCoreOffset;
    shapeInfo.ky = cubeS1RealSize;
    shapeInfo.kx = s2RealSize;
    __gm__ TYPE* query = (__gm__ TYPE*)queryGm.GetPhyAddr();
    __gm__ TYPE* key = (__gm__ TYPE*)keyGm.GetPhyAddr();
    __gm__ float* out = (__gm__ float*)mm1Res.GetPhyAddr();

    SetFlag();
    Cube1Compute(shapeInfo, query, key, out, needNz2Nd);
    WaitFlag();
}

template <typename TYPE, LayOutTypeEnum layOutType>
__aicore__ inline void
CubeOp1<TYPE, layOutType>::Cube2Process(int64_t vCoreOffset, int32_t cubeS1RealSize, int32_t s2RealSize, 
                                GlobalTensor<TYPE> stage1Res, GlobalTensor<TYPE> valueGm, GlobalTensor<float> mm2Res)
{
    AddrInfo shapeInfo;
    shapeInfo.left = 0;
    shapeInfo.right = vCoreOffset;
    shapeInfo.ky = cubeS1RealSize;
    shapeInfo.kx = s2RealSize;
    __gm__ TYPE* stage1Res1 = (__gm__ TYPE*)stage1Res.GetPhyAddr();
    __gm__ TYPE* value = (__gm__ TYPE*)valueGm.GetPhyAddr();
    __gm__ float* out = (__gm__ float*)mm2Res.GetPhyAddr();
    
    SetFlag();
    Cube2Compute(shapeInfo, stage1Res1, value, out);
    WaitFlag();
}

#include "cube_modules/cube1_op.h"
#include "cube_modules/cube2_op.h"

} // namespace CUBE_OP1
#endif