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
 * \file mhc_sinkhorn_backward.h
 * \brief mhc_sinkhorn_backward
 */

#ifndef ASCENDC_MHC_SINKHORN_BACKWARD_H
#define ASCENDC_MHC_SINKHORN_BACKWARD_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "mhc_sinkhorn_backward_struct.h"
#include "mhc_sinkhorn_backward_tiling_key.h"

namespace MhcSinkhornBackward {
using namespace AscendC;

constexpr int64_t MASK_BUFFER_SIZE = 64;
constexpr int64_t TEMP_BUFFER_SIZE = 256;
constexpr uint32_t MASK_4 = 0b00000000000000001111111111111111;
constexpr uint32_t MASK_6 = 0b00000000111111111111111111111111;
constexpr uint32_t MASK_8 = 0b11111111111111111111111111111111;
// constexpr uint32_t MASK_4 = 0b11111111111111110000000000000000;
// constexpr uint32_t MASK_6 = 0b11111111000000000000000000000000;
// constexpr uint32_t MASK_8 = 0b00000000000000000000000000000000;
constexpr int64_t MASK_NUM = 8;
constexpr int64_t INDEX_BLOCK_LEN = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0)
    {
        return 0;
    }
    return (a + b - 1) / b;
}

class MhcSinkhornBackwardSimd {
public:
    __aicore__ inline MhcSinkhornBackwardSimd(TPipe &pipe, const MhcSinkhornBackwardTilingData &tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR norm, GM_ADDR sum, GM_ADDR gradInput, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcSoftmaxBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                               __local_mem__ float *gradInputAddr,__local_mem__ uint32_t *maskAddr, 
                                               __local_mem__ float *tempAddr, uint32_t n, uint32_t perT);

    __aicore__ inline void CalcColBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                           __local_mem__ uint32_t *maskAddr, __local_mem__ float *tempAddr,
                                           uint32_t n, uint32_t perT);

    __aicore__ inline void CalcRowBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                           __local_mem__ uint32_t *maskAddr, __local_mem__ float *tempAddr,
                                           uint32_t n, uint32_t perT);
    
    template <HardEvent EVENT>
    __aicore__ inline void SetWaitEvent(HardEvent event);

    __aicore__ inline void CopyInNormAndSum(const uint64_t normOffset, const uint32_t normProcessNum, const uint64_t sumOffset, const uint32_t sumProcessNum);

    GlobalTensor<float> gradY_;
    GlobalTensor<float> norm_;
    GlobalTensor<float> sum_;
    GlobalTensor<float> gradInput_;
    TPipe &pipe_;
    const MhcSinkhornBackwardTilingData &tilingData_;
    int64_t blockIdx_; // 核id
    int64_t loop_; // 循环次数,单次处理normCorePerLoopTSize个矩阵，需要loop_次
    int64_t loopTSize_; // 整核/尾核 整块处理的矩阵数量
    int64_t tailLoopTSize_; // 整核/尾核 尾块处理的矩阵数量
    int64_t nSize_;
    int64_t nAlignSize_;
    TQue<QuePosition::VECIN, 1> inputGradYQue_;
    TQue<QuePosition::VECIN, 1> inputNormQue_;
    TQue<QuePosition::VECIN, 1> inputSumQue_;
    TQue<QuePosition::VECOUT, 1> outputQue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> tempBuffer_;
};

__aicore__ inline void MhcSinkhornBackwardSimd::Init(GM_ADDR gradY, GM_ADDR norm, GM_ADDR sum, GM_ADDR gradInput,
                                                      GM_ADDR tiling)
{
    blockIdx_ = GetBlockIdx();
    gradY_.SetGlobalBuffer((__gm__ float *)(gradY));
    norm_.SetGlobalBuffer((__gm__ float *)(norm));
    sum_.SetGlobalBuffer((__gm__ float *)(sum));
    gradInput_.SetGlobalBuffer((__gm__ float *)(gradInput));
    
    nSize_ = tilingData_.nSize;
    nAlignSize_ = tilingData_.nAlignSize;
    
    // 如果是主核的话
    loop_ = tilingData_.normCoreTLoops; // 循环次数
    loopTSize_ = tilingData_.normCorePerLoopTSize;  // 主块T
    tailLoopTSize_ = tilingData_.normCoreLastLoopTSize; // 尾块T
    // 如果是尾核的话
    if (blockIdx_ == tilingData_.needCoreNum - 1) {
        loop_ = tilingData_.tailCoreLoops;
        loopTSize_ = tilingData_.tailCorePerLoopTSize;
        tailLoopTSize_ = tilingData_.tailCoreLastLoopTSize;
    }

    // printf("[debug] loop_=%lld\n", loop_);
    // printf("[debug] loopTSize_=%lld\n", loopTSize_);
    // printf("[debug] tailLoopTSize_=%lld\n", tailLoopTSize_);
    // printf("[debug] nSize_=%lld\n", nSize_);
    // printf("[debug] nAlignSize_=%lld\n", nAlignSize_);
    // printf("[debug] tilingData_.numIters=%lld\n", tilingData_.numIters);

    pipe_.InitBuffer(maskBuffer_, MASK_BUFFER_SIZE);
    pipe_.InitBuffer(tempBuffer_, TEMP_BUFFER_SIZE);
    pipe_.InitBuffer(inputGradYQue_, DOUBLE_BUFFER, tilingData_.coreTSize * nSize_ * nSize_ * sizeof(float));
    pipe_.InitBuffer(inputNormQue_, DOUBLE_BUFFER, tilingData_.coreTSize * nSize_ * nSize_ * sizeof(float));
    pipe_.InitBuffer(inputSumQue_, DOUBLE_BUFFER, tilingData_.coreTSize * nSize_ * nSize_ * sizeof(float));
    pipe_.InitBuffer(outputQue_, DOUBLE_BUFFER, tilingData_.coreTSize * nSize_ * nSize_ * sizeof(float));
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcRowBackward(__local_mem__ float *gradYAddr,
                                                                 __local_mem__ float *normAddr,
                                                                 __local_mem__ float *sumAddr,
                                                                 __local_mem__ float *gradInputAddr,
                                                                 __local_mem__ uint32_t *maskAddr,
                                                                 __local_mem__ float *tempAddr,
                                                                 uint32_t n, uint32_t perT)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::LoadAlign(maskReg, maskAddr);

        MicroAPI::RegTensor<float> gradYReg;
        MicroAPI::RegTensor<float> normReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<float> gradInputReg;
        MicroAPI::RegTensor<float> sumGradYReg;
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> orderReg;
        MicroAPI::RegTensor<int32_t> duplicateReg1;
        MicroAPI::RegTensor<int32_t> duplicateReg2;
        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(perT);

        // 生成索引 indexReg
        MicroAPI::Duplicate(duplicateReg1, INDEX_BLOCK_LEN); // 填充 8
        MicroAPI::Duplicate(duplicateReg2, static_cast<int32_t>(n * n)); // 填充：n*n
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0)); // 生成索引 以0为起始值
        MicroAPI::Div(tmpReg1, orderReg, duplicateReg1, maskReg); // orderReg ÷ 8 -> temp1
        MicroAPI::Mul(tmpReg2, tmpReg1, duplicateReg1, maskReg); // temp1 * 8 -> temp2
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg); // orderReg - temp2 -> temp2
        MicroAPI::Mul(tmpReg3, tmpReg1, duplicateReg2, maskReg); // temp1 * n*n -> temp3
        MicroAPI::Add(indexReg, tmpReg2, tmpReg3, maskReg); // temp2 + temp3 -> indexReg

        MicroAPI::DataCopyGather(gradYReg, gradYAddr, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg); // 根据索引位置将输入gradYAddr对应元素搬入gradYReg
        // norm 和 sum 直接搬运？搬运多少数据量？
        MicroAPI::DataCopy(normReg, normAddr + perT * INDEX_BLOCK_LEN * sizeof(float));
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumReg, sumAddr + perT * sizeof(float));
        
        MicroAPI::Mul(gradInputReg, gradYReg, normReg, maskReg);
        MicroAPI::ReduceSumWithDataBlock(sumGradYReg, gradInputReg, maskReg);

        // 先要搬出去再搬进来广播 todo:能不能之间广播？
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(tempAddr, sumGradYReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumGradYReg, tempAddr);

        MicroAPI::Sub(gradInputReg, gradYReg, sumGradYReg, maskReg);
        MicroAPI::Div(gradInputReg, gradInputReg, sumReg, maskReg);

        MicroAPI::DataCopyScatter(gradInputAddr, gradInputReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}


__aicore__ inline void MhcSinkhornBackwardSimd::CalcColBackward(__local_mem__ float *gradYAddr,
                                                                 __local_mem__ float *normAddr,
                                                                 __local_mem__ float *sumAddr,
                                                                 __local_mem__ float *gradInputAddr,
                                                                 __local_mem__ uint32_t *maskAddr,
                                                                 __local_mem__ float *tempAddr,
                                                                 uint32_t n, uint32_t perT)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::LoadAlign(maskReg, maskAddr);

        MicroAPI::RegTensor<float> gradYReg;
        MicroAPI::RegTensor<float> normReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<float> gradInputReg;
        MicroAPI::RegTensor<float> sumGradYReg;
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> orderReg;
        MicroAPI::RegTensor<int32_t> duplicateReg1;
        MicroAPI::RegTensor<int32_t> duplicateReg2;
        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(perT);

        // 生成索引 indexReg
        MicroAPI::Duplicate(duplicateReg1, INDEX_BLOCK_LEN); // 填充 8
        MicroAPI::Duplicate(duplicateReg2, static_cast<int32_t>(n * n)); // 填充：n*n
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0)); // 生成索引 以0为起始值
        MicroAPI::Div(tmpReg1, orderReg, duplicateReg1, maskReg); // orderReg ÷ 8 -> temp1
        MicroAPI::Mul(tmpReg2, tmpReg1, duplicateReg1, maskReg); // temp1 * 8 -> temp2
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg); // orderReg - temp2 -> temp2
        MicroAPI::Mul(tmpReg3, tmpReg1, duplicateReg2, maskReg); // temp1 * n*n -> temp3
        MicroAPI::Add(indexReg, tmpReg2, tmpReg3, maskReg); // temp2 + temp3 -> indexReg

        MicroAPI::DataCopyGather(gradYReg, gradYAddr, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg); // 根据索引位置将输入gradYAddr对应元素搬入gradYReg
        // norm 和 sum 直接搬运？搬运多少数据量？
        MicroAPI::DataCopy(normReg, normAddr + perT * INDEX_BLOCK_LEN * sizeof(float));
        MicroAPI::DataCopy(sumReg, sumAddr + perT * INDEX_BLOCK_LEN * sizeof(float));
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        
        MicroAPI::Mul(gradInputReg, gradYReg, normReg, maskReg);
        MicroAPI::ReduceSumWithDataBlock(sumGradYReg, gradInputReg, maskReg); // 8*8 ->  1*8
        // 1*8 -> 8*8
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(tempAddr, sumGradYReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumGradYReg, tempAddr); // sumGradYReg ->8*8

        MicroAPI::Sub(gradInputReg, gradYReg, sumGradYReg, maskReg);
        MicroAPI::Div(gradInputReg, gradInputReg, sumReg, maskReg);

        MicroAPI::DataCopyScatter(gradInputAddr, gradInputReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcSoftmaxBackward(__local_mem__ float *gradYAddr, 
                                                                    __local_mem__ float *normAddr,
                                                                    __local_mem__ float *gradInputAddr,
                                                                    __local_mem__ uint32_t *maskAddr, 
                                                                    __local_mem__ float *tempAddr,
                                                                    uint32_t n, uint32_t perT)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::LoadAlign(maskReg, maskAddr);

        MicroAPI::RegTensor<float> gradYReg; 
        MicroAPI::RegTensor<float> normReg;   
        MicroAPI::RegTensor<float> mulResultReg;    
        MicroAPI::RegTensor<float> dotProdReg;
        MicroAPI::RegTensor<float> subResultReg;
        MicroAPI::RegTensor<float> gradInputReg;

        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> orderReg; 

        MicroAPI::RegTensor<int32_t> duplicateReg1;
        MicroAPI::RegTensor<int32_t> duplicateReg2;

        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(perT);

        // indexReg：使用DataCopyGather生成制定索引
        MicroAPI::Duplicate(duplicateReg1, static_cast<int32_t>(INDEX_BLOCK_LEN));  // 填充8,以8个float数为一个block
        MicroAPI::Duplicate(duplicateReg2, static_cast<int32_t>(n * n));    // 填充n*n
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0));    // orderReg：生成顺序索引:0,1,2,3,...
        MicroAPI::Div(tmpReg1, orderReg, duplicateReg1, maskReg);   // tmpReg1:标识每一个数属于哪一个block
        MicroAPI::Mul(tmpReg2, tmpReg1, duplicateReg1, maskReg);    // tmpReg2:过渡
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg); // tmpReg2:标识每一个数在block块内的相对索引
        MicroAPI::Mul(tmpReg3, tmpReg1, duplicateReg2, maskReg);   //tmpReg3:标识每一个数所在块的起始索引
        MicroAPI::Add(indexReg, tmpReg2, tmpReg3, maskReg); // indexReg:标识每一个数gather时的索引

        MicroAPI::DataCopyGather(gradYReg, gradYAddr, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg); // 按照indeReg中存储索引的顺序将数据搬到gradYReg,此时gradYReg [n, t, align_n]

        // 搬运norm， norm的shape应该是[n,t,aligin]
        // MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(normReg, normAddr);
        MicroAPI::DataCopy(normReg, normAddr + perT * INDEX_BLOCK_LEN * sizeof(float));

        MicroAPI::Mul(mulResultReg, gradYReg, normReg, maskReg);
        MicroAPI::ReduceSumWithDataBlock(dotProdReg, mulResultReg, maskReg); // 1*8

        // 将dotProdReg [1,t,align_n] 广播到 [n,t,align_n]
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(tempAddr, dotProdReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dotProdReg, tempAddr); // 8*8

        MicroAPI::Sub(subResultReg, gradYReg, dotProdReg, maskReg);
        MicroAPI::Mul(gradInputReg, subResultReg, normReg, maskReg);

        MicroAPI::DataCopyScatter(gradInputAddr, gradInputReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::CopyInNormAndSum(const uint64_t normOffset,
                                                                 const uint32_t normProcessNum,
                                                                 const uint64_t sumOffset,
                                                                 const uint32_t sumProcessNum)
{
    LocalTensor<float> normLocal = inputNormQue_.AllocTensor<float>();
    LocalTensor<float> sumLocal = inputSumQue_.AllocTensor<float>();
    
    DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
    DataCopyExtParams normParams{1, static_cast<uint32_t>(normProcessNum * sizeof(float)), 0, 0, 0};
    DataCopyExtParams sumParams{1, static_cast<uint32_t>(sumProcessNum  * sizeof(float)), 0, 0, 0};

    DataCopyPad(normLocal, norm_[normOffset], normParams, dataCopyPadExtParams);
    DataCopyPad(sumLocal, sum_[sumOffset], sumParams, dataCopyPadExtParams);

    inputNormQue_.EnQue(normLocal);
    inputSumQue_.EnQue(sumLocal);
}

__aicore__ inline void MhcSinkhornBackwardSimd::Process()
{
    // printf("=================进入MhcSinkhornBackwardSimd！\n");
    if (blockIdx_ >= tilingData_.needCoreNum) {
        return;
    }

    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    LocalTensor<float> tempLocal = tempBuffer_.Get<float>();

    uint32_t mask = 0;  // mask=0b00000000000000001111111111111111
    if (nSize_ == N_VALID_4) {
        mask = MASK_4;
    } else if (nSize_ == N_VALID_6) {
        mask = MASK_6;
    } else if (nSize_ == N_VALID_8) {
        mask = MASK_8;
    }
    Duplicate(maskLocal, mask, MASK_NUM); //Duplicate(maskLocal, mask, 8)
    // printf("=================111111111111111111\n");

    printf("[debug] blockIdx_=%lld\n", blockIdx_);
    for (int64_t i = 0; i < loop_; i++) {
        printf("[debug] loop_i=%lld\n", i);
        // 计算gradY起始偏移地址以及搬运的数据量
        LocalTensor<float> gradYLocal = inputGradYQue_.AllocTensor<float>();
        LocalTensor<float> gradInputLocal = outputQue_.AllocTensor<float>();
        // LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
        // LocalTensor<float> sumLocal = inputSumQue_.DeQue<float>();
        uint64_t gradYOffset = blockIdx_ * tilingData_.coreTSize * nSize_ * nSize_ +
                               i * loopTSize_ * nSize_ * nSize_;
        uint32_t loopSize = (i == loop_ - 1) ? tailLoopTSize_ : loopTSize_; // 整块处理的矩阵个数
        uint32_t gradYProcessNum = loopSize * (nSize_ * nSize_); // grad_y从GM搬到UB上的元素个数

        // printf("[debug] gradYOffset=%lld\n", gradYOffset);
        // printf("[debug] loopSize=%lld\n", loopSize);
        // printf("[debug] gradYProcessNum=%lld\n", gradYProcessNum);

        DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyExtParams gradYParams{1, static_cast<uint32_t>(gradYProcessNum * sizeof(float)), 0, 0, 0}; //n=4和8时，搬运天然32字节对齐，如果n=6时,可能会多搬4个脏数据，搬出搬同样数据量大小即可
        DataCopyPad(gradYLocal, gradY_[gradYOffset], gradYParams, dataCopyPadExtParams);

        SetWaitEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        
        __local_mem__ float *gradYAddr = (__local_mem__ float *)gradYLocal.GetPhyAddr();
        __local_mem__ float *gradInputAddr = (__local_mem__ float *)gradInputLocal.GetPhyAddr();
        __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();
        __local_mem__ float *tempAddr = (__local_mem__ float *)tempLocal.GetPhyAddr();
        // __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
        // __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();

        // 计算UB -> reg 循环次数
        // 每一次能搬几个T
        uint32_t repeatSize = Ops::Base::GetVRegSize() / BLOCK_SIZE; // 256/32=8
        uint16_t repeatTimes = Ceil(loopSize, repeatSize); // 总T / 每次T = 循环次数  假设总T=13  循环次数=2， 第一次处理8 第二次处理3

        // printf("[debug] repeatSize=%lld\n", repeatSize);
        // printf("[debug] repeatTimes=%lld\n", repeatTimes);

        // norm_out、sum_out搬运数据量：GM -> UB [行,列]
        uint32_t normSize = loopSize * nSize_ * nAlignSize_; // norm_out从GM搬到UB上的元素个数：
        uint32_t sumSize = loopSize * nAlignSize_; // sum_out从GM搬到UB上的元素个数：
        // printf("[debug] normSize=%lld\n", normSize);
        // printf("[debug] sumSize=%lld\n", sumSize);
        // printf("=================2222222222222222222222\n");
        // 迭代numIters-1次，迭代19次 列 -> 行 -> 列 -> 行 -> 列 -> ....
        for (int64_t iter = 20 - 1; iter > 0; iter--) {
        // for (int64_t iter = tilingData_.numIters - 1; iter > 0; iter--) {  // 参考A3写法，迭代下标从19开始
            printf("[debug] iter=%lld\n", iter);
            // norm_out、sum_out和iters迭代次数有关系，所以要在tilingData_.numIters中计算norm_out、sum_out起始偏移地址
            int64_t colOffset = 2 * iter + 1;   // 39 37 35
            int64_t rowOffset = 2 * iter;       // 38 36 34
            printf("[debug] colOffset=%lld\n", colOffset);
            printf("[debug] rowOffset=%lld\n", rowOffset);
            //列归一
            // Norm_out、Sum_out的偏移(起始索引)
            uint64_t normOffset = blockIdx_ * nSize_ * nAlignSize_ * 40 * tilingData_.coreTSize +
                                  colOffset * nSize_ * nAlignSize_ * tilingData_.coreTSize +
                                  i * loopTSize_ * nSize_ * nAlignSize_ * 40;
            uint64_t sumOffset = blockIdx_ * nAlignSize_ * 40 * tilingData_.coreTSize +
                                 colOffset * nAlignSize_ * tilingData_.coreTSize +
                                 i * loopTSize_ * nAlignSize_ * 40;

            // uint64_t normOffset = colOffset * tilingData_.totalLength * nSize_ * nAlignSize_ +
            //                       blockIdx_ * tilingData_.coreTSize * nSize_ * nAlignSize_ +
            //                       i * loopTSize_ * nSize_ * nAlignSize_;
            // uint64_t sumOffset = colOffset * tilingData_.totalLength * nAlignSize_ +
            //                      blockIdx_ * tilingData_.coreTSize * nAlignSize_ +
            //                      i * loopTSize_ * nAlignSize_;
            // printf("[debug][iter=%lld] normOffset=%lld\n", iter, normOffset);
            // printf("[debug][iter=%lld] sumOffset=%lld\n", iter, sumOffset);
            // printf("[debug] normSize=%lld\n", normSize);
            // printf("[debug] sumSize=%lld\n", sumSize);
            CopyInNormAndSum(normOffset, normSize, sumOffset, sumSize);
            LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
            LocalTensor<float> sumLocal = inputSumQue_.DeQue<float>();
            __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
            __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();

            for (uint16_t j = 0; j < repeatTimes; j++) {
                // printf("=================333333333333333333333333\n");
                uint32_t perT = min(repeatSize, loopSize - (repeatSize * j));
                // printf("[debug] perT=%lld\n", perT);
                // printf("[debug] repeatSize=%lld\n", repeatSize);
                // printf("[debug] repeatTimes=%lld\n", repeatTimes);
                // printf("[debug] loopSize=%lld\n", loopSize);
                // printf("[debug] loopSize - (repeatSize * repeatTimes)=%lld\n", loopSize - (repeatSize * j));
                // // printf("=================3.1====================\n");
                for (uint16_t k = 0; k < nSize_; k++) {
                    // // printf("=================3.2===================\n");
                    // 每一列计算的起始偏移（每个寄存器）
                    // printf("[debug] curGradYAddr=%lld\n", j * repeatSize * nSize_ + k * nSize_);
                    // printf("[debug] curNormAddr=%lld\n", j * repeatSize * nAlignSize_ + k * perT * nAlignSize_);
                    // printf("[debug] curSumAddr=%lld\n", j * repeatSize * nAlignSize_);
                    auto curGradYAddr = gradYAddr + j * repeatSize * nSize_ + k * nSize_;
                    auto curNormAddr = normAddr + j * repeatSize * nAlignSize_ + k * perT * nAlignSize_;
                    auto curSumAddr = sumAddr + j * repeatSize * nAlignSize_; // 和k无关
                    auto curGradInputAddr = gradInputAddr + j * repeatSize * nSize_ + k * nSize_;
                    // // printf("=================3.3===================\n");
                    CalcColBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, tempAddr, static_cast<uint32_t>(nSize_), perT);
                    // // printf("=================3.4===================\n");
                }
                __VEC_SCOPE__
                {
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
            }
            // // printf("=================3.5====================\n");
            // 算完释放
            inputNormQue_.FreeTensor<float>(normLocal);
            inputSumQue_.FreeTensor<float>(sumLocal);

            // 行归一
            // Norm_out、Sum_out的偏移
            // // printf("=================3.6====================\n");
            normOffset = blockIdx_ * nSize_ * nAlignSize_ * 40 * tilingData_.coreTSize +
                         rowOffset * nSize_ * nAlignSize_ * tilingData_.coreTSize +
                         i * loopTSize_ * nSize_ * nAlignSize_;
            sumOffset = blockIdx_ * nAlignSize_ * 40 * tilingData_.coreTSize +
                        rowOffset * nAlignSize_ * tilingData_.coreTSize +
                        i * loopTSize_ * nAlignSize_;
            // normOffset = rowOffset * tilingData_.totalLength * nSize_ * nAlignSize_ +
            //              blockIdx_ * tilingData_.coreTSize * nSize_ * nAlignSize_ +
            //              i * loopTSize_ * nSize_ * nAlignSize_;
            // sumOffset = rowOffset * tilingData_.totalLength * nAlignSize_ +
            //             blockIdx_ * tilingData_.coreTSize * nAlignSize_ +
            //             i * loopTSize_ * nAlignSize_;
            // norm_out、sum_out搬运数据量：GM -> UB
            CopyInNormAndSum(normOffset, normSize, sumOffset, sumSize);
            // // printf("=================3.7====================\n");

            normLocal = inputNormQue_.DeQue<float>();
            sumLocal = inputSumQue_.DeQue<float>();
            normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
            sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();  
            // // printf("=================3.8====================\n");

            for (uint16_t j = 0; j < repeatTimes; j++) {
                // // printf("=================44444444444444444444444\n");
                // 行归一
                // 计算偏移
                uint32_t perT = min(repeatSize, loopSize - (repeatSize * j));
                for (uint16_t k = 0; k < nSize_; k++) {
                    auto curGradYAddr = gradYAddr + j * repeatSize * nSize_ + k * nSize_;
                    auto curNormAddr = normAddr + j * repeatSize * nAlignSize_ + k * perT * nAlignSize_;
                    auto curSumAddr = sumAddr + j * nAlignSize_ + k * loopSize;
                    auto curGradInputAddr = gradInputAddr + j * repeatSize * nSize_ + k * nSize_;
                    CalcRowBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, tempAddr, static_cast<uint32_t>(nSize_), perT);
                }
                __VEC_SCOPE__
                {
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
            }
            inputNormQue_.FreeTensor<float>(normLocal);
            inputSumQue_.FreeTensor<float>(sumLocal);
        }

        // 最后一次迭归一 列 
        // Norm_out、Sum_out的偏移
        uint64_t normOffset = blockIdx_ * nSize_ * nAlignSize_ * 40 * tilingData_.coreTSize +
                              1 * nSize_ * nAlignSize_ * tilingData_.coreTSize +
                              i * loopTSize_ * nSize_ * nAlignSize_;
        uint64_t sumOffset = blockIdx_ * nAlignSize_ * 40 * tilingData_.coreTSize +
                             1 * nAlignSize_ * tilingData_.coreTSize +
                             i * loopTSize_ * nAlignSize_;
        // int64_t normOffset = 1 * tilingData_.totalLength * nSize_ * nAlignSize_ +
        //                      blockIdx_ * tilingData_.coreTSize * nSize_ * nAlignSize_ +
        //                      i * loopTSize_ * nSize_ * nAlignSize_;
        // int64_t sumOffset = 1 * tilingData_.totalLength * nAlignSize_ +
        //                     blockIdx_ * tilingData_.coreTSize * nAlignSize_ +
        //                     i * loopTSize_ * nAlignSize_;
        // int64_t normOffset = 1 * tilingData_.totalLength * nSize_ * nAlignSize_ +
        //                      2 * blockIdx_ * tilingData_.coreTSize * nSize_ * nAlignSize_ +
        //                      i * loopTSize_ * nSize_ * nAlignSize_;
        // int64_t sumOffset = 1 * tilingData_.totalLength * nSize_ * nAlignSize_ +
        //                     2 * blockIdx_ * tilingData_.coreTSize * nAlignSize_ +
        //                     i * loopTSize_ * nAlignSize_;
        
        // norm_out、sum_out搬运数据量：GM -> UB
        CopyInNormAndSum(normOffset, normSize, sumOffset, sumSize);

        LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
        LocalTensor<float> sumLocal = inputSumQue_.DeQue<float>();
        __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
        __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();   

        for (uint16_t j = 0; j < repeatTimes; j++) {
            // // printf("=================5555555555555555555555\n");
            uint32_t perT = min(repeatSize, loopSize - (repeatSize * j));
            for (uint16_t k = 0; k < nSize_; k++) {
                auto curGradYAddr = gradYAddr + j * repeatSize * nSize_ + k * nSize_;
                auto curNormAddr = normAddr + j * repeatSize * nAlignSize_ + k * perT * nAlignSize_;
                auto curSumAddr = sumAddr + j * repeatSize * nAlignSize_;
                auto curGradInputAddr = gradInputAddr + j * repeatSize * nSize_ + k * nSize_;
                CalcColBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, tempAddr, static_cast<uint32_t>(nSize_), perT);
            }
            __VEC_SCOPE__
            {
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }
        }
        inputNormQue_.FreeTensor<float>(normLocal);
        inputSumQue_.FreeTensor<float>(sumLocal);

        // 最后一次迭代Softmax 行
        normOffset = blockIdx_ * nSize_ * nAlignSize_ * 40 * tilingData_.coreTSize +
                     i * loopTSize_ * nSize_ * nAlignSize_;
        // normOffset = blockIdx_ * tilingData_.coreTSize * nSize_ * nAlignSize_ +
        //              i * loopTSize_ * nSize_ * nAlignSize_;
        // normOffset = 0 * tilingData_.totalLength * nSize_ * nAlignSize_ + 
        //             2 * blockIdx_ * tilingData_.coreTSize * nSize_ * nAlignSize_ + 
        //             i * loopTSize_ * nSize_ * nAlignSize_;
        // norm_out、sum_out搬运数据量：GM -> UB
        DataCopyExtParams normParams = {1, static_cast<uint32_t>(normSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(normLocal, norm_[normOffset], normParams, dataCopyPadExtParams);

        for (uint16_t j = 0; j < repeatTimes; j++) {
            // // printf("=================666666666666666666666666\n");
            uint32_t perT = min(repeatSize, loopSize - (repeatSize * j));
            for (uint16_t k = 0; k < nSize_; k++) {
                // 计算偏移
                auto curGradYAddr = gradYAddr + j * repeatSize * nSize_ + k * nSize_;
                auto curNormAddr = normAddr + j * repeatSize * nAlignSize_ + k * perT * nAlignSize_;
                auto curGradInputAddr = gradInputAddr + j * repeatSize * nSize_ + k * nSize_;
                CalcSoftmaxBackward(curGradYAddr, curNormAddr, curGradInputAddr, maskAddr, tempAddr, static_cast<uint32_t>(nSize_), perT);
            }
            __VEC_SCOPE__
            {
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }
        }

        __VEC_SCOPE__
        {
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        SetWaitEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        // // printf("=================7777777777777777777777777777\n");
        // CopyOut
        // 计算 grad_input 的地址偏移
        int64_t grad_input =  blockIdx_ * tilingData_.coreTSize * nSize_ * nSize_ + 
                        i * loopTSize_ * nSize_ * nSize_;
                    
        // grad_input搬运数据量：UB -> GM
        uint32_t gradInputdSize = loopSize * (nSize_ * nSize_); // grad_y从GM搬到UB上的元素个数：5*4*4
        DataCopyExtParams gradInputParams{1, static_cast<uint32_t>(gradInputdSize * sizeof(float)), 0, 0, 0}; //n=4和8时，搬运天然32字节对齐，如果n=6时,可能会多搬4个脏数据，搬出搬同样数据量大小即可

        DataCopyPad(gradInput_[grad_input], gradInputLocal, gradInputParams);
        inputGradYQue_.FreeTensor<float>(gradYLocal);
        inputNormQue_.FreeTensor<float>(normLocal);
        inputSumQue_.FreeTensor<float>(sumLocal);
        outputQue_.FreeTensor<float>(gradInputLocal);
    }
}

template <HardEvent EVENT>
__aicore__ inline void MhcSinkhornBackwardSimd::SetWaitEvent(HardEvent event)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<EVENT>(eventId);
    WaitFlag<EVENT>(eventId);
}

} // namespace MhcSinkhornBackward

#endif // ASCENDC_MHC_SINKHORN_BACKWARD_H