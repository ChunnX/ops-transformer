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
constexpr int64_t MAX_BUFFER_SIZE = 256;
constexpr uint32_t MASK_4 = 0b00000000000000001111111111111111;
constexpr uint32_t MASK_6 = 0b00000000111111111111111111111111;
constexpr uint32_t MASK_8 = 0b11111111111111111111111111111111;
constexpr int64_t MASK_NUM = 8;
constexpr int64_t INDEX_BLOCK_LEN = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

class MhcSinkhornBackwardSimd {
public:
    __aicore__ inline MhcSinkhornBackwardSimd(TPipe &pipe, const MhcSinkhornBackwardTilingData &tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR norm, GM_ADDR sum, GM_ADDR gradInput, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcSoftmaxBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                               __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                               __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                               uint32_t dataLen, uint32_t n);
    __aicore__ inline void CalcColBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                           __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                           uint32_t dataLen, uint32_t n);
    __aicore__ inline void CalcRowBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                           __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                           uint32_t dataLen, uint32_t n);

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
    TQue<QuePosition::VECIN, 1> inputGradYQue_;
    TQue<QuePosition::VECIN, 1> inputNormQue_;
    TQue<QuePosition::VECIN, 1> inputSumQue_;
    TQue<QuePosition::VECOUT, 1> outputQue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> maxBuffer_;
};

__aicore__ inline void MhcSinkhornBackwardSimd::Init(GM_ADDR gradY, GM_ADDR norm, GM_ADDR sum, GM_ADDR gradInput,
                                                      GM_ADDR tiling)
{
    blockIdx_ = GetBlockIdx();
    gradY_.SetGlobalBuffer((__gm__ float *)(gradY));
    norm_.SetGlobalBuffer((__gm__ float *)(norm));
    sum_.SetGlobalBuffer((__gm__ float *)(sum));
    gradInput_.SetGlobalBuffer((__gm__ float *)(gradInput));
    pipe_.InitBuffer(maskBuffer_, MASK_BUFFER_SIZE);
    pipe_.InitBuffer(maxBuffer_, MAX_BUFFER_SIZE);
    pipe_.InitBuffer(inputGradYQue_, DOUBLE_BUFFER, tilingData_.coreTSize * tilingData_.nSize * tilingData_.nSize * sizeof(float));
    pipe_.InitBuffer(inputNormQue_, DOUBLE_BUFFER, tilingData_.coreTSize * tilingData_.nSize * tilingData_.nSize * sizeof(float));
    pipe_.InitBuffer(inputSumQue_, DOUBLE_BUFFER, tilingData_.coreTSize * tilingData_.nSize * tilingData_.nSize * sizeof(float));
    pipe_.InitBuffer(outputQue_, DOUBLE_BUFFER, tilingData_.coreTSize * tilingData_.nSize * tilingData_.nSize * sizeof(float));
    // 如果是主核的话
    loop_ = tilingData_.normCoreTLoops;
    loopTSize_ = tilingData_.normCorePerLoopTSize;
    tailLoopTSize_ = tilingData_.normCoreLastLoopTsize;
    // 如果是尾核的话
    if (blockIdx_ == tilingData_.needCoreNum - 1) {
        loop_ = tilingData_.tailCoreLoops;
        loopTSize_ = tilingData_.tailCorePerLoopTSize;
        tailLoopTSize_ = tilingData_.tailCoreLastLoopTSize;
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcRowBackward(__local_mem__ float *gradYAddr,
                                                                 __local_mem__ float *normAddr,
                                                                 __local_mem__ float *sumAddr,
                                                                 __local_mem__ float *gradInputAddr,
                                                                 __local_mem__ uint32_t *maskAddr,
                                                                 __local_mem__ float *maxAddr,
                                                                 uint32_t dataLen, uint32_t n, uint32_t perLoopT)
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
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen);

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
        MicroAPI::LoadAlign(normReg, normAddr + perLoopT * 8 * n);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_E2B_B32>>(sumReg, sumAddr + perLoopT * 8);
        
        MicroAPI::Mul(gradInputReg, gradYReg, normReg, maskReg);
        MicroAPI::ReduceSumWithDataBlock(sumGradYReg, gradInputReg, maskReg);

        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(maxAddr, sumGradYReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumGradYReg, maxAddr);

        MicroAPI::Sub(gradInputReg, gradYReg, sumGradYReg, maskReg);
        // sumReg广播

        MicroAPI::Div(gradInputReg, gradInputReg, sumReg, maskReg);

        MicroAPI::DataCopyScatter(gradInputAddr, gradInputReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}


__aicore__ inline void MhcSinkhornBackwardSimd::CalcColBackward(__local_mem__ float *gradYAddr,
                                                                 __local_mem__ float *normAddr,
                                                                 __local_mem__ float *sumAddr,
                                                                 __local_mem__ float *gradInputAddr,
                                                                 __local_mem__ uint32_t *maskAddr,
                                                                 __local_mem__ float *maxAddr,
                                                                 uint32_t dataLen, uint32_t n, uint32_t perLoopT)
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
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen);

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
        MicroAPI::LoadAlign(normReg, normAddr + perLoopT * 8 * n);
        MicroAPI::LoadAlign(sumReg, sumAddr + perLoopT * 8);
        
        MicroAPI::Mul(gradInputReg, gradYReg, normReg, maskReg);
        MicroAPI::ReduceSumWithDataBlock(sumGradYReg, gradInputReg, maskReg);

        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(maxAddr, sumGradYReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumGradYReg, maxAddr);

        MicroAPI::Sub(gradInputReg, gradYReg, sumGradYReg, maskReg);
        MicroAPI::Div(gradInputReg, gradInputReg, sumReg, maskReg);

        MicroAPI::DataCopyScatter(gradInputAddr, gradInputReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::CalcSoftmaxBackward(__local_mem__ float *gradYAddr, 
                                                                    __local_mem__ float *normAddr,
                                                                    __local_mem__ float *sumAddr,
                                                                    __local_mem__ float *gradInputAddr,
                                                                    __local_mem__ uint32_t *maskAddr, 
                                                                    __local_mem__ float *maxAddr,
                                                                    uint32_t dataLen, uint32_t n)
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
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen);

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
        MicroApi::LoadAlign(normReg, normAddr + perLoopT * 8 * n);

        MicroAPI::Mul(mulResultReg, gradYReg, normReg, maskReg);
        MicroAPI::ReduceSumWithDataBlock(dotProdReg, mulResultReg, maskReg);

        // 将dotProdReg [1,t,align_n] 广播到 [n,t,align_n]
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(maxAddr, dotProdReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dotProdReg, maxAddr);

        MicroAPI::Sub(subResultReg, gradYReg, dotProdReg, maskReg);
        MicroAPI::Mul(gradInputReg, subResultReg, normReg, maskReg);

        MicroAPI::DataCopyScatter(gradInputAddr, gradInputReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::Process()
{
    if (blockIdx_ >= tilingData_.needCoreNum) {
        return;
    }

    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    LocalTensor<float> maxLocal = maxBuffer_.Get<float>();

    uint32_t mask = 0;  // mask=0b00000000000000001111111111111111
    if (tilingData_.nSize == N_VALID_4) {
        mask = MASK_4;
    } else if (tilingData_.nSize == N_VALID_6) {
        mask = MASK_6;
    } else if (tilingData_.nSize == N_VALID_8) {
        mask = MASK_8;
    }
    Duplicate(maskLocal, mask, MASK_NUM); //Duplicate(maskLocal, mask, 8)

    for (int64_t i = 0; i < loop_; i++) {
        LocalTensor<float> gradYLocal = inputGradYQue_.AllocTensor<float>();
        LocalTensor<float> normLocal = inputNormQue_.AllocTensor<float>();
        LocalTensor<float> sumLocal = inputSumQue_.AllocTensor<float>();
        LocalTensor<float> gradInputLocal = outputQue_.AllocTensor<float>();

        // 计算起始偏移地址
        // 存疑
        int64_t inputOffsetGradY =  i * tilingData.normCorePerLoopTSize * tilingData_.nSize * tilingData_.nSize;
        
        
        
        uint32_t loopSize = (i == loop_ - 1) ? tailloopTSize_ : loopTSize_; // 整块处理的矩阵个数：5
        uint32_t copyInGradYSize = loopSize * (tilingData_.nSize * tilingData_.nSize); // grad_y从GM搬到UB上的元素个数：5*4*4


        // CopyIn：GM -> UB
        DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyExtParamsGradY{1, static_cast<uint32_t>(copyInGradYSize * sizeof(float)), 0, 0, 0}; //n=4和8时，搬运天然32字节对齐，如果n=6时,可能会多搬4个脏数据，搬出搬同样数据量大小即可

        DataCopyPad(gradYLocal, gradY_[inputOffsetGradY], dataCopyExtParamsGradY, dataCopyPadExtParams);


        auto MTE2ToVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(MTE2ToVEventID);
        WaitFlag<HardEvent::MTE2_V>(MTE2ToVEventID);
       


        // 计算reg循环次数
        uint32_t perLoopT = Ops::Base::GetVRegSize() / sizeof(float) / tilingData_.nSize / tilingData_.nAlignSize ; //计算一次可以搬运多少个矩阵到Register:256/4/8/4=2
        uint32_t repeatSize = perLoopT * tilingData_.nSize * tilingData_.nAlignSize; // 一次可以从UB搬到Reg中的元素个数：2*4*8   2*4*8  1*4*8
        uint16_t repeatTimes = Ops::Base::CeilDiv(loopSize * tilingData_.nSize * tilingData_.nAlignSize, repeatSize); // 循环次数：需要循环多少次才可以把UB中数据搬运完到Reg中
        //  loopSize 5   
        // repeatSize 2*4*8  
        // perLoopT 2 2 1 
        // repeatSize 64 64 32
        // repeatTimes 3

        // 迭代numIters-1次，迭代19次 列 -> 行 -> 列 -> 行 -> 列 -> ....
        for (int64_t iter = tilingData_.numIters - 1; iter > 0; iter--) {  // 为什么下标从18开始，不从19开始
            // norm_out、sum_out和iters迭代次数有关系
            int64_t inputOffsetNorm = blockIdx_ * tilingData_.coreTSize * tilingData_.nSize * tilingData_.nAlignSize + 
                              i * loopTSize_ * tilingData_.nSize * tilingData_.nAlignSize;
            int64_t inputOffsetSum = blockIdx_ * tilingData_.coreTSize * tilingData_.nAlignSize + 
                              i * loopTSize_ * tilingData_.nAlignSize;

            uint32_t copyInNormSize = 2 * loopSize * (tilingData_.nSize * tilingData_.nAlignSize); // norm_out从GM搬到UB上的元素个数：2*5*4*8
            uint32_t copyInSumSize = 2* loopSize * tilingData_.nAlignSize; // sum_out从GM搬到UB上的元素个数：2*5*8

            DataCopyExtParams dataCopyExtParamsNorm{1, static_cast<uint32_t>(copyInNormSize * sizeof(float)), 0, 0, 0};
            DataCopyExtParams dataCopyExtParamsSum{1, static_cast<uint32_t>(copyInSumSize  * sizeof(float)), 0, 0, 0};

            DataCopyPad(normLocal, norm_[inputOffsetNorm], dataCopyExtParamsNorm, dataCopyPadExtParams);
            DataCopyPad(sumLocal, sum_[inputOffsetSum], dataCopyExtParamsSum, dataCopyPadExtParams);    

            __local_mem__ float *gradYAddr = (__local_mem__ float *)gradYLocal.GetPhyAddr();
            __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
            __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();
            __local_mem__ float *gradInputAddr = (__local_mem__ float *)gradInputLocal.GetPhyAddr();
            __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();
            __local_mem__ float *maxAddr = (__local_mem__ float *)maxLocal.GetPhyAddr();

            // copyIn: UB -> Register
            for (uint16_t i = 0; i < repeatTimes; i++) {
                // 列归一
                auto curGradYAddr = gradYAddr + i * repeatSize;
                auto curNormAddr = normAddr + i * repeatSize;
                auto curSumAddr = sumAddr + i * repeatSize;
                auto curGradInputAddr = gradInputAddr + i * repeatSize;
                CalcColBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                static_cast<uint32_t>(loopSize), static_cast<uint32_t>(tilingData_.nSize), min(perLoopT, loopSize - perLoopT * (repeatTimes + 1)));
                __VEC_SCOPE__
                {
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
                // 行归一
                // 计算偏移
                auto curGradYAddr = gradYAddr + i * repeatSize + j * min(perLoopT, dataLen - perLoopT * (repeatTimes + 1)) * INDEX_BLOCK_LEN;
                auto curNormAddr = normAddr + i * repeatSize + j * min(perLoopT, dataLen - perLoopT * (repeatTimes + 1)) * INDEX_BLOCK_LEN;
                auto curSumAddr = sumAddr + i * repeatSize + j * min(perLoopT, dataLen - perLoopT * (repeatTimes + 1)) * INDEX_BLOCK_LEN;
                auto curGradInputAddr = gradInputAddr + i * repeatSize + j * tilingData_.nSize;

                CalcRowBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize), min(perLoopT, dataLen - perLoopT * (repeatTimes + 1)));
                __VEC_SCOPE__
                {
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
            }
        }

        // 最后一次迭代列
        for (uint16_t i = 0; i < repeatTimes; i++) {
            for (uint16_t j = 0; j < tilingData_.nSize; j++) {
                auto curGradYAddr = gradYAddr + i * repeatSize + j;
                auto curNormAddr = normAddr + i * repeatSize + j;
                auto curSumAddr = sumAddr + i * repeatSize + j;
                auto curGradInputAddr = gradInputAddr + i * repeatSize + j;
                CalcColBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                               static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize), min(perLoopT, dataLen - perLoopT * (repeatTimes + 1)));
            }
        }
        __VEC_SCOPE__
        {
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        // 最后一次迭代Softmax
        for (uint16_t i = 0; i < repeatTimes; i++) {
            for (uint16_t j = 0; j < tilingData_.nSize; j++) {
                auto curGradYAddr = gradYAddr + i * repeatSize + j * tilingData_.nSize;
                auto curNormAddr = normAddr + i * repeatSize + j * tilingData_.nSize;
                auto curSumAddr = sumAddr + i * repeatSize + j * tilingData_.nSize;
                auto curGradInputAddr = gradInputAddr + i * repeatSize + j * tilingData_.nSize;
                CalcSoftmaxBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                    static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize));
            }
        }
        inputGradYQue_.FreeTensor<float>(gradYLocal);
        inputNormQue_.FreeTensor<float>(normLocal);
        inputSumQue_.FreeTensor<float>(sumLocal);
        __VEC_SCOPE__
        {
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        auto VToMTE3EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(VToMTE3EventID);
        WaitFlag<HardEvent::V_MTE3>(VToMTE3EventID);

        // CopyOut
        DataCopyPad(gradInput_[inputOffset], gradInputLocal, dataCopyExtParams);
        outputQue_.FreeTensor<float>(gradInputLocal);
    }
}
} // namespace MhcSinkhornBackward

#endif // ASCENDC_MHC_SINKHORN_BACKWARD_H
