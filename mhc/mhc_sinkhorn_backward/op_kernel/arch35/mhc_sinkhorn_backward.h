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
                                               __local_mem__ float *maxAddr, uint32_t dataLen, uint32_t n);

    __aicore__ inline void CalcColBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                           __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                           uint32_t dataLen, uint32_t n);

    __aicore__ inline void CalcRowBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                           __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                           uint32_t dataLen, uint32_t n);
    
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
    tailLoopTSize_ = tilingData_.normCoreLastLoopTSize;
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
                                                                 uint32_t dataLen, uint32_t n)
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
        MicroAPI::LoadAlign(normReg, normAddr);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumReg, sumAddr);
        
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
                                                                 uint32_t dataLen, uint32_t n)
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
        MicroAPI::LoadAlign(normReg, normAddr );
        MicroAPI::LoadAlign(sumReg, sumAddr);
        
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
        MicroAPI::LoadAlign(normReg, normAddr);

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

__aicore__ inline void MhcSinkhornBackwardSimd::CopyInNormAndSum(const uint64_t normOffset, const uint32_t normProcessNum, const uint64_t sumOffset, const uint32_t sumProcessNum)
{
    LocalTensor<float> normLocal = inputNormQue_.AllocTensor<float>();
    LocalTensor<float> sumLocal = inputSumQue_.AllocTensor<float>();
    DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};

    DataCopyExtParams dataCopyExtParamsNorm{1, static_cast<uint32_t>(normProcessNum * sizeof(float)), 0, 0, 0};
    DataCopyExtParams dataCopyExtParamsSum{1, static_cast<uint32_t>(sumProcessNum  * sizeof(float)), 0, 0, 0};

    DataCopyPad(normLocal, norm_[normOffset], dataCopyExtParamsNorm, dataCopyPadExtParams);
    DataCopyPad(sumLocal, sum_[sumOffset], dataCopyExtParamsSum, dataCopyPadExtParams);

    inputNormQue_.EnQue(normLocal);
    inputSumQue_.EnQue(sumLocal);
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
        LocalTensor<float> gradInputLocal = outputQue_.AllocTensor<float>();

        // grad_y搬运数据量：GM -> UB
        uint32_t loopSize = (i == loop_ - 1) ? tailLoopTSize_ : loopTSize_; // loopSize:整核/尾核每次循环处理的矩阵个数:11
    
        // 计算grad_y起始偏移地址
        int64_t inputOffsetGradY =  blockIdx_ * tilingData_.coreTSize * tilingData_.nSize * tilingData_.nSize + 
                                i * loopSize * tilingData_.nSize * tilingData_.nSize;

        uint32_t copyInGradYSize = loopSize * (tilingData_.nSize * tilingData_.nSize); // grad_y从GM搬到UB上的元素个数：11*4*4
        
        DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyExtParamsGradY{1, static_cast<uint32_t>(copyInGradYSize * sizeof(float)), 0, 0, 0}; //n=4和8时，搬运天然32字节对齐，如果n=6时,可能会多搬4个脏数据，搬出搬同样数据量大小即可
        DataCopyPad(gradYLocal, gradY_[inputOffsetGradY], dataCopyExtParamsGradY, dataCopyPadExtParams);

        SetWaitEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        
        // grad_y、grad_input、mask、max
        __local_mem__ float *gradYAddr = (__local_mem__ float *)gradYLocal.GetPhyAddr();
        __local_mem__ float *gradInputAddr = (__local_mem__ float *)gradInputLocal.GetPhyAddr();
        __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();
        __local_mem__ float *maxAddr = (__local_mem__ float *)maxLocal.GetPhyAddr();

        uint32_t perLoopT = 8 ; // 每个loop_循环整块每次固定处理8个矩阵
        uint32_t repeatSize = 64; // 整块一次可以从UB搬到Reg中元素个数：8*8=64
        uint16_t repeatTimes = Ceil(loopSize * tilingData_.nSize * tilingData_.nAlignSize, repeatSize); // // 计算UB -> reg 循环次数 3

        // 迭代numIters-1次，迭代19次 列 -> 行 -> 列 -> 行 -> 列 -> ....
        for (int64_t iter = tilingData_.numIters - 1; iter > 0; iter--) {  // 参考A3写法，迭代下标从19开始
            // norm_out、sum_out和iters迭代次数有关系，所以要在tilingData_.numIters中计算norm_out、sum_out起始偏移地址
            int64_t colOffset = 2 * iter + 1;   // 39 37 35
            int64_t rowOffset = 2 * iter;       // 38 36 34
            
            //列归一
            // Norm_out、Sum_out的偏移
            int64_t inputOffsetNorm = colOffset * tilingData_.totalLength * tilingData_.nSize * tilingData_.nAlignSize + 
                                2 * blockIdx_ * tilingData_.coreTSize * tilingData_.nSize * tilingData_.nAlignSize + 
                                i * loopSize * tilingData_.nSize * tilingData_.nAlignSize; // 改动点：loopTSize_ -> loopSize，结果应该是一样的，但可以区别整核/尾核每次循环处理矩阵个数不同
            int64_t inputOffsetSum = colOffset * tilingData_.totalLength * tilingData_.nSize * tilingData_.nAlignSize + 
                                2 * blockIdx_ * tilingData_.coreTSize * tilingData_.nAlignSize + 
                                i * loopSize * tilingData_.nAlignSize;
            
            // norm_out、sum_out搬运数据量：GM -> UB
            uint32_t copyInNormSize = 2 * loopSize * tilingData_.nSize * tilingData_.nAlignSize; // norm_out从GM搬到UB上的元素个数：2*11*4*8, 这里就是文档里说的loop_每次循环处理矩阵的个数
            uint32_t copyInSumSize = 2 * loopSize * tilingData_.nAlignSize; // sum_out从GM搬到UB上的元素个数：2*11*8
            CopyInNormAndSum(inputOffsetNorm, copyInNormSize, inputOffsetSum, copyInSumSize); // norm、sum一块搬到normLocal、sumLocal上

            LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
            LocalTensor<float> sumLocal = inputSumQue_.DeQue<float>();

            __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
            __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();

            for (uint16_t j = 0; j < repeatTimes; j++) {    // 每次repeatTimes处理8个矩阵
                for(uint16_t n = 0; n < tilingData_.n; n++){
                    // grad_y和grad_input的取址方式一样，因为都是用gather收集索引，只要知道起始下标就可以
                    auto curGradYAddr = gradYAddr + j * repeatSize / 2 + n * tilingData_.nSize;   // gradYAddr：grad_y在当前loop_循环内的偏移地址
                    auto curGradInputAddr = gradInputAddr + j * repeatSize + n * tilingData_.nSize;
                    
                    // norm、sum的取址方式一样，因为内存排布已经连续，顺着取就可以
                    perLoopT =  min(perLoopT, loopSize - perLoopT * (repeatTimes + 1));
                    uint32_t dataLen = perLoopT * tilingData_.nAlignSize; //  变动搬运数据量
                    auto curNormAddr = normAddr + j * dataLen + n * perLoopT * tilingData_.nAlignSize; 
                    auto curSumAddr = sumAddr + j * dataLen; //sum_out_col常驻
                    
                    CalcColBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                    static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize));
                    __VEC_SCOPE__
                    {
                        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                    }
                }
            }

            inputNormQue_.FreeTensor<float>(normLocal);
            inputSumQue_.FreeTensor<float>(sumLocal);

            // 行归一
            // Norm_out、Sum_out的偏移
            inputOffsetNorm = rowOffset * tilingData_.totalLength * tilingData_.nSize * tilingData_.nAlignSize + 
                                2 * blockIdx_ * tilingData_.coreTSize * tilingData_.nSize * tilingData_.nAlignSize + 
                                i * loopSize * tilingData_.nSize * tilingData_.nAlignSize;
            inputOffsetSum = rowOffset * tilingData_.totalLength * tilingData_.nSize * tilingData_.nAlignSize + 
                                2 * blockIdx_ * tilingData_.coreTSize * tilingData_.nAlignSize + 
                                i * loopSize * tilingData_.nAlignSize;
            // norm_out、sum_out搬运数据量：GM -> UB
            copyInNormSize = 2 * loopSize * tilingData_.nSize * tilingData_.nAlignSize; // norm_out从GM搬到UB上的元素个数：2*11*4*8
            copyInSumSize = 2 * loopSize * tilingData_.nAlignSize; // sum_out从GM搬到UB上的元素个数：2*11*8
            CopyInNormAndSum(inputOffsetNorm, copyInNormSize, inputOffsetSum, copyInSumSize); // norm、sum一块搬到normLocal、sumLocal上

            LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
            LocalTensor<float> sumLocal = inputSumQue_.DeQue<float>();

            normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
            sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();   
            
            for (uint16_t j = 0; j < repeatTimes; j++) {
                for(uint16_t n=0; n < tilingData_.n; n++){
                    auto curGradYAddr = gradYAddr + j * repeatSize / 2 + n * tilingData_.nSize;   // gradYAddr：grad_y在当前loop_循环内的偏移地址   j * perLoopT * tilingData_.nSize * tilingData_.nSize：前面循环了多少个整块(8)  n * tilingData_.nSize：取所有矩阵第n行的起始索引
                    auto curGradInputAddr = gradInputAddr + j * repeatSize + n * tilingData_.nSize;
                    
                    perLoopT =  min(perLoopT, loopSize - perLoopT * (repeatTimes + 1));

                    auto curNormAddr = normAddr + j * dataLen + n * perLoopT * tilingData_.nAlignSize; 
                    auto curSumAddr = sumAddr + n * loopSize + j * 8; //sum_out_row需广播

                    uint32_t dataLen = perLoopT * tilingData_.nAlignSize; //  变动搬运数据量

                    CalcRowBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize)); // 这个地方应该要新增perLoopT类似的参数
                }
                __VEC_SCOPE__
                {
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
            }
            inputNormQue_.FreeTensor<float>(normLocal);
            inputSumQue_.FreeTensor<float>(sumLocal);
        }

        // 最后一次列归一
        // Norm_out、Sum_out的偏移
        int64_t inputOffsetNorm = 1 * tilingData_.totalLength * tilingData_.nSize * tilingData_.nAlignSize + 
                                2 * blockIdx_ * tilingData_.coreTSize * tilingData_.nSize * tilingData_.nAlignSize + 
                                i * loopSize * tilingData_.nSize * tilingData_.nAlignSize;;

        int64_t inputOffsetSum = 1 * tilingData_.totalLength * tilingData_.nSize * tilingData_.nAlignSize + 
                            2 * blockIdx_ * tilingData_.coreTSize * tilingData_.nAlignSize + 
                            i * loopSize * tilingData_.nAlignSize;
        
        // norm_out、sum_out搬运数据量：GM -> UB
        uint32_t copyInNormSize = 2 * loopSize * tilingData_.nSize * tilingData_.nAlignSize; // norm_out从GM搬到UB上的元素个数：2*11*4*8, 这里就是文档里说的loop_每次循环处理矩阵的个数
        uint32_t copyInSumSize = 2 * loopSize * tilingData_.nAlignSize; // sum_out从GM搬到UB上的元素个数：2*11*8
        CopyInNormAndSum(inputOffsetNorm, copyInNormSize, inputOffsetSum, copyInSumSize); // norm、sum一块搬到normLocal、sumLocal上

        LocalTensor<float> normLocal = inputNormQue_.DeQue<float>();
        LocalTensor<float> sumLocal = inputSumQue_.DeQue<float>();
        normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
        sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();   

        for (uint16_t j = 0; j < repeatTimes; j++) {    // 每次repeatTimes处理8个矩阵
            for(uint16_t n = 0; n < tilingData_.n; n++){
                // grad_y和grad_input的取址方式一样，因为都是用gather收集索引，只要知道起始下标就可以
                auto curGradYAddr = gradYAddr + j * repeatSize / 2 + n * tilingData_.nSize;   // gradYAddr：grad_y在当前loop_循环内的偏移地址   j * perLoopT * tilingData_.nSize * tilingData_.nSize：前面循环了多少个整块(8)  n * tilingData_.nSize：取所有矩阵第n行的起始索引
                auto curGradInputAddr = gradInputAddr + j * repeatSize + n * tilingData_.nSize;
                
                // norm、sum的取址方式一样，因为内存排布已经连续，顺着取就可以
                perLoopT =  min(perLoopT, loopSize - perLoopT * (repeatTimes + 1));
                uint32_t dataLen = perLoopT * tilingData_.nAlignSize; //  变动搬运数据量
                auto curNormAddr = normAddr + j * dataLen + n * perLoopT * tilingData_.nAlignSize; 
                auto curSumAddr = sumAddr + j * dataLen; //sum_out_col常驻
                
                CalcColBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize));
                __VEC_SCOPE__
                {
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
            }
        }

        inputNormQue_.FreeTensor<float>(normLocal);
        inputSumQue_.FreeTensor<float>(sumLocal);

        // 最后一次迭代Softmax
        inputOffsetNorm = 0 * tilingData_.totalLength * tilingData_.nSize + tilingData_.nAlignSize + 
                            2 * blockIdx_ * tilingData_.coreTSize * tilingData_.nSize * tilingData_.nAlignSize + 
                            i * loopSize * tilingData_.nSize * tilingData_.nAlignSize;
        // norm_out、sum_out搬运数据量：GM -> UB
        copyInNormSize = 2 * loopSize * tilingData_.nSize * tilingData_.nAlignSize; // norm_out从GM搬到UB上的元素个数：2*5*4*8
        dataCopyExtParamsNorm = {1, static_cast<uint32_t>(copyInNormSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(normLocal, norm_[inputOffsetNorm], dataCopyExtParamsNorm, dataCopyPadExtParams);

        for (uint16_t j = 0; j < repeatTimes; j++) {
            for(uint16_t n=0; n < tilingData_.n; n++){
                auto curGradYAddr = gradYAddr + j * repeatSize / 2 + n * tilingData_.nSize;   // gradYAddr：grad_y在当前loop_循环内的偏移地址   j * perLoopT * tilingData_.nSize * tilingData_.nSize：前面循环了多少个整块(8)  n * tilingData_.nSize：取所有矩阵第n行的起始索引
                auto curGradInputAddr = gradInputAddr + j * repeatSize + n * tilingData_.nSize;
                
                perLoopT =  min(perLoopT, loopSize - perLoopT * (repeatTimes + 1));
                uint32_t dataLen = perLoopT * tilingData_.nAlignSize; //  变动搬运数据量
                auto curNormAddr = normAddr + j * dataLen + n * perLoopT * tilingData_.nAlignSize; 
                CalcSoftmaxBackward(curGradYAddr, curNormAddr, curGradInputAddr, maskAddr, maxAddr, static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize));
            }
            __VEC_SCOPE__
            {
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }
        }

        inputGradYQue_.FreeTensor<float>(gradYLocal);
        inputNormQue_.FreeTensor<float>(normLocal);
        inputSumQue_.FreeTensor<float>(sumLocal);
        __VEC_SCOPE__
        {
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        SetWaitEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        // CopyOut
        // 计算 grad_input 的地址偏移
        int64_t grad_input =  blockIdx_ * tilingData_.coreTSize * tilingData_.nSize * tilingData_.nSize + 
                        i * tilingData_.normCorePerLoopTSize * tilingData_.nSize * tilingData_.nSize;
                    
        // grad_input搬运数据量：UB -> GM
        uint32_t copyOutGradInputdSize = loopSize * (tilingData_.nSize * tilingData_.nSize); // grad_y从GM搬到UB上的元素个数：5*4*4
        
        DataCopyExtParams dataCopyExtParamsGradInput{1, static_cast<uint32_t>(copyOutGradInputdSize * sizeof(float)), 0, 0, 0}; //n=4和8时，搬运天然32字节对齐，如果n=6时,可能会多搬4个脏数据，搬出搬同样数据量大小即可
        dataCopyPadExtParams = {false, 0, 0, 0};

        DataCopyPad(gradInput_[grad_input], gradInputLocal, dataCopyExtParamsGradInput);
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