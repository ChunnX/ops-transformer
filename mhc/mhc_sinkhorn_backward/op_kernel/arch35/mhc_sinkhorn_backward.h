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
                                               uint32_t dataLen, uint32_t n, float eps);
    __aicore__ inline void CalcColBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                           __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                           uint32_t dataLen, uint32_t n, float eps);
    __aicore__ inline void CalcRowBackward(__local_mem__ float *gradYAddr, __local_mem__ float *normAddr,
                                           __local_mem__ float *sumAddr, __local_mem__ float *gradInputAddr,
                                           __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                           uint32_t dataLen, uint32_t n, float eps);

    GlobalTensor<float> gradY_;
    GlobalTensor<float> norm_;
    GlobalTensor<float> sum_;
    GlobalTensor<float> gradInput_;
    TPipe &pipe_;
    const MhcSinkhornBackwardTilingData &tilingData_;
    int64_t blockIdx_; // 核id
    int64_t loop_; // 循环次数
    int64_t loopTSize_; // 主块处理数据量
    int64_t tailLoopTSize_; // 尾块处理的数据量
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
    tailloopTSize_ = tilingData_.normCoreLastLoopTsize;
    // 如果是尾核的话
    if (blockIdx_ == tilingData_.needCoreNum - 1) {
        loop_ = tilingData_.tailCoreLoops;
        loopTSize_ = tilingData_.tailCorePerLoopTSize;
        tailloopTSize_ = tilingData_.tailCoreLastLoopTSize;
    }
}

__aicore__ inline void MhcSinkhornBackwardSimd::Process()
{
    if (blockIdx_ >= tilingData_.needCoreNum) {
        return;
    }

    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    LocalTensor<float> maxLocal = maxBuffer_.Get<float>();

    uint32_t mask = 0;
    if (tilingData_.nSize == N_VALID_4) {
        mask = MASK_4;
    } else if (tilingData_.nSize == N_VALID_6) {
        mask = MASK_6;
    } else if (tilingData_.nSize == N_VALID_8) {
        mask = MASK_8;
    }
    Duplicate(maskLocal, mask, MASK_NUM);

    for (int64_t i = 0; i < loop_; i++) {
        LocalTensor<float> gradYLocal = inputGradYQue_.AllocTensor<float>();
        LocalTensor<float> normLocal = inputNormQue_.AllocTensor<float>();
        LocalTensor<float> sumLocal = inputSumQue_.AllocTensor<float>();
        LocalTensor<float> gradInputLocal = outputQue_.AllocTensor<float>();

        // 计算起始偏移地址

        int64_t inputOffset = blockIdx_ * tilingData_.coreTSize * tilingData_.nSize * tilingData_.nSize + 
                              i * loopTSize_ * tilingData_.nSize * tilingData_.nSize;
        
        uint32_t loopSize = (i == loop_ - 1) ? tailloopTSize_ : loopTSize_; // T的大小
        loopSize = loopSize * (tilingData_.nSize * tilingData_.nSize); // 总的数据量

        // CopyIn
        DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(loopSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(gradYLocal, gradY_[inputOffset], dataCopyExtParams, dataCopyPadExtParams);
        DataCopyPad(normLocal, norm_[inputOffset], dataCopyExtParams, dataCopyPadExtParams);
        DataCopyPad(sumLocal, sum_[inputOffset], dataCopyExtParams, dataCopyPadExtParams);

        auto MTE2ToVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(MTE2ToVEventID);
        WaitFlag<HardEvent::MTE2_V>(MTE2ToVEventID);

        // 计算reg循环次数
        uint32_t repeatSize = Ops::Base::GetVRegSize() / tilingData_.nSize * tilingData_.nSize * BLOCK_SIZE; // 每次循环处理的数据量
        uint16_t repeatTimes = Ops::Base::CeilDiv(loopSize, repeatSize); // 总数据量 / 每次循环处理的数据量 = 循环次数

        __local_mem__ float *gradYAddr = (__local_mem__ float *)gradYLocal.GetPhyAddr();
        __local_mem__ float *normAddr = (__local_mem__ float *)normLocal.GetPhyAddr();
        __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();
        __local_mem__ float *gradInputAddr = (__local_mem__ float *)gradInputLocal.GetPhyAddr();
        __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();
        __local_mem__ float *maxAddr = (__local_mem__ float *)maxLocal.GetPhyAddr();

        auto dataLen = loopSize / tilingData_.nSize / tilingData_.nSize; // t大小

        // 迭代numIters次
        for (int64_t iter = tilingData_.numIters - 2; iter >= 0; iter--) {
            for (uint16_t i = 0; i < repeatTimes; i++) {
                // 列方向计算
                for (uint16_t j = 0; j < tilingData_.nSize; j++) {
                    // 计算偏移
                    auto curGradYAddr = gradYAddr + i * repeatSize + j;
                    auto curNormAddr = normAddr + i * repeatSize + j;
                    auto curSumAddr = sumAddr + i * repeatSize + j;
                    auto curGradInputAddr = gradInputAddr + i * repeatSize + j;

                    CalcColBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                   static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize),
                                   tilingData_.eps);
                }
                __VEC_SCOPE__
                {
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                          AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
                // 行方向计算
                for (uint16_t j = 0; j < tilingData_.nSize; j++) {
                    // 计算偏移
                    auto curGradYAddr = gradYAddr + i * repeatSize + j * tilingData_.nSize;
                    auto curNormAddr = normAddr + i * repeatSize + j * tilingData_.nSize;
                    auto curSumAddr = sumAddr + i * repeatSize + j * tilingData_.nSize;
                    auto curGradInputAddr = gradInputAddr + i * repeatSize + j * tilingData_.nSize;

                    CalcRowBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                   static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize),
                                   tilingData_.eps);
                }
            }
            __VEC_SCOPE__
            {
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }
        }

        // 列
        for (uint16_t i = 0; i < repeatTimes; i++) {
            for (uint16_t j = 0; j < tilingData_.nSize; j++) {
                auto curGradYAddr = gradYAddr + i * repeatSize + j;
                auto curNormAddr = normAddr + i * repeatSize + j;
                auto curSumAddr = sumAddr + i * repeatSize + j;
                auto curGradInputAddr = gradInputAddr + i * repeatSize + j;
                CalcColBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                               static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize),
                               tilingData_.eps);
            }
        }
        __VEC_SCOPE__
        {
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }

        // Softmax行
        for (uint16_t i = 0; i < repeatTimes; i++) {
            for (uint16_t j = 0; j < tilingData_.nSize; j++) {
                auto curGradYAddr = gradYAddr + i * repeatSize + j * tilingData_.nSize;
                auto curNormAddr = normAddr + i * repeatSize + j * tilingData_.nSize;
                auto curSumAddr = sumAddr + i * repeatSize + j * tilingData_.nSize;
                auto curGradInputAddr = gradInputAddr + i * repeatSize + j * tilingData_.nSize;
                CalcSoftmaxBackward(curGradYAddr, curNormAddr, curSumAddr, curGradInputAddr, maskAddr, maxAddr,
                                    static_cast<uint32_t>(dataLen), static_cast<uint32_t>(tilingData_.nSize),
                                    tilingData_.eps);
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
