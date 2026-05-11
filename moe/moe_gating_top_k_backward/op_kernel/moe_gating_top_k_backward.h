/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_backward.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_E_K_GENERALIZED_H
#define MOE_GATING_TOP_K_E_K_GENERALIZED_H
#include "kernel_operator.h"
#include "common.h"
#include "basic_api/kernel_operator_utils_intf.h"
namespace MoeGatingTopKBackward {
using namespace AscendC;

template <typename T>
class MoeGatingTopKBackwardKernel {
public:
    __aicore__ inline MoeGatingTopKBackwardKernel(){};
    __aicore__ inline void Init(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx, GM_ADDR gradX, GM_ADDR out, const MoeGatingTopKBackwardTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GetGatherOffsetIndex(); // fqy
    __aicore__ inline void CopyInGradY(); // lc
    __aicore__ inline void CopyInXNorm(); // fqy
    __aicore__ inline void CopyInExpertIdx(); // fqy
    __aicore__ inline void Sigmoid(); // fqy
    __aicore__ inline void GetGradXNorm(); // lc
    __aicore__ inline void GetGradX(); // lc
    __aicore__ inline void CopyOut();  // lc

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> gradYQue_;
    TQue<QuePosition::VECOUT, 1> indicesQue_;
    TQue<QuePosition::VECOUT, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    TBuf<TPosition::VECCALC> bufk0_;
    TBuf<TPosition::VECCALC> bufk1_;
    TBuf<TPosition::VECCALC> bufn2_;
    TBuf<TPosition::VECCALC> bufn3_;
    TBuf<TPosition::VECCALC> bufs_;
    TBuf<TPosition::VECCALC> bufk4Add_;
    TBuf<TPosition::VECCALC> bufk4Index_;
    TBuf<TPosition::VECCALC> bufk4RecipSumW_;

    GlobalTensor<float> xNormGm_;
    GlobalTensor<T> gradYGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> gradXGm_;

    LocalTensor<float> xNormLocal_;
    LocalTensor<float> gradYLocal_;
    LocalTensor<float> expertIdxLocal_;

    int64_t blockIdx_ = 0;
    int64_t loopTimes_ = 0;
    int64_t tailRows_ = 0;
    int64_t curRows_ = 0;
    // int64_t blockIdx_ = 0;
    // int64_t perCoreRowCount_ = 0;
    // int64_t curCoreRowCount_ = 0;
    // int64_t expertCount_ = 0;

    const MoeGatingTopKBackwardTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::Init(GM_ADDR xNorm, GM_ADDR gradY, GM_ADDR expertIdx, GM_ADDR gradX, GM_ADDR out, const MoeGatingTopKBackwardTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    // 获得当前核所需要的处理的loop次数、尾loop搬入的token数。
    if (blockIdx_ == GetBlockNum() - 1) {
        loopTimes_ = tilingData_->lastLoopTimes;
        tailRows_ = tilingData_->lastTailRows;

    } else {
        loopTimes_ = tilingData_->perLoopTimes;
        tailRows_ = tilingData_->perTailRows;
    }

    // init gm buf
    xNormGm_.SetGlobalBuffer((__gm__ float *)xNorm + tilingData_->perCoreRows * tilingData_->expertCount * blockIdx_);
    gradYGm_.SetGlobalBuffer((__gm__ T *)gradY + tilingData_->perCoreRows * tilingData_->k * blockIdx_);
    expertIdx.SetGlobalBuffer((__gm__ int32_t *)expertIdx + tilingData_->perCoreRows * tilingData_->k * blockIdx_);
    gradX.SetGlobalBuffer((__gm__ T *)gradX + tilingData_->perCoreRows * tilingData_->expertCount * blockIdx_);

    // init que
    pipe_->InitBuffer(gradYQue_, NUM_TWO, AlignBytes(tilingData_->baseRows * tilingData_->k, tilingData_->gradYDtype));
    pipe_->InitBuffer(indicesQue_, NUM_TWO, AlignBytes(tilingData_->baseRows * tilingData_->k, SIZE_OF_INT32));
    pipe_->InitBuffer(xQue_, NUM_TWO, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(outQue_, NUM_ONE, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, tilingData_->gradYDtype));
    
    pipe_->InitBuffer(bufk0_, AlignBytes(tilingData_->baseRows * tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk1_, AlignBytes(tilingData_->baseRows * tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk4Add_, AlignBytes(tilingData_->baseRows * tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk4Index_, AlignBytes(tilingData_->baseRows * tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufk4RecipSumW_, AlignBytes(tilingData_->baseRows * tilingData_->k, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufn2_, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufn3_, AlignBytes(tilingData_->baseRows * tilingData_->expertCount, SIZE_OF_FLOAT32));
    pipe_->InitBuffer(bufs_, AlignBytes(tilingData_->baseRows * 1, SIZE_OF_FLOAT32));
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::Process()
{
    GetGatherOffsetIndex(); // fqy
    for(int64_t loopIdx = 0; loopIdx < loopTimes_; loopIdx++) {
        curRows_ = loopIdx == loopTimes_ - 1 ? tailRows_ : tilingData_->baseRows;
        CopyInGradY(); // lc
        CopyInXNorm(); // fqy
        CopyInExpertIdx(); // fqy
        
        Sigmoid(); // fqy
        GetGradXNorm(); // lc
        GetGradX(); // lc

        CopyOut();  // lc
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::GetGatherOffsetIndex()
{
    LocalTensor<int32_t> bufk0 = bufk0_.Get<int32_t>();
    LocalTensor<int32_t> bufk4Add = bufk4Add_.Get<int32_t>();
    Arange(bufk0, static_cast<int32_t>(0), static_cast<int32_t>(tilingData_->expertCount), static_cast<int32_t>(tilingData_->baseRows));
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V); // Arange中存在S和V，不同count结束的操作流不一样
    PipeBarrier<PIPE_V>();
    uint32_t dstShape[NUM_TWO] = { tilingData_->baseRows, tilingData_->k };
    uint32_t srcShape[NUM_TWO] = { tilingData_->baseRows, 1 };
    BroadCast<int32_t, NUM_TWO, 1>(bufk4Add, bufk0, dstShape, srcShape); //下一次使用bufk4Add前PIPE_V
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::CopyInGradY()
{
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::CopyInXNorm()
{
    xNormLocal_ = xQue_.AllocTensor<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = static_cast<uint32_t>(curRows_ * tilingData_->expertCount * sizeof(float));
    copyParams.srcStride = static_cast<uint32_t>(0);
    copyParams.dstStride = static_cast<uint32_t>(0);
    DataCopyPadExtParams<float> padParams{true, 0, 0, 0}; // 不能为false？，必须填充数值0，需要计算右侧填充的元素个数吗？待测试
    DataCopyPad(xNormLocal_, this->xNormGm_, copyParams, padParams); // init中已经给Gm设定核间偏移量了，不需要再设置

    xQue_.EnQue(xNormLocal_);
    xNormLocal_ = xQue_.DeQue<float>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::CopyInExpertIdx()
{
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::Sigmoid()
{
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::GetGradXNorm()
{
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::GetGradX()
{
}

template <typename T>
__aicore__ inline void MoeGatingTopKBackwardKernel<T>::CopyOut()
{
}

} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_E_K_GENERALIZED_H