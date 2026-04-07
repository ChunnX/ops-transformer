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
 * \file masked_causal_conv1d_backward.h
 * \brief Arch35 AICore kernel for masked_causal_conv1d_backward (W=3)
 */

#ifndef MASK_EDCAUSAL_CONV1D_BACKWARD_H
#define MASK_EDCAUSAL_CONV1D_BACKWARD_H

#include "kernel_operator.h"
#include "vf/compute.h"
#include "masked_causal_conv1d_backward_struct.h"

namespace MaskedCausalConv1dBackwardKernelNS {
using namespace AscendC;
using MaskedCausalConv1dBackwardArch35Tiling::MaskedCausalConv1dBackwardTilingDataV35;
namespace {
    using namespace AscendC;
    #define _USE_DBG_PRINT 1
    template<typename T>
    __aicore__ inline void DisplayTensor(const LocalTensor<T>& tsr, const uint32_t count, const uint32_t type, const __gm__ char* note) {
        printf("Current block: %d , display %d elements", GetBlockIdx(), count);
        printf("%s:\n", (note ? note : "(null)"));  // 打印备注信息
        const uint32_t elementsPerRow = 16;         // 每16个元素

        for (uint32_t i = 0; i < count; ++i) {
            // 根据类型打印元素（整数或浮点数）
            if (type == 0) {
                // 关键修改：%2u 表示占2个字符宽度，不足时左侧补空格
                printf("[%u]: %d ", i, tsr.GetValue(i));  // 整数格式
            } else {
                // 关键修改：%2u 实现索引对齐
                printf("[%u]: %f ", i, tsr.GetValue(i));  // 浮点数格式
            }

            // 每行满16个元素后换行（最后一行不足16个也会在结束时换行）
            if ((i + 1) % elementsPerRow == 0) {
                printf("");
            }
        }
        // 如果总元素数不是16的倍数，最后一行末尾补充换行
        if (count % elementsPerRow != 0) {
            printf("\n");
        }
    }

    template<typename T>
    __aicore__ inline void DisplayTensor(const GlobalTensor<T>& tsr, const uint32_t count, const uint32_t type, const __gm__ char* note) {
        printf("Current block: %d , display %d elements", GetBlockIdx(), count);
        printf("%s:\n", (note ? note : "(null)"));  // 打印备注信息
        const uint32_t elementsPerRow = 16;         // 每16个元素

        for (uint32_t i = 0; i < count; ++i) {
            // 根据类型打印元素（整数或浮点数）
            if (type == 0) {
                // 关键修改：%2u 表示占2个字符宽度，不足时左侧补空格
                printf("[%u]: %d ", i, tsr.GetValue(i));  // 整数格式
            } else {
                // 关键修改：%2u 实现索引对齐
                printf("[%u]: %f ", i, tsr.GetValue(i));  // 浮点数格式
            }

            // 每行满16个元素后换行（最后一行不足16个也会在结束时换行）
            if ((i + 1) % elementsPerRow == 0) {
                printf("");
            }
        }
        // 如果总元素数不是16的倍数，最后一行末尾补充换行
        if (count % elementsPerRow != 0) {
            printf("\n");
        }
    }
}
template <typename DT>
class MaskedCausalConv1dBackwardKernel {
public:
    static constexpr int kW = 3;
    static constexpr int kBufferNum = 2;
    static constexpr int kAlignBytes = 32;

    // Declarations
    __aicore__ inline MaskedCausalConv1dBackwardKernel();

    __aicore__ inline void Init(GM_ADDR grad_output,
                                GM_ADDR input,
                                GM_ADDR weight,
                                GM_ADDR mask,
                                GM_ADDR grad_input,
                                GM_ADDR grad_weight,
                                const MaskedCausalConv1dBackwardTilingDataV35 *td,
                                TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInGradOutput(int64_t bTile, int64_t sTile, int64_t hTile,
                                            int64_t sEff, int64_t hLenThis);
    __aicore__ inline void CopyInInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                       int64_t sLen, int64_t hLenThis);
    __aicore__ inline void CopyInWeight(int64_t hTile, int64_t hLenThis);
    __aicore__ inline void CopyInMask(int64_t bTile, int64_t sTile, int64_t sLen);
    __aicore__ inline void ComputeGradOutputMask(int64_t sLen, int64_t hLenThis);
    __aicore__ inline void Compute(int64_t bTile, int64_t sTile, int64_t hTile,
                                   int64_t sEff, int64_t sLen, int64_t hLenThis,
                                   LocalTensor<DT> &wLocal);
    __aicore__ inline void CopyOutGradInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                            int64_t sLen, int64_t hLenThis);
    __aicore__ inline void CopyOutGradWeight(int64_t hTile, int64_t hLenThis);

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    __aicore__ inline void printTiling(){
        printf("kernel tiling...");
        printf("hMainCoreCnt_=%ld", hMainCoreCnt_);
        printf("hTailCoreCnt_=%ld", hTailCoreCnt_);
        printf("hMainSize_=%ld", hMainSize_);
        printf("hTailSize_=%ld", hTailSize_);
        printf("hLoopCnt_=%ld", hLoopCnt_);
        printf("bLoopCnt_=%ld", bLoopCnt_);
        printf("sLoopCnt_=%ld", sLoopCnt_);
        printf("ubMainFactorH_=%ld", ubMainFactorH_);
        printf("ubTailFactorH_=%ld", ubTailFactorH_);
        printf("ubMainFactorB_=%ld", ubMainFactorB_);
        printf("ubTailFactorB_=%ld", ubTailFactorB_);
        printf("ubMainFactorS_=%ld", ubMainFactorS_);
        printf("ubTailFactorS_=%ld", ubTailFactorS_);
        printf("tailHLoopCnt_=%ld", tailHLoopCnt_);
        printf("tailBLoopCnt_=%ld", tailBLoopCnt_);
        printf("tailSLoopCnt_=%ld", tailSLoopCnt_);
        printf("tailCoreUbMainFactorH_=%ld", tailCoreUbMainFactorH_);
        printf("tailCoreUbTailFactorH_=%ld", tailCoreUbTailFactorH_);
        printf("tailCoreUbMainFactorB_=%ld", tailCoreUbMainFactorB_);
        printf("tailCoreUbTailFactorB_=%ld", tailCoreUbTailFactorB_);
        printf("tailCoreUbMainFactorS_=%ld", tailCoreUbMainFactorS_);
        printf("tailCoreUbTailFactorS_=%ld", tailCoreUbTailFactorS_);
        printf("hasMask_=%ld", hasMask_);
        printf("S_=%ld", S_);
        printf("B_=%ld", B_);
        printf("H_=%ld", H_);
        printf("W_=%ld", W_);
    }

    __aicore__ inline void printBuffer(){
        uint64_t gradOutQSize = kBufferNum * (ubMainFactorH_ * ubMainFactorB_ * (ubMainFactorS_ + 2) * sizeof(DT));
        uint64_t inputQSize = kBufferNum * (ubMainFactorH_ * ubMainFactorB_ * ubMainFactorS_ * sizeof(DT));
        uint64_t weightQSize = kBufferNum * (ubMainFactorH_ * kW * sizeof(DT));
        uint64_t gradInQSize = kBufferNum * (ubMainFactorH_ * ubMainFactorB_ * ubMainFactorS_ * sizeof(DT));
        uint64_t gradWeightQSize = kBufferNum * (ubMainFactorH_ * kW * sizeof(DT));
        uint64_t tmpBufSize = kW * ubMainFactorH_ * B_ * sizeof(float);
        uint64_t maskQSize = 0; 
        if (hasMask_) {
            uint32_t maskBytes = static_cast<uint32_t>(ubMainFactorB_ * (ubMainFactorS_ + 2));
            uint32_t maskBufSize = (maskBytes + (kAlignBytes - 1)) / kAlignBytes * kAlignBytes;
            maskQSize = kBufferNum * maskBufSize;
        }
        printf("gradOutQSize=%ld", gradOutQSize);
        printf("inputQSize=%ld", inputQSize);
        printf("weightQSize=%ld", weightQSize);
        printf("gradInQSize=%ld", gradInQSize);
        printf("gradWeightQSize=%ld", gradWeightQSize);
        printf("tmpBufSize=%ld", tmpBufSize);
        printf("maskQSize=%ld", maskQSize);
        printf("allSize=%ld", gradOutQSize + inputQSize + weightQSize + gradInQSize + gradWeightQSize + tmpBufSize + maskQSize);
        // ubSize is 253952
        // availableUbSize is 245760
    }

private:

    TPipe *pipe_;
    const MaskedCausalConv1dBackwardTilingDataV35 *td_;
    TQue<QuePosition::VECIN, kBufferNum> gradOutQ_;
    TQue<QuePosition::VECIN, kBufferNum> inputQ_;
    TQue<QuePosition::VECIN, kBufferNum> weightQ_;
    TQue<QuePosition::VECIN, kBufferNum> maskQ_;
    TQue<QuePosition::VECOUT, kBufferNum> gradInQ_;
    TQue<QuePosition::VECOUT, kBufferNum> gradWeightQ_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    GlobalTensor<DT> gradOutGm_;
    GlobalTensor<DT> inputGm_;
    GlobalTensor<DT> weightGm_;
    GlobalTensor<DT> gradInGm_;
    GlobalTensor<DT> gradWeightGm_;
    GlobalTensor<bool> maskGm_;

    // tiling values
    int64_t hMainCoreCnt_{0};           // h维度主核核数
    int64_t hTailCoreCnt_{0};           // h维度尾核核数
    int64_t hMainSize_{0};              // h维度主核处理的大小
    int64_t hTailSize_{0};              // h维度尾核处理的大小

    // 主核循环参数
    int64_t hLoopCnt_{0};               // 主核UB内h维度循环次数
    int64_t bLoopCnt_{0};               // 主核UB内b维度循环次数
    int64_t sLoopCnt_{0};               // 主核UB内s维度循环次数

    // 主核UB切块参数
    int64_t ubMainFactorH_{0};          // 主核UB内h维度主块大小
    int64_t ubTailFactorH_{0};          // 主核UB内h维度尾块大小
    int64_t ubMainFactorB_{0};          // 主核UB内b维度主块大小
    int64_t ubTailFactorB_{0};          // 主核UB内b维度尾块大小
    int64_t ubMainFactorS_{0};          // 主核UB内s维度主块大小
    int64_t ubTailFactorS_{0};          // 主核UB内s维度尾块大小

    // 尾核循环参数
    int64_t tailHLoopCnt_{0};           // 尾核UB内h维度循环次数
    int64_t tailBLoopCnt_{0};           // 尾核UB内b维度循环次数
    int64_t tailSLoopCnt_{0};           // 尾核UB内s维度循环次数

    // 尾核UB切块参数
    int64_t tailCoreUbMainFactorH_{0};  // 尾核UB内h维度主块大小
    int64_t tailCoreUbTailFactorH_{0};  // 尾核UB内h维度尾块大小
    int64_t tailCoreUbMainFactorB_{0};  // 尾核UB内b维度主块大小
    int64_t tailCoreUbTailFactorB_{0};  // 尾核UB内b维度尾块大小
    int64_t tailCoreUbMainFactorS_{0};  // 尾核UB内s维度主块大小
    int64_t tailCoreUbTailFactorS_{0};  // 尾核UB内s维度尾块大小

    // 全局参数
    int64_t hasMask_{0};                // 1，有mask；0，无mask
    int64_t S_{0};                      // S维度大小
    int64_t B_{0};                      // B维度大小
    int64_t H_{0};                      // H维度大小
    int64_t W_{0};                      // W维度大小

    int64_t hStart_{0}, hLen_{0};

    // whether this block is a main core (true) or tail core (false)
    bool isMainCore_{true};

    // per-core selected loop counts and UB factors
    int64_t hLoopCntCur_{0};
    int64_t bLoopCntCur_{0};
    int64_t sLoopCntCur_{0};

    int64_t ubMainHCur_{0};
    int64_t ubTailHCur_{0};
    int64_t ubMainBCur_{0};
    int64_t ubTailBCur_{0};
    int64_t ubMainSCur_{0};
    int64_t ubTailSCur_{0};

    // fp32 accumulators for grad_weight
    LocalTensor<float> gwAccF32_[kW];
};


template <typename DT>
__aicore__ inline MaskedCausalConv1dBackwardKernel<DT>::MaskedCausalConv1dBackwardKernel() {}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::Init(GM_ADDR grad_output,
                                                           GM_ADDR input,
                                                           GM_ADDR weight,
                                                           GM_ADDR mask,
                                                           GM_ADDR grad_input,
                                                           GM_ADDR grad_weight,
                                                           const MaskedCausalConv1dBackwardTilingDataV35 *td,
                                                           TPipe *pipe)
{
    pipe_ = pipe;
    td_ = td;

    // cache tiling fields
    hMainCoreCnt_ = td->hMainCoreCnt;           // h维度主核核数
    hTailCoreCnt_ = td->hTailCoreCnt;           // h维度尾核核数
    hMainSize_ = td->hMainSize;              // h维度主核处理的大小
    hTailSize_ = td->hTailSize;              // h维度尾核处理的大小
    hLoopCnt_ = td->hLoopCnt;               // 主核UB内h维度循环次数
    bLoopCnt_ = td->bLoopCnt;               // 主核UB内b维度循环次数
    sLoopCnt_ = td->sLoopCnt;               // 主核UB内s维度循环次数
    ubMainFactorH_ = td->ubMainFactorH;          // 主核UB内h维度主块大小
    ubTailFactorH_ = td->ubTailFactorH;          // 主核UB内h维度尾块大小
    ubMainFactorB_ = td->ubMainFactorB;          // 主核UB内b维度主块大小
    ubTailFactorB_ = td->ubTailFactorB;          // 主核UB内b维度尾块大小
    ubMainFactorS_ = td->ubMainFactorS;          // 主核UB内s维度主块大小
    ubTailFactorS_ = td->ubTailFactorS;          // 主核UB内s维度尾块大小
    tailHLoopCnt_ = td->tailHLoopCnt;           // 尾核UB内h维度循环次数
    tailBLoopCnt_ = td->tailBLoopCnt;           // 尾核UB内b维度循环次数
    tailSLoopCnt_ = td->tailSLoopCnt;           // 尾核UB内s维度循环次数
    tailCoreUbMainFactorH_ = td->tailCoreUbMainFactorH;  // 尾核UB内h维度主块大小
    tailCoreUbTailFactorH_ = td->tailCoreUbTailFactorH;  // 尾核UB内h维度尾块大小
    tailCoreUbMainFactorB_ = td->tailCoreUbMainFactorB;  // 尾核UB内b维度主块大小
    tailCoreUbTailFactorB_ = td->tailCoreUbTailFactorB;  // 尾核UB内b维度尾块大小
    tailCoreUbMainFactorS_ = td->tailCoreUbMainFactorS;  // 尾核UB内s维度主块大小
    tailCoreUbTailFactorS_ = td->tailCoreUbTailFactorS;  // 尾核UB内s维度尾块大小
    hasMask_ = td->hasMask;                // 1，有mask；0，无mask
    S_ = td->S;                      // S维度大小
    B_ = td->B;                      // B维度大小
    H_ = td->H;                      // H维度大小
    W_ = td->W;                      // W维度大小


    // compute this core's H start/len
    uint64_t blkIdx = GetBlockIdx();
    isMainCore_ = (blkIdx < static_cast<uint64_t>(hMainCoreCnt_));
    if (isMainCore_) {
        hStart_ = blkIdx * hMainSize_;
        hLen_ = hMainSize_;
    } else {
        uint64_t tailIdx = blkIdx - static_cast<uint64_t>(hMainCoreCnt_);
        hStart_ = static_cast<uint64_t>(hMainCoreCnt_) * hMainSize_ + tailIdx * hTailSize_;
        hLen_ = hTailSize_;
    }

    // set per-core loop counts and UB factors
    hLoopCntCur_ = isMainCore_ ? hLoopCnt_ : tailHLoopCnt_;
    bLoopCntCur_ = isMainCore_ ? bLoopCnt_ : tailBLoopCnt_;
    sLoopCntCur_ = isMainCore_ ? sLoopCnt_ : tailSLoopCnt_;

    ubMainHCur_ = isMainCore_ ? ubMainFactorH_ : tailCoreUbMainFactorH_;
    ubTailHCur_ = isMainCore_ ? ubTailFactorH_ : tailCoreUbTailFactorH_;
    ubMainBCur_ = isMainCore_ ? ubMainFactorB_ : tailCoreUbMainFactorB_;
    ubTailBCur_ = isMainCore_ ? ubTailFactorB_ : tailCoreUbTailFactorB_;
    ubMainSCur_ = isMainCore_ ? ubMainFactorS_ : tailCoreUbMainFactorS_;
    ubTailSCur_ = isMainCore_ ? ubTailFactorS_ : tailCoreUbTailFactorS_;

    // bind GM tensors (assume contiguous ND layout)
    gradOutGm_.SetGlobalBuffer((__gm__ DT *)grad_output, S_ * B_ * H_);
    inputGm_.SetGlobalBuffer((__gm__ DT *)input, S_ * B_ * H_);
    weightGm_.SetGlobalBuffer((__gm__ DT *)weight, W_ * H_);
    gradInGm_.SetGlobalBuffer((__gm__ DT *)grad_input, S_ * B_ * H_);
    gradWeightGm_.SetGlobalBuffer((__gm__ DT *)grad_weight, W_ * H_);
    if (hasMask_) {
        maskGm_.SetGlobalBuffer((__gm__ bool *)mask, B_ * S_);
    }

    // init queues and tmp buffers
    pipe_->InitBuffer(gradOutQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * ubMainBCur_ * (ubMainSCur_ + 2) * sizeof(DT)));
    pipe_->InitBuffer(inputQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * ubMainBCur_ * ubMainSCur_ * sizeof(DT)));
    pipe_->InitBuffer(weightQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * kW * sizeof(DT)));
    if (hasMask_) {
        uint32_t maskBytes = static_cast<uint32_t>(ubMainBCur_ * (ubMainSCur_ + 2));
        uint32_t maskBufSize = (maskBytes + (kAlignBytes - 1)) / kAlignBytes * kAlignBytes;
        pipe_->InitBuffer(maskQ_, kBufferNum, maskBufSize);
    }
    pipe_->InitBuffer(gradInQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * ubMainBCur_ * ubMainSCur_ * sizeof(DT)));
    pipe_->InitBuffer(gradWeightQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * kW * sizeof(DT)));

    // tmp buffer holds fp32 accumulators for grad_weight per-B: W * ubMainHCur_ * B_
    pipe_->InitBuffer(tmpBuf_, static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float)));
    uint32_t off = 0;
    for (int32_t k = 0; k < kW; ++k) {
        // gwAccF32_[k] points to slice for current W, starting at batch 0; per-batch offset will be applied in process
        gwAccF32_[k] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(ubMainHCur_ * B_), off);
        off += static_cast<uint32_t>(ubMainHCur_ * B_ * sizeof(float));
    }

    // if (blkIdx == 0) {
    //     printTiling();
    //     printBuffer();
    // }
    
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::Process()
{
    // zero fp32 accumulators for this h-tile (batch-wise contiguous slices)
    Duplicate(gwAccF32_[0], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));
    Duplicate(gwAccF32_[1], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));
    Duplicate(gwAccF32_[2], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));
    
    // Accumulate grad_weight in fp32 across all (b,s) tiles per h-tile, then cast+store once per h-tile
    for (int64_t hTile = 0; hTile < hLoopCntCur_; ++hTile) {
        int64_t hLenThis = (hTile == hLoopCntCur_ - 1 && ubTailHCur_ > 0) ? ubTailHCur_ : ubMainHCur_;

        // Load weight once per h-tile
        CopyInWeight(hTile, hLenThis);
        LocalTensor<DT> wLocal = weightQ_.DeQue<DT>();

        for (int64_t bTile = 0; bTile < B_; ++bTile) {
            for (int64_t sTile = 0; sTile < sLoopCntCur_; ++sTile) {
                int64_t sLen = (sTile == sLoopCntCur_ - 1 && ubTailSCur_ > 0) ? ubTailSCur_ : ubMainSCur_;
                int64_t sEff = (sTile == sLoopCntCur_ - 1) ? sLen : (sLen + 2);
                
                CopyInGradOutput(bTile, sTile, hTile, sEff, hLenThis);

                CopyInInput(bTile, sTile, hTile, sLen, hLenThis);

                Compute(bTile, sTile, hTile, sEff, sLen, hLenThis, wLocal);

                CopyOutGradInput(bTile, sTile, hTile, sLen, hLenThis);

            }
        }

        weightQ_.FreeTensor(wLocal);
        CopyOutGradWeight(hTile, hLenThis);
    }
}
template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::Compute(int64_t bTile, int64_t sTile, int64_t hTile,
                                                              int64_t sEff, int64_t sLen,
                                                              int64_t hLenThis, LocalTensor<DT> &wLocal)
{
    LocalTensor<DT> goLocal = gradOutQ_.DeQue<DT>();
    LocalTensor<DT> inLocal = inputQ_.DeQue<DT>();
    LocalTensor<DT> giLocal = gradInQ_.AllocTensor<DT>();

    // Compute grad_input using VF (reuse weight already loaded for this h-tile)
    // sEff accounts for 2-row overlap between adjacent s-tiles
    AggHiddenGradVF::DoGradInput<DT>(goLocal, wLocal, giLocal,
                                     static_cast<uint32_t>(sEff), static_cast<uint32_t>(sLen),
                                     static_cast<uint32_t>(hLenThis));

    // Accumulate grad_weight in fp32 using VF Acc variant
    // Create lvalue slices for accumulators to satisfy ref parameters
    LocalTensor<float> acc0Slice = gwAccF32_[0][static_cast<uint32_t>(bTile * ubMainHCur_)];
    LocalTensor<float> acc1Slice = gwAccF32_[1][static_cast<uint32_t>(bTile * ubMainHCur_)];
    LocalTensor<float> acc2Slice = gwAccF32_[2][static_cast<uint32_t>(bTile * ubMainHCur_)];
    AggHiddenGradVF::DoGradWeightAcc<DT>(
        goLocal, inLocal,
        acc0Slice, acc1Slice, acc2Slice,
        static_cast<uint32_t>(sEff),
        static_cast<uint32_t>(sLen), static_cast<uint32_t>(hLenThis));
        
    // pipe_barrier(PIPE_ALL);
    gradOutQ_.FreeTensor(goLocal);
    inputQ_.FreeTensor(inLocal);
    gradInQ_.EnQue<DT>(giLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::ComputeGradOutputMask(int64_t sLen, int64_t hLenThis)
{
    LocalTensor<bool> mLocal = maskQ_.DeQue<bool>();
    LocalTensor<DT> goLocal = gradOutQ_.DeQue<DT>();
    for (int64_t s = 0; s < sLen; ++s) {
        bool flag = mLocal.GetValue(static_cast<uint32_t>(s));
        if (!flag) {
            uint32_t offset = static_cast<uint32_t>((s) * hLenThis);
            LocalTensor<DT> row = goLocal[offset];
            Duplicate(row, static_cast<DT>(0), static_cast<uint32_t>(hLenThis));
        }
    }
    maskQ_.FreeTensor(mLocal);
    gradOutQ_.EnQue<DT>(goLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyInGradOutput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                       int64_t sEff, int64_t hLenThis)
{
    LocalTensor<DT> goLocal = gradOutQ_.AllocTensor<DT>();

    int64_t sStart = sTile * ubMainSCur_; // 12 * 315
    int64_t bStart = bTile * ubMainBCur_; // 0
    int64_t hOff = hTile * ubMainHCur_; // 0
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> dstBlock = goLocal[0];
    DataCopyExtParams inParams{static_cast<uint16_t>(sEff), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0, 0};

    DataCopyPad(dstBlock, gradOutGm_[base], inParams, padParams);

    // For last S-tile, zero the remaining 2 halo rows in buffer (batch clear)
    if (sTile == sLoopCntCur_ - 1) {
        uint32_t tailOff = static_cast<uint32_t>(sEff * hLenThis);
        LocalTensor<DT> tailRows = goLocal[tailOff];
        Duplicate(tailRows, static_cast<DT>(0), static_cast<uint32_t>(2 * hLenThis));
    }

    gradOutQ_.EnQue<DT>(goLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyInInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                  int64_t sLen, int64_t hLenThis)
{
    LocalTensor<DT> inLocal = inputQ_.AllocTensor<DT>();

    int64_t sStart = sTile * ubMainSCur_;
    int64_t bStart = bTile * ubMainBCur_;
    int64_t hOff = hTile * ubMainHCur_;
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> dstBlock = inLocal[0];
    DataCopyExtParams inParams{static_cast<uint16_t>(sLen), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0, 0};
    DataCopyPad(dstBlock, inputGm_[base], inParams, padParams);

    inputQ_.EnQue<DT>(inLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyInWeight(int64_t hTile, int64_t hLenThis)
{
    LocalTensor<DT> wLocal = weightQ_.AllocTensor<DT>();
    DataCopyExtParams inParams{static_cast<uint16_t>(kW), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)), 0, 0};
    int64_t hOff = hTile * ubMainHCur_;
    int64_t base = (hStart_ + hOff);
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
    DataCopyPad(wLocal, weightGm_[base], inParams, padParams);
    weightQ_.EnQue<DT>(wLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyInMask(int64_t bTile, int64_t sTile,
                                                                 int64_t sLen)
{
    LocalTensor<bool> mLocal = maskQ_.AllocTensor<bool>();
    // bool is 1B on Ascend: mask has shape [B,S] with 1 byte per element
    DataCopyExtParams inParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sLen),
                               static_cast<uint32_t>(S_ - sLen), 0, 0};
    int64_t sStart = sTile * ubMainFactorS_;
    int64_t bStart = bTile * ubMainFactorB_;
    int64_t base = bStart * S_ + sStart;
    DataCopyPadExtParams<bool> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, maskGm_[base], inParams, padParams);
    maskQ_.EnQue<bool>(mLocal);
}



template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyOutGradInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                       int64_t sLen,
                                                                       int64_t hLenThis)
{
    LocalTensor<DT> giLocal = gradInQ_.DeQue<DT>();

    int64_t sStart = sTile * ubMainSCur_;
    int64_t bStart = bTile * ubMainBCur_;
    int64_t hOff = hTile * ubMainHCur_;

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> srcBlock = giLocal[0];
    DataCopyExtParams outParams{static_cast<uint16_t>(sLen), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               0, static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0};
    DataCopyPad(gradInGm_[base], srcBlock, outParams);

    // pipe_barrier(PIPE_ALL);
    gradInQ_.FreeTensor(giLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyOutGradWeight(int64_t hTile, int64_t hLenThis)
{
    // Reduce across B in fp32, cast to DT, then write to GM: layout [W,H]
    int64_t hOff = hTile * ubMainHCur_;
    int64_t baseH = hStart_ + hOff;

    // Allocate UB buffer to hold casted [W,H] slices for this h-tile
    LocalTensor<DT> gwCast = gradWeightQ_.AllocTensor<DT>();
    LocalTensor<DT> out0 = gwCast;                                      // k=0 slice at offset 0
    LocalTensor<DT> out1 = gwCast[static_cast<uint32_t>(ubMainHCur_)];  // k=1 slice
    LocalTensor<DT> out2 = gwCast[static_cast<uint32_t>(2 * ubMainHCur_)]; // k=2 slice

    // Perform B-reduction on fp32 accumulators and cast to DT into out0/out1/out2
    AggHiddenGradVF::DoReduceBToHAndCast<DT>(
        gwAccF32_[0], gwAccF32_[1], gwAccF32_[2],
        static_cast<uint32_t>(B_), static_cast<uint32_t>(hLenThis),
        out0, out1, out2);

    // Store each W slice to GM
    for (int32_t k = 0; k < kW; ++k) {
        int64_t base = static_cast<int64_t>(k) * H_ + baseH;
        DataCopyExtParams outParams{static_cast<uint16_t>(1), static_cast<uint32_t>(hLenThis * sizeof(DT)), 0, 0, 0};
        LocalTensor<DT> gwSlice = gwCast[static_cast<uint32_t>(k * ubMainHCur_)];
        DataCopyPad(gradWeightGm_[base], gwSlice, outParams);
    }
    gradWeightQ_.FreeTensor(gwCast);
}

} // namespace MaskedCausalConv1dBackwardKernelNS

#endif // MASK_EDCAUSAL_CONV1D_BACKWARD_H
