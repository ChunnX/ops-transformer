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
 * \file matmul_all_reduce_add_x3_arch35.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_ADD_X3_ARCH35_H
#define MATMUL_ALL_REDUCE_ADD_X3_ARCH35_H

namespace AscendC {
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t ALIGN_BYTES = 32;

template <class T>
class MatmulAllReduceAddX3 {
public:
    __aicore__ inline MatmulAllReduceAddX3()
    {
    }
    __aicore__ inline void Init(GM_ADDR mmOutput, GM_ADDR add, uint64_t totalCnt, uint64_t tileCnt, TPipe *tPipe,
                                uint32_t coreNum)
    {
        pipe_ = tPipe;
        this->blockCnt_ = totalCnt / coreNum;
        uint64_t blockAddr = this->blockCnt_ * GetBlockIdx();
        if ((coreNum - 1) == GetBlockIdx()) {
            this->blockCnt_ = totalCnt - this->blockCnt_ * GetBlockIdx();
        }
        this->tileNum_ = Ceil(this->blockCnt_, tileCnt);
        this->tileCnt_ = tileCnt;

        mmOutGm_.SetGlobalBuffer((__gm__ T *)mmOutput + blockAddr, this->blockCnt_);
        addGm_.SetGlobalBuffer((__gm__ T *)add + blockAddr, this->blockCnt_);
        pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER, tileCnt * sizeof(T));
        pipe_->InitBuffer(inQueueY_, DOUBLE_BUFFER, tileCnt * sizeof(T));
        pipe_->InitBuffer(outQueueZ_, DOUBLE_BUFFER, tileCnt * sizeof(T));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#else
        if (std::is_same<T, bfloat16_t>::value) {
            pipe_->InitBuffer(tempQueOutFp32_, tileCnt * sizeof(float));
            pipe_->InitBuffer(tempQueAddFp32_, tileCnt * sizeof(float));
        }
#endif
    }
    __aicore__ inline void Process(uint64_t progress)
    {
        if (this->blockCnt_ == 0) {
            return;
        }
        uint64_t calcCnt =
            (progress == (this->tileNum_ - 1)) ? (this->blockCnt_ - progress * this->tileCnt_) : this->tileCnt_;
        DataCopyParams copyParams = {1, static_cast<uint16_t>(calcCnt * sizeof(T)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        LocalTensor<T> mmOutLocal = inQueueX_.AllocTensor<T>();
        DataCopyPad(mmOutLocal, mmOutGm_[progress * this->tileCnt_], copyParams, padParams);
        inQueueX_.EnQue(mmOutLocal);

        LocalTensor<T> addLocal = inQueueY_.AllocTensor<T>();
        DataCopyPad(addLocal, addGm_[progress * this->tileCnt_], copyParams, padParams);
        inQueueY_.EnQue(addLocal);

        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        Add(zLocal, xLocal, yLocal, calcCnt);
        PipeBarrier<PIPE_V>();
#else
        if (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> outFp32LocalTmp = tempQueOutFp32_.Get<float>();
            LocalTensor<float> addFp32LocalTmp = tempQueAddFp32_.Get<float>();
            Cast(outFp32LocalTmp, xLocal, RoundMode::CAST_NONE, calcCnt);
            Cast(addFp32LocalTmp, yLocal, RoundMode::CAST_NONE, calcCnt);
            PipeBarrier<PIPE_V>();
            Add(outFp32LocalTmp, outFp32LocalTmp, addFp32LocalTmp, calcCnt);
            PipeBarrier<PIPE_V>();
            Cast(zLocal, outFp32LocalTmp, RoundMode::CAST_RINT, calcCnt);
            PipeBarrier<PIPE_V>();
        } else if (std::is_same<T, half>::value) {
            Add(zLocal, xLocal, yLocal, calcCnt);
            PipeBarrier<PIPE_V>();
        }
#endif
        outQueueZ_.EnQue<T>(zLocal);
        inQueueX_.FreeTensor(mmOutLocal);
        inQueueY_.FreeTensor(addLocal);

        LocalTensor<T> outLocal = outQueueZ_.DeQue<T>();
        DataCopyPad(mmOutGm_[progress * this->tileCnt_], outLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

    TPipe *pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueY_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueZ_;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#else
    TBuf<TPosition::VECCALC> tempQueOutFp32_;
    TBuf<TPosition::VECCALC> tempQueAddFp32_;
#endif
    GlobalTensor<T> mmOutGm_;
    GlobalTensor<T> addGm_;
    uint64_t blockCnt_;
    uint64_t tileNum_;
    uint64_t tileCnt_;
};

template <class T>
__aicore__ inline void MatmulAllReduceAddX3Kernel(GM_ADDR mmOutput, GM_ADDR add, uint64_t totalCnt, uint64_t tileCnt,
                                                  TPipe *tPipe)
{
    uint32_t coreNum = GetBlockNum() * GetTaskRation();
    if (g_coreType == AIC || (GetBlockIdx() >= coreNum)) {
        return;
    }
    tPipe->Reset();
    MatmulAllReduceAddX3<T> op;
    op.Init(mmOutput, add, totalCnt, tileCnt, tPipe, coreNum);
    for (uint64_t i = 0; i < op.tileNum_; i++) {
        op.Process(i);
    }
}

template <class T>
class MatmulAllReduceAddBiasX3 {
public:
    __aicore__ inline MatmulAllReduceAddBiasX3() {}

    __aicore__ inline void Init(GM_ADDR mmOutput, GM_ADDR bias, GM_ADDR add, uint64_t totalElements,
                                uint64_t nValue, uint64_t tileCnt, TPipe *tPipe, uint32_t coreNum, bool hasX3,
                                uint32_t tileN)
    {
        pipe_ = tPipe;
        nValue_ = nValue;
        hasX3_ = hasX3;
        alignElements_ = static_cast<uint64_t>(ALIGN_BYTES / sizeof(T));
        alignedN_ = CeilAlign(nValue_, alignElements_);

        uint64_t totalRows = totalElements / nValue;
        blockRows_ = totalRows / coreNum;
        rowOffset_ = blockRows_ * GetBlockIdx();
        if ((coreNum - 1) == GetBlockIdx()) {
            blockRows_ = totalRows - blockRows_ * GetBlockIdx();
        }
        tileCnt_ = tileCnt;
        tileNum_ = Ceil(blockRows_, tileCnt_);

        mmOutGm_.SetGlobalBuffer((__gm__ T *)mmOutput + rowOffset_ * nValue_, blockRows_ * nValue_);
        biasGm_.SetGlobalBuffer((__gm__ T *)bias, nValue_);
        if (hasX3_) {
            addGm_.SetGlobalBuffer((__gm__ T *)add + rowOffset_ * nValue_, blockRows_ * nValue_);
        }

        if (tileN == 0U || tileN >= alignedN_) {
            nChunked_ = false;
            effTileN_ = alignedN_;
            pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER, tileCnt_ * alignedN_ * sizeof(T));
            pipe_->InitBuffer(biasBuf_, alignedN_ * sizeof(T));
            pipe_->InitBuffer(outQueueZ_, DOUBLE_BUFFER, tileCnt_ * alignedN_ * sizeof(T));
            if (hasX3_) {
                pipe_->InitBuffer(inQueueY_, DOUBLE_BUFFER, tileCnt_ * alignedN_ * sizeof(T));
            }
            LocalTensor<T> biasLocal = biasBuf_.Get<T>();
            DataCopyExtParams biasCopyParams = {1, static_cast<uint32_t>(nValue_ * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> biasPadParams = {false, 0, 0, 0};
            DataCopyPad(biasLocal, biasGm_, biasCopyParams, biasPadParams);
            PipeBarrier<PIPE_V>();
        } else {
            nChunked_ = true;
            effTileN_ = tileN;
            nChunks_ = Ceil(nValue_, static_cast<uint64_t>(tileN));
            pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER, tileCnt_ * effTileN_ * sizeof(T));
            pipe_->InitBuffer(biasBuf_, effTileN_ * sizeof(T));
            pipe_->InitBuffer(outQueueZ_, DOUBLE_BUFFER, tileCnt_ * effTileN_ * sizeof(T));
            if (hasX3_) {
                pipe_->InitBuffer(inQueueY_, DOUBLE_BUFFER, tileCnt_ * effTileN_ * sizeof(T));
            }
        }
    }

    __aicore__ inline void Process(uint64_t progress)
    {
        if (blockRows_ == 0) {
            return;
        }
        uint64_t calcRows = (progress == (tileNum_ - 1)) ? (blockRows_ - progress * tileCnt_) : tileCnt_;

        if (!nChunked_) {
            ProcessFullRow(progress, calcRows);
        } else {
            ProcessChunked(progress, calcRows);
        }
    }

private:
    __aicore__ inline void ProcessFullRow(uint64_t progress, uint64_t calcRows)
    {
        uint32_t blockLen = static_cast<uint32_t>(nValue_ * sizeof(T));
        uint32_t padBytes = (ALIGN_BYTES - blockLen % ALIGN_BYTES) % ALIGN_BYTES;
        uint8_t rightPad = static_cast<uint8_t>(padBytes / static_cast<uint32_t>(sizeof(T)));
        DataCopyExtParams rowCopyParams = {static_cast<uint16_t>(calcRows), blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {rightPad != 0, 0, rightPad, 0};

        LocalTensor<T> mmOutLocal = inQueueX_.AllocTensor<T>();
        DataCopyPad(mmOutLocal, mmOutGm_[progress * tileCnt_ * nValue_], rowCopyParams, padParams);
        inQueueX_.EnQue(mmOutLocal);

        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();
        LocalTensor<T> biasLocal = biasBuf_.Get<T>();

        for (uint64_t row = 0; row < calcRows; row++) {
            Add(zLocal[row * alignedN_], xLocal[row * alignedN_], biasLocal, nValue_);
        }
        PipeBarrier<PIPE_V>();

        if (hasX3_) {
            LocalTensor<T> addLocal = inQueueY_.AllocTensor<T>();
            DataCopyPad(addLocal, addGm_[progress * tileCnt_ * nValue_], rowCopyParams, padParams);
            inQueueY_.EnQue(addLocal);
            LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
            Add(zLocal, zLocal, yLocal, calcRows * alignedN_);
            PipeBarrier<PIPE_V>();
            inQueueY_.FreeTensor(addLocal);
        }

        outQueueZ_.EnQue<T>(zLocal);
        inQueueX_.FreeTensor(mmOutLocal);

        DataCopyExtParams outCopyParams = {static_cast<uint16_t>(calcRows), blockLen, 0, 0, 0};
        LocalTensor<T> outLocal = outQueueZ_.DeQue<T>();
        DataCopyPad<T>(mmOutGm_[progress * tileCnt_ * nValue_], outLocal, outCopyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

    __aicore__ inline void ProcessChunked(uint64_t progress, uint64_t calcRows)
    {
        for (uint64_t nc = 0; nc < nChunks_; nc++) {
            uint64_t nOffset = nc * effTileN_;
            uint64_t nCount = (nc == nChunks_ - 1) ? (nValue_ - nOffset) : effTileN_;
            uint32_t blockLen = static_cast<uint32_t>(nCount * sizeof(T));

            LocalTensor<T> biasLocal = biasBuf_.Get<T>();
            DataCopyExtParams biasCopyParams = {1, blockLen, 0, 0, 0};
            DataCopyPadExtParams<T> biasPadParams = {false, 0, 0, 0};
            DataCopyPad(biasLocal, biasGm_[nOffset], biasCopyParams, biasPadParams);
            PipeBarrier<PIPE_V>();

            uint32_t srcStrideBytes = static_cast<uint32_t>((nValue_ - nCount) * sizeof(T));
            DataCopyExtParams rowCopyParams = {static_cast<uint16_t>(calcRows), blockLen, srcStrideBytes, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

            LocalTensor<T> mmOutLocal = inQueueX_.AllocTensor<T>();
            DataCopyPad(mmOutLocal, mmOutGm_[progress * tileCnt_ * nValue_ + nOffset], rowCopyParams, padParams);
            inQueueX_.EnQue(mmOutLocal);

            LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
            LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();

            for (uint64_t row = 0; row < calcRows; row++) {
                Add(zLocal[row * effTileN_], xLocal[row * effTileN_], biasLocal, nCount);
            }
            PipeBarrier<PIPE_V>();

            if (hasX3_) {
                LocalTensor<T> addLocal = inQueueY_.AllocTensor<T>();
                DataCopyPad(addLocal, addGm_[progress * tileCnt_ * nValue_ + nOffset], rowCopyParams, padParams);
                inQueueY_.EnQue(addLocal);
                LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
                Add(zLocal, zLocal, yLocal, calcRows * effTileN_);
                PipeBarrier<PIPE_V>();
                inQueueY_.FreeTensor(addLocal);
            }

            outQueueZ_.EnQue<T>(zLocal);
            inQueueX_.FreeTensor(mmOutLocal);

            DataCopyExtParams outCopyParams = {static_cast<uint16_t>(calcRows), blockLen, 0, srcStrideBytes, 0};
            LocalTensor<T> outLocal = outQueueZ_.DeQue<T>();
            DataCopyPad<T>(mmOutGm_[progress * tileCnt_ * nValue_ + nOffset], outLocal, outCopyParams);
            outQueueZ_.FreeTensor(zLocal);
        }
    }

public:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueY_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueZ_;
    GlobalTensor<T> mmOutGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> addGm_;
    uint64_t nValue_;
    uint64_t alignElements_;
    uint64_t alignedN_;
    uint64_t blockRows_;
    uint64_t rowOffset_;
    uint64_t tileNum_;
    uint64_t tileCnt_;
    bool hasX3_;
    bool nChunked_ = false;
    uint64_t effTileN_ = 0;
    uint64_t nChunks_ = 0;
};

template <class T>
__aicore__ inline void MatmulAllReduceAddBiasX3Kernel(GM_ADDR mmOutput, GM_ADDR bias, GM_ADDR add,
                                                      uint64_t totalCnt, uint64_t nValue, uint64_t tileCnt,
                                                      TPipe *tPipe, bool hasX3, uint32_t tileN)
{
    uint32_t coreNum = GetBlockNum() * GetTaskRation();
    if (g_coreType == AIC || (GetBlockIdx() >= coreNum)) {
        return;
    }
    tPipe->Reset();
    MatmulAllReduceAddBiasX3<T> op;
    op.Init(mmOutput, bias, add, totalCnt, nValue, tileCnt, tPipe, coreNum, hasX3, tileN);
    for (uint64_t i = 0; i < op.tileNum_; i++) {
        op.Process(i);
    }
}
} // namespace AscendC
#endif // MATMUL_ALL_REDUCE_ADD_X3_H
