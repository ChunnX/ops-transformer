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
 * \file matmul_all_reduce_add_x3.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_ADD_X3_H
#define MATMUL_ALL_REDUCE_ADD_X3_H

namespace AscendC {

constexpr uint32_t DOUBLE_BUFFER = 2;

template <class T>
class MatmulAllReduceAddX3 {
public:
    __aicore__ inline MatmulAllReduceAddX3()
    {
    }
    __aicore__ inline void Init(GM_ADDR mmOutput, GM_ADDR add, uint64_t totalCnt, uint64_t tileCnt, TPipe *tPipe,
                                uint32_t coreNum)
    {
        pipe = tPipe;
        this->blockCnt = totalCnt / coreNum;
        uint64_t blockAddr = this->blockCnt * GetBlockIdx();
        if ((coreNum - 1) == GetBlockIdx()) {
            this->blockCnt = totalCnt - this->blockCnt * GetBlockIdx();
        }
        this->tileNum = Ceil(this->blockCnt, tileCnt);
        this->tileCnt = tileCnt;

        mmOutGm.SetGlobalBuffer((__gm__ T *)mmOutput + blockAddr, this->blockCnt);
        addGm.SetGlobalBuffer((__gm__ T *)add + blockAddr, this->blockCnt);
        pipe->InitBuffer(inQueueXmm, DOUBLE_BUFFER, tileCnt * sizeof(T));
        pipe->InitBuffer(inQueueY, DOUBLE_BUFFER, tileCnt * sizeof(T));
        pipe->InitBuffer(outQueueZmm, DOUBLE_BUFFER, tileCnt * sizeof(T));
        if (std::is_same<T, bfloat16_t>::value) {
            pipe->InitBuffer(tempQueOutFp32, tileCnt * sizeof(float));
            pipe->InitBuffer(tempQueAddFp32, tileCnt * sizeof(float));
        }
    }
    __aicore__ inline void Process(uint64_t progress)
    {
        if (this->blockCnt == 0) {
            return;
        }
        uint64_t calcCnt =
            (progress == (this->tileNum - 1)) ? (this->blockCnt - progress * this->tileCnt) : this->tileCnt;
        DataCopyParams copyParams = {1, static_cast<uint16_t>(calcCnt * sizeof(T)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        LocalTensor<T> mmOutLocal_ = inQueueXmm.AllocTensor<T>();
        DataCopyPad(mmOutLocal_, mmOutGm[progress * this->tileCnt], copyParams, padParams);
        inQueueXmm.EnQue(mmOutLocal_);

        LocalTensor<T> addLocal = inQueueY.AllocTensor<T>();
        DataCopyPad(addLocal, addGm[progress * this->tileCnt], copyParams, padParams);
        inQueueY.EnQue(addLocal);

        LocalTensor<T> xLocal = inQueueXmm.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal_ = outQueueZmm.AllocTensor<T>();

        if (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> outFp32LocalTmp = tempQueOutFp32.Get<float>();
            LocalTensor<float> addFp32LocalTmp = tempQueAddFp32.Get<float>();
            Cast(outFp32LocalTmp, xLocal, RoundMode::CAST_NONE, calcCnt);
            Cast(addFp32LocalTmp, yLocal, RoundMode::CAST_NONE, calcCnt);
            PipeBarrier<PIPE_V>();
            Add(outFp32LocalTmp, outFp32LocalTmp, addFp32LocalTmp, calcCnt);
            PipeBarrier<PIPE_V>();
            Cast(zLocal_, outFp32LocalTmp, RoundMode::CAST_RINT, calcCnt);
            PipeBarrier<PIPE_V>();
        } else if (std::is_same<T, half>::value) {
            Add(zLocal_, xLocal, yLocal, calcCnt);
            PipeBarrier<PIPE_V>();
        }

        outQueueZmm.EnQue<T>(zLocal_);
        inQueueXmm.FreeTensor(mmOutLocal_);
        inQueueY.FreeTensor(addLocal);

        LocalTensor<T> outLocal = outQueueZmm.DeQue<T>();
        DataCopyPad(mmOutGm[progress * this->tileCnt], outLocal, copyParams);
        outQueueZmm.FreeTensor(zLocal_);
    }

    TPipe *pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueXmm;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueZmm;
    TBuf<TPosition::VECCALC> tempQueOutFp32;
    TBuf<TPosition::VECCALC> tempQueAddFp32;
    GlobalTensor<T> mmOutGm;
    GlobalTensor<T> addGm;
    uint64_t blockCnt;
    uint64_t tileNum;
    uint64_t tileCnt;
};

template <class T>
__aicore__ inline void Matmul_All_Reduce_Add_X3(GM_ADDR mmOutput, GM_ADDR add, uint64_t totalCnt, uint64_t tileCnt,
                                                TPipe *tPipe)
{
    uint32_t coreNum = GetBlockNum() * GetTaskRation();
    if (g_coreType == AIC || (GetBlockIdx() >= coreNum)) {
        return;
    }
    tPipe->Reset();
    MatmulAllReduceAddX3<T> op;
    op.Init(mmOutput, add, totalCnt, tileCnt, tPipe, coreNum);
    for (uint64_t i = 0; i < op.tileNum; i++) {
        op.Process(i);
    }
}
} // namespace AscendC
#endif // MATMUL_ALL_REDUCE_ADD_X3_H
