/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
using namespace AscendC;

namespace BSA_ARC35 {

template <typename BSA_TYPE>
class CubeOp {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    TBuf<> l1_buffer_;

public:
    __aicore__ inline CubeOp(){};
    __aicore__ inline void Init(const TILING_CLASS *tilingData, TPipe *tPipe)
    {
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        tPipe->InitBuffer(l1_buffer_, 183 * 1024);
    }


    __aicore__ inline void ComputeMM12(GlobalTensor<INPUT_TYPE> leftGm, GlobalTensor<INPUT_TYPE> rightGm,
                                       RunTimeInfo *runTimeInfo)
    {
        int32_t mLoop = CeilDiv(runTimeInfo->s1Len, 128);
        int32_t nLoop = CeilDiv(runTimeInfo->s2Len, 128);
        int32_t mTail = (runTimeInfo->s1Len % 128) == 0 ? 128 : (runTimeInfo->s1Len % 128);
        int32_t nTail = (runTimeInfo->s2Len % 128) == 0 ? 128 : (runTimeInfo->s2Len % 128);

        for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
            int32_t nProcess = (nIdx == nLoop - 1) ? nTail : 128;
            for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
                int32_t mProcess = (mIdx == mLoop - 1) ? mTail : 128;
            }
        }
    }
};

} // namespace BSA_ARC35
