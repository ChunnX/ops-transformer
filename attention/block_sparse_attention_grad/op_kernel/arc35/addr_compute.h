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
class AddrComputeModule {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    int32_t batch_num;
    int32_t q_seq_len;
    int32_t kv_seq_len;
    int32_t q_group;
    int32_t q_head_num;
    int32_t kv_head_num;
    int32_t head_dim;

public:
    __aicore__ inline void Init(const TILING_CLASS *tilingData)
    {
        this->batch_num = tilingData->batchNum;
        this->q_seq_len = tilingData->qSeqLen;
        this->kv_seq_len = tilingData->kvSeqLen;
        this->q_group = tilingData->qGroup;
        this->q_head_num = tilingData->qHeadNum;
        this->kv_head_num = tilingData->kvHeadNum;
        this->head_dim = tilingData->headDim;

        printf("batch_num: %d, q_seq_len: %d, kv_seq_len: %d, q_group: %d, q_head_num: %d, kv_head_num: %d, head_dim: "
               "%d\n",
               batch_num, q_seq_len, kv_seq_len, q_group, q_head_num, kv_head_num, head_dim);
    }

    __aicore__ inline void GetRunTimeInfo(RunTimeInfo *runTimeInfo)
    {
    }
};
} // namespace BSA_ARC35
