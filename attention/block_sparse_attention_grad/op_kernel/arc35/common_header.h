/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

namespace BSA_ARC35 {

struct RunTimeInfo {
    int32_t taskId{0};
    int32_t bIdx{0};        // 当前计算的batch id
    int32_t n1Idx{0};       // 当前计算的q head id
    int32_t s1Idx{0};       // 需要计算的左上角顶点s1Idx
    int32_t s2Idx{0};       // 需要计算的左上角顶点s2Idx
    int32_t s1Len{0};       // 需要计算S1方向的长度
    int32_t s2Len{0};       // 需要计算S2方向的长度
    int32_t loop_finish{0}; // 是否遍历完所有任务
};


__aicore__ inline int64_t getSeqLen(int32_t i, __gm__ uint8_t *seq_Len)
{
    int64_t actualSeqlen;
    if (i == 0) {
        actualSeqlen = ((__gm__ int64_t *)seq_Len)[0];
    } else {
        actualSeqlen = ((__gm__ int64_t *)seq_Len)[i] - ((__gm__ int64_t *)seq_Len)[i - 1];
    }
    return actualSeqlen;
}

__aicore__ inline int64_t getTotalLen(int32_t i, __gm__ uint8_t *seq_Len)
{
    int64_t actualTotalSeqlen = ((__gm__ int64_t *)seq_Len)[i];
    return actualTotalSeqlen;
}

} // namespace BSA_ARC35
