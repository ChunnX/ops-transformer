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
    int32_t s1Idx{0}; // 需要计算的左上角顶点s1Idx
    int32_t s2Idx{0}; // 需要计算的左上角顶点s2Idx
    int32_t s1Len{0}; // 需要计算S1方向的长度
    int32_t s2Len{0}; // 需要计算S2方向的长度
};
} // namespace BSA_ARC35
