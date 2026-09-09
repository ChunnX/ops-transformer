/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>
#define __aicore__
enum pipe_t {
    PIPE_S,
    PIPE_FIX,
    PIPE_MTE2,
    PIPE_MTE3
};
namespace AscendC {
constexpr int AIC = 0;
void TestSet(uint8_t, pipe_t, uint16_t);
void TestWait(uint8_t, pipe_t, uint16_t);
template <uint8_t M, pipe_t P>
void CrossCoreSetFlag(uint16_t id)
{
    TestSet(M, P, id);
}
template <uint8_t M, pipe_t P>
void CrossCoreWaitFlag(uint16_t id)
{
    TestWait(M, P, id);
}
} // namespace AscendC
constexpr int g_coreType = AscendC::AIC;
