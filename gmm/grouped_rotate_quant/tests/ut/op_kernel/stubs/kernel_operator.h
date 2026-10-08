// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// Host-only interface substitutes. No device instructions or numerical backend.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#define __aicore__
#define __global__
using GM_ADDR = uint8_t *;
struct half {};
struct bfloat16_t {};
namespace AscendC {
struct fp4x2_e2m1_t {};
struct fp8_e4m3fn_t {};
struct fp8_e5m2_t {};
struct fp8_e8m0_t {};
namespace Std = std;
inline int activePipes = 0;
struct TPipe {
    TPipe() { ++activePipes; }
    ~TPipe() { --activePipes; }
};
} // namespace AscendC
inline uint64_t testTilingKey = 0;
inline int testTaskType = 0;
#define ASSERT(condition) do { if (!(condition)) throw std::runtime_error("device assertion"); } while (false)
#define REGISTER_TILING_DEFAULT(type) static_assert(sizeof(type) == 128)
#define KERNEL_TYPE_MIX_AIC_1_2 12
#define KERNEL_TASK_TYPE_DEFAULT(type) testTaskType = (type)
#define TILING_KEY_IS(key) (testTilingKey == (key))
