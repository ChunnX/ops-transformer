// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
#include <cstdint>

// Host 和 Bisheng 共享唯一入口声明；FP8 scale 非空时绕过输入量化。
extern "C" void mega_moe_kernel_do(uint32_t block_dim, uint32_t ub_bytes, void* stream, bool is_a8w4, uint8_t* context,
                                   uint8_t* x, uint8_t* x_scales, uint8_t* topk_ids, uint8_t* topk_weights,
                                   uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1, uint8_t* weight_scales2,
                                   uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
                                   uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums,
                                   uint8_t* workspace, uint8_t* timer, const uint8_t* tiling_host, uint64_t ffts_addr);
