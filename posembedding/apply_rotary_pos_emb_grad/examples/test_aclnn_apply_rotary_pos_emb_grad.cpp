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
 * \file test_aclnn_apply_rotary_pos_emb_grad.cpp
 * \brief aclnn 调用示例
 */
#include <aclnn/aclnn_base.h>
#include <aclnn/acl.h>
#include <iostream>
#include <vector>

// aclnnApplyRotaryPosEmbGrad 接口
// 接口签名:
// aclnnStatus aclnnApplyRotaryPosEmbGrad(
//     const aclTensor* dyQ, const aclTensor* dyK,
//     const aclTensor* cos, const aclTensor* sin,
//     const aclTensor* xQ, const aclTensor* xK,
//     int64_t mode,
//     aclTensor* dxQ, aclTensor* dxK,
//     aclTensor* dcos, aclTensor* dsin,
//     aclrtStream stream);

int main()
{
    // ==========================================================================
    // 示例: BNSD layout, shape (B, N, S, D)
    // dy_q: (1, 32, 4096, 128)  float16
    // dy_k: (1, 8,  4096, 128)  float16  (GQA)
    // cos:  (1, 1,  4096, 128)  float16  (11SD broadcast)
    // sin:  (1, 1,  4096, 128)  float16
    // x_q:  (1, 32, 4096, 128)  float16  (optional, 需要 dcos/dsin 时传入)
    // x_k:  (1, 8,  4096, 128)  float16  (optional)
    //
    // mode: 0 (half)
    // ==========================================================================

    // 1. 创建输入 tensor（省略实际 acl 初始化代码）
    // aclTensor* dyQ = ...;
    // aclTensor* dyK = ...;
    // aclTensor* cos  = ...;
    // aclTensor* sin  = ...;
    // aclTensor* xQ   = ...;  // 可为 nullptr（不计算 dcos/dsin）
    // aclTensor* xK   = ...;  // 可为 nullptr
    //
    // 2. 创建输出 tensor
    // aclTensor* dxQ  = ...;
    // aclTensor* dxK  = ...;
    // aclTensor* dcos = ...;
    // aclTensor* dsin = ...;
    //
    // 3. 调用算子
    // aclnnStatus ret = aclnnApplyRotaryPosEmbGrad(
    //     dyQ, dyK, cos, sin, xQ, xK, 0, dxQ, dxK, dcos, dsin, stream);

    std::cout << "ApplyRotaryPosEmbGrad example — 双路 Q+K rope 反向梯度" << std::endl;
    std::cout << "输入: dy_q, dy_k, cos, sin, [x_q, x_k]" << std::endl;
    std::cout << "输出: dx_q, dx_k, [dcos, dsin]" << std::endl;
    std::cout << "仅 Ascend 950 支持, half mode 仅支持 mode=0" << std::endl;

    return 0;
}
