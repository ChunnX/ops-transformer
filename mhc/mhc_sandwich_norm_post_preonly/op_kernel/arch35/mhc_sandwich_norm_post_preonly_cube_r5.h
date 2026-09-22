/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_CUBE_R5_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_CUBE_R5_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

// O13第一阶段只验证MIX任务中的AIC Matmul初始化和workspace接线。
// 结果尚未接入R5，AIV仍使用已验证的Vector归约，避免改变功能语义。
class MhcSandwichNormPostPreonlyCubeR5Probe {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

    __aicore__ inline void Init(const TCubeTiling *tiling, GM_ADDR workspace)
    {
        if ASCEND_IS_AIC {
            mm_.Init(tiling, &pipe_);
            (void)workspace;
        }
    }

private:
    TPipe pipe_;
    matmul::MatmulImpl<AType, BType, CType, CType> mm_;
};

#endif // MHC_SANDWICH_NORM_POST_PREONLY_CUBE_R5_H_
