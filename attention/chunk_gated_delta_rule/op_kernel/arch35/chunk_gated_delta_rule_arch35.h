/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file chunk_gated_delta_rule_arch35.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_ARCH35_H
#define CHUNK_GATED_DELTA_RULE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_fused_exact_arch35.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// arch35 入口绑定 TND 输入输出并启动融合调度，shape/类型约束由 host 校验。
// T=sum(seq_lens)，Nk/Nv 为 QK/V 头数，Dk/Dv 为特征维；每个状态矩阵 S 的方向为 [Dv,Dk]。
// Q/K/V/beta/O 固定为 BF16，g 固定为 FP32；仅状态类型与 g 是否存在参与模板分发。
template <typename stateType, bool hasGamma>
class CGDR {
public:
    __aicore__ inline CGDR(const CGDRInitParams &addresses, const ChunkGatedDeltaRuleTilingData &tiling)
        : context_(addresses, tiling, hasGamma)
    {}

    __aicore__ inline void Process()
    {
        SyncAll<false>();
        FusedStage123Exact<stateType, hasGamma>::Process(context_);
    }

private:
    KernelContext<stateType> context_;
};

} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_ARCH35_H
