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
 * \file fullmesh_checker.h
 * \brief MOE Distribute Dispatch FullMeshV2 可选特性校验
 */

#ifndef MOE_DISPATCH_FULLMESH_CHECKER_H
#define MOE_DISPATCH_FULLMESH_CHECKER_H

#include "defs.h"

namespace fullmesh_check {

// ========== 约束规则校验 ==========

inline ge::graphStatus CheckConstraints(
    const char* nodeName, uint32_t bs, uint32_t k, uint32_t tpWorldSize) {
    // BS <= 256
    if (bs > static_cast<uint32_t>(moe_dispatch_defs::FULLMESH_BS_MAX)) {
        OPS_LOG_E(nodeName, "When comm_alg is fullmesh_v2, bs should be between [1, %u], but got %u.",
            moe_dispatch_defs::FULLMESH_BS_MAX, bs);
        return ge::GRAPH_FAILED;
    }

    // K <= 12
    if (k > static_cast<uint32_t>(moe_dispatch_defs::FULLMESH_K_MAX)) {
        OPS_LOG_E(nodeName, "When comm_alg is fullmesh_v2, topK should be between [1, %u], but got %u.",
            moe_dispatch_defs::FULLMESH_K_MAX, k);
        return ge::GRAPH_FAILED;
    }

    // tp != 2 (FullMesh 不支持 TP=2)
    if (tpWorldSize == moe_dispatch_defs::TP_WORLD_SIZE_TWO) {
        OPS_LOG_E(nodeName, "When comm_alg is fullmesh_v2, tp_world_size cannot be 2.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== FullMesh 标志位判断 ==========

inline bool IsFullMeshV2(const char* commAlg) {
    if (commAlg == nullptr) {
        return false;
    }
    // commAlg 包含 "fullmesh" 表示启用 fullmesh
    return (std::string(commAlg).find("fullmesh") != std::string::npos);
}

} // namespace fullmesh_check

#endif // MOE_DISPATCH_FULLMESH_CHECKER_H