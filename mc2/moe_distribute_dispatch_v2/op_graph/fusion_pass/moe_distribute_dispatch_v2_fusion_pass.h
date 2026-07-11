/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_MOE_DISTRIBUTE_DISPATCH_V2_FUSION_PASS_H
#define OPS_TRANSFORMER_MOE_DISTRIBUTE_DISPATCH_V2_FUSION_PASS_H

#include "version/cann_version.h"
#define GRAPH_FUSION_SUPPORT_VERSION 85000000
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION

#include "ge/fusion/pass/decompose_pass.h"

namespace ops {

class __attribute__((visibility("default"))) MoeDistributeDispatchV2FusionPass
    : public ge::fusion::DecomposePass {
public:
    explicit MoeDistributeDispatchV2FusionPass(const std::vector<ge::AscendString> &op_types)
        : DecomposePass(op_types) {}

protected:
    bool MeetRequirements(const ge::GNode &matched_node) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode &matched_node) override;
};

}  // namespace ops
#endif  // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#endif  // OPS_TRANSFORMER_MOE_DISTRIBUTE_DISPATCH_V2_FUSION_PASS_H
