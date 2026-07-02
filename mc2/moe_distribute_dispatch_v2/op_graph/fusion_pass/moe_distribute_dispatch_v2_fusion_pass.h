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
 * \file moe_distribute_dispatch_v2_fusion_pass.h
 * \brief MoeDistributeDispatchV2 → MoeDistributeDispatchV3 融合 pass
 */
#ifndef OPS_TRANSFORMER_MOE_DISTRIBUTE_DISPATCH_V2_FUSION_PASS_H
#define OPS_TRANSFORMER_MOE_DISTRIBUTE_DISPATCH_V2_FUSION_PASS_H
#include "version/cann_version.h"
#define GRAPH_FUSION_SUPPORT_VERSION 90000000
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "ge/fusion/pass/fusion_base_pass.h"
#include "graph/graph.h"

namespace ops {
class __attribute__((visibility("default"))) MoeDistributeDispatchV2FusionPass : public ge::fusion::FusionBasePass {
public:
    ge::graphStatus Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context) override;

private:
    ge::graphStatus Fusion(ge::Graph &graph, ge::GNode &moeDistributeDispatchNode, ge::GNode &sharedCtxNode);
    ge::graphStatus CreateFusionNode(ge::Graph &graph, ge::GNode &moeDistributeDispatchNode,
                                     int64_t &hccl_buff_size, const ge::TensorDesc &tensorDesc,
                                     ge::GNode &fusionNode);
    ge::graphStatus AddAttr(ge::GNode &moeDistributeDispatchNode, ge::GNode &fusionNode,
                            int64_t &hccl_buff_size) const;
    ge::graphStatus AddEdge(ge::Graph &graph, ge::GNode &moeNodePtr, ge::GNode &fusionNode,
                            ge::GNode &contextNodePtr) const;
    ge::graphStatus FusionNode(ge::Graph &graph, ge::GNode &moeDistributeDispatchNode,
                               ge::GNode &contextNode, int64_t &hccl_buff_size,
                               const ge::TensorDesc &tensorDesc);
};
}  // namespace ops
#endif  // CANN_VERSION_NUM
#endif  // OPS_TRANSFORMER_MOE_DISTRIBUTE_DISPATCH_V2_FUSION_PASS_H
