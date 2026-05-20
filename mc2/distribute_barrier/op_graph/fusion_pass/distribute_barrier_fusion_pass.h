/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * ...
 */

#ifndef OPS_TRANSFORMER_DISTRIBUTE_BARRIER_FUSION_PASS_H
#define OPS_TRANSFORMER_DISTRIBUTE_BARRIER_FUSION_PASS_H
#include "version/cann_version.h"
#define GRAPH_FUSION_SUPPORT_VERSION 90000000
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "ge/fusion/pass/fusion_base_pass.h"

namespace ops {
class __attribute__((visibility("default"))) DistributeBarrierFusionPass
    : public ge::fusion::FusionBasePass {
public:
    ge::graphStatus Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context) override;
};
}  // namespace ops
#endif // CANN_VERSION_NUM
#endif
