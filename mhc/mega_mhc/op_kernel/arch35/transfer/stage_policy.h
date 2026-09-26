/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_STAGE_POLICY_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_STAGE_POLICY_H
#include "../execution_policy.h"
#include "gm_stage.h"
#include "ub_l1_stage.h"

namespace AiInfraMegaMhc::Transfer {
template <ProjectionTransport TRANSPORT> struct StagePolicy;
template <> struct StagePolicy<ProjectionTransport::GmStage> { using Type = GmStage; };
template <> struct StagePolicy<ProjectionTransport::UbToL1> { using Type = UbL1Stage; };
template <ProjectionTransport TRANSPORT> using StagePolicyT = typename StagePolicy<TRANSPORT>::Type;
} // namespace AiInfraMegaMhc::Transfer
#endif
