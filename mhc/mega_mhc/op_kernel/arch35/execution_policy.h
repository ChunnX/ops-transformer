/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_EXECUTION_POLICY_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_EXECUTION_POLICY_H
#include <cstdint>

namespace AiInfraMegaMhc {
/** Compile-time transport selection shared by host planning and the scheduling templates. */
enum class ProjectionTransport : uint32_t { GmStage = 0U, UbToL1 = 1U };
constexpr ProjectionTransport kFourProjectionTransport = ProjectionTransport::GmStage;
constexpr ProjectionTransport kFiveProjectionTransport = ProjectionTransport::GmStage;
} // namespace AiInfraMegaMhc
#endif
