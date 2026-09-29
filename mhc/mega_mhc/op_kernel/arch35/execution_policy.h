/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_EXECUTION_POLICY_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_EXECUTION_POLICY_H
#include <cstdint>

namespace MegaMhc {
/** Compile-time transport selection shared by host planning and the scheduling templates. */
enum class ProjectionTransport : uint32_t {
    GmStage = 0U,
    UbToL1 = 1U
};
constexpr ProjectionTransport kFourProjectionTransport = ProjectionTransport::UbToL1;
constexpr ProjectionTransport kFiveProjectionTransport = ProjectionTransport::GmStage;
} // namespace MegaMhc
#endif
