/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SINGLE_CORE_H
#define MEGA_MOE_SINGLE_CORE_H

// Compatibility aggregation header. Implementations are grouped by execution
// role in the directories and composite scheduling headers below.
#include "single_core_context.h"
#include "aiv_compute/quant_process.h"
#include "aiv_compute/shared_expert_prepare.h"
#include "aiv_comm/send_mask.h"
#include "aiv_comm/token_dispatch.h"
#include "workspace_reset.h"
#include "dispatch_gmm1_swiglu.h"
#include "shared_expert_gmm1_swiglu.h"

#endif // MEGA_MOE_SINGLE_CORE_H
