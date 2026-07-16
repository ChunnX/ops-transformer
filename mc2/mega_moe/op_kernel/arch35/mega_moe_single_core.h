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
// function in the subdirectories below.
#include "single_core_context.h"
#include "aiv_compute/quant_process.h"
#include "aiv_compute/mega_moe_combine_send_compute.h"
#include "aiv_compute/mega_moe_combine_reduce.h"
#include "aiv_comm/dispatch_prepare.h"
#include "aiv_comm/dispatch.h"
#include "aiv_comm/mega_moe_combine_send_comm.h"
#include "aiv_comm/mega_moe_combine_sync.h"
#include "block/mega_moe_gmm2_block.h"
#include "epilogue/dispatch_gmm1_swiglu.h"
#include "epilogue/mega_moe_combine_send_epilogue.h"
#include "epilogue/mega_moe_gmm2_epilogue.h"

#endif // MEGA_MOE_SINGLE_CORE_H
