/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// The CPU kernel-test runtime cannot model HCCL peer windows or device
// CATLASS execution. Compile the MTE ABI contract here; complete device-kernel
// instantiation is validated by the A2 and A3 operator package builds.
#include "../../../op_kernel/arch22/grouped_mat_mul_allto_allv_mte_tiling.h"

static_assert(MC2KernelTemplate::GMMA2AV_MTE_MAX_COUNT_NUM == 1024U,
              "MTE count-table ABI must cover EP128 with eight local experts");
