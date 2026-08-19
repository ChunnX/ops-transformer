/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH92_QUANT_COMPAT_H
#define MEGA_MOE_ARCH92_QUANT_COMPAT_H

#if __has_include("../../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../../moe_distribute_dispatch_v2/quantize_functions.h"
#elif __has_include("../../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h")
#include "../../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#else
#include <quantize_functions.h>
#endif

// The A6 yellow-zone quant helper uses lowercase `quant`; existing MegaMoe
// code uses `Quant`. Keep the compatibility alias visible before stage headers.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
namespace Quant = quant;
#endif

#endif // MEGA_MOE_ARCH92_QUANT_COMPAT_H

