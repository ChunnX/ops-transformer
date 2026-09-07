/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

// The CANN CPU kernel stub cannot model HCCL peer windows or CATLASS device
// execution. Compile the V2 host/device contract here; real device compilation
// is covered by the opkernel build.
#include "allto_allv_grouped_mat_mul_hccl_context_stub.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_aiv_mode.h"
