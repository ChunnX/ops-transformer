/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Arch22 MTE pipeline composition. Runtime state, low-level operations,
// stages and wave scheduling have separate ownership, as in mega_moe A5.
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H

#include "common/gmma2av_mte_context.h"
#include "common/gmma2av_mte_workspace.h"
#include "common/gmma2av_mte_copy.h"
#include "common/gmma2av_mte_sync.h"
#include "stage/gmma2av_mte_prepare.h"
#include "stage/gmma2av_mte_stage.h"
#include "stage/gmma2av_mte_combine.h"
#include "grouped_mat_mul_allto_allv_mte_wave.h"

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H
