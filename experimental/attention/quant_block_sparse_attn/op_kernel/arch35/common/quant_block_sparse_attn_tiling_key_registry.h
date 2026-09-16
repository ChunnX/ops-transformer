/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TILING_KEY_REGISTRY_H
#define QUANT_BLOCK_SPARSE_ATTN_TILING_KEY_REGISTRY_H

#include "../qbsa_fp8/qbsa_fp8_tiling_key.h"
#include "../qbsa_mxfp8/qbsa_mxfp8_tiling_key.h"

ASCENDC_TPL_SEL(QBSA_FP8_TPL_SELECTION, QBSA_MXFP8_TPL_SELECTION, );

#endif // QUANT_BLOCK_SPARSE_ATTN_TILING_KEY_REGISTRY_H
