/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_v3.h
 * \brief
 */
#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_V3
#define FUSED_INFER_ATTENTION_SCORE_TILING_V3

#include <exe_graph/runtime/tiling_context.h>
#include "../../incre_flash_attention/op_host/incre_flash_attention_tiling_context.h"

#ifdef ASCENDC_OP_TEST
#define FIA_EXTERN_C extern "C"
#else
#define FIA_EXTERN_C
#endif
namespace optiling {

FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context);
bool RouteToFia(gert::TilingContext *context, IncreFlashAttentionContext &ifaContext);

} // namespace optiling
#endif // FUSED_INFER_ATTENTION_SCORE_TILING_V3