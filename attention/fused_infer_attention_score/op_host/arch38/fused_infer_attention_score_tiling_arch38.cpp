/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_arch38.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_arch38.h"
// #include "fused_infer_attention_score_tiling_check.h"
// #include "../checkers/fia_checker.h"
// #include "../fused_infer_attention_score_tiling_info_parser.h"
// #include "../../../common/op_host/arch22/fia_tiling_nonquant_mla.h"
// #include "../../../common/op_host/arch22/fia_tiling_nonquant.h"
// #include "../../../common/op_host/arch22/fia_tiling_empty_tensor.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace AscendC;
namespace optiling {

FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreArch38(gert::TilingContext *context)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
