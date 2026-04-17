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
 * \file attention_pioneer_tiling.cpp
 * \brief
 */

#include "attention_pioneer_tiling.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "arch35/attention_pioneer_tiling_v2.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

AP_EXTERN_C ge::graphStatus DoOpTilingAttentionPioneer(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("AttentionPioneer", "Tiling context is null."),
        return ge::GRAPH_FAILED);
    return TilingAttentionPioneerV2(context);
}

extern "C" {
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingAttentionPioneer(
    gert::TilingContext *context)
{
    return DoOpTilingAttentionPioneer(context);
}
}
} // namespace optiling
