/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_tiling.cpp
 * \brief
 */

#include "flash_attn_tiling_reg.h"
#include "flash_attn_tiling_impl.h"
#include "../flash_attn_tiling_info_parser.h"
#include "../checkers/fa_checker.h"

#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "../../op_kernel/arch35/flash_attn_tiling_data.h"
#include "../../op_kernel/arch35/flash_attn_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus TilingFlashAttnReg(gert::TilingContext *context)
{
    FaTilingInfo faInfo;
    FaInfoParser faInfoParser(context);
    FlashAttnTilingImpl fa(context);
    if (faInfoParser.Parse(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    flash_attn::FAChecker faChecker;
    faChecker.Init(faInfo);
    // Check函数只做校验，不能修改faInfo中的信息
    // if (faChecker.Process(faInfo) != ge::GRAPH_SUCCESS) {
    //     return ge::GRAPH_FAILED;
    // }

    if (fa.DoOpTiling(&faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling