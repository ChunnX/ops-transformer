/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_CHECK_H_
#define FLASH_ATTN_GRAD_CHECK_H_

#include "exe_graph/runtime/tiling_context.h"
#include "flash_attn_grad_tiling_info.h"
#include "log/log.h"

namespace optiling {

// 输入/属性索引与 mask_mode 取值统一定义在 flash_attn_grad_tiling_info.h，
// 此处不再重复声明，避免两份常量各自漂移。

class FlashAttnGradCheck {
public:
    static ge::graphStatus CheckParams(gert::TilingContext *context);

private:
    static bool IsTensorExist(gert::TilingContext *context, size_t index);
    static ge::graphStatus CheckInputExistence(gert::TilingContext *context);
    static ge::graphStatus CheckDtypeConsistency(gert::TilingContext *context);
    static ge::graphStatus CheckAttrs(gert::TilingContext *context, int64_t &maskMode, int64_t &winLeft,
                                       int64_t &winRight, std::string &layoutQStr, std::string &layoutKvStr,
                                       std::string &layoutOutStr);
    static ge::graphStatus CheckMaskAndAttnMask(gert::TilingContext *context, int64_t maskMode,
                                                 int64_t winLeft, int64_t winRight);
    static ge::graphStatus CheckLayout(gert::TilingContext *context, const std::string &layoutQStr,
                                       const std::string &layoutKvStr, const std::string &layoutOutStr);
    static ge::graphStatus CheckShape(gert::TilingContext *context, const std::string &layoutQStr);
    static ge::graphStatus CheckDeterministic(gert::TilingContext *context);
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECK_H_
