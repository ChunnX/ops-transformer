/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_HOST_TILING_H_
#define FLASH_MLA_HOST_TILING_H_
#include "exe_graph/runtime/tiling_context.h"
#include "../op_kernel/flash_mla_metadata_protocol.h"
namespace optiling {
ge::graphStatus PrepareMlaStaticTiling(gert::TilingContext *context, FlashMla::MlaStaticTiling &tiling);
ge::graphStatus PrepareMlaStaticTiling(gert::TilingContext *context, const FlashMla::MlaPlatformInfo &platform,
                                       FlashMla::MlaStaticTiling &tiling);
ge::graphStatus TilingFlashMla(gert::TilingContext *context);
} // namespace optiling
#endif
