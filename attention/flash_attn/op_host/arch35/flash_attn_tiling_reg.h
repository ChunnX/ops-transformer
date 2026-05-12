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
 * \file flash_attn_tiling.h
 * \brief
 */

#ifndef FLASH_ATTN_TILING_REG_H_
#define FLASH_ATTN_TILING_REG_H_
#include "register/tilingdata_base.h"
#include "../fa_tiling_base.h"
#include "../fa_tiling_info.h"

namespace optiling {
ge::graphStatus TilingFlashAttnReg(gert::TilingContext *context);

} // namespace optiling
#endif // FLASH_ATTN_TILING_REG_H_