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
 * \file ffn_to_attension_v2_tiling.h
 * \brief V2 tiling delegates to V1's FFNToAttentionTilingFunc via index mapping,
 *        then overrides the tiling key from A3 to A5.
 */

#ifndef FFN_TO_ATTENSION_V2_TILING_H
#define FFN_TO_ATTENSION_V2_TILING_H

#include "register/tilingdata_base.h"

namespace MC2Tiling {
ge::graphStatus FFNToAttentionV2TilingFunc(gert::TilingContext *context);
ge::graphStatus FFNToAttentionTilingFunc(gert::TilingContext *context);
} // namespace MC2Tiling

#endif // FFN_TO_ATTENSION_V2_TILING_H
