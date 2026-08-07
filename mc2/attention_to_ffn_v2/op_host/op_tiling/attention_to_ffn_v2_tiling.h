/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTENTION_TO_FFn_V2_HOST_TILING_H
#define ATTENTION_TO_FFn_V2_HOST_TILING_H

#include "../../../attention_to_ffn/op_host/op_tiling/attention_to_ffn_tiling_base.h"

namespace MC2Tiling {
ge::graphStatus AttentionToFFnV2TilingFunc(gert::TilingContext *context);
} // namespace MC2Tiling

#endif // ATTENTION_TO_FFn_V2_HOST_TILING_H
