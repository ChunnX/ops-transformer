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
 * \file ffn_to_attention_v2_tiling.h
 * \brief
 */

#ifndef FFN_TO_ATTENTION_TILING_H
#define FFN_TO_ATTENTION_TILING_H

namespace optiling {
class FFNToAttentionV2TilingFuncA5 : public FFNToAttentionTilingFunc {
public:
    ge::graphStatus FFNToAttentionTilingFunc(gert::TilingContext *context) override;
};
} // namespace optiling
#endif