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
 * \file attention_pioneer_tiling_v2.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ATTENTIONPIONEER_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ATTENTIONPIONEER_V2_H_
#include "register/tilingdata_base.h"
#include "../common/fia_tiling_base.h"

namespace optiling {
ge::graphStatus TilingAttentionPioneerV2(gert::TilingContext *context);
class AttentionPioneerTilingV2 : public FiaTilingBase{
public:
    explicit AttentionPioneerTilingV2(gert::TilingContext *context): FiaTilingBase(context) {}
    ~AttentionPioneerTilingV2() override = default;

protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override {}
    bool IsCapable() override {return true;}
    ge::graphStatus DoOpTiling() override;
};

ge::graphStatus TilingAttentionPioneerV2(gert::TilingContext *context);

} // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_ATTENTIONPIONEER_V2_H_