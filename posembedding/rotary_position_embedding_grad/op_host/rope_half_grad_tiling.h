/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rope_half_grad.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_HALF_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_HALF_GRAD_H

#include "rotary_position_embedding_grad_tiling.h"

namespace optiling {

class RopeRotateHalfGradTlingClass : public RotaryPosEmbeddingGradMembaseTilingClass
{
public:
    explicit RopeRotateHalfGradTlingClass(gert::TilingContext* context)
        : RotaryPosEmbeddingGradMembaseTilingClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_HALF_GRAD_H
