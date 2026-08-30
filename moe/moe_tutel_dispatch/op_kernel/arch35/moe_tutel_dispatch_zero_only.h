/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_dispatch_zero_only.h
 * \brief Non-empty output with no routes: initialize y to positive zero only.
 */

#ifndef MOE_TUTEL_DISPATCH_ZERO_ONLY_H_
#define MOE_TUTEL_DISPATCH_ZERO_ONLY_H_

#include "moe_tutel_dispatch_base.h"

namespace MoeTutelDispatchNs {

template <typename TByte>
class MoeTutelDispatchZeroOnly : public MoeTutelDispatchBase<TByte> {
    using Base = MoeTutelDispatchBase<TByte>;

public:
    __aicore__ inline MoeTutelDispatchZeroOnly(TPipe *pipe, const MoeTutelDispatchTilingData *__restrict tiling)
        : Base(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR y)
    {
        Base::InitCommon(x, gates, indices, locations, y);
    }

    __aicore__ inline void Process()
    {
        Base::InitOutputZero();
    }
};
} // namespace MoeTutelDispatchNs

#endif // MOE_TUTEL_DISPATCH_ZERO_ONLY_H_
