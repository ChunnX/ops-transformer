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
 * \file block_balance.h
 * \brief BlockBalance对外接口
 */

#ifndef BLOCK_BALANCE_H
#define BLOCK_BALANCE_H

#include "block_balance_def.h"
#include "block_balance_impl.h"

namespace load_balance {

class BlockBalance {
public:
    inline static uint32_t Compute(const BlockBalanceParam &param, BlockBalanceResult &result);
};

inline uint32_t BlockBalance::Compute(const BlockBalanceParam &param, BlockBalanceResult &result)
{
    BlockBalanceImpl impl {};
    auto ret = impl.SetParam(param);
    if (ret != BLOCK_BALANCE_SUCCESS) {
        return ret;
    }

    result = impl.Compute();
    return BLOCK_BALANCE_SUCCESS;
}

} // namespace load_balance

#endif
