/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_MX_QUERY_LAYOUT_H
#define QBSA_MX_QUERY_LAYOUT_H

#include "../../quant_block_sparse_attn_const.h"

namespace regbasemx {
// tokenBase is cu_seqlens_q[b] for TND, b * physical Sq for padded layouts.
// Return the token/head slot; multiply by D, Dv or scale width at the call site.
template <QBSALayout layout>
__aicore__ inline constexpr uint64_t MxQuerySlot(uint64_t tokenBase, uint32_t s, uint32_t n, uint32_t heads, uint32_t sq)
{
    if constexpr (layout == QBSALayout::BNSD) {
        return tokenBase * heads + static_cast<uint64_t>(n) * sq + s;
    } else {
        return (tokenBase + s) * heads + n;
    }
}

template <QBSALayout layout>
__aicore__ inline constexpr uint64_t MxLseSlot(uint64_t tokenBase, uint32_t s, uint32_t n, uint32_t heads, uint32_t sq)
{
    if constexpr (layout == QBSALayout::BSND || layout == QBSALayout::BNSD) {
        return tokenBase * heads + static_cast<uint64_t>(n) * sq + s;
    } else {
        return (tokenBase + s) * heads + n;
    }
}
} // namespace regbasemx
#endif
