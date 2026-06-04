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
 * \file quant_block_sparse_attention_check.h
 * \brief QuantBlockSparseAttention parameter validation class.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTENTION_CHECK_H_
#define QUANT_BLOCK_SPARSE_ATTENTION_CHECK_H_

#include <exe_graph/runtime/tiling_context.h>
#include "quant_block_sparse_attention_info_parser.h"

namespace optiling {

class QuantBlockSparseAttentionCheck {
public:
    explicit QuantBlockSparseAttentionCheck(const QuantBlockSparseAttentionTilingInfo &tilingInfo);
    ~QuantBlockSparseAttentionCheck() = default;

    ge::graphStatus Process();

private:
    ge::graphStatus CheckDtype() const;
    ge::graphStatus CheckBlockSize() const;
    ge::graphStatus CheckShapeConsistency() const;
    ge::graphStatus CheckMaskMode() const;

    const QuantBlockSparseAttentionTilingInfo &tilingInfo_;
};

} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTENTION_CHECK_H_
