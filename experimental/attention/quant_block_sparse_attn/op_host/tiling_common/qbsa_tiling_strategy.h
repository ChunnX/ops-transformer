/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_TILING_STRATEGY_H_
#define QBSA_TILING_STRATEGY_H_

#include <cstdint>

#include <exe_graph/runtime/tiling_context.h>
#include "qbsa_tiling_context.h"
#include "../../op_kernel/arch35/common/quant_block_sparse_attn_tiling_data.h"

namespace optiling {
class QbsaTilingStrategy {
public:
    explicit QbsaTilingStrategy(gert::TilingContext *context);
    virtual ~QbsaTilingStrategy() = default;

    ge::graphStatus Run();

protected:
    virtual ge::graphStatus CheckAttributes() const = 0;
    virtual ge::graphStatus CheckTensorMetadata() const = 0;
    virtual ge::graphStatus ParseShapeFacts(QbsaTilingContextParser &parser) = 0;
    virtual ge::graphStatus CheckShapeRelations() const = 0;
    virtual ge::graphStatus BuildTilingData() = 0;

    uint32_t GetAicCoreNum() const;

    gert::TilingContext *context_ = nullptr;
    QuantBlockSparseAttnTilingInfo tilingInfo_;
    QuantBlockSparseAttnTilingData tilingData_;
    uint32_t usedCoreNum_ = 0U;
    uint64_t totalTaskNum_ = 0U;
    uint64_t tilingKey_ = 0U;

private:
    ge::graphStatus PublishTilingData();
    void PrintTilingData() const;
};
} // namespace optiling

#endif // QBSA_TILING_STRATEGY_H_
