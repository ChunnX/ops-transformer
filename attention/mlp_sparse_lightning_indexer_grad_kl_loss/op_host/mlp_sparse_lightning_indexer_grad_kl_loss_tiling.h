/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MLP_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_H
#define MLP_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
struct MlpSparseLightningIndexerGradKLLossCompileInfo {
    uint32_t aicCoreNum = 0;
};

BEGIN_TILING_DATA_DEF(MlpSparseLightningIndexerGradKLLossTilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalSize);
    TILING_DATA_FIELD_DEF(int64_t, s1Size);
    TILING_DATA_FIELD_DEF(int64_t, s2Size);
    TILING_DATA_FIELD_DEF(int64_t, kSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MlpSparseLightningIndexerGradKLLoss,
                           MlpSparseLightningIndexerGradKLLossTilingData);
} // namespace optiling

#endif
