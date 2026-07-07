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
 * \file lightning_indexer_kl_loss.cpp
 * \brief AscendC kernel entry for LightningIndexerKL_Loss
 */

#include "lightning_indexer_kl_loss.h"
#include "lightning_indexer_kl_loss_tiling_key.h"

using namespace NsLightningIndexerKlLoss;

/*
 * Kernel entry: 4 TilingKey modes via two boolean template args:
 *   DeterType=false, DataType=SchModeDataTypeEnum::FLOAT  → non-deterministic, fp32
 *   DeterType=true,  DataType=SchModeDataTypeEnum::FLOAT  → deterministic, fp32
 *   DeterType=false, DataType=SchModeDataTypeEnum::HALF   → non-deterministic, half (fp16/bf16)
 *   DeterType=true,  DataType=SchModeDataTypeEnum::HALF   → deterministic, half (fp16/bf16)
 */
template <bool isDeterministic, bool isHalf>
__global__ __aicore__ void lightning_indexer_kl_loss(GM_ADDR targetScore, GM_ADDR indexProbs,
    GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LightningIndexerKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(LightningIndexerKlLossTilingData, tilingData, tiling);

    if constexpr (isHalf) {
        LightningIndexerKlLoss<Half> op;
        op.Init(targetScore, indexProbs, loss, workspace, &tilingData);
        op.InitHalfBufs();
        if constexpr (isDeterministic) {
            op.InitWorkspaceDet(workspace);
        }
        op.Process(true);
        if constexpr (isDeterministic) {
            op.WriteBackDet();
        } else {
            op.WriteBackNonDet();
        }
    } else {
        LightningIndexerKlLoss<float> op;
        op.Init(targetScore, indexProbs, loss, workspace, &tilingData);
        if constexpr (isDeterministic) {
            op.InitWorkspaceDet(workspace);
        }
        op.Process(false);
        if constexpr (isDeterministic) {
            op.WriteBackDet();
        } else {
            op.WriteBackNonDet();
        }
    }
}