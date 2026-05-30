/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_basic_block.h"

namespace QkvRmsNormRopeCacheWithKScale {

class D128FastController {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        tensors_ = tensors;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Process()
    {
        const TokenRange tokenRange = MakeCoreTokenRange(tilingData_->totalTokens, GetCubeIndex(), GetCubeGroupNum());

        D128FastBasicBlock basicBlock;
        basicBlock.Init(tensors_, tilingData_);
        uint32_t localTileId = 0U;
        for (uint32_t tokenOffset = tokenRange.begin; tokenOffset < tokenRange.end;) {
            const uint32_t tokenSize = MinU32(tilingData_->tokenTile, tokenRange.end - tokenOffset);
            if (tokenSize == 0U) {
                break;
            }
            const TileParam tile = MakeTileParam(localTileId, tokenOffset, tokenSize, tilingData_);
            basicBlock.ComputeTile(tile);
            tokenOffset += tokenSize;
            ++localTileId;
        }
        basicBlock.End();
    }

private:
    GlobalTensors tensors_;
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_
