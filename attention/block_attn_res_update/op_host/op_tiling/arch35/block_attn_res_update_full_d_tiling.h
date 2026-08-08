/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_FULL_D_TILING_H
#define BLOCK_ATTN_RES_UPDATE_FULL_D_TILING_H

#include <cstdint>
#include "block_attn_res_update_tiling_base.h"
#include "../../../op_kernel/arch35/block_attn_res_update_tiling_data.h"

namespace optiling {
namespace block_attn_res_update {

class BlockAttnResUpdateFullDTiling : public BlockAttnResUpdateTilingBase {
public:
    explicit BlockAttnResUpdateFullDTiling(gert::TilingContext *context)
        : BlockAttnResUpdateTilingBase(context)
    {}
    ~BlockAttnResUpdateFullDTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus SelectUbTiling(uint64_t maxTPerCore);
    bool TryUbTiling(uint64_t maxTPerCore);
    uint64_t CalcUbBytes(uint32_t tileT, uint32_t &statsTStride, uint8_t &phase2Mode) const;

    BlockAttnResUpdateTilingData *tilingData_ = nullptr;
    uint64_t selectedUbBytes_ = 0;
    uint32_t blockDim_ = 1;
};

} // namespace block_attn_res_update
} // namespace optiling

#endif // BLOCK_ATTN_RES_UPDATE_FULL_D_TILING_H
