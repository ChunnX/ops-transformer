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
 * \file block_attn_res_prepare_mix_tiling.h
 * \brief Mixed Cube/Vector tiling template for BlockAttnResPrepare.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_MIX_TILING_H
#define BLOCK_ATTN_RES_PREPARE_MIX_TILING_H

#include "block_attn_res_prepare_base_tiling.h"
#include "../../op_kernel/arch35/block_attn_res_prepare_tiling_data.h"

namespace optiling {

class BlockAttnResPrepareMixTiling final : public BlockAttnResPrepareBaseTiling {
public:
    explicit BlockAttnResPrepareMixTiling(gert::TilingContext *context)
        : BlockAttnResPrepareBaseTiling(context)
    {}
    ~BlockAttnResPrepareMixTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    TilingDataView GetTilingDataView() const override;

private:
    bool SelectMixTileShape();

    uint32_t baseT_ = 1;
    uint32_t baseS_ = 0;
    uint32_t baseD_ = 0;
    uint32_t baseDAlign_ = 0;
    uint32_t sTileNum_ = 0;
    uint32_t dTileNum_ = 0;
    uint32_t sAlign_ = 0;
    uint32_t nAlign_ = 0;
    uint32_t mm1NAlign_ = 0;
    uint32_t dAlign_ = 0;
    uint32_t qL1BufferNum_ = 1;
    uint32_t vL1BufferNum_ = 1;
    uint32_t vUbBufferNum_ = 1;
    uint64_t qL1Elems_ = 0;
    uint64_t vL1Elems_ = 0;
    uint64_t eL1Elems_ = 0;
    uint64_t vUbElems_ = 0;
    uint64_t dotUbElems_ = 0;
    uint64_t reduceUbElems_ = 0;
    uint64_t softmaxUbElems_ = 0;
    uint64_t workspacePerCoreElems_ = 0;

    BlockAttnResPrepareCubeTilingData tilingData_{};
};

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_MIX_TILING_H
