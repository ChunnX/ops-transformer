/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_TILING_H
#define BLOCK_ATTN_RES_PREPARE_TILING_H

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "../op_kernel/arch35/block_attn_res_prepare_tiling_data.h"

namespace optiling {

struct BlockAttnResPrepareCompileInfo {
    uint64_t aicCoreNum = 0;
    uint64_t aivCoreNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t l0CSize = 0;
    uint64_t systemWorkspaceSize = 0;
};

ge::graphStatus TilingBlockAttnResPrepare(gert::TilingContext *context);
ge::graphStatus TilingPrepareBlockAttnResPrepare(gert::TilingParseContext *context);

class BlockAttnResPrepareTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockAttnResPrepareTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~BlockAttnResPrepareTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

protected:
    ge::graphStatus CheckInputShapes();
    ge::graphStatus CheckInputDtypes() const;

    uint64_t totalT_ = 0;
    uint64_t totalN_ = 0;
    uint64_t totalS_ = 0;
    uint64_t totalD_ = 0;
    float epsilon_ = 1e-6F;

    uint64_t aicCoreNum_ = 0;
    uint64_t aivCoreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t l0ASize_ = 0;
    uint64_t l0BSize_ = 0;
    uint64_t l0CSize_ = 0;
    uint64_t systemWorkspaceSize_ = 0;

private:
    bool SelectVectorBaseD(bool hasMultipleWorkRounds);

    uint32_t usedCoreNum_ = 0;
    uint32_t baseD_ = 0;
    uint32_t qBufferNum_ = 1;
    uint32_t vBufferNum_ = 1;
    uint32_t oBufferNum_ = 1;
    uint32_t vCacheRows_ = 0;

    BlockAttnResPrepareTilingData tilingData_{};
};

class BlockAttnResPrepareCubeTiling : public BlockAttnResPrepareTiling {
public:
    explicit BlockAttnResPrepareCubeTiling(gert::TilingContext *context)
        : BlockAttnResPrepareTiling(context)
    {}
    ~BlockAttnResPrepareCubeTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    bool SelectCubeBaseD();
    void FillCubeTiling(TCubeTiling &cubeTiling, uint32_t m, uint32_t n, uint32_t k, uint32_t baseM,
                        uint32_t baseN, uint32_t baseK) const;

    uint32_t usedCoreNum_ = 0;
    uint32_t baseT_ = 1;
    uint32_t baseS_ = 0;
    uint32_t baseD_ = 0;
    uint32_t baseDAlign_ = 0;
    uint32_t sTileNum_ = 0;
    uint32_t dTileNum_ = 0;
    uint32_t sAlign_ = 0;
    uint32_t nAlign_ = 0;
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
    uint64_t workspaceSize_ = 0;

    BlockAttnResPrepareCubeTilingData tilingData_{};
};

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_TILING_H
