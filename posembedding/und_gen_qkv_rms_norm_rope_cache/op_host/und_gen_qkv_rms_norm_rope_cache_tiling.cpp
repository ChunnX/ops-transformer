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
 * \file und_gen_qkv_rms_norm_rope_cache_tiling.cpp
 * \brief regbase（arch35 / Ascend 950）模板 tiling
 */

#include "und_gen_qkv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include <iostream>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {

bool UndGenQkvRmsNormRopeCacheRegbaseTiling::IsCapable()
{
    // 本算子当前仅有一套 regbase 模板，覆盖全部已校验的 shape
    return true;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheRegbaseTiling::CalBlockTiling()
{
    // TODO(tiling): 多核切分。按输出 token 维 total 切分，每个 out_t 独立、无跨核依赖，
    //               slot_mapping[out_t] 唯一保证 KV Cache 无写冲突。
    //               需产出 usedCoreNum / blockFactor / tailBlockFactor。
    tilingData_.set_usedCoreNum(0);
    tilingData_.set_blockFactor(0);
    tilingData_.set_tailBlockFactor(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheRegbaseTiling::CalUbTiling()
{
    // TODO(tiling): UB 切分。单 token 需驻留 Q/K/V(float) + 权重 + 3 轴 cos_sin + 中间 buffer，
    //               依据 ubSize_ 反推每次可处理的 token 数 ubFactor。
    tilingData_.set_ubFactor(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheRegbaseTiling::DoOpTiling()
{
    tilingData_.set_totalTokens(totalTokens_);
    tilingData_.set_undLen(undLen_);
    tilingData_.set_genLen(genLen_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_numHeadQ(numHeadQ_);
    tilingData_.set_numHeadK(numHeadK_);
    tilingData_.set_numHeadV(numHeadV_);
    tilingData_.set_headDim(headDim_);
    tilingData_.set_maxPos(maxPos_);
    tilingData_.set_blockNum(blockNum_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_hasGen(static_cast<int64_t>(hasGen_));
    tilingData_.set_hasCatIndices(static_cast<int64_t>(hasCatIndices_));
    tilingData_.set_mropeSectionT(mropeSection_[DIM_ZERO]);
    tilingData_.set_mropeSectionH(mropeSection_[DIM_ONE]);
    tilingData_.set_mropeSectionW(mropeSection_[DIM_TWO]);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_reciprocal(reciprocal_);

    OP_CHECK_IF(CalBlockTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalBlockTiling failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalUbTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalUbTiling failed."),
                return ge::GRAPH_FAILED);

    // 当前只有一套模板，tilingKey 固定为 0；后续按 hasGen/hasCatIndices 等分支扩展
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
}

void UndGenQkvRmsNormRopeCacheRegbaseTiling::PrintTilingData()
{
    std::cout << "========== UndGenQkvRmsNormRopeCache TilingData ==========" << std::endl;
    std::cout << "totalTokens:     " << tilingData_.get_totalTokens() << std::endl;
    std::cout << "undLen:          " << tilingData_.get_undLen() << std::endl;
    std::cout << "genLen:          " << tilingData_.get_genLen() << std::endl;
    std::cout << "numHead:         " << tilingData_.get_numHead() << std::endl;
    std::cout << "numHeadQ:        " << tilingData_.get_numHeadQ() << std::endl;
    std::cout << "numHeadK:        " << tilingData_.get_numHeadK() << std::endl;
    std::cout << "numHeadV:        " << tilingData_.get_numHeadV() << std::endl;
    std::cout << "headDim:         " << tilingData_.get_headDim() << std::endl;
    std::cout << "maxPos:          " << tilingData_.get_maxPos() << std::endl;
    std::cout << "blockNum:        " << tilingData_.get_blockNum() << std::endl;
    std::cout << "blockSize:       " << tilingData_.get_blockSize() << std::endl;
    std::cout << "hasGen:          " << tilingData_.get_hasGen() << std::endl;
    std::cout << "hasCatIndices:   " << tilingData_.get_hasCatIndices() << std::endl;
    std::cout << "mropeSection:    [" << tilingData_.get_mropeSectionT() << ", " << tilingData_.get_mropeSectionH()
              << ", " << tilingData_.get_mropeSectionW() << "]" << std::endl;
    std::cout << "epsilon:         " << tilingData_.get_epsilon() << std::endl;
    std::cout << "reciprocal:      " << tilingData_.get_reciprocal() << std::endl;
    std::cout << "usedCoreNum:     " << tilingData_.get_usedCoreNum() << std::endl;
    std::cout << "blockFactor:     " << tilingData_.get_blockFactor() << std::endl;
    std::cout << "tailBlockFactor: " << tilingData_.get_tailBlockFactor() << std::endl;
    std::cout << "ubFactor:        " << tilingData_.get_ubFactor() << std::endl;
    std::cout << "=========================================================" << std::endl;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheRegbaseTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    // TODO(tiling): usedCoreNum 由 CalBlockTiling 产出后，这里直接使用其值
    int64_t blockDim = tilingData_.get_usedCoreNum() > 0 ? tilingData_.get_usedCoreNum() : 1;
    context_->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    // PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(UndGenQkvRmsNormRopeCache, UndGenQkvRmsNormRopeCacheRegbaseTiling, 1000);
} // namespace optiling
