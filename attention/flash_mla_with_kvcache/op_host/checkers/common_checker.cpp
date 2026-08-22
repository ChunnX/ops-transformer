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
 * \file common_checker.cpp
 * \brief Common checker for layout, shape, dtype, and scalar attr parameters
 *
 * MLA contract:
 *   - Inputs: q, k_cache, block_table, cache_seqlens, cu_seqlens_q, seqused_q,
 *     attn_mask, metadata. No v / q_rope / k_rope; rope is merged into q and
 *     k_cache (last dim 576 = nope 512 + rope 64), k_cache carries BOTH key and
 *     value data (k == v, single latent KV head: kvHeadNum == n2Size == 1).
 *   - Layouts restricted to the FIA-validated set:
 *     q=TND (cu_seqlens_q required), kv=PA_NZ (block_table required),
 *     out=NTD. BSND/BNSD query, continuous KV, non-NTD out are rejected.
 *     Gate mirror: fused_infer_attention_score/op_host/arch35/
 *     fia_tiling_nonquant_mla.cpp:79-109 (IsCapableSparseLayoutCheckMla).
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "common_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;
using namespace Ops::Base;

// ============================================================================
// Layout — SinglePara (only the FIA-validated routing set)
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaLayout(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    // the fia*mla four-piece has only ever been
    // exercised with q=TND / kv=PA_NZ / out=NTD (fia_tiling_nonquant_mla.cpp:99-107).
    // BSND/BNSD query, continuous KV and any other output layout are rejected.
    const std::vector<FlashMlaWithKvcacheLayout> supportedQLayouts = {FlashMlaWithKvcacheLayout::TND};
    // kv 侧接受 "NZ" 与 "PA_NZ" 两个别名（默认 layout_kv="NZ"；parser 会把 PA_NZ 归一为 NZ，
    // 见 arch35/flash_mla_with_kvcache_tiling.cpp 的 FlashMlaWithKvcacheLayout::NZ 断言 —— 实测两者都必须放行）
    const std::vector<FlashMlaWithKvcacheLayout> supportedKvLayouts = {FlashMlaWithKvcacheLayout::NZ, FlashMlaWithKvcacheLayout::PA_NZ};
    const std::vector<FlashMlaWithKvcacheLayout> supportedOutLayouts = {FlashMlaWithKvcacheLayout::NTD};

    if (std::find(supportedQLayouts.begin(), supportedQLayouts.end(), faInfo.qLayout) ==
        supportedQLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_q",
                                              LayoutToSerialString(faInfo.qLayout).c_str(),
                                              "The value of layout_q must be TND (BSND/BNSD query is not supported)");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) ==
                    supportedKvLayouts.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    faInfo.opName, "layout_kv", LayoutToSerialString(faInfo.kvLayout).c_str(),
                    "The value of layout_kv must be PA_NZ (continuous/other paged KV is not supported)"),
                return ge::GRAPH_FAILED);

    if (std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), faInfo.outLayout) ==
        supportedOutLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_out",
                                              LayoutToSerialString(faInfo.outLayout).c_str(),
                                              "The value of layout_out must be NTD (non-NTD output is not supported)");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Attr — SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (CheckSingleParaLayout(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus CommonChecker::CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.query.desc == nullptr || faInfo.opParamInfo.query.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "query"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.kCache.desc == nullptr || faInfo.opParamInfo.kCache.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "k_cache"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.attnOut.desc == nullptr || faInfo.opParamInfo.attnOut.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "attn_out"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Dtype
// ============================================================================

ge::graphStatus CommonChecker::CheckNonQuantDataType(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.kCache.desc, K_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.kCache.desc, K_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDtypeConsistency(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    const gert::CompileTimeTensorDesc *queryDesc = faInfo.opParamInfo.query.desc;
    const gert::CompileTimeTensorDesc *kCacheDesc = faInfo.opParamInfo.kCache.desc;
    const gert::CompileTimeTensorDesc *attnOutDesc = faInfo.opParamInfo.attnOut.desc;

    ge::DataType queryDtype = queryDesc->GetDataType();

    if (kCacheDesc != nullptr) {
        if (kCacheDesc->GetDataType() != queryDtype) {
            std::string reason =
                "The dtype of k_cache must be the same as dtype(" + ToString(queryDtype) + ") of query";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "k_cache",
                                                  ToString(kCacheDesc->GetDataType()).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (attnOutDesc != nullptr) {
        if (attnOutDesc->GetDataType() != queryDtype) {
            std::string reason = "The dtype of attn_out must be the same as dtype(" +
                                 ToString(queryDtype) + ") of query";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "attn_out",
                                                  ToString(attnOutDesc->GetDataType()).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// HeadNum
// ============================================================================

ge::graphStatus CommonChecker::CheckNonQuantHeadNum(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if ((faInfo.n1Size < 0) || (faInfo.n2Size < 0)) {
        std::string shapeStr = ToString(faInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                               ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(faInfo.opName, "query and k_cache",
                                               shapeStr.c_str(), "N of query and k_cache must be greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    if (faInfo.n1Size < faInfo.n2Size) {
        std::string shapeStr = ToString(faInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                               ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(faInfo.opName, "query and k_cache",
                                               shapeStr.c_str(), "N of query must be greater than or equal to the same axis of k_cache");
        return ge::GRAPH_FAILED;
    }
    if (faInfo.n1Size % faInfo.n2Size != 0) {
        std::string shapeStr = ToString(faInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                               ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(faInfo.opName, "query and k_cache",
                                               shapeStr.c_str(), "N of query must be an integer multiple of the same axis of k_cache");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Axis
// ============================================================================

ge::graphStatus CommonChecker::CheckAxis(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (faInfo.bSize >= B_LIMIT || faInfo.bSize <= 0) {
        std::string reason = "The value of B must be within the range (0, " + std::to_string(B_LIMIT) + ")";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "axis B",
                                              std::to_string(faInfo.bSize).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if (faInfo.qLayout == FlashMlaWithKvcacheLayout::TND) {
        OP_CHECK_IF(faInfo.qTSize <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "query",
                                                          ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                                                          "T of query must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(faInfo.n1Size <= 0 || faInfo.n1Size > NUM_128,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "query",
                                                      ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                                                      "N of query must be within the range (0, 128] (FIA MLA gate)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.n2Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "k_cache",
                                                      ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape()).c_str(),
                                                      "N of k_cache must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s1Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "query",
                                                      ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                                                      "S of query must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s2Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "k_cache",
                                                      ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape()).c_str(),
                                                      "S of k_cache must be greater than 0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MLA geometry (hard constraints)
// ============================================================================

ge::graphStatus CommonChecker::CheckMlaGeometry(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    // kvHeadNum == 1: single latent KV head (kv == n2Size, FIA gate n2Size==1)
    OP_CHECK_IF(faInfo.n2Size != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "k_cache",
                    ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape()).c_str(),
                    "kvHeadNum(N of k_cache) must be 1 (MLA single latent KV head)"),
                return ge::GRAPH_FAILED);

    // qkHeadDim == 512 (nope segment of the merged 576-wide q)
    OP_CHECK_IF(faInfo.qkHeadDim != static_cast<int64_t>(DSIZE_512),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "axis D of query and k_cache",
                                                      std::to_string(faInfo.qkHeadDim).c_str(),
                                                      "qkHeadDim(nope segment) must be 512 in MLA"),
                return ge::GRAPH_FAILED);

    // vHeadDim == 512 (nope/value width inside k_cache, == head_dim_v attr)
    OP_CHECK_IF(faInfo.vHeadDim != static_cast<int64_t>(DSIZE_512),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "axis D of value",
                                                      std::to_string(faInfo.vHeadDim).c_str(),
                                                      "vHeadDim must be 512 in MLA (== head_dim_v)"),
                return ge::GRAPH_FAILED);

    // head_dim_v attr: required, == 512, and head_dim_v + 64 == 576 (rope merged into k_cache)
    const int64_t *headDimV = faInfo.opParamInfo.headDimV;
    OP_CHECK_IF(headDimV == nullptr,
                OP_LOGE(faInfo.opName, "head_dim_v is required but is null!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*headDimV != static_cast<int64_t>(DSIZE_512),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    faInfo.opName, "head_dim_v", std::to_string(*headDimV).c_str(),
                    "The value of head_dim_v must be 512 (nope/value width inside k_cache)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*headDimV + static_cast<int64_t>(MLA_ROPE_D_DIM_64) != static_cast<int64_t>(DSIZE_576),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    faInfo.opName, "head_dim_v", std::to_string(*headDimV).c_str(),
                    "head_dim_v + rope(64) must equal 576 (q/k_cache last dim = nope 512 + rope 64)"),
                return ge::GRAPH_FAILED);

    // q last dim == 576, k_cache last dim == 576 (shape-level, merged nope+rope)
    const gert::Shape &qShape = faInfo.opParamInfo.query.shape->GetStorageShape();
    int64_t qLastDim = qShape.GetDim(qShape.GetDimNum() - 1);
    OP_CHECK_IF(qLastDim != static_cast<int64_t>(DSIZE_576),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "query", ToString(qShape).c_str(),
                    "The last dim of q must be 576 (nope 512 + rope 64)"),
                return ge::GRAPH_FAILED);

    const gert::Shape &kShape = faInfo.opParamInfo.kCache.shape->GetStorageShape();
    const bool isPaNz = (faInfo.kvLayout == FlashMlaWithKvcacheLayout::NZ || faInfo.kvLayout == FlashMlaWithKvcacheLayout::PA_NZ);
    if (!isPaNz) {
        // 连续 KV 的 k_cache 最后维 == 576；PA_NZ 5-D [Bn, N, D0, Bs, 16] 由 CheckKVShapeForPageAttention
        // 的 Bn/N/Bs/D/D0 参数比较覆盖（D0=32/typeSize，shape 级验证）
        int64_t kLastDim = kShape.GetDim(kShape.GetDimNum() - 1);
        OP_CHECK_IF(kLastDim != static_cast<int64_t>(DSIZE_576),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        faInfo.opName, "k_cache", ToString(kShape).c_str(),
                        "The last dim of k_cache must be 576 (nope 512 + rope 64)"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Layout — MultiPara constraint table (single valid combo)
// ============================================================================

struct LayoutConstraintConfig {
    std::vector<FlashMlaWithKvcacheLayout> supportedKvLayouts;
    std::vector<FlashMlaWithKvcacheLayout> supportedOutLayouts;
};

static const std::map<FlashMlaWithKvcacheLayout, LayoutConstraintConfig> LAYOUT_CONSTRAINT_TABLE = {
    // kv 两别名：默认 "NZ"；parser 归一 FlashMlaWithKvcacheLayout::NZ（arch35/flash_mla_with_kvcache_tiling.cpp），PA_NZ 保留
    {FlashMlaWithKvcacheLayout::TND, {{FlashMlaWithKvcacheLayout::NZ, FlashMlaWithKvcacheLayout::PA_NZ}, {FlashMlaWithKvcacheLayout::NTD}}},
};

ge::graphStatus CommonChecker::CheckMultiParaLayout(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    auto it = LAYOUT_CONSTRAINT_TABLE.find(faInfo.qLayout);
    OP_CHECK_IF(it == LAYOUT_CONSTRAINT_TABLE.end(),
                OP_LOGE(faInfo.opName, "layout_q %s is not supported", LayoutToSerialString(faInfo.qLayout).c_str()),
                return ge::GRAPH_FAILED);

    const auto &config = it->second;
    const std::string qLayoutStr = LayoutToSerialString(faInfo.qLayout);

    OP_CHECK_IF(std::find(config.supportedKvLayouts.begin(), config.supportedKvLayouts.end(), faInfo.kvLayout) ==
                    config.supportedKvLayouts.end(),
                OP_LOGE(faInfo.opName, "When layout_q is %s, layout_kv must match constraint, but got %s",
                        qLayoutStr.c_str(), LayoutToSerialString(faInfo.kvLayout).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(config.supportedOutLayouts.begin(), config.supportedOutLayouts.end(), faInfo.outLayout) ==
                    config.supportedOutLayouts.end(),
                OP_LOGE(faInfo.opName, "When layout_q is %s, layout_out must match constraint, but got %s",
                        qLayoutStr.c_str(), LayoutToSerialString(faInfo.outLayout).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Shape Compare
// ============================================================================

void CommonChecker::SetFaShapeCompare(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    queryShapeCmp_ = std::make_shared<FlashMlaWithKvcacheTilingShapeCompare>(faInfo.opParamInfo.query.shape->GetStorageShape(),
                                                                             faInfo.qLayout, QUERY_NAME, faInfo.opName);
    keyShapeCmp_ = std::make_shared<FlashMlaWithKvcacheTilingShapeCompare>(faInfo.opParamInfo.kCache.shape->GetStorageShape(),
                                                                           faInfo.kvLayout, K_CACHE_NAME, faInfo.opName);
    attnOutShapeCmp_ = std::make_shared<FlashMlaWithKvcacheTilingShapeCompare>(faInfo.opParamInfo.attnOut.shape->GetStorageShape(),
                                                                               faInfo.outLayout, ATTN_OUT_NAME, faInfo.opName);
}

ge::graphStatus CommonChecker::CheckQueryShape(const FlashMlaWithKvcacheTilingInfo &faInfo) const
{
    FlashMlaWithKvcacheTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    // q TND carries the merged width: nope 512 + rope 64 = 576
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim + MLA_ROPE_D_DIM_64);
    shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus CommonChecker::CheckKVShapeForPageAttention(const FlashMlaWithKvcacheTilingInfo &faInfo) const
{
    ge::DataType kvDtype = faInfo.opParamInfo.kCache.desc->GetDataType();
    uint32_t kvBlockElemNum = 32 / FlashMlaWithKvcacheBaseChecker::GetTypeSize(kvDtype);

    if (faInfo.blockSize % kvBlockElemNum != 0) {
        std::string reason = "The value of block_size must be a multiple of " + std::to_string(kvBlockElemNum) +
                             " (32 / sizeof(kv_dtype))";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "block_size",
                                              std::to_string(faInfo.blockSize).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    FlashMlaWithKvcacheTilingShapeCompareParam shapeParams;
    shapeParams.Bn = static_cast<int64_t>(faInfo.totalBlockNum);
    shapeParams.N = static_cast<int64_t>(faInfo.n2Size);
    shapeParams.Bs = static_cast<int64_t>(faInfo.blockSize);
    // k_cache PA_NZ carries the merged width (nope 512 + rope 64); D0 = 32B / elem
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim + MLA_ROPE_D_DIM_64);
    shapeParams.D0 = static_cast<int64_t>(kvBlockElemNum);

    return keyShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus CommonChecker::CheckKVShape(const FlashMlaWithKvcacheTilingInfo &faInfo) const
{
    // only PA_NZ KV is supported; block_table drives the paging（kvLayout 可能为归一后的 NZ 别名）
    if (faInfo.kvLayout == FlashMlaWithKvcacheLayout::PA_NZ || faInfo.kvLayout == FlashMlaWithKvcacheLayout::NZ) {
        if (faInfo.pageAttentionFlag) {
            return CheckKVShapeForPageAttention(faInfo);
        }
        std::string reason = "block_table cannot be empty when layout_kv is PA_NZ";
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "block_table", reason.c_str());
        return ge::GRAPH_FAILED;
    }

    std::string reason = "layout_kv: " + LayoutToSerialString(faInfo.kvLayout) +
                         " is not supported (only PA_NZ KV)";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_kv",
                                          LayoutToSerialString(faInfo.kvLayout).c_str(), reason.c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus CommonChecker::CheckAttnOutShape(const FlashMlaWithKvcacheTilingInfo &faInfo) const
{
    FlashMlaWithKvcacheTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    // attn_out carries only the value (nope) width; rope is not part of the output
    shapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    if (attnOutShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckShapeConsistency(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    SetFaShapeCompare(faInfo);
    if (ge::GRAPH_SUCCESS != CheckQueryShape(faInfo) || ge::GRAPH_SUCCESS != CheckKVShape(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — combined
// ============================================================================

ge::graphStatus CommonChecker::CheckMultiPara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (CheckMultiParaLayout(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonQuantDataType(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonQuantHeadNum(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtypeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAxis(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMlaGeometry(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttnOutShape(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla_with_kvcache
} // namespace optiling
