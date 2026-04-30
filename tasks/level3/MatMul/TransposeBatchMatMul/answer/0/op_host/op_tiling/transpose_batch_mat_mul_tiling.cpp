/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the CANN Open Software License Agreement Version 2.0 (the "License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_batch_mat_mul_tiling.cpp
 * \brief TransposeBatchMatMul tiling entry (migrated)
 */
#include "transpose_batch_mat_mul_tiling.h"

#include <type_traits>

#include "matmul_v3/matmul_v3_platform_common.h"
#include "transpose_batch_mat_mul_simplifiedkey.h"
// REMOVED: #include "./arch35/transpose_batch_mat_mul_tiling_advanced.h" (non-target arch)
#include "transpose_batch_mat_mul_base_tiling.h"
#include "transpose_batch_mat_mul_einsum_tiling.h"
#include "platform/platform_infos_def.h"
// REMOVED: #include "op_cache_tiling.h" (framework-internal, not available in single-op build)

#include "common/tiling_templates_registry.h"
#include "register/op_def_registry.h"

using namespace optiling::transpose_batch_mat_mul;
using namespace optiling::matmul_v3;
using Ops::NN::Optiling::TilingRegistry;
// REMOVED: using Ops::NN::TilingPrepareForOpCache; (framework-internal)

namespace optiling {

REGISTER_TILING_TEMPLATE("TransposeBatchMatMul", TransposeBatchMatMulBaseTiling, 0);

static ge::graphStatus IsEinsumMode(gert::TilingContext* context)
{
    constexpr size_t ALLOW_DIM = 3;
    constexpr size_t ATTR_NUM = 5;
    constexpr bool EINSUM_SUPPORT_ENABLE_HF32 = false;
    constexpr int32_t EINSUM_SUPPORT_BATCHSPLIT_FACTOR = 1;
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(idx));
    idx++;
    OP_TILING_CHECK((context->GetOptionalInputShape(idx)!= nullptr ||
        context->GetOptionalInputShape(idx + 1)!= nullptr),
        OP_LOGI(context->GetNodeName(), "Einsum mode not support bias or scale"),
        return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    idx = static_cast<size_t>(0);
    auto aPermList_ = attrs->GetAttrPointer<gert::ContinuousVector>(idx);
    const int64_t* perm_x1 = reinterpret_cast<const int64_t*>(aPermList_->GetData());
    OP_TILING_CHECK((PermDecode(perm_x1, aPermList_->GetSize()) != 213L),
        OP_LOGI(context->GetNodeName(), "Einsum mode only support permA={1,0,2}"),
        return ge::GRAPH_FAILED);
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        idx++;
        OP_TILING_CHECK(
            (*(attrs->GetAttrPointer<int32_t>(ATTR_NUM - idx)) != EINSUM_SUPPORT_BATCHSPLIT_FACTOR),
            OP_LOGI(context->GetNodeName(), "Einsum mode only support batch_split_factor=1"),
            return ge::GRAPH_FAILED);
        idx++;
        OP_TILING_CHECK(
            (*(attrs->GetAttrPointer<bool>(ATTR_NUM - idx)) != EINSUM_SUPPORT_ENABLE_HF32),
            OP_LOGI(context->GetNodeName(), "Einsum mode only support ENABLE_HF32=false"),
            return ge::GRAPH_FAILED);
    }
    const gert::Shape &aShape = context->GetInputShape(0)->GetOriginShape();
    const gert::Shape &bShape = context->GetInputShape(1)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    if ((aDimNum == ALLOW_DIM) && (bDimNum == ALLOW_DIM)) {
        if ((context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) &&
            (context->GetInputDesc(1)->GetDataType() == ge::DT_FLOAT) &&
            (context->GetOutputDesc(0)->GetDataType() == ge::DT_FLOAT)) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TbmmEinsumTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr,
        CUBE_INNER_ERR_REPORT("TbmmEinsum", "context is null"), return ge::GRAPH_FAILED);
    size_t sysWorkspaceSize = static_cast<size_t>(24 * 1024 * 1024);  // 24M same as ppmatmul tiling
    size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    OP_LOGI(context->GetNodeName(), "TbmmEinsum Tiling start.");
    TransposeBatchMatMulEinsumTiling tbmmEinsumTiling(context);
    tbmmEinsumTiling.DoTiling();
    auto res = tbmmEinsumTiling.PostTiling();
    OP_LOGI(context->GetNodeName(), "TbmmEinsum Tiling end.");
    return res;
}

static ge::graphStatus TransposeBatchMatMulTilingFunc(gert::TilingContext* context) {
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    // REMOVED: IsAdvancedSocVersion branch (non-target arch, only arch35/910_95)
    if (IsEinsumMode(context) == ge::GRAPH_SUCCESS) {
        return TbmmEinsumTilingFunc(context);
    }
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

// REMOVED: TilingPrepareForTransposeBatchMatMul and .TilingParse<MatmulV3CompileInfo>(...)
// aclnn direct-call path does not trigger TilingParse; platform info is obtained from runtime context.

IMPL_OP_OPTILING(TransposeBatchMatMul)
    .Tiling(TransposeBatchMatMulTilingFunc)
    .GenSimplifiedKey(transpose_batch_matmul::GenSimplifiedKey);
} // namespace optiling
