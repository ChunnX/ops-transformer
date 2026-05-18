/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_tiling.cpp
 * \brief
 */
#include "grouped_matmul_tiling_utils.h"

#include "register/op_impl_registry.h"
#include "arch35/grouped_weight_quant_batch_matmul_tiling.h"
#include "arch35/grouped_no_quant_matmul_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "err/ops_err.h"
using namespace Ops::Transformer::OpTiling;
using namespace ge;
using namespace AscendC;
using namespace GroupedMatmul;

namespace optiling {
namespace {
bool IsGMMFloatDType(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT;
}

bool IsGMMA16DType(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

ge::graphStatus RunDefaultGMMTiling(gert::TilingContext *context)
{
    GMMDefaultTiling tiling(context);
    return tiling.RunFusionKernelTiling(context);
}

ge::graphStatus RunNonDavGMMTiling(gert::TilingContext *context, ge::DataType xDType, ge::DataType weightDtype)
{
    if (xDType == ge::DT_INT8 && weightDtype == ge::DT_INT4) {
        GMMA8W4MsdTiling msdTiling(context);
        ge::graphStatus result = msdTiling.RunFusionKernelTiling(context);
        if (result != ge::GRAPH_PARAM_INVALID) {
            return result;
        }
        GMMA8W4FakeTiling fakeTiling(context);
        return fakeTiling.RunFusionKernelTiling(context);
    }
    if (xDType == ge::DT_INT8 && weightDtype == ge::DT_INT8) {
        GMMA8W8Tiling tiling(context);
        ge::graphStatus result = tiling.RunFusionKernelTiling(context);
        return result != ge::GRAPH_PARAM_INVALID ? result : RunDefaultGMMTiling(context);
    }
    if (xDType == ge::DT_INT4 && weightDtype == ge::DT_INT4) {
        GMMA4W4Tiling tiling(context);
        ge::graphStatus result = tiling.RunFusionKernelTiling(context);
        return result != ge::GRAPH_PARAM_INVALID ? result : RunDefaultGMMTiling(context);
    }
    if (IsGMMA16DType(xDType) && weightDtype == ge::DT_INT8) {
        GMMA16W8Tiling tiling(context);
        ge::graphStatus result = tiling.RunFusionKernelTiling(context);
        return result != ge::GRAPH_PARAM_INVALID ? result : RunDefaultGMMTiling(context);
    }
    if (IsGMMA16DType(xDType) && weightDtype == ge::DT_INT4) {
        GMMA16W4Tiling tiling(context);
        ge::graphStatus result = tiling.RunFusionKernelTiling(context);
        return result != ge::GRAPH_PARAM_INVALID ? result : RunDefaultGMMTiling(context);
    }
    if (xDType == weightDtype && IsGMMFloatDType(xDType)) {
        GMMFloatTiling tiling(context);
        ge::graphStatus result = tiling.RunFusionKernelTiling(context);
        return result != ge::GRAPH_PARAM_INVALID ? result : RunDefaultGMMTiling(context);
    }
    return RunDefaultGMMTiling(context);
}
}  // namespace

ASCENDC_EXTERN_C ge::graphStatus TilingGMM(gert::TilingContext* context) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);  // check xDesc is not null
    ge::DataType xDType = xDesc->GetDataType();
    auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    ge::DataType weightDtype = w0Desc->GetDataType();
    auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    if (compileInfoPtr->npuArch == NpuArch::DAV_3510) {
        // 全量化：双8bits或双4bits(不会有A4W2)
        bool isQuant = xDType == ge::DT_FLOAT4_E2M1 || xDType == ge::DT_INT4 || xDType == ge::DT_FLOAT4_E1M2 ||
                       (ge::GetSizeByDataType(xDType) == 1 && ge::GetSizeByDataType(weightDtype) == 1);
        if (isQuant) {
            std::vector<int32_t> registerList = {0, 1};
            return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
        } else if (xDType != weightDtype) {
            GroupedWeightQuantBatchMatmulTiling groupedWeightQuantTiling;
            OP_CHECK_IF(!groupedWeightQuantTiling.SetTiling(context),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetTiling failed."),
                        return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        }
        bool isUnQuant =
            (xDType == ge::DT_FLOAT16 || xDType == ge::DT_BF16 || xDType == ge::DT_FLOAT) && (xDType == weightDtype);
        if (isUnQuant) {
            GroupedNoQuantMatmulTiling groupedNoQuantMatmulTiling;
            OP_CHECK_IF(!groupedNoQuantMatmulTiling.SetTiling(context),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetTiling failed."),
                        return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        }
    }
    return RunNonDavGMMTiling(context, xDType, weightDtype);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGMM(gert::TilingParseContext* context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context);
  fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
  auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
  compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
  compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
  compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

  OP_CHECK_IF((compileInfoPtr->npuArch != NpuArch::DAV_3510 && compileInfoPtr->aivNum == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "platform info is invalid, aicNum=%u, aivNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
             compileInfoPtr->aicNum, compileInfoPtr->aivNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
             compileInfoPtr->l0CSize, compileInfoPtr->l0ASize, compileInfoPtr->l0BSize),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF((compileInfoPtr->aicNum == 0 || compileInfoPtr->ubSize == 0 || \
             compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 || \
             compileInfoPtr->l0BSize == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "platform info is invalid, aicNum=%u, aivNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
             compileInfoPtr->aicNum, compileInfoPtr->aivNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
             compileInfoPtr->l0CSize, compileInfoPtr->l0ASize, compileInfoPtr->l0BSize),
             return ge::GRAPH_FAILED);

  OP_LOGI(context->GetNodeName(), "Parse compile info success, soc: %d",
            static_cast<int>(compileInfoPtr->socVersion));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmul)
.Tiling(TilingGMM)
.TilingParse<GMMCompileInfo>(TilingPrepareForGMM);  // regist into the framework
}  // namespace optiling
