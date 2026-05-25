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
 * \file moe_distribute_dispatch_v2_tiling_arch22_a3.cpp
 * \brief A3 tiling implementation
 */

#include "moe_distribute_dispatch_v2_tiling_arch22.h"

#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
#include "mc2_exception_dump.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {

uint64_t MoeDistributeDispatchV2TilingFuncA3::CalTilingKey(const DispatchV2TilingCtx &tc)
{
    uint32_t templateDispatch = TILINGKEY_NO_FULLMESH;
    bool tp = false;
    uint32_t tilingKeyQuantMode = tc.quantMode;
    bool scaleMode = false;
    uint64_t tilingKey;
    uint32_t commMode = TILINGKEY_TPL_MTE;
    if (tc.isScales) {
        scaleMode = true;
    }
    if (tc.isSetFullMeshV2) {
        templateDispatch = TILINGKEY_ENABLE_FULLMESH;
    }
    if (tc.isLayered) {
        templateDispatch = TILINGKEY_ENABLE_HIERARCHY;
    }
    if (tc.tpWorldSize == 2) {
        tp = true;
    }
    tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode,
        templateDispatch, commMode, TILINGKEY_TPL_A3);
    return tilingKey;
}

bool MoeDistributeDispatchV2TilingFuncA3::CheckTensorDataType(
    const gert::TilingContext *context, const char *nodeName,
    DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);

    if (tc.quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
            return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(nodeName, "dynamicScales dataType is invalid, dataType should be float, but is %s.",
            Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str()), return false);
    }
    if (tc.isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(nodeName, "scales dataType is invalid, dataType should be float, but is %s.",
            Ops::Base::ToString(scalesDesc->GetDataType()).c_str()), return false);
    }
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    if (tc.quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be int8, but is %s.",
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    } else {
        OP_TILING_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType %s, but is %s.",
            Ops::Base::ToString(xDesc->GetDataType()).c_str(), Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
            return false);
    }

    OP_TILING_CHECK(!CheckCommomOtherInputTensorDataType(context, nodeName, tc, config),
        OP_LOGE(nodeName, "CheckCommomOtherInputTensorDataType failed."), return false);
    OP_TILING_CHECK(!CheckCommomOutputTensorDataType(context, nodeName),
        OP_LOGE(nodeName, "CheckCommomOutputTensorDataType failed."), return false);

    return true;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA3::CheckQuantModeMatchScales(
    gert::TilingContext *context, const char *nodeName,
    DispatchV2TilingCtx &tc, DispatchV2Config &config)
{
    (void)config;
    OP_TILING_CHECK(tc.quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT),
        OP_LOGE(nodeName, "cannot support static quant now."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tc.isScales && (tc.quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT))) ||
        ((!tc.isScales) && (tc.quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))),
        OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
        static_cast<int32_t>(tc.isScales), tc.quantMode), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA3::CheckCommAlgPtr(const char* commAlgPtr, const char *nodeName)
{
    OP_TILING_CHECK(((strlen(commAlgPtr) != 0) && (strcmp(commAlgPtr, "fullmesh_v1") != 0) &&
        (strcmp(commAlgPtr, "fullmesh_v2")  != 0) && (strcmp(commAlgPtr, "hierarchy") != 0)),
        OP_LOGE(nodeName,
            "Attr commAlg is invalid, current only support hierarchy, "
            "fullmesh_v1 and fullmesh_v2, but got commAlg = %s.",
            commAlgPtr), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA3::CheckQuantModePtr(const int64_t* quantModePtr,
    const char *nodeName)
{
    OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
    (*quantModePtr > static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)),
    OP_LOGE(nodeName, "quantMode is invalid, only support [0, %ld], but got quantMode=%ld.",
    static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT), *quantModePtr), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA3::MoeDistributeDispatchV2TilingFunc(gert::TilingContext* context)
{
    DispatchV2Config config;
    config.xIndex = 0U;
    config.expertIdsIndex = 1U;
    config.scalesIndex = 2U;
    config.xActiveMaskIndex = 3U;
    config.expertScalesIndex = 4U;
    config.elasticInfoIndex = 5U;
    config.performanceInfoIndex = 6U;
    config.attrGroupEpIndex = 0;
    config.attrEpWorldSizeIndex = 1;
    config.attrEpRankIdIndex = 2;
    config.attrMoeExpertNumIndex = 3;
    config.attrGroupTpIndex = 4;
    config.attrTpWorldSizeIndex = 5;
    config.attrTpRankIdIndex = 6;
    config.attrExpertSharedTypeIndex = 7;
    config.attrSharedExpertNumIndex = 8;
    config.attrSharedExpertRankNumIndex = 9;
    config.attrQuantModeIndex = 10;
    config.attrGlobalBsIndex = 11;
    config.attrExpertTokenNumsTypeIndex = 12;
    config.attrCommAlgIndex = 13;
    config.attrZeroExpertNumIndex = 14;
    config.attrCopyExpertNumIndex = 15;
    config.attrConstExpertNumIndex = 16;
    config.isMc2Context = false;
    DispatchV2TilingCtx tc;
    return MoeDistributeDispatchA3TilingFuncImplPublicV2(context, config, tc,
        [this](gert::TilingContext *ctx, const char *name, DispatchV2TilingCtx &t, DispatchV2Config &cfg) {
            return CheckQuantModeMatchScales(ctx, name, t, cfg);
        },
        [this](const DispatchV2TilingCtx &t) { return CalTilingKey(t); },
        [this](const gert::TilingContext *ctx, const char *name, DispatchV2TilingCtx &t, DispatchV2Config &cfg) {
            return CheckTensorDataType(ctx, name, t, cfg);
        },
        [this](const int64_t *ptr, const char *name) { return CheckQuantModePtr(ptr, name); },
        [this](const char *ptr, const char *name) { return CheckCommAlgPtr(ptr, name); });
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchV2TilingFuncImpl(gert::TilingContext* context)
{
    MoeDistributeDispatchV2TilingFuncA3 impl;
    return impl.MoeDistributeDispatchV2TilingFunc(context);
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2TilingFuncImpl)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2);
} // namespace optiling
