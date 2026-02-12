/*!
 * \file adaptive_max_pool3d_grad_tiling.cpp
 * \brief
 */
#include "adaptive_max_pool3d_grad_tiling.h"
#include <iostream>

namespace optiling {
using namespace OpTiling;

static ge::graphStatus Tiling4AdaptiveMaxPool3DGrad(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4AdaptiveMaxPool3DGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepare4AdaptiveMaxPool3DGrad.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<Tiling4AdaptiveMaxPool3DGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    OP_CHECK_IF(
        (compileInfoPtr->totalCoreNum <= 0), OP_LOGE(context, "Failed to get corenum size"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (compileInfoPtr->maxUbSize <= 0), OP_LOGE(context, "Failed to get maxub size"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaptiveMaxPool3DGrad)
    .Tiling(Tiling4AdaptiveMaxPool3DGrad)
    .TilingParse<Tiling4AdaptiveMaxPool3DGradCompileInfo>(TilingPrepare4AdaptiveMaxPool3DGrad);
} // namespace optiling