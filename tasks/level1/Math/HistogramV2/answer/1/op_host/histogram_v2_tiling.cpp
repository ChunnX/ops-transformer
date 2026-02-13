/*!
 * \file histogram_v2_tiling.cpp
 * \brief
 */
#include "histogram_v2_tiling.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "common/log.h"
#include "common/tiling_templates_registry.h"

namespace optiling {
using namespace OpTiling;

ge::graphStatus TilingHistogramV2(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForHistogramV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For HistogramV2 start.");
    auto compileInfo = context->GetCompiledInfo<HistogramV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context->GetNodeName(), "sysWorkspaceSize is %ld.", compileInfo->sysWorkspaceSize);
    compileInfo->npuArch = ascendcPlatform.GetCurNpuArch();
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context->GetNodeName(), "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For HistogramV2 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HistogramV2)
    .Tiling(TilingHistogramV2)
    .TilingParse<HistogramV2CompileInfo>(TilingPrepareForHistogramV2);
} // namespace optiling