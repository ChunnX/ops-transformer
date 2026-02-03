/*!
 * \file histogram_v2_base_tiling.cpp
 * \brief
 */
#include "histogram_v2_tiling.h"
#include "common/log.h"

namespace optiling {
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

ge::graphStatus HistogramV2BaseClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        npuArch = ascendcPlatform.GetCurNpuArch();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        aicoreParams_.ubSize = ubSizePlatform;
    } else {
        auto compileInfoPtr = reinterpret_cast<const HistogramV2CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        npuArch = compileInfoPtr->npuArch;
        aicoreParams_.ubSize = compileInfoPtr->ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2BaseClass::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2BaseClass::GetWorkspaceSize()
{
    // 计算workspace大小，无需workspace临时空间，不存在多核同步，预留固定大小即可
    workspaceSize_ = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling