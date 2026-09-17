/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Tiling registration for the Ascend 950 MXFP8 weight-NZ V3 path.
 */

#include <vector>

#include "grouped_matmul_swiglu_quant_weight_nz_v3_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace Ops::Transformer::OpTiling;
using namespace optiling::GroupedMatmulSwigluQuantV3Tiling;

namespace optiling {
namespace {
constexpr int32_t V3_950_TILING_TEMPLATE = 0;

ge::graphStatus GroupedMatmulSwigluQuantV3TilingFunc(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return TilingRegistry::GetInstance().DoTilingImpl(context, {V3_950_TILING_TEMPLATE});
}

ASCENDC_EXTERN_C graphStatus TilingPrepareForGroupedMatmulSwigluQuantV3(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto compileInfo = context->GetCompiledInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo->l0CSize);
    return GRAPH_SUCCESS;
}
} // namespace

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV3, GroupedMatmulSwigluQuantV3Tiling950,
                             V3_950_TILING_TEMPLATE);

IMPL_OP_OPTILING(GroupedMatmulSwigluQuantV3)
    .Tiling(GroupedMatmulSwigluQuantV3TilingFunc)
    .TilingParse<GMMCompileInfo>(TilingPrepareForGroupedMatmulSwigluQuantV3);
} // namespace optiling
