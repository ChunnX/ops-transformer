#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/bsa_select_block_mask_tiling_data.h"
#include "../op_kernel/bsa_select_block_mask_tiling_key.h"
#include "bsa_select_block_mask_tiling.h"

namespace optiling {

static ge::graphStatus BSASelectBlockMaskTilingFunc(gert::TilingContext* context)
{
    BSASelectBlockMaskTilingBase bsaTiling(context);
    bsaTiling.DoTiling();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBSASelectBlockMask([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start parsing tiling for BSASelectBlockMask.");
    auto compileInfoPtr = context->GetCompiledInfo<BSASelectBlockMaskCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
        OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BSASelectBlockMask)
    .Tiling(BSASelectBlockMaskTilingFunc)
    .TilingParse<BSASelectBlockMaskCompileInfo>(TilingParseForBSASelectBlockMask);
} // namespace optiling
