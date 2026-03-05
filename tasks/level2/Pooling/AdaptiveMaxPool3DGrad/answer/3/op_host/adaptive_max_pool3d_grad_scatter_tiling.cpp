/*!
 * \file adaptive_max_pool3d_grad_scatter_tiling.cpp
 * \brief
 */
#include "adaptive_max_pool3d_grad_tiling.h"
#include <iostream>
#include "max_pool3d_grad_tiling_common.h"

namespace optiling {
ge::graphStatus AdaptiveMaxPool3DGradScatterTiling::GetShapeAttrsInfo()
{
    AdaptiveMaxPool3DGradTilingBase::GetShapeAttrsInfo();
    maxPoolGradParams.ncRound = 0UL;
    maxPoolGradParams.ncRoundTail = 0UL;
    maxPoolGradParams.totalRound = 0UL;
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveMaxPool3DGradScatterTiling::IsCapable()
{
    return true;
}

bool AdaptiveMaxPool3DGradScatterTiling::SetScatterTilingParams()
{
    return CalculateScatterTilingParams(
        maxPoolGradParams,
        maxPoolGradParams.doDim,
        maxPoolGradParams.hoDim,
        maxPoolGradParams.woDim,
        maxPoolGradParams.xDtypeSize,
        maxPoolGradParams.indexDtypeSize,
        MAX_BLOCK_COUNT);
}

void AdaptiveMaxPool3DGradScatterTiling::SetOtherTilingParams()
{
    SetCntTailTilingParams();
    
    CalculateRoundParams(
        maxPoolGradParams,
        true,
        maxPoolGradParams.diDim,
        maxPoolGradParams.hiDim,
        maxPoolGradParams.wiDim);
}

void AdaptiveMaxPool3DGradScatterTiling::SetScatterTilingData()
{
    SetScatterTilingDataCommon(tilingData, maxPoolGradParams);
}

void AdaptiveMaxPool3DGradScatterTiling::PrintScatterTilingData()
{
    PrintScatterTilingDataCommon(context_->GetNodeName(), tilingData);
    std::cout << "*******************START*******************" << std::endl;
    std::cout << "ncRound = " << tilingData.get_ncRound() << std::endl;
    std::cout << "ncRoundTail = " << tilingData.get_ncRoundTail() << std::endl;
    std::cout << "totalRound = " << tilingData.get_totalRound() << std::endl;
    std::cout << "preCoreNum = " << tilingData.get_preCoreNum() << std::endl;
    std::cout << "*******************END*******************" << std::endl;
}

ge::graphStatus AdaptiveMaxPool3DGradScatterTiling::DoOpTiling()
{
    bool res = SetScatterTilingParams();
    OP_CHECK_IF(!res, OP_LOGE(context_->GetNodeName(), "Scatter cal tiling params failed."), return ge::GRAPH_FAILED);
    maxPoolGradParams.tilingType = TILING_TYPE_SCATTER;
    SetOtherTilingParams();
    SetBaseTilingData();
    SetScatterTilingData();
    PrintTilingData();
    PrintScatterTilingData();
    
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("AdaptiveMaxPool3DGrad", AdaptiveMaxPool3DGradScatterTiling, 2);
} // namespace optiling