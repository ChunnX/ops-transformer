#include <iostream>
#include "max_pool3d_grad_with_argmax_tiling.h"
#include "max_pool3d_grad_tiling_common.h"

namespace optiling {
ge::graphStatus MaxPool3DGradWithArgmaxScatterTiling::GetShapeAttrsInfo()
{
    MaxPool3DGradWithArgmaxTilingBase::GetShapeAttrsInfo();
    maxPoolGradParams.ncRound = 0UL;
    maxPoolGradParams.ncRoundTail = 0UL;
    maxPoolGradParams.totalRound = 0UL;
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradWithArgmaxScatterTiling::IsCapable()
{
    return true;
}

bool MaxPool3DGradWithArgmaxScatterTiling::SetScatterTilingParams()
{
    return CalculateScatterTilingParams(
        maxPoolGradParams,
        maxPoolGradParams.doDim,
        maxPoolGradParams.hoDim,
        maxPoolGradParams.woDim,
        maxPoolGradParams.xDtypeSize,
        maxPoolGradParams.indexDtypeSize,
        MAX_BLOCK_COUNT,
        BLOCK_SIZE);
}

void MaxPool3DGradWithArgmaxScatterTiling::SetOtherTilingParams()
{
    SetCntTailTilingParams();
    
    CalculateRoundParams(
        maxPoolGradParams,
        maxPoolGradParams.isOverLap,
        maxPoolGradParams.diDim,
        maxPoolGradParams.hiDim,
        maxPoolGradParams.wiDim);
}

void MaxPool3DGradWithArgmaxScatterTiling::SetScatterTilingData()
{
    SetScatterTilingDataCommon(tilingData, maxPoolGradParams);
}

void MaxPool3DGradWithArgmaxScatterTiling::PrintScatterTilingData()
{
    PrintScatterTilingDataCommon(context_->GetNodeName(), tilingData);
}

ge::graphStatus MaxPool3DGradWithArgmaxScatterTiling::DoOpTiling()
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

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", 
                         MaxPool3DGradWithArgmaxScatterTiling, 6);

} // namespace optiling