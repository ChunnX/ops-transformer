/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_all_reduce_tiling_310_general.cc
 * \brief
 */
#include "quant_matmul_all_reduce_tiling_310_general.h"
#include "mc2_log.h"
#include "op_mc2.h"
using namespace Mc2Log;

namespace optiling {
bool QuantMatmulAllReduceTiling310General::IsCapable()
{
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }
    if (args_.aType == matmul_tiling::DataType::DT_INT8 && args_.bType == matmul_tiling::DataType::DT_INT8) {
        OP_LOGI(opName_, "start with 310p quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "skip 310p quant tiling as dtype not support");
    return false;
}

ge::graphStatus QuantMatmulAllReduceTiling310General::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckA8W8());
    DoRCSTiling();
    DoSplitMTiling();
    GE_ASSERT_GRAPH_SUCCESS(DoQuantTiling());
    DoAllReduceTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantMatmulAllReduceTiling310General::GetTilingKey() const
{
    uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGI(opName_, "QuantMatmulAllReduceTiling310General get tilingKey %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus QuantMatmulAllReduceTiling310General::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWorkspaceSize());
    myWorkSpaceSize_ = myWorkSpaceSize_ + (workspaceSize_ - libApiWorkSpaceSize_);
    OP_LOGI(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTiling310General::PostTiling()
{
    OP_LOGD(
        opName_, "final tiling data size: %zu and context capacity size: %zu ",
        quantMatmulAllReduceTilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(quantMatmulAllReduceTilingData_.GetDataSize());
    OP_TILING_CHECK(
        quantMatmulAllReduceTilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "tiling data size[%zu] not aligned to 8", quantMatmulAllReduceTilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    if (MutableRCSTilingData().get_rankID() == 0) {
        PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingData());
        PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTileTilingData());
        PrintMc2MsgData(context_->GetNodeName(), MutableMc2MsgData());
        if (MutableRCSTilingData().get_tailM() > 0) {
            OP_LOGD(opName_, "have tail");
            PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTailTilingData());
        }
    }
    context_->SetBlockDim(args_.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

Mc2Msg& QuantMatmulAllReduceTiling310General::MutableMc2MsgData()
{
    return quantMatmulAllReduceTilingData_.msg;
}

RCSTiling& QuantMatmulAllReduceTiling310General::MutableRCSTilingData()
{
    return quantMatmulAllReduceTilingData_.param;
}

TCubeTiling& QuantMatmulAllReduceTiling310General::MutableTCubeTileTilingData()
{
    return quantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}

TCubeTiling& QuantMatmulAllReduceTiling310General::MutableTCubeTailTilingData()
{
    return quantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus QuantMatmulAllReduceTiling310General::DoQuantTiling()
{
    args_.mValue = tileMValue_;
    QuantTilingTransferHelper mmTile(*this, quantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        return mmTile.DoTiling();
    } else {
        GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
        if (MutableRCSTilingData().get_tailCnt() == 0) {
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        QuantTilingTransferHelper mmTail(*this, quantMatmulAllReduceTilingData_.tailmatmulTiling);
        return mmTail.DoTiling();
    }
}
} // namespace optiling
