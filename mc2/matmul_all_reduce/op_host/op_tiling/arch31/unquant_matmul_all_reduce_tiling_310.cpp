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
 * \file unquant_matmul_all_reduce_tiling_310.cc
 * \brief
 */
#include "unquant_matmul_all_reduce_tiling_310.h"
#include "mc2_log.h"
#include "op_mc2.h"
using namespace Mc2Log;

namespace optiling {
bool UnQuantMatmulAllReduceTiling310::IsCapable()
{
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }
    auto weightTensor = context_->GetInputDesc(static_cast<size_t>(ParamValue::WEIGHT));
    OP_TILING_CHECK(
        weightTensor == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "weight tensor is invalid"),
        return false);
    auto format = weightTensor->GetStorageFormat();
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND310P || format == ge::Format::FORMAT_ND) {
        OP_LOGI(opName_, "skip normalized unquant tiling when is not 310p or not weight nz[%d].", format);
        return false;
    }
    if (args_.aType == matmul_tiling::DataType::DT_FLOAT16 && args_.bType == matmul_tiling::DataType::DT_FLOAT16) {
        OP_LOGI(opName_, "start with 310p normalized weight unquant tiling.");
        return true;
    }
    OP_LOGI(opName_, "skip 310p weight unquant tiling as dtype not support");
    return false;
}

ge::graphStatus UnQuantMatmulAllReduceTiling310::DoOpTiling()
{
    DoRCSTiling();
    DoSplitMTiling();
    if (isKZero_) {
        MutableTCubeTileTilingData().set_M(args_.orgMValue);
        MutableTCubeTileTilingData().set_isBias(args_.isBias);
        MutableTCubeTileTilingData().set_usedCoreNum(1);
        DoAllReduceTiling();
        return ge::GRAPH_SUCCESS;
    }
    GE_ASSERT_GRAPH_SUCCESS(DoUnQuantTiling());
    DoAllReduceTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t UnQuantMatmulAllReduceTiling310::GetTilingKey() const
{
    if (isKZero_) {
        const uint64_t emptyTensorKey = 2100000UL;
        OP_LOGI(opName_, "UnQuantMatmulAllReduceTiling310 get tilingKey %lu", emptyTensorKey);
        return emptyTensorKey;
    }
    uint64_t tilingKey = context_->GetTilingKey() + 2000U;
    OP_LOGI(opName_, "UnQuantMatmulAllReduceTiling310 get tilingKey %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus UnQuantMatmulAllReduceTiling310::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWorkspaceSize());
    myWorkSpaceSize_ = myWorkSpaceSize_ + (workspaceSize_ - libApiWorkSpaceSize_);
    OP_LOGI(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnQuantMatmulAllReduceTiling310::PostTiling()
{
    OP_LOGD(
        opName_, "final tiling data size: %zu and context capacity size: %zu ",
        unquantMatmulAllReduceTilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(unquantMatmulAllReduceTilingData_.GetDataSize());
    OP_TILING_CHECK(
        unquantMatmulAllReduceTilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "tiling data size[%zu] not aligned to 8", unquantMatmulAllReduceTilingData_.GetDataSize()),
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

Mc2Msg& UnQuantMatmulAllReduceTiling310::MutableMc2MsgData()
{
    return unquantMatmulAllReduceTilingData_.msg;
}

RCSTiling& UnQuantMatmulAllReduceTiling310::MutableRCSTilingData()
{
    return unquantMatmulAllReduceTilingData_.param;
}

TCubeTiling& UnQuantMatmulAllReduceTiling310::MutableTCubeTileTilingData()
{
    return unquantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}

TCubeTiling& UnQuantMatmulAllReduceTiling310::MutableTCubeTailTilingData()
{
    return unquantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus UnQuantMatmulAllReduceTiling310::DoUnQuantTiling()
{
    args_.mValue = tileMValue_;
    UnQuantTilingTransferHelper mmTile(*this, unquantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        return mmTile.DoTiling();
    } else {
        GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
        if (MutableRCSTilingData().get_tailCnt() == 0) {
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        UnQuantTilingTransferHelper mmTail(*this, unquantMatmulAllReduceTilingData_.tailmatmulTiling);
        return mmTail.DoTiling();
    }
}
} // namespace optiling
