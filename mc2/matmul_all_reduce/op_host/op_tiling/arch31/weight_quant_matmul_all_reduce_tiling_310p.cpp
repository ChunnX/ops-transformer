/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_matmul_all_reduce_tiling_310p.cc
 * \brief
 */
#include "weight_quant_matmul_all_reduce_tiling_310p.h"
#include "../../../op_kernel/matmul_all_reduce_tiling_key.h"
#include "mc2_log.h"
#include "op_mc2.h"
using namespace Mc2Log;

namespace optiling {
bool WeightQuantMatmulAllReduceTiling310P::IsCapable()
{
    auto weightTensor = context_->GetInputDesc(static_cast<size_t>(ParamValue::WEIGHT));
    OP_TILING_CHECK(
        weightTensor == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "weight tensor is invalid"),
        return false);
    auto format = weightTensor->GetStorageFormat();
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND310P || format == ge::Format::FORMAT_ND) {
        OP_LOGI(opName_, "skip weight quant tiling when is not 310p or is not weight nz[%d].", format);
        return false;
    }
    OP_LOGI(opName_, "sformat is %d.", format);
    if ((args_.aType == matmul_tiling::DataType::DT_FLOAT16 || args_.aType == matmul_tiling::DataType::DT_BFLOAT16) &&
        (args_.bType == matmul_tiling::DataType::DT_INT4 || args_.bType == matmul_tiling::DataType::DT_INT8)) {
        OP_LOGI(opName_, "start with 310p weight quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "skip 310p weight quant tiling as dtype not support");
    return false;
}

void WeightQuantMatmulAllReduceTiling310P::UpdateCommOffset()
{
    auto&& args = MutableMc2MsgData();
    auto debugMode = mc2tiling::Mc2TilingUtils::GetDebugMode();
    // 只通信不计算模式下，如果K < N，sendOff的offset和sendCnt需要根据K计算
    auto columnNum = args_.orgNValue;
    if (debugMode == MC2_DEBUG_ONLY_AICPU && args_.orgKValue < args_.orgNValue) {
        columnNum = args_.orgKValue;
    }
    // AllReduce
    args.set_sendOff(tileMValue_ * args_.orgNValue * args_.outputDtypeSize);
    args.set_recvOff(tileMValue_ * columnNum * args_.outputDtypeSize);
    args.set_sendCnt(tileMValue_ * args_.orgNValue);
    args.set_recvCnt(tileMValue_ * columnNum);

    // 通信公式化Tiling计算中，可能有多个尾块
    args.set_tailSendOff(tailMValue_ * args_.orgNValue * args_.outputDtypeSize);
    args.set_tailRecvOff(tailMValue_ * columnNum * args_.outputDtypeSize);
    args.set_tailSendCnt(tailMValue_ * args_.orgNValue);
    args.set_tailRecvCnt(tailMValue_ * columnNum);
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckA16W8());
    DoRCSTiling();
    DoSplitMTiling();
    weightQuantMatmulAllReduceTilingData_.tilematmulTiling.set_cubeBlockDimM(1);
    weightQuantMatmulAllReduceTilingData_.tilematmulTiling.set_cubeBlockDimN(1);
    if (isKZero_) {
        MutableTCubeTileTilingData().set_M(args_.orgMValue);
        MutableTCubeTileTilingData().set_isBias(args_.isBias);
        MutableTCubeTileTilingData().set_usedCoreNum(1);
        DoAllReduceTiling();
        return ge::GRAPH_SUCCESS;
    }
    GE_ASSERT_GRAPH_SUCCESS(DoWeightQuantTiling());
    DoAllReduceTiling();
    UpdateCommOffset();
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantMatmulAllReduceTiling310P::GetTilingKey() const
{
    if (isKZero_) {
        const uint64_t emptyTensorKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_310P),
            static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
            static_cast<uint64_t>(isKZero_),
            MATMUL_ALLREDUCE_INT8_COMM_F,
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            SET_NOT_USE_FM_MM_TPL_TILING,
            SET_NOT_USE_QUANT_MM_TPL_TILING,
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(FORMAT_B_NZ));

        OP_LOGI(opName_, "WeightQuantMatmulAllReduceTiling310P get tilingKey %lu. isKZero_ %lu.", emptyTensorKey, isKZero_);
        return emptyTensorKey;
    }

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(ASCEND_310P),
        static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
        MATMUL_ALLREDUCE_EMPTY_INPUT_F,
        MATMUL_ALLREDUCE_INT8_COMM_F,
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        SET_NOT_USE_FM_MM_TPL_TILING,
        SET_NOT_USE_QUANT_MM_TPL_TILING,
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        weightQuantMatmul310TPLParam_.hasAntiquantOffset,
        weightQuantMatmul310TPLParam_.antiQuantType,
        weightQuantMatmul310TPLParam_.transA,
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(FORMAT_B_NZ));

    OP_LOGI(opName_, "WeightQuantMatmulAllReduceTiling310P get tilingKey %lu. AntiQuantType %lu, hasAntiquantOffset %lu, transA %lu.", tilingKey,
        weightQuantMatmul310TPLParam_.antiQuantType, weightQuantMatmul310TPLParam_.hasAntiquantOffset, weightQuantMatmul310TPLParam_.transA);
    return tilingKey;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWorkspaceSize());
    myWorkSpaceSize_ = myWorkSpaceSize_ + (workspaceSize_ - libApiWorkSpaceSize_);
    OP_LOGI(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::PostTiling()
{
    OP_LOGD(
        opName_, "final tiling data size: %zu and context capacity size: %zu ",
        weightQuantMatmulAllReduceTilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(weightQuantMatmulAllReduceTilingData_.GetDataSize());

    OP_TILING_CHECK(
        weightQuantMatmulAllReduceTilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "tiling data size[%zu] not aligned to 8", weightQuantMatmulAllReduceTilingData_.GetDataSize()),
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

    int32_t tile_core_num = weightQuantMatmulAllReduceTilingData_.tilematmulTiling.get_cubeBlockDimM() *
                            weightQuantMatmulAllReduceTilingData_.tilematmulTiling.get_cubeBlockDimN();
    int32_t tail_core_num = weightQuantMatmulAllReduceTilingData_.tailmatmulTiling.get_cubeBlockDimM() *
                            weightQuantMatmulAllReduceTilingData_.tailmatmulTiling.get_cubeBlockDimN();
    int32_t max_core_num = tile_core_num > tail_core_num ? tile_core_num : tail_core_num;
    OP_LOGI(
        opName_, " PostTiling tile_core_num:%d tail_core_num:%d max_core_num:%d", tile_core_num, tail_core_num,
        max_core_num);
    context_->SetBlockDim(max_core_num);
    return ge::GRAPH_SUCCESS;
}

Mc2Msg& WeightQuantMatmulAllReduceTiling310P::MutableMc2MsgData()
{
    return weightQuantMatmulAllReduceTilingData_.msg;
}

RCSTiling& WeightQuantMatmulAllReduceTiling310P::MutableRCSTilingData()
{
    return weightQuantMatmulAllReduceTilingData_.param;
}

TCubeTiling& WeightQuantMatmulAllReduceTiling310P::MutableTCubeTileTilingData()
{
    return weightQuantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}

TCubeTiling& WeightQuantMatmulAllReduceTiling310P::MutableTCubeTailTilingData()
{
    return weightQuantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::DoWeightQuantTiling()
{
    args_.mValue = tileMValue_;
    OP_LOGI(opName_, "DoWeightQuantTiling tileMValue_:%lu", tileMValue_);
    WeightQuantTilingTransferHelper mmTile(*this, weightQuantMatmulAllReduceTilingData_.tilematmulTiling);
    OP_LOGI(opName_, "DoWeightQuantTiling enableSplitK:%d", args_.enableSplitK);
    if (args_.enableSplitK) {
        auto res = mmTile.DoTiling();
        weightQuantMatmul310TPLParam_ = mmTile.GetWeightQuantMatmul310TPLParam();
        return res;
    } else {
        OP_LOGI(opName_, "DoWeightQuantTiling tailMValue_:%lu", tailMValue_);
        GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
        weightQuantMatmul310TPLParam_ = mmTile.GetWeightQuantMatmul310TPLParam();
        tileTilingKey_ = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_310P),
            static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
            MATMUL_ALLREDUCE_EMPTY_INPUT_F,
            MATMUL_ALLREDUCE_INT8_COMM_F,
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            SET_NOT_USE_FM_MM_TPL_TILING,
            SET_NOT_USE_QUANT_MM_TPL_TILING,
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            weightQuantMatmul310TPLParam_.hasAntiquantOffset,
            weightQuantMatmul310TPLParam_.antiQuantType,
            weightQuantMatmul310TPLParam_.transA,
            static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(FORMAT_B_NZ));
        OP_LOGI(opName_, " tilematmulTiling tilingKey %lu. AntiQuantType %lu, hasAntiquantOffset %lu, transA %lu.", tileTilingKey_,
            weightQuantMatmul310TPLParam_.antiQuantType, weightQuantMatmul310TPLParam_.hasAntiquantOffset, weightQuantMatmul310TPLParam_.transA);
        if (MutableRCSTilingData().get_tailCnt() == 0) {
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        WeightQuantTilingTransferHelper mmTail(*this, weightQuantMatmulAllReduceTilingData_.tailmatmulTiling);
        auto res = mmTail.DoTiling();
        weightQuantMatmul310TPLParam_ = mmTile.GetWeightQuantMatmul310TPLParam();
        return res;
    }
}

//注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce,WeightQuantMatmulAllReduceTiling310P,static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND310P),0);
} // namespace optiling
