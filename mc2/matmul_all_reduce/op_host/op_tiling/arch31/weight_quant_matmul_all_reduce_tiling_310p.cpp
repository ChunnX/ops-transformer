/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_matmul_all_reduce_til_310p.cc
 * \brief
 */
#include "weight_quant_matmul_all_reduce_til_310p.h"
#include "../../../op_kernel/matmul_all_reduce_til_key.h"
#include "mc2_log.h"
#include "common/utils/op_mc2.h"
using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optil {
bool WeightQuantMatmulAllReduceTiling310P::IsCapable()
{
    auto weightTensor = tilCtx_->GetInputDesc(static_cast<size_t>(ParamValue::WEIGHT));
    OP_TILING_CHECK(
        weightTensor == nullptr, VECTOR_INNER_ERR_REPORT_TILING(tilCtx_->GetNodeName(), "weight tensor is invalid"),
        return false);
    auto format = weightTensor->GetStorageFormat();
    if (npuArch_ != NpuArch::DAV_2002 || format == ge::Format::FORMAT_ND) {
        OP_LOGI(opLogName_, "skip weight quant til when is not 310p or is not weight nz[%d].", format);
        return false;
    }
    OP_LOGI(opLogName_, "sformat is %d.", format);
    if ((args_.aType == matmul_til::DataType::DT_FLOAT16 || args_.aType == matmul_til::DataType::DT_BFLOAT16) &&
        (args_.bType == matmul_til::DataType::DT_INT4 || args_.bType == matmul_til::DataType::DT_INT8)) {
        OP_LOGI(opLogName_, "start with 310p weight quant til.");
        return true;
    }
    OP_LOGI(opLogName_, "skip 310p weight quant til as dtype not support");
    return false;
}

void WeightQuantMatmulAllReduceTiling310P::UpdateCommOffset()
{
    auto&& args = MutableMc2MsgData();
    auto debugMode = mc2til::Mc2TilingUtils::GetDebugMode();
    // 只通信不计算模式下，如果K < N，sendOff的offset和sendCnt需要根据K计算
    auto columnNum = args_.orgNValue;
    if (debugMode == MC2_DEBUG_ONLY_AICPU && args_.orgKValue < args_.orgNValue) {
        columnNum = args_.orgKValue;
    }
    // AllReduce
    args.sendOff = tileMValue_ * args_.orgNValue * args_.outputDtypeSize;
    args.recvOff = tileMValue_ * columnNum * args_.outputDtypeSize;
    args.sendCnt = tileMValue_ * args_.orgNValue;
    args.recvCnt = tileMValue_ * columnNum;

    // 通信公式化Tiling计算中，可能有多个尾块
    args.tailSendOff = tailMValue_ * args_.orgNValue * args_.outputDtypeSize;
    args.tailRecvOff = tailMValue_ * columnNum * args_.outputDtypeSize;
    args.tailSendCnt = tailMValue_ * args_.orgNValue;
    args.tailRecvCnt = tailMValue_ * columnNum;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckA16W8());
    DoRCSTiling();
    DoSplitMTiling();
    wqMARData_.tilematmulTiling.cubeBlockDimM = 1;
    wqMARData_.tilematmulTiling.cubeBlockDimN = 1;
    if (isKZeroFlag_) {
        MutableTCubeTileTilingData().M = args_.orgMValue;
        MutableTCubeTileTilingData().isBias = args_.isBias;
        MutableTCubeTileTilingData().usedCoreNum = 1;
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
    if (isKZeroFlag_) {
        const uint64_t emptyKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_310P),
            static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
            static_cast<uint64_t>(isKZeroFlag_),
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

        OP_LOGI(opLogName_, "WeightQuantMatmulAllReduceTiling310P get tilKey %lu. isKZeroFlag_ %lu.", emptyKey, isKZeroFlag_);
        return emptyKey;
    }

    const uint64_t tilKey = GET_TPL_TILING_KEY(
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

    OP_LOGI(opLogName_, "WeightQuantMatmulAllReduceTiling310P get tilKey %lu. AntiQuantType %lu, hasAntiquantOffset %lu, transA %lu.", tilKey,
        weightQuantMatmul310TPLParam_.antiQuantType, weightQuantMatmul310TPLParam_.hasAntiquantOffset, weightQuantMatmul310TPLParam_.transA);
    return tilKey;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::GetWSSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWSSize());
    calcWorkspaceSz_ = calcWorkspaceSz_ + (workSpcSz_ - libApiWorkSpaceSize_);
    OP_LOGI(opLogName_, " set max workspace size %lu to context", calcWorkspaceSz_);
    size_t* wsPtr = tilCtx_->GetWSSizes(1); // set workspace
    wsPtr[0] = calcWorkspaceSz_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::PostTiling()
{
    size_t tilDataSz = sizeof(WeightQuantMatmulAllReduceNzTilingData);
    OP_LOGD(
        opLogName_, "final til data size: %zu and context capacity size: %zu ",
        tilDataSz, tilCtx_->GetRawTilingData()->GetCapacity());

    OP_TILING_CHECK(
        tilDataSz % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opLogName_, "til data size[%zu] not aligned to 8", tilDataSz),
        return ge::GRAPH_FAILED);
    tilCtx_->GetRawTilingData()->SetDataSize(tilDataSz);

    errno_t ret = memcpy_s(tilCtx_->GetRawTilingData()->GetData(), tilCtx_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void *>(&wqMARData_), tilDataSz);
    if (ret != EOK){
        OP_LOGE(tilCtx_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    if (MutableRCSTilingData().rankID == 0) {
        PrintRCSTilingData(tilCtx_->GetNodeName(), MutableRCSTilingData());
        PrintTCubeTilingData(tilCtx_->GetNodeName(), MutableTCubeTileTilingData());
        PrintMc2MsgData(tilCtx_->GetNodeName(), MutableMc2MsgData());
        if (MutableRCSTilingData().tailM > 0) {
            OP_LOGD(opLogName_, "have tail");
            PrintTCubeTilingData(tilCtx_->GetNodeName(), MutableTCubeTailTilingData());
        }
    }

    int32_t tile_core_num = wqMARData_.tilematmulTiling.cubeBlockDimM *
                            wqMARData_.tilematmulTiling.cubeBlockDimN;
    int32_t tail_core_num = wqMARData_.tailmatmulTiling.cubeBlockDimM *
                            wqMARData_.tailmatmulTiling.cubeBlockDimN;
    int32_t max_core_num = tile_core_num > tail_core_num ? tile_core_num : tail_core_num;
    OP_LOGI(
        opLogName_, " PostTiling tile_core_num:%d tail_core_num:%d max_core_num:%d", tile_core_num, tail_core_num,
        max_core_num);
    tilCtx_->SetBlockDim(max_core_num);

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    ret = tilCtx_->SetScheduleMode(batch_mode);
    GE_ASSERT_GRAPH_SUCCESS(ret); 
    
    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::Mc2Msg& WeightQuantMatmulAllReduceTiling310P::MutableMc2MsgData()
{
    return wqMARData_.msg;
}

Mc2Tiling::RCSTiling& WeightQuantMatmulAllReduceTiling310P::MutableRCSTilingData()
{
    return wqMARData_.param;
}

AscendC::til::TCubeTiling& WeightQuantMatmulAllReduceTiling310P::MutableTCubeTileTilingData()
{
    return wqMARData_.tilematmulTiling.matmulTiling;
}

AscendC::til::TCubeTiling& WeightQuantMatmulAllReduceTiling310P::MutableTCubeTailTilingData()
{
    return wqMARData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::DoWeightQuantTiling()
{
    args_.mValue = tileMValue_;
    OP_LOGI(opLogName_, "DoWeightQuantTiling tileMValue_:%lu", tileMValue_);
    WeightQuantTilingTransferHelper mmTile(*this, wqMARData_.tilematmulTiling);
    OP_LOGI(opLogName_, "DoWeightQuantTiling enableSplitK:%d", args_.enableSplitK);
    if (args_.enableSplitK) {
        auto res = mmTile.DoTiling();
        weightQuantMatmul310TPLParam_ = mmTile.GetWeightQuantMatmul310TPLParam();
        return res;
    } else {
        OP_LOGI(opLogName_, "DoWeightQuantTiling tailMValue_:%lu", tailMValue_);
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
        OP_LOGI(opLogName_, " tilematmulTiling tilKey %lu. AntiQuantType %lu, hasAntiquantOffset %lu, transA %lu.", tileTilingKey_,
            weightQuantMatmul310TPLParam_.antiQuantType, weightQuantMatmul310TPLParam_.hasAntiquantOffset, weightQuantMatmul310TPLParam_.transA);
        if (MutableRCSTilingData().tailCnt == 0) {
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        WeightQuantTilingTransferHelper mmTail(*this, wqMARData_.tailmatmulTiling);
        auto res = mmTail.DoTiling();
        weightQuantMatmul310TPLParam_ = mmTile.GetWeightQuantMatmul310TPLParam();
        return res;
    }
}

CutResult WeightQuantMatmulAllReduceTiling310P::GetTilingResult()
{
    CutResult mCutAllreduce;
    SocVersion inputSocVersion = SocVersion::SOC910_B;
    SetMCutSocVersion(inputSocVersion);
    const gert::StorageShape* commQScaleShp5 = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape* commQScaleShp5b = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQScaleShp5 != nullptr) && (commQScaleShp5b != nullptr)) { // low-bit comm
        OP_LOGD(opLogName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantRdcTilHccl4(
            args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVersion);
        quantRdcTilHccl4.GetTiling();
        mCutAllreduce = quantRdcTilHccl4.tilM_.cutRes;
    } else {
        MMPlusAllReduce rdcTilHccl4(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVersion, isPerBlock_);
        rdcTilHccl4.GetTiling();
        mCutAllreduce = rdcTilHccl4.tilM_.cutRes;
    }
    return mCutAllreduce;
}

//注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce, WeightQuantMatmulAllReduceTiling310P, \
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND310P), 0);
} // namespace optil
