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
 * \file unquant_matmul_all_reduce_tiling_310.cc
 * \brief
 */
#include "unquant_matmul_all_reduce_tiling_310.h"
#include "../../../op_kernel/matmul_all_reduce_tiling_key.h"
#include "mc2_log.h"
#include "common/utils/op_mc2.h"
using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optiling {
bool UnQuantMatmulAllReduceTiling310::IsCapable()
{
    auto weightTensor = context_->GetInputDesc(static_cast<size_t>(ParamValue::WEIGHT));
    OP_TILING_CHECK(
        weightTensor == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "weight tensor is invalid"),
        return false);
    auto format = weightTensor->GetStorageFormat();
    if ((npuArch_ != NpuArch::DAV_2002) || (format == ge::Format::FORMAT_ND)) {
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
    if (isKZeroFl_) {
        MutableTCubeTileTilingData().M = args_.orgMValue;
        MutableTCubeTileTilingData().isBias = args_.isBias;
        MutableTCubeTileTilingData().usedCoreNum = 1;
        DoAllReduceTiling();
        return ge::GRAPH_SUCCESS;
    }
    GE_ASSERT_GRAPH_SUCCESS(DoUnQuantTiling());
    DoAllReduceTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t UnQuantMatmulAllReduceTiling310::GetTilingKey() const
{
    if (isKZeroFl_) {
        const uint64_t emptyKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_310P),
            static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_FP_MM),
            static_cast<uint64_t>(isKZeroFl_),
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

        OP_LOGI(opName_, "UnQuantMatmulAllReduceTiling310 get tilingKey %lu. isKZeroFl_ %lu", emptyKey, isKZeroFl_);
        return emptyKey;
    }

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(ASCEND_310P),
        static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_FP_MM),
        MATMUL_ALLREDUCE_EMPTY_INPUT_F,
        MATMUL_ALLREDUCE_INT8_COMM_F,
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        matmulTPLParam_.disableMixNd2nz,
        SET_NOT_USE_QUANT_MM_TPL_TILING,
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(FORMAT_B_NZ));

    OP_LOGI(opName_, "UnQuantMatmulAllReduceTiling310 get tilingKey %lu. disableMixNd2nz %lu", tilingKey, matmulTPLParam_.disableMixNd2nz);
    return tilingKey;
}

ge::graphStatus UnQuantMatmulAllReduceTiling310::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWorkspaceSize());
    calcWorkspaceSz_ = calcWorkspaceSz_ + (workspaceSize_ - libApiWorkSpaceSize_);
    OP_LOGI(opName_, " set max workspace size %lu to context", calcWorkspaceSz_);
    size_t* wsPtr = context_->GetWorkspaceSizes(1); // set workspace
    wsPtr[0] = calcWorkspaceSz_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnQuantMatmulAllReduceTiling310::PostTiling()
{
    size_t tilDataSz = sizeof(UnQuantMatmulAllReduceTilingData);
    OP_LOGD(
        opName_, "final tiling data size: %zu and context capacity size: %zu ",
        tilDataSz, context_->GetRawTilingData()->GetCapacity());
    OP_TILING_CHECK(
        tilDataSz % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "tiling data size[%zu] not aligned to 8", tilDataSz),
        return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilDataSz);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void *>(&unquantMARData_), tilDataSz);
    if (ret != EOK){
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    if (MutableRCSTilingData().rankID == 0) {
        PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingData());
        PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTileTilingData());
        PrintMc2MsgData(context_->GetNodeName(), MutableMc2MsgData());
        if (MutableRCSTilingData().tailM > 0) {
            OP_LOGD(opName_, "have tail");
            PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTailTilingData());
        }
    }

    context_->SetBlockDim(args_.aicCoreNum);

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    ret = context_->SetScheduleMode(batch_mode);
    GE_ASSERT_GRAPH_SUCCESS(ret); 

    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::Mc2Msg& UnQuantMatmulAllReduceTiling310::MutableMc2MsgData()
{
    return unquantMARData_.msg;
}

Mc2Tiling::RCSTiling& UnQuantMatmulAllReduceTiling310::MutableRCSTilingData()
{
    return unquantMARData_.param;
}

AscendC::tiling::TCubeTiling& UnQuantMatmulAllReduceTiling310::MutableTCubeTileTilingData()
{
    return unquantMARData_.tilematmulTiling.matmulTiling;
}

AscendC::tiling::TCubeTiling& UnQuantMatmulAllReduceTiling310::MutableTCubeTailTilingData()
{
    return unquantMARData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus UnQuantMatmulAllReduceTiling310::DoUnQuantTiling()
{
    args_.mValue = tileMValue_;
    UnQuantTilingTransferHelper mmTile(*this, unquantMARData_.tilematmulTiling);
    if (args_.enableSplitK) {
        auto res = mmTile.DoTiling();
        matmulTPLParam_ = mmTile.GetMatmulTPLParam();
        return res;
    } else {
        GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
        if (MutableRCSTilingData().tailCnt == 0) {
            matmulTPLParam_ = mmTile.GetMatmulTPLParam();

            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        UnQuantTilingTransferHelper mmTail(*this, unquantMARData_.tailmatmulTiling);
        auto res = mmTail.DoTiling();
        matmulTPLParam_ = mmTail.GetMatmulTPLParam();
        return res;
    }
}

CutResult UnQuantMatmulAllReduceTiling310::GetTilingResult()
{
    CutResult mCutRdc3;
    SocVersion inSocVer3 = SocVersion::SOC910_B;
    SetMCutSocVersion(inSocVer3);
    const gert::StorageShape* commQScaleShp4 = mmrCtx3_.comm_quant_scale_1_shape;
    const gert::StorageShape* commQScaleShp4b = mmrCtx3_.comm_quant_scale_2_shape;
    if ((commQScaleShp4 != nullptr) && (commQScaleShp4b != nullptr)) { // low-bit comm
        OP_LOGD(opName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantRdcTilHccl3(
            args_, args_.rankDim, KernelType::ALL_REDUCE, inSocVer3);
        quantRdcTilHccl3.GetTiling();
        mCutRdc3 = quantRdcTilHccl3.tilingM_.cutRes;
    } else {
        MMPlusAllReduce rdcTilHccl3(args_, args_.rankDim, KernelType::ALL_REDUCE, inSocVer3, isPerBlock_);
        rdcTilHccl3.GetTiling();
        mCutRdc3 = rdcTilHccl3.tilingM_.cutRes;
    }
    return mCutRdc3;
}

//注册Tiling
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce, UnQuantMatmulAllReduceTiling310, \
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND310P), 2);
} // namespace optiling
