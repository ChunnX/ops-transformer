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
 * \file matmul_all_reduce_tiling_910.cc
 * \brief
 */
#include "matmul_all_reduce_tiling_910.h"
#include "common/utils/op_mc2.h"

namespace optiling {
using namespace Mc2Tiling;
bool MatmulAllReduceTiling910::IsCapable()
{
    OP_LOGI(opLogName_, "start with MatmulAllReduceTiling910 tiling.");
    return true;
}

void MatmulAllReduceTiling910::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().M = args_.orgMValue;
    MutableTCubeTileTilingData().N = args_.orgNValue;
    MutableTCubeTileTilingData().isBias = args_.isBias;
    MutableTCubeTileTilingData().usedCoreNum = 1;
}

ge::graphStatus MatmulAllReduceTiling910::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckA16W16());
    GE_ASSERT_GRAPH_SUCCESS(CheckInput());
    DoRCSTiling();
    DoSplitMTiling();
    if (!isKZeroFlag_) {
        GE_ASSERT_GRAPH_SUCCESS(Do910Tiling());
    } else {
        DoEmptyTensorTiling();
    }
    DoAllReduceTiling(true);
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceTiling910::GetTilingKey() const
{
    if (unlikely(isKZeroFlag_)) {
        uint64_t tilingKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_910B), static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_FP_MM), isKZeroFlag_,
            MATMUL_ALLREDUCE_INT8_COMM_F,
            0UL, // ENABLE_L2_CACHE
            0UL, // SHARE_MM
            SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING);
        OP_LOGI(opLogName_, "Get tiling key %lu for empty tensor.", tilingKey);

        return tilingKey;
    }

    uint64_t MM_type = (matmulTPLParam_.disableMixNd2nz == MAT_MUL_V3_MIXND2NZ_FALSE && !enableBiasConvert_ &&
                        !matmulAllReduce910TilingData_.param.isAdd) ?
                           MATMUL_ALLREDUCE_MM_TYPE_FP_MM_CUBE_ONLY :
                           MATMUL_ALLREDUCE_MM_TYPE_FP_MM;

    uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(ASCEND_910B), MM_type, MATMUL_ALLREDUCE_EMPTY_INPUT_F, MATMUL_ALLREDUCE_INT8_COMM_F,
        0UL, // ENABLE_L2_CACHE
        0UL, // SHARE_MM
        matmulTPLParam_.disableMixNd2nz, SET_NOT_USE_QUANT_MM_TPL_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING);
    OP_LOGI(opLogName_, "Get tiling key %lu. Cube only flag is %lu", tilingKey, MM_type);

    return tilingKey;
}

ge::graphStatus MatmulAllReduceTiling910::GetWorkspaceSize()
{
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::GetWorkspaceSize());
    OP_LOGI(opLogName_, "select max workspace size to context, calcWorkspaceSz_:%lu, workspaceSize_:%lu", calcWorkspaceSz_,
            workspaceSize_);
    calcWorkspaceSz_ = std::max(calcWorkspaceSz_, workspaceSize_);
    size_t *wsPtr = tilCtx8_->GetWorkspaceSizes(1);
    wsPtr[0] = calcWorkspaceSz_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::PostTiling()
{
    size_t tilDataSz = sizeof(MatmulAllReduce910TilingData);
    OP_LOGD(opLogName_, "final tiling data size: %zu and context capacity size: %zu ", tilDataSz,
            tilCtx8_->GetRawTilingData()->GetCapacity());
    OP_TILING_CHECK(tilDataSz % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILING(opLogName_, "tiling data size[%zu] not aligned to 8", tilDataSz),
                    return ge::GRAPH_FAILED);
    tilCtx8_->GetRawTilingData()->SetDataSize(tilDataSz);

    errno_t ret = memcpy_s(tilCtx8_->GetRawTilingData()->GetData(), tilCtx8_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&matmulAllReduce910TilingData_), tilDataSz);
    if (ret != EOK) {
        OP_LOGE(tilCtx8_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    PrintTilingData();

    tilCtx8_->SetBlockDim(args_.aicCoreNum);

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    ret = tilCtx8_->SetScheduleMode(batch_mode);
    GE_ASSERT_GRAPH_SUCCESS(ret);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::Do910Tiling()
{
    args_.mValue = tileMValue_;
    TilingTransferHelper mmTile(*this, matmulAllReduce910TilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        OP_LOGD(opLogName_, "Enable SplitK Tiling.");
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
        TilingTransferHelper mmTail(*this, matmulAllReduce910TilingData_.tailmatmulTiling);
        auto res = mmTail.DoTiling();
        matmulTPLParam_ = mmTail.GetMatmulTPLParam();
        return res;
    }
}

Mc2Tiling::Mc2Msg &MatmulAllReduceTiling910::MutableMc2MsgData()
{
    return matmulAllReduce910TilingData_.msg;
}

Mc2Tiling::RCSTiling &MatmulAllReduceTiling910::MutableRCSTilingData()
{
    return matmulAllReduce910TilingData_.param;
}

AscendC::tiling::TCubeTiling &MatmulAllReduceTiling910::MutableTCubeTileTilingData()
{
    return matmulAllReduce910TilingData_.tilematmulTiling.matmulTiling;
}

AscendC::tiling::TCubeTiling &MatmulAllReduceTiling910::MutableTCubeTailTilingData()
{
    return matmulAllReduce910TilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus MatmulAllReduceTiling910::CheckAxisSize()
{
    const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
    OP_TILING_CHECK(
        m > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(), "The size of m-axis(%lu) exceeds the upper limit.", m),
        return ge::GRAPH_FAILED);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    OP_TILING_CHECK(
        k > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(), "The size of k-axis(%lu) exceeds the upper limit.", k),
        return ge::GRAPH_FAILED);
    const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
    OP_TILING_CHECK(
        n > static_cast<uint64_t>(INT32_MAX),
        VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(), "The size of n-axis(%lu) exceeds the upper limit.", n),
        return ge::GRAPH_FAILED);

    return CheckEmptyTensor();
}

ge::graphStatus MatmulAllReduceTiling910::CheckInputDtype()
{
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtx8_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(x2DimNum != DIM_NUM_TWO,
                    VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(),
                                                   "In the not quant scenario, Expect x2 dim to be 2, but "
                                                   " got x2 dim:[%lu].",
                                                   x2DimNum),
                    return ge::GRAPH_FAILED);
    auto x1Type = mmrCtx8_.x1->GetDataType();
    //  x1 为fp16 或者bf16
    OP_TILING_CHECK(!((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
                    VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(),
                                                   "In the not quant scenario, type of x1 should be"
                                                   " fp16 or bf16."),
                    return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x2Type = mmrCtx8_.x2->GetDataType();
    OP_TILING_CHECK(x1Type != x2Type,
                    VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(),
                                                   "In the not quant scenario, type of x1 and x2"
                                                   " should be same"),
                    return ge::GRAPH_FAILED);
    // x1,bias数据类型相同
    if (mmrCtx8_.bias_shape != nullptr) {
        auto biasType = mmrCtx8_.bias->GetDataType();
        OP_TILING_CHECK(x1Type != biasType,
                        VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(),
                                                       "In the not quant scenario, type of x1 and bias should be"
                                                       " same."),
                        return ge::GRAPH_FAILED);
    }
    OP_LOGD(opLogName_, "Check Input Dtype Success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::CheckInputFormat()
{
    // 非量化场景B矩阵不支持NZ
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(mmrCtx8_.x2->GetStorageFormat())) ==
            ge::Format::FORMAT_FRACTAL_NZ,
        VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(), "In the not quant scenario, weigth dont support NZ"),
        return ge::GRAPH_FAILED);
    OP_LOGD(opLogName_, "Check Input Format Success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::CheckInputShape()
{
    auto outputDimNum = mmrCtx8_.y_shape->GetStorageShape().GetDimNum();
    if (mmrCtx8_.x3_shape != nullptr) {
        auto x3DimNum = mmrCtx8_.x3_shape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(outputDimNum != x3DimNum,
                        VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(),
                                                       "In the not quant scenario, shape of x3 and output should be"
                                                       " same."),
                        return ge::GRAPH_FAILED);
        for (size_t i = 0U; i < outputDimNum; i++) {
            OP_TILING_CHECK(mmrCtx8_.y_shape->GetStorageShape().GetDim(i) !=
                                mmrCtx8_.x3_shape->GetStorageShape().GetDim(i),
                            VECTOR_INNER_ERR_REPORT_TILING(tilCtx8_->GetNodeName(),
                                                           "In the not quant scenario, shape of x3 and output should be"
                                                           " same."),
                            return ge::GRAPH_FAILED);
        }
    }
    OP_LOGD(opLogName_, "Check Input Shape Success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAllReduceTiling910::CheckInput()
{
    OP_LOGD(opLogName_, "Begin Check Input.");
    GE_ASSERT_GRAPH_SUCCESS(MatmulAllReduceTilingBase::CheckInput());
    GE_ASSERT_GRAPH_SUCCESS(CheckInputDtype());

    //  非量化场景不支持B矩阵Nz格式 除了310P
    if (npuArch_ != NpuArch::DAV_2002) {
        GE_ASSERT_GRAPH_SUCCESS(CheckInputFormat());
    }

    GE_ASSERT_GRAPH_SUCCESS(CheckInputShape());

    return CheckAxisSize();
}

MatmulAllReduceTiling910::MatmulAllReduceTiling910(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context), matmulAllReduce910TilingData_(matmulAllReduce910TilingDataSelf_)
{
}

MatmulAllReduceTiling910::MatmulAllReduceTiling910(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                                   MatmulAllReduce910TilingData *out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo), matmulAllReduce910TilingData_(*out)
{
}

ge::graphStatus TilingTransferHelper::GetShapeAttrsInfo()
{
    auto &&tilArg2 = tilingProcesser_.args_;
    args_.opName = tilingProcesser_.opLogName_;
    args_.isATrans = tilArg2.isATrans;
    args_.isBTrans = tilArg2.isBTrans;
    args_.hasBias = tilArg2.isBias;
    args_.aType = tilArg2.geAType;
    args_.bType = tilArg2.geBType;
    args_.cType = tilArg2.geCType;
    args_.biasType = tilArg2.isBias ? tilArg2.geBiasType : ge::DT_INT32;

    args_.aFormat = ge::FORMAT_ND;
    args_.bFormat = ge::FORMAT_ND;
    args_.outFormat = ge::FORMAT_ND;

    args_.mValue = tilArg2.mValue;
    args_.kValue = tilArg2.kValue;
    args_.nValue = tilArg2.nValue;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingTransferHelper::PostTiling()
{
    tilingProcesser_.calcWorkspaceSz_ = std::max(tilingProcesser_.calcWorkspaceSz_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opLogName_, " set mm workspace size %lu to mc2", tilingProcesser_.calcWorkspaceSz_);
    return ge::GRAPH_SUCCESS;
}

MatmulTPLParam TilingTransferHelper::GetMatmulTPLParam()
{
    MatmulTPLParam param;
    param.disableMixNd2nz = static_cast<uint64_t>(GetMixNd2nzType());
    // 1: disable mix nd2nz 0: enable mix nd2nz
    return param;
}

TilingTransferHelper::TilingTransferHelper(MatmulAllReduceTiling910 &matmulAllReduceTiling910,
                                           Mc2MatmulV3TilingData &data)
    : Mc2MatmulV3BaseTiling(matmulAllReduceTiling910.tilCtx8_, &data), tilingProcesser_(matmulAllReduceTiling910)
{
}

CutResult MatmulAllReduceTiling910::GetTilingResult()
{
    CutResult mCutRdc7;
    SocVersion inSocVer7 = SocVersion::SOC910_B;
    SetMCutSocVersion(inSocVer7);
    const gert::StorageShape *commQuantScaleShape1 = mmrCtx8_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleShape2 = mmrCtx8_.comm_quant_scale_2_shape;
    if ((commQuantScaleShape1 != nullptr) && (commQuantScaleShape2 != nullptr)) { // low-bit comm
        OP_LOGD(opLogName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantRdcTilHccl7(args_, args_.rankDim, KernelType::ALL_REDUCE, inSocVer7);
        quantRdcTilHccl7.GetTiling();
        mCutRdc7 = quantRdcTilHccl7.tilingM_.cutRes;
    } else {
        MMPlusAllReduce rdcTilHccl7(args_, args_.rankDim, KernelType::ALL_REDUCE, inSocVer7, isPerBlock_);
        rdcTilHccl7.GetTiling();
        mCutRdc7 = rdcTilHccl7.tilingM_.cutRes;
    }
    return mCutRdc7;
}


// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce, MatmulAllReduceTiling910,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B), 2);
} // namespace optiling
