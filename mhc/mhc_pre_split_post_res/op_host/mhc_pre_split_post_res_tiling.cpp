/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res_tiling.cpp
 * \brief
 */

#include "mhc_pre_split_post_res_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

// ------------------算子原型索引常量定义----------------
// Inputs Index
static constexpr int64_t MHC_X_INDEX = 0;
static constexpr int64_t MHC_PHI_INDEX = 1;
static constexpr int64_t MHC_ALPHA_INDEX = 2;
static constexpr int64_t MHC_BIAS_INDEX = 3;

// Outputs Index
static constexpr int64_t MHC_H_POST_INDEX = 0;
static constexpr int64_t MHC_H_RES_INDEX = 1;

// Attr Index
static constexpr int64_t MHC_NORM_EPS_INDEX = 0;

// Dim Index
static constexpr int64_t MHC_DIM_ZERO = 0;
static constexpr int64_t MHC_DIM_ONE = 1;
static constexpr int64_t MHC_DIM_TWO = 2;
static constexpr int64_t MHC_DIM_THREE = 3;

// Num constants
static constexpr int64_t MHC_NUM_ZERO = 0;
static constexpr int64_t MHC_NUM_ONE = 1;
static constexpr int64_t MHC_NUM_TWO = 2;
static constexpr int64_t MHC_NUM_THREE = 3;
static constexpr int64_t MHC_NUM_FOUR = 4;

// 固定参数
static constexpr int64_t MHC_N_VALUE = 4;          // N 固定为4
static constexpr int64_t MHC_ALIGN_ELEM = 128;     // 256B / 2 (bf16/fp16)，D方向对齐粒度
static constexpr int64_t MHC_DIM_TILE_MAX = 2560;  // UB单次能处理的最大D大小

// 入参限制
static constexpr int64_t MHC_MIN_T = 1;
static constexpr int64_t MHC_MAX_T = 512 * 1024;
// D 仅支持 2560 和 5120 两个值
static constexpr int64_t MHC_D_2560 = 2560;
static constexpr int64_t MHC_D_5120 = 5120;

static constexpr int64_t MIN_FP32_ALIGN_ELEM_SIZE = 8;

// --------------------------GetOpName-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("MhcPreSplitPostRes", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

// --------------------------GetNpuInfo-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    if (platformInfo_ == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MhcPreSplitPostResCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
        OP_LOGD(context_->GetNodeName(), "GetPlatformInfo: platformInfo == nullptr, using compileInfo");
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
        OP_CHECK_IF(aicNum == 0 || coreNum_ == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;

        socVersion_ = ascendcPlatform.GetSocVersion();
    }

    OP_CHECK_IF(coreNum_ == 0 || ubSize_ == 0,
        OP_LOGE(opName_, "coreNum=%lu, ubSize=%lu obtained from platform/compile info is invalid.", coreNum_, ubSize_),
        return GRAPH_FAILED);
    OP_CHECK_IF((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
                    (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93),
        OP_LOGE(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_),
        return GRAPH_FAILED);

    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OP_LOGE(opName_, "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
        OP_LOGE(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    // workspace 大小在 GenerateTiling 中根据 blockDim/N/D 动态确定，此处仅做 nullptr 检查

    return ge::GRAPH_SUCCESS;
}

// --------------------------GetAttrInfo-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::GetAttrInfo()
{
    auto attrsPtr = context_->GetAttrs();
    OP_CHECK_IF(
        attrsPtr == nullptr, OP_LOGE(context_->GetNodeName(), "GetAttrs() returned nullptr."), return ge::GRAPH_FAILED);

    auto normEpsPtr = attrsPtr->GetFloat(MHC_NORM_EPS_INDEX);
    normEps_ = (normEpsPtr != nullptr ? *normEpsPtr : 1e-6f);

    return ge::GRAPH_SUCCESS;
}

// --------------------------CheckXValid-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::CheckXValid()
{
    const gert::StorageShape *xShapePtr = context_->GetInputShape(MHC_X_INDEX);
    OP_CHECK_IF(
        xShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "x InputShape is nullptr."), return ge::GRAPH_FAILED);

    int64_t xDimNum = static_cast<int64_t>(xShapePtr->GetStorageShape().GetDimNum());
    OP_CHECK_IF(xDimNum != MHC_NUM_THREE && xDimNum != MHC_NUM_FOUR,
        OP_LOGE(context_->GetNodeName(), "x: number of dimensions must be 3 (TND) or 4 (BSND)."),
        return ge::GRAPH_FAILED);

    if (xDimNum == MHC_NUM_FOUR) {
        // BSND: [B, S, N, D]
        int64_t B = xShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO);
        int64_t S = xShapePtr->GetStorageShape().GetDim(MHC_DIM_ONE);
        N_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_TWO);
        D_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_THREE);
        totalLength_ = B * S;
    } else {
        // TND: [T, N, D]
        totalLength_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO);
        N_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_ONE);
        D_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_TWO);
    }

    bool isValid = true;
    isValid = isValid && (totalLength_ >= MHC_MIN_T && totalLength_ <= MHC_MAX_T);
    isValid = isValid && (N_ == MHC_N_VALUE);
    isValid = isValid && (D_ == MHC_D_2560 || D_ == MHC_D_5120);
    OP_CHECK_IF(!isValid,
        OP_LOGE(context_->GetNodeName(),
            "x shape is invalid: T=%ld, N=%ld, D=%ld (D must be 2560 or 5120).",
            totalLength_,
            N_,
            D_),
        return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(MHC_X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return ge::GRAPH_FAILED);
    ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DataType::DT_BF16 && xDtype != ge::DataType::DT_FLOAT16,
        OP_LOGE(context_->GetNodeName(), "x only supports bfloat16 and float16."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// --------------------------CheckPhiValid-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::CheckPhiValid()
{
    const gert::StorageShape *phiShapePtr = context_->GetInputShape(MHC_PHI_INDEX);
    OP_CHECK_IF(phiShapePtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "phi InputShape is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(phiShapePtr->GetStorageShape().GetDimNum() != MHC_NUM_TWO,
        OP_LOGE(context_->GetNodeName(), "phi: number of dimensions must be 2."),
        return ge::GRAPH_FAILED);

    // phi shape: [N^2+N, N*D]
    int64_t phiDim0 = phiShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO);
    int64_t phiDim1 = phiShapePtr->GetStorageShape().GetDim(MHC_DIM_ONE);
    phiRows_ = N_ * N_ + N_;

    bool isValid = true;
    isValid = isValid && (phiDim0 == phiRows_);
    isValid = isValid && (phiDim1 == N_ * D_);
    OP_CHECK_IF(!isValid,
        OP_LOGE(context_->GetNodeName(),
            "phi shape invalid: expected [%ld, %ld], got [%ld, %ld].",
            phiRows_,
            N_ * D_,
            phiDim0,
            phiDim1),
        return ge::GRAPH_FAILED);

    auto phiDesc = context_->GetInputDesc(MHC_PHI_INDEX);
    OP_CHECK_IF(phiDesc == nullptr, OP_LOGE(context_->GetNodeName(), "phiDesc is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(phiDesc->GetDataType() != ge::DataType::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "phi dtype must be float32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// --------------------------CheckAlphaValid-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::CheckAlphaValid()
{
    const gert::StorageShape *alphaShapePtr = context_->GetInputShape(MHC_ALPHA_INDEX);
    OP_CHECK_IF(alphaShapePtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "alpha InputShape is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(alphaShapePtr->GetStorageShape().GetDimNum() != MHC_NUM_ONE,
        OP_LOGE(context_->GetNodeName(), "alpha: number of dimensions must be 1."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(alphaShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO) != MHC_NUM_TWO,
        OP_LOGE(context_->GetNodeName(), "alpha shape must be [2]."),
        return ge::GRAPH_FAILED);

    auto alphaDesc = context_->GetInputDesc(MHC_ALPHA_INDEX);
    OP_CHECK_IF(
        alphaDesc == nullptr, OP_LOGE(context_->GetNodeName(), "alphaDesc is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(alphaDesc->GetDataType() != ge::DataType::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "alpha dtype must be float32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// --------------------------CheckBiasValid-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::CheckBiasValid()
{
    const gert::StorageShape *biasShapePtr = context_->GetInputShape(MHC_BIAS_INDEX);
    OP_CHECK_IF(biasShapePtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "bias InputShape is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(biasShapePtr->GetStorageShape().GetDimNum() != MHC_NUM_ONE,
        OP_LOGE(context_->GetNodeName(), "bias: number of dimensions must be 1."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(biasShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO) != phiRows_,
        OP_LOGE(context_->GetNodeName(),
            "bias shape must be [%ld], got [%ld].",
            phiRows_,
            biasShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO)),
        return ge::GRAPH_FAILED);

    auto biasDesc = context_->GetInputDesc(MHC_BIAS_INDEX);
    OP_CHECK_IF(biasDesc == nullptr, OP_LOGE(context_->GetNodeName(), "biasDesc is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(biasDesc->GetDataType() != ge::DataType::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "bias dtype must be float32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// --------------------------GenerateTiling-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::CalcTokenGrouping()
{
    // T轴按核组分配（一组 = coresPerToken 个核处理一个token范围）
    int64_t coreGroups = static_cast<int64_t>(coreNum_) / coresPerToken_;
    OP_CHECK_IF(coreGroups == MHC_NUM_ZERO,
        OP_LOGE(opName_, "coreGroups is 0, coreNum=%lu, coresPerToken=%ld.", coreNum_, coresPerToken_),
        return ge::GRAPH_FAILED);

    baseT_ = (totalLength_ + coreGroups - MHC_NUM_ONE) / coreGroups;
    coreTAct_ = (totalLength_ + baseT_ - MHC_NUM_ONE) / baseT_;
    tailT_ = totalLength_ - (coreTAct_ - MHC_NUM_ONE) * baseT_;

    // 实际启动核数
    blockDim_ = coreTAct_ * coresPerToken_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSplitPostResInfoParser::CalcReduceParams()
{
    // 保留，暂未使用
    // -------------------------------------------------------
    // ReduceSum work tensor 大小（float）：
    //   xTileElems = N * dimTile
    //   elementsPerRepeat = 256B / sizeof(float) = 64
    //   elementsPerBlock  =  32B / sizeof(float) = 8
    //   workNeed = RoundUp(xTileElems / 64, 8) * 8
    // -------------------------------------------------------
    int64_t xTileElems = N_ * dimTile_;
    int64_t workNeed = ((xTileElems / 64 + 7) / 8) * 8;
    if (workNeed < MIN_FP32_ALIGN_ELEM_SIZE)
        workNeed = MIN_FP32_ALIGN_ELEM_SIZE;  // 最小 FP32_ALIGN_ELEM = 8
    reduceWorkSize_ = workNeed;

    // invK = 1/(N*D)，RmsNorm 均值系数；host 侧浮点运算，无 aicore 类型限制
    invK_ = 1.0f / static_cast<float>(N_ * D_);

    OP_LOGD(context_->GetNodeName(), "reduceWorkSize: %ld (xTileElems=%ld)", reduceWorkSize_, xTileElems);
    OP_LOGD(context_->GetNodeName(), "invK: %f (N=%ld, D=%ld)", invK_, N_, D_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSplitPostResInfoParser::CheckUbBudget()
{
    const int64_t xElemBytes = 2;
    const int64_t fp32Bytes = 4;
    const int64_t alignedPhiRows = ((phiRows_ + MIN_FP32_ALIGN_ELEM_SIZE - 1) / MIN_FP32_ALIGN_ELEM_SIZE) *
                                   MIN_FP32_ALIGN_ELEM_SIZE;
    const int64_t xTileElemsUb = N_ * dimTile_;
    const int64_t fixedUbBytes = 2 * (alignedPhiRows + MIN_FP32_ALIGN_ELEM_SIZE) * fp32Bytes +
                                 2 * (MIN_FP32_ALIGN_ELEM_SIZE + reduceWorkSize_) * fp32Bytes +
                                 MIN_FP32_ALIGN_ELEM_SIZE * fp32Bytes;
    // 线性增长 buffer（对应 kernel InitCommonBuffers）：xInQueue_(2B/元素，SINGLE depth1/MULTI depth2)、
    // xFp32Buf_(4B)、squareBuf_(4B，仅 SINGLE 分配)、phiRowQue_(4B，depth2)
    const bool isSingleTile = (dimLoop_ == MHC_NUM_ONE);
    const int64_t xInQueueBytes = xTileElemsUb * xElemBytes * (isSingleTile ? MHC_NUM_ONE : MHC_NUM_TWO);
    const int64_t xFp32BufBytes = xTileElemsUb * fp32Bytes;
    const int64_t squareBufBytes = isSingleTile ? xTileElemsUb * fp32Bytes : MHC_NUM_ZERO;
    const int64_t phiRowQueBytes = xTileElemsUb * fp32Bytes * MHC_NUM_TWO;
    const int64_t ubUsageBytes = xInQueueBytes + xFp32BufBytes + squareBufBytes + phiRowQueBytes + fixedUbBytes;
    OP_CHECK_IF(ubUsageBytes > static_cast<int64_t>(ubSize_),
        OP_LOGE(opName_,
            "UB usage %ld exceeds ubSize %lu (dimTile=%ld, dimLoop=%ld), check tiling/UB budget.",
            ubUsageBytes, ubSize_, dimTile_, dimLoop_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSplitPostResInfoParser::GenerateTiling()
{
    // D轴分块：设计文档规定 dimTile = min(D, 2560)，对齐128元素
    dimTile_ = D_ > MHC_DIM_TILE_MAX ? MHC_DIM_TILE_MAX : D_;

    dimTile_ = (dimTile_ / MHC_ALIGN_ELEM) * MHC_ALIGN_ELEM;
    if (dimTile_ == MHC_NUM_ZERO) {
        dimTile_ = MHC_ALIGN_ELEM;
    }

    // D轴需要几个tile
    dimLoop_ = (D_ + dimTile_ - MHC_NUM_ONE) / dimTile_;

    // 每token使用核数：仅当 D<=2560（dimLoop==1）且 token 数不足以填满一半核数时启用双核
    // token 充足（totalLength >= coreNum/2）时单核已能饱和所有核，双核只引入额外开销
    bool needDualCore = (dimLoop_ == MHC_NUM_ONE) && (totalLength_ < static_cast<int64_t>(coreNum_) / MHC_NUM_TWO);
    coresPerToken_ = needDualCore ? MHC_NUM_TWO : MHC_NUM_ONE;

    // phi行分配：双核模式下，phiRowMid是前半部分的行数
    // 前半行[0, phiRowMid) 对应 h_post(N行) + comb前半(N*N/2行)
    // 实际上双核按 phiRows/2 对半分
    phiRowMid_ = (coresPerToken_ == MHC_NUM_TWO) ? (phiRows_ / MHC_NUM_TWO) : phiRows_;

    // tilingKey
    tilingKey_ = (dimLoop_ == MHC_NUM_ONE) ? MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE : MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;

    if (CalcTokenGrouping() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const size_t wsBytes = 32;
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr,
        OP_LOGE(opName_, "currentWorkspace is nullptr"),
        return ge::GRAPH_FAILED);
    currentWorkspace[0] = wsBytes;
    OP_LOGD(context_->GetNodeName(), "wsBytes: %zu (blockDim=%ld, N=%ld, D=%ld)", wsBytes, blockDim_, N_, D_);

    if (CalcReduceParams() != ge::GRAPH_SUCCESS || CheckUbBudget() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// --------------------------ParseAndCheck-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::ParseAndCheck()
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetAttrInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckXValid() || ge::GRAPH_SUCCESS != CheckPhiValid() ||
        ge::GRAPH_SUCCESS != CheckAlphaValid() || ge::GRAPH_SUCCESS != CheckBiasValid()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GenerateTiling()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForMhcPreSplitPostRes(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------DoTiling-------------------------------------
ge::graphStatus MhcPreSplitPostResTiling::DoTiling()
{
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_N(N_);
    tilingData_.set_D(D_);
    tilingData_.set_phiRows(phiRows_);
    tilingData_.set_dimTile(dimTile_);
    tilingData_.set_dimLoop(dimLoop_);
    tilingData_.set_coresPerToken(coresPerToken_);
    tilingData_.set_phiRowMid(phiRowMid_);
    tilingData_.set_blockDim(blockDim_);
    tilingData_.set_baseT(baseT_);
    tilingData_.set_tailT(tailT_);
    tilingData_.set_coreTAct(coreTAct_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_reduceWorkSize(reduceWorkSize_);
    tilingData_.set_invK(invK_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(blockDim_);

    OP_LOGD(context_->GetNodeName(), "totalLength: %ld", totalLength_);
    OP_LOGD(context_->GetNodeName(), "N: %ld", N_);
    OP_LOGD(context_->GetNodeName(), "D: %ld", D_);
    OP_LOGD(context_->GetNodeName(), "phiRows: %ld", phiRows_);
    OP_LOGD(context_->GetNodeName(), "dimTile: %ld", dimTile_);
    OP_LOGD(context_->GetNodeName(), "dimLoop: %ld", dimLoop_);
    OP_LOGD(context_->GetNodeName(), "coresPerToken: %ld", coresPerToken_);
    OP_LOGD(context_->GetNodeName(), "phiRowMid: %ld", phiRowMid_);
    OP_LOGD(context_->GetNodeName(), "blockDim: %ld", blockDim_);
    OP_LOGD(context_->GetNodeName(), "baseT: %ld", baseT_);
    OP_LOGD(context_->GetNodeName(), "tailT: %ld", tailT_);
    OP_LOGD(context_->GetNodeName(), "coreTAct: %ld", coreTAct_);
    OP_LOGD(context_->GetNodeName(), "normEps: %f", normEps_);
    OP_LOGD(context_->GetNodeName(), "invK: %f", invK_);
    OP_LOGD(context_->GetNodeName(), "reduceWorkSize: %ld", reduceWorkSize_);
    OP_LOGD(context_->GetNodeName(), "tilingKey: %lu", tilingKey_);
    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
static ge::graphStatus TilingForMhcPreSplitPostRes(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("MhcPreSplitPostRes", "Tiling context is null."),
        return ge::GRAPH_FAILED);
    MhcPreSplitPostResTiling tiling(context);
    if (tiling.ParseAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tiling.DoTiling();
}

// --------------------------注册Tiling函数及TilingPrepare函数--------
IMPL_OP_OPTILING(MhcPreSplitPostRes)
    .Tiling(TilingForMhcPreSplitPostRes)
    .TilingParse<MhcPreSplitPostResCompileInfo>(TilingPrepareForMhcPreSplitPostRes);

}  // namespace optiling
