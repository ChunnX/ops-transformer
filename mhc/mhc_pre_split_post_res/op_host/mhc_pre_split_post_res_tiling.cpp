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
 * \brief MhcPreSplitPostRes 算子 Tiling 切分与校验实现
 */

#include <cmath>

#include "mhc_pre_split_post_res_const.h"
#include "mhc_pre_split_post_res_tiling.h"
#include "../op_kernel/arch35/mhc_pre_split_post_res_tiling_key.h"
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
static constexpr int64_t MHC_H_RES_INDEX = 1;

// Attr Index
static constexpr int64_t MHC_NORM_EPS_INDEX = 0;

// Num constants
static constexpr int64_t MHC_NUM_ZERO = 0;
static constexpr int64_t MHC_NUM_ONE = 1;
static constexpr int64_t MHC_NUM_TWO = 2;

// 规格常量（MHC_N_VALUE/D 白名单/MHC_DIM_* 等）统一来自 mhc_pre_split_post_res_const.h

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
        OP_CHECK_IF(aicNum == 0 || coreNum_ == 0, OP_LOGE(opName_, "number of cores obtained is 0."),
            return GRAPH_FAILED);

        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;

        socVersion_ = ascendcPlatform.GetSocVersion();
    }

    OP_CHECK_IF(coreNum_ == 0 || ubSize_ == 0,
        OP_LOGE(opName_, "coreNum=%lu, ubSize=%lu obtained from platform/compile info are invalid.", coreNum_, ubSize_),
        return GRAPH_FAILED);
    OP_CHECK_IF(socVersion_ != platform_ascendc::SocVersion::ASCEND950,
        OP_LOGE(opName_, "SOC Version[%d] is not supported.", static_cast<int32_t>(socVersion_)),
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
    if (normEpsPtr == nullptr) {
        // 实测（2026-10-09）：APT 链路下 aclnn executor attrs 未能传入 tiling（GetFloat 取空），
        // 静默落默认会掩盖"调用方传了非默认 eps 却不生效"；此处告警保留可观测性（行为不变）。
        OP_LOGW(context_->GetNodeName(), "norm_eps attr missing in tiling attrs, fallback to default %f.",
                MHC_DEFAULT_NORM_EPS);
    }
    normEps_ = (normEpsPtr != nullptr ? *normEpsPtr : MHC_DEFAULT_NORM_EPS);
    OP_CHECK_IF(!std::isfinite(normEps_) || normEps_ <= 0.0f,
        OP_LOGE(context_->GetNodeName(), "norm_eps(%f) must be finite and greater than 0.", normEps_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// --------------------------CheckXValid-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::CheckXValid()
{
    const gert::StorageShape *xShapePtr = context_->GetInputShape(MHC_X_INDEX);
    OP_CHECK_IF(
        xShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "x InputShape is nullptr."), return ge::GRAPH_FAILED);

    size_t xDimNum = xShapePtr->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != MHC_DIM_NUM_3 && xDimNum != MHC_DIM_NUM_4,
        OP_LOGE(context_->GetNodeName(), "x: number of dimensions must be 3 (TND) or 4 (BSND)."),
        return ge::GRAPH_FAILED);

    if (xDimNum == MHC_DIM_NUM_4) {
        // BSND: [B, S, N, D]
        int64_t batchSize = xShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO);
        int64_t seqLen = xShapePtr->GetStorageShape().GetDim(MHC_DIM_ONE);
        n_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_TWO);
        d_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_THREE);
        totalLength_ = batchSize * seqLen;
    } else {
        // TND: [T, N, D]
        totalLength_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO);
        n_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_ONE);
        d_ = xShapePtr->GetStorageShape().GetDim(MHC_DIM_TWO);
    }

    bool isValid = true;
    isValid = isValid && (totalLength_ >= MHC_MIN_T);
    isValid = isValid && (n_ == MHC_N_VALUE);
    isValid = isValid && (d_ == MHC_D_2560 || d_ == MHC_D_5120 || d_ == MHC_D_6144 || d_ == MHC_D_7168);
    OP_CHECK_IF(!isValid,
        OP_LOGE(context_->GetNodeName(),
            "x shape is invalid: T=%ld, N=%ld, D=%ld (N must be 4; D must be one of "
            "2560/5120/6144/7168).",
            totalLength_,
            n_,
            d_),
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
    OP_CHECK_IF(phiShapePtr->GetStorageShape().GetDimNum() != MHC_DIM_NUM_2,
        OP_LOGE(context_->GetNodeName(), "phi: number of dimensions must be 2."),
        return ge::GRAPH_FAILED);

    // phi shape 两种形态：[N^2+N, N*D]（完整，输出 h_res）/ [N, N*D]（仅 post，无 h_res）
    int64_t phiDim0 = phiShapePtr->GetStorageShape().GetDim(MHC_DIM_ZERO);
    int64_t phiDim1 = phiShapePtr->GetStorageShape().GetDim(MHC_DIM_ONE);
    phiRows_ = phiDim0;
    hasRes_ = (phiDim0 == n_ * n_ + n_);

    bool isValid = true;
    isValid = isValid && (phiDim0 == n_ * n_ + n_ || phiDim0 == n_);
    isValid = isValid && (phiDim1 == n_ * d_);
    OP_CHECK_IF(!isValid,
        OP_LOGE(context_->GetNodeName(),
            "phi shape invalid: expected [%ld, %ld] or [%ld, %ld], got [%ld, %ld].",
            n_ * n_ + n_,
            n_ * d_,
            n_,
            n_ * d_,
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
    OP_CHECK_IF(alphaShapePtr->GetStorageShape().GetDimNum() != MHC_DIM_NUM_1,
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
    OP_CHECK_IF(biasShapePtr->GetStorageShape().GetDimNum() != MHC_DIM_NUM_1,
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

ge::graphStatus MhcPreSplitPostResInfoParser::CheckUbBudget()
{
    const int64_t xElemBytes = 2;
    const int64_t fp32Bytes = 4;
    const int64_t alignedPhiRows = ((phiRows_ + MIN_FP32_ALIGN_ELEM_SIZE - 1) / MIN_FP32_ALIGN_ELEM_SIZE) *
                                   MIN_FP32_ALIGN_ELEM_SIZE;
    const int64_t xTileElemsUb = n_ * dimTile_;
    const int64_t alignedN =
        ((n_ + MIN_FP32_ALIGN_ELEM_SIZE - 1) / MIN_FP32_ALIGN_ELEM_SIZE) * MIN_FP32_ALIGN_ELEM_SIZE;
    const int64_t alignedNN =
        ((n_ * n_ + MIN_FP32_ALIGN_ELEM_SIZE - 1) / MIN_FP32_ALIGN_ELEM_SIZE) * MIN_FP32_ALIGN_ELEM_SIZE;
    // fixedUbBytes 逐项镜像 kernel InitCommonBuffers 固定分配（arch35/ascend950 形态，本算子仅注册 950）：
    // xHatBuf_/biasBuf_（AlignUp(phiRows)+8）、dotBuf_/sumSqDotBuf_（仅 8 结果槽，无 work 段）、
    // sumSqBuf_（8）、alphaBuf_（8）、hPostOutQueue_/hResOutQueue_（AlignUp(N)/AlignUp(N*N)）
    const int64_t fixedUbBytes = MHC_NUM_TWO * (alignedPhiRows + MIN_FP32_ALIGN_ELEM_SIZE) * fp32Bytes +
                                 MHC_NUM_TWO * MIN_FP32_ALIGN_ELEM_SIZE * fp32Bytes +
                                 MHC_NUM_TWO * MIN_FP32_ALIGN_ELEM_SIZE * fp32Bytes +
                                 (alignedN + alignedNN) * fp32Bytes;
    // 线性增长 buffer（对应 kernel InitCommonBuffers）：xInQueue_(2B/元素，SINGLE depth1/MULTI depth2)、
    // xFp32Buf_(4B，arch35 全量 n*d 按 tile 拼接持久化——第二轮遍历免除 x 的 GM 重读)、
    // phiRowQue_(4B，depth2)
    const bool isSingleTile = (dimLoop_ == MHC_NUM_ONE);
    const int64_t xInQueueBytes = xTileElemsUb * xElemBytes * (isSingleTile ? MHC_NUM_ONE : MHC_NUM_TWO);
    const int64_t xFp32BufBytes = n_ * d_ * fp32Bytes;
    const int64_t phiRowQueBytes = xTileElemsUb * fp32Bytes * MHC_NUM_TWO;
    const int64_t ubUsageBytes = xInQueueBytes + xFp32BufBytes + phiRowQueBytes + fixedUbBytes;
    OP_CHECK_IF(ubUsageBytes > static_cast<int64_t>(ubSize_),
        OP_LOGE(opName_,
            "UB usage %ld exceeds ubSize %lu (dimTile=%ld, dimLoop=%ld), check tiling/UB budget.",
            ubUsageBytes, ubSize_, dimTile_, dimLoop_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSplitPostResInfoParser::GenerateTiling()
{
    // D轴分块：dimTile = min(D, 2560)。D 已由 CheckXValid 保证白名单取值；
    // 尾块不足 dimTile 时由 kernel 的 TileCopyD 按 D - dimOff 计算
    dimTile_ = d_ > MHC_DIM_TILE_MAX ? MHC_DIM_TILE_MAX : d_;

    // D轴需要几个tile
    dimLoop_ = (d_ + dimTile_ - MHC_NUM_ONE) / dimTile_;

    // 每token使用核数：token 数不足以填满一半核数（T*2 <= coreNum，含 T==coreNum/2 边界——
    // 该点双核恰可放满全部核，单核则半数核空闲）时启用双核 phi 行并行
    // （SINGLE/MULTI 路径统一，D>2560 场景同样生效，对齐设计 2.7：小 T 时把 phi 读负载摊到 2*T 核）
    // token 充足（T >= coreNum）时单核即可用满所有核，双核只引入额外开销（T∈[coreNum/2, coreNum)
    // 区间为单核过渡区，仅 T 个核活跃）。
    // 仅 post 形态（phiRows==N，无 h_res）恒单核：双核 phi 行对半（phiRowMid=N/2）会把
    // h_post 的 N 行劈到两个核，而 h_post 段仅 core0 写出（xHat 各核只算自己的半行），
    // 且该形态 phi 仅 N 行搬运量小、单核足够。
    bool needDualCore = hasRes_ && (totalLength_ * MHC_NUM_TWO <= static_cast<int64_t>(coreNum_));
    coresPerToken_ = needDualCore ? MHC_NUM_TWO : MHC_NUM_ONE;

    // phi行分配：双核模式下，phiRowMid是前半部分的行数
    // 前半行[0, phiRowMid) 对应 h_post(N行) + comb前半(N*N/2行)
    // 实际上双核按 phiRows/2 对半分
    phiRowMid_ = (coresPerToken_ == MHC_NUM_TWO) ? (phiRows_ / MHC_NUM_TWO) : phiRows_;

    // tilingKey
    // schMode 经 GET_TPL_TILING_KEY 编码进 TilingKey（单参数 1bit，编码值即 0/1），
    // kernel 侧按 arch35/mhc_pre_split_post_res_tiling_key.h 的 SEL 表实例化模板入口
    tilingKey_ = (dimLoop_ == MHC_NUM_ONE) ? GET_TPL_TILING_KEY(MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE)
                                           : GET_TPL_TILING_KEY(MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE);

    // h_res 为可选输出（对齐 mhc_pre tiling 的 hasResi 校验）：完整形态（phi 行数 N^2+N）
    // 必须提供 h_res 输出 desc；仅 post 形态（hasRes_==false）不校验。
    if (hasRes_) {
        OP_CHECK_IF(context_->GetOutputDesc(MHC_H_RES_INDEX) == nullptr,
            OP_LOGE(opName_, "h_res output is required when the number of phi rows is N^2+N (%ld).", phiRows_),
            return ge::GRAPH_FAILED);
    }

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

    const size_t wsBytes = 32;
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr,
        OP_LOGE(opName_, "currentWorkspace is nullptr"),
        return ge::GRAPH_FAILED);
    currentWorkspace[0] = wsBytes;
    OP_LOGD(context_->GetNodeName(), "wsBytes: %zu (blockDim=%ld, N=%ld, D=%ld)", wsBytes, blockDim_, n_, d_);

    // invK = 1/(N*D)，RmsNorm 均值系数；host 侧浮点运算，无 aicore 类型限制
    invK_ = 1.0f / static_cast<float>(n_ * d_);
    if (CheckUbBudget() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// --------------------------ParseAndCheck-------------------------------------
ge::graphStatus MhcPreSplitPostResInfoParser::ParseAndCheck()
{
    if (GetOpName() != ge::GRAPH_SUCCESS || GetNpuInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAttrInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckXValid() != ge::GRAPH_SUCCESS || CheckPhiValid() != ge::GRAPH_SUCCESS ||
        CheckAlphaValid() != ge::GRAPH_SUCCESS || CheckBiasValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GenerateTiling() != ge::GRAPH_SUCCESS) {
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
    tilingData_.set_n(n_);
    tilingData_.set_d(d_);
    tilingData_.set_phiRows(phiRows_);
    tilingData_.set_dimTile(dimTile_);
    tilingData_.set_dimLoop(dimLoop_);
    tilingData_.set_coresPerToken(coresPerToken_);
    tilingData_.set_phiRowMid(phiRowMid_);
    tilingData_.set_baseT(baseT_);
    tilingData_.set_tailT(tailT_);
    tilingData_.set_coreTAct(coreTAct_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_invK(invK_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(blockDim_);

    OP_LOGD(context_->GetNodeName(), "totalLength: %ld", totalLength_);
    OP_LOGD(context_->GetNodeName(), "N: %ld", n_);
    OP_LOGD(context_->GetNodeName(), "D: %ld", d_);
    OP_LOGD(context_->GetNodeName(), "phiRows: %ld", phiRows_);
    OP_LOGD(context_->GetNodeName(), "dimTile: %ld", dimTile_);
    OP_LOGD(context_->GetNodeName(), "dimLoop: %ld", dimLoop_);
    OP_LOGD(context_->GetNodeName(), "coresPerToken: %ld", coresPerToken_);
    OP_LOGD(context_->GetNodeName(), "phiRowMid: %ld", phiRowMid_);
    OP_LOGD(context_->GetNodeName(), "baseT: %ld", baseT_);
    OP_LOGD(context_->GetNodeName(), "tailT: %ld", tailT_);
    OP_LOGD(context_->GetNodeName(), "coreTAct: %ld", coreTAct_);
    OP_LOGD(context_->GetNodeName(), "normEps: %f", normEps_);
    OP_LOGD(context_->GetNodeName(), "invK: %f", invK_);
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
