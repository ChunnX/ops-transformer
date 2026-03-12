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
 * \file mainifold_constrained_hyper_connection_tiling.cpp
 * \brief
 */

#include "mhc_pre_backward_tiling.h"
#include "../../../common/include/tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../common/include/err/ops_err.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;
const constexpr int64_t BSD_DIM_NUM = 3;
const constexpr int64_t BSNN_DIM_NUM = 4;
const constexpr int64_t TD_DIM_NUM = 2;
const constexpr int64_t TN_DIM_NUM = 2;
const constexpr int64_t TNN_DIM_NUM = 3;
const constexpr uint32_t H_IN_GRAD_INDEX = 3;
const constexpr uint32_t H_POST_GRAD_INDEX = 4;
const constexpr uint32_t H_COMB_BEFORE_GRAD_INDEX = 5;

const constexpr int64_t INDEX_B = 0;
const constexpr int64_t INDEX_S = 1;
const constexpr int64_t INDEX_D = 2;

const constexpr int64_t INDEX_T = 0;
const constexpr int64_t INDEX_N = 1;
const constexpr int64_t INDEX_D_TND = 1;

const constexpr int32_t C0_BASE_M = 256;
const constexpr int32_t C0_BASE_N = 128;
const constexpr int32_t C0_BASE_K = 32;
const constexpr uint32_t MAX_D_LENTH = 8192;
const constexpr uint32_t D_ALIGN = 64;

REGISTER_TILING_TEMPLATE("MhcPreBackward", MhcPreBackwardBaseTiling, 1000);

ge::graphStatus MhcPreBackwardBaseTiling::GetInputShape()
{
    auto hInGradTensor = context_->GetDynamicInputTensor(H_IN_GRAD_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hInGradTensor);
    auto hPostGradTensor = context_->GetDynamicInputTensor(H_POST_GRAD_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hPostGradTensor);
    auto hCombBeforeGradTensor = context_->GetDynamicInputTensor(H_COMB_BEFORE_GRAD_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hCombBeforeGradTensor);

    auto hInDims = hInGradTensor->GetStorageShape().GetDimNum();
    auto hPostDims = hPostGradTensor->GetStorageShape().GetDimNum();
    auto hCombDims = hCombBeforeGradTensor->GetStorageShape().GetDimNum();
    
    if ((hInDims != BSD_DIM_NUM && hInDims != TD_DIM_NUM) ||
        (hPostDims != BSD_DIM_NUM && hPostDims != TN_DIM_NUM) ||
        (hCombDims != BSNN_DIM_NUM && hCombDims != TNN_DIM_NUM)) {
        OP_LOGE(context_->GetNodeName(), "input dims invalid for MhcPreBackward");
        return ge::GRAPH_FAILED;
    }

    // 检查输入维度是否一致
    if (hInDims != hPostDims) {
        OP_LOGE(context_->GetNodeName(), "h_in_grad and h_post_grad must have the same dim num");
        return ge::GRAPH_FAILED;
    }

    if (hInDims == BSD_DIM_NUM) {
        // BSND格式: h_in_grad [B,S,D], h_post_grad [B,S,N], h_comb_before_grad [B,S,N,N]
        uint64_t batch = hInGradTensor->GetStorageShape().GetDim(INDEX_B);
        uint64_t sequence = hInGradTensor->GetStorageShape().GetDim(INDEX_S);
        D_ = hInGradTensor->GetStorageShape().GetDim(INDEX_D);
        N_ = hPostGradTensor->GetStorageShape().GetDim(INDEX_D);
        
        if (hPostGradTensor->GetStorageShape().GetDim(INDEX_B) != batch ||
            hPostGradTensor->GetStorageShape().GetDim(INDEX_S) != sequence) {
            OP_LOGE(context_->GetNodeName(), "h_post_grad shape must align with h_in_grad on B and S dims");
            return ge::GRAPH_FAILED;
        }
        if (hCombBeforeGradTensor->GetStorageShape().GetDim(0) != batch ||
            hCombBeforeGradTensor->GetStorageShape().GetDim(1) != sequence ||
            hCombBeforeGradTensor->GetStorageShape().GetDim(2) != N_ ||
            hCombBeforeGradTensor->GetStorageShape().GetDim(3) != N_) {
            OP_LOGE(context_->GetNodeName(), "h_comb_before_grad shape must be [B, S, N, N]");
            return ge::GRAPH_FAILED;
        }
        totalLength_ = batch * sequence;
    } else if (hInDims == TD_DIM_NUM) {
        // TND格式: h_in_grad [T,D], h_post_grad [T,N], h_comb_before_grad [T,N,N]
        uint64_t t = hInGradTensor->GetStorageShape().GetDim(INDEX_T);
        D_ = hInGradTensor->GetStorageShape().GetDim(INDEX_D_TND);
        N_ = hPostGradTensor->GetStorageShape().GetDim(INDEX_N);
        
        if (hPostGradTensor->GetStorageShape().GetDim(INDEX_T) != t) {
            OP_LOGE(context_->GetNodeName(), "h_post_grad shape must align with h_in_grad on T dim");
            return ge::GRAPH_FAILED;
        }
        if (hCombBeforeGradTensor->GetStorageShape().GetDim(0) != t ||
            hCombBeforeGradTensor->GetStorageShape().GetDim(1) != N_ ||
            hCombBeforeGradTensor->GetStorageShape().GetDim(2) != N_) {
            OP_LOGE(context_->GetNodeName(), "h_comb_before_grad shape must be [T, N, N]");
            return ge::GRAPH_FAILED;
        }
        totalLength_ = t;
    }

    static const std::vector<uint64_t> legalN = {4, 6, 8};
    if (std::find(legalN.begin(), legalN.end(), N_) == legalN.end()) {
        OP_LOGE(context_->GetNodeName(), "Invalid input shape N=%lu. Expected one of {4,6,8}", N_);
        return ge::GRAPH_FAILED;
    }

    if (D_ > MAX_D_LENTH || D_ % D_ALIGN != 0) {
        OP_LOGE(context_->GetNodeName(), "Invalid input shape D=%lu. Expected to be less than or equal to 8192 and 64-element alignment", D_);
        return ge::GRAPH_FAILED;
    }

    matN_ = N_ * D_;
    matM_ = totalLength_;
    matK_ = (2 * N_) + (N_ * N_);
    fusionSize_ = (2 * N_) + (N_ * N_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardBaseTiling::ParseInputAndAttr()
{
    uint64_t ubSize, l1Size, l0CSize;

    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "get input shape failed");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);

    blockDim_ = ascendcPlatform.GetCoreNumAic();
    vecCoreNum_ = ascendcPlatform.GetCoreNumAiv();

    auto attrs = context_->GetAttrs();
    auto hcEpsPtr_ = attrs->GetAttrPointer<float>(0);
    if (hcEpsPtr_ != nullptr) {
        hcEps_ = *hcEpsPtr_;
    } else {
        hcEps_ = 1e-6;
    }

    return ge::GRAPH_SUCCESS;
}


void MhcPreBackwardBaseTiling::FillTilingData()
{
    // 矩阵计算剩余部分
    // ProcessC0: M = B*S, N = n*D, K = 2n+n*n
    tilingData_.matmulTilingC0.set_dbL0C(1);
    tilingData_.matmulTilingC0.set_stepKa(1);
    tilingData_.matmulTilingC0.set_stepKb(1);
    tilingData_.matmulTilingC0.set_depthA1(1);
    tilingData_.matmulTilingC0.set_depthB1(1);
    tilingData_.matmulTilingC0.set_stepM(1);
    tilingData_.matmulTilingC0.set_stepN(1);
    tilingData_.matmulTilingC0.set_baseM(C0_BASE_M);
    tilingData_.matmulTilingC0.set_baseN(C0_BASE_N);
    tilingData_.matmulTilingC0.set_baseK(C0_BASE_K);

    // ProcessC1: M = 2n+n*n, N = n*D, K = B*S
    tilingData_.matmulTilingC1.set_dbL0C(1);
    tilingData_.matmulTilingC1.set_stepKa(1);
    tilingData_.matmulTilingC1.set_stepKb(1);
    tilingData_.matmulTilingC1.set_depthA1(1);
    tilingData_.matmulTilingC1.set_depthB1(1);
    tilingData_.matmulTilingC1.set_stepM(1);
    tilingData_.matmulTilingC1.set_stepN(1);

    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_vecCoreNum(vecCoreNum_);
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_nD(N_ * D_);
    tilingData_.set_fusionSize(fusionSize_);
    tilingData_.set_N(N_);
    tilingData_.set_D(D_);
    tilingData_.set_hcEps(hcEps_);
}

uint64_t MhcPreBackwardBaseTiling::CalculateWorkspaceSize(
    uint64_t totalLength, uint64_t fusionSize, uint64_t coreNum, uint64_t elementSize)
{
    uint64_t v1Elements = totalLength * fusionSize +           // h_mix_grad
                             3 * coreNum +                        // alpha_grad
                             coreNum * fusionSize +               // bias_grad
                             totalLength;                          // inv_rms_grad


    uint32_t v2Elements = 1024 * 128 * 24 * 2 +                 // x_rs_grad_mm
                            1024 * 128 * 24 * 2 +               // x_rs
                            2 * 1024 * 1024;
    uint64_t totalElements = ((v1Elements + 32 - 1) / 32 * 32) + v2Elements;
    return totalElements * elementSize;
}

ge::graphStatus MhcPreBackwardBaseTiling::TilingProcess()
{
    // 计算workspace大小（float32，每个元素4字节）
    size_t userWorkspaceSize = CalculateWorkspaceSize(totalLength_, matN_, blockDim_, sizeof(float));
    
    size_t systemWorkspaceSize = 40 * 1024 * 1024; // 10M

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    matmul_tiling::MatmulApiTiling mm0_(ascendcPlatform);
    matmul_tiling::MatmulApiTiling mm1_(ascendcPlatform);

    // C0
    mm0_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm0_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm0_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm0_.SetBias(false);
    mm0_.SetShape(1024, 128, N_ * N_ + 2 * N_);                     // [sM(=BSBlockSize), sN(=NDBlockSize), sK(=N*N + 2*N)]
    mm0_.SetOrgShape(totalLength_, N_ * D_, N_ * N_ + 2 * N_);      // 原始[M(=BS), N(=nD), K(=N*N + 2*N)]
    mm0_.SetBufferSpace(-1, -1, -1);
    if (mm0_.GetTiling(tilingData_.matmulTilingC0) == -1) {
        OP_LOGE(context_->GetNodeName(), "ProcessC0 Get Tiling Failed!, m = %lu, n = %lu, k = %lu", totalLength_, N_ * D_, N_ * N_ + 2 * N_);
        return ge::GRAPH_FAILED;
    }

    // C1
    mm1_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm1_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm1_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm1_.SetBias(false);
    mm1_.SetShape(N_ * N_ + 2 * N_, 128, 1024);                 // [sM(=N*N + 2*N), sN(=NDBlockSize), sK(=BSBlockSize)]
    mm1_.SetOrgShape(N_ * N_ + 2 * N_, N_* D_, totalLength_);   // 原始[M(=N*N + 2*N), N(=nD), K(=BS)]
    mm1_.SetBufferSpace(-1, -1, -1);
    if (mm1_.GetTiling(tilingData_.matmulTilingC1) == -1) {
        OP_LOGE(context_->GetNodeName(), "ProcessC1 Get Tiling Failed!, m = %lu, n = %lu, k = %lu", N_ * N_ + 2 * N_, N_ * D_, totalLength_);
        return ge::GRAPH_FAILED;
    }
    
    // 后续模板按位处理
    tilingKey_ = 0UL;

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MhcPreBackwardBaseTiling::DoOpTiling()
{
    auto inputHInGradDesc = context_->GetInputDesc(0);
    if (inputHInGradDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer: h_in_grad");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void MhcPreBackwardBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "blockDim: [%d]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "totalLength: [%d]", tilingData_.get_totalLength());
    OP_LOGD(context_->GetNodeName(), "nD: [%d]", tilingData_.get_nD());
    OP_LOGD(context_->GetNodeName(), "matN: [%d]", tilingData_.get_fusionSize());
    OP_LOGD(context_->GetNodeName(), "hcEps: [%d]", tilingData_.get_hcEps());
}

uint64_t MhcPreBackwardBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MhcPreBackwardBaseTiling::PostTiling()
{
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(1);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
        return ge::GRAPH_FAILED);

    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc4mHCPreGrad(gert::TilingContext* context)
{
     OP_CHECK_IF(context == nullptr,
        OPS_REPORT_CUBE_INNER_ERR("[mHCPreBackwardTilingFunc]", " context is null"),
        return ge::GRAPH_FAILED);

    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}


static ge::graphStatus TilingPrepare4mHCPreGrad(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("[TilingPrepare4mHC]", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MhcPreBackwardCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);

    OP_LOGI(context->GetNodeName(),
            "parse compile info success l1Size:%lu, l2Size:%lu, coreNum:%lu",
            compileInfoPtr->l1Size,
            compileInfoPtr->l2Size,
            compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPreBackward)
    .Tiling(TilingFunc4mHCPreGrad)
    .TilingParse<MhcPreBackwardCompileInfo>(TilingPrepare4mHCPreGrad);
}
