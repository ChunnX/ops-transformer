/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mhc_pre_sinkhorn_backward_arch35_tiling_base.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.h"
#include "log/log.h"

using namespace AscendC;
namespace optiling {

constexpr uint8_t GRAD_HIN_IDX = 0;
constexpr uint8_t GRAD_H_POST_IDX = 1;
constexpr uint8_t GRAD_H_RES_IDX = 2;
constexpr uint8_t INPUT_X_IDX = 3;
constexpr uint8_t PHI_IDX = 4;
constexpr uint8_t ALPHA_IDX = 5;
constexpr uint8_t BIAS_IDX = 6;
constexpr uint8_t H_PRE_IDX = 7;
constexpr uint8_t HC_BEFORE_NORM_IDX = 8;
constexpr uint8_t INV_RMS_IDX = 9;
constexpr uint8_t SUM_OUT_IDX = 10;
constexpr uint8_t NORM_OUT_IDX = 11;
constexpr uint8_t GRAD_X_IDX = 0;
constexpr uint8_t GRAD_PHI_IDX = 1;
constexpr uint8_t GRAD_ALPHA_IDX = 2;
constexpr uint8_t GRAD_BIAS_IDX = 3;
constexpr uint8_t BATCH_SIZE_DIM_IDX = 0;
constexpr uint8_t SEQ_LENGTH_DIM_IDX = 1;
constexpr uint8_t N_DIM_IDX = 2;
constexpr uint8_t C_DIM_IDX = 3;
constexpr float DEFAULT_EPS = 1e-6f;
constexpr uint8_t ITER_COUNT_IDX = 0;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint8_t BF16_BYTE_SIZE = 2;
constexpr int64_t FP32_BYTE_SIZE = 4;
constexpr int64_t ALPHA_SIZE_3 = 3;
constexpr int64_t N_SIZE_4 = 4;
constexpr int64_t ALIGN_32 = 32;

template<typename T>
constexpr T AlignUp(T value, T align) {
    return (value + align - 1) / align * align;
}

using namespace ge;
using namespace std;
using namespace AscendC;

bool MhcPreSinkhornBackwardArch35DeterminiticTiling::IsCapable()
{
    auto isDeterministic = (context_->GetDeterministic() == 1);
    if (!isDeterministic) {
        return false;
    }
    
    return true;
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::SetTilingData()
{
    MhcPreSinkhornBackwardArch35DeterminiticTilingData* tilingData =
    context_->GetTilingData<MhcPreSinkhornBackwardArch35DeterminiticTilingData>();
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);

    auto floatDataType = matmul_tiling::DataType::DT_FLOAT;
    auto bf16DataType = matmul_tiling::DataType::DT_BF16;

    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling mm2Tiling(ascendPlatformInfo);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bf16DataType);
    mm1Tiling.SetOrgShape(mm1M_, mm1N_, mm1K_);
    mm1Tiling.SetShape(mm1M_, mm1N_, mm1K_);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    mm1Tiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN);
    if (mm1Tiling.GetTiling(tilingData->mm1TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    mm2Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType, true);
    mm2Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm2Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm2Tiling.SetOrgShape(mm2M_, mm2N_, mm2K_);
    mm2Tiling.SetShape(mm2M_, mm2N_, mm2K_);
    mm2Tiling.SetBias(false);
    mm2Tiling.SetBufferSpace(-1, -1, -1);
    mm2Tiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN);
    if (mm2Tiling.GetTiling(tilingData->mm2TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    tilingData->batchSize = batchSize_;
    tilingData->seqLength = seqLength_;
    tilingData->c = c_;
    tilingData->n = n_;
    tilingData->aivNum = coreNumAiv_;
    tilingData->skIterCount = skIterCount_;
    tilingData->eps = hcEps_;

    return ge::GRAPH_SUCCESS;  
}

void MhcPreSinkhornBackwardArch35DeterminiticTiling::DoUbTiling()
{
    int64_t totalBs = batchSize_ * seqLength_;
    bsTaskCount_ = totalBs / coreNumAiv_;
    tailBsTaskCount_ = totalBs - bsTaskCount_ * coreNumAiv_;
    cLoopDataLen_ = c_;

    int64_t nAlignFp32 = AlignUp(n_ * FP32_BYTE_SIZE, ALIGN_32) / FP32_BYTE_SIZE;
    int64_t cAlignFp32 = AlignUp(c_ * FP32_BYTE_SIZE, ALIGN_32) / FP32_BYTE_SIZE;
    int64_t cAlignFp16 = AlignUp(c_ * BF16_BYTE_SIZE, ALIGN_32) / BF16_BYTE_SIZE;
    
    int64_t hPreBuffer = nAlignFp32 * FP32_BYTE_SIZE;
    int64_t xBuffer = cAlignFp16 * n_ * BF16_BYTE_SIZE;
    int64_t gradHinBuffer = cAlignFp16 * BF16_BYTE_SIZE;
    int64_t xFp32Buffer = cAlignFp32 * n_ * FP32_BYTE_SIZE;
    int64_t gradXFromHinBuffer = cAlignFp32 * n_ * FP32_BYTE_SIZE;
    int64_t gradHPreBuffer = nAlignFp32 * FP32_BYTE_SIZE;
    
    int64_t totalUbPerBs = (hPreBuffer + xBuffer + gradHinBuffer + xFp32Buffer + gradXFromHinBuffer + gradHPreBuffer) * DOUBLE_BUFFER;
    int64_t maxBsLoopDataLen = static_cast<int64_t>(ubSize_ * 0.9 / totalUbPerBs);
    
    bsLoopDataLen_ = std::min(maxBsLoopDataLen, bsTaskCount_);
    bsLoopDataLen_ = std::max(bsLoopDataLen_, static_cast<int64_t>(1));
    
    cLoopDataLen_ = c_;
    tile_ = bsLoopDataLen_;
    
    OP_LOGI(opName, "DoUbTiling: totalBs=%lld, bsTaskCount=%lld, tailBsTaskCount=%lld, bsLoopDataLen=%lld, cLoopDataLen=%lld, nAlign=%lld, cAlign=%lld",
        totalBs, bsTaskCount_, tailBsTaskCount_, bsLoopDataLen_, cLoopDataLen_, nAlignFp32, cAlignFp32);
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::DoOpTiling()
{
    DoUbTiling();
    return SetTilingData();
}

uint64_t MhcPreSinkhornBackwardArch35DeterminiticTiling::GetTilingKey() const
{
    bool isDeterministic = true;
    return GET_TPL_TILING_KEY(isDeterministic);
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::PostTiling()
{
    context_->SetDynUBufSize(ubSize_);
    context_->SetBlockDim(coreNumAic_);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);
    size_t gradHat2Workspace = batchSize_ * seqLength_ * n_ * c_ * 10 * sizeof(float);
    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = gradHat2Workspace;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(opName, "fail to GetWorkspaceSizes"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35DeterminiticTiling::DumpTilingInfo()
{

    mm1K_ = n_ * n_ + 2 * n_;
    mm1M_ = tile_ * 2;
    mm1N_ = n_ * c_;

    mm2K_ = tile_ * 2;
    mm2M_ = n_ * n_ + 2 * n_;
    mm2N_ = n_ * c_;
    std::ostringstream info;
    info << "batchSize: " << batchSize_ << std::endl;
    info << "seqLength: " << seqLength_ << std::endl;
    info << "c: " << c_ << std::endl;
    info << "n: " << n_ << std::endl;
    info << "c0: " << c0_ << std::endl;
    info << "c1: " << c1_ << std::endl;
    info << "aivNum: " << coreNumAiv_ << std::endl;
    info << "skIterCount: " << skIterCount_ << std::endl;
    info << "tileSize: " << tile_ << std::endl;
    info << "mm1K: " << mm1K_ << std::endl;
    info << "mm1M: " << mm1M_ << std::endl;
    info << "mm1N: " << mm1N_ << std::endl;
    info << "mm2K: " << mm2K_ << std::endl;
    info << "mm2M: " << mm2M_ << std::endl;
    info << "mm2N: " << mm2N_ << std::endl;

    OP_LOGI(opName, "Tiling info is: %s", info.str().c_str());    
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MhcPreSinkhornBackward, MhcPreSinkhornBackwardArch35DeterminiticTiling, 0);
}