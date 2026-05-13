/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tri_inverse_tiling.cpp
 * \brief
 */

#include "tri_inverse_tiling.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(TriInverse, TriInverseTiling, 0);

namespace {
constexpr size_t CC_IN_INDEX = 0;
constexpr size_t CC_OUT_INDEX = 0;
constexpr size_t TRI_INVERSE_DIM_NUM = 3;
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr int64_t CHUNK_SIZE = 64;
constexpr int64_t STAGE_ONE_PARA_NUM = 4;
constexpr uint32_t MATMUL_BASE_M = 128;
constexpr uint32_t MATMUL_BASE_K = 128;
constexpr uint32_t MATMUL_BASE_N = 128;
} // namespace

void TriInverseTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    if (compileInfo_.aivNum == 0 || compileInfo_.aicNum == 0) {
        OP_LOGE(context_->GetNodeName(), "invalid core num, aiv=%lu, aic=%lu", compileInfo_.aivNum,
                compileInfo_.aicNum);
        return;
    }
    tilingData_.aiCoreNum = static_cast<int64_t>(compileInfo_.aicNum);
}

ge::graphStatus TriInverseTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::CheckContext() const
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(CC_IN_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(CC_IN_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(CC_OUT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(CC_OUT_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::AnalyzeDtype() const
{
    auto inputDtype = context_->GetInputDesc(CC_IN_INDEX)->GetDataType();
    auto outputDtype = context_->GetOutputDesc(CC_OUT_INDEX)->GetDataType();
    OP_CHECK_IF(inputDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "cc_in dtype must be bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "cc_out dtype must be bfloat16"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::AnalyzeShapes()
{
    const auto &inputShape = context_->GetInputShape(CC_IN_INDEX)->GetOriginShape();
    const auto &outputShape = context_->GetOutputShape(CC_OUT_INDEX)->GetOriginShape();
    OP_CHECK_IF(inputShape.GetDimNum() != TRI_INVERSE_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "cc_in must be 3D, but got %zuD", inputShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputShape.GetDimNum() != TRI_INVERSE_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "cc_out must be 3D, but got %zuD", outputShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputShape.GetDim(DIM_0) <= 0,
                OP_LOGE(context_->GetNodeName(), "tile_num must be positive, got %ld", inputShape.GetDim(DIM_0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputShape.GetDim(DIM_1) != CHUNK_SIZE || inputShape.GetDim(DIM_2) != CHUNK_SIZE,
                OP_LOGE(context_->GetNodeName(), "cc_in shape must be [tile_num, 64, 64]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputShape != inputShape, OP_LOGE(context_->GetNodeName(), "cc_out shape must equal cc_in shape"),
                return ge::GRAPH_FAILED);

    tilingData_.tileNum = inputShape.GetDim(DIM_0);
    return ge::GRAPH_SUCCESS;
}

bool TriInverseTiling::CheckFormat(const gert::CompileTimeTensorDesc *desc, const std::string &name) const
{
    auto primaryFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    OP_CHECK_IF(primaryFormat == ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(), "%s format %s is not supported", name.c_str(),
                        ge::TypeUtils::FormatToSerialString(primaryFormat).c_str()),
                return false);
    return true;
}

ge::graphStatus TriInverseTiling::AnalyzeFormat() const
{
    if (!CheckFormat(context_->GetInputDesc(CC_IN_INDEX), "cc_in") ||
        !CheckFormat(context_->GetOutputDesc(CC_OUT_INDEX), "cc_out")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "invalid context"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "invalid dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "invalid shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "invalid format"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::DoOpTiling()
{
    tilingData_.chunkSize = CHUNK_SIZE;
    tilingData_.stageOneParaNum = STAGE_ONE_PARA_NUM;
    workspaceSize_ = 0;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::DoMatmulTiling()
{
    matmul_tiling::MultiCoreMatmulTiling mm;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    mm.SetBufferSpace(l1Size, l0CSize, ubSize);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BFLOAT16, true);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BFLOAT16, true);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BFLOAT16);
    mm.SetBias(false);
    mm.SetDim(1);
    mm.SetShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
    mm.SetOrgShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
    mm.SetFixSplit(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
    if (mm.GetTiling(tilingData_.matmulTilingFp32) == -1) {
        OP_LOGE(context_->GetNodeName(), "TriInverse: Get matmul tiling failed");
        return ge::GRAPH_FAILED;
    }
    tilingData_.matmulTilingFp32.dbL0C = 1;
    tilingData_.matmulTilingFp32.stepKa = 1;
    tilingData_.matmulTilingFp32.stepKb = 1;
    tilingData_.matmulTilingFp32.depthA1 = 1;
    tilingData_.matmulTilingFp32.depthB1 = 1;
    tilingData_.matmulTilingFp32.stepM = 1;
    tilingData_.matmulTilingFp32.stepN = 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::DoLibApiTiling()
{
    tilingKey_ = 0;
    return DoMatmulTiling();
}

uint64_t TriInverseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus TriInverseTiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::SetScheduleConfig()
{
    constexpr uint32_t batchMode = 1U;
    auto ret = context_->SetScheduleMode(batchMode);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetScheduleMode failed, ret=%u", ret),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriInverseTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.aiCoreNum);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData->GetData());

    auto tilingDataSize = sizeof(TriInverse::TriInverseTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataSize,
                OP_LOGE(context_->GetNodeName(), "raw tiling data capacity %zu is smaller than %zu",
                        rawTilingData->GetCapacity(), tilingDataSize),
                return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    OP_CHECK_IF(SetScheduleConfig() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetScheduleMode failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void TriInverseTiling::PrintTilingData() const
{
    OP_LOGD(context_->GetNodeName(), "aiCoreNum: [%ld]", tilingData_.aiCoreNum);
    OP_LOGD(context_->GetNodeName(), "tileNum: [%ld]", tilingData_.tileNum);
    OP_LOGD(context_->GetNodeName(), "chunkSize: [%ld]", tilingData_.chunkSize);
    OP_LOGD(context_->GetNodeName(), "stageOneParaNum: [%ld]", tilingData_.stageOneParaNum);
}

static ge::graphStatus TriInverseTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("TriInverse", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForTriInverse(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("TriInverse", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<TriInverseCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TriInverse)
    .Tiling(TriInverseTilingFunc)
    .TilingParse<TriInverseCompileInfo>(TilingPrepareForTriInverse);
} // namespace optiling
