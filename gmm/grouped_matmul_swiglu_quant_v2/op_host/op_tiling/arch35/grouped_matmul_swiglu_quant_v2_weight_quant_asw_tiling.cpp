/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/platform_infos_def.h"
#include "op_host/tiling_templates_registry.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_asw/grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling_key.h"
#include "../../../../grouped_matmul/op_host/grouped_matmul_host_util.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmulSwigluQuantV2Tiling;

namespace {
constexpr uint64_t KB_UNIT = 1024;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t STEP_K_4 = 4;
} // namespace

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantASWTiling::GetShapeAttrsInfo()
{
    const auto *compileInfoPtr = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "CompileInfo is null."), return ge::GRAPH_FAILED);
    compileInfoPtr_ = compileInfoPtr;
    OP_LOGI(context_, "Compile info: npuArch(%u) aicNum(%u) aivNum(%u) ubSize(%lu) l0cSize(%lu).",
            static_cast<uint32_t>(compileInfoPtr_->npuArch_), compileInfoPtr_->aicNum_, compileInfoPtr_->aivNum_,
            compileInfoPtr_->ubSize_, compileInfoPtr_->l0cSize_);
    const NpuArch npuArch = static_cast<NpuArch>(compileInfoPtr_->npuArch_);
    return ValidateWeightQuantAsw(*context_, inputParams_, npuArch, sizeof(tilingData_));
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantASWTiling::GetPlatformInfo()
{
    // GetShapeAttrsInfo在GetPlatformInfo之前调用，但需要用到npuarch，compileInfoPtr_获取提到GetShapeAttrsInfo中
    // 此处仅保留AIC/AIV核数校验
    const NpuArch npuArch = static_cast<NpuArch>(compileInfoPtr_->npuArch_);
    OP_CHECK_IF(!CheckWeightQuantAswCoreNum(npuArch, compileInfoPtr_->aicNum_, compileInfoPtr_->aivNum_),
                OP_LOGE(context_, "Invalid core number ratio."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulSwigluQuantV2WeightQuantASWTiling::IsCapable()
{
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantASWTiling::DoOpTiling()
{
    tilingData_.gmmWeightQuantAswParam.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmWeightQuantAswParam.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmWeightQuantAswParam.l0cSize = static_cast<uint32_t>(compileInfoPtr_->l0cSize_ / KB_UNIT);

    OP_LOGD(context_, "DoOpTiling: groupNum=%u, groupListType=%u, l0cSize=%u.",
            tilingData_.gmmWeightQuantAswParam.groupNum,
            static_cast<uint32_t>(tilingData_.gmmWeightQuantAswParam.groupListType),
            tilingData_.gmmWeightQuantAswParam.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantASWTiling::DoLibApiTiling()
{
    tilingData_.mmTilingData.M = static_cast<uint32_t>(inputParams_.mSize);
    tilingData_.mmTilingData.N = static_cast<uint32_t>(inputParams_.nSize);
    tilingData_.mmTilingData.Ka = static_cast<uint32_t>(inputParams_.kSize);
    tilingData_.mmTilingData.Kb = static_cast<uint32_t>(inputParams_.kSize);
    tilingData_.mmTilingData.usedCoreNum = compileInfoPtr_->aicNum_;
    tilingData_.mmTilingData.baseM = BASIC_BLOCK_SIZE_256;
    tilingData_.mmTilingData.baseN = BASIC_BLOCK_SIZE_256;
    tilingData_.mmTilingData.baseK = BASIC_BLOCK_SIZE_128;
    tilingData_.mmTilingData.stepKa = STEP_K_4;
    tilingData_.mmTilingData.stepKb = STEP_K_4;
    tilingData_.mmTilingData.isBias = 0;
    OP_LOGD(context_,
            "DoLibApiTiling: M=%d, N=%d, Ka=%d, Kb=%d, usedCoreNum=%d, baseM=%d, baseN=%d, baseK=%d, stepKa=%d, "
            "stepKb=%d, isBias=%d.",
            tilingData_.mmTilingData.M, tilingData_.mmTilingData.N, tilingData_.mmTilingData.Ka,
            tilingData_.mmTilingData.Kb, tilingData_.mmTilingData.usedCoreNum, tilingData_.mmTilingData.baseM,
            tilingData_.mmTilingData.baseN, tilingData_.mmTilingData.baseK, tilingData_.mmTilingData.stepKa,
            tilingData_.mmTilingData.stepKb, tilingData_.mmTilingData.isBias);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantASWTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_, "workspaces is nullptr."), return ge::GRAPH_FAILED);
    workspaces[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantASWTiling::PostTiling()
{
    context_->SetBlockDim(compileInfoPtr_->aicNum_);
    context_->SetScheduleMode(1);
    const auto *rawTiling = context_->GetRawTilingData();
    OP_CHECK_IF(rawTiling == nullptr, OP_LOGE(context_, "RawTilingData is nullptr."), return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(const_cast<void *>(rawTiling->GetData()), rawTiling->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_, "memcpy_s failed, ret = %d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    OP_LOGD(context_, "PostTiling: blockDim=%u, tilingSize=%zu.", compileInfoPtr_->aicNum_, sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2WeightQuantASWTiling::GetTilingKey() const
{
    constexpr uint64_t bTrans = GMM_SWIGLU_QUANT_TRANS;
    constexpr uint64_t aTrans = GMM_SWIGLU_QUANT_NO_TRANS;
    return GET_TPL_TILING_KEY(bTrans, aTrans);
}

void GroupedMatmulSwigluQuantV2WeightQuantASWTiling::Reset()
{
    tilingData_ = GMMSQArch35Tiling::GMMSQWeightQuantAswTilingData();
    const auto *rawTiling = context_->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr) {
        return;
    }
    OP_CHECK_IF(memset_s(const_cast<void *>(rawTiling->GetData()), rawTiling->GetCapacity(), 0,
                         rawTiling->GetCapacity()) != EOK,
                OP_LOGE(context_, "Fail to clear tiling data."), return);
}
} // namespace optiling
