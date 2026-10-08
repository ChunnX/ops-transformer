/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "grouped_rotate_quant_tiling.h"
#include "../../grouped_rotate_quant_context.h"
#include "securec.h"

namespace optiling {
namespace {
void ReadHardware(platform_ascendc::PlatformAscendC &platform, GroupedRotateQuantCompileInfo &info)
{
    info.npuArch = static_cast<int32_t>(platform.GetCurNpuArch());
    auto &h = info.hardware;
    h.aicNum = platform.GetCoreNumAic();
    h.aivNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, h.l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, h.l0ASize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, h.l0BSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, h.l0CSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, h.ubSize);
}

bool ReadOutputs(gert::TilingContext *context, size_t irIndex, ge::DataType dtype,
                 std::vector<grq::Shape> &shapes)
{
    const auto *info = context->GetIrOutputInstanceInfo(irIndex);
    if (info == nullptr) {
        return false;
    }
    for (size_t i = 0; i < info->GetInstanceNum(); ++i) {
        const size_t index = info->GetInstanceStart() + i;
        const auto *shape = context->GetOutputShape(index);
        const auto *desc = context->GetOutputDesc(index);
        if (shape == nullptr || desc == nullptr || desc->GetDataType() != dtype ||
            desc->GetStorageFormat() != ge::FORMAT_ND || desc->GetOriginFormat() != ge::FORMAT_ND ||
            shape->GetStorageShape() != shape->GetOriginShape()) {
            return false;
        }
        shapes.push_back(grq::ReadShape(*shape));
    }
    return true;
}
} // namespace

ge::graphStatus GroupedRotateQuantTiling950::GetShapeAttrsInfo()
{
    inputParams_ = {};
    ge::DataType dstType;
    std::string error;
    if (!grq::ReadInputs(context_, inputParams_, dstType, error) ||
        !grq::ValidateInputs(inputParams_, false, error)) {
        OP_LOGE(context_, "%s", error.c_str());
        return ge::GRAPH_FAILED;
    }
    // Tensor API uses ND descriptor strides. Reject padded physical shapes rather
    // than silently using logical dimensions for GM strides.
    const auto *node = context_->GetComputeNodeInfo();
    if (node == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < node->GetInputsNum(); ++i) {
        const auto *shape = context_->GetInputShape(i);
        if (shape != nullptr && shape->GetStorageShape() != shape->GetOriginShape()) {
            OP_LOGE(context_, "input storage and origin shapes must agree for ND");
            return ge::GRAPH_FAILED;
        }
    }
    std::vector<grq::Shape> y;
    std::vector<grq::Shape> scale;
    if (!ReadOutputs(context_, grq::Y, dstType, y) ||
        !ReadOutputs(context_, grq::SCALE, ge::DT_FLOAT8_E8M0, scale)) {
        OP_LOGE(context_, "invalid output metadata/dtype/ND storage shape");
        return ge::GRAPH_FAILED;
    }
    if (!grq::ValidateOutputs(inputParams_, y, scale, error)) {
        OP_LOGE(context_, "%s", error.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedRotateQuantTiling950::GetPlatformInfo()
{
    const auto *info = context_->GetCompileInfo<GroupedRotateQuantCompileInfo>();
    if (info != nullptr && info->hardware.aicNum > 0) {
        compileInfo_ = *info;
    } else {
        auto *platformInfo = context_->GetPlatformInfo();
        if (platformInfo == nullptr) {
            OP_LOGE(context_, "missing platform/compile information");
            return ge::GRAPH_FAILED;
        }
        platform_ascendc::PlatformAscendC platform(platformInfo);
        ReadHardware(platform, compileInfo_);
    }
    if (!IsCapable()) {
        OP_LOGE(context_, "GroupedRotateQuant only supports DAV_3510");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool GroupedRotateQuantTiling950::IsCapable()
{
    return compileInfo_.npuArch == static_cast<int32_t>(NpuArch::DAV_3510);
}

ge::graphStatus GroupedRotateQuantTiling950::DoOpTiling()
{
    std::string error;
    if (!grq::BuildPlan(inputParams_, compileInfo_.hardware, tilingData_, error)) {
        OP_LOGE(context_, "%s", error.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedRotateQuantTiling950::GetWorkspaceSize()
{
    // The planned Tensor API path uses Fixpipe L0C -> UB and writes y/scale directly.
    // No full-M*N intermediate or group-shape scratch allocation is required.
    workspaceSize_ = grq::SYSTEM_WORKSPACE_BYTES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedRotateQuantTiling950::PostTiling()
{
    auto *raw = context_->GetRawTilingData();
    auto *workspaces = context_->GetWorkspaceSizes(1);
    if (raw == nullptr || raw->GetData() == nullptr || raw->GetCapacity() < sizeof(tilingData_) ||
        workspaces == nullptr) {
        OP_LOGE(context_, "missing workspace/tiling buffer or insufficient tiling capacity");
        return ge::GRAPH_FAILED;
    }
    if (memcpy_s(raw->GetData(), raw->GetCapacity(), &tilingData_, sizeof(tilingData_)) != EOK) {
        OP_LOGE(context_, "copying tiling data failed");
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(tilingData_));
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(tilingData_.usedCoreNum);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGroupedRotateQuant(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // One Ascend950 Tensor API strategy in v1; no unrelated quantized-GMM fallback.
    GroupedRotateQuantTiling950 tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareGroupedRotateQuant(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto *info = context->GetCompiledInfo<GroupedRotateQuantCompileInfo>();
    auto *platformInfo = context->GetPlatformInfo();
    if (info == nullptr || platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *info = {};
    platform_ascendc::PlatformAscendC platform(platformInfo);
    ReadHardware(platform, *info);
    return info->npuArch == static_cast<int32_t>(NpuArch::DAV_3510) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

IMPL_OP_OPTILING(GroupedRotateQuant)
    .Tiling(TilingGroupedRotateQuant)
    .TilingParse<GroupedRotateQuantCompileInfo>(TilingPrepareGroupedRotateQuant);
} // namespace optiling
