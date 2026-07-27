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
 * \file broadcast_nddma_bench_tiling.cpp
 * \brief BroadcastNddmaBench 手写 tiling（free-function 模式）
 *
 * tiling 逻辑（对应原直调 host main 语义）：
 *   - dim0/dim1 从输入 shape[0]/[1] 读取（广播轴从不折叠这两维）
 *   - dim2/dim3/dim4/scene/is_mte3 从 OpDef Attr 读取（广播轴折叠使 input shape 丢失真实长度）
 *   - dtype 决定 TilingKey：fp32=0 / fp16=1 / int8=2
 *   - blockNum = min(dim0, coreNum)，SetBlockDim；kernel 内 per/rem 覆盖任意 dim0
 *   - workspace = 0（kernel 内无 workspace 使用）
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "broadcast_nddma_bench_tiling.h"

namespace {
constexpr uint64_t TILING_KEY_FP32 = 0;
constexpr uint64_t TILING_KEY_FP16 = 1;
constexpr uint64_t TILING_KEY_INT8 = 2;

constexpr uint32_t X_INDEX = 0;

// OpDef Attr 声明顺序 → attr index（is_mte3, scene, dim2, dim3, dim4）
constexpr uint32_t ATTR_IS_MTE3_INDEX = 0;
constexpr uint32_t ATTR_SCENE_INDEX = 1;
constexpr uint32_t ATTR_DIM2_INDEX = 2;
constexpr uint32_t ATTR_DIM3_INDEX = 3;
constexpr uint32_t ATTR_DIM4_INDEX = 4;

constexpr uint32_t DIM0_AXIS = 0;
constexpr uint32_t DIM1_AXIS = 1;
constexpr uint32_t MIN_RANK = 2;
} // namespace

namespace optiling {

struct BroadcastNddmaBenchCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

static uint64_t GetTilingKeyByDtype(ge::DataType dtype)
{
    if (dtype == ge::DataType::DT_FLOAT) {
        return TILING_KEY_FP32;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_FP16;
    }
    // DT_INT8
    return TILING_KEY_INT8;
}

static uint32_t GetAttrU32(const gert::RuntimeAttrs *attrs, uint32_t index, uint32_t defVal)
{
    if (attrs == nullptr) {
        return defVal;
    }
    const int64_t *ptr = attrs->GetAttrPointer<int64_t>(index);
    return (ptr != nullptr) ? static_cast<uint32_t>(*ptr) : defVal;
}

static ge::graphStatus BroadcastNddmaBenchTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BroadcastNddmaBench", "context is null"),
                return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(xShapePtr == nullptr, OP_LOGE(context, "input x shape is null"),
                return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() < MIN_RANK,
                OP_LOGE(context, "input x rank must be >= 2, got %zu", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // dim0/dim1 从 input shape 读取（广播轴从不折叠这两维）
    uint32_t dim0 = static_cast<uint32_t>(xShape.GetDim(DIM0_AXIS));
    uint32_t dim1 = static_cast<uint32_t>(xShape.GetDim(DIM1_AXIS));

    // dim2/dim3/dim4/scene/is_mte3 从 Attr 读取
    auto attrs = context->GetAttrs();
    uint32_t isMte3 = GetAttrU32(attrs, ATTR_IS_MTE3_INDEX, 1U);
    uint32_t scene = GetAttrU32(attrs, ATTR_SCENE_INDEX, 0U);
    uint32_t dim2 = GetAttrU32(attrs, ATTR_DIM2_INDEX, 0U);
    uint32_t dim3 = GetAttrU32(attrs, ATTR_DIM3_INDEX, 0U);
    uint32_t dim4 = GetAttrU32(attrs, ATTR_DIM4_INDEX, 0U);

    ge::DataType xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    uint64_t tilingKey = GetTilingKeyByDtype(xDtype);
    context->SetTilingKey(tilingKey);

    // blockNum = min(dim0, coreNum)（用例保证 dim0 <= coreNum，clamp 兜底）
    uint32_t coreNum = 0;
    auto compileInfo = reinterpret_cast<const BroadcastNddmaBenchCompileInfo *>(context->GetCompileInfo());
    if (compileInfo != nullptr) {
        coreNum = compileInfo->coreNum;
    }
    uint32_t blockDim = (coreNum > 0 && dim0 > coreNum) ? coreNum : dim0;
    context->SetBlockDim(blockDim);

    BroadcastNddmaBenchTilingData tilingData;
    tilingData.set_dim0(dim0);
    tilingData.set_dim1(dim1);
    tilingData.set_dim2(dim2);
    tilingData.set_dim3(dim3);
    tilingData.set_dim4(dim4);
    tilingData.set_isMte3(isMte3);
    tilingData.set_scene(scene);

    OP_LOGD(context, "BroadcastNddmaBench tiling: out(%u,%u,%u,%u,%u) scene=%u isMte3=%u tilingKey=%lu blockDim=%u",
            dim0, dim1, dim2, dim3, dim4, scene, isMte3, tilingKey, blockDim);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context, "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4BroadcastNddmaBench(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<BroadcastNddmaBenchCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context, "compileInfo is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
        compileInfo->ubSize = ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BroadcastNddmaBench)
    .Tiling(BroadcastNddmaBenchTilingFunc)
    .TilingParse<BroadcastNddmaBenchCompileInfo>(TilingPrepare4BroadcastNddmaBench);
} // namespace optiling
