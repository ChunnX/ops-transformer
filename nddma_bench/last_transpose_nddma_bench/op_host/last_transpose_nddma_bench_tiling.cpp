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
 * \file last_transpose_nddma_bench_tiling.cpp
 * \brief LastTransposeNddmaBench 手写 tiling（free-function 模式）
 *
 * tiling 逻辑（对应原直调 host main 语义）：
 *   - 输入 shape 就是 (dim0, dim1, dim2, dim3)，直接读回填 TilingData
 *   - dtype 决定 TilingKey：fp32=0 / fp16=1 / int8=2
 *   - blockNum = dim0（用例保证 dim0 <= coreNum），SetBlockDim(dim0)
 *   - workspace = 0（kernel 内无 workspace 使用）
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "last_transpose_nddma_bench_tiling.h"

namespace {
constexpr uint64_t TILING_KEY_FP32 = 0;
constexpr uint64_t TILING_KEY_FP16 = 1;
constexpr uint64_t TILING_KEY_INT8 = 2;

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t ATTR_IS_MTE3_INDEX = 0;

constexpr uint32_t DIM0_AXIS = 0;
constexpr uint32_t DIM1_AXIS = 1;
constexpr uint32_t DIM2_AXIS = 2;
constexpr uint32_t DIM3_AXIS = 3;
constexpr uint32_t EXPECT_RANK = 4;
} // namespace

namespace optiling {

struct LastTransposeNddmaBenchCompileInfo {
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

// dtype 元素字节数（fp32=4 / fp16=2 / int8=1），与 GetTilingKeyByDtype 的 dtype 分支一致。
static uint32_t GetDtypeSize(ge::DataType dtype)
{
    if (dtype == ge::DataType::DT_FLOAT) {
        return 4U;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return 2U;
    }
    // DT_INT8
    return 1U;
}

static ge::graphStatus LastTransposeNddmaBenchTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("LastTransposeNddmaBench", "context is null"),
                return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(xShapePtr == nullptr, OP_LOGE(context, "input x shape is null"),
                return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != EXPECT_RANK,
                OP_LOGE(context, "input x rank must be 4, got %zu", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    uint32_t dim0 = static_cast<uint32_t>(xShape.GetDim(DIM0_AXIS));
    uint32_t dim1 = static_cast<uint32_t>(xShape.GetDim(DIM1_AXIS));
    uint32_t dim2 = static_cast<uint32_t>(xShape.GetDim(DIM2_AXIS));
    uint32_t dim3 = static_cast<uint32_t>(xShape.GetDim(DIM3_AXIS));

    // is_mte3 可选属性，默认 1
    uint32_t isMte3 = 1U;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *isMte3Ptr = attrs->GetAttrPointer<int64_t>(ATTR_IS_MTE3_INDEX);
        if (isMte3Ptr != nullptr) {
            isMte3 = static_cast<uint32_t>(*isMte3Ptr);
        }
    }

    ge::DataType xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    uint64_t tilingKey = GetTilingKeyByDtype(xDtype);
    context->SetTilingKey(tilingKey);

    // 容量校验（下沉自 kernel）：单块 tileElems = dim2*dim3，tileBytes = tileElems * dtypeSize，
    //   DB 双缓冲每半区容量 halfCapBytes = ubSize/2；tileBytes 超限则报错返回。
    //   ubSize 取运行期真值（compileInfo->ubSize，来自 GetCoreMemSize(UB)）。
    uint32_t dtypeSize = GetDtypeSize(xDtype);
    auto compileInfo = reinterpret_cast<const LastTransposeNddmaBenchCompileInfo *>(context->GetCompileInfo());
    if (compileInfo != nullptr && compileInfo->ubSize > 0) {
        uint64_t tileBytes = static_cast<uint64_t>(dim2) * dim3 * dtypeSize;
        uint64_t halfCapBytes = compileInfo->ubSize / 2;
        OP_CHECK_IF(tileBytes > halfCapBytes,
                    OP_LOGE(context,
                            "UB tile capacity exceeded: tileBytes=%lu (dim2=%u dim3=%u dtypeSize=%u) "
                            "> halfCapBytes=%lu (ubSize=%lu, DB)",
                            tileBytes, dim2, dim3, dtypeSize, halfCapBytes, compileInfo->ubSize),
                    return ge::GRAPH_FAILED);
    }

    // blockNum = dim0（用例保证 dim0 <= coreNum）
    context->SetBlockDim(dim0);

    LastTransposeNddmaBenchTilingData tilingData;
    tilingData.set_dim0(dim0);
    tilingData.set_dim1(dim1);
    tilingData.set_dim2(dim2);
    tilingData.set_dim3(dim3);
    tilingData.set_isMte3(isMte3);

    OP_LOGD(context, "LastTransposeNddmaBench tiling: (%u,%u,%u,%u) isMte3=%u tilingKey=%lu blockDim=%u",
            dim0, dim1, dim2, dim3, isMte3, tilingKey, dim0);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context, "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4LastTransposeNddmaBench(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<LastTransposeNddmaBenchCompileInfo>();
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

IMPL_OP_OPTILING(LastTransposeNddmaBench)
    .Tiling(LastTransposeNddmaBenchTilingFunc)
    .TilingParse<LastTransposeNddmaBenchCompileInfo>(TilingPrepare4LastTransposeNddmaBench);
} // namespace optiling
