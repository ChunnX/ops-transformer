/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

/*!
 * \file cumprod_tiling.cpp
 * \brief Cumprod tiling（PyPTO kernel 的 host 侧 tiling，AscendC 方式）。
 *
 * 职责（全部异常显式拦截）：
 *   1. 平台信息：AIV 核数（全核 launch——分段 SCAN 路径含 kernel 内 INTER_BLOCK
 *      阶段屏障，参与者为本次 launch 的全部 AIV 核，部分 launch 会破坏屏障语义）
 *   2. 输入校验：dtype ∈ {FLOAT, FLOAT16, BF16}；rank ∈ [1, 8]；axis ∈ [-rank, rank)
 *   3. axis 常量读取（输入1，TilingInputsDataDependency；aclnn 流程经 executor
 *      ConvertToTensor 物化为 host 常量张量，GE 流程为常量折叠 Const）
 *   4. 三折轴分解 [outer, scan, inner] + exclusive / reverse 属性读取
 *   5. 路径选择：inner>1 → FOLD（TilingKey mode=0）；inner==1 → SCAN（mode=1，
 *      低并行度时长轴两段式分段：kernel 内 pass A → 跨核屏障 → pass B）
 *   6. TilingKey 按 "bit0=mode" 手工打包（对应 op_kernel/cumprod.py 的
 *      CumprodTilingKey，方式同 dense_lightning_indexer_softmax_lse_v2_tiling.cpp）
 *   7. Workspace：LibApi 保留区 + 分段 SCAN 路径的 [seg_count, outer] fp32 段聚合数组
 */

#include <algorithm>
#include <limits>

#include "cumprod_tiling.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

using namespace ge;

namespace optiling {

// ── kernel 常量镜像（op_kernel/cumprod.py；两者必须同步修改）──────────────────
static constexpr int64_t CUMPROD_MIN_RANK = 1;
static constexpr int64_t CUMPROD_MAX_RANK = 8;
static constexpr int64_t CUMPROD_MAX_SEGMENTS = 64;   // MAXP in the kernel
static constexpr int64_t CUMPROD_ROWS_FP32 = 64;      // ROWS32 in the kernel
static constexpr int64_t CUMPROD_ROWS_NARROW = 128;   // ROWS16 in the kernel
static constexpr int64_t CUMPROD_CHUNK_FP32 = 64;     // CHUNK32 in the kernel (FOLD)
static constexpr int64_t CUMPROD_CHUNK_NARROW = 128;  // CHUNK16 in the kernel (FOLD)

// Attr indices (cumprod_def.cpp): 0 = exclusive, 1 = reverse
static constexpr uint32_t CUMPROD_ATTR_EXCLUSIVE = 0;
static constexpr uint32_t CUMPROD_ATTR_REVERSE = 1;

static bool CumprodGetConstAxis(gert::TilingContext *context, int64_t &axis)
{
    const gert::Tensor *axisTensor = context->GetInputTensor(1);
    if (axisTensor == nullptr) {
        return false;
    }
    // 仅 host 常量张量可读：aclnn 层把 api 标量经 executor ConvertToTensor 物化为
    // host 常量张量；GE 流程为常量折叠 Const。设备侧张量（图上计算所得）本 kernel
    // 不支持（三折轴必须在 host 侧折算进 tiling）。
    if (axisTensor->GetPlacement() != gert::kOnHost && axisTensor->GetPlacement() != gert::kFollowing) {
        return false;
    }
    const ge::DataType dtype = axisTensor->GetDataType();
    if (dtype == ge::DT_INT32) {
        const int32_t *p = axisTensor->GetData<int32_t>();
        if (p == nullptr) {
            return false;
        }
        axis = static_cast<int64_t>(p[0]);
        return true;
    }
    if (dtype == ge::DT_INT64) {
        const int64_t *p = axisTensor->GetData<int64_t>();
        if (p == nullptr) {
            return false;
        }
        axis = p[0];
        return true;
    }
    OP_LOGE(context->GetNodeName(), "axis dtype must be int32 or int64, but got %d.",
            static_cast<int32_t>(dtype));
    return false;
}

static int64_t CumprodCeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

static ge::graphStatus CumprodTilingFunc(gert::TilingContext *context)
{
    // 1、平台信息：AIV/AIC 核数
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aivNum == 0,
                OP_LOGE(context->GetNodeName(), "num of aiv core obtained is 0."),
                return ge::GRAPH_PARAM_INVALID);

    // 全核 launch：分段 SCAN 路径含 kernel 内 INTER_BLOCK 阶段屏障（参与者为本次
    // launch 的全部 AIV 核）；无屏障路径上空闲核的 job 循环零迭代，无害。
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // 2、输入 dtype 校验（kernel 仅支持 FLOAT/FLOAT16/BF16 三种数值路径）
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "input x desc is null."),
                return ge::GRAPH_PARAM_INVALID);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16,
                OP_LOGE(context->GetNodeName(),
                        "The dtype of input x must be within the range [DT_FLOAT, DT_FLOAT16, DT_BF16], but got %d.",
                        static_cast<int32_t>(xDtype)),
                return ge::GRAPH_PARAM_INVALID);
    const bool isFp32 = (xDtype == ge::DT_FLOAT);

    // 3、读取 axis 常量值（输入1，host 常量张量；显式拦截不可读情形）
    int64_t dim = 0;
    OP_CHECK_IF(!CumprodGetConstAxis(context, dim),
                OP_LOGE(context->GetNodeName(),
                        "failed to get the const value of axis (must be a host const int32/int64 tensor)."),
                return ge::GRAPH_PARAM_INVALID);

    // 4、获取并校验 x 的 shape（rank ∈ [1, 8]）
    auto xShape = context->GetInputShape(0);
    OP_CHECK_IF(xShape == nullptr,
                OP_LOGE(context->GetNodeName(), "input x shape is null."),
                return ge::GRAPH_PARAM_INVALID);
    int64_t rank = static_cast<int64_t>(xShape->GetStorageShape().GetDimNum());
    OP_CHECK_IF(rank < CUMPROD_MIN_RANK || rank > CUMPROD_MAX_RANK,
                OP_LOGE(context->GetNodeName(), "rank of x must be in [1, 8], but got %ld.", rank),
                return ge::GRAPH_PARAM_INVALID);

    // 5、属性 exclusive / reverse（运行时 bool，四组合共享同一 kernel 变体）
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE(context->GetNodeName(), "GetAttrs is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    const bool *exclusivePtr = attrs->GetAttrPointer<bool>(CUMPROD_ATTR_EXCLUSIVE);
    OP_CHECK_IF(exclusivePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "attr exclusive is nullptr."), return ge::GRAPH_PARAM_INVALID);
    const bool *reversePtr = attrs->GetAttrPointer<bool>(CUMPROD_ATTR_REVERSE);
    OP_CHECK_IF(reversePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "attr reverse is nullptr."), return ge::GRAPH_PARAM_INVALID);

    // 6、负轴折算 + 范围拦截 + 三折轴分解 [outer, scan, inner]
    if (dim < 0) {
        dim += rank;
    }
    OP_CHECK_IF(dim < 0 || dim >= rank,
                OP_LOGE(context->GetNodeName(), "axis must be in [-rank, rank) with rank=%ld, but got %ld.",
                        rank, dim),
                return ge::GRAPH_PARAM_INVALID);
    int64_t outer = 1;
    int64_t scanLen = xShape->GetStorageShape().GetDim(dim);
    int64_t inner = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer *= xShape->GetStorageShape().GetDim(i);
    }
    for (int64_t i = dim + 1; i < rank; i++) {
        inner *= xShape->GetStorageShape().GetDim(i);
    }

    // 7、workspace 与 tiling data
    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    // Workspace: LibApi 保留区 + 用户区（分段 SCAN 路径溢出 [seg_count, outer] fp32 段聚合）
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    // CumprodTiling 来自 pypto kernel（op_kernel/cumprod.py dataclass），经 force-include 可见
    CumprodTiling *tilingData = context->GetTilingData<CumprodTiling>();
    OP_CHECK_IF(tilingData == nullptr,
                OP_LOGE(context->GetNodeName(), "GetTilingData<CumprodTiling> is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    tilingData->outer = outer;
    tilingData->scan = scanLen;
    tilingData->inner = inner;
    tilingData->exclusive = *exclusivePtr;
    tilingData->reverse = *reversePtr;

    int64_t segCount = 1;
    int64_t segLen = scanLen;
    int64_t mode = 1; // 0 = FOLD (inner > 1), 1 = SCAN (inner == 1)
    if (outer == 0 || scanLen == 0 || inner == 0) {
        // 空 tensor：零 job（kernel 各循环界为 0，不产生任何 GM 访问），无用户 workspace
        mode = 0;
        segCount = 1;
        segLen = 0;
    } else if (inner > 1) {
        // ── FOLD（inner>1，mode=0）：lane=inner 逐扫描行折叠；无用户 workspace ──
        mode = 0;
        segCount = 1;
        segLen = scanLen;
    } else {
        // ── SCAN（inner==1，mode=1）：lane=扫描线逐列扫描；
        //    低并行度时长轴两段式分段（pass A 段聚合 → 跨核屏障 → pass B 主路径，
        //    同一 launch 内完成），目标并行度 = 2×AIV 核数 ──
        const int64_t rows = isFp32 ? CUMPROD_ROWS_FP32 : CUMPROD_ROWS_NARROW;
        const int64_t jobsBase = std::max(CumprodCeilDiv(outer, rows), static_cast<int64_t>(1));
        const int64_t target = 2 * static_cast<int64_t>(aivNum);
        segCount = std::min(std::min(CUMPROD_MAX_SEGMENTS, scanLen),
                            std::max(CumprodCeilDiv(target, jobsBase), static_cast<int64_t>(1)));
        segLen = CumprodCeilDiv(scanLen, segCount);
        segCount = CumprodCeilDiv(scanLen, segLen); // 重取整
        if (segCount > 1) {
            // 显式异常拦截：用户 workspace 字节数溢出（seg_count*outer*4）
            constexpr int64_t maxElems = std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(float));
            OP_CHECK_IF(segCount > maxElems || outer > maxElems || segCount * outer > maxElems,
                        OP_LOGE(context->GetNodeName(), "workspace size overflow, seg_count=%ld, outer=%ld.",
                                segCount, outer),
                        return ge::GRAPH_PARAM_INVALID);
            workspaceSize += static_cast<size_t>(segCount) * static_cast<size_t>(outer) * sizeof(float);
        }
    }
    tilingData->seg_count = segCount;
    tilingData->seg_len = segLen;

    // 8、TilingKey（对应 op_kernel/cumprod.py 的 CumprodTilingKey）：
    //    bit0 = mode（0 = FOLD, 1 = SCAN）。手工按位打包后下发，
    //    方式同 dense_lightning_indexer_softmax_lse_v2_tiling.cpp。
    uint64_t tilingKey = static_cast<uint64_t>(mode);
    context->SetTilingKey(tilingKey);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "Cumprod tiling completed: blockDim=%u, aivNum=%u, aicNum=%u, dim=%ld, outer=%ld, "
            "scan=%ld, inner=%ld, mode=%ld, segCount=%ld, segLen=%ld, exclusive=%d, reverse=%d, "
            "tilingKey=%lu, workspaceSize=%zu.",
            blockDim, aivNum, aicNum, dim, outer, scanLen, inner, mode, segCount, segLen,
            static_cast<int32_t>(*exclusivePtr), static_cast<int32_t>(*reversePtr), tilingKey, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCumprod([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling 注册入口；axis（输入1）为 tiling 数据依赖（host 侧读取其常量值）
IMPL_OP_OPTILING(Cumprod)
    .TilingInputsDataDependency({1})
    .Tiling(CumprodTilingFunc)
    .TilingParse<CumprodCompileInfo>(TilingParseForCumprod);

} // namespace optiling
