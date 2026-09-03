/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cumprod_tiling.cpp
 * \brief Cumprod tiling: read the axis input (host const in the aclnn flow after
 *        ConvertToTensor materialization, or a GE constant), fold shape to
 *        [outer, scan, inner], compute the segmentation for the low-outer/long-scan
 *        path, set TilingKey (axis_known / scan_mode), BlockDim (full AIV launch:
 *        the in-kernel INTER_BLOCK stage barrier requires all-core participation)
 *        and Workspace.
 */

#include "cumprod_tiling.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

using namespace ge;

namespace optiling {

static constexpr int64_t CUMPROD_MIN_RANK = 1;
static constexpr int64_t CUMPROD_MAX_RANK = 8;
static constexpr int64_t CUMPROD_MAX_SEGMENTS = 64;   // MAXP in the kernel
static constexpr int64_t CUMPROD_ROWS_FP32 = 64;      // ROWS32 in the kernel
static constexpr int64_t CUMPROD_ROWS_NARROW = 128;   // ROWS16 in the kernel

// Attr indices (cumprod_def.cpp): 0 = exclusive, 1 = reverse
static constexpr uint32_t CUMPROD_ATTR_EXCLUSIVE = 0;
static constexpr uint32_t CUMPROD_ATTR_REVERSE = 1;

static bool CumprodGetConstAxis(gert::TilingContext *context, int64_t &axis)
{
    const gert::Tensor *axisTensor = context->GetInputTensor(1);
    if (axisTensor == nullptr) {
        return false;
    }
    // Only host-resident tensors are readable here. The aclnn layer materializes the
    // api scalar into a host const tensor (executor ConvertToTensor); GE constants are
    // host-resident as well. A genuine device tensor (if a graph feeds one) is resolved
    // by the kernel on device (TilingKey axis_known = 0).
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

    // Full AIV launch for every input shape: the segmented SCAN path contains
    // an in-kernel INTER_BLOCK stage barrier whose participants are all AIV
    // cores, so partial launches would deadlock. Idle cores on small shapes
    // are harmless (job loops simply come up empty).
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // Attrs: exclusive / reverse (optional, default false; kernel implements inclusive forward)
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE(context->GetNodeName(), "GetAttrs is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    const bool *exclusivePtr = attrs->GetAttrPointer<bool>(CUMPROD_ATTR_EXCLUSIVE);
    const bool *reversePtr = attrs->GetAttrPointer<bool>(CUMPROD_ATTR_REVERSE);
    if ((exclusivePtr != nullptr && *exclusivePtr) || (reversePtr != nullptr && *reversePtr)) {
        OP_LOGE(context->GetNodeName(), "Cumprod only supports exclusive=false and reverse=false.");
        return ge::GRAPH_PARAM_INVALID;
    }

    // axis: input tensor (index 1). In the aclnn flow the api scalar is materialized
    // into a host const tensor (executor ConvertToTensor); in the GE flow it is a
    // constant-folded Const. Host-resolved values drive the TilingKey specialization;
    // a device-resident axis falls back to the kernel's device-side fold (every
    // tiling key stays correct).
    int64_t dim = 0;
    const bool axisIsConst = CumprodGetConstAxis(context, dim);

    auto xShape = context->GetInputShape(0);
    OP_CHECK_IF(xShape == nullptr,
                OP_LOGE(context->GetNodeName(), "input x shape is null."),
                return ge::GRAPH_PARAM_INVALID);
    int64_t rank = static_cast<int64_t>(xShape->GetStorageShape().GetDimNum());
    OP_CHECK_IF(rank < CUMPROD_MIN_RANK || rank > CUMPROD_MAX_RANK,
                OP_LOGE(context->GetNodeName(), "rank of x must be in [1, 8], but got %ld.", rank),
                return ge::GRAPH_PARAM_INVALID);

    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "input x desc is null."),
                return ge::GRAPH_PARAM_INVALID);
    const ge::DataType xDtype = xDesc->GetDataType();
    const bool isFp32 = (xDtype == ge::DT_FLOAT);

    // CumprodTiling comes from the pypto kernel (.py dataclass) via force-include.
    CumprodTiling *tilingData = context->GetTilingData<CumprodTiling>();
    tilingData->rank = rank;
    for (int64_t i = 0; i < rank; i++) {
        tilingData->dims[i] = xShape->GetStorageShape().GetDim(i);
    }
    for (int64_t i = rank; i < CUMPROD_MAX_RANK; i++) {
        tilingData->dims[i] = 0;
    }

    // Workspace: system reserve + user part (segmented SCAN path spills
    // [seg_count, outer] fp32 segment aggregates).
    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    int64_t outer = 1;
    int64_t scanLen = 1;
    int64_t inner = 1;
    int64_t segCount = 1;
    int64_t segLen = 0;
    int64_t mode = 1; // 0 = FOLD (inner > 1), 1 = SCAN (inner == 1)

    if (axisIsConst) {
        if (dim < 0) {
            dim += rank;
        }
        OP_CHECK_IF(dim < 0 || dim >= rank,
                    OP_LOGE(context->GetNodeName(),
                            "axis must be in [-rank, rank) with rank=%ld, but got %ld.",
                            rank, dim),
                    return ge::GRAPH_PARAM_INVALID);
        for (int64_t i = 0; i < dim; i++) {
            outer *= xShape->GetStorageShape().GetDim(i);
        }
        scanLen = xShape->GetStorageShape().GetDim(dim);
        for (int64_t i = dim + 1; i < rank; i++) {
            inner *= xShape->GetStorageShape().GetDim(i);
        }

        if (inner > 1) {
            // FOLD: jobs = outer * ceil(inner / chunk); no workspace needed.
            mode = 0;
        } else {
            // SCAN: rows per job depend on the dtype class.
            const int64_t rows = isFp32 ? CUMPROD_ROWS_FP32 : CUMPROD_ROWS_NARROW;
            const int64_t jobsBase = std::max(CumprodCeilDiv(outer, rows), static_cast<int64_t>(1));
            const int64_t target = 2 * static_cast<int64_t>(aivNum);
            segCount = std::min(std::min(CUMPROD_MAX_SEGMENTS, scanLen),
                                std::max(CumprodCeilDiv(target, jobsBase), static_cast<int64_t>(1)));
            if (scanLen < 1) {
                segCount = 1;
                segLen = 0;
            } else {
                segLen = CumprodCeilDiv(scanLen, segCount);
                segCount = CumprodCeilDiv(scanLen, segLen); // re-tighten
            }
            if (segCount > 1) {
                // Segmented SCAN: pass A spills fp32 segment aggregates.
                workspaceSize += static_cast<size_t>(segCount) * static_cast<size_t>(outer) * sizeof(float);
            }
        }
        tilingData->outer = outer;
        tilingData->scan = scanLen;
        tilingData->inner = inner;
        tilingData->seg_count = segCount;
        tilingData->seg_len = segLen;
    } else {
        // Device-side axis resolution: the kernel folds the axes and computes the
        // segmentation itself (same formula, cores = launched AIV count).
        // Workspace bound: seg_count * outer <= (4*aivNum - 1) * rows, since
        // seg_count > 1 implies jobsBase < 2*aivNum (outer <= jobsBase*rows)
        // and seg_count <= ceil(2*aivNum / jobsBase).
        tilingData->outer = 0;
        tilingData->scan = 0;
        tilingData->inner = 0;
        tilingData->seg_count = 1;
        tilingData->seg_len = 0;
        workspaceSize += static_cast<size_t>(4 * aivNum - 1) *
                         static_cast<size_t>(CUMPROD_ROWS_NARROW) * sizeof(float);
    }

    // TilingKey: bit0 = axis_known (host-resolved axes), bit1 = scan_mode
    //            (0 = FOLD, 1 = SCAN; only meaningful when axis_known = 1).
    //            Matches CumprodTilingKey in op_kernel/cumprod.py; GET_TPL_TILING_KEY
    //            packs each field per its candidate-value index.
    uint64_t tilingKey = GET_TPL_TILING_KEY(axisIsConst ? 1 : 0, mode);
    context->SetTilingKey(tilingKey);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "Cumprod tiling completed: blockDim=%u, aivNum=%u, axisIsConst=%d, dim=%ld, outer=%ld, "
            "scan=%ld, inner=%ld, mode=%ld, segCount=%ld, segLen=%ld, tilingKey=%lu, workspaceSize=%zu.",
            blockDim, aivNum, static_cast<int32_t>(axisIsConst), dim, outer, scanLen, inner, mode,
            segCount, segLen, tilingKey, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCumprod(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Cumprod)
    .TilingInputsDataDependency({1})
    .Tiling(CumprodTilingFunc)
    .TilingParse<CumprodCompileInfo>(TilingParseForCumprod);

} // namespace optiling
