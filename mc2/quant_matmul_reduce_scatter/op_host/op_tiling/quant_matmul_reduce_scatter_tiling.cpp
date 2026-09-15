/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_scatter_tiling.cpp
 * \brief host侧tiling实现 (QuantMatmulReduceScatter, MX-quant FP8, apace UDMA path, compute-first)
 *
 * 移植自直调形态 ST host 逻辑（apace/tests/st/matmul_reduce_scatter/src/main.cpp）：
 *   - K 不切分（每卡持有完整 K），M 轴按 rankSize 切分用于 AllToAll 通信
 *   - 通信数据为 CType (BF16)，非 AType (FP8)；通信量 = m * n * sizeof(BF16)
 *   - 通信 tile 沿 M 轴按 headMSize(默认 512) 切分，尾块须 16 对齐（Blaze 最小行粒度）
 *   - workspace = libApi + m * n * sizeof(BF16)（AIC 写 C_i、AIV PUT 源缓冲）
 */

#include <string>
#include <climits>
#include <cstdint>
#include <cstring>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "graph/types.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "ascendc/host_api/tiling/template_argument.h"
#include "apace/kernel/fusions/matmul_reduce_scatter/quant_matmul_reduce_scatter_urma_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../op_kernel/arch35/quant_matmul_reduce_scatter_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace Mc2Tiling {

// ---- input/output indices ----
constexpr uint32_t IDX_INPUT_CONTEXT = 0U;
constexpr uint32_t IDX_INPUT_X1 = 1U;
constexpr uint32_t IDX_INPUT_X2 = 2U;
constexpr uint32_t IDX_INPUT_X1_SCALE = 3U;
constexpr uint32_t IDX_INPUT_X2_SCALE = 4U;
constexpr uint32_t IDX_OUTPUT_Y = 0U;

// ---- attr indices (must match def order) ----
constexpr uint32_t ATTR_GROUP_INDEX = 0U;
constexpr uint32_t ATTR_HCCL_BUFFER_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_IS_TRANS_A_INDEX = 2U;
constexpr uint32_t ATTR_IS_TRANS_B_INDEX = 3U;
constexpr uint32_t ATTR_RANK_SIZE_INDEX = 4U;
constexpr uint32_t ATTR_GROUP_SIZE_INDEX = 5U;
constexpr uint32_t ATTR_COMM_MODE_INDEX = 7U;

// ---- constants ----
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SCALE_DIM_NUM = 3U;
constexpr uint64_t MX_SCALE_BLOCK = 64UL;
constexpr uint64_t SCALE_LAST_DIM = 2UL;
constexpr int64_t MAX_INT32_VAL = 2147483647;
constexpr int64_t K_MIN_VAL = 256;
constexpr int64_t K_MAX_VAL = 65535;
constexpr uint64_t COMM_TILE_M = 512UL;   // 通信头块 M（与直调形态默认值一致）
constexpr uint64_t M_ALIGN = 16UL;        // Blaze matmul 最小行粒度
constexpr uint64_t HCCL_BUFFER_RESERVED_BYTES = 2UL * 1024UL * 1024UL; // 通信 buffer 预留量（2MB）
constexpr uint64_t BYTES_PER_CTYPE = 2UL; // BF16

constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr uint64_t GROUP_N_OFFSET = 16UL;
constexpr uint64_t GROUP_M_OFFSET = 32UL;
constexpr uint64_t MX_GROUP_M = 1UL;
constexpr uint64_t MX_GROUP_N = 1UL;
constexpr uint64_t MX_GROUP_K = 32UL;

struct ShapeInfo {
    int64_t m{0}; // 每卡输入 x1 的完整 M（通信前），M 轴按 rankSize 切分
    int64_t k{0};
    int64_t n{0};
    uint64_t rankSize{0};
};

/**
 * @brief 校验tensor指针非空
 */
static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context)
{
    auto contextDesc = context->GetInputDesc(IDX_INPUT_CONTEXT);
    auto x1Desc = context->GetInputDesc(IDX_INPUT_X1);
    auto x2Desc = context->GetInputDesc(IDX_INPUT_X2);
    auto yDesc = context->GetOutputDesc(IDX_OUTPUT_Y);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    // scale 在 def 中为 OPTIONAL，但本算子仅支持 MX 量化，tiling 强制要求必传
    auto x1ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
    auto x2ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
    OP_TILING_CHECK(x1ScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x1_scale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2ScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x2_scale"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor数据类型
 */
static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    auto x1Desc = context->GetInputDesc(IDX_INPUT_X1);
    auto x2Desc = context->GetInputDesc(IDX_INPUT_X2);
    auto yDesc = context->GetOutputDesc(IDX_OUTPUT_Y);

    OP_TILING_CHECK(x1Desc->GetDataType() != ge::DT_FLOAT8_E4M3FN,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "x1", Ops::Base::ToString(x1Desc->GetDataType()).c_str(),
                        "The dtype of x1 must be DT_FLOAT8_E4M3FN"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Desc->GetDataType() != ge::DT_FLOAT8_E4M3FN,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "x2", Ops::Base::ToString(x2Desc->GetDataType()).c_str(),
                        "The dtype of x2 must be DT_FLOAT8_E4M3FN"),
                    return ge::GRAPH_FAILED);

    auto x1ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
    OP_TILING_CHECK(x1ScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x1_scale",
                                                          Ops::Base::ToString(x1ScaleDesc->GetDataType()).c_str(),
                                                          "The dtype of x1_scale must be DT_FLOAT8_E8M0"),
                    return ge::GRAPH_FAILED);
    auto x2ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
    OP_TILING_CHECK(x2ScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x2_scale",
                                                          Ops::Base::ToString(x2ScaleDesc->GetDataType()).c_str(),
                                                          "The dtype of x2_scale must be DT_FLOAT8_E8M0"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(yDesc->GetDataType() != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "y",
                                                          Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                                          "The dtype of y must be DT_BF16"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor格式
 */
static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    const std::initializer_list<std::pair<const gert::StorageShape *, const char *>> checkedShapes = {};
    (void)checkedShapes;

    auto contextDesc = context->GetInputDesc(IDX_INPUT_CONTEXT);
    ge::Format contextFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat()));
    OP_TILING_CHECK(contextFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "context", Ops::Base::ToString(contextFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto x1Desc = context->GetInputDesc(IDX_INPUT_X1);
    ge::Format x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x1", Ops::Base::ToString(x1Format).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto x2Desc = context->GetInputDesc(IDX_INPUT_X2);
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(x2Format != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x2", Ops::Base::ToString(x2Format).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto yDesc = context->GetOutputDesc(IDX_OUTPUT_Y);
    ge::Format yFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat()));
    OP_TILING_CHECK(yFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "y", Ops::Base::ToString(yFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto x1ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
    if (x1ScaleDesc != nullptr) {
        ge::Format x1ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x1ScaleDesc->GetStorageFormat()));
        OP_TILING_CHECK(
            x1ScaleFormat != ge::FORMAT_ND,
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x1_scale", Ops::Base::ToString(x1ScaleFormat).c_str(), "ND"),
            return ge::GRAPH_FAILED);
    }

    auto x2ScaleDesc = context->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
    if (x2ScaleDesc != nullptr) {
        ge::Format x2ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2ScaleDesc->GetStorageFormat()));
        OP_TILING_CHECK(
            x2ScaleFormat != ge::FORMAT_ND,
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x2_scale", Ops::Base::ToString(x2ScaleFormat).c_str(), "ND"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验单个 MX scale shape: [dim0, Ceil(K/64), 2]
 */
static ge::graphStatus CheckOneScaleShape(const gert::StorageShape *scaleShape, const char *nodeName,
                                          const char *paramName, int64_t dim0, int64_t scaleKDim)
{
    OP_TILING_CHECK(scaleShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, paramName), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scaleShape->GetStorageShape().GetDimNum() != SCALE_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, paramName, (std::to_string(scaleShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of scale must be 3D"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scaleShape->GetStorageShape().GetDim(0) != dim0 ||
                        scaleShape->GetStorageShape().GetDim(1) != scaleKDim ||
                        scaleShape->GetStorageShape().GetDim(2) != static_cast<int64_t>(SCALE_LAST_DIM),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, paramName,
                        (std::string("[") + std::to_string(scaleShape->GetStorageShape().GetDim(0)) + "," +
                         std::to_string(scaleShape->GetStorageShape().GetDim(1)) + "," +
                         std::to_string(scaleShape->GetStorageShape().GetDim(2)) + "]")
                            .c_str(),
                        (std::string("expected [") + std::to_string(dim0) + "," + std::to_string(scaleKDim) + "," +
                         std::to_string(SCALE_LAST_DIM) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 读取 rank_size attr（必须由外界显式传入，不做 group 反查）
 */
static ge::graphStatus ResolveRankSize(const gert::TilingContext *context, int64_t &rankSize)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    auto rankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE_INDEX);
    OP_TILING_CHECK(rankSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "rank_size"), return ge::GRAPH_FAILED);
    rankSize = *rankSizePtr;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 x1/x2 维度与取值，并提取基础 shape 信息（m/k/n/rankSize）
 */
static ge::graphStatus CheckInputShape(const gert::TilingContext *context, ShapeInfo &shapeInfo)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *x1Shape = context->GetInputShape(IDX_INPUT_X1);
    OP_TILING_CHECK(x1Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x1Shape->GetStorageShape().GetDimNum() != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x1", (std::to_string(x1Shape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of x1 must be 2D"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *x2Shape = context->GetInputShape(IDX_INPUT_X2);
    OP_TILING_CHECK(x2Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x2"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Shape->GetStorageShape().GetDimNum() != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x2", (std::to_string(x2Shape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of x2 must be 2D"),
                    return ge::GRAPH_FAILED);

    int64_t m = x1Shape->GetStorageShape().GetDim(0);
    int64_t kX1 = x1Shape->GetStorageShape().GetDim(1);
    int64_t x2N = x2Shape->GetStorageShape().GetDim(0);
    int64_t kX2 = x2Shape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(
        m == 0 || kX1 == 0 || x2N == 0 || kX2 == 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1/x2",
                                  (std::string("x1=[") + std::to_string(m) + "," + std::to_string(kX1) + "] x2=[" +
                                   std::to_string(x2N) + "," + std::to_string(kX2) + "]")
                                      .c_str(),
                                  "non-zero"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(kX1 != kX2,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "x1 and x2", (std::to_string(kX1) + " and " + std::to_string(kX2)).c_str(),
                        "The k-axis of x1 and x2 must be the same (K is not split across ranks)"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        kX1 < K_MIN_VAL || kX1 >= K_MAX_VAL,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "K", std::to_string(kX1).c_str(),
            (std::string("in range [") + std::to_string(K_MIN_VAL) + ", " + std::to_string(K_MAX_VAL) + ")").c_str()),
        return ge::GRAPH_FAILED);

    // MX scale 对齐约束：CeilDiv(k, 64) 必须为偶数
    OP_TILING_CHECK(((kX1 + static_cast<int64_t>(MX_SCALE_BLOCK) - 1) / static_cast<int64_t>(MX_SCALE_BLOCK)) % 2 != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "K", std::to_string(kX1).c_str(),
                                               "CeilDiv(K, 64) must be an even number in MXFP8 scene"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        m > MAX_INT32_VAL || x2N > MAX_INT32_VAL,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1.M/x2.N", (std::to_string(m) + "/" + std::to_string(x2N)).c_str(),
                                  "must not exceed INT32_MAX"),
        return ge::GRAPH_FAILED);

    // rank_size: 直接读取外界传入的 attr
    int64_t rankSize = 0;
    OP_TILING_CHECK(ResolveRankSize(context, rankSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "resolve rank_size failed"), return ge::GRAPH_FAILED);
    // 先拦截非正值，防止下方 m % rankSize / m / rankSize 除零（{2,4,8,16} 合法性由 CheckRankSizeAttr 把关）
    OP_TILING_CHECK(rankSize <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "rank_size", std::to_string(rankSize).c_str(), "positive"),
                    return ge::GRAPH_FAILED);

    // M 轴按 rankSize 切分用于 AllToAll 通信，必须整除
    OP_TILING_CHECK(m % rankSize != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1.M / rank_size",
                                              (std::to_string(m) + " / " + std::to_string(rankSize)).c_str(),
                                              "x1.M must be divisible by rank_size"),
                    return ge::GRAPH_FAILED);

    // chunkM = m / rankSize 须 16 对齐（Blaze matmul 最小行粒度，尾块约束）
    int64_t chunkM = m / rankSize;
    OP_TILING_CHECK(chunkM % static_cast<int64_t>(M_ALIGN) != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1.M / rank_size",
                                              (std::to_string(chunkM) + " (M/rank_size)").c_str(),
                                              "must be 16-aligned"),
                    return ge::GRAPH_FAILED);

    shapeInfo.m = m;
    shapeInfo.k = kX1;
    shapeInfo.n = x2N;
    shapeInfo.rankSize = static_cast<uint64_t>(rankSize);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 y shape: [M / rank_size, N]
 */
static ge::graphStatus CheckOutputShape(const gert::TilingContext *context, const ShapeInfo &shapeInfo)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *yShape = context->GetOutputShape(IDX_OUTPUT_Y);
    OP_TILING_CHECK(yShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yShape->GetStorageShape().GetDimNum() != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "y", (std::to_string(yShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of y must be 2D"),
                    return ge::GRAPH_FAILED);

    int64_t yM = yShape->GetStorageShape().GetDim(0);
    int64_t yN = yShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        yM != shapeInfo.m / static_cast<int64_t>(shapeInfo.rankSize) || yN != shapeInfo.n,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yM) + "," + std::to_string(yN) + "]").c_str(),
            (std::string("expected [") + std::to_string(shapeInfo.m / static_cast<int64_t>(shapeInfo.rankSize)) + "," +
             std::to_string(shapeInfo.n) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 scale shapes: x1_scale [M, Ceil(K/64), 2]; x2_scale [N, Ceil(K/64), 2]
 */
static ge::graphStatus CheckScaleShapes(const gert::TilingContext *context, const ShapeInfo &shapeInfo)
{
    const char *nodeName = context->GetNodeName();

    int64_t scaleKDim = (shapeInfo.k + static_cast<int64_t>(MX_SCALE_BLOCK) - 1) / static_cast<int64_t>(MX_SCALE_BLOCK);
    OP_TILING_CHECK(CheckOneScaleShape(context->GetOptionalInputShape(IDX_INPUT_X1_SCALE), nodeName, "x1_scale",
                                       shapeInfo.m, scaleKDim) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check x1_scale shape failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOneScaleShape(context->GetOptionalInputShape(IDX_INPUT_X2_SCALE), nodeName, "x2_scale",
                                       shapeInfo.n, scaleKDim) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check x2_scale shape failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor维度和shape
 */
static ge::graphStatus CheckTensorShape(const gert::TilingContext *context, ShapeInfo &shapeInfo)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckInputShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check input shape failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check output shape failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckScaleShapes(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check scale shape failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor (ptr + dtype + format + shape)
 */
static ge::graphStatus CheckTensor(const gert::TilingContext *context, ShapeInfo &shapeInfo)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckTensorPtrNullptr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check tensor nullptr failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDataType(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check tensor dtype failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorFormat(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check tensor format failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check tensor shape failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 rank_size: 合法取值且不超过 1:1 核配比下实际可用 block 数
 */
static ge::graphStatus CheckRankSizeAttr(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    // rank_size: 仅校验外界传入的 attr，不做 group 反查
    int64_t rankSize = 0;
    OP_TILING_CHECK(ResolveRankSize(context, rankSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "resolve rank_size failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSize != 2 && rankSize != 4 && rankSize != 8 && rankSize != 16,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "rank_size", std::to_string(rankSize).c_str(), "2/4/8/16"),
                    return ge::GRAPH_FAILED);
    // 拦截：1:1 核配比下实际可用 block 数 = min(aicNum, aivNum)
    // rankSize 不能超过实际可用 block 数，否则通信挂死
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t actualBlockNum = std::min(aicNum, aivNum);
    OP_TILING_CHECK(
        static_cast<uint32_t>(rankSize) > actualBlockNum,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "rank_size",
                                  (std::to_string(rankSize) + " > min(aicNum=" + std::to_string(aicNum) +
                                   ", aivNum=" + std::to_string(aivNum) + ")=" + std::to_string(actualBlockNum))
                                      .c_str(),
                                  "<= min(aicNum, aivNum)"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 is_trans_a / is_trans_b: 仅支持 x1 [M,K] + x2 [N,K]（TransA=false, TransB=true）
 */
static ge::graphStatus CheckTransposeAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto isTransAPtr = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_A_INDEX);
    OP_TILING_CHECK(isTransAPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "is_trans_a"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*isTransAPtr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "is_trans_a", "true", "false"),
                    return ge::GRAPH_FAILED);

    auto isTransBPtr = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B_INDEX);
    OP_TILING_CHECK(isTransBPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "is_trans_b"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!(*isTransBPtr),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "is_trans_b", "false", "true (x2 must be [N,K] layout)"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 comm_mode: 仅支持 aiv_urma
 */
static ge::graphStatus CheckCommModeAttr(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto commModePtr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE_INDEX);
    OP_TILING_CHECK(commModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "comm_mode"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::strcmp(commModePtr, "aiv_urma") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "comm_mode", commModePtr, "aiv_urma"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 group_size (MX quant: [1,1,32])，本算子仅支持 MX 量化，
 *        值为 0 的维度自动从 scale shape 推导，推导后校验是否为 [1,1,32]
 */
static ge::graphStatus CheckGroupSizeAttr(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group_size"),
                    return ge::GRAPH_FAILED);
    uint64_t gs = static_cast<uint64_t>(*groupSizePtr);
    uint64_t gsK = gs & GROUP_MNK_BIT_SIZE;
    uint64_t gsN = (gs >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t gsM = (gs >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;

    // 自动推导：值为 0 的维度从 scale shape 反推
    const gert::StorageShape *x1Shape = context->GetInputShape(IDX_INPUT_X1);
    const gert::StorageShape *x2Shape = context->GetInputShape(IDX_INPUT_X2);
    auto x1ScaleShape = context->GetOptionalInputShape(IDX_INPUT_X1_SCALE);
    auto x2ScaleShape = context->GetOptionalInputShape(IDX_INPUT_X2_SCALE);
    int64_t mValue = x1Shape->GetStorageShape().GetDim(0);
    int64_t kValue = x1Shape->GetStorageShape().GetDim(1);
    int64_t nValue = x2Shape->GetStorageShape().GetDim(0);

    if (gsM == 0) {
        int64_t scaleM = x1ScaleShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(scaleM == 0, OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1_scale M", "0", "non-zero"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            mValue % scaleM != 0,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1 M / x1_scale M",
                                      (std::to_string(mValue) + " / " + std::to_string(scaleM)).c_str(), "divisible"),
            return ge::GRAPH_FAILED);
        gsM = static_cast<uint64_t>(mValue / scaleM);
    }
    if (gsN == 0) {
        int64_t scaleN = x2ScaleShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(scaleN == 0, OP_LOGE_FOR_INVALID_VALUE(nodeName, "x2_scale N", "0", "non-zero"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            nValue % scaleN != 0,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "x2 N / x2_scale N",
                                      (std::to_string(nValue) + " / " + std::to_string(scaleN)).c_str(), "divisible"),
            return ge::GRAPH_FAILED);
        gsN = static_cast<uint64_t>(nValue / scaleN);
    }
    if (gsK == 0) {
        // mxfp scale shape [M, CeilDiv(K,64), 2]，K维 group size = K / (CeilDiv(K,64) * 2)
        int64_t scaleKDim = x1ScaleShape->GetStorageShape().GetDim(1);
        int64_t scaleKValue = scaleKDim * 2;
        OP_TILING_CHECK(scaleKValue == 0, OP_LOGE_FOR_INVALID_VALUE(nodeName, "x1_scale K dim", "0", "non-zero"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(kValue % scaleKValue != 0,
                        OP_LOGE_FOR_INVALID_VALUE(
                            nodeName, "x1 K / x1_scale K",
                            (std::to_string(kValue) + " / " + std::to_string(scaleKValue)).c_str(), "divisible"),
                        return ge::GRAPH_FAILED);
        gsK = static_cast<uint64_t>(kValue / scaleKValue);
    }

    OP_LOGI(nodeName, "group_size: attr=[%lu,%lu,%lu], inferred [M=%lu N=%lu K=%lu]",
            (gs >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE, (gs >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE,
            gs & GROUP_MNK_BIT_SIZE, gsM, gsN, gsK);
    OP_TILING_CHECK(gsM != MX_GROUP_M || gsN != MX_GROUP_N || gsK != MX_GROUP_K,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "group_size",
                                              (std::string("[") + std::to_string(gsM) + "," + std::to_string(gsN) +
                                               "," + std::to_string(gsK) + "]")
                                                  .c_str(),
                                              "[1,1,32]"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验通信数据量 + 预留量 <= hccl_buffer_size
 *
 * hccl_buffer_size 为 torch 层 get_hccl_buffer_size() 透传的实际 HCCL buffer 大小（受 HCCL_BUFFSIZE 环境变量控制）。
 * 与 all_gather_matmul_v3（通信 A 数据）不同，本算子计算在前，通信对象为 CType (BF16)：
 *   win 区布局 = R 个 rank 的 C 分片连续存放 = [m, n]（rank-major chunks，与 workspace 一致）
 *   通信数据量 = m * n * sizeof(BF16)
 * 预留 2MB（与通信框架开销/同步字段有关，对齐 torch 层拦截口径）。
 */
static ge::graphStatus CheckHcclBufferSizeAttr(const gert::TilingContext *context, const ShapeInfo &shapeInfo)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto hcclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HCCL_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(hcclBufferSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "hccl_buffer_size"),
                    return ge::GRAPH_FAILED);
    int64_t hcclBufferSize = *hcclBufferSizePtr;
    OP_TILING_CHECK(hcclBufferSize <= 0,
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "hccl_buffer_size", std::to_string(hcclBufferSize).c_str(),
                                              "positive integer"),
                    return ge::GRAPH_FAILED);

    uint64_t m = static_cast<uint64_t>(shapeInfo.m);
    uint64_t n = static_cast<uint64_t>(shapeInfo.n);
    uint64_t commDataBytes = m * n * BYTES_PER_CTYPE;
    uint64_t needBytes = commDataBytes + HCCL_BUFFER_RESERVED_BYTES;

    OP_TILING_CHECK(needBytes > static_cast<uint64_t>(hcclBufferSize),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "hccl_buffer_size",
                        (std::string("need ") + std::to_string(needBytes) + " bytes = comm_data " +
                         std::to_string(commDataBytes) + " + reserved " + std::to_string(HCCL_BUFFER_RESERVED_BYTES) +
                         ", but got " + std::to_string(hcclBufferSize))
                            .c_str(),
                        ">= comm_data + 2MB; please increase HCCL_BUFFSIZE environment variable"),
                    return ge::GRAPH_FAILED);

    OP_LOGI(nodeName,
            "hccl_buffer_size check: hcclBufferSize=%ld bytes, commDataBytes=%lu, reserved=%lu, needBytes=%lu, PASS",
            hcclBufferSize, commDataBytes, HCCL_BUFFER_RESERVED_BYTES, needBytes);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验算子属性
 */
static ge::graphStatus CheckAttrs(const gert::TilingContext *context, const ShapeInfo &shapeInfo)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckRankSizeAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check rank_size attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTransposeAttrs(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check transpose attrs failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckCommModeAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check comm_mode attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckGroupSizeAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check group_size attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckHcclBufferSizeAttr(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check hccl_buffer_size attr failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling数据
 */
static ge::graphStatus SetTilingData(gert::TilingContext *context, const ShapeInfo &shapeInfo)
{
    const char *nodeName = context->GetNodeName();

    auto *rawTilingData = context->GetRawTilingData();
    OP_TILING_CHECK(rawTilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "rawTilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rawTilingData->GetCapacity() < sizeof(QuantMatmulReduceScatterTilingData),
                    OP_LOGE(nodeName, "rawTilingData capacity %zu < %zu", rawTilingData->GetCapacity(),
                            sizeof(QuantMatmulReduceScatterTilingData)),
                    return ge::GRAPH_FAILED);

    memset_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), 0,
             rawTilingData->GetCapacity()); // 校验非空并tilingdata 缓冲区清零
    auto *tilingData = reinterpret_cast<QuantMatmulReduceScatterTilingData *>(rawTilingData->GetData());

    uint64_t m = static_cast<uint64_t>(shapeInfo.m);
    uint64_t k = static_cast<uint64_t>(shapeInfo.k);
    uint64_t n = static_cast<uint64_t>(shapeInfo.n);
    uint64_t rankSize = shapeInfo.rankSize;

    OP_LOGI(nodeName, "quant dtype: x1=FP8_E4M3FN x2=FP8_E4M3FN y=BF16, m=%lu k=%lu n=%lu rankSize=%lu", m, k, n,
            rankSize);

    // 全量 mm tiling via SWAT engine（完整 m，无 TP 切分；T=1 退化路径共用）
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                              mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetPlatformInfoPtr(context->GetPlatformInfo());
        tilingEngine.SetCoreLimit(ascendcPlatform.GetCoreNumAic(), ascendcPlatform.GetCoreNumAiv());
        tilingEngine.GetTilingData(m, n, k, false, true, tilingData->mmTilingData);
    }
    OP_LOGD(nodeName, "mmTilingData(swat): baseM=%u baseN=%u baseK=%u dbL0c=%u usedCoreNum=%u",
            static_cast<uint32_t>(tilingData->mmTilingData.baseM), static_cast<uint32_t>(tilingData->mmTilingData.baseN),
            static_cast<uint32_t>(tilingData->mmTilingData.baseK), static_cast<uint32_t>(tilingData->mmTilingData.dbL0c),
            static_cast<uint32_t>(tilingData->mmTilingData.usedCoreNum));

    // 通信 tiling: PUT 模式多 tile（切分轴 = M，非切分轴 = N）
    // chunkM = m / rankSize，按 headMSize（默认 512，16 对齐）切分为头块 + 可选 1 个尾块
    uint64_t chunkM = m / rankSize;
    uint64_t headMSize = COMM_TILE_M;
    if (headMSize > chunkM) {
        headMSize = chunkM;
    }
    headMSize = (headMSize / M_ALIGN) * M_ALIGN;
    if (headMSize == 0) {
        headMSize = M_ALIGN;
    }
    uint64_t headTileCnt = chunkM / headMSize;
    uint64_t tailMSize = chunkM % headMSize; // chunkM 与 headMSize 均 16 对齐，尾块自然 16 对齐
    uint64_t tailTileCnt = (tailMSize > 0) ? 1 : 0;
    uint64_t tileCnt = headTileCnt + tailTileCnt;

    tilingData->commTilingData.splitAxisTileCnt = headTileCnt;
    tilingData->commTilingData.splitAxisTileSize = headMSize;
    tilingData->commTilingData.splitAxisTailSize = tailMSize;
    tilingData->commTilingData.splitAxisTailCnt = tailTileCnt;
    tilingData->commTilingData.nonSplitAxisSize = n;

    // 子问题 tiling: m=headMSize，用于 T>1 per-tile head tile（baseM 减半提升负载均衡，同 allto_all_matmul_v2）
    if (tileCnt > 1) {
        {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                                  mm::BiasDataType::DT_FLOAT>
                subTilingEngine;
            subTilingEngine.SetPlatformInfoPtr(context->GetPlatformInfo());
            subTilingEngine.SetCoreLimit(ascendcPlatform.GetCoreNumAic(), ascendcPlatform.GetCoreNumAiv());
            subTilingEngine.EnableBaseMHalving(true);
            subTilingEngine.GetTilingData(headMSize, n, k, false, true, tilingData->subMmTilingData);
        }
        // tail tile tiling: m = rankSize * tailMSize（tailMSize 已保证 16 对齐）
        if (tailTileCnt > 0 && tailMSize > 0) {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                                  mm::BiasDataType::DT_FLOAT>
                tailTilingEngine;
            tailTilingEngine.SetPlatformInfoPtr(context->GetPlatformInfo());
            tailTilingEngine.SetCoreLimit(ascendcPlatform.GetCoreNumAic(), ascendcPlatform.GetCoreNumAiv());
            uint64_t tailLogicalM = rankSize * tailMSize;
            tailTilingEngine.GetTilingData(tailLogicalM, n, k, false, true, tilingData->tailMmTilingData);
        }
        // tailTileCnt == 0 时 tailMmTilingData 保持 memset 清零状态
    }
    // tileCnt == 1 时 subMmTilingData/tailMmTilingData 保持清零状态（T=1 走 mmTilingData 退化路径）

    // AIV 侧核分工：后 rankSize 个核做 AllToAll PUT，前 usedCoreNum-rankSize 个核做 reduceSum。
    // 要求 usedCoreNum > rankSize（至少保留计算核），否则 reduceSum 无核执行；以 2*rankSize 为下限。
    uint32_t swatUsedCoreNum = tilingData->mmTilingData.usedCoreNum;
    uint32_t minCoreNum = static_cast<uint32_t>(rankSize) * 2U;
    uint32_t usedCoreNum = (swatUsedCoreNum < minCoreNum) ? minCoreNum : swatUsedCoreNum;
    tilingData->mmTilingData.usedCoreNum = usedCoreNum;

    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context->SetScheduleMode(1);
    context->SetBlockDim(usedCoreNum);
    rawTilingData->SetDataSize(rawTilingData->GetCapacity());

    OP_LOGI(nodeName,
            "QuantMatmulReduceScatter tiling: m=%lu k=%lu n=%lu rankSize=%lu usedCoreNum=%u(swat=%u) "
            "commTile[tileSize=%lu tileCnt=%lu tailSize=%lu tailCnt=%lu]",
            m, k, n, rankSize, usedCoreNum, swatUsedCoreNum, tilingData->commTilingData.splitAxisTileSize,
            tilingData->commTilingData.splitAxisTileCnt, tilingData->commTilingData.splitAxisTailSize,
            tilingData->commTilingData.splitAxisTailCnt);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling key
 */
static void SetTilingKey(gert::TilingContext *context)
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(MX_QUANT_MODE);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "tilingKey is [%lu]", tilingKey);
}

/**
 * @brief 设置workspace大小
 *
 * workspace 布局（前段 C_i 缓冲，libApi workspace 附于其后，总量计入 workspaces[0]）：
 *   - C_i 缓冲: [m, n] * sizeof(BF16)，AIC 写入本 rank 完整 C，AIV 从此处 PUT 到其他 rank win 区
 *   - libApi: 框架预留
 */
static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const ShapeInfo &shapeInfo)
{
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    uint64_t libApiWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ciBufferSize = static_cast<uint64_t>(shapeInfo.m) * static_cast<uint64_t>(shapeInfo.n) * BYTES_PER_CTYPE;
    uint64_t workspaceSize = libApiWorkSpaceSize + ciBufferSize;
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(context->GetNodeName(), "workSpaces is nullptr"),
                    return ge::GRAPH_FAILED);
    workSpaces[0] = workspaceSize;
    OP_LOGD(context->GetNodeName(), "workspace size is %lu (libApi=%lu, ciBuffer=%lu)", workspaceSize,
            libApiWorkSpaceSize, ciBufferSize);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief QuantMatmulReduceScatter 算子的 tiling 函数
 */
static ge::graphStatus QuantMatmulReduceScatterTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("QuantMatmulReduceScatter", "failed to get tiling context"),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "nodeName"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Enter QuantMatmulReduceScatter tiling func");

    // 1. tensor check (ptr + dtype + format + shape)
    ShapeInfo shapeInfo;
    OP_TILING_CHECK(CheckTensor(context, shapeInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check input/output failed"), return ge::GRAPH_FAILED);

    // 2. attr check
    OP_TILING_CHECK(CheckAttrs(context, shapeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check attrs failed"),
                    return ge::GRAPH_FAILED);

    // 3. set tiling data
    OP_TILING_CHECK(SetTilingData(context, shapeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set tiling data failed"),
                    return ge::GRAPH_FAILED);

    // 4. set tiling key
    SetTilingKey(context);

    // 5. set workspace
    OP_TILING_CHECK(SetWorkSpace(context, shapeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set workspace failed"),
                    return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "QuantMatmulReduceScatter tiling end");
    return ge::GRAPH_SUCCESS;
}

struct QuantMatmulReduceScatterCompileInfo {};

static ge::graphStatus TilingParseForQuantMatmulReduceScatter(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantMatmulReduceScatter)
    .Tiling(QuantMatmulReduceScatterTilingFunc)
    .TilingParse<QuantMatmulReduceScatterCompileInfo>(TilingParseForQuantMatmulReduceScatter);

} // namespace Mc2Tiling
