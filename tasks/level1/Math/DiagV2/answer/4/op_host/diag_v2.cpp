/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file diag_v2_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "diag_v2_tiling.h"
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

template <typename T1, typename T2>
T1 CeilDiv(T1 a, T2 b) {
    a = int64_t(a);
    b = int64_t(b);
    return T1(b == 0 ? a : (a + b - 1) / b);
};

namespace optiling {

static const int32_t SIZE_1 = 1;
static const int32_t SIZE_2 = 2;
static const int32_t SIZE_4 = 4;
static const int32_t SIZE_8 = 8;
static const int32_t SIZE_16 = 16;
static const int32_t LENGTH_1024 = 1024;
static const int32_t LENGTH_128 = 128;
static const int32_t LENGTH_64 = 64;
static const int32_t LENGTH_32 = 32;

inline static ge::graphStatus DiagV2SetTilingData(gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    auto rawTilingData = context->GetRawTilingData();
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static inline int64_t CalcLeastNumPerCore(const int32_t typeSize, const gert::TilingContext* context)
{
    OP_CHECK_IF(
        (typeSize <= 0), OP_LOGE(context, "Tiling4DiagV2 typeSize is invalid %d, please check.", typeSize), return -1);

    int64_t leastNumPerCore = BLOCK_SIZE / typeSize;
    return (leastNumPerCore > 0) ? leastNumPerCore : 1;
}

static ge::graphStatus Tiling4DiagV2(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4DiagV2 running begin");
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    auto inputShape = inputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        (inputShape.GetDimNum() != SIZE_2 && inputShape.GetDimNum() != 1),
        OP_LOGE(
            context, "Tiling4DiagV2 get input shape dim(=%zu) is not 1 or 2, please check.", inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    DiagV2TilingData tilingData;
    tilingData.set_xHeight(inputShape.GetDim(0));
    tilingData.set_xWidth(inputShape.GetDim(1));
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, diagonalPtr);
    const int64_t diagonal = *diagonalPtr;
    OP_CHECK_IF(
        (diagonal >= 0 && diagonal > tilingData.get_xWidth()) ||
            (diagonal < 0 && std::abs(diagonal) > tilingData.get_xHeight()),
        OP_LOGE(
            context, "Tiling4DiagV2 attr diagonal(=%ld) is wrong, please check. w=%ld, h=%ld", diagonal,
            tilingData.get_xWidth(), tilingData.get_xHeight()),
        return ge::GRAPH_FAILED);
    tilingData.set_gmOffset(diagonal > 0 ? diagonal : std::abs(diagonal) * tilingData.get_xWidth());
    tilingData.set_numOut(
        (diagonal >= 0) ? std::min((tilingData.get_xWidth() - diagonal), tilingData.get_xHeight()) :
                          std::min((tilingData.get_xHeight() + diagonal), tilingData.get_xWidth()));
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dataType = inputDesc->GetDataType();
    const int32_t typeSize = ge::GetSizeByDataType(dataType);
    int64_t leastNumPerCore = CalcLeastNumPerCore(typeSize, context);
    OP_CHECK_IF(
        (leastNumPerCore <= 0), OP_LOGE(context, "Tiling4DiagV2 leastNumPerCore is invalid, please check."),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNum();

    int64_t tmpRealCoreNum = tilingData.get_numOut() < coreNum ? tilingData.get_numOut() : coreNum;
    int64_t tmpCorePerNum = CeilDiv(tilingData.get_numOut(), tmpRealCoreNum);
    tilingData.set_numPerCore(CeilDiv(tmpCorePerNum, leastNumPerCore) * leastNumPerCore);
    tilingData.set_realCoreNum(CeilDiv(tilingData.get_numOut(), tilingData.get_numPerCore()));
    tilingData.set_tailNum(tilingData.get_numOut() - (tilingData.get_realCoreNum() - 1) * tilingData.get_numPerCore());
    tilingData.set_tilingKey(static_cast<int64_t>(2104));
    tilingData.set_matrixRowLength(LENGTH_64);
    OP_CHECK_IF(
        DiagV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "DiagV2SetTilingData set tiling data fail."), return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    std::cout << "*******************START*******************" << std::endl;
    std::cout << "xWidth = " << tilingData.get_xWidth() << std::endl;
    std::cout << "xHeight = " << tilingData.get_xHeight() << std::endl;
    std::cout << "gmOffset = " << tilingData.get_gmOffset() << std::endl;
    std::cout << "numOut = " << tilingData.get_numOut() << std::endl;
    std::cout << "realCoreNum = " << tilingData.get_realCoreNum() << std::endl;
    std::cout << "numPerCore = " << tilingData.get_numPerCore() << std::endl;
    std::cout << "tailNum = " << tilingData.get_tailNum() << std::endl;
    std::cout << "tilingKey = " << tilingData.get_tilingKey() << std::endl;
    std::cout << "matrixRowLength = " << tilingData.get_matrixRowLength() << std::endl;
    std::cout << "*******************END*******************" << std::endl;

    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class DiagV2 : public OpDef {
public:
    explicit DiagV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,
                 ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_BOOL})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,
                 ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_BOOL})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND});
        this->Attr("diagonal").AttrType(OPTIONAL).Int(0);
        this->AICore().SetTiling(optiling::Tiling4DiagV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(DiagV2);
} // namespace optiling
