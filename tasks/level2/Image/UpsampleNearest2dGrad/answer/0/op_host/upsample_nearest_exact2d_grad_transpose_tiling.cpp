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
 * \file upsample_nearest_exact2d_grad_transpose_tiling.cpp
 * \brief
 */

#include "upsample_nearest_exact2d_grad_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }

namespace optiling {

constexpr int64_t SLIDE_SIZE = 16000;
constexpr uint8_t TILING_BASE_VALUE = 100;
constexpr uint8_t TILING_RESIZE_SMALL_VALUE = 10;
constexpr uint8_t TILING_ALIGN_VALUE = 1;
constexpr int32_t BUFFERS_NUM = 2;
constexpr int32_t SIZE_FLOAT = 4;
constexpr int32_t RESERVED_SIZE = 8 * 1024 * 1024;
constexpr uint64_t WORKSPACE_SIZE = 1 * 1024 * 1024;

class UpsampleNearestExact2dGradTransposeTiling {
public:
    explicit UpsampleNearestExact2dGradTransposeTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    void setScale();
    void getWorkSpace();
    void getOutputShape();
    uint8_t GetTilingKeyVal();
    void SetSlideSize();
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    bool isIntergerResize(int64_t input, int64_t output, float scale);
    void FillTilingData();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x);

    template <typename T1>
    inline T1 Min(T1 a, T1 b);

private:
    UpsampleNearestExact2dGradTransposeTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t funcType = 0;
    gert::Shape input_shape;
    const float* scaleH = nullptr;
    const float* scaleW = nullptr;
    float realScaleH = 0.0f;
    float realScaleW = 0.0f;
    const gert::ContinuousVector* output_size = nullptr;
    const gert::ContinuousVector* input_size = nullptr;

    int64_t batches = 0;
    int64_t outputShapes[4] = {0};
    int64_t inputShapes[4] = {0};

    int64_t startW[MAX_CORE_CONT] = {0};
    int64_t endW[MAX_CORE_CONT] = {0};
    int64_t startH[MAX_CORE_CONT] = {0};
    int64_t endH[MAX_CORE_CONT] = {0};
    int64_t startBatches[MAX_CORE_CONT] = {0};
    int64_t endBatches[MAX_CORE_CONT] = {0};

    int64_t eachCoreW = 0;
    int64_t needCoreNumW = 0;
    int64_t eachCoreH = 0;
    int64_t needCoreNumH = 0;
    int64_t eachCoreBatch = 0;
    int64_t needCoreNumBatch = 0;
    uint32_t needCoreNum = 0;

    bool isWResizeSmall = false;
    bool isHResizeSmall = false;
    bool isWAlign = false;
    bool isHAlign = false;
};

void UpsampleNearestExact2dGradTransposeTiling::setScale()
{
    const int64_t* output_size_array = reinterpret_cast<const int64_t*>(output_size->GetData());
    realScaleH = compute_scale_value(inputShapes[H_INDEX], outputShapes[H_INDEX], scaleH);
    realScaleW = compute_scale_value(inputShapes[W_INDEX], outputShapes[W_INDEX], scaleW);

    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);
}

ge::graphStatus UpsampleNearestExact2dGradTransposeTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = temp->GetDataType();

    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    input_size = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, input_size);
    output_size = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, output_size);
    scaleH = attrs->GetAttrPointer<float>(H_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, scaleH);
    scaleW = attrs->GetAttrPointer<float>(W_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, scaleW);

    auto src_shape = tilingContext->GetInputShape(0);
    input_shape = src_shape->GetOriginShape();

    getOutputShape();
    setScale();
    SetSlideSize();
    auto ascendc_platform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t coreNumPlatform = ascendc_platform.GetCoreNumAiv();
    needCoreNum = GetNeedCoreNum(coreNumPlatform);
    getWorkSpace();
    tilingContext->SetBlockDim(CeilA2B(needCoreNum, 2)); // 设置组合数
    tilingContext->SetTilingKey(GetTilingKeyVal());
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearestExact2dGradTransposeTiling::getWorkSpace()
{
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;
}

void UpsampleNearestExact2dGradTransposeTiling::getOutputShape()
{
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        inputShapes[i] = tilingContext->GetInputShape(0)->GetStorageShape().GetDim(i);
        outputShapes[i] = tilingContext->GetOutputShape(0)->GetStorageShape().GetDim(i);
    }
    batches = inputShapes[N_INDEX] * inputShapes[C_INDEX];
    tilingData.set_input_shapes(inputShapes);
    tilingData.set_output_shapes(outputShapes);
    tilingData.set_batches(batches);
}

template <typename T1, typename T2>
inline auto UpsampleNearestExact2dGradTransposeTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleNearestExact2dGradTransposeTiling::Ceil(T1 x)
{
    int32_t floor_x = int32_t(x);
    if (x == floor_x) {
        return floor_x;
    }
    return floor_x + 1;
}

template <typename T1>
inline T1 UpsampleNearestExact2dGradTransposeTiling::Min(T1 a, T1 b)
{
    return a < b ? a : b;
}

uint8_t UpsampleNearestExact2dGradTransposeTiling::GetTilingKeyVal()
{
    uint8_t tilingKey = TILING_BASE_VALUE;
    if (isWResizeSmall) {
        tilingKey += TILING_RESIZE_SMALL_VALUE;
        return tilingKey;
    }
    if (isWAlign) {
        tilingKey += TILING_ALIGN_VALUE;
    }
    return tilingKey;
}

void UpsampleNearestExact2dGradTransposeTiling::SetSlideSize()
{
    auto ascendc_platform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint64_t ubSize;
    ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize = ubSize - RESERVED_SIZE;
    int64_t slideSize = SLIDE_SIZE;
    if (slideSize * SIZE_FLOAT * BUFFERS_NUM > ubSize) {
        slideSize /= 2;
    }
    tilingData.set_slideSize(slideSize);
}

uint32_t UpsampleNearestExact2dGradTransposeTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    int64_t oh = outputShapes[2];
    int64_t ow = outputShapes[3];
    int64_t ih = inputShapes[2];
    int64_t iw = inputShapes[3];
    isWResizeSmall = (ow < iw || realScaleW > 1) ? true : false;
    isHResizeSmall = (oh < ih || realScaleH > 1) ? true : false;
    isWAlign = isWResizeSmall ? false : isIntergerResize(iw, ow, realScaleW);
    isHAlign = isHResizeSmall ? false : isIntergerResize(ih, oh, realScaleH);

    eachCoreW = CeilA2B(ow, coreNumPlatform);
    if (isWAlign) {
        int64_t m = ow / iw;
        eachCoreW = CeilA2B(eachCoreW, m) * m;
    }
    needCoreNumW = CeilA2B(ow, eachCoreW);
    eachCoreH = oh;
    needCoreNumH = 1;
    eachCoreBatch = batches;
    needCoreNumBatch = 1;

    if (needCoreNumW > 0 && coreNumPlatform / needCoreNumW >= 2) {
        int64_t CoreH = coreNumPlatform / needCoreNumW;
        eachCoreH = CeilA2B(oh, CoreH);
        if (isHAlign) {
            int64_t n = oh / ih;
            eachCoreH = CeilA2B(eachCoreH, n) * n;
        }
        needCoreNumH = CeilA2B(oh, eachCoreH);
    }

    if (coreNumPlatform / (needCoreNumW * needCoreNumH) >= 2) {
        int64_t CoreBatch = coreNumPlatform / (needCoreNumW * needCoreNumH);
        eachCoreBatch = CeilA2B(batches, CoreBatch);
        needCoreNumBatch = CeilA2B(batches, eachCoreBatch);
    }
    int64_t usedCore = needCoreNumW * needCoreNumH * needCoreNumBatch;
    return usedCore;
}

bool UpsampleNearestExact2dGradTransposeTiling::isIntergerResize(int64_t input, int64_t output, float scale)
{
    if (input == 0) {
        return false;
    }
    std::string opType(tilingContext->GetNodeType());
    float realScale = output != 0 ? static_cast<float>(input) / output : 0;
    return input > 0 && output > 0 && output % input == 0 && FloatEqual(realScale, scale);
}

void UpsampleNearestExact2dGradTransposeTiling::FillTilingData()
{
    for (int64_t i = 0; i < needCoreNum; i++) {
        startW[i] = (i % needCoreNumW) * eachCoreW;
        endW[i] = Min(startW[i] + eachCoreW, outputShapes[3]);
        startH[i] = i % (needCoreNumW * needCoreNumH) / needCoreNumW * eachCoreH;
        endH[i] = Min(startH[i] + eachCoreH, outputShapes[2]);
        startBatches[i] = i / (needCoreNumW * needCoreNumH) * eachCoreBatch;
        endBatches[i] = Min(startBatches[i] + eachCoreBatch, batches);
    }

    int64_t slideSizeH = isHAlign ? outputShapes[2] / inputShapes[2] : 1;
    int64_t slideSizeW = isWResizeSmall ? 1 : 127;
    if (isWAlign) {
        int64_t resizeW = outputShapes[3] / inputShapes[3];
        slideSizeW = slideSizeW / resizeW * resizeW;
    }

    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_slideSizeH(slideSizeH);
    tilingData.set_slideSizeW(slideSizeW);

    tilingData.set_isHResizeSmall(isHResizeSmall);
    tilingData.set_isWResizeSmall(isWResizeSmall);
    tilingData.set_isHAlign(isHAlign);
    tilingData.set_isWAlign(isWAlign);
    tilingData.set_startW(startW);
    tilingData.set_endW(endW);
    tilingData.set_startH(startH);
    tilingData.set_endH(endH);
    tilingData.set_startBatches(startBatches);
    tilingData.set_endBatches(endBatches);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

ge::graphStatus tiling4UpsampleNearestExact2dGradTransposeTiling(gert::TilingContext* context)
{
    UpsampleNearestExact2dGradTransposeTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}
} // namespace optiling
