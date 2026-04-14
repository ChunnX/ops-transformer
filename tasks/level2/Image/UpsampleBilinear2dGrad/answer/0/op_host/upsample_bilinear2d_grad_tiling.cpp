/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_bilinear2d_grad_tiling.cpp
 * \brief
 */
#include <cmath>

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_bilinear2d_grad_tiling.h"

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)

namespace optiling {
constexpr uint32_t BEST_PERFORMANCE_SIZE = 16;
constexpr uint32_t BYTE = 8;
constexpr uint32_t BYTE_REPEAT = 256;
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_BF16 = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr float MIN_SUPPORT_SCALE = 0.02;

constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_HALF = 3;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;

constexpr int32_t DOUBLE_VALUE = 2;
constexpr int32_t RESERVED_VALUE = 4;
constexpr int32_t EIGHT_VALUE = 8;
constexpr uint8_t SCHEDULE_MODE = 1;

class UpsampleBilinear2dGradTiling {
public:
    explicit UpsampleBilinear2dGradTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus Init() const;
    ge::graphStatus RunBigKernelTiling();

private:
    void setScale();
    inline float compute_scale_value(const int64_t in_size, const int64_t out_size,
                                     const bool align_corner, const float *scale);
    void getWorkSpace(uint32_t needCoreNum);
    void getOutputShape();
    void getSlideSize();
    uint8_t GetDataTypeSize() const;
    uint64_t GetDataTypeVal() const;
    uint64_t GetTilingKeyVal() const;
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatForm);
    uint32_t GetNeedCoreNumW(const uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumH(const uint32_t coreNumPlatform);

    void setSingleCoreK();
    void FillTilingData();
    void getTCubeTiling_w();
    void getTCubeTiling_h();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    int64_t slide_size = 16;
    UpsampleBilinear2dGradTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize = 0;
    gert::Shape input_shape;
    uint8_t dim = 0;
    const bool *align_corners = nullptr;
    const float *scale_h = nullptr;
    const float *scale_w = nullptr;
    float realScale_h = 0.0f;
    float realScale_w = 0.0f;
    const gert::ContinuousVector *output_size = nullptr;
    const gert::ContinuousVector *input_size = nullptr;
    int64_t slideStartList_w[MAX_CORE_CONT] = {0};
    int64_t slideEndList_w[MAX_CORE_CONT] = {0};
    int64_t tailRowStartList_w[MAX_CORE_CONT] = {0};
    int64_t tailRowEndList_w[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartList_w[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndList_w[MAX_CORE_CONT] = {0};

    int64_t slideStartList_h[MAX_CORE_CONT] = {0};
    int64_t slideEndList_h[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartList_h[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndList_h[MAX_CORE_CONT] = {0};
    int64_t tailRowStartList_h[MAX_CORE_CONT] = {0};
    int64_t tailRowEndList_h[MAX_CORE_CONT] = {0};
    int32_t tailBatchStartList_h[MAX_CORE_CONT] = {0};
    int32_t tailBatchEndList_h[MAX_CORE_CONT] = {0};
    int64_t output_shapes[4] = {0};
    int64_t input_shapes[4] = {0};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int32_t singleCoreK_w = 0;
    int32_t singleCoreK_h = 0;
};

inline bool FloatEqual(const float a, const float b)
{
    const float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

ge::graphStatus UpsampleBilinear2dGradTiling::Init() const
{
    return ge::GRAPH_SUCCESS;
}

void UpsampleBilinear2dGradTiling::setScale()
{
    if (dim == H_INDEX) {
        const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());
        realScale_h =
            compute_scale_value(input_shape.GetDim(H_INDEX), output_size_array[H_INDEX], *align_corners, scale_h);
        realScale_w =
            compute_scale_value(input_shape.GetDim(W_INDEX), output_size_array[W_INDEX], *align_corners, scale_w);
        if (*align_corners) {
            tilingData.set_align_corners(C_INDEX);
        } else {
            tilingData.set_align_corners(H_INDEX);
        }
        tilingData.set_scale_h(realScale_h);
        tilingData.set_scale_w(realScale_w);
    }
}

inline float UpsampleBilinear2dGradTiling::compute_scale_value(
    const int64_t in_size, const int64_t out_size, const bool align_corner, const float *scale)
{
    if (align_corner) {
        if (in_size > 1) {
            return static_cast<float>(out_size - 1) / static_cast<float>(in_size - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        if (scale != nullptr && *scale > 0) {
            return static_cast<float>(*scale);
        } else {
            return in_size != 0 ? static_cast<float>(out_size) / static_cast<float>(in_size) : 1;
        }
    }
}

void UpsampleBilinear2dGradTiling::setSingleCoreK()
{
    getSlideSize();
    setScale();
    getOutputShape();
    if (!FloatEqual(realScale_w, static_cast<float>(0.0))) {
        singleCoreK_w = Ceil((slide_size + RESERVED_VALUE) / realScale_w) + DOUBLE_VALUE * RESERVED_VALUE;
        singleCoreK_w = (singleCoreK_w + EIGHT_VALUE - 1) / EIGHT_VALUE * EIGHT_VALUE;
        if (singleCoreK_w < slide_size) {
            singleCoreK_w = slide_size;
        }
    } else {
        singleCoreK_w = input_shapes[W_INDEX];
        if (singleCoreK_w < slide_size) {
            singleCoreK_w = slide_size;
        }
    }
    if (!FloatEqual(realScale_h, static_cast<float>(0.0))) {
        singleCoreK_h = Ceil((slide_size + RESERVED_VALUE) / realScale_h) + DOUBLE_VALUE * RESERVED_VALUE;
        singleCoreK_h = (singleCoreK_h + EIGHT_VALUE - 1) / EIGHT_VALUE * EIGHT_VALUE;
        if (singleCoreK_h < slide_size) {
            singleCoreK_h = slide_size;
        }
    } else {
        singleCoreK_h = input_shapes[H_INDEX];
        if (singleCoreK_h < slide_size) {
            singleCoreK_h = slide_size;
        }
    }
}

ge::graphStatus UpsampleBilinear2dGradTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    input_size = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_IF(input_size == nullptr, OP_LOGE(tilingContext->GetNodeName(), "input_size == nullptr"),
        return ge::GRAPH_FAILED);
    output_size = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    OP_CHECK_IF(output_size == nullptr, OP_LOGE(tilingContext->GetNodeName(), "output_size == nullptr"),
        return ge::GRAPH_FAILED);

    align_corners = attrs->GetAttrPointer<bool>(H_INDEX);
    OP_CHECK_IF(align_corners == nullptr, OP_LOGE(tilingContext->GetNodeName(), "align_corners == nullptr"),
        return ge::GRAPH_FAILED);

    scale_h = attrs->GetAttrPointer<float>(W_INDEX);
    OP_CHECK_IF(scale_h == nullptr, OP_LOGE(tilingContext->GetNodeName(), "scale_h == nullptr"), return ge::GRAPH_FAILED);
    scale_w = attrs->GetAttrPointer<float>(RESERVED_VALUE);
    OP_CHECK_IF(scale_w == nullptr, OP_LOGE(tilingContext->GetNodeName(), "scale_w == nullptr"), return ge::GRAPH_FAILED);
    auto tmp = tilingContext->GetInputDesc(0);
    if (tmp == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType srcDtype = ge::DT_UNDEFINED;
    srcDtype = tmp->GetDataType();

    if (dataType == ge::DT_UNDEFINED) {
        dataType = srcDtype;
        dataTypeSize = GetDataTypeSize();
    } else if (srcDtype != dataType) {
        return ge::GRAPH_FAILED;
    }

    auto srcShape = tilingContext->GetInputShape(0);
    // 固定是2
    dim = srcShape->GetStorageShape().GetDimNum() - 2;

    input_shape = srcShape->GetOriginShape();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    const uint32_t coreNumPlatForm = ascendcPlatform.GetCoreNum();

    tilingContext->SetTilingKey(GetTilingKeyVal());

    setSingleCoreK();

    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatForm);

    // 计算workspace，每个核的系数矩阵，中间矩阵大小
    getWorkSpace(needCoreNum);

    tilingContext->SetBlockDim(needCoreNum);

    FillTilingData();

    return ge::GRAPH_SUCCESS;
}

uint32_t UpsampleBilinear2dGradTiling::GetNeedCoreNumH(const uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[2];
    int64_t output_w = output_shapes[3];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = 0;
    int64_t remainder = 0;
    if (coreNumPlatform != 0) {
        eachCoreSlideNum = slideNum / coreNumPlatform;
        remainder = slideNum % coreNumPlatform;
    }

    int64_t inputBatch = input_shapes[0] * input_shapes[1];
    int64_t tailAvergingBatch = 0;
    int64_t input_w = output_shapes[3];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = BEST_PERFORMANCE_SIZE;

    if (remainder != 0) {
        // 获取最小分行数
        int64_t minAvergingRows = BEST_PERFORMANCE_SIZE;
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingBatch = CeilA2B(inputBatch, groupCoreNum);
        tailAvergingRows = std::max(CeilA2B(output_w, groupCoreNum), minAvergingRows);
        // 单个batch，每个分组的核数
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputBatch, tailAvergingBatch));
    }

    uint32_t needCoreNum_h = 0;

    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartList_h[coreIndex] = coreIndex * eachCoreSlideNum * slide_size;
        slideEndList_h[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slide_size;
        if (remainder != 0) {
            // 尾块处理, 核数不全都一样
            int64_t groupIndex = groupCoreNum != 0 ? coreIndex / groupCoreNum : 1;
            if (groupIndex < remainder) {
                // 算出第几个分组
                tailSlideStartList_h[coreIndex] = (tailStartSlideNum + groupIndex) * slide_size;
                tailSlideEndList_h[coreIndex] =
                    std::min(tailSlideStartList_h[coreIndex] + slide_size, static_cast<int64_t>(outputSize));

                int64_t coreIndexInGroup = groupCoreNum != 0 ? coreIndex % groupCoreNum : 1;

                tailRowStartList_h[coreIndex] = coreIndexInGroup * tailAvergingRows;
                tailRowEndList_h[coreIndex] =
                    std::min(tailRowStartList_h[coreIndex] + tailAvergingRows, static_cast<int64_t>(input_w));
                tailBatchStartList_h[coreIndex] = coreIndexInGroup * tailAvergingBatch;
                tailBatchEndList_h[coreIndex] =
                    std::min(tailBatchStartList_h[coreIndex] + tailAvergingBatch, static_cast<int64_t>(inputBatch));
                needCoreNum_h++;
            }
        }
    }

    if (eachCoreSlideNum > 0) {
        needCoreNum_h = coreNumPlatform;
    }

    tilingData.set_need_core_num_h(needCoreNum_h);

    return needCoreNum_h;
}

void UpsampleBilinear2dGradTiling::getTCubeTiling_h()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);

    matmul_tiling::MatmulApiTiling mmTiling_h;
    mmTiling_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_h.SetOrgShape(output_shapes[H_INDEX], output_shapes[W_INDEX], input_shapes[W_INDEX]);
    mmTiling_h.SetShape(slide_size, output_shapes[W_INDEX], singleCoreK_h);

    if (mmTiling_h.GetTiling(tilingData.matmulTiling_h) == -1) {
        return;
    }
}

void UpsampleBilinear2dGradTiling::getTCubeTiling_w()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetOrgShape(input_shape.GetDim(N_INDEX) * input_shape.GetDim(C_INDEX) * input_shape.GetDim(H_INDEX),
        output_shapes[W_INDEX],
        input_shape.GetDim(W_INDEX));
    mmTiling_w.SetShape(
        input_shape.GetDim(0) * input_shape.GetDim(C_INDEX) * input_shape.GetDim(H_INDEX), slide_size, singleCoreK_w);

    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        return;
    }
}

// 先只算w方向
void UpsampleBilinear2dGradTiling::getWorkSpace(uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    // 中间tensor
    uint64_t intermediate_matrix_size = output_shapes[0] * output_shapes[1] * input_shape.GetDim(2) * output_shapes[3];

    uint32_t radioMatrixWorkspaceSize = slide_size * singleCoreK_w;
    uint32_t radioMatrixWorkspaceSize_h = slide_size * singleCoreK_h;
    uint32_t maxRadioMatrixWorkspaceSize = std::max(radioMatrixWorkspaceSize, radioMatrixWorkspaceSize_h);
    workspaces[0] = (intermediate_matrix_size + maxRadioMatrixWorkspaceSize * needCoreNum) * BYTE + WORK_SPACE_SIZE;
    tilingData.set_radio_matrix_size(radioMatrixWorkspaceSize);
    tilingData.set_radio_matrix_size_h(radioMatrixWorkspaceSize_h);
    tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
}

void UpsampleBilinear2dGradTiling::getOutputShape()
{
    const int64_t *output_size_arr = reinterpret_cast<const int64_t *>(output_size->GetData());
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        input_shapes[i] = input_shape.GetDim(i);
        output_shapes[i] = input_shape.GetDim(i);
        if (i > 1) {
            output_shapes[i] = output_size_arr[i];
        }
    }
    tilingData.set_input_shapes(input_shapes);
    tilingData.set_output_shapes(output_shapes);
}

void UpsampleBilinear2dGradTiling::getSlideSize()
{
    slide_size = BEST_PERFORMANCE_SIZE;
    tilingData.set_slide_size(slide_size);
}

template <typename T1, typename T2>
inline auto UpsampleBilinear2dGradTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleBilinear2dGradTiling::Ceil(T1 x) const
{
    return static_cast<int32_t>(std::ceil(x));
}

uint8_t UpsampleBilinear2dGradTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

uint64_t UpsampleBilinear2dGradTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return DATE_TYPE_FLOAT;
        case ge::DT_BF16:
            return DATE_TYPE_HALF;
        case ge::DT_FLOAT16:
            return DATE_TYPE_FLOAT16;
        default:
            return 0;
    }
}

uint64_t UpsampleBilinear2dGradTiling::GetTilingKeyVal() const
{
    switch (dataType) {
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        default:
            return 0;
    }
}
uint32_t UpsampleBilinear2dGradTiling::GetNeedCoreNum(const uint32_t coreNumPlatForm)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (!FloatEqual(realScale_w, 1.0)) {
        needCoreNumW = GetNeedCoreNumW(coreNumPlatForm);

        getTCubeTiling_w();
        tilingData.set_need_core_num_w(needCoreNumW);
    }

    if ((!FloatEqual(realScale_h, 1.0)) || (FloatEqual(realScale_w, 1.0))) {
        needCoreNumH = GetNeedCoreNumH(coreNumPlatForm);

        getTCubeTiling_h();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}
uint32_t UpsampleBilinear2dGradTiling::GetNeedCoreNumW(const uint32_t coreNumPlatform)
{
    // 反向保持一致，还是按结果切
    //  先简单处理，直接均分给每个核；如果计算系数矩阵bound，可以优化，先按照批次分组分配，再分配给各个核，可以减少计算系数矩阵次数。

    int64_t outputSize = output_shapes[3];

    int64_t slideNum = CeilA2B(outputSize, slide_size);

    int64_t eachCoreSlideNum = coreNumPlatform != 0 ? slideNum / coreNumPlatform : 1;

    int64_t remainder = coreNumPlatform != 0 ? slideNum % coreNumPlatform : 1;
    // H维度总数
    int64_t input_h = input_shape.GetDim(0) * input_shape.GetDim(1) * input_shape.GetDim(2);
    // 获取最小分行数
    int64_t minAvergingRows = slide_size;

    // 按照剩余尾块数给核分组，然后每组核再均分行数
    int64_t groupCoreNum = coreNumPlatform / remainder;

    int64_t tailAvergingRows = std::max(CeilA2B(input_h, groupCoreNum), minAvergingRows);

    groupCoreNum = std::min(groupCoreNum, CeilA2B(input_h, tailAvergingRows));

    int64_t needCore = 0;

    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartList_w[coreIndex] = coreIndex * eachCoreSlideNum * slide_size;
        slideEndList_w[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slide_size;
        // 尾块处理
        int64_t groupIndex = groupCoreNum != 0 ? coreIndex / groupCoreNum : 1;
        if (groupIndex < remainder) {
            // 算出第几个分组
            tailSlideStartList_w[coreIndex] = (tailStartSlideNum + groupIndex) * slide_size;
            tailSlideEndList_w[coreIndex] =
                std::min(tailSlideStartList_w[coreIndex] + slide_size, static_cast<int64_t>(outputSize));
            int64_t coreIndexInGroup = groupCoreNum != 0 ? coreIndex % groupCoreNum : 1;
            tailRowStartList_w[coreIndex] = coreIndexInGroup * tailAvergingRows;
            tailRowEndList_w[coreIndex] =
                std::min(tailRowStartList_w[coreIndex] + tailAvergingRows, static_cast<int64_t>(input_h));

            needCore++;
        }
    }

    if (eachCoreSlideNum > 0) {
        needCore = coreNumPlatform;
    }

    return needCore;
}

void UpsampleBilinear2dGradTiling::FillTilingData()
{
    tilingData.set_slideStartList_w(slideStartList_w);
    tilingData.set_slideEndList_w(slideEndList_w);
    tilingData.set_tailSlideStartList_w(tailSlideStartList_w);
    tilingData.set_tailSlideEndList_w(tailSlideEndList_w);
    tilingData.set_tailRowStartList_w(tailRowStartList_w);
    tilingData.set_tailRowEndList_w(tailRowEndList_w);
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.set_slideStartList_h(slideStartList_h);
    tilingData.set_slideEndList_h(slideEndList_h);
    tilingData.set_tailSlideStartList_h(tailSlideStartList_h);
    tilingData.set_tailSlideEndList_h(tailSlideEndList_h);
    tilingData.set_tailRowStartList_h(tailRowStartList_h);
    tilingData.set_tailRowEndList_h(tailRowEndList_h);
    tilingData.set_tailBatchStartList_h(tailBatchStartList_h);
    tilingData.set_tailBatchEndList_h(tailBatchEndList_h);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleBilinear2dGradTiling(gert::TilingContext *context)
{
    UpsampleBilinear2dGradTiling tilingObject(context);
    context->SetScheduleMode(SCHEDULE_MODE);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(UpsampleBilinear2dGrad)
    .Tiling(tiling4UpsampleBilinear2dGradTiling);
}  // namespace optiling
