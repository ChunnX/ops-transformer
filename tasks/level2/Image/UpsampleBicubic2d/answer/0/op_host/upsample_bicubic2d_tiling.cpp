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
 * \file upsample_bicubic2d_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_bicubic2d_tiling.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)

namespace optiling {
constexpr uint32_t BEST_PERFORMANCE_SIZE_1 = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_2 = 32;
constexpr uint32_t BEST_PERFORMANCE_SIZE_3 = 48;
constexpr uint32_t BEST_PERFORMANCE_SIZE_4 = 64;
constexpr uint32_t SCALE_FIVE = 5;
constexpr uint32_t SCALE_EIGHT = 8;
constexpr uint32_t SCALE_TWENTY = 20;
constexpr uint32_t BYTE = 8;
constexpr uint32_t BYTE_REPEAT = 256;  // The amount of data that can be processed by a repeat.
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr int8_t DIM_SIZE = 2;
constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;

constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_BF16 = 3;

constexpr uint32_t OUTPUT_SIZE_ATTR = 0;
constexpr uint32_t ALIGN_CORNERS_ATTR = 1;
constexpr uint32_t SCALE_H_ATTR = 2;
constexpr uint32_t SCALE_W_ATTR = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LENGTH_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint32_t DIM_LEN = 4;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;

constexpr uint8_t SCHEDULE_MODE = 1;

class UpsampleBicubic2dTiling {
public:
    explicit UpsampleBicubic2dTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    void setScale();
    inline float compute_scale_value(const int64_t input_size, const int64_t out_size, const float *scale);
    void getWorkSpace(uint32_t needCoreNum);
    void getShapes();
    void getSlideSize();
    uint8_t GetDataTypeSize();
    uint64_t GetDataTypeVal();
    uint32_t GetNeedCoreNum();
    uint32_t GetNeedCoreNumW(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumH(uint32_t coreNumPlatform);
    void FillTilingData();
    void getTCubeTiling_w();
    void getTCubeTiling_h();
    bool CheckShapes() const;

    template <typename T1, typename T2>
    inline T1 CeilA2B(const T1 a, const T2 b) const;

    template <typename T1>
    inline int32_t Ceil(const T1 x) const;

private:
    int64_t slide_size = BEST_PERFORMANCE_SIZE_1;
    UpsampleBicubic2dTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize = BYTE_LENGTH_4;
    gert::Shape input_shape;
    uint8_t dim = 0;
    const bool *align_corners = nullptr;
    const float *scale_h = nullptr;
    const float *scale_w = nullptr;
    float realScale_h = 0.0;
    float realScale_w = 0.0;
    const gert::ContinuousVector *output_size = nullptr;
    int32_t slideStartList_w[MAX_CORE_CONT] = {0};
    int32_t slideEndList_w[MAX_CORE_CONT] = {0};
    int32_t tailSlideStartList_w[MAX_CORE_CONT] = {0};
    int32_t tailSlideEndList_w[MAX_CORE_CONT] = {0};
    int32_t tailRowStartList_w[MAX_CORE_CONT] = {0};
    int32_t tailRowEndList_w[MAX_CORE_CONT] = {0};

    int32_t slideStartList_h[MAX_CORE_CONT] = {0};
    int32_t slideEndList_h[MAX_CORE_CONT] = {0};
    int32_t tailSlideStartList_h[MAX_CORE_CONT] = {0};
    int32_t tailSlideEndList_h[MAX_CORE_CONT] = {0};
    int32_t tailRowStartList_h[MAX_CORE_CONT] = {0};
    int32_t tailRowEndList_h[MAX_CORE_CONT] = {0};
    int32_t tailBatchStartList_h[MAX_CORE_CONT] = {0};
    int32_t tailBatchEndList_h[MAX_CORE_CONT] = {0};

    int64_t output_shapes[4] = {0};
    int64_t input_shapes[4] = {0};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int64_t singleCoreK_w = 0;
    int64_t singleCoreK_h = 0;
};

void UpsampleBicubic2dTiling::setScale()
{
    if (dim == DIM_SIZE) {
        const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());

        realScale_h = compute_scale_value(input_shape.GetDim(H_INDEX), output_size_array[N_INDEX], scale_h);
        realScale_w = compute_scale_value(input_shape.GetDim(W_INDEX), output_size_array[C_INDEX], scale_w);

        tilingData.set_scale_h(realScale_h);
        tilingData.set_scale_w(realScale_w);

        int16_t max_interp_size_w = 4;
        int16_t max_interp_size_h = 4;

        tilingData.set_max_interp_size_w(max_interp_size_w);
        tilingData.set_max_interp_size_h(max_interp_size_h);
    }
}

inline float UpsampleBicubic2dTiling::compute_scale_value(
    const int64_t input_size, const int64_t out_size, const float *scale)
{
    if (out_size == input_size) {
        return static_cast<float>(1);
    }
    if (*align_corners) {
        if (out_size > 1) {
            return static_cast<float>(input_size - 1) / static_cast<float>(out_size - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return (scale != nullptr && *scale > 0) ? *scale
                                                : (static_cast<float>(input_size) / static_cast<float>(out_size));
    }
}

inline bool FloatEqual(float x, float y)
{
    float closeTo0 = float(1e-6);
    if (x > y) {
        return x - y < closeTo0;
    } else {
        return y - x < closeTo0;
    }
}

ge::graphStatus UpsampleBicubic2dTiling::RunBigKernelTiling()
{
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(tilingContext->GetNodeName(), "attrs == nullptr"), return ge::GRAPH_FAILED);

    output_size = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_ATTR);
    OP_CHECK_IF(output_size == nullptr, OP_LOGE(tilingContext->GetNodeName(), "output_size == nullptr"),
        return ge::GRAPH_FAILED);
    align_corners = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    OP_CHECK_IF(align_corners == nullptr, OP_LOGE(tilingContext->GetNodeName(), "align_corners == nullptr"),
        return ge::GRAPH_FAILED);
    scale_h = attrs->GetAttrPointer<float>(SCALE_H_ATTR);
    OP_CHECK_IF(scale_h == nullptr, OP_LOGE(tilingContext->GetNodeName(), "scale_h == nullptr"), return ge::GRAPH_FAILED);
    scale_w = attrs->GetAttrPointer<float>(SCALE_W_ATTR);
    OP_CHECK_IF(scale_w == nullptr, OP_LOGE(tilingContext->GetNodeName(), "scale_w == nullptr"), return ge::GRAPH_FAILED);

    auto tempInputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(tempInputDesc == nullptr, OP_LOGE(tilingContext->GetNodeName(), "InputDesc == nullptr"),
        return ge::GRAPH_FAILED);

    ge::DataType srcDtype = ge::DT_UNDEFINED;
    srcDtype = tempInputDesc->GetDataType();

    // Determine whether all data types are consistent.
    if (dataType == ge::DT_UNDEFINED) {
        dataType = srcDtype;
        dataTypeSize = GetDataTypeSize();
    }
    OP_CHECK_IF(
        srcDtype != dataType, OP_LOGE(tilingContext->GetNodeName(), "srcDtype != dataType"), return ge::GRAPH_FAILED);

    auto src_shape = tilingContext->GetInputShape(0);
    OP_CHECK_IF(
        src_shape == nullptr, OP_LOGE(tilingContext->GetNodeName(), "src_shape == nullptr"), return ge::GRAPH_FAILED);
    dim = src_shape->GetStorageShape().GetDimNum() - 2;  // fixed to 2

    input_shape = src_shape->GetOriginShape();

    OP_CHECK_IF(CheckShapes() == false, OP_LOGE(tilingContext->GetNodeName(), "CheckShapes() == false"),
        return ge::GRAPH_FAILED);

    tilingContext->SetTilingKey(1);

    setScale();
    getSlideSize();
    getShapes();

    // workspace, core number for coefficient matrices and intermediate matrix
    getWorkSpace(GetNeedCoreNum());
    tilingContext->SetBlockDim(GetNeedCoreNum());

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

bool UpsampleBicubic2dTiling::CheckShapes() const
{
    if (input_shape.GetDimNum() != SHAPE_SIZE) {
        OP_LOGE(tilingContext->GetNodeName(), "input_shape.GetDimNum() != SHAPE_SIZE");
        return false;
    }

    const int64_t *outputSizeArray = reinterpret_cast<const int64_t *>(output_size->GetData());
    int64_t inputH = input_shape.GetDim(H_INDEX);
    int64_t inputW = input_shape.GetDim(W_INDEX);
    int64_t outH = outputSizeArray[0];
    int64_t outW = outputSizeArray[W_INDEX - H_INDEX];

    if (!(inputH > 0 && inputW > 0 && outH > 0 && outW > 0)) {
        OP_LOGE(tilingContext->GetNodeName(), "!(inputH > 0 && inputW > 0 && outH > 0 && outW > 0)");
        return false;
    }

    return true;
}

void UpsampleBicubic2dTiling::getTCubeTiling_w()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetOrgShape(input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shape[H_INDEX],
        output_shapes[W_INDEX], input_shapes[W_INDEX]);
    mmTiling_w.SetShape(
        input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shape[H_INDEX], slide_size, singleCoreK_w);

    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        OP_LOGE(tilingContext->GetNodeName(), "getTCubeTiling_w Error, please Check inputShapes.");
        return;
    }
}

void UpsampleBicubic2dTiling::getTCubeTiling_h()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_h;
    mmTiling_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_h.SetOrgShape(output_shapes[H_INDEX], output_shapes[W_INDEX], input_shapes[W_INDEX]);
    mmTiling_h.SetShape(slide_size, output_shapes[W_INDEX], singleCoreK_h);

    if (mmTiling_h.GetTiling(tilingData.matmulTiling_h) == -1) {
        OP_LOGE(tilingContext->GetNodeName(), "getTCubeTiling_h Error, please Check inputShapes.");
        return;
    }
}

void UpsampleBicubic2dTiling::getWorkSpace(uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        // intermediate tensor
        uint64_t intermediate_matrix_size = output_shapes[N_INDEX] * output_shapes[C_INDEX] * input_shapes[H_INDEX] *
                                            output_shapes[W_INDEX] * dataTypeSize;
        intermediate_matrix_size = (intermediate_matrix_size + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

        // coefficient matrix per core, two workspace spaces per core to avoid overwriting
        int64_t singleCoreK = singleCoreK_w > singleCoreK_h ? singleCoreK_w : singleCoreK_h;

        uint32_t ratioMatrixWorkspaceSize = slide_size * singleCoreK * dataTypeSize;
        workspaces[0] = intermediate_matrix_size + ratioMatrixWorkspaceSize * needCoreNum + WORK_SPACE_SIZE;
        tilingData.set_ratio_matrix_size_w(slide_size * singleCoreK_w);
        tilingData.set_ratio_matrix_size_h(slide_size * singleCoreK_h);
        tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
    }
}

void UpsampleBicubic2dTiling::getShapes()
{
    const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        input_shapes[i] = input_shape.GetDim(i);
        output_shapes[i] = input_shape.GetDim(i);
        if (i > C_INDEX) {
            output_shapes[i] = output_size_array[i - H_INDEX];
        }
    }
    tilingData.set_input_shapes(input_shapes);
    tilingData.set_output_shapes(output_shapes);
}

void UpsampleBicubic2dTiling::getSlideSize()
{
    auto maxScale = realScale_h > realScale_w ? realScale_h : realScale_w;
    if (maxScale <= SCALE_FIVE) {
        slide_size = BEST_PERFORMANCE_SIZE_4;
    } else if (maxScale <= SCALE_EIGHT) {
        slide_size = BEST_PERFORMANCE_SIZE_3;
    } else if (maxScale <= SCALE_TWENTY) {
        slide_size = BEST_PERFORMANCE_SIZE_2;
    } else {
        slide_size = BEST_PERFORMANCE_SIZE_1;
    }
    tilingData.set_slide_size(slide_size);
}

template <typename T1, typename T2>
inline auto UpsampleBicubic2dTiling::CeilA2B(const T1 a, const T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleBicubic2dTiling::Ceil(const T1 x) const
{
    int32_t floor_x = int32_t(x);
    if (x == floor_x) {
        return floor_x;
    }
    return floor_x + 1;
}

uint8_t UpsampleBicubic2dTiling::GetDataTypeSize()
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LENGTH_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LENGTH_4;
    }
}

uint64_t UpsampleBicubic2dTiling::GetDataTypeVal()
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return DATE_TYPE_FLOAT;
        case ge::DT_FLOAT16:
            return DATE_TYPE_FLOAT16;
        case ge::DT_BF16:
            return DATE_TYPE_BF16;
        default:
            return 0;
    }
}

uint32_t UpsampleBicubic2dTiling::GetNeedCoreNum()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint16_t maxCoreNum = ascendcPlatform.GetCoreNumAiv();

    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (!FloatEqual(realScale_w, 1.0)) {
        singleCoreK_w = Ceil(slide_size * realScale_w) + Ceil(tilingData.get_max_interp_size_w());
        needCoreNumW = GetNeedCoreNumW(maxCoreNum);
        getTCubeTiling_w();
    }

    if (!FloatEqual(realScale_h, 1.0) || FloatEqual(realScale_w, 1.0)) {
        singleCoreK_h = Ceil(slide_size * realScale_h) + Ceil(tilingData.get_max_interp_size_h());
        needCoreNumH = GetNeedCoreNumH(maxCoreNum);
        getTCubeTiling_h();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

uint32_t UpsampleBicubic2dTiling::GetNeedCoreNumW(uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[3];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // H dimension total count
    int64_t input_h = input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shapes[H_INDEX];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = slide_size;

    if (remainder != 0) {
        int64_t minAvergingRows = slide_size;
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRows = std::max(CeilA2B(input_h, groupCoreNum), minAvergingRows);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(input_h, tailAvergingRows));
    }
    int64_t needCoreNum = 0;

    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    if (groupCoreNum == 0) {
        return coreNumPlatform;
    }
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartList_w[coreIndex] = coreIndex * eachCoreSlideNum * slide_size;
        slideEndList_w[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slide_size;
        if (remainder != 0) {
            int64_t groupIndex = coreIndex / groupCoreNum;
            if (groupIndex < remainder) {
                tailSlideStartList_w[coreIndex] = (tailStartSlideNum + groupIndex) * slide_size;
                tailSlideEndList_w[coreIndex] =
                    std::min(tailSlideStartList_w[coreIndex] + slide_size, static_cast<int64_t>(outputSize));
                int64_t coreIndexInGroup = coreIndex % groupCoreNum;
                tailRowStartList_w[coreIndex] = coreIndexInGroup * tailAvergingRows;
                tailRowEndList_w[coreIndex] =
                    std::min(tailRowStartList_w[coreIndex] + tailAvergingRows, static_cast<int64_t>(input_h));
                needCoreNum++;
            }
        }
    }

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    }

    tilingData.set_need_core_num_w(needCoreNum);

    return needCoreNum;
}

uint32_t UpsampleBicubic2dTiling::GetNeedCoreNumH(uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[H_INDEX];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // Batch and W dimension total count
    int64_t batch = input_shapes[N_INDEX] * input_shapes[C_INDEX];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingBatch = slide_size;
    if (remainder != 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingBatch = CeilA2B(batch, groupCoreNum);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(batch, tailAvergingBatch));
    }

    int64_t needCoreNum = 0;

    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    if (groupCoreNum == 0) {
        return coreNumPlatform;
    }
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartList_h[coreIndex] = coreIndex * eachCoreSlideNum * slide_size;
        slideEndList_h[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slide_size;
        if (remainder != 0) {
            int64_t groupIndex = coreIndex / groupCoreNum;
            if (groupIndex < remainder) {
                tailSlideStartList_h[coreIndex] = (tailStartSlideNum + groupIndex) * slide_size;
                tailSlideEndList_h[coreIndex] =
                    std::min(tailSlideStartList_h[coreIndex] + slide_size, static_cast<int64_t>(outputSize));
                int64_t coreIndexInGroup = coreIndex % groupCoreNum;
                tailRowStartList_h[coreIndex] = coreIndexInGroup * tailAvergingBatch;
                tailRowEndList_h[coreIndex] =
                    std::min(tailRowStartList_h[coreIndex] + tailAvergingBatch, static_cast<int64_t>(batch));
                needCoreNum++;
            }
        }
    }

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    }

    tilingData.set_need_core_num_h(needCoreNum);

    return needCoreNum;
}

void UpsampleBicubic2dTiling::FillTilingData()
{
    tilingData.set_slideStartList_w(slideStartList_w);
    tilingData.set_tailSlideStartList_w(tailSlideStartList_w);
    tilingData.set_tailRowStartList_w(tailRowStartList_w);
    tilingData.set_slideEndList_w(slideEndList_w);
    tilingData.set_tailSlideEndList_w(tailSlideEndList_w);
    tilingData.set_tailRowEndList_w(tailRowEndList_w);
    tilingData.set_dataType(GetDataTypeVal());

    tilingData.set_slideStartList_h(slideStartList_h);
    tilingData.set_slideEndList_h(slideEndList_h);
    tilingData.set_tailSlideStartList_h(tailSlideStartList_h);
    tilingData.set_tailSlideEndList_h(tailSlideEndList_h);
    tilingData.set_tailRowStartList_h(tailRowStartList_h);
    tilingData.set_tailRowEndList_h(tailRowEndList_h);

    tilingData.set_align_corners(*align_corners);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleBicubic2dTiling(gert::TilingContext *context)
{
    UpsampleBicubic2dTiling tilingObject(context);
    context->SetScheduleMode(SCHEDULE_MODE);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(UpsampleBicubic2d)
    .Tiling(tiling4UpsampleBicubic2dTiling);

}  // namespace optiling
