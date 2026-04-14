/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_bilinear2d_aa_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "upsample_bilinear2d_aa_tiling.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }
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

constexpr uint32_t BEST_PERFORMANCE_SCALE_1 = 50;
constexpr uint32_t BEST_PERFORMANCE_SCALE_2 = 20;
constexpr uint32_t BEST_PERFORMANCE_SCALE_3 = 8;
constexpr uint32_t BEST_PERFORMANCE_SCALE_4 = 5;

constexpr uint32_t BYTE = 8;
constexpr uint32_t BYTE_REPEAT = 256;  // The amount of data that can be processed by a repeat.
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;

constexpr uint32_t OUTPUT_SIZE_ATTR = 0;
constexpr uint32_t ALIGN_CORNERS_ATTR = 1;
constexpr uint32_t SCALE_H_ATTR = 2;
constexpr uint32_t SCALE_W_ATTR = 3;

constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_HALF = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_FOUR = 4;
constexpr uint32_t BYTE_LEN_TWO = 2;

constexpr uint32_t DIM_LEN = 4;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;

constexpr float MAX_SUPPORT_SCALE = 50.0f;
constexpr uint8_t SCHEDULE_MODE = 1;

class UpsampleBilinearAATiling {
public:
    explicit UpsampleBilinearAATiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling(gert::TilingContext *context);
private:
    void setScale();
    inline float compute_scale_value(int64_t input_size, int64_t out_size,
                                     bool align_corner, const float *scale) const;
    bool getWorkSpace(uint32_t needCoreNum);
    void getShapes();
    void getSlideSize();
    uint8_t GetDataTypeSize() const;
    uint64_t GetDataTypeVal() const;
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatForm);
    uint32_t GetNeedCoreNumW(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumH(uint32_t coreNumPlatform);
    void FillTilingData();
    void getTCubeTiling_w();
    void getTCubeTiling_h();
    inline bool CheckScales(const gert::TilingContext *context, float scales_w, float scales_h);

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    int64_t slide_size{16};
    UpsampleBilinearAATilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize{4};
    gert::Shape input_shape;
    const bool *align_corners{nullptr};
    const float *scale_h{nullptr};
    const float *scale_w{nullptr};
    float realScale_h{0.0f};
    float realScale_w{0.0f};
    const gert::ContinuousVector *output_size{nullptr};

    int64_t output_shapes[4] = {0};
    int64_t input_shapes[4] = {0};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int64_t singleCoreK_w = 0;
    int64_t singleCoreK_h = 0;
};

void UpsampleBilinearAATiling::setScale()
{
    const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());

    realScale_h = compute_scale_value(input_shape.GetDim(H_INDEX), output_size_array[0], *align_corners, scale_h);
    realScale_w = compute_scale_value(input_shape.GetDim(W_INDEX), output_size_array[1], *align_corners, scale_w);

    tilingData.set_scale_h(realScale_h);
    tilingData.set_scale_w(realScale_w);

    float support_w = (realScale_w >= 1.0) ? realScale_w : 1.0;
    float support_h = (realScale_h >= 1.0) ? realScale_h : 1.0;

    tilingData.set_support_w(support_w);
    tilingData.set_support_h(support_h);

    int16_t max_interp_size_w = Ceil(support_w) * 2 + 1;
    int16_t max_interp_size_h = Ceil(support_h) * 2 + 1;

    tilingData.set_max_interp_size_w(max_interp_size_w);
    tilingData.set_max_interp_size_h(max_interp_size_h);

    float invscale_w = (realScale_w >= 1.0) ? 1.0 / realScale_w : 1.0;
    float invscale_h = (realScale_h >= 1.0) ? 1.0 / realScale_h : 1.0;

    tilingData.set_invscale_w(invscale_w);
    tilingData.set_invscale_h(invscale_h);
}

inline float UpsampleBilinearAATiling::compute_scale_value(
    int64_t input_size, int64_t out_size, bool align_corner, const float *scale) const
{
    if (out_size == input_size) {
        return static_cast<float>(1);
    }
    if (align_corner) {
        if (out_size > 1) {
            return static_cast<float>(input_size - 1) / (out_size - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return (scale != nullptr && *scale > 0) ? static_cast<float>(*scale)
                                                : (static_cast<float>(input_size) / out_size);
    }
}

inline bool UpsampleBilinearAATiling::CheckScales(const gert::TilingContext *context, float scales_w, float scales_h)
{
    const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());
    int64_t inputH = input_shape.GetDim(H_INDEX);
    int64_t inputW = input_shape.GetDim(W_INDEX);
    int64_t outputH = output_size_array[0];
    int64_t outputW = output_size_array[1];
    float tmpScalesH = scales_h > 0 ? scales_h : static_cast<float>(inputH / outputH);
    float tmpScalesW = scales_w > 0 ? scales_w : static_cast<float>(inputW / outputW);
    OP_CHECK_IF((tmpScalesH > MAX_SUPPORT_SCALE || tmpScalesW > MAX_SUPPORT_SCALE),
        OP_LOGE(context->GetNodeName(),
            "Scales should not exceed 50, but got scale (scales_w: %f, scales_h: %f) ",
            tmpScalesW,
            tmpScalesH),
        return false);
    return true;
}

inline bool FloatEqual(float a, float b)
{
    float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

ge::graphStatus UpsampleBilinearAATiling::RunBigKernelTiling(gert::TilingContext *context)
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    OP_CHECK_IF(srcTensor == nullptr, OP_LOGE(tilingContext->GetNodeName(), "srcTensor == nullptr"),
        return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(tilingContext->GetNodeName(), "attrs == nullptr"), return ge::GRAPH_FAILED);
    output_size = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_ATTR);
    OP_CHECK_IF(output_size == nullptr, OP_LOGE(tilingContext->GetNodeName(), "output_size == nullptr"), return ge::GRAPH_FAILED);
    align_corners = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    OP_CHECK_IF(align_corners == nullptr, OP_LOGE(tilingContext->GetNodeName(), "align_corners == nullptr"), return ge::GRAPH_FAILED);
    scale_h = attrs->GetAttrPointer<float>(SCALE_H_ATTR);
    OP_CHECK_IF(scale_h == nullptr, OP_LOGE(tilingContext->GetNodeName(), "scale_h == nullptr"), return ge::GRAPH_FAILED);
    scale_w = attrs->GetAttrPointer<float>(SCALE_W_ATTR);
    OP_CHECK_IF(scale_w == nullptr, OP_LOGE(tilingContext->GetNodeName(), "scale_w == nullptr"), return ge::GRAPH_FAILED);

    auto temp = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(temp == nullptr, OP_LOGE(tilingContext->GetNodeName(), "temp == nullptr"), return ge::GRAPH_FAILED);

    ge::DataType srcDtype = ge::DT_UNDEFINED;
    srcDtype = temp->GetDataType();

    // Determine whether all data types are consistent.
    if (dataType == ge::DT_UNDEFINED) {
        dataTypeSize = GetDataTypeSize();
        dataType = srcDtype;
    } else if (srcDtype != dataType) {
        return ge::GRAPH_FAILED;
    }

    auto src_shape = tilingContext->GetInputShape(0);
    input_shape = src_shape->GetOriginShape();

    if (!CheckScales(tilingContext, *scale_w, *scale_h)) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t coreNumPlatForm = ascendcPlatform.GetCoreNumAiv();

    tilingContext->SetTilingKey(1);

    setScale();
    getSlideSize();
    getShapes();
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatForm);
    // 计算workspace，每个核的系数矩阵（是否要乘2，避免doubelBuffer矩阵相互影响？），中间矩阵大小
    if (!getWorkSpace(needCoreNum)) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetBlockDim(needCoreNum);

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

uint32_t UpsampleBilinearAATiling::GetNeedCoreNum(uint32_t coreNumPlatForm)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (!FloatEqual(realScale_w, 1.0)) {
        singleCoreK_w = Ceil(slide_size * realScale_w) + Ceil(tilingData.get_max_interp_size_w());
        if (singleCoreK_w > input_shapes[W_INDEX]) {
            singleCoreK_w = input_shapes[W_INDEX];
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatForm);
        getTCubeTiling_w();
    }

    if (!FloatEqual(realScale_h, 1.0) || FloatEqual(realScale_w, 1.0)) {
        singleCoreK_h = Ceil(slide_size * realScale_h) + Ceil(tilingData.get_max_interp_size_h());
        if (singleCoreK_h > input_shapes[H_INDEX]) {
            singleCoreK_h = input_shapes[H_INDEX];
        }
        needCoreNumH = GetNeedCoreNumH(coreNumPlatForm);
        getTCubeTiling_h();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

void UpsampleBilinearAATiling::getTCubeTiling_w()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetOrgShape(input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shape[H_INDEX],
        output_shapes[W_INDEX],
        input_shapes[W_INDEX]);
    mmTiling_w.SetShape(
        input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shape[H_INDEX], slide_size, singleCoreK_w);
    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        OP_LOGE(tilingContext->GetNodeName(), "getTCubeTiling_w Error, please Check inputShapes.");
        return;
    }
}

void UpsampleBilinearAATiling::getTCubeTiling_h()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingH;
    mmTilingH.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingH.SetOrgShape(output_shapes[H_INDEX], output_shapes[W_INDEX], input_shapes[W_INDEX]);
    mmTilingH.SetShape(slide_size, output_shapes[W_INDEX], singleCoreK_h);

    if (mmTilingH.GetTiling(tilingData.matmulTiling_h) == -1) {
        OP_LOGE(tilingContext->GetNodeName(), "getTCubeTiling_h Error, please Check inputShapes.");
        return;
    }
}

// 先只算w方向
bool UpsampleBilinearAATiling::getWorkSpace(uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return false;
    }
    // 中间tensor
    uint64_t intermediate_matrix_size =
        output_shapes[0] * output_shapes[1] * input_shapes[2] * output_shapes[3] * dataTypeSize;
    intermediate_matrix_size = (intermediate_matrix_size + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    // 每个核的系数矩阵，每个核申请两个workspace空间，避免相互覆盖
    int64_t singleCoreK = singleCoreK_w > singleCoreK_h ? singleCoreK_w : singleCoreK_h;

    uint32_t radioMatrixWorkspaceSize = slide_size * singleCoreK * dataTypeSize;
    workspaces[0] = intermediate_matrix_size + radioMatrixWorkspaceSize * needCoreNum + WORK_SPACE_SIZE;
    tilingData.set_radio_matrix_size_w(slide_size * singleCoreK_w);
    tilingData.set_radio_matrix_size_h(slide_size * singleCoreK_h);
    tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
    return true;
}

void UpsampleBilinearAATiling::getShapes()
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

void UpsampleBilinearAATiling::getSlideSize()
{
    auto max_scale = realScale_h > realScale_w ? realScale_h : realScale_w;
    if (max_scale <= BEST_PERFORMANCE_SCALE_4) {
        slide_size = BEST_PERFORMANCE_SIZE_4;
    } else if (max_scale <= BEST_PERFORMANCE_SCALE_3) {
        slide_size = BEST_PERFORMANCE_SIZE_3;
    } else if (max_scale <= BEST_PERFORMANCE_SCALE_2) {
        slide_size = BEST_PERFORMANCE_SIZE_2;
    } else {
        slide_size = BEST_PERFORMANCE_SIZE_1;
    }
    tilingData.set_slide_size(slide_size);
}

template <typename T1, typename T2>
inline auto UpsampleBilinearAATiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleBilinearAATiling::Ceil(T1 x) const
{
    int32_t floor_x = int32_t(x);
    if (x == floor_x) {
        return floor_x;
    }
    return floor_x + 1;
}

uint8_t UpsampleBilinearAATiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_FOUR;
        case ge::DT_FLOAT16:
            return BYTE_LEN_TWO;
        case ge::DT_BF16:
            return BYTE_LEN_TWO;
        default:
            return BYTE_LEN_FOUR;
    }
}

uint64_t UpsampleBilinearAATiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return DATE_TYPE_FLOAT16;
        case ge::DT_BF16:
            return DATE_TYPE_HALF;
        case ge::DT_FLOAT:
            return DATE_TYPE_FLOAT;
        default:
            return 0;
    }
}

uint32_t UpsampleBilinearAATiling::GetNeedCoreNumW(uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[3];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainderW = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // H维度总数
    int64_t input_h = input_shapes[0] * input_shapes[1] * input_shapes[2];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = slide_size;

    if (remainderW != 0) {
        // 获取最小分行数
        int64_t minAvergingRows = slide_size;
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainderW;
        tailAvergingRows = std::max(CeilA2B(input_h, groupCoreNum), minAvergingRows);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(input_h, tailAvergingRows));
    }
    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;

    tilingData.set_eachCoreSlideNumW(eachCoreSlideNum);
    tilingData.set_tailStartSlideNumW(tailStartSlideNum);
    tilingData.set_slideNumW(slideNum);
    tilingData.set_groupCoreNumW(groupCoreNum);
    tilingData.set_tailAvergingRowsW(tailAvergingRows);
    tilingData.set_remainderW(remainderW);

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainderW != 0) {
        for (uint32_t index = 0; index < coreNumPlatform; index++) {
            groupCoreNum = groupCoreNum == 0 ? 1 : groupCoreNum;
            // 尾块处理
            int64_t groupIndex = index / groupCoreNum;
            if (groupIndex < remainderW) {
                needCoreNum++;
            }
        }
    }

    tilingData.set_need_core_num_w(needCoreNum);

    return needCoreNum;
}

uint32_t UpsampleBilinearAATiling::GetNeedCoreNumH(uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[2];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // Batch和W维度总数
    int64_t batch = input_shapes[0] * input_shapes[1];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingBatch = slide_size;
    if (remainder != 0) {
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingBatch = CeilA2B(batch, groupCoreNum);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(batch, tailAvergingBatch));
    }

    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;

    tilingData.set_eachCoreSlideNumH(eachCoreSlideNum);
    tilingData.set_tailStartSlideNumH(tailStartSlideNum);
    tilingData.set_slideNumH(slideNum);
    tilingData.set_groupCoreNumH(groupCoreNum);
    tilingData.set_tailAvergingRowsH(tailAvergingBatch);
    tilingData.set_remainderH(remainder);

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainder != 0) {
        for (uint32_t indexH = 0; indexH < coreNumPlatform; indexH++) {
            groupCoreNum = groupCoreNum == 0 ? 1 : groupCoreNum;
            // 尾块处理, 核数不全都一样
            int64_t groupIndex = indexH / groupCoreNum;
            if (groupIndex < remainder) {
                needCoreNum++;
            }
        }
    }

    tilingData.set_need_core_num_h(needCoreNum);

    return needCoreNum;
}

void UpsampleBilinearAATiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleBilinearAATiling(gert::TilingContext *context)
{
    UpsampleBilinearAATiling tilingObject(context);
    context->SetScheduleMode(SCHEDULE_MODE);
    return tilingObject.RunBigKernelTiling(context);
}

IMPL_OP_OPTILING(UpsampleBilinear2dAA)
    .Tiling(tiling4UpsampleBilinearAATiling);

}  // namespace optiling
