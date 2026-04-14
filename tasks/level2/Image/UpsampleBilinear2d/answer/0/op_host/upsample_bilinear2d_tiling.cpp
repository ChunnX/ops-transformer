/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FIT FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_bilinear2d_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_bilinear2d_tiling.h"

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)

namespace optiling {
constexpr uint32_t BEST_PERFORMANCE_SIZE_128 = 128;
constexpr uint32_t BEST_PERFORMANCE_SIZE_32 = 32;
constexpr uint32_t BEST_PERFORMANCE_SIZE_48 = 48;
constexpr uint32_t BEST_PERFORMANCE_SIZE_16 = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_64 = 64;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_20 = 20;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_1 = 1;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_8 = 8;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_5 = 5;
constexpr int8_t SHAPE_SIZE = 4;
constexpr float HALF_NUM = 0.5;
constexpr uint32_t BYTE = 8;

constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t N_INDEX = 0;
constexpr int8_t W_INDEX = 3;

constexpr int8_t DIM_ONE = 1;
constexpr int8_t DIM_TWO = 2;
constexpr int8_t DIM_ZERO = 0;
constexpr int8_t DIM_THREE = 3;

constexpr uint32_t ALIGN_CORNERS_ATTR = 0;
constexpr uint32_t SCALES_H_ATTR = 1;
constexpr uint32_t SCALES_W_ATTR = 2;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_HALF = 3;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;
constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr int8_t NUM_TWO = 2;
constexpr int8_t NUM_FIVE = 5;
constexpr int8_t NUM_ONE = 1;
constexpr int64_t NUM_1024 = 1024;
constexpr int64_t max_interp_size = 2;
constexpr int64_t max_interp_size_10 = 10;
constexpr float MAX_SUPPORT_SHRINK_SCALE = 50.0f;
constexpr uint8_t SCHEDULE_MODE = 1;

class UpsampleBilinear2dTiling {
public:
    explicit UpsampleBilinear2dTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    void setScale();
    void get_scale_from_out();
    inline float compute_scale_value(
        const int64_t inputSize, const int64_t outputSize, const bool alignCorners, const float scale) const;
    bool getWorkSpace(const uint32_t needCoreNum);
    void getShapes();
    void setSlideSize(const uint32_t coreNumPlatFormInfo);
    inline int64_t calculateSlideSize(const uint32_t coreNumPlatFormInfo, uint8_t direction);
    inline int64_t getSlideSizeByScale(const uint32_t coreNumPlatFormInfo, uint8_t direction, float realScale);
    uint8_t GetDataTypeSize() const;
    uint64_t GetDataTypeVal() const;
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatFormInfo);
    uint32_t GetNeedCoreNumW(const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size);
    uint32_t GetNeedCoreNumH(const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size);
    void FillTilingData();
    void getTCubeTiling_w();
    void getTCubeTiling_h();
    inline bool CheckScales(
        const gert::TilingContext *context, const float scales_w, const float scales_h) const;
    inline int64_t getSingleCoreK(const int64_t slideSize, const float scale, const bool alignCorners) const;
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 m, T2 n) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    int64_t slide_size_w{16};
    int64_t slide_size_h{16};
    UpsampleBilinear2dTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize{4};
    gert::Shape input_shape;
    gert::Shape output_shape;
    const bool *align_corners{nullptr};
    float scale_h = 0.0f;
    float scale_w = 0.0f;
    float realScale_h{0.0f};
    float realScale_w{0.0f};

    int64_t output_shapes[4] = {0};
    int64_t input_shapes[4] = {0};

    int64_t slide_size_list[5] = {BEST_PERFORMANCE_SIZE_16,
        BEST_PERFORMANCE_SIZE_32,
        BEST_PERFORMANCE_SIZE_48,
        BEST_PERFORMANCE_SIZE_64,
        BEST_PERFORMANCE_SIZE_128};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int64_t singleCoreK_w = 0;
    int64_t singleCoreK_h = 0;
    uint32_t coreNumPlatForm = 20;
};

void UpsampleBilinear2dTiling::setScale()
{
    realScale_h = compute_scale_value(input_shapes[H_INDEX], output_shapes[H_INDEX], *align_corners, scale_h);
    realScale_w = compute_scale_value(input_shapes[W_INDEX], output_shapes[W_INDEX], *align_corners, scale_w);
    tilingData.set_scale_h(realScale_h);
    tilingData.set_scale_w(realScale_w);
}

void UpsampleBilinear2dTiling::get_scale_from_out()
{
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    align_corners = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    const float *scalesHPtr = attrs->GetAttrPointer<float>(SCALES_H_ATTR);
    const float *scalesWPtr = attrs->GetAttrPointer<float>(SCALES_W_ATTR);
    scale_h = (scalesHPtr != nullptr) ? *scalesHPtr : 0.0f;
    scale_w = (scalesWPtr != nullptr) ? *scalesWPtr : 0.0f;
}

inline float UpsampleBilinear2dTiling::compute_scale_value(
    const int64_t inputSize, const int64_t outputSize, const bool alignCorners, const float scale) const
{
    if (outputSize == inputSize) {
        return static_cast<float>(1);
    }
    if (alignCorners) {
        if (outputSize > 1) {
            return static_cast<float>(inputSize - 1) / (outputSize - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return (scale > 0) ? static_cast<float>(scale) : (static_cast<float>(inputSize) / outputSize);
    }
}

inline bool UpsampleBilinear2dTiling::CheckScales(
    const gert::TilingContext *context, const float scales_w, const float scales_h) const
{
    // 2D限制50倍
    OP_CHECK_IF((scales_h > MAX_SUPPORT_SHRINK_SCALE || scales_w > MAX_SUPPORT_SHRINK_SCALE),
        OP_LOGE(context->GetNodeName(),
            "Scales should not exceed 50, but got scale (scales_w: %f, scales_h: %f) ",
            scales_w,
            scales_h),
        return false);
    return true;
}

inline bool FloatEqual(const float m, const float n)
{
    const float closeTo0 = float(1e-6);
    if (m > n) {
        return m - n < closeTo0;
    } else {
        return n - m < closeTo0;
    }
};

ge::graphStatus UpsampleBilinear2dTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    get_scale_from_out();

    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType xDtype = ge::DT_UNDEFINED;
    xDtype = temp->GetDataType();

    // Determine whether all data types are consistent.
    if (dataType == ge::DT_UNDEFINED) {
        dataType = xDtype;
        dataTypeSize = GetDataTypeSize();
    } else if (xDtype != dataType) {
        return ge::GRAPH_FAILED;
    }
    auto srcShape = tilingContext->GetInputShape(0);
    auto dstShape = tilingContext->GetOutputShape(0);

    input_shape = srcShape->GetOriginShape();
    output_shape = dstShape->GetOriginShape();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t coreNumPlatFormInfo = ascendcPlatform.GetCoreNumAiv();
    if (coreNumPlatFormInfo < 1) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(1);
    tilingData.set_align_corners(*align_corners);
    getShapes();
    setScale();
    if (!CheckScales(tilingContext, realScale_w, realScale_h)) {
        return ge::GRAPH_FAILED;
    }

    setSlideSize(coreNumPlatFormInfo);
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatFormInfo);
    // 计算workspace，每个核的系数矩阵，中间矩阵大小
    if (!getWorkSpace(needCoreNum)) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetBlockDim(needCoreNum);
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

uint32_t UpsampleBilinear2dTiling::GetNeedCoreNum(const uint32_t coreNumPlatFormInfo)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (!FloatEqual(realScale_w, 1.0)) {
        int64_t kMinValue = getSingleCoreK(0, realScale_w, *align_corners);
        int64_t kMaxValue = getSingleCoreK(slide_size_w, realScale_w, *align_corners);
        singleCoreK_w = kMaxValue - kMinValue + Ceil(max_interp_size_10);
        if (singleCoreK_w > input_shapes[W_INDEX]) {
            singleCoreK_w = input_shapes[W_INDEX];
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatFormInfo, NUM_TWO, slide_size_w);
        getTCubeTiling_w();
    }

    if (!FloatEqual(realScale_h, 1.0) || FloatEqual(realScale_w, 1.0)) {
        singleCoreK_h = Ceil(slide_size_h * realScale_h) + Ceil(max_interp_size);
        if (singleCoreK_h > input_shapes[H_INDEX]) {
            singleCoreK_h = input_shapes[H_INDEX];
        }
        needCoreNumH = GetNeedCoreNumH(coreNumPlatFormInfo, NUM_TWO, slide_size_h);
        getTCubeTiling_h();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

void UpsampleBilinear2dTiling::getTCubeTiling_w()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetOrgShape(input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shapes[H_INDEX],
        output_shapes[W_INDEX],
        input_shapes[W_INDEX]);
    mmTiling_w.SetShape(
        input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shapes[H_INDEX], slide_size_w, singleCoreK_w);
    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        return;
    }
}

void UpsampleBilinear2dTiling::getTCubeTiling_h()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_h;
    mmTiling_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_h.SetOrgShape(output_shapes[H_INDEX], output_shapes[W_INDEX], input_shapes[W_INDEX]);
    mmTiling_h.SetShape(slide_size_h, output_shapes[W_INDEX], singleCoreK_h);

    if (mmTiling_h.GetTiling(tilingData.matmulTiling_h) == -1) {
        return;
    }
}

// 先只算w方向
bool UpsampleBilinear2dTiling::getWorkSpace(const uint32_t needCoreNum)
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
    int64_t slide_size = std::max(slide_size_w, slide_size_h);
    uint32_t radioMatrixWorkspaceSize = slide_size * singleCoreK * dataTypeSize;
    workspaces[0] = intermediate_matrix_size + radioMatrixWorkspaceSize * needCoreNum + WORK_SPACE_SIZE;
    tilingData.set_radio_matrix_size_w(slide_size_w * singleCoreK_w);
    tilingData.set_radio_matrix_size_h(slide_size_h * singleCoreK_h);
    tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
    return true;
}

void UpsampleBilinear2dTiling::getShapes()
{
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        input_shapes[i] = input_shape.GetDim(i);
        output_shapes[i] = input_shape.GetDim(i);
        if (i > C_INDEX) {
            output_shapes[i] = output_shape.GetDim(i);
        }
    }

    tilingData.set_input_shapes(input_shapes);
    tilingData.set_output_shapes(output_shapes);
}

void UpsampleBilinear2dTiling::setSlideSize(const uint32_t coreNumPlatFormInfo)
{
    slide_size_w = getSlideSizeByScale(coreNumPlatFormInfo, NUM_ONE, realScale_w);
    slide_size_h = getSlideSizeByScale(coreNumPlatFormInfo, NUM_TWO, realScale_h);
    tilingData.set_slide_size_w(slide_size_w);
    tilingData.set_slide_size_h(slide_size_h);
}

inline int64_t UpsampleBilinear2dTiling::getSlideSizeByScale(
    const uint32_t coreNumPlatFormInfo, uint8_t direction, float realScale)
{
    int64_t slideSize = 16;

    int64_t slideSizeBysize = calculateSlideSize(coreNumPlatFormInfo, direction);
    if (input_shapes[DIM_TWO] == NUM_ONE && input_shapes[DIM_THREE] == NUM_ONE) {
        slideSizeBysize = NUM_1024;
    }
    if (realScale <= BEST_PERFORMANCE_SCALE_BELOW_1) {
        slideSize = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_128), slideSizeBysize);
    } else if (realScale <= BEST_PERFORMANCE_SCALE_BELOW_5) {
        slideSize = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_64), slideSizeBysize);
    } else if (realScale <= BEST_PERFORMANCE_SCALE_BELOW_8) {
        slideSize = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_48), slideSizeBysize);
    } else if (realScale <= BEST_PERFORMANCE_SCALE_BELOW_20) {
        slideSize = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_32), slideSizeBysize);
    } else {
        slideSize = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_16), slideSizeBysize);
    }
    return slideSize;
}

inline int64_t UpsampleBilinear2dTiling::calculateSlideSize(const uint32_t coreNumPlatFormInfo, uint8_t direction)
{
    int64_t slide_size = BEST_PERFORMANCE_SIZE_16;
    uint32_t neeCoreNumMax = 0;

    for (uint32_t coreIndex = 0; coreIndex < NUM_FIVE; coreIndex++) {
        uint32_t res = 0;
        if (direction == NUM_ONE) {
            res = GetNeedCoreNumW(coreNumPlatFormInfo, NUM_ONE, slide_size_list[coreIndex]);
        } else {
            res = GetNeedCoreNumH(coreNumPlatFormInfo, NUM_ONE, slide_size_list[coreIndex]);
        }
        slide_size = res >= neeCoreNumMax ? slide_size_list[coreIndex] : slide_size;
        neeCoreNumMax = std::max(res, neeCoreNumMax);
    }
    return slide_size;
}

template <typename T1, typename T2>
inline auto UpsampleBilinear2dTiling::CeilA2B(T1 m, T2 n) const -> T1
{
    if (n != 0) {
        return (m + n - 1) / n;
    } else {
        return m;
    }
}

template <typename T1>
inline int32_t UpsampleBilinear2dTiling::Ceil(T1 x) const
{
    int32_t floor_x = int32_t(x);
    if (x == static_cast<T1>(floor_x)) {
        return floor_x;
    }
    return floor_x + 1;
}

inline int64_t UpsampleBilinear2dTiling::getSingleCoreK(
    const int64_t slideSize, const float scale, const bool alignCorners) const
{
    float relIdx = 0;
    float calSize = std::max(0.0f, static_cast<float>(slideSize - 1));
    if (alignCorners) {
        relIdx = scale * calSize;
    } else {
        relIdx = std::max(static_cast<float>((calSize + HALF_NUM) * scale - HALF_NUM), static_cast<float>(0.0));
    }
    int64_t floorRelIdx = int32_t(relIdx);
    return floorRelIdx;
};

uint8_t UpsampleBilinear2dTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_BF16:
            return BYTE_LEN_2;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        default:
            return BYTE_LEN_4;
    }
}

uint64_t UpsampleBilinear2dTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_BF16:
            return DATE_TYPE_HALF;
        case ge::DT_FLOAT16:
            return DATE_TYPE_FLOAT16;
        case ge::DT_FLOAT:
            return DATE_TYPE_FLOAT;
        default:
            return 0;
    }
}

uint32_t UpsampleBilinear2dTiling::GetNeedCoreNumW(
    const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size)
{
    int64_t outputSizeW = output_shapes[3];
    int64_t slideNumW = CeilA2B(outputSizeW, slide_size);
    int64_t eachCoreSlideNumW = coreNumPlatform > 0 ? slideNumW / coreNumPlatform : 0;
    int64_t remainderW = coreNumPlatform > 0 ? slideNumW % coreNumPlatform : 0;

    // H维度总数
    int64_t inputNCH = input_shapes[0] * input_shapes[1] * input_shapes[2];
    int64_t groupCoreNumW = coreNumPlatform;
    int64_t tailAvergingRows = slide_size;

    if (remainderW != 0) {
        // 获取最小分行数
        int64_t minAvergingRows = slide_size;
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNumW = coreNumPlatform / remainderW;
        tailAvergingRows = std::max(CeilA2B(inputNCH, groupCoreNumW), minAvergingRows);
        groupCoreNumW = std::min(groupCoreNumW, CeilA2B(inputNCH, tailAvergingRows));
    }
    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNumW * coreNumPlatform;

    if (eachCoreSlideNumW > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainderW != 0) {
        for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
            groupCoreNumW = groupCoreNumW == 0 ? 1 : groupCoreNumW;
            // 尾块处理
            int64_t groupIndex = coreIndex / groupCoreNumW;
            if (groupIndex < remainderW) {
                needCoreNum++;
            }
        }
    }

    if (isCalculate == NUM_TWO) {
        tilingData.set_eachCoreSlideNumW(eachCoreSlideNumW);
        tilingData.set_tailStartSlideNumW(tailStartSlideNum);
        tilingData.set_slideNumW(slideNumW);
        tilingData.set_groupCoreNumW(groupCoreNumW);
        tilingData.set_tailAvergingRowsW(tailAvergingRows);
        tilingData.set_remainderW(remainderW);
        tilingData.set_need_core_num_w(needCoreNum);
    }

    return needCoreNum;
}

uint32_t UpsampleBilinear2dTiling::GetNeedCoreNumH(
    const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size)
{
    int64_t outputSize = output_shapes[2];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainderH = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // Batch和W维度总数
    int64_t batch = input_shapes[0] * input_shapes[1];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingBatch = slide_size;
    if (remainderH != 0) {
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainderH;
        tailAvergingBatch = CeilA2B(batch, groupCoreNum);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(batch, tailAvergingBatch));
    }

    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainderH != 0) {
        for (uint32_t coreIndexH = 0; coreIndexH < coreNumPlatform; coreIndexH++) {
            groupCoreNum = groupCoreNum == 0 ? 1 : groupCoreNum;
            // 尾块处理, 核数不全都一样
            int64_t groupIndex = coreIndexH / groupCoreNum;
            if (groupIndex < remainderH) {
                needCoreNum++;
            }
        }
    }
    if (isCalculate == NUM_TWO) {
        tilingData.set_eachCoreSlideNumH(eachCoreSlideNum);
        tilingData.set_tailStartSlideNumH(tailStartSlideNum);
        tilingData.set_slideNumH(slideNum);
        tilingData.set_groupCoreNumH(groupCoreNum);
        tilingData.set_tailAvergingRowsH(tailAvergingBatch);
        tilingData.set_remainderH(remainderH);
        tilingData.set_need_core_num_h(needCoreNum);
    }
    return needCoreNum;
}

void UpsampleBilinear2dTiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleBilinear2dTiling(gert::TilingContext *context)
{
    UpsampleBilinear2dTiling tilingObject(context);
    context->SetScheduleMode(SCHEDULE_MODE);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(UpsampleBilinear2d)
    .Tiling(tiling4UpsampleBilinear2dTiling);
}  // namespace optiling
