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
 * \file upsample_linear1d_tiling.cpp
 * \brief
 */
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_linear1d_tiling.h"

namespace optiling {
constexpr uint32_t BEST_PERFORMANCE_SIZE_16 = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_32 = 32;
constexpr uint32_t BEST_PERFORMANCE_SIZE_64 = 64;
constexpr uint32_t BEST_PERFORMANCE_SIZE_128 = 128;

constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_50 = 50;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_20 = 20;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_8 = 8;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_5 = 5;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_1 = 1;

constexpr float HALF_NUM = 0.5;

constexpr uint32_t BYTE = 8;
constexpr uint32_t BYTE_REPEAT = 256;  // The amount of data that can be processed by a repeat.
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t W_INDEX = 2;
constexpr int8_t DIM_ZERO = 0;
constexpr int8_t DIM_ONE = 1;
constexpr int8_t DIM_TWO = 2;
constexpr int8_t DIM_THREE = 3;

constexpr uint32_t ALIGN_CORNERS_ATTR = 0;
constexpr uint32_t SCALES_ATTR = 1;
constexpr uint64_t DATA_TYPE_FP32_SIZE = 4;

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BYTE_LENGTH_4 = 4;
constexpr uint32_t BYTE_LENGTH_2 = 2;

constexpr uint32_t DIM_LEN = 4;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;

constexpr int8_t NUM_ONE = 1;
constexpr int8_t NUM_TWO = 2;
constexpr int8_t NUM_FOUR = 4;
constexpr int64_t NUM_1024 = 1024;

constexpr float MAX_SUPPORT_SHRINK_SCALE = 50.0f;
constexpr float MAX_SUPPORT_ZOOM_SCALE = 800.0f;
constexpr float MAX_SUPPORT_ZOOM_SCALE_REV = 0.00125f;

constexpr float support = 1.0;
constexpr int64_t max_interp_size = 2;
constexpr int64_t max_interp_size_10 = 10;
constexpr uint8_t SCHEDULE_MODE = 1;
constexpr int64_t UB_FREE = 1024;

class UpsampleLinear1dTiling {
public:
    explicit UpsampleLinear1dTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    void setScale();
    void get_scale_from_out();
    inline float compute_scale_value(
        const int64_t input_size, const int64_t output_size, const bool align_corner, const float scale) const;
    bool getWorkSpace(const uint32_t needCoreNum, const int64_t ubSize);
    void getWorkspaceBlock(const uint64_t input_h, const uint64_t mPerTime, uint64_t& matmulBlockPerTime);
    uint64_t getWorkspaceBlockPart1(const uint64_t input_h, const uint64_t mPerTime, uint64_t& matmulBlockPerTime);
    void getWorkspaceRemainder(uint64_t input_h, uint64_t mPerTime, uint64_t matmulBlockPerTime);
    bool getWorkLoopInfo(const int64_t ubSize, const uint64_t radioMatrixWorkspaceSize, const uint64_t singleCoreKAlign, const uint64_t slide_size, const uint32_t needCoreNum);
    void getShapes();
    void setSlideSize(const uint32_t coreNumPlatFormInfo);
    inline int64_t calculateSlideSize(const uint32_t coreNumPlatFormInfo);
    inline int64_t getSlideSizeByScale(const uint32_t coreNumPlatFormInfo, float real_scale);
    uint8_t GetDataTypeSize() const;
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatFormInfo);
    uint32_t GetNeedCoreNumW(const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size);
    void FillTilingData();
    void getTCubeTiling_w();
    inline bool CheckScales(
        const gert::TilingContext *context, const float scales_w) const;
    inline int64_t getSingleCoreK(const int64_t slideSize, const float scale, const bool alignCorners) const;
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    int64_t slide_size_w{16};
    UpsampleLinear1dTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize{4};
    gert::Shape input_shape;
    gert::Shape output_shape;
    const bool *align_corners{nullptr};
    float scale_w = 0.0f;
    float realScale_w{0.0f};

    int64_t output_shapes[3] = {0};
    int64_t input_shapes[3] = {0};

    int64_t slide_size_list[4] = {BEST_PERFORMANCE_SIZE_16,
        BEST_PERFORMANCE_SIZE_32,
        BEST_PERFORMANCE_SIZE_64,
        BEST_PERFORMANCE_SIZE_128};

    TCubeTiling matmulTiling_w;
    int64_t singleCoreK_w = 0;
};

void UpsampleLinear1dTiling::setScale()
{
    realScale_w = compute_scale_value(input_shapes[W_INDEX], output_shapes[W_INDEX], *align_corners, scale_w);
    tilingData.set_scale_w(realScale_w);
}

void UpsampleLinear1dTiling::get_scale_from_out()
{
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    align_corners = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    const float *scales = attrs->GetAttrPointer<float>(SCALES_ATTR);
    scale_w = *scales;
}

inline float UpsampleLinear1dTiling::compute_scale_value(
    const int64_t input_size, const int64_t output_size, const bool align_corner, const float scale) const
{
    if (output_size == input_size) {
        return static_cast<float>(1);
    }
    if (align_corner) {
        if (output_size > 1) {
            return static_cast<float>(input_size - 1) / (output_size - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return (scale > 0) ? static_cast<float>(scale) : (static_cast<float>(input_size) / output_size);
    }
}

inline bool FloatEqual(const float a, const float b)
{
    const float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

inline bool UpsampleLinear1dTiling::CheckScales(
    const gert::TilingContext *context, const float scales_w) const
{
    return true;
}

ge::graphStatus UpsampleLinear1dTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    auto temp = tilingContext->GetInputDesc(0);
    if (srcTensor == nullptr || temp == nullptr) {
        return ge::GRAPH_FAILED;
    }
    get_scale_from_out();

    ge::DataType srcDtype = ge::DT_UNDEFINED;
    srcDtype = temp->GetDataType();

    // Determine whether all data types are consistent.
    if (dataType == ge::DT_UNDEFINED) {
        dataType = srcDtype;
        dataTypeSize = GetDataTypeSize();
    } else if (srcDtype != dataType) {
        return ge::GRAPH_FAILED;
    }
    auto src_shape = tilingContext->GetInputShape(0);
    auto dst_shape = tilingContext->GetOutputShape(0);

    input_shape = src_shape->GetOriginShape();
    output_shape = dst_shape->GetOriginShape();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t coreNumPlatFormInfo = ascendcPlatform.GetCoreNumAiv();
    if (coreNumPlatFormInfo < 1) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (dataTypeSize == DATA_TYPE_FP32_SIZE) {
        tilingContext->SetTilingKey(2);
    } else {
        tilingContext->SetTilingKey(1);
    }
    tilingData.set_align_corners(*align_corners);
    getShapes();
    setScale();
    if (!CheckScales(tilingContext, realScale_w)) {
        return ge::GRAPH_FAILED;
    }

    setSlideSize(coreNumPlatFormInfo);
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatFormInfo);
    // 计算workspace，每个核的系数矩阵，中间矩阵大小
    if (!getWorkSpace(needCoreNum, ubSize)) {
        return ge::GRAPH_FAILED;
    }

    if (!FloatEqual(realScale_w, 1.0)) {
        getTCubeTiling_w();
    }
    tilingContext->SetBlockDim(needCoreNum);
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

uint32_t UpsampleLinear1dTiling::GetNeedCoreNum(const uint32_t coreNumPlatFormInfo)
{
    uint32_t needCoreNumW = 0;
    if (!FloatEqual(realScale_w, 1.0)) {
        int64_t kMinValue = getSingleCoreK(0, realScale_w, *align_corners);
        int64_t kMaxValue = getSingleCoreK(slide_size_w, realScale_w, *align_corners);
        singleCoreK_w = kMaxValue - kMinValue + Ceil(max_interp_size_10);
        if (singleCoreK_w > input_shapes[W_INDEX]) {
            singleCoreK_w = input_shapes[W_INDEX];
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatFormInfo, NUM_TWO, slide_size_w);
    }

    uint32_t needCoreNum = needCoreNumW;
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

void UpsampleLinear1dTiling::getTCubeTiling_w()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling_w.SetOrgShape(input_shapes[N_INDEX] * input_shapes[C_INDEX],
        output_shapes[W_INDEX],
        input_shapes[W_INDEX]);
    if(dataTypeSize == DATA_TYPE_FP32_SIZE) {
        mmTiling_w.SetShape(input_shapes[N_INDEX] * input_shapes[C_INDEX], slide_size_w, singleCoreK_w);
    } else {
        uint64_t matmulBlockPerTime = tilingData.get_matmulBlockPerTime();
        mmTiling_w.SetShape(matmulBlockPerTime, slide_size_w, singleCoreK_w);
    }
    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        return;
    }
}

// 先只算w方向
bool UpsampleLinear1dTiling::getWorkSpace(const uint32_t needCoreNum, const int64_t ubSize)
{
    // 每个核的系数矩阵，每个核申请两个workspace空间，避免相互覆盖
    uint64_t blockSizeNum = (32 / dataTypeSize);
    uint64_t singleCoreK = singleCoreK_w;
    uint64_t slide_size = slide_size_w;
    uint64_t singleCoreKAlign = (singleCoreK + blockSizeNum - 1) / blockSizeNum * blockSizeNum;
    uint64_t radio_matrix_num_w = slide_size_w * singleCoreK;
    radio_matrix_num_w = (radio_matrix_num_w + BEST_PERFORMANCE_SIZE_128 - 1) / BEST_PERFORMANCE_SIZE_128 * BEST_PERFORMANCE_SIZE_128;
    uint64_t radioMatrixWorkspaceSize = radio_matrix_num_w * 4;
    tilingData.set_blockSizeNum(blockSizeNum);
    tilingData.set_radio_matrix_size_w(radio_matrix_num_w);
    if (dataTypeSize == DATA_TYPE_FP32_SIZE) {
        size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            return false;
        }
        workspaces[0] = radioMatrixWorkspaceSize * needCoreNum + WORK_SPACE_SIZE;
        return true;
    } else {
        bool res = getWorkLoopInfo(ubSize, radioMatrixWorkspaceSize, singleCoreKAlign, slide_size, needCoreNum);
        return res;
    }
}

uint64_t UpsampleLinear1dTiling::getWorkspaceBlockPart1(
    const uint64_t input_h, const uint64_t mPerTime, uint64_t& matmulBlockPerTime)
{
    // 搬入搬出循环次数
    uint64_t loopTimes_0 = 1;
    uint64_t loopTimes_1 = 1;
    // 搬入搬出尾块大小
    uint64_t loopTail_0 = 0;
    uint64_t loopTail_1 = 0;
    // matmul循环次数
    uint64_t matmulLoopTimes = 1;
    uint64_t matmulBlockTail = 0;
    // input_h = matmulLoopTimes * matmulBlockPerTime + matmulBlockTail
    // 一次可以算完
    if (matmulBlockPerTime > input_h) {
        matmulLoopTimes = 1;
        matmulBlockPerTime = input_h;
        matmulBlockTail = 0;
    } else {
        matmulLoopTimes = matmulBlockPerTime == 0 ? 0 : input_h / matmulBlockPerTime;
        matmulBlockTail = input_h - matmulLoopTimes * matmulBlockPerTime;
    }

    // matmulBlockPerTime = (loopTimes_0 + loopTimes_1) * mPerTime + loopTail_0 + loopTail_1
    if (matmulBlockPerTime >= mPerTime) {
        // 每个aiv核单次需要处理多少数据
        int64_t mm_0 = (matmulBlockPerTime / 2) > 0 ? (matmulBlockPerTime / 2) : matmulBlockPerTime;
        int64_t mm_1 = matmulBlockPerTime - mm_0;
        loopTimes_0 = mPerTime == 0 ? 0 : mm_0 / mPerTime;
        loopTimes_1 = mPerTime == 0 ? 0 : mm_1 / mPerTime;
        // 尾块只在0核有, 避免出现分不均匀
        uint64_t loopTailNum = matmulBlockPerTime - (loopTimes_0 + loopTimes_1) * mPerTime;
        loopTail_0 = mm_0 - loopTimes_0 * mPerTime;
        loopTail_1 = mm_1 - loopTimes_1 * mPerTime;
    } else {
        // 如果 matmulBlockPerTime == mPerTime, 那么只需要0核一次就处理完了
        matmulBlockPerTime = mPerTime;
        loopTimes_0 = 1;
        loopTimes_1 = 0;
        loopTail_0 = 0;
        loopTail_1 = 0;
    }
    int64_t matmulBlockTail_0 = (matmulBlockTail / 2) > 0 ? (matmulBlockTail / 2) : matmulBlockTail;
    int64_t matmulBlockPerTime_0 = (matmulBlockPerTime / 2) > 0 ? (matmulBlockPerTime / 2) : matmulBlockPerTime;
    tilingData.set_loopTimes0(loopTimes_0);
    tilingData.set_loopTimes1(loopTimes_1);
    tilingData.set_loopTail0(loopTail_0);
    tilingData.set_loopTail1(loopTail_1);
    tilingData.set_matmulLoopTimes(matmulLoopTimes);
    tilingData.set_matmulBlockPerTime(matmulBlockPerTime);
    tilingData.set_matmulBlockPerTime0(matmulBlockPerTime_0);
    tilingData.set_matmulBlockTail(matmulBlockTail);
    tilingData.set_matmulBlockTail0(matmulBlockTail_0);
    return matmulBlockTail;
}

void UpsampleLinear1dTiling::getWorkspaceBlock(
    const uint64_t input_h, const uint64_t mPerTime, uint64_t& matmulBlockPerTime) {
    // matmul循环次数
    uint64_t matmulBlockTail = getWorkspaceBlockPart1(input_h, mPerTime, matmulBlockPerTime);
    // 尾块搬入搬出循环次数
    uint64_t loopTailTimes_0 = 1;
    uint64_t loopTailTimes_1 = 1;
    // 搬入搬出尾块大小
    uint64_t loopTailTail_0 = 0;
    uint64_t loopTailTail_1 = 0;
    // matmulBlockTail = (loopTailTimes_0 + loopTailTimes_1) * mPerTime + loopTailTail_0 + loopTailTail_1
    if (matmulBlockTail > mPerTime) {
        int64_t mm_tail_0 = (matmulBlockTail / 2) > 0 ? (matmulBlockTail / 2) : matmulBlockTail;
        int64_t mm_tail_1 = matmulBlockTail - mm_tail_0;
        loopTailTimes_0 = mPerTime == 0 ? 0 : mm_tail_0 / mPerTime;
        loopTailTimes_1 = mPerTime == 0 ? 0 : mm_tail_1 / mPerTime;
        loopTailTail_0 = mm_tail_0 - loopTailTimes_0 * mPerTime;
        loopTailTail_1 = mm_tail_1 - loopTailTimes_1 * mPerTime;
    } else {
        // 如果 matmulBlockTail == loopTailTail, 那么只需要0核一次就处理完了
        loopTailTimes_0 = 0;
        loopTailTimes_1 = 0;
        loopTailTail_0 = (matmulBlockTail / 2) > 0 ? (matmulBlockTail / 2) : matmulBlockTail;
        loopTailTail_1 = matmulBlockTail - loopTailTail_0;
    }
    tilingData.set_loopTailTimes0(loopTailTimes_0);
    tilingData.set_loopTailTimes1(loopTailTimes_1);
    tilingData.set_loopTailTail0(loopTailTail_0);
    tilingData.set_loopTailTail1(loopTailTail_1);
}

void UpsampleLinear1dTiling::getWorkspaceRemainder(
    uint64_t input_h, uint64_t mPerTime, uint64_t matmulBlockPerTime)
{
    uint64_t tailAvergingRowsW = std::min(static_cast<uint64_t>(tilingData.get_tailAvergingRowsW()), input_h);
    uint64_t remainderMatmulLoopTimes = 1;
    uint64_t remainderMatmulBlockTail = 0;
    // tailAvergingRowsW = remainderMatmulLoopTimes * matmulBlockPerTime + remainderMatmulBlockTail
    if (matmulBlockPerTime > tailAvergingRowsW) {
        remainderMatmulLoopTimes = 0;
        remainderMatmulBlockTail = tailAvergingRowsW;
    } else {
        remainderMatmulLoopTimes = matmulBlockPerTime > 0 ? tailAvergingRowsW / matmulBlockPerTime : 0;
        remainderMatmulBlockTail = tailAvergingRowsW - remainderMatmulLoopTimes * matmulBlockPerTime;
    }
    int64_t remainderMMTailHalf = remainderMatmulBlockTail / 2;
    int64_t remainderMMTail_0 = remainderMMTailHalf > 0 ? remainderMMTailHalf : remainderMatmulBlockTail;
    int64_t remainderMMTail_1 = remainderMatmulBlockTail - remainderMMTail_0;
    uint64_t remainderLoopTailTimes0 = mPerTime > 0 ? (remainderMMTail_0 / mPerTime) : 0;
    uint64_t remainderLoopTailTimes1 = mPerTime > 0 ? (remainderMMTail_1 / mPerTime) : 0;
    uint64_t remainderLoopTailTail0 = remainderMMTail_0 - remainderLoopTailTimes0 * mPerTime;
    uint64_t remainderLoopTailTail1 = remainderMMTail_1 - remainderLoopTailTimes1 * mPerTime;

    tilingData.set_remainderMatmulLoopTimes(remainderMatmulLoopTimes);
    tilingData.set_remainderMatmulBlockTail(remainderMatmulBlockTail);
    tilingData.set_remainderMatmulBlockTail0(remainderMMTail_0);
    tilingData.set_remainderLoopTailTimes0(remainderLoopTailTimes0);
    tilingData.set_remainderLoopTailTimes1(remainderLoopTailTimes1);
    tilingData.set_remainderLoopTailTail0(remainderLoopTailTail0);
    tilingData.set_remainderLoopTailTail1(remainderLoopTailTail1);
}

bool UpsampleLinear1dTiling::getWorkLoopInfo(
    const int64_t ubSize, const uint64_t radioMatrixWorkspaceSize, const uint64_t singleCoreKAlign, const uint64_t slide_size, const uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return false;
    }
    uint64_t blockSizeNum = (32 / dataTypeSize);
    int64_t input_h = input_shapes[N_INDEX] * input_shapes[C_INDEX];
    uint64_t ubEnable = static_cast<uint64_t>(ubSize) - UB_FREE - radioMatrixWorkspaceSize;
    // 单aic核可以使用的内存大小
    uint64_t workspaceEnablePerCore = (1024 * 1024 * 16) / 20;
    // ub下可以使用的最大内存大小
    uint64_t mPerTime = ubEnable / (4 * singleCoreKAlign + 4 * slide_size);
    mPerTime = mPerTime / blockSizeNum * blockSizeNum;

    mPerTime = mPerTime >= input_h ? input_h : mPerTime;
    // workspaceEnablePerCore = radioMatrixWorkspaceSize + sizeof(float) * matmulBlockPerTime * (singleCoreKAlign + slide_size)
    uint64_t matmulBlockPerTime = (workspaceEnablePerCore - radioMatrixWorkspaceSize) / (4 * (singleCoreKAlign + slide_size));
    matmulBlockPerTime = matmulBlockPerTime / blockSizeNum * blockSizeNum;
    getWorkspaceBlock(input_h, mPerTime, matmulBlockPerTime);
    getWorkspaceRemainder(input_h, mPerTime, matmulBlockPerTime);
    uint64_t mPerTimeUpAlign = (mPerTime + blockSizeNum - 1) / blockSizeNum * blockSizeNum;
    uint64_t inputUbSize = mPerTimeUpAlign * singleCoreKAlign * 4;
    uint64_t outputUbSize = mPerTimeUpAlign * slide_size * 4;
    tilingData.set_inputUbSize(inputUbSize);
    tilingData.set_outputUbSize(outputUbSize);
    tilingData.set_inputH(input_h);
    tilingData.set_mPerTime(mPerTime);

    uint64_t mmInputNum = matmulBlockPerTime * singleCoreKAlign;
    uint64_t mmOutputNum = matmulBlockPerTime * slide_size;
    uint64_t radio_matrix_size_w = tilingData.get_radio_matrix_size_w();
    uint64_t mmtotalPerCoreNum = radio_matrix_size_w + mmInputNum + mmOutputNum;
    tilingData.set_mmInputNum(mmInputNum);
    tilingData.set_mmtotalPerCoreNum(mmtotalPerCoreNum);
    uint64_t inputWorkSpaceSize = mmInputNum * 4 * needCoreNum;
    uint64_t outputWorkSpaceSize = mmOutputNum * 4 * needCoreNum;
    uint64_t middleRatioSize = radioMatrixWorkspaceSize * needCoreNum;
    workspaces[0] = middleRatioSize + inputWorkSpaceSize + outputWorkSpaceSize + WORK_SPACE_SIZE;
    return true;
};

void UpsampleLinear1dTiling::getShapes()
{
    for (int8_t i = 0; i < DIM_THREE; i++) {
        input_shapes[i] = input_shape.GetDim(i);
        output_shapes[i] = output_shape.GetDim(i);
    }
    tilingData.set_input_shapes(input_shapes);
    tilingData.set_output_shapes(output_shapes);
}

void UpsampleLinear1dTiling::setSlideSize(const uint32_t coreNumPlatFormInfo)
{
    slide_size_w = getSlideSizeByScale(coreNumPlatFormInfo, realScale_w);
    tilingData.set_slide_size_w(slide_size_w);
}

inline int64_t UpsampleLinear1dTiling::getSlideSizeByScale(
    const uint32_t coreNumPlatFormInfo, float real_scale)
{
    int64_t slide_size = 16;

    int64_t slideSizeBysize = calculateSlideSize(coreNumPlatFormInfo);
    if (input_shapes[DIM_TWO] == NUM_ONE) {
        slideSizeBysize = NUM_1024;
    }
    if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_1) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_128), slideSizeBysize);
    } else if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_5) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_64), slideSizeBysize);
    } else if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_20) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_32), slideSizeBysize);
    } else {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_16), slideSizeBysize);
    }
    return slide_size;
}

inline int64_t UpsampleLinear1dTiling::calculateSlideSize(const uint32_t coreNumPlatFormInfo)
{
    int64_t slide_size = BEST_PERFORMANCE_SIZE_16;
    uint32_t neeCoreNumMax = 0;

    for (uint32_t coreIndex = 0; coreIndex < NUM_FOUR; coreIndex++) {
        uint32_t res = 0;
        res = GetNeedCoreNumW(coreNumPlatFormInfo, NUM_ONE, slide_size_list[coreIndex]);
        slide_size = res >= neeCoreNumMax ? slide_size_list[coreIndex] : slide_size;
        neeCoreNumMax = std::max(res, neeCoreNumMax);
    }
    return slide_size;
}

template <typename T1, typename T2>
inline auto UpsampleLinear1dTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleLinear1dTiling::Ceil(T1 x) const
{
    int32_t floor_x = int32_t(x);
    if (x == static_cast<T1>(floor_x)) {
        return floor_x;
    }
    return floor_x + 1;
}

inline int64_t UpsampleLinear1dTiling::getSingleCoreK(
    const int64_t slideSize, const float scale, const bool alignCorners) const
{
    float rel_idx = 0;
    float calNum = std::max(0.0f, static_cast<float>(slideSize - 1));
    if (alignCorners) {
        rel_idx = scale * calNum;
    } else {
        rel_idx = std::max(static_cast<float>((calNum + HALF_NUM) * scale - HALF_NUM), static_cast<float>(0.0));
    }
    int64_t floor_rel_idx = int32_t(rel_idx);
    return floor_rel_idx;
};

uint8_t UpsampleLinear1dTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LENGTH_4;
        case ge::DT_FLOAT16:
            return BYTE_LENGTH_2;
        case ge::DT_BF16:
            return BYTE_LENGTH_2;
        default:
            return BYTE_LENGTH_4;
    }
}

uint32_t UpsampleLinear1dTiling::GetNeedCoreNumW(
    const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size)
{
    int64_t outputSize = output_shapes[W_INDEX];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // H维度总数
    int64_t input_h = input_shapes[N_INDEX] * input_shapes[C_INDEX];
    int64_t tailAvergingRows = slide_size;
    int64_t groupCoreNum = coreNumPlatform;

    if (remainder != 0) {
        // 获取最小分行数
        int64_t minAvergingRows = slide_size;
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRows = std::max(CeilA2B(input_h, groupCoreNum), minAvergingRows);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(input_h, tailAvergingRows));
    }

    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    int64_t needCoreNum = 0;

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainder != 0) {
        for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
            groupCoreNum = groupCoreNum == 0 ? 1 : groupCoreNum;
            // 尾块处理
            int64_t groupIndex = coreIndex / groupCoreNum;
            if (groupIndex < remainder) {
                needCoreNum++;
            }
        }
    }

    if (isCalculate == NUM_TWO) {
        tilingData.set_eachCoreSlideNumW(eachCoreSlideNum);
        tilingData.set_tailStartSlideNumW(tailStartSlideNum);
        tilingData.set_slideNumW(slideNum);
        tilingData.set_groupCoreNumW(groupCoreNum);
        tilingData.set_tailAvergingRowsW(tailAvergingRows);
        tilingData.set_remainderW(remainder);
        tilingData.set_need_core_num_w(needCoreNum);
    }

    return needCoreNum;
}

void UpsampleLinear1dTiling::FillTilingData()
{
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleLinear1dTiling(gert::TilingContext *context)
{
    UpsampleLinear1dTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(UpsampleLinear1d)
    .Tiling(tiling4UpsampleLinear1dTiling);
}  // namespace optiling
